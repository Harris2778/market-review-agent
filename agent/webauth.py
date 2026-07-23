"""
公开网站的用户认证 / 额度 / 对话存储模块（纯标准库实现）。

仅用 sqlite3 + hashlib(pbkdf2_hmac) + secrets + hmac，零第三方依赖，
与 Docker 镜像 python:3.13-slim 兼容，不增加 requirements。

数据库路径解析：
  env WEB_DB_PATH 显式设置优先；缺省为 ${DATA_DIR:-data}/webapp.db，
  与 main.py / agent.push 等模块的 DATA_DIR 约定一致（Railway 挂卷 /data）。

表结构：
  users           注册用户（用户名唯一，PBKDF2 加盐哈希存密码；
                  is_admin / quota_limit override / register_ip 为后加列，
                  老库由 _migrate 的 ALTER TABLE 补齐）
  tokens          会话令牌（Bearer token，30 天过期，惰性清理）
  monthly_usage   周期额度用量（主键 user_id+month；month 列存周期起始日
                  ISO 字符串，周期以用户 created_at 为锚的周年月，跨周期天然重置）
  device_accounts 设备→账号绑定（注册防多号，同 device_id 限额）
  conversations   对话（归属用户）
  messages        对话消息（归属对话）

额度语义：
  每用户每周期定额 WEB_MONTHLY_QUOTA（默认 30 条消息）；users.quota_limit
  override 非空时以 override 为准（管理员可按用户调整，0 也合法=完全禁用）。
  重置周期以用户 created_at 为锚的周年月：7/24 注册 → 8/24 重置；
  月末边界向目标月最后一天收敛（1/31 注册 → 2/28(29) 重置）。
  加油包功能已整体下线：无 packs 表、无 topup，扣减只看周期内用量。

注册防多号：
  device_id（可选）：同一 device_id 注册满 DEVICE_MAX_ACCOUNTS（默认 2）
    个账号后再注册 → DeviceLimitError（路由层 409 {error:'device_limit'}）；
  register_ip：同 IP 24h 内注册满 IP_MAX_REGISTER_PER_DAY（默认 5）个后
    再注册 → IpLimitError（路由层 429 {error:'ip_limit'}）。

管理员体系：
  users.is_admin 标记管理员；注册时用户名命中 env ADMIN_USERNAMES
  （默认 'yoozo'，逗号分隔）自动置 1，启动时 mark_admins() 幂等补齐存量用户。

所有 SQL 全部参数化，用户名/密码在入口做长度与字符校验。
"""

import calendar
import hmac
import os
import re
import secrets
import sqlite3
import hashlib
import threading
import uuid
from datetime import date, datetime, timedelta
from typing import Optional

# ── 配置 ──

TOKEN_TTL_DAYS = 30          # 令牌有效期（天）
PBKDF2_ITERATIONS = 200_000  # PBKDF2 迭代次数
PASSWORD_MIN_LEN = 6         # 密码最小长度

# 用户名：3-32 位，字母 / 数字 / 下划线 / 中文
_USERNAME_RE = re.compile(r"^[0-9A-Za-z_一-鿿]{3,32}$")

# 同一进程内串行化建表与额度扣减（SQLite 单写者，BEGIN IMMEDIATE 兜底并发）
_lock = threading.Lock()


# ── 异常 ──

class ValidationError(ValueError):
    """输入校验失败（用户名格式、密码长度等）。"""


class UserExistsError(Exception):
    """注册时用户名已存在。"""


class AuthError(Exception):
    """登录失败（用户不存在或密码错误）。"""


class DeviceLimitError(Exception):
    """注册时同一 device_id 绑定账号数已达上限。"""


class IpLimitError(Exception):
    """注册时同一 IP 24h 内注册数已达上限。"""


# ── 时间与路径（独立成函数，便于测试 monkeypatch）──

def _now() -> datetime:
    """当前本地时间。独立成函数便于测试注入（如 token 过期场景）。"""
    return datetime.now()


def _env_int(name: str, default: int) -> int:
    """读取正整数环境变量；缺失或非法时回退默认值。每次调用实时读取，便于测试。"""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def monthly_quota() -> int:
    """每周期定额消息数（env WEB_MONTHLY_QUOTA，默认 30）。"""
    return _env_int("WEB_MONTHLY_QUOTA", 30)


def device_max_accounts() -> int:
    """同一 device_id 可注册账号数上限（env DEVICE_MAX_ACCOUNTS，默认 2）。"""
    return _env_int("DEVICE_MAX_ACCOUNTS", 2)


def ip_max_register_per_day() -> int:
    """同一 IP 24h 内注册数上限（env IP_MAX_REGISTER_PER_DAY，默认 5）。"""
    return _env_int("IP_MAX_REGISTER_PER_DAY", 5)


def admin_usernames() -> list:
    """管理员用户名列表（env ADMIN_USERNAMES，默认 'yoozo'，逗号分隔）。"""
    raw = os.getenv("ADMIN_USERNAMES", "yoozo")
    return [u.strip() for u in raw.split(",") if u.strip()]


def _db_path() -> str:
    """DB 路径解析：env WEB_DB_PATH > ${DATA_DIR:-data}/webapp.db。每次调用实时读取。"""
    explicit = os.getenv("WEB_DB_PATH")
    if explicit and explicit.strip():
        return explicit.strip()
    return os.path.join(os.getenv("DATA_DIR") or "data", "webapp.db")


# ── 数据库连接与建表 ──

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    salt          TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    is_admin      INTEGER NOT NULL DEFAULT 0,
    quota_limit   INTEGER,
    register_ip   TEXT
);
CREATE TABLE IF NOT EXISTS tokens (
    token      TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS monthly_usage (
    user_id INTEGER NOT NULL REFERENCES users(id),
    month   TEXT NOT NULL,
    used    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, month)
);
CREATE TABLE IF NOT EXISTS device_accounts (
    device_id  TEXT NOT NULL,
    user_id    INTEGER NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id),
    title      TEXT NOT NULL,
    pinned     INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tokens_user ON tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_device_accounts_device ON device_accounts(device_id);
CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, id);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """
    老库结构迁移（在 _SCHEMA 之后执行）。

    conversations.pinned 与 users.is_admin / users.quota_limit /
    users.register_ip 均为后加列：新库已由 _SCHEMA 建好带列结构；
    Railway 上已存在的老库（CREATE TABLE IF NOT EXISTS 不会改已有表）
    靠这里的 ALTER TABLE 补列。列已存在时 SQLite 抛 OperationalError，
    捕获即幂等，重复执行/新旧库混跑都不报错。
    """
    for ddl in (
        "ALTER TABLE conversations ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE users ADD COLUMN quota_limit INTEGER",
        "ALTER TABLE users ADD COLUMN register_ip TEXT",
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass  # 列已存在（新库或已迁移过的老库）


def _connect() -> sqlite3.Connection:
    """新建一个数据库连接（每次调用独立连接，线程安全），并确保表结构存在。"""
    path = _db_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    with _lock:
        conn.executescript(_SCHEMA)
        _migrate(conn)
    return conn


def _iso(dt: datetime) -> str:
    # 毫秒精度：秒级会让同一秒内的连续操作 updated_at 打平，对话排序不稳定
    return dt.isoformat(timespec="milliseconds")


# ── 输入校验与密码哈希 ──

def validate_username(username) -> str:
    """校验并返回 strip 后的用户名；非法时抛 ValidationError。"""
    if not isinstance(username, str):
        raise ValidationError("用户名必须是字符串")
    username = username.strip()
    if not _USERNAME_RE.match(username):
        raise ValidationError("用户名须为 3-32 位字母、数字、下划线或中文")
    return username


def validate_password(password) -> str:
    """校验密码；非法时抛 ValidationError。"""
    if not isinstance(password, str):
        raise ValidationError("密码必须是字符串")
    if len(password) < PASSWORD_MIN_LEN:
        raise ValidationError(f"密码最少 {PASSWORD_MIN_LEN} 位")
    return password


def _hash_password(password: str, salt_hex: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), PBKDF2_ITERATIONS
    )
    return digest.hex()


# ── 注册 / 登录 / 登出 / 令牌解析 ──

def _validate_device_id(device_id) -> Optional[str]:
    """device_id 可选：须为字符串，去空白；空串按未提供处理；超长按非法处理。"""
    if device_id is None:
        return None
    if not isinstance(device_id, str):
        raise ValidationError("device_id 必须是字符串")
    device_id = device_id.strip()
    if not device_id:
        return None
    if len(device_id) > 128:
        raise ValidationError("device_id 最长 128 字符")
    return device_id


def _validate_ip(register_ip) -> Optional[str]:
    """register_ip 可选：须为字符串，去空白；空串按未提供处理。"""
    if register_ip is None:
        return None
    if not isinstance(register_ip, str):
        raise ValidationError("register_ip 必须是字符串")
    register_ip = register_ip.strip()[:64]
    return register_ip or None


def register(
    username: str,
    password: str,
    device_id: Optional[str] = None,
    register_ip: Optional[str] = None,
) -> str:
    """
    注册用户并直接签发令牌。返回 token 字符串。

    - 用户名已存在 → UserExistsError（路由层映射 409）
    - 输入非法 → ValidationError（路由层映射 400）
    - device_id 绑定账号数达 DEVICE_MAX_ACCOUNTS → DeviceLimitError（409 device_limit）
    - register_ip 24h 内注册数达 IP_MAX_REGISTER_PER_DAY → IpLimitError（429 ip_limit）
    - 用户名命中 ADMIN_USERNAMES 自动标记 is_admin=1
    """
    username = validate_username(username)
    password = validate_password(password)
    device_id = _validate_device_id(device_id)
    register_ip = _validate_ip(register_ip)
    salt = secrets.token_hex(16)
    pw_hash = _hash_password(password, salt)
    now = _now()
    now_iso = _iso(now)
    conn = _connect()
    try:
        if device_id is not None:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM device_accounts WHERE device_id = ?",
                (device_id,),
            ).fetchone()["c"]
            if count >= device_max_accounts():
                raise DeviceLimitError(f"同一设备注册账号数已达上限: {device_max_accounts()}")
        if register_ip is not None:
            cutoff = _iso(now - timedelta(hours=24))
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM users WHERE register_ip = ? AND created_at > ?",
                (register_ip, cutoff),
            ).fetchone()["c"]
            if count >= ip_max_register_per_day():
                raise IpLimitError(f"同一 IP 24h 内注册数已达上限: {ip_max_register_per_day()}")
        is_admin = 1 if username in admin_usernames() else 0
        try:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, salt, created_at, "
                "is_admin, register_ip) VALUES (?,?,?,?,?,?)",
                (username, pw_hash, salt, now_iso, is_admin, register_ip),
            )
        except sqlite3.IntegrityError:
            raise UserExistsError(f"用户名已存在: {username}")
        user_id = cur.lastrowid
        if device_id is not None:
            conn.execute(
                "INSERT INTO device_accounts (device_id, user_id, created_at) "
                "VALUES (?,?,?)",
                (device_id, user_id, now_iso),
            )
        token = _issue_token(conn, user_id)
        conn.commit()
        return token
    finally:
        conn.close()


def mark_admins() -> int:
    """
    按 env ADMIN_USERNAMES 把已存在用户幂等标记 is_admin=1（启动时调用）。
    返回受影响行数；列表为空时不做任何事。
    """
    names = admin_usernames()
    if not names:
        return 0
    conn = _connect()
    try:
        placeholders = ",".join("?" for _ in names)
        cur = conn.execute(
            f"UPDATE users SET is_admin = 1 WHERE username IN ({placeholders}) "
            "AND is_admin = 0",
            names,
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def login(username: str, password: str) -> str:
    """
    登录并签发令牌。返回 token 字符串。
    用户不存在或密码错误统一抛 AuthError（不区分，防用户枚举）。
    """
    username = validate_username(username)
    password = validate_password(password)
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, password_hash, salt FROM users WHERE username = ?", (username,)
        ).fetchone()
        ok = False
        if row is not None:
            candidate = _hash_password(password, row["salt"])
            ok = hmac.compare_digest(candidate, row["password_hash"])
        if not ok:
            raise AuthError("用户名或密码错误")
        token = _issue_token(conn, row["id"])
        conn.commit()
        return token
    finally:
        conn.close()


def _issue_token(conn: sqlite3.Connection, user_id: int) -> str:
    """在给定连接上签发一个 30 天有效的令牌（调用方负责 commit）。"""
    now = _now()
    token = secrets.token_urlsafe(32)
    conn.execute(
        "INSERT INTO tokens (token, user_id, created_at, expires_at) VALUES (?,?,?,?)",
        (token, user_id, _iso(now), _iso(now + timedelta(days=TOKEN_TTL_DAYS))),
    )
    # 惰性清理该用户的过期令牌
    conn.execute(
        "DELETE FROM tokens WHERE user_id = ? AND expires_at < ?", (user_id, _iso(now))
    )
    return token


def logout(token: str) -> None:
    """登出：删除令牌（幂等，令牌不存在也视为成功）。"""
    if not token:
        return
    conn = _connect()
    try:
        conn.execute("DELETE FROM tokens WHERE token = ?", (token,))
        conn.commit()
    finally:
        conn.close()


def resolve_token(token: str) -> Optional[dict]:
    """
    解析 Bearer token → {"id": user_id, "username": ..., "is_admin": bool}；
    无效或过期返回 None。过期令牌顺手删除（惰性清理）。
    """
    if not token or not isinstance(token, str):
        return None
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT t.user_id, t.expires_at, u.username, u.is_admin "
            "FROM tokens t JOIN users u ON u.id = t.user_id WHERE t.token = ?",
            (token,),
        ).fetchone()
        if row is None:
            return None
        if row["expires_at"] < _iso(_now()):
            conn.execute("DELETE FROM tokens WHERE token = ?", (token,))
            conn.commit()
            return None
        return {
            "id": row["user_id"],
            "username": row["username"],
            "is_admin": bool(row["is_admin"]),
        }
    finally:
        conn.close()


# ── 额度（周年月周期）──

def _clamp_day(year: int, month: int, day: int) -> date:
    """year-month 月内最接近 day 的日期：day 超出当月天数时收敛到月末
    （1/31 注册的锚在 2 月收敛为 2/28 或 2/29）。"""
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


def _add_months(year: int, month: int, delta: int) -> tuple:
    """(year, month) 平移 delta 个月，返回 (year, month)。"""
    total = (year * 12 + (month - 1)) + delta
    return total // 12, total % 12 + 1


def _period_start(anchor: date, today: date) -> date:
    """
    以 anchor（注册日）为锚的周年月周期中，today 所属周期的起始日：
    本月锚日 <= today 则本月锚日，否则上月锚日（锚日按目标月天数收敛）。
    注册 7/24：today 8/23 → 7/24；today 8/24 → 8/24。
    """
    cand = _clamp_day(today.year, today.month, anchor.day)
    if cand <= today:
        return cand
    py, pm = _add_months(today.year, today.month, -1)
    return _clamp_day(py, pm, anchor.day)


def _next_reset(anchor: date, period_start: date) -> date:
    """period_start 所属周期的下一周年日（即 reset_date）。"""
    ny, nm = _add_months(period_start.year, period_start.month, 1)
    return _clamp_day(ny, nm, anchor.day)


def _user_snapshot(conn: sqlite3.Connection, user_row: sqlite3.Row) -> dict:
    """
    在给定连接上计算某用户的周期额度快照：
    {quota_limit, quota_used, quota_remaining, reset_date}。

    quota_limit：users.quota_limit override 非空优先，否则 WEB_MONTHLY_QUOTA；
    quota_used：以 created_at 为锚的当前周期（monthly_usage.month = 周期起始日 ISO）用量；
    reset_date：下一周年日 ISO。
    """
    anchor = datetime.fromisoformat(user_row["created_at"]).date()
    period = _period_start(anchor, _now().date())
    row = conn.execute(
        "SELECT used FROM monthly_usage WHERE user_id = ? AND month = ?",
        (user_row["id"], period.isoformat()),
    ).fetchone()
    used = row["used"] if row else 0
    limit = (
        user_row["quota_limit"]
        if user_row["quota_limit"] is not None
        else monthly_quota()
    )
    return {
        "quota_limit": limit,
        "quota_used": used,
        "quota_remaining": max(0, limit - used),
        "reset_date": _next_reset(anchor, period).isoformat(),
    }


def get_quota(user_id: int) -> dict:
    """
    查询用户额度快照：
    {quota_limit, quota_used, quota_remaining, reset_date}
    """
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, created_at, quota_limit FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if row is None:
            raise LookupError(f"用户不存在: {user_id}")
        return _user_snapshot(conn, row)
    finally:
        conn.close()


def consume_quota(user_id: int) -> bool:
    """
    扣减 1 条消息额度（只看当前周期用量，quota_limit override 优先）。
    扣减成功返回 True；周期额度耗尽返回 False（不扣任何计数）。

    单连接 + BEGIN IMMEDIATE：并发下不会出现超扣。
    """
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT id, created_at, quota_limit FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if row is None:
            raise LookupError(f"用户不存在: {user_id}")
        anchor = datetime.fromisoformat(row["created_at"]).date()
        period = _period_start(anchor, _now().date()).isoformat()
        limit = row["quota_limit"] if row["quota_limit"] is not None else monthly_quota()
        conn.execute(
            "INSERT OR IGNORE INTO monthly_usage (user_id, month, used) VALUES (?,?,0)",
            (user_id, period),
        )
        cur = conn.execute(
            "UPDATE monthly_usage SET used = used + 1 "
            "WHERE user_id = ? AND month = ? AND used < ?",
            (user_id, period, limit),
        )
        conn.commit()
        return cur.rowcount > 0
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── 管理员 ──

def _admin_user_item(conn: sqlite3.Connection, row: sqlite3.Row) -> dict:
    """users 行 → 管理员视图对象（quota_limit 为生效值）。"""
    return {
        "username": row["username"],
        "created_at": row["created_at"],
        "is_admin": bool(row["is_admin"]),
        **_user_snapshot(conn, row),
    }


def admin_list_users() -> list:
    """
    全部用户的管理员视图列表（按注册先后升序）：
    [{username, created_at, is_admin, quota_limit(生效值), quota_used,
      quota_remaining, reset_date}]
    """
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM users ORDER BY id"
        ).fetchall()
        return [_admin_user_item(conn, r) for r in rows]
    finally:
        conn.close()


def admin_set_quota_limit(username: str, quota_limit) -> Optional[dict]:
    """
    设置某用户的 quota_limit override（实时生效），返回更新后的管理员视图对象；
    用户不存在返回 None（路由层 404）。

    quota_limit 须为 0-100000 的整数（bool 判非法），否则 ValidationError；
    传 0 合法（该用户周期内额度完全禁用）。
    """
    if (
        not isinstance(quota_limit, int)
        or isinstance(quota_limit, bool)
        or not 0 <= quota_limit <= 100000
    ):
        raise ValidationError("quota_limit 须为 0-100000 的整数")
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE users SET quota_limit = ? WHERE id = ?", (quota_limit, row["id"])
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (row["id"],)
        ).fetchone()
        return _admin_user_item(conn, row)
    finally:
        conn.close()


def admin_user_questions(username: str, limit: int = 50, offset: int = 0) -> Optional[dict]:
    """
    某用户所有对话中 role=user 的消息，按时间倒序：
    {total, items:[{content, created_at, conversation_title}]}。
    用户不存在返回 None（路由层 404）。
    """
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row is None:
            return None
        total = conn.execute(
            "SELECT COUNT(*) AS c FROM messages m "
            "JOIN conversations c ON c.id = m.conversation_id "
            "WHERE c.user_id = ? AND m.role = 'user'",
            (row["id"],),
        ).fetchone()["c"]
        items = conn.execute(
            "SELECT m.content, m.created_at, c.title AS conversation_title "
            "FROM messages m JOIN conversations c ON c.id = m.conversation_id "
            "WHERE c.user_id = ? AND m.role = 'user' "
            "ORDER BY m.created_at DESC, m.id DESC LIMIT ? OFFSET ?",
            (row["id"], limit, offset),
        ).fetchall()
        return {"total": total, "items": [dict(i) for i in items]}
    finally:
        conn.close()


def admin_impersonate(username: str) -> Optional[str]:
    """
    为某用户签发一个普通登录令牌（管理员免密码切换身份用）。
    用户不存在返回 None（路由层 404）。是否管理员由路由层 require_admin 校验，
    本函数不复查（与 admin_set_quota_limit 等同层函数保持一致的职责划分）。
    """
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row is None:
            return None
        token = _issue_token(conn, row["id"])
        conn.commit()
        return token
    finally:
        conn.close()


# ── 对话 CRUD ──

def create_conversation(user_id: int, title: str) -> dict:
    """新建对话，返回 {id, title, created_at, updated_at}。"""
    if not isinstance(title, str) or not title.strip():
        raise ValidationError("对话标题不能为空")
    title = title.strip()[:100]
    now = _iso(_now())
    conv_id = uuid.uuid4().hex
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO conversations (id, user_id, title, created_at, updated_at) "
            "VALUES (?,?,?,?,?)",
            (conv_id, user_id, title, now, now),
        )
        conn.commit()
    finally:
        conn.close()
    return {"id": conv_id, "title": title, "created_at": now, "updated_at": now}


def list_conversations(user_id: int) -> list:
    """
    列出用户全部对话：置顶优先，同组内按 updated_at 降序（同刻按 id 稳定次序）。
    每项含 pinned 布尔字段。
    """
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, title, pinned, created_at, updated_at FROM conversations "
            "WHERE user_id = ? ORDER BY pinned DESC, updated_at DESC, id",
            (user_id,),
        ).fetchall()
        return [{**dict(r), "pinned": bool(r["pinned"])} for r in rows]
    finally:
        conn.close()


def update_conversation(
    user_id: int,
    conversation_id: str,
    title: Optional[str] = None,
    pinned: Optional[bool] = None,
) -> Optional[dict]:
    """
    修改对话标题 / 置顶状态。返回更新后的
    {id, title, pinned, created_at, updated_at}（pinned 为布尔）。

    - title 与 pinned 至少提供其一，否则 ValidationError（路由层 400）；
    - title 去空白后须为 1-60 字符，否则 ValidationError；
    - 对话不存在或不属于该用户返回 None（路由层 404，不暴露越权差异）；
    - updated_at 仅在改 title 时刷新；单独 pin/unpin 不动 updated_at
      （置顶由 pinned 列承担排序权重，不应扰乱同组内的时间排序）。
    """
    if title is None and pinned is None:
        raise ValidationError("title 与 pinned 至少提供其一")
    new_title: Optional[str] = None
    if title is not None:
        if not isinstance(title, str):
            raise ValidationError("对话标题必须是字符串")
        new_title = title.strip()
        if not 1 <= len(new_title) <= 60:
            raise ValidationError("对话标题须为 1-60 个字符")
    if pinned is not None and not isinstance(pinned, bool):
        raise ValidationError("pinned 须为布尔值")

    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        ).fetchone()
        if row is None:
            return None

        sets, params = [], []
        if new_title is not None:
            sets.append("title = ?")
            params.append(new_title)
        if pinned is not None:
            sets.append("pinned = ?")
            params.append(1 if pinned else 0)
        if new_title is not None:
            sets.append("updated_at = ?")
            params.append(_iso(_now()))
        params.append(conversation_id)
        conn.execute(
            f"UPDATE conversations SET {', '.join(sets)} WHERE id = ?", params
        )
        conn.commit()

        row = conn.execute(
            "SELECT id, title, pinned, created_at, updated_at FROM conversations "
            "WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        return {**dict(row), "pinned": bool(row["pinned"])}
    finally:
        conn.close()


def get_conversation(user_id: int, conversation_id: str) -> Optional[dict]:
    """
    取对话详情（含消息列表）。不存在或不属于该用户一律返回 None
    （路由层映射 404，不暴露越权差异）。
    """
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        ).fetchone()
        if row is None:
            return None
        msgs = conn.execute(
            "SELECT role, content, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        return {
            "id": row["id"],
            "title": row["title"],
            "messages": [dict(m) for m in msgs],
        }
    finally:
        conn.close()


def delete_conversation(user_id: int, conversation_id: str) -> bool:
    """删除对话及其消息；不存在或不属于该用户返回 False（路由层 404）。"""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        ).fetchone()
        if row is None:
            return False
        # 先删子表消息再删对话（messages 有外键引用 conversations）
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        return True
    finally:
        conn.close()


def add_message(conversation_id: str, role: str, content: str) -> None:
    """向对话追加一条消息，并刷新对话 updated_at。"""
    now = _iso(_now())
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) "
            "VALUES (?,?,?,?)",
            (conversation_id, role, content, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_history(conversation_id: str, limit: int = 20) -> list:
    """取对话最近消息作为多轮历史（升序返回，最多 limit 条）。"""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    finally:
        conn.close()
