"""
公开网站的用户认证 / 额度 / 对话存储模块（纯标准库实现）。

仅用 sqlite3 + hashlib(pbkdf2_hmac) + secrets + hmac，零第三方依赖，
与 Docker 镜像 python:3.13-slim 兼容，不增加 requirements。

数据库路径解析：
  env WEB_DB_PATH 显式设置优先；缺省为 ${DATA_DIR:-data}/webapp.db，
  与 main.py / agent.push 等模块的 DATA_DIR 约定一致（Railway 挂卷 /data）。

表结构：
  users          注册用户（用户名唯一，PBKDF2 加盐哈希存密码）
  tokens         会话令牌（Bearer token，30 天过期，惰性清理）
  monthly_usage  月度额度用量（主键 user_id+month，跨月天然重置）
  packs          加油包余额（不过期，可累积）
  conversations  对话（归属用户）
  messages       对话消息（归属对话）

额度语义：
  每月定额 WEB_MONTHLY_QUOTA（默认 100 条消息），每月 1 日按 month 键自然重置；
  加油包每包 WEB_PACK_SIZE（默认 50 条），不过期可累积；
  扣减顺序先月度后包；total_remaining = monthly_remaining + pack_credits。

所有 SQL 全部参数化，用户名/密码在入口做长度与字符校验。
"""

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
    """每月定额消息数（env WEB_MONTHLY_QUOTA，默认 100）。"""
    return _env_int("WEB_MONTHLY_QUOTA", 100)


def pack_size() -> int:
    """加油包每包消息数（env WEB_PACK_SIZE，默认 50）。"""
    return _env_int("WEB_PACK_SIZE", 50)


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
    created_at    TEXT NOT NULL
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
CREATE TABLE IF NOT EXISTS packs (
    user_id INTEGER PRIMARY KEY REFERENCES users(id),
    credits INTEGER NOT NULL DEFAULT 0
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
CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, id);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """
    老库结构迁移（在 _SCHEMA 之后执行）。

    conversations.pinned 为后加列：新库已由 _SCHEMA 建好带列结构；
    Railway 上已存在的老库（CREATE TABLE IF NOT EXISTS 不会改已有表）
    靠这里的 ALTER TABLE 补列。列已存在时 SQLite 抛 OperationalError，
    捕获即幂等，重复执行/新旧库混跑都不报错。
    """
    try:
        conn.execute(
            "ALTER TABLE conversations ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0"
        )
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

def register(username: str, password: str) -> str:
    """
    注册用户并直接签发令牌。返回 token 字符串。

    - 用户名已存在 → UserExistsError（路由层映射 409）
    - 输入非法 → ValidationError（路由层映射 400）
    """
    username = validate_username(username)
    password = validate_password(password)
    salt = secrets.token_hex(16)
    pw_hash = _hash_password(password, salt)
    now = _iso(_now())
    conn = _connect()
    try:
        try:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?,?,?,?)",
                (username, pw_hash, salt, now),
            )
        except sqlite3.IntegrityError:
            raise UserExistsError(f"用户名已存在: {username}")
        user_id = cur.lastrowid
        token = _issue_token(conn, user_id)
        conn.commit()
        return token
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
    解析 Bearer token → {"id": user_id, "username": ...}；无效或过期返回 None。
    过期令牌顺手删除（惰性清理）。
    """
    if not token or not isinstance(token, str):
        return None
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT t.user_id, t.expires_at, u.username "
            "FROM tokens t JOIN users u ON u.id = t.user_id WHERE t.token = ?",
            (token,),
        ).fetchone()
        if row is None:
            return None
        if row["expires_at"] < _iso(_now()):
            conn.execute("DELETE FROM tokens WHERE token = ?", (token,))
            conn.commit()
            return None
        return {"id": row["user_id"], "username": row["username"]}
    finally:
        conn.close()


# ── 额度 ──

def _month_key(d: date) -> str:
    return d.strftime("%Y-%m")


def _reset_date(d: date) -> str:
    """次月 1 日的 ISO 日期。"""
    if d.month == 12:
        return date(d.year + 1, 1, 1).isoformat()
    return date(d.year, d.month + 1, 1).isoformat()


def get_quota(user_id: int) -> dict:
    """
    查询用户额度快照：
    {monthly_quota, monthly_used, monthly_remaining, pack_credits,
     total_remaining, reset_date}
    """
    quota = monthly_quota()
    month = _month_key(_now().date())
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT used FROM monthly_usage WHERE user_id = ? AND month = ?",
            (user_id, month),
        ).fetchone()
        used = row["used"] if row else 0
        row = conn.execute(
            "SELECT credits FROM packs WHERE user_id = ?", (user_id,)
        ).fetchone()
        credits = row["credits"] if row else 0
    finally:
        conn.close()
    monthly_remaining = max(0, quota - used)
    return {
        "monthly_quota": quota,
        "monthly_used": used,
        "monthly_remaining": monthly_remaining,
        "pack_credits": credits,
        "total_remaining": monthly_remaining + credits,
        "reset_date": _reset_date(_now().date()),
    }


def consume_quota(user_id: int) -> bool:
    """
    扣减 1 条消息额度：先扣月度，月度用尽再扣加油包。
    扣减成功返回 True；两者皆空返回 False（不扣任何计数）。

    单连接 + BEGIN IMMEDIATE：并发下不会出现超扣。
    """
    quota = monthly_quota()
    month = _month_key(_now().date())
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT OR IGNORE INTO monthly_usage (user_id, month, used) VALUES (?,?,0)",
            (user_id, month),
        )
        conn.execute(
            "INSERT OR IGNORE INTO packs (user_id, credits) VALUES (?,0)", (user_id,)
        )
        cur = conn.execute(
            "UPDATE monthly_usage SET used = used + 1 "
            "WHERE user_id = ? AND month = ? AND used < ?",
            (user_id, month, quota),
        )
        if cur.rowcount == 0:
            cur = conn.execute(
                "UPDATE packs SET credits = credits - 1 "
                "WHERE user_id = ? AND credits > 0",
                (user_id,),
            )
        conn.commit()
        return cur.rowcount > 0
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def topup(user_id: int, pack_count: int = 1) -> dict:
    """
    加油包充值（当前为 mock 支付直接到账）。
    # TODO(支付): 接入真实支付渠道后，本函数应改为支付回调确认后调用，
    #   pack_count 由支付订单的购买数量决定，并落支付流水表防重放。
    """
    if not isinstance(pack_count, int) or isinstance(pack_count, bool) or pack_count < 1:
        raise ValidationError("pack_count 须为正整数")
    if pack_count > 100:
        raise ValidationError("单次充值包数不能超过 100")
    credits = pack_count * pack_size()
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO packs (user_id, credits) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET credits = credits + excluded.credits",
            (user_id, credits),
        )
        conn.commit()
    finally:
        conn.close()
    return get_quota(user_id)


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
