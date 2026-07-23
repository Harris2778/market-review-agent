"""
公开网站后端测试（agent/webauth.py + main.py 的 /api/* 端点）。

覆盖范围：
1. webauth 单元层：注册/登录/重复注册/错误密码/token 过期/登出、
   周年月额度（默认 30、跨周期重置、月末收敛、override 优先）、
   注册防多号（device_limit / ip_limit）、管理员视图（列表/改额度/提问记录）、
   对话 CRUD 与越权隔离
2. 端点层（FastAPI TestClient）：
   - POST /api/auth/register → 200 / 409（重名/device_limit）/ 429（ip_limit）/ 400
   - POST /api/auth/login → 200 / 401
   - POST /api/auth/logout → token 失效
   - GET /api/me → 新字段形状（username/is_admin/quota_*）无加油包字段
   - 对话 CRUD：列表降序、详情带消息、越权 404、删除 200/404
   - POST /api/topup → 410（加油包已下线）
   - POST /api/chat → SSE 流式（mock agent）、落库、成功才扣额度、
     额度耗尽 429、流失败不扣额度
   - /api/admin/* → 非管理员 403、未登录 401、列表/改额度/提问记录
   - GET / → web/index.html 存在时 200，缺失时 503 占位
   - GET /api/root-info → 原根路由健康 JSON

测试通过 monkeypatch WEB_DB_PATH 指向 tmp_path 临时库，绝不触达真实
data/webapp.db；/api/chat 的 agent 全部 mock，不依赖 LLM 与 .env。
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# conftest 会设置 AGENT_API_KEY 测试假值；此处 setdefault 仅作兜底
os.environ.setdefault("AGENT_API_KEY", "test-fake-agent-key-for-pytest")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main  # noqa: E402
from agent import webauth  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


# ── 共享 fixture ──

@pytest.fixture()
def db(tmp_path, monkeypatch):
    """每个测试用例一个独立的临时 DB。"""
    monkeypatch.setenv("WEB_DB_PATH", str(tmp_path / "webapp_test.db"))
    return tmp_path / "webapp_test.db"


@pytest.fixture()
def client(db):
    return TestClient(main.app)


def _register(client, username="testuser", password="secret123"):
    return client.post("/api/auth/register", json={"username": username, "password": password})


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _tick_clock(monkeypatch):
    """
    把 webauth._now 换成逐次调用 +1 秒的假时钟。
    updated_at 只到毫秒精度，快速连续建对话会打平，排序退化为按 id（uuid 随机序），
    凡断言默认 updated_at 降序的测试都用它消除时序抖动。
    """
    real_now = webauth._now
    state = {"n": 0}

    def fake_now():
        state["n"] += 1
        return real_now() + timedelta(seconds=state["n"])

    monkeypatch.setattr(webauth, "_now", fake_now)


class _FakeAgent:
    """模拟流式 agent：process_message(stream=True) 返回异步迭代器。"""

    def __init__(self, chunks=("你好", "，世界"), fail=False):
        self.chunks = chunks
        self.fail = fail
        self.calls = []

    async def process_message(self, message, stream=False, history=None):
        self.calls.append({"message": message, "stream": stream, "history": history})

        async def _gen():
            if self.fail:
                yield self.chunks[0]
                raise RuntimeError("上游模型炸了")
            for c in self.chunks:
                yield c

        return _gen()


# ════════════════════════════════════════════════════════════
# 1) webauth 单元层
# ════════════════════════════════════════════════════════════

class TestWebauthUnit:
    def test_register_and_login(self, db):
        token = webauth.register("alice", "password1")
        assert isinstance(token, str) and token
        user = webauth.resolve_token(token)
        assert user["username"] == "alice"
        token2 = webauth.login("alice", "password1")
        assert webauth.resolve_token(token2)["id"] == user["id"]

    def test_register_duplicate_raises(self, db):
        webauth.register("alice", "password1")
        with pytest.raises(webauth.UserExistsError):
            webauth.register("alice", "password2")

    @pytest.mark.parametrize("bad_name", ["ab", "a" * 33, "has space", "bad!name", "", 123])
    def test_register_invalid_username(self, db, bad_name):
        with pytest.raises(webauth.ValidationError):
            webauth.register(bad_name, "password1")

    @pytest.mark.parametrize("name", ["abc", "user_name9", "张三丰", "用户_01"])
    def test_register_valid_usernames(self, db, name):
        assert webauth.resolve_token(webauth.register(name, "password1"))["username"] == name

    def test_register_short_password(self, db):
        with pytest.raises(webauth.ValidationError):
            webauth.register("alice", "12345")

    def test_login_wrong_password(self, db):
        webauth.register("alice", "password1")
        with pytest.raises(webauth.AuthError):
            webauth.login("alice", "password2")

    def test_login_unknown_user(self, db):
        with pytest.raises(webauth.AuthError):
            webauth.login("nobody", "password1")

    def test_logout_invalidates_token(self, db):
        token = webauth.register("alice", "password1")
        webauth.logout(token)
        assert webauth.resolve_token(token) is None

    def test_token_expired(self, db, monkeypatch):
        token = webauth.register("alice", "password1")
        real_now = webauth._now
        # 时间快进到 31 天后（token 有效期 30 天）
        monkeypatch.setattr(
            webauth, "_now", lambda: real_now() + timedelta(days=31)
        )
        assert webauth.resolve_token(token) is None

    def test_register_admin_flag(self, db):
        """注册时用户名命中 ADMIN_USERNAMES（默认 'yoozo'）自动置 is_admin。"""
        admin = webauth.resolve_token(webauth.register("yoozo", "password1"))
        assert admin["is_admin"] is True
        normal = webauth.resolve_token(webauth.register("alice", "password1"))
        assert normal["is_admin"] is False

    def test_mark_admins_idempotent(self, db, monkeypatch):
        """mark_admins 把存量用户幂等补齐 is_admin=1，重复执行安全。"""
        monkeypatch.setenv("ADMIN_USERNAMES", "alice,carol")
        webauth.register("alice", "password1")  # 注册时即命中
        bob = webauth.resolve_token(webauth.register("bob", "password1"))
        assert bob["is_admin"] is False
        monkeypatch.setenv("ADMIN_USERNAMES", "alice,bob")
        assert webauth.mark_admins() == 1  # alice 已是 admin，仅补 bob
        assert webauth.mark_admins() == 0  # 幂等
        token = webauth.login("bob", "password1")
        assert webauth.resolve_token(token)["is_admin"] is True

    def test_quota_defaults(self, db):
        user = webauth.resolve_token(webauth.register("alice", "password1"))
        quota = webauth.get_quota(user["id"])
        assert quota["quota_limit"] == 30  # 新默认定额
        assert quota["quota_used"] == 0
        assert quota["quota_remaining"] == 30
        # reset_date 为注册日下一周年日（今天注册 → 下月同日，月末收敛）
        today = datetime.now().date()
        period = webauth._period_start(today, today)
        assert period == today
        assert quota["reset_date"] == webauth._next_reset(today, today).isoformat()
        # 加油包字段彻底移除
        assert "pack_credits" not in quota
        assert "total_remaining" not in quota
        assert not hasattr(webauth, "topup")
        assert not hasattr(webauth, "pack_size")

    def test_consume_up_to_limit_then_exhausted(self, db, monkeypatch):
        """扣减只看周期内用量：用满 quota 后再扣返回 False，且不扣任何计数。"""
        monkeypatch.setenv("WEB_MONTHLY_QUOTA", "2")
        user = webauth.resolve_token(webauth.register("alice", "password1"))
        assert webauth.consume_quota(user["id"]) is True
        assert webauth.consume_quota(user["id"]) is True
        q = webauth.get_quota(user["id"])
        assert (q["quota_used"], q["quota_remaining"]) == (2, 0)
        assert webauth.consume_quota(user["id"]) is False
        assert webauth.get_quota(user["id"])["quota_used"] == 2

    def test_anniversary_reset_cross_period(self, db, monkeypatch):
        """周年月重置：注册 7/24，mock 时间 +31 天（8/24）即跨周期，用量归零。"""
        monkeypatch.setattr(webauth, "_now", lambda: datetime(2025, 7, 24, 10, 0, 0))
        user = webauth.resolve_token(webauth.register("alice", "password1"))
        q = webauth.get_quota(user["id"])
        assert q["reset_date"] == "2025-08-24"

        assert webauth.consume_quota(user["id"]) is True
        assert webauth.get_quota(user["id"])["quota_used"] == 1

        # 8/23 23:59 仍未重置
        monkeypatch.setattr(webauth, "_now", lambda: datetime(2025, 8, 23, 23, 59, 0))
        assert webauth.get_quota(user["id"])["quota_used"] == 1

        # 注册日 +31 天 = 8/24，跨入新周期：用量归零，reset 推到 9/24
        monkeypatch.setattr(webauth, "_now", lambda: datetime(2025, 8, 24, 0, 0, 1))
        q = webauth.get_quota(user["id"])
        assert (q["quota_used"], q["quota_remaining"]) == (0, 30)
        assert q["reset_date"] == "2025-09-24"
        assert webauth.consume_quota(user["id"]) is True
        assert webauth.get_quota(user["id"])["quota_used"] == 1

    def test_anniversary_month_end_clamp(self, db, monkeypatch):
        """月末边界：1/31 注册 → 2/28 重置（平年）；3/1 时周期锚收敛到 2/28。"""
        monkeypatch.setattr(webauth, "_now", lambda: datetime(2025, 1, 31, 10, 0, 0))
        user = webauth.resolve_token(webauth.register("alice", "password1"))
        assert webauth.get_quota(user["id"])["reset_date"] == "2025-02-28"
        assert webauth.consume_quota(user["id"]) is True

        # 2/27 仍是旧周期；2/28 重置；3/1 处于 2/28 起的新周期，3/31 再重置
        monkeypatch.setattr(webauth, "_now", lambda: datetime(2025, 2, 27, 12, 0, 0))
        assert webauth.get_quota(user["id"])["quota_used"] == 1
        monkeypatch.setattr(webauth, "_now", lambda: datetime(2025, 2, 28, 0, 0, 0))
        assert webauth.get_quota(user["id"])["quota_used"] == 0
        monkeypatch.setattr(webauth, "_now", lambda: datetime(2025, 3, 1, 0, 0, 0))
        q = webauth.get_quota(user["id"])
        assert q["quota_used"] == 0
        assert q["reset_date"] == "2025-03-31"

        # 闰年：2024/1/31 注册 → 2024/2/29 重置
        monkeypatch.setattr(webauth, "_now", lambda: datetime(2024, 1, 31, 10, 0, 0))
        leap = webauth.resolve_token(webauth.register("leapuser", "password1"))
        assert webauth.get_quota(leap["id"])["reset_date"] == "2024-02-29"

    def test_quota_override_takes_priority(self, db):
        """quota_limit override 非空优先于默认 30，改完实时生效；0 = 完全禁用。"""
        user = webauth.resolve_token(webauth.register("alice", "password1"))
        assert webauth.get_quota(user["id"])["quota_limit"] == 30

        webauth.admin_set_quota_limit("alice", 2)
        q = webauth.get_quota(user["id"])
        assert q["quota_limit"] == 2
        assert webauth.consume_quota(user["id"]) is True
        assert webauth.consume_quota(user["id"]) is True
        assert webauth.consume_quota(user["id"]) is False  # 按 override=2 计算

        webauth.admin_set_quota_limit("alice", 0)
        assert webauth.consume_quota(user["id"]) is False
        assert webauth.get_quota(user["id"])["quota_remaining"] == 0

    def test_admin_set_quota_limit_validation(self, db):
        webauth.register("alice", "password1")
        for bad in (-1, 100001, True, "10", None, 3.5):
            with pytest.raises(webauth.ValidationError):
                webauth.admin_set_quota_limit("alice", bad)
        assert webauth.admin_set_quota_limit("ghost", 10) is None

    def test_admin_list_users(self, db):
        webauth.register("yoozo", "password1")
        webauth.register("alice", "password1")
        webauth.consume_quota(webauth.resolve_token(webauth.login("alice", "password1"))["id"])
        users = webauth.admin_list_users()
        assert [u["username"] for u in users] == ["yoozo", "alice"]
        assert users[0]["is_admin"] is True
        assert users[1]["is_admin"] is False
        assert users[1]["quota_used"] == 1
        assert users[1]["quota_limit"] == 30
        for key in ("username", "created_at", "is_admin", "quota_limit",
                    "quota_used", "quota_remaining", "reset_date"):
            assert key in users[0]

    def test_admin_user_questions(self, db):
        alice = webauth.resolve_token(webauth.register("alice", "password1"))
        c1 = webauth.create_conversation(alice["id"], "对话一")
        c2 = webauth.create_conversation(alice["id"], "对话二")
        webauth.add_message(c1["id"], "user", "问题1")
        webauth.add_message(c1["id"], "assistant", "回答1")
        webauth.add_message(c2["id"], "user", "问题2")

        result = webauth.admin_user_questions("alice")
        assert result["total"] == 2
        assert {i["content"] for i in result["items"]} == {"问题1", "问题2"}
        assert all("conversation_title" in i and "created_at" in i for i in result["items"])
        # 倒序 + 分页
        assert result["items"][0]["content"] == "问题2"
        page = webauth.admin_user_questions("alice", limit=1, offset=1)
        assert page["total"] == 2
        assert len(page["items"]) == 1
        assert page["items"][0]["content"] == "问题1"

        assert webauth.admin_user_questions("ghost") is None

    def test_device_limit(self, db, monkeypatch):
        """同一 device_id 注册满 DEVICE_MAX_ACCOUNTS（默认 2）后再注册抛 DeviceLimitError。"""
        monkeypatch.setenv("DEVICE_MAX_ACCOUNTS", "2")
        webauth.register("alice", "password1", device_id="dev-1")
        webauth.register("bob", "password1", device_id="dev-1")
        with pytest.raises(webauth.DeviceLimitError):
            webauth.register("carol", "password1", device_id="dev-1")
        # 别的设备 / 不带 device_id 不受影响
        webauth.register("dave", "password1", device_id="dev-2")
        webauth.register("erin", "password1")

    def test_device_id_validation(self, db):
        with pytest.raises(webauth.ValidationError):
            webauth.register("alice", "password1", device_id=123)
        with pytest.raises(webauth.ValidationError):
            webauth.register("alice", "password1", device_id="x" * 129)
        # 空白 device_id 按未提供处理
        assert webauth.resolve_token(
            webauth.register("alice", "password1", device_id="   ")
        )["username"] == "alice"

    def test_ip_limit(self, db, monkeypatch):
        """同 IP 24h 内注册满 IP_MAX_REGISTER_PER_DAY（默认 5）后再注册抛 IpLimitError。"""
        monkeypatch.setenv("IP_MAX_REGISTER_PER_DAY", "5")
        for i in range(5):
            webauth.register(f"user{i:02d}", "password1", register_ip="1.2.3.4")
        with pytest.raises(webauth.IpLimitError):
            webauth.register("user99", "password1", register_ip="1.2.3.4")
        # 别的 IP / 24h 前的注册不占名额
        webauth.register("other1", "password1", register_ip="5.6.7.8")
        monkeypatch.setattr(
            webauth, "_now", lambda: datetime.now() + timedelta(hours=25)
        )
        webauth.register("other2", "password1", register_ip="1.2.3.4")

    def test_conversation_crud_and_isolation(self, db):
        alice = webauth.resolve_token(webauth.register("alice", "password1"))
        bob = webauth.resolve_token(webauth.register("bob", "password1"))

        conv = webauth.create_conversation(alice["id"], "我的复盘")
        assert conv["title"] == "我的复盘"
        webauth.add_message(conv["id"], "user", "今日复盘")
        webauth.add_message(conv["id"], "assistant", "这是回复")

        detail = webauth.get_conversation(alice["id"], conv["id"])
        assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
        assert detail["messages"][0]["content"] == "今日复盘"

        # 越权：bob 看不到、删不掉 alice 的对话
        assert webauth.get_conversation(bob["id"], conv["id"]) is None
        assert webauth.delete_conversation(bob["id"], conv["id"]) is False

        assert webauth.delete_conversation(alice["id"], conv["id"]) is True
        assert webauth.get_conversation(alice["id"], conv["id"]) is None

    def test_get_history_recent_first_limited(self, db):
        alice = webauth.resolve_token(webauth.register("alice", "password1"))
        conv = webauth.create_conversation(alice["id"], "t")
        for i in range(25):
            webauth.add_message(conv["id"], "user", f"m{i}")
        history = webauth.get_history(conv["id"], limit=20)
        assert len(history) == 20
        assert history[0]["content"] == "m5"   # 升序返回最近 20 条
        assert history[-1]["content"] == "m24"

    def test_list_conversations_pinned_first(self, db, monkeypatch):
        alice = webauth.resolve_token(webauth.register("alice", "password1"))
        # 假时钟逐次 +1s：同毫秒建两条对话会让 updated_at 打平，排序退化为按 id（uuid 随机序）
        _tick_clock(monkeypatch)
        c1 = webauth.create_conversation(alice["id"], "旧")
        c2 = webauth.create_conversation(alice["id"], "新")
        convs = webauth.list_conversations(alice["id"])
        assert [c["id"] for c in convs] == [c2["id"], c1["id"]]  # 默认按 updated_at 降序
        assert all(c["pinned"] is False for c in convs)

        webauth.update_conversation(alice["id"], c1["id"], pinned=True)
        convs = webauth.list_conversations(alice["id"])
        assert convs[0]["id"] == c1["id"] and convs[0]["pinned"] is True

        webauth.update_conversation(alice["id"], c1["id"], pinned=False)
        convs = webauth.list_conversations(alice["id"])
        assert [c["id"] for c in convs] == [c2["id"], c1["id"]]

    def test_update_conversation_rename(self, db, monkeypatch):
        alice = webauth.resolve_token(webauth.register("alice", "password1"))
        conv = webauth.create_conversation(alice["id"], "原名")
        later = webauth._now() + timedelta(seconds=5)
        monkeypatch.setattr(webauth, "_now", lambda: later)
        result = webauth.update_conversation(alice["id"], conv["id"], title="  新名  ")
        assert result["title"] == "新名"  # 去空白
        assert result["pinned"] is False
        assert result["created_at"] == conv["created_at"]
        assert result["updated_at"] > conv["updated_at"]  # 改名刷新 updated_at

    def test_update_conversation_pin_keeps_updated_at(self, db, monkeypatch):
        alice = webauth.resolve_token(webauth.register("alice", "password1"))
        conv = webauth.create_conversation(alice["id"], "t")
        later = webauth._now() + timedelta(seconds=5)
        monkeypatch.setattr(webauth, "_now", lambda: later)
        result = webauth.update_conversation(alice["id"], conv["id"], pinned=True)
        assert result["pinned"] is True
        assert result["updated_at"] == conv["updated_at"]  # pin 不动 updated_at

    def test_update_conversation_validation(self, db):
        alice = webauth.resolve_token(webauth.register("alice", "password1"))
        conv = webauth.create_conversation(alice["id"], "t")
        # 两者都不给
        with pytest.raises(webauth.ValidationError):
            webauth.update_conversation(alice["id"], conv["id"])
        # 空名 / 纯空白 / 超长 / 非字符串
        for bad in ("", "   ", "x" * 61, 123):
            with pytest.raises(webauth.ValidationError):
                webauth.update_conversation(alice["id"], conv["id"], title=bad)
        # pinned 非布尔
        with pytest.raises(webauth.ValidationError):
            webauth.update_conversation(alice["id"], conv["id"], pinned=1)

    def test_update_conversation_cross_user_returns_none(self, db):
        alice = webauth.resolve_token(webauth.register("alice", "password1"))
        bob = webauth.resolve_token(webauth.register("bob", "password1"))
        conv = webauth.create_conversation(alice["id"], "私有")
        assert webauth.update_conversation(bob["id"], conv["id"], pinned=True) is None
        assert webauth.update_conversation(bob["id"], conv["id"], title="劫持") is None
        assert webauth.update_conversation(alice["id"], "nonexistent", pinned=True) is None
        # alice 的对话未被改动
        assert webauth.list_conversations(alice["id"])[0]["pinned"] is False


def test_migrate_old_db_without_pinned(tmp_path, monkeypatch):
    """
    老库迁移兼容：手工建一个无 pinned / is_admin / quota_limit / register_ip
    列的老结构库（模拟 Railway 线上存量库），首次 _connect 时
    ensure_schema(_SCHEMA)+_migrate 应无报错补齐全部后加列，
    且存量数据 pinned 默认为 False、is_admin 默认为 False、额度按默认 30；
    改名/置顶功能正常；重复执行幂等。
    """
    import sqlite3 as raw_sqlite3

    db_path = tmp_path / "old_webapp.db"
    conn = raw_sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE monthly_usage (
            user_id INTEGER NOT NULL REFERENCES users(id),
            month TEXT NOT NULL,
            used INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, month)
        );
        CREATE TABLE packs (
            user_id INTEGER PRIMARY KEY REFERENCES users(id),
            credits INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?,?,?,?)",
        ("legacy", "x", "y", "2024-01-01T00:00:00.000"),
    )
    conn.execute(
        "INSERT INTO conversations (id, user_id, title, created_at, updated_at) "
        "VALUES (?,?,?,?,?)",
        ("c1", 1, "老对话", "2024-01-01T00:00:00.000", "2024-01-01T00:00:00.000"),
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("WEB_DB_PATH", str(db_path))
    # 第一次访问触发迁移：老数据 pinned 默认为 False
    convs = webauth.list_conversations(1)
    assert len(convs) == 1
    assert convs[0]["pinned"] is False
    assert convs[0]["title"] == "老对话"
    # 置顶 / 改名在新列上正常工作
    result = webauth.update_conversation(1, "c1", pinned=True)
    assert result["pinned"] is True
    assert result["updated_at"] == "2024-01-01T00:00:00.000"  # pin 不动 updated_at
    # 迁移幂等：再次连接（列已存在，ALTER 抛错被吞）不报错
    assert webauth.list_conversations(1)[0]["pinned"] is True

    # users 后加列迁移：存量用户按默认额度 30、非管理员；遗留 packs 表无害忽略
    item = webauth.admin_list_users()[0]
    assert item["username"] == "legacy"
    assert item["is_admin"] is False
    assert item["quota_limit"] == 30
    quota = webauth.get_quota(1)
    assert quota["quota_used"] == 0 and quota["quota_remaining"] == 30
    assert webauth.consume_quota(1) is True


# ════════════════════════════════════════════════════════════
# 2) 认证端点
# ════════════════════════════════════════════════════════════

class TestAuthEndpoints:
    def test_register_200(self, client):
        resp = _register(client)
        assert resp.status_code == 200
        data = resp.json()
        assert data["user"]["username"] == "testuser"
        assert isinstance(data["token"], str) and data["token"]

    def test_register_duplicate_409(self, client):
        assert _register(client).status_code == 200
        resp = _register(client)
        assert resp.status_code == 409

    def test_register_validation_400(self, client):
        assert _register(client, username="ab").status_code == 400
        assert _register(client, password="12345").status_code == 400

    def test_login_200(self, client):
        _register(client)
        resp = client.post(
            "/api/auth/login", json={"username": "testuser", "password": "secret123"}
        )
        assert resp.status_code == 200
        assert resp.json()["user"]["username"] == "testuser"

    def test_login_wrong_password_401(self, client):
        _register(client)
        resp = client.post(
            "/api/auth/login", json={"username": "testuser", "password": "wrong-pass"}
        )
        assert resp.status_code == 401

    def test_login_unknown_user_401(self, client):
        resp = client.post(
            "/api/auth/login", json={"username": "ghost", "password": "secret123"}
        )
        assert resp.status_code == 401

    def test_logout_then_me_401(self, client):
        token = _register(client).json()["token"]
        assert client.post("/api/auth/logout", headers=_auth(token)).status_code == 200
        assert client.get("/api/me", headers=_auth(token)).status_code == 401

    def test_me_requires_auth(self, client):
        assert client.get("/api/me").status_code == 401
        assert client.get("/api/me", headers=_auth("bad-token")).status_code == 401

    def test_me_fields(self, client):
        token = _register(client).json()["token"]
        resp = client.get("/api/me", headers=_auth(token))
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "testuser"
        assert data["is_admin"] is False
        assert data["quota_limit"] == 30
        assert data["quota_used"] == 0
        assert data["quota_remaining"] == 30
        assert "reset_date" in data
        assert data["quota_remaining"] == data["quota_limit"] - data["quota_used"]
        # 加油包字段彻底移除
        for key in ("pack_credits", "total_remaining", "monthly_quota",
                    "monthly_used", "monthly_remaining"):
            assert key not in data

    def test_me_admin_flag(self, client):
        token = _register(client, username="yoozo").json()["token"]
        data = client.get("/api/me", headers=_auth(token)).json()
        assert data["is_admin"] is True


# ════════════════════════════════════════════════════════════
# 2.5) 注册防多号端点（device_limit / ip_limit）
# ════════════════════════════════════════════════════════════

class TestRegisterLimits:
    def test_device_limit_409(self, client, monkeypatch):
        monkeypatch.setenv("DEVICE_MAX_ACCOUNTS", "2")
        payload = {"password": "secret123", "device_id": "dev-1"}
        assert client.post("/api/auth/register", json={"username": "alice", **payload}).status_code == 200
        assert client.post("/api/auth/register", json={"username": "bob", **payload}).status_code == 200
        resp = client.post("/api/auth/register", json={"username": "carol", **payload})
        assert resp.status_code == 409
        assert resp.json() == {"error": "device_limit"}
        # 不带 device_id / 换设备不受影响
        assert client.post("/api/auth/register", json={"username": "dave", "password": "secret123"}).status_code == 200
        assert client.post(
            "/api/auth/register",
            json={"username": "erin", "password": "secret123", "device_id": "dev-2"},
        ).status_code == 200

    def test_ip_limit_429(self, client, monkeypatch):
        monkeypatch.setenv("IP_MAX_REGISTER_PER_DAY", "5")
        headers = {"X-Forwarded-For": "9.9.9.9, 10.0.0.1"}  # 取第一个
        for i in range(5):
            resp = client.post(
                "/api/auth/register",
                json={"username": f"user{i:02d}", "password": "secret123"},
                headers=headers,
            )
            assert resp.status_code == 200
        resp = client.post(
            "/api/auth/register",
            json={"username": "user99", "password": "secret123"},
            headers=headers,
        )
        assert resp.status_code == 429
        assert resp.json() == {"error": "ip_limit"}
        # 换个来源 IP 立即可注册（X-Forwarded-For 取第一个地址）
        resp = client.post(
            "/api/auth/register",
            json={"username": "user98", "password": "secret123"},
            headers={"X-Forwarded-For": "8.8.8.8"},
        )
        assert resp.status_code == 200


# ════════════════════════════════════════════════════════════
# 3) 对话端点
# ════════════════════════════════════════════════════════════

class TestConversationEndpoints:
    def test_create_and_list_desc(self, client):
        token = _register(client).json()["token"]
        c1 = client.post(
            "/api/conversations", json={"title": "第一条"}, headers=_auth(token)
        ).json()
        c2 = client.post(
            "/api/conversations", json={"title": "第二条"}, headers=_auth(token)
        ).json()
        assert c1["id"] != c2["id"]
        for key in ("id", "title", "created_at", "updated_at"):
            assert key in c1

        resp = client.get("/api/conversations", headers=_auth(token))
        assert resp.status_code == 200
        convs = resp.json()
        assert len(convs) == 2
        assert convs[0]["updated_at"] >= convs[1]["updated_at"]  # 降序

        # 向旧对话发消息（落库刷新 updated_at）后它应排到最前
        web_auth = webauth
        user = web_auth.resolve_token(token)
        web_auth.add_message(c1["id"], "user", " bump ")
        convs = client.get("/api/conversations", headers=_auth(token)).json()
        assert convs[0]["id"] == c1["id"]

    def test_get_detail_with_messages(self, client):
        token = _register(client).json()["token"]
        conv = client.post(
            "/api/conversations", json={"title": "t"}, headers=_auth(token)
        ).json()
        user = webauth.resolve_token(token)
        webauth.add_message(conv["id"], "user", "问题")
        webauth.add_message(conv["id"], "assistant", "回答")
        resp = client.get(f"/api/conversations/{conv['id']}", headers=_auth(token))
        assert resp.status_code == 200
        data = resp.json()
        assert [m["role"] for m in data["messages"]] == ["user", "assistant"]
        assert all("created_at" in m for m in data["messages"])

    def test_cross_user_404(self, client):
        token_a = _register(client, "alice").json()["token"]
        token_b = _register(client, "bob").json()["token"]
        conv = client.post(
            "/api/conversations", json={"title": "私有"}, headers=_auth(token_a)
        ).json()
        assert client.get(
            f"/api/conversations/{conv['id']}", headers=_auth(token_b)
        ).status_code == 404
        assert client.delete(
            f"/api/conversations/{conv['id']}", headers=_auth(token_b)
        ).status_code == 404

    def test_delete_200_then_404(self, client):
        token = _register(client).json()["token"]
        conv = client.post(
            "/api/conversations", json={"title": "t"}, headers=_auth(token)
        ).json()
        resp = client.delete(f"/api/conversations/{conv['id']}", headers=_auth(token))
        assert resp.status_code == 200
        assert resp.json() == {}
        assert client.get(
            f"/api/conversations/{conv['id']}", headers=_auth(token)
        ).status_code == 404
        assert client.get("/api/conversations", headers=_auth(token)).json() == []

    def test_endpoints_require_auth(self, client):
        assert client.get("/api/conversations").status_code == 401
        assert client.post("/api/conversations", json={"title": "t"}).status_code == 401
        assert client.get("/api/conversations/whatever").status_code == 401
        assert client.patch("/api/conversations/whatever", json={"pinned": True}).status_code == 401
        assert client.delete("/api/conversations/whatever").status_code == 401


class TestPatchConversationEndpoint:
    def _setup(self, client, username="testuser"):
        token = _register(client, username).json()["token"]
        return token

    def _create(self, client, token, title):
        return client.post(
            "/api/conversations", json={"title": title}, headers=_auth(token)
        ).json()

    def test_pin_sorting_priority(self, client, monkeypatch):
        _tick_clock(monkeypatch)
        token = self._setup(client)
        c1 = self._create(client, token, "旧对话")
        c2 = self._create(client, token, "新对话")
        convs = client.get("/api/conversations", headers=_auth(token)).json()
        assert [c["id"] for c in convs] == [c2["id"], c1["id"]]  # 默认 updated_at 降序
        assert all(c["pinned"] is False for c in convs)

        resp = client.patch(
            f"/api/conversations/{c1['id']}", json={"pinned": True}, headers=_auth(token)
        )
        assert resp.status_code == 200
        assert resp.json()["pinned"] is True

        convs = client.get("/api/conversations", headers=_auth(token)).json()
        assert convs[0]["id"] == c1["id"]  # 置顶优先于 updated_at
        assert convs[0]["pinned"] is True

    def test_unpin_restores_order(self, client, monkeypatch):
        _tick_clock(monkeypatch)
        token = self._setup(client)
        c1 = self._create(client, token, "旧对话")
        c2 = self._create(client, token, "新对话")
        client.patch(f"/api/conversations/{c1['id']}", json={"pinned": True}, headers=_auth(token))
        resp = client.patch(
            f"/api/conversations/{c1['id']}", json={"pinned": False}, headers=_auth(token)
        )
        assert resp.status_code == 200
        assert resp.json()["pinned"] is False
        convs = client.get("/api/conversations", headers=_auth(token)).json()
        assert [c["id"] for c in convs] == [c2["id"], c1["id"]]

    def test_rename_success(self, client, monkeypatch):
        token = self._setup(client)
        conv = self._create(client, token, "原名")
        later = datetime.now() + timedelta(seconds=5)
        monkeypatch.setattr(webauth, "_now", lambda: later)
        resp = client.patch(
            f"/api/conversations/{conv['id']}", json={"title": "  新名字  "}, headers=_auth(token)
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "新名字"  # 去空白
        assert data["pinned"] is False
        assert data["created_at"] == conv["created_at"]
        assert data["updated_at"] > conv["updated_at"]  # 改名刷新 updated_at
        for key in ("id", "title", "pinned", "created_at", "updated_at"):
            assert key in data

    def test_rename_400_empty_and_too_long(self, client):
        token = self._setup(client)
        conv = self._create(client, token, "t")
        for payload in (
            {"title": ""},            # 空名
            {"title": "   "},         # 纯空白
            {"title": "x" * 61},      # 超长（上限 60）
            {"title": 123},           # 非字符串
            {},                       # title/pinned 都缺
            {"pinned": "yes"},        # pinned 非布尔
        ):
            resp = client.patch(
                f"/api/conversations/{conv['id']}", json=payload, headers=_auth(token)
            )
            assert resp.status_code == 400, payload
        # 边界：恰好 60 字符合法
        resp = client.patch(
            f"/api/conversations/{conv['id']}", json={"title": "x" * 60}, headers=_auth(token)
        )
        assert resp.status_code == 200

    def test_cross_user_404(self, client):
        token_a = self._setup(client, "alice")
        token_b = self._setup(client, "bob")
        conv = self._create(client, token_a, "私有")
        for payload in ({"pinned": True}, {"title": "劫持"}):
            resp = client.patch(
                f"/api/conversations/{conv['id']}", json=payload, headers=_auth(token_b)
            )
            assert resp.status_code == 404
        # 不存在的对话对 owner 也 404
        resp = client.patch(
            "/api/conversations/nonexistent", json={"pinned": True}, headers=_auth(token_a)
        )
        assert resp.status_code == 404
        # alice 的对话未被改动
        convs = client.get("/api/conversations", headers=_auth(token_a)).json()
        assert convs[0]["pinned"] is False
        assert convs[0]["title"] == "私有"

    def test_pin_does_not_change_updated_at(self, client, monkeypatch):
        token = self._setup(client)
        conv = self._create(client, token, "t")
        later = datetime.now() + timedelta(seconds=5)
        monkeypatch.setattr(webauth, "_now", lambda: later)
        resp = client.patch(
            f"/api/conversations/{conv['id']}", json={"pinned": True}, headers=_auth(token)
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["pinned"] is True
        assert data["updated_at"] == conv["updated_at"]  # pin 不动 updated_at
        # 列表里的 updated_at 同样不变
        convs = client.get("/api/conversations", headers=_auth(token)).json()
        assert convs[0]["updated_at"] == conv["updated_at"]


# ════════════════════════════════════════════════════════════
# 4) 加油包下线与 /api/chat
# ════════════════════════════════════════════════════════════

class TestTopupRemoved:
    def test_topup_returns_410(self, client):
        """加油包功能整体下线：端点保留但一律 410，无需鉴权也 410。"""
        token = _register(client).json()["token"]
        resp = client.post("/api/topup", json={"pack_count": 1}, headers=_auth(token))
        assert resp.status_code == 410
        assert resp.json()["error"] == "topup_removed"
        assert client.post("/api/topup", json={"pack_count": 1}).status_code == 410


class TestWebChat:
    def _setup(self, client):
        token = _register(client).json()["token"]
        conv = client.post(
            "/api/conversations", json={"title": "chat"}, headers=_auth(token)
        ).json()
        return token, conv

    def test_stream_success_persists_and_consumes(self, client):
        token, conv = self._setup(client)
        fake = _FakeAgent(chunks=("今日", "复盘摘要"))
        with patch.object(main, "_agent_loaded", True), \
             patch.object(main, "get_agent", return_value=fake):
            resp = client.post(
                "/api/chat",
                json={"conversation_id": conv["id"], "message": "今日复盘"},
                headers=_auth(token),
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        body = resp.text
        assert "data: [DONE]" in body
        assert "今日" in body and "复盘摘要" in body
        assert "风险提示" in body  # 免责条款
        assert '"role": "assistant"' in body or '"role":"assistant"' in body

        # 落库：user + assistant 各一条
        detail = client.get(f"/api/conversations/{conv['id']}", headers=_auth(token)).json()
        msgs = detail["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[0]["content"] == "今日复盘"
        assert "今日复盘摘要" in msgs[1]["content"]
        assert "风险提示" in msgs[1]["content"]

        # 流正常完成扣 1 次额度
        me = client.get("/api/me", headers=_auth(token)).json()
        assert me["quota_used"] == 1
        assert me["quota_remaining"] == me["quota_limit"] - 1

    def test_stream_failure_no_quota_consumed(self, client):
        token, conv = self._setup(client)
        fake = _FakeAgent(chunks=("半截",), fail=True)
        with patch.object(main, "_agent_loaded", True), \
             patch.object(main, "get_agent", return_value=fake):
            resp = client.post(
                "/api/chat",
                json={"conversation_id": conv["id"], "message": "会失败"},
                headers=_auth(token),
            )
        assert resp.status_code == 200
        assert "stream_error" in resp.text

        # 流失败：不扣额度，助手消息不落库（用户消息仍在）
        me = client.get("/api/me", headers=_auth(token)).json()
        assert me["quota_used"] == 0
        detail = client.get(f"/api/conversations/{conv['id']}", headers=_auth(token)).json()
        assert [m["role"] for m in detail["messages"]] == ["user"]

    def test_quota_exhausted_429(self, client, monkeypatch):
        monkeypatch.setenv("WEB_MONTHLY_QUOTA", "1")
        token, conv = self._setup(client)
        fake = _FakeAgent()
        with patch.object(main, "_agent_loaded", True), \
             patch.object(main, "get_agent", return_value=fake):
            r1 = client.post(
                "/api/chat",
                json={"conversation_id": conv["id"], "message": "第一条"},
                headers=_auth(token),
            )
            assert r1.status_code == 200
            resp = client.post(
                "/api/chat",
                json={"conversation_id": conv["id"], "message": "第二条"},
                headers=_auth(token),
            )
        assert resp.status_code == 429
        data = resp.json()
        assert data["error"] == "quota_exhausted"
        assert data["quota_remaining"] == 0
        assert "reset_date" in data
        # 429 的请求不落库
        detail = client.get(f"/api/conversations/{conv['id']}", headers=_auth(token)).json()
        assert len(detail["messages"]) == 2  # 仅第一轮 user+assistant

    def test_chat_requires_auth(self, client):
        resp = client.post(
            "/api/chat", json={"conversation_id": "x", "message": "hi"}
        )
        assert resp.status_code == 401

    def test_chat_conversation_not_found_404(self, client):
        token = _register(client).json()["token"]
        resp = client.post(
            "/api/chat",
            json={"conversation_id": "nonexistent", "message": "hi"},
            headers=_auth(token),
        )
        assert resp.status_code == 404

    def test_chat_bad_request_400(self, client):
        token, conv = self._setup(client)
        for payload in (
            {"conversation_id": "", "message": "hi"},
            {"conversation_id": conv["id"], "message": ""},
            {"conversation_id": conv["id"]},
        ):
            resp = client.post("/api/chat", json=payload, headers=_auth(token))
            assert resp.status_code == 400

    def test_chat_history_passed_to_agent(self, client):
        """多轮：agent 收到的 history 应包含前轮消息（不含本轮）。"""
        token, conv = self._setup(client)
        fake = _FakeAgent()
        user = webauth.resolve_token(token)
        webauth.add_message(conv["id"], "user", "旧问题")
        webauth.add_message(conv["id"], "assistant", "旧回答")
        with patch.object(main, "_agent_loaded", True), \
             patch.object(main, "get_agent", return_value=fake):
            client.post(
                "/api/chat",
                json={"conversation_id": conv["id"], "message": "新问题"},
                headers=_auth(token),
            )
        history = fake.calls[0]["history"]
        assert [h["content"] for h in history] == ["旧问题", "旧回答"]


# ════════════════════════════════════════════════════════════
# 4.5) 管理员端点 /api/admin/*
# ════════════════════════════════════════════════════════════

class TestAdminEndpoints:
    def _admin_token(self, client):
        return _register(client, username="yoozo").json()["token"]

    def test_non_admin_403(self, client):
        token = _register(client).json()["token"]  # 普通用户
        assert client.get("/api/admin/users", headers=_auth(token)).status_code == 403
        assert client.patch(
            "/api/admin/users/testuser/quota",
            json={"quota_limit": 100}, headers=_auth(token),
        ).status_code == 403
        assert client.get(
            "/api/admin/users/testuser/questions", headers=_auth(token)
        ).status_code == 403

    def test_unauthenticated_401(self, client):
        assert client.get("/api/admin/users").status_code == 401
        assert client.patch(
            "/api/admin/users/x/quota", json={"quota_limit": 1}
        ).status_code == 401
        assert client.get("/api/admin/users/x/questions").status_code == 401

    def test_list_users(self, client):
        admin = self._admin_token(client)
        _register(client, username="alice")
        resp = client.get("/api/admin/users", headers=_auth(admin))
        assert resp.status_code == 200
        users = resp.json()
        assert [u["username"] for u in users] == ["yoozo", "alice"]
        assert users[0]["is_admin"] is True
        assert users[1]["is_admin"] is False
        for key in ("username", "created_at", "is_admin", "quota_limit",
                    "quota_used", "quota_remaining", "reset_date"):
            assert key in users[0]
        assert users[1]["quota_limit"] == 30
        assert users[1]["quota_used"] == 0
        assert users[1]["quota_remaining"] == 30

    def test_patch_quota_realtime(self, client):
        admin = self._admin_token(client)
        user_token = _register(client, username="alice").json()["token"]

        resp = client.patch(
            "/api/admin/users/alice/quota",
            json={"quota_limit": 100}, headers=_auth(admin),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "alice"
        assert data["quota_limit"] == 100
        assert data["quota_remaining"] == 100

        # 实时生效：用户侧 /api/me 立刻看到新上限
        me = client.get("/api/me", headers=_auth(user_token)).json()
        assert me["quota_limit"] == 100
        assert me["quota_remaining"] == 100

        # 0 合法：完全禁用
        resp = client.patch(
            "/api/admin/users/alice/quota",
            json={"quota_limit": 0}, headers=_auth(admin),
        )
        assert resp.status_code == 200
        assert resp.json()["quota_limit"] == 0
        assert client.get("/api/me", headers=_auth(user_token)).json()["quota_remaining"] == 0

    def test_patch_quota_validation_and_404(self, client):
        admin = self._admin_token(client)
        _register(client, username="alice")
        for bad in (-1, 100001, True, "10", None, 3.5):
            resp = client.patch(
                "/api/admin/users/alice/quota",
                json={"quota_limit": bad}, headers=_auth(admin),
            )
            assert resp.status_code == 400, bad
        # 边界值合法
        assert client.patch(
            "/api/admin/users/alice/quota",
            json={"quota_limit": 100000}, headers=_auth(admin),
        ).status_code == 200
        # 用户不存在 404
        assert client.patch(
            "/api/admin/users/ghost/quota",
            json={"quota_limit": 10}, headers=_auth(admin),
        ).status_code == 404

    def test_user_questions(self, client):
        admin = self._admin_token(client)
        user_token = _register(client, username="alice").json()["token"]
        c1 = client.post(
            "/api/conversations", json={"title": "对话一"}, headers=_auth(user_token)
        ).json()
        c2 = client.post(
            "/api/conversations", json={"title": "对话二"}, headers=_auth(user_token)
        ).json()
        user = webauth.resolve_token(user_token)
        webauth.add_message(c1["id"], "user", "问题1")
        webauth.add_message(c1["id"], "assistant", "回答1")
        webauth.add_message(c2["id"], "user", "问题2")

        resp = client.get("/api/admin/users/alice/questions", headers=_auth(admin))
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert [i["content"] for i in data["items"]] == ["问题2", "问题1"]  # 倒序
        assert data["items"][0]["conversation_title"] == "对话二"
        assert all("created_at" in i for i in data["items"])

        # 分页
        page = client.get(
            "/api/admin/users/alice/questions?limit=1&offset=1", headers=_auth(admin)
        ).json()
        assert page["total"] == 2
        assert [i["content"] for i in page["items"]] == ["问题1"]

        # 用户不存在 404；分页参数非法 400
        assert client.get(
            "/api/admin/users/ghost/questions", headers=_auth(admin)
        ).status_code == 404
        assert client.get(
            "/api/admin/users/alice/questions?limit=0", headers=_auth(admin)
        ).status_code in (400, 422)
        assert client.get(
            "/api/admin/users/alice/questions?offset=-1", headers=_auth(admin)
        ).status_code in (400, 422)

    def test_impersonate(self, client):
        admin = self._admin_token(client)
        user_token = _register(client, username="alice").json()["token"]
        # 管理员免密切换：拿到 alice 的普通令牌，且确实能以 alice 身份使用
        resp = client.post("/api/admin/users/alice/impersonate", headers=_auth(admin))
        assert resp.status_code == 200
        body = resp.json()
        assert body["user"]["username"] == "alice" and body["token"]
        me = client.get("/api/me", headers=_auth(body["token"])).json()
        assert me["username"] == "alice" and me["is_admin"] is False
        # 非管理员 403 / 未登录 401 / 用户不存在 404
        assert client.post(
            "/api/admin/users/alice/impersonate", headers=_auth(user_token)
        ).status_code == 403
        assert client.post("/api/admin/users/alice/impersonate").status_code == 401
        assert client.post(
            "/api/admin/users/ghost/impersonate", headers=_auth(admin)
        ).status_code == 404


# ════════════════════════════════════════════════════════════
# 5) 静态站点与 /api/root-info
# ════════════════════════════════════════════════════════════

class TestStaticSite:
    def test_root_503_when_web_missing(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(
            main, "_web_index_path", lambda: str(tmp_path / "no_such_index.html")
        )
        resp = client.get("/")
        assert resp.status_code == 503
        assert resp.json()["error"] == "web_not_deployed"

    def test_root_serves_index_html(self, client, tmp_path, monkeypatch):
        index = tmp_path / "index.html"
        index.write_text("<html><body>市场复盘</body></html>", encoding="utf-8")
        monkeypatch.setattr(main, "_web_index_path", lambda: str(index))
        resp = client.get("/")
        assert resp.status_code == 200
        assert "市场复盘" in resp.text

    def test_root_info_json(self, client):
        resp = client.get("/api/root-info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == main.AGENT_NAME
        assert data["version"] == main.AGENT_VERSION
        assert data["status"] in ("running", "error")

    def test_static_mount_registered(self):
        names = [getattr(r, "name", None) for r in main.app.routes]
        assert "static" in names

    def test_health_unchanged(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


def test_index_html_assets_use_absolute_static_path():
    """防回归：index.html 本地资源必须用 /static/ 绝对路径。
    相对路径在根路由 / 下会被浏览器解析为 /style.css 等 → 404 → 空白页
    （已发生两次：初版与 Kimi 风格重写各回退一次）。"""
    import re
    from pathlib import Path
    html = (Path(__file__).resolve().parent.parent / "web" / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href)="([^"]+)"', html)
    local = [r for r in refs if not r.startswith(("http://", "https://", "data:", "/", "#"))]
    assert local == [], f"发现相对路径资源引用: {local}"
