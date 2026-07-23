"""
公开网站后端测试（agent/webauth.py + main.py 的 /api/* 端点）。

覆盖范围：
1. webauth 单元层：注册/登录/重复注册/错误密码/token 过期/登出、
   额度快照/扣减顺序（先月度后包）/加油包累加、对话 CRUD 与越权隔离
2. 端点层（FastAPI TestClient）：
   - POST /api/auth/register → 200 / 409 / 400
   - POST /api/auth/login → 200 / 401
   - POST /api/auth/logout → token 失效
   - GET /api/me → 额度字段完整
   - 对话 CRUD：列表降序、详情带消息、越权 404、删除 200/404
   - POST /api/topup → 包额度累加
   - POST /api/chat → SSE 流式（mock agent）、落库、成功才扣额度、
     额度耗尽 429、流失败不扣额度
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

    def test_quota_defaults(self, db):
        user = webauth.resolve_token(webauth.register("alice", "password1"))
        quota = webauth.get_quota(user["id"])
        assert quota["monthly_quota"] == 100
        assert quota["monthly_used"] == 0
        assert quota["monthly_remaining"] == 100
        assert quota["pack_credits"] == 0
        assert quota["total_remaining"] == 100
        # reset_date 为次月 1 日
        today = datetime.now().date()
        expected = (
            f"{today.year + 1}-01-01" if today.month == 12
            else f"{today.year}-{today.month + 1:02d}-01"
        )
        assert quota["reset_date"] == expected

    def test_consume_order_monthly_before_pack(self, db, monkeypatch):
        """扣减顺序：先扣月度，月度用尽后再扣加油包。"""
        monkeypatch.setenv("WEB_MONTHLY_QUOTA", "2")
        monkeypatch.setenv("WEB_PACK_SIZE", "10")
        user = webauth.resolve_token(webauth.register("alice", "password1"))
        webauth.topup(user["id"], 1)  # +10 包额度

        assert webauth.consume_quota(user["id"]) is True
        q = webauth.get_quota(user["id"])
        assert (q["monthly_used"], q["pack_credits"]) == (1, 10)

        assert webauth.consume_quota(user["id"]) is True
        q = webauth.get_quota(user["id"])
        assert (q["monthly_used"], q["pack_credits"]) == (2, 10)  # 月度用尽，包未动

        assert webauth.consume_quota(user["id"]) is True
        q = webauth.get_quota(user["id"])
        assert (q["monthly_used"], q["pack_credits"]) == (2, 9)  # 开始扣包

    def test_consume_exhausted_returns_false(self, db, monkeypatch):
        monkeypatch.setenv("WEB_MONTHLY_QUOTA", "1")
        user = webauth.resolve_token(webauth.register("alice", "password1"))
        assert webauth.consume_quota(user["id"]) is True
        assert webauth.consume_quota(user["id"]) is False
        # 失败不扣任何计数
        assert webauth.get_quota(user["id"])["total_remaining"] == 0

    def test_topup_accumulates(self, db, monkeypatch):
        monkeypatch.setenv("WEB_PACK_SIZE", "50")
        user = webauth.resolve_token(webauth.register("alice", "password1"))
        q1 = webauth.topup(user["id"], 1)
        assert q1["pack_credits"] == 50
        q2 = webauth.topup(user["id"], 2)
        assert q2["pack_credits"] == 150
        assert q2["total_remaining"] == 100 + 150

    def test_topup_invalid_pack_count(self, db):
        user = webauth.resolve_token(webauth.register("alice", "password1"))
        for bad in (0, -1, True, "1"):
            with pytest.raises(webauth.ValidationError):
                webauth.topup(user["id"], bad)

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
        for key in (
            "monthly_quota", "monthly_used", "monthly_remaining",
            "pack_credits", "total_remaining", "reset_date",
        ):
            assert key in data
        assert data["monthly_remaining"] == data["monthly_quota"] - data["monthly_used"]
        assert data["total_remaining"] == data["monthly_remaining"] + data["pack_credits"]


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
        assert client.delete("/api/conversations/whatever").status_code == 401


# ════════════════════════════════════════════════════════════
# 4) 充值与 /api/chat
# ════════════════════════════════════════════════════════════

class TestTopup:
    def test_topup_accumulates(self, client, monkeypatch):
        monkeypatch.setenv("WEB_PACK_SIZE", "50")
        token = _register(client).json()["token"]
        resp = client.post("/api/topup", json={"pack_count": 1}, headers=_auth(token))
        assert resp.status_code == 200
        assert resp.json() == {"pack_credits": 50, "total_remaining": 100 + 50}
        resp = client.post("/api/topup", json={"pack_count": 2}, headers=_auth(token))
        assert resp.json()["pack_credits"] == 150

    def test_topup_requires_auth(self, client):
        assert client.post("/api/topup", json={"pack_count": 1}).status_code == 401


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
        assert me["monthly_used"] == 1
        assert me["total_remaining"] == me["monthly_quota"] - 1

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
        assert me["monthly_used"] == 0
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
        assert data["total_remaining"] == 0
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
