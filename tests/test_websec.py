"""
安全模块测试（agent/websec.py，纯 stdlib ASGI 中间件 + 登录锁定）。

用 FastAPI 迷你 app + TestClient 验证，不 import main.py，不触达
真实数据库与外部 API：

1. SecurityHeadersMiddleware：普通响应 5 个安全头齐全、CSP 值精确匹配；
   SSE（text/event-stream）响应同样带头且流式 body 原样透传不被缓冲改体。
2. IPRateLimitMiddleware：滑动窗口超限 → 429 {"error": "rate_limited"}；
   敏感路径（/api/auth/login、/api/auth/register）限额独立于全局；
   X-Forwarded-For 首跳作为限流维度，不同来源 IP 互不影响。
3. 登录锁定：连续 5 次 record_login_fail → is_locked 返回剩余秒数；
   clear_login_fail 清零；monkeypatch 时间流逝后锁定自动过期并清零计数。
4. BlockedPathsMiddleware：/.env、/.git、/wp-*、/phpmyadmin 等 → 404，
   正常路径不受影响。
5. ROBOTS_TXT 常量内容。
"""

import os
import sys
import time
from pathlib import Path

import pytest

# conftest 会设置 AGENT_API_KEY 测试假值；本文件不 import main，仅作兜底
os.environ.setdefault("AGENT_API_KEY", "test-fake-agent-key-for-pytest")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import websec  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


# ── 迷你 app 工厂 ──

def make_app(global_limit=300, path_limits=None):
    """只挂 websec 中间件的最小 FastAPI app（与 main.py 无关）。"""
    app = FastAPI()

    @app.get("/ok")
    async def ok():
        return {"ok": True}

    @app.post("/api/auth/login")
    async def login():
        return {"token": "fake"}

    @app.post("/api/auth/register")
    async def register():
        return {"username": "fake"}

    @app.get("/api/chat")
    async def chat():
        async def gen():
            for i in range(3):
                yield f"data: chunk-{i}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    # add_middleware 后挂的在外层；顺序不影响本测试断言
    app.add_middleware(websec.SecurityHeadersMiddleware)
    app.add_middleware(
        websec.IPRateLimitMiddleware,
        global_limit=global_limit,
        path_limits=path_limits,
    )
    app.add_middleware(websec.BlockedPathsMiddleware)
    return app


@pytest.fixture(autouse=True)
def _clean_login_failures():
    """登录锁定是模块级内存状态，用例间互斥清理。"""
    websec._failures.clear()
    yield
    websec._failures.clear()


# ── 1. 安全响应头 ──

class TestSecurityHeaders:
    EXPECTED = {
        "x-content-type-options": "nosniff",
        "x-frame-options": "DENY",
        "referrer-policy": "strict-origin-when-cross-origin",
        "permissions-policy": "camera=(), microphone=(), geolocation=()",
        "content-security-policy": websec.CSP_VALUE,
    }

    def test_headers_present_on_normal_response(self):
        client = TestClient(make_app())
        resp = client.get("/ok")
        assert resp.status_code == 200
        for name, value in self.EXPECTED.items():
            assert resp.headers.get(name) == value, f"缺失或不符: {name}"

    def test_csp_value_exact(self):
        assert websec.CSP_VALUE == (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "font-src 'self'; "
            "frame-ancestors 'none'"
        )

    def test_headers_on_sse_without_touching_body(self):
        """SSE 长连接：头照常注入，流式 body 必须原样透传。"""
        client = TestClient(make_app())
        resp = client.get("/api/chat")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        for name, value in self.EXPECTED.items():
            assert resp.headers.get(name) == value, f"SSE 缺失头: {name}"
        assert resp.text == "data: chunk-0\n\ndata: chunk-1\n\ndata: chunk-2\n\n"

    def test_route_set_header_not_overridden(self):
        """路由自行设置的同名头不被中间件覆盖、不重复。"""
        app = FastAPI()

        @app.get("/custom")
        async def custom():
            return JSONResponse(
                {"ok": True},
                headers={"X-Frame-Options": "SAMEORIGIN"},
            )

        app.add_middleware(websec.SecurityHeadersMiddleware)
        resp = TestClient(app).get("/custom")
        assert resp.headers["x-frame-options"] == "SAMEORIGIN"
        # 其余头仍正常注入
        assert resp.headers["x-content-type-options"] == "nosniff"


# ── 2. IP 限流 ──

class TestIPRateLimit:
    def test_global_limit_triggers_429(self):
        client = TestClient(make_app(global_limit=3))
        for _ in range(3):
            assert client.get("/ok").status_code == 200
        resp = client.get("/ok")
        assert resp.status_code == 429
        assert resp.json() == {"error": "rate_limited"}

    def test_login_path_stricter_limit(self):
        client = TestClient(make_app(
            global_limit=100,
            path_limits={"/api/auth/login": 2, "/api/auth/register": 5},
        ))
        assert client.post("/api/auth/login").status_code == 200
        assert client.post("/api/auth/login").status_code == 200
        resp = client.post("/api/auth/login")
        assert resp.status_code == 429
        assert resp.json() == {"error": "rate_limited"}
        # 其它路径不受 login 专用桶影响
        assert client.get("/ok").status_code == 200

    def test_register_path_limit_default(self):
        """缺省 path_limits：/api/auth/register 限 5/min。"""
        client = TestClient(make_app(global_limit=100))
        for _ in range(5):
            assert client.post("/api/auth/register").status_code == 200
        resp = client.post("/api/auth/register")
        assert resp.status_code == 429
        assert resp.json() == {"error": "rate_limited"}

    def test_x_forwarded_for_first_hop_is_bucket_key(self):
        client = TestClient(make_app(global_limit=2))
        h1 = {"X-Forwarded-For": "1.1.1.1, 10.0.0.1"}
        h2 = {"X-Forwarded-For": "2.2.2.2"}
        assert client.get("/ok", headers=h1).status_code == 200
        assert client.get("/ok", headers=h1).status_code == 200
        # 同一首跳超限
        assert client.get("/ok", headers=h1).status_code == 429
        # 不同首跳互不影响
        assert client.get("/ok", headers=h2).status_code == 200

    def test_window_slides(self):
        """窗口滑过后限额恢复（构造极小窗口验证滑动逻辑）。"""
        app = FastAPI()

        @app.get("/ok")
        async def ok():
            return {"ok": True}

        app.add_middleware(websec.IPRateLimitMiddleware,
                           global_limit=1, path_limits={}, window_seconds=1)
        client = TestClient(app)
        assert client.get("/ok").status_code == 200
        assert client.get("/ok").status_code == 429
        time.sleep(1.1)
        assert client.get("/ok").status_code == 200


# ── 3. 登录锁定 ──

class TestLoginLockout:
    USER = "alice"

    def test_under_threshold_not_locked(self):
        for _ in range(websec.MAX_CONSECUTIVE_FAILS - 1):
            assert websec.record_login_fail(self.USER) == 0
            assert websec.is_locked(self.USER) == 0

    def test_fifth_fail_locks_15_minutes(self):
        for _ in range(websec.MAX_CONSECUTIVE_FAILS - 1):
            websec.record_login_fail(self.USER)
        remaining = websec.record_login_fail(self.USER)
        assert remaining == websec.LOCK_SECONDS
        locked = websec.is_locked(self.USER)
        assert 0 < locked <= websec.LOCK_SECONDS

    def test_clear_resets_counter(self):
        for _ in range(websec.MAX_CONSECUTIVE_FAILS):
            websec.record_login_fail(self.USER)
        assert websec.is_locked(self.USER) > 0
        websec.clear_login_fail(self.USER)
        assert websec.is_locked(self.USER) == 0
        # 清零后重新计数，一两次失败不再锁
        assert websec.record_login_fail(self.USER) == 0
        assert websec.is_locked(self.USER) == 0

    def test_unknown_user_not_locked(self):
        assert websec.is_locked("nobody") == 0

    def test_lock_expires_and_counter_resets(self, monkeypatch):
        fake_now = time.time()
        monkeypatch.setattr(websec, "_now", lambda: fake_now)
        for _ in range(websec.MAX_CONSECUTIVE_FAILS):
            websec.record_login_fail(self.USER)
        assert websec.is_locked(self.USER) > 0
        # 时间快进到锁定期满之后
        monkeypatch.setattr(websec, "_now",
                            lambda: fake_now + websec.LOCK_SECONDS + 1)
        assert websec.is_locked(self.USER) == 0
        # 过期后失败计数已清零：一次失败不会立即再锁
        assert websec.record_login_fail(self.USER) == 0
        assert websec.is_locked(self.USER) == 0

    def test_usernames_independent(self):
        for _ in range(websec.MAX_CONSECUTIVE_FAILS):
            websec.record_login_fail("bob")
        assert websec.is_locked("bob") > 0
        assert websec.is_locked("carol") == 0


# ── 4. 扫描路径黑名单 ──

class TestBlockedPaths:
    @pytest.mark.parametrize("path", [
        "/.env",
        "/.env.local",
        "/.git/config",
        "/.git/HEAD",
        "/.svn/entries",
        "/wp-admin/setup-config.php",
        "/wp-login.php",
        "/wordpress/wp-config.php",
        "/phpmyadmin/index.php",
        "/pma/index.php",
        "/server-status",
    ])
    def test_scanner_paths_404(self, path):
        client = TestClient(make_app())
        resp = client.get(path)
        assert resp.status_code == 404

    def test_normal_paths_unaffected(self):
        client = TestClient(make_app())
        assert client.get("/ok").status_code == 200
        assert client.post("/api/auth/login").status_code == 200


# ── 5. robots.txt 常量 ──

def test_robots_txt_constant():
    assert websec.ROBOTS_TXT == "User-agent: *\nDisallow: /\n"
    assert "Disallow: /" in websec.ROBOTS_TXT
