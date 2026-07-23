"""
公开网站的安全模块（纯标准库实现）。

仅用标准库实现 ASGI 中间件与登录锁定原语，零第三方依赖，
与 Docker 镜像 python:3.13-slim 兼容，不增加 requirements。
（中间件直接实现 ASGI 协议，不经过 starlette BaseHTTPMiddleware，
因此对 /api/chat 这类 text/event-stream 长连接只做头部注入，
绝不缓冲、绝不改写响应体。）

提供能力：
  1. SecurityHeadersMiddleware  安全响应头（nosniff / DENY / Referrer-Policy /
     Permissions-Policy / CSP），仅作用于 HTTP 响应，不碰 SSE body。
  2. IPRateLimitMiddleware      内存滑动窗口限流：全局 300 req/min/IP，
     /api/auth/login 10/min/IP，/api/auth/register 5/min/IP，
     超限返回 429 {"error": "rate_limited"}。
  3. 登录锁定                   同 username 连续 5 次密码错误锁 15 分钟，
     供 main.py 调用 record_login_fail / clear_login_fail / is_locked，
     锁定期登录应直接返回 423 {"error": "locked", "retry_after": <秒>}。
  4. ROBOTS_TXT                 robots.txt 内容常量（全站 Disallow）。
  5. BlockedPathsMiddleware     常见扫描路径黑名单（/.env、/.git、/wp-*、
     /phpmyadmin 等），直接 404 快速返回，不进路由。

接线方式（由 main.py 完成，本模块不修改 main.py）：
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(IPRateLimitMiddleware)
    app.add_middleware(BlockedPathsMiddleware)

    @app.get("/robots.txt")
    async def robots_txt():
        return PlainTextResponse(websec.ROBOTS_TXT)

    # 登录端点内：
    retry_after = websec.is_locked(username)
    if retry_after:
        return JSONResponse({"error": "locked", "retry_after": retry_after},
                            status_code=423)
    ...校验密码...
    失败 → websec.record_login_fail(username)
    成功 → websec.clear_login_fail(username)

注意：限流与登录锁定均为**单 worker 内存实现**。若以后在 Railway 上
扩到多 worker / 多实例，需要换成共享存储（如 Redis），否则各进程
计数互不可见、限制形同虚设。当前部署为单 worker，内存实现足够。
"""

import json
import math
import threading
import time
from collections import deque
from typing import Dict, Optional, Tuple

__all__ = [
    "SecurityHeadersMiddleware",
    "IPRateLimitMiddleware",
    "BlockedPathsMiddleware",
    "record_login_fail",
    "clear_login_fail",
    "is_locked",
    "ROBOTS_TXT",
    "SECURITY_HEADERS",
    "BLOCKED_PATH_PREFIXES",
]


# ══════════════════════════════════════════════════════════════════
# 1. 安全响应头
# ══════════════════════════════════════════════════════════════════

# 纯手写前端（web/ 无构建、无外部 CDN），CSP 可以收得比较紧；
# style-src 保留 'unsafe-inline' 因为页面里有少量 inline style。
CSP_VALUE = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "font-src 'self'; "
    "frame-ancestors 'none'"
)

SECURITY_HEADERS: Dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": CSP_VALUE,
}


class SecurityHeadersMiddleware:
    """给所有 HTTP 响应注入安全头的纯 ASGI 中间件。

    只拦截 http.response.start 消息追加响应头，body 消息原样透传，
    因此 /api/chat 的 SSE 流（text/event-stream）不会被缓冲或改体。
    路由已自行设置的同名头不会被覆盖（避免重复 CSP）。
    """

    def __init__(self, app, headers: Optional[Dict[str, str]] = None):
        self.app = app
        # 预编码为 ASGI raw header 形式
        source = headers if headers is not None else SECURITY_HEADERS
        self._raw_headers: Tuple[Tuple[bytes, bytes], ...] = tuple(
            (k.lower().encode("latin-1"), v.encode("latin-1"))
            for k, v in source.items()
        )

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw_headers = self._raw_headers

        async def send_with_security_headers(message):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                existing = {k.lower() for k, _ in headers}
                for name, value in raw_headers:
                    if name not in existing:
                        headers.append((name, value))
            await send(message)

        await self.app(scope, receive, send_with_security_headers)


# ══════════════════════════════════════════════════════════════════
# 2. IP 滑动窗口限流
# ══════════════════════════════════════════════════════════════════

DEFAULT_WINDOW_SECONDS = 60
DEFAULT_GLOBAL_LIMIT = 300                      # req/min/IP，全站
DEFAULT_PATH_LIMITS = {                         # 敏感端点收紧（req/min/IP）
    "/api/auth/login": 10,
    "/api/auth/register": 5,
}


def _client_ip(scope) -> str:
    """取客户端 IP：优先 X-Forwarded-For 首跳，回退 ASGI client.host。

    部署在 Railway（前置代理）时，X-Forwarded-For 由平台注入，首跳为真实
    客户端 IP；直连场景回退到 TCP 对端地址。
    """
    for name, value in scope.get("headers", []):
        if name.lower() == b"x-forwarded-for":
            first = value.decode("latin-1").split(",")[0].strip()
            if first:
                return first
    client = scope.get("client")
    if client and client[0]:
        return client[0]
    return "unknown"


async def _send_json(send, status: int, payload: dict,
                     extra_headers: Tuple[Tuple[bytes, bytes], ...] = ()):
    """直接在 ASGI 层返回 JSON 响应（不经过路由）。"""
    body = json.dumps(payload).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("latin-1")),
        *extra_headers,
    ]
    await send({"type": "http.response.start", "status": status,
                "headers": headers})
    await send({"type": "http.response.body", "body": body})


class IPRateLimitMiddleware:
    """内存滑动窗口限流（纯 ASGI）。

    规则：每个 (bucket, IP) 维护一个时间戳 deque，窗口 DEFAULT_WINDOW_SECONDS
    （默认 60s）内超过上限即 429 {"error": "rate_limited"}。
    敏感路径（path_limits 精确匹配）与全局限额同时计数、同时生效。

    单 worker 内存实现：多 worker / 多实例部署时必须换共享存储（Redis 等），
    否则每个进程各记各的账，限额实际放大 worker 数倍。
    """

    def __init__(
        self,
        app,
        global_limit: int = DEFAULT_GLOBAL_LIMIT,
        path_limits: Optional[Dict[str, int]] = None,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ):
        self.app = app
        self.global_limit = global_limit
        self.path_limits = (
            dict(path_limits) if path_limits is not None
            else dict(DEFAULT_PATH_LIMITS)
        )
        self.window_seconds = window_seconds
        # key: (bucket_name, ip) → deque[monotonic_ts]
        self._buckets: Dict[Tuple[str, str], deque] = {}
        self._lock = threading.Lock()

    def _hit(self, bucket: str, ip: str, limit: int, now: float) -> bool:
        """记录一次访问；返回 True 表示已超限。"""
        key = (bucket, ip)
        cutoff = now - self.window_seconds
        with self._lock:
            dq = self._buckets.get(key)
            if dq is None:
                dq = deque()
                self._buckets[key] = dq
            while dq and dq[0] <= cutoff:
                dq.popleft()
            if len(dq) >= limit:
                return True
            dq.append(now)
            if not dq:
                self._buckets.pop(key, None)
            return False

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        ip = _client_ip(scope)
        path = scope.get("path", "")
        now = time.monotonic()

        # 敏感路径限额更严，先查；再查全局限额
        path_limit = self.path_limits.get(path)
        if path_limit is not None and self._hit(f"path:{path}", ip,
                                                path_limit, now):
            await _send_json(send, 429, {"error": "rate_limited"},
                             ((b"retry-after", str(self.window_seconds)
                               .encode("latin-1")),))
            return
        if self._hit("global", ip, self.global_limit, now):
            await _send_json(send, 429, {"error": "rate_limited"},
                             ((b"retry-after", str(self.window_seconds)
                               .encode("latin-1")),))
            return

        await self.app(scope, receive, send)


# ══════════════════════════════════════════════════════════════════
# 3. 登录锁定（连续密码错误 → 临时锁定）
# ══════════════════════════════════════════════════════════════════

MAX_CONSECUTIVE_FAILS = 5       # 连续错误次数上限
LOCK_SECONDS = 15 * 60          # 锁定时长：15 分钟

# username → [consecutive_fails, locked_until_epoch(0=未锁)]
_failures: Dict[str, list] = {}
_failures_lock = threading.Lock()


def _now() -> float:
    """当前 epoch 秒。独立成函数便于测试 monkeypatch 时间流逝。"""
    return time.time()


def record_login_fail(username: str) -> int:
    """记录一次密码错误；达到上限即锁定 15 分钟。

    返回当前剩余锁定秒数（0 = 未锁定）。main.py 在密码校验失败时调用。
    """
    with _failures_lock:
        entry = _failures.setdefault(username, [0, 0.0])
        entry[0] += 1
        if entry[0] >= MAX_CONSECUTIVE_FAILS:
            entry[1] = _now() + LOCK_SECONDS
            return LOCK_SECONDS
        return 0


def clear_login_fail(username: str) -> None:
    """登录成功后清零该用户的失败计数与锁定状态。"""
    with _failures_lock:
        _failures.pop(username, None)


def is_locked(username: str) -> int:
    """返回剩余锁定秒数（0 = 未锁定）。

    锁定期满后惰性清零失败计数，给用户全新的一组尝试机会。
    main.py 在登录入口先调本函数，非 0 时直接 423 拒绝。
    """
    with _failures_lock:
        entry = _failures.get(username)
        if not entry or entry[1] <= 0:
            return 0
        remaining = entry[1] - _now()
        if remaining <= 0:
            # 锁定期满：重置计数，允许重新尝试
            del _failures[username]
            return 0
        return math.ceil(remaining)


# ══════════════════════════════════════════════════════════════════
# 4. robots.txt 内容常量
# ══════════════════════════════════════════════════════════════════

# 全站禁止抓取：站内内容为登录用户的个人对话，无公开索引价值
ROBOTS_TXT = "User-agent: *\nDisallow: /\n"


# ══════════════════════════════════════════════════════════════════
# 5. 扫描路径黑名单
# ══════════════════════════════════════════════════════════════════

# 公网常见扫描器/漏扫的探测路径前缀（小写匹配），一律 404 快速返回，
# 不进 FastAPI 路由与异常处理链，省 CPU 也不泄露路由存在性。
BLOCKED_PATH_PREFIXES: Tuple[str, ...] = (
    "/.env",
    "/.git",
    "/.svn",
    "/.hg",
    "/.aws",
    "/.ssh",
    "/.vscode",
    "/.idea",
    "/wp-",
    "/wordpress",
    "/phpmyadmin",
    "/pma",
    "/server-status",
    "/.well-known/change-password",
)


class BlockedPathsMiddleware:
    """命中扫描路径黑名单的请求直接 404（纯 ASGI，快速返回）。"""

    def __init__(self, app,
                 prefixes: Tuple[str, ...] = BLOCKED_PATH_PREFIXES):
        self.app = app
        self.prefixes = tuple(p.lower() for p in prefixes)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "").lower()
            if any(path.startswith(p) for p in self.prefixes):
                body = b"Not Found"
                await send({
                    "type": "http.response.start",
                    "status": 404,
                    "headers": [
                        (b"content-type", b"text/plain; charset=utf-8"),
                        (b"content-length", str(len(body)).encode("latin-1")),
                    ],
                })
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)
