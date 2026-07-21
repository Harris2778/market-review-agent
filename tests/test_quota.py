"""
第六波工程化：每分钟限流 / 每日配额 / /v1/usage / 版本号测试（main.py）。

覆盖范围：
1. 每分钟限流：时间注入下触发 429（rate_limited）、被拒请求不消耗计数、
   跨入下一分钟自动恢复
2. 每日配额：触发 429（quota_exceeded）、被拒请求不消耗计数
3. 跨日重置：Asia/Shanghai 自然日边界（23:59:50 → 次日 00:00:05）自动归零
4. /v1/usage：计数正确、查询本身不消耗计数、跨分钟/跨日惰性归零、Bearer 鉴权
5. 鉴权失败（401）的请求不计数（无 Authorization 头与错误 token 两种）
6. AGENT_VERSION 已升级为 1.3.0（根端点与 app.version 同步）

零网络：agent 全部 mock；时间通过 patch main._quota_now 注入；
计数器为 main 模块级内存 dict，每个用例前后在锁内清空，互不污染。
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

# conftest 会注入 AGENT_API_KEY 测试假值；setdefault 兜底保证本文件可独立运行
os.environ.setdefault("AGENT_API_KEY", "test-fake-agent-key-for-pytest")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

INVALID_AUTH = {"Authorization": "Bearer wrong-token-definitely-not-the-key"}
SHANGHAI = ZoneInfo("Asia/Shanghai")


def sh_ts(year, month, day, hour=12, minute=0, second=0):
    """构造 Asia/Shanghai 时区某时刻对应的 epoch 秒。"""
    return datetime(year, month, day, hour, minute, second, tzinfo=SHANGHAI).timestamp()


@pytest.fixture(scope="module")
def client():
    return TestClient(main.app)


@pytest.fixture()
def auth():
    # 与 main 模块实际加载到的 key 保持一致（conftest 注入的假值）
    return {"Authorization": f"Bearer {main.AGENT_API_KEY}"}


@pytest.fixture(autouse=True)
def reset_quota_state():
    """每个用例前后清空模块级计数器（持锁操作），避免用例间与跨测试文件污染。"""
    with main._quota_lock:
        main._quota_state["minutes"].clear()
        main._quota_state["daily"]["date"] = None
        main._quota_state["daily"]["count"] = 0
    yield
    with main._quota_lock:
        main._quota_state["minutes"].clear()
        main._quota_state["daily"]["date"] = None
        main._quota_state["daily"]["count"] = 0


def _patched_agent():
    """返回 patch main.get_agent / main._agent_loaded 的上下文管理器组合。"""
    agent = MagicMock()
    agent.process_message = AsyncMock(return_value={"content": "今日市场复盘内容"})
    agent.cache_warm = True
    return (
        patch.object(main, "get_agent", return_value=agent),
        patch.object(main, "_agent_loaded", True),
    )


def _post_chat(client, auth):
    """发一次已鉴权的非流式对话请求（需在 _patched_agent 上下文中调用）。"""
    return client.post(
        "/v1/chat/completions",
        json={
            "model": "market-review-agent",
            "messages": [{"role": "user", "content": "今日复盘"}],
            "stream": False,
        },
        headers=auth,
    )


# ── 1) 每分钟限流：触发与恢复（时间注入）──

def test_rate_limit_trigger_and_recovery(client, auth, monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT_PER_MIN", 3)
    t0 = sh_ts(2026, 7, 20, 15, 30, 10)
    get_agent_patch, loaded_patch = _patched_agent()
    with get_agent_patch, loaded_patch:
        monkeypatch.setattr(main, "_quota_now", lambda: t0)

        # 阈值内全部放行
        codes = [_post_chat(client, auth).status_code for _ in range(3)]
        assert codes == [200, 200, 200], f"限流阈值内的请求应全部放行，实际 {codes}"

        # 超出阈值 → 429 rate_limited，响应体为契约约定的 error 结构
        resp = _post_chat(client, auth)
        assert resp.status_code == 429
        assert resp.json() == {
            "error": {"message": "请求过于频繁，请稍后再试", "code": "rate_limited"}
        }

        # 被拒绝的请求不得消耗任何计数
        usage = client.get("/v1/usage", headers=auth).json()
        assert usage["minute_used"] == 3
        assert usage["today_used"] == 3

        # 时间注入跨入下一分钟 → 滑动窗口推进，自动恢复
        monkeypatch.setattr(main, "_quota_now", lambda: t0 + 60)
        assert _post_chat(client, auth).status_code == 200
        usage = client.get("/v1/usage", headers=auth).json()
        assert usage["minute_used"] == 1
        assert usage["today_used"] == 4


# ── 2) 每日配额：触发 ──

def test_daily_quota_trigger(client, auth, monkeypatch):
    monkeypatch.setattr(main, "QUOTA_DAILY", 2)
    monkeypatch.setattr(main, "RATE_LIMIT_PER_MIN", 1000)  # 排除分钟限流干扰
    t0 = sh_ts(2026, 7, 20, 15, 40, 0)
    get_agent_patch, loaded_patch = _patched_agent()
    with get_agent_patch, loaded_patch:
        monkeypatch.setattr(main, "_quota_now", lambda: t0)

        assert _post_chat(client, auth).status_code == 200
        assert _post_chat(client, auth).status_code == 200

        resp = _post_chat(client, auth)
        assert resp.status_code == 429
        assert resp.json() == {
            "error": {"message": "今日配额已用完", "code": "quota_exceeded"}
        }

        # 被拒请求不消耗计数
        usage = client.get("/v1/usage", headers=auth).json()
        assert usage["today_used"] == 2
        assert usage["minute_used"] == 2


def test_rate_limit_checked_before_quota(client, auth, monkeypatch):
    """分钟限流与每日配额同时超限时，优先报 rate_limited（更瞬时的限制先报）。"""
    monkeypatch.setattr(main, "RATE_LIMIT_PER_MIN", 2)
    monkeypatch.setattr(main, "QUOTA_DAILY", 2)
    t0 = sh_ts(2026, 7, 20, 14, 0, 0)
    get_agent_patch, loaded_patch = _patched_agent()
    with get_agent_patch, loaded_patch:
        monkeypatch.setattr(main, "_quota_now", lambda: t0)
        assert _post_chat(client, auth).status_code == 200
        assert _post_chat(client, auth).status_code == 200
        resp = _post_chat(client, auth)
        assert resp.status_code == 429
        assert resp.json()["error"]["code"] == "rate_limited"


# ── 3) 跨日重置（Asia/Shanghai 自然日）──

def test_daily_quota_resets_across_shanghai_midnight(client, auth, monkeypatch):
    monkeypatch.setattr(main, "QUOTA_DAILY", 2)
    monkeypatch.setattr(main, "RATE_LIMIT_PER_MIN", 1000)
    # 上海时间 2026-07-20 23:59:50 → 仅过 15 秒上海已跨自然日
    t1 = sh_ts(2026, 7, 20, 23, 59, 50)
    t2 = t1 + 15
    assert main._quota_today(t1) == "2026-07-20"
    assert main._quota_today(t2) == "2026-07-21"

    get_agent_patch, loaded_patch = _patched_agent()
    with get_agent_patch, loaded_patch:
        monkeypatch.setattr(main, "_quota_now", lambda: t1)
        assert _post_chat(client, auth).status_code == 200
        assert _post_chat(client, auth).status_code == 200
        assert _post_chat(client, auth).status_code == 429  # 当日配额用完

        # 上海跨过零点 → 配额自动归零重置
        monkeypatch.setattr(main, "_quota_now", lambda: t2)
        assert _post_chat(client, auth).status_code == 200
        usage = client.get("/v1/usage", headers=auth).json()
        assert usage["today_used"] == 1


# ── 4) /v1/usage：计数正确、查询不消耗、鉴权 ──

def test_usage_endpoint_counts_correctly(client, auth, monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT_PER_MIN", 30)
    monkeypatch.setattr(main, "QUOTA_DAILY", 500)
    t0 = sh_ts(2026, 7, 20, 10, 0, 0)
    get_agent_patch, loaded_patch = _patched_agent()
    with get_agent_patch, loaded_patch:
        monkeypatch.setattr(main, "_quota_now", lambda: t0)
        for _ in range(3):
            assert _post_chat(client, auth).status_code == 200

    resp = client.get("/v1/usage", headers=auth)
    assert resp.status_code == 200
    assert resp.json() == {
        "today_used": 3,
        "daily_quota": 500,
        "minute_used": 3,
        "rate_limit": 30,
    }

    # 查询本身不消耗计数：再查一次数值不变
    assert client.get("/v1/usage", headers=auth).json() == resp.json()

    # 跨入下一分钟：minute_used 归零，today_used 保留
    monkeypatch.setattr(main, "_quota_now", lambda: t0 + 60)
    usage = client.get("/v1/usage", headers=auth).json()
    assert usage["minute_used"] == 0
    assert usage["today_used"] == 3

    # 跨日：today_used 惰性归零（不依赖 chat 请求触发）
    monkeypatch.setattr(main, "_quota_now", lambda: sh_ts(2026, 7, 21, 9, 0, 0))
    usage = client.get("/v1/usage", headers=auth).json()
    assert usage["today_used"] == 0
    assert usage["minute_used"] == 0


def test_usage_endpoint_requires_auth(client):
    resp = client.get("/v1/usage")
    assert resp.status_code == 401
    assert resp.json()["detail"]["code"] == "invalid_api_key"

    resp = client.get("/v1/usage", headers=INVALID_AUTH)
    assert resp.status_code == 401
    assert resp.json()["detail"]["code"] == "invalid_api_key"


# ── 5) 鉴权失败（401）的请求不计数 ──

def test_unauthorized_requests_not_counted(client, auth, monkeypatch):
    t0 = sh_ts(2026, 7, 20, 11, 0, 0)
    monkeypatch.setattr(main, "_quota_now", lambda: t0)
    payload = {
        "model": "market-review-agent",
        "messages": [{"role": "user", "content": "今日复盘"}],
    }

    # 默认 RATE_LIMIT_PER_MIN=30 阈值下，灌入远超阈值的 401 请求（无头 + 错钥）
    for _ in range(50):
        assert client.post("/v1/chat/completions", json=payload).status_code == 401
    for _ in range(50):
        assert client.post(
            "/v1/chat/completions", json=payload, headers=INVALID_AUTH
        ).status_code == 401

    # 计数器零消耗
    usage = client.get("/v1/usage", headers=auth).json()
    assert usage["minute_used"] == 0
    assert usage["today_used"] == 0

    # 100 次 401 之后合法请求仍正常放行（未误触限流）
    get_agent_patch, loaded_patch = _patched_agent()
    with get_agent_patch, loaded_patch:
        assert _post_chat(client, auth).status_code == 200


# ── 6) 版本号 1.3.0（生产健康检查确认新部署）──

def test_agent_version_bumped_to_1_3_0():
    assert main.AGENT_VERSION == "1.3.0"
    assert main.app.version == "1.3.0"


def test_root_endpoint_reports_new_version(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["version"] == "1.3.0"
