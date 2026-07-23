"""
FastAPI 接口层测试（main.py）。

覆盖范围：
1. 全部 12 个 /debug/* 端点无 Authorization 头 → 401（参数化）
2. 错误的 Bearer token → 401
3. /health 无需鉴权 → 200
4. POST /v1/chat/completions：无鉴权 401；agent.process_message 抛异常时
   返回结构化错误 JSON（含 error 字段，HTTP 502）而非裸 500 traceback
   —— 非流式兜底防回归
5. /debug/futures 响应 JSON 不包含 token 相关字段（防 token 泄漏回归）
6. 服务信息端点 /api/root-info 返回的 JSON 不包含 traceback / 服务器路径信息

所有外部调用（requests / tushare / data_fetcher / orchestrator）全部 mock，
绝不发起真实网络请求。
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# conftest 会设置 AGENT_API_KEY 测试假值；此处 setdefault 仅作兜底，
# 保证本文件在 conftest 缺席时也可独立运行（main.py 启动强制要求该变量）。
os.environ.setdefault("AGENT_API_KEY", "test-fake-agent-key-for-pytest")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# 与 main.py 中实际注册的 12 个调试端点逐一核对过
DEBUG_ENDPOINTS = [
    "/debug/mcp-test",
    "/debug/hot",
    "/debug/sina-news",
    "/debug/sector-stocks",
    "/debug/derivatives",
    "/debug/macro",
    "/debug/mcp-news",
    "/debug/stock-all",
    "/debug/futures",
    "/debug/news-count",
    "/debug/pipeline",
    "/debug/tushare",
]


@pytest.fixture(scope="module")
def client():
    return TestClient(main.app)


@pytest.fixture()
def valid_auth():
    # 以 main 模块实际加载到的 key 为准，与 conftest 注入的假值保持一致
    return {"Authorization": f"Bearer {main.AGENT_API_KEY}"}


INVALID_AUTH = {"Authorization": "Bearer wrong-token-definitely-not-the-key"}


def _iter_keys(obj):
    """递归产出 JSON 结构中所有 dict 键。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _iter_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_keys(item)


# ── 1) 12 个 /debug/* 端点：无 Authorization 头 → 401 ──

@pytest.mark.parametrize("endpoint", DEBUG_ENDPOINTS)
def test_debug_endpoint_requires_auth(client, endpoint):
    resp = client.get(endpoint)
    assert resp.status_code == 401, f"{endpoint} 未鉴权时应返回 401，实际 {resp.status_code}"
    detail = resp.json().get("detail")
    assert isinstance(detail, dict), f"{endpoint} 的 401 detail 应为结构化 JSON"
    assert detail.get("code") == "invalid_api_key"
    assert "error" in detail


# ── 2) 错误的 Bearer token → 401 ──

@pytest.mark.parametrize("endpoint", DEBUG_ENDPOINTS)
def test_debug_endpoint_rejects_wrong_token(client, endpoint):
    resp = client.get(endpoint, headers=INVALID_AUTH)
    assert resp.status_code == 401, f"{endpoint} 错误 token 时应返回 401，实际 {resp.status_code}"
    assert resp.json()["detail"]["code"] == "invalid_api_key"


def test_debug_endpoint_rejects_malformed_auth_header(client):
    """缺少 Bearer 前缀的 header 同样应拒绝。"""
    resp = client.get("/debug/tushare", headers={"Authorization": main.AGENT_API_KEY})
    assert resp.status_code == 401


# ── 3) /health 无需鉴权 → 200 ──

def test_health_no_auth_required(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "apis" in data
    assert "agent" in data


# ── 4) /v1/chat/completions ──

def test_chat_completions_no_auth(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "market-review-agent", "messages": [{"role": "user", "content": "今日复盘"}]},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"]["code"] == "invalid_api_key"


def test_chat_completions_wrong_token(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "market-review-agent", "messages": [{"role": "user", "content": "今日复盘"}]},
        headers=INVALID_AUTH,
    )
    assert resp.status_code == 401


def test_chat_completions_empty_messages(client, valid_auth):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "market-review-agent", "messages": []},
        headers=valid_auth,
    )
    assert resp.status_code == 400


def _patched_agent(process_message):
    """返回 patch main.get_agent / main._agent_loaded 的上下文管理器组合。"""
    agent = MagicMock()
    agent.process_message = process_message
    agent.cache_warm = True
    return (
        patch.object(main, "get_agent", return_value=agent),
        patch.object(main, "_agent_loaded", True),
    )


def test_chat_completions_success_structure(client, valid_auth):
    """非流式成功路径：返回 OpenAI 兼容的 chat.completion 结构。"""
    get_agent_patch, loaded_patch = _patched_agent(
        AsyncMock(return_value={"content": "今日市场复盘内容"})
    )
    with get_agent_patch, loaded_patch:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "market-review-agent",
                "messages": [{"role": "user", "content": "今日复盘"}],
                "stream": False,
            },
            headers=valid_auth,
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl-")
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "今日市场复盘内容"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert "usage" in data


def test_chat_completions_agent_exception_returns_structured_error(client, valid_auth):
    """
    防回归：非流式处理中 agent.process_message 抛异常时，
    必须返回结构化错误 JSON（含 error 字段，HTTP 502），
    不得泄露 traceback / 服务器路径 / 原始异常信息。
    """
    secret_bearing_error = RuntimeError(
        "DeepSeek API 连接失败，详见 /Users/harriszhang/market-review-agent/secret.py"
    )
    get_agent_patch, loaded_patch = _patched_agent(
        AsyncMock(side_effect=secret_bearing_error)
    )
    with get_agent_patch, loaded_patch:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "market-review-agent",
                "messages": [{"role": "user", "content": "今日复盘"}],
                "stream": False,
            },
            headers=valid_auth,
        )

    # 不是裸 500
    assert resp.status_code == 502, f"异常时应返回 502，实际 {resp.status_code}"
    data = resp.json()
    assert "error" in data, "错误响应必须包含 error 字段"
    assert data["error"]["type"] == "upstream_error"
    assert data["error"]["code"] == "agent_process_error"
    assert data["error"]["message"]

    body = resp.text
    assert "Traceback" not in body, "响应不得包含 traceback"
    assert "/Users/" not in body, "响应不得包含服务器本地路径"
    assert "secret.py" not in body, "响应不得泄露原始异常信息"
    assert "DeepSeek API 连接失败" not in body


def test_chat_completions_not_json_body(client, valid_auth):
    resp = client.post(
        "/v1/chat/completions",
        content="not-a-json-body",
        headers={**valid_auth, "Content-Type": "application/json"},
    )
    assert resp.status_code == 400


# ── 5) /debug/futures：响应 JSON 不包含 token 相关字段 ──

def _fetch_futures_stub(*args, **kwargs):
    return {"symbol": "AU0", "name": "沪金主力", "price": 560.5, "pct": 0.8}


def _fetch_stock_quote_stub(*args, **kwargs):
    return {"name": "贵州茅台", "code": "sh600519", "price": 1500.0, "pct": 1.2}


def test_debug_futures_response_has_no_token_fields(client, valid_auth):
    with patch("agent.data_fetcher.fetch_futures", side_effect=_fetch_futures_stub), \
         patch("agent.data_fetcher.fetch_stock_quote", side_effect=_fetch_stock_quote_stub):
        resp = client.get("/debug/futures", headers=valid_auth)

    assert resp.status_code == 200
    data = resp.json()
    assert data["futures"]["symbol"] == "AU0"
    assert data["stock"]["name"] == "贵州茅台"

    all_keys = [k.lower() for k in _iter_keys(data)]
    assert "token_prefix" not in all_keys, "响应不得包含 token_prefix 键"
    assert not any("token" in k for k in all_keys), (
        f"响应不得包含任何 token 相关字段，实际键：{all_keys}"
    )


def test_debug_futures_token_leak_regression_would_be_caught(client, valid_auth):
    """
    负向验证：若数据层 mock 返回中混入 token_prefix（模拟历史泄漏 bug），
    上面的递归键扫描必须能发现它——验证断言本身有效。
    """
    with patch(
        "agent.data_fetcher.fetch_futures",
        return_value={"symbol": "AU0", "token_prefix": "abcd1234"},
    ), patch("agent.data_fetcher.fetch_stock_quote", return_value={}):
        resp = client.get("/debug/futures", headers=valid_auth)
    assert resp.status_code == 200
    all_keys = [k.lower() for k in _iter_keys(resp.json())]
    # 该用例确认端点是透传的，因此主用例的"无 token 键"断言具有回归意义
    assert "token_prefix" in all_keys


# ── 6) 服务信息端点 /api/root-info：不泄露 traceback / 服务器路径 ──
# （网站上线后原根路由 / 的 JSON 挪到 /api/root-info，/ 改为静态站点首页）

def test_root_info_no_traceback_or_server_path(client):
    resp = client.get("/api/root-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == main.AGENT_NAME
    assert "version" in data
    assert data["status"] in ("running", "error")

    body = resp.text
    assert "Traceback" not in body
    assert 'File "' not in body
    assert "/Users/" not in body
    assert "site-packages" not in body


def test_root_info_agent_load_failure_still_safe(client):
    """
    即使智能体加载失败，/api/root-info 也只返回通用错误文案，
    不得把 _agent_error（含 traceback）写进响应体。
    """
    fake_tb = 'Traceback (most recent call last):\n  File "/Users/harriszhang/x.py", line 1'
    with patch.object(main, "_agent_loaded", False), \
         patch.object(main, "_agent_error", fake_tb):
        resp = client.get("/api/root-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "error"
    assert "error" in data
    body = resp.text
    assert "Traceback" not in body
    assert "/Users/" not in body
    assert "x.py" not in body


# ── 附加：带鉴权的 debug 端点 happy-path（数据源全部 mock）──

def test_debug_sina_news_with_mocked_fetcher(client, valid_auth):
    news = [{"title": f"样例新闻标题{i}号" * 3} for i in range(5)]
    with patch("agent.data_fetcher.fetch_sina_news", return_value=news) as m:
        resp = client.get("/debug/sina-news", headers=valid_auth)
    assert resp.status_code == 200
    data = resp.json()
    assert data["d1_count"] == 5
    assert data["d2_count"] == 5
    assert len(data["d1_sample"]) == 3
    assert m.call_count == 2


def test_debug_sector_stocks_with_mocked_fetcher(client, valid_auth):
    with patch("agent.data_fetcher.fetch_sector_stock_detail", return_value={"stocks": []}) as m:
        resp = client.get("/debug/sector-stocks", headers=valid_auth)
    assert resp.status_code == 200
    assert resp.json() == {"sector": "食品饮料", "detail": {"stocks": []}}
    m.assert_called_once()


def test_debug_stock_all_with_mocked_fetchers(client, valid_auth):
    with patch("agent.data_fetcher.fetch_stock_quote", return_value={"name": "贵州茅台"}), \
         patch("agent.data_fetcher.fetch_stock_kline", return_value=[1, 2, 3, 4, 5]), \
         patch("agent.data_fetcher.fetch_stock_news", return_value=["n1", "n2"]):
        resp = client.get("/debug/stock-all", headers=valid_auth)
    assert resp.status_code == 200
    assert resp.json() == {"quote": True, "kline": 5, "news": 2}


def test_debug_news_count_with_mocked_fetchers(client, valid_auth):
    with patch("agent.data_fetcher.fetch_eastmoney_news", return_value=[1] * 80), \
         patch("agent.data_fetcher.fetch_eastmoney_news_page2", return_value=[1] * 40), \
         patch("agent.data_fetcher.fetch_sina_news", return_value=[1] * 30):
        resp = client.get("/debug/news-count", headers=valid_auth)
    assert resp.status_code == 200
    data = resp.json()
    assert data["em_p1"] == 80
    assert data["em_p2"] == 40
    assert data["sina_d1"] == 30
    assert data["sina_d2"] == 30
    assert data["total_sina"] == 60


def test_debug_pipeline_with_mocked_fetchers(client, valid_auth):
    # 生产 bug 已修复：main.py 的 import 已从 fetch_cls_news 改为实际存在的 fetch_cls_telegraph
    fake_trade_date = datetime(2026, 7, 20)
    with patch("agent.orchestrator._get_latest_trade_date", return_value=fake_trade_date), \
         patch("agent.data_fetcher.fetch_a_share_indices", return_value={"上证指数": {"close": 3900}}), \
         patch("agent.data_fetcher.fetch_shenwan_sectors", return_value=[{"name": "食品饮料"}]), \
         patch("agent.data_fetcher.fetch_fund_flows", return_value={"north": 1.5}), \
         patch("agent.data_fetcher.fetch_global_indices", return_value={"道指": {"close": 44000}}), \
         patch("agent.data_fetcher.fetch_us_macro", return_value={"cpi": 3.0}), \
         patch("agent.data_fetcher.fetch_cls_telegraph", return_value=["a", "b", "c", "d", "e"]):
        resp = client.get("/debug/pipeline", headers=valid_auth)

    assert resp.status_code == 200
    data = resp.json()
    results = data["pipeline_test"]
    assert results["indices"]["count"] == 1
    assert results["indices"]["date_used"] == "20260720"
    assert results["sectors"]["count"] == 1
    assert results["global"]["count"] == 1
    assert results["news_cls"] == "5 items"
    assert data["dates"]["trade_date_used"] == "20260720"


def test_debug_tushare_without_token(client, valid_auth, monkeypatch):
    """TUSHARE_TOKEN 未配置时应返回 no_token，而不是抛异常或泄漏信息。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "")
    resp = client.get("/debug/tushare", headers=valid_auth)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "no_token"
    assert "token" not in json.dumps(data).lower().replace("no_token", "").replace("tushare_token", "")


def test_debug_derivatives_without_token(client, valid_auth, monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "")
    resp = client.get("/debug/derivatives", headers=valid_auth)
    assert resp.status_code == 200
    assert resp.json() == {"status": "no_token"}


def test_debug_macro_without_token(client, valid_auth, monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "")
    resp = client.get("/debug/macro", headers=valid_auth)
    assert resp.status_code == 200
    assert resp.json() == {"status": "no_token"}


def _mock_mcp_session(monkeypatch, tool_call_json):
    """mock 掉端点内部对新浪 MCP 的两次 requests.post（initialize + tools/call）。"""
    init_resp = MagicMock()
    init_resp.headers = {"Mcp-Session-Id": "fake-session-id"}
    init_resp.status_code = 200

    call_resp = MagicMock()
    call_resp.json.return_value = tool_call_json

    post_mock = MagicMock(side_effect=[init_resp, call_resp])
    monkeypatch.setattr("requests.post", post_mock)
    return post_mock


def test_debug_mcp_test_with_mocked_requests(client, valid_auth, monkeypatch):
    monkeypatch.setenv("SINA_MCP_TOKEN", "fake-sina-token")
    post_mock = _mock_mcp_session(monkeypatch, {"result": {"content": []}})
    resp = client.get("/debug/mcp-test", headers=valid_auth)
    assert resp.status_code == 200
    data = resp.json()
    assert data["tool"] == "cnMarketUpdownDistribution"
    assert "response" in data
    assert post_mock.call_count == 2  # initialize + tools/call


def test_debug_hot_with_mocked_requests(client, valid_auth, monkeypatch):
    monkeypatch.setenv("SINA_MCP_TOKEN", "fake-sina-token")
    post_mock = _mock_mcp_session(monkeypatch, {"result": {"content": []}})
    resp = client.get("/debug/hot", headers=valid_auth)
    assert resp.status_code == 200
    assert "text" in resp.json()
    assert post_mock.call_count == 2


def test_debug_mcp_news_with_mocked_requests(client, valid_auth, monkeypatch):
    monkeypatch.setenv("SINA_MCP_TOKEN", "fake-sina-token")
    inner = {"result": {"data": {"data": [{"title": "银行板块新闻一"}, {"title": "银行板块新闻二"}]}}}
    tool_call_json = {"result": {"content": [{"text": json.dumps(inner)}]}}
    _mock_mcp_session(monkeypatch, tool_call_json)
    resp = client.get("/debug/mcp-news", headers=valid_auth)
    assert resp.status_code == 200
    data = resp.json()
    assert data["token_exists"] is True
    assert data["session"] is True
    assert data["count"] == 2
    assert data["sample"][0].startswith("银行板块新闻")
