"""MCP 结果归一化助手 + _mcp_call HTTP 错误归一化的单元测试。

覆盖对象（agent/data_fetcher.py）：
- compact_mcp_result(data) -> dict：递归剔除占位字段、句子边界截断、非 dict 输入包装
- is_mcp_error(data) -> bool：HTTP 错误结构 / status.code 非成功 / data 空 / 全 "--" 占位
- mcp_error_brief(data) -> str：单行人类可读错误摘要
- _mcp_call：HTTP 非 200 返回 {"error": "HTTP_<code>"}，正常返回结构不变

背景回归点：生产上 hkFinanceReportsByIndex 返回
{"result":{"data":[],"status":{"code":11,"msg":"Input error"}}}、
cnCompanyBasicInfo 返回全 "--" 占位字段时，模型识别不出失败反复重试，
最终把原始 JSON dump 给用户。

规则：全部 mock（requests / 环境变量），零网络。
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent import data_fetcher
from agent.data_fetcher import (
    _mcp_call,
    compact_mcp_result,
    is_mcp_error,
    mcp_error_brief,
)


# ════════════════════════════════════════════════════════════════
# 工具：构造 _mcp_call 两步 JSON-RPC 的 mock 序列
# ════════════════════════════════════════════════════════════════

def _init_resp(status=200, with_sid=True):
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"Mcp-Session-Id": "fake-sid"} if with_sid else {}
    return resp


def _call_resp(status=200, payload=None, text=None, json_raises=False):
    """tools/call 响应。payload 为 dict 时包成 content[0].text JSON；
    text 为 str 时原样塞入；json_raises 模拟 .json() 抛错。"""
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {}
    if json_raises:
        resp.json.side_effect = ValueError("not json")
    elif payload is not None:
        resp.json.return_value = {
            "result": {"content": [{"text": json.dumps(payload, ensure_ascii=False)}]}
        }
    elif text is not None:
        resp.json.return_value = {"result": {"content": [{"text": text}]}}
    else:
        resp.json.return_value = {"result": {"content": []}}
    return resp


def _run_mcp_call(resps, monkeypatch=None):
    with patch.object(data_fetcher, "requests") as mreq:
        mreq.post = MagicMock(side_effect=resps)
        return _mcp_call("hkFinanceReportsByIndex", {"symbol": "hk02333"})


# 生产事故现场的两种典型失败载荷
HK_INPUT_ERROR = {"result": {"data": [], "status": {"code": 11, "msg": "Input error"}}}
CN_PLACEHOLDER = {
    "result": {"data": {
        "name": "--", "industry": "--", "pe": "--", "pb": "--",
        "market_value": "--", "list_date": "--",
    }, "status": {"code": 0, "msg": "success"}}
}


# ════════════════════════════════════════════════════════════════
# 1. compact_mcp_result
# ════════════════════════════════════════════════════════════════

class TestCompactMcpResult:

    def test_strips_none_empty_string_dashdash(self):
        raw = {"a": None, "b": "", "c": "--", "d": "  ", "e": "有效值"}
        assert compact_mcp_result(raw) == {"e": "有效值"}

    def test_strips_empty_dict_and_list_recursively(self):
        raw = {
            "result": {
                "data": [],
                "meta": {"page": {}, "rows": []},
                "status": {"code": 0, "msg": "success"},
            },
            "items": [[], {"x": None}],
        }
        # status 是数据本身不是本函数职责（不过滤），但空壳 result/items 应剔除
        assert compact_mcp_result(raw) == {"result": {"status": {"code": 0, "msg": "success"}}}

    def test_keeps_zero_and_false(self):
        raw = {"pct": 0, "flag": False, "price": 0.0, "name": "x"}
        out = compact_mcp_result(raw)
        assert out == {"pct": 0, "flag": False, "price": 0.0, "name": "x"}

    def test_nested_placeholder_records_removed(self):
        out = compact_mcp_result(CN_PLACEHOLDER)
        # 全 "--" 的 data 应整个消失，仅剩 status
        assert out == {"result": {"status": {"code": 0, "msg": "success"}}}

    def test_list_elements_compacted_and_placeholders_dropped(self):
        raw = {"rows": [{"name": "A", "v": "--"}, {"name": "--"}, {"name": "B", "v": 2}]}
        out = compact_mcp_result(raw)
        assert out == {"rows": [{"name": "A"}, {"name": "B", "v": 2}]}

    def test_long_string_truncated_at_sentence_boundary(self):
        long_text = "第一句完整的话。" * 60 + "尾巴" * 200
        out = compact_mcp_result({"content": long_text})
        c = out["content"]
        assert len(c) <= 301  # 300 + 省略号
        assert c.endswith("…")
        # 在标点边界截断：截断点之后紧跟的原文第 1 个字符必须是句末标点
        body = c[:-1]
        assert long_text.startswith(body)
        assert long_text[len(body)] == "。"

    def test_short_string_verbatim(self):
        assert compact_mcp_result({"t": "短句"}) == {"t": "短句"}

    def test_non_dict_input_wrapped(self):
        assert compact_mcp_result([{"a": 1}, {"b": "--"}]) == {"items": [{"a": 1}]}
        assert compact_mcp_result("hello") == {"value": "hello"}
        assert compact_mcp_result(123) == {"value": 123}

    def test_placeholder_only_input_returns_empty_dict(self):
        assert compact_mcp_result({}) == {}
        assert compact_mcp_result([]) == {}
        assert compact_mcp_result("--") == {}
        assert compact_mcp_result(None) == {}
        assert compact_mcp_result({"a": "--", "b": []}) == {}

    def test_hk_input_error_structure_preserved_for_diagnosis(self):
        """错误现场数据（code=11）本身要保留，供 is_mcp_error/mcp_error_brief 诊断。"""
        out = compact_mcp_result(HK_INPUT_ERROR)
        assert out == {"result": {"status": {"code": 11, "msg": "Input error"}}}


# ════════════════════════════════════════════════════════════════
# 2. is_mcp_error
# ════════════════════════════════════════════════════════════════

class TestIsMcpError:

    def test_http_error_struct(self):
        assert is_mcp_error({"error": "HTTP_502"}) is True
        assert is_mcp_error({"error": "HTTP_401"}) is True

    def test_status_code_non_success(self):
        assert is_mcp_error(HK_INPUT_ERROR) is True
        # 顶层 status 形态
        assert is_mcp_error({"data": [{"x": 1}], "status": {"code": 11, "msg": "Input error"}}) is True
        # 字符串 code
        assert is_mcp_error({"status": {"code": "500"}}) is True

    def test_status_code_success_with_data_is_not_error(self):
        ok = {"result": {"data": [{"name": "鲟龙科技", "pe": "15.2"}],
                         "status": {"code": 0, "msg": "success"}}}
        assert is_mcp_error(ok) is False
        assert is_mcp_error({"data": [{"x": 1}], "status": {"code": 200}}) is False

    def test_empty_data_variants(self):
        assert is_mcp_error({"data": []}) is True
        assert is_mcp_error({"data":[]}) is True  # noqa: E201 — 显式回归无空格形态
        assert is_mcp_error({"result": {"data": []}}) is True
        assert is_mcp_error({"result": {"data": {"s_list": []}}}) is True
        assert is_mcp_error({"result": {"data": [], "status": {"code": 0, "msg": "success"}}}) is True

    def test_all_placeholder_fields(self):
        assert is_mcp_error(CN_PLACEHOLDER) is True
        assert is_mcp_error({"name": "--", "pe": "--"}) is True

    def test_normal_payloads_are_not_error(self):
        assert is_mcp_error({"raw": "a,b,c,d", "type": "csv"}) is False
        assert is_mcp_error({"data": {"price": "10.5", "pct": 0}}) is False
        assert is_mcp_error({"result": {"data": {"data": [{"title": "新闻"}]}}}) is False
        assert is_mcp_error([{"a": 1}]) is False

    def test_trivial_empty_and_scalar(self):
        assert is_mcp_error({}) is True
        assert is_mcp_error([]) is True
        assert is_mcp_error(None) is True
        assert is_mcp_error("--") is True
        assert is_mcp_error("") is True
        assert is_mcp_error("正常文本") is False
        assert is_mcp_error(0) is False      # 数字标量视为有效数据
        assert is_mcp_error(3.14) is False

    def test_empty_raw_csv_is_error(self):
        assert is_mcp_error({"raw": "", "type": "csv"}) is True

    def test_error_key_falsy_not_flagged(self):
        """error 字段存在但为 None/空串 不算错误（有有效数据时）。"""
        assert is_mcp_error({"error": None, "data": {"x": 1}}) is False
        assert is_mcp_error({"error": "", "data": {"x": 1}}) is False


# ════════════════════════════════════════════════════════════════
# 3. mcp_error_brief
# ════════════════════════════════════════════════════════════════

class TestMcpErrorBrief:

    def test_input_error_with_tool_name(self):
        data = dict(HK_INPUT_ERROR)
        data["tool"] = "hkFinanceReportsByIndex"
        brief = mcp_error_brief(data)
        assert brief == "hkFinanceReportsByIndex 参数错误(code=11)"

    def test_input_error_without_tool_name(self):
        brief = mcp_error_brief(HK_INPUT_ERROR)
        assert "参数错误" in brief
        assert "code=11" in brief

    def test_http_error(self):
        assert "502" in mcp_error_brief({"error": "HTTP_502"})
        assert "HTTP" in mcp_error_brief({"error": "HTTP_502"})

    def test_empty_data_brief(self):
        assert "无有效数据" in mcp_error_brief({"result": {"data": []}})

    def test_placeholder_fields_brief(self):
        assert "占位" in mcp_error_brief(CN_PLACEHOLDER)

    def test_single_line(self):
        for data in (HK_INPUT_ERROR, CN_PLACEHOLDER, {"error": "HTTP_500"}, {"data": []}):
            assert "\n" not in mcp_error_brief(data)

    def test_non_error_returns_empty_string(self):
        assert mcp_error_brief({"data": {"price": "10.5"}}) == ""
        assert mcp_error_brief({"raw": "a,b", "type": "csv"}) == ""


# ════════════════════════════════════════════════════════════════
# 4. _mcp_call：HTTP 错误归一化 + 正常结构回归
# ════════════════════════════════════════════════════════════════

class TestMcpCallHttpNormalization:

    def test_no_token_returns_empty(self, monkeypatch):
        monkeypatch.setenv("SINA_MCP_TOKEN", "")
        with patch.object(data_fetcher, "requests") as mreq:
            assert _mcp_call("anyTool", {}) == {}
            mreq.post.assert_not_called()

    def test_initialize_http_500(self):
        out = _run_mcp_call([_init_resp(status=500, with_sid=False)])
        assert out == {"error": "HTTP_500"}

    def test_tools_call_http_502(self):
        out = _run_mcp_call([_init_resp(), _call_resp(status=502)])
        assert out == {"error": "HTTP_502"}

    def test_tools_call_http_429(self):
        out = _run_mcp_call([_init_resp(), _call_resp(status=429)])
        assert out == {"error": "HTTP_429"}

    def test_http_error_struct_compact_and_detectable(self):
        """归一化结构可被助手识别：is_mcp_error=True 且 brief 单行。"""
        out = _run_mcp_call([_init_resp(), _call_resp(status=503)])
        assert is_mcp_error(out) is True
        assert "503" in mcp_error_brief(out)
        assert compact_mcp_result(out) == out  # 已是紧凑结构

    def test_error_struct_breaks_mcp_news_pagination(self):
        """fetch_mcp_news 对 {"error": ...} 不 crash 且停止翻页（兼容现有调用方）。"""
        with patch.object(data_fetcher, "_mcp_call", return_value={"error": "HTTP_502"}):
            assert data_fetcher.fetch_mcp_news("银行", 60) == []

    def test_normal_json_path_unchanged(self):
        """JSON 正常返回结构逐字节保持历史行为。"""
        payload = {"result": {"data": [{"name": "鲟龙科技", "pe": "15.2"}],
                              "status": {"code": 0, "msg": "success"}}}
        out = _run_mcp_call([_init_resp(), _call_resp(payload=payload)])
        assert out == payload

    def test_status_code_error_body_preserved(self):
        """status.code=11 的返回体原样保留（不改写），由 is_mcp_error 识别。"""
        out = _run_mcp_call([_init_resp(), _call_resp(payload=HK_INPUT_ERROR)])
        assert out == HK_INPUT_ERROR
        assert is_mcp_error(out) is True

    def test_csv_var_text_path_unchanged(self):
        out = _run_mcp_call([_init_resp(), _call_resp(text='var xxx="a,b,c,d"')])
        assert out == {"raw": "a,b,c,d", "type": "csv"}
        assert is_mcp_error(out) is False

    def test_plain_text_path_unchanged(self):
        out = _run_mcp_call([_init_resp(), _call_resp(text="纯文本响应不是JSON")])
        assert out == {"raw": "纯文本响应不是JSON", "type": "text"}

    def test_empty_content_returns_empty(self):
        out = _run_mcp_call([_init_resp(), _call_resp()])
        assert out == {}

    def test_no_session_id_returns_empty(self):
        out = _run_mcp_call([_init_resp(with_sid=False)])
        assert out == {}

    def test_network_exception_returns_empty(self):
        with patch.object(data_fetcher, "requests") as mreq:
            mreq.post = MagicMock(side_effect=ConnectionError("boom"))
            assert _mcp_call("anyTool", {}) == {}

    def test_mock_without_status_code_treated_as_success(self):
        """回归防护：裸 MagicMock 响应（未设 status_code，现有测试的 mock 风格）
        不得被误判为 HTTP 错误。"""
        init = MagicMock()
        init.headers = {"Mcp-Session-Id": "fake-sid"}
        call = MagicMock()
        call.headers = {}
        payload = {"result": {"data": [{"x": 1}]}}
        call.json.return_value = {
            "result": {"content": [{"text": json.dumps(payload, ensure_ascii=False)}]}
        }
        out = _run_mcp_call([init, call])
        assert out == payload

    def test_real_status_200_passes(self):
        payload = {"data": [{"x": 1}]}
        out = _run_mcp_call([_init_resp(status=200), _call_resp(status=200, payload=payload)])
        assert out == payload
