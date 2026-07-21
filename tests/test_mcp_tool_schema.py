"""tests/test_mcp_tool_schema.py — MCP 工具 schema 透传专项。

根因：_generic_mcp 的 DeepSeek function calling 链路丢弃了 MCP 工具
inputSchema 的参数描述/枚举/required，模型不知道 cnFinanceReportsFull 的
source 只能填 lrb/fzb/llb/gjzb/zxzb，瞎填导致 code=11 Input error。

覆盖：
1. get_mcp_tools 缓存条目保留完整 schema（properties 含 description/enum/
   type、required 列表），name/desc/params 旧键不变；描述截断 300 字。
2. _generic_mcp ds_tools 用真实 schema 构建 parameters。
3. 旧格式缓存条目（无 schema 键 / schema.properties 为空）降级到原 params
   逻辑。

规则：全部 mock，零网络。
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from agent import data_fetcher
from agent.orchestrator import MarketReviewAgent


# ── fixtures ─────────────────────────────────────────────────

LONG_DESC = "报表来源：" + "很" * 400  # >300 字，验证截断

FAKE_TOOLS = [
    {
        "name": "cnFinanceReportsFull",
        "description": "获取A股完整财务报表。" + "长" * 300,  # >200 字
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码，如 sz002956"},
                "date": {"type": "string", "description": "报告期，如 2025-03-31"},
                "source": {
                    "type": "string",
                    "description": LONG_DESC,
                    "enum": ["lrb", "fzb", "llb", "gjzb", "zxzb"],
                },
            },
            "required": ["symbol", "source", "ghost"],  # ghost 不在 properties
        },
    },
    {
        "name": "noSchemaTool",
        "description": "无 schema 的工具",
        # 没有 inputSchema
    },
]


def _reset_cache():
    data_fetcher._mcp_tools_cache = None


def _fake_completion(content: str = "正文", finish_reason: str = "stop", tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=tool_calls),
            finish_reason=finish_reason,
        )]
    )


# ── 1. get_mcp_tools schema 保留 ─────────────────────────────

class TestGetMcpToolsSchema:

    def _run_get(self):
        _reset_cache()
        resp_init = MagicMock()
        resp_init.headers = {"Mcp-Session-Id": "sid-1"}
        resp_list = MagicMock()
        resp_list.json.return_value = {"result": {"tools": FAKE_TOOLS}}
        with patch.object(data_fetcher, "_env", return_value="fake-token"), \
             patch("agent.data_fetcher.requests.post",
                   side_effect=[resp_init, resp_list]):
            return data_fetcher.get_mcp_tools()

    def teardown_method(self):
        _reset_cache()

    def test_schema_preserved_with_enum_and_required(self):
        tools = self._run_get()
        assert len(tools) == 2
        t = tools[0]
        # 旧键不变，兼容其他调用方
        assert t["name"] == "cnFinanceReportsFull"
        assert len(t["desc"]) == 200
        assert t["params"] == ["symbol", "date", "source"]
        # 新 schema 键：枚举、required 完整
        schema = t["schema"]
        assert schema["properties"]["source"]["enum"] == ["lrb", "fzb", "llb", "gjzb", "zxzb"]
        assert schema["properties"]["symbol"]["description"] == "股票代码，如 sz002956"
        assert schema["properties"]["symbol"]["type"] == "string"
        # ghost 不在 properties 中，从 required 剔除
        assert schema["required"] == ["symbol", "source"]

    def test_param_description_truncated_to_300(self):
        tools = self._run_get()
        desc = tools[0]["schema"]["properties"]["source"]["description"]
        assert len(desc) == 300
        assert desc == LONG_DESC[:300]

    def test_tool_without_input_schema_gets_empty_schema(self):
        tools = self._run_get()
        t = tools[1]
        assert t["name"] == "noSchemaTool"
        assert t["params"] == []
        assert t["schema"] == {"properties": {}, "required": []}


# ── 2/3. _generic_mcp ds_tools 构建 ─────────────────────────

class TestDsToolsBuild:

    def _captured_tools(self, mcp_tools):
        """跑一次 _generic_mcp（首轮即 stop），返回传给 DeepSeek 的 tools。"""
        agent = MarketReviewAgent()
        agent.client = MagicMock()
        create = AsyncMock(return_value=_fake_completion("回答"))
        agent.client.chat.completions.create = create
        with patch("agent.data_fetcher.get_mcp_tools", MagicMock(return_value=mcp_tools)), \
             patch("agent.data_fetcher._mcp_call", MagicMock(return_value={})):
            asyncio.run(agent._generic_mcp("查询财务指标", False))
        kwargs = create.call_args.kwargs
        return kwargs["tools"]

    def test_real_schema_used_when_present(self):
        mcp_tools = [{
            "name": "cnFinanceReportsFull",
            "desc": "获取A股完整财务报表",
            "params": ["symbol", "source"],
            "schema": {
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码"},
                    "source": {
                        "type": "string",
                        "description": "报表来源",
                        "enum": ["lrb", "fzb", "llb", "gjzb", "zxzb"],
                    },
                },
                "required": ["symbol", "source"],
            },
        }]
        tools = self._captured_tools(mcp_tools)
        fn = tools[0]["function"]
        params = fn["parameters"]
        assert params["type"] == "object"
        # 真实描述与枚举透传，不再是「描述=参数名」
        assert params["properties"]["source"]["description"] == "报表来源"
        assert params["properties"]["source"]["enum"] == ["lrb", "fzb", "llb", "gjzb", "zxzb"]
        assert params["required"] == ["symbol", "source"]

    def test_legacy_entry_without_schema_falls_back(self):
        """旧缓存格式（无 schema 键）→ 退回 params 逻辑，描述=参数名。"""
        mcp_tools = [{"name": "globalStockSearchSymbols", "desc": "搜索股票代码",
                      "params": ["keyword"]}]
        tools = self._captured_tools(mcp_tools)
        params = tools[0]["function"]["parameters"]
        assert params == {
            "type": "object",
            "properties": {"keyword": {"type": "string", "description": "keyword"}},
        }

    def test_empty_schema_properties_also_falls_back(self):
        """schema 键存在但 properties 为空 → 同样走降级路径。"""
        mcp_tools = [{"name": "noSchemaTool", "desc": "无参数", "params": [],
                      "schema": {"properties": {}, "required": []}}]
        tools = self._captured_tools(mcp_tools)
        params = tools[0]["function"]["parameters"]
        assert params == {"type": "object", "properties": {}}

    def test_model_can_fill_enum_from_schema_end_to_end(self):
        """新链路下模型带着枚举信息调 cnFinanceReportsFull（mock 模型按 schema 填参）。"""
        mcp_tools = [{
            "name": "cnFinanceReportsFull", "desc": "获取A股完整财务报表",
            "params": ["symbol", "source"],
            "schema": {
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码"},
                    "source": {"type": "string", "description": "报表来源",
                               "enum": ["lrb", "fzb", "llb", "gjzb", "zxzb"]},
                },
                "required": ["symbol", "source"],
            },
        }]
        agent = MarketReviewAgent()
        agent.client = MagicMock()
        tc = SimpleNamespace(
            id="call_1", type="function",
            function=SimpleNamespace(
                name="cnFinanceReportsFull",
                arguments=json.dumps({"symbol": "sz002956", "source": "gjzb"},
                                     ensure_ascii=False),
            ),
        )
        agent.client.chat.completions.create = AsyncMock(side_effect=[
            _fake_completion(content=None, finish_reason="tool_calls", tool_calls=[tc]),
            _fake_completion(content="营业总收入 1.23 亿"),
        ])
        mcp_call = MagicMock(return_value={"result": {"data": [{"营业总收入": "1.23亿"}]}})
        with patch("agent.data_fetcher.get_mcp_tools", MagicMock(return_value=mcp_tools)), \
             patch("agent.data_fetcher._mcp_call", mcp_call):
            out = asyncio.run(agent._generic_mcp("查西麦食品财务指标", False))
        # _mcp_call 收到的参数带合法枚举值
        mcp_call.assert_called_once_with("cnFinanceReportsFull",
                                         {"symbol": "sz002956", "source": "gjzb"})
        assert "营业总收入" in out["content"]
