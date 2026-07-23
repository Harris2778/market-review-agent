"""DSML 文本工具调用解析与出口剥除测试。

背景：DeepSeek 偶发把工具调用以纯文本 DSML 标记输出
（<｜｜DSML｜｜tool_calls>…</｜｜DSML｜｜tool_calls>，｜ 为全角竖线 U+FF5C），
而非结构化 message.tool_calls。未处理时原始标记直接泄漏给用户
（生产事故：问「介绍一下鲟龙科技公司」收到一整段 DSML 标记）。

覆盖：
1. _parse_dsml_tool_calls：完整块解析（真实事故样本）、参数类型还原
   （string="true" 保留字符串 / string="false" JSON 解析数字布尔 / 解析失败回退）、
   多 invoke、正文与 DSML 混排时正文保留。
2. _strip_dsml：完整块删除、未闭合块删除、孤立标记行删除、无 DSML 原文不动。
3. 编排集成：_agent_query 第一轮返回 DSML 文本 → 解析成合成工具调用并真正
   执行工具 → 最终答复不含任何 DSML 标记；assistant 回显消息携带 tool_calls
   且 tool 结果消息 tool_call_id 配对。
4. 出口兜底：超轮强制成文与 _call_llm 出口均剥除残留 DSML 标记。

规则（与项目其他测试一致）：
- 所有外部调用全部 mock，绝不发起真实网络请求。
- 无 pytest-asyncio，异步函数一律用 asyncio.run 驱动。
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import agent.orchestrator as orchestrator
from agent.orchestrator import (
    MarketReviewAgent,
    _parse_dsml_tool_calls,
    _strip_dsml,
)

# 用户生产事故中的真实 DSML 样本（一字不差）
REAL_WORLD_DSML = (
    '<｜｜DSML｜｜tool_calls>\n'
    '<｜｜DSML｜｜invoke name="get_stock_kline">\n'
    '<｜｜DSML｜｜parameter name="market" string="true">hk</｜｜DSML｜｜parameter>\n'
    '<｜｜DSML｜｜parameter name="symbol" string="true">06715</｜｜DSML｜｜parameter>\n'
    '<｜｜DSML｜｜parameter name="days" string="false">30</｜｜DSML｜｜parameter>\n'
    '</｜｜DSML｜｜invoke>\n'
    '</｜｜DSML｜｜tool_calls>'
)


def _make_agent() -> MarketReviewAgent:
    agent = MarketReviewAgent()
    agent.client = MagicMock()
    return agent


def _completion_text(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=None),
            finish_reason="stop",
        )]
    )


# ════════════════════════════════════════════════════════════════
# 1. _parse_dsml_tool_calls
# ════════════════════════════════════════════════════════════════

class TestParseDsmlToolCalls:

    def test_real_world_sample(self):
        """生产事故样本：get_stock_kline(hk, 06715, 30) 完整还原。"""
        calls, remainder = _parse_dsml_tool_calls(REAL_WORLD_DSML)
        assert calls == [("get_stock_kline", {
            "market": "hk",
            "symbol": "06715",
            "days": 30,  # string="false" → JSON 数字
        })]
        assert isinstance(calls[0][1]["days"], int)
        assert remainder == "", f"纯 DSML 内容剔除后应为空: {remainder!r}"

    def test_string_true_keeps_leading_zero(self):
        """string="true" 的 06715 必须保留前导零（港股代码）。"""
        calls, _ = _parse_dsml_tool_calls(REAL_WORLD_DSML)
        assert calls[0][1]["symbol"] == "06715"

    def test_bool_and_float_params(self):
        text = (
            '<｜｜DSML｜｜tool_calls>'
            '<｜｜DSML｜｜invoke name="t">'
            '<｜｜DSML｜｜parameter name="ratio" string="false">0.75</｜｜DSML｜｜parameter>'
            '<｜｜DSML｜｜parameter name="flag" string="false">true</｜｜DSML｜｜parameter>'
            '</｜｜DSML｜｜invoke>'
            '</｜｜DSML｜｜tool_calls>'
        )
        calls, _ = _parse_dsml_tool_calls(text)
        assert calls == [("t", {"ratio": 0.75, "flag": True})]

    def test_non_json_false_param_falls_back_to_string(self):
        text = (
            '<｜｜DSML｜｜tool_calls>'
            '<｜｜DSML｜｜invoke name="t">'
            '<｜｜DSML｜｜parameter name="x" string="false">abc</｜｜DSML｜｜parameter>'
            '</｜｜DSML｜｜invoke>'
            '</｜｜DSML｜｜tool_calls>'
        )
        calls, _ = _parse_dsml_tool_calls(text)
        assert calls == [("t", {"x": "abc"})]

    def test_multiple_invokes(self):
        text = (
            '<｜｜DSML｜｜tool_calls>'
            '<｜｜DSML｜｜invoke name="a">'
            '<｜｜DSML｜｜parameter name="p" string="true">1</｜｜DSML｜｜parameter>'
            '</｜｜DSML｜｜invoke>'
            '<｜｜DSML｜｜invoke name="b">'
            '<｜｜DSML｜｜parameter name="q" string="true">2</｜｜DSML｜｜parameter>'
            '</｜｜DSML｜｜invoke>'
            '</｜｜DSML｜｜tool_calls>'
        )
        calls, _ = _parse_dsml_tool_calls(text)
        assert calls == [("a", {"p": "1"}), ("b", {"q": "2"})]

    def test_mixed_prose_preserved(self):
        """DSML 块前后的正文文本必须保留在 remainder。"""
        text = "我先查一下数据。\n" + REAL_WORLD_DSML + "\n请稍等。"
        calls, remainder = _parse_dsml_tool_calls(text)
        assert len(calls) == 1
        assert "我先查一下数据。" in remainder
        assert "请稍等。" in remainder
        assert "DSML" not in remainder

    def test_no_dsml_passthrough(self):
        calls, remainder = _parse_dsml_tool_calls("普通回答，没有标记。")
        assert calls == []
        assert remainder == "普通回答，没有标记。"

    def test_empty_and_none(self):
        assert _parse_dsml_tool_calls("") == ([], "")
        assert _parse_dsml_tool_calls(None) == ([], "")


# ════════════════════════════════════════════════════════════════
# 2. _strip_dsml
# ════════════════════════════════════════════════════════════════

class TestStripDsml:

    def test_full_block_removed(self):
        assert _strip_dsml(REAL_WORLD_DSML) == ""

    def test_unclosed_block_removed(self):
        """未闭合的 tool_calls 块（截断输出）也要剥到结尾。"""
        text = '正文开头。\n<｜｜DSML｜｜tool_calls>\n<｜｜DSML｜｜invoke name="x">'
        out = _strip_dsml(text)
        assert out == "正文开头。"

    def test_orphan_marker_lines_removed(self):
        """孤立的闭合标记行（块外残留）逐行剔除。"""
        text = '回答正文。\n</｜｜DSML｜｜invoke>\n</｜｜DSML｜｜tool_calls>\n第二行正文。'
        out = _strip_dsml(text)
        assert "DSML" not in out
        assert "回答正文。" in out
        assert "第二行正文。" in out

    def test_halfwidth_pipe_variant(self):
        """兼容半角 || 变体。"""
        text = '<||DSML||tool_calls><||DSML||invoke name="t"></||DSML||invoke></||DSML||tool_calls>'
        assert _strip_dsml(text) == ""

    def test_no_dsml_unchanged(self):
        assert _strip_dsml("正常文本") == "正常文本"
        assert _strip_dsml("") == ""
        assert _strip_dsml(None) == ""


# ════════════════════════════════════════════════════════════════
# 3. 编排集成：DSML 文本 → 合成工具调用 → 真正执行 → 答复无标记
# ════════════════════════════════════════════════════════════════

class TestAgentLoopDsmlIntegration:

    def test_dsml_round_executes_tool_and_clean_answer(self):
        agent = _make_agent()
        agent.client.chat.completions.create = AsyncMock(side_effect=[
            _completion_text(REAL_WORLD_DSML),      # 第一轮：DSML 文本工具调用
            _completion_text("鲟龙科技近30日K线平稳。"),  # 第二轮：收敛成文
        ])
        exec_mock = MagicMock(return_value={"ok": True, "data": {"close": 12.3}})
        with patch("agent.orchestrator.execute_tool", exec_mock):
            result = asyncio.run(
                agent._agent_query("介绍一下鲟龙科技公司的相关情况", stream=False)
            )
        # 工具被真正执行，且参数正确还原
        exec_mock.assert_called_once_with(
            "get_stock_kline", {"market": "hk", "symbol": "06715", "days": 30}
        )
        # 最终答复不含任何 DSML 标记
        assert "DSML" not in result["content"]
        assert "｜｜" not in result["content"]
        assert "鲟龙科技" in result["content"]

    def test_assistant_echo_pairs_tool_call_ids(self):
        """合成 assistant 回显消息携带 tool_calls，tool 结果 tool_call_id 配对。"""
        agent = _make_agent()
        agent.client.chat.completions.create = AsyncMock(side_effect=[
            _completion_text(REAL_WORLD_DSML),
            _completion_text("成文。"),
        ])
        exec_mock = MagicMock(return_value={"ok": True, "data": {}})
        with patch("agent.orchestrator.execute_tool", exec_mock):
            asyncio.run(agent._agent_query("查一下", stream=False))
        second_messages = agent.client.chat.completions.create.await_args_list[1].kwargs["messages"]
        assistant_msgs = [m for m in second_messages if m.get("role") == "assistant" and m.get("tool_calls")]
        tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        assert len(assistant_msgs) == 1, f"应有一条带 tool_calls 的 assistant 消息: {second_messages}"
        assert len(tool_msgs) == 1
        tc = assistant_msgs[0]["tool_calls"][0]
        assert tc["function"]["name"] == "get_stock_kline"
        assert json.loads(tc["function"]["arguments"]) == {
            "market": "hk", "symbol": "06715", "days": 30,
        }
        assert tool_msgs[0]["tool_call_id"] == tc["id"]
        # assistant 回显的 content 不得含 DSML 残留
        assert not assistant_msgs[0]["content"] or "DSML" not in assistant_msgs[0]["content"]


# ════════════════════════════════════════════════════════════════
# 4. 出口兜底：任何残留 DSML 都到不了用户
# ════════════════════════════════════════════════════════════════

class TestExitSafetyNet:

    def test_final_text_exit_strips_orphan_dsml(self):
        """工具循环收敛轮的纯文本若混入孤立 DSML 行，出口剥除。"""
        agent = _make_agent()
        agent.client.chat.completions.create = AsyncMock(side_effect=[
            _completion_text("正常回答。\n</｜｜DSML｜｜invoke>"),
        ])
        result = asyncio.run(agent._agent_query("随便问", stream=False))
        assert "DSML" not in result["content"]
        assert "正常回答。" in result["content"]

    def test_overround_exit_strips_dsml(self):
        """超轮强制成文出口同样剥除 DSML。"""
        agent = _make_agent()
        # 8 轮都返回 DSML 工具调用（耗尽轮数）+ 强制成文轮返回带残留标记文本
        rounds = [_completion_text(REAL_WORLD_DSML)] * agent._AGENT_MAX_ROUNDS
        rounds.append(_completion_text("强制成文。\n<｜｜DSML｜｜tool_calls>未闭合"))
        agent.client.chat.completions.create = AsyncMock(side_effect=rounds)
        exec_mock = MagicMock(return_value={"ok": True, "data": {}})
        with patch("agent.orchestrator.execute_tool", exec_mock):
            result = asyncio.run(agent._agent_query("查一下", stream=False))
        assert "DSML" not in result["content"]
        assert "强制成文。" in result["content"]

    def test_call_llm_exit_strips_dsml(self):
        """_call_llm 普通对话出口剥除 DSML。"""
        agent = _make_agent()
        agent.client.chat.completions.create = AsyncMock(side_effect=[
            _completion_text("闲聊回答。\n" + REAL_WORLD_DSML),
        ])
        result = asyncio.run(agent._chat("你好", stream=False))
        assert "DSML" not in result["content"]
        assert "闲聊回答。" in result["content"]
