"""数字溯源校验层 orchestrator 接入测试（tests/test_orchestrator_validators.py）。

覆盖第三波『输出后数字校验层』在 agent/orchestrator.py 的三处挂载点：

1. 非流式深挖路径（_sector_deep_dive stream=False → _critique_and_revise）：
   validators 检出违规时，critique 调用的 user 消息末尾包含
   「【确定性校验结果】」段落与违规数字原文；无违规时消息不追加该段落。
2. 降级路径：orchestrator.validators 为 None（模拟 validators.py import 失败）
   或 find_unsourced_numbers 抛异常时，critique 流程照常执行、user 消息
   不含校验段落、不向上抛异常。
3. 流式路径（_sector_deep_dive 与 _agent_query 的 stream 分支）：
   log-only 接入——仅 logger.warning 记录违规数量与前几条，
   不修改流式输出、不阻塞，validators 异常也不影响输出。

规则（与项目其他测试一致）：
- 所有外部调用全部 mock（DeepSeek 客户端 / 数据采集 / execute_tool），
  绝不发起真实网络请求。
- 无 pytest-asyncio，异步函数一律用 asyncio.run 驱动。
- fixture 与构造 helper 全部写在本文件内，不依赖其他测试文件。
"""

import asyncio
import json
import logging
from contextlib import ExitStack
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agent.data_fetcher as data_fetcher  # noqa: F401  （对齐项目测试的显式导入习惯）
import agent.orchestrator as orchestrator
import agent.validators as validators_mod
from agent.orchestrator import MarketReviewAgent

# 固定交易日，避免 _get_latest_trade_date 触达真实 tushare
FIXED_TRADE_DATE = datetime(2025, 1, 10)  # 周五


# ─────────────────────────────────────────────
# 构造 helper（本文件自包含）
# ─────────────────────────────────────────────

def _make_agent() -> MarketReviewAgent:
    """构造一个 agent，DeepSeek 客户端替换为 mock，防止任何真实 HTTP 调用。"""
    agent = MarketReviewAgent()
    agent.client = MagicMock()
    return agent


def _completion_text(text: str):
    """伪造纯文本的 create 返回（草稿 / 审查结论 / 终稿）。"""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=None),
            finish_reason="stop",
        )]
    )


def _make_tool_call(name: str, arguments: dict, call_id: str):
    """伪造 OpenAI 格式的 tool_call 对象。"""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
    )


def _completion_with_tool_calls(tool_calls):
    """伪造带 tool_calls 的 create 返回（模型要求调用工具）。"""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=None, tool_calls=tool_calls),
            finish_reason="tool_calls",
        )]
    )


async def _fake_chunk_stream(*pieces):
    """伪造流式 chunk 序列（无 finish_reason，一轮结束）。"""
    for p in pieces:
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=p))]
        )


def _sector_deep_dive_patches(stack: ExitStack) -> None:
    """板块深挖数据采集侧的全部 mock（接线方式对齐 test_agent_loop）。"""
    stack.enter_context(patch(
        "agent.orchestrator._get_latest_trade_date",
        return_value=FIXED_TRADE_DATE,
    ))
    stack.enter_context(patch(
        "agent.orchestrator.collect_market_snapshot",
        AsyncMock(return_value=MagicMock(name="snapshot")),
    ))
    stack.enter_context(patch(
        "agent.orchestrator.format_market_data_for_prompt", return_value="DATA"
    ))
    # 估值/资金/景气度附加数据：_fetch_sector_extras 内局部 import，
    # 因此 patch data_fetcher 命名空间（函数已存在，不用 create=True）。
    # 数字 10.0 / 1.0 与草稿中的 999.99 距离远超容差，不会误命中。
    stack.enter_context(patch(
        "agent.data_fetcher.fetch_sector_valuation",
        MagicMock(return_value={"pe": 10.0}),
    ))
    stack.enter_context(patch(
        "agent.data_fetcher.fetch_sector_moneyflow",
        MagicMock(return_value={"main_net": 1.0}),
    ))
    stack.enter_context(patch(
        "agent.data_fetcher.fetch_sector_earnings",
        MagicMock(return_value={"note": ""}),
    ))


def _import_tools():
    """导入第二波 agent.tools 模块（已由并行代理交付）。"""
    import agent.tools as tools_mod
    return tools_mod


def _user_msg_of(call_kwargs: dict) -> str:
    """从 create 调用 kwargs 的 messages 中取 user 消息文本。"""
    messages = call_kwargs["messages"]
    user_msgs = [m for m in messages if m.get("role") == "user"]
    assert user_msgs, f"messages 中缺少 user 消息: {messages!r}"
    return user_msgs[-1]["content"]


# ─────────────────────────────────────────────
# 测试用草稿/上下文
# ─────────────────────────────────────────────

# 无数字填充句（重复拉长用，避免引入额外数字干扰违规计数）
_PAD = "煤炭板块今日整体平稳，估值处于历史中枢附近，资金观望情绪较浓，短线缺乏新增催化。"

# 长草稿（>500 字），含一个必然无出处的数字 999.99 亿
LONG_DRAFT_UNSOURCED = (
    "【草稿】煤炭板块日报：板块全天成交 999.99 亿，主力资金分歧明显。"
    + _PAD * 13
)

# 长草稿（>500 字），数字 3891.25 / 1234.56 均能在 SOURCED_CONTEXT 找到出处
SOURCED_CONTEXT = "上证指数 3891.25 点，成交额 1234.56 亿。"
LONG_DRAFT_SOURCED = (
    "市场整体平稳，上证指数收报 3891.25 点，全天成交 1234.56 亿，观望情绪浓厚。" * 13
)

assert len(LONG_DRAFT_UNSOURCED) > 500, "测试草稿必须超过 500 字审查护栏"
assert len(LONG_DRAFT_SOURCED) > 500, "测试草稿必须超过 500 字审查护栏"


# ════════════════════════════════════════════════════════════════
# 0. 接线冒烟：validators 已挂到 orchestrator 顶部
# ════════════════════════════════════════════════════════════════


class TestValidatorsWiring:
    def test_validators_imported_in_orchestrator(self):
        """orchestrator 顶部 try/except import 成功时，validators 应为真实模块。"""
        assert orchestrator.validators is validators_mod, (
            "agent.orchestrator.validators 未接线到 agent.validators——"
            "若 validators.py 已交付，import 失败应如实暴露"
        )
        assert hasattr(orchestrator.validators, "find_unsourced_numbers")
        assert hasattr(orchestrator.validators, "format_violations_for_critique")


# ════════════════════════════════════════════════════════════════
# 1. 非流式深挖路径：critique user 消息携带确定性校验段落
# ════════════════════════════════════════════════════════════════


class TestCritiqueValidatorsAppendix:
    """非流式 _sector_deep_dive / _critique_and_revise 的校验段落追加。"""

    def test_deep_dive_critique_message_contains_validator_section(self):
        """草稿含无出处数字 → critique 调用的 user 消息末尾含【确定性校验结果】段落。"""
        agent = _make_agent()
        # 第 1 次 create 返回含无出处数字的长草稿；第 2 次（critique）返回「通过」
        agent.client.chat.completions.create = AsyncMock(
            side_effect=[
                _completion_text(LONG_DRAFT_UNSOURCED),
                _completion_text("通过"),
            ]
        )

        with ExitStack() as stack:
            _sector_deep_dive_patches(stack)
            result = asyncio.run(agent._sector_deep_dive("煤炭", stream=False))

        assert isinstance(result, dict), f"应返回 dict: {result!r}"
        create = agent.client.chat.completions.create
        assert create.await_count == 2, (
            f"草稿 + critique（通过→跳过修正）应恰好 2 次 LLM 调用，实际 {create.await_count}"
        )
        # 第 2 次调用即 critique：system 为 CRITIQUE_PROMPT，user 末尾追加校验段落
        critique_kwargs = create.await_args_list[1].kwargs
        user_msg = _user_msg_of(critique_kwargs)
        assert "【数据上下文】" in user_msg and "【待审查初稿】" in user_msg, (
            f"critique user 消息结构异常: {user_msg[:200]!r}"
        )
        assert "【确定性校验结果】" in user_msg, (
            f"critique user 消息缺少确定性校验段落: {user_msg[-400:]!r}"
        )
        assert "以下数字在数据上下文中未找到出处，审查时必须要求删除或改写" in user_msg, (
            f"critique user 消息缺少校验段落前缀说明: {user_msg[-400:]!r}"
        )
        assert "999.99" in user_msg, (
            f"校验段落应包含违规数字 999.99: {user_msg[-400:]!r}"
        )
        # 校验段落必须追加在初稿之后（消息末尾）
        assert user_msg.index("【确定性校验结果】") > user_msg.index("【待审查初稿】"), (
            "校验段落应位于【待审查初稿】之后"
        )
        # critique 返回「通过」→ 最终输出为原草稿
        assert "【草稿】" in result["content"]

    def test_critique_no_violations_no_appendix(self):
        """草稿数字全部有出处 → critique user 消息不含【确定性校验结果】。"""
        agent = _make_agent()
        agent.client.chat.completions.create = AsyncMock(
            return_value=_completion_text("通过")
        )

        result = asyncio.run(
            agent._critique_and_revise(LONG_DRAFT_SOURCED, SOURCED_CONTEXT)
        )

        assert result == LONG_DRAFT_SOURCED, "critique「通过」时应返回原草稿"
        create = agent.client.chat.completions.create
        assert create.await_count == 1
        user_msg = _user_msg_of(create.await_args.kwargs)
        assert "【确定性校验结果】" not in user_msg, (
            f"无违规时不应追加校验段落: {user_msg[-300:]!r}"
        )

    def test_appendix_uses_validators_api_and_sits_at_end(self):
        """mock validators：校验段落 = 前缀行 + format_violations_for_critique 输出，
        且 find_unsourced_numbers 以 (draft, context) 同源文本调用。"""
        agent = _make_agent()
        agent.client.chat.completions.create = AsyncMock(
            return_value=_completion_text("通过")
        )
        mock_v = MagicMock()
        mock_v.find_unsourced_numbers.return_value = [
            {"value": 999.99, "raw": "999.99 亿",
             "snippet": "板块全天成交 999.99 亿", "reason": "测试违规"}
        ]
        mock_v.format_violations_for_critique.return_value = "MOCKED_FORMAT_BLOCK"

        with patch("agent.orchestrator.validators", mock_v):
            asyncio.run(agent._critique_and_revise(LONG_DRAFT_UNSOURCED, "CTX"))

        mock_v.find_unsourced_numbers.assert_called_once_with(
            LONG_DRAFT_UNSOURCED, "CTX"
        )
        mock_v.format_violations_for_critique.assert_called_once_with(
            mock_v.find_unsourced_numbers.return_value
        )
        user_msg = _user_msg_of(
            agent.client.chat.completions.create.await_args.kwargs
        )
        assert "【确定性校验结果】" in user_msg
        assert user_msg.endswith("MOCKED_FORMAT_BLOCK"), (
            f"format_violations_for_critique 输出应位于 user 消息末尾: {user_msg[-200:]!r}"
        )


# ════════════════════════════════════════════════════════════════
# 2. 降级路径：validators 缺失 / 抛异常时审查流程照常
# ════════════════════════════════════════════════════════════════


class TestValidatorsDegradation:
    def test_validators_none_degrades_gracefully(self):
        """validators 为 None（模拟 import 失败）→ 无校验段落，critique 照常。"""
        agent = _make_agent()
        agent.client.chat.completions.create = AsyncMock(
            return_value=_completion_text("通过")
        )

        with patch("agent.orchestrator.validators", None):
            result = asyncio.run(
                agent._critique_and_revise(LONG_DRAFT_UNSOURCED, "CTX")
            )

        assert result == LONG_DRAFT_UNSOURCED, (
            "validators 缺失时 critique「通过」应返回原草稿"
        )
        create = agent.client.chat.completions.create
        assert create.await_count == 1, "validators 缺失不应影响 critique 调用"
        user_msg = _user_msg_of(create.await_args.kwargs)
        assert "【确定性校验结果】" not in user_msg

    def test_validators_exception_degrades_gracefully(self, caplog):
        """find_unsourced_numbers 抛异常 → 仅记 log，critique 照常、无校验段落。"""
        agent = _make_agent()
        agent.client.chat.completions.create = AsyncMock(
            return_value=_completion_text("通过")
        )
        mock_v = MagicMock()
        mock_v.find_unsourced_numbers.side_effect = RuntimeError("validators boom")

        with patch("agent.orchestrator.validators", mock_v):
            with caplog.at_level(logging.WARNING, logger="agent.orchestrator"):
                result = asyncio.run(
                    agent._critique_and_revise(LONG_DRAFT_UNSOURCED, "CTX")
                )

        assert result == LONG_DRAFT_UNSOURCED, "validators 异常时应降级返回原草稿"
        create = agent.client.chat.completions.create
        assert create.await_count == 1, "validators 异常不应阻断 critique 调用"
        user_msg = _user_msg_of(create.await_args.kwargs)
        assert "【确定性校验结果】" not in user_msg
        assert any(
            "数字溯源校验执行异常" in r.getMessage() for r in caplog.records
        ), "validators 异常应记录 warning 日志"


# ════════════════════════════════════════════════════════════════
# 3. 流式路径：log-only，不改输出、不阻塞
# ════════════════════════════════════════════════════════════════


class TestStreamLogOnly:
    def test_sector_deep_dive_stream_log_only(self, caplog):
        """深挖流式分支：跑一次校验并 warning 记录，流式输出与原逻辑完全一致。"""
        agent = _make_agent()
        mock_v = MagicMock()
        mock_v.find_unsourced_numbers.return_value = [
            {"value": 999.99, "raw": "999.99 亿",
             "snippet": "板块全天成交 999.99 亿", "reason": "测试违规"},
            {"value": 88.8, "raw": "88.8 亿",
             "snippet": "另有 88.8 亿", "reason": "测试违规2"},
        ]
        agent.client.chat.completions.create = AsyncMock(
            return_value=_fake_chunk_stream("板块", "正文")
        )

        with ExitStack() as stack:
            _sector_deep_dive_patches(stack)
            stack.enter_context(patch("agent.orchestrator.validators", mock_v))
            with caplog.at_level(logging.WARNING, logger="agent.orchestrator"):
                async def run():
                    gen = await agent._sector_deep_dive("煤炭", stream=True)
                    return [piece async for piece in gen]
                chunks = asyncio.run(run())

        # 输出未被修改
        assert chunks == ["板块", "正文"], f"流式输出不应被校验逻辑改动: {chunks!r}"
        create = agent.client.chat.completions.create
        assert create.await_count == 1
        assert create.await_args.kwargs.get("stream") is True
        # 校验恰好执行一次：被检查文本为 system+user 拼接，上下文为完整 user_prompt
        mock_v.find_unsourced_numbers.assert_called_once()
        text_arg, ctx_arg = mock_v.find_unsourced_numbers.call_args.args
        assert "DATA" in ctx_arg, "深挖流式校验的上下文应为完整 user_prompt（含数据块）"
        assert "DATA" in text_arg, "深挖流式校验的被检查文本应包含 user_prompt"
        # log-only：warning 含数量与违规原文，且明确标注不拦截
        warnings = [r.getMessage() for r in caplog.records]
        assert any(
            "log-only" in m and "2 个" in m and "999.99" in m for m in warnings
        ), f"缺少 log-only 校验 warning: {warnings!r}"

    def test_sector_deep_dive_stream_validators_exception_still_streams(self, caplog):
        """深挖流式分支：validators 抛异常 → 仅记 log，流式输出照常。"""
        agent = _make_agent()
        mock_v = MagicMock()
        mock_v.find_unsourced_numbers.side_effect = RuntimeError("validators boom")
        agent.client.chat.completions.create = AsyncMock(
            return_value=_fake_chunk_stream("板块", "正文")
        )

        with ExitStack() as stack:
            _sector_deep_dive_patches(stack)
            stack.enter_context(patch("agent.orchestrator.validators", mock_v))
            with caplog.at_level(logging.WARNING, logger="agent.orchestrator"):
                async def run():
                    gen = await agent._sector_deep_dive("煤炭", stream=True)
                    return [piece async for piece in gen]
                chunks = asyncio.run(run())

        assert chunks == ["板块", "正文"], "validators 异常不应影响流式输出"
        assert any(
            "数字溯源校验异常" in r.getMessage() for r in caplog.records
        ), "流式分支 validators 异常应记录 warning 日志"

    def test_agent_query_stream_log_only(self, caplog):
        """Agent 流式分支：上下文用 tool_context_parts 拼接，log-only 不改输出。"""
        tools_mod = _import_tools()
        tool_name = tools_mod.TOOL_REGISTRY[0]["function"]["name"]
        agent = _make_agent()
        agent.client.chat.completions.create = AsyncMock(
            side_effect=[
                # 第 1 轮：模型要求调工具
                _completion_with_tool_calls(
                    [_make_tool_call(tool_name, {}, "call_1")]
                ),
                # 第 2 轮：纯文本 → 流式分支收尾（该返回不被消费）
                _completion_text("流式终稿"),
                # 第 3 次：_stream_response 内部的流式调用
                _fake_chunk_stream("终稿", "流式"),
            ]
        )
        exec_mock = MagicMock(
            return_value={"ok": True, "data": {"pe_compare": "baijiu_lower"}}
        )
        mock_v = MagicMock()
        mock_v.find_unsourced_numbers.return_value = [
            {"value": 88.8, "raw": "88.8 亿", "snippet": "x", "reason": "y"},
        ]

        with ExitStack() as stack:
            # execute_tool 在 orchestrator 顶部被 import 进本命名空间，两处都 patch
            stack.enter_context(patch("agent.orchestrator.execute_tool", exec_mock))
            stack.enter_context(patch("agent.tools.execute_tool", exec_mock))
            stack.enter_context(patch("agent.orchestrator.validators", mock_v))
            with caplog.at_level(logging.WARNING, logger="agent.orchestrator"):
                async def run():
                    gen = await agent._agent_query(
                        "比较一下白酒和半导体的估值", stream=True
                    )
                    return [piece async for piece in gen]
                chunks = asyncio.run(run())

        # 流式输出未被修改
        assert "".join(chunks) == "终稿流式", (
            f"Agent 流式输出不应被校验逻辑改动: {chunks!r}"
        )
        # 校验恰好一次：上下文必须含工具返回（tool_context_parts 拼接）
        mock_v.find_unsourced_numbers.assert_called_once()
        ctx_arg = mock_v.find_unsourced_numbers.call_args.args[1]
        assert "baijiu_lower" in ctx_arg, (
            f"Agent 流式校验的上下文应由 tool_context_parts 拼接（含工具返回）: {ctx_arg[:200]!r}"
        )
        warnings = [r.getMessage() for r in caplog.records]
        assert any(
            "Agent 查询" in m and "log-only" in m and "88.8" in m for m in warnings
        ), f"缺少 Agent 流式 log-only 校验 warning: {warnings!r}"

    def test_agent_query_stream_validators_none_still_streams(self):
        """Agent 流式分支：validators 为 None → 直接跳过校验，流式输出照常。"""
        tools_mod = _import_tools()
        tool_name = tools_mod.TOOL_REGISTRY[0]["function"]["name"]
        agent = _make_agent()
        agent.client.chat.completions.create = AsyncMock(
            side_effect=[
                _completion_with_tool_calls(
                    [_make_tool_call(tool_name, {}, "call_1")]
                ),
                _completion_text("流式终稿"),
                _fake_chunk_stream("甲", "乙"),
            ]
        )
        exec_mock = MagicMock(return_value={"ok": True, "data": {"n": 1}})

        with ExitStack() as stack:
            stack.enter_context(patch("agent.orchestrator.execute_tool", exec_mock))
            stack.enter_context(patch("agent.tools.execute_tool", exec_mock))
            stack.enter_context(patch("agent.orchestrator.validators", None))

            async def run():
                gen = await agent._agent_query("比较一下白酒和半导体", stream=True)
                return [piece async for piece in gen]
            chunks = asyncio.run(run())

        assert "".join(chunks) == "甲乙", (
            f"validators 缺失时 Agent 流式输出应照常: {chunks!r}"
        )
