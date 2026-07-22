"""tests/test_social_intent.py — 社媒舆情/投资人格意图路由 + [UNSOURCED] 泄漏修复测试（全 mock 零网络）。

背景：清小搭平台实测 7 问全崩——detect_intent 没有新能力入口，「市场情绪/个股人气/
股吧/微博/B站/逆向投资」类问题被旧意图劫持（全市场复盘/个股行情卡）或落入
general_chat→_chat 纯模型闲聊拒答；复盘提示词里的 [UNSOURCED] 标记直接泄漏给用户。

本文件钉死：
1. detect_intent 新增 social_sentiment / persona 高优先级意图的命中矩阵
   （判定位置：自选股之后、行业提取与复盘关键词之前）；
2. Q1-Q7 用户实测问题原文的逐条路由断言；
3. 防误判矩阵：复盘/板块深挖/个股/自选股/新闻/显式数据查询指令不被抢
   （含『复盘』二字优先复盘、含 查询/列出/排名 等指令词保持旧 mcp_query 路由）；
4. process_message 接线：两意图走 _agent_query 且 hint 透传（流式/非流式两路径），
   _agent_query 把 hint 拼进 AGENT_QUERY_PROMPT 系统消息尾部（mock LLM 验证）；
5. [UNSOURCED] 修复：复盘提示词改为「数据未覆盖」+ 复盘出口专属的残留清洗
   （非流式替换 / 流式跨 chunk 安全替换）。

规则（与项目其他测试一致）：全部 mock 零网络；无 pytest-asyncio，异步用 asyncio.run 驱动。
"""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agent.orchestrator as orchestrator
from agent.orchestrator import MarketReviewAgent, detect_intent
from agent.system_prompts import AGENT_QUERY_PROMPT

# 固定交易日，避免 _get_latest_trade_date 触达真实 tushare
FIXED_TRADE_DATE = datetime(2025, 1, 10)  # 周五


def _make_agent() -> MarketReviewAgent:
    """构造一个 agent，DeepSeek 客户端替换为 mock，防止任何真实 HTTP 调用。"""
    agent = MarketReviewAgent()
    agent.client = MagicMock()
    return agent


def _completion_text(text: str):
    """伪造纯文本的 create 返回（工具循环终点 / 普通生成）。"""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=None),
            finish_reason="stop",
        )]
    )


def _fake_stream(chunks):
    """伪造 OpenAI 流式响应（async 可迭代，末 chunk 带 finish_reason）。"""
    class _Stream:
        def __aiter__(self):
            async def _gen():
                for t in chunks:
                    yield SimpleNamespace(choices=[SimpleNamespace(
                        delta=SimpleNamespace(content=t), finish_reason=None,
                    )])
                yield SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=None), finish_reason="stop",
                )])
            return _gen()
    return _Stream()


async def _collect(gen) -> str:
    parts = []
    async for piece in gen:
        parts.append(piece)
    return "".join(parts)


def _patch_market_data(stack):
    """_market_review 数据层 mock：固定交易日 + 假快照 + 假格式化文本。"""
    stack.enter_context(patch(
        "agent.orchestrator._get_latest_trade_date", return_value=FIXED_TRADE_DATE
    ))
    stack.enter_context(patch(
        "agent.orchestrator.collect_market_snapshot",
        AsyncMock(return_value=MagicMock(name="snapshot")),
    ))
    stack.enter_context(patch(
        "agent.orchestrator.format_market_data_for_prompt", return_value="DATA"
    ))


# ════════════════════════════════════════════════════════════════
# 1. social_sentiment 强信号：单独命中即判，无需市场语境
# ════════════════════════════════════════════════════════════════


class TestSocialStrongSignals:
    @pytest.mark.parametrize(
        "message, keyword",
        [
            ("股吧里大家都在讨论什么股票", "股吧"),
            ("微博上大家都在说什么", "微博"),
            ("知乎上怎么看这件事", "知乎"),
            ("抖音上刷到好多人在聊", "抖音"),
            ("B站视频都在讲这个", "B站"),
            ("b站弹幕怎么说", "b站"),
            ("哔哩哔哩上有什么内容", "哔哩"),
            ("小红书上大家都在晒什么", "小红书"),
            ("今天的热搜有哪些", "热搜"),
            ("热榜上都有什么", "热榜"),
            ("最新舆情怎么样", "舆情"),
            ("舆论现在什么风向", "舆论"),
        ],
    )
    def test_strong_signal_alone_triggers(self, message, keyword):
        intent, sector = detect_intent(message)
        assert intent == "social_sentiment", (
            f"强信号 {keyword!r} 单独命中应判 social_sentiment，"
            f"消息 {message!r} 实际 {intent}"
        )
        assert sector is None


# ════════════════════════════════════════════════════════════════
# 2. social_sentiment 弱信号：需市场语境（市场语境词或个股/行业实体）
# ════════════════════════════════════════════════════════════════


class TestSocialWeakSignals:
    @pytest.mark.parametrize(
        "message, keyword",
        [
            ("今天市场情绪怎么样", "情绪"),
            ("茅台现在人气怎么样", "人气"),
            ("半导体板块的热度如何", "热度"),
            ("这只股票值得关注吗", "关注"),
            ("大家怎么讨论白酒板块的", "讨论"),
            ("宁德时代的评论都在说什么", "评论"),
            ("今天涨停多少家", "涨停"),
            ("今天跌停多少家", "跌停"),
            ("市场炸板率高不高", "炸板"),
            ("今天打板行情怎么样", "打板"),
            ("市场最高连板几板了", "连板"),
        ],
    )
    def test_weak_signal_with_market_context(self, message, keyword):
        intent, _ = detect_intent(message)
        assert intent == "social_sentiment", (
            f"弱信号 {keyword!r} + 市场语境应判 social_sentiment，"
            f"消息 {message!r} 实际 {intent}"
        )

    @pytest.mark.parametrize(
        "message",
        [
            "我的人气怎么样",      # 人气，无市场语境
            "这个话题热度真高",    # 热度，无市场语境
            "评论区全是玩梗的",    # 评论，无市场语境
        ],
    )
    def test_weak_signal_without_context_not_social(self, message):
        intent, _ = detect_intent(message)
        assert intent != "social_sentiment", (
            f"弱信号无市场语境不得判 social_sentiment，消息 {message!r} 实际 {intent}"
        )
        assert intent == "general_chat", (
            f"消息 {message!r} 应回落 general_chat，实际 {intent}"
        )


# ════════════════════════════════════════════════════════════════
# 3. persona 意图：人格/框架关键词 + 市场语境才判
# ════════════════════════════════════════════════════════════════


class TestPersonaIntent:
    @pytest.mark.parametrize(
        "message, keyword",
        [
            ("用价值投资的眼光看看白酒板块", "价值投资"),
            ("成长投资框架下宁德时代怎么样", "成长投资"),
            ("趋势交易的角度看大盘", "趋势交易"),
            ("用逆向投资的思路看看白酒板块能不能抄底", "逆向投资"),
            ("巴菲特会怎么看现在的A股", "巴菲特"),
            ("格雷厄姆的估值方法分析茅台", "格雷厄姆"),
            ("我的投资框架适不适合现在的行情", "投资框架"),
            ("这种投资风格在股市里行不行", "投资风格"),
        ],
    )
    def test_persona_keyword_with_market_context(self, message, keyword):
        intent, sector = detect_intent(message)
        assert intent == "persona", (
            f"人格关键词 {keyword!r} + 市场语境应判 persona，"
            f"消息 {message!r} 实际 {intent}"
        )
        assert sector is None

    @pytest.mark.parametrize(
        "message",
        [
            "巴菲特是谁",        # 巴菲特，无市场语境
            "介绍一下价值投资",  # 价值投资，无市场语境
        ],
    )
    def test_persona_keyword_without_context_not_persona(self, message):
        intent, _ = detect_intent(message)
        assert intent != "persona", (
            f"人格关键词无市场语境不得判 persona，消息 {message!r} 实际 {intent}"
        )
        assert intent == "general_chat", (
            f"消息 {message!r} 应回落 general_chat，实际 {intent}"
        )


# ════════════════════════════════════════════════════════════════
# 4. Q1-Q7 用户实测问题原文逐条断言
# ════════════════════════════════════════════════════════════════


class TestUserReportedQuestions:
    @pytest.mark.parametrize(
        "qid, message, expected_intent",
        [
            ("Q1", "今天市场情绪怎么样", "social_sentiment"),
            ("Q2", "茅台现在人气怎么样", "social_sentiment"),
            ("Q3", "股吧里大家都在讨论什么股票", "social_sentiment"),
            ("Q4", "微博知乎抖音B站上有什么股市热点", "social_sentiment"),
            ("Q5", "B站上关于半导体的评论都在说什么", "social_sentiment"),
            ("Q6", "用逆向投资的思路看看白酒板块能不能抄底", "persona"),
        ],
    )
    def test_q1_to_q6_detect_intent(self, qid, message, expected_intent):
        intent, _ = detect_intent(message)
        assert intent == expected_intent, (
            f"{qid} 实测消息 {message!r} 期望 {expected_intent}，实际 {intent}"
        )

    def test_q7_comparison_routes_to_agent_loop(self):
        """Q7『比较一下白酒和半导体的情绪面和技术面』：含比较+两实体，
        必须走 Agent 工具循环路径（_agent_query），不得落入 _chat/复盘/深挖。"""
        agent = _make_agent()
        agent._agent_query = AsyncMock(
            return_value={"role": "assistant", "content": "agent答案"}
        )
        agent._chat = AsyncMock(return_value={"role": "assistant", "content": "闲聊答案"})
        agent._market_review = AsyncMock(
            return_value={"role": "assistant", "content": "复盘答案"}
        )
        agent._sector_deep_dive = AsyncMock(
            return_value={"role": "assistant", "content": "深挖答案"}
        )

        result = asyncio.run(agent.process_message(
            "比较一下白酒和半导体的情绪面和技术面", stream=False
        ))

        assert agent._agent_query.await_count == 1, (
            "Q7 比较两板块情绪面技术面应路由到 _agent_query 工具循环"
        )
        assert agent._chat.await_count == 0, "Q7 不应落入纯闲聊 _chat"
        assert agent._market_review.await_count == 0, "Q7 不应被劫持去全市场复盘"
        assert agent._sector_deep_dive.await_count == 0, "Q7 不应落入单板块深挖"
        assert result["content"] == "agent答案"


# ════════════════════════════════════════════════════════════════
# 5. 防误判矩阵：旧意图不被新意图抢
# ════════════════════════════════════════════════════════════════


class TestNoMisjudgment:
    @pytest.mark.parametrize(
        "message, expected_intent",
        [
            # ── 复盘：含『复盘』二字一律优先复盘 ──
            ("今日复盘", "market_review"),
            ("复盘", "market_review"),
            ("今天市场怎么样", "market_review"),   # 不含任何新关键词，旧路径不变
            ("复盘今天的市场情绪", "market_review"),  # 复盘+情绪：复盘优先
            # ── 板块深挖 ──
            ("煤炭板块复盘", "sector_deep_dive"),
            ("半导体板块怎么样", "sector_deep_dive"),
            ("聚焦半导体板块", "sector_deep_dive"),
            # ── 个股查询 ──
            ("茅台怎么样", "stock_query"),
            ("分析一下比亚迪", "stock_query"),
            # ── 自选股（判定在新意图之前）──
            ("我的自选股", "watchlist"),
            ("加自选 茅台", "watchlist"),
            # ── 新闻意图 ──
            ("今天有什么新闻", "news_only"),
            ("银行板块新闻", "news_only"),
            # ── 显式数据查询指令：保持旧 mcp_query 路由 ──
            ("今天涨停家数查询", "mcp_query"),
            ("今天涨停家数列出", "mcp_query"),
            # ── 纯闲聊 ──
            ("给我讲个笑话", "general_chat"),
            ("红烧肉怎么做", "general_chat"),
        ],
    )
    def test_legacy_intents_not_hijacked(self, message, expected_intent):
        intent, _ = detect_intent(message)
        assert intent == expected_intent, (
            f"消息 {message!r} 期望 {expected_intent}（不被新意图抢），实际 {intent}"
        )


# ════════════════════════════════════════════════════════════════
# 6. process_message 接线：两意图走 _agent_query 且 hint 透传（流式/非流式）
# ════════════════════════════════════════════════════════════════


class TestRoutingHintWiring:
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        "message, intent, hint_marker",
        [
            ("今天市场情绪怎么样", "social_sentiment", "get_market_sentiment"),
            ("用逆向投资的思路看看白酒板块能不能抄底", "persona", "analyze_with_persona"),
        ],
    )
    def test_intent_routes_to_agent_query_with_hint(
        self, message, intent, hint_marker, stream
    ):
        agent = _make_agent()
        agent._agent_query = AsyncMock(
            return_value={"role": "assistant", "content": "agent答案"}
        )
        agent._chat = AsyncMock(return_value={"role": "assistant", "content": "闲聊答案"})
        agent._market_review = AsyncMock(
            return_value={"role": "assistant", "content": "复盘答案"}
        )

        result = asyncio.run(agent.process_message(message, stream=stream))

        assert agent._agent_query.await_count == 1, (
            f"意图 {intent} 应路由到 _agent_query（stream={stream}）"
        )
        assert agent._chat.await_count == 0, f"意图 {intent} 不应落入 _chat"
        assert agent._market_review.await_count == 0, (
            f"意图 {intent} 不应被劫持去全市场复盘"
        )
        _, kwargs = agent._agent_query.await_args
        hint = kwargs.get("hint")
        assert hint, f"_agent_query 应收到非空 hint: {kwargs}"
        assert hint == orchestrator._AGENT_ROUTE_HINTS[intent], (
            f"hint 应为 _AGENT_ROUTE_HINTS[{intent!r}] 原文"
        )
        assert hint.startswith("【本问题路由提示】"), "hint 应以路由提示头开头"
        assert hint_marker in hint, (
            f"意图 {intent} 的 hint 应引导优先使用 {hint_marker}"
        )
        assert result["content"] == "agent答案"


# ════════════════════════════════════════════════════════════════
# 7. _agent_query hint 透传：拼进 AGENT_QUERY_PROMPT 系统消息尾部（mock LLM）
# ════════════════════════════════════════════════════════════════


class TestAgentQueryHintPassthrough:
    def test_hint_appended_to_system_message(self):
        """非流式：hint 非空时追加到系统消息尾部，模型实际收到。"""
        agent = _make_agent()
        agent.client.chat.completions.create = AsyncMock(
            return_value=_completion_text("舆情分析正文")
        )
        hint = orchestrator._AGENT_ROUTE_HINTS["social_sentiment"]

        result = asyncio.run(agent._agent_query(
            "今天市场情绪怎么样", stream=False, hint=hint
        ))

        create = agent.client.chat.completions.create
        assert create.await_count == 1
        system_msg = create.await_args.kwargs["messages"][0]["content"]
        assert system_msg.startswith(AGENT_QUERY_PROMPT), (
            "系统消息应以 AGENT_QUERY_PROMPT 开头"
        )
        assert system_msg.endswith(hint), (
            f"hint 应追加到系统消息尾部: ...{system_msg[-80:]!r}"
        )
        assert "【本问题路由提示】" in system_msg
        assert isinstance(result, dict) and "舆情分析正文" in result["content"]

    def test_hint_none_keeps_system_message_unchanged(self):
        """hint=None（缺省）时系统消息与原来完全一致，无路由提示。"""
        agent = _make_agent()
        agent.client.chat.completions.create = AsyncMock(
            return_value=_completion_text("比较分析正文")
        )

        asyncio.run(agent._agent_query("比较一下白酒和半导体的估值", stream=False))

        system_msg = (
            agent.client.chat.completions.create.await_args.kwargs["messages"][0]["content"]
        )
        assert system_msg.startswith(AGENT_QUERY_PROMPT)
        assert "【本问题路由提示】" not in system_msg, (
            "hint=None 时系统消息不得出现路由提示"
        )

    def test_hint_appended_on_stream_path(self):
        """流式：hint 同样拼进系统消息（工具循环与最终流式成文共用同一 system）。"""
        agent = _make_agent()
        agent.client.chat.completions.create = AsyncMock(side_effect=[
            _completion_text("流式成文触发"),   # 工具循环第 1 轮：纯文本收敛
            _fake_stream(["流式舆情正文"]),      # _stream_response 的流式调用
        ])
        hint = orchestrator._AGENT_ROUTE_HINTS["persona"]

        gen = asyncio.run(agent._agent_query(
            "用逆向投资的思路看看白酒板块能不能抄底", stream=True, hint=hint
        ))
        output = asyncio.run(_collect(gen))

        create = agent.client.chat.completions.create
        assert create.await_count == 2, "工具循环 1 轮 + 流式成文 1 轮"
        # 两轮调用共用同一 messages 列表，系统消息都带 hint
        for call in create.await_args_list:
            system_msg = call.kwargs["messages"][0]["content"]
            assert system_msg.endswith(hint), (
                f"流式路径系统消息尾部应带 hint: ...{system_msg[-80:]!r}"
            )
        assert "【本问题路由提示】" in system_msg
        assert "流式舆情正文" in output


# ════════════════════════════════════════════════════════════════
# 8. [UNSOURCED] 泄漏修复：提示词文案 + 复盘出口专属清洗
# ════════════════════════════════════════════════════════════════


class TestUnsourcedLeakFix:
    def test_review_prompt_no_longer_instructs_unsourced(self):
        """复盘 user prompt 不得再出现 [UNSOURCED] 指令，应为「数据未覆盖」。"""
        from contextlib import ExitStack
        agent = _make_agent()
        agent._call_llm = AsyncMock(
            return_value={"role": "assistant", "content": "复盘正文"}
        )
        with ExitStack() as stack:
            _patch_market_data(stack)
            asyncio.run(agent._market_review(stream=False))

        user_prompt = agent._call_llm.await_args.args[1]
        assert "[UNSOURCED]" not in user_prompt, (
            f"复盘提示词不得再含 [UNSOURCED] 指令: ...{user_prompt[-120:]!r}"
        )
        assert "数据缺失写「数据未覆盖」" in user_prompt

    def test_replace_unsourced_unit(self):
        """_replace_unsourced：替换/无标记透传/空串兜底。"""
        assert orchestrator._replace_unsourced("涨幅[UNSOURCED]待核") == "涨幅（数据未覆盖）待核"
        assert orchestrator._replace_unsourced("无标记文本") == "无标记文本"
        assert orchestrator._replace_unsourced("") == ""
        assert orchestrator._replace_unsourced(None) == ""

    def test_non_stream_review_output_sanitized(self):
        """非流式复盘出口：模型仍输出 [UNSOURCED] 时被替换为（数据未覆盖）。"""
        from contextlib import ExitStack
        agent = _make_agent()
        agent._call_llm = AsyncMock(return_value={
            "role": "assistant",
            "content": "今日成交[UNSOURCED]，31 行业[UNSOURCED]。",
        })
        with ExitStack() as stack:
            _patch_market_data(stack)
            result = asyncio.run(agent._market_review(stream=False))

        content = result["content"]
        assert "[UNSOURCED]" not in content, f"[UNSOURCED] 不得泄漏到用户: {content!r}"
        assert content.count("（数据未覆盖）") == 2

    def test_stream_review_output_sanitized_across_chunks(self):
        """流式复盘出口：[UNSOURCED] 被拆到两个 chunk 时也能完整替换。"""
        from contextlib import ExitStack
        agent = _make_agent()

        async def _fake_gen():
            for piece in ["今日成交[UN", "SOURCED]，情", "绪[UNSOUR", "CED]。"]:
                yield piece

        agent._call_llm = AsyncMock(return_value=_fake_gen())
        with ExitStack() as stack:
            _patch_market_data(stack)
            gen = asyncio.run(agent._market_review(stream=True))
            output = asyncio.run(_collect(gen))

        assert "[UNSOURCED]" not in output, (
            f"跨 chunk 的 [UNSOURCED] 不得泄漏到用户: {output!r}"
        )
        assert output == "今日成交（数据未覆盖），情绪（数据未覆盖）。", (
            f"替换后正文应与原文逐字一致（仅标记被换）: {output!r}"
        )

    def test_stream_passthrough_without_token(self):
        """流式清洗对无标记 chunk 零影响（逐字透传）。"""
        async def _fake_gen():
            for piece in ["第一段。", "第二段，", "完。"]:
                yield piece

        output = asyncio.run(_collect(
            orchestrator._replace_unsourced_stream(_fake_gen())
        ))
        assert output == "第一段。第二段，完。"
