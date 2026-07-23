"""第七波集成测试：orchestrator × watchlist / industry_kb / history_lens。

覆盖范围：
1. detect_intent 自选股意图：含『自选』+ 动作词才判；裸『自选股』按列表处理；
   不含『自选』二字一律不判（不误伤既有意图）。
2. watchlist 四类动作（加/删/列表/复盘）路由与回显（mock watchlist 模块与
   DeepSeek client）；参数缺失用法文案；空自选股引导文案。
3. industry_kb 注入块出现在板块深挖 user_prompt【五】之后；返回 None 不注入。
4. history_lens 以史为鉴块出现在全市场复盘 / 板块深挖 user_prompt 末尾。
5. 三模块任一降级（置 None / 抛异常）时主流程不炸。
6. 自选股意图不误伤上下文继承（『那半导体呢』仍继承板块深挖）。

规则（与项目其他测试一致）：
- 所有外部依赖全部 mock（watchlist / data_fetcher / DeepSeek 客户端），零网络。
- _get_latest_trade_date 统一 patch 为固定日期，避免触达真实 tushare。
- 无 pytest-asyncio，异步函数一律用 asyncio.run 驱动。
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agent.orchestrator as orchestrator
from agent.orchestrator import (
    MarketReviewAgent,
    detect_intent,
    _resolve_contextual_intent,
)

# 固定交易日，避免 _get_latest_trade_date 触达真实 tushare
FIXED_TRADE_DATE = datetime(2025, 1, 10)  # 周五

SAMPLE_STOCKS = [
    {"code": "sh600519", "name": "贵州茅台", "market": "cn", "added_at": "2025-01-09T15:00:00"},
    {"code": "sz300750", "name": "宁德时代", "market": "cn", "added_at": "2025-01-09T15:01:00"},
]

SAMPLE_WATCHLIST_BLOCK = (
    "【用户自选股】共 2 只（复盘时请优先纳入分析）：\n"
    "1. 贵州茅台（sh600519）\n"
    "2. 宁德时代（sz300750）"
)

SAMPLE_KB_BLOCK = "【六、行业知识库（背景知识，数据以数据块为准）】\n测试行业档案"

SAMPLE_HISTORY_NOTE = "【以史为鉴：本智能体历史判断回顾】\n01-09 偏多 命中 测试依据"


def _make_agent() -> MarketReviewAgent:
    """构造 agent，DeepSeek 客户端替换为 mock，杜绝真实 HTTP。"""
    agent = MarketReviewAgent()
    agent.client = MagicMock()
    return agent


def _watchlist_mock(stocks=None):
    """mock agent.orchestrator.watchlist 模块（stocks=None 表示空清单）。"""
    m = MagicMock(name="watchlist")
    stocks = list(stocks or [])
    m.list_stocks.return_value = [dict(s) for s in stocks]
    m.is_empty.return_value = not stocks
    m.add_stock.return_value = (True, "已添加自选股：贵州茅台（sh600519）")
    m.remove_stock.return_value = (True, "已删除自选股：贵州茅台（sh600519）")
    m.format_watchlist_block.return_value = SAMPLE_WATCHLIST_BLOCK if stocks else None
    return m


def _run_sector_deep_dive(agent, sector="电子") -> str:
    """驱动非流式板块深挖并返回 LLM 收到的 user_prompt。"""
    agent._call_llm = AsyncMock(return_value={"role": "assistant", "content": "深挖正文"})
    agent._critique_and_revise = AsyncMock(side_effect=lambda draft, ctx: draft)
    with patch(
        "agent.orchestrator._get_latest_trade_date", return_value=FIXED_TRADE_DATE
    ), patch(
        "agent.orchestrator.collect_market_snapshot",
        AsyncMock(return_value=MagicMock(name="snapshot")),
    ), patch(
        "agent.orchestrator.format_market_data_for_prompt", return_value="DATA"
    ), patch.object(
        agent, "_fetch_sector_extras", AsyncMock(return_value=(None, None, None))
    ):
        asyncio.run(agent._sector_deep_dive(sector, stream=False))
    assert agent._call_llm.await_count == 1
    return agent._call_llm.await_args.args[1]


def _run_market_review(agent) -> str:
    """驱动非流式全市场复盘并返回 LLM 收到的 user_prompt。"""
    agent._call_llm = AsyncMock(return_value={"role": "assistant", "content": "复盘正文"})
    with patch(
        "agent.orchestrator._get_latest_trade_date", return_value=FIXED_TRADE_DATE
    ), patch(
        "agent.orchestrator.collect_market_snapshot",
        AsyncMock(return_value=MagicMock(name="snapshot")),
    ), patch(
        "agent.orchestrator.format_market_data_for_prompt", return_value="DATA"
    ):
        asyncio.run(agent._market_review(stream=False))
    assert agent._call_llm.await_count == 1
    return agent._call_llm.await_args.args[1]


async def _collect_stream(gen) -> str:
    parts = []
    async for chunk in gen:
        parts.append(chunk)
    return "".join(parts)


# ════════════════════════════════════════════════════════════════
# 1. detect_intent 自选股意图
# ════════════════════════════════════════════════════════════════


class TestWatchlistIntent:
    """含『自选』+ 动作词判 watchlist；不含『自选』二字一律不判。"""

    @pytest.mark.parametrize(
        "message",
        [
            "加自选 茅台",
            "添加自选 600519",
            "把茅台加入自选股",
            "删自选 茅台",
            "移除自选 600519",
            "删除自选股里的茅台",
            "我的自选股",
            "自选股列表",
            "复盘我的自选股",
            "看看自选股",
            "自选股怎么样",
            "自选股",
            "自选",
        ],
    )
    def test_watchlist_positive(self, message):
        intent, sector = detect_intent(message)
        assert intent == "watchlist", f"消息 {message!r} 期望 watchlist，实际 {intent}"
        assert sector is None

    @pytest.mark.parametrize(
        "message, expected_intent",
        [
            ("复盘", "market_review"),
            ("今天大盘怎么样", "market_review"),
            ("煤炭板块复盘", "sector_deep_dive"),
            ("茅台怎么样", "stock_query"),
            ("给我讲个笑话", "general_chat"),
        ],
    )
    def test_no_watchlist_word_no_watchlist_intent(self, message, expected_intent):
        """不含『自选』二字的消息绝不被自选股意图截获。"""
        intent, _ = detect_intent(message)
        assert intent == expected_intent, f"消息 {message!r} 期望 {expected_intent}，实际 {intent}"

    def test_watchlist_word_without_action_is_conservative(self):
        """含『自选』但无动作词时不判 watchlist（保守规则）。"""
        intent, _ = detect_intent("今天自选")
        assert intent != "watchlist"


# ════════════════════════════════════════════════════════════════
# 2. watchlist 四类动作路由与回显
# ════════════════════════════════════════════════════════════════


class TestWatchlistRouting:
    """process_message 路由到 _watchlist 后的加/删/列表/复盘行为。"""

    def test_add_route_echoes_code_name_for_confirmation(self):
        agent = _make_agent()
        wl = _watchlist_mock(stocks=SAMPLE_STOCKS)
        with patch.object(orchestrator, "watchlist", wl):
            result = asyncio.run(agent.process_message("加自选 茅台"))
        assert isinstance(result, dict)
        wl.add_stock.assert_called_once_with("茅台")
        content = result["content"]
        assert "已添加自选股" in content
        assert "贵州茅台" in content and "sh600519" in content  # 回显 (code, name)
        assert "核对" in content  # 让用户确认

    def test_add_keyword_fallback_before_verb(self):
        """『把茅台加入自选股』：股票名在动作词前，兜底提取。"""
        agent = _make_agent()
        wl = _watchlist_mock()
        with patch.object(orchestrator, "watchlist", wl):
            asyncio.run(agent.process_message("把茅台加入自选股"))
        wl.add_stock.assert_called_once_with("茅台")

    def test_add_failure_echo(self):
        agent = _make_agent()
        wl = _watchlist_mock()
        wl.add_stock.return_value = (False, "未找到匹配的股票：茅台")
        with patch.object(orchestrator, "watchlist", wl):
            result = asyncio.run(agent.process_message("加自选 茅台"))
        assert "未找到匹配的股票" in result["content"]
        assert "核对" not in result["content"]  # 未成功不追加确认行

    def test_remove_route(self):
        agent = _make_agent()
        wl = _watchlist_mock(stocks=SAMPLE_STOCKS)
        with patch.object(orchestrator, "watchlist", wl):
            result = asyncio.run(agent.process_message("删自选 茅台"))
        wl.remove_stock.assert_called_once_with("茅台")
        assert "已删除自选股" in result["content"]

    def test_list_route_plain_text_no_llm(self):
        agent = _make_agent()
        agent._call_llm = AsyncMock()  # 列表路径绝不走 LLM
        wl = _watchlist_mock(stocks=SAMPLE_STOCKS)
        with patch.object(orchestrator, "watchlist", wl):
            result = asyncio.run(agent.process_message("我的自选股"))
        assert isinstance(result, dict)
        content = result["content"]
        assert "共 2 只" in content
        assert "贵州茅台" in content and "sh600519" in content
        assert "宁德时代" in content and "sz300750" in content
        assert agent._call_llm.await_count == 0

    def test_list_route_stream_wraps_dict(self):
        """流式模式下列表 dict 包装为生成器，文本完整。"""
        agent = _make_agent()
        wl = _watchlist_mock(stocks=SAMPLE_STOCKS)
        with patch.object(orchestrator, "watchlist", wl):
            gen = asyncio.run(agent.process_message("我的自选股", stream=True))
            text = asyncio.run(_collect_stream(gen))
        assert "贵州茅台" in text and "宁德时代" in text

    def test_review_route_builds_block_and_calls_llm(self):
        agent = _make_agent()
        agent._call_llm = AsyncMock(return_value={"role": "assistant", "content": "复盘结果"})
        wl = _watchlist_mock(stocks=SAMPLE_STOCKS)
        quote = {"price": "1500.0", "pct": "+1.20", "high": "1510.0", "low": "1490.0"}
        with patch.object(orchestrator, "watchlist", wl), patch(
            "agent.data_fetcher.fetch_stock_quote", MagicMock(return_value=quote)
        ):
            result = asyncio.run(agent.process_message("复盘我的自选股"))
        assert result["content"].startswith("复盘结果")
        assert "风险提示" in result["content"]  # 出口统一兜底追加
        assert agent._call_llm.await_count == 1
        system_prompt = agent._call_llm.await_args.args[0]
        user_prompt = agent._call_llm.await_args.args[1]
        assert isinstance(system_prompt, str) and system_prompt.strip()
        assert "【用户自选股】" in user_prompt
        assert "贵州茅台" in user_prompt and "宁德时代" in user_prompt
        assert "现价" in user_prompt  # 行情块进入 prompt

    def test_review_single_quote_failure_degrades(self):
        """单只行情获取失败降级为『数据暂不可用』，主流程不炸。"""
        agent = _make_agent()
        agent._call_llm = AsyncMock(return_value={"role": "assistant", "content": "复盘结果"})
        wl = _watchlist_mock(stocks=SAMPLE_STOCKS)
        with patch.object(orchestrator, "watchlist", wl), patch(
            "agent.data_fetcher.fetch_stock_quote",
            MagicMock(side_effect=Exception("network boom")),
        ):
            result = asyncio.run(agent.process_message("复盘我的自选股"))
        assert result["content"].startswith("复盘结果")
        assert "风险提示" in result["content"]  # 出口统一兜底追加
        user_prompt = agent._call_llm.await_args.args[1]
        assert "行情数据暂不可用" in user_prompt

    def test_empty_watchlist_review_guide(self):
        """空自选股复盘：返回引导文案，不走 LLM。"""
        agent = _make_agent()
        agent._call_llm = AsyncMock()
        wl = _watchlist_mock(stocks=[])
        with patch.object(orchestrator, "watchlist", wl):
            result = asyncio.run(agent.process_message("复盘我的自选股"))
        assert "还没有添加自选股" in result["content"]
        assert "加自选" in result["content"]
        assert agent._call_llm.await_count == 0

    def test_empty_watchlist_list_guide(self):
        agent = _make_agent()
        wl = _watchlist_mock(stocks=[])
        with patch.object(orchestrator, "watchlist", wl):
            result = asyncio.run(agent.process_message("我的自选股"))
        assert "还没有添加自选股" in result["content"]

    def test_add_missing_param_returns_usage(self):
        agent = _make_agent()
        wl = _watchlist_mock()
        with patch.object(orchestrator, "watchlist", wl):
            result = asyncio.run(agent.process_message("加自选"))
        assert "用法" in result["content"]
        wl.add_stock.assert_not_called()

    def test_remove_missing_param_returns_usage(self):
        agent = _make_agent()
        wl = _watchlist_mock()
        with patch.object(orchestrator, "watchlist", wl):
            result = asyncio.run(agent.process_message("删自选"))
        assert "用法" in result["content"]
        wl.remove_stock.assert_not_called()

    def test_watchlist_module_none_degrades(self):
        """watchlist 模块未就绪（None）：返回降级文案，绝不抛出。"""
        agent = _make_agent()
        with patch.object(orchestrator, "watchlist", None):
            result = asyncio.run(agent.process_message("我的自选股"))
        assert isinstance(result, dict)
        assert "暂不可用" in result["content"]

    def test_add_stock_raises_degrades(self):
        """watchlist.add_stock 抛异常：降级文案，绝不炸主流程。"""
        agent = _make_agent()
        wl = _watchlist_mock()
        wl.add_stock.side_effect = Exception("disk boom")
        with patch.object(orchestrator, "watchlist", wl):
            result = asyncio.run(agent.process_message("加自选 茅台"))
        assert "失败" in result["content"]

    def test_watchlist_system_prompt_fallback_on_exception(self):
        agent = _make_agent()
        with patch("agent.orchestrator.get_system_prompt", side_effect=Exception("boom")):
            assert agent._watchlist_system_prompt() == orchestrator._DEFAULT_WATCHLIST_SYSTEM_PROMPT

    def test_watchlist_system_prompt_fallback_on_empty(self):
        agent = _make_agent()
        with patch("agent.orchestrator.get_system_prompt", return_value=""):
            assert agent._watchlist_system_prompt() == orchestrator._DEFAULT_WATCHLIST_SYSTEM_PROMPT


# ════════════════════════════════════════════════════════════════
# 3. 行业知识库注入（板块深挖）
# ════════════════════════════════════════════════════════════════


class TestIndustryKbInjection:
    """format_kb_block 注入到深挖 user_prompt【五】之后；None/异常/未就绪不注入。"""

    def test_kb_block_injected_after_section_five(self):
        agent = _make_agent()
        kb = MagicMock(name="industry_kb")
        kb.format_kb_block.return_value = SAMPLE_KB_BLOCK
        with patch.object(orchestrator, "industry_kb", kb):
            user_prompt = _run_sector_deep_dive(agent, sector="电子")
        kb.format_kb_block.assert_called_once_with("电子")
        assert "【六、行业知识库" in user_prompt
        idx5 = user_prompt.index("【五、新闻与景气背景】")
        idx6 = user_prompt.index("【六、行业知识库")
        assert idx5 < idx6, "知识库块必须位于【五、新闻与景气背景】之后"

    def test_kb_none_not_injected(self):
        agent = _make_agent()
        kb = MagicMock(name="industry_kb")
        kb.format_kb_block.return_value = None
        with patch.object(orchestrator, "industry_kb", kb):
            user_prompt = _run_sector_deep_dive(agent, sector="电子")
        assert "【六、行业知识库" not in user_prompt

    def test_kb_module_none_degrades(self):
        agent = _make_agent()
        with patch.object(orchestrator, "industry_kb", None):
            user_prompt = _run_sector_deep_dive(agent, sector="电子")
        assert "【六、行业知识库" not in user_prompt
        assert "【五、新闻与景气背景】" in user_prompt  # 主流程结构完好

    def test_kb_raises_degrades(self):
        agent = _make_agent()
        kb = MagicMock(name="industry_kb")
        kb.format_kb_block.side_effect = Exception("kb boom")
        with patch.object(orchestrator, "industry_kb", kb):
            user_prompt = _run_sector_deep_dive(agent, sector="电子")
        assert "【六、行业知识库" not in user_prompt
        assert "【五、新闻与景气背景】" in user_prompt


# ════════════════════════════════════════════════════════════════
# 4. 以史为鉴注入（全市场复盘 + 板块深挖）
# ════════════════════════════════════════════════════════════════


class TestHistoryNoteInjection:
    """get_history_note 注入 user_prompt 末尾；None/异常/未就绪不注入。"""

    def test_history_note_in_market_review(self):
        agent = _make_agent()
        hl = MagicMock(name="history_lens")
        hl.get_history_note.return_value = SAMPLE_HISTORY_NOTE
        with patch.object(orchestrator, "history_lens", hl):
            user_prompt = _run_market_review(agent)
        hl.get_history_note.assert_called_once_with(sector=None, mode="market_review")
        assert "【以史为鉴：本智能体历史判断回顾】" in user_prompt
        assert user_prompt.endswith(SAMPLE_HISTORY_NOTE), "以史为鉴块必须在 user_prompt 末尾"

    def test_history_note_in_sector_deep_dive(self):
        agent = _make_agent()
        hl = MagicMock(name="history_lens")
        hl.get_history_note.return_value = SAMPLE_HISTORY_NOTE
        with patch.object(orchestrator, "history_lens", hl):
            user_prompt = _run_sector_deep_dive(agent, sector="电子")
        hl.get_history_note.assert_called_once_with(sector="电子", mode="sector_deep_dive")
        assert user_prompt.endswith(SAMPLE_HISTORY_NOTE)

    def test_history_none_not_injected(self):
        agent = _make_agent()
        hl = MagicMock(name="history_lens")
        hl.get_history_note.return_value = None
        with patch.object(orchestrator, "history_lens", hl):
            user_prompt = _run_market_review(agent)
        assert "【以史为鉴" not in user_prompt

    def test_history_module_none_degrades(self):
        agent = _make_agent()
        with patch.object(orchestrator, "history_lens", None):
            user_prompt = _run_market_review(agent)
        assert "【以史为鉴" not in user_prompt
        assert "生成A股市场复盘" in user_prompt  # 主流程结构完好

    def test_history_raises_degrades(self):
        agent = _make_agent()
        hl = MagicMock(name="history_lens")
        hl.get_history_note.side_effect = Exception("history boom")
        with patch.object(orchestrator, "history_lens", hl):
            user_prompt = _run_sector_deep_dive(agent, sector="电子")
        assert "【以史为鉴" not in user_prompt
        assert "深度分析电子板块" in user_prompt  # 主流程结构完好


# ════════════════════════════════════════════════════════════════
# 5. 意图继承防误伤
# ════════════════════════════════════════════════════════════════


class TestIntentInheritanceGuard:
    """自选股规则不得破坏既有上下文意图继承。"""

    def test_bare_sector_still_inherits_deep_dive(self):
        """『那半导体呢』在上文板块深挖语境下仍继承深挖并切换行业。"""
        history = [
            {"role": "user", "content": "煤炭板块复盘"},
            {"role": "assistant", "content": "煤炭板块深挖正文……"},
        ]
        intent, sector, label = _resolve_contextual_intent("那半导体呢", history)
        assert intent == "sector_deep_dive"
        assert sector == "电子"
        assert label is not None

    def test_watchlist_intent_not_hijacked_by_history(self):
        """上文是全市场复盘时，『复盘我的自选股』仍判 watchlist 而非继承复盘。"""
        history = [
            {"role": "user", "content": "今日复盘"},
            {"role": "assistant", "content": "全市场复盘正文……"},
        ]
        intent, sector, label = _resolve_contextual_intent("复盘我的自选股", history)
        assert intent == "watchlist"
        assert sector is None
        assert label is None
