"""残留1修复测试：_resolve_time_reference 解析矩阵 + 单日交易日对齐。

覆盖范围：
1. _resolve_time_reference 全矩阵——单日类（今天/今日/昨天/昨日/前天/
   上周一~上周五/本周一~本周五）→ ('day', date)；周区间类（这周/本周/
   近一周/这周以来/上周/上礼拜）→ ('week', 周一起始, 周日或今日结束)；
   无引用 → ('none', None)。含跨周末/跨年边界与『上周五』优先于『上周』的
   单日/周区间判定顺序。
2. 单日对齐：market_review/sector_deep_dive/news_only 路径下，
   _get_latest_trade_date 必须以解析出的目标日期为 ref 调用
   （『昨天』场景数据真的是昨天的），无引用时保持 datetime.now() 现状。

规则（与项目其他测试一致）：
- 所有外部依赖全部 mock，绝不发起真实网络请求。
- 无 pytest-asyncio，异步函数一律用 asyncio.run 驱动。
"""

import asyncio
from datetime import datetime, timedelta, date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.orchestrator import (
    MarketReviewAgent,
    _resolve_time_reference,
)

# 固定锚点：2025-01-10 周五（所在周 01-06周一~01-12周日，上周跨年到 2024-12-30）
ANCHOR_FRI = datetime(2025, 1, 10)
# 2025-01-12 周日 / 2025-01-13 周一 / 2025-01-01 周三（跨年周）
ANCHOR_SUN = datetime(2025, 1, 12)
ANCHOR_MON = datetime(2025, 1, 13)
ANCHOR_NEWYEAR = datetime(2025, 1, 1)

FIXED_TRADE_DATE = datetime(2025, 1, 10)


def _make_agent() -> MarketReviewAgent:
    agent = MarketReviewAgent()
    agent.client = MagicMock()
    return agent


# ════════════════════════════════════════════════════════════════
# 1. _resolve_time_reference 解析矩阵
# ════════════════════════════════════════════════════════════════


class TestResolveTimeReferenceMatrix:
    """以 2025-01-10（周五）为锚点的单日/周区间/无引用全矩阵。"""

    @pytest.mark.parametrize(
        "msg, expected",
        [
            # ── 单日相对词 ──
            ("今天市场怎么样", ("day", date(2025, 1, 10))),
            ("今日大盘", ("day", date(2025, 1, 10))),
            ("昨天市场怎么样", ("day", date(2025, 1, 9))),
            ("昨日行情", ("day", date(2025, 1, 9))),
            ("前天白酒板块", ("day", date(2025, 1, 8))),
            # ── 本周X（一~五）单日 ──
            ("本周一市场", ("day", date(2025, 1, 6))),
            ("本周二行情", ("day", date(2025, 1, 7))),
            ("本周三半导体", ("day", date(2025, 1, 8))),
            ("本周四盘面", ("day", date(2025, 1, 9))),
            ("本周五复盘", ("day", date(2025, 1, 10))),
            ("这周三白酒", ("day", date(2025, 1, 8))),
            # ── 上周X（一~五）单日（跨年：上周一为 2024-12-30）──
            ("上周一市场", ("day", date(2024, 12, 30))),
            ("上周二行情", ("day", date(2024, 12, 31))),
            ("上周三新能源", ("day", date(2025, 1, 1))),
            ("上周四盘面", ("day", date(2025, 1, 2))),
            ("上周五白酒板块怎么样", ("day", date(2025, 1, 3))),
            ("上礼拜三市场", ("day", date(2025, 1, 1))),
            # ── 周区间：上周（完整上一周，跨年）──
            ("上周白酒板块怎么样", ("week", date(2024, 12, 30), date(2025, 1, 5))),
            ("上礼拜半导体", ("week", date(2024, 12, 30), date(2025, 1, 5))),
            # ── 周区间：本周（周一~今日，周五锚点即整周）──
            ("这周半导体周报", ("week", date(2025, 1, 6), date(2025, 1, 10))),
            ("本周市场行情", ("week", date(2025, 1, 6), date(2025, 1, 10))),
            ("近一周新能源板块", ("week", date(2025, 1, 6), date(2025, 1, 10))),
            ("这周以来白酒", ("week", date(2025, 1, 6), date(2025, 1, 10))),
            ("本周以来大盘", ("week", date(2025, 1, 6), date(2025, 1, 10))),
            # ── 无引用 ──
            ("茅台怎么样", ("none", None)),
            ("复盘", ("none", None)),
            ("给我讲个笑话", ("none", None)),
            ("", ("none", None)),
        ],
    )
    def test_anchor_friday(self, msg, expected):
        assert _resolve_time_reference(msg, ANCHOR_FRI) == expected

    def test_anchor_none_message(self):
        """None/非字符串消息按无引用处理，绝不抛出。"""
        assert _resolve_time_reference(None, ANCHOR_FRI) == ("none", None)

    def test_friday_specific_beats_week_range(self):
        """『上周五』必须判为单日而不是周区间（判定顺序防回归）。"""
        kind = _resolve_time_reference("上周五白酒板块", ANCHOR_FRI)
        assert kind[0] == "day"
        assert kind[1] == date(2025, 1, 3)


class TestResolveTimeReferenceBoundaries:
    """跨周末 / 跨年边界。"""

    def test_sunday_anchor_current_week_ends_today(self):
        """周日锚点：本周结束日 = min(周日, 今日) = 周日当天。"""
        assert _resolve_time_reference("本周白酒板块", ANCHOR_SUN) == (
            "week", date(2025, 1, 6), date(2025, 1, 12),
        )

    def test_sunday_anchor_yesterday_is_saturday(self):
        """周日锚点：『昨天』= 周六（交易日对齐由 _get_latest_trade_date 负责）。"""
        assert _resolve_time_reference("昨天市场怎么样", ANCHOR_SUN) == (
            "day", date(2025, 1, 11),
        )

    def test_monday_anchor_last_week(self):
        """周一锚点：『上周』= 刚过去的完整一周；『本周』只有今天一天。"""
        assert _resolve_time_reference("上周白酒板块", ANCHOR_MON) == (
            "week", date(2025, 1, 6), date(2025, 1, 12),
        )
        assert _resolve_time_reference("这周市场", ANCHOR_MON) == (
            "week", date(2025, 1, 13), date(2025, 1, 13),
        )
        assert _resolve_time_reference("本周一行情", ANCHOR_MON) == (
            "day", date(2025, 1, 13),
        )

    def test_new_year_boundary(self):
        """跨年周：2025-01-01（周三）所在周周一是 2024-12-30。"""
        assert _resolve_time_reference("本周一市场", ANCHOR_NEWYEAR) == (
            "day", date(2024, 12, 30),
        )
        assert _resolve_time_reference("上周白酒板块", ANCHOR_NEWYEAR) == (
            "week", date(2024, 12, 23), date(2024, 12, 29),
        )
        assert _resolve_time_reference("上周一行情", ANCHOR_NEWYEAR) == (
            "day", date(2024, 12, 23),
        )
        assert _resolve_time_reference("昨天市场怎么样", ANCHOR_NEWYEAR) == (
            "day", date(2024, 12, 31),
        )


# ════════════════════════════════════════════════════════════════
# 2. 单日交易日对齐：_get_latest_trade_date 的 ref 传参正确性
# ════════════════════════════════════════════════════════════════


class TestSingleDayAlignment:
    """『昨天』场景必须让数据真的是昨天的——验证 ref 传参。"""

    def test_market_review_yesterday_aligns_trade_date(self):
        """message='昨天市场怎么样' → _get_latest_trade_date 以昨天日期为 ref。"""
        agent = _make_agent()
        agent._call_llm = AsyncMock(return_value={"role": "assistant", "content": "ok"})
        trade_mock = MagicMock(return_value=FIXED_TRADE_DATE)

        with patch("agent.orchestrator._get_latest_trade_date", trade_mock), patch(
            "agent.orchestrator.collect_market_snapshot",
            AsyncMock(return_value=SimpleNamespace()),
        ), patch(
            "agent.orchestrator.format_market_data_for_prompt", return_value="DATA"
        ):
            asyncio.run(agent._market_review(stream=False, message="昨天市场怎么样"))

        expected = (datetime.now() - timedelta(days=1)).date()
        assert trade_mock.call_count >= 1
        ref_arg = trade_mock.call_args.args[0]
        assert ref_arg == expected, (
            f"『昨天』场景应以 {expected} 为 ref 对齐交易日，实际 {ref_arg}"
        )
        # 回答头部声明：prompt 含数据对应日期注明指令
        user_prompt = agent._call_llm.await_args.args[1]
        assert "数据对应的交易日期" in user_prompt

    def test_market_review_no_reference_keeps_now(self):
        """无时间引用 → 保持现状以 datetime.now() 为 ref（防回归）。"""
        agent = _make_agent()
        agent._call_llm = AsyncMock(return_value={"role": "assistant", "content": "ok"})
        trade_mock = MagicMock(return_value=FIXED_TRADE_DATE)

        with patch("agent.orchestrator._get_latest_trade_date", trade_mock), patch(
            "agent.orchestrator.collect_market_snapshot",
            AsyncMock(return_value=SimpleNamespace()),
        ), patch(
            "agent.orchestrator.format_market_data_for_prompt", return_value="DATA"
        ):
            asyncio.run(agent._market_review(stream=False))

        ref_arg = trade_mock.call_args.args[0]
        # datetime.now() 是 datetime 实例；解析出的目标日期是 date 实例——
        # isinstance(x, datetime) 可区分（date 不是 datetime 子类）
        assert isinstance(ref_arg, datetime), (
            f"无引用场景应保持 datetime.now() 现状，实际 {type(ref_arg)}"
        )

    def test_sector_deep_dive_day_before_yesterday(self):
        """message='前天白酒板块' → _get_latest_trade_date 以前天日期为 ref。"""
        agent = _make_agent()
        agent._call_llm = AsyncMock(return_value={"role": "assistant", "content": "ok"})
        trade_mock = MagicMock(return_value=FIXED_TRADE_DATE)

        with patch("agent.orchestrator._get_latest_trade_date", trade_mock), patch(
            "agent.orchestrator.collect_market_snapshot",
            AsyncMock(return_value=SimpleNamespace()),
        ), patch(
            "agent.orchestrator.format_market_data_for_prompt", return_value="DATA"
        ), patch(
            "agent.data_fetcher.fetch_sector_valuation", MagicMock(return_value={})
        ), patch(
            "agent.data_fetcher.fetch_sector_moneyflow", MagicMock(return_value={})
        ), patch(
            "agent.data_fetcher.fetch_sector_earnings", MagicMock(return_value={})
        ):
            asyncio.run(
                agent._sector_deep_dive("食品饮料", stream=False, message="前天白酒板块怎么样")
            )

        expected = (datetime.now() - timedelta(days=2)).date()
        ref_arg = trade_mock.call_args.args[0]
        assert ref_arg == expected, (
            f"『前天』场景应以 {expected} 为 ref 对齐交易日，实际 {ref_arg}"
        )

    def test_news_only_yesterday_aligns_header_date(self):
        """message='昨天有什么新闻' → 头部日期以昨天为 ref 对齐。"""
        agent = _make_agent()
        agent._call_llm = AsyncMock(return_value={"role": "assistant", "content": "ok"})
        trade_mock = MagicMock(return_value=FIXED_TRADE_DATE)

        with patch("agent.orchestrator._get_latest_trade_date", trade_mock), patch(
            "agent.data_fetcher.fetch_news_pool",
            MagicMock(return_value={"mcp": [], "flash": []}),
        ):
            asyncio.run(agent._news_only(None, stream=False, message="昨天有什么新闻"))

        expected = (datetime.now() - timedelta(days=1)).date()
        ref_arg = trade_mock.call_args.args[0]
        assert ref_arg == expected, (
            f"『昨天』新闻场景应以 {expected} 为 ref 对齐交易日，实际 {ref_arg}"
        )
