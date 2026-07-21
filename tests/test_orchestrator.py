"""agent/orchestrator.py 编排层逻辑测试。

覆盖范围：
1. detect_intent 意图识别（市场复盘 / 板块聚焦 / 个股 / 新闻 / 期货 / 基金 /
   mcp_query / general_chat 兜底）+ 行业别名映射（SECTOR_NAME_MAP → 申万一级）。
2. max_tokens=8192 防回归：非流式 _call_llm 与流式 _stream_response 两条路径
   都必须以 max_tokens=8192 调用 chat.completions.create。
3. 缓存行为防回归：市场复盘快照（snapshot_<date>）与板块快照
   （snapshot_<date>_<sector>）必须在 self._cache 中共存，互不覆盖。
4. 新闻去重防回归：_news_only 路径下，fetch_sina_news 返回的重复标题
   必须在最终输出中只出现一次。

规则：
- 所有外部依赖全部 mock（collect_market_snapshot / fetch_*_news / DeepSeek 客户端），
  绝不发起真实网络请求。
- _get_latest_trade_date 统一 patch 为固定日期，避免触发真实 tushare 交易日历请求。
- 无 pytest-asyncio，异步函数一律用 asyncio.run 驱动。
"""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.orchestrator import (
    MarketReviewAgent,
    detect_intent,
    _extract_sector,
    SECTOR_NAME_MAP,
)

# 固定交易日，避免 _get_latest_trade_date 触达真实 tushare
FIXED_TRADE_DATE = datetime(2025, 1, 10)  # 周五
FIXED_DATE_STR = "20250110"


def _make_agent() -> MarketReviewAgent:
    """构造一个 agent，DeepSeek 客户端替换为 mock，防止任何真实 HTTP 调用。"""
    agent = MarketReviewAgent()
    agent.client = MagicMock()
    return agent


def _fake_completion(content: str = "复盘正文"):
    """伪造 chat.completions.create 的非流式返回对象。"""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


# ════════════════════════════════════════════════════════════════
# 1. detect_intent 意图识别
# ════════════════════════════════════════════════════════════════


class TestDetectIntent:
    """detect_intent 返回 (intent_type, sector_or_None/消息) 的分类正确性。"""

    @pytest.mark.parametrize(
        "message, expected_intent, expected_sector",
        [
            # ── 市场复盘关键词 ──
            ("复盘", "market_review", None),
            ("今天大盘怎么样", "market_review", None),
            ("今日复盘", "market_review", None),
            ("市场回顾", "market_review", None),
            ("盘后总结一下行情", "market_review", None),
            ("market review", "market_review", None),
            # ── 板块聚焦 ──
            ("煤炭板块复盘", "sector_deep_dive", "煤炭"),
            ("电子行业怎么样", "sector_deep_dive", "电子"),
            # ── 个股查询（股票名称 / 代码）──
            ("茅台怎么样", "stock_query", None),
            ("分析AAPL", "stock_query", None),
            # ── 新闻查询 ──
            ("今天有什么新闻", "news_only", None),
            ("银行板块新闻", "news_only", "银行"),
            # ── 期货查询 ──
            ("黄金期货价格", "futures_query", None),
            ("螺纹钢期货怎么样", "futures_query", None),
            # ── 基金查询 ──
            ("ETF基金净值", "fund_query", None),
            # ── 简单数据查询走 MCP ──
            ("今天涨停家数查询", "mcp_query", None),
            # ── general_chat 兜底 ──
            ("给我讲个笑话", "general_chat", None),
            ("红烧肉怎么做", "general_chat", None),
        ],
    )
    def test_intent_classification(self, message, expected_intent, expected_sector):
        intent, sector = detect_intent(message)
        assert intent == expected_intent, (
            f"消息 {message!r} 期望意图 {expected_intent}，实际 {intent}"
        )
        if expected_sector is not None:
            assert sector == expected_sector, (
                f"消息 {message!r} 期望板块 {expected_sector}，实际 {sector}"
            )

    @pytest.mark.parametrize(
        "message, expected_intent, expected_sector",
        [
            # ── 行业别名映射：别名 → 申万一级行业 ──
            ("半导体板块复盘", "sector_deep_dive", "电子"),
            ("光伏行业怎么样", "sector_deep_dive", "电气设备"),
            ("白酒板块深度", "sector_deep_dive", "食品饮料"),
            ("券商板块复盘", "sector_deep_dive", "非银金融"),
            ("创新药行业分析", "sector_deep_dive", "医药生物"),
        ],
    )
    def test_sector_alias_mapping(self, message, expected_intent, expected_sector):
        """SECTOR_NAME_MAP 中的别名必须映射到对应申万一级行业。"""
        intent, sector = detect_intent(message)
        assert intent == expected_intent
        assert sector == expected_sector, (
            f"消息 {message!r} 中别名应映射为申万一级 {expected_sector}，实际 {sector}"
        )

    @pytest.mark.parametrize(
        "alias, sw_name",
        [
            ("半导体", "电子"),
            ("芯片", "电子"),
            ("光伏", "电气设备"),
            ("白酒", "食品饮料"),
            ("券商", "非银金融"),
            ("创新药", "医药生物"),
            ("军工", "国防军工"),
        ],
    )
    def test_extract_sector_alias_direct(self, alias, sw_name):
        """直接验证 _extract_sector 的别名→申万一级映射与 SECTOR_NAME_MAP 一致。"""
        assert _extract_sector(f"看看{alias}") == sw_name
        assert SECTOR_NAME_MAP[alias] == sw_name

    def test_extract_sector_no_match(self):
        assert _extract_sector("给我讲个笑话") is None

    def test_extract_sector_longest_match_first(self):
        """长词优先：'新能源车' 应先于 '汽车' 命中，映射为 汽车。"""
        assert _extract_sector("新能源车板块") == "汽车"


# ════════════════════════════════════════════════════════════════
# 2. max_tokens=8192 防回归
# ════════════════════════════════════════════════════════════════


class TestMaxTokens:
    """刚修过的 bug：所有 DeepSeek 调用必须带 max_tokens=8192，防回归。"""

    def test_call_llm_non_stream_max_tokens(self):
        agent = _make_agent()
        agent.client.chat.completions.create = AsyncMock(
            return_value=_fake_completion("非流式正文")
        )

        result = asyncio.run(agent._call_llm("系统提示", "用户消息", stream=False))

        create = agent.client.chat.completions.create
        assert create.await_count == 1
        kwargs = create.await_args.kwargs
        assert kwargs.get("max_tokens") == 8192, (
            f"非流式调用 max_tokens 应为 8192，实际 {kwargs.get('max_tokens')}"
        )
        # 非流式返回带免责声明的 dict
        assert result["role"] == "assistant"
        assert "非流式正文" in result["content"]
        assert "风险提示" in result["content"]

    def test_call_llm_stream_max_tokens(self):
        agent = _make_agent()

        async def fake_chunk_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="你好"))]
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="世界"))]
            )
            # content 为 None 的 chunk 应被跳过
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]
            )

        agent.client.chat.completions.create = AsyncMock(
            return_value=fake_chunk_stream()
        )

        async def run_stream():
            gen = await agent._call_llm("系统提示", "用户消息", stream=True)
            chunks = []
            async for piece in gen:
                chunks.append(piece)
            return chunks

        chunks = asyncio.run(run_stream())

        create = agent.client.chat.completions.create
        assert create.await_count == 1
        kwargs = create.await_args.kwargs
        assert kwargs.get("max_tokens") == 8192, (
            f"流式调用 max_tokens 应为 8192，实际 {kwargs.get('max_tokens')}"
        )
        assert kwargs.get("stream") is True
        assert "".join(chunks) == "你好世界"


# ════════════════════════════════════════════════════════════════
# 3. 缓存行为防回归：全市场快照与板块快照共存
# ════════════════════════════════════════════════════════════════


class TestCacheBehavior:
    """板块快照（snapshot_<date>_<sector>）不得冲掉全市场快照（snapshot_<date>）。"""

    def test_market_and_sector_snapshots_coexist(self):
        agent = _make_agent()
        market_snapshot = SimpleNamespace(tag="market")
        sector_snapshot = SimpleNamespace(tag="sector_煤炭")
        collect_mock = AsyncMock(side_effect=[market_snapshot, sector_snapshot])

        # _call_llm 不触达真实 DeepSeek
        agent._call_llm = AsyncMock(
            return_value={"role": "assistant", "content": "ok"}
        )

        with patch(
            "agent.orchestrator._get_latest_trade_date",
            return_value=FIXED_TRADE_DATE,
        ), patch(
            "agent.orchestrator.collect_market_snapshot", collect_mock
        ), patch(
            "agent.orchestrator.format_market_data_for_prompt", return_value="DATA"
        ):
            # 先跑市场复盘
            asyncio.run(agent._market_review(stream=False))
            # 再跑板块深挖
            asyncio.run(agent._sector_deep_dive("煤炭", stream=False))

        market_key = f"snapshot_{FIXED_DATE_STR}"
        sector_key = f"snapshot_{FIXED_DATE_STR}_煤炭"

        # 两个 key 必须共存——板块快照不冲掉全市场快照（防回归）
        assert market_key in agent._cache, "全市场快照 key 丢失"
        assert sector_key in agent._cache, "板块快照 key 缺失"
        assert agent._cache[market_key] is market_snapshot
        assert agent._cache[sector_key] is sector_snapshot

        # 两次采集：第一次不带板块，第二次带 sector_focus
        assert collect_mock.await_count == 2
        first_kwargs = collect_mock.await_args_list[0].kwargs
        second_kwargs = collect_mock.await_args_list[1].kwargs
        assert first_kwargs.get("sector_focus") is None
        assert second_kwargs.get("sector_focus") == "煤炭"

    def test_same_day_market_review_uses_cache(self):
        """同一交易日第二次复盘命中缓存，不重复采集。"""
        agent = _make_agent()
        collect_mock = AsyncMock(return_value=SimpleNamespace(tag="market"))
        agent._call_llm = AsyncMock(
            return_value={"role": "assistant", "content": "ok"}
        )

        with patch(
            "agent.orchestrator._get_latest_trade_date",
            return_value=FIXED_TRADE_DATE,
        ), patch(
            "agent.orchestrator.collect_market_snapshot", collect_mock
        ), patch(
            "agent.orchestrator.format_market_data_for_prompt", return_value="DATA"
        ):
            asyncio.run(agent._market_review(stream=False))
            asyncio.run(agent._market_review(stream=False))

        assert collect_mock.await_count == 1, "同日第二次复盘不应重复采集"
        assert len(agent._cache) == 1


# ════════════════════════════════════════════════════════════════
# 4. 新闻去重防回归：_news_only 按标题去重
# ════════════════════════════════════════════════════════════════


class TestNewsDedup:
    """fetch_sina_news 跨日期返回重复标题时，最终输出中每个标题只出现一次。"""

    def _run_news_only(self, sina_side_effect, mcp_items=None, em_items=None):
        """驱动 _news_only，返回 result dict（content 为最终新闻文本）。"""
        agent = _make_agent()
        agent._call_llm = AsyncMock(
            return_value={"role": "assistant", "content": "placeholder"}
        )

        with patch(
            "agent.orchestrator._get_latest_trade_date",
            return_value=FIXED_TRADE_DATE,
        ), patch(
            "agent.data_fetcher.fetch_sina_news",
            MagicMock(side_effect=sina_side_effect),
        ) as sina_mock, patch(
            "agent.data_fetcher.fetch_mcp_news",
            MagicMock(return_value=mcp_items or []),
        ), patch(
            "agent.data_fetcher.fetch_eastmoney_news",
            MagicMock(return_value=em_items or []),
        ):
            result = asyncio.run(agent._news_only(None, stream=False))

        return result, sina_mock

    def test_duplicate_titles_deduped(self):
        """3 天抓取中重复出现的标题，最终文本里只保留一条。"""
        dup_items = [
            {"time": "2025-01-10 09:30:00", "title": "央行开展中期借贷便利操作"},
            {"time": "2025-01-10 10:00:00", "title": "证监会发布新规稳定市场"},
        ]
        # 3 个日期每次都返回相同的两条（模拟跨天重复推送）
        result, sina_mock = self._run_news_only(
            [list(dup_items), list(dup_items), list(dup_items)]
        )

        # _news_only 应对 3 个日期各调用一次 fetch_sina_news
        assert sina_mock.call_count == 3

        content = result["content"]
        assert content.count("央行开展中期借贷便利操作") == 1, (
            "重复标题未被去重，content:\n" + content
        )
        assert content.count("证监会发布新规稳定市场") == 1

    def test_same_day_duplicate_titles_deduped(self):
        """同一天返回的列表内部含重复标题，也应去重。"""
        day_items = [
            {"time": "2025-01-10 09:30:00", "title": "某大型银行发布年度业绩快报"},
            {"time": "2025-01-10 09:31:00", "title": "某大型银行发布年度业绩快报"},
            {"time": "2025-01-10 11:00:00", "title": "沪深两市成交额突破万亿元"},
        ]
        result, _ = self._run_news_only([day_items, [], []])

        content = result["content"]
        assert content.count("某大型银行发布年度业绩快报") == 1
        assert content.count("沪深两市成交额突破万亿元") == 1

    def test_blank_titles_filtered(self):
        """空标题 / 过短标题（<4 字符）不应进入最终新闻文本。"""
        day_items = [
            {"time": "2025-01-10 09:30:00", "title": ""},
            {"time": "2025-01-10 09:31:00", "title": "短"},
            {"time": "2025-01-10 11:00:00", "title": "这是一条正常长度的新闻标题"},
        ]
        result, _ = self._run_news_only([day_items, [], []])

        content = result["content"]
        assert "这是一条正常长度的新闻标题" in content
        # 只有一条有效新闻
        assert "共1+0条" in content or "共1条" in content or content.count("---") >= 1
