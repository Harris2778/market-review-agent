"""tests/test_snapshot_news_mcp.py — 复盘快照新闻切智研 MCP（分工A）契约测试。

覆盖范围：
1. collect_market_snapshot 新闻出口契约：news_items 键固定为
   mcp/flash/cls_telegraph/global（旧 eastmoney/sina/ts_news 六源键消失）。
2. 多关键词并行：_SNAPSHOT_NEWS_KEYWORDS 每词各调一次 fetch_mcp_news，
   快讯源 fetch_mcp_flash 按 _SNAPSHOT_FLASH_LIMIT 调用。
3. 出口去重：跨关键词同标题经 _dedup_news_fuzzy 只保留一条；
   快讯与关键词源跨源模糊键重复的不重复计入。
4. 单 task 失败降级为空，不影响其他新闻 task 与非新闻 task（现有模式保持）。
5. 广度保证：关键词数 × 每词条数 + 快讯条数 ≥ 400（去重前理论上限）。
6. sector_focus 下 mcp 新闻经 filter_news_by_sector 过滤（正文 content 参与匹配）。
7. format_market_data_for_prompt：渲染智研双源新闻段，旧源段落标题不再出现。

规则（与 tests/test_news.py 一致）：
- 所有外部依赖全部 mock，绝不发起真实网络请求。
- 无 pytest-asyncio，异步函数一律用 asyncio.run 驱动。
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from agent import data_fetcher
from agent.data_fetcher import MarketSnapshot

DAY0 = "2026-07-21"

# collect_market_snapshot 内除新闻外的全部 task 依赖，统一 mock 为空
_EMPTY_FETCHES = {
    "fetch_a_share_indices": {},
    "fetch_shenwan_sectors": [],
    "fetch_fund_flows": {},
    "fetch_global_indices": {},
    "fetch_broker_recommendations": [],
    "fetch_forex": {},
    "fetch_shibor": {},
    "fetch_north_holdings": [],
    "fetch_top_list": [],
    "fetch_china_macro": {},
    "fetch_us_macro": {},
    "fetch_finnhub_news": [],
    "fetch_economic_calendar": [],
    "fetch_market_breadth": {},
    "fetch_hot_stocks": [],
    "fetch_us_breadth": {},
    "fetch_limit_up_pool": [],
    "fetch_lian_ban": [],
    "fetch_forecast": [],
    "fetch_express": [],
    "fetch_block_trades_tushare": [],
    "fetch_ggt_daily": [],
    "fetch_repurchase": [],
    "fetch_share_float": [],
    "fetch_fund_list": [],
    "fetch_strong_sectors": [],
    "fetch_northbound_flow": [],
    "fetch_us_sectors": [],
    "fetch_hk_sectors": [],
    "fetch_cls_telegraph": [],
    "fetch_sector_volume_all": {},
    "fetch_sector_stock_detail": None,
}


def _item(title, time=f"{DAY0} 09:30:00", source="智研", **extra):
    it = {"title": title, "time": time, "source": source}
    it.update(extra)
    return it


def _run_snapshot(mcp_side_effect=None, mcp_return=None, flash=None,
                  sector_focus=None, **overrides):
    """以全 mock 数据源驱动 collect_market_snapshot。"""
    patchers = {name: MagicMock(return_value=ret)
                for name, ret in _EMPTY_FETCHES.items()}
    if mcp_side_effect is not None:
        patchers["fetch_mcp_news"] = MagicMock(side_effect=mcp_side_effect)
    else:
        patchers["fetch_mcp_news"] = MagicMock(return_value=mcp_return or [])
    patchers["fetch_mcp_flash"] = MagicMock(return_value=flash or [])
    for name, ret in overrides.items():
        patchers[name] = MagicMock(return_value=ret)

    with patch.multiple(data_fetcher, **patchers):
        snap = asyncio.run(data_fetcher.collect_market_snapshot(
            DAY0.replace("-", ""), sector_focus=sector_focus))
    return snap, patchers


# ════════════════════════════════════════════════════════════════
# 1. 新闻出口契约：智研双源键 + 保留源键，旧六源键消失
# ════════════════════════════════════════════════════════════════

class TestSnapshotNewsKeys:
    def test_news_items_keys_are_mcp_contract(self):
        cls = [_item("财联社电报样例标题", source="财联社电报", brief="电报正文")]
        fh = [_item("Global markets rally", source="Finnhub")]
        snap, _ = _run_snapshot(mcp_return=[_item("央行开展MLF操作")],
                                flash=[_item("沪深两市成交额突破万亿", source="智研快讯")],
                                fetch_cls_telegraph=cls,
                                fetch_finnhub_news=fh)

        assert set(snap.news_items.keys()) == {"mcp", "flash", "cls_telegraph", "global"}
        for old_key in ("eastmoney", "sina", "ts_news"):
            assert old_key not in snap.news_items, f"旧源键 {old_key} 应已移除"
        assert snap.news_items["cls_telegraph"] == cls, "财联社电报源应保留"
        assert snap.news_items["global"] == fh, "fh_news（Finnhub）不在弃用范围，应保留"

    def test_mcp_and_flash_content_routed(self):
        snap, _ = _run_snapshot(
            mcp_return=[_item("央行开展5000亿元MLF操作")],
            flash=[_item("北向资金净流入超百亿", source="智研快讯")])
        assert [it["title"] for it in snap.news_items["mcp"]] == ["央行开展5000亿元MLF操作"]
        assert [it["title"] for it in snap.news_items["flash"]] == ["北向资金净流入超百亿"]


# ════════════════════════════════════════════════════════════════
# 2. 多关键词并行 + 抓取量参数
# ════════════════════════════════════════════════════════════════

class TestKeywordFanout:
    def test_each_keyword_fetched_once_with_per_keyword_limit(self):
        _, patchers = _run_snapshot()
        mock_news = patchers["fetch_mcp_news"]
        called_kws = sorted(c.args[0] for c in mock_news.call_args_list)
        assert called_kws == sorted(data_fetcher._SNAPSHOT_NEWS_KEYWORDS), (
            f"每个关键词应各调一次 fetch_mcp_news: {called_kws}"
        )
        for c in mock_news.call_args_list:
            assert c.args[1] == data_fetcher._SNAPSHOT_NEWS_PER_KEYWORD

    def test_flash_fetched_with_flash_limit(self):
        _, patchers = _run_snapshot()
        patchers["fetch_mcp_flash"].assert_called_once_with(
            data_fetcher._SNAPSHOT_FLASH_LIMIT)

    def test_breadth_contract_at_least_400_pre_dedup(self):
        """广度只能多不能少：关键词×每词条数 + 快讯条数 ≥ 400（去重前上限）。"""
        theoretical = (len(data_fetcher._SNAPSHOT_NEWS_KEYWORDS)
                       * data_fetcher._SNAPSHOT_NEWS_PER_KEYWORD
                       + data_fetcher._SNAPSHOT_FLASH_LIMIT)
        assert theoretical >= 400, f"去重前理论上限 {theoretical} < 400"


# ════════════════════════════════════════════════════════════════
# 3. 出口去重：跨关键词 + 跨源模糊去重
# ════════════════════════════════════════════════════════════════

class TestSnapshotNewsDedup:
    def test_cross_keyword_fuzzy_dedup(self):
        dup = "央行开展5000亿元中期借贷便利操作"

        def _by_kw(keyword, limit=30):
            if keyword == "A股":
                return [_item(dup), _item("沪深两市成交额突破一万五千亿元")]
            if keyword == "央行":
                return [_item(f"【快讯】{dup}")]  # 换皮同题：剥【】后模糊键相同
            return []

        snap, _ = _run_snapshot(mcp_side_effect=_by_kw)
        titles = [it["title"] for it in snap.news_items["mcp"]]
        assert len(titles) == 2, f"跨关键词同题应模糊去重为一条: {titles}"
        assert titles[0] == dup  # 保留先出现者

    def test_flash_deduped_against_mcp_cross_source(self):
        dup = "证监会就市值管理指引公开征求意见"
        snap, _ = _run_snapshot(
            mcp_side_effect=lambda kw, limit=30: [_item(dup)] if kw == "A股" else [],
            flash=[_item(dup, source="智研快讯"),
                   _item("北向资金单日净流入超百亿", source="智研快讯")])
        assert [it["title"] for it in snap.news_items["mcp"]] == [dup]
        assert [it["title"] for it in snap.news_items["flash"]] == [
            "北向资金单日净流入超百亿"], "与 mcp 模糊键重复的快讯不应重复计入"


# ════════════════════════════════════════════════════════════════
# 4. 单 task 失败降级为空，不影响其他 task
# ════════════════════════════════════════════════════════════════

class TestFailureDegradation:
    def test_flash_failure_keeps_mcp(self):
        snap, patchers = _run_snapshot(
            mcp_side_effect=lambda kw, limit=30: [_item(f"{keyword}相关新闻标题")] if kw == "A股" else [],
        )
        # 重新跑：让 flash 抛异常
        with patch.multiple(data_fetcher, **{
                **{n: MagicMock(return_value=r) for n, r in _EMPTY_FETCHES.items()},
                "fetch_mcp_news": MagicMock(
                    side_effect=lambda kw, limit=30: [_item("央行降准释放流动性")] if kw == "央行" else []),
                "fetch_mcp_flash": MagicMock(side_effect=RuntimeError("智研快讯超时")),
        }):
            snap = asyncio.run(data_fetcher.collect_market_snapshot(DAY0.replace("-", "")))
        assert snap.news_items["flash"] == [], "失败源应降级为空列表"
        assert [it["title"] for it in snap.news_items["mcp"]] == ["央行降准释放流动性"]

    def test_single_keyword_failure_keeps_other_keywords(self):
        def _by_kw(keyword, limit=30):
            if keyword == "政策":
                raise RuntimeError("qNewsSearch 限流")
            if keyword == "A股":
                return [_item("沪深两市成交额突破万亿")]
            return []

        snap, _ = _run_snapshot(mcp_side_effect=_by_kw)
        assert [it["title"] for it in snap.news_items["mcp"]] == ["沪深两市成交额突破万亿"], (
            "单词失败不应影响其他关键词 task"
        )


# ════════════════════════════════════════════════════════════════
# 5. sector_focus：mcp 新闻按行业关键词过滤（content 参与匹配）
# ════════════════════════════════════════════════════════════════

class TestSectorFocusFilter:
    def test_mcp_filtered_by_sector_keywords(self):
        elec = _item("半导体设备龙头发布新一代光刻机", content="芯片产业链持续景气")
        liquor = _item("贵州茅台白酒春节销量超预期")

        def _by_kw(keyword, limit=30):
            return {"A股": [elec], "公司": [liquor]}.get(keyword, [])

        snap, _ = _run_snapshot(mcp_side_effect=_by_kw, sector_focus="电子")
        titles = [it["title"] for it in snap.news_items["mcp"]]
        assert elec["title"] in titles
        assert liquor["title"] not in titles, "无关行业标题应被过滤"

    def test_mcp_content_field_participates_in_sector_match(self):
        """标题不含关键词但 content 含关键词的智研条目也应被行业过滤保留。"""
        vague = _item("某龙头企业发布重要公告", content="公司主营半导体晶圆代工业务")
        snap, _ = _run_snapshot(
            mcp_side_effect=lambda kw, limit=30: [vague] if kw == "公司" else [],
            sector_focus="电子")
        assert [it["title"] for it in snap.news_items["mcp"]] == [vague["title"]]


# ════════════════════════════════════════════════════════════════
# 6. format_market_data_for_prompt：智研双源渲染段
# ════════════════════════════════════════════════════════════════

class TestFormatPromptNewsSections:
    def _snapshot(self, news_items):
        snap = MarketSnapshot(date=DAY0.replace("-", ""))
        snap.news_items = news_items
        snap.macro_data = {}
        return snap

    def test_renders_mcp_and_flash_sections(self):
        snap = self._snapshot({
            "mcp": [_item("央行开展5000亿元MLF操作", content="人民银行今日开展中期借贷便利操作，中标利率不变。")],
            "flash": [_item("北向资金净流入超百亿", source="智研快讯")],
            "cls_telegraph": [],
            "global": [],
        })
        text = data_fetcher.format_market_data_for_prompt(snap)
        assert "### 智研7x24快讯（共1条" in text
        assert "北向资金净流入超百亿" in text
        assert "### 智研财经新闻（多关键词去重，共1条" in text
        assert "央行开展5000亿元MLF操作" in text
        # mcp 条目带 content 摘要（对齐旧 ts_news 渲染）
        assert "人民银行今日开展中期借贷便利操作" in text

    def test_old_source_sections_gone(self):
        snap = self._snapshot({
            "mcp": [_item("央行开展5000亿元MLF操作")],
            "flash": [_item("北向资金净流入超百亿", source="智研快讯")],
            "cls_telegraph": [],
            "global": [],
        })
        text = data_fetcher.format_market_data_for_prompt(snap)
        for old_header in ("新浪财经历史新闻", "东方财富7x24实时快讯", "财联社新闻（Tushare"):
            assert old_header not in text, f"旧源渲染段 {old_header!r} 应已移除"

    def test_empty_news_renders_nothing_and_no_crash(self):
        snap = self._snapshot({"mcp": [], "flash": [], "cls_telegraph": [], "global": []})
        text = data_fetcher.format_market_data_for_prompt(snap)
        assert "智研" not in text
