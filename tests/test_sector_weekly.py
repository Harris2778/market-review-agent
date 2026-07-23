"""板块周报（sector_weekly 新功能）测试。

覆盖范围：
1. detect_intent 周区间识别：本周/上周+板块 → sector_weekly；显式新闻/基金
   诉求护栏不被抢；『上周市场』（无板块）保持 market_review。
2. 周度行情采集：fetch_shenwan_sectors 按当周各交易日逐日取数，节假日
   重复行按签名去重，周末不取数；周涨跌幅按逐日复合计算。
3. 当周新闻：fetch_news_pool 按周区间过滤 + 板块别名关键词匹配 + 跨源去重。
4. 券商研报：search_reports(industry=板块, days=14) 检索，三家券商以上
   分组聚合（评级/评级变动/目标价/EPS）；无研报时诚实降级
   「券商研报数据未覆盖」，禁止编造。
5. 周报主路径 _sector_weekly_review 端到端接线（全 mock）。

规则（与项目其他测试一致）：
- 所有外部依赖全部 mock，绝不发起真实网络请求。
- 无 pytest-asyncio，异步函数一律用 asyncio.run 驱动。
"""

import asyncio
from datetime import datetime, timedelta, date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.orchestrator import (
    MarketReviewAgent,
    detect_intent,
    _collect_sector_weekly_quotes,
    _format_sector_weekly_market_block,
    _collect_sector_weekly_news,
    _format_weekly_broker_block,
    _sector_news_keywords,
)


def _make_agent() -> MarketReviewAgent:
    agent = MarketReviewAgent()
    agent.client = MagicMock()
    return agent


def _sw_row(name, pct, close, amount, vol):
    """伪造 fetch_shenwan_sectors 的单板块行（字段口径与 data_fetcher 一致）。"""
    return {
        "name": name, "pct_chg": pct, "close": close,
        "amount": amount, "vol": vol, "tag": "中性",
    }


# 固定周：2025-01-06(周一)~2025-01-10(周五)
WEEK_START = date(2025, 1, 6)
WEEK_END = date(2025, 1, 10)
WEEK_DATES = [WEEK_START + timedelta(days=i) for i in range(5)]  # 周一~周五


# ════════════════════════════════════════════════════════════════
# 1. detect_intent 周区间识别
# ════════════════════════════════════════════════════════════════


class TestWeeklyIntent:
    """本周/上周+板块 → sector_weekly；护栏词让位既有路由。"""

    @pytest.mark.parametrize(
        "message, expected_intent, expected_sector",
        [
            ("上周白酒板块怎么样", "sector_weekly", "食品饮料"),
            ("这周半导体周报", "sector_weekly", "电子"),
            ("本周新能源板块表现", "sector_weekly", "电气设备"),
            ("近一周医药板块如何", "sector_weekly", "医药生物"),
            ("上礼拜券商板块回顾", "sector_weekly", "非银金融"),
            # ── 护栏：显式新闻/基金诉求不被周报抢走 ──
            ("上周白酒板块新闻", "news_only", "食品饮料"),
            ("上周白酒板块基金净值", "fund_query", None),
            # ── 无板块：全市场保持既有 market_review（全市场周报不做，见报告）──
            ("上周市场怎么样", "market_review", None),
            # ── 无周区间：板块问题保持既有板块深挖 ──
            ("白酒板块怎么样", "sector_deep_dive", "食品饮料"),
            ("昨天白酒板块怎么样", "sector_deep_dive", "食品饮料"),
        ],
    )
    def test_weekly_intent_classification(self, message, expected_intent, expected_sector):
        intent, sector = detect_intent(message)
        assert intent == expected_intent, (
            f"消息 {message!r} 期望意图 {expected_intent}，实际 {intent}"
        )
        if expected_sector is not None:
            assert sector == expected_sector, (
                f"消息 {message!r} 期望板块 {expected_sector}，实际 {sector}"
            )


# ════════════════════════════════════════════════════════════════
# 2. 周度行情采集：逐日取数 + 节假日去重 + 周末跳过 + 复合周涨幅
# ════════════════════════════════════════════════════════════════


class TestWeeklyQuotesCollection:
    def test_holiday_duplicate_rows_deduped(self):
        """01-07（模拟节假日）返回与 01-06 相同的行 → 按签名去重只计一次。"""
        rows_by_date = {
            "20250106": [_sw_row("食品饮料", 1.0, 100.0, 50.0, 10.0), _sw_row("煤炭", 0.5, 80.0, 30.0, 5.0)],
            "20250107": [_sw_row("食品饮料", 1.0, 100.0, 50.0, 10.0)],  # 节假日对齐重复行
            "20250108": [_sw_row("食品饮料", 2.0, 102.0, 55.0, 11.0)],
            "20250109": [_sw_row("食品饮料", -1.0, 100.98, 48.0, 9.0)],
            "20250110": [_sw_row("食品饮料", 0.5, 101.48, 52.0, 10.0)],
        }
        fetch_mock = MagicMock(side_effect=lambda d: rows_by_date.get(d, []))

        with patch("agent.data_fetcher.fetch_shenwan_sectors", fetch_mock):
            daily = _collect_sector_weekly_quotes("食品饮料", WEEK_START, WEEK_END, WEEK_END)

        assert [d["date"] for d in daily] == [
            "2025-01-06", "2025-01-08", "2025-01-09", "2025-01-10",
        ], "节假日重复行未去重或日期归属错误"
        # 板块过滤正确：煤炭行不进入
        assert all(d["date"] != "" for d in daily)

        block = _format_sector_weekly_market_block(daily)
        # 复合周涨幅：1.01 * 1.02 * 0.99 * 1.005 - 1
        cum = 1.01 * 1.02 * 0.99 * 1.005
        expected_pct = f"{(cum - 1) * 100:+.2f}%"
        assert "当周累计涨跌幅（4个交易日复合）" in block
        assert expected_pct in block
        # 成交额合计：50+55+48+52 = 205 亿元
        assert "205.00" in block

    def test_weekend_days_never_fetched(self):
        """周区间跨周末时，周六/周日不发起取数。"""
        fetch_mock = MagicMock(return_value=[])
        sunday = date(2025, 1, 12)
        with patch("agent.data_fetcher.fetch_shenwan_sectors", fetch_mock):
            _collect_sector_weekly_quotes("食品饮料", WEEK_START, sunday, sunday)
        called_dates = [c.args[0] for c in fetch_mock.call_args_list]
        assert called_dates == [
            "20250106", "20250107", "20250108", "20250109", "20250110",
        ], f"周末不应取数，实际调用 {called_dates}"

    def test_empty_quotes_honest_degradation(self):
        """数据源全空 → 诚实写「数据未覆盖」，不编造行情。"""
        with patch("agent.data_fetcher.fetch_shenwan_sectors", MagicMock(return_value=[])):
            daily = _collect_sector_weekly_quotes("食品饮料", WEEK_START, WEEK_END, WEEK_END)
        assert daily == []
        assert "数据未覆盖" in _format_sector_weekly_market_block(daily)


# ════════════════════════════════════════════════════════════════
# 3. 当周新闻：周区间过滤 + 别名关键词匹配 + 跨源去重
# ════════════════════════════════════════════════════════════════


class TestWeeklyNewsCollection:
    def test_range_filter_keyword_match_and_dedup(self):
        in_range_t = "2025-01-08 10:00:00"
        pool = {
            "mcp": [
                {"title": "白酒春节备货启动渠道反馈积极", "time": in_range_t},
                {"title": "半导体设备招标放量超预期", "time": in_range_t},  # 非板块
                {"title": "白酒上上周的旧闻不应出现", "time": "2025-01-02 09:00:00"},  # 区间外
            ],
            "flash": [
                {"title": "白酒春节备货启动渠道反馈积极", "time": in_range_t},  # 跨源重复
                {"title": "消费龙头密集发布年度经营公告", "time": "2025-01-10 18:00:00"},  # 别名命中
            ],
        }
        with patch(
            "agent.data_fetcher.fetch_news_pool", MagicMock(return_value=pool)
        ) as pool_mock:
            items = _collect_sector_weekly_news("食品饮料", WEEK_START, WEEK_END)

        pool_mock.assert_called_once()
        kws = pool_mock.call_args.kwargs.get("sector_keywords") or pool_mock.call_args.args[0]
        assert "食品饮料" in kws and "白酒" in kws, "检索关键词应含申万名与别名"

        titles = [it["title"] for it in items]
        assert titles.count("白酒春节备货启动渠道反馈积极") == 1, "跨源重复标题未去重"
        assert "消费龙头密集发布年度经营公告" in titles, "板块别名（消费）命中的新闻丢失"
        assert "半导体设备招标放量超预期" not in titles, "非板块新闻未被过滤"
        assert "白酒上上周的旧闻不应出现" not in titles, "周区间外新闻未被过滤"

    def test_pool_failure_returns_empty(self):
        """新闻池异常 → 空列表，绝不抛出。"""
        with patch(
            "agent.data_fetcher.fetch_news_pool", MagicMock(side_effect=RuntimeError("boom"))
        ):
            assert _collect_sector_weekly_news("食品饮料", WEEK_START, WEEK_END) == []


# ════════════════════════════════════════════════════════════════
# 4. 券商研报分组聚合与诚实降级
# ════════════════════════════════════════════════════════════════


THREE_BROKER_REPORTS = {
    "total": 4,
    "reports": [
        {"org": "中信证券", "title": "白酒行业周报：批价企稳", "date": "2025-01-09",
         "rating": "买入", "rating_change": "维持", "target_price": "",
         "eps_forecast": None, "stock_name": ""},
        {"org": "华泰证券", "title": "食品饮料：春节备货调研", "date": "2025-01-08",
         "rating": "增持", "rating_change": "", "target_price": "",
         "eps_forecast": 2.5, "stock_name": ""},
        {"org": "国泰海通", "title": "白酒渠道跟踪周报", "date": "2025-01-07",
         "rating": "", "rating_change": "", "target_price": "1800元",
         "eps_forecast": None, "stock_name": "贵州茅台"},
        {"org": "中信证券", "title": "食品饮料年度策略", "date": "2025-01-06",
         "rating": "买入", "rating_change": "上调", "target_price": "",
         "eps_forecast": None, "stock_name": ""},
    ],
}


class TestWeeklyBrokerBlock:
    def test_three_brokers_grouped(self):
        """三家券商以上按券商分组，评级/评级变动/目标价/EPS 逐家列出。"""
        block = _format_weekly_broker_block(THREE_BROKER_REPORTS)
        assert "覆盖 3 家券商" in block
        assert "【中信证券】" in block and "【华泰证券】" in block and "【国泰海通】" in block
        # 中信证券两篇归在同一组
        assert block.count("【中信证券】") == 1
        assert "评级：买入（上调）" in block, "评级变动未标注"
        assert "评级：买入（维持）" in block
        assert "EPS预测：2.5" in block
        assert "目标价：1800元" in block
        assert "贵州茅台" in block

    def test_no_reports_honest_degradation(self):
        """无研报 → 诚实声明未覆盖，绝不编造券商观点。"""
        block = _format_weekly_broker_block({"total": 0, "reports": []})
        assert "券商研报数据未覆盖" in block

    def test_reports_without_org_grouped_as_unknown(self):
        """缺券商名的记录归入『未知券商』组，不丢弃、不崩溃。"""
        block = _format_weekly_broker_block({
            "total": 1,
            "reports": [{"org": "", "title": "某行业周报", "date": "2025-01-08",
                         "rating": "买入", "rating_change": "", "target_price": "",
                         "eps_forecast": None, "stock_name": ""}],
        })
        assert "【未知券商】" in block


# ════════════════════════════════════════════════════════════════
# 5. 周报主路径端到端接线（全 mock）
# ════════════════════════════════════════════════════════════════


class TestSectorWeeklyReviewPath:
    """_sector_weekly_review 数据组装与 LLM 接线。"""

    def _week_dates_last_week(self):
        today = datetime.now().date()
        monday_prev = today - timedelta(days=today.weekday()) - timedelta(days=7)
        return [monday_prev + timedelta(days=i) for i in range(5)]

    def _run_weekly(self, reports_result):
        week_dates = self._week_dates_last_week()
        pcts = [1.0, 2.0, -1.0, 0.5, 1.5]
        rows_by_date = {
            d.strftime("%Y%m%d"): [
                _sw_row("食品饮料", pct, 100.0 + i, 50.0 + i, 10.0),
                _sw_row("煤炭", 0.1, 80.0, 30.0, 5.0),
            ]
            for i, (d, pct) in enumerate(zip(week_dates, pcts))
        }
        pool = {
            "mcp": [
                {"title": "白酒春节备货启动渠道反馈积极",
                 "time": week_dates[2].strftime("%Y-%m-%d") + " 10:00:00"},
            ],
            "flash": [
                {"title": "消费龙头密集发布年度经营公告",
                 "time": week_dates[4].strftime("%Y-%m-%d") + " 18:00:00"},
            ],
        }
        agent = _make_agent()
        agent._call_llm = AsyncMock(return_value={"role": "assistant", "content": "周报正文"})
        search_mock = MagicMock(return_value=reports_result)
        pool_mock = MagicMock(return_value=pool)
        quotes_mock = MagicMock(side_effect=lambda d: rows_by_date.get(d, []))

        with patch("agent.data_fetcher.fetch_shenwan_sectors", quotes_mock), patch(
            "agent.data_fetcher.fetch_news_pool", pool_mock
        ), patch(
            "agent.report_library.search_reports", search_mock
        ), patch(
            "agent.data_fetcher.fetch_sector_valuation", MagicMock(return_value={})
        ), patch(
            "agent.data_fetcher.fetch_sector_moneyflow",
            MagicMock(return_value={"main_net": 3.2, "stock_count": 100})
        ), patch(
            "agent.data_fetcher.fetch_sector_earnings", MagicMock(return_value={})
        ), patch(
            "agent.orchestrator._get_latest_trade_date",
            MagicMock(return_value=datetime(2025, 1, 10)),
        ):
            result = asyncio.run(
                agent._sector_weekly_review(
                    "食品饮料", stream=False, message="上周白酒板块怎么样"
                )
            )
        return result, agent, search_mock, pool_mock, quotes_mock, week_dates, pcts

    def test_weekly_assembly_three_brokers(self):
        """三家券商以上分组聚合进入 prompt；行情/新闻/研报/资金四路齐全。"""
        result, agent, search_mock, pool_mock, quotes_mock, week_dates, pcts = \
            self._run_weekly(THREE_BROKER_REPORTS)

        assert result["role"] == "assistant"
        assert "周报正文" in result["content"]

        # 研报检索：industry=板块、days=14 回溯（覆盖当周+前一周）
        search_mock.assert_called_once()
        assert search_mock.call_args.kwargs.get("industry") == "食品饮料"
        assert search_mock.call_args.kwargs.get("days") == 14

        # 新闻池：板块关键词含申万名与别名
        kws = pool_mock.call_args.kwargs.get("sector_keywords")
        assert "食品饮料" in kws and "白酒" in kws

        # 行情：上周 5 个工作日逐日取数
        assert quotes_mock.call_count == 5

        user_prompt = agent._call_llm.await_args.args[1]
        # 头部统计区间（上周一~上周日）
        monday_prev = week_dates[0]
        sunday_prev = monday_prev + timedelta(days=6)
        assert monday_prev.strftime("%Y年%m月%d日") in user_prompt
        assert sunday_prev.strftime("%Y年%m月%d日") in user_prompt
        # 周度行情复合涨幅
        cum = 1.0
        for p in pcts:
            cum *= 1 + p / 100
        assert f"{(cum - 1) * 100:+.2f}%" in user_prompt
        # 当周新闻两源各一条进入
        assert "白酒春节备货启动渠道反馈积极" in user_prompt
        assert "消费龙头密集发布年度经营公告" in user_prompt
        # 资金流向块进入
        assert "主力资金" in user_prompt
        # 三家券商分组聚合
        assert "覆盖 3 家券商" in user_prompt
        assert "【中信证券】" in user_prompt
        assert "【华泰证券】" in user_prompt
        assert "【国泰海通】" in user_prompt
        # 五段结构指令
        assert "多家券商观点对比" in user_prompt
        assert "下周关注" in user_prompt

    def test_weekly_no_reports_honest_degradation(self):
        """研报库无覆盖 → prompt 如实写「券商研报数据未覆盖」，禁止编造。"""
        _, agent, search_mock, _, _, _, _ = self._run_weekly({"total": 0, "reports": []})
        user_prompt = agent._call_llm.await_args.args[1]
        assert "券商研报数据未覆盖" in user_prompt
        # 禁止编造的指令必须在场
        assert "禁止编造" in user_prompt or "严禁虚构" in user_prompt
