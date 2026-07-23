"""金融功能 QA 六根因定向修复测试。

背景：金融功能全维度 QA（89 题，12 类）暴露六大系统性根因，本文件锁定对应修复：
1. 海外市场路由：美股/港股信号词、前导零港股代码、美股 ticker → us_hk_query，
   不被 A 股复盘模板劫持、不被 fund_query『基金』碰撞。
2. 分析型个股路由：技术分析/估值/财报/公告/多实体 → agent_analyze，
   不被确定性行情卡短路；纯行情诉求（茅台走势）保持 stock_query 不变。
3. general_chat 金融兜底：北向/沪指/成交额/CPI 等漏路由词 → agent_analyze；
   闲聊（笑话/你好）零影响。
4. 风险提示统一兜底：stock_query 等模板路径必带；general_chat 按内容信号判定；
   campus_kb 永不追加；幂等判重。
5. 出口卫生：函数名软泄漏剥除、『修正后的完整正文』元话语剥除（全路径）。
6. _stock_query 模板：K线 date 缺失不输出裸冒号、取数全失败不输出空白模板。
7. 上下文继承：上文 stock_query + 指代追问（那它的财报怎么样）→ agent_analyze。

规则（与项目其他测试一致）：
- 所有外部调用全部 mock，绝不发起真实网络请求。
- 无 pytest-asyncio，异步函数一律用 asyncio.run 驱动。
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import agent.orchestrator as orchestrator
from agent.orchestrator import (
    MarketReviewAgent,
    _ensure_disclaimer,
    _finance_fallback_hit,
    _has_stock_entity,
    _resolve_contextual_intent,
    _strip_function_names,
    _strip_meta_openings,
    detect_intent,
)


def _make_agent() -> MarketReviewAgent:
    agent = MarketReviewAgent()
    agent.client = MagicMock()
    return agent


# ════════════════════════════════════════════════════════════════
# 1. 海外市场路由
# ════════════════════════════════════════════════════════════════

class TestOverseasRouting:

    def test_nasdaq_not_hijacked_by_astock_review(self):
        intent, _ = detect_intent("纳斯达克指数怎么样")
        assert intent == "us_hk_query"

    def test_hk_leading_zero_code(self):
        assert detect_intent("00700现在行情怎么样")[0] == "us_hk_query"
        assert detect_intent("06715这只股票如何")[0] == "us_hk_query"

    def test_us_ticker(self):
        assert detect_intent("AAPL走势如何")[0] == "us_hk_query"

    def test_fed_not_fund_query_collision(self):
        """『联邦基金利率』含『基金』，不得落入 fund_query。"""
        intent, _ = detect_intent("美联储联邦基金利率最新是多少")
        assert intent == "us_hk_query"

    def test_hk_company_names(self):
        assert detect_intent("腾讯控股最近怎么样")[0] == "us_hk_query"
        assert detect_intent("港股通资金流向")[0] == "us_hk_query"
        assert detect_intent("恒生指数最近怎么样")[0] == "us_hk_query"


# ════════════════════════════════════════════════════════════════
# 2. 分析型个股路由
# ════════════════════════════════════════════════════════════════

class TestAnalyticStockRouting:

    def test_technical_analysis_goes_agent(self):
        assert detect_intent("贵州茅台技术分析")[0] == "agent_analyze"
        assert detect_intent("宁德时代MACD怎么看")[0] == "agent_analyze"
        assert detect_intent("中芯国际支撑位压力位")[0] == "agent_analyze"

    def test_valuation_goes_agent(self):
        assert detect_intent("比亚迪估值怎么样")[0] == "agent_analyze"
        assert detect_intent("比亚迪现在贵不贵")[0] == "agent_analyze"
        assert detect_intent("600519市盈率多少")[0] == "agent_analyze"

    def test_fundamentals_news_goes_agent(self):
        assert detect_intent("600519财务数据")[0] == "agent_analyze"
        assert detect_intent("平安银行最新公告新闻")[0] == "agent_analyze"
        assert detect_intent("介绍一下鲟龙科技公司")[0] == "agent_analyze"

    def test_multi_entity_goes_agent(self):
        assert detect_intent("茅台和五粮液估值对比")[0] == "agent_analyze"

    def test_plain_quote_stays_stock_query(self):
        """纯行情诉求保持确定性行情卡不变。"""
        assert detect_intent("茅台走势")[0] == "stock_query"

    def test_company_keyword_entity(self):
        assert _has_stock_entity("介绍一下鲟龙科技公司的相关情况")
        assert not _has_stock_entity("今天天气不错")


# ════════════════════════════════════════════════════════════════
# 3. general_chat 金融兜底探针
# ════════════════════════════════════════════════════════════════

class TestFinanceFallbackProbe:

    def test_finance_signals_hit(self):
        assert _finance_fallback_hit("最近北向资金动向如何")
        assert _finance_fallback_hit("最近沪指走势怎么样")
        assert _finance_fallback_hit("今天两市成交额有多少")
        assert _finance_fallback_hit("最近一期CPI数据是多少")
        assert _finance_fallback_hit("LPR最近有调整吗")
        assert _finance_fallback_hit("最近有什么财经大事")
        assert _finance_fallback_hit("最近有哪些政策利好")

    def test_chitchat_untouched(self):
        assert not _finance_fallback_hit("你好")
        assert not _finance_fallback_hit("你是谁")
        assert not _finance_fallback_hit("你能做什么")
        assert not _finance_fallback_hit("讲个笑话")
        assert not _finance_fallback_hit("给我讲个股市笑话")  # 豁免词优先

    def test_short_message_ignored(self):
        assert not _finance_fallback_hit("股")


# ════════════════════════════════════════════════════════════════
# 4. 风险提示统一兜底
# ════════════════════════════════════════════════════════════════

class TestEnsureDisclaimer:

    def test_template_paths_get_disclaimer(self):
        r = {"role": "assistant", "content": "贵州茅台行情：价格1305.00"}
        out = _ensure_disclaimer(r, "stock_query")
        assert "风险提示" in out["content"]
        # 原 dict 不被原地修改
        assert "风险提示" not in r["content"]

    def test_general_chat_finance_content_gets_disclaimer(self):
        r = {"role": "assistant", "content": "最近大盘上涨，成交活跃。"}
        out = _ensure_disclaimer(r, "general_chat")
        assert "风险提示" in out["content"]

    def test_general_chat_chitchat_untouched(self):
        r = {"role": "assistant", "content": "你好！有什么可以帮你的？"}
        out = _ensure_disclaimer(r, "general_chat")
        assert "风险提示" not in out["content"]

    def test_campus_never_gets_disclaimer(self):
        r = {"role": "assistant", "content": "清华宿舍分紫荆和南区。"}
        out = _ensure_disclaimer(r, "campus_kb")
        assert "风险提示" not in out["content"]

    def test_idempotent(self):
        r = {"role": "assistant", "content": "复盘内容。\n\n风险提示：已有"}
        out = _ensure_disclaimer(r, "market_review")
        assert out["content"].count("风险提示") == 1

    def test_generator_passthrough(self):
        gen = (x for x in ["a", "b"])
        assert _ensure_disclaimer(gen, "stock_query") is gen


# ════════════════════════════════════════════════════════════════
# 5. 出口卫生：函数名 / 元话语剥除
# ════════════════════════════════════════════════════════════════

class TestOutputHygiene:

    def test_function_names_stripped(self):
        text = "我通过 get_market_sentiment 获取了情绪数据，再通过 get_us_macro 查了宏观。"
        out = _strip_function_names(text)
        assert "get_market_sentiment" not in out
        assert "get_us_macro" not in out
        assert "情绪数据" in out  # 句子其余部分保留

    def test_normal_text_untouched(self):
        text = "正常的中文回答，没有标记。"
        assert _strip_function_names(text) == text

    def test_revision_meta_stripped(self):
        text = "以下为修正后的完整正文：\n茅台今日收盘价1305元。"
        out = _strip_meta_openings(text)
        assert "修正后" not in out
        assert "茅台今日收盘价1305元" in out

    def test_agent_query_strips_function_names(self):
        """_agent_query 出口（全路径）剥除函数名泄漏。"""
        agent = _make_agent()
        agent.client.chat.completions.create = AsyncMock(side_effect=[
            SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content="我用 get_stock_kline 查了K线，走势平稳。", tool_calls=None),
                finish_reason="stop",
            )]),
        ])
        result = asyncio.run(agent._agent_query("茅台怎么样", stream=False))
        assert "get_stock_kline" not in result["content"]
        assert "走势平稳" in result["content"]


# ════════════════════════════════════════════════════════════════
# 6. _stock_query 模板修复
# ════════════════════════════════════════════════════════════════

class TestStockQueryTemplate:

    def _run_stock_query(self, quote, kline, news):
        agent = _make_agent()
        with patch("agent.data_fetcher.fetch_stock_quote", return_value=quote), \
             patch("agent.data_fetcher.fetch_stock_kline", return_value=kline), \
             patch("agent.data_fetcher.fetch_stock_news", return_value=news):
            return asyncio.run(agent._stock_query("茅台走势", False))

    def test_missing_date_no_bare_colon(self):
        """date 缺失时不输出 ':1305.000' 裸冒号。"""
        result = self._run_stock_query(
            {"price": 1305.0, "pct": 0.5, "open": 1300, "high": 1310, "low": 1295},
            [{"close": 1305.0}, {"trade_date": "2026-07-22", "close": 1308.0}],
            [],
        )
        content = result["content"]
        assert ":1305" not in content.replace("收盘1305", "")  # 无裸冒号
        assert "07-22:1308" in content  # trade_date 兜底生效

    def test_all_empty_no_blank_template(self):
        """取数全失败：如实声明，不输出『价格? 涨跌?%』空白模板。"""
        result = self._run_stock_query({}, [], [])
        assert "暂不可用" in result["content"]
        assert "价格?" not in result["content"]

    def test_empty_kline_shows_uncovered(self):
        result = self._run_stock_query(
            {"price": 1305.0, "pct": 0.5, "open": 1, "high": 2, "low": 1}, [], [],
        )
        assert "数据未覆盖" in result["content"]


# ════════════════════════════════════════════════════════════════
# 7. 上下文继承：个股追问
# ════════════════════════════════════════════════════════════════

class TestContextualInheritance:

    def test_stock_followup_goes_agent(self):
        history = [
            {"role": "user", "content": "茅台走势"},
            {"role": "assistant", "content": "贵州茅台行情…"},
        ]
        intent, _, label = _resolve_contextual_intent("那它的财报怎么样", history)
        assert intent == "agent_analyze"
        assert label == "个股查询"

    def test_market_followup_unchanged(self):
        history = [
            {"role": "user", "content": "今天A股复盘"},
            {"role": "assistant", "content": "复盘…"},
        ]
        intent, _, _ = _resolve_contextual_intent("资金呢", history)
        assert intent == "market_review"

    def test_no_history_passthrough(self):
        intent, _, label = _resolve_contextual_intent("那它的财报怎么样", [])
        assert intent != "agent_analyze"
        assert label is None


# ════════════════════════════════════════════════════════════════
# 8. 路由回归：既有意图零影响
# ════════════════════════════════════════════════════════════════

class TestRoutingRegression:

    def test_existing_intents_unchanged(self):
        assert detect_intent("今天A股复盘")[0] == "market_review"
        assert detect_intent("半导体板块怎么样")[0] == "sector_deep_dive"
        assert detect_intent("现在市场情绪如何")[0] == "social_sentiment"
        assert detect_intent("你好")[0] == "general_chat"
        assert detect_intent("清华大学宿舍怎么样")[0] == "campus_kb"
        assert detect_intent("黄金期货价格")[0] == "futures_query"

    def test_process_message_finance_fallback_integration(self):
        """general_chat 金融问题经兜底改道 _agent_query（不走无工具闲聊）。"""
        agent = _make_agent()
        called = {}

        async def fake_agent_query(msg, stream, history=None, hint=None, disclaimer=True):
            called["hit"] = True
            return {"role": "assistant", "content": "北向资金数据整理。"}

        agent._agent_query = fake_agent_query
        # 阻止校园探针干扰（北向问题在校园库零命中，但 mock 掉更稳）
        with patch.object(orchestrator, "_campus_fallback_hit", return_value=False):
            result = asyncio.run(agent.process_message("最近北向资金动向如何"))
        assert called.get("hit"), "金融兜底应改道 _agent_query"


# ════════════════════════════════════════════════════════════════
# 9. 复测二轮补丁：校园探针让位 / 宏观护栏 / 内部字段 / 分隔线
# ════════════════════════════════════════════════════════════════

class TestRetestRound2Patches:

    def test_macro_keywords_not_stock_query(self):
        """GDP/CPI 等大写缩写不得撞 stock_patterns 的 [A-Z] 模式。"""
        assert detect_intent("GDP增速怎么样")[0] == "agent_analyze"
        assert detect_intent("最新CPI数据")[0] == "agent_analyze"
        assert detect_intent("LPR最近调整了吗")[0] == "agent_analyze"

    def test_campus_probe_yields_to_finance(self):
        """金融消息在校园库命中书籍笔记时不得误判 campus_kb。"""
        agent = _make_agent()
        called = {}

        async def fake_agent_query(msg, stream, history=None, hint=None, disclaimer=True):
            called["disclaimer"] = disclaimer
            return {"role": "assistant", "content": "整理。"}

        agent._agent_query = fake_agent_query
        # 校园探针 mock 为命中（模拟书籍笔记误命中场景），金融让位后不应生效
        with patch.object(orchestrator, "_campus_fallback_hit", return_value=True):
            asyncio.run(agent.process_message("推荐几只股票"))
        assert called.get("disclaimer") is True, "金融问题不得走 campus_kb（disclaimer=False）"

    def test_internal_fields_stripped(self):
        text = "结论如下。signal: bullish, confidence: 0.8, disclaimer: 仅供参考"
        out = _strip_function_names(text)
        assert "signal" not in out and "confidence" not in out and "disclaimer" not in out
        assert "结论如下。" in out

    def test_horizontal_rule_stripped(self):
        out = orchestrator._clean_markdown("第一段。\n---\n第二段。")
        assert "---" not in out
        assert "第一段。" in out and "第二段。" in out

    def test_meta_opening_extended(self):
        text = "现在数据已经够了，可以给出结论。\n茅台今日上涨。"
        out = _strip_meta_openings(text)
        assert "已经够了" not in out
        assert "茅台今日上涨" in out
