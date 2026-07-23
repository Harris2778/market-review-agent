"""港美股数据源接线定向测试（残留4 修复）。

背景：QA 发现港美股问题普遍取不到数，实锤根因：
1. 工具 schema 教 LLM 港股代码用 hk00700 格式，智研接口只认裸 00700 → 行情信息不存在；
2. 智研港美股 K 线日期键是 day 而非 date → 日期取空、K线裸冒号；
3. 注册表缺港股财报/资金流/美股资金流/美股涨跌分布/重大事项工具；
4. yfinance 未安装（全球指数兜底路径）、FRED/Tushare 报障实为代理瞬时不稳定。

覆盖：
- _normalize_market_symbol：hk/us 前缀剥离、USD 护栏、A股不动；
- fetch_stock_kline：day 键兜底、change_pct/pct 多键兜底；
- 4 个新 fetch 包装：MCP 工具名与参数映射（含 market 0/1/2 映射）；
- execute_tool 分发：5 个新工具 ok 路径与参数适配；
- TOOL_REGISTRY：新工具注册与必填参数声明。

规则（与项目其他测试一致）：全部 mock，绝不发起真实网络请求。
"""

from unittest.mock import patch

import agent.data_fetcher as data_fetcher
import agent.tools as tools_mod


# ════════════════════════════════════════════════════════════════
# 1. 代码归一
# ════════════════════════════════════════════════════════════════

class TestNormalizeMarketSymbol:

    def test_hk_prefix_stripped(self):
        assert data_fetcher._normalize_market_symbol("hk", "hk00700") == "00700"
        assert data_fetcher._normalize_market_symbol("hk", "HK00700") == "00700"

    def test_us_prefix_stripped(self):
        assert data_fetcher._normalize_market_symbol("us", "usAAPL") == "AAPL"
        assert data_fetcher._normalize_market_symbol("us", "USAAPL") == "AAPL"

    def test_bare_symbols_unchanged(self):
        assert data_fetcher._normalize_market_symbol("hk", "00700") == "00700"
        assert data_fetcher._normalize_market_symbol("us", "AAPL") == "AAPL"
        assert data_fetcher._normalize_market_symbol("us", "aapl") == "aapl"

    def test_cn_untouched(self):
        assert data_fetcher._normalize_market_symbol("cn", "sh600519") == "sh600519"

    def test_usd_guard(self):
        """USD 开头（汇率对场景）不被误剥。"""
        assert data_fetcher._normalize_market_symbol("us", "USDCNY") == "USDCNY"


# ════════════════════════════════════════════════════════════════
# 2. K线 day 键兜底
# ════════════════════════════════════════════════════════════════

class TestKlineDayKeyFallback:

    def test_day_key_mapped(self):
        rows = [{"day": "2026-07-16", "close": "484.0", "change_pct": "1.2"}]
        with patch.object(data_fetcher, "_mcp_call", return_value={"data": rows}) as m:
            out = data_fetcher.fetch_stock_kline("hk", "hk00700", 5)
        assert out == [{"date": "2026-07-16", "close": "484.0", "pct": "1.2"}]
        # hk 前缀已剥离后再调 MCP
        assert m.call_args[0][1]["symbol"] == "00700"

    def test_date_key_still_works(self):
        rows = [{"date": "2026-07-22", "close": "1305.0", "change_pct": "0.5"}]
        with patch.object(data_fetcher, "_mcp_call", return_value={"data": rows}):
            out = data_fetcher.fetch_stock_kline("cn", "sh600519", 5)
        assert out[0]["date"] == "2026-07-22"

    def test_pct_fallback_keys(self):
        rows = [{"day": "2026-07-16", "close": "1.0", "pct": "2.0"}]
        with patch.object(data_fetcher, "_mcp_call", return_value={"data": rows}):
            out = data_fetcher.fetch_stock_kline("hk", "00700", 5)
        assert out[0]["pct"] == "2.0"


# ════════════════════════════════════════════════════════════════
# 3. 新 fetch 包装：MCP 工具名与参数映射
# ════════════════════════════════════════════════════════════════

class TestNewFetchWrappers:

    def test_hk_finance_report_call(self):
        with patch.object(data_fetcher, "_mcp_call",
                          return_value={"result": {"data": [{"year": "2025"}]}}) as m:
            out = data_fetcher.fetch_hk_finance_report("hk00700", "净利润", 1)
        name, args = m.call_args[0]
        assert name == "hk_finance_all"
        assert args == {"symbol": "00700", "frType": "净利润", "yearNum": "1"}
        assert out["data"] == [{"year": "2025"}]

    def test_hk_fund_flow_call(self):
        with patch.object(data_fetcher, "_mcp_call", return_value={"data": {"x": 1}}) as m:
            data_fetcher.fetch_hk_fund_flow("00700", 10)
        name, args = m.call_args[0]
        assert name == "hkTradingMainFundsHistory"
        assert args == {"symbol": "00700", "days": "10"}

    def test_us_market_breadth_call(self):
        with patch.object(data_fetcher, "_mcp_call", return_value={"data": {"up": 100}}) as m:
            out = data_fetcher.fetch_us_market_breadth()
        assert m.call_args[0][0] == "usMarketStatisticsUpdown"
        assert out == {"up": 100}

    def test_major_events_market_mapping(self):
        with patch.object(data_fetcher, "_mcp_call", return_value={"data": []}) as m:
            data_fetcher.fetch_stock_major_events("hk", "hk00700", 5)
        name, args = m.call_args[0]
        assert name == "globalStockMajorEvents"
        assert args == {"market": "1", "symbols": "00700", "pageSize": "5"}

    def test_us_fund_flow_prefix_stripped(self):
        with patch.object(data_fetcher, "_mcp_call", return_value={"data": {}}) as m:
            data_fetcher.fetch_us_fund_flow("usAAPL")
        assert m.call_args[0][1]["symbol"] == "aapl"


# ════════════════════════════════════════════════════════════════
# 4. execute_tool 分发：新工具 ok 路径
# ════════════════════════════════════════════════════════════════

class TestNewToolDispatch:

    def test_registry_contains_new_tools(self):
        names = {t["function"]["name"] for t in tools_mod.TOOL_REGISTRY}
        for n in ("get_hk_finance_report", "get_hk_fund_flow", "get_us_fund_flow",
                  "get_us_market_breadth", "get_stock_major_events"):
            assert n in names, f"{n} 未注册"

    def test_dispatch_hk_finance(self):
        with patch.object(data_fetcher, "fetch_hk_finance_report",
                          return_value={"data": []}) as f:
            r = tools_mod.execute_tool(
                "get_hk_finance_report", {"symbol": "00700", "indicator": "营业收入"})
        assert r["ok"] is True
        f.assert_called_once_with(symbol="00700", indicator="营业收入", years=1)

    def test_dispatch_us_breadth(self):
        with patch.object(data_fetcher, "fetch_us_market_breadth", return_value={"up": 1}):
            r = tools_mod.execute_tool("get_us_market_breadth", {})
        assert r["ok"] is True and r["data"] == {"up": 1}

    def test_dispatch_major_events(self):
        with patch.object(data_fetcher, "fetch_stock_major_events", return_value=[]) as f:
            r = tools_mod.execute_tool(
                "get_stock_major_events", {"market": "hk", "symbol": "00700"})
        assert r["ok"] is True
        f.assert_called_once_with(market="hk", symbol="00700", limit=10)

    def test_dispatch_missing_required(self):
        r = tools_mod.execute_tool("get_hk_finance_report", {})
        assert r["ok"] is False and "symbol" in r["error"]
