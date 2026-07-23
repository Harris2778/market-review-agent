"""17 个智研 MCP 新数据工具 + 新闻模糊去重/聚合池的单元测试。

覆盖对象：
- agent/data_fetcher.py 新增 17 个智研包装函数
  （fetch_company_profile / fetch_company_managers / fetch_shareholder_count /
  fetch_financial_report_full / fetch_stock_valuation / fetch_lockup_schedule /
  fetch_margin_detail / fetch_block_trades / fetch_connect_holdings /
  fetch_fund_info / fetch_fund_networth / fetch_fund_holdings / fetch_fund_dividend /
  fetch_forex_quote / fetch_commodity_futures_list / fetch_hk_special_ranking /
  fetch_us_fund_flow_history）
- 模糊去重 _fuzzy_title_key / _dedup_news_fuzzy
- 新闻聚合池 fetch_news_pool（只含 mcp/flash 两键、跨源模糊去重）
- agent/tools.py execute_tool 对 17 个新工具的路由、TOOL_REGISTRY 数量与目录

规则：全部用 unittest.mock.patch 掉 agent.data_fetcher._mcp_call（或
fetch_mcp_news / fetch_mcp_flash），零网络、零外部依赖。

已确认的真实签名（grep 自 agent/data_fetcher.py）：
- fetch_shareholder_count(code: str, type: str = "amount")
  第一参数名为 code（不是 symbol），Code 参数不带 sh/sz 前缀。
"""

from unittest.mock import patch

import pytest

from agent import data_fetcher
from agent.data_fetcher import (
    _dedup_news_fuzzy,
    fetch_block_trades,
    fetch_commodity_futures_list,
    fetch_company_managers,
    fetch_company_profile,
    fetch_connect_holdings,
    fetch_financial_report_full,
    fetch_forex_quote,
    fetch_fund_dividend,
    fetch_fund_holdings,
    fetch_fund_info,
    fetch_fund_networth,
    fetch_hk_special_ranking,
    fetch_lockup_schedule,
    fetch_margin_detail,
    fetch_news_pool,
    fetch_shareholder_count,
    fetch_stock_valuation,
    fetch_us_fund_flow_history,
)

MCP = "agent.data_fetcher._mcp_call"


def _empty_result(payload):
    """把 result.data 包装成 _mcp_call 的标准返回结构。"""
    return {"result": {"data": payload}}


# ════════════════════════════════════════════════════════════════
# ① 17 个包装逐个断言 MCP 工具名与参数映射
# ════════════════════════════════════════════════════════════════

class TestMcpToolNameAndParams:
    def test_fetch_company_profile(self):
        with patch(MCP, return_value=_empty_result({})) as m:
            fetch_company_profile("sh600519")
        m.assert_called_once_with("cnCompanyBasicInfo", {"symbol": "sh600519"})

    def test_fetch_company_managers(self):
        with patch(MCP, return_value=_empty_result({})) as m:
            fetch_company_managers("sz000002")
        m.assert_called_once_with("cnCompanyManagerInfo", {"symbol": "sz000002"})

    def test_fetch_shareholder_count_strips_prefix(self):
        """调 cnCompanyShareholderHistory，Code 不带 sh/sz 前缀。"""
        with patch(MCP, return_value=_empty_result({})) as m:
            fetch_shareholder_count("sh600519")
        m.assert_called_once_with(
            "cnCompanyShareholderHistory", {"Code": "600519", "Type": "amount"})

    def test_fetch_shareholder_count_plain_code_and_type(self):
        with patch(MCP, return_value=_empty_result({})) as m:
            fetch_shareholder_count("600519", type="average")
        m.assert_called_once_with(
            "cnCompanyShareholderHistory", {"Code": "600519", "Type": "average"})

    def test_fetch_financial_report_full_empty_rdate_two_calls(self):
        """r_date='' 时先调 cnFinanceReportDateList（参数名 paperCode）
        取 result.data.dList[0].date_value，再调 cnFinanceReportsFull。"""
        date_resp = {"result": {"data": {"dList": [{"date_value": "2024-12-31"}]}}}
        full_resp = {"result": {"data": {"report_list": {}}}}
        with patch(MCP, side_effect=[date_resp, full_resp]) as m:
            fetch_financial_report_full("sz000002", r_date="")
        assert m.call_count == 2
        m.assert_any_call("cnFinanceReportDateList", {"paperCode": "sz000002"})
        m.assert_any_call("cnFinanceReportsFull", {
            "paperCode": "sz000002", "rDate": "2024-12-31", "source": "gjzb"})

    def test_fetch_financial_report_full_given_rdate_single_call(self):
        with patch(MCP, return_value={"result": {"data": {"report_list": {}}}}) as m:
            fetch_financial_report_full("sz000002", source="lrb", r_date="2024-09-30")
        m.assert_called_once_with("cnFinanceReportsFull", {
            "paperCode": "sz000002", "rDate": "2024-09-30", "source": "lrb"})

    def test_fetch_stock_valuation(self):
        with patch(MCP, return_value=_empty_result({"dp": []})) as m:
            fetch_stock_valuation("sh600519", vtype="sjl", rank="y3")
        m.assert_called_once_with("cnStockValuationDetail", {
            "symbol": "sh600519", "type": "sjl", "rank": "y3"})

    def test_fetch_stock_valuation_crop_structure(self):
        """返回 latest / points[:20] / total 的裁剪结构（gg=个股口径）。"""
        gg = [{"day": f"2024-01-{i:02d}", "val": i} for i in range(1, 31)]
        dp = [{"day": "2024-01-01", "val": 999}]  # 大盘基准：不得混入个股字段
        hy = [{"day": "2024-01-01", "val": 888}]
        with patch(MCP, return_value=_empty_result({"gg": gg, "dp": dp, "hy": hy})):
            out = fetch_stock_valuation("sh600519")
        assert out["latest"] == gg[0]
        assert out["points"] == gg[:20]
        assert len(out["points"]) == 20
        assert out["total"] == 30
        assert out["benchmark_market"] == dp[0]
        assert out["benchmark_industry"] == hy[0]
        assert out["type"] == "syl" and out["rank"] == "y1"

    def test_fetch_stock_valuation_dp_is_benchmark_not_stock(self):
        """QA 实锤防回归：dp 是大盘基准序列（所有股票同值），latest 必须取 gg。
        旧版误用 dp 导致茅台/五粮液 PE 都答成 17.88。"""
        payload = {"gg": [{"day": "2026-07-23", "val": "19.53"}],
                   "dp": [{"day": "2026-07-23", "val": "17.88"}]}
        with patch(MCP, return_value=_empty_result(payload)):
            out = fetch_stock_valuation("sh600519")
        assert out["latest"]["val"] == "19.53"
        assert out["benchmark_market"]["val"] == "17.88"
        # gg 缺失时 latest=None，绝不回退用 dp 冒充个股
        with patch(MCP, return_value=_empty_result({"gg": [], "dp": payload["dp"]})):
            out2 = fetch_stock_valuation("sh600519")
        assert out2["latest"] is None and out2["total"] == 0

    def test_fetch_lockup_schedule(self):
        with patch(MCP, return_value=_empty_result({})) as m:
            fetch_lockup_schedule("sh600519")
        m.assert_called_once_with("cnStockLockupFuture", {
            "symbol": "sh600519", "num": "10", "page": "1"})

    def test_fetch_margin_detail(self):
        with patch(MCP, return_value=_empty_result([])) as m:
            fetch_margin_detail("sh600519")
        m.assert_called_once_with("cnStockTradingMarginList", {
            "symbol": "sh600519", "num": "10", "page": "1"})

    def test_fetch_block_trades_pagination_param_is_p(self):
        """调 cnTradingBlockList，分页参数名是 p 不是 page。"""
        with patch(MCP, return_value=_empty_result({"data": []})) as m:
            fetch_block_trades("sh600519")
        m.assert_called_once_with("cnTradingBlockList", {
            "symbol": "sh600519", "num": "10", "p": "1"})
        positional = m.call_args[0]
        assert "page" not in positional[1]

    def test_fetch_connect_holdings_five_params(self):
        with patch(MCP, return_value=_empty_result({"s_list": []})) as m:
            fetch_connect_holdings(type="hk", sort="hold_num", num=50)
        m.assert_called_once_with("cnStockConnectHoldings", {
            "type": "hk", "sort": "hold_num", "asc": "0", "num": "50", "page": "1"})

    def test_fetch_fund_info(self):
        with patch(MCP, return_value=_empty_result({})) as m:
            fetch_fund_info("110022")
        m.assert_called_once_with("fund_info", {"symbol": "110022"})

    def test_fetch_fund_networth(self):
        with patch(MCP, return_value=_empty_result([])) as m:
            fetch_fund_networth("110022", limit=5)
        m.assert_called_once_with("fund_networth", {"symbol": "110022", "num": "5"})

    def test_fetch_fund_holdings(self):
        with patch(MCP, return_value=_empty_result({"data": []})) as m:
            fetch_fund_holdings("110022")
        m.assert_called_once_with("fund_heavy_stock", {"symbol": "110022", "num": "10"})

    def test_fetch_fund_dividend(self):
        with patch(MCP, return_value=_empty_result({"cf": [], "fh": []})) as m:
            fetch_fund_dividend("110022")
        m.assert_called_once_with("fund_dividend", {"symbol": "110022"})

    def test_fetch_forex_quote_top_level_data(self):
        """该接口返回在顶层 data 键（无 result 包装），应照样取值。"""
        quote = {"symbol": "USDCNY", "price": "7.2456"}
        with patch(MCP, return_value={"data": quote}) as m:
            out = fetch_forex_quote("USDCNY")
        m.assert_called_once_with("forexQuoteLatest", {"symbol": "USDCNY"})
        assert out == quote

    def test_fetch_forex_quote_uppercases_symbol(self):
        with patch(MCP, return_value={"data": {}}) as m:
            fetch_forex_quote("usdcny")
        m.assert_called_once_with("forexQuoteLatest", {"symbol": "USDCNY"})

    def test_fetch_commodity_futures_list(self):
        with patch(MCP, return_value=_empty_result({"data": []})) as m:
            fetch_commodity_futures_list("shfe")
        m.assert_called_once_with("futureCommodityList", {"type": "shfe"})

    def test_fetch_hk_special_ranking_five_params(self):
        """node/sort/asc/num/page 五参数全传。"""
        with patch(MCP, return_value=_empty_result({"data": []})) as m:
            fetch_hk_special_ranking(node="gqg_hk", sort="changepercent", num=20)
        m.assert_called_once_with("hkStockSpecialRanking", {
            "node": "gqg_hk", "sort": "changepercent", "asc": "0",
            "num": "20", "page": "1"})

    def test_fetch_us_fund_flow_history_leading_space_key(self):
        """调 usTradingFundFlow60Days，参数键是 ' symbol' 带前导空格。"""
        with patch(MCP, return_value=_empty_result({"history": []})) as m:
            fetch_us_fund_flow_history("AAPL", days=20)
        m.assert_called_once_with("usTradingFundFlow60Days", {
            " symbol": "AAPL", "days": "20"})


# ════════════════════════════════════════════════════════════════
# ② _mcp_call 返回 {} 或抛异常：每个包装不抛出、返回空结构
# ════════════════════════════════════════════════════════════════

# (函数, 位置参数, 空结构期望值)
_WRAPPERS_EMPTY_CASES = [
    (fetch_company_profile, ("sh600519",), {}),
    (fetch_company_managers, ("sh600519",),
     {"companyinfo": {}, "incumbent": [], "change": []}),
    (fetch_shareholder_count, ("600519",), []),
    (fetch_financial_report_full, ("sh600519",), {}),
    (fetch_stock_valuation, ("sh600519",),
     {"type": "syl", "rank": "y1", "latest": None, "points": [], "total": 0,
      "benchmark_market": None, "benchmark_industry": None}),
    (fetch_lockup_schedule, ("sh600519",), {"data": [], "rowCount": 0}),
    (fetch_margin_detail, ("sh600519",), []),
    (fetch_block_trades, ("sh600519",), []),
    (fetch_connect_holdings, (), []),
    (fetch_fund_info, ("110022",), {}),
    (fetch_fund_networth, ("110022",), []),
    (fetch_fund_holdings, ("110022",), []),
    (fetch_fund_dividend, ("110022",), {"cf": [], "fh": []}),
    (fetch_forex_quote, ("USDCNY",), {}),
    (fetch_commodity_futures_list, ("dce",), []),
    (fetch_hk_special_ranking, (), []),
    (fetch_us_fund_flow_history, ("AAPL",), []),
]


# 异常路径与空响应路径空结构不同的包装（fetch_company_managers 异常时返回 {}）
_WRAPPERS_EXCEPTION_EXPECTED = {
    "fetch_company_managers": {},
}


@pytest.mark.parametrize("fn,args,expected", _WRAPPERS_EMPTY_CASES,
                         ids=[f[0].__name__ for f in _WRAPPERS_EMPTY_CASES])
def test_wrapper_returns_empty_on_empty_response(fn, args, expected):
    with patch(MCP, return_value={}):
        assert fn(*args) == expected


@pytest.mark.parametrize("fn,args,expected", _WRAPPERS_EMPTY_CASES,
                         ids=[f[0].__name__ for f in _WRAPPERS_EMPTY_CASES])
def test_wrapper_returns_empty_on_exception(fn, args, expected):
    expected = _WRAPPERS_EXCEPTION_EXPECTED.get(fn.__name__, expected)
    with patch(MCP, side_effect=RuntimeError("boom")):
        assert fn(*args) == expected


# ════════════════════════════════════════════════════════════════
# ③ fetch_shareholder_count：数字字符串键 dict 转 list 并按 EndDate 降序
# ════════════════════════════════════════════════════════════════

class TestShareholderCountTransform:
    def test_numeric_key_dict_to_sorted_list(self):
        payload = {
            "0": {"ANum": "50000", "Num": "100000", "EndDate": "2023-12-31", "close": "1700"},
            "1": {"ANum": "52000", "Num": "98000", "EndDate": "2024-06-30", "close": "1600"},
            "2": {"ANum": "48000", "Num": "105000", "EndDate": "2024-03-31", "close": "1650"},
        }
        with patch(MCP, return_value=_empty_result(payload)):
            out = fetch_shareholder_count("600519")
        assert isinstance(out, list)
        assert len(out) == 3
        assert [r["EndDate"] for r in out] == ["2024-06-30", "2024-03-31", "2023-12-31"]

    def test_caps_at_eight_periods(self):
        payload = {str(i): {"EndDate": f"202{i}-12-31"} for i in range(10)}
        with patch(MCP, return_value=_empty_result(payload)):
            out = fetch_shareholder_count("600519")
        assert len(out) == 8
        assert out[0]["EndDate"] == "2029-12-31"


# ════════════════════════════════════════════════════════════════
# ④ execute_tool 对 17 个新工具可路由；缺必填参数时 ok=False
# ════════════════════════════════════════════════════════════════

# (工具名, 合法参数, 必填参数名或 None)
_TOOL_ROUTE_CASES = [
    ("get_company_profile", {"symbol": "sh600519"}, "symbol"),
    ("get_company_managers", {"symbol": "sh600519"}, "symbol"),
    ("get_shareholder_count", {"code": "600519"}, "code"),
    ("get_financial_report_full", {"paper_code": "sh600519"}, "paper_code"),
    ("get_stock_valuation", {"symbol": "sh600519"}, "symbol"),
    ("get_lockup_schedule", {"symbol": "sh600519"}, "symbol"),
    ("get_margin_detail", {"symbol": "sh600519"}, "symbol"),
    ("get_block_trades", {"symbol": "sh600519"}, "symbol"),
    ("get_connect_holdings", {"type": "sh"}, None),
    ("get_fund_info", {"symbol": "110022"}, "symbol"),
    ("get_fund_networth", {"symbol": "110022"}, "symbol"),
    ("get_fund_holdings", {"symbol": "110022"}, "symbol"),
    ("get_fund_dividend", {"symbol": "110022"}, "symbol"),
    ("get_forex_quote", {"symbol": "USDCNY"}, "symbol"),
    ("get_commodity_futures_list", {"market": "dce"}, None),
    ("get_hk_special_ranking", {"node": "gqg_hk"}, None),
    ("get_us_fund_flow_history", {"symbol": "AAPL"}, "symbol"),
]


@pytest.mark.parametrize("tool_name,args,required", _TOOL_ROUTE_CASES,
                         ids=[c[0] for c in _TOOL_ROUTE_CASES])
def test_execute_tool_routes_new_tools(tool_name, args, required):
    """patch _mcp_call 返回 {} 后调用不应报'未注册'类错误。"""
    from agent.tools import execute_tool
    with patch(MCP, return_value={}):
        res = execute_tool(tool_name, dict(args))
    assert isinstance(res, dict) and "ok" in res
    assert "未注册" not in str(res.get("error", "")), \
        f"{tool_name} 未被 execute_tool 识别: {res}"
    assert res["ok"] is True, f"{tool_name} 路由失败: {res}"


@pytest.mark.parametrize("tool_name,args,required", _TOOL_ROUTE_CASES,
                         ids=[c[0] for c in _TOOL_ROUTE_CASES])
def test_execute_tool_missing_required_param(tool_name, args, required):
    if required is None:
        pytest.skip(f"{tool_name} 无必填参数")
    from agent.tools import execute_tool
    bad_args = {k: v for k, v in args.items() if k != required}
    with patch(MCP, return_value={}):
        res = execute_tool(tool_name, bad_args)
    assert res["ok"] is False
    assert required in str(res.get("error", ""))


# ════════════════════════════════════════════════════════════════
# ⑤ 模糊去重 _dedup_news_fuzzy
# ════════════════════════════════════════════════════════════════

class TestFuzzyDedup:
    def test_column_prefix_variant_deduped(self):
        items = [
            {"title": "中邮证券：维持贵州茅台“买入”评级，市场化定价持续兑现"},
            {"title": "研报掘金丨中邮证券：维持贵州茅台“买入”评级，市场化定价持续兑现"},
        ]
        out = _dedup_news_fuzzy(items)
        assert len(out) == 1
        assert out[0]["title"] == items[0]["title"]  # 保留先出现者

    def test_different_headlines_not_killed(self):
        items = [
            {"title": "贵州茅台跌1.00%，成交额43.93亿元"},
            {"title": "贵州茅台涨2.3%创新高"},
        ]
        out = _dedup_news_fuzzy(items)
        assert len(out) == 2

    def test_empty_title_kept(self):
        items = [{"title": ""}, {"title": ""}]
        assert len(_dedup_news_fuzzy(items)) == 2


# ════════════════════════════════════════════════════════════════
# ⑥ fetch_news_pool：只含 mcp/flash 两键，跨源近似标题被合并
# ════════════════════════════════════════════════════════════════

class TestNewsPool:
    def test_keys_and_cross_source_dedup(self):
        mcp_items = [
            {"title": "中邮证券：维持贵州茅台“买入”评级，市场化定价持续兑现",
             "time": "2025-01-01 10:00:00", "source": "智研", "content": "正文甲"},
        ]
        flash_items = [
            {"title": "研报掘金丨中邮证券：维持贵州茅台“买入”评级，市场化定价持续兑现",
             "time": "2025-01-01 10:05:00", "source": "智研快讯"},
            {"title": "央行开展 5000 亿元 MLF 操作",
             "time": "2025-01-01 09:00:00", "source": "智研快讯"},
        ]
        with patch("agent.data_fetcher.fetch_mcp_news", return_value=mcp_items), \
             patch("agent.data_fetcher.fetch_mcp_flash", return_value=flash_items):
            pool = fetch_news_pool(["白酒"])
        assert set(pool.keys()) == {"mcp", "flash"}
        assert len(pool["mcp"]) == 1
        # 快讯里的换皮重复被合并，只留真正不同的 1 条
        assert len(pool["flash"]) == 1
        assert pool["flash"][0]["title"] == "央行开展 5000 亿元 MLF 操作"

    def test_single_source_failure_degrades(self):
        with patch("agent.data_fetcher.fetch_mcp_news", side_effect=RuntimeError("x")), \
             patch("agent.data_fetcher.fetch_mcp_flash", return_value=[
                 {"title": "快讯甲", "time": "", "source": "智研快讯"}]):
            pool = fetch_news_pool()
        assert set(pool.keys()) == {"mcp", "flash"}
        assert pool["mcp"] == []
        assert len(pool["flash"]) == 1


# ════════════════════════════════════════════════════════════════
# ⑦ TOOL_REGISTRY 数量与目录
# ════════════════════════════════════════════════════════════════

NEW_TOOL_NAMES = [c[0] for c in _TOOL_ROUTE_CASES]


def test_tool_registry_size_and_catalog():
    from agent.tools import TOOL_REGISTRY, get_tool_catalog
    assert len(TOOL_REGISTRY) == 54
    catalog = get_tool_catalog()
    registered = {t["function"]["name"] for t in TOOL_REGISTRY}
    for name in NEW_TOOL_NAMES:
        assert name in registered, f"{name} 不在 TOOL_REGISTRY"
        assert name in catalog, f"{name} 不在 get_tool_catalog 输出"


class TestEnsureWritableHome:
    """Railway 只读 /root 实锤：Tushare SDK 写 tk.csv 崩初始化（PermissionError）。
    _ensure_writable_home 在 HOME 不可写时改指 /tmp，可写时零副作用。"""

    def test_readonly_home_redirected_to_tmp(self, monkeypatch):
        import agent.data_fetcher as df
        monkeypatch.setenv("HOME", "/nonexistent_readonly_home")
        monkeypatch.setattr(df.os.path, "expanduser", lambda p: "/nonexistent_readonly_home")
        monkeypatch.setattr(df.os, "access", lambda p, m: False)
        df._ensure_writable_home()
        assert df.os.environ["HOME"] == "/tmp"

    def test_writable_home_untouched(self, monkeypatch):
        import agent.data_fetcher as df
        monkeypatch.setenv("HOME", "/writable_home")
        monkeypatch.setattr(df.os.path, "expanduser", lambda p: "/writable_home")
        monkeypatch.setattr(df.os, "access", lambda p, m: True)
        df._ensure_writable_home()
        assert df.os.environ["HOME"] == "/writable_home"

    def test_get_pro_calls_ensure_before_init(self, monkeypatch):
        import agent.data_fetcher as df
        calls = []
        monkeypatch.setattr(df, "_ensure_writable_home", lambda: calls.append("ensure"))
        monkeypatch.setattr(df, "_env", lambda k, d="": "fake_token")
        import sys, types
        fake_ts = types.SimpleNamespace(
            set_token=lambda t: calls.append(("set_token", t)),
            pro_api=lambda: "PRO",
        )
        monkeypatch.setitem(sys.modules, "tushare", fake_ts)
        assert df._get_pro() == "PRO"
        assert calls[0] == "ensure"
        assert calls[1] == ("set_token", "fake_token")
