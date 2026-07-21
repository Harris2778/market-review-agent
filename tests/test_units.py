"""
数据采集层单位换算回归测试（tests/test_units.py）。

历史背景：本项目曾出现 10 倍量级的单位偏差 bug（如申万行业成交额
误用 /1e8 而非 /1e5、把成交量标成成交额）。本文件通过 mock 全部外部
数据源（tushare pro、requests/新浪 MCP），对 agent/data_fetcher.py 的
单位换算逻辑做精确断言，防止回归。

统一换算口径（以生产代码注释为准）：
- index_daily.vol    : 手   → 万手  (/10000)
- index_daily.amount : 千元 → 亿元  (/1e5)
- moneyflow *_amount : 万元 → 亿元  (/10000)
- moneyflow_hsgt     : 万元 → 亿元  (/10000)
- margin.rzye        : 元   → 亿元  (/1e8)
- hk_hold.vol        : 股   → 亿股  (/1e8)

运行：/usr/local/bin/python3 -m pytest tests/test_units.py -v
绝不发起真实网络请求。
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# 保证无论 conftest.py 是否就绪，都能从项目根导入 agent 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import data_fetcher  # noqa: E402
from agent.data_fetcher import MarketSnapshot  # noqa: E402

DATE = "20240115"


# ─────────────────────────────────────────────
# 测试数据构造辅助
# ─────────────────────────────────────────────

def _make_index_daily(rows=30, close=3900.0, pct_chg=0.5,
                      vol=50000.0, amount=3.5e8):
    """构造 index_daily 返回的 DataFrame（vol 单位：手；amount 单位：千元）。"""
    dates = pd.date_range("2023-10-01", periods=rows).strftime("%Y%m%d")
    return pd.DataFrame({
        "trade_date": dates,
        "open": [close] * rows,
        "high": [close] * rows,
        "low": [close] * rows,
        "close": [close] * rows,
        "pct_chg": [pct_chg] * rows,
        "vol": [vol] * rows,
        "amount": [amount] * rows,
    })


def _mock_pro(**overrides):
    """构造 mock 的 tushare pro 对象，可按接口名覆盖返回值。"""
    pro = MagicMock(name="tushare_pro")
    for attr, val in overrides.items():
        setattr(pro, attr, val)
    return pro


# ─────────────────────────────────────────────
# 1) fetch_a_share_indices：vol /10000（万手）、amount /1e5（千元→亿）
# ─────────────────────────────────────────────

class TestFetchAShareIndices:
    RAW_VOL = 50000.0        # 手
    RAW_AMOUNT = 3.5e8       # 千元

    def _run(self):
        pro = _mock_pro()
        pro.index_daily.return_value = _make_index_daily(
            vol=self.RAW_VOL, amount=self.RAW_AMOUNT)
        with patch.object(data_fetcher, "_get_pro", return_value=pro):
            return data_fetcher.fetch_a_share_indices(DATE)

    def test_vol_divided_by_10000(self):
        """vol（手）必须换算为万手：原始值 / 10000。"""
        result = self._run()
        assert "上证综指" in result
        assert result["上证综指"]["vol"] == pytest.approx(
            round(self.RAW_VOL / 10000, 2))

    def test_amount_divided_by_1e5(self):
        """amount（千元）必须换算为亿元：原始值 / 1e5，而不是 /1e8 或原样。"""
        result = self._run()
        amt = result["上证综指"]["amount"]
        assert amt == pytest.approx(round(self.RAW_AMOUNT / 1e5, 2))
        # 防回归：绝不能是 /1e8（10 倍偏差）或原始值
        assert amt != pytest.approx(round(self.RAW_AMOUNT / 1e8, 2))
        assert amt != pytest.approx(self.RAW_AMOUNT)

    def test_all_seven_indices_converted(self):
        """7 大指数全部返回且单位一致换算。"""
        result = self._run()
        assert len(result) == len(data_fetcher.A_INDEX_CODES)
        for name, d in result.items():
            assert d["vol"] == pytest.approx(round(self.RAW_VOL / 10000, 2)), name
            assert d["amount"] == pytest.approx(round(self.RAW_AMOUNT / 1e5, 2)), name


# ─────────────────────────────────────────────
# 2) fetch_shenwan_sectors：amount 必须 /1e5（历史 bug：曾误用 /1e8）
# ─────────────────────────────────────────────

class TestFetchShenwanSectors:
    RAW_VOL = 800000.0       # 手
    RAW_AMOUNT = 1.23e9      # 千元

    def _run(self):
        pro = _mock_pro()
        pro.index_daily.return_value = _make_index_daily(
            vol=self.RAW_VOL, amount=self.RAW_AMOUNT)
        with patch.object(data_fetcher, "_get_pro", return_value=pro):
            return data_fetcher.fetch_shenwan_sectors(DATE)

    def test_amount_divided_by_1e5_not_1e8(self):
        """申万行业成交额：千元→亿 必须 /1e5。这是历史上出 10 倍偏差 bug 的地方。"""
        sectors = self._run()
        assert len(sectors) == len(data_fetcher.SW_SECTOR_CODES)
        expected = round(self.RAW_AMOUNT / 1e5, 2)      # 12300.0 亿
        wrong = round(self.RAW_AMOUNT / 1e7, 2)          # 123.0（历史 bug 值，差 100 倍）
        assert expected != wrong  # 测试数据本身能区分两种口径
        for s in sectors:
            assert s["amount"] == pytest.approx(expected), s["name"]
            assert s["amount"] != pytest.approx(wrong), s["name"]

    def test_vol_divided_by_10000(self):
        """申万行业成交量：手→万手 /10000。"""
        sectors = self._run()
        for s in sectors:
            assert s["vol"] == pytest.approx(round(self.RAW_VOL / 10000, 2)), s["name"]


# ─────────────────────────────────────────────
# 3) 个股资金流 moneyflow：大/中/小单净额 /10000（万元→亿）
#    经由 fetch_sector_stock_detail（moneyflow 唯一使用点）
# ─────────────────────────────────────────────

class TestMoneyflowUnits:
    # 单位：万元
    BUY_LG, SELL_LG = 150000.0, 50000.0    # 大单净 +100000 万 = +10.0 亿
    BUY_MD, SELL_MD = 30000.0, 80000.0     # 中单净 -50000 万 = -5.0 亿
    BUY_SM, SELL_SM = 10000.0, 40000.0     # 小单净 -30000 万 = -3.0 亿
    RAW_AMOUNT = 2.0e7                     # 个股日成交额（千元）
    RAW_VOL = 100000.0                     # 个股日成交量（手）

    def _run(self):
        pro = _mock_pro()
        pro.stock_basic.return_value = pd.DataFrame({
            "ts_code": ["000001.SZ", "000002.SZ"],
            "name": ["平安银行", "万科A"],
            "industry": ["银行", "房地产"],
        })
        pro.index_member.return_value = pd.DataFrame({
            "con_code": ["000001.SZ", "000002.SZ"],
        })
        pro.daily.return_value = pd.DataFrame({
            "ts_code": ["000001.SZ", "000002.SZ"],
            "open": [10.0, 20.0], "high": [11.0, 21.0],
            "low": [9.5, 19.5], "close": [10.5, 20.5],
            "pre_close": [10.0, 20.0],
            "pct_chg": [5.0, 2.5],
            "vol": [self.RAW_VOL, self.RAW_VOL],
            "amount": [self.RAW_AMOUNT, self.RAW_AMOUNT],
        })
        pro.moneyflow.return_value = pd.DataFrame({
            "ts_code": ["000001.SZ"],
            "buy_lg_amount": [self.BUY_LG], "sell_lg_amount": [self.SELL_LG],
            "buy_md_amount": [self.BUY_MD], "sell_md_amount": [self.SELL_MD],
            "buy_sm_amount": [self.BUY_SM], "sell_sm_amount": [self.SELL_SM],
        })
        with patch.object(data_fetcher, "_get_pro", return_value=pro):
            # 清掉模块级名称缓存，保证 stock_basic 走 mock
            with patch.dict(data_fetcher._stock_name_cache, {}, clear=True), \
                 patch.dict(data_fetcher._stock_sector_cache, {}, clear=True):
                return data_fetcher.fetch_sector_stock_detail("电子", DATE)

    def test_lg_md_sm_net_divided_by_10000(self):
        """大/中/小单净额：万元→亿 必须 /10000。"""
        result = self._run()
        ff = result["fund_flow"]
        assert ff["lg_net"] == pytest.approx(
            round((self.BUY_LG - self.SELL_LG) / 10000, 2))   # +10.0 亿
        assert ff["md_net"] == pytest.approx(
            round((self.BUY_MD - self.SELL_MD) / 10000, 2))   # -5.0 亿
        assert ff["sm_net"] == pytest.approx(
            round((self.BUY_SM - self.SELL_SM) / 10000, 2))   # -3.0 亿
        # 防回归：绝不能是万元原值（差 1e4 倍）
        assert ff["lg_net"] != pytest.approx(self.BUY_LG - self.SELL_LG)

    def test_total_amount_divided_by_1e5_and_vol_by_10000(self):
        """板块合计成交额：千元→亿 /1e5；合计成交量：手→万手 /10000。"""
        result = self._run()
        assert result["total_amount"] == pytest.approx(
            round(2 * self.RAW_AMOUNT / 1e5, 2))
        assert result["total_vol"] == pytest.approx(
            round(2 * self.RAW_VOL / 10000, 2))

    def test_top_gainers_vol_divided_by_10000(self):
        """领涨股 vol 字段同样是万手口径。"""
        result = self._run()
        assert result["top_gainers"], "应至少有 1 只领涨股"
        for s in result["top_gainers"]:
            assert s["vol"] == pytest.approx(round(self.RAW_VOL / 10000, 2))


# ─────────────────────────────────────────────
# 4) fetch_fund_flows：north/south_money /10000（万元→亿）、rzye /1e8（元→亿）
# ─────────────────────────────────────────────

class TestFetchFundFlows:
    NORTH = 856000.0      # 万元 → 85.6 亿
    SOUTH = 423000.0      # 万元 → 42.3 亿
    RZYE = 1.58e12        # 元   → 15800.0 亿

    def _run(self, north=NORTH, south=SOUTH):
        pro = _mock_pro()
        pro.moneyflow_hsgt.return_value = pd.DataFrame({
            "trade_date": [DATE],
            "north_money": [north],
            "south_money": [south],
        })
        pro.margin.return_value = pd.DataFrame({
            "exchange_id": ["SSE", "SZSE"],
            "rzye": [self.RZYE * 0.6, self.RZYE * 0.4],
        })
        with patch.object(data_fetcher, "_get_pro", return_value=pro):
            return data_fetcher.fetch_fund_flows(DATE)

    def test_north_money_divided_by_10000(self):
        """北向资金：万元→亿 /10000。"""
        result = self._run()
        assert result["north_bound"] == pytest.approx(round(self.NORTH / 10000, 2))
        assert result["north_bound"] != pytest.approx(self.NORTH)  # 防万元原值回归

    def test_south_money_divided_by_10000(self):
        """南向资金：万元→亿 /10000。"""
        result = self._run()
        assert result["south_bound"] == pytest.approx(round(self.SOUTH / 10000, 2))

    def test_margin_rzye_divided_by_1e8(self):
        """融资余额 rzye：元→亿 /1e8，且为两所合计。"""
        result = self._run()
        assert result["margin_bal"] == pytest.approx(round(self.RZYE / 1e8, 2))
        assert result["margin_bal"] != pytest.approx(round(self.RZYE / 1e5, 2))

    def test_negative_flows_are_included(self):
        """净流出（负值）必须写入结果——北向净卖出是重要信号，不得丢弃；0 才视为无数据。"""
        result = self._run(north=0.0, south=-100.0)
        assert "north_bound" not in result  # 0 表示无数据/暂停，不写入
        assert result["south_bound"] == round(-100.0 / 10000, 2)  # 净流出保留


# ─────────────────────────────────────────────
# 6 前置) fetch_north_holdings：hk_hold.vol /1e8（股→亿股）
# ─────────────────────────────────────────────

class TestFetchNorthHoldings:
    def test_vol_divided_by_1e8(self):
        """北向持仓股数：股→亿股 /1e8。"""
        raw_vol = 1.5e8  # 股 → 1.5 亿股
        pro = _mock_pro()
        pro.hk_hold.return_value = pd.DataFrame({
            "name": ["贵州茅台", "宁德时代"],
            "ts_code": ["600519.SH", "300750.SZ"],
            "vol": [raw_vol, raw_vol / 2],
            "ratio": [7.5, 3.2],
        })
        with patch.object(data_fetcher, "_get_pro", return_value=pro):
            items = data_fetcher.fetch_north_holdings()
        assert len(items) == 2
        # 按 vol 降序，第一只是贵州茅台
        assert items[0]["name"] == "贵州茅台"
        assert items[0]["vol"] == pytest.approx(round(raw_vol / 1e8, 2))
        assert items[0]["vol"] != pytest.approx(raw_vol)  # 防股数原值回归
        assert items[1]["vol"] == pytest.approx(round(raw_vol / 2 / 1e8, 2))


# ─────────────────────────────────────────────
# 5) format_market_data_for_prompt 指数行：『成交额X亿』是真实换算值
# ─────────────────────────────────────────────

class TestFormatPromptIndexLines:
    def _snapshot(self, indices):
        snap = MarketSnapshot(date=DATE)
        snap.indices = indices
        # 注：dataclass 默认 news_items=list / macro_data=dict，
        # 但 format_market_data_for_prompt 对两者都用 .get()，
        # 实际运行时被赋值为 dict，这里按真实运行形态构造。
        snap.news_items = {}
        snap.macro_data = {}
        return snap

    def test_amount_rendered_as_chengjiaoe(self):
        """指数行必须输出『成交额X亿』，X 是 amount 字段的真实换算值。"""
        snap = self._snapshot({
            "上证综指": {"close": 3900.0, "pct_chg": 0.5, "vol": 5.0,
                         "amount": 35.0, "chg_5d": None, "chg_20d": None,
                         "vol_ratio": None, "mas_above": "—"},
        })
        text = data_fetcher.format_market_data_for_prompt(snap)
        assert "成交额35.0亿" in text
        # 防回归：有 amount 时不得退化成『成交量...万手』
        idx_line = next(l for l in text.splitlines() if "上证综指" in l)
        assert "成交量" not in idx_line

    def test_amount_is_real_converted_value_not_vol(self):
        """端到端：fetch_a_share_indices 的 amount 换算值原样进入 prompt，
        不得把成交量数值标成成交额（历史 bug）。"""
        raw_vol, raw_amount = 50000.0, 3.5e8   # → 5.0 万手 / 35.0 亿
        pro = _mock_pro()
        pro.index_daily.return_value = _make_index_daily(
            vol=raw_vol, amount=raw_amount)
        with patch.object(data_fetcher, "_get_pro", return_value=pro):
            indices = data_fetcher.fetch_a_share_indices(DATE)
        snap = self._snapshot(indices)
        text = data_fetcher.format_market_data_for_prompt(snap)
        assert "成交额3500.0亿" in text        # amount/1e5 的真实值
        assert "成交额5.0亿" not in text       # 不得把 vol 当成交额
        assert "成交额50000" not in text       # 不得出现未换算原值

    def test_fallback_to_vol_when_amount_missing(self):
        """amount 为 0/缺失时回退为『成交量X万手』标注。"""
        snap = self._snapshot({
            "深证成指": {"close": 12000.0, "pct_chg": -0.3, "vol": 8.5,
                         "amount": 0, "chg_5d": None, "chg_20d": None,
                         "vol_ratio": None, "mas_above": "—"},
        })
        text = data_fetcher.format_market_data_for_prompt(snap)
        assert "成交量8.5万手" in text


# ─────────────────────────────────────────────
# 6) 北向持仓在 prompt 中标注为『亿股』
# ─────────────────────────────────────────────

class TestFormatPromptNorthHoldings:
    def _snapshot(self):
        snap = MarketSnapshot(date=DATE)
        snap.news_items = {}
        snap.macro_data = {}
        return snap

    def test_north_holdings_labeled_yigu(self):
        """北向持仓 TOP 列表必须以『亿股』为单位标注。"""
        snap = self._snapshot()
        snap._north_hold = [
            {"name": "贵州茅台", "code": "600519.SH", "vol": 1.5, "ratio": 7.5},
        ]
        snap._north_sector = {}
        text = data_fetcher.format_market_data_for_prompt(snap)
        assert "持仓1.5亿股" in text
        assert "占比7.5%" in text

    def test_north_sector_distribution_labeled_yigu(self):
        """北向行业分布同样以『亿股』标注。"""
        snap = self._snapshot()
        snap._north_hold = [
            {"name": "贵州茅台", "code": "600519.SH", "vol": 1.5, "ratio": 7.5},
        ]
        snap._north_sector = {"食品饮料": 12.0}
        text = data_fetcher.format_market_data_for_prompt(snap)
        assert "食品饮料: 12亿股" in text


# ─────────────────────────────────────────────
# 7) fetch_market_breadth：新浪 MCP CSV 解析 + 字段不足容错
# ─────────────────────────────────────────────

# 13 字段 CSV：f0=忽略, f1=日期, f2..f6=跌档, f7=平, f8..f12=涨档
GOOD_CSV = "x,20240115,5,10,20,100,500,50,600,200,30,15,8"


class TestFetchMarketBreadth:
    def test_normal_parse_via_requests(self):
        """从 requests 层 mock 整个新浪 MCP 会话，验证 CSV 正常解析与汇总。"""
        r_init = MagicMock()
        r_init.headers = {"Mcp-Session-Id": "test-session"}
        r_call = MagicMock()
        r_call.json.return_value = {
            "result": {"content": [
                {"text": f'var cnMarketUpdownDistribution="{GOOD_CSV}"'}]}}

        with patch.dict(os.environ, {"SINA_MCP_TOKEN": "test-token"}), \
             patch.object(data_fetcher.requests, "post",
                          side_effect=[r_init, r_call]) as mock_post:
            result = data_fetcher.fetch_market_breadth()

        assert mock_post.call_count == 2  # initialize + tools/call，均已被拦截
        assert result["date"] == "20240115"
        assert result["跌停"] == "5"
        assert result["涨停"] == "8"
        assert result["平"] == "50"
        # total_up = 600+200+30+15+8, total_down = 5+10+20+100+500
        assert result["total_up"] == "853"
        assert result["total_down"] == "635"

    def test_short_csv_returns_empty_dict_no_indexerror(self):
        """字段数 < 13 时返回 {}，绝不抛 IndexError（历史 bug 回归点）。"""
        with patch.object(data_fetcher, "_mcp_call",
                          return_value={"raw": "a,b,c", "type": "csv"}):
            result = data_fetcher.fetch_market_breadth()
        assert result == {}

    def test_twelve_fields_still_returns_empty(self):
        """恰好 12 个字段（差 1 个）同样安全返回 {}。"""
        raw = ",".join(["x"] * 12)
        with patch.object(data_fetcher, "_mcp_call",
                          return_value={"raw": raw, "type": "csv"}):
            assert data_fetcher.fetch_market_breadth() == {}

    def test_empty_raw_returns_empty_dict(self):
        """raw 为空或不含逗号时返回 {}。"""
        with patch.object(data_fetcher, "_mcp_call", return_value={}):
            assert data_fetcher.fetch_market_breadth() == {}
        with patch.object(data_fetcher, "_mcp_call",
                          return_value={"raw": "nocomma", "type": "text"}):
            assert data_fetcher.fetch_market_breadth() == {}

    def test_non_numeric_fields_returns_empty_dict(self):
        """字段数足够但含非数字时，ValueError 被捕获并返回 {}。"""
        raw = "x,20240115,abc,10,20,100,500,50,600,200,30,15,8"
        with patch.object(data_fetcher, "_mcp_call",
                          return_value={"raw": raw, "type": "csv"}):
            assert data_fetcher.fetch_market_breadth() == {}
