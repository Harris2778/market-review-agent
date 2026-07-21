"""板块深度分析（第一期）测试。

覆盖范围：
1. fetch_sector_valuation：成分股总市值加权 PE/PB 的手算正确性
   （非成分股剔除、pe<=0 不参与 PE 加权）+ 近一年月度采样历史分位。
2. fetch_sector_moneyflow：主力净流入=(特大单+大单买)-(特大单+大单卖)，
   万元→亿 ÷10000 换算，top_inflow/top_outflow 排序。
3. 失败路径：index_member 抛异常 / 返回空 / _get_pro 为 None 时，
   两个函数都返回 note 非空的安全结构而不抛异常。
4. 板块深度分析 prompt：五维度框架关键词、数据真实性红线、禁用词清单。
5. orchestrator 接线：_sector_deep_dive 必须调用两个新数据函数，
   且估值/资金段落进入最终 user prompt。

规则（与项目其他测试一致）：
- 所有外部调用全部 mock（_get_pro 返回 mock pro 对象 / DeepSeek 客户端），
  绝不发起真实网络请求。
- 不用 create=True patch 不存在的函数——接线缺失必须表现为测试失败。
- 无 pytest-asyncio，异步函数一律用 asyncio.run 驱动。

背景契约（生产代码由并行代理实现）：
- fetch_sector_valuation(sector_name, trade_date) ->
  {'pe','pb','pe_percentile','pb_percentile','sample_count','note'}
- fetch_sector_moneyflow(sector_name, trade_date) ->
  {'main_net','retail_net','top_inflow','top_outflow','stock_count','note'}
- moneyflow 金额单位万元（÷10000=亿）；daily_basic.total_mv 单位万元。
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

import agent.data_fetcher as data_fetcher
import agent.orchestrator as orchestrator
from agent.data_fetcher import (
    fetch_sector_earnings,
    fetch_sector_moneyflow,
    fetch_sector_valuation,
)
from agent.system_prompts import get_system_prompt

# ── 板块 extras 当日缓存：进程级状态，每个用例前后清空，避免跨用例污染 ──


@pytest.fixture(autouse=True)
def _clear_sector_extras_cache():
    data_fetcher._sector_extras_cache.clear()
    data_fetcher._sector_extras_key_locks.clear()
    yield
    data_fetcher._sector_extras_cache.clear()
    data_fetcher._sector_extras_key_locks.clear()


# ── 测试数据常量 ──

SECTOR = "煤炭"
IDX_CODE = data_fetcher.SW_INDEX_MAP[SECTOR]  # 801950.SI
TRADE_DATE = "20250718"  # 周五

STOCK_A = "601001.SH"
STOCK_B = "601002.SH"
STOCK_C = "601003.SH"
STOCK_D = "600999.SH"  # 非成分股，必须被剔除

MEMBERS = [STOCK_A, STOCK_B, STOCK_C]

# 当日估值数据（total_mv 单位：万元）。
# C 的 pe<=0 → 不参与 PE 加权（但 pb>0，仍参与 PB 加权）；
# D 不是成分股 → 任何指标都不参与。
CURR_VALUATION = {
    STOCK_A: {"pe": 10.0, "pe_ttm": 10.0, "pb": 1.0, "total_mv": 10000.0},
    STOCK_B: {"pe": 20.0, "pe_ttm": 20.0, "pb": 2.0, "total_mv": 20000.0},
    STOCK_C: {"pe": -5.0, "pe_ttm": -5.0, "pb": 3.0, "total_mv": 30000.0},
    STOCK_D: {"pe": 999.0, "pe_ttm": 999.0, "pb": 99.0, "total_mv": 900000.0},
}

# 历史月度采样日估值（全部低于当日 → 当日分位应为 100 或接近）。
HIST_VALUATION = {
    STOCK_A: {"pe": 8.0, "pe_ttm": 8.0, "pb": 0.8, "total_mv": 10000.0},
    STOCK_B: {"pe": 15.0, "pe_ttm": 15.0, "pb": 1.5, "total_mv": 20000.0},
    STOCK_C: {"pe": 10.0, "pe_ttm": 10.0, "pb": 2.0, "total_mv": 30000.0},
    STOCK_D: {"pe": 500.0, "pe_ttm": 500.0, "pb": 50.0, "total_mv": 900000.0},
}

HIST_DATES = ["20250115", "20250214", "20250314", "20250415", "20250515", "20250613"]

# 手算期望值：
# PE = (10×10000 + 20×20000) / (10000+20000) = 500000/30000（C 的 pe<=0 剔除）
EXPECTED_PE = 500000.0 / 30000.0           # ≈ 16.667
# PB = (1×10000 + 2×20000 + 3×30000) / 60000
EXPECTED_PB = 140000.0 / 60000.0           # ≈ 2.333


def _member_df():
    """index_member 返回的成分股表（3 只）。"""
    return pd.DataFrame({
        "index_code": [IDX_CODE] * len(MEMBERS),
        "con_code": list(MEMBERS),
    })


def _daily_basic_rows(trade_date: str) -> list:
    table = CURR_VALUATION if trade_date == TRADE_DATE else HIST_VALUATION
    return [
        {"ts_code": code, "trade_date": trade_date, **vals}
        for code, vals in table.items()
    ]


def _fake_daily_basic(**kwargs):
    """模拟 pro.daily_basic：当日返回当日估值，历史日期返回历史估值。

    同时支持按 trade_date 单日查询和按 start_date/end_date 区间查询，
    以及按 ts_code 过滤——兼容生产端不同的采样调用方式。
    """
    trade_date = kwargs.get("trade_date")
    dates = [trade_date] if trade_date else list(HIST_DATES)
    rows = []
    for d in dates:
        rows.extend(_daily_basic_rows(d))
    df = pd.DataFrame(rows)
    ts_code = kwargs.get("ts_code")
    if ts_code:
        codes = set(str(ts_code).split(","))
        df = df[df["ts_code"].isin(codes)].reset_index(drop=True)
    return df


# ── 资金流向测试数据（金额单位：万元）──
# 主力净额 = (buy_elg + buy_lg) - (sell_elg + sell_lg)
#   A: (50000+40000)-(30000+20000) = +40000 万（流入第一）
#   B: (10000+10000)-(40000+30000) = -50000 万（流出第一）
#   C: (20000+20000)-(10000+10000) = +20000 万
# 合计主力净流入 = +10000 万 = 1.0 亿
# 中单全部置 0 → 散户净额（无论定义为小单还是中小单）= -5000+15000+2000 = +12000 万 = 1.2 亿
MONEYFLOW_ROWS = [
    {"ts_code": STOCK_A, "trade_date": TRADE_DATE,
     "buy_elg_amount": 50000.0, "sell_elg_amount": 30000.0,
     "buy_lg_amount": 40000.0, "sell_lg_amount": 20000.0,
     "buy_md_amount": 0.0, "sell_md_amount": 0.0,
     "buy_sm_amount": 10000.0, "sell_sm_amount": 15000.0},
    {"ts_code": STOCK_B, "trade_date": TRADE_DATE,
     "buy_elg_amount": 10000.0, "sell_elg_amount": 40000.0,
     "buy_lg_amount": 10000.0, "sell_lg_amount": 30000.0,
     "buy_md_amount": 0.0, "sell_md_amount": 0.0,
     "buy_sm_amount": 20000.0, "sell_sm_amount": 5000.0},
    {"ts_code": STOCK_C, "trade_date": TRADE_DATE,
     "buy_elg_amount": 20000.0, "sell_elg_amount": 10000.0,
     "buy_lg_amount": 20000.0, "sell_lg_amount": 10000.0,
     "buy_md_amount": 0.0, "sell_md_amount": 0.0,
     "buy_sm_amount": 2000.0, "sell_sm_amount": 0.0},
    # 非成分股：巨额资金，若未剔除会严重污染聚合结果
    {"ts_code": STOCK_D, "trade_date": TRADE_DATE,
     "buy_elg_amount": 999999.0, "sell_elg_amount": 0.0,
     "buy_lg_amount": 999999.0, "sell_lg_amount": 0.0,
     "buy_md_amount": 0.0, "sell_md_amount": 0.0,
     "buy_sm_amount": 999999.0, "sell_sm_amount": 0.0},
]

EXPECTED_MAIN_NET = 1.0    # 亿（+10000 万 ÷ 10000）
EXPECTED_RETAIL_NET = 1.2  # 亿（+12000 万 ÷ 10000）


def _fake_moneyflow(**kwargs):
    """模拟 pro.moneyflow：返回全市场资金流向，可按 ts_code 过滤。"""
    df = pd.DataFrame(MONEYFLOW_ROWS)
    ts_code = kwargs.get("ts_code")
    if ts_code:
        codes = set(str(ts_code).split(","))
        df = df[df["ts_code"].isin(codes)].reset_index(drop=True)
    return df


def _make_pro(**overrides):
    """构造 mock 的 tushare pro 对象，默认接好 index_member。"""
    pro = MagicMock(name="tushare_pro")
    pro.index_member.return_value = _member_df()
    for attr, val in overrides.items():
        setattr(pro, attr, val)
    return pro


def _entry_code(entry) -> str:
    """从 top_inflow/top_outflow 条目里提取股票代码，兼容 dict/tuple/str。"""
    if isinstance(entry, dict):
        for key in ("ts_code", "code", "股票代码", "symbol"):
            if key in entry:
                return str(entry[key])
        return str(entry)
    if isinstance(entry, (list, tuple)) and entry:
        return str(entry[0])
    return str(entry)


def _entry_codes(entries) -> list:
    return [_entry_code(e) for e in entries]


# ════════════════════════════════════════════════════════════════
# 1. fetch_sector_valuation：加权估值 + 历史分位
# ════════════════════════════════════════════════════════════════


class TestFetchSectorValuation:
    """成分股总市值加权 PE/PB + 近一年月度采样历史分位。"""

    def _run(self):
        pro = _make_pro()
        pro.daily_basic.side_effect = _fake_daily_basic
        with patch.object(data_fetcher, "_get_pro", return_value=pro):
            return fetch_sector_valuation(SECTOR, TRADE_DATE)

    def test_contract_keys(self):
        result = self._run()
        for key in ("pe", "pb", "pe_percentile", "pb_percentile",
                    "sample_count", "note"):
            assert key in result, f"返回结构缺少契约字段 {key}: {result}"

    def test_weighted_pe_excludes_nonpositive_pe(self):
        """加权 PE 手算值：pe<=0 的成分股 C 不参与 PE 加权。"""
        result = self._run()
        # 实现保留 2 位小数，容差取 0.005
        assert result["pe"] == pytest.approx(EXPECTED_PE, abs=0.005), (
            f"加权 PE 应为 {EXPECTED_PE:.3f}（剔除 pe<=0 的 C），实际 {result['pe']}"
        )
        # 若错误地把 C 的市值留在分母，会得到 500000/60000≈8.33——必须能区分
        assert result["pe"] != pytest.approx(500000.0 / 60000.0, rel=1e-3)

    def test_weighted_pb_includes_all_members(self):
        """加权 PB 手算值：3 只成分股全部参与（pb 均>0）。"""
        result = self._run()
        assert result["pb"] == pytest.approx(EXPECTED_PB, abs=0.005), (
            f"加权 PB 应为 {EXPECTED_PB:.3f}，实际 {result['pb']}"
        )

    def test_non_member_excluded(self):
        """非成分股 D（pe=999, total_mv=90亿）若混入会把加权 PE 拉到 ~900 以上。"""
        result = self._run()
        assert result["pe"] < 100, (
            f"非成分股未被剔除，加权 PE 异常：{result['pe']}"
        )
        assert result["pb"] < 50, (
            f"非成分股未被剔除，加权 PB 异常：{result['pb']}"
        )

    def test_sample_count(self):
        """sample_count 至少覆盖两只 pe>0 的成分股。"""
        result = self._run()
        assert isinstance(result["sample_count"], int)
        assert result["sample_count"] >= 2

    def test_percentile_near_100_when_current_above_all_history(self):
        """当日加权 PE/PB 高于全部历史月度样本 → 分位应为 100 或接近。"""
        result = self._run()
        assert result["pe_percentile"] >= 95, (
            f"当前 PE 高于所有历史样本，分位应接近 100，实际 {result['pe_percentile']}"
        )
        assert result["pb_percentile"] >= 95, (
            f"当前 PB 高于所有历史样本，分位应接近 100，实际 {result['pb_percentile']}"
        )


# ════════════════════════════════════════════════════════════════
# 2. fetch_sector_moneyflow：主力净流入 + 换算 + 排序
# ════════════════════════════════════════════════════════════════


class TestFetchSectorMoneyflow:
    """主力（特大单+大单）净流入，万元→亿 ÷10000。"""

    def _run(self):
        pro = _make_pro()
        pro.moneyflow.side_effect = _fake_moneyflow
        with patch.object(data_fetcher, "_get_pro", return_value=pro):
            return fetch_sector_moneyflow(SECTOR, TRADE_DATE)

    def test_contract_keys(self):
        result = self._run()
        for key in ("main_net", "retail_net", "top_inflow",
                    "top_outflow", "stock_count", "note"):
            assert key in result, f"返回结构缺少契约字段 {key}: {result}"

    def test_main_net_formula(self):
        """主力净流入 = (特大单买+大单买) - (特大单卖+大单卖)，3 只成分股合计。"""
        result = self._run()
        assert result["main_net"] == pytest.approx(EXPECTED_MAIN_NET, abs=0.01), (
            f"主力净流入应为 {EXPECTED_MAIN_NET} 亿，实际 {result['main_net']}"
        )

    def test_main_net_unit_converted(self):
        """显式防回归：必须 ÷10000 换算为亿，不能是未换算的万元值。"""
        result = self._run()
        assert result["main_net"] != pytest.approx(10000.0, rel=0.1), (
            "main_net 疑似未做 万元→亿 换算（÷10000）"
        )
        assert result["main_net"] == pytest.approx(EXPECTED_MAIN_NET, abs=0.01)

    def test_retail_net(self):
        """散户净流入（中单全 0，两种散户口径下都等于小单净额）= 1.2 亿。"""
        result = self._run()
        assert result["retail_net"] == pytest.approx(EXPECTED_RETAIL_NET, abs=0.01), (
            f"散户净流入应为 {EXPECTED_RETAIL_NET} 亿，实际 {result['retail_net']}"
        )

    def test_non_member_excluded(self):
        """非成分股 D 有 ~200 亿主力流入，若未剔除 main_net 会爆量。"""
        result = self._run()
        assert abs(result["main_net"]) < 10, (
            f"非成分股未被剔除，main_net 异常：{result['main_net']}"
        )

    def test_top_inflow_outflow_ordering(self):
        """top_inflow 第一是 A（+4亿主力净额），top_outflow 第一是 B（-5亿）。"""
        result = self._run()
        inflow_codes = _entry_codes(result["top_inflow"])
        outflow_codes = _entry_codes(result["top_outflow"])

        assert inflow_codes, "top_inflow 不应为空"
        assert outflow_codes, "top_outflow 不应为空"
        assert inflow_codes[0] == STOCK_A, (
            f"流入第一应为 {STOCK_A}，实际 {inflow_codes}"
        )
        assert outflow_codes[0] == STOCK_B, (
            f"流出第一应为 {STOCK_B}，实际 {outflow_codes}"
        )
        # B 是净流出，不应出现在流入榜；A 不应出现在流出榜
        assert STOCK_B not in inflow_codes
        assert STOCK_A not in outflow_codes
        # 非成分股 D 不得出现在任何榜单
        assert STOCK_D not in inflow_codes
        assert STOCK_D not in outflow_codes

    def test_stock_count(self):
        """stock_count 应等于有资金数据的成分股数量（3）。"""
        result = self._run()
        assert result["stock_count"] == 3


# ════════════════════════════════════════════════════════════════
# 3. 失败路径：安全结构 + note 非空，绝不抛异常
# ════════════════════════════════════════════════════════════════


VALUATION_KEYS = ("pe", "pb", "pe_percentile", "pb_percentile",
                  "sample_count", "note")
MONEYFLOW_KEYS = ("main_net", "retail_net", "top_inflow",
                  "top_outflow", "stock_count", "note")


class TestFailurePaths:
    """index_member 异常/空、_get_pro 为 None：都返回 note 非空的安全结构。"""

    def _assert_safe(self, result, keys, ctx):
        assert isinstance(result, dict), f"{ctx}: 应返回 dict，实际 {type(result)}"
        for key in keys:
            assert key in result, f"{ctx}: 安全结构缺少字段 {key}: {result}"
        assert isinstance(result["note"], str) and result["note"].strip(), (
            f"{ctx}: 失败时 note 必须非空说明原因，实际 {result.get('note')!r}"
        )

    def test_valuation_index_member_raises(self):
        pro = _make_pro()
        pro.index_member.side_effect = Exception("tushare down")
        pro.daily_basic.side_effect = _fake_daily_basic
        with patch.object(data_fetcher, "_get_pro", return_value=pro):
            result = fetch_sector_valuation(SECTOR, TRADE_DATE)  # 不得抛异常
        self._assert_safe(result, VALUATION_KEYS, "index_member 抛异常")

    def test_valuation_index_member_empty(self):
        pro = _make_pro()
        pro.index_member.return_value = pd.DataFrame(columns=["con_code"])
        pro.daily_basic.side_effect = _fake_daily_basic
        with patch.object(data_fetcher, "_get_pro", return_value=pro):
            result = fetch_sector_valuation(SECTOR, TRADE_DATE)
        self._assert_safe(result, VALUATION_KEYS, "index_member 返回空")

    def test_valuation_pro_unavailable(self):
        with patch.object(data_fetcher, "_get_pro", return_value=None):
            result = fetch_sector_valuation(SECTOR, TRADE_DATE)
        self._assert_safe(result, VALUATION_KEYS, "_get_pro 为 None")

    def test_moneyflow_index_member_raises(self):
        pro = _make_pro()
        pro.index_member.side_effect = Exception("tushare down")
        pro.moneyflow.side_effect = _fake_moneyflow
        with patch.object(data_fetcher, "_get_pro", return_value=pro):
            result = fetch_sector_moneyflow(SECTOR, TRADE_DATE)  # 不得抛异常
        self._assert_safe(result, MONEYFLOW_KEYS, "index_member 抛异常")

    def test_moneyflow_index_member_empty(self):
        pro = _make_pro()
        pro.index_member.return_value = pd.DataFrame(columns=["con_code"])
        pro.moneyflow.side_effect = _fake_moneyflow
        with patch.object(data_fetcher, "_get_pro", return_value=pro):
            result = fetch_sector_moneyflow(SECTOR, TRADE_DATE)
        self._assert_safe(result, MONEYFLOW_KEYS, "index_member 返回空")

    def test_moneyflow_pro_unavailable(self):
        with patch.object(data_fetcher, "_get_pro", return_value=None):
            result = fetch_sector_moneyflow(SECTOR, TRADE_DATE)
        self._assert_safe(result, MONEYFLOW_KEYS, "_get_pro 为 None")


# ════════════════════════════════════════════════════════════════
# 4. 板块深度分析 prompt 内容断言
# ════════════════════════════════════════════════════════════════


class TestSectorDeepDivePrompt:
    """板块深度分析 prompt：五维框架 + 数据真实性红线 + 禁用词清单。"""

    def _prompt(self) -> str:
        # 与 orchestrator._sector_deep_dive 实际取 prompt 的方式保持一致
        return get_system_prompt("sector_deep_dive", SECTOR)

    def test_five_dimension_framework(self):
        """五维度框架：趋势位置/估值水位/资金博弈/景气度/催化与风险+综合判断。"""
        prompt = self._prompt()
        for keyword in ("趋势", "估值", "资金", "景气"):
            assert keyword in prompt, (
                f"板块深度分析 prompt 缺少五维框架关键词「{keyword}」"
            )
        assert ("催化" in prompt) or ("综合判断" in prompt), (
            "板块深度分析 prompt 缺少「催化」或「综合判断」维度"
        )

    def test_data_authenticity_redline(self):
        """数据真实性红线：只用数据块中的数字 / 不得编造（意思相近即可）。"""
        prompt = self._prompt()
        assert any(w in prompt for w in ("编造", "虚构", "捏造")), (
            "prompt 缺少「不得编造/虚构」类数据真实性红线"
        )
        assert any(w in prompt for w in
                   ("数据块", "提供的数据", "上述数据", "给出的数据", "已有数据")), (
            "prompt 缺少「只使用所提供数据」类约束"
        )

    def test_banned_buzzwords(self):
        """禁用词清单：至少包含 护城河/飞轮/赋能 等 AI 味词汇中的 3 个。"""
        prompt = self._prompt()
        candidates = ["护城河", "飞轮", "赋能", "抓手", "闭环",
                      "颗粒度", "生态化反", "降维打击"]
        hits = [w for w in candidates if w in prompt]
        assert len(hits) >= 3, (
            f"prompt 的禁用词清单至少应包含 3 个禁用词，实际仅命中 {hits}"
        )


# ════════════════════════════════════════════════════════════════
# 5. orchestrator 接线：_sector_deep_dive 调用估值/资金函数并进 prompt
# ════════════════════════════════════════════════════════════════

FIXED_TRADE_DATE = datetime(2025, 7, 18)  # 周五，与 TRADE_DATE 一致

VALUATION_MOCK_RET = {
    "pe": 16.67, "pb": 2.33,
    "pe_percentile": 100.0, "pb_percentile": 96.0,
    "sample_count": 3, "note": "",
}
MONEYFLOW_MOCK_RET = {
    "main_net": 1.0, "retail_net": 1.2,
    "top_inflow": [{"ts_code": STOCK_A, "net": 4.0}],
    "top_outflow": [{"ts_code": STOCK_B, "net": -5.0}],
    "stock_count": 3, "note": "",
}


def _patch_target(name: str) -> str:
    """定位 patch 目标：orchestrator 顶层导入则 patch orchestrator 命名空间，
    否则（函数内局部 import）patch data_fetcher 命名空间。
    不用 create=True——两处都不存在就让 patch  loudly 失败，暴露接线缺失。
    """
    if hasattr(orchestrator, name):
        return f"agent.orchestrator.{name}"
    return f"agent.data_fetcher.{name}"


class TestOrchestratorWiring:
    """板块深挖必须把估值/资金/景气度数据接进最终 user prompt。"""

    def test_sector_deep_dive_calls_valuation_moneyflow_and_earnings(self):
        agent = orchestrator.MarketReviewAgent()
        agent.client = MagicMock()  # 双保险：不触达真实 DeepSeek
        agent._call_llm = AsyncMock(
            return_value={"role": "assistant", "content": "ok"}
        )

        val_mock = MagicMock(return_value=dict(VALUATION_MOCK_RET))
        mf_mock = MagicMock(return_value=dict(MONEYFLOW_MOCK_RET))
        earn_mock = MagicMock(return_value=dict(EARNINGS_MOCK_RET))

        with patch(
            "agent.orchestrator._get_latest_trade_date",
            return_value=FIXED_TRADE_DATE,
        ), patch(
            "agent.orchestrator.collect_market_snapshot",
            AsyncMock(return_value=MagicMock(name="snapshot")),
        ), patch(
            "agent.orchestrator.format_market_data_for_prompt",
            return_value="DATA",
        ), patch(
            _patch_target("fetch_sector_valuation"), val_mock
        ), patch(
            _patch_target("fetch_sector_moneyflow"), mf_mock
        ), patch(
            _patch_target("fetch_sector_earnings"), earn_mock
        ):
            asyncio.run(agent._sector_deep_dive(SECTOR, stream=False))

        # 三个数据函数都被调用，且都拿到了板块名
        assert val_mock.call_count >= 1, (
            "_sector_deep_dive 未调用 fetch_sector_valuation——接线缺失"
        )
        assert mf_mock.call_count >= 1, (
            "_sector_deep_dive 未调用 fetch_sector_moneyflow——接线缺失"
        )
        assert earn_mock.call_count >= 1, (
            "_sector_deep_dive 未调用 fetch_sector_earnings——接线缺失"
        )
        assert SECTOR in str(val_mock.call_args), (
            f"fetch_sector_valuation 调用参数应包含板块名：{val_mock.call_args}"
        )
        assert SECTOR in str(mf_mock.call_args), (
            f"fetch_sector_moneyflow 调用参数应包含板块名：{mf_mock.call_args}"
        )
        assert SECTOR in str(earn_mock.call_args), (
            f"fetch_sector_earnings 调用参数应包含板块名：{earn_mock.call_args}"
        )

        # 最终 user prompt 必须包含估值、资金、景气度段落标记
        assert agent._call_llm.await_count == 1
        user_prompt = agent._call_llm.await_args.args[1]
        assert "估值" in user_prompt, (
            f"user prompt 缺少估值段落标记：\n{user_prompt}"
        )
        assert "资金" in user_prompt, (
            f"user prompt 缺少资金段落标记：\n{user_prompt}"
        )
        assert "景气度" in user_prompt, (
            f"user prompt 缺少景气度段落标记（如【四、板块景气度】）：\n{user_prompt}"
        )

    def test_fetch_sector_extras_returns_three_items(self):
        """_fetch_sector_extras 必须从二元组升级为三元组（估值/资金/景气度）。"""
        agent = orchestrator.MarketReviewAgent()
        with patch(
            _patch_target("fetch_sector_valuation"),
            MagicMock(return_value=dict(VALUATION_MOCK_RET)),
        ), patch(
            _patch_target("fetch_sector_moneyflow"),
            MagicMock(return_value=dict(MONEYFLOW_MOCK_RET)),
        ), patch(
            _patch_target("fetch_sector_earnings"),
            MagicMock(return_value=dict(EARNINGS_MOCK_RET)),
        ):
            extras = asyncio.run(
                agent._fetch_sector_extras(SECTOR, TRADE_DATE)
            )
        assert isinstance(extras, tuple) and len(extras) == 3, (
            f"_fetch_sector_extras 应返回三元组（估值, 资金, 景气度），实际 {extras!r}"
        )


# ════════════════════════════════════════════════════════════════
# 6. fetch_sector_earnings：板块景气度（业绩预告 + 业绩快报聚合）
# ════════════════════════════════════════════════════════════════
#
# 背景契约（生产代码由并行代理实现）：
#   fetch_sector_earnings(sector_name, trade_date) ->
#   {'total_forecast','positive_count','negative_count','positive_ratio',
#    'median_change','top_improvers','top_decliners','express_count',
#    'period','note'}
# - pro.forecast 的 p_change_min/max 是百分数数值；type 为 预增/预减/扭亏/首亏 等；
# - 同一股票多次公告只保留 ann_date 最新一条（去重后再统计）；
# - 预喜 = 预增 + 扭亏（+ 减亏/续盈若出现），预忧 = 预减 + 首亏（+ 续亏 等）；
# - 任何失败返回 note 非空的安全结构，绝不抛异常。

EARNINGS_KEYS = ("total_forecast", "positive_count", "negative_count",
                 "positive_ratio", "median_change", "top_improvers",
                 "top_decliners", "express_count", "period", "note")

FORECAST_PERIOD = "20250630"  # 中报预告期

FORECAST_COLUMNS = ["ts_code", "ann_date", "end_date", "type",
                    "p_change_min", "p_change_max",
                    "net_profit_min", "net_profit_max"]

# 5 条预告（2 预增 1 扭亏 1 首亏 1 预减），覆盖 3 只成分股：
# - A 有两天公告（20250701 预增10% / 20250710 预增50~80%），
#   去重后只留最新的 20250710 一条 → 变动中值 65%；
# - C 有两天公告（20250702 预减 / 20250712 首亏-60~-40%），
#   去重后留最新的首亏 → 变动中值 -50%；
# - B 一条扭亏，p_change_min==max==30 → 无论按 min/max/中值口径都是 30。
# 去重后 3 条：A 预增（中值65）、B 扭亏（30）、C 首亏（-50）。
# 期望：total=3，预喜=2（预增+扭亏），预忧=1（首亏），预喜比例=2/3，
# median_change=30（三种口径下中位数都是 B 的 30）。
FORECAST_ROWS = [
    {"ts_code": STOCK_A, "ann_date": "20250701", "end_date": FORECAST_PERIOD,
     "type": "预增", "p_change_min": 10.0, "p_change_max": 10.0,
     "net_profit_min": 5000.0, "net_profit_max": 5000.0},
    {"ts_code": STOCK_A, "ann_date": "20250710", "end_date": FORECAST_PERIOD,
     "type": "预增", "p_change_min": 50.0, "p_change_max": 80.0,
     "net_profit_min": 10000.0, "net_profit_max": 15000.0},
    {"ts_code": STOCK_B, "ann_date": "20250711", "end_date": FORECAST_PERIOD,
     "type": "扭亏", "p_change_min": 30.0, "p_change_max": 30.0,
     "net_profit_min": 8000.0, "net_profit_max": 8000.0},
    {"ts_code": STOCK_C, "ann_date": "20250702", "end_date": FORECAST_PERIOD,
     "type": "预减", "p_change_min": -20.0, "p_change_max": -10.0,
     "net_profit_min": 3000.0, "net_profit_max": 4000.0},
    {"ts_code": STOCK_C, "ann_date": "20250712", "end_date": FORECAST_PERIOD,
     "type": "首亏", "p_change_min": -60.0, "p_change_max": -40.0,
     "net_profit_min": -6000.0, "net_profit_max": -4000.0},
]

# 2 条业绩快报（A、B）→ express_count 期望为 2
EXPRESS_ROWS = [
    {"ts_code": STOCK_A, "ann_date": "20250715", "end_date": FORECAST_PERIOD,
     "revenue": 100000.0, "n_income": 12000.0},
    {"ts_code": STOCK_B, "ann_date": "20250716", "end_date": FORECAST_PERIOD,
     "revenue": 80000.0, "n_income": 9000.0},
]

EXPECTED_POSITIVE_RATIO = 2 / 3   # 预喜 2 / 去重后总数 3
EXPECTED_MEDIAN_CHANGE = 30.0     # 去重后 [65, 30, -50] 的中位数（min/max/中值口径一致）


def _fake_forecast(**kwargs):
    """模拟 pro.forecast：兼容按 ann_date 单日 / start-end 区间 / ts_code 过滤。

    生产实现按周采样 ann_date（120 天回溯，每 7 天一次），单日查询可能落在
    两条公告之间——因此 ann_date 单日查询按「该日起往前 6 天」的周窗口匹配，
    避免测试与具体采样偏移耦合。生产端按 ann_date 去重留最新，不受影响。
    """
    df = pd.DataFrame(FORECAST_ROWS, columns=FORECAST_COLUMNS)
    ann_date = kwargs.get("ann_date")
    if ann_date:
        target = datetime.strptime(str(ann_date), "%Y%m%d")

        def _in_week_window(a) -> bool:
            try:
                delta = target - datetime.strptime(str(a), "%Y%m%d")
            except ValueError:
                return False
            return timedelta(0) <= delta <= timedelta(days=6)

        df = df[df["ann_date"].map(_in_week_window)]
    start_date = kwargs.get("start_date")
    if start_date:
        df = df[df["ann_date"] >= str(start_date)]
    end_date = kwargs.get("end_date")
    if end_date:
        df = df[df["ann_date"] <= str(end_date)]
    period = kwargs.get("period")
    if period:
        df = df[df["end_date"] == str(period)]
    ts_code = kwargs.get("ts_code")
    if ts_code:
        codes = set(str(ts_code).split(","))
        df = df[df["ts_code"].isin(codes)]
    return df.reset_index(drop=True)


def _fake_express(**kwargs):
    """模拟 pro.express：返回 A、B 两只成分股的快报，可按 ts_code 过滤。"""
    df = pd.DataFrame(EXPRESS_ROWS)
    ts_code = kwargs.get("ts_code")
    if ts_code:
        codes = set(str(ts_code).split(","))
        df = df[df["ts_code"].isin(codes)]
    return df.reset_index(drop=True)


def _empty_forecast_df():
    return pd.DataFrame(columns=FORECAST_COLUMNS)


class TestFetchSectorEarnings:
    """业绩预告聚合：去重、预喜/预忧统计、中位数、改善/恶化榜。"""

    def _run(self, pro=None):
        if pro is None:
            pro = MagicMock(name="tushare_pro")
            pro.forecast.side_effect = _fake_forecast
            pro.express.side_effect = _fake_express
        # _stock_name 置为恒等：榜单条目里的名称即 ts_code，与名称缓存状态无关；
        # time.sleep 打桩：跳过生产端防限流等待，加速按周采样循环。
        with patch.object(data_fetcher, "_get_pro", return_value=pro), \
             patch.object(data_fetcher, "_get_sector_member_codes",
                          return_value=list(MEMBERS)), \
             patch.object(data_fetcher, "_stock_name",
                          side_effect=lambda c: c), \
             patch("time.sleep"):
            return fetch_sector_earnings(SECTOR, TRADE_DATE)

    def test_contract_keys(self):
        result = self._run()
        for key in EARNINGS_KEYS:
            assert key in result, f"返回结构缺少契约字段 {key}: {result}"

    def test_dedup_keeps_latest_announcement(self):
        """同一股票两天公告只保留最新一条：total_forecast=3 而非 5。"""
        result = self._run()
        assert result["total_forecast"] == 3, (
            f"5 条预告去重后应为 3（每只股票留最新一条），实际 {result['total_forecast']}"
        )
        # 若错误地保留了 A 的旧公告（预增10%），A 的中值会变 10，
        # top_improvers 第一就不再是 A —— 由排序用例兜底验证。
        assert result["positive_count"] == 2, (
            f"预喜（预增+扭亏）应为 2，实际 {result['positive_count']}"
        )
        assert result["negative_count"] == 1, (
            f"预忧（首亏）应为 1，实际 {result['negative_count']}"
        )

    def test_positive_ratio(self):
        """预喜比例 = 2/3 ≈ 66.7%（兼容小数 0-1 或百分数 0-100 两种表示）。"""
        result = self._run()
        ratio = result["positive_ratio"]
        ok = (ratio == pytest.approx(EXPECTED_POSITIVE_RATIO, abs=0.01)) or \
             (ratio == pytest.approx(EXPECTED_POSITIVE_RATIO * 100, abs=1.0))
        assert ok, (
            f"预喜比例应为 2/3≈0.667（或 66.7%），实际 {ratio!r}"
        )

    def test_median_change(self):
        """变动中位数 = 30（B 的扭亏预告，min/max/中值三种口径下都是 30）。"""
        result = self._run()
        assert result["median_change"] == pytest.approx(
            EXPECTED_MEDIAN_CHANGE, abs=0.5), (
            f"median_change 应为 {EXPECTED_MEDIAN_CHANGE}，实际 {result['median_change']}"
        )

    def test_top_improvers_decliners_ordering(self):
        """改善榜第一 = A（预增中值65%），恶化榜第一 = C（首亏中值-50%）。"""
        result = self._run()
        improver_codes = _entry_codes(result["top_improvers"])
        decliner_codes = _entry_codes(result["top_decliners"])

        assert improver_codes, "top_improvers 不应为空"
        assert decliner_codes, "top_decliners 不应为空"
        assert improver_codes[0] == STOCK_A, (
            f"改善榜第一应为 {STOCK_A}（最新预增50~80%），实际 {improver_codes}"
        )
        assert decliner_codes[0] == STOCK_C, (
            f"恶化榜第一应为 {STOCK_C}（最新首亏-60~-40%），实际 {decliner_codes}"
        )
        # 首亏的 C 不应出现在改善榜；预增的 A 不应出现在恶化榜
        assert STOCK_C not in improver_codes
        assert STOCK_A not in decliner_codes

    def test_express_count(self):
        """业绩快报（A、B 两条）也应计入统计。"""
        result = self._run()
        assert result["express_count"] == 2, (
            f"快报条数应为 2，实际 {result['express_count']}"
        )

    def test_disclosure_vacuum_no_exception(self):
        """披露真空期：forecast/express 都为空 → total=0、note 非空、不抛异常。"""
        pro = MagicMock(name="tushare_pro")
        pro.forecast.return_value = _empty_forecast_df()
        pro.express.return_value = pd.DataFrame(
            columns=["ts_code", "ann_date", "end_date"])
        result = self._run(pro)  # 不得抛异常
        assert result["total_forecast"] == 0, (
            f"真空期 total_forecast 应为 0，实际 {result['total_forecast']}"
        )
        assert isinstance(result["note"], str) and result["note"].strip(), (
            f"真空期 note 必须非空说明原因，实际 {result.get('note')!r}"
        )

    def test_member_fetch_failure_safe_structure(self):
        """成分股获取失败（index_member 抛异常）→ 返回安全结构，不向上抛。

        与估值/资金的 TestFailurePaths 同一约定：_get_sector_member_codes
        内部 catch 异常返回 None，生产函数据此返回 note 非空的安全结构。
        """
        pro = MagicMock(name="tushare_pro")
        pro.index_member.side_effect = Exception("member api down")
        pro.forecast.side_effect = _fake_forecast
        pro.express.side_effect = _fake_express
        with patch.object(data_fetcher, "_get_pro", return_value=pro), \
             patch("time.sleep"):
            result = fetch_sector_earnings(SECTOR, TRADE_DATE)  # 不得抛异常
        assert isinstance(result, dict), f"应返回 dict，实际 {type(result)}"
        for key in EARNINGS_KEYS:
            assert key in result, f"安全结构缺少契约字段 {key}: {result}"
        assert isinstance(result["note"], str) and result["note"].strip(), (
            f"失败时 note 必须非空说明原因，实际 {result.get('note')!r}"
        )


# ── orchestrator 景气度 mock 返回值（接线测试用）──

EARNINGS_MOCK_RET = {
    "total_forecast": 3, "positive_count": 2, "negative_count": 1,
    "positive_ratio": 0.667, "median_change": 30.0,
    "top_improvers": [{"ts_code": STOCK_A, "change": 65.0}],
    "top_decliners": [{"ts_code": STOCK_C, "change": -50.0}],
    "express_count": 2, "period": FORECAST_PERIOD, "note": "",
}


# ════════════════════════════════════════════════════════════════
# 7. 板块深度分析 prompt：景气度一节的内容要求
# ════════════════════════════════════════════════════════════════


class TestSectorDeepDiveEarningsPrompt:
    """SECTOR_DEEP_DIVE_PROMPT 的景气度一节：预喜比例解读指引 + 样本局限提示。"""

    def _prompt(self) -> str:
        return get_system_prompt("sector_deep_dive", SECTOR)

    def test_positive_ratio_threshold_guidance(self):
        """景气度一节必须给出预喜比例的高低阈值解读指引（>60% / <40% 或相近表述）。"""
        prompt = self._prompt()
        assert "预喜" in prompt, (
            "prompt 景气度一节缺少「预喜」概念（预喜比例解读指引缺失）"
        )
        has_high = any(w in prompt for w in ("60%", "六成", "60 %"))
        has_low = any(w in prompt for w in ("40%", "四成", "40 %"))
        assert has_high and has_low, (
            "prompt 景气度一节缺少预喜比例阈值指引（如 >60% 景气向好 / <40% 景气承压）"
        )

    def test_sample_limitation_hint(self):
        """景气度一节必须提示样本局限（披露不全/样本有限/不代表全板块等）。"""
        prompt = self._prompt()
        assert any(w in prompt for w in
                   ("样本", "披露率", "代表性", "不代表", "局限")), (
            "prompt 景气度一节缺少样本局限提示（如预告披露不全、样本不代表全板块）"
        )
