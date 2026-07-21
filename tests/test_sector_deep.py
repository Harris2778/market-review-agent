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
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

import agent.data_fetcher as data_fetcher
import agent.orchestrator as orchestrator
from agent.data_fetcher import fetch_sector_moneyflow, fetch_sector_valuation
from agent.system_prompts import get_system_prompt

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
    """板块深挖必须把估值/资金数据接进最终 user prompt。"""

    def test_sector_deep_dive_calls_valuation_and_moneyflow(self):
        agent = orchestrator.MarketReviewAgent()
        agent.client = MagicMock()  # 双保险：不触达真实 DeepSeek
        agent._call_llm = AsyncMock(
            return_value={"role": "assistant", "content": "ok"}
        )

        val_mock = MagicMock(return_value=dict(VALUATION_MOCK_RET))
        mf_mock = MagicMock(return_value=dict(MONEYFLOW_MOCK_RET))

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
        ):
            asyncio.run(agent._sector_deep_dive(SECTOR, stream=False))

        # 两个数据函数都被调用，且都拿到了板块名
        assert val_mock.call_count >= 1, (
            "_sector_deep_dive 未调用 fetch_sector_valuation——接线缺失"
        )
        assert mf_mock.call_count >= 1, (
            "_sector_deep_dive 未调用 fetch_sector_moneyflow——接线缺失"
        )
        assert SECTOR in str(val_mock.call_args), (
            f"fetch_sector_valuation 调用参数应包含板块名：{val_mock.call_args}"
        )
        assert SECTOR in str(mf_mock.call_args), (
            f"fetch_sector_moneyflow 调用参数应包含板块名：{mf_mock.call_args}"
        )

        # 最终 user prompt 必须包含估值与资金段落标记
        assert agent._call_llm.await_count == 1
        user_prompt = agent._call_llm.await_args.args[1]
        assert "估值" in user_prompt, (
            f"user prompt 缺少估值段落标记：\n{user_prompt}"
        )
        assert "资金" in user_prompt, (
            f"user prompt 缺少资金段落标记：\n{user_prompt}"
        )
