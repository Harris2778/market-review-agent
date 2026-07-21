"""板块 extras 当日缓存 + 景气度智能采样测试。

覆盖范围：
1. 进程内当日缓存（_sector_extras_cached）：
   - 缓存命中后第二次调用不再触发 pro.* 调用（mock 计数不变）；
   - 缓存 key 含函数名 + 板块名 + 交易日（换日期/换板块重新调）；
   - 降级 dict（note 说明原因）也允许缓存；底层抛异常不缓存；
   - 线程安全：并发同 key 只计算一次。
2. 景气度智能采样（fetch_sector_earnings）：
   - 优先按报告期直查 pro.forecast(period=YYYYMMDD)，1~2 次调用；
   - 最近期间为空时尝试次近期间；period 抛异常/全部为空时降级按周采样；
   - 聚合语义与按周采样一致；note 注明采样方式；披露真空期语义保留。
3. 三个 extras 函数返回字段契约不变。

规则（与项目其他测试一致）：所有外部调用全部 mock，绝不发起真实网络请求；
time.sleep 打桩加速；每个用例前后清空进程级缓存，避免跨用例污染。
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import agent.data_fetcher as data_fetcher
from agent.data_fetcher import (
    fetch_sector_earnings,
    fetch_sector_moneyflow,
    fetch_sector_valuation,
)

# ── 测试数据常量 ──

SECTOR = "煤炭"
OTHER_SECTOR = "银行"
IDX_CODE = data_fetcher.SW_INDEX_MAP[SECTOR]  # 801950.SI
TRADE_DATE = "20250718"  # 周五
OTHER_DATE = "20250717"

STOCK_A = "601001.SH"
STOCK_B = "601002.SH"
STOCK_C = "601003.SH"
MEMBERS = [STOCK_A, STOCK_B, STOCK_C]

FORECAST_PERIOD = "20250630"  # 最近已过季度末（中报）
PREV_PERIOD = "20250331"      # 次近季度末（一季报）

VALUATION_KEYS = {"pe", "pb", "pe_percentile", "pb_percentile",
                  "sample_count", "note"}
MONEYFLOW_KEYS = {"main_net", "retail_net", "top_inflow",
                  "top_outflow", "stock_count", "note"}
EARNINGS_KEYS = {"total_forecast", "positive_count", "negative_count",
                 "positive_ratio", "median_change", "top_improvers",
                 "top_decliners", "express_count", "period", "note"}

FORECAST_COLUMNS = ["ts_code", "ann_date", "end_date", "type",
                    "p_change_min", "p_change_max",
                    "net_profit_min", "net_profit_max"]

# 与 test_sector_deep 同款的 5 条预告：去重后 3 条
# A 预增（中值65）、B 扭亏（30）、C 首亏（-50）→ 预喜2 预忧1 中位数30
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

# 次近期间（一季报）的 2 条预告：仅用于「最近期间为空→尝试次近期间」用例
PREV_FORECAST_ROWS = [
    {"ts_code": STOCK_A, "ann_date": "20250410", "end_date": PREV_PERIOD,
     "type": "预增", "p_change_min": 20.0, "p_change_max": 40.0,
     "net_profit_min": 6000.0, "net_profit_max": 7000.0},
    {"ts_code": STOCK_B, "ann_date": "20250415", "end_date": PREV_PERIOD,
     "type": "预减", "p_change_min": -30.0, "p_change_max": -10.0,
     "net_profit_min": 2000.0, "net_profit_max": 3000.0},
]

EXPRESS_ROWS = [
    {"ts_code": STOCK_A, "ann_date": "20250715", "end_date": FORECAST_PERIOD,
     "revenue": 100000.0, "n_income": 12000.0},
    {"ts_code": STOCK_B, "ann_date": "20250716", "end_date": FORECAST_PERIOD,
     "revenue": 80000.0, "n_income": 9000.0},
]


# ── 共享 fixture：清空进程级缓存 + 打桩 sleep ──


@pytest.fixture(autouse=True)
def _clear_sector_extras_cache():
    data_fetcher._sector_extras_cache.clear()
    data_fetcher._sector_extras_key_locks.clear()
    yield
    data_fetcher._sector_extras_cache.clear()
    data_fetcher._sector_extras_key_locks.clear()


@pytest.fixture(autouse=True)
def _no_sleep():
    with patch("time.sleep"):
        yield


# ── mock 构造 ──


def _fake_forecast(**kwargs):
    """模拟 pro.forecast：period 精确匹配 end_date；ann_date 按周窗口匹配。"""
    df = pd.DataFrame(FORECAST_ROWS, columns=FORECAST_COLUMNS)
    period = kwargs.get("period")
    if period:
        df = df[df["end_date"] == str(period)]
        return df.reset_index(drop=True)
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
    return df.reset_index(drop=True)


def _fake_express(**kwargs):
    return pd.DataFrame(EXPRESS_ROWS)


def _fake_daily_basic(**kwargs):
    rows = [
        {"ts_code": STOCK_A, "pe": 10.0, "pb": 1.0, "total_mv": 10000.0},
        {"ts_code": STOCK_B, "pe": 20.0, "pb": 2.0, "total_mv": 20000.0},
        {"ts_code": STOCK_C, "pe": 15.0, "pb": 1.5, "total_mv": 30000.0},
    ]
    return pd.DataFrame(rows)


def _fake_moneyflow(**kwargs):
    rows = [
        {"ts_code": STOCK_A,
         "buy_elg_amount": 50000.0, "sell_elg_amount": 30000.0,
         "buy_lg_amount": 40000.0, "sell_lg_amount": 20000.0,
         "buy_md_amount": 0.0, "sell_md_amount": 0.0,
         "buy_sm_amount": 10000.0, "sell_sm_amount": 15000.0},
        {"ts_code": STOCK_B,
         "buy_elg_amount": 10000.0, "sell_elg_amount": 40000.0,
         "buy_lg_amount": 10000.0, "sell_lg_amount": 30000.0,
         "buy_md_amount": 0.0, "sell_md_amount": 0.0,
         "buy_sm_amount": 20000.0, "sell_sm_amount": 5000.0},
        {"ts_code": STOCK_C,
         "buy_elg_amount": 20000.0, "sell_elg_amount": 10000.0,
         "buy_lg_amount": 20000.0, "sell_lg_amount": 10000.0,
         "buy_md_amount": 0.0, "sell_md_amount": 0.0,
         "buy_sm_amount": 2000.0, "sell_sm_amount": 0.0},
    ]
    return pd.DataFrame(rows)


def _make_pro():
    """构造 mock 的 tushare pro 对象。"""
    pro = MagicMock(name="tushare_pro")
    pro.index_member.return_value = pd.DataFrame({
        "index_code": [IDX_CODE] * len(MEMBERS),
        "con_code": list(MEMBERS),
    })
    pro.forecast.side_effect = _fake_forecast
    pro.express.side_effect = _fake_express
    pro.daily_basic.side_effect = _fake_daily_basic
    pro.moneyflow.side_effect = _fake_moneyflow
    return pro


def _run(fn, pro, sector=SECTOR, date=TRADE_DATE):
    with patch.object(data_fetcher, "_get_pro", return_value=pro):
        return fn(sector, date)


def _run_earnings(pro, sector=SECTOR, date=TRADE_DATE):
    """景气度专用：_stock_name 置恒等，榜单条目名称即 ts_code。"""
    with patch.object(data_fetcher, "_get_pro", return_value=pro), \
         patch.object(data_fetcher, "_stock_name", side_effect=lambda c: c):
        return fetch_sector_earnings(sector, date)


# ════════════════════════════════════════════════════════════════
# 1. 进程内当日缓存
# ════════════════════════════════════════════════════════════════


class TestSameDayCache:
    """命中缓存后第二次调用不再触发任何 pro.* 调用。"""

    def test_valuation_cache_hit_no_repeat_api_calls(self):
        pro = _make_pro()
        r1 = _run(fetch_sector_valuation, pro)
        assert r1["pe"] is not None and r1["sample_count"] == 3
        member_calls = pro.index_member.call_count
        basic_calls = pro.daily_basic.call_count
        assert member_calls >= 1 and basic_calls >= 1

        r2 = _run(fetch_sector_valuation, pro)
        assert pro.index_member.call_count == member_calls, (
            "缓存命中后 index_member 不应再次调用")
        assert pro.daily_basic.call_count == basic_calls, (
            "缓存命中后 daily_basic 不应再次调用")
        assert r2 == r1

    def test_moneyflow_cache_hit_no_repeat_api_calls(self):
        pro = _make_pro()
        r1 = _run(fetch_sector_moneyflow, pro)
        assert r1["main_net"] is not None and r1["stock_count"] == 3
        member_calls = pro.index_member.call_count
        flow_calls = pro.moneyflow.call_count
        assert member_calls >= 1 and flow_calls >= 1

        r2 = _run(fetch_sector_moneyflow, pro)
        assert pro.index_member.call_count == member_calls
        assert pro.moneyflow.call_count == flow_calls, (
            "缓存命中后 moneyflow 不应再次调用")
        assert r2 == r1

    def test_earnings_cache_hit_no_repeat_api_calls(self):
        pro = _make_pro()
        r1 = _run_earnings(pro)
        assert r1["total_forecast"] == 3
        forecast_calls = pro.forecast.call_count
        express_calls = pro.express.call_count
        assert forecast_calls >= 1 and express_calls >= 1

        r2 = _run_earnings(pro)
        assert pro.forecast.call_count == forecast_calls, (
            "缓存命中后 forecast 不应再次调用")
        assert pro.express.call_count == express_calls, (
            "缓存命中后 express 不应再次调用")
        assert r2 == r1

    def test_cache_key_contains_trade_date(self):
        """换交易日 → 缓存不命中，重新调用 pro.*。"""
        pro = _make_pro()
        _run(fetch_sector_valuation, pro, date=TRADE_DATE)
        calls_after_first = pro.daily_basic.call_count
        _run(fetch_sector_valuation, pro, date=OTHER_DATE)
        assert pro.daily_basic.call_count > calls_after_first, (
            "换日期后缓存应失效，daily_basic 应再次调用")

        pro2 = _make_pro()
        _run_earnings(pro2, date=TRADE_DATE)
        calls_after_first = pro2.forecast.call_count
        _run_earnings(pro2, date=OTHER_DATE)
        assert pro2.forecast.call_count > calls_after_first, (
            "换日期后缓存应失效，forecast 应再次调用")

    def test_cache_key_contains_sector_name(self):
        """换板块 → 缓存不命中，重新调用 pro.*。"""
        pro = _make_pro()
        _run(fetch_sector_moneyflow, pro, sector=SECTOR)
        calls_after_first = pro.moneyflow.call_count
        _run(fetch_sector_moneyflow, pro, sector=OTHER_SECTOR)
        assert pro.moneyflow.call_count > calls_after_first, (
            "换板块后缓存应失效，moneyflow 应再次调用")

    def test_cache_key_contains_func_name(self):
        """同板块同日期，不同 extras 函数互不共享缓存。"""
        pro = _make_pro()
        _run(fetch_sector_valuation, pro)
        flow_calls_before = pro.moneyflow.call_count
        r = _run(fetch_sector_moneyflow, pro)
        assert pro.moneyflow.call_count > flow_calls_before, (
            "估值的缓存不应被资金流函数复用")
        assert r["main_net"] is not None

    def test_degraded_result_also_cached(self):
        """降级 dict（note 说明原因）同样允许缓存：当日数据不会变。"""
        pro = _make_pro()
        pro.index_member.side_effect = Exception("tushare down")
        r1 = _run(fetch_sector_valuation, pro)
        assert r1["pe"] is None and r1["note"].strip()
        assert pro.index_member.call_count == 1

        r2 = _run(fetch_sector_valuation, pro)
        assert pro.index_member.call_count == 1, (
            "降级结果命中缓存后不应重试 index_member")
        assert r2 == r1

    def test_exception_not_cached(self):
        """底层抛异常的调用不应被缓存：恢复后重调应真正执行。"""
        pro = _make_pro()
        with patch.object(data_fetcher, "_get_pro", return_value=pro), \
             patch.object(data_fetcher, "_get_sector_member_codes",
                          side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                fetch_sector_valuation(SECTOR, TRADE_DATE)
        key = ("fetch_sector_valuation", SECTOR, TRADE_DATE)
        assert key not in data_fetcher._sector_extras_cache, (
            "抛异常的调用不应写入缓存")

        r = _run(fetch_sector_valuation, pro)
        assert r["pe"] is not None
        assert pro.index_member.call_count == 1, (
            "异常未缓存，恢复后应真正执行一次完整采集")

    def test_thread_safety_concurrent_same_key_single_compute(self):
        """16 线程并发同 key 调用：只计算一次，全部拿到一致结果。"""
        pro = _make_pro()
        with patch.object(data_fetcher, "_get_pro", return_value=pro), \
             patch.object(data_fetcher, "_stock_name",
                          side_effect=lambda c: c):
            with ThreadPoolExecutor(max_workers=8) as ex:
                results = list(ex.map(
                    lambda _: fetch_sector_earnings(SECTOR, TRADE_DATE),
                    range(16)))
        assert pro.forecast.call_count == 1, (
            f"并发同 key 应只直查一次 forecast，实际 {pro.forecast.call_count} 次")
        assert pro.express.call_count == 1, (
            f"并发同 key 应只查一次 express，实际 {pro.express.call_count} 次")
        assert all(r == results[0] for r in results)
        assert results[0]["total_forecast"] == 3


# ════════════════════════════════════════════════════════════════
# 2. 景气度智能采样：报告期直查优先 + 按周采样降级
# ════════════════════════════════════════════════════════════════


class TestEarningsPeriodSampling:
    """优先 pro.forecast(period=)；失败/空时降级按周 ann_date 采样。"""

    def test_period_direct_query_preferred(self):
        """直查命中：第一次 forecast 调用即带 period，且不产生 ann_date 调用。"""
        pro = _make_pro()
        r = _run_earnings(pro)
        first = pro.forecast.call_args_list[0]
        assert first.kwargs.get("period") == FORECAST_PERIOD, (
            f"首次 forecast 调用应按最近报告期直查，实际 {first}")
        ann_calls = [c for c in pro.forecast.call_args_list
                     if c.kwargs.get("ann_date")]
        assert not ann_calls, "直查命中后不应再按周采样 ann_date"
        assert pro.forecast.call_count == 1, (
            f"直查命中应 1 次调用覆盖全量预告，实际 {pro.forecast.call_count} 次")
        assert "直查" in r["note"], f"note 应注明报告期直查：{r['note']!r}"

    def test_period_direct_aggregation_semantics(self):
        """直查方式下聚合语义不变：去重留最新、预喜预忧、中位数、榜单。"""
        pro = _make_pro()
        r = _run_earnings(pro)
        assert r["total_forecast"] == 3, (
            f"5 条预告去重后应为 3，实际 {r['total_forecast']}")
        assert r["positive_count"] == 2
        assert r["negative_count"] == 1
        assert r["positive_ratio"] == pytest.approx(2 / 3 * 100, abs=0.5)
        assert r["median_change"] == pytest.approx(30.0, abs=0.5)
        # 去重留最新：A 保留 0710 的预增50~80%（中值65）→ 改善榜第一
        assert r["top_improvers"][0] == (STOCK_A, "预增", 65.0)
        assert r["top_decliners"][0] == (STOCK_C, "首亏", -50.0)
        assert r["period"] == "2025中报"
        assert r["express_count"] == 2

    def test_recent_period_empty_tries_older_period(self):
        """最近期间为空 → 尝试次近期间（第 2 次调用）并命中。"""
        def _fake_older_only(**kwargs):
            period = kwargs.get("period")
            if period == FORECAST_PERIOD:
                return pd.DataFrame(columns=FORECAST_COLUMNS)
            if period == PREV_PERIOD:
                return pd.DataFrame(PREV_FORECAST_ROWS,
                                    columns=FORECAST_COLUMNS)
            return pd.DataFrame(columns=FORECAST_COLUMNS)

        pro = _make_pro()
        pro.forecast.side_effect = _fake_older_only
        r = _run_earnings(pro)
        period_calls = [c.kwargs.get("period")
                        for c in pro.forecast.call_args_list
                        if c.kwargs.get("period")]
        assert period_calls == [FORECAST_PERIOD, PREV_PERIOD], (
            f"应先查最近期间、空则查次近期间，实际 {period_calls}")
        assert r["total_forecast"] == 2
        assert r["period"] == "2025一季报"
        assert "直查" in r["note"]

    def test_period_raises_falls_back_to_weekly(self):
        """period 直查抛异常 → 降级按周采样，聚合结果与直查一致。"""
        def _forecast_period_raises(**kwargs):
            if kwargs.get("period"):
                raise RuntimeError("period param unsupported")
            return _fake_forecast(**kwargs)

        pro = _make_pro()
        pro.forecast.side_effect = _forecast_period_raises
        r = _run_earnings(pro)
        ann_calls = [c for c in pro.forecast.call_args_list
                     if c.kwargs.get("ann_date")]
        assert len(ann_calls) == 17, (
            f"降级后应按周采样 17 次，实际 {len(ann_calls)} 次")
        assert "按周采样" in r["note"] and "降级" in r["note"], (
            f"note 应注明按周采样降级：{r['note']!r}")
        assert r["total_forecast"] == 3
        assert r["positive_count"] == 2 and r["negative_count"] == 1
        assert r["median_change"] == pytest.approx(30.0, abs=0.5)

    def test_period_all_empty_falls_back_to_weekly(self):
        """全部期间返回空 → 降级按周采样。"""
        def _forecast_period_empty(**kwargs):
            if kwargs.get("period"):
                return pd.DataFrame(columns=FORECAST_COLUMNS)
            return _fake_forecast(**kwargs)

        pro = _make_pro()
        pro.forecast.side_effect = _forecast_period_empty
        r = _run_earnings(pro)
        period_calls = [c for c in pro.forecast.call_args_list
                        if c.kwargs.get("period")]
        assert len(period_calls) == 2, (
            f"两个期间都为空才降级，period 调用应 2 次，实际 {len(period_calls)}")
        ann_calls = [c for c in pro.forecast.call_args_list
                     if c.kwargs.get("ann_date")]
        assert len(ann_calls) == 17
        assert r["total_forecast"] == 3
        assert "按周采样" in r["note"] and "降级" in r["note"]

    def test_weekly_fallback_counts_api_failures(self):
        """降级后按周采样全部失败 → note 保留失败计数语义。"""
        pro = _make_pro()
        pro.forecast.side_effect = Exception("api down")
        pro.express.side_effect = Exception("api down")
        r = _run_earnings(pro)  # 不得抛异常
        assert r["total_forecast"] == 0
        assert "17 次调用失败" in r["note"], (
            f"按周采样 17 次全败应计入 note：{r['note']!r}")

    def test_vacuum_semantics_preserved(self):
        """直查与降级均无数据 → 保留「近120天无预告→披露真空期」语义。"""
        pro = _make_pro()
        pro.forecast.side_effect = None
        pro.forecast.return_value = pd.DataFrame(columns=FORECAST_COLUMNS)
        pro.express.side_effect = None
        pro.express.return_value = pd.DataFrame(
            columns=["ts_code", "ann_date", "end_date"])
        r = _run_earnings(pro)  # 不得抛异常
        assert r["total_forecast"] == 0
        assert "真空期" in r["note"], (
            f"披露真空期语义应保留：{r['note']!r}")
        assert "按周采样" in r["note"], (
            f"真空期经历了降级按周采样，note 应注明：{r['note']!r}")


# ════════════════════════════════════════════════════════════════
# 3. 返回字段契约不变
# ════════════════════════════════════════════════════════════════


class TestContractUnchanged:
    """缓存 + 采样优化后，三个函数的返回字段契约与原先完全一致。"""

    def test_valuation_contract_keys(self):
        r = _run(fetch_sector_valuation, _make_pro())
        assert set(r.keys()) == VALUATION_KEYS

    def test_moneyflow_contract_keys(self):
        r = _run(fetch_sector_moneyflow, _make_pro())
        assert set(r.keys()) == MONEYFLOW_KEYS

    def test_earnings_contract_keys_period_mode(self):
        r = _run_earnings(_make_pro())
        assert set(r.keys()) == EARNINGS_KEYS

    def test_earnings_contract_keys_weekly_fallback(self):
        def _forecast_period_raises(**kwargs):
            if kwargs.get("period"):
                raise RuntimeError("period param unsupported")
            return _fake_forecast(**kwargs)

        pro = _make_pro()
        pro.forecast.side_effect = _forecast_period_raises
        r = _run_earnings(pro)
        assert set(r.keys()) == EARNINGS_KEYS

    def test_earnings_contract_keys_failure_path(self):
        """失败路径（pro 不可用）字段契约同样不变。"""
        with patch.object(data_fetcher, "_get_pro", return_value=None):
            r = fetch_sector_earnings(SECTOR, TRADE_DATE)
        assert set(r.keys()) == EARNINGS_KEYS
        assert r["note"].strip()
