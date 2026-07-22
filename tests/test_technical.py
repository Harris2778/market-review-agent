"""
确定性技术分析层测试（tests/test_technical.py）。

覆盖 agent/technical.py 全部核心逻辑：
- MA / MACD / RSI / 乖离率：手工构造可验算的小序列，数值断言容差 1e-6；
- 均线排列七态全覆盖 + ma60 缺失降级 + ma20 缺失「未知」；
- 量能五态 + 边界 ratio + 缺量 missing；
- 支撑/压力摆动点 + 回退路径 + 字段缺失 None；
- 综合评分：已知结构精确分、权重覆盖、0-100 边界、score_breakdown 结构；
- verdict_from_score：评分带边界（79/80/81 等）、护栏降级矩阵、config 覆盖；
- is_trade_day：calendar 注入 / 周中启发式 / holidays 扣除三路径；
- 输入容错：数据不足、脏行、非法类型，绝不抛异常。

全 mock 零网络（模块本身零 I/O）。
运行：/usr/local/bin/python3 -m pytest tests/test_technical.py -v
"""

import sys
from datetime import date
from pathlib import Path

import pytest

# 保证无论 conftest.py 是否就绪，都能从项目根导入 agent 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import technical  # noqa: E402
from agent.technical import (  # noqa: E402
    DATA_QUALITY_CAPS,
    DEFAULT_BANDS,
    DEFAULT_SCORE_WEIGHTS,
    compute_indicators,
    is_trade_day,
    verdict_from_score,
)

TOL = 1e-6


# ─────────────────────────────────────────────
# 测试数据构造辅助
# ─────────────────────────────────────────────

def make_rows(closes, vols=None, highs=None, lows=None, start="2024-01-01"):
    """按收盘价列表构造升序日线（date 用递增序号占位，指标计算不依赖真实日历）。"""
    rows = []
    n = len(closes)
    for i, c in enumerate(closes):
        row = {
            "date": f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
            "open": c, "high": c if highs is None else highs[i],
            "low": c if lows is None else lows[i], "close": c,
        }
        if vols is not None:
            row["vol"] = vols[i]
        rows.append(row)
    assert n >= 1
    return rows


# ─────────────────────────────────────────────
# MA
# ─────────────────────────────────────────────

class TestMA:
    def test_ma5_exact(self):
        # MA5 = (10+11+12+13+14)/5 = 12.0，手工可验算
        result = compute_indicators(make_rows([10, 11, 12, 13, 14]))
        assert result["ok"] is True
        assert result["ma"]["ma5"] == pytest.approx(12.0, abs=TOL)

    def test_ma60_none_when_short(self):
        result = compute_indicators(make_rows(list(range(1, 31))))  # 30 行
        assert result["ma"]["ma5"] is not None
        assert result["ma"]["ma10"] is not None
        assert result["ma"]["ma20"] is not None
        assert result["ma"]["ma60"] is None

    def test_constant_closes_all_ma_equal(self):
        result = compute_indicators(make_rows([10.0] * 70))
        for key in ("ma5", "ma10", "ma20", "ma60"):
            assert result["ma"][key] == pytest.approx(10.0, abs=TOL)

    def test_tushare_trade_date_field(self):
        rows = [{"trade_date": f"202401{10 + i}", "close": 10 + i,
                 "vol": "12345"} for i in range(5)]
        result = compute_indicators(rows)
        assert result["ok"] is True
        assert result["as_of"] == "2024-01-14"
        assert result["close"] == pytest.approx(14.0, abs=TOL)


# ─────────────────────────────────────────────
# 均线排列七态（规则见 MA_ALIGNMENT_TABLE）
# ─────────────────────────────────────────────

class TestMAAlignment:
    def test_perfect_bull(self):
        # 严格递增 1..70：close=70 > ma5=68 > ma10=65.5 > ma20=60.5 > ma60=40.5
        result = compute_indicators(make_rows(list(range(1, 71))))
        assert result["ma_alignment"] == "完美多头"

    def test_perfect_bear(self):
        result = compute_indicators(make_rows(list(range(70, 0, -1))))
        assert result["ma_alignment"] == "完美空头"

    def test_bull_with_close_below_ma5(self):
        # 1..69 后回落至 64：ma5=66>ma10=64>ma20=59.25>ma60≈39.42 且 close=64<ma5
        closes = list(range(1, 70)) + [64]
        result = compute_indicators(make_rows(closes))
        assert result["ma_alignment"] == "多头"

    def test_bear_with_close_above_ma5(self):
        # 70..2 后反弹至 6：ma5=4<ma10=7<ma20=11.75<ma60≈31.58 且 close=6>ma5
        closes = list(range(70, 1, -1)) + [6]
        result = compute_indicators(make_rows(closes))
        assert result["ma_alignment"] == "空头"

    def test_tangled(self):
        # 全部同价：close 与均线完全粘合
        result = compute_indicators(make_rows([10.0] * 30))
        assert result["ma_alignment"] == "纠缠"

    def test_weak_bull(self):
        # 20 行（无 ma60）：50,48,...,32 后 33,...,42
        # close=42 > ma20=39.25 且 ma5=40 > ma10=37.5
        closes = [50 - 2 * i for i in range(10)] + [33 + i for i in range(10)]
        result = compute_indicators(make_rows(closes))
        assert result["ma_alignment"] == "弱多头"

    def test_weak_bear(self):
        # 30,32,...,48 后 47,...,38：close=38 < ma20=40.75 且 ma5=40 < ma10=42.5
        closes = [30 + 2 * i for i in range(10)] + [47 - i for i in range(10)]
        result = compute_indicators(make_rows(closes))
        assert result["ma_alignment"] == "弱空头"

    def test_unknown_when_ma20_missing(self):
        result = compute_indicators(make_rows([10, 11, 12, 13, 14]))  # 仅 5 行
        assert result["ma_alignment"] == "未知"


# ─────────────────────────────────────────────
# 乖离率
# ─────────────────────────────────────────────

class TestBias:
    def test_bias20_exact(self):
        # 24 根 10 + 末根 11：ma20 = (10*19+11)/20 = 10.05
        # bias20 = 0.95/10.05*100 = 19/201*100 ≈ 9.452736
        closes = [10.0] * 24 + [11.0]
        result = compute_indicators(make_rows(closes))
        assert result["bias"]["bias20"] == pytest.approx(
            19.0 / 201.0 * 100.0, abs=TOL)
        assert result["bias"]["bias60"] is None  # 25 行 < 60

    def test_bias60_present_with_60_rows(self):
        closes = [10.0] * 69 + [11.0]
        result = compute_indicators(make_rows(closes))
        # ma60 = (10*59+11)/60 ≈ 10.016667；bias60 ≈ 9.3170
        assert result["bias"]["bias60"] == pytest.approx(
            (11.0 - (10.0 * 59 + 11.0) / 60.0) / ((10.0 * 59 + 11.0) / 60.0) * 100,
            abs=TOL)


# ─────────────────────────────────────────────
# 量能五态
# ─────────────────────────────────────────────

class TestVolumeState:
    def _run(self, last_vol):
        vols = [100.0] * 24 + [last_vol]
        return compute_indicators(make_rows(list(range(1, 26)), vols=vols))

    def test_huge_volume(self):
        assert self._run(250.0)["volume_state"] == "巨量"  # ratio 2.5

    def test_huge_volume_boundary(self):
        assert self._run(200.0)["volume_state"] == "巨量"  # ratio 恰好 2.0

    def test_expanding_volume(self):
        assert self._run(160.0)["volume_state"] == "放量"  # ratio 1.6

    def test_expanding_volume_boundary(self):
        assert self._run(150.0)["volume_state"] == "放量"  # ratio 恰好 1.5

    def test_flat_volume(self):
        assert self._run(100.0)["volume_state"] == "平量"  # ratio 1.0

    def test_flat_volume_lower_boundary(self):
        assert self._run(80.0)["volume_state"] == "平量"  # ratio 恰好 0.8

    def test_shrinking_volume(self):
        assert self._run(60.0)["volume_state"] == "缩量"  # ratio 0.6

    def test_ground_volume(self):
        assert self._run(30.0)["volume_state"] == "地量"  # ratio 0.3

    def test_missing_without_vol_field(self):
        result = compute_indicators(make_rows(list(range(1, 26))))
        assert result["volume_state"] == "missing"

    def test_missing_when_history_too_short(self):
        # 仅 5 行：历史量 4 根 < VOLUME_MIN_HISTORY(5)
        result = compute_indicators(make_rows([1, 2, 3, 4, 5],
                                              vols=[100.0] * 5))
        assert result["volume_state"] == "missing"


# ─────────────────────────────────────────────
# MACD（EMA 以首值播种，hist = 2*(dif-dea)）
# ─────────────────────────────────────────────

class TestMACD:
    def test_golden_cross_exact_values(self):
        # closes=[1,1,1,1,2]：ema12₅=15/13, ema26₅=29/27
        # dif₅ = 28/351；dea₅ = 0.2*28/351 = 28/1755；hist = 224/1755
        result = compute_indicators(make_rows([1.0, 1.0, 1.0, 1.0, 2.0]))
        macd = result["macd"]
        assert macd["dif"] == pytest.approx(28.0 / 351.0, abs=TOL)
        assert macd["dea"] == pytest.approx(28.0 / 1755.0, abs=TOL)
        assert macd["hist"] == pytest.approx(224.0 / 1755.0, abs=TOL)
        assert macd["state"] == "金叉"

    def test_dead_cross_exact_values(self):
        result = compute_indicators(make_rows([2.0, 2.0, 2.0, 2.0, 1.0]))
        macd = result["macd"]
        assert macd["dif"] == pytest.approx(-28.0 / 351.0, abs=TOL)
        assert macd["dea"] == pytest.approx(-28.0 / 1755.0, abs=TOL)
        assert macd["hist"] == pytest.approx(-224.0 / 1755.0, abs=TOL)
        assert macd["state"] == "死叉"

    def test_bull_state_on_steady_rise(self):
        # 线性上涨：dif 持续在 dea 上方（交叉已发生）→ 多头
        result = compute_indicators(make_rows([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        assert result["macd"]["state"] == "多头"
        assert result["macd"]["dif"] > result["macd"]["dea"]

    def test_bear_state_on_steady_fall(self):
        result = compute_indicators(make_rows([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]))
        assert result["macd"]["state"] == "空头"
        assert result["macd"]["dif"] < result["macd"]["dea"]

    def test_constant_closes_zero_dif(self):
        result = compute_indicators(make_rows([10.0] * 10))
        assert result["macd"]["dif"] == pytest.approx(0.0, abs=TOL)
        assert result["macd"]["hist"] == pytest.approx(0.0, abs=TOL)
        assert result["macd"]["state"] == "多头"  # dif >= dea 的确定性归属


# ─────────────────────────────────────────────
# RSI（简单平均法）
# ─────────────────────────────────────────────

class TestRSI:
    def test_all_up_is_100(self):
        closes = [10.0 + i for i in range(7)]  # 6 个变动全涨
        result = compute_indicators(make_rows(closes))
        assert result["rsi"]["rsi6"] == pytest.approx(100.0, abs=TOL)

    def test_all_down_is_0(self):
        closes = [16.0 - i for i in range(7)]
        result = compute_indicators(make_rows(closes))
        assert result["rsi"]["rsi6"] == pytest.approx(0.0, abs=TOL)

    def test_alternating_is_60(self):
        # 变动 +1,-1,+1,-1,+1（5 个，不足 6 根用全部）：
        # avg_gain=3/5, avg_loss=2/5 → RSI = 100*0.6/(0.6+0.4) = 60
        closes = [10.0, 11.0, 10.0, 11.0, 10.0, 11.0]
        result = compute_indicators(make_rows(closes))
        assert result["rsi"]["rsi6"] == pytest.approx(60.0, abs=TOL)
        assert result["rsi"]["rsi12"] == pytest.approx(60.0, abs=TOL)

    def test_flat_is_50(self):
        result = compute_indicators(make_rows([10.0] * 10))
        assert result["rsi"]["rsi6"] == pytest.approx(50.0, abs=TOL)

    def test_rsi12_uses_last_12_changes(self):
        # 前 5 个变动全跌、后 12 个变动全涨 → RSI12=100，RSI6=100
        closes = [20.0 - i for i in range(6)] + [15.0 + i for i in range(1, 13)]
        result = compute_indicators(make_rows(closes))
        assert result["rsi"]["rsi12"] == pytest.approx(100.0, abs=TOL)


# ─────────────────────────────────────────────
# 支撑位 / 压力位
# ─────────────────────────────────────────────

class TestSwingLevels:
    def test_swing_low_high(self):
        # 摆动低点 8(i=2)/9(i=4)/9(i=7) → 支撑=max(低于close=11)=9
        # 摆动高点 15(i=2)/16(i=6) → 压力=min(高于close=11)=15
        closes = [11.0] * 10
        lows = [9, 10, 8, 10, 9, 10, 11, 9, 10, 11]
        highs = [13, 14, 15, 14, 13, 14, 16, 14, 13, 12]
        result = compute_indicators(make_rows(closes, highs=highs, lows=lows))
        assert result["support"] == pytest.approx(9.0, abs=TOL)
        assert result["resistance"] == pytest.approx(15.0, abs=TOL)

    def test_fallback_to_window_extremes(self):
        # close=9 低于所有摆动低点 → 支撑回退窗口最低价 10
        closes = [9.0] * 7
        lows = [10.0] * 7
        highs = [20.0] * 7
        result = compute_indicators(make_rows(closes, highs=highs, lows=lows))
        assert result["support"] == pytest.approx(10.0, abs=TOL)
        assert result["resistance"] == pytest.approx(20.0, abs=TOL)

    def test_none_when_high_low_missing(self):
        rows = [{"date": "2024-01-0%d" % (i + 1), "close": 10.0 + i}
                for i in range(6)]
        result = compute_indicators(rows)
        assert result["ok"] is True
        assert result["support"] is None
        assert result["resistance"] is None


# ─────────────────────────────────────────────
# 综合评分
# ─────────────────────────────────────────────

class TestScore:
    def test_perfect_bear_exact_score(self):
        # closes = [30-0.1i]，70 行：完美多头反向
        # 排列 0 + MACD空头 30*25 + bias(-3.95%)→50*15 + 缺量 50*15 + RSI(0)→20*15
        # = (750+750+750+300)/100 = 25.5
        closes = [30.0 - 0.1 * i for i in range(70)]
        result = compute_indicators(make_rows(closes))
        assert result["ma_alignment"] == "完美空头"
        assert result["score"] == pytest.approx(25.5, abs=TOL)

    def test_gentle_rise_high_score(self):
        # 温和上行（bias20≈5.96% 处于最佳区间）+ 放量 → 高分强势
        closes = [10.0 + 0.1 * i for i in range(70)]
        vols = [100.0] * 69 + [180.0]
        result = compute_indicators(make_rows(closes, vols=vols))
        assert result["ma_alignment"] == "完美多头"
        assert result["score"] >= 80.0
        assert result["score"] <= 100.0

    def test_score_always_in_0_100(self):
        for closes in ([1.0] * 70, list(range(1, 71)),
                       list(range(70, 0, -1)), [10, 12, 9, 13, 8, 14, 7]):
            result = compute_indicators(make_rows(list(closes)))
            assert 0.0 <= result["score"] <= 100.0

    def test_score_breakdown_structure(self):
        result = compute_indicators(make_rows(list(range(1, 71))))
        breakdown = result["score_breakdown"]
        assert set(breakdown) == {"ma_alignment", "macd", "bias", "volume", "rsi"}
        total_weight = 0
        for key, item in breakdown.items():
            assert item["weight"] == DEFAULT_SCORE_WEIGHTS[key]
            assert 0.0 <= item["score"] <= 100.0
            assert isinstance(item["note"], str) and item["note"]
            total_weight += item["weight"]
        assert total_weight == 100

    def test_config_weights_override(self):
        # 仅留 RSI 权重：完美空头场景 RSI6=0 → 子分 20 → 总分即 20
        closes = [30.0 - 0.1 * i for i in range(70)]
        result = compute_indicators(make_rows(closes), config={
            "weights": {"ma_alignment": 0, "macd": 0, "bias": 0,
                        "volume": 0, "rsi": 100}})
        assert result["score"] == pytest.approx(20.0, abs=TOL)

    def test_config_weights_illegal_ignored(self):
        closes = [30.0 - 0.1 * i for i in range(70)]
        result = compute_indicators(make_rows(closes), config={
            "weights": {"ma_alignment": -5, "macd": "abc"}})
        # 非法权重回退默认 → 与默认结果一致
        assert result["score"] == pytest.approx(25.5, abs=TOL)


# ─────────────────────────────────────────────
# 输入容错（绝不抛）
# ─────────────────────────────────────────────

class TestRobustness:
    def test_too_few_rows(self):
        result = compute_indicators(make_rows([10, 11, 12, 13]))
        assert result["ok"] is False
        assert "数据不足" in result["note"]

    def test_empty_rows(self):
        result = compute_indicators([])
        assert result["ok"] is False

    def test_none_rows(self):
        result = compute_indicators(None)
        assert result["ok"] is False

    def test_garbage_rows_never_raise(self):
        result = compute_indicators(["junk", 42, None, {"close": "bad"}])
        assert result["ok"] is False

    def test_dirty_rows_skipped(self):
        rows = [{"close": "bad"}, {"nope": 1}, "junk"]
        rows += [{"date": "2024-01-0%d" % (i + 1), "close": 10.0 + i}
                 for i in range(5)]
        result = compute_indicators(rows)
        assert result["ok"] is True
        assert result["close"] == pytest.approx(14.0, abs=TOL)

    def test_result_keys_complete(self):
        result = compute_indicators(make_rows(list(range(1, 71))))
        for key in ("ok", "as_of", "close", "ma", "ma_alignment", "bias",
                    "volume_state", "macd", "rsi", "support", "resistance",
                    "score", "score_breakdown", "source"):
            assert key in result

    def test_module_constants_exported(self):
        assert sum(DEFAULT_SCORE_WEIGHTS.values()) == 100
        assert [b["min"] for b in DEFAULT_BANDS] == [80, 60, 40, 20, 0]
        assert {r["key"] for r in DATA_QUALITY_CAPS} == {
            "stale", "no_volume", "insufficient"}


# ─────────────────────────────────────────────
# verdict_from_score：评分带边界 + 护栏矩阵
# ─────────────────────────────────────────────

class TestVerdict:
    @pytest.mark.parametrize("score,band", [
        (100, "强势"), (81, "强势"), (80, "强势"),
        (79, "偏多"), (60, "偏多"),
        (59.9, "中性"), (40, "中性"),
        (39.9, "偏空"), (20, "偏空"),
        (19.9, "弱势"), (0, "弱势"),
    ])
    def test_band_boundaries(self, score, band):
        result = verdict_from_score(score)
        assert result["band"] == band
        assert isinstance(result["action"], str) and result["action"]
        assert result["confidence_cap"] == 1.0
        assert result["guardrail_reason"] is None

    def test_stale_cap(self):
        result = verdict_from_score(90, {"stale": True})
        assert result["confidence_cap"] == 0.5
        assert "非最新交易日" in result["guardrail_reason"]

    def test_no_volume_cap(self):
        result = verdict_from_score(90, {"no_volume": True})
        assert result["confidence_cap"] == 0.7
        assert "量能" in result["guardrail_reason"]

    def test_insufficient_cap(self):
        result = verdict_from_score(90, {"insufficient": True})
        assert result["confidence_cap"] == 0.3

    def test_multiple_flags_take_strictest(self):
        result = verdict_from_score(90, {"stale": True, "insufficient": True,
                                         "no_volume": True})
        assert result["confidence_cap"] == 0.3  # 三条命中取最严
        assert "；" in result["guardrail_reason"]  # 多条原因连接

    def test_stale_plus_no_volume(self):
        result = verdict_from_score(90, {"stale": True, "no_volume": True})
        assert result["confidence_cap"] == 0.5

    def test_none_data_quality(self):
        result = verdict_from_score(90, None)
        assert result["confidence_cap"] == 1.0
        assert result["guardrail_reason"] is None

    def test_custom_bands_override(self):
        result = verdict_from_score(75, config={"bands": [
            {"min": 50, "band": "强", "action": "买"},
            {"min": 0, "band": "弱", "action": "卖"},
        ]})
        assert result["band"] == "强"
        assert result["action"] == "买"

    def test_illegal_score_falls_to_lowest_band(self):
        assert verdict_from_score("abc")["band"] == "弱势"
        assert verdict_from_score(None)["band"] == "弱势"


# ─────────────────────────────────────────────
# is_trade_day 三路径
# ─────────────────────────────────────────────

class TestIsTradeDay:
    def test_calendar_membership(self):
        calendar = {"2024-01-15", "2024-01-16"}
        assert is_trade_day("2024-01-15", calendar=calendar) is True
        assert is_trade_day("2024-01-17", calendar=calendar) is False

    def test_calendar_overrides_weekend(self):
        # 调休工作日：周日但在日历中 → True（注入日历优先于启发式）
        assert is_trade_day("2024-01-14", calendar={"2024-01-14"}) is True

    def test_calendar_accepts_yyyymmdd_entries(self):
        assert is_trade_day("2024-01-15", calendar={"20240115"}) is True

    def test_heuristic_weekday(self):
        assert is_trade_day("2024-01-15") is True   # 周一
        assert is_trade_day("2024-01-19") is True   # 周五

    def test_heuristic_weekend(self):
        assert is_trade_day("2024-01-14") is False  # 周日
        assert is_trade_day("2024-01-13") is False  # 周六

    def test_heuristic_minus_holidays(self):
        assert is_trade_day("2024-01-15", holidays={"2024-01-15"}) is False
        assert is_trade_day("2024-01-16", holidays={"2024-01-15"}) is True

    def test_accepts_date_object_and_compact_string(self):
        assert is_trade_day(date(2024, 1, 15)) is True
        assert is_trade_day("20240115") is True

    def test_invalid_input_returns_false(self):
        assert is_trade_day("not-a-date") is False
        assert is_trade_day(None) is False
        assert is_trade_day(20240115) is False  # int 不在契约内
