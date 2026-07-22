"""
第十一波『确定性技术分析层与决策护栏』模块（纯 stdlib，零网络，零 LLM，fail-safe）。

职责：把调用方注入的升序日线 K 线序列，确定性地计算为结构化技术指标与
评分结论，供 Stage 2 工具层包装后注入智能体分析流程。设计思路借鉴常见
趋势交易技术分析框架（均线排列 / MACD / RSI / 乖离率 / 量能 / 支撑压力），
全部代码与文案均为本项目原创撰写。

输入契约（rows，升序，兼容 Tushare daily 字段名）：
    [{"date": "YYYY-MM-DD" 或 "trade_date": "YYYYMMDD",
      "open": x, "high": x, "low": x, "close": x,
      "vol": x(可选), "amount": x(可选)}, ...]

公开契约（签名固定，供 Stage 2 与测试引用）：
- compute_indicators(rows, config=None) -> dict
- verdict_from_score(score, data_quality=None, config=None) -> dict
- is_trade_day(day, calendar=None, holidays=None) -> bool
- 常量：DEFAULT_SCORE_WEIGHTS / DEFAULT_BANDS / DATA_QUALITY_CAPS

工程约定（与项目铁律一致）：
- 纯函数、零 I/O、零网络、不依赖 pandas（纯 Python 列表计算）；
- 字段缺失优雅降级（相关指标置 None / 状态标记 missing / 评分取中性），
  任何输入绝不抛异常，失败一律返回 {"ok": False, "note": "..."}；
- 结果带 source 字段标明出处（合规要求：每个数字必须有数据块出处）。
"""

import logging
import math
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 模块常量（Stage 2 与测试可直接引用）
# ─────────────────────────────────────────────

# 最少 K 线行数：低于此数任何指标都不可靠，直接判定数据不足。
MIN_ROWS = 5

# MACD 标准参数（12, 26, 9）；hist 采用通达信惯例 MACD 柱 = 2 * (DIF - DEA)。
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# RSI 周期（简单平均法 / Cutler's RSI，便于手工验算；见 _rsi docstring）。
RSI_SHORT = 6
RSI_LONG = 12

# 支撑/压力摆动点观察窗口（近 60 根 K 线）。
SWING_WINDOW = 60

# 量能五态判定表：最新成交量 / 前 20 日均量（不含最新一根，至少需 5 根历史量
# 否则样本太少标记 missing）。按 ratio 从大到小匹配，写成常量表便于审阅调整。
VOLUME_STATE_TABLE: Tuple[Dict[str, Any], ...] = (
    {"min_ratio": 2.0, "state": "巨量", "comment": "量能异常放大（≥2 倍均量），常见于消息刺激或变盘"},
    {"min_ratio": 1.5, "state": "放量", "comment": "温和放大（1.5~2 倍），趋势配合度好"},
    {"min_ratio": 0.8, "state": "平量", "comment": "与均量相当（0.8~1.5 倍），常态"},
    {"min_ratio": 0.5, "state": "缩量", "comment": "明显萎缩（0.5~0.8 倍），观望情绪浓"},
    {"min_ratio": 0.0, "state": "地量", "comment": "极度萎缩（<0.5 倍），常见于趋势末端"},
)
# 量能历史样本下限：前序有效量少于该值时标记 missing。
VOLUME_MIN_HISTORY = 5

# 均线排列七态判定表（注释即规则文档；实现见 _ma_alignment，按表内顺序
# 自上而下优先匹配，规则未命中任何一条时回退「纠缠」）。
# 约定：ma60 缺失时跳过依赖 ma60 的四个强态（完美多头/完美空头/多头/空头），
# 仅用 ma5/ma10/ma20 判定弱态；ma20 缺失（行数 <20）直接返回「未知」。
MA_ALIGNMENT_TABLE: Tuple[Dict[str, str], ...] = (
    {"state": "完美多头", "rule": "close > ma5 > ma10 > ma20 > ma60（严格单调）",
     "comment": "价格领跑全部均线，最强趋势形态"},
    {"state": "完美空头", "rule": "close < ma5 < ma10 < ma20 < ma60（严格单调）",
     "comment": "价格垫底全部均线，最弱趋势形态"},
    {"state": "多头", "rule": "ma5 > ma10 > ma20 > ma60（收盘价允许回落到 ma5 下方）",
     "comment": "均线多头排列，趋势仍向上"},
    {"state": "空头", "rule": "ma5 < ma10 < ma20 < ma60（收盘价允许反弹到 ma5 上方）",
     "comment": "均线空头排列，趋势仍向下"},
    {"state": "纠缠",
     "rule": "|close-ma20|/ma20 < 1% 且 (max-min)(ma5,ma10,ma20)/ma20 < 2%",
     "comment": "价格与均线粘合，方向待选择"},
    {"state": "弱多头", "rule": "close > ma20 且 ma5 > ma10",
     "comment": "站上中期均线且短期向上，但排列未成型"},
    {"state": "弱空头", "rule": "close < ma20 且 ma5 < ma10",
     "comment": "跌破中期均线且短期向下，但排列未成型"},
)

# 默认评分权重（合计 100；config["weights"] 可按键覆盖）。
# 权重设计偏向趋势交易：均线排列（趋势结构）与 MACD（趋势动量）合计 55 分
# 占绝对主导；乖离率用于惩罚追高/超跌的极端偏离；量能与 RSI 作为辅助确认。
DEFAULT_SCORE_WEIGHTS: Dict[str, int] = {
    "ma_alignment": 30,  # 趋势结构：趋势交易的核心信号，权重最高
    "macd": 25,          # 趋势动量：MACD 金叉/死叉与多空状态确认
    "bias": 15,          # 乖离率：偏离过大（过热/超卖）扣分，防追高杀低
    "volume": 15,        # 量能配合：放量确认趋势，缺量时取中性分
    "rsi": 15,           # RSI 超买超卖：辅助判断短期位置
}

# 各分项 0-100 子分映射表（实现见 _score_* 系列函数，常量表形式便于审阅）。
ALIGNMENT_SCORE_TABLE: Dict[str, float] = {
    "完美多头": 100.0, "多头": 85.0, "弱多头": 65.0, "纠缠": 50.0,
    "弱空头": 35.0, "空头": 15.0, "完美空头": 0.0, "未知": 50.0,
}
MACD_SCORE_TABLE: Dict[str, float] = {
    "金叉": 90.0, "多头": 70.0, "空头": 30.0, "死叉": 10.0,
}
# bias20 分段打分：(下限含, 上限不含, 子分)。适度正乖离最佳；过热与深跌均扣分。
BIAS_SCORE_TABLE: Tuple[Tuple[float, float, float], ...] = (
    (-math.inf, -10.0, 10.0),   # 深度负乖离：趋势严重受损
    (-10.0, -5.0, 30.0),
    (-5.0, -2.0, 50.0),
    (-2.0, 2.0, 70.0),          # 贴近均线：可上可下，中性偏稳
    (2.0, 6.0, 100.0),          # 温和正乖离：趋势交易最佳区间
    (6.0, 10.0, 60.0),          # 偏离偏大：追高风险显现
    (10.0, math.inf, 30.0),     # 过热：显著追高风险
)
VOLUME_SCORE_TABLE: Dict[str, float] = {
    "放量": 90.0, "巨量": 70.0, "平量": 60.0, "缩量": 40.0,
    "地量": 20.0, "missing": 50.0,  # 缺量取中性分，惩罚交由护栏层处理
}
# RSI6 分段打分：(下限含, 上限不含, 子分)。50~80 为趋势友好区，超买超卖均扣分。
RSI_SCORE_TABLE: Tuple[Tuple[float, float, float], ...] = (
    (80.0, math.inf, 35.0),     # 超买：短期过热
    (60.0, 80.0, 90.0),         # 强势区：趋势交易最佳
    (50.0, 60.0, 75.0),
    (40.0, 50.0, 55.0),
    (20.0, 40.0, 35.0),         # 弱势区
    (-math.inf, 20.0, 20.0),    # 超卖：趋势严重受损（反弹与否交给趋势项判断）
)

# 评分带 → 结论映射（默认带，config["bands"] 可整体覆盖；按 min 降序排列）。
DEFAULT_BANDS: Tuple[Dict[str, Any], ...] = (
    {"min": 80, "band": "强势", "action": "趋势强劲，可顺势持有或逢回调分批介入"},
    {"min": 60, "band": "偏多", "action": "偏多格局，持股为主，回调不破支撑可加仓"},
    {"min": 40, "band": "中性", "action": "方向不明，观望为主，严格控制仓位"},
    {"min": 20, "band": "偏空", "action": "偏弱格局，减仓避险，等待企稳信号"},
    {"min": 0, "band": "弱势", "action": "趋势走弱，回避为主，不宜左侧抄底"},
)

# 数据质量护栏表（注释即规则文档；实现见 verdict_from_score）。
# 任一规则命中即把 confidence_cap 压到对应上限；多条命中取最严（最小值）。
DATA_QUALITY_CAPS: Tuple[Dict[str, Any], ...] = (
    {"key": "insufficient", "cap": 0.3,
     "reason": "K线数据不足，指标可信度低，结论仅供参考"},
    {"key": "stale", "cap": 0.5,
     "reason": "数据非最新交易日（as_of 滞后），结论可能已过时"},
    {"key": "no_volume", "cap": 0.7,
     "reason": "缺少量能数据，量价配合无法验证"},
)

# 合规出处标注：本模块所有数字均为对调用方注入 K 线的本地确定性计算结果。
SOURCE_NOTE = "本地确定性计算（输入K线由调用方注入，如 Tushare daily）"


# ─────────────────────────────────────────────
# 输入归一化（全部容错，绝不抛）
# ─────────────────────────────────────────────

def _to_float(value: Any) -> Optional[float]:
    """宽松转 float；失败/None/NaN/inf 一律返回 None。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _normalize_date(value: Any) -> Optional[str]:
    """日期归一化为 'YYYY-MM-DD'；兼容 'YYYYMMDD' 与 date/datetime；失败返回 None。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        s = value.strip()
        if len(s) == 8 and s.isdigit():
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            return s
    return None


def _normalize_rows(rows: Any) -> List[Dict[str, Optional[float]]]:
    """把原始 rows 归一化为内部结构；无法解析的行跳过。

    返回列表元素：{"date": str|None, "open"/"high"/"low"/"close": float|None,
    "vol": float|None, "amount": float|None}。仅保留 close 可解析的行
    （close 是所有价格指标的锚），其余字段允许 None 优雅降级。
    """
    normalized: List[Dict[str, Optional[float]]] = []
    if not isinstance(rows, (list, tuple)):
        return normalized
    for row in rows:
        if not isinstance(row, dict):
            continue
        close = _to_float(row.get("close"))
        if close is None:
            continue
        normalized.append({
            "date": _normalize_date(row.get("date") or row.get("trade_date")),
            "open": _to_float(row.get("open")),
            "high": _to_float(row.get("high")),
            "low": _to_float(row.get("low")),
            "close": close,
            "vol": _to_float(row.get("vol")),
            "amount": _to_float(row.get("amount")),
        })
    return normalized


# ─────────────────────────────────────────────
# 指标计算（纯函数）
# ─────────────────────────────────────────────

def _sma(values: Sequence[float], period: int) -> Optional[float]:
    """末 period 项简单平均；样本不足返回 None。"""
    if period <= 0 or len(values) < period:
        return None
    return sum(values[-period:]) / period


def _ema_series(values: Sequence[float], period: int) -> List[float]:
    """标准 EMA 序列：alpha = 2/(period+1)，以首值播种。与 values 等长。"""
    alpha = 2.0 / (period + 1)
    out: List[float] = []
    prev = 0.0
    for i, v in enumerate(values):
        prev = v if i == 0 else alpha * v + (1 - alpha) * prev
        out.append(prev)
    return out


def _macd(closes: Sequence[float]) -> Dict[str, Any]:
    """MACD(12,26,9)。dif = EMA12-EMA26；dea = DIF 的 EMA9；
    hist = 2*(dif-dea)（通达信惯例）。
    状态：金叉（上穿）/死叉（下穿）/多头（dif>=dea 且无新交叉）/空头（dif<dea）。
    """
    ema_fast = _ema_series(closes, MACD_FAST)
    ema_slow = _ema_series(closes, MACD_SLOW)
    dif_series = [f - s for f, s in zip(ema_fast, ema_slow)]
    dea_series = _ema_series(dif_series, MACD_SIGNAL)
    dif, dea = dif_series[-1], dea_series[-1]
    prev_dif, prev_dea = dif_series[-2], dea_series[-2]
    hist = 2.0 * (dif - dea)
    if prev_dif <= prev_dea and dif > dea:
        state = "金叉"
    elif prev_dif >= prev_dea and dif < dea:
        state = "死叉"
    elif dif >= dea:
        state = "多头"
    else:
        state = "空头"
    return {"dif": dif, "dea": dea, "hist": hist, "state": state}


def _rsi(closes: Sequence[float], period: int) -> Optional[float]:
    """RSI（简单平均法 / Cutler's RSI）：RSI = 100*avg_gain/(avg_gain+avg_loss)，
    对末 period 根 K 线的涨跌幅取简单平均；可用变动不足 period 根时用已有变动
    计算（至少 1 根）。无涨跌（全平）返回 50；平均跌幅为 0（只涨）返回 100。
    选择简单平均而非 Wilder 平滑，是为了让小样本序列可手工验算。
    """
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    if not changes:
        return None
    window = changes[-period:] if len(changes) >= period else changes
    gains = sum(c for c in window if c > 0) / len(window)
    losses = sum(-c for c in window if c < 0) / len(window)
    if gains == 0 and losses == 0:
        return 50.0
    if losses == 0:
        return 100.0
    return 100.0 * gains / (gains + losses)


def _ma_alignment(close: float, ma: Dict[str, Optional[float]]) -> str:
    """均线排列七态，规则见 MA_ALIGNMENT_TABLE（按表序优先匹配）。"""
    ma5, ma10, ma20, ma60 = ma["ma5"], ma["ma10"], ma["ma20"], ma["ma60"]
    if ma20 is None or ma5 is None or ma10 is None:
        return "未知"
    if ma60 is not None:
        if close > ma5 > ma10 > ma20 > ma60:
            return "完美多头"
        if close < ma5 < ma10 < ma20 < ma60:
            return "完美空头"
        if ma5 > ma10 > ma20 > ma60:
            return "多头"
        if ma5 < ma10 < ma20 < ma60:
            return "空头"
    near_price = abs(close - ma20) / ma20 < 0.01 if ma20 else False
    spread = (max(ma5, ma10, ma20) - min(ma5, ma10, ma20)) / ma20 if ma20 else math.inf
    if near_price and spread < 0.02:
        return "纠缠"
    if close > ma20 and ma5 > ma10:
        return "弱多头"
    if close < ma20 and ma5 < ma10:
        return "弱空头"
    return "纠缠"  # 表内规则均未命中的确定性回退


def _volume_state(vols: Sequence[Optional[float]]) -> str:
    """量能五态，规则见 VOLUME_STATE_TABLE。

    基准 = 最新一根之前最多 20 根有效量的均值（不含最新一根，避免自我抬升）；
    历史有效量 < VOLUME_MIN_HISTORY 或最新量缺失时返回 'missing'。
    """
    if not vols or vols[-1] is None:
        return "missing"
    history = [v for v in vols[:-1] if v is not None]
    if len(history) < VOLUME_MIN_HISTORY:
        return "missing"
    base = sum(history[-20:]) / len(history[-20:])
    if base <= 0:
        return "missing"
    ratio = vols[-1] / base
    for row in VOLUME_STATE_TABLE:
        if ratio >= row["min_ratio"]:
            return row["state"]
    return "missing"  # 防御性回退（表覆盖全部非负 ratio，正常不会到达）


def _swing_levels(bars: Sequence[Dict[str, Optional[float]]],
                  close: float) -> Tuple[Optional[float], Optional[float]]:
    """近 60 根摆动高低点求支撑/压力（确定性算法）。

    摆动低点：low[i] <= low[i-1] 且 low[i] <= low[i+1]（窗口内部点）。
    支撑 = 低于收盘价的摆动低点中最高者；无候选则回退窗口最低价。
    压力 = 高于收盘价的摆动高点中最低者；无候选则回退窗口最高价。
    high/low 字段缺失的行跳过；全缺时对应结果返回 None。
    """
    window = list(bars[-SWING_WINDOW:])
    lows = [b["low"] for b in window]
    highs = [b["high"] for b in window]

    swing_lows: List[float] = []
    swing_highs: List[float] = []
    for i in range(1, len(window) - 1):
        lo, lo_prev, lo_next = lows[i], lows[i - 1], lows[i + 1]
        if lo is not None and lo_prev is not None and lo_next is not None \
                and lo <= lo_prev and lo <= lo_next:
            swing_lows.append(lo)
        hi, hi_prev, hi_next = highs[i], highs[i - 1], highs[i + 1]
        if hi is not None and hi_prev is not None and hi_next is not None \
                and hi >= hi_prev and hi >= hi_next:
            swing_highs.append(hi)

    below = [v for v in swing_lows if v < close]
    above = [v for v in swing_highs if v > close]
    valid_lows = [v for v in lows if v is not None]
    valid_highs = [v for v in highs if v is not None]
    support = max(below) if below else (min(valid_lows) if valid_lows else None)
    resistance = min(above) if above else (max(valid_highs) if valid_highs else None)
    return support, resistance


# ─────────────────────────────────────────────
# 评分（权重 config 可覆盖；缺失分项取中性 50）
# ─────────────────────────────────────────────

_NEUTRAL_SCORE = 50.0


def _score_bias(bias20: Optional[float]) -> float:
    if bias20 is None:
        return _NEUTRAL_SCORE
    for lo, hi, score in BIAS_SCORE_TABLE:
        if lo <= bias20 < hi:
            return score
    return _NEUTRAL_SCORE


def _score_rsi(rsi6: Optional[float]) -> float:
    if rsi6 is None:
        return _NEUTRAL_SCORE
    for lo, hi, score in RSI_SCORE_TABLE:
        if lo <= rsi6 < hi:
            return score
    return _NEUTRAL_SCORE


def _composite_score(alignment: str, macd_state: str,
                     bias20: Optional[float], volume_state: str,
                     rsi6: Optional[float],
                     config: Optional[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    """加权综合分。config["weights"] 按键覆盖 DEFAULT_SCORE_WEIGHTS；
    非法权重（非数/负数）忽略回退默认。各分项恒参与加权（缺失取中性 50），
    保证总分稳定可比。"""
    weights = dict(DEFAULT_SCORE_WEIGHTS)
    if isinstance(config, dict) and isinstance(config.get("weights"), dict):
        for key, value in config["weights"].items():
            f = _to_float(value)
            if key in weights and f is not None and f >= 0:
                weights[key] = f

    components: Dict[str, Dict[str, Any]] = {
        "ma_alignment": {
            "score": ALIGNMENT_SCORE_TABLE.get(alignment, _NEUTRAL_SCORE),
            "note": f"均线排列：{alignment}",
        },
        "macd": {
            "score": MACD_SCORE_TABLE.get(macd_state, _NEUTRAL_SCORE),
            "note": f"MACD 状态：{macd_state}",
        },
        "bias": {
            "score": _score_bias(bias20),
            "note": "bias20 缺失取中性分" if bias20 is None else f"bias20={bias20:.2f}%",
        },
        "volume": {
            "score": VOLUME_SCORE_TABLE.get(volume_state, _NEUTRAL_SCORE),
            "note": "量能缺失取中性分" if volume_state == "missing" else f"量能状态：{volume_state}",
        },
        "rsi": {
            "score": _score_rsi(rsi6),
            "note": "RSI6 缺失取中性分" if rsi6 is None else f"RSI6={rsi6:.2f}",
        },
    }

    total_weight = sum(weights.values())
    if total_weight <= 0:
        total_weight = 1.0  # 防御：全部权重被覆盖为 0 时避免除零
    weighted = 0.0
    breakdown: Dict[str, Any] = {}
    for key, comp in components.items():
        w = weights[key]
        weighted += w * comp["score"]
        breakdown[key] = {"weight": w, "score": comp["score"], "note": comp["note"]}
    return round(weighted / total_weight, 2), breakdown


# ─────────────────────────────────────────────
# 公开函数
# ─────────────────────────────────────────────

def compute_indicators(rows: Any, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """计算升序日线序列的技术指标与综合评分。

    参数：
        rows: 升序日线列表（字段契约见模块 docstring，兼容 Tushare daily）。
        config: 可选 {"weights": {...}} 覆盖 DEFAULT_SCORE_WEIGHTS 单项权重。

    返回（rows 有效行 ≥ MIN_ROWS）：
        {"ok": True, "as_of", "close", "source",
         "ma": {"ma5","ma10","ma20","ma60"}, "ma_alignment",
         "bias": {"bias20","bias60"}, "volume_state",
         "macd": {"dif","dea","hist","state"}, "rsi": {"rsi6","rsi12"},
         "support", "resistance", "score", "score_breakdown"}
    有效行不足或输入非法时返回 {"ok": False, "note": "数据不足..."}。
    字段缺失优雅降级（置 None / missing / 中性分），绝不抛异常。
    """
    try:
        bars = _normalize_rows(rows)
        if len(bars) < MIN_ROWS:
            return {
                "ok": False,
                "note": f"数据不足：有效K线仅 {len(bars)} 行（至少需 {MIN_ROWS} 行）",
                "source": SOURCE_NOTE,
            }

        closes = [b["close"] for b in bars if b["close"] is not None]
        close = closes[-1]
        ma = {
            "ma5": _sma(closes, 5),
            "ma10": _sma(closes, 10),
            "ma20": _sma(closes, 20),
            "ma60": _sma(closes, 60),
        }
        alignment = _ma_alignment(close, ma)
        bias20 = (close - ma["ma20"]) / ma["ma20"] * 100.0 if ma["ma20"] else None
        bias60 = (close - ma["ma60"]) / ma["ma60"] * 100.0 if ma["ma60"] else None
        volume_state = _volume_state([b["vol"] for b in bars])
        macd = _macd(closes)
        rsi6 = _rsi(closes, RSI_SHORT)
        rsi12 = _rsi(closes, RSI_LONG)
        support, resistance = _swing_levels(bars, close)
        score, breakdown = _composite_score(
            alignment, macd["state"], bias20, volume_state, rsi6, config)

        return {
            "ok": True,
            "as_of": bars[-1]["date"],
            "close": close,
            "ma": ma,
            "ma_alignment": alignment,
            "bias": {"bias20": bias20, "bias60": bias60},
            "volume_state": volume_state,
            "macd": macd,
            "rsi": {"rsi6": rsi6, "rsi12": rsi12},
            "support": support,
            "resistance": resistance,
            "score": score,
            "score_breakdown": breakdown,
            "source": SOURCE_NOTE,
        }
    except Exception as exc:  # 绝不抛出：任何未预期异常降级为失败说明
        logger.warning("compute_indicators 未预期异常（已降级）: %r", exc)
        return {"ok": False, "note": f"指标计算失败：{exc!r}", "source": SOURCE_NOTE}


def verdict_from_score(score: Any, data_quality: Optional[Dict[str, Any]] = None,
                       config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """评分带 → 结论映射 + 数据质量护栏。

    参数：
        score: 0-100 综合分（非法输入按 0 处理，落入最低带）。
        data_quality: 可选质量标记 {"stale": bool, "no_volume": bool,
                      "insufficient": bool}，键缺失视为 False。
        config: 可选 {"bands": [...]} 整体覆盖 DEFAULT_BANDS
                （每项需含 min/band/action，按 min 降序）。

    返回：
        {"band": 评分带, "action": 一句话操作建议,
         "confidence_cap": 0~1 置信度上限,
         "guardrail_reason": str | None（多条护栏原因以「；」连接）}
    护栏规则见 DATA_QUALITY_CAPS：任一命中即压上限，多条命中取最严。
    """
    try:
        numeric_score = _to_float(score)
        if numeric_score is None:
            numeric_score = 0.0

        bands = DEFAULT_BANDS
        if isinstance(config, dict) and isinstance(config.get("bands"), (list, tuple)):
            custom = [b for b in config["bands"]
                      if isinstance(b, dict) and "min" in b and "band" in b]
            if custom:
                bands = tuple(sorted(custom, key=lambda b: -float(b["min"])))

        band, action = bands[-1]["band"], bands[-1].get("action", "")
        for row in bands:
            if numeric_score >= float(row["min"]):
                band, action = row["band"], row.get("action", "")
                break

        cap = 1.0
        reasons: List[str] = []
        quality = data_quality if isinstance(data_quality, dict) else {}
        for rule in DATA_QUALITY_CAPS:
            if quality.get(rule["key"]):
                cap = min(cap, float(rule["cap"]))
                reasons.append(rule["reason"])

        return {
            "band": band,
            "action": action,
            "confidence_cap": cap,
            "guardrail_reason": "；".join(reasons) if reasons else None,
        }
    except Exception as exc:  # 绝不抛出
        logger.warning("verdict_from_score 未预期异常（已降级）: %r", exc)
        return {
            "band": "弱势",
            "action": "结论生成失败，按最保守处理",
            "confidence_cap": 0.0,
            "guardrail_reason": f"结论映射异常：{exc!r}",
        }


def is_trade_day(day: Any, calendar: Optional[Iterable[str]] = None,
                 holidays: Optional[Iterable[str]] = None) -> bool:
    """判断是否交易日。

    参数：
        day: 'YYYY-MM-DD' / 'YYYYMMDD' / date / datetime。
        calendar: 可注入的真实交易日集合（'YYYY-MM-DD' 字符串，如来自
                  Tushare trade_cal）；注入时唯一依据为集合成员关系。
        holidays: 可注入的节假日集合（'YYYY-MM-DD' 字符串），仅在无
                  calendar 时用于从启发式结果中扣除。

    重要说明：未注入 calendar 时为「周一~周五且不在 holidays 中」的
    **启发式**判断，无法覆盖调休工作日与未声明的节假日；调用方应优先
    注入真实交易日历（Tushare trade_cal），启发式仅供离线兜底与测试。
    无法解析的输入一律返回 False，绝不抛异常。
    """
    try:
        iso = _normalize_date(day)
        if iso is None:
            return False
        if calendar is not None:
            cal: Set[str] = set()
            for d in calendar:
                normalized = _normalize_date(d)
                if normalized is not None:
                    cal.add(normalized)
            return iso in cal
        weekday = datetime.strptime(iso, "%Y-%m-%d").weekday()
        if weekday >= 5:  # 周六=5 / 周日=6
            return False
        if holidays:
            for h in holidays:
                if _normalize_date(h) == iso:
                    return False
        return True
    except Exception:
        return False
