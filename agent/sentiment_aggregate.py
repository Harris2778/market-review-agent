"""agent/sentiment_aggregate.py 舆情分布聚合与快照趋势层（纯函数 + SQLite）。

职责（舆情分析「分布化」改造的聚合层，零 LLM、零网络直连）：
1. aggregate_distribution：把已打分条目（sentiment 标签）聚合为整体情绪分布
   ——计数分布 + 点赞加权分布 + 样本量置信度分档，替代「只引用个别帖子」的
   轶事式描述。
2. pick_representatives：每个情感桶按 likes 降序取代表性样本（瘦身字段），
   单条引用降级为分布主体的点缀。
3. save_snapshot / get_trend：复用 social.db 落盘情绪快照
   （表 sentiment_snapshots，主键 platform+target+date，幂等 REPLACE），
   并按近两个有效快照的乐观占比差判定趋势方向。

标签兼容（全局契约）：
- LLM 体系「乐观/中性/悲观」与词典体系「利好/利空/中性」双兼容：
  利好→乐观、利空→悲观归并；未知标签归入中性并计入 unknown_count。

权重契约：条目 metrics.likes（缺省回退顶层 likes，再缺省 0）+ 1 为权重
（0 赞也有基础票）。

置信度分档（全局契约）：n<30 → 低（声明样本不足）；30≤n<100 → 中；n≥100 → 高。

铁律：所有公开函数任何异常记 warning 并返回安全值，绝不抛出。
"""

import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional

from agent import social_store

logger = logging.getLogger(__name__)

METHOD = "aggregate_v1"

# 情感桶（固定三桶，输出键序）
BUCKET_POS = "乐观"
BUCKET_NEU = "中性"
BUCKET_NEG = "悲观"
BUCKETS = (BUCKET_POS, BUCKET_NEU, BUCKET_NEG)

# 标签归并表：词典体系 → LLM 体系
_LABEL_MAP = {
    "乐观": BUCKET_POS,
    "利好": BUCKET_POS,
    "中性": BUCKET_NEU,
    "悲观": BUCKET_NEG,
    "利空": BUCKET_NEG,
}

# 置信度分档边界（全局契约）
CONFIDENCE_LOW_MAX = 30    # n < 30 → 低
CONFIDENCE_HIGH_MIN = 100  # n >= 100 → 高

# 趋势判定：近两个有效快照乐观占比差绝对值超过该值（百分点）视为转向
TREND_SHIFT_PCT = 15.0

_SNAPSHOT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sentiment_snapshots (
    platform   TEXT NOT NULL,
    target     TEXT NOT NULL,
    date       TEXT NOT NULL,
    n          INTEGER NOT NULL DEFAULT 0,
    pos        REAL NOT NULL DEFAULT 0,
    neu        REAL NOT NULL DEFAULT 0,
    neg        REAL NOT NULL DEFAULT 0,
    w_pos      REAL NOT NULL DEFAULT 0,
    w_neu      REAL NOT NULL DEFAULT 0,
    w_neg      REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (platform, target, date)
)
"""


# ═══════════════════════════════════════════
# 纯函数工具
# ═══════════════════════════════════════════

def _clean_str(value) -> str:
    """归一字符串字段：None → ''，其余 strip。绝不抛。"""
    if value is None:
        return ""
    try:
        return str(value).strip()
    except Exception:
        return ""


def _to_float(value, default: float = 0.0) -> float:
    """宽松 float 转换；失败返回 default，绝不抛。"""
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default: int = 0) -> int:
    """宽松 int 转换；失败返回 default，绝不抛。"""
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _norm_label(value) -> Optional[str]:
    """标签归并：返回 '乐观/中性/悲观'；未知标签返回 None（由调用方计入 unknown）。"""
    label = _clean_str(value)
    if not label:
        return None
    return _LABEL_MAP.get(label)


def _item_likes(item: dict, likes_key: str = "likes") -> int:
    """提取点赞数：metrics[likes_key] 优先，回退顶层 likes_key，再缺省 0；
    负值截断到 0，绝不抛。"""
    try:
        metrics = item.get("metrics")
        raw = None
        if isinstance(metrics, dict):
            raw = metrics.get(likes_key)
        if raw is None:
            raw = item.get(likes_key)
        return max(0, _to_int(raw, 0))
    except Exception:
        return 0


def _pct(part: float, total: float) -> float:
    """安全百分比（0-100，保留 1 位小数）；total<=0 返回 0.0。"""
    if total <= 0:
        return 0.0
    return round(part / total * 100.0, 1)


# ═══════════════════════════════════════════
# 1. 情绪分布聚合
# ═══════════════════════════════════════════

def aggregate_distribution(items, weight_likes_key: str = "likes") -> dict:
    """聚合已打分条目为整体情绪分布（纯函数，绝不抛异常）。

    入参 items：已打分条目列表（dict，含 sentiment 标签——兼容
    「乐观/悲观/中性」与词典的「利好/利空」（利好→乐观、利空→悲观归并；
    未知/缺失标签归入中性并计入 unknown_count）；likes 取 metrics.likes
    或顶层 likes，缺省 0，权重 = likes + 1）。

    返回：
        {n, dist:{乐观:{count,pct},中性:{...},悲观:{...}},
         weighted_dist:{乐观:pct,中性:pct,悲观:pct},
         unknown_count,
         confidence:{level:'低/中/高', reason},
         method:'aggregate_v1'}
    n=0（空样本）→ {'n': 0, 'note': '样本为空'}。
    """
    try:
        counts: Dict[str, int] = {b: 0 for b in BUCKETS}
        weights: Dict[str, float] = {b: 0.0 for b in BUCKETS}
        unknown_count = 0
        n = 0
        for item in items or []:
            if not isinstance(item, dict):
                continue  # 防御：非字典条目跳过（不计入 n）
            n += 1
            bucket = _norm_label(item.get("sentiment"))
            if bucket is None:
                bucket = BUCKET_NEU  # 未知标签归入中性
                unknown_count += 1
            counts[bucket] += 1
            weights[bucket] += _item_likes(item, weight_likes_key) + 1.0

        if n == 0:
            return {"n": 0, "note": "样本为空"}

        dist = {b: {"count": counts[b], "pct": _pct(counts[b], n)}
                for b in BUCKETS}
        total_weight = sum(weights.values())
        weighted_dist = {b: _pct(weights[b], total_weight) for b in BUCKETS}

        if n < CONFIDENCE_LOW_MAX:
            level = "低"
            reason = f"样本量 n={n} < {CONFIDENCE_LOW_MAX}，样本不足，分布仅供参考"
        elif n < CONFIDENCE_HIGH_MIN:
            level = "中"
            reason = (f"样本量 {CONFIDENCE_LOW_MAX} ≤ n={n} < "
                      f"{CONFIDENCE_HIGH_MIN}，分布有参考价值")
        else:
            level = "高"
            reason = f"样本量 n={n} ≥ {CONFIDENCE_HIGH_MIN}，分布置信度高"

        return {
            "n": n,
            "dist": dist,
            "weighted_dist": weighted_dist,
            "unknown_count": unknown_count,
            "confidence": {"level": level, "reason": reason},
            "method": METHOD,
        }
    except Exception as e:  # 铁律：绝不抛
        logger.warning("aggregate_distribution 异常（降级空样本）: %s", e,
                       exc_info=True)
        return {"n": 0, "note": "样本为空"}


# ═══════════════════════════════════════════
# 2. 代表性样本抽取
# ═══════════════════════════════════════════

def _slim_item(item: dict, likes_key: str = "likes") -> dict:
    """瘦身代表性样本：title（空则 content）截 80 字 + likes + platform + url。"""
    title = _clean_str(item.get("title"))
    text = title if title else _clean_str(item.get("content"))
    return {
        "text": text[:80],
        "likes": _item_likes(item, likes_key),
        "platform": _clean_str(item.get("platform")),
        "url": _clean_str(item.get("url")),
    }


def pick_representatives(items, per_bucket: int = 2,
                         weight_likes_key: str = "likes") -> dict:
    """每个情感桶（乐观/悲观/中性）按 likes 降序取前 per_bucket 条代表性样本。

    标签归并与 aggregate_distribution 一致（利好→乐观、利空→悲观、
    未知→中性）。样本瘦身见 _slim_item。任何异常降级为空桶，绝不抛。
    返回 {'乐观': [...], '悲观': [...], '中性': [...]}。
    """
    result: Dict[str, List[dict]] = {BUCKET_POS: [], BUCKET_NEG: [], BUCKET_NEU: []}
    try:
        per = max(0, _to_int(per_bucket, 2))
        buckets: Dict[str, List[dict]] = {b: [] for b in BUCKETS}
        for item in items or []:
            if not isinstance(item, dict):
                continue
            bucket = _norm_label(item.get("sentiment")) or BUCKET_NEU
            buckets[bucket].append(item)
        for bucket in (BUCKET_POS, BUCKET_NEG, BUCKET_NEU):
            ranked = sorted(
                buckets[bucket],
                key=lambda it: _item_likes(it, weight_likes_key),
                reverse=True,
            )
            result[bucket] = [_slim_item(it, weight_likes_key)
                              for it in ranked[:per]]
        return result
    except Exception as e:  # 铁律：绝不抛
        logger.warning("pick_representatives 异常（降级空桶）: %s", e,
                       exc_info=True)
        return {BUCKET_POS: [], BUCKET_NEG: [], BUCKET_NEU: []}


# ═══════════════════════════════════════════
# 3. 快照存储与趋势（复用 social.db）
# ═══════════════════════════════════════════

def _resolve_db_path(db_path: Optional[str] = None) -> str:
    """路径解析：db_path 参数 > env SOCIAL_DB_PATH > ${DATA_DIR:-data}/social.db
    （复用 social_store 的解析逻辑，保证同库）。"""
    return social_store._resolve_db_path(db_path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_snapshot_table(conn: sqlite3.Connection) -> None:
    conn.execute(_SNAPSHOT_SCHEMA_SQL)


def _snapshot_numbers(snapshot: dict) -> dict:
    """从快照 dict 提取落盘数值，兼容两种形态（绝不抛）：

    1. 扁平：{platform, target, date, n, pos, neu, neg, w_pos, w_neu, w_neg}
    2. aggregate_distribution 输出形态：dist.{乐观,中性,悲观}.pct 与
       weighted_dist.{乐观,中性,悲观} 自动映射到 pos/neu/neg、w_*。
    """
    snap = snapshot if isinstance(snapshot, dict) else {}
    dist = snap.get("dist") if isinstance(snap.get("dist"), dict) else {}
    wdist = (snap.get("weighted_dist")
             if isinstance(snap.get("weighted_dist"), dict) else {})

    def _dist_pct(bucket: str) -> float:
        entry = dist.get(bucket)
        if isinstance(entry, dict):
            return _to_float(entry.get("pct"), 0.0)
        return _to_float(entry, 0.0)

    pos = snap.get("pos")
    neu = snap.get("neu")
    neg = snap.get("neg")
    return {
        "platform": _clean_str(snap.get("platform")),
        "target": _clean_str(snap.get("target")),
        "date": _clean_str(snap.get("date")),
        "n": max(0, _to_int(snap.get("n"), 0)),
        "pos": _to_float(pos, _dist_pct(BUCKET_POS)) if pos is not None
               else _dist_pct(BUCKET_POS),
        "neu": _to_float(neu, _dist_pct(BUCKET_NEU)) if neu is not None
               else _dist_pct(BUCKET_NEU),
        "neg": _to_float(neg, _dist_pct(BUCKET_NEG)) if neg is not None
               else _dist_pct(BUCKET_NEG),
        "w_pos": _to_float(snap.get("w_pos"),
                           _to_float(wdist.get(BUCKET_POS), 0.0)),
        "w_neu": _to_float(snap.get("w_neu"),
                           _to_float(wdist.get(BUCKET_NEU), 0.0)),
        "w_neg": _to_float(snap.get("w_neg"),
                           _to_float(wdist.get(BUCKET_NEG), 0.0)),
    }


def save_snapshot(snapshot, db_path: Optional[str] = None) -> bool:
    """落盘一条情绪快照（幂等 REPLACE，主键 platform+target+date）。

    路径解析：db_path > env SOCIAL_DB_PATH > ${DATA_DIR:-data}/social.db。
    缺 platform/target/date 记 warning 返回 False；任何 SQL 异常返回 False，
    绝不抛。成功返回 True。
    """
    try:
        nums = _snapshot_numbers(snapshot)
        if not nums["platform"] or not nums["target"] or not nums["date"]:
            logger.warning("save_snapshot 缺 platform/target/date，跳过: %r",
                           snapshot)
            return False
        path = _resolve_db_path(db_path)
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with sqlite3.connect(path) as conn:
            _ensure_snapshot_table(conn)
            conn.execute(
                "INSERT OR REPLACE INTO sentiment_snapshots "
                "(platform, target, date, n, pos, neu, neg, "
                "w_pos, w_neu, w_neg, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (nums["platform"], nums["target"], nums["date"], nums["n"],
                 nums["pos"], nums["neu"], nums["neg"],
                 nums["w_pos"], nums["w_neu"], nums["w_neg"], _now_iso()),
            )
        return True
    except Exception as e:  # 铁律：绝不抛
        logger.warning("save_snapshot 落盘失败（返回 False）: %s", e,
                       exc_info=True)
        return False


def get_trend(platform, target, days: int = 7,
              db_path: Optional[str] = None) -> dict:
    """读取情绪快照序列并判定趋势方向（绝不抛，异常降级安全值）。

    返回 {
        series: [{date, n, dist:{乐观,中性,悲观},
                  weighted:{乐观,中性,悲观}}]（按日期升序，至多 days 条）,
        direction: '转向乐观' / '转向悲观' / '基本稳定' / '数据不足',
        note: str,
    }
    direction 判定：取近 2 个有效快照（n>0），乐观占比差 > +15pct → 转向乐观，
    < -15pct → 转向悲观，否则基本稳定；有效快照 < 2 个 → 数据不足。
    """
    safe = {"series": [], "direction": "数据不足", "note": ""}
    try:
        plat = _clean_str(platform)
        tgt = _clean_str(target)
        if not plat or not tgt:
            safe["note"] = "platform/target 为空，无法查询趋势"
            return safe
        days_int = _to_int(days, 7)
        if days_int <= 0:
            days_int = 7
        path = _resolve_db_path(db_path)
        if not os.path.exists(path):
            safe["note"] = "快照库不存在，暂无历史数据"
            return safe
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            _ensure_snapshot_table(conn)
            rows = conn.execute(
                "SELECT * FROM sentiment_snapshots "
                "WHERE platform = ? AND target = ? "
                "ORDER BY date DESC LIMIT ?",
                (plat, tgt, days_int),
            ).fetchall()

        series: List[dict] = []
        for row in reversed(rows):  # 升序
            d = dict(row)
            series.append({
                "date": _clean_str(d.get("date")),
                "n": _to_int(d.get("n"), 0),
                "dist": {
                    BUCKET_POS: _to_float(d.get("pos"), 0.0),
                    BUCKET_NEU: _to_float(d.get("neu"), 0.0),
                    BUCKET_NEG: _to_float(d.get("neg"), 0.0),
                },
                "weighted": {
                    BUCKET_POS: _to_float(d.get("w_pos"), 0.0),
                    BUCKET_NEU: _to_float(d.get("w_neu"), 0.0),
                    BUCKET_NEG: _to_float(d.get("w_neg"), 0.0),
                },
            })
        safe["series"] = series

        valid = [s for s in series if s["n"] > 0]
        if len(valid) < 2:
            safe["direction"] = "数据不足"
            safe["note"] = (f"有效快照 {len(valid)} 个（<2），"
                            "不足以判定趋势")
            return safe
        prev, latest = valid[-2], valid[-1]
        diff = round(latest["dist"][BUCKET_POS] - prev["dist"][BUCKET_POS], 1)
        if diff > TREND_SHIFT_PCT:
            safe["direction"] = "转向乐观"
        elif diff < -TREND_SHIFT_PCT:
            safe["direction"] = "转向悲观"
        else:
            safe["direction"] = "基本稳定"
        safe["note"] = (f"近两个有效快照乐观占比变化 {diff:+.1f}pct"
                        f"（{prev['date']} → {latest['date']}），"
                        f"阈值 ±{TREND_SHIFT_PCT:.0f}pct")
        return safe
    except Exception as e:  # 铁律：绝不抛
        logger.warning("get_trend 查询失败（降级安全值）: %s", e, exc_info=True)
        safe["note"] = "快照查询异常，已降级为空序列"
        return safe
