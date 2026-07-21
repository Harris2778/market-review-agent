"""
事后打分层：自我问责系统的评分引擎（纯 stdlib，零网络，可独立测试）。

第四波『自我问责系统』组成：
- 存档层（agent/archive.py，另一工程师并行开发）：把每次分析产出追加为 JSONL。
- 打分层（本模块）：对距今足够久的存档记录，取实际行情涨跌幅做事后核对，
  写回 score / scored_at / score_note。

存档格式（与存档层约定一致；本模块独立实现同格式读写，不 import archive.py）：
- 目录解析：ARCHIVE_DIR 显式设置时优先；缺省推导为 ${DATA_DIR:-data}/archive
  （DATA_DIR 为全项目统一的数据根目录约定，Railway 挂卷后设 DATA_DIR=/data，
  见 DEPLOY.md）。目录解析时若在 Railway 上未挂卷（RAILWAY_ENVIRONMENT 存在
  且最终目录不以 RAILWAY_VOLUME_PREFIX /data 开头），logger.warning 提醒
  数据将在重启后丢失，每模块仅警告一次（模块级标志位）。
- 文件 archive_YYYYMMDD.jsonl，每行一个 JSON 对象：
  {"id": str, "ts": str, "trade_date": "YYYYMMDD",
   "mode": "market_review|sector_deep_dive|agent_query",
   "sector": str | None, "content": str, "context_excerpt": str,
   "numbers": list, "score": None | "hit" | "miss" | "neutral",
   "scored_at": None | ISO8601 str, "score_note": None | str}
"""

import glob
import json
import logging
import os
import tempfile
import threading
from datetime import date, datetime

logger = logging.getLogger(__name__)

ARCHIVE_DIR_ENV = "ARCHIVE_DIR"
DATA_DIR_ENV = "DATA_DIR"
DEFAULT_DATA_DIR = "data"
# 旧缺省常量（= os.path.join(DEFAULT_DATA_DIR, "archive")），保留以兼容外部引用；
# 新代码请用 default_archive_dir() 动态推导
DEFAULT_ARCHIVE_DIR = os.path.join("data", "archive")
RAILWAY_ENV_ENV = "RAILWAY_ENVIRONMENT"
# Railway 挂载卷路径前缀判定：最终目录以此开头才视为已挂卷持久化。
# 判定规则集中在这一处常量，如需调整（例如更换挂载路径）改这里即可。
RAILWAY_VOLUME_PREFIX = "/data"
ARCHIVE_FILE_GLOB = "archive_*.jsonl"

# 临时存储警告只发一次的模块级标志位（测试可用 monkeypatch 重置）
_EPHEMERAL_WARNED = False

# 命中/落空的方向性阈值：实际区间涨跌幅绝对值须严格大于 1% 才计 hit/miss。
HIT_THRESHOLD_PCT = 1.0

# 方向词典（见 extract_direction docstring 的完整规则）。
_BULLISH_WORDS = ("偏多", "乐观", "强势", "看好")
_BEARISH_WORDS = ("偏空", "谨慎", "弱势", "回避")
# 优先扫描的标记词：出现这些词的语句被认为承载了最终判断。
_PRIORITY_MARKERS = ("综合判断", "总体")


def _data_dir() -> str:
    """数据根目录：环境变量 DATA_DIR，缺省 "data"（空串回退缺省）。"""
    return os.getenv(DATA_DIR_ENV) or DEFAULT_DATA_DIR


def _warn_if_ephemeral_storage(path) -> None:
    """Railway 临时存储警告：未挂卷时提醒数据将在重启后丢失（每模块仅警告一次）。

    判定规则：环境变量 RAILWAY_ENVIRONMENT 存在（Railway 运行时自动注入），
    且最终目录不以 RAILWAY_VOLUME_PREFIX（默认 "/data"，挂载卷路径前缀）开头。
    挂载路径前缀的判定集中在 RAILWAY_VOLUME_PREFIX 常量，如需调整改该常量。
    本函数自身 fail-safe：任何异常静默吞掉，绝不影响目录解析主流程。
    """
    global _EPHEMERAL_WARNED
    if _EPHEMERAL_WARNED:
        return
    try:
        if not os.getenv(RAILWAY_ENV_ENV):
            return
        if str(path).startswith(RAILWAY_VOLUME_PREFIX):
            return
        logger.warning(
            "运行在 Railway 但未挂卷，重启后数据将丢失"
            "（当前存档目录=%r，不在挂载卷路径 %s 下；"
            "请挂载 Volume 到 %s 并设置 %s=%s，详见 DEPLOY.md）",
            path, RAILWAY_VOLUME_PREFIX,
            RAILWAY_VOLUME_PREFIX, DATA_DIR_ENV, RAILWAY_VOLUME_PREFIX,
        )
        _EPHEMERAL_WARNED = True
    except Exception:
        pass


def default_archive_dir() -> str:
    """存档目录：ARCHIVE_DIR 显式设置优先，缺省推导为 ${DATA_DIR:-data}/archive。"""
    path = os.getenv(ARCHIVE_DIR_ENV) or os.path.join(_data_dir(), "archive")
    _warn_if_ephemeral_storage(path)
    return path


def _parse_trade_date(value):
    """"YYYYMMDD" -> date；非法/缺失返回 None。"""
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value.strip(), "%Y%m%d").date()
    except ValueError:
        return None


def find_pending(records, days=5, today=None):
    """筛选待打分记录。

    规则：score 为 null 且 trade_date 距今（自然日）>= days 天。
    - trade_date 缺失或格式非法的记录不视为 pending（无法定位行情窗口）。
    - today 参数用于测试注入"今天"，默认取系统当天日期。
    """
    if today is None:
        today = date.today()
    pending = []
    for rec in records:
        if rec.get("score") is not None:
            continue
        td = _parse_trade_date(rec.get("trade_date"))
        if td is None:
            continue
        if (today - td).days >= days:
            pending.append(rec)
    return pending


def _count_direction_words(text, words):
    return sum(text.count(w) for w in words)


def _judge_scope(text):
    """在给定文本范围内按词频裁决方向。无方向词返回 None，平局返回 "neutral"。"""
    bull = _count_direction_words(text, _BULLISH_WORDS)
    bear = _count_direction_words(text, _BEARISH_WORDS)
    if bull == 0 and bear == 0:
        return None
    if bull > bear:
        return "bullish"
    if bear > bull:
        return "bearish"
    return "neutral"


def extract_direction(content):
    """从分析全文启发式提取方向判断，返回 "bullish" | "bearish" | "neutral"。

    规则（按优先级）：
    1. 优先范围：含『综合判断』或『总体』的行，连同其下一行（标题与结论常分两行）。
       在该范围内统计方向词，词频多者胜出；有方向词但多空平局 → "neutral"。
    2. 兜底范围：优先范围内没有任何方向词时，对全文做同样的词频裁决。
    3. 全文也无方向词 → "neutral"。

    方向词典：偏多/乐观/强势/看好 → bullish；偏空/谨慎/弱势/回避 → bearish。
    注意：按词出现次数计数（同一词重复出现累加）；多空打平一律保守记 neutral，
    因为模棱两可的判断不应参与 hit/miss 问责。
    """
    if not content:
        return "neutral"
    lines = content.splitlines()
    priority_chunks = []
    for i, line in enumerate(lines):
        if any(marker in line for marker in _PRIORITY_MARKERS):
            chunk = line
            if i + 1 < len(lines):
                chunk = chunk + "\n" + lines[i + 1]
            priority_chunks.append(chunk)
    if priority_chunks:
        verdict = _judge_scope("\n".join(priority_chunks))
        if verdict is not None:
            return verdict
    verdict = _judge_scope(content)
    return verdict if verdict is not None else "neutral"


def score_record(record, pct_change_5d):
    """对单条记录打分，返回 (score, note)。

    判定矩阵（pct 为实际区间涨跌幅百分比，阈值 ±1% 严格不等）：
    - pct > +1% 且方向 bullish → "hit"；pct < -1% 且方向 bearish → "hit"
    - pct < -1% 且方向 bullish → "miss"；pct > +1% 且方向 bearish → "miss"
    - 其余一律 "neutral"：方向 neutral、或涨跌幅落在 ±1% 区间（含恰为 ±1%）。
    """
    direction = extract_direction(record.get("content") or "")
    pct = float(pct_change_5d)
    if pct > HIT_THRESHOLD_PCT and direction == "bullish":
        score = "hit"
    elif pct < -HIT_THRESHOLD_PCT and direction == "bearish":
        score = "hit"
    elif pct < -HIT_THRESHOLD_PCT and direction == "bullish":
        score = "miss"
    elif pct > HIT_THRESHOLD_PCT and direction == "bearish":
        score = "miss"
    else:
        score = "neutral"
    note = (
        f"方向判断={direction}；实际区间涨跌幅={pct:+.2f}%；"
        f"阈值=±{HIT_THRESHOLD_PCT:.1f}%（严格大于才计 hit/miss）→ {score}"
    )
    return score, note


def _read_jsonl(path):
    """读 JSONL 文件，返回 (records, bad_lines)。坏行按原样保留以便回写不丢数据。"""
    records = []
    bad_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                bad_lines.append(stripped)
    return records, bad_lines


def _write_jsonl(path, records, bad_lines):
    """原子回写：先写临时文件再 os.replace，避免中途崩溃截断存档。

    写入前 makedirs exist_ok 补强：目录不存在时自动创建（挂载卷首次写入
    或目录被外部清理的场景），已存在则不动。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=".archive_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for line in bad_lines:
                f.write(line + "\n")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def apply_scores(archive_dir, pct_fn, days=5, writer_lock=None, today=None):
    """遍历存档目录，对 pending 记录逐条打分并回写。

    流程（读全量 → 改写文件，threading.Lock 保护回写）：
    1. 按文件名排序遍历 archive_*.jsonl；
    2. find_pending 筛出待打分记录；
    3. 对每条 pending 调 pct_fn(record) -> float | None 取实际区间涨跌幅；
       返回 None（行情数据还取不到，例如未满 5 个交易日）则跳过、不写回；
    4. score_record 得出 hit/miss/neutral，写回 score/scored_at/score_note；
    5. 仅当文件有改动才回写（原子替换）；坏行原样保留。

    writer_lock：可选的外部 threading.Lock（便于与存档层共用同一把锁）；
    缺省由本函数自建。返回 {"scored": [...], "skipped": [...], "files_rewritten": int}，
    scored/skipped 元素为 {"id", "trade_date", "mode", "score"|None, "note"|str, "reason"|str}。
    """
    if writer_lock is None:
        writer_lock = threading.Lock()
    scored = []
    skipped = []
    files_rewritten = 0

    pattern = os.path.join(str(archive_dir), ARCHIVE_FILE_GLOB)
    for path in sorted(glob.glob(pattern)):
        records, bad_lines = _read_jsonl(path)
        pending = find_pending(records, days=days, today=today)
        if not pending:
            continue
        dirty = False
        for rec in pending:
            pct = pct_fn(rec)
            if pct is None:
                skipped.append({
                    "id": rec.get("id"),
                    "trade_date": rec.get("trade_date"),
                    "mode": rec.get("mode"),
                    "reason": "pct_fn 返回 None（实际行情暂不可取），跳过不写回",
                })
                continue
            score, note = score_record(rec, pct)
            rec["score"] = score
            rec["scored_at"] = datetime.now().isoformat(timespec="seconds")
            rec["score_note"] = note
            dirty = True
            scored.append({
                "id": rec.get("id"),
                "trade_date": rec.get("trade_date"),
                "mode": rec.get("mode"),
                "score": score,
                "note": note,
            })
        if dirty:
            with writer_lock:
                _write_jsonl(path, records, bad_lines)
            files_rewritten += 1

    return {"scored": scored, "skipped": skipped, "files_rewritten": files_rewritten}
