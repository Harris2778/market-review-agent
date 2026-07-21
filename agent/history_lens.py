"""
第七波『以史为鉴』历史判断回顾模块（纯 stdlib，零网络，fail-safe）。

职责：把问责存档中的历史判断回注入新分析的 prompt 上下文（以史为鉴），
并为后续 dashboard 提供命中率汇总。

公开契约（供集成者，如 orchestrator / dashboard 梯队）：
- get_history_note(sector=None, mode=None, limit=5) -> str | None
  扫描存档目录全部日文件，筛同 sector（None 则不限）且同 mode（None 则不限）
  的记录，按 trade_date 倒序取 limit 条，格式化为 ≤300 字注入块：
      【以史为鉴：本智能体历史判断回顾】
      MM-DD 方向 打分 一句话依据
      ...
  方向 ∈ {偏多, 偏空, 中性}（从 content 提取，简易规则，参考『综合判断』附近）；
  打分 ∈ {命中, 偏差, 中性, 待评分}（由 score: hit/miss/neutral/null 映射）；
  依据取自 score_note，单行化后截断 40 字（无 score_note 则省略依据段）。
  无匹配记录（或目录不存在）返回 None。
- get_accuracy_summary() -> dict
  全部已打分记录（score ∈ hit/miss/neutral）按 mode 分组的计数与命中率：
      {"total":   {"hit", "miss", "neutral", "scored", "hit_rate"},
       "by_mode": {mode: 同上结构}}
  hit_rate = hit / scored（0~1 浮点，保留 4 位小数；scored 为 0 时为 None）。
  mode 缺失的记录归入 "unknown" 组；未打分记录不计入。

存档格式（与 archive.py / scorer.py 的契约一致；按既有惯例本模块自实现
同格式 JSONL 读取，不 import agent 内其他模块，避免跨梯队耦合）：
- 目录：环境变量 ARCHIVE_DIR 优先；缺省 ${DATA_DIR:-data}/archive，
  每次调用动态读取，便于测试注入与部署切换；
- 文件 archive_YYYYMMDD.jsonl，每行一个 JSON 对象：
  {"id", "ts", "trade_date", "mode", "sector", "content",
   "context_excerpt", "numbers",
   "score": null | "hit" | "miss" | "neutral",
   "scored_at", "score_note"}

设计说明：
- 全部 fail-safe：目录不存在、文件不可读、坏行（非法 JSON / 非 dict）
  一律跳过或返回空结果；任何未预期异常只记 log，get_history_note 返回
  None、get_accuracy_summary 返回全零结构，绝不抛出。
- 300 字上限：逐行累加，加入下一行将超限即停止（头部长度计入）；
  连一条记录行都放不下时返回 None（防御性，正常配置不会发生）。
"""

import glob
import json
import logging
import os

logger = logging.getLogger(__name__)

ARCHIVE_DIR_ENV = "ARCHIVE_DIR"
DATA_DIR_ENV = "DATA_DIR"
DEFAULT_DATA_DIR = "data"
ARCHIVE_FILE_GLOB = "archive_*.jsonl"

HEADER = "【以史为鉴：本智能体历史判断回顾】"
MAX_NOTE_CHARS = 300
BASIS_MAX_CHARS = 40
DEFAULT_LIMIT = 5

# score → 展示标签；None 及任何未知值一律显示「待评分」
_SCORE_LABELS = {"hit": "命中", "miss": "偏差", "neutral": "中性"}
_PENDING_LABEL = "待评分"

# 方向词典与优先标记：与 scorer.extract_direction 的判定惯例保持一致，
# 仅输出端改为中文标签（偏多/偏空/中性）。
_BULLISH_WORDS = ("偏多", "乐观", "强势", "看好")
_BEARISH_WORDS = ("偏空", "谨慎", "弱势", "回避")
_PRIORITY_MARKERS = ("综合判断", "总体")


def _archive_dir() -> str:
    """存档目录：ARCHIVE_DIR 优先，缺省 ${DATA_DIR:-data}/archive。"""
    archive_dir = os.getenv(ARCHIVE_DIR_ENV)
    if archive_dir:
        return archive_dir
    return os.path.join(os.getenv(DATA_DIR_ENV, DEFAULT_DATA_DIR), "archive")


def _iter_records(archive_dir: str):
    """遍历存档目录全部日文件，yield 每条 dict 记录。

    坏行（JSON 解析失败 / 非 dict）记 log 后跳过；文件不可读整文件跳过；
    目录不存在时不产生任何记录。
    """
    pattern = os.path.join(str(archive_dir), ARCHIVE_FILE_GLOB)
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        obj = json.loads(stripped)
                    except json.JSONDecodeError:
                        logger.warning(
                            "存档文件 %s 第 %d 行 JSON 解析失败（跳过）", path, lineno)
                        continue
                    if isinstance(obj, dict):
                        yield obj
                    else:
                        logger.warning(
                            "存档文件 %s 第 %d 行非 JSON 对象（跳过）", path, lineno)
        except OSError as e:
            logger.warning("存档文件读取失败 %s（整文件跳过）: %s", path, e, exc_info=True)
            continue


def _sort_key(rec: dict) -> tuple:
    """排序键：trade_date 为主、ts 为辅（均为字符串比较，YYYYMMDD/ISO8601 天然有序）。"""
    return (str(rec.get("trade_date") or ""), str(rec.get("ts") or ""))


def _mm_dd(trade_date) -> str:
    """"YYYYMMDD" -> "MM-DD"；非法/缺失返回 "??-??"。"""
    s = str(trade_date or "").strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[4:6]}-{s[6:8]}"
    return "??-??"


def _count_words(text: str, words) -> int:
    return sum(text.count(w) for w in words)


def _judge_scope(text: str):
    """在文本范围内按方向词词频裁决：无方向词返回 None，平局返回 "中性"。"""
    bull = _count_words(text, _BULLISH_WORDS)
    bear = _count_words(text, _BEARISH_WORDS)
    if bull == 0 and bear == 0:
        return None
    if bull > bear:
        return "偏多"
    if bear > bull:
        return "偏空"
    return "中性"


def _extract_direction_label(content) -> str:
    """从分析全文提取方向标签："偏多" | "偏空" | "中性"。

    规则（与 scorer 方向判定惯例一致）：
    1. 优先范围：含『综合判断』或『总体』的行连同其下一行，词频多者胜出；
       有方向词但多空平局 → "中性"。
    2. 兜底范围：优先范围内无方向词时对全文裁决。
    3. 全文也无方向词 → "中性"。
    """
    if not content or not isinstance(content, str):
        return "中性"
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
    return verdict if verdict is not None else "中性"


def _one_line(text) -> str:
    """单行化：压扁换行与连续空白，便于注入块保持一行一条。"""
    return " ".join(str(text).split())


def _format_line(rec: dict) -> str:
    """格式化单条记录：『MM-DD 方向 打分 一句话依据』（依据截断 40 字，可省略）。"""
    mm_dd = _mm_dd(rec.get("trade_date"))
    direction = _extract_direction_label(rec.get("content"))
    label = _SCORE_LABELS.get(rec.get("score"), _PENDING_LABEL)
    note = _one_line(rec.get("score_note") or "")[:BASIS_MAX_CHARS]
    if note:
        return f"{mm_dd} {direction} {label} {note}"
    return f"{mm_dd} {direction} {label}"


def _normalize_limit(limit) -> int:
    """limit 兜底：非正整数/不可解析一律回退 DEFAULT_LIMIT。"""
    try:
        n = int(limit)
    except (TypeError, ValueError):
        return DEFAULT_LIMIT
    return n if n > 0 else DEFAULT_LIMIT


def get_history_note(sector=None, mode=None, limit=5):
    """生成『以史为鉴』注入块（≤300 字），无匹配记录或任何失败返回 None。

    sector: 板块名精确匹配；None（或空串）则不限板块。
    mode:   "market_review" | "sector_deep_dive" | "agent_query"；None（或空串）不限。
    limit:  最多纳入的记录条数（按 trade_date 倒序），非法值回退 5。
    """
    try:
        records = []
        for rec in _iter_records(_archive_dir()):
            if sector and rec.get("sector") != sector:
                continue
            if mode and rec.get("mode") != mode:
                continue
            records.append(rec)
        if not records:
            return None
        records.sort(key=_sort_key, reverse=True)
        records = records[:_normalize_limit(limit)]

        parts = [HEADER]
        used = len(HEADER)
        for rec in records:
            line = _format_line(rec)
            extra = 1 + len(line)  # +1 为行间换行符
            if used + extra > MAX_NOTE_CHARS:
                break
            parts.append(line)
            used += extra
        if len(parts) == 1:
            # 防御性：连一条记录行都放不下时不出半空注入块
            return None
        return "\n".join(parts)
    except Exception as e:
        logger.warning("以史为鉴注入块生成失败（返回 None）: %s", e, exc_info=True)
        return None


def get_accuracy_summary() -> dict:
    """已打分记录按 mode 分组的 hit/miss/neutral 计数与命中率（供 dashboard 用）。

    返回 {"total": bucket, "by_mode": {mode: bucket}}；
    bucket = {"hit", "miss", "neutral", "scored", "hit_rate"}，
    hit_rate = hit / scored（保留 4 位小数；scored 为 0 时为 None）。
    任何失败返回全零结构，绝不抛出。
    """
    def _bucket():
        return {"hit": 0, "miss": 0, "neutral": 0, "scored": 0, "hit_rate": None}

    total = _bucket()
    by_mode = {}
    try:
        for rec in _iter_records(_archive_dir()):
            score = rec.get("score")
            if score not in _SCORE_LABELS:
                continue  # 未打分（None）或未知值不计入问责统计
            mode = rec.get("mode") or "unknown"
            bucket = by_mode.setdefault(mode, _bucket())
            for b in (total, bucket):
                b[score] += 1
                b["scored"] += 1
        for b in [total, *by_mode.values()]:
            if b["scored"] > 0:
                b["hit_rate"] = round(b["hit"] / b["scored"], 4)
    except Exception as e:
        logger.warning("命中率汇总失败（返回全零结构）: %s", e, exc_info=True)
    return {"total": total, "by_mode": by_mode}
