#!/usr/bin/env python3
"""课程点评综合总结模块（校园知识库 · thucourse_summary 生成器）。

职责：给定一门课程的全部点评条目（thucourse_review 条目字典列表，
字段遵循全局契约 8 字段），生成一段全面的中文综合总结，并构造
source='thucourse_summary' 的 kb 条目供 upsert 落库。

公开 API（全局契约，不得擅自改动）：
- summarize_course_reviews(course_title, reviews, llm_fn=None) -> dict
    返回 {summary_text, rating_avg, rating_dist, review_count,
          highlights: list[str], method: 'llm'|'fallback'}
- build_summary_entry(course_sqid, course_title, summary_dict) -> dict
    生成 thucourse_summary 条目，source_id='thucourse:summary:{sqid}'

设计要点：
1. llm_fn 可注入（签名 llm_fn(prompt: str) -> str）。注入时走 LLM 路径：
   构造含课程名、评分分布、点评原文（超长按评分分层抽样控制在
   PROMPT_REVIEW_CHAR_BUDGET≈6000 字内）的 prompt，要求 LLM 严格输出
   JSON {"summary_text": ..., "highlights": [...]}（须覆盖
   工作量/给分/教学质量/考核方式/适合人群 五维度）。LLM 调用抛异常或
   返回解析失败时自动降级 fallback 确定性路径。
2. fallback 路径不依赖 jieba：标点切句 + 中文 2~4 gram / 英文词
   词频统计（按句文档频率），按评分分层（高/中/低/无评分）挑选
   代表性句子 3~5 条作为 highlights，模板化生成总结文本，
   明确标注「基于 N 条点评的自动摘要」。
3. 评分归一：从条目 metadata_json 中提取 rating/score/stars/rate/评分/分数
   数值；>5 且 ≤10 视为十分制折半归一到五分制；其余非法值忽略。
4. 绝不抛异常：空/None/畸形输入返回 review_count=0 的空总结结构。
"""

import json
import logging
import math
import re
from collections import Counter
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── 全局常量 ──

# LLM prompt 中点评原文的字符预算（超长时按评分分层抽样截断）
PROMPT_REVIEW_CHAR_BUDGET = 6000
# 单条点评进入 prompt 的截断长度
PROMPT_REVIEW_SNIPPET_MAX = 800
# fallback highlights 条数区间
HIGHLIGHTS_MIN = 3
HIGHLIGHTS_MAX = 5
# 单条 highlight 展示截断长度
HIGHLIGHT_TEXT_MAX = 120

# LLM 输出要求覆盖的五维度（仅用于 prompt 文案与文档说明）
FIVE_DIMENSIONS = ("工作量", "给分", "教学质量", "考核方式", "适合人群")

# metadata_json 中评分的候选键
_RATING_KEYS = ("rating", "score", "stars", "rate", "评分", "分数")

# 标点切句
_SENT_SPLIT_RE = re.compile(r"[。！？!?；;…\n]+")
# 连续中文段 / 连续英文数字词
_CJK_RUN_RE = re.compile(r"[一-鿿]+")
_ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*")

# 极简停用词（虚词/无信息量代词；不含领域词，避免误伤观点词）
_STOPWORDS = frozenset(
    "的 了 是 我 你 他 她 它 们 和 与 在 就 都 很 还 但 不 有 个 会 这 那 "
    "之 其 及 或 而 且 啊 呢 吧 嘛 哦 嗯 也 又 再 最 太 更 被 把 对 向 从 "
    "因为 所以 但是 不过 然后 就是 真的 觉得 感觉 比较 一个 一些 没有 什么 "
    "怎么 这样 那样 这个 那个 可以 可能 应该 还是 虽然 如果 的话 自己 大家 "
    "以及 或者 而且 因此 并且 只是 只有 甚至 已经 一直 非常 十分 相当 特别"
    .split()
)

LlmFn = Callable[[str], str]


# ═══════════════════════════════════════════
# 纯函数工具：评分 / 切句 / n-gram
# ═══════════════════════════════════════════

def _parse_metadata(entry: dict) -> dict:
    """条目 metadata_json（JSON 字符串）容错解析；任何失败返回 {}。"""
    raw = entry.get("metadata_json")
    if isinstance(raw, dict):  # 容忍直接给 dict 的调用方
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_rating(entry: dict) -> Optional[float]:
    """从条目 metadata_json 提取评分并归一到五分制。

    >5 且 ≤10 视为十分制折半；∈(0,5] 原样；其余（缺失/0/负/越界/非数值）
    返回 None。布尔值不算评分。"""
    meta = _parse_metadata(entry)
    for key in _RATING_KEYS:
        if key not in meta:
            continue
        value = meta.get(key)
        if isinstance(value, bool):
            continue
        try:
            rating = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(rating) or math.isinf(rating):
            continue
        if 5 < rating <= 10:
            rating = rating / 2.0
        if 0 < rating <= 5:
            return rating
    return None


def _rating_tier(rating: Optional[float]) -> str:
    """评分分层：high(≥4) / mid(≥3) / low(<3) / none(无评分)。"""
    if rating is None:
        return "none"
    if rating >= 4:
        return "high"
    if rating >= 3:
        return "mid"
    return "low"


def _rating_stats(reviews: List[dict]) -> dict:
    """评分统计：avg（两位小数，无评分 None）+ dist（"1".."5" 计数）+ count。"""
    ratings = [r for r in (_extract_rating(e) for e in reviews) if r is not None]
    dist = {str(i): 0 for i in range(1, 6)}
    for r in ratings:
        bucket = min(5, max(1, int(round(r))))
        dist[str(bucket)] += 1
    dist = {k: v for k, v in dist.items() if v > 0}
    avg = round(sum(ratings) / len(ratings), 2) if ratings else None
    return {"rating_avg": avg, "rating_dist": dist, "rating_count": len(ratings)}


def _split_sentences(text: str) -> List[str]:
    """按中英文标点/换行切句；过滤过短句（<6 字，信息量不足）。"""
    sentences = []
    for seg in _SENT_SPLIT_RE.split(text or ""):
        seg = seg.strip(" \t，,、：:\"'“”‘’（）()")
        if len(seg) >= 6:
            sentences.append(seg)
    return sentences


def _ngrams(sentence: str) -> List[str]:
    """jieba 不可用时的简易分词：中文 2~4 gram + 英文词小写。
    停用词与单字由调用方过滤。"""
    tokens: List[str] = []
    for run in _CJK_RUN_RE.findall(sentence):
        for n in (2, 3, 4):
            if len(run) >= n:
                tokens.extend(run[i:i + n] for i in range(len(run) - n + 1))
    tokens.extend(w.lower() for w in _ASCII_WORD_RE.findall(sentence))
    return [t for t in tokens if t not in _STOPWORDS]


# ═══════════════════════════════════════════
# fallback 确定性摘要
# ═══════════════════════════════════════════

def _pick_highlights(reviews: List[dict]) -> Tuple[List[str], List[str]]:
    """词频 + 评分分层挑选代表性句子。

    返回 (highlights, keywords)：
    - 词频按「句文档频率」统计（一句内重复只计一次，避免长文刷频）；
    - 句子得分 = 句内词频(≥2)词得分之和 / (1+log10(句长))，
      得分相同按句子文本字典序稳定排序（确定性）；
    - 按 high → mid → low → none 分层轮转挑选，去重后取 3~5 条。
    """
    freq: Counter = Counter()
    tiered: Dict[str, List[Tuple[str, float]]] = {
        t: [] for t in ("high", "mid", "low", "none")}
    seen = set()
    for entry in reviews:
        content = str(entry.get("content") or "")
        tier = _rating_tier(_extract_rating(entry))
        for sent in _split_sentences(content):
            if sent in seen:
                continue
            seen.add(sent)
            grams = _ngrams(sent)
            freq.update(set(grams))
            tiered[tier].append((sent, grams))  # type: ignore[arg-type]

    keywords = [w for w, c in freq.most_common() if c >= 2][:8]

    def _score(item) -> Tuple[float, str]:
        sent, grams = item
        raw = sum(freq[g] for g in set(grams) if freq[g] >= 2)
        score = raw / (1.0 + math.log10(len(sent) + 1))
        return (-score, sent)  # 负分排序=降序，文本字典序兜底

    for tier in tiered:
        tiered[tier].sort(key=_score)

    picked: List[str] = []
    order = [t for t in ("high", "mid", "low", "none") if tiered[t]]
    idx = 0
    while order and len(picked) < HIGHLIGHTS_MAX:
        tier = order[idx % len(order)]
        if tiered[tier]:
            picked.append(tiered[tier].pop(0)[0])
            if not tiered[tier]:
                order.remove(tier)
                idx %= max(1, len(order))
                continue
        idx += 1
    highlights = [s[:HIGHLIGHT_TEXT_MAX] for s in picked[:HIGHLIGHTS_MAX]]
    return highlights, keywords


def _dist_text(dist: dict) -> str:
    """评分分布 → 紧凑文本，如 '5分×3/4分×1'。"""
    if not dist:
        return ""
    return "/".join(f"{k}分×{v}" for k, v in
                    sorted(dist.items(), key=lambda kv: kv[0], reverse=True))


def _fallback_summary(course_title: str, reviews: List[dict], stats: dict) -> dict:
    """确定性模板摘要：评分统计 + 高频关键词 + 分层代表性观点。"""
    n = len(reviews)
    highlights, keywords = _pick_highlights(reviews)
    title = course_title or "该课程"

    parts = [f"基于 {n} 条点评的自动摘要：《{title}》"]
    avg, dist, rc = stats["rating_avg"], stats["rating_dist"], stats["rating_count"]
    if avg is not None:
        parts.append(f"平均评分 {avg}/5（{rc} 人给出评分，分布：{_dist_text(dist)}）")
    else:
        parts.append("点评未提供结构化评分")
    if keywords:
        parts.append("点评高频提及：" + "、".join(keywords))
    if highlights:
        quoted = "；".join(f"{i + 1}）{h}" for i, h in enumerate(highlights))
        parts.append(f"代表性观点：{quoted}")
    parts.append("（本摘要为程序基于点评文本自动统计生成，不含人工研判，仅供参考）")

    return {
        "summary_text": "。".join(parts),
        "rating_avg": avg,
        "rating_dist": dist,
        "review_count": n,
        "highlights": highlights,
        "method": "fallback",
    }


# ═══════════════════════════════════════════
# LLM 路径
# ═══════════════════════════════════════════

def _sample_reviews_for_prompt(reviews: List[dict]) -> List[Tuple[Optional[float], str]]:
    """点评原文按评分分层轮转抽样（高/中/低/无评分均衡覆盖），
    单条截断 PROMPT_REVIEW_SNIPPET_MAX 字，总量控制在
    PROMPT_REVIEW_CHAR_BUDGET 字内。返回 [(rating, snippet)]，顺序确定。"""
    tiered: Dict[str, List[Tuple[Optional[float], str]]] = {
        t: [] for t in ("high", "mid", "low", "none")}
    for entry in reviews:
        rating = _extract_rating(entry)
        snippet = str(entry.get("content") or "").strip()[:PROMPT_REVIEW_SNIPPET_MAX]
        if snippet:
            tiered[_rating_tier(rating)].append((rating, snippet))

    sampled: List[Tuple[Optional[float], str]] = []
    budget = PROMPT_REVIEW_CHAR_BUDGET
    order = [t for t in ("high", "mid", "low", "none") if tiered[t]]
    idx = 0
    while order and budget > 0:
        tier = order[idx % len(order)]
        if tiered[tier]:
            item = tiered[tier].pop(0)
            if len(item[1]) <= budget:
                sampled.append(item)
                budget -= len(item[1])
            # 超预算的条目丢弃；该层耗尽则移出轮转
            if not tiered[tier]:
                order.remove(tier)
                idx %= max(1, len(order))
                continue
        else:
            order.remove(tier)
            idx %= max(1, len(order))
            continue
        idx += 1
    return sampled


def build_llm_prompt(course_title: str, reviews: List[dict], stats: dict) -> str:
    """构造 LLM 总结 prompt（课程名 + 评分分布 + 分层抽样后的点评原文）。"""
    n = len(reviews)
    avg, dist = stats["rating_avg"], stats["rating_dist"]
    dist_line = (f"平均 {avg}/5，分布 {_dist_text(dist)}"
                 if avg is not None else "无结构化评分")
    sampled = _sample_reviews_for_prompt(reviews)

    lines = [
        "你是课程点评总结助手。请根据以下清华大学课程的学生点评，生成一份综合总结。",
        "",
        f"课程：《{course_title or '未知课程'}》",
        f"点评总数：{n} 条（评分：{dist_line}）",
        "",
        "点评原文：",
    ]
    for i, (rating, snippet) in enumerate(sampled, 1):
        tag = f"评分{rating:g}" if rating is not None else "无评分"
        lines.append(f"【点评{i} · {tag}】{snippet}")
    if len(sampled) < n:
        lines.append(f"（注：点评共 {n} 条，受长度限制按评分分层抽样展示 "
                     f"{len(sampled)} 条）")
    lines += [
        "",
        "总结要求：从以下五个维度综合归纳——工作量、给分、教学质量、考核方式、"
        "适合人群；观点须忠于点评原文，不编造。",
        "严格输出如下 JSON（不要输出任何多余文字或 markdown 代码围栏）：",
        '{"summary_text": "300字以内的中文综合总结，须覆盖上述五维度", '
        '"highlights": ["代表性观点1", "代表性观点2", "代表性观点3"]}',
    ]
    return "\n".join(lines)


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_llm_output(raw) -> Optional[dict]:
    """解析 LLM 返回：剥离代码围栏后提取首个 JSON 对象，校验
    summary_text（非空字符串）与 highlights（字符串列表，缺省 []）。
    任何一步失败返回 None（调用方降级 fallback）。"""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = _FENCE_RE.sub("", raw.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    summary_text = payload.get("summary_text")
    if not isinstance(summary_text, str) or not summary_text.strip():
        return None
    highlights = payload.get("highlights")
    if not isinstance(highlights, list):
        highlights = []
    highlights = [str(h).strip()[:HIGHLIGHT_TEXT_MAX]
                  for h in highlights if str(h).strip()][:HIGHLIGHTS_MAX]
    return {"summary_text": summary_text.strip(), "highlights": highlights}


def _try_llm(course_title: str, reviews: List[dict], stats: dict,
             llm_fn: LlmFn) -> Optional[dict]:
    """LLM 路径：构造 prompt → 调用 → 解析。调用抛异常或解析失败返回 None。"""
    try:
        prompt = build_llm_prompt(course_title, reviews, stats)
    except Exception:
        logger.warning("LLM prompt 构造失败，降级 fallback", exc_info=True)
        return None
    try:
        raw = llm_fn(prompt)
    except Exception:
        logger.warning("llm_fn 调用失败，降级 fallback", exc_info=True)
        return None
    parsed = _parse_llm_output(raw)
    if parsed is None:
        logger.warning("llm_fn 返回解析失败，降级 fallback")
        return None
    return {
        "summary_text": parsed["summary_text"],
        "rating_avg": stats["rating_avg"],
        "rating_dist": stats["rating_dist"],
        "review_count": len(reviews),
        "highlights": parsed["highlights"],
        "method": "llm",
    }


# ═══════════════════════════════════════════
# 公开 API
# ═══════════════════════════════════════════

def _empty_summary(course_title: str) -> dict:
    """空/畸形输入的空总结结构（review_count=0）。"""
    title = course_title or "该课程"
    return {
        "summary_text": (f"基于 0 条点评的自动摘要：《{title}》暂无可用的课程点评，"
                         "无法生成综合总结。（本摘要为程序自动生成）"),
        "rating_avg": None,
        "rating_dist": {},
        "review_count": 0,
        "highlights": [],
        "method": "fallback",
    }


def summarize_course_reviews(course_title: str, reviews: Optional[List[dict]],
                             llm_fn: Optional[LlmFn] = None) -> dict:
    """给定一门课程的全部点评条目，生成综合总结。

    返回 {summary_text, rating_avg, rating_dist, review_count,
          highlights: list[str], method: 'llm'|'fallback'}。
    llm_fn 注入时走 LLM 路径（失败自动降级 fallback）；绝不抛异常，
    空点评列表返回 review_count=0 的空总结结构。"""
    try:
        if not isinstance(course_title, str):
            course_title = str(course_title or "")
        clean = [r for r in (reviews or []) if isinstance(r, dict)]
        if not clean:
            return _empty_summary(course_title)
        stats = _rating_stats(clean)
        if llm_fn is not None:
            result = _try_llm(course_title, clean, stats, llm_fn)
            if result is not None:
                return result
        return _fallback_summary(course_title, clean, stats)
    except Exception:  # 双保险：绝不向调用方抛出
        logger.exception("summarize_course_reviews 意外异常，返回空总结")
        return _empty_summary(course_title if isinstance(course_title, str)
                              else "该课程")


def _now_iso() -> str:
    """本地时区 ISO 时间戳（秒级）。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build_summary_entry(course_sqid, course_title, summary_dict) -> dict:
    """生成 source='thucourse_summary' 的 kb 条目（全局契约 8 字段）。

    source_id='thucourse:summary:{sqid}'；content 取 summary_text；
    评分/条数/highlights/method 等结构化字段放入 metadata_json。
    绝不抛异常：畸形输入返回占位条目。"""
    try:
        sqid = str(course_sqid).strip() if course_sqid is not None else ""
        sqid = sqid or "unknown"
        title = str(course_title or "").strip()
        summary = summary_dict if isinstance(summary_dict, dict) else {}
        metadata = {
            "course_sqid": sqid,
            "course_title": title,
            "rating_avg": summary.get("rating_avg"),
            "rating_dist": summary.get("rating_dist") or {},
            "review_count": summary.get("review_count", 0),
            "highlights": summary.get("highlights") or [],
            "method": summary.get("method", "fallback"),
            "generator": "agent.review_summary",
        }
        return {
            "source": "thucourse_summary",
            "source_id": f"thucourse:summary:{sqid}",
            "title": f"{title} · 点评综合总结" if title else "课程点评综合总结",
            "content": str(summary.get("summary_text") or ""),
            "url": f"thucourse:course:{sqid}",  # 课程页相对引用
            "metadata_json": json.dumps(metadata, ensure_ascii=False),
            "updated_at": _now_iso(),
        }
    except Exception:  # 双保险：绝不向调用方抛出
        logger.exception("build_summary_entry 意外异常，返回占位条目")
        return {
            "source": "thucourse_summary",
            "source_id": "thucourse:summary:unknown",
            "title": "课程点评综合总结",
            "content": "",
            "url": "thucourse:course:unknown",
            "metadata_json": "{}",
            "updated_at": _now_iso(),
        }
