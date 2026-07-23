"""确定性输出护栏（agent/output_guard.py）。

修「残留2：数据未覆盖时 LLM 仍下结论」。COMPLIANCE_PROMPT 的
「数据口径纪律」仅靠提示词仍会被违反，本模块在报告产出后做纯 stdlib
的确定性修复，把「声明了数据未覆盖、却又给出方向性结论」的矛盾输出
改写为只保留未覆盖声明的保守文本。

唯一公开函数：
- fix_uncovered_conclusions(text)  修复未覆盖声明与结论并存的矛盾文本

四条规则（按执行顺序）：
① 同句矛盾：按 。；！？\\n 切句后，一句内同时含未覆盖声明词与方向性
   结论词时，按「，」切分句，删除结论分句、保留未覆盖声明分句与中性
   分句（例：『RSI6数据未覆盖，进入超买区域，短期有回调压力』
   →『RSI6数据未覆盖。』）。若同一分句内矛盾无法安全拆分，整句不动。
② 跨句主题矛盾：某句声明主题 T 未覆盖（主题词与未覆盖词同句共现）后，
   扫描全文其他句子：含 T 主题词且含方向性结论词（含「整体呈净流入/
   流出态势」类）的句子整句删除；含未覆盖词的声明句与明确说不下结论
   的豁免句不动。
③ 头部结论特例：全文前 200 字内的结论句含「更便宜/更贵/更优」且估值
   主题已被声明未覆盖时，整句替换为「估值数据未覆盖，无法判断」。
④ 豁免与清理：只陈述未覆盖或说明局限的句子一律不动；上述删除导致
   【】小节标题下无任何内容时，空标题一并清掉（仅在发生过修复时执行）。

设计原则：
- 纯函数：不读文件、不访问网络、不依赖全局状态。
- 幂等：对输出再次调用结果不变。
- 保守：拿不准（分句内矛盾混合、豁免词共现等）一律不改。
- fail-safe：None、空串、非字符串等异常输入原样返回；任何内部异常
  捕获后原样返回，绝不抛出。
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# 未覆盖声明词：命中其一即视为「本句在声明数据缺口」
UNCOVERED_WORDS = (
    "数据未覆盖",
    "未覆盖",
    "数据暂缺",
    "暂无数据",
    "无数据",
    "数据缺失",
    "未能获取",
    "未获取",
)

# 方向性结论词：命中其一即视为「本句在下结论」
CONCLUSION_WORDS = (
    "超买",
    "超卖",
    "回调压力",
    "反弹压力",
    "净流入",
    "净流出",
    "更便宜",
    "更贵",
    "高估",
    "低估",
    "看好",
    "看空",
    "利多",
    "利空",
    "黄金坑",
    "机会窗口",
)

# 豁免词：明确说不下结论的句子不动
NO_CONCLUSION_WORDS = ("无法判断", "不下结论", "不做判断", "难以判断", "不足以")

# 主题映射：主题名 -> 主题词（用于跨句矛盾检测）
TOPIC_KEYWORDS = {
    "北向资金": ("北向",),
    "南向/港股通": ("南向", "港股通"),
    "估值": ("估值", "PE", "PB", "市盈率", "市净率"),
    "RSI": ("RSI",),
    "MACD": ("MACD",),
    "两融": ("两融", "融资余额", "融券"),
    "业绩预告": ("业绩预告", "预增", "预减"),
    "龙虎榜": ("龙虎榜",),
    "SHIBOR": ("SHIBOR", "拆借利率"),
    "CPI": ("CPI",),
    "PMI": ("PMI",),
    "M2": ("M2", "社融"),
    "资金流向": ("主力资金", "资金流向", "资金净"),
}

# 规则③：头部比较级结论词与扫描窗口
HEAD_COMPARE_WORDS = ("更便宜", "更贵", "更优")
HEAD_WINDOW_CHARS = 200
HEAD_REPLACEMENT = "估值数据未覆盖，无法判断"

_SENT_SEPS = "。；！？\n"
_CLAUSE_SEP = "，"
_HEADER_RE = re.compile(r"^\s*【[^】]*】\s*$")


def _contains_any(text, words):
    return any(w in text for w in words)


def _split_sentences(text):
    """按 。；！？\\n 切句，返回 [[正文, 分隔符], ...]，分隔符随句保留以便无损重组。"""
    sentences = []
    buf = []
    for ch in text:
        if ch in _SENT_SEPS:
            sentences.append(["".join(buf), ch])
            buf = []
        else:
            buf.append(ch)
    if buf:
        sentences.append(["".join(buf), ""])
    return sentences


def _fix_same_sentence(body):
    """规则①：同句矛盾修复。返回 (新句子, 是否修复)。"""
    if not _contains_any(body, UNCOVERED_WORDS):
        return body, False
    if not _contains_any(body, CONCLUSION_WORDS):
        return body, False
    if _contains_any(body, NO_CONCLUSION_WORDS):
        return body, False
    clauses = body.split(_CLAUSE_SEP)
    has_uncovered = [_contains_any(c, UNCOVERED_WORDS) for c in clauses]
    has_conclusion = [_contains_any(c, CONCLUSION_WORDS) for c in clauses]
    # 同一分句内既声明未覆盖又下结论，无法安全拆分，保守整句不改
    if any(u and c for u, c in zip(has_uncovered, has_conclusion)):
        return body, False
    kept = [c for c, concl in zip(clauses, has_conclusion) if not concl]
    new_body = _CLAUSE_SEP.join(kept)
    if new_body == body or not new_body.strip():
        return body, False
    return new_body, True


def _find_uncovered_topics(sentences):
    """找出已被声明未覆盖的主题集合（主题词与未覆盖词同句共现）。"""
    topics = set()
    for body, _sep in sentences:
        if not _contains_any(body, UNCOVERED_WORDS):
            continue
        for topic, keywords in TOPIC_KEYWORDS.items():
            if _contains_any(body, keywords):
                topics.add(topic)
    return topics


def _fix_head_compare(sentences, uncovered_topics):
    """规则③：头部比较级结论句在估值未覆盖时替换为保守表述。"""
    if "估值" not in uncovered_topics:
        return False
    changed = False
    pos = 0
    for sent in sentences:
        body, sep = sent
        start = pos
        pos += len(body) + len(sep)
        if start >= HEAD_WINDOW_CHARS:
            continue
        if not _contains_any(body, HEAD_COMPARE_WORDS):
            continue
        # 已是未覆盖声明句或明确不下结论的句子不动
        if _contains_any(body, UNCOVERED_WORDS):
            continue
        if _contains_any(body, NO_CONCLUSION_WORDS):
            continue
        logger.info("output_guard: 规则③ 头部比较结论替换: %s -> %s", body[:50], HEAD_REPLACEMENT)
        sent[0] = HEAD_REPLACEMENT
        changed = True
    return changed


def _fix_cross_sentence(sentences, uncovered_topics):
    """规则②：删除与未覆盖主题矛盾的方向性结论句。返回是否有删除。"""
    if not uncovered_topics:
        return False
    keywords = tuple(w for t in uncovered_topics for w in TOPIC_KEYWORDS[t])
    kept = []
    changed = False
    for body, sep in sentences:
        if (
            _contains_any(body, keywords)
            and _contains_any(body, CONCLUSION_WORDS)
            and not _contains_any(body, UNCOVERED_WORDS)
            and not _contains_any(body, NO_CONCLUSION_WORDS)
        ):
            logger.info("output_guard: 规则② 删除跨句矛盾结论句: %s", body[:50])
            changed = True
            continue
        kept.append([body, sep])
    if changed:
        sentences[:] = kept
    return changed


def _drop_empty_headers(text):
    """规则④：清理【】行后（跳过空行）无任何内容的空小节标题。"""
    lines = text.split("\n")
    n = len(lines)
    keep = [True] * n
    for i, line in enumerate(lines):
        if not _HEADER_RE.match(line):
            continue
        j = i + 1
        while j < n and not lines[j].strip():
            j += 1
        if j >= n or _HEADER_RE.match(lines[j]):
            logger.info("output_guard: 规则④ 清理空小节标题: %s", line.strip())
            keep[i] = False
    return "\n".join(line for line, k in zip(lines, keep) if k)


def fix_uncovered_conclusions(text: str) -> str:
    """修复「数据未覆盖却仍下结论」的矛盾输出。

    纯函数、幂等、保守（拿不准不改）、绝不抛出异常：
    非字符串或空输入原样返回，内部任何异常捕获后原样返回。
    """
    if not isinstance(text, str) or not text.strip():
        return text
    try:
        sentences = _split_sentences(text)
        changed = False

        # 规则①：同句矛盾
        for sent in sentences:
            new_body, fixed = _fix_same_sentence(sent[0])
            if fixed:
                logger.info("output_guard: 规则① 同句矛盾修复: %s -> %s", sent[0][:50], new_body[:50])
                sent[0] = new_body
                changed = True

        uncovered_topics = _find_uncovered_topics(sentences)

        # 规则③：头部比较级结论特例（先于规则②，替换优于整句删除）
        if _fix_head_compare(sentences, uncovered_topics):
            changed = True

        # 规则②：跨句主题矛盾
        if _fix_cross_sentence(sentences, uncovered_topics):
            changed = True

        if not changed:
            return text

        result = "".join(body + sep for body, sep in sentences)
        # 规则④：删除后留下的空小节标题一并清掉
        result = _drop_empty_headers(result)
        return result
    except Exception:
        logger.exception("output_guard: 护栏内部异常，原样返回")
        return text
