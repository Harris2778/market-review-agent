"""离线评估 rubric 集合（确定性检查器，零网络、零 LLM、可 CI）。

五个 rubric 对应项目质量红线：

  name              判定内容
  ----------------  ----------------------------------------------------------
  number_sourcing   数字溯源：output 里每个数字都必须能在 fixture_context
                    （注入 prompt 的数据块文本）里找到出处。直接复用
                    agent.validators.find_unsourced_numbers，不重造。
  banned_words      禁用词命中：清单镜像 agent/system_prompts.py 四个分析类
                    prompt 的「禁用词」段（护城河/飞轮/赋能/格局/综上所述等）。
  markdown_table    markdown 管道表格残留：_clean_markdown 之后最终输出里
                    不应再出现「| 管道表格行 / |---| 分隔行」。
  risk_disclaimer   风险提示语存在性：非透传类输出必须含「不构成任何投资建议」。
  sector_structure  sector_deep_dive 模式的五维结构标记：
                    趋势 / 估值 / 资金 / 景气度 / 催化，缺一即失败。
                    其他 mode 本项不适用，自动通过。

统一签名：check(output, context="", mode="") -> dict
返回结构：{"name": str, "passed": bool, "detail": str}（中文明细）。
所有函数 fail-safe：异常输入一律转为安全结果，绝不抛出。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# 本文件既会被 run_eval.py 以同目录方式 import，也会被
# tests/test_eval_offline.py 用 importlib 按文件路径加载。
# 这里自行为 agent 包兜底 sys.path，保证两种入口都能 import agent.validators。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent.validators import find_unsourced_numbers  # noqa: E402

# ── 禁用词清单 ──
# 镜像 agent/system_prompts.py 中 SECTOR_DEEP_DIVE_PROMPT / NEWS_ANALYSIS_PROMPT /
# AGENT_QUERY_PROMPT / CRITIQUE_PROMPT 共用的「禁用词」段。prompt 侧清单变更时
# 必须同步本表；tests/test_eval_offline.py 会校验本表每个词仍出现在
# system_prompts 文本中，防止两侧漂移。
BANNED_WORDS = [
    "护城河", "飞轮", "赋能", "格局", "至关重要", "值得注意的是",
    "综上所述", "不仅…而且", "深度", "全方位", "拥抱", "长期主义",
    "黄金坑", "戴维斯双击",
]

# 「不仅…而且」是成对结构：两者同现才算命中（与 CRITIQUE_PROMPT 语义一致）。
_PAIR_BANNED = {"不仅…而且": ("不仅", "而且")}

# ── 风险提示语（与 orchestrator/main 追加的免责声明关键句一致）──
RISK_DISCLAIMER = "不构成任何投资建议"

# ── sector_deep_dive 五维结构标记（对应输出模板五个维度小节）──
SECTOR_DIMENSIONS = ["趋势", "估值", "资金", "景气度", "催化"]

_VALID_MODES = ("market_review", "sector_deep_dive", "news_only")


def _result(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _text(x) -> str:
    try:
        return x if isinstance(x, str) else ("" if x is None else str(x))
    except Exception:
        return ""


# ── 1. 数字溯源 ──

def check_number_sourcing(output, context: str = "", mode: str = "") -> dict:
    """output 中存在在 fixture_context 里找不到出处的数字即失败。"""
    try:
        violations = find_unsourced_numbers(_text(output), _text(context))
        if violations:
            items = "；".join(
                "「%s」（…%s…）" % (v.get("raw", "?"), v.get("snippet", ""))
                for v in violations[:5]
            )
            return _result(
                "number_sourcing", False,
                "发现 %d 个无出处数字：%s" % (len(violations), items),
            )
        return _result("number_sourcing", True, "全部数字均可在数据块中找到出处")
    except Exception as exc:  # fail-safe：校验器自身异常不炸评估
        return _result("number_sourcing", False, "校验器异常，按失败处理：%r" % exc)


# ── 2. 禁用词 ──

def _banned_hits(text: str) -> list:
    hits = []
    for word in BANNED_WORDS:
        if word in _PAIR_BANNED:
            a, b = _PAIR_BANNED[word]
            if a in text and b in text:
                hits.append(word)
        elif word in text:
            hits.append(word)
    return hits


def check_banned_words(output, context: str = "", mode: str = "") -> dict:
    try:
        hits = _banned_hits(_text(output))
        if hits:
            return _result("banned_words", False, "命中禁用词：%s" % "、".join(hits))
        return _result("banned_words", True, "未命中禁用词")
    except Exception as exc:
        return _result("banned_words", False, "校验器异常，按失败处理：%r" % exc)


# ── 3. markdown 管道表格残留 ──

# 分隔行形态：| --- | --- | 或 |:--:|--: 等（至少一个管道符 + 连续 3 个以上连字符）
_TABLE_SEP_RE = re.compile(r"\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?")


def _table_residue_lines(text: str) -> list:
    hits = []
    for line in text.splitlines():
        s = line.strip()
        if not s or "|" not in s:
            continue
        # 一行里 ≥2 个管道符 → 管道表格行；或命中分隔行形态
        if s.count("|") >= 2 or _TABLE_SEP_RE.search(s):
            hits.append(s)
    return hits


def check_markdown_table(output, context: str = "", mode: str = "") -> dict:
    try:
        hits = _table_residue_lines(_text(output))
        if hits:
            shown = " / ".join(hits[:3])
            return _result(
                "markdown_table", False,
                "发现 %d 行管道表格残留：%s" % (len(hits), shown),
            )
        return _result("markdown_table", True, "无管道表格残留")
    except Exception as exc:
        return _result("markdown_table", False, "校验器异常，按失败处理：%r" % exc)


# ── 4. 风险提示语 ──

def check_risk_disclaimer(output, context: str = "", mode: str = "") -> dict:
    try:
        if RISK_DISCLAIMER in _text(output):
            return _result("risk_disclaimer", True, "含风险提示语「%s」" % RISK_DISCLAIMER)
        return _result(
            "risk_disclaimer", False,
            "缺少风险提示语「%s」" % RISK_DISCLAIMER,
        )
    except Exception as exc:
        return _result("risk_disclaimer", False, "校验器异常，按失败处理：%r" % exc)


# ── 5. sector_deep_dive 五维结构 ──

def check_sector_structure(output, context: str = "", mode: str = "") -> dict:
    try:
        if mode != "sector_deep_dive":
            return _result(
                "sector_structure", True,
                "mode=%s 非板块五维模式，本项不适用，自动通过" % (mode or "?"),
            )
        text = _text(output)
        missing = [d for d in SECTOR_DIMENSIONS if d not in text]
        if missing:
            return _result(
                "sector_structure", False,
                "五维结构缺失标记：%s（要求：%s）"
                % ("、".join(missing), "/".join(SECTOR_DIMENSIONS)),
            )
        return _result(
            "sector_structure", True,
            "五维结构标记齐全（%s）" % "/".join(SECTOR_DIMENSIONS),
        )
    except Exception as exc:
        return _result("sector_structure", False, "校验器异常，按失败处理：%r" % exc)


# ── 注册表与批量入口 ──

RUBRICS = {
    "number_sourcing": check_number_sourcing,
    "banned_words": check_banned_words,
    "markdown_table": check_markdown_table,
    "risk_disclaimer": check_risk_disclaimer,
    "sector_structure": check_sector_structure,
}


def run_all(output, context: str = "", mode: str = "") -> list:
    """对一条 output 跑全部 rubric，按注册顺序返回结果清单。"""
    return [check(output, context, mode) for check in RUBRICS.values()]
