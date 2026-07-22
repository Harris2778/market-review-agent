"""DeepSeek 批量情感打分器（词典法的升级替代，供聚合层经 scorer 注入使用）。

职责：将社交帖子/评论等文本批量送给 DeepSeek（openai 兼容协议）做情感打分，
标签体系为「乐观/中性/悲观」（词典法「利好/利空/中性」的扩展升级），失败时
整体回退 agent.sentiment 的确定性词典法，保证任何路径绝不抛异常。

设计要点（对齐全局契约）：

1. 批量调用：每 batch_size 条拼一次 chat.completions.create，编号列表进
   prompt，要求模型只返回 JSON 数组 [{"i":编号,"label":"乐观|中性|悲观",
   "score":-1~1}]。标签定义写死在 prompt：乐观=对标的/市场前景看多、利好、
   期待；悲观=看空、利空、担忧、嘲讽看空；中性=纯信息/无关/无法判断。
2. 解析容忍：模型输出可带 ```json 代码块包裹/前后废话——正则提取第一个
   「[」到最后一个「]」再 json.loads；数组中单条解析失败（非 dict、编号
   非法、label 不在三值内、score 非数值）该条标中性 score=0；编号错位/缺条
   按缺条补中性；整个响应提不出合法 JSON 数组视为本批 LLM 失效 → 词典回退。
3. 降级契约：每条返回 {index:int(全局), label:str, score:float, method:
   'llm'|'fallback'}；LLM 调用异常/超时（timeout=30）或响应不可解析 →
   该批次整体降级，内部直接调 agent.sentiment.score_news_sentiment 词典
   回退填好 label/score（利好→乐观、利空→悲观归并），method 标 'fallback'。
4. client 可注入（openai 风格 client.chat.completions.create）；未注入时
   惰性自建：读 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL 环境变量，构建失败
   （缺 key/缺 openai 包）→ 全量词典回退，绝不抛。
5. 限速：批间间隔 0.5s + 0~0.3s 随机抖动（首个批次不限速），sleep 可注入，
   测试用 fake 记录。
6. 工程纪律：公开函数绝不向调用方抛异常；空输入返回 []。
"""

import json
import logging
import os
import random
import re
import time
from typing import Callable, Dict, List, Optional

import agent.sentiment as sentiment

logger = logging.getLogger(__name__)

# ── 公共常量 ──

DEFAULT_MODEL = "deepseek-chat"   # DeepSeek 默认聊天模型
DEFAULT_BATCH_SIZE = 20           # 单次调用打包含量
LLM_TIMEOUT = 30                  # 单次 LLM 调用超时（秒）
BATCH_RATE = 0.5                  # 批间限速基线（秒）
BATCH_JITTER = 0.3                # 批间随机抖动上限（秒）

VALID_LABELS = ("乐观", "中性", "悲观")

# 词典法标签 → LLM 标签体系归并（全局契约：利好→乐观、利空→悲观）
_DICT_TO_LLM_LABEL = {"利好": "乐观", "利空": "悲观", "中性": "中性"}

# 贪婪匹配：第一个 「[」 到最后一个 「]」（容忍代码块包裹/前后废话）
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

# ── prompt 模板（标签定义写死，编号列表由 _build_messages 拼装）──

_SYSTEM_PROMPT = (
    "你是 A 股舆情情感分析器，专注判断散户帖子/评论对标的股票或市场前景的"
    "情感倾向。严格按用户要求的 JSON 格式输出，不要输出任何其他内容。"
)

_USER_PROMPT_TEMPLATE = """对下面编号列表中的每条文本做情感打分。

标签定义：
- 乐观：对标的/市场前景表达看多、利好、期待
- 悲观：看空、利空、担忧、嘲讽看空
- 中性：纯信息/无关/无法判断

要求：
1. 只返回一个 JSON 数组，不要任何其他文字、解释或代码块标记；
2. 数组元素格式：{{"i": 编号, "label": "乐观|中性|悲观", "score": -1到1的小数}}；
3. score 表征强度：越接近 1 越乐观，越接近 -1 越悲观，0 附近为中性；
4. 每条编号都必须给出结果。

待打分文本：
{numbered_texts}"""


# ═══════════════════════════════════════════
# 纯函数工具
# ═══════════════════════════════════════════

def _build_messages(texts: List[str]) -> List[dict]:
    """拼装 openai 风格 messages：system 角色定义 + user 编号列表 prompt。"""
    numbered = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(
            numbered_texts=numbered)},
    ]


def _response_content(resp) -> str:
    """从 openai 风格响应防御式提取文本内容；失败返回 ''，绝不抛。"""
    try:
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "")
        return content if isinstance(content, str) else ""
    except Exception:  # 防御：异形响应对象
        return ""


def _extract_json_array(content: str) -> Optional[list]:
    """从模型输出提取 JSON 数组：正则取第一个 [ 到最后一个 ] 再解析。

    容忍 ```json 代码块包裹与前后废话；无匹配或解析失败返回 None
    （调用方据此判定本批 LLM 失效），绝不抛。
    """
    if not isinstance(content, str) or not content:
        return None
    m = _JSON_ARRAY_RE.search(content)
    if not m:
        return None
    try:
        payload = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, list) else None


def _parse_entries(payload: list) -> Dict[int, dict]:
    """解析模型 JSON 数组 → {编号: {label, score}}。

    单条容错：非 dict / 编号非法 / label 不在三值内 / score 非数值的条目
    一律丢弃（调用方按缺条补中性）；score 截断到 [-1, 1]；重复编号后者
    覆盖前者（等价忽略）。绝不抛。
    """
    parsed: Dict[int, dict] = {}
    if not isinstance(payload, list):
        return parsed
    for entry in payload:
        if not isinstance(entry, dict):
            continue  # 单条解析失败：丢弃，由缺条补齐逻辑兜底
        idx = sentiment._to_int(entry.get("i"))
        label = str(entry.get("label", "")).strip()
        score = sentiment._to_float(entry.get("score"))
        if idx is None or label not in VALID_LABELS or score is None:
            continue  # 单条解析失败：丢弃
        parsed[idx] = {
            "label": label,
            "score": round(sentiment._clamp(score, -1.0, 1.0), 4),
        }
    return parsed


def _fallback_batch(texts: List[str], start_index: int) -> List[dict]:
    """批次整体降级：内部调用词典法 score_news_sentiment 填好 label/score。

    词典标签归并到 LLM 标签体系（利好→乐观、利空→悲观），method 标
    'fallback'。词典法为纯本地确定性实现，自身也不抛；此处仍套防御。
    """
    results: List[dict] = []
    try:
        items = [{"title": t} for t in texts]
        scored = sentiment.score_news_sentiment(items)
    except Exception as e:  # 防御：词典回退自身异常 → 全中性
        logger.warning("sentiment_llm 词典回退异常，降级全中性: %s", e)
        scored = []
    for offset, _text in enumerate(texts):
        label, score = "中性", 0.0
        if offset < len(scored) and isinstance(scored[offset], dict):
            raw_label = str(scored[offset].get("sentiment", "")).strip()
            label = _DICT_TO_LLM_LABEL.get(raw_label, "中性")
            raw_score = sentiment._to_float(scored[offset].get("sentiment_score"))
            if raw_score is not None:
                score = round(sentiment._clamp(raw_score, -1.0, 1.0), 4)
        results.append({
            "index": start_index + offset,
            "label": label,
            "score": score,
            "method": "fallback",
        })
    return results


# ═══════════════════════════════════════════
# client 构建（惰性，可注入绕过）
# ═══════════════════════════════════════════

def _build_default_client():
    """惰性自建 openai 风格 DeepSeek client。

    读 DEEPSEEK_API_KEY（必需）/ DEEPSEEK_BASE_URL（缺省官方端点）环境
    变量；openai 包缺失或构建失败（如缺 key）记 warning 返回 None，绝不抛
    （调用方据此走全量词典回退）。
    """
    try:
        import openai
    except ImportError:
        logger.warning("sentiment_llm: openai 包不可用，LLM 打分降级词典法")
        return None
    api_key = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    if not api_key:
        logger.warning("sentiment_llm: 未配置 DEEPSEEK_API_KEY，LLM 打分降级词典法")
        return None
    base_url = (os.environ.get("DEEPSEEK_BASE_URL")
                or "https://api.deepseek.com").strip()
    try:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        logger.warning("sentiment_llm: DeepSeek client 构建失败: %s", e)
        return None


# ═══════════════════════════════════════════
# 公开接口
# ═══════════════════════════════════════════

def score_texts_batch(texts: List[str], client=None,
                      model: str = DEFAULT_MODEL,
                      batch_size: int = DEFAULT_BATCH_SIZE,
                      sleep: Optional[Callable[[float], None]] = None) -> List[dict]:
    """DeepSeek 批量情感打分：texts 为 str 列表，逐批调用、容错解析、降级回退。

    返回 [{index:int(全局下标), label:'乐观|中性|悲观', score:float(-1~1),
    method:'llm'|'fallback'}]，顺序与输入一致。LLM 调用异常/超时或响应不可
    解析 → 该批次整体词典回退（method='fallback'，label/score 已由
    agent.sentiment.score_news_sentiment 填好）；编号错位/缺条按缺条补
    中性（method 仍为 'llm'）。批间限速 0.5s+抖动（sleep 可注入，首个批次
    不限速）。空输入返回 []；任何路径绝不抛异常。
    """
    try:
        norm_texts = [t if isinstance(t, str) else str(t) for t in (texts or [])]
    except Exception:  # 防御：异形入参
        return []
    if not norm_texts:
        return []

    try:
        size = int(batch_size)
    except (TypeError, ValueError):
        size = DEFAULT_BATCH_SIZE
    if size <= 0:
        size = DEFAULT_BATCH_SIZE
    model_name = str(model or DEFAULT_MODEL)
    sleep_fn = sleep if callable(sleep) else time.sleep

    # client 注入优先；未注入时惰性自建（失败 → 全量词典回退）
    if client is None:
        client = _build_default_client()

    results: List[dict] = []
    first_batch = True
    for start in range(0, len(norm_texts), size):
        batch = norm_texts[start:start + size]
        # 批间限速：首个批次不限速
        if not first_batch:
            try:
                sleep_fn(BATCH_RATE + random.uniform(0, BATCH_JITTER))
            except Exception as e:  # 防御：注入的 sleep 异常不阻断打分
                logger.warning("sentiment_llm 限速 sleep 异常: %s", e)
        first_batch = False

        if client is None:
            results.extend(_fallback_batch(batch, start))
            continue

        # LLM 调用：异常/超时/响应不可解析 → 本批整体词典回退
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=_build_messages(batch),
                timeout=LLM_TIMEOUT,
            )
            content = _response_content(resp)
            payload = _extract_json_array(content)
        except Exception as e:
            logger.warning("sentiment_llm LLM 调用失败（批次起始 %s），"
                           "降级词典法: %s", start, e)
            payload = None

        if payload is None:
            results.extend(_fallback_batch(batch, start))
            continue

        parsed = _parse_entries(payload)
        for offset in range(len(batch)):
            entry = parsed.get(offset)
            if entry is None:
                # 编号错位/缺条/单条解析失败：补中性
                entry = {"label": "中性", "score": 0.0}
            results.append({
                "index": start + offset,
                "label": entry["label"],
                "score": entry["score"],
                "method": "llm",
            })
    return results


def make_llm_scorer(client=None) -> Callable[[dict], dict]:
    """构造符合 sentiment.score_news_sentiment scorer 注入签名的打分函数。

    返回 scorer(item) -> {'sentiment':..., 'sentiment_score':..., 'hits':[]}；
    内部走批量打分（单条也走批量接口 batch_size=1），client 透传注入。
    任何异常降级中性 0.0，绝不抛（score_news_sentiment 对 scorer 异常
    也有兜底，此为双保险）。
    """

    def _item_text(item: dict) -> str:
        """拼接条目可打分文本：title + summary + content（对齐词典法口径）。"""
        if not isinstance(item, dict):
            return ""
        parts = []
        for key in ("title", "summary", "content"):
            value = item.get(key)
            if value is not None:
                text = str(value).strip()
                if text:
                    parts.append(text)
        return " ".join(parts)

    def scorer(item: dict) -> dict:
        try:
            results = score_texts_batch([_item_text(item)], client=client,
                                        batch_size=1)
            if results:
                entry = results[0]
                label = entry.get("label")
                if label not in VALID_LABELS:
                    label = "中性"
                score = sentiment._to_float(entry.get("score"))
                score = (round(sentiment._clamp(score, -1.0, 1.0), 4)
                         if score is not None else 0.0)
                return {"sentiment": label, "sentiment_score": score,
                        "hits": []}
        except Exception as e:  # 防御：任何异常降级中性
            logger.warning("sentiment_llm scorer 打分异常，降级中性: %s", e)
        return {"sentiment": "中性", "sentiment_score": 0.0, "hits": []}

    return scorer
