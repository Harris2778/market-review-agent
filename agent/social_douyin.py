"""抖音社媒舆情爬取层（微舆/BettaFish 式 · 纯原创实现，只借思路）。

端点定案依据：research/social_endpoints_recon.md（2026-07-22 本机实测定案）。

唯一可用源（v1 仅热榜，无搜索/评论）：
- GET https://www.douyin.com/aweme/v1/web/hot/search/list/ —— 桌面端热榜，
  200 ✅ status_code=0，必带桌面 UA + Referer: https://www.douyin.com/，
  无需 Cookie / 无需 X-Bogus 签名（全新 Session 实测通过）。
  响应结构（实测 49 条）：
    data.word_list[]:
      word         热点词
      hot_value    热度值（如 11541217）
      position     排名（本层不消费，按返回顺序截断 limit）
      event_time   事件时间（unix 秒，转 ISO8601 填入 published_at）
      sentence_id  热点 ID（拼 URL 用；兼作 post_id）
      label        标签码（3=热 等；本层不消费）
  url 拼接：https://www.douyin.com/hot/{sentence_id}

⚠️⚠️ 「无签名直连红利」——本接口极度脆弱，随时可能失效 ⚠️⚠️
该端点历史上随抖音风控策略摇摆，随时可能要求 X-Bogus/a_bogus 签名或
登录态。本模块因此**必须内置降级**：非 200 / status_code != 0 /
响应非 JSON / data.word_list 结构不符，一律记 warning 并返回空列表，
绝不抛异常；调用方须按 warning 观测失效并切换兜底源（如 newsnow-douyin）。

⚠️ 视频搜索 / 评论需 X-Bogus 签名 + 登录态，超出"无登录静态请求"合规边界，
v1 明确放弃，故本模块不提供 search() / fetch_comments()。

合规注记：仅访问无登录公开数据、低频调用（≤1 req/s + 随机抖动）、
无任何登录态自动化；平台条款变动可能导致接口失效，需按 warning 观测。

统一 Post 契约（四方对齐）：
{platform:'douyin', post_id:str(sentence_id，缺失时 sha1(word)[:12]),
 title:str, content:'', author:'',
 metrics:{'heat': hot_value}（hot_value 缺失则省略该键）,
 url:str(sentence_id 缺失时为空串), published_at:str(event_time→ISO 或空串),
 source:'douyin_hot'}
"""

import hashlib
import json
import logging
import random
import re
import time
from datetime import datetime, timezone
from typing import Callable, List, Optional

try:
    import requests
except ImportError:  # pragma: no cover - 依赖缺失时仅默认 session 不可用
    requests = None

logger = logging.getLogger(__name__)

# ── 端点与公共参数 ──

HOT_LIST_URL = "https://www.douyin.com/aweme/v1/web/hot/search/list/"
REFERER = "https://www.douyin.com/"
HOT_PAGE_PREFIX = "https://www.douyin.com/hot/"
SOURCE = "douyin_hot"

DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT = 10
DEFAULT_RATE = 1.0    # 单轮内请求间隔下限（秒）
DEFAULT_JITTER = 0.5  # 随机抖动上限（秒）

_TAG_RE = re.compile(r"<[^>]+>")


# ═══════════════════════════════════════════
# HTTP 基础设施（session 可注入；模式照抄 agent/sentiment.py 的 _RateGate）
# ═══════════════════════════════════════════

class _RateGate:
    """单轮限速门：同一轮内首个请求不限速，其后间隔 RATE + random(0, JITTER)。"""

    def __init__(self, sleep: Callable[[float], None],
                 rate: float = DEFAULT_RATE, jitter: float = DEFAULT_JITTER):
        self._sleep = sleep
        self._rate = max(0.0, rate)
        self._jitter = max(0.0, jitter)
        self._used = False

    def wait(self) -> None:
        if not self._used:
            self._used = True
            return
        self._sleep(self._rate + random.uniform(0, self._jitter))


def _new_session():
    """生产自建 session：trust_env=False 绕开本机系统代理 + 桌面浏览器 UA。"""
    if requests is None:
        raise RuntimeError("requests 库不可用")
    sess = requests.Session()
    sess.trust_env = False
    sess.headers.update({
        "User-Agent": DEFAULT_UA,
        "Accept": "application/json, text/plain, */*",
        "Referer": REFERER,
    })
    return sess


def _resp_json(resp) -> Optional[dict]:
    """从 response 提取 JSON 对象：优先 resp.json()，回退 text/content 解析；
    顶层非 dict 或解析失败返回 None，绝不抛出。"""
    payload = None
    try:
        payload = resp.json()
    except Exception:
        content = getattr(resp, "text", None)
        if content is None:
            raw = getattr(resp, "content", b"")
            content = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
    return payload if isinstance(payload, dict) else None


def _clean_text(value) -> str:
    """文本清洗：去 HTML 标签（防御性，热榜词实测无标签）+ 去首尾空白。"""
    if not isinstance(value, str):
        return ""
    return _TAG_RE.sub("", value).strip()


def _to_int(value) -> Optional[int]:
    """宽松转 int；失败返回 None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        if re.fullmatch(r"-?\d+", s):
            return int(s)
    return None


def _to_iso(value) -> str:
    """unix 秒 → ISO8601（UTC）；缺失/非法返回空串。"""
    ts = _to_int(value)
    if ts is None or ts <= 0:
        return ""
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return ""


# ═══════════════════════════════════════════
# 热榜采集
# ═══════════════════════════════════════════

def _parse_word_list(payload: dict) -> List[dict]:
    """防御式解析 data.word_list[]：结构漂移记 warning 返回空，字段缺失跳过该条。"""
    if payload.get("status_code") != 0:
        logger.warning("douyin_hot 业务状态异常：status_code=%r（接口风控可能已变更）",
                       payload.get("status_code"))
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        logger.warning("douyin_hot 结构漂移：data 非对象（got %s）", type(data).__name__)
        return []
    word_list = data.get("word_list")
    if not isinstance(word_list, list):
        logger.warning("douyin_hot 结构漂移：data.word_list 非列表（got %s）",
                       type(word_list).__name__)
        return []

    posts: List[dict] = []
    for row in word_list:
        if not isinstance(row, dict):
            continue  # 防御：非对象条目跳过
        word = _clean_text(row.get("word") or "")
        if not word:
            continue  # 防御：热点词缺失跳过该条
        sentence_id = _clean_text(str(row.get("sentence_id") or ""))
        metrics = {}
        hot_value = _to_int(row.get("hot_value"))
        if hot_value is not None:
            metrics["heat"] = hot_value
        posts.append({
            "platform": "douyin",
            "post_id": sentence_id or hashlib.sha1(word.encode("utf-8")).hexdigest()[:12],
            "title": word,
            "content": "",
            "author": "",
            "metrics": metrics,
            "url": HOT_PAGE_PREFIX + sentence_id if sentence_id else "",
            "published_at": _to_iso(row.get("event_time")),
            "source": SOURCE,
        })
    return posts


def fetch_hot(limit: int = 20, session=None, sleep=None) -> List[dict]:
    """抓取抖音热榜，返回统一 Post 列表（按返回顺序截断 limit）。

    「无签名直连红利」脆弱接口：任何失败（网络异常 / 非 200 /
    status_code != 0 / 非 JSON / 结构漂移）记 warning 并返回空列表
    或部分结果，绝不向调用方抛异常。
    """
    sess = session or _new_session()
    gate = _RateGate(sleep or time.sleep)
    gate.wait()
    try:
        resp = sess.get(HOT_LIST_URL, headers={"Referer": REFERER},
                        timeout=DEFAULT_TIMEOUT)
    except Exception as e:
        logger.warning("douyin_hot 请求失败 %s: %s", HOT_LIST_URL, e)
        return []
    sc = getattr(resp, "status_code", None)
    if sc != 200:
        logger.warning("douyin_hot HTTP %s %s（接口风控可能已变更）", sc, HOT_LIST_URL)
        return []
    payload = _resp_json(resp)
    if payload is None:
        logger.warning("douyin_hot 响应不是合法 JSON 对象：%s", HOT_LIST_URL)
        return []
    posts = _parse_word_list(payload)
    return posts[:max(0, int(limit))]
