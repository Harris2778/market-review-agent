"""微博社媒舆情爬取层（微舆/BettaFish 式 · 纯原创实现，只借思路）。

端点定案依据：research/social_endpoints_recon.md（2026-07-22 本机实测定案）。

唯一可用源（v1 仅热榜，无搜索/评论）：
- GET https://weibo.com/ajax/side/hotSearch —— 桌面端热搜，200 ✅，
  必带 UA + Referer: https://weibo.com/，无需 Cookie。
  响应结构（实测 51 条 realtime + 1 条 hotgov 置顶）：
    data.realtime[]:
      word        话题词（如 "别再给AI乱传文件了"）
      note        同 word（展示文本，作 word 缺失时的回退）
      num         热度数值（如 2241781）
      label_name  标签（"热"/"新"/"沸"/"爆"，可能为空；本层不消费）
      rank/realpos 排名（本层不消费，按返回顺序截断 limit）
  ⚠️ realtime 项无 URL 也无 mid，url 自行拼接
     https://s.weibo.com/weibo?q={quote(word)}；
     post_id 用 word 的 sha1[:12] 合成；published_at 无数据，统一空串。

⚠️ 移动端全线死刑（2026-07-22 实测，不要实现、不要复活）：
- m.weibo.cn 热搜 / 搜索 / 评论 hotflow 全部 432 → Sina Visitor System 拦截；
- s.weibo.com 桌面搜索 HTML 同样 Visitor 拦截；
- passport.weibo.cn/visitor/genvisitor 被 wbBotDetector JS 反爬挡回。
  搜索/评论需 headless 浏览器或 Cookie 池，超出"无登录静态请求"合规边界，
  v1 明确放弃，故本模块不提供 search() / fetch_comments()。

合规注记：仅访问无登录公开数据、低频调用（≤1 req/s + 随机抖动）、
无任何登录态自动化；该接口属未授权公开端点，平台条款/风控策略变动
可能导致失效——任何异常只记 warning 并返回空列表，调用方须按 warning 观测。

统一 Post 契约（四方对齐）：
{platform:'weibo', post_id:str, title:str, content:'', author:'',
 metrics:{'heat': num}（num 缺失则省略该键）, url:str,
 published_at:''（热搜实时无发布时间）, source:'weibo_hot'}
"""

import hashlib
import json
import logging
import random
import re
import time
from datetime import datetime, timezone  # noqa: F401  (契约保留：本模块热搜无时间字段)
from typing import Callable, List, Optional
from urllib.parse import quote

try:
    import requests
except ImportError:  # pragma: no cover - 依赖缺失时仅默认 session 不可用
    requests = None

logger = logging.getLogger(__name__)

# ── 端点与公共参数 ──

HOT_SEARCH_URL = "https://weibo.com/ajax/side/hotSearch"
REFERER = "https://weibo.com/"
SEARCH_URL_PREFIX = "https://s.weibo.com/weibo?q="
SOURCE = "weibo_hot"

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
    """生产自建 session：trust_env=False 绕开本机系统代理 + 浏览器 UA。"""
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
    """文本清洗：去 HTML 标签（防御性，热搜词实测无标签）+ 去首尾空白。"""
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


# ═══════════════════════════════════════════
# 热榜采集
# ═══════════════════════════════════════════

def _parse_realtime(payload: dict) -> List[dict]:
    """防御式解析 data.realtime[]：结构漂移/字段缺失记 warning 或跳过该条。"""
    data = payload.get("data")
    if not isinstance(data, dict):
        logger.warning("weibo_hot 结构漂移：data 非对象（got %s）", type(data).__name__)
        return []
    realtime = data.get("realtime")
    if not isinstance(realtime, list):
        logger.warning("weibo_hot 结构漂移：data.realtime 非列表（got %s）",
                       type(realtime).__name__)
        return []

    posts: List[dict] = []
    for row in realtime:
        if not isinstance(row, dict):
            continue  # 防御：非对象条目跳过
        word = _clean_text(row.get("word") or row.get("note") or "")
        if not word:
            continue  # 防御：话题词缺失跳过该条
        metrics = {}
        num = _to_int(row.get("num"))
        if num is not None:
            metrics["heat"] = num
        posts.append({
            "platform": "weibo",
            "post_id": hashlib.sha1(word.encode("utf-8")).hexdigest()[:12],
            "title": word,
            "content": "",
            "author": "",
            "metrics": metrics,
            "url": SEARCH_URL_PREFIX + quote(word),
            "published_at": "",  # 热搜为实时榜单，端点不提供发布时间
            "source": SOURCE,
        })
    return posts


def fetch_hot(limit: int = 20, session=None, sleep=None) -> List[dict]:
    """抓取微博实时热搜榜，返回统一 Post 列表（按返回顺序截断 limit）。

    任何失败（网络异常 / 非 200 / 非 JSON / 结构漂移）记 warning 并返回
    空列表或部分结果，绝不向调用方抛异常。
    """
    sess = session or _new_session()
    gate = _RateGate(sleep or time.sleep)
    gate.wait()
    try:
        resp = sess.get(HOT_SEARCH_URL, headers={"Referer": REFERER},
                        timeout=DEFAULT_TIMEOUT)
    except Exception as e:
        logger.warning("weibo_hot 请求失败 %s: %s", HOT_SEARCH_URL, e)
        return []
    sc = getattr(resp, "status_code", None)
    if sc != 200:
        logger.warning("weibo_hot HTTP %s %s", sc, HOT_SEARCH_URL)
        return []
    payload = _resp_json(resp)
    if payload is None:
        logger.warning("weibo_hot 响应不是合法 JSON 对象：%s", HOT_SEARCH_URL)
        return []
    posts = _parse_realtime(payload)
    return posts[:max(0, int(limit))]
