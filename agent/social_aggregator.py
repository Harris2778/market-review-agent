"""newsnow 聚合器兜底采集层（微舆/BettaFish 式舆情子系统 · 聚合兜底模块 · 纯原创实现）。

模块定位：**全局兜底源，不是主链路**。newsnow（newsnow.busiyi.world）是第三方
公共聚合服务，响应 `status:"cache"` 表明有缓存层——**有缓存、无 SLA、数据
新鲜度不受控，仅在平台直连失败时降级使用**，同时可作交叉校验参照。

端点定案（2026-07-22 实网侦查，见 research/social_endpoints_recon.md §6）：

- `GET https://newsnow.busiyi.world/api/s?id={source_id}&latest` → 200，
  无需 Cookie/签名，统一结构：
  `{status: "cache"|"success", id, updatedTime(毫秒), items[]:
  {id, title, url, mobileUrl?, extra?}}`。
- 源 ID 映射（实测定案，仅这四个可用）：
  weibo→'weibo'、zhihu→'zhihu'、douyin→'douyin'、bilibili→'bilibili-hot-search'。
  **小红书实测 500 Invalid source id，聚合器无此源，不支持**；其余非法
  platform 入参直接记 warning 返回空列表，不发请求。
- items 附加字段（实测）：知乎 `extra.info`（热度文本如 "461 万热度"）+
  `extra.hover`（摘要）；微博/B站仅 `extra.icon`；抖音无 extra。

统一 Post 字典（全局契约）：{platform(传什么就是什么，原样透传), post_id:str,
title, content(=extra.hover 可空串), author:''(聚合器无作者字段),
metrics(只放实际拿到的键：heat=extra.info 解析值，提不出省略该键),
url, published_at(顶层 updatedTime 毫秒→ISO), source:'newsnow_{source_id}'}。

合规注记：仅采集无登录公开数据；低频访问（≤1 req/s + 随机抖动）；无登录
自动化；第三方服务可用性与平台条款变动可能导致端点失效——所有失败只记
warning 并返回空/部分结果，绝不向调用方抛异常，可用性需按 warning 观测。
"""

import json
import logging
import random
import re
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

try:
    import requests
except ImportError:  # pragma: no cover - 依赖缺失时仅默认 session 不可用
    requests = None

logger = logging.getLogger(__name__)

# ── 端点与源映射（2026-07-22 实测定案）──

NEWSNOW_API_URL = "https://newsnow.busiyi.world/api/s"

# 统一平台名 → newsnow 源 ID；小红书实测 500 不存在，刻意缺席
SOURCE_ID_MAP: Dict[str, str] = {
    "weibo": "weibo",
    "zhihu": "zhihu",
    "douyin": "douyin",
    "bilibili": "bilibili-hot-search",
}

DEFAULT_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
DEFAULT_TIMEOUT = 10
DEFAULT_RATE = 1.0    # 同一轮内连续请求间隔下限（秒）：≤1 req/s
DEFAULT_JITTER = 0.3  # 随机抖动上限（秒）

# ═══════════════════════════════════════════
# 通用小工具（与兄弟模块各自独立实现，互不 import）
# ═══════════════════════════════════════════

_TAG_RE = re.compile(r"<[^>]+>")
# 热度文本解析："461 万热度" / "1.5亿" / 纯数字；单位支持 万/亿
_HEAT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(亿|万)?")


def _to_int(value) -> Optional[int]:
    """宽松 int 转换；非法值返回 None，绝不抛出。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _clean_str(value) -> str:
    """归一字符串字段：None → ''，其余 strip。"""
    if value is None:
        return ""
    return str(value).strip()


def _strip_html(text: str) -> str:
    """清洗 HTML 标签（如 <em> 高亮），压缩多余空白。"""
    if not text:
        return ""
    return re.sub(r"\s+", " ", _TAG_RE.sub("", text)).strip()


def _parse_heat(text) -> Optional[int]:
    """从热度文本提取绝对数值（万×1e4、亿×1e8）；提不出返回 None。"""
    if not text:
        return None
    m = _HEAT_RE.search(str(text))
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2)
    if unit == "万":
        value *= 10000
    elif unit == "亿":
        value *= 100000000
    return int(value)


def _iso_from_unix(value, *, ms: bool = False) -> str:
    """unix 时间戳 → ISO8601（UTC）；非法/缺失返回空串，绝不抛出。"""
    if value is None or isinstance(value, bool):
        return ""
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return ""
    if ts <= 0:
        return ""
    if ms:
        ts /= 1000.0
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return ""


# ═══════════════════════════════════════════
# HTTP 基础设施（session/sleep 可注入，对齐 sentiment._RateGate 模式）
# ═══════════════════════════════════════════

def _default_session():
    """生产 Session：trust_env=False（绕开本机系统代理）+ 浏览器 UA。"""
    if requests is None:
        return None
    s = requests.Session()
    s.trust_env = False
    s.headers.update({"User-Agent": DEFAULT_UA})
    return s


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


def _resp_json(resp) -> Optional[dict]:
    """从 response 提取 JSON 对象：优先 resp.json()，回退 text 解析；
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


def _request_json(url: str, *, session, gate: _RateGate,
                  tag: str = "", **kw) -> Optional[dict]:
    """统一请求入口：限速 + 状态码检查 + JSON 解析。失败记 warning 返回 None。"""
    gate.wait()
    if session is None:
        logger.warning("%s 无可用 session（requests 缺失？）：%s", tag, url)
        return None
    kw.setdefault("timeout", DEFAULT_TIMEOUT)
    try:
        resp = session.get(url, **kw)
    except Exception as e:
        logger.warning("%s 请求失败 %s: %s", tag, url, e)
        return None
    sc = getattr(resp, "status_code", None)
    if isinstance(sc, int) and sc >= 400:
        logger.warning("%s HTTP %s %s", tag, sc, url)
        return None
    payload = _resp_json(resp)
    if payload is None:
        logger.warning("%s 响应不是合法 JSON 对象：%s", tag, url)
    return payload


# ═══════════════════════════════════════════
# 公开接口：newsnow 聚合热榜兜底（仅此一项能力；无搜索/评论）
# ═══════════════════════════════════════════

def fetch_hot(platform: str, limit: int = 20, session=None, sleep=None) -> List[Dict]:
    """从 newsnow 聚合器抓取指定平台热榜兜底数据，返回统一 Post 字典列表。

    - platform：统一平台名（'weibo'/'zhihu'/'douyin'/'bilibili'），映射到
      实测可用的 newsnow 源 ID；**不支持小红书（实测 500）**，非法值记
      warning 返回空列表，不发请求；
    - limit：返回条数上限（客户端截断，端点本身一次全量）；
    - session：可注入 requests.Session（测试注入 fake）；未注入时自建
      trust_env=False + 浏览器 UA 的 Session；
    - sleep：可注入限速休眠函数（测试注入 fake）。

    任何失败记 warning 并返回（可能为空的）列表，绝不抛异常。
    """
    platform_name = _clean_str(platform).lower()
    source_id = SOURCE_ID_MAP.get(platform_name)
    if source_id is None:
        logger.warning(
            "newsnow 不支持的平台 %r（支持：%s；小红书实测 500 无此源）",
            platform, sorted(SOURCE_ID_MAP),
        )
        return []

    n = _to_int(limit)
    limit = n if n and n > 0 else 20
    gate = _RateGate(sleep or time.sleep)
    sess = session if session is not None else _default_session()

    payload = _request_json(
        NEWSNOW_API_URL,
        session=sess,
        gate=gate,
        params={"id": source_id, "latest": ""},
        tag=f"newsnow_{source_id}",
    )
    if payload is None:
        return []
    items = payload.get("items")
    if not isinstance(items, list):
        logger.warning("newsnow_%s items 字段非 list：%r",
                       source_id, type(items).__name__)
        return []

    # 顶层 updatedTime 为毫秒时间戳，作为全批次条目的 published_at
    published_at = _iso_from_unix(payload.get("updatedTime"), ms=True)

    posts: List[Dict] = []
    for item in items:
        if len(posts) >= limit:
            break
        if not isinstance(item, dict):
            continue
        iid = item.get("id")
        title = _strip_html(_clean_str(item.get("title")))
        if iid is None or not title:
            continue  # 无 id / 无标题，逐条跳过

        extra = item.get("extra")
        if not isinstance(extra, dict):
            extra = {}

        metrics: Dict[str, int] = {}
        heat = _parse_heat(extra.get("info"))  # 如 "461 万热度"；提不出省略该键
        if heat is not None:
            metrics["heat"] = heat

        posts.append({
            "platform": str(platform),  # 传什么就是什么，原样透传
            "post_id": str(iid),
            "title": title,
            "content": _strip_html(_clean_str(extra.get("hover"))),
            "author": "",  # 聚合器热榜条目无作者字段
            "metrics": metrics,
            "url": _clean_str(item.get("url")),
            "published_at": published_at,
            "source": f"newsnow_{source_id}",
        })
    return posts
