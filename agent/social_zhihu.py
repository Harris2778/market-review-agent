"""知乎热榜采集层（微舆/BettaFish 式舆情子系统 · 知乎模块 · 纯原创实现）。

端点定案（2026-07-22 实网侦查，见 research/social_endpoints_recon.md §2）：

- **唯一可用源**：`GET https://api.zhihu.com/topstory/hot-list?limit=N` → 200，
  必带 UA + `Referer: https://www.zhihu.com/billboard`，无需 Cookie/签名。
- **已判死刑（不要复活）**：
  - billboard HTML（`https://www.zhihu.com/billboard`）：403 zse-ck 反爬，
    带首页预热 Cookie 仍 403；
  - 搜索 `api/v4/search_v3?t=general&q={kw}`：400 `{"HitLabels": null}`，
    需签名/登录态。
  → **知乎关键词搜索 v1 缺席**：本模块刻意不提供 `search()` / `fetch_comments()`；
    搜索缺口由 newsnow 聚合器（agent/social_aggregator.py）热榜兜底 +
    Stage 3 提示词向用户显式说明。

响应结构（2026-07-22 实测）：顶层 `data[]`，条目级 `detail_text` 为热度文本
（如 "498 万热度"，可能缺席）；`data[].target`：
`{id, title, url(API), excerpt, created(unix 秒), answer_count,
follower_count, comment_count, author.name}`。前端 URL 按定案拼接：
`https://www.zhihu.com/question/{target.id}`。

统一 Post 字典（全局契约）：{platform:'zhihu', post_id:str, title,
content(=excerpt), author(=author.name 可空串), metrics(只放实际拿到的键：
comments=answer_count、heat=detail_text 解析值), url, published_at(created→ISO),
source:'zhihu_hot'}。

合规注记：仅采集无登录公开数据；低频访问（≤1 req/s + 随机抖动）；无登录
自动化、不碰签名破解；平台条款/风控变动可能导致端点失效——所有失败只记
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

# ── 端点与公共参数（2026-07-22 实测定案）──

ZHIHU_HOT_LIST_URL = "https://api.zhihu.com/topstory/hot-list"
ZHIHU_REFERER = "https://www.zhihu.com/billboard"
ZHIHU_QUESTION_URL = "https://www.zhihu.com/question/{qid}"

DEFAULT_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
DEFAULT_TIMEOUT = 10
DEFAULT_RATE = 1.0    # 同一轮内连续请求间隔下限（秒）：≤1 req/s
DEFAULT_JITTER = 0.3  # 随机抖动上限（秒）

# ═══════════════════════════════════════════
# 通用小工具（与兄弟模块各自独立实现，互不 import）
# ═══════════════════════════════════════════

_TAG_RE = re.compile(r"<[^>]+>")
# 热度文本解析："498 万热度" / "1.5亿" / 纯数字；单位支持 万/亿
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
# 公开接口：知乎热榜（v1 仅此一项能力；无搜索/评论，见模块 docstring）
# ═══════════════════════════════════════════

def fetch_hot(limit: int = 20, session=None, sleep=None) -> List[Dict]:
    """抓取知乎热榜，返回统一 Post 字典列表。

    - limit：返回条数上限（同时作为端点 limit 参数，默认 20）；
    - session：可注入 requests.Session（测试注入 fake）；未注入时自建
      trust_env=False + 浏览器 UA 的 Session；
    - sleep：可注入限速休眠函数（测试注入 fake）。

    任何失败记 warning 并返回（可能为空的）列表，绝不抛异常。
    """
    n = _to_int(limit)
    limit = n if n and n > 0 else 20
    gate = _RateGate(sleep or time.sleep)
    sess = session if session is not None else _default_session()

    payload = _request_json(
        ZHIHU_HOT_LIST_URL,
        session=sess,
        gate=gate,
        params={"limit": limit},
        headers={"Referer": ZHIHU_REFERER},
        tag="zhihu_hot",
    )
    if payload is None:
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        logger.warning("zhihu_hot data 字段非 list：%r", type(data).__name__)
        return []

    posts: List[Dict] = []
    for entry in data:
        if len(posts) >= limit:
            break
        if not isinstance(entry, dict):
            continue
        target = entry.get("target")
        if not isinstance(target, dict):
            continue
        qid = target.get("id")
        title = _strip_html(_clean_str(target.get("title")))
        if qid is None or not title:
            continue  # 无 id 无法拼 URL / 无标题无意义，逐条跳过

        metrics: Dict[str, int] = {}
        answers = _to_int(target.get("answer_count"))
        if answers is not None:
            metrics["comments"] = answers
        # 热度：实测在条目级 detail_text（如 "498 万热度"）；兼容 metrics_text
        heat = _parse_heat(entry.get("detail_text") or entry.get("metrics_text"))
        if heat is not None:
            metrics["heat"] = heat

        author = target.get("author")
        author_name = _clean_str(author.get("name")) if isinstance(author, dict) else ""

        posts.append({
            "platform": "zhihu",
            "post_id": str(qid),
            "title": title,
            "content": _strip_html(_clean_str(target.get("excerpt"))),
            "author": author_name,
            "metrics": metrics,
            "url": ZHIHU_QUESTION_URL.format(qid=qid),
            "published_at": _iso_from_unix(target.get("created")),
            "source": "zhihu_hot",
        })
    return posts
