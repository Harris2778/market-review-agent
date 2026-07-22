"""东方财富股吧舆情采集层（微舆/BettaFish 式 · 纯原创实现）。

端点 2026-07-22 实测定案（详见 research/guba_endpoints_recon.md）：

1. 个股吧帖子列表（JSON API）：POST
   https://gbapi.eastmoney.com/webarticlelist/api/Article/Articlelist，
   **必带魔法参数 deviceid=Wap10.0.0.1 + version=200**（缺失时 rc=0 且
   re=[]，提示「系统繁忙[00003]」）；业务参数 code=6 位代码、sorttype=0、
   ps（≤100）、p 分页。成功判据 rc==1；字段路径 re[].post_id /
   post_title / user_nickname / post_click_count(阅读) /
   post_comment_count(评论) / post_forward_count(转发) /
   post_publish_time（北京时间字符串，需 localize +08:00）。
   **列表项无点赞字段**（post_like_count 恒为 None）。
2. 帖子详情（HTML SSR，v1 唯一可行路径）：详情 JSON API 被端点级 WAF
   403 封死，改走 GET https://guba.eastmoney.com/news,{code},{post_id}.html，
   SSR 内嵌 `var post_article={...}` 完整 JSON（花括号配平提取后
   json.loads），含 post_content（HTML 正文）/ post_abstract（纯文本摘要）/
   post_like_count（**唯一点赞来源**）/ post_user.user_nickname 等 90+ 字段。
3. 评论 / 全站热榜 / 吧内搜索：2026-07-22 实测全部判死（需登录态 token
   或数据加密），v1 不实现。

合规注记：仅采集无登录公开数据；低频访问（≤1 req/s + 随机抖动）；
无登录自动化、无签名破解；平台条款/风控策略变动可能导致端点失效，
一律按 warning 日志观测并降级，绝不向调用方抛异常。

统一 Post 契约：
    {platform: 'guba', post_id: str, title: str, content: str(可空串),
     author: str(可空串), metrics: dict（只放实际拿到的键
     views/comments/shares/likes）, url: str,
     published_at: str(ISO 带 +08:00 时区或空串), source: str}
"""

import html
import json
import logging
import random
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:  # pragma: no cover - 依赖缺失时仅默认会话不可用
    requests = None

logger = logging.getLogger(__name__)

PLATFORM = "guba"

ARTICLE_LIST_URL = "https://gbapi.eastmoney.com/webarticlelist/api/Article/Articlelist"
DETAIL_URL_TMPL = "https://guba.eastmoney.com/news,{code},{post_id}.html"

# 魔法参数（2026-07-22 实测硬性门槛：缺失时 rc=0 空数据「系统繁忙[00003]」）
MAGIC_DEVICE_ID = "Wap10.0.0.1"
MAGIC_VERSION = "200"

DEFAULT_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
DEFAULT_TIMEOUT = 10
DEFAULT_RATE = 1.0    # 同一轮内连续请求间隔下限（秒）：≤1 req/s
DEFAULT_JITTER = 0.3  # 随机抖动上限（秒）
PAGE_SIZE_MAX = 100   # ps 实测上限
CONTENT_MAX_LEN = 2000  # 详情正文清洗后截断长度

CN_TZ = timezone(timedelta(hours=8))  # 北京时间 +08:00

_CODE_RE = re.compile(r"^\d{6}$")
_NEWS_URL_RE = re.compile(r"news,(\d{6}),(\d+)\.html")
_POST_ARTICLE_RE = re.compile(r"var\s+post_article\s*=")
_TAG_RE = re.compile(r"<[^>]+>")
_BLOCK_TAG_RE = re.compile(r"<(?:br|/p|/div|/li|/tr|/h[1-6])[^>]*>", re.IGNORECASE)
_WS_RE = re.compile(r"[ \t　]+")


# ═══════════════════════════════════════════
# 基础工具
# ═══════════════════════════════════════════

class _RateGate:
    """单轮限速门：同一轮内首个请求不限速，其后间隔 RATE + random(0, JITTER)。
    （模式照抄 agent/sentiment.py 的 _RateGate）"""

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
    """自建会话：绕开本机系统代理（trust_env=False）+ 浏览器 UA。"""
    if requests is None:
        raise RuntimeError("requests 库不可用")
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": DEFAULT_UA})
    return session


def _resolve_session(session):
    """返回可用会话；注入优先，未注入则自建。失败返回 None。"""
    if session is not None:
        return session
    try:
        return _new_session()
    except Exception as e:  # pragma: no cover - requests 缺失兜底
        logger.warning("guba 无法创建会话: %s", e)
        return None


def _to_int(value) -> Optional[int]:
    """宽松整数化：None/bool/'-'/'--'/非数值 → None，绝不抛出。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _clean_str(value) -> str:
    """归一字符串字段：None/占位符 → ''，其余 strip。"""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text in ("-", "--") else text


def _put_metric(metrics: dict, key: str, value) -> None:
    """仅在实际拿到数值时写入 metrics（契约：只放拿到的键）。"""
    num = _to_int(value)
    if num is not None:
        metrics[key] = num


def _to_beijing_iso(value) -> str:
    """'YYYY-MM-DD HH:MM:SS'（北京时间）→ ISO8601 带 +08:00；非法 → ''。"""
    text = _clean_str(value)
    if not text:
        return ""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text, fmt).replace(tzinfo=CN_TZ)
            return dt.isoformat()
        except ValueError:
            continue
    return ""


def _resp_json(resp) -> Optional[dict]:
    """从 response 提取 JSON 对象；顶层非 dict 或解析失败返回 None，绝不抛出。"""
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


def _resp_text(resp) -> str:
    """从 response 提取文本；失败返回 ''，绝不抛出。"""
    content = getattr(resp, "text", None)
    if content is not None:
        return str(content)
    raw = getattr(resp, "content", b"")
    return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)


def _strip_html_to_text(value, max_len: int = CONTENT_MAX_LEN) -> str:
    """正文 HTML → 纯文本：块级标签换行、去全部标签、反转义实体、
    折叠空白，截断到 max_len 字。"""
    if value is None:
        return ""
    text = str(value)
    text = _BLOCK_TAG_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    lines = [_WS_RE.sub(" ", ln).strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return text[:max_len]


# ═══════════════════════════════════════════
# 1. 帖子列表（Articlelist JSON API）
# ═══════════════════════════════════════════

def _parse_bar_post_item(row: dict, code: str) -> Optional[Dict]:
    """防御式解析单条列表项为 Post；缺 post_id 的行跳过返回 None。
    列表无点赞字段 → metrics 不含 likes。"""
    if not isinstance(row, dict):
        return None
    post_id_raw = row.get("post_id")
    if post_id_raw is None or _clean_str(post_id_raw) == "":
        return None
    post_id = _clean_str(post_id_raw)
    metrics: Dict[str, int] = {}
    _put_metric(metrics, "views", row.get("post_click_count"))
    _put_metric(metrics, "comments", row.get("post_comment_count"))
    _put_metric(metrics, "shares", row.get("post_forward_count"))
    return {
        "platform": PLATFORM,
        "post_id": post_id,
        "title": _clean_str(row.get("post_title")),
        "content": "",
        "author": _clean_str(row.get("user_nickname")),
        "metrics": metrics,
        "url": DETAIL_URL_TMPL.format(code=code, post_id=post_id),
        "published_at": _to_beijing_iso(row.get("post_publish_time")),
        "source": "guba_list",
    }


def _parse_bar_post_page(payload: dict, code: str) -> List[Dict]:
    """防御式解析一页列表；rc!=1 或非 dict 由调用方处理，这里只看 re。"""
    rows = payload.get("re")
    if not isinstance(rows, list):
        return []
    posts: List[Dict] = []
    for row in rows:
        post = _parse_bar_post_item(row, code)
        if post is not None:
            posts.append(post)
    return posts


def fetch_bar_posts(code, limit: int = 30, session=None,
                    sleep: Optional[Callable[[float], None]] = None) -> List[Dict]:
    """抓取个股吧帖子列表（POST Articlelist，分页聚合到 limit）。

    必带魔法参数 deviceid=Wap10.0.0.1 + version=200（缺失时服务端返回
    rc=0 空数据）。成功判据 rc==1；rc 非 1 / data 空 / 字段缺失一律降级：
    保留已抓到的部分并记 warning，绝不抛异常。非法 code（非 6 位数字）
    记 warning 返回 []。返回 Post 列表，source='guba_list'，metrics 无 likes。
    """
    code_text = _clean_str(code)
    if not _CODE_RE.match(code_text):
        logger.warning("guba_list 非法股票代码入参: %r", code)
        return []
    limit = max(1, _to_int(limit) or 30)
    sess = _resolve_session(session)
    if sess is None:
        return []
    gate = _RateGate(sleep or time.sleep)

    posts: List[Dict] = []
    page = 1
    while len(posts) < limit:
        page_size = min(PAGE_SIZE_MAX, limit - len(posts))
        form = {
            "deviceid": MAGIC_DEVICE_ID,   # 魔法参数必带
            "version": MAGIC_VERSION,      # 魔法参数必带
            "code": code_text,
            "sorttype": "0",
            "ps": str(page_size),
            "p": str(page),
            "type": "1",
            "m": "1",
        }
        gate.wait()
        try:
            resp = sess.post(ARTICLE_LIST_URL, data=form, timeout=DEFAULT_TIMEOUT)
        except Exception as e:
            logger.warning("guba_list 请求失败 p=%s: %s", page, e)
            break  # 请求失败：保留已抓到的部分
        sc = getattr(resp, "status_code", None)
        if isinstance(sc, int) and sc >= 400:
            logger.warning("guba_list HTTP %s p=%s", sc, page)
            break
        payload = _resp_json(resp)
        if payload is None:
            logger.warning("guba_list 响应不是合法 JSON 对象 p=%s", page)
            break
        rc = _to_int(payload.get("rc"))
        if rc != 1:
            logger.warning("guba_list rc=%s（成功判据 rc==1）p=%s message=%s",
                           payload.get("rc"), page,
                           _clean_str(payload.get("message") or payload.get("msg")))
            break
        page_posts = _parse_bar_post_page(payload, code_text)
        if not page_posts:
            break  # 空页终止
        posts.extend(page_posts)
        if len(page_posts) < page_size:
            break  # 本页不足 pageSize 视为最后一页
        page += 1
    return posts[:limit]


# ═══════════════════════════════════════════
# 2. 帖子详情（HTML SSR · var post_article 配平提取）
# ═══════════════════════════════════════════

def _extract_balanced_json(text: str, start: int) -> Optional[str]:
    """花括号配平法：从 text[start]（须为 '{'）起扫描，返回首个配平的
    JSON 对象子串。正确处理字符串内的花括号与转义引号（\\" \\\\），
    字符串外单/双引号均视作字符串定界。配平失败返回 None。"""
    if start < 0 or start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_str: Optional[str] = None
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str is not None:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
        else:
            if ch in ('"', "'"):
                in_str = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def _extract_post_article(page_text: str) -> Optional[dict]:
    """从详情页 HTML 提取 `var post_article={...}` 内嵌 JSON 并 json.loads。
    定位失败 / 配平失败 / JSON 非法均返回 None，绝不抛出。"""
    m = _POST_ARTICLE_RE.search(page_text)
    if m is None:
        return None
    brace_at = page_text.find("{", m.end())
    if brace_at < 0:
        return None
    blob = _extract_balanced_json(page_text, brace_at)
    if blob is None:
        return None
    try:
        payload = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _parse_detail(post_id: str, code: str, article: dict, url: str) -> Dict:
    """post_article JSON → 详情 dict。字段缺失降级为空串/缺键，绝不抛出。"""
    post_user = article.get("post_user")
    if not isinstance(post_user, dict):
        post_user = {}
    metrics: Dict[str, int] = {}
    _put_metric(metrics, "likes", article.get("post_like_count"))   # 唯一点赞来源
    _put_metric(metrics, "views", article.get("post_click_count"))
    _put_metric(metrics, "comments", article.get("post_comment_count"))
    _put_metric(metrics, "shares", article.get("post_forward_count"))
    return {
        "post_id": _clean_str(article.get("post_id")) or post_id,
        "title": _clean_str(article.get("post_title")),
        "content": _strip_html_to_text(article.get("post_content")),
        "abstract": _clean_str(article.get("post_abstract")),
        "metrics": metrics,
        "author": _clean_str(post_user.get("user_nickname")),
        "published_at": _to_beijing_iso(article.get("post_publish_time")),
        "url": url,
    }


def _fetch_detail(sess, url: str, *, gate: _RateGate, tag: str) -> Optional[dict]:
    """统一详情 GET：限速 → 请求 → 状态码 → 配平提取 → 解析。
    任何失败记 warning 返回 None，绝不抛出。"""
    gate.wait()
    try:
        resp = sess.get(url, timeout=DEFAULT_TIMEOUT)
    except Exception as e:
        logger.warning("%s 请求失败 %s: %s", tag, url, e)
        return None
    sc = getattr(resp, "status_code", None)
    if isinstance(sc, int) and sc >= 400:
        logger.warning("%s HTTP %s %s", tag, sc, url)
        return None
    page_text = _resp_text(resp)
    if not page_text:
        logger.warning("%s 响应为空：%s", tag, url)
        return None
    article = _extract_post_article(page_text)
    if article is None:
        logger.warning("%s 未能从页面提取 post_article JSON：%s", tag, url)
        return None
    return article


def fetch_post_detail(post_id, code: Optional[str] = None, session=None,
                      sleep: Optional[Callable[[float], None]] = None) -> Optional[Dict]:
    """抓取帖子详情（GET news,{code},{post_id}.html SSR 页）。

    用花括号配平法提取 `var post_article=` 后的 JSON 对象并 json.loads
    （配平正确处理字符串内花括号与转义引号）。返回 {post_id, title,
    content(正文 HTML 清洗成纯文本，截 2000 字), abstract, metrics:{likes,
    views, comments, shares}（只放拿到的键）, author, published_at, url}；
    code 缺失时可从 URL 形态的 post_id（news,{code},{id}.html）解析；
    提取失败 / HTTP 错误 / 参数不足记 warning 返回 None，绝不抛出。
    """
    pid = _clean_str(post_id)
    code_text = _clean_str(code)
    # code 缺失时尝试从 URL 形态的 post_id 解析（news,{code},{post_id}.html）
    m = _NEWS_URL_RE.search(pid)
    if m is not None:
        if not code_text:
            code_text = m.group(1)
        pid = m.group(2)
    if not pid:
        logger.warning("guba_detail 非法 post_id 入参: %r", post_id)
        return None
    if not _CODE_RE.match(code_text):
        logger.warning("guba_detail 缺少合法 6 位 code（post_id=%r, code=%r）",
                       post_id, code)
        return None
    sess = _resolve_session(session)
    if sess is None:
        return None
    url = DETAIL_URL_TMPL.format(code=code_text, post_id=pid)
    gate = _RateGate(sleep or time.sleep)
    article = _fetch_detail(sess, url, gate=gate, tag="guba_detail")
    if article is None:
        return None
    return _parse_detail(pid, code_text, article, url)


# ═══════════════════════════════════════════
# 3. 详情回填（点赞数唯一点来源）
# ═══════════════════════════════════════════

def _code_from_post(post: dict) -> str:
    """从 Post.url（news,{code},{id}.html）解析 6 位 code；失败返回 ''。"""
    m = _NEWS_URL_RE.search(_clean_str(post.get("url")))
    return m.group(1) if m else ""


def enrich_posts(posts: List[Dict], top_n: int = 3, session=None,
                 sleep: Optional[Callable[[float], None]] = None) -> List[Dict]:
    """对按 metrics.comments 降序的前 top_n 条抓详情，回填 content 与
    metrics.likes（列表无点赞字段，详情 post_like_count 是唯一来源）。

    详情失败（None）的帖子保留原样，不进 notes；返回与入参等长同序的
    **副本列表**（入参不被修改）。任何单帖失败记 warning 并继续，绝不抛。
    """
    if not posts:
        return []
    top_n = max(0, _to_int(top_n) or 0)
    result = [dict(p) if isinstance(p, dict) else p for p in posts]
    for p in result:
        if isinstance(p, dict) and isinstance(p.get("metrics"), dict):
            p["metrics"] = dict(p["metrics"])
    if top_n <= 0:
        return result

    def _comment_count(idx_post: Tuple[int, dict]) -> int:
        _, p = idx_post
        if not isinstance(p, dict):
            return 0
        metrics = p.get("metrics")
        if not isinstance(metrics, dict):
            return 0
        return _to_int(metrics.get("comments")) or 0

    ranked = sorted(enumerate(result), key=lambda t: _comment_count(t),
                    reverse=True)
    targets = [idx for idx, _ in ranked[:top_n]]

    sess = _resolve_session(session)
    if sess is None:
        return result
    gate = _RateGate(sleep or time.sleep)
    for idx in targets:
        post = result[idx]
        pid = _clean_str(post.get("post_id"))
        code_text = _code_from_post(post)
        if not pid or not code_text:
            logger.warning("guba_enrich 帖子缺 post_id/code，跳过回填: %r",
                           post.get("post_id"))
            continue
        url = DETAIL_URL_TMPL.format(code=code_text, post_id=pid)
        article = _fetch_detail(sess, url, gate=gate, tag="guba_enrich")
        if article is None:
            continue  # 详情失败：保留原帖，不进 notes
        detail = _parse_detail(pid, code_text, article, url)
        if detail["content"]:
            post["content"] = detail["content"]
        likes = detail["metrics"].get("likes")
        if likes is not None:
            post["metrics"]["likes"] = likes
    return result
