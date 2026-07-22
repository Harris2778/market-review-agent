"""B 站社媒舆情采集层（微舆/BettaFish 式 · 纯原创实现）。

端点 2026-07-22 实测定案（见 research/social_endpoints_recon.md 第 3 节）：

1. 热搜 square：GET https://api.bilibili.com/x/web-interface/search/square?limit=N
   → data.trending.list[] { keyword, show_name, icon, heat_score }，200 直通。
2. 热门 popular（可选合并，默认关）：GET .../x/web-interface/popular?ps=N&pn=1
   → data.list[] { aid, bvid, title, desc, pubdate, owner.name,
   stat.{view,danmaku,reply,favorite,coin,share,like} }，六维指标最全，无需 Cookie。
3. 搜索 type：GET .../x/web-interface/search/type?search_type=video|article&keyword=
   → data.result[]（过滤 type 匹配项）。裸请求 412「request was banned」，
   须先 GET https://www.bilibili.com 热身拿 buvid3 访客 Cookie 后重试；
   video 条目 title 含 <em class="keyword"> 高亮标签，必须清洗。
4. 评论 reply（plain 版）：GET https://api.bilibili.com/x/v2/reply?type=1&oid={aid}&sort=1&ps=N
   → data.replies[] { content.message, member.uname, like, rcount, ctime }；
   需 buvid3 预热；wbi 变体（reply/wbi/main）匿名实测 code=-403，但登录态
   （BILI_SESSDATA cookie + wbi 签名）可用——匿名每视频仅约 3 条热评，
   登录态走签名路径拉全量，签名材料失败降级 plain 带 cookie 请求。
   post_id 接受 aid 纯数字或 bvid（bvid 先经 x/web-interface/view?bvid= 转 aid）。

合规注记：仅采集无登录公开数据；低频访问（≤1 req/s + 随机抖动）；
无登录自动化、无签名破解；平台条款/风控变动可能导致端点失效，
一律按 warning 日志观测并降级，绝不向调用方抛异常。
"""

import hashlib
import html
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional
from urllib.parse import quote, urlencode

try:
    import requests
except ImportError:  # pragma: no cover - 依赖缺失时仅默认会话不可用
    requests = None

logger = logging.getLogger(__name__)

PLATFORM = "bilibili"

HOT_SQUARE_URL = "https://api.bilibili.com/x/web-interface/search/square"
POPULAR_URL = "https://api.bilibili.com/x/web-interface/popular"
SEARCH_TYPE_URL = "https://api.bilibili.com/x/web-interface/search/type"
REPLY_URL = "https://api.bilibili.com/x/v2/reply"
REPLY_WBI_URL = "https://api.bilibili.com/x/v2/reply/wbi/main"
NAV_URL = "https://api.bilibili.com/x/web-interface/nav"
VIEW_URL = "https://api.bilibili.com/x/web-interface/view"
WARMUP_URL = "https://www.bilibili.com"
REFERER = "https://www.bilibili.com/"

DEFAULT_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
DEFAULT_TIMEOUT = 10
DEFAULT_RATE = 1.0    # 同一轮内连续请求间隔下限（秒）：≤1 req/s
DEFAULT_JITTER = 0.3  # 随机抖动上限（秒）

# 评论翻页参数：plain 版 x/v2/reply 响应带 data.page{num,size,count,acount}
# 分页元信息，确认支持 pn 翻页（recon 报告第 3 节）；每页固定 ps=20。
REPLY_PAGE_SIZE = 20   # 翻页每页大小
REPLY_PAGE_MAX = 25    # 翻页页数安全上限（25 页 × 20 条 = 最多 500 条）

_TAG_RE = re.compile(r"<[^>]+>")
_AID_RE = re.compile(r"^\d+$")

# wbi 签名混淆表（标准 64 元素表，img_key+sub_key 重排后取前 32 位为 mixin key）
_MIXIN_KEY_ENC_TAB = (
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
)


# ── 基础工具 ──

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


def _strip_html(text) -> str:
    """清洗 HTML 高亮标签（如 <em class="keyword">）并反转义实体。"""
    if text is None:
        return ""
    cleaned = _TAG_RE.sub("", str(text))
    return html.unescape(cleaned).strip()


def _to_iso(ts) -> str:
    """unix 秒 → ISO8601（UTC）；缺失/非法 → ''。"""
    try:
        value = int(ts)
    except (TypeError, ValueError):
        return ""
    if value <= 0:
        return ""
    try:
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return ""


def _to_int(value) -> Optional[int]:
    """宽松整数化：'-'/'--'/None/非数值 → None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


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


def _warmup(session) -> bool:
    """会话预热：GET https://www.bilibili.com 拿 buvid3 访客 Cookie。

    搜索/评论端点裸请求会 412（缺 buvid3），预热后 Session 复用即可。
    任何异常记 warning 返回 False，绝不抛出。测试可 monkeypatch 本函数跳过。
    """
    try:
        resp = session.get(WARMUP_URL, timeout=DEFAULT_TIMEOUT,
                           headers={"Referer": REFERER})
    except Exception as e:
        logger.warning("bilibili 热身请求失败 %s: %s", WARMUP_URL, e)
        return False
    sc = getattr(resp, "status_code", None)
    if not isinstance(sc, int) or sc >= 400:
        logger.warning("bilibili 热身响应异常 HTTP %s", sc)
        return False
    return True


def _get_json(session, url: str, *, gate: _RateGate, tag: str,
              params: Optional[dict] = None, _retried: bool = False) -> Optional[dict]:
    """统一 GET：限速 → 请求 → 412 热身重试一次 → 状态码/code 校验。
    任何失败记 warning 返回 None，绝不抛出。"""
    gate.wait()
    try:
        resp = session.get(url, params=params, timeout=DEFAULT_TIMEOUT,
                           headers={"Referer": REFERER})
    except Exception as e:
        logger.warning("%s 请求失败 %s: %s", tag, url, e)
        return None
    sc = getattr(resp, "status_code", None)
    payload = _resp_json(resp)
    banned = sc == 412 or (isinstance(payload, dict) and payload.get("code") == -412)
    if banned:
        if _retried:
            logger.warning("%s 热身后仍 412，放弃：%s", tag, url)
            return None
        logger.warning("%s HTTP/JSON 412 风控（缺 buvid3），热身后重试一次：%s", tag, url)
        if not _warmup(session):
            logger.warning("%s 热身失败，降级放弃：%s", tag, url)
            return None
        return _get_json(session, url, gate=gate, tag=tag, params=params, _retried=True)
    if isinstance(sc, int) and sc >= 400:
        logger.warning("%s HTTP %s %s", tag, sc, url)
        return None
    if payload is None:
        logger.warning("%s 响应不是合法 JSON 对象：%s", tag, url)
        return None
    if "code" in payload and payload.get("code") != 0:
        logger.warning("%s code=%s message=%s：%s", tag,
                       payload.get("code"), payload.get("message"), url)
        return None
    return payload


def _resolve_session(session):
    """返回可用会话；注入优先，未注入则自建。失败返回 None。"""
    if session is not None:
        return session
    try:
        return _new_session()
    except Exception as e:  # pragma: no cover - requests 缺失兜底
        logger.warning("bilibili 无法创建会话: %s", e)
        return None


def _put_metric(metrics: dict, key: str, value) -> None:
    """仅在实际拿到数值时写入 metrics。"""
    num = _to_int(value)
    if num is not None:
        metrics[key] = num


# ── 登录态（BILI_SESSDATA）与 wbi 签名 ──
#
# 实测（2026-07-23）：评论接口匿名状态每视频仅返回约 3 条热评且多为一年
# 前旧评；配置 BILI_SESSDATA 环境变量（登录 cookie）后走 wbi 签名路径
# （x/v2/reply/wbi/main）可拉全量评论。无 SESSDATA 时行为与既有完全一致。


def _get_sessdata() -> str:
    """读 BILI_SESSDATA 环境变量；未配置返回 ''。绝不抛。"""
    try:
        return (os.environ.get("BILI_SESSDATA") or "").strip()
    except Exception:
        return ""


def _attach_sessdata(session, sessdata: str) -> bool:
    """把 SESSDATA 挂载到会话：优先 cookie jar，退化 dict/headers。绝不抛。"""
    try:
        session.cookies.set("SESSDATA", sessdata, domain=".bilibili.com")
        return True
    except Exception:
        pass
    try:
        session.cookies["SESSDATA"] = sessdata
        return True
    except Exception:
        pass
    try:
        session.headers.update({"Cookie": f"SESSDATA={sessdata}"})
        return True
    except Exception as e:
        logger.warning("bilibili SESSDATA 挂载失败: %s", e)
        return False


def _wbi_key_from_url(url) -> str:
    """从 wbi_img 的 img_url/sub_url 提取文件名（去扩展名）作为 key。"""
    try:
        name = str(url or "").rsplit("/", 1)[-1]
        return name.split(".", 1)[0]
    except Exception:
        return ""


def _get_wbi_mixin_key(session, gate: "_RateGate") -> Optional[str]:
    """经 x/web-interface/nav 取 wbi_img 推导 mixin key；失败返回 None，绝不抛。"""
    payload = _get_json(session, NAV_URL, gate=gate, tag="bilibili_nav")
    if payload is None:
        return None
    try:
        data = payload.get("data")
        wbi_img = data.get("wbi_img") if isinstance(data, dict) else None
        if not isinstance(wbi_img, dict):
            return None
        img_key = _wbi_key_from_url(wbi_img.get("img_url"))
        sub_key = _wbi_key_from_url(wbi_img.get("sub_url"))
        if not img_key or not sub_key:
            return None
        raw = img_key + sub_key
        return "".join(raw[i] for i in _MIXIN_KEY_ENC_TAB)[:32]
    except Exception as e:
        logger.warning("bilibili wbi mixin key 推导失败: %s", e)
        return None


def _sign_wbi_params(params: dict, mixin_key: str) -> dict:
    """wbi 签名：参数加 wts → 按键排序 urlencode → md5(query+mixin_key)=w_rid。"""
    signed = {k: v for k, v in params.items()}
    signed["wts"] = int(time.time())
    query = urlencode(sorted(signed.items()))
    signed["w_rid"] = hashlib.md5(
        (query + mixin_key).encode("utf-8")).hexdigest()
    return signed


def _fetch_comments_wbi(session, aid: str, limit: int, gate: "_RateGate",
                        mixin_key: str) -> List[Dict]:
    """登录态 wbi 签名评论路径（x/v2/reply/wbi/main，cursor 翻页）。

    单页失败保留已抓部分即止损；绝不抛。返回评论列表（可能为空）。
    """
    comments: List[Dict] = []
    next_cursor = 0
    for page_no in range(1, REPLY_PAGE_MAX + 1):
        if len(comments) >= limit:
            break
        params = _sign_wbi_params(
            {"type": 1, "oid": aid, "mode": 3,
             "ps": REPLY_PAGE_SIZE, "next": next_cursor},
            mixin_key)
        payload = _get_json(session, REPLY_WBI_URL, gate=gate,
                            tag="bilibili_reply_wbi", params=params)
        if payload is None:
            logger.warning("bilibili_reply_wbi 第 %s 页抓取失败，保留已抓 %s 条",
                           page_no, len(comments))
            break
        page = _parse_replies(payload, REPLY_PAGE_SIZE, aid)
        if not page:
            break  # 空页终止
        comments.extend(page)
        data = payload.get("data")
        cursor = data.get("cursor") if isinstance(data, dict) else None
        if isinstance(cursor, dict):
            if cursor.get("is_end"):
                break  # 游标到尾终止
            nxt = _to_int(cursor.get("next"))
            if nxt is None or nxt == next_cursor:
                break  # 游标不前进，防死循环
            next_cursor = nxt
        else:
            break  # 无游标信息，无法继续翻页
        if len(page) < REPLY_PAGE_SIZE:
            break  # 短页即末页
    return comments[:limit]


# ── 解析器（全部防御式）──

def _parse_trending(payload: dict, limit: int) -> List[Dict]:
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    trending = data.get("trending")
    if not isinstance(trending, dict):
        return []
    items = trending.get("list")
    if not isinstance(items, list):
        return []
    posts: List[Dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        keyword = str(item.get("keyword") or item.get("show_name") or "").strip()
        if not keyword:
            continue
        title = str(item.get("show_name") or keyword).strip()
        metrics: Dict[str, int] = {}
        _put_metric(metrics, "heat", item.get("heat_score"))
        posts.append({
            "platform": PLATFORM,
            "post_id": keyword,
            "title": title,
            "content": "",
            "author": "",
            "metrics": metrics,
            "url": "https://search.bilibili.com/all?keyword=" + quote(keyword),
            "published_at": "",
            "source": "bilibili_hot_search",
        })
        if len(posts) >= limit:
            break
    return posts


def _parse_popular(payload: dict, limit: int) -> List[Dict]:
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    items = data.get("list")
    if not isinstance(items, list):
        return []
    posts: List[Dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        bvid = str(item.get("bvid") or "").strip()
        aid = item.get("aid")
        post_id = bvid or (str(aid) if aid is not None else "")
        if not post_id:
            continue
        owner = item.get("owner") if isinstance(item.get("owner"), dict) else {}
        stat = item.get("stat") if isinstance(item.get("stat"), dict) else {}
        metrics: Dict[str, int] = {}
        _put_metric(metrics, "views", stat.get("view"))
        _put_metric(metrics, "likes", stat.get("like"))
        _put_metric(metrics, "comments", stat.get("reply"))
        _put_metric(metrics, "shares", stat.get("share"))
        _put_metric(metrics, "favorites", stat.get("favorite"))
        _put_metric(metrics, "coins", stat.get("coin"))
        posts.append({
            "platform": PLATFORM,
            "post_id": post_id,
            "title": _strip_html(item.get("title")),
            "content": str(item.get("desc") or "").strip(),
            "author": str(owner.get("name") or "").strip(),
            "metrics": metrics,
            "url": ("https://www.bilibili.com/video/" + bvid) if bvid else "",
            "published_at": _to_iso(item.get("pubdate")),
            "source": "bilibili_popular",
        })
        if len(posts) >= limit:
            break
    return posts


def _parse_search_result(payload: dict, limit: int, search_type: str) -> List[Dict]:
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    items = data.get("result")
    if not isinstance(items, list):
        return []
    posts: List[Dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != search_type:
            continue
        if search_type == "video":
            post = _parse_search_video(item)
        else:
            post = _parse_search_article(item)
        if post is None:
            continue
        posts.append(post)
        if len(posts) >= limit:
            break
    return posts


def _parse_search_video(item: dict) -> Optional[Dict]:
    bvid = str(item.get("bvid") or "").strip()
    aid = item.get("aid")
    post_id = bvid or (str(aid) if aid is not None else "")
    if not post_id:
        return None
    metrics: Dict[str, int] = {}
    _put_metric(metrics, "views", item.get("play"))        # play 可能是 "-"
    _put_metric(metrics, "comments", item.get("review"))
    _put_metric(metrics, "likes", item.get("like"))
    _put_metric(metrics, "favorites", item.get("favorites"))
    return {
        "platform": PLATFORM,
        "post_id": post_id,
        "title": _strip_html(item.get("title")),
        "content": _strip_html(item.get("description")),
        "author": str(item.get("author") or "").strip(),
        "metrics": metrics,
        "url": ("https://www.bilibili.com/video/" + bvid) if bvid else "",
        "published_at": _to_iso(item.get("pubdate")),
        "source": "bilibili_search_video",
    }


def _parse_search_article(item: dict) -> Optional[Dict]:
    article_id = item.get("id")
    if article_id is None:
        return None
    post_id = str(article_id)
    metrics: Dict[str, int] = {}
    _put_metric(metrics, "views", item.get("view"))
    _put_metric(metrics, "comments", item.get("reply"))
    _put_metric(metrics, "likes", item.get("like"))
    return {
        "platform": PLATFORM,
        "post_id": post_id,
        "title": _strip_html(item.get("title")),
        "content": _strip_html(item.get("desc")),
        "author": str(item.get("author") or "").strip(),
        "metrics": metrics,
        "url": "https://www.bilibili.com/read/cv" + post_id,
        "published_at": _to_iso(item.get("pubdate")),
        "source": "bilibili_search_article",
    }


def _parse_replies(payload: dict, limit: int, post_id: str) -> List[Dict]:
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    replies = data.get("replies")
    if not isinstance(replies, list):
        return []
    comments: List[Dict] = []
    for reply in replies:
        if not isinstance(reply, dict):
            continue
        content = reply.get("content") if isinstance(reply.get("content"), dict) else {}
        member = reply.get("member") if isinstance(reply.get("member"), dict) else {}
        message = str(content.get("message") or "").strip()
        if not message:
            continue
        comments.append({
            "platform": PLATFORM,
            "post_id": post_id,
            "author": str(member.get("uname") or "").strip(),
            "content": message,
            "likes": _to_int(reply.get("like")) or 0,
            "published_at": _to_iso(reply.get("ctime")),
        })
        if len(comments) >= limit:
            break
    return comments


# ── 公开 API ──

def fetch_hot(limit: int = 20, session=None, sleep: Optional[Callable[[float], None]] = None,
              include_popular: bool = False) -> List[Dict]:
    """B 站热搜榜（search/square，data.trending.list[]）。

    include_popular=True 时追加热门视频榜（popular，stat 六维指标最全）。
    任何单源失败记 warning 并返回已拿到的部分结果，绝不抛异常。
    """
    if limit <= 0:
        return []
    sess = _resolve_session(session)
    if sess is None:
        return []
    gate = _RateGate(sleep or time.sleep)
    posts: List[Dict] = []
    payload = _get_json(sess, HOT_SQUARE_URL, gate=gate, tag="bilibili_hot",
                        params={"limit": limit})
    if payload is not None:
        posts.extend(_parse_trending(payload, limit))
    if include_popular:
        popular = _get_json(sess, POPULAR_URL, gate=gate, tag="bilibili_popular",
                            params={"ps": limit, "pn": 1})
        if popular is not None:
            posts.extend(_parse_popular(popular, limit))
    return posts


def search(keyword: str, limit: int = 20, session=None,
           sleep: Optional[Callable[[float], None]] = None,
           search_type: str = "video", order: Optional[str] = None) -> List[Dict]:
    """B 站关键词搜索（search/type，search_type=video|article）。

    order=None（缺省）时行为与既有完全一致（综合排序）；传 order="pubdate"
    时请求参数带 order=pubdate（按发布时间倒序），仅用于采样路径。
    裸请求可能 412（缺 buvid3），自动热身重试一次。失败记 warning 返回 []，绝不抛。
    """
    keyword = str(keyword or "").strip()
    if not keyword or limit <= 0:
        return []
    if search_type not in ("video", "article"):
        logger.warning("bilibili_search 不支持的 search_type=%s", search_type)
        return []
    sess = _resolve_session(session)
    if sess is None:
        return []
    gate = _RateGate(sleep or time.sleep)
    params = {"search_type": search_type, "keyword": keyword, "page": 1}
    if order is not None:
        params["order"] = str(order)
    payload = _get_json(sess, SEARCH_TYPE_URL, gate=gate, tag="bilibili_search",
                        params=params)
    if payload is None:
        return []
    return _parse_search_result(payload, limit, search_type)


def fetch_comments(post_id, limit: int = 20, session=None,
                   sleep: Optional[Callable[[float], None]] = None) -> List[Dict]:
    """B 站评论（默认 x/v2/reply plain 版；登录态走 wbi 签名路径）。

    post_id 接受 aid 纯数字或 bvid（bvid 先经 view 接口转 aid）。
    失败记 warning 返回 []，绝不抛。

    登录态增强：配置 BILI_SESSDATA 环境变量时，会话挂载 SESSDATA cookie
    并走 wbi 签名路径（x/v2/reply/wbi/main，mode=3，cursor 翻页）拉全量
    评论——匿名状态每视频仅返回约 3 条热评。wbi 签名材料（nav wbi_img）
    获取失败时降级为不带签名的带 cookie 请求（plain 路径）。无 SESSDATA
    时行为与既有完全一致。

    翻页：plain 版响应带 data.page 分页元信息，确认支持 pn 翻页。
    limit<=20 维持单页直取（ps=limit，与既有行为一致）；limit>20 进入
    内部翻页循环（ps=20 每页，页间经 _RateGate 限速），终止条件：
    空页 / data.cursor.next==0 或 is_end / 短页（不足 ps 即末页）/
    页数达 REPLY_PAGE_MAX（25 页 = 500 条上限）/ 单页请求失败
    （失败保留已抓部分，绝不抛）。
    """
    pid = str(post_id or "").strip()
    if not pid or limit <= 0:
        return []
    sess = _resolve_session(session)
    if sess is None:
        return []
    gate = _RateGate(sleep or time.sleep)
    if _AID_RE.match(pid):
        aid = pid
    else:
        view = _get_json(sess, VIEW_URL, gate=gate, tag="bilibili_view",
                         params={"bvid": pid})
        if view is None:
            return []
        data = view.get("data")
        aid_val = data.get("aid") if isinstance(data, dict) else None
        if aid_val is None:
            logger.warning("bilibili_view 响应缺少 aid：bvid=%s", pid)
            return []
        aid = str(aid_val)
    # 登录态路径：SESSDATA + wbi 签名（签名材料失败降级 plain 带 cookie）
    sessdata = _get_sessdata()
    if sessdata:
        _attach_sessdata(sess, sessdata)
        mixin_key = _get_wbi_mixin_key(sess, gate)
        if mixin_key is not None:
            return _fetch_comments_wbi(sess, aid, limit, gate, mixin_key)
        logger.warning("bilibili wbi 签名材料获取失败，"
                       "降级为不带签名的带 cookie 请求（plain 路径）")
    if limit <= REPLY_PAGE_SIZE:
        payload = _get_json(sess, REPLY_URL, gate=gate, tag="bilibili_reply",
                            params={"type": 1, "oid": aid, "sort": 1, "ps": limit})
        if payload is None:
            return []
        return _parse_replies(payload, limit, aid)
    # 翻页循环：ps=20 每页，保留已抓部分，任何单页失败即止损退出
    comments: List[Dict] = []
    pn = 1
    while len(comments) < limit and pn <= REPLY_PAGE_MAX:
        payload = _get_json(sess, REPLY_URL, gate=gate, tag="bilibili_reply",
                            params={"type": 1, "oid": aid, "sort": 1,
                                    "ps": REPLY_PAGE_SIZE, "pn": pn})
        if payload is None:
            logger.warning("bilibili_reply 第 %s 页抓取失败，保留已抓 %s 条",
                           pn, len(comments))
            break
        page = _parse_replies(payload, REPLY_PAGE_SIZE, aid)
        if not page:
            break  # 空页终止
        comments.extend(page)
        data = payload.get("data")
        cursor = data.get("cursor") if isinstance(data, dict) else None
        if isinstance(cursor, dict):
            if _to_int(cursor.get("next")) == 0 or cursor.get("is_end"):
                break  # 游标到尾终止
        if len(page) < REPLY_PAGE_SIZE:
            break  # 短页即末页
        pn += 1
    return comments[:limit]
