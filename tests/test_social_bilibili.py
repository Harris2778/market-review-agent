"""agent/social_bilibili.py 单元测试（全 mock 零网络）。

fixture 按 research/social_endpoints_recon.md（2026-07-22 实测定案）与
research/recon_raw*.json 抓包结构构造。
"""

import json
import logging

import pytest

import agent.social_bilibili as sb


# ── 测试替身 ──

class FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False) if payload is not None else ""

    def json(self):
        if self._payload is None:
            raise ValueError("No JSON")
        return self._payload


class FakeSession:
    """按 URL 子串路由到响应队列；记录全部调用。cookies 模拟热身种 Cookie。"""

    def __init__(self, routes=None, on_get=None):
        self.routes = routes or {}
        self.on_get = on_get
        self.calls = []
        self.cookies = {}

    def get(self, url, params=None, timeout=None, headers=None, **kw):
        self.calls.append({"url": url, "params": params, "headers": headers})
        if self.on_get is not None:
            self.on_get(url, self)
        for key, queue in self.routes.items():
            if key in url:
                if queue:
                    return queue.pop(0)
                return FakeResp(500, None)
        return FakeResp(404, None)

    def urls(self, needle):
        return [c["url"] for c in self.calls if needle in c["url"]]


class FakeSleeps:
    def __init__(self):
        self.values = []

    def __call__(self, seconds):
        self.values.append(seconds)


# ── fixture 原型（结构对齐 recon 抓包）──

def trending_payload():
    return {"code": 0, "message": "OK", "ttl": 1, "data": {"trending": {
        "title": "bilibili热搜", "trackid": "12259440515728026421",
        "list": [
            {"keyword": "A股 大涨", "show_name": "A股大涨 沪指创新高", "icon": "",
             "uri": "", "goto": "", "heat_score": 36487390},
            {"keyword": "半导体 反弹", "show_name": "半导体板块反弹", "icon": "",
             "uri": "", "goto": "", "heat_score": 1584455},
        ]}}}


def popular_payload():
    return {"code": 0, "message": "OK", "ttl": 1, "data": {"list": [
        {"aid": 116953150725617, "bvid": "BV1AnKs6tE2n", "cid": 40140474831,
         "tname": "财经商业", "title": "A股复盘：放量大涨", "desc": "今日复盘",
         "pubdate": 1784685600,
         "owner": {"mid": 1654550386, "name": "财经UP主", "face": "https://x/f.jpg"},
         "stat": {"view": 4390464, "danmaku": 2225, "reply": 2886,
                  "favorite": 118118, "coin": 104380, "share": 3749, "like": 275328}},
    ]}}


def search_video_payload():
    return {"code": 0, "message": "OK", "ttl": 1, "data": {
        "numResults": 1000, "numPages": 50, "result": [
            {"type": "video", "id": 116956774797518, "aid": 116956774797518,
             "bvid": "BV1TAK86VEAJ", "author": "擒龙先生", "typename": "财经商业",
             "arcurl": "http://www.bilibili.com/video/av116956774797518",
             "title": '<em class="keyword">a股</em>惊天大逆转！ETF大涨19%！',
             "description": "半导体<em class=\"keyword\">反弹</em>先锋",
             "play": 113873, "video_review": 368, "favorites": 418,
             "review": 1659, "like": 2800, "pubdate": 1784619114},
            {"type": "video", "id": 2, "aid": 2, "bvid": "BV2xx411c7mD",
             "author": "某UP", "title": "无高亮标题", "description": "",
             "play": "-", "review": 3, "like": 4, "favorites": 5,
             "pubdate": 1784619000},
            {"type": "live_room", "id": 99},  # 非 video 应被过滤
        ]}}


def search_article_payload():
    return {"code": 0, "message": "OK", "ttl": 1, "data": {
        "numResults": 10, "numPages": 1, "result": [
            {"type": "article", "id": 40123456, "title": '<em class="keyword">A股</em>周报',
             "desc": "本周市场回顾", "author": "专栏作者", "view": 12345,
             "like": 321, "reply": 54, "pubdate": 1784610000,
             "category_name": "财经"},
        ]}}


def view_payload(aid=116963821095272):
    return {"code": 0, "message": "OK", "ttl": 1, "data": {"aid": aid,
            "bvid": "BV1bXKsztEQ1", "title": "某视频"}}


def reply_payload():
    return {"code": 0, "message": "OK", "ttl": 1, "data": {
        "page": {"num": 1, "size": 5, "count": 2, "acount": 2},
        "replies": [
            {"rpid_str": "310415642848", "ctime": 1784726572, "like": 12,
             "rcount": 3,
             "content": {"message": "满仓半导体，冲！"},
             "member": {"mid": "3690990567163923", "uname": "启航财经"}},
            {"rpid_str": "310415642849", "ctime": 1784726600, "like": 0,
             "rcount": 0,
             "content": {"message": "观望中"},
             "member": {"mid": "1", "uname": "路人甲"}},
        ]}}


BANNED = FakeResp(412, {"code": -412, "message": "request was banned", "ttl": 1})


@pytest.fixture(autouse=True)
def no_real_warmup(monkeypatch):
    """默认把热身 mock 掉（成功且种 buvid3），需要测热身本体的用例自行恢复。"""
    def fake_warmup(session):
        session.cookies["buvid3"] = "FAKE-BUVID3"
        return True
    monkeypatch.setattr(sb, "_warmup", fake_warmup)


# ── fetch_hot ──

def test_hot_parse_basic():
    sess = FakeSession({"search/square": [FakeResp(200, trending_payload())]})
    posts = sb.fetch_hot(limit=20, session=sess, sleep=lambda s: None)
    assert len(posts) == 2
    p = posts[0]
    assert p["platform"] == "bilibili"
    assert p["post_id"] == "A股 大涨"
    assert p["title"] == "A股大涨 沪指创新高"
    assert p["metrics"] == {"heat": 36487390}
    assert p["url"] == "https://search.bilibili.com/all?keyword=A%E8%82%A1%20%E5%A4%A7%E6%B6%A8"
    assert p["published_at"] == ""
    assert p["source"] == "bilibili_hot_search"


def test_hot_limit_slice():
    sess = FakeSession({"search/square": [FakeResp(200, trending_payload())]})
    posts = sb.fetch_hot(limit=1, session=sess, sleep=lambda s: None)
    assert len(posts) == 1


def test_hot_code_nonzero_degrades():
    sess = FakeSession({"search/square": [FakeResp(200, {"code": -404, "message": "啥都木有"})]})
    assert sb.fetch_hot(session=sess, sleep=lambda s: None) == []


def test_hot_http_500_degrades():
    sess = FakeSession({"search/square": [FakeResp(500, None)]})
    assert sb.fetch_hot(session=sess, sleep=lambda s: None) == []


def test_hot_malformed_json_degrades():
    sess = FakeSession({"search/square": [FakeResp(200, None)]})
    assert sb.fetch_hot(session=sess, sleep=lambda s: None) == []


def test_hot_missing_trending_degrades():
    sess = FakeSession({"search/square": [FakeResp(200, {"code": 0, "data": {}})]})
    assert sb.fetch_hot(session=sess, sleep=lambda s: None) == []


def test_hot_include_popular_merges_six_metrics():
    sess = FakeSession({
        "search/square": [FakeResp(200, trending_payload())],
        "popular": [FakeResp(200, popular_payload())],
    })
    posts = sb.fetch_hot(limit=20, session=sess, sleep=lambda s: None, include_popular=True)
    assert len(posts) == 3
    pop = posts[-1]
    assert pop["source"] == "bilibili_popular"
    assert pop["post_id"] == "BV1AnKs6tE2n"
    assert pop["author"] == "财经UP主"
    assert pop["metrics"] == {"views": 4390464, "likes": 275328, "comments": 2886,
                              "shares": 3749, "favorites": 118118, "coins": 104380}
    assert pop["url"] == "https://www.bilibili.com/video/BV1AnKs6tE2n"
    assert pop["published_at"].startswith("2026-07-")  # pubdate 1784685600 → ISO


def test_hot_default_excludes_popular():
    sess = FakeSession({"search/square": [FakeResp(200, trending_payload())],
                        "popular": [FakeResp(200, popular_payload())]})
    posts = sb.fetch_hot(session=sess, sleep=lambda s: None)
    assert all(p["source"] == "bilibili_hot_search" for p in posts)
    assert sess.urls("popular") == []


def test_hot_popular_failure_keeps_hot():
    sess = FakeSession({
        "search/square": [FakeResp(200, trending_payload())],
        "popular": [FakeResp(500, None)],
    })
    posts = sb.fetch_hot(session=sess, sleep=lambda s: None, include_popular=True)
    assert len(posts) == 2


# ── search ──

def test_search_video_parse_and_em_clean():
    sess = FakeSession({"search/type": [FakeResp(200, search_video_payload())]})
    posts = sb.search("a股", session=sess, sleep=lambda s: None)
    assert len(posts) == 2  # live_room 被过滤
    p = posts[0]
    assert p["title"] == "a股惊天大逆转！ETF大涨19%！"
    assert "<em" not in p["title"]
    assert p["content"] == "半导体反弹先锋"
    assert p["author"] == "擒龙先生"
    assert p["metrics"] == {"views": 113873, "comments": 1659, "likes": 2800,
                            "favorites": 418}
    assert p["url"] == "https://www.bilibili.com/video/BV1TAK86VEAJ"
    assert p["published_at"].startswith("2026-07-")
    assert p["source"] == "bilibili_search_video"


def test_search_play_dash_no_views_metric():
    sess = FakeSession({"search/type": [FakeResp(200, search_video_payload())]})
    posts = sb.search("a股", session=sess, sleep=lambda s: None)
    assert "views" not in posts[1]["metrics"]  # play="-" 不入 metrics


def test_search_article_parse():
    sess = FakeSession({"search/type": [FakeResp(200, search_article_payload())]})
    posts = sb.search("A股", session=sess, sleep=lambda s: None, search_type="article")
    assert len(posts) == 1
    p = posts[0]
    assert p["post_id"] == "40123456"
    assert p["title"] == "A股周报"
    assert p["metrics"] == {"views": 12345, "comments": 54, "likes": 321}
    assert p["url"] == "https://www.bilibili.com/read/cv40123456"
    assert p["source"] == "bilibili_search_article"


def test_search_bad_type_returns_empty():
    sess = FakeSession()
    assert sb.search("a股", session=sess, sleep=lambda s: None, search_type="live") == []
    assert sess.calls == []


def test_search_empty_keyword_returns_empty():
    sess = FakeSession()
    assert sb.search("  ", session=sess, sleep=lambda s: None) == []
    assert sess.calls == []


def test_search_412_warmup_retry_chain(monkeypatch):
    """裸请求 412 → 热身拿 buvid3 → 重试成功。"""
    warmups = []
    def spy_warmup(session):
        warmups.append(1)
        session.cookies["buvid3"] = "REAL-ISH"
        return True
    monkeypatch.setattr(sb, "_warmup", spy_warmup)
    sess = FakeSession({"search/type": [BANNED, FakeResp(200, search_video_payload())]})
    posts = sb.search("a股", session=sess, sleep=lambda s: None)
    assert len(warmups) == 1
    assert len(posts) == 2
    assert len(sess.urls("search/type")) == 2  # 原请求 + 重试


def test_search_412_twice_gives_up(monkeypatch):
    warmups = []
    monkeypatch.setattr(sb, "_warmup", lambda s: warmups.append(1) or True)
    sess = FakeSession({"search/type": [BANNED, BANNED]})
    assert sb.search("a股", session=sess, sleep=lambda s: None) == []
    assert len(warmups) == 1  # 只热身一次
    assert len(sess.urls("search/type")) == 2  # 只重试一次


def test_search_412_warmup_failure_degrades(monkeypatch, caplog):
    monkeypatch.setattr(sb, "_warmup", lambda s: False)
    sess = FakeSession({"search/type": [BANNED]})
    with caplog.at_level(logging.WARNING, logger="agent.social_bilibili"):
        assert sb.search("a股", session=sess, sleep=lambda s: None) == []
    assert any("热身失败" in r.message for r in caplog.records)
    assert len(sess.urls("search/type")) == 1  # 不重试


def test_search_code_nonzero_degrades():
    sess = FakeSession({"search/type": [FakeResp(200, {"code": -400, "message": "请求错误"})]})
    assert sb.search("a股", session=sess, sleep=lambda s: None) == []


def test_search_result_not_list_degrades():
    sess = FakeSession({"search/type": [FakeResp(200, {"code": 0, "data": {"result": None}})]})
    assert sb.search("a股", session=sess, sleep=lambda s: None) == []


# ── fetch_comments ──

def test_comments_with_aid_direct():
    sess = FakeSession({"x/v2/reply": [FakeResp(200, reply_payload())]})
    comments = sb.fetch_comments("116963821095272", session=sess, sleep=lambda s: None)
    assert len(comments) == 2
    c = comments[0]
    assert c["platform"] == "bilibili"
    assert c["post_id"] == "116963821095272"
    assert c["author"] == "启航财经"
    assert c["content"] == "满仓半导体，冲！"
    assert c["likes"] == 12
    assert c["published_at"].startswith("2026-07-")
    assert sess.urls("web-interface/view") == []  # aid 直通，不查 view
    params = sess.calls[0]["params"]
    assert params["oid"] == "116963821095272" and params["type"] == 1


def test_comments_with_bvid_resolves_aid_first():
    sess = FakeSession({
        "web-interface/view": [FakeResp(200, view_payload(aid=116963821095272))],
        "x/v2/reply": [FakeResp(200, reply_payload())],
    })
    comments = sb.fetch_comments("BV1bXKsztEQ1", session=sess, sleep=lambda s: None)
    assert len(comments) == 2
    assert len(sess.urls("web-interface/view")) == 1
    reply_call = [c for c in sess.calls if "x/v2/reply" in c["url"]][0]
    assert reply_call["params"]["oid"] == "116963821095272"
    assert comments[0]["post_id"] == "116963821095272"


def test_comments_bvid_view_fails_degrades():
    sess = FakeSession({"web-interface/view": [FakeResp(500, None)]})
    assert sb.fetch_comments("BV1bXKsztEQ1", session=sess, sleep=lambda s: None) == []
    assert sess.urls("x/v2/reply") == []


def test_comments_bvid_view_missing_aid_degrades():
    sess = FakeSession({"web-interface/view": [FakeResp(200, {"code": 0, "data": {}})]})
    assert sb.fetch_comments("BV1bXKsztEQ1", session=sess, sleep=lambda s: None) == []


def test_comments_412_warmup_retry(monkeypatch):
    warmups = []
    monkeypatch.setattr(sb, "_warmup", lambda s: warmups.append(1) or True)
    sess = FakeSession({"x/v2/reply": [BANNED, FakeResp(200, reply_payload())]})
    comments = sb.fetch_comments("116963821095272", session=sess, sleep=lambda s: None)
    assert len(warmups) == 1
    assert len(comments) == 2


def test_comments_code_nonzero_degrades():
    sess = FakeSession({"x/v2/reply": [FakeResp(200, {"code": -403, "message": "访问权限不足"})]})
    assert sb.fetch_comments("116963821095272", session=sess, sleep=lambda s: None) == []


def test_comments_replies_null_degrades():
    sess = FakeSession({"x/v2/reply": [FakeResp(200, {"code": 0, "data": {"replies": None}})]})
    assert sb.fetch_comments("116963821095272", session=sess, sleep=lambda s: None) == []


def test_comments_empty_message_skipped():
    payload = reply_payload()
    payload["data"]["replies"][0]["content"]["message"] = ""
    sess = FakeSession({"x/v2/reply": [FakeResp(200, payload)]})
    comments = sb.fetch_comments("116963821095272", session=sess, sleep=lambda s: None)
    assert len(comments) == 1
    assert comments[0]["content"] == "观望中"


def test_comments_limit_slice():
    sess = FakeSession({"x/v2/reply": [FakeResp(200, reply_payload())]})
    comments = sb.fetch_comments("116963821095272", limit=1, session=sess, sleep=lambda s: None)
    assert len(comments) == 1


# ── 基础设施：session 注入 / 自建 / 限速 / 热身本体 ──

def test_session_injection_used():
    sess = FakeSession({"search/square": [FakeResp(200, trending_payload())]})
    sb.fetch_hot(session=sess, sleep=lambda s: None)
    assert len(sess.calls) == 1
    assert sess.calls[0]["headers"]["Referer"] == "https://www.bilibili.com/"


def test_new_session_defaults(monkeypatch):
    """未注入 session 时自建：trust_env=False + 浏览器 UA（不发真实请求）。"""
    if sb.requests is None:
        pytest.skip("requests 不可用")
    created = []
    real_session = sb.requests.Session
    def spy_session():
        s = real_session()
        created.append(s)
        return s
    monkeypatch.setattr(sb.requests, "Session", spy_session)
    # 自建 session 发请求会失败（网络不可达），fetch_hot 应降级返回 [] 且不抛
    class Boom:
        def __init__(self): self.trust_env = True; self.headers = {}
        def get(self, *a, **k): raise RuntimeError("no network in tests")
    monkeypatch.setattr(sb, "_new_session", lambda: created.append(Boom()) or created[-1])
    assert sb.fetch_hot(sleep=lambda s: None) == []
    assert created and created[-1].trust_env is True  # Boom 的默认值，只验证被使用
    # _new_session 本体：trust_env=False + UA
    monkeypatch.undo()
    s = sb._new_session()
    assert s.trust_env is False
    assert "Mozilla" in s.headers.get("User-Agent", "")


def test_rate_gate_first_free_then_sleep():
    sleeps = FakeSleeps()
    sess = FakeSession({
        "search/square": [FakeResp(200, trending_payload())],
        "popular": [FakeResp(200, popular_payload())],
    })
    sb.fetch_hot(session=sess, sleep=sleeps, include_popular=True)
    assert len(sleeps.values) == 1  # 首个请求不限速，第二次限速
    assert 1.0 <= sleeps.values[0] <= 1.3


def test_rate_gate_comments_bvid_two_requests():
    sleeps = FakeSleeps()
    sess = FakeSession({
        "web-interface/view": [FakeResp(200, view_payload())],
        "x/v2/reply": [FakeResp(200, reply_payload())],
    })
    sb.fetch_comments("BV1bXKsztEQ1", session=sess, sleep=sleeps)
    assert len(sleeps.values) == 1


def test_warmup_real_gets_homepage(monkeypatch):
    """热身本体：GET https://www.bilibili.com，2xx 视为成功。"""
    monkeypatch.undo()  # 还原 fixture 的 _warmup 替换，测本体
    sess = FakeSession({"www.bilibili.com": [FakeResp(200, None)]})
    assert sb._warmup(sess) is True
    assert sess.calls[0]["url"] == "https://www.bilibili.com"


def test_warmup_real_http_error(monkeypatch):
    monkeypatch.undo()  # 还原 fixture 的 _warmup 替换
    sess = FakeSession({"www.bilibili.com": [FakeResp(503, None)]})
    assert sb._warmup(sess) is False


def test_warmup_real_exception(monkeypatch):
    monkeypatch.undo()

    class ExplodingSession:
        def get(self, *a, **k):
            raise RuntimeError("boom")

    assert sb._warmup(ExplodingSession()) is False


# ── 工具函数 ──

def test_strip_html_nested_and_entities():
    assert sb._strip_html('<em class="keyword">a股</em>&amp;<b>牛市</b>') == "a股&牛市"
    assert sb._strip_html(None) == ""


def test_to_iso_invalid():
    assert sb._to_iso(None) == ""
    assert sb._to_iso("abc") == ""
    assert sb._to_iso(-1) == ""
    assert sb._to_iso(1784619114).startswith("2026-07-21")


def test_never_raises_on_garbage(caplog):
    """全链路垃圾输入：任何失败都是 warning + 空结果，绝不抛。"""
    sess = FakeSession()  # 所有路由 404
    with caplog.at_level(logging.WARNING, logger="agent.social_bilibili"):
        assert sb.fetch_hot(session=sess, sleep=lambda s: None) == []
        assert sb.search("a股", session=sess, sleep=lambda s: None) == []
        assert sb.fetch_comments("BVxxx", session=sess, sleep=lambda s: None) == []
    assert len(caplog.records) > 0


def test_limit_zero_returns_empty():
    sess = FakeSession()
    assert sb.fetch_hot(limit=0, session=sess, sleep=lambda s: None) == []
    assert sb.search("a股", limit=0, session=sess, sleep=lambda s: None) == []
    assert sb.fetch_comments("123", limit=0, session=sess, sleep=lambda s: None) == []
    assert sess.calls == []
