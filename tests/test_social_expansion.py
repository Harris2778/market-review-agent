"""舆情采样扩容测试（全 mock 零网络）。

覆盖：
1. agent/social_bilibili.fetch_comments 的 pn 翻页增强
   （多页聚合 / next==0 终止 / 空页终止 / 短页终止 / 单页失败保留已抓 /
   bvid 解析后翻页 / 限速 / 默认行为不变）；
2. agent/social_media.collect_keyword_samples 合并与部分失败降级；
3. agent/social_media.collect_guba_samples 透传与 enrich 降级。

所有网络走 FakeSession / 假模块注入（sys.modules），零真实请求。
"""

import json
import sys
import types

import pytest

import agent.social_bilibili as sb
import agent.social_media as sm


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
    """按 URL 子串路由到响应队列；记录全部调用。"""

    def __init__(self, routes=None):
        self.routes = routes or {}
        self.calls = []

    def get(self, url, params=None, timeout=None, headers=None, **kw):
        self.calls.append({"url": url, "params": params, "headers": headers})
        for key, queue in self.routes.items():
            if key in url:
                if queue:
                    return queue.pop(0)
                return FakeResp(500, None)
        return FakeResp(404, None)

    def urls(self, needle):
        return [c["url"] for c in self.calls if needle in c["url"]]

    def reply_params(self):
        return [c["params"] for c in self.calls if "x/v2/reply" in c["url"]]


class FakeSleeps:
    def __init__(self):
        self.values = []

    def __call__(self, seconds):
        self.values.append(seconds)


@pytest.fixture(autouse=True)
def no_real_warmup(monkeypatch):
    monkeypatch.setattr(sb, "_warmup", lambda session: True)


# ── 翻页 payload 构造 ──

def reply_page(tag, n, next_val=None, is_end=False):
    """构造一页评论 payload；next_val 非 None 时附 data.cursor。"""
    replies = [
        {"rpid_str": f"{tag}{i}", "ctime": 1784726572 + i, "like": i,
         "rcount": 0,
         "content": {"message": f"评论{tag}-{i}"},
         "member": {"mid": "1", "uname": f"用户{tag}{i}"}}
        for i in range(n)
    ]
    data = {"page": {"num": 1, "size": n, "count": n, "acount": n},
            "replies": replies}
    if next_val is not None:
        data["cursor"] = {"all_count": 1000, "is_begin": False, "prev": 0,
                          "next": next_val, "is_end": is_end}
    return {"code": 0, "message": "OK", "ttl": 1, "data": data}


# ══ 1. fetch_comments 翻页增强 ══

def test_default_limit_single_page_unchanged():
    """默认 limit=20：单页直取 ps=20，不带 pn，行为与增强前一致。"""
    sess = FakeSession({"x/v2/reply": [FakeResp(200, reply_page("a", 20))]})
    comments = sb.fetch_comments("123", session=sess, sleep=lambda s: None)
    assert len(comments) == 20
    params = sess.reply_params()
    assert len(params) == 1
    assert params[0]["ps"] == 20
    assert "pn" not in params[0]


def test_small_limit_ps_equals_limit_single_request():
    sess = FakeSession({"x/v2/reply": [FakeResp(200, reply_page("a", 5))]})
    comments = sb.fetch_comments("123", limit=5, session=sess, sleep=lambda s: None)
    assert len(comments) == 5
    params = sess.reply_params()
    assert len(params) == 1 and params[0]["ps"] == 5


def test_pagination_multi_page_collects_all():
    """limit=45：20+20+5 三页聚合，pn 递增、ps 固定 20。"""
    sess = FakeSession({"x/v2/reply": [
        FakeResp(200, reply_page("a", 20, next_val=2)),
        FakeResp(200, reply_page("b", 20, next_val=3)),
        FakeResp(200, reply_page("c", 5, next_val=0)),
    ]})
    comments = sb.fetch_comments("123", limit=45, session=sess, sleep=lambda s: None)
    assert len(comments) == 45
    params = sess.reply_params()
    assert [p["pn"] for p in params] == [1, 2, 3]
    assert all(p["ps"] == sb.REPLY_PAGE_SIZE for p in params)
    assert comments[0]["content"] == "评论a-0"
    assert comments[-1]["content"] == "评论c-4"


def test_pagination_terminates_on_next_zero():
    """cursor.next==0 立即终止，不再请求后续页。"""
    sess = FakeSession({"x/v2/reply": [
        FakeResp(200, reply_page("a", 20, next_val=2)),
        FakeResp(200, reply_page("b", 20, next_val=0)),
        FakeResp(200, reply_page("c", 20, next_val=0)),  # 不应被消费
    ]})
    comments = sb.fetch_comments("123", limit=100, session=sess, sleep=lambda s: None)
    assert len(comments) == 40
    assert len(sess.reply_params()) == 2


def test_pagination_terminates_on_is_end():
    sess = FakeSession({"x/v2/reply": [
        FakeResp(200, reply_page("a", 20, next_val=2, is_end=True)),
        FakeResp(200, reply_page("b", 20, next_val=3)),
    ]})
    comments = sb.fetch_comments("123", limit=100, session=sess, sleep=lambda s: None)
    assert len(comments) == 20
    assert len(sess.reply_params()) == 1


def test_pagination_terminates_on_empty_page():
    sess = FakeSession({"x/v2/reply": [
        FakeResp(200, reply_page("a", 20, next_val=2)),
        FakeResp(200, {"code": 0, "message": "OK", "data": {"replies": []}}),
    ]})
    comments = sb.fetch_comments("123", limit=100, session=sess, sleep=lambda s: None)
    assert len(comments) == 20
    assert len(sess.reply_params()) == 2


def test_pagination_terminates_on_short_page():
    """无 cursor 时短页（不足 ps）视为末页。"""
    sess = FakeSession({"x/v2/reply": [
        FakeResp(200, reply_page("a", 20)),
        FakeResp(200, reply_page("b", 7)),
        FakeResp(200, reply_page("c", 20)),  # 不应被消费
    ]})
    comments = sb.fetch_comments("123", limit=100, session=sess, sleep=lambda s: None)
    assert len(comments) == 27
    assert len(sess.reply_params()) == 2


def test_pagination_http_failure_keeps_collected():
    """第 2 页 HTTP 500：保留第 1 页已抓结果，绝不抛。"""
    sess = FakeSession({"x/v2/reply": [
        FakeResp(200, reply_page("a", 20, next_val=2)),
        FakeResp(500, None),
    ]})
    comments = sb.fetch_comments("123", limit=100, session=sess, sleep=lambda s: None)
    assert len(comments) == 20


def test_pagination_code_error_keeps_collected():
    """第 2 页 code=-403：保留已抓部分。"""
    sess = FakeSession({"x/v2/reply": [
        FakeResp(200, reply_page("a", 20, next_val=2)),
        FakeResp(200, {"code": -403, "message": "访问权限不足"}),
    ]})
    comments = sb.fetch_comments("123", limit=100, session=sess, sleep=lambda s: None)
    assert len(comments) == 20


def test_pagination_first_page_failure_returns_empty():
    sess = FakeSession({"x/v2/reply": [FakeResp(500, None)]})
    assert sb.fetch_comments("123", limit=100, session=sess, sleep=lambda s: None) == []


def test_pagination_slices_to_limit():
    """翻页聚合超过 limit 时截断到 limit。"""
    sess = FakeSession({"x/v2/reply": [
        FakeResp(200, reply_page("a", 20, next_val=2)),
        FakeResp(200, reply_page("b", 20, next_val=3)),
    ]})
    comments = sb.fetch_comments("123", limit=25, session=sess, sleep=lambda s: None)
    assert len(comments) == 25
    assert comments[-1]["content"] == "评论b-4"


def test_pagination_bvid_resolves_then_paginates():
    """bvid 先经 view 转 aid，再翻页拉评论。"""
    sess = FakeSession({
        "web-interface/view": [FakeResp(200, {"code": 0, "data": {"aid": 999}})],
        "x/v2/reply": [
            FakeResp(200, reply_page("a", 20, next_val=2)),
            FakeResp(200, reply_page("b", 20, next_val=0)),
        ],
    })
    comments = sb.fetch_comments("BV1xx411c7mD", limit=40,
                                 session=sess, sleep=lambda s: None)
    assert len(comments) == 40
    assert len(sess.urls("web-interface/view")) == 1
    assert all(p["oid"] == "999" for p in sess.reply_params())
    assert all(c["post_id"] == "999" for c in comments)


def test_pagination_rate_gate_sleeps_between_pages():
    """3 次请求（首请求免费）→ sleep 恰好 2 次。"""
    sleeps = FakeSleeps()
    sess = FakeSession({"x/v2/reply": [
        FakeResp(200, reply_page("a", 20, next_val=2)),
        FakeResp(200, reply_page("b", 20, next_val=3)),
        FakeResp(200, reply_page("c", 20, next_val=0)),
    ]})
    sb.fetch_comments("123", limit=60, session=sess, sleep=sleeps)
    assert len(sleeps.values) == 2
    assert all(1.0 <= v <= 1.3 for v in sleeps.values)


def test_pagination_page_cap():
    """页数安全上限 REPLY_PAGE_MAX：页面无限续供时也不会失控。"""
    pages = [FakeResp(200, reply_page(f"p{i}", 20, next_val=i + 2))
             for i in range(sb.REPLY_PAGE_MAX + 5)]
    sess = FakeSession({"x/v2/reply": pages})
    comments = sb.fetch_comments("123", limit=10 ** 6,
                                 session=sess, sleep=lambda s: None)
    assert len(sess.reply_params()) == sb.REPLY_PAGE_MAX
    assert len(comments) == sb.REPLY_PAGE_MAX * sb.REPLY_PAGE_SIZE


def test_comment_contract_keys_in_pagination():
    sess = FakeSession({"x/v2/reply": [FakeResp(200, reply_page("a", 3))]})
    comments = sb.fetch_comments("123", session=sess, sleep=lambda s: None)
    c = comments[0]
    assert set(c) == {"platform", "post_id", "author", "content",
                      "likes", "published_at"}
    assert c["platform"] == "bilibili"
    assert isinstance(c["likes"], int)


# ══ 2. collect_keyword_samples ══

def fake_video(pid, title):
    return {"platform": "bilibili", "post_id": pid, "title": title,
            "content": "", "author": "UP主", "metrics": {}, "url": "",
            "published_at": "", "source": "bilibili_search_video"}


def fake_comment(pid, text):
    return {"platform": "bilibili", "post_id": pid, "author": "网友",
            "content": text, "likes": 1, "published_at": "2026-07-22T00:00:00+00:00"}


def make_bilibili_module(videos, comments_map, calls=None):
    """构造假 social_bilibili 模块；comments_map: post_id → list/异常。"""
    mod = types.ModuleType("social_bilibili")

    def search(keyword, limit=20, sleep=None, **kw):
        if calls is not None:
            calls.append(("search", keyword, limit))
        return videos

    def fetch_comments(post_id, limit=20, sleep=None, **kw):
        if calls is not None:
            calls.append(("fetch_comments", post_id, limit))
        result = comments_map.get(post_id, [])
        if isinstance(result, Exception):
            raise result
        return [dict(c) for c in result]

    mod.search = search
    mod.fetch_comments = fetch_comments
    return mod


@pytest.fixture
def inject_bilibili(monkeypatch):
    def _inject(mod):
        monkeypatch.setitem(sys.modules, "social_bilibili", mod)
    return _inject


def test_keyword_samples_happy_merges_with_source_video(inject_bilibili):
    videos = [fake_video("v1", "视频甲"), fake_video("v2", "视频乙")]
    comments_map = {
        "v1": [fake_comment("v1", "看多"), fake_comment("v1", "满仓")],
        "v2": [fake_comment("v2", "观望"), fake_comment("v2", "割肉"),
               fake_comment("v2", "躺平")],
    }
    inject_bilibili(make_bilibili_module(videos, comments_map))
    out = sm.collect_keyword_samples("半导体", sleep=lambda s: None)
    assert out["keyword"] == "半导体"
    assert len(out["comments"]) == 5
    assert [c["source_video"] for c in out["comments"]] == \
        ["视频甲", "视频甲", "视频乙", "视频乙", "视频乙"]
    assert out["videos_used"] == [
        {"post_id": "v1", "title": "视频甲", "comments": 2},
        {"post_id": "v2", "title": "视频乙", "comments": 3},
    ]
    assert out["notes"] == []


def test_keyword_samples_partial_failure_keeps_others(inject_bilibili):
    """视频乙评论为空 → 跳过进 notes，视频甲评论保留。"""
    videos = [fake_video("v1", "视频甲"), fake_video("v2", "视频乙")]
    inject_bilibili(make_bilibili_module(videos,
                                         {"v1": [fake_comment("v1", "看多")],
                                          "v2": []}))
    out = sm.collect_keyword_samples("半导体", sleep=lambda s: None)
    assert len(out["comments"]) == 1
    assert out["comments"][0]["source_video"] == "视频甲"
    assert len(out["videos_used"]) == 1
    assert any("视频乙" in n and "跳过" in n for n in out["notes"])


def test_keyword_samples_fetch_raises_degrades(inject_bilibili):
    """某视频 fetch_comments 抛异常 → 该视频跳过进 notes，其他保留，绝不抛。"""
    videos = [fake_video("v1", "视频甲"), fake_video("v2", "视频乙")]
    inject_bilibili(make_bilibili_module(
        videos, {"v1": [fake_comment("v1", "看多")],
                 "v2": RuntimeError("boom")}))
    out = sm.collect_keyword_samples("半导体", sleep=lambda s: None)
    assert len(out["comments"]) == 1
    assert any("视频乙" in n for n in out["notes"])


def test_keyword_samples_limits_passed_through(inject_bilibili):
    calls = []
    videos = [fake_video("v1", "视频甲")]
    inject_bilibili(make_bilibili_module(
        videos, {"v1": [fake_comment("v1", "看多")]}, calls=calls))
    sm.collect_keyword_samples("半导体", video_limit=3,
                               comments_per_video=77, sleep=lambda s: None)
    assert ("search", "半导体", 3) in calls
    assert ("fetch_comments", "v1", 77) in calls


def test_keyword_samples_search_empty_notes(inject_bilibili):
    inject_bilibili(make_bilibili_module([], {}))
    out = sm.collect_keyword_samples("冷门词", sleep=lambda s: None)
    assert out["comments"] == []
    assert out["videos_used"] == []
    assert any("无视频结果" in n for n in out["notes"])


def test_keyword_samples_empty_keyword_no_calls(inject_bilibili):
    calls = []
    inject_bilibili(make_bilibili_module([fake_video("v1", "视频甲")],
                                         {}, calls=calls))
    out = sm.collect_keyword_samples("   ", sleep=lambda s: None)
    assert out["keyword"] == ""
    assert out["comments"] == []
    assert calls == []
    assert any("关键词为空" in n for n in out["notes"])


def test_keyword_samples_search_capability_missing(monkeypatch):
    mod = types.ModuleType("social_bilibili")  # 无 search/fetch_comments
    monkeypatch.setitem(sys.modules, "social_bilibili", mod)
    out = sm.collect_keyword_samples("半导体", sleep=lambda s: None)
    assert out["comments"] == []
    assert any("search" in n for n in out["notes"])


def test_keyword_samples_fetch_comments_capability_missing(monkeypatch):
    mod = types.ModuleType("social_bilibili")
    mod.search = lambda keyword, limit=20, sleep=None: [fake_video("v1", "视频甲")]
    monkeypatch.setitem(sys.modules, "social_bilibili", mod)
    out = sm.collect_keyword_samples("半导体", sleep=lambda s: None)
    assert out["comments"] == []
    assert out["videos_used"] == []
    assert any("fetch_comments" in n for n in out["notes"])


def test_keyword_samples_search_raises_degrades(inject_bilibili):
    mod = types.ModuleType("social_bilibili")

    def search(keyword, limit=20, sleep=None):
        raise RuntimeError("search boom")

    mod.search = search
    mod.fetch_comments = lambda post_id, limit=20, sleep=None: []
    inject_bilibili(mod)
    out = sm.collect_keyword_samples("半导体", sleep=lambda s: None)
    assert out["comments"] == []
    assert any("无视频结果" in n or "降级" in n for n in out["notes"])


def test_keyword_samples_video_missing_post_id_skipped(inject_bilibili):
    bad = fake_video("", "无ID视频")
    good = fake_video("v1", "视频甲")
    inject_bilibili(make_bilibili_module(
        [bad, good], {"v1": [fake_comment("v1", "看多")]}))
    out = sm.collect_keyword_samples("半导体", sleep=lambda s: None)
    assert len(out["comments"]) == 1
    assert any("post_id" in n for n in out["notes"])


def test_keyword_samples_video_limit_slices(inject_bilibili):
    videos = [fake_video(f"v{i}", f"视频{i}") for i in range(5)]
    comments_map = {f"v{i}": [fake_comment(f"v{i}", "看多")] for i in range(5)}
    inject_bilibili(make_bilibili_module(videos, comments_map))
    out = sm.collect_keyword_samples("半导体", video_limit=2,
                                     sleep=lambda s: None)
    assert len(out["videos_used"]) == 2
    assert len(out["comments"]) == 2


def test_keyword_samples_all_videos_fail_notes(inject_bilibili):
    videos = [fake_video("v1", "视频甲"), fake_video("v2", "视频乙")]
    inject_bilibili(make_bilibili_module(videos, {}))
    out = sm.collect_keyword_samples("半导体", sleep=lambda s: None)
    assert out["comments"] == []
    assert out["videos_used"] == []
    assert any("全部视频评论采样失败" in n for n in out["notes"])


def test_keyword_samples_never_raises_on_garbage(inject_bilibili):
    inject_bilibili(make_bilibili_module([{"no_post_id": True}, "junk", None],
                                         {}))
    out = sm.collect_keyword_samples("半导体", sleep=lambda s: None)
    assert isinstance(out, dict)
    assert out["comments"] == []


# ══ 3. collect_guba_samples ══

def fake_guba_post(pid, content=""):
    return {"platform": "guba", "post_id": pid, "title": f"帖子{pid}",
            "content": content, "author": "吧友", "metrics": {},
            "url": f"https://guba/{pid}", "published_at": "",
            "source": "guba_list"}


def make_guba_module(posts, enriched=None, calls=None, with_enrich=True):
    mod = types.ModuleType("social_guba")

    def fetch_bar_posts(code, limit=30, sleep=None, **kw):
        if calls is not None:
            calls.append(("fetch_bar_posts", code, limit))
        if isinstance(posts, Exception):
            raise posts
        return [dict(p) for p in posts]

    mod.fetch_bar_posts = fetch_bar_posts
    if with_enrich:
        def enrich_posts(items, top_n=3, sleep=None, **kw):
            if calls is not None:
                calls.append(("enrich_posts", top_n))
            if isinstance(enriched, Exception):
                raise enriched
            return enriched if enriched is not None else items
        mod.enrich_posts = enrich_posts
    return mod


@pytest.fixture
def inject_guba(monkeypatch):
    def _inject(mod):
        monkeypatch.setitem(sys.modules, "social_guba", mod)
    return _inject


def test_guba_samples_passthrough(inject_guba):
    posts = [fake_guba_post("1"), fake_guba_post("2"), fake_guba_post("3")]
    calls = []
    inject_guba(make_guba_module(posts, calls=calls))
    out = sm.collect_guba_samples("600519", post_limit=100,
                                  sleep=lambda s: None)
    assert out["code"] == "600519"
    assert out["posts"] == posts
    assert out["notes"] == []
    assert ("fetch_bar_posts", "600519", 100) in calls


def test_guba_samples_enrich_zero_skips_enrich(inject_guba):
    posts = [fake_guba_post("1")]
    calls = []
    inject_guba(make_guba_module(posts, calls=calls))
    out = sm.collect_guba_samples("600519", enrich=0, sleep=lambda s: None)
    assert len(out["posts"]) == 1
    assert not any(c[0] == "enrich_posts" for c in calls)


def test_guba_samples_enrich_applies(inject_guba):
    posts = [fake_guba_post("1"), fake_guba_post("2")]
    enriched = [dict(posts[0], content="正文回填",
                     metrics={"likes": 42}), posts[1]]
    calls = []
    inject_guba(make_guba_module(posts, enriched=enriched, calls=calls))
    out = sm.collect_guba_samples("600519", enrich=2, sleep=lambda s: None)
    assert ("enrich_posts", 2) in calls
    assert out["posts"][0]["content"] == "正文回填"
    assert out["posts"][0]["metrics"]["likes"] == 42


def test_guba_samples_enrich_capability_missing(monkeypatch):
    mod = make_guba_module([fake_guba_post("1")], with_enrich=False)
    monkeypatch.setitem(sys.modules, "social_guba", mod)
    out = sm.collect_guba_samples("600519", enrich=3, sleep=lambda s: None)
    assert len(out["posts"]) == 1
    assert any("enrich_posts" in n for n in out["notes"])


def test_guba_samples_enrich_raises_keeps_list(inject_guba):
    posts = [fake_guba_post("1")]
    inject_guba(make_guba_module(posts, enriched=RuntimeError("enrich boom")))
    out = sm.collect_guba_samples("600519", enrich=3, sleep=lambda s: None)
    assert out["posts"] == posts  # 保留列表原始数据
    assert any("富化失败" in n for n in out["notes"])


def test_guba_samples_invalid_code_no_calls(inject_guba):
    calls = []
    inject_guba(make_guba_module([fake_guba_post("1")], calls=calls))
    out = sm.collect_guba_samples("不是代码", sleep=lambda s: None)
    assert out["code"] == ""
    assert out["posts"] == []
    assert calls == []
    assert any("非法" in n for n in out["notes"])


def test_guba_samples_fetch_capability_missing(monkeypatch):
    mod = types.ModuleType("social_guba")  # 无 fetch_bar_posts
    monkeypatch.setitem(sys.modules, "social_guba", mod)
    out = sm.collect_guba_samples("600519", sleep=lambda s: None)
    assert out["code"] == "600519"
    assert out["posts"] == []
    assert any("fetch_bar_posts" in n for n in out["notes"])


def test_guba_samples_empty_posts_notes(inject_guba):
    inject_guba(make_guba_module([]))
    out = sm.collect_guba_samples("600519", sleep=lambda s: None)
    assert out["posts"] == []
    assert any("未抓到帖子" in n for n in out["notes"])


def test_guba_samples_fetch_raises_degrades(inject_guba):
    inject_guba(make_guba_module(RuntimeError("fetch boom")))
    out = sm.collect_guba_samples("600519", sleep=lambda s: None)
    assert out["posts"] == []
    assert any("未抓到帖子" in n for n in out["notes"])


def test_guba_samples_post_limit_clamped(inject_guba):
    calls = []
    inject_guba(make_guba_module([fake_guba_post("1")], calls=calls))
    sm.collect_guba_samples("600519", post_limit="abc", sleep=lambda s: None)
    assert ("fetch_bar_posts", "600519", 100) in calls  # 非法值回落默认 100
