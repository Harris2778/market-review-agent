"""agent/social_zhihu.py 知乎热榜采集层测试（全 mock 零网络）。

覆盖范围：
1. 正常解析：2026-07-22 实测结构 fixture（data[].target + 条目级 detail_text），
   字段映射 title/content(=excerpt)/author/url 拼接/published_at(created→ISO)/
   metrics.comments(=answer_count)/metrics.heat(detail_text 万单位解析)。
2. 字段缺失防御：缺 excerpt/author/created/detail_text → 空串或省略键；
   缺 id/title → 逐条跳过；data 非 list → 空列表。
3. HTTP 错误降级：5xx/网络异常/非法 JSON/顶层非 dict → 空列表不抛。
4. limit 截断与非法 limit 入参。
5. session 注入：请求 URL/params(limit)/headers(Referer) 断言。
6. 限速注入：fake sleep 与 _RateGate 行为。
7. 默认 session 工厂：trust_env=False + UA（monkeypatch requests.Session）。
8. 契约断言：不提供 search / fetch_comments（知乎搜索 v1 缺席）。
9. HTML 标签清洗（excerpt/title 含 <em>）。

所有 HTTP 由 fake session 注入，sleep 由 fake 注入，绝不触达真实网络。
"""

import json
from datetime import datetime, timezone

import pytest

import agent.social_zhihu as zh


# ── 公共工具 ──

class FakeResp:
    """最小 response 替身：status_code + JSON body。"""

    def __init__(self, payload=None, status_code=200, raw_text=None):
        self.status_code = status_code
        self._payload = payload
        if raw_text is not None:
            self.text = raw_text
        else:
            self.text = json.dumps(payload, ensure_ascii=False) if payload is not None else ""

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """最小 session 替身：记录 get 调用，按 handler 返回 FakeResp 或抛异常。"""

    def __init__(self, handler):
        self.calls = []
        self._handler = handler

    def get(self, url, **kw):
        self.calls.append((url, kw))
        return self._handler(url, kw)


def make_sleep(record):
    """构造 fake sleep：记录每次休眠秒数。"""
    def _sleep(seconds):
        record.append(seconds)
    return _sleep


# ── fixture：2026-07-22 实测 hot-list 结构（recon_raw.json zhihu_hot_api 原型）──

def hot_entry(qid, title, *, excerpt="摘要文本", created=1784520137,
              answer_count=307, follower_count=596, author_name="用户",
              detail_text="498 万热度"):
    """实测形态条目：热度文本在条目级 detail_text，指标在 target 内。"""
    entry = {
        "type": "hot_list_feed",
        "card_id": f"Q_{qid}",
        "target": {
            "id": qid,
            "title": title,
            "url": f"https://api.zhihu.com/questions/{qid}",
            "type": "question",
            "created": created,
            "answer_count": answer_count,
            "follower_count": follower_count,
            "comment_count": 0,
            "excerpt": excerpt,
            "author": {"type": "people", "id": "0", "name": author_name},
        },
    }
    if detail_text is not None:
        entry["detail_text"] = detail_text
    return entry


def hot_payload(entries):
    return {"data": entries}


# ═══════════════════════════════════════════
# 1. 正常解析
# ═══════════════════════════════════════════

def test_fetch_hot_parses_real_structure():
    """实测结构正常解析：统一 Post 字典全字段映射正确。"""
    entries = [
        hot_entry(2062507625783226699, "14 岁少年纹身，家长索赔 20 万，如何看待？"),
        hot_entry(2063187091773773522, "滔搏终止耐克线上销售，影响有多大？",
                  detail_text="458 万热度"),
    ]
    sess = FakeSession(lambda url, kw: FakeResp(hot_payload(entries)))
    posts = zh.fetch_hot(limit=20, session=sess, sleep=make_sleep([]))

    assert len(posts) == 2
    p = posts[0]
    assert p["platform"] == "zhihu"
    assert p["post_id"] == "2062507625783226699"          # int → str
    assert p["title"] == "14 岁少年纹身，家长索赔 20 万，如何看待？"
    assert p["content"] == "摘要文本"
    assert p["author"] == "用户"
    assert p["url"] == "https://www.zhihu.com/question/2062507625783226699"
    assert p["source"] == "zhihu_hot"
    assert p["metrics"]["comments"] == 307                 # answer_count → comments
    assert p["metrics"]["heat"] == 4980000                 # "498 万热度" → 绝对值
    expected_iso = datetime.fromtimestamp(1784520137, tz=timezone.utc).isoformat(timespec="seconds")
    assert p["published_at"] == expected_iso
    assert posts[1]["metrics"]["heat"] == 4580000


def test_fetch_hot_heat_missing_detail_text():
    """条目无 detail_text：metrics 省略 heat 键，其余字段正常。"""
    entries = [hot_entry(123, "无热度条目", detail_text=None)]
    sess = FakeSession(lambda url, kw: FakeResp(hot_payload(entries)))
    posts = zh.fetch_hot(session=sess, sleep=make_sleep([]))
    assert len(posts) == 1
    assert "heat" not in posts[0]["metrics"]
    assert posts[0]["metrics"]["comments"] == 307


def test_fetch_hot_heat_unparseable_text():
    """detail_text 无数字（如 '热'）：省略 heat 键不抛。"""
    entries = [hot_entry(123, "热度文本无数字", detail_text="热度飙升")]
    sess = FakeSession(lambda url, kw: FakeResp(hot_payload(entries)))
    posts = zh.fetch_hot(session=sess, sleep=make_sleep([]))
    assert "heat" not in posts[0]["metrics"]


def test_fetch_hot_heat_fallback_metrics_text():
    """detail_text 缺席时兼容 metrics_text 字段取热度。"""
    entry = hot_entry(123, "metrics_text 兜底", detail_text=None)
    entry["metrics_text"] = "1.5 亿热度"
    sess = FakeSession(lambda url, kw: FakeResp(hot_payload([entry])))
    posts = zh.fetch_hot(session=sess, sleep=make_sleep([]))
    assert posts[0]["metrics"]["heat"] == 150000000


# ═══════════════════════════════════════════
# 2. 字段缺失防御
# ═══════════════════════════════════════════

def test_fetch_hot_missing_optional_fields():
    """缺 excerpt/author/created/answer_count：空串或省略键，不抛。"""
    entry = {"type": "hot_list_feed",
             "target": {"id": 999, "title": "只剩标题和 id"}}
    sess = FakeSession(lambda url, kw: FakeResp(hot_payload([entry])))
    posts = zh.fetch_hot(session=sess, sleep=make_sleep([]))
    assert len(posts) == 1
    p = posts[0]
    assert p["content"] == ""
    assert p["author"] == ""
    assert p["published_at"] == ""
    assert p["metrics"] == {}


def test_fetch_hot_skips_entries_without_id_or_title():
    """缺 id 或缺 title 的条目逐条跳过，其余保留。"""
    entries = [
        {"target": {"title": "无 id 条目", "excerpt": "x"}},
        hot_entry(0, ""),  # 空标题
        hot_entry(456, "正常条目"),
        "not-a-dict",
        {"target": "not-a-dict"},
    ]
    sess = FakeSession(lambda url, kw: FakeResp(hot_payload(entries)))
    posts = zh.fetch_hot(session=sess, sleep=make_sleep([]))
    assert [p["post_id"] for p in posts] == ["456"]


def test_fetch_hot_data_not_list():
    """顶层 data 非 list（dict/None/字符串）→ 空列表 + warning。"""
    for bad in [{"data": {"x": 1}}, {"data": None}, {"error": "boom"}, {}]:
        sess = FakeSession(lambda url, kw, b=bad: FakeResp(b))
        assert zh.fetch_hot(session=sess, sleep=make_sleep([])) == []


def test_fetch_hot_created_invalid_yields_empty_published():
    """created 非法（负数/字符串垃圾）→ published_at 空串。"""
    entries = [hot_entry(123, "负时间", created=-5),
               hot_entry(456, "垃圾时间", created="abc")]
    sess = FakeSession(lambda url, kw: FakeResp(hot_payload(entries)))
    posts = zh.fetch_hot(session=sess, sleep=make_sleep([]))
    assert [p["published_at"] for p in posts] == ["", ""]


# ═══════════════════════════════════════════
# 3. HTTP 错误降级
# ═══════════════════════════════════════════

def test_fetch_hot_http_error_returns_empty():
    """HTTP 4xx/5xx → 空列表不抛。"""
    for sc in (403, 429, 500, 503):
        sess = FakeSession(lambda url, kw, s=sc: FakeResp(None, status_code=s))
        assert zh.fetch_hot(session=sess, sleep=make_sleep([])) == []


def test_fetch_hot_network_exception_returns_empty():
    """session.get 抛异常（连接错误/超时）→ 空列表不抛。"""
    def boom(url, kw):
        raise ConnectionError("network down")
    sess = FakeSession(boom)
    assert zh.fetch_hot(session=sess, sleep=make_sleep([])) == []


def test_fetch_hot_invalid_json_returns_empty():
    """响应非 JSON / 顶层非 dict → 空列表不抛。"""
    sess = FakeSession(lambda url, kw: FakeResp(None, raw_text="<html>403</html>"))
    assert zh.fetch_hot(session=sess, sleep=make_sleep([])) == []
    sess2 = FakeSession(lambda url, kw: FakeResp(None, raw_text="[1,2,3]"))
    assert zh.fetch_hot(session=sess2, sleep=make_sleep([])) == []


# ═══════════════════════════════════════════
# 4. limit 行为 + 请求参数断言
# ═══════════════════════════════════════════

def test_fetch_hot_limit_truncates_and_passes_to_params():
    """limit 既作端点参数又作本地截断；请求带 Referer 头。"""
    entries = [hot_entry(i, f"条目{i}") for i in range(1, 6)]
    sess = FakeSession(lambda url, kw: FakeResp(hot_payload(entries)))
    posts = zh.fetch_hot(limit=2, session=sess, sleep=make_sleep([]))
    assert len(posts) == 2

    url, kw = sess.calls[0]
    assert url == zh.ZHIHU_HOT_LIST_URL
    assert kw["params"]["limit"] == 2
    assert kw["headers"]["Referer"] == "https://www.zhihu.com/billboard"
    assert kw["timeout"] == zh.DEFAULT_TIMEOUT


def test_fetch_hot_invalid_limit_falls_back_to_default():
    """非法 limit（None/0/负数/垃圾）→ 默认 20，不抛。"""
    entries = [hot_entry(i, f"条目{i}") for i in range(1, 4)]
    for bad_limit in (None, 0, -3, "abc"):
        sess = FakeSession(lambda url, kw: FakeResp(hot_payload(entries)))
        posts = zh.fetch_hot(limit=bad_limit, session=sess, sleep=make_sleep([]))
        assert len(posts) == 3
        assert sess.calls[0][1]["params"]["limit"] == 20


# ═══════════════════════════════════════════
# 5. 限速注入与 RateGate
# ═══════════════════════════════════════════

def test_fetch_hot_single_request_no_sleep():
    """单请求场景：RateGate 首请求不限速，fake sleep 不被调用。"""
    record = []
    sess = FakeSession(lambda url, kw: FakeResp(hot_payload([hot_entry(1, "x")])))
    zh.fetch_hot(session=sess, sleep=make_sleep(record))
    assert record == []


def test_rate_gate_sleeps_from_second_request():
    """_RateGate：第 2/3 次 wait 才休眠，间隔 ∈ [RATE, RATE+JITTER]。"""
    record = []
    gate = zh._RateGate(make_sleep(record))
    gate.wait()
    gate.wait()
    gate.wait()
    assert len(record) == 2
    for s in record:
        assert zh.DEFAULT_RATE <= s <= zh.DEFAULT_RATE + zh.DEFAULT_JITTER


# ═══════════════════════════════════════════
# 6. 默认 session 工厂
# ═══════════════════════════════════════════

def test_default_session_trust_env_false_and_ua(monkeypatch):
    """未注入 session 时自建：trust_env=False（绕开系统代理）+ 浏览器 UA。"""
    created = {}

    class FakeRealSession:
        def __init__(self):
            self.headers = {}
            self.trust_env = True
            created["s"] = self

    monkeypatch.setattr(zh.requests, "Session", FakeRealSession)
    sess = zh._default_session()
    assert sess.trust_env is False
    assert "Chrome" in sess.headers["User-Agent"]


def test_fetch_hot_without_session_uses_default(monkeypatch):
    """fetch_hot 不传 session：走 _default_session 工厂（可 monkeypatch）。"""
    fake = FakeSession(lambda url, kw: FakeResp(hot_payload([hot_entry(7, "默认会话")])))
    monkeypatch.setattr(zh, "_default_session", lambda: fake)
    posts = zh.fetch_hot(sleep=make_sleep([]))
    assert len(posts) == 1 and posts[0]["post_id"] == "7"
    assert len(fake.calls) == 1


# ═══════════════════════════════════════════
# 7. 契约与清洗
# ═══════════════════════════════════════════

def test_no_search_or_comments_functions():
    """知乎搜索/评论 v1 缺席（billboard 403 / search_v3 400 已判死刑）：
    模块刻意不提供 search / fetch_comments。"""
    assert not hasattr(zh, "search")
    assert not hasattr(zh, "fetch_comments")


def test_fetch_hot_strips_html_tags():
    """title/excerpt 含 HTML 标签（如 <em> 高亮）时清洗。"""
    entries = [hot_entry(123, "<em>茅台</em> 股价创新高",
                         excerpt="<b>酱香</b>科技 <em>大涨</em>")]
    sess = FakeSession(lambda url, kw: FakeResp(hot_payload(entries)))
    posts = zh.fetch_hot(session=sess, sleep=make_sleep([]))
    assert posts[0]["title"] == "茅台 股价创新高"
    assert posts[0]["content"] == "酱香科技 大涨"
