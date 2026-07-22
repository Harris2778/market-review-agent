"""agent/social_douyin.py 抖音热榜采集测试（全 mock 零网络）。

覆盖范围：
1. 正常解析：fixture 照 2026-07-22 recon 实测结构（status_code=0 /
   data.word_list[]，条目含 word/hot_value/position/event_time/sentence_id/
   group_id/label/label_url）；字段映射（title=word、metrics.heat=hot_value、
   post_id=sentence_id、url=douyin.com/hot/{sentence_id}、
   published_at=event_time→ISO8601 UTC、source=douyin_hot）、limit 截断。
2. 字段缺失防御：word 缺失跳过、sentence_id 缺失 sha1 回退 + url 空串、
   hot_value 缺失省略 heat、event_time 缺失/非法 published_at 空串、
   非 dict 条目跳过。
3. 「无签名直连红利」降级：HTTP 非 200 / status_code != 0 / 非 JSON /
   顶层非 dict / data 非 dict / word_list 非 list / 请求异常
   → warning + 空列表，绝不抛。
4. 机制：session 注入（Referer/timeout 断言）、自建 session trust_env=False +
   桌面 UA、_RateGate 限速 sleep 注入、HTML 标签清洗、无 search/comments。
"""

import hashlib
import json
import logging
from datetime import datetime, timezone

import pytest

import agent.social_douyin as dy


# ── fixture：照 recon_raw5.json douyin_hot_full 实测结构 ──

def _wl_item(word, hot_value, position, sentence_id, event_time, label=3):
    """构造一条 word_list 条目，字段集与实测一致。"""
    return {
        "article_detail_count": 0,
        "can_extend_detail": False,
        "discuss_video_count": 1,
        "display_style": 0,
        "event_time": event_time,
        "group_id": "7663473166690161961",
        "hot_value": hot_value,
        "hotlist_param": "{\"version\":1}",
        "label": label,
        "label_url": ("https://lf3-static.bytednsdoc.com/obj/eden-cn/"
                      "hotspot_detail_page/3.png"),
        "max_rank": 1,
        "position": position,
        "sentence_id": sentence_id,
        "sentence_tag": 9000,
        "video_count": 3,
        "word": word,
        "word_cover": {"uri": "tos-cn-p-0015/x", "url_list": []},
    }


DOUYIN_PAYLOAD = {
    "status_code": 0,
    "data": {
        "word_list": [
            _wl_item("广州早茶已经next level了", 11541217, 1, "2580683", 1784691491),
            _wl_item("引力一号一箭九星发射", 11190871, 2, "2580893", 1784700683),
            _wl_item("A股三大指数集体收涨", 9876543, 3, "2581465", 1784710000,
                     label=0),
        ]
    },
    "extra": {"logid": "20260722120000AAAA"},
    "log_pb": {"impr_id": "x"},
}


# ── 公共工具 ──

class FakeResp:
    """最小 response 替身：status_code + text/content + json()。"""

    def __init__(self, content=b"", status_code=200):
        if isinstance(content, str):
            content = content.encode("utf-8")
        self.content = content
        self.status_code = status_code
        self.text = content.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.text)


class FakeSession:
    """最小 session 替身：记录 get 调用，返回 handler 产物。"""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []
        self.headers = {}
        self.trust_env = True  # 仿 requests 默认值

    def get(self, url, headers=None, timeout=None, **kw):
        self.calls.append({"url": url, "headers": headers or {},
                           "timeout": timeout, "kw": kw})
        return self.handler(url)


def ok_handler(payload=None):
    body = json.dumps(payload if payload is not None else DOUYIN_PAYLOAD,
                      ensure_ascii=False)
    return lambda url: FakeResp(body, 200)


def make_sleep(record):
    def _sleep(seconds):
        record.append(seconds)
    return _sleep


# ── 1. 正常解析 ──

def test_fetch_hot_parses_word_list_fields():
    posts = dy.fetch_hot(session=FakeSession(ok_handler()), sleep=lambda s: None)
    assert len(posts) == 3
    p = posts[0]
    assert p["platform"] == "douyin"
    assert p["title"] == "广州早茶已经next level了"
    assert p["content"] == ""
    assert p["author"] == ""
    assert p["source"] == "douyin_hot"
    assert set(p.keys()) == {"platform", "post_id", "title", "content", "author",
                             "metrics", "url", "published_at", "source"}


def test_metrics_heat_from_hot_value():
    posts = dy.fetch_hot(session=FakeSession(ok_handler()), sleep=lambda s: None)
    assert posts[0]["metrics"] == {"heat": 11541217}
    assert posts[1]["metrics"] == {"heat": 11190871}


def test_post_id_is_sentence_id():
    posts = dy.fetch_hot(session=FakeSession(ok_handler()), sleep=lambda s: None)
    assert posts[0]["post_id"] == "2580683"
    assert posts[2]["post_id"] == "2581465"


def test_url_from_sentence_id():
    posts = dy.fetch_hot(session=FakeSession(ok_handler()), sleep=lambda s: None)
    assert posts[0]["url"] == "https://www.douyin.com/hot/2580683"


def test_published_at_from_event_time_iso():
    posts = dy.fetch_hot(session=FakeSession(ok_handler()), sleep=lambda s: None)
    expect = datetime.fromtimestamp(1784691491, tz=timezone.utc).isoformat()
    assert posts[0]["published_at"] == expect
    assert posts[0]["published_at"].startswith("2026-")
    assert posts[0]["published_at"].endswith("+00:00")


def test_event_time_string_digits_accepted():
    payload = {"status_code": 0, "data": {"word_list": [
        _wl_item("字符串时间", 100, 1, "999", "1784691491"),
    ]}}
    posts = dy.fetch_hot(session=FakeSession(ok_handler(payload)),
                         sleep=lambda s: None)
    assert posts[0]["published_at"] == datetime.fromtimestamp(
        1784691491, tz=timezone.utc).isoformat()


def test_limit_truncates_in_order():
    posts = dy.fetch_hot(limit=2, session=FakeSession(ok_handler()),
                         sleep=lambda s: None)
    assert [p["title"] for p in posts] == ["广州早茶已经next level了",
                                           "引力一号一箭九星发射"]


def test_limit_default_20():
    big = {"status_code": 0, "data": {"word_list": [
        _wl_item(f"热点{i}", 1000 - i, i + 1, str(1000 + i), 1784691491 + i)
        for i in range(30)]}}
    posts = dy.fetch_hot(session=FakeSession(ok_handler(big)), sleep=lambda s: None)
    assert len(posts) == 20


# ── 2. 字段缺失防御 ──

def test_missing_word_skipped():
    payload = {"status_code": 0, "data": {"word_list": [
        {"hot_value": 100, "sentence_id": "1", "event_time": 1784691491},
        {"word": "   ", "hot_value": 99, "sentence_id": "2"},
        _wl_item("幸存热点", 88, 3, "2580999", 1784691491),
    ]}}
    posts = dy.fetch_hot(session=FakeSession(ok_handler(payload)),
                         sleep=lambda s: None)
    assert len(posts) == 1
    assert posts[0]["title"] == "幸存热点"


def test_missing_sentence_id_sha1_fallback_and_empty_url():
    payload = {"status_code": 0, "data": {"word_list": [
        {"word": "无ID热点", "hot_value": 100, "event_time": 1784691491},
    ]}}
    posts = dy.fetch_hot(session=FakeSession(ok_handler(payload)),
                         sleep=lambda s: None)
    expect = hashlib.sha1("无ID热点".encode("utf-8")).hexdigest()[:12]
    assert posts[0]["post_id"] == expect
    assert posts[0]["url"] == ""


def test_missing_hot_value_omits_heat():
    payload = {"status_code": 0, "data": {"word_list": [
        {"word": "无热度", "sentence_id": "1", "event_time": 1784691491},
        {"word": "热度字符串", "hot_value": "777", "sentence_id": "2",
         "event_time": 1784691491},
        {"word": "热度乱码", "hot_value": "xyz", "sentence_id": "3",
         "event_time": 1784691491},
    ]}}
    posts = dy.fetch_hot(session=FakeSession(ok_handler(payload)),
                         sleep=lambda s: None)
    assert posts[0]["metrics"] == {}
    assert posts[1]["metrics"] == {"heat": 777}
    assert posts[2]["metrics"] == {}


def test_missing_or_bad_event_time_empty_published_at():
    payload = {"status_code": 0, "data": {"word_list": [
        {"word": "无时间", "sentence_id": "1", "hot_value": 1},
        {"word": "时间乱码", "sentence_id": "2", "hot_value": 1,
         "event_time": "不是数字"},
        {"word": "时间为负", "sentence_id": "3", "hot_value": 1,
         "event_time": -5},
    ]}}
    posts = dy.fetch_hot(session=FakeSession(ok_handler(payload)),
                         sleep=lambda s: None)
    assert [p["published_at"] for p in posts] == ["", "", ""]


def test_non_dict_row_skipped():
    payload = {"status_code": 0, "data": {"word_list": [
        "垃圾行", None, 42, _wl_item("幸存热点", 66, 1, "777", 1784691491),
    ]}}
    posts = dy.fetch_hot(session=FakeSession(ok_handler(payload)),
                         sleep=lambda s: None)
    assert len(posts) == 1
    assert posts[0]["title"] == "幸存热点"


def test_html_tags_cleaned_in_word():
    payload = {"status_code": 0, "data": {"word_list": [
        {"word": "<b>带标签</b>热点", "sentence_id": "1", "hot_value": 10},
    ]}}
    posts = dy.fetch_hot(session=FakeSession(ok_handler(payload)),
                         sleep=lambda s: None)
    assert posts[0]["title"] == "带标签热点"


# ── 3. 「无签名直连红利」降级路径 ──

def test_http_error_returns_empty_and_warns(caplog):
    sess = FakeSession(lambda url: FakeResp("<html>blocked</html>", 403))
    with caplog.at_level(logging.WARNING, logger="agent.social_douyin"):
        posts = dy.fetch_hot(session=sess, sleep=lambda s: None)
    assert posts == []
    assert any("403" in r.message for r in caplog.records)


def test_status_code_nonzero_returns_empty_and_warns(caplog):
    """抖音业务层拒绝（风控信号）→ 降级空列表。"""
    payload = {"status_code": 2154, "status_msg": "blocked", "data": {}}
    sess = FakeSession(ok_handler(payload))
    with caplog.at_level(logging.WARNING, logger="agent.social_douyin"):
        posts = dy.fetch_hot(session=sess, sleep=lambda s: None)
    assert posts == []
    assert any("status_code" in r.message for r in caplog.records)


def test_non_json_returns_empty_and_warns(caplog):
    sess = FakeSession(lambda url: FakeResp("<html>verify</html>", 200))
    with caplog.at_level(logging.WARNING, logger="agent.social_douyin"):
        posts = dy.fetch_hot(session=sess, sleep=lambda s: None)
    assert posts == []
    assert any("JSON" in r.message for r in caplog.records)


def test_top_level_list_returns_empty_and_warns(caplog):
    sess = FakeSession(lambda url: FakeResp("[1,2]", 200))
    with caplog.at_level(logging.WARNING, logger="agent.social_douyin"):
        posts = dy.fetch_hot(session=sess, sleep=lambda s: None)
    assert posts == []
    assert caplog.records


def test_data_not_dict_returns_empty_and_warns(caplog):
    sess = FakeSession(ok_handler({"status_code": 0, "data": []}))
    with caplog.at_level(logging.WARNING, logger="agent.social_douyin"):
        posts = dy.fetch_hot(session=sess, sleep=lambda s: None)
    assert posts == []
    assert any("data" in r.message for r in caplog.records)


def test_word_list_not_list_returns_empty_and_warns(caplog):
    sess = FakeSession(ok_handler(
        {"status_code": 0, "data": {"word_list": {"x": 1}}}))
    with caplog.at_level(logging.WARNING, logger="agent.social_douyin"):
        posts = dy.fetch_hot(session=sess, sleep=lambda s: None)
    assert posts == []
    assert any("word_list" in r.message for r in caplog.records)


def test_request_exception_returns_empty_and_warns(caplog):
    def boom(url):
        raise TimeoutError("mock 超时")
    with caplog.at_level(logging.WARNING, logger="agent.social_douyin"):
        posts = dy.fetch_hot(session=FakeSession(boom), sleep=lambda s: None)
    assert posts == []
    assert any("mock 超时" in r.message for r in caplog.records)


def test_never_raises_on_garbage():
    garbage = [b"", b"null", b"{}", b'{"status_code":0}',
               b'{"status_code":0,"data":{"word_list":{}}}']
    for body in garbage:
        sess = FakeSession(lambda url, b=body: FakeResp(b, 200))
        assert dy.fetch_hot(session=sess, sleep=lambda s: None) == []


# ── 4. 机制：session 注入 / UA / trust_env / 限速 ──

def test_injected_session_referer_and_timeout():
    sess = FakeSession(ok_handler())
    dy.fetch_hot(session=sess, sleep=lambda s: None)
    assert len(sess.calls) == 1
    call = sess.calls[0]
    assert call["url"] == dy.HOT_LIST_URL
    assert call["headers"]["Referer"] == "https://www.douyin.com/"
    assert call["timeout"] == 10


def test_default_session_trust_env_false_and_ua(monkeypatch):
    made = []

    def factory():
        sess = FakeSession(ok_handler())
        made.append(sess)
        return sess

    monkeypatch.setattr(dy.requests, "Session", factory)
    posts = dy.fetch_hot(sleep=lambda s: None)
    assert len(posts) == 3
    sess = made[0]
    assert sess.trust_env is False
    assert sess.headers["User-Agent"] == dy.DEFAULT_UA
    assert "Mozilla" in sess.headers["User-Agent"]


def test_rate_gate_first_wait_free_second_sleeps():
    record = []
    gate = dy._RateGate(make_sleep(record))
    gate.wait()
    assert record == []
    gate.wait()
    assert len(record) == 1
    assert 1.0 <= record[0] <= 1.5


def test_fetch_hot_single_request_no_sleep():
    """热榜为单请求端点：一轮内不应触发限速休眠。"""
    record = []
    dy.fetch_hot(session=FakeSession(ok_handler()), sleep=make_sleep(record))
    assert record == []


# ── 5. 能力边界：v1 仅热榜 ──

def test_no_search_no_fetch_comments():
    assert not hasattr(dy, "search")
    assert not hasattr(dy, "fetch_comments")
    assert callable(dy.fetch_hot)
