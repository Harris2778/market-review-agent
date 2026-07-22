"""agent/social_weibo.py 微博热搜采集测试（全 mock 零网络）。

覆盖范围：
1. 正常解析：fixture 照 2026-07-22 recon 实测结构（ok/data.hotgov/data.realtime[]，
   条目含 realpos/flag/label_name/word/word_scheme/num/note/rank）；
   字段映射（title=word、metrics.heat=num、post_id=sha1(word)[:12]、
   url=s.weibo.com 拼接 quote(word)、published_at/content/author 空串、
   source=weibo_hot）、limit 截断。
2. 字段缺失防御：word 缺失回退 note、word+note 均缺跳过、num 缺失省略 heat、
   非 dict 条目跳过。
3. 降级：HTTP 非 200 / 非 JSON / 顶层非 dict / data 非 dict / realtime 非 list /
   请求异常 → warning + 空列表，绝不抛。
4. 机制：session 注入（Referer/timeout 断言）、自建 session trust_env=False +
   浏览器 UA、_RateGate 限速 sleep 注入、HTML 标签清洗、无 search/comments。
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from urllib.parse import quote

import pytest

import agent.social_weibo as wb


# ── fixture：照 recon_raw3.json weibo_hot_desktop_full 实测结构 ──

def _rt_item(word, num, realpos, label_name="", note=None):
    """构造一条 realtime 条目，字段集与实测一致。"""
    item = {
        "realpos": realpos,
        "flag": 2 if label_name else 0,
        "label_name": label_name,
        "word": word,
        "icon_desc": label_name,
        "word_scheme": word,
        "num": num,
        "note": word if note is None else note,
        "emoticon": "",
        "rank": realpos - 1,
    }
    if label_name:
        item["icon_desc_color"] = "#ff9406"
        item["icon"] = "https://simg.s.weibo.com/moter/flags/2_0.png"
        item["icon_width"] = 24
        item["icon_height"] = 24
        item["small_icon_desc"] = label_name
        item["small_icon_desc_color"] = "#ff9406"
    return item


WEIBO_PAYLOAD = {
    "ok": 1,
    "data": {
        "hotgov": {
            "word": "全国两会召开",
            "note": "全国两会召开",
            "mid": "5123456789",
            "url": "https://s.weibo.com/weibo?q=...",
            "stime": 1784690000,
            "is_gov": 1,
            "icon_desc": "置顶",
        },
        "realtime": [
            _rt_item("别再给AI乱传文件了", 2248139, 1, "热"),
            _rt_item("两个AI演员比内娱待爆艺人都火", 957120, 2),
            _rt_item("以旧换新带动消费1.1万亿元", 885718, 3, "新"),
        ],
    },
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
    body = json.dumps(payload if payload is not None else WEIBO_PAYLOAD,
                      ensure_ascii=False)
    return lambda url: FakeResp(body, 200)


def make_sleep(record):
    def _sleep(seconds):
        record.append(seconds)
    return _sleep


# ── 1. 正常解析 ──

def test_fetch_hot_parses_realtime_fields():
    sess = FakeSession(ok_handler())
    posts = wb.fetch_hot(session=sess, sleep=lambda s: None)
    assert len(posts) == 3
    p = posts[0]
    assert p["platform"] == "weibo"
    assert p["title"] == "别再给AI乱传文件了"
    assert p["content"] == ""
    assert p["author"] == ""
    assert p["published_at"] == ""
    assert p["source"] == "weibo_hot"
    assert set(p.keys()) == {"platform", "post_id", "title", "content", "author",
                             "metrics", "url", "published_at", "source"}


def test_metrics_heat_from_num():
    posts = wb.fetch_hot(session=FakeSession(ok_handler()), sleep=lambda s: None)
    assert posts[0]["metrics"] == {"heat": 2248139}
    assert posts[2]["metrics"] == {"heat": 885718}


def test_post_id_is_sha1_of_word():
    posts = wb.fetch_hot(session=FakeSession(ok_handler()), sleep=lambda s: None)
    expect = hashlib.sha1("别再给AI乱传文件了".encode("utf-8")).hexdigest()[:12]
    assert posts[0]["post_id"] == expect
    assert len(posts[0]["post_id"]) == 12


def test_url_is_quoted_search_url():
    posts = wb.fetch_hot(session=FakeSession(ok_handler()), sleep=lambda s: None)
    assert posts[0]["url"] == (
        "https://s.weibo.com/weibo?q=" + quote("别再给AI乱传文件了"))
    assert posts[0]["url"].startswith("https://s.weibo.com/weibo?q=")


def test_url_quotes_special_chars():
    payload = {"ok": 1, "data": {"realtime": [
        _rt_item("A股 茅台&五粮液", 12345, 1),
    ]}}
    posts = wb.fetch_hot(session=FakeSession(ok_handler(payload)),
                         sleep=lambda s: None)
    assert posts[0]["url"] == (
        "https://s.weibo.com/weibo?q=" + quote("A股 茅台&五粮液"))
    assert "&" not in posts[0]["url"].split("q=", 1)[1]


def test_label_name_not_required():
    """label_name 为空串（实测常见）不影响解析。"""
    posts = wb.fetch_hot(session=FakeSession(ok_handler()), sleep=lambda s: None)
    assert posts[1]["title"] == "两个AI演员比内娱待爆艺人都火"


def test_limit_truncates_in_order():
    sess = FakeSession(ok_handler())
    posts = wb.fetch_hot(limit=2, session=sess, sleep=lambda s: None)
    assert [p["title"] for p in posts] == ["别再给AI乱传文件了",
                                           "两个AI演员比内娱待爆艺人都火"]


def test_limit_default_20_and_zero():
    big = {"ok": 1, "data": {"realtime": [
        _rt_item(f"话题{i}", 1000 - i, i + 1) for i in range(30)]}}
    posts = wb.fetch_hot(session=FakeSession(ok_handler(big)), sleep=lambda s: None)
    assert len(posts) == 20
    posts0 = wb.fetch_hot(limit=0, session=FakeSession(ok_handler(big)),
                          sleep=lambda s: None)
    assert posts0 == []


# ── 2. 字段缺失防御 ──

def test_note_fallback_when_word_missing():
    payload = {"ok": 1, "data": {"realtime": [
        {"note": "只有note的话题", "num": 100, "realpos": 1, "rank": 0},
    ]}}
    posts = wb.fetch_hot(session=FakeSession(ok_handler(payload)),
                         sleep=lambda s: None)
    assert posts[0]["title"] == "只有note的话题"
    assert posts[0]["metrics"] == {"heat": 100}


def test_missing_word_and_note_skipped():
    payload = {"ok": 1, "data": {"realtime": [
        {"num": 100, "realpos": 1},                      # 无 word/note → 跳过
        {"word": "  ", "note": "", "num": 99},           # 全空白 → 跳过
        _rt_item("正常话题", 88, 2),
    ]}}
    posts = wb.fetch_hot(session=FakeSession(ok_handler(payload)),
                         sleep=lambda s: None)
    assert len(posts) == 1
    assert posts[0]["title"] == "正常话题"


def test_missing_num_omits_heat():
    payload = {"ok": 1, "data": {"realtime": [
        {"word": "无热度话题", "note": "无热度话题", "realpos": 1},
        {"word": "热度是字符串", "num": "123456", "realpos": 2},
        {"word": "热度是乱码", "num": "abc", "realpos": 3},
    ]}}
    posts = wb.fetch_hot(session=FakeSession(ok_handler(payload)),
                         sleep=lambda s: None)
    assert posts[0]["metrics"] == {}
    assert posts[1]["metrics"] == {"heat": 123456}
    assert posts[2]["metrics"] == {}


def test_non_dict_row_skipped():
    payload = {"ok": 1, "data": {"realtime": [
        "垃圾行", None, 42, _rt_item("幸存话题", 66, 1),
    ]}}
    posts = wb.fetch_hot(session=FakeSession(ok_handler(payload)),
                         sleep=lambda s: None)
    assert len(posts) == 1
    assert posts[0]["title"] == "幸存话题"


def test_html_tags_cleaned_in_word():
    payload = {"ok": 1, "data": {"realtime": [
        {"word": "<em>带标签</em>话题", "note": "<em>带标签</em>话题",
         "num": 10, "realpos": 1},
    ]}}
    posts = wb.fetch_hot(session=FakeSession(ok_handler(payload)),
                         sleep=lambda s: None)
    assert posts[0]["title"] == "带标签话题"
    assert "<" not in posts[0]["title"]


# ── 3. HTTP 错误 / 非 JSON / 结构漂移降级 ──

def test_http_error_returns_empty_and_warns(caplog):
    sess = FakeSession(lambda url: FakeResp("<html>Visitor</html>", 432))
    with caplog.at_level(logging.WARNING, logger="agent.social_weibo"):
        posts = wb.fetch_hot(session=sess, sleep=lambda s: None)
    assert posts == []
    assert any("432" in r.message for r in caplog.records)


def test_non_json_returns_empty_and_warns(caplog):
    sess = FakeSession(lambda url: FakeResp("<html>not json</html>", 200))
    with caplog.at_level(logging.WARNING, logger="agent.social_weibo"):
        posts = wb.fetch_hot(session=sess, sleep=lambda s: None)
    assert posts == []
    assert any("JSON" in r.message for r in caplog.records)


def test_top_level_list_returns_empty_and_warns(caplog):
    sess = FakeSession(lambda url: FakeResp("[1,2,3]", 200))
    with caplog.at_level(logging.WARNING, logger="agent.social_weibo"):
        posts = wb.fetch_hot(session=sess, sleep=lambda s: None)
    assert posts == []
    assert caplog.records


def test_data_not_dict_returns_empty_and_warns(caplog):
    sess = FakeSession(ok_handler({"ok": 1, "data": None}))
    with caplog.at_level(logging.WARNING, logger="agent.social_weibo"):
        posts = wb.fetch_hot(session=sess, sleep=lambda s: None)
    assert posts == []
    assert any("data" in r.message for r in caplog.records)


def test_realtime_not_list_returns_empty_and_warns(caplog):
    sess = FakeSession(ok_handler({"ok": 1, "data": {"realtime": {"x": 1}}}))
    with caplog.at_level(logging.WARNING, logger="agent.social_weibo"):
        posts = wb.fetch_hot(session=sess, sleep=lambda s: None)
    assert posts == []
    assert any("realtime" in r.message for r in caplog.records)


def test_request_exception_returns_empty_and_warns(caplog):
    def boom(url):
        raise ConnectionError("mock 断网")
    with caplog.at_level(logging.WARNING, logger="agent.social_weibo"):
        posts = wb.fetch_hot(session=FakeSession(boom), sleep=lambda s: None)
    assert posts == []
    assert any("mock 断网" in r.message for r in caplog.records)


def test_never_raises_on_garbage():
    garbage = [b"", b"null", b"{}", b'{"data":[]}', b'{"data":{"realtime":{}}}']
    for body in garbage:
        sess = FakeSession(lambda url, b=body: FakeResp(b, 200))
        assert wb.fetch_hot(session=sess, sleep=lambda s: None) == []


# ── 4. 机制：session 注入 / UA / trust_env / 限速 ──

def test_injected_session_referer_and_timeout():
    sess = FakeSession(ok_handler())
    wb.fetch_hot(session=sess, sleep=lambda s: None)
    assert len(sess.calls) == 1
    call = sess.calls[0]
    assert call["url"] == wb.HOT_SEARCH_URL
    assert call["headers"]["Referer"] == "https://weibo.com/"
    assert call["timeout"] == 10


def test_default_session_trust_env_false_and_ua(monkeypatch):
    made = []

    def factory():
        sess = FakeSession(ok_handler())
        made.append(sess)
        return sess

    monkeypatch.setattr(wb.requests, "Session", factory)
    posts = wb.fetch_hot(sleep=lambda s: None)
    assert len(posts) == 3
    sess = made[0]
    assert sess.trust_env is False
    assert sess.headers["User-Agent"] == wb.DEFAULT_UA
    assert "Mozilla" in sess.headers["User-Agent"]


def test_rate_gate_first_wait_free_second_sleeps():
    record = []
    gate = wb._RateGate(make_sleep(record))
    gate.wait()
    assert record == []
    gate.wait()
    assert len(record) == 1
    assert 1.0 <= record[0] <= 1.5


def test_fetch_hot_single_request_no_sleep():
    """热榜为单请求端点：一轮内不应触发限速休眠。"""
    record = []
    wb.fetch_hot(session=FakeSession(ok_handler()), sleep=make_sleep(record))
    assert record == []


# ── 5. 能力边界：v1 仅热榜 ──

def test_no_search_no_fetch_comments():
    assert not hasattr(wb, "search")
    assert not hasattr(wb, "fetch_comments")
    assert callable(wb.fetch_hot)
