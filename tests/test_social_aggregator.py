"""agent/social_aggregator.py newsnow 聚合器兜底层测试（全 mock 零网络）。

覆盖范围：
1. 正常解析：2026-07-22 实测结构 fixture（status/updatedTime(ms)/items[]，
   知乎 extra.info+extra.hover、微博/B站 extra.icon、抖音无 extra），字段映射
   platform 原样透传/post_id/content(=hover)/metrics.heat(info 万单位解析，
   提不出省略该键)/published_at(updatedTime 毫秒→ISO)/source='newsnow_{id}'。
2. source_id 映射：weibo/zhihu/douyin/bilibili→bilibili-hot-search 四源请求
   参数断言；小红书（实测 500 无此源）与任意非法 platform → 空列表 + 不发请求。
3. 字段缺失防御：无 extra/缺 hover/缺 updatedTime/缺 id/title 逐条跳过；
   items 非 list → 空列表。
4. HTTP 错误降级：500（newsnow 非法源同款）/网络异常/非法 JSON → 空列表不抛。
5. limit 截断与非法 limit 入参。
6. 限速注入：fake sleep 与 _RateGate 行为。
7. 默认 session 工厂：trust_env=False + UA。
8. 契约断言：不提供 search / fetch_comments；HTML 标签清洗。

所有 HTTP 由 fake session 注入，sleep 由 fake 注入，绝不触达真实网络。
"""

import json
from datetime import datetime, timezone

import pytest

import agent.social_aggregator as agg


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


# ── fixture：2026-07-22 实测 newsnow 结构（recon_raw.json 四源原型）──

def newsnow_item(iid, title, url, *, info=None, hover=None):
    """实测条目：extra 可选；知乎带 info（热度文本）+ hover（摘要）。"""
    item = {"id": iid, "title": title, "url": url}
    extra = {}
    if info is not None:
        extra["info"] = info
    if hover is not None:
        extra["hover"] = hover
    if extra:
        item["extra"] = extra
    return item


def newsnow_payload(items, source_id="zhihu", updated_time=1784725742588, status="cache"):
    payload = {"status": status, "id": source_id, "items": items}
    if updated_time is not None:
        payload["updatedTime"] = updated_time
    return payload


ZHIHU_ITEMS = [
    newsnow_item("2062507625783226699", "14 岁少年纹身，家长索赔 20 万，如何看待？",
                 "https://www.zhihu.com/question/2062507625783226699",
                 info="461 万热度", hover="近日，据高女士反映……"),
    newsnow_item("2063187091773773522", "滔搏终止耐克线上销售，影响有多大？",
                 "https://www.zhihu.com/question/2063187091773773522",
                 info="458 万热度", hover="7 月 22 日，滔搏国际公告称……"),
]


# ═══════════════════════════════════════════
# 1. 正常解析
# ═══════════════════════════════════════════

def test_fetch_hot_parses_zhihu_structure():
    """知乎源实测结构：全字段映射 + platform 原样透传 + 毫秒时间戳转 ISO。"""
    sess = FakeSession(lambda url, kw: FakeResp(newsnow_payload(ZHIHU_ITEMS)))
    posts = agg.fetch_hot("zhihu", session=sess, sleep=make_sleep([]))

    assert len(posts) == 2
    p = posts[0]
    assert p["platform"] == "zhihu"
    assert p["post_id"] == "2062507625783226699"
    assert p["title"] == "14 岁少年纹身，家长索赔 20 万，如何看待？"
    assert p["content"] == "近日，据高女士反映……"          # extra.hover
    assert p["author"] == ""                                # 聚合器无作者字段
    assert p["url"] == "https://www.zhihu.com/question/2062507625783226699"
    assert p["source"] == "newsnow_zhihu"
    assert p["metrics"]["heat"] == 4610000                  # "461 万热度" → 绝对值
    expected_iso = datetime.fromtimestamp(1784725742588 / 1000, tz=timezone.utc).isoformat(timespec="seconds")
    assert p["published_at"] == expected_iso                # 顶层 updatedTime 毫秒


def test_fetch_hot_weibo_items_without_heat():
    """微博源实测结构（仅 extra.icon）：content 空串、metrics 无 heat 键。"""
    items = [newsnow_item("别再给AI乱传文件了", "别再给AI乱传文件了",
                          "https://s.weibo.com/weibo?q=x&t=31")]
    sess = FakeSession(lambda url, kw: FakeResp(newsnow_payload(items, "weibo")))
    posts = agg.fetch_hot("weibo", session=sess, sleep=make_sleep([]))
    assert len(posts) == 1
    p = posts[0]
    assert p["content"] == ""
    assert p["metrics"] == {}
    assert p["source"] == "newsnow_weibo"
    assert p["platform"] == "weibo"


def test_fetch_hot_douyin_items_without_extra():
    """抖音源实测结构（无 extra 字段）：正常解析不抛。"""
    items = [{"id": "2580760", "title": "在家也能复刻成吉思鸡",
              "url": "https://www.douyin.com/hot/2580760"}]
    sess = FakeSession(lambda url, kw: FakeResp(newsnow_payload(items, "douyin")))
    posts = agg.fetch_hot("douyin", session=sess, sleep=make_sleep([]))
    assert len(posts) == 1
    assert posts[0]["post_id"] == "2580760"
    assert posts[0]["metrics"] == {}
    assert posts[0]["content"] == ""


def test_fetch_hot_heat_unparseable_omits_key():
    """extra.info 无数字 → 省略 heat 键不抛。"""
    items = [newsnow_item("1", "标题", "https://x", info="热度飙升", hover="摘要")]
    sess = FakeSession(lambda url, kw: FakeResp(newsnow_payload(items)))
    posts = agg.fetch_hot("zhihu", session=sess, sleep=make_sleep([]))
    assert "heat" not in posts[0]["metrics"]
    assert posts[0]["content"] == "摘要"


# ═══════════════════════════════════════════
# 2. source_id 映射与非法 platform
# ═══════════════════════════════════════════

def test_fetch_hot_source_id_mapping():
    """四源映射实测定案：bilibili→'bilibili-hot-search'，请求带 id + latest。"""
    expected = {"weibo": "weibo", "zhihu": "zhihu", "douyin": "douyin",
                "bilibili": "bilibili-hot-search"}
    for platform, source_id in expected.items():
        sess = FakeSession(lambda url, kw: FakeResp(newsnow_payload([], source_id)))
        agg.fetch_hot(platform, session=sess, sleep=make_sleep([]))
        url, kw = sess.calls[0]
        assert url == agg.NEWSNOW_API_URL
        assert kw["params"]["id"] == source_id
        assert "latest" in kw["params"]
        assert kw["timeout"] == agg.DEFAULT_TIMEOUT


def test_fetch_hot_invalid_platform_no_request():
    """非法 platform（含小红书：实测 500 无此源）→ 空列表 + warning + 零请求。"""
    for bad in ("xiaohongshu", "rednote", "zhihu2", "", None, 123):
        sess = FakeSession(lambda url, kw: FakeResp(newsnow_payload([])))
        assert agg.fetch_hot(bad, session=sess, sleep=make_sleep([])) == []
        assert sess.calls == []


def test_fetch_hot_platform_case_insensitive_and_passthrough():
    """platform 大小写/空白宽容映射；Post.platform 保留调用方原值。"""
    items = [newsnow_item("1", "标题", "https://x")]
    sess = FakeSession(lambda url, kw: FakeResp(newsnow_payload(items)))
    posts = agg.fetch_hot("  ZhiHu ", session=sess, sleep=make_sleep([]))
    assert len(posts) == 1
    assert posts[0]["platform"] == "  ZhiHu "               # 传什么就是什么
    assert posts[0]["source"] == "newsnow_zhihu"
    assert sess.calls[0][1]["params"]["id"] == "zhihu"


# ═══════════════════════════════════════════
# 3. 字段缺失防御
# ═══════════════════════════════════════════

def test_fetch_hot_missing_updated_time():
    """顶层缺 updatedTime → published_at 空串不抛。"""
    items = [newsnow_item("1", "标题", "https://x")]
    sess = FakeSession(lambda url, kw: FakeResp(
        newsnow_payload(items, updated_time=None)))
    posts = agg.fetch_hot("zhihu", session=sess, sleep=make_sleep([]))
    assert posts[0]["published_at"] == ""


def test_fetch_hot_skips_items_without_id_or_title():
    """缺 id / 缺 title / 非 dict 条目逐条跳过，其余保留。"""
    items = [
        {"title": "无 id", "url": "https://x"},
        newsnow_item("2", "", "https://x"),
        "not-a-dict",
        newsnow_item("3", "正常条目", "https://x"),
    ]
    sess = FakeSession(lambda url, kw: FakeResp(newsnow_payload(items)))
    posts = agg.fetch_hot("zhihu", session=sess, sleep=make_sleep([]))
    assert [p["post_id"] for p in posts] == ["3"]


def test_fetch_hot_items_not_list():
    """items 非 list（dict/None/缺席）→ 空列表 + warning。"""
    for bad in [{"items": {"x": 1}}, {"items": None}, {"status": "cache"}]:
        sess = FakeSession(lambda url, kw, b=bad: FakeResp(b))
        assert agg.fetch_hot("zhihu", session=sess, sleep=make_sleep([])) == []


# ═══════════════════════════════════════════
# 4. HTTP 错误降级
# ═══════════════════════════════════════════

def test_fetch_hot_http_error_returns_empty():
    """HTTP 4xx/5xx（含 newsnow 非法源同款 500）→ 空列表不抛。"""
    for sc in (400, 429, 500, 503):
        sess = FakeSession(lambda url, kw, s=sc: FakeResp(
            {"error": "Invalid source id"}, status_code=s))
        assert agg.fetch_hot("zhihu", session=sess, sleep=make_sleep([])) == []


def test_fetch_hot_network_exception_returns_empty():
    """session.get 抛异常（连接错误/超时）→ 空列表不抛。"""
    def boom(url, kw):
        raise TimeoutError("timeout")
    sess = FakeSession(boom)
    assert agg.fetch_hot("weibo", session=sess, sleep=make_sleep([])) == []


def test_fetch_hot_invalid_json_returns_empty():
    """响应非 JSON / 顶层非 dict → 空列表不抛。"""
    sess = FakeSession(lambda url, kw: FakeResp(None, raw_text="Bad Gateway"))
    assert agg.fetch_hot("douyin", session=sess, sleep=make_sleep([])) == []
    sess2 = FakeSession(lambda url, kw: FakeResp(None, raw_text='["a","b"]'))
    assert agg.fetch_hot("douyin", session=sess2, sleep=make_sleep([])) == []


# ═══════════════════════════════════════════
# 5. limit 行为
# ═══════════════════════════════════════════

def test_fetch_hot_limit_truncates():
    """端点一次全量，limit 在客户端截断。"""
    items = [newsnow_item(str(i), f"条目{i}", f"https://x/{i}") for i in range(30)]
    sess = FakeSession(lambda url, kw: FakeResp(newsnow_payload(items)))
    posts = agg.fetch_hot("weibo", limit=5, session=sess, sleep=make_sleep([]))
    assert len(posts) == 5


def test_fetch_hot_invalid_limit_falls_back_to_default():
    """非法 limit（None/0/负数/垃圾）→ 默认 20，不抛。"""
    items = [newsnow_item(str(i), f"条目{i}", f"https://x/{i}") for i in range(25)]
    for bad_limit in (None, 0, -3, "abc"):
        sess = FakeSession(lambda url, kw: FakeResp(newsnow_payload(items)))
        posts = agg.fetch_hot("zhihu", limit=bad_limit, session=sess, sleep=make_sleep([]))
        assert len(posts) == 20


# ═══════════════════════════════════════════
# 6. 限速注入与 RateGate
# ═══════════════════════════════════════════

def test_fetch_hot_single_request_no_sleep():
    """单请求场景：RateGate 首请求不限速，fake sleep 不被调用。"""
    record = []
    sess = FakeSession(lambda url, kw: FakeResp(newsnow_payload([])))
    agg.fetch_hot("zhihu", session=sess, sleep=make_sleep(record))
    assert record == []


def test_rate_gate_sleeps_from_second_request():
    """_RateGate：第 2/3 次 wait 才休眠，间隔 ∈ [RATE, RATE+JITTER]。"""
    record = []
    gate = agg._RateGate(make_sleep(record))
    gate.wait()
    gate.wait()
    gate.wait()
    assert len(record) == 2
    for s in record:
        assert agg.DEFAULT_RATE <= s <= agg.DEFAULT_RATE + agg.DEFAULT_JITTER


# ═══════════════════════════════════════════
# 7. 默认 session 工厂
# ═══════════════════════════════════════════

def test_default_session_trust_env_false_and_ua(monkeypatch):
    """未注入 session 时自建：trust_env=False（绕开系统代理）+ 浏览器 UA。"""
    created = {}

    class FakeRealSession:
        def __init__(self):
            self.headers = {}
            self.trust_env = True
            created["s"] = self

    monkeypatch.setattr(agg.requests, "Session", FakeRealSession)
    sess = agg._default_session()
    assert sess.trust_env is False
    assert "Chrome" in sess.headers["User-Agent"]


def test_fetch_hot_without_session_uses_default(monkeypatch):
    """fetch_hot 不传 session：走 _default_session 工厂（可 monkeypatch）。"""
    fake = FakeSession(lambda url, kw: FakeResp(newsnow_payload(
        [newsnow_item("9", "默认会话条目", "https://x")])))
    monkeypatch.setattr(agg, "_default_session", lambda: fake)
    posts = agg.fetch_hot("bilibili", sleep=make_sleep([]))
    assert len(posts) == 1 and posts[0]["post_id"] == "9"
    assert posts[0]["source"] == "newsnow_bilibili-hot-search"
    assert len(fake.calls) == 1


# ═══════════════════════════════════════════
# 8. 契约与清洗
# ═══════════════════════════════════════════

def test_no_search_or_comments_functions():
    """聚合器仅热榜兜底：刻意不提供 search / fetch_comments。"""
    assert not hasattr(agg, "search")
    assert not hasattr(agg, "fetch_comments")


def test_fetch_hot_strips_html_tags():
    """title/hover 含 HTML 标签时清洗。"""
    items = [newsnow_item("1", "<em>茅台</em>提价", "https://x",
                          hover="<b>酱香</b>科技")]
    sess = FakeSession(lambda url, kw: FakeResp(newsnow_payload(items)))
    posts = agg.fetch_hot("zhihu", session=sess, sleep=make_sleep([]))
    assert posts[0]["title"] == "茅台提价"
    assert posts[0]["content"] == "酱香科技"
