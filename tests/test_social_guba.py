"""agent/social_guba.py 单元测试（全 mock 零网络）。

fixture 按 research/guba_endpoints_recon.md（2026-07-22 实测定案）结构构造：
- 列表：POST Articlelist，rc==1 为成功判据，re[] 条目含 post_id/post_title/
  user_nickname/post_click_count/post_comment_count/post_forward_count/
  post_publish_time；列表无点赞字段。
- 详情：HTML SSR 内嵌 var post_article={...}，花括号配平提取后 json.loads。
"""

import json
import logging

import pytest

import agent.social_guba as sg


# ── 测试替身 ──

class FakeResp:
    def __init__(self, status=200, payload=None, text=None):
        self.status_code = status
        self._payload = payload
        if text is not None:
            self.text = text
        elif payload is not None:
            self.text = json.dumps(payload, ensure_ascii=False)
        else:
            self.text = ""

    def json(self):
        if self._payload is None:
            raise ValueError("No JSON")
        return self._payload


class FakeSession:
    """按 URL 子串路由到响应队列；记录全部 post/get 调用。"""

    def __init__(self, post_routes=None, get_routes=None):
        self.post_routes = post_routes or {}
        self.get_routes = get_routes or {}
        self.post_calls = []
        self.get_calls = []

    def post(self, url, data=None, timeout=None, headers=None, **kw):
        self.post_calls.append({"url": url, "data": data, "timeout": timeout})
        return self._dispatch(self.post_routes, url)

    def get(self, url, timeout=None, headers=None, **kw):
        self.get_calls.append({"url": url, "timeout": timeout})
        return self._dispatch(self.get_routes, url)

    @staticmethod
    def _dispatch(routes, url):
        for needle, queue in routes.items():
            if needle in url:
                if isinstance(queue, list):
                    if queue:
                        return queue.pop(0)
                    return FakeResp(500, text="")
                return queue
        return FakeResp(404, text="")


class RaisingSession:
    def __init__(self):
        self.post_calls = []
        self.get_calls = []

    def post(self, url, **kw):
        self.post_calls.append({"url": url})
        raise ConnectionError("boom")

    def get(self, url, **kw):
        self.get_calls.append({"url": url})
        raise ConnectionError("boom")


class FakeSleeps:
    def __init__(self):
        self.values = []

    def __call__(self, seconds):
        self.values.append(seconds)


# ── fixture 原型（结构对齐 recon 实测定案）──

CODE = "600519"
PID = 1748053530


def list_row(**over):
    row = {
        "post_id": PID,
        "post_title": "茅台中报前瞻：稳健增长可期",
        "post_content": "",
        "stockbar_code": CODE,
        "stockbar_name": "贵州茅台吧",
        "user_id": 987654,
        "user_nickname": "价值投资者老王",
        "post_click_count": 1395,
        "post_comment_count": 8,
        "post_forward_count": 3,
        "post_like_count": None,          # 列表无点赞字段（实测为 None）
        "post_publish_time": "2026-07-22 19:10:41",
        "post_last_time": "2026-07-22 21:03:15",
    }
    row.update(over)
    return row


def list_payload(rows=None, rc=1, count=19638):
    return {"rc": rc, "rt": 11, "sSuccess": 1 if rc == 1 else 0,
            "message": "成功" if rc == 1 else "系统繁忙[00003]",
            "count": count,
            "re": rows if rows is not None else [list_row()]}


def detail_article(**over):
    art = {
        "post_id": PID,
        "post_title": "茅台中报前瞻：稳健增长可期",
        "post_content": ('<div data-type="abstract">摘要 div</div>'
                         '<p>正文第一段<br>第二行 &amp; 符号</p>'),
        "post_abstract": "茅台中报前瞻摘要，约一百字左右的纯文本。",
        "post_like_count": 3,             # 唯一点赞来源
        "post_click_count": 1395,
        "post_comment_count": 8,
        "post_forward_count": 3,
        "post_publish_time": "2026-07-22 19:10:41",
        "post_user": {"user_id": 987654, "user_nickname": "价值投资者老王",
                      "user_age": "13.5年", "user_influ_level": 9},
    }
    art.update(over)
    return art


def detail_page(article, trailing=""):
    return ("<html><head><title>股吧</title></head><body><script>\n"
            "var post_article=" + json.dumps(article, ensure_ascii=False)
            + ";\n" + trailing + "</script></body></html>")


def make_post(pid, comments, code=CODE):
    return {"platform": "guba", "post_id": str(pid), "title": f"帖{pid}",
            "content": "", "author": "某用户",
            "metrics": {"views": 100, "comments": comments, "shares": 1},
            "url": f"https://guba.eastmoney.com/news,{code},{pid}.html",
            "published_at": "2026-07-22T19:10:41+08:00",
            "source": "guba_list"}


# ═══════════════════════════════════════════
# 列表：正常解析与契约
# ═══════════════════════════════════════════

def test_list_normal_field_mapping():
    sess = FakeSession(post_routes={"Articlelist": [FakeResp(200, list_payload())]})
    posts = sg.fetch_bar_posts(CODE, limit=30, session=sess, sleep=FakeSleeps())
    assert len(posts) == 1
    p = posts[0]
    assert p["platform"] == "guba"
    assert p["post_id"] == str(PID)            # int → str
    assert p["title"] == "茅台中报前瞻：稳健增长可期"
    assert p["content"] == ""
    assert p["author"] == "价值投资者老王"
    assert p["metrics"] == {"views": 1395, "comments": 8, "shares": 3}
    assert "likes" not in p["metrics"]          # 列表无点赞字段
    assert p["url"] == f"https://guba.eastmoney.com/news,{CODE},{PID}.html"
    assert p["published_at"] == "2026-07-22T19:10:41+08:00"  # 北京时间 +08:00
    assert p["source"] == "guba_list"


def test_list_magic_params_always_sent():
    sess = FakeSession(post_routes={"Articlelist": [FakeResp(200, list_payload())]})
    sg.fetch_bar_posts(CODE, limit=30, session=sess, sleep=FakeSleeps())
    assert len(sess.post_calls) == 1
    form = sess.post_calls[0]["data"]
    assert form["deviceid"] == "Wap10.0.0.1"    # 魔法参数必带
    assert form["version"] == "200"             # 魔法参数必带
    assert form["code"] == CODE
    assert form["p"] == "1"
    assert form["ps"] == "30"


def test_list_pagination_aggregates_across_pages():
    rows_p1 = [list_row(post_id=1000 + i) for i in range(100)]
    rows_p2 = [list_row(post_id=2000 + i) for i in range(20)]
    sess = FakeSession(post_routes={"Articlelist": [
        FakeResp(200, list_payload(rows_p1)),
        FakeResp(200, list_payload(rows_p2)),
    ]})
    posts = sg.fetch_bar_posts(CODE, limit=120, session=sess, sleep=FakeSleeps())
    assert len(posts) == 120
    assert len(sess.post_calls) == 2
    assert sess.post_calls[0]["data"]["p"] == "1"
    assert sess.post_calls[1]["data"]["p"] == "2"
    assert sess.post_calls[0]["data"]["ps"] == "100"   # ps 上限 100
    assert sess.post_calls[1]["data"]["ps"] == "20"
    # 两页都必须带魔法参数
    for call in sess.post_calls:
        assert call["data"]["deviceid"] == "Wap10.0.0.1"
        assert call["data"]["version"] == "200"


def test_list_terminates_on_short_page():
    rows = [list_row(post_id=1000 + i) for i in range(10)]
    sess = FakeSession(post_routes={"Articlelist": [
        FakeResp(200, list_payload(rows))]})
    posts = sg.fetch_bar_posts(CODE, limit=30, session=sess, sleep=FakeSleeps())
    assert len(posts) == 10
    assert len(sess.post_calls) == 1            # 本页不足 ps 即终止，不翻页


def test_list_terminates_on_empty_re():
    sess = FakeSession(post_routes={"Articlelist": [
        FakeResp(200, list_payload(rows=[]))]})
    posts = sg.fetch_bar_posts(CODE, limit=30, session=sess, sleep=FakeSleeps())
    assert posts == []
    assert len(sess.post_calls) == 1


def test_list_truncates_to_limit():
    rows = [list_row(post_id=1000 + i) for i in range(5)]
    sess = FakeSession(post_routes={"Articlelist": [
        FakeResp(200, list_payload(rows))]})
    posts = sg.fetch_bar_posts(CODE, limit=3, session=sess, sleep=FakeSleeps())
    assert len(posts) == 3                      # re 多于 limit 时截断


def test_list_string_counts_coerced():
    sess = FakeSession(post_routes={"Articlelist": [FakeResp(200, list_payload(
        rows=[list_row(post_click_count="1395", post_comment_count="8",
                       post_forward_count="3")]))]})
    posts = sg.fetch_bar_posts(CODE, session=sess, sleep=FakeSleeps())
    assert posts[0]["metrics"] == {"views": 1395, "comments": 8, "shares": 3}


def test_list_missing_post_id_row_skipped():
    rows = [list_row(post_id=None), list_row(post_id=2222)]
    sess = FakeSession(post_routes={"Articlelist": [
        FakeResp(200, list_payload(rows))]})
    posts = sg.fetch_bar_posts(CODE, session=sess, sleep=FakeSleeps())
    assert [p["post_id"] for p in posts] == ["2222"]


def test_list_missing_metrics_keys_absent():
    row = list_row()
    del row["post_click_count"]
    del row["post_forward_count"]
    sess = FakeSession(post_routes={"Articlelist": [
        FakeResp(200, list_payload([row]))]})
    posts = sg.fetch_bar_posts(CODE, session=sess, sleep=FakeSleeps())
    assert posts[0]["metrics"] == {"comments": 8}   # 只放拿到的键


def test_list_invalid_publish_time_gives_empty():
    sess = FakeSession(post_routes={"Articlelist": [FakeResp(200, list_payload(
        rows=[list_row(post_publish_time="not-a-time")]))]})
    posts = sg.fetch_bar_posts(CODE, session=sess, sleep=FakeSleeps())
    assert posts[0]["published_at"] == ""


# ═══════════════════════════════════════════
# 列表：降级路径（rc / HTTP / 异常 / 非法 code）
# ═══════════════════════════════════════════

def test_list_rc_zero_degrades(caplog):
    """缺魔法参数时服务端表现（rc=0 空数据）→ 返回 [] + warning。"""
    sess = FakeSession(post_routes={"Articlelist": [
        FakeResp(200, list_payload(rows=[], rc=0))]})
    with caplog.at_level(logging.WARNING, logger="agent.social_guba"):
        posts = sg.fetch_bar_posts(CODE, session=sess, sleep=FakeSleeps())
    assert posts == []
    assert any("rc=" in r.message for r in caplog.records)


def test_list_rc_other_nonzero_degrades(caplog):
    sess = FakeSession(post_routes={"Articlelist": [
        FakeResp(200, list_payload(rc=2))]})
    with caplog.at_level(logging.WARNING, logger="agent.social_guba"):
        posts = sg.fetch_bar_posts(CODE, session=sess, sleep=FakeSleeps())
    assert posts == []
    assert any("rc=" in r.message for r in caplog.records)


def test_list_http_error(caplog):
    sess = FakeSession(post_routes={"Articlelist": [FakeResp(500, text="")]})
    with caplog.at_level(logging.WARNING, logger="agent.social_guba"):
        posts = sg.fetch_bar_posts(CODE, session=sess, sleep=FakeSleeps())
    assert posts == []
    assert any("HTTP 500" in r.message for r in caplog.records)


def test_list_request_exception(caplog):
    sess = RaisingSession()
    with caplog.at_level(logging.WARNING, logger="agent.social_guba"):
        posts = sg.fetch_bar_posts(CODE, session=sess, sleep=FakeSleeps())
    assert posts == []
    assert any("请求失败" in r.message for r in caplog.records)


def test_list_invalid_json_body(caplog):
    sess = FakeSession(post_routes={"Articlelist": [
        FakeResp(200, text="<html>not json</html>")]})
    with caplog.at_level(logging.WARNING, logger="agent.social_guba"):
        posts = sg.fetch_bar_posts(CODE, session=sess, sleep=FakeSleeps())
    assert posts == []
    assert any("JSON" in r.message for r in caplog.records)


@pytest.mark.parametrize("bad", ["60051", "6005190", "abcdef", "", None, 600519.5])
def test_list_invalid_code_returns_empty_no_request(bad, caplog):
    sess = FakeSession(post_routes={"Articlelist": [FakeResp(200, list_payload())]})
    with caplog.at_level(logging.WARNING, logger="agent.social_guba"):
        posts = sg.fetch_bar_posts(bad, session=sess, sleep=FakeSleeps())
    assert posts == []
    assert sess.post_calls == []                # 非法 code 不发请求
    assert any("非法股票代码" in r.message for r in caplog.records)


def test_list_partial_failure_keeps_collected():
    """第二页失败时保留第一页已抓到的部分（防御式降级）。"""
    rows_p1 = [list_row(post_id=1000 + i) for i in range(100)]
    sess = FakeSession(post_routes={"Articlelist": [
        FakeResp(200, list_payload(rows_p1)),
        FakeResp(502, text=""),
    ]})
    posts = sg.fetch_bar_posts(CODE, limit=120, session=sess, sleep=FakeSleeps())
    assert len(posts) == 100


# ═══════════════════════════════════════════
# 限速与 session 注入
# ═══════════════════════════════════════════

def test_rate_gate_sleeps_between_pages():
    rows_p1 = [list_row(post_id=1000 + i) for i in range(100)]
    rows_p2 = [list_row(post_id=2000 + i) for i in range(20)]
    sleeps = FakeSleeps()
    sess = FakeSession(post_routes={"Articlelist": [
        FakeResp(200, list_payload(rows_p1)),
        FakeResp(200, list_payload(rows_p2)),
    ]})
    sg.fetch_bar_posts(CODE, limit=120, session=sess, sleep=sleeps)
    assert len(sleeps.values) == 1              # 首轮不限速，其后限速
    assert 1.0 <= sleeps.values[0] <= 1.0 + sg.DEFAULT_JITTER


def test_rate_gate_no_sleep_single_request():
    sleeps = FakeSleeps()
    sess = FakeSession(post_routes={"Articlelist": [FakeResp(200, list_payload())]})
    sg.fetch_bar_posts(CODE, limit=30, session=sess, sleep=sleeps)
    assert sleeps.values == []                  # 首个请求不限速


def test_session_injection_used(monkeypatch):
    def _boom():  # 若走默认会话创建即失败，证明注入会话被使用
        raise AssertionError("should not create default session")
    monkeypatch.setattr(sg, "_new_session", _boom)
    sess = FakeSession(post_routes={"Articlelist": [FakeResp(200, list_payload())]})
    posts = sg.fetch_bar_posts(CODE, session=sess, sleep=FakeSleeps())
    assert len(posts) == 1
    assert sess.post_calls[0]["timeout"] == sg.DEFAULT_TIMEOUT


# ═══════════════════════════════════════════
# 纯函数：配平提取 / HTML 清洗 / 时间转换
# ═══════════════════════════════════════════

def test_balanced_json_basic():
    text = 'prefix {"a": 1, "b": {"c": 2}} suffix'
    assert sg._extract_balanced_json(text, text.index("{")) == '{"a": 1, "b": {"c": 2}}'


def test_balanced_json_braces_inside_string_and_escapes():
    blob = '{"s": "他说\\"{关键}\\"，路径 C:\\\\x\\\\{y}，}}}", "n": 1}'
    start = blob.index("{")
    assert sg._extract_balanced_json(blob, start) == blob
    assert json.loads(blob)["s"] == '他说"{关键}"，路径 C:\\x\\{y}，}}}'


def test_balanced_json_unbalanced_returns_none():
    assert sg._extract_balanced_json('{"a": "未闭合', 0) is None
    assert sg._extract_balanced_json("", 0) is None


def test_strip_html_entities_blocks_and_ws():
    raw = '<div>甲 &amp; 乙</div><p>丙<br>丁&nbsp;戊</p>'
    assert sg._strip_html_to_text(raw) == "甲 & 乙\n丙\n丁\xa0戊"


def test_strip_html_truncation_and_none():
    assert sg._strip_html_to_text(None) == ""
    assert sg._strip_html_to_text("x" * 3000) == "x" * 2000
    assert sg._strip_html_to_text("x" * 3000, max_len=10) == "x" * 10


def test_to_beijing_iso_variants():
    assert sg._to_beijing_iso("2026-07-22 19:10:41") == "2026-07-22T19:10:41+08:00"
    assert sg._to_beijing_iso("2026-07-22 19:10") == "2026-07-22T19:10:00+08:00"
    assert sg._to_beijing_iso(None) == ""
    assert sg._to_beijing_iso("garbage") == ""


# ═══════════════════════════════════════════
# 详情：配平提取与字段契约
# ═══════════════════════════════════════════

def test_detail_normal_full_mapping():
    sess = FakeSession(get_routes={
        "news,600519,1748053530": FakeResp(200, text=detail_page(detail_article()))})
    d = sg.fetch_post_detail(str(PID), code=CODE, session=sess, sleep=FakeSleeps())
    assert d is not None
    assert d["post_id"] == str(PID)
    assert d["title"] == "茅台中报前瞻：稳健增长可期"
    assert d["content"] == "摘要 div\n正文第一段\n第二行 & 符号"  # HTML → 纯文本
    assert d["abstract"] == "茅台中报前瞻摘要，约一百字左右的纯文本。"
    assert d["metrics"] == {"likes": 3, "views": 1395, "comments": 8, "shares": 3}
    assert d["author"] == "价值投资者老王"
    assert d["published_at"] == "2026-07-22T19:10:41+08:00"
    assert d["url"] == f"https://guba.eastmoney.com/news,{CODE},{PID}.html"


def test_detail_adversarial_braces_and_escapes_in_content():
    """对抗样例：正文带花括号与转义引号，配平不得提前截断。"""
    nasty = '他说\\"目标价{1800元}不变\\"，路径 C:\\\\data\\\\{cache}，}}}'
    page = detail_page(detail_article(post_content=nasty),
                       trailing='var post_other={"x":{"y":1}};')
    sess = FakeSession(get_routes={
        "news,600519,1748053530": FakeResp(200, text=page)})
    d = sg.fetch_post_detail(str(PID), code=CODE, session=sess, sleep=FakeSleeps())
    assert d is not None
    assert '目标价{1800元}不变' in d["content"]
    assert "}}}" in d["content"]
    assert "post_other" not in json.dumps(d, ensure_ascii=False)  # 正确停在配平点


def test_detail_content_truncated_to_2000():
    sess = FakeSession(get_routes={"news,600519": FakeResp(
        200, text=detail_page(detail_article(post_content="<p>" + "长" * 3000 + "</p>")))})
    d = sg.fetch_post_detail(str(PID), code=CODE, session=sess, sleep=FakeSleeps())
    assert d is not None
    assert len(d["content"]) == 2000


def test_detail_missing_like_count_no_likes_key():
    art = detail_article()
    del art["post_like_count"]
    sess = FakeSession(get_routes={"news,600519": FakeResp(200, text=detail_page(art))})
    d = sg.fetch_post_detail(str(PID), code=CODE, session=sess, sleep=FakeSleeps())
    assert d is not None
    assert "likes" not in d["metrics"]          # 只放拿到的键


def test_detail_missing_post_user_author_empty():
    art = detail_article()
    del art["post_user"]
    sess = FakeSession(get_routes={"news,600519": FakeResp(200, text=detail_page(art))})
    d = sg.fetch_post_detail(str(PID), code=CODE, session=sess, sleep=FakeSleeps())
    assert d is not None
    assert d["author"] == ""


# ═══════════════════════════════════════════
# 详情：失败路径
# ═══════════════════════════════════════════

def test_detail_no_post_article_marker(caplog):
    sess = FakeSession(get_routes={"news,600519": FakeResp(
        200, text="<html><body>无内嵌数据</body></html>")})
    with caplog.at_level(logging.WARNING, logger="agent.social_guba"):
        d = sg.fetch_post_detail(str(PID), code=CODE, session=sess, sleep=FakeSleeps())
    assert d is None
    assert any("post_article" in r.message for r in caplog.records)


def test_detail_unbalanced_json_returns_none(caplog):
    page = "<script>var post_article={\"post_id\": 1, \"title\": \"未闭合</script>"
    sess = FakeSession(get_routes={"news,600519": FakeResp(200, text=page)})
    with caplog.at_level(logging.WARNING, logger="agent.social_guba"):
        d = sg.fetch_post_detail(str(PID), code=CODE, session=sess, sleep=FakeSleeps())
    assert d is None


def test_detail_invalid_json_returns_none(caplog):
    page = "<script>var post_article={not valid json};</script>"
    sess = FakeSession(get_routes={"news,600519": FakeResp(200, text=page)})
    with caplog.at_level(logging.WARNING, logger="agent.social_guba"):
        d = sg.fetch_post_detail(str(PID), code=CODE, session=sess, sleep=FakeSleeps())
    assert d is None


def test_detail_http_403_waf(caplog):
    """端点级 WAF 403（recon 观察：首次即封，非限流）→ None + warning。"""
    sess = FakeSession(get_routes={"news,600519": FakeResp(403, text="")})
    with caplog.at_level(logging.WARNING, logger="agent.social_guba"):
        d = sg.fetch_post_detail(str(PID), code=CODE, session=sess, sleep=FakeSleeps())
    assert d is None
    assert any("HTTP 403" in r.message for r in caplog.records)


def test_detail_request_exception(caplog):
    sess = RaisingSession()
    with caplog.at_level(logging.WARNING, logger="agent.social_guba"):
        d = sg.fetch_post_detail(str(PID), code=CODE, session=sess, sleep=FakeSleeps())
    assert d is None
    assert any("请求失败" in r.message for r in caplog.records)


def test_detail_code_parsed_from_url_post_id():
    """code 缺失时可从 URL 形态 post_id 解析。"""
    sess = FakeSession(get_routes={
        "news,600519,1748053530": FakeResp(200, text=detail_page(detail_article()))})
    d = sg.fetch_post_detail(
        f"https://guba.eastmoney.com/news,{CODE},{PID}.html",
        session=sess, sleep=FakeSleeps())
    assert d is not None
    assert d["post_id"] == str(PID)
    assert d["url"].endswith(f"news,{CODE},{PID}.html")


def test_detail_missing_code_returns_none(caplog):
    sess = FakeSession(get_routes={"news": FakeResp(200, text=detail_page(detail_article()))})
    with caplog.at_level(logging.WARNING, logger="agent.social_guba"):
        d = sg.fetch_post_detail(str(PID), session=sess, sleep=FakeSleeps())
    assert d is None
    assert sess.get_calls == []                 # 参数不足不发请求
    assert any("code" in r.message for r in caplog.records)


def test_detail_single_request_no_sleep():
    sleeps = FakeSleeps()
    sess = FakeSession(get_routes={
        "news,600519": FakeResp(200, text=detail_page(detail_article()))})
    sg.fetch_post_detail(str(PID), code=CODE, session=sess, sleep=sleeps)
    assert sleeps.values == []                  # 首个请求不限速


# ═══════════════════════════════════════════
# enrich_posts：回填 / 排序 / 失败保留
# ═══════════════════════════════════════════

def test_enrich_backfills_top_n_by_comments():
    posts = [make_post(11, comments=5), make_post(22, comments=99),
             make_post(33, comments=50)]
    art = detail_article(post_id=22, post_like_count=42,
                         post_content="<p>热帖正文</p>")
    sess = FakeSession(get_routes={"news,600519,22": FakeResp(200, text=detail_page(art))})
    out = sg.enrich_posts(posts, top_n=1, session=sess, sleep=FakeSleeps())
    assert len(sess.get_calls) == 1             # 只抓 comments 最高的 22
    enriched = {p["post_id"]: p for p in out}
    assert enriched["22"]["content"] == "热帖正文"
    assert enriched["22"]["metrics"]["likes"] == 42
    assert enriched["11"]["content"] == ""      # 未进 top_n 的保持原样
    assert "likes" not in enriched["11"]["metrics"]


def test_enrich_only_top_n_calls_and_shared_gate():
    posts = [make_post(i, comments=100 - i) for i in range(5)]
    page = detail_page(detail_article(post_like_count=7, post_content="<p>x</p>"))
    sleeps = FakeSleeps()
    sess = FakeSession(get_routes={"news,600519": FakeResp(200, text=page)})
    sg.enrich_posts(posts, top_n=3, session=sess, sleep=sleeps)
    assert len(sess.get_calls) == 3             # 恰好 top_n 次详情请求
    assert len(sleeps.values) == 2              # 同一限速门：首个不限速
    assert all(1.0 <= v <= 1.0 + sg.DEFAULT_JITTER for v in sleeps.values)


def test_enrich_failure_keeps_original_no_notes():
    posts = [make_post(11, comments=10), make_post(22, comments=20)]
    page = detail_page(detail_article(post_id=11, post_like_count=9,
                                      post_content="<p>成功正文</p>"))
    sess = FakeSession(get_routes={
        "news,600519,11": FakeResp(200, text=page),
        "news,600519,22": FakeResp(403, text=""),   # 22 详情失败
    })
    out = sg.enrich_posts(posts, top_n=2, session=sess, sleep=FakeSleeps())
    enriched = {p["post_id"]: p for p in out}
    assert enriched["22"]["content"] == ""      # 失败保留原帖
    assert "likes" not in enriched["22"]["metrics"]
    assert "notes" not in enriched["22"]        # 不进 notes
    assert enriched["11"]["metrics"]["likes"] == 9


def test_enrich_does_not_mutate_input():
    posts = [make_post(22, comments=20)]
    page = detail_page(detail_article(post_id=22, post_like_count=5,
                                      post_content="<p>新正文</p>"))
    sess = FakeSession(get_routes={"news,600519,22": FakeResp(200, text=page)})
    out = sg.enrich_posts(posts, top_n=1, session=sess, sleep=FakeSleeps())
    assert out[0]["content"] == "新正文"        # 副本被回填
    assert posts[0]["content"] == ""            # 入参不被修改
    assert "likes" not in posts[0]["metrics"]


def test_enrich_preserves_order_and_length():
    posts = [make_post(11, comments=5), make_post(22, comments=99),
             make_post(33, comments=50)]
    sess = FakeSession(get_routes={"news": FakeResp(403, text="")})
    out = sg.enrich_posts(posts, top_n=2, session=sess, sleep=FakeSleeps())
    assert [p["post_id"] for p in out] == ["11", "22", "33"]
    assert len(out) == len(posts)


def test_enrich_top_n_zero_no_calls():
    posts = [make_post(11, comments=10)]
    sess = FakeSession(get_routes={"news": FakeResp(200, text=detail_page(detail_article()))})
    out = sg.enrich_posts(posts, top_n=0, session=sess, sleep=FakeSleeps())
    assert sess.get_calls == []
    assert out[0]["content"] == ""


def test_enrich_empty_input():
    assert sg.enrich_posts([], top_n=3, session=FakeSession()) == []


def test_enrich_post_without_code_skipped(caplog):
    bad = make_post(44, comments=100)
    bad["url"] = "https://example.com/no-code-here"
    posts = [bad, make_post(55, comments=10)]
    page = detail_page(detail_article(post_id=55, post_like_count=1,
                                      post_content="<p>ok</p>"))
    sess = FakeSession(get_routes={"news,600519,55": FakeResp(200, text=page)})
    with caplog.at_level(logging.WARNING, logger="agent.social_guba"):
        out = sg.enrich_posts(posts, top_n=2, session=sess, sleep=FakeSleeps())
    enriched = {p["post_id"]: p for p in out}
    assert enriched["44"]["content"] == ""      # 无法解析 code → 跳过
    assert enriched["55"]["metrics"]["likes"] == 1
    assert any("跳过回填" in r.message for r in caplog.records)
