"""tests/test_bilibili_sessdata.py — B 站登录态评论（BILI_SESSDATA + wbi 签名）测试。

实测背景（2026-07-23）：评论接口匿名每视频仅约 3 条热评；登录态
（SESSDATA cookie + wbi 签名）走 x/v2/reply/wbi/main 拉全量。

覆盖：
1. 有 SESSDATA：fetch_comments 走签名路径（请求带 cookie 与 w_rid/wts，
   命中 reply/wbi/main，评论正常解析，plain 端点不被调用）。
2. 签名材料失败（nav 请求失败 / 响应缺 wbi_img）→ 降级不带签名的带
   cookie 请求（plain 路径，无 w_rid）。
3. 无 SESSDATA：行为与既有完全一致（不调 nav/wbi，无 cookie，plain 路径）。
4. wbi 路径翻页：cursor 翻页合并、单页失败保留已抓部分。
5. social_media 匿名降级透明化：无 BILI_SESSDATA 时 collect_keyword_samples
   notes 含说明；配置后不含。

全 mock 零网络：假 session 路由响应，monkeypatch 环境变量与 _warmup。
"""

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

    def json(self):
        if self._payload is None:
            raise ValueError("No JSON")
        return self._payload


class FakeSession:
    """按 URL 子串路由（routes 按插入序匹配）；记录全部请求。cookies 用 dict。"""

    def __init__(self, routes=None):
        self.routes = routes or {}
        self.calls = []
        self.cookies = {}
        self.headers = {}

    def get(self, url, params=None, timeout=None, headers=None, **kw):
        self.calls.append({"url": url, "params": dict(params or {})})
        for key, queue in self.routes.items():
            if key in url:
                if queue:
                    return queue.pop(0)
                return FakeResp(500, None)
        return FakeResp(404, None)

    def calls_for(self, needle):
        return [c for c in self.calls if needle in c["url"]]


@pytest.fixture(autouse=True)
def _clean_env_and_warmup(monkeypatch):
    """确定性：默认无 BILI_SESSDATA；热身 mock 成功（零网络）。"""
    monkeypatch.delenv("BILI_SESSDATA", raising=False)
    monkeypatch.setattr(sb, "_warmup", lambda session: True)


# ── fixture 原型 ──

def nav_payload():
    return {"code": 0, "message": "OK", "data": {
        "wbi_img": {
            "img_url": "https://i0.hdslb.com/bfs/wbi/7cd084941338484aae1ad9425b84077c.png",
            "sub_url": "https://i0.hdslb.com/bfs/wbi/4932caff0ff746eab6f01bf08b70ac45.png",
        }}}


def wbi_reply_payload(texts, is_end=True, next_cursor=0):
    return {"code": 0, "message": "OK", "data": {
        "cursor": {"is_end": is_end, "next": next_cursor},
        "replies": [
            {"rpid_str": str(i), "ctime": 1784726572, "like": i, "rcount": 0,
             "content": {"message": t},
             "member": {"mid": str(i), "uname": f"网友{i}"}}
            for i, t in enumerate(texts)]}}


def plain_reply_payload(texts):
    return {"code": 0, "message": "OK", "data": {
        "page": {"num": 1, "size": 20, "count": len(texts), "acount": len(texts)},
        "replies": [
            {"rpid_str": str(i), "ctime": 1784726572, "like": i, "rcount": 0,
             "content": {"message": t},
             "member": {"mid": str(i), "uname": f"网友{i}"}}
            for i, t in enumerate(texts)]}}


# ═══════════════════════════════════════════
# 1. 有 SESSDATA → wbi 签名路径
# ═══════════════════════════════════════════

def test_sessdata_signed_path(monkeypatch):
    """有 SESSDATA：走 reply/wbi/main，请求带 cookie 与 w_rid/wts。"""
    monkeypatch.setenv("BILI_SESSDATA", "fake-sessdata-123")
    sess = FakeSession({
        "x/web-interface/nav": [FakeResp(200, nav_payload())],
        "reply/wbi/main": [FakeResp(200, wbi_reply_payload(["全量评论甲", "全量评论乙"]))],
        "x/v2/reply": [FakeResp(200, plain_reply_payload(["plain 不应被用到"]))],
    })
    # 路由匹配顺序：reply/wbi/main 含 "x/v2/reply" 子串？不含——
    # "x/v2/reply/wbi/main" 含 "x/v2/reply"，FakeSession 按插入序先匹配 nav，
    # 再匹配 "reply/wbi/main"，最后才匹配 "x/v2/reply"，顺序已保证。
    out = sb.fetch_comments("123", session=sess, sleep=lambda s: None)
    assert [c["content"] for c in out] == ["全量评论甲", "全量评论乙"]
    # cookie 已挂载
    assert sess.cookies.get("SESSDATA") == "fake-sessdata-123"
    # 签名请求：w_rid（32 位 hex）+ wts
    wbi_calls = sess.calls_for("reply/wbi/main")
    assert len(wbi_calls) == 1
    params = wbi_calls[0]["params"]
    assert "w_rid" in params and len(params["w_rid"]) == 32
    assert "wts" in params
    assert params["type"] == 1 and params["oid"] == "123"
    # plain 端点未被调用
    assert sess.calls_for("x/v2/reply?") == []
    plain_calls = [c for c in sess.calls_for("x/v2/reply")
                   if "wbi" not in c["url"]]
    assert plain_calls == []


def test_sessdata_wbi_pagination(monkeypatch):
    """wbi 路径 cursor 翻页：两页合并，is_end 终止。"""
    monkeypatch.setenv("BILI_SESSDATA", "s")
    sess = FakeSession({
        "x/web-interface/nav": [FakeResp(200, nav_payload())],
        "reply/wbi/main": [
            FakeResp(200, wbi_reply_payload(
                [f"第1页评论{i}" for i in range(20)],
                is_end=False, next_cursor=1)),
            FakeResp(200, wbi_reply_payload(["第2页评论"])),
        ],
    })
    out = sb.fetch_comments("123", limit=30, session=sess,
                            sleep=lambda s: None)
    assert len(out) == 21
    assert out[-1]["content"] == "第2页评论"
    wbi_calls = sess.calls_for("reply/wbi/main")
    assert len(wbi_calls) == 2
    assert wbi_calls[1]["params"]["next"] == 1  # cursor 前进


def test_sessdata_wbi_page_failure_keeps_partial(monkeypatch):
    """wbi 第二页失败：保留第一页已抓部分，绝不抛。"""
    monkeypatch.setenv("BILI_SESSDATA", "s")
    sess = FakeSession({
        "x/web-interface/nav": [FakeResp(200, nav_payload())],
        "reply/wbi/main": [
            FakeResp(200, wbi_reply_payload(
                [f"评{i}" for i in range(20)], is_end=False, next_cursor=1)),
            FakeResp(500, None),
        ],
    })
    out = sb.fetch_comments("123", limit=30, session=sess,
                            sleep=lambda s: None)
    assert len(out) == 20


# ═══════════════════════════════════════════
# 2. 签名材料失败 → 降级 plain 带 cookie
# ═══════════════════════════════════════════

def test_sign_material_nav_fails_fallback_plain(monkeypatch):
    """nav 请求失败（404）：降级 plain 路径（无 w_rid），cookie 仍挂载。"""
    monkeypatch.setenv("BILI_SESSDATA", "s")
    sess = FakeSession({
        "x/web-interface/nav": [FakeResp(404, None)],
        "x/v2/reply": [FakeResp(200, plain_reply_payload(["plain 评论"]))],
    })
    out = sb.fetch_comments("123", session=sess, sleep=lambda s: None)
    assert [c["content"] for c in out] == ["plain 评论"]
    assert sess.cookies.get("SESSDATA") == "s"
    plain_calls = sess.calls_for("x/v2/reply")
    assert len(plain_calls) == 1
    assert "w_rid" not in plain_calls[0]["params"]


def test_sign_material_bad_structure_fallback_plain(monkeypatch):
    """nav 响应缺 wbi_img：同样降级 plain 路径。"""
    monkeypatch.setenv("BILI_SESSDATA", "s")
    sess = FakeSession({
        "x/web-interface/nav": [FakeResp(200, {"code": 0, "data": {}})],
        "x/v2/reply": [FakeResp(200, plain_reply_payload(["降级评论"]))],
    })
    out = sb.fetch_comments("123", session=sess, sleep=lambda s: None)
    assert [c["content"] for c in out] == ["降级评论"]
    assert sess.calls_for("reply/wbi/main") == []


# ═══════════════════════════════════════════
# 3. 无 SESSDATA → 行为完全不变
# ═══════════════════════════════════════════

def test_no_sessdata_unchanged():
    """无 SESSDATA：不调 nav/wbi，无 cookie，plain 路径与既有一致。"""
    sess = FakeSession({
        "x/v2/reply": [FakeResp(200, plain_reply_payload(["匿名评论"]))],
    })
    out = sb.fetch_comments("123", session=sess, sleep=lambda s: None)
    assert [c["content"] for c in out] == ["匿名评论"]
    assert sess.calls_for("web-interface/nav") == []
    assert sess.calls_for("wbi") == []
    assert "SESSDATA" not in sess.cookies
    assert len(sess.calls) == 1
    assert "w_rid" not in sess.calls[0]["params"]


# ═══════════════════════════════════════════
# 4. social_media 匿名降级透明化
# ═══════════════════════════════════════════

def _inject_bilibili(monkeypatch):
    mod = types.ModuleType("social_bilibili")
    mod.search = lambda keyword, limit=20, sleep=None, **kw: [
        {"platform": "bilibili", "post_id": "v1", "title": "视频",
         "content": "", "author": "", "metrics": {}, "url": "",
         "published_at": "", "source": "bilibili_search_video"}]
    mod.fetch_comments = lambda post_id, limit=20, sleep=None, **kw: [
        {"platform": "bilibili", "post_id": post_id, "author": "网友",
         "content": "看多", "likes": 1, "published_at": ""}]
    monkeypatch.setitem(sys.modules, "social_bilibili", mod)
    monkeypatch.setitem(sys.modules, "agent.social_bilibili", mod)


def test_anonymous_note_when_no_sessdata(monkeypatch):
    """无 BILI_SESSDATA：notes 含匿名热评受限说明。"""
    _inject_bilibili(monkeypatch)
    out = sm.collect_keyword_samples("半导体", sleep=lambda s: None)
    assert any("BILI_SESSDATA" in n and "匿名" in n for n in out["notes"]), \
        out["notes"]


def test_no_anonymous_note_when_sessdata_set(monkeypatch):
    """配置 BILI_SESSDATA：notes 不含匿名说明。"""
    monkeypatch.setenv("BILI_SESSDATA", "s")
    _inject_bilibili(monkeypatch)
    out = sm.collect_keyword_samples("半导体", sleep=lambda s: None)
    assert not any("BILI_SESSDATA" in n for n in out["notes"]), out["notes"]
