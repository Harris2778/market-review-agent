"""tests/test_sentiment_upgrade.py — 舆情分布升级（四桶/时间窗/深度档）测试（全 mock 零网络）。

覆盖：
1. social_bilibili.search 新增 order 参数：缺省不带 order（行为与现状一致），
   order="pubdate" 时请求参数带 order=pubdate。
2. collect_keyword_samples 采样路径调用 search 传 order="pubdate"；
   collect_keyword_samples / collect_guba_samples 的 since_days 时间窗过滤
   （近期保留 / 超窗丢弃 / 时间缺失保留并进 notes）。
3. get_sentiment_distribution：depth="deep" 参数透传（post_limit=300 /
   video_limit=8 / comments_per_video=50 / 评论总量 400 截断）、standard
   缺省量级不变、非法 depth 降级 standard 进 notes；window 输出（from/to
   ISO、无时间信息双 None 进 notes）；bull_bear 透传与合并路径重算。
4. tools 层：get_stock_sentiment / search_social_media schema 含 depth
   枚举，执行分支 depth 透传到 get_sentiment_distribution，缺省 standard。

零网络保证：LLM/聚合层用假模块注入 sys.modules；采集层用 monkeypatch 或
假平台模块；B 站 search 用本地 FakeSession。
"""

import json
import sys
import types
from datetime import datetime, timedelta, timezone

import pytest

import agent.social_bilibili as sb
import agent.social_media as sm
import agent.tools as tools_mod


def _iso_days_ago(days: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


# ════════════════════════════════════════════════════════════════
# 1. social_bilibili.search 的 order 参数
# ════════════════════════════════════════════════════════════════

class _FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False) if payload is not None else ""

    def json(self):
        if self._payload is None:
            raise ValueError("No JSON")
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self.calls = []
        self._payload = payload

    def get(self, url, params=None, timeout=None, headers=None, **kw):
        self.calls.append({"url": url, "params": params})
        if "search/type" in url:
            return _FakeResp(200, self._payload)
        return _FakeResp(404, None)


def _search_payload():
    return {"code": 0, "message": "OK", "ttl": 1, "data": {
        "numResults": 1, "numPages": 1, "result": [
            {"type": "video", "id": 1, "aid": 1, "bvid": "BV1xx",
             "author": "某UP", "title": "A股复盘", "description": "",
             "play": 10, "review": 2, "like": 3, "favorites": 1,
             "pubdate": 1784619114},
        ]}}


@pytest.fixture
def _no_warmup(monkeypatch):
    monkeypatch.setattr(sb, "_warmup", lambda session: True)


def test_search_default_no_order_param(_no_warmup):
    """不传 order：请求参数不带 order 键（行为与现状完全一致）。"""
    sess = _FakeSession(_search_payload())
    posts = sb.search("a股", session=sess, sleep=lambda s: None)
    assert len(posts) == 1
    params = sess.calls[0]["params"]
    assert "order" not in params
    assert params["search_type"] == "video" and params["keyword"] == "a股"


def test_search_pubdate_order_param(_no_warmup):
    """order="pubdate"：请求参数带 order=pubdate（按发布时间倒序）。"""
    sess = _FakeSession(_search_payload())
    posts = sb.search("a股", session=sess, sleep=lambda s: None,
                      order="pubdate")
    assert len(posts) == 1
    assert sess.calls[0]["params"]["order"] == "pubdate"


# ════════════════════════════════════════════════════════════════
# 2. 采集层：order=pubdate 透传 + since_days 时间窗过滤
# ════════════════════════════════════════════════════════════════

def _video(pid, title="视频"):
    return {"platform": "bilibili", "post_id": pid, "title": title,
            "content": "", "author": "UP", "metrics": {}, "url": "",
            "published_at": "", "source": "bilibili_search_video"}


def _comment(pid, text, published_at):
    return {"platform": "bilibili", "post_id": pid, "author": "网友",
            "content": text, "likes": 1, "published_at": published_at}


def _inject_bilibili(monkeypatch, videos, comments_map, search_calls=None):
    mod = types.ModuleType("social_bilibili")

    def search(keyword, limit=20, sleep=None, **kw):
        if search_calls is not None:
            search_calls.append({"keyword": keyword, "limit": limit, **kw})
        return videos

    def fetch_comments(post_id, limit=20, sleep=None, **kw):
        return [dict(c) for c in comments_map.get(post_id, [])]

    mod.search = search
    mod.fetch_comments = fetch_comments
    monkeypatch.setitem(sys.modules, "social_bilibili", mod)
    monkeypatch.setitem(sys.modules, "agent.social_bilibili", mod)
    return mod


def _inject_guba(monkeypatch, posts):
    mod = types.ModuleType("social_guba")
    mod.fetch_bar_posts = lambda code, limit=30, sleep=None, **kw: \
        [dict(p) for p in posts]
    monkeypatch.setitem(sys.modules, "social_guba", mod)
    monkeypatch.setitem(sys.modules, "agent.social_guba", mod)
    return mod


def test_keyword_samples_search_uses_pubdate_order(monkeypatch):
    """采样路径调用 search 时传 order="pubdate"。"""
    calls = []
    _inject_bilibili(monkeypatch, [_video("v1")],
                     {"v1": [_comment("v1", "看多", _iso_days_ago(1))]},
                     search_calls=calls)
    sm.collect_keyword_samples("半导体", sleep=lambda s: None)
    assert calls and calls[0]["order"] == "pubdate"


def test_keyword_samples_since_days_filters_old(monkeypatch):
    """since_days 过滤：近期保留、超窗丢弃、时间缺失保留并进 notes。"""
    videos = [_video("v1")]
    comments_map = {"v1": [
        _comment("v1", "今天的评论", _iso_days_ago(1)),
        _comment("v1", "三十天前的评论", _iso_days_ago(30)),
        _comment("v1", "无时间评论", ""),
    ]}
    _inject_bilibili(monkeypatch, videos, comments_map)
    out = sm.collect_keyword_samples("半导体", sleep=lambda s: None,
                                     since_days=7)
    texts = [c["content"] for c in out["comments"]]
    assert "今天的评论" in texts
    assert "三十天前的评论" not in texts
    assert "无时间评论" in texts  # 时间缺失保留，不误杀
    assert any("缺少可解析时间" in n for n in out["notes"])
    assert any("时间窗" in n and "过滤掉" in n for n in out["notes"])


def test_keyword_samples_since_days_custom_window(monkeypatch):
    """自定义 since_days=40 时 30 天前的评论保留。"""
    videos = [_video("v1")]
    comments_map = {"v1": [_comment("v1", "三十天前", _iso_days_ago(30))]}
    _inject_bilibili(monkeypatch, videos, comments_map)
    out = sm.collect_keyword_samples("半导体", sleep=lambda s: None,
                                     since_days=40)
    assert len(out["comments"]) == 1


def test_guba_samples_since_days_filters_old(monkeypatch):
    """股吧采样：超窗帖子丢弃，缺失时间帖子保留并记 notes。"""
    posts = [
        {"platform": "guba", "post_id": "p1", "title": "新帖", "content": "",
         "author": "", "metrics": {}, "url": "",
         "published_at": _iso_days_ago(2), "source": "guba_list"},
        {"platform": "guba", "post_id": "p2", "title": "旧帖", "content": "",
         "author": "", "metrics": {}, "url": "",
         "published_at": _iso_days_ago(60), "source": "guba_list"},
        {"platform": "guba", "post_id": "p3", "title": "无时间帖", "content": "",
         "author": "", "metrics": {}, "url": "",
         "published_at": "", "source": "guba_list"},
    ]
    _inject_guba(monkeypatch, posts)
    out = sm.collect_guba_samples("600519", sleep=lambda s: None, since_days=7)
    ids = [p["post_id"] for p in out["posts"]]
    assert ids == ["p1", "p3"]
    assert any("缺少可解析时间" in n for n in out["notes"])
    assert any("过滤掉 1 条" in n for n in out["notes"])


# ════════════════════════════════════════════════════════════════
# 3. get_sentiment_distribution：depth / since_days / window / bull_bear
# ════════════════════════════════════════════════════════════════

@pytest.fixture
def fake_llm_agg(monkeypatch):
    """假 LLM（全乐观）+ 假聚合（四桶 + bull_bear，记录调用）。"""
    llm = types.ModuleType("sentiment_llm")
    llm.score_texts_batch = lambda texts, client=None, **kw: [
        {"index": i, "label": "乐观", "score": 0.5, "method": "llm"}
        for i in range(len(texts))]
    monkeypatch.setitem(sys.modules, "sentiment_llm", llm)
    monkeypatch.setitem(sys.modules, "agent.sentiment_llm", llm)

    agg = types.ModuleType("sentiment_aggregate")
    agg.calls = {"snapshot": []}

    def aggregate_distribution(items):
        items = list(items)
        n = len(items)
        counts = {b: 0 for b in ("乐观", "悲观", "中性", "无关")}
        for it in items:
            label = it.get("sentiment")
            counts[label if label in counts else "中性"] += 1
        dist = {b: {"count": c, "pct": round(c / n * 100, 1) if n else 0.0}
                for b, c in counts.items()}
        pos_neg = counts["乐观"] + counts["悲观"]
        bull_bear = ({"乐观_pct": round(counts["乐观"] / pos_neg * 100, 1),
                      "悲观_pct": round(counts["悲观"] / pos_neg * 100, 1)}
                     if pos_neg else
                     {"乐观_pct": None, "悲观_pct": None,
                      "note": "样本中无明确多空观点"})
        return {"n": n, "dist": dist,
                "weighted_dist": {b: 0.0 for b in counts},
                "bull_bear": bull_bear,
                "confidence": {"level": "低", "reason": f"n={n}"},
                "method": "fake"}

    agg.aggregate_distribution = aggregate_distribution
    agg.pick_representatives = lambda items, per_bucket=2: {}
    agg.save_snapshot = lambda snapshot, db_path=None: \
        agg.calls["snapshot"].append(snapshot)
    agg.get_trend = lambda platform, target, days=7, db_path=None: None
    monkeypatch.setitem(sys.modules, "sentiment_aggregate", agg)
    monkeypatch.setitem(sys.modules, "agent.sentiment_aggregate", agg)
    return agg


@pytest.fixture
def recording_collectors(monkeypatch):
    """记录型假采集函数（接受全部新kwargs），返回可控样本。"""
    calls = {"guba": [], "keyword": []}
    samples = {"guba_posts": [], "keyword_comments": []}

    def collect_guba_samples(code, post_limit=80, enrich=0, sleep=None,
                             since_days=7, **kw):
        calls["guba"].append({"code": code, "post_limit": post_limit,
                              "since_days": since_days})
        return {"code": code, "posts": samples["guba_posts"], "notes": []}

    def collect_keyword_samples(keyword, video_limit=5,
                                comments_per_video=30, sleep=None,
                                since_days=7, **kw):
        calls["keyword"].append({
            "keyword": keyword, "video_limit": video_limit,
            "comments_per_video": comments_per_video,
            "since_days": since_days})
        return {"keyword": keyword, "videos_used": 1,
                "comments": samples["keyword_comments"], "notes": []}

    monkeypatch.setattr(sm, "collect_guba_samples",
                        collect_guba_samples, raising=False)
    monkeypatch.setattr(sm, "collect_keyword_samples",
                        collect_keyword_samples, raising=False)
    return calls, samples


def _guba_post(pid="p1", published_at="2026-08-01T00:00:00+00:00"):
    return {"platform": "guba", "post_id": pid, "title": "看好", "content": "",
            "author": "", "metrics": {}, "url": "",
            "published_at": published_at, "source": "guba_list"}


def _bili_comment(cid="c1", published_at="2026-08-12T00:00:00+00:00"):
    return {"platform": "bilibili", "post_id": "v1", "author": "网友",
            "content": "看好", "likes": 1, "published_at": published_at}


class TestDepthTiers:
    def test_standard_defaults_unchanged(self, fake_llm_agg,
                                         recording_collectors):
        """standard 缺省：post_limit=80 / video_limit=5 /
        comments_per_video=comment_limit(120) / since_days=7。"""
        calls, samples = recording_collectors
        samples["guba_posts"] = [_guba_post()]
        samples["keyword_comments"] = [_bili_comment()]
        sm.get_sentiment_distribution(code="600519", keyword="茅台")
        assert calls["guba"][-1]["post_limit"] == 80
        kw_call = calls["keyword"][-1]
        assert kw_call["video_limit"] == 5
        assert kw_call["comments_per_video"] == 120
        assert kw_call["since_days"] == 7
        assert calls["guba"][-1]["since_days"] == 7

    def test_deep_tier_expands_limits(self, fake_llm_agg,
                                      recording_collectors):
        """deep 档：post_limit=300 / video_limit=15 / comments_per_video=50。"""
        calls, samples = recording_collectors
        samples["guba_posts"] = [_guba_post()]
        samples["keyword_comments"] = [_bili_comment()]
        sm.get_sentiment_distribution(code="600519", keyword="茅台",
                                      depth="deep")
        assert calls["guba"][-1]["post_limit"] == 300
        kw_call = calls["keyword"][-1]
        assert kw_call["video_limit"] == 15
        assert kw_call["comments_per_video"] == 50

    def test_deep_comment_total_cap_400(self, fake_llm_agg,
                                        recording_collectors):
        """deep 档评论总量上限 400：超出截断并进 notes。"""
        calls, samples = recording_collectors
        samples["keyword_comments"] = [
            _bili_comment(f"c{i}") for i in range(500)]
        result = sm.get_sentiment_distribution(keyword="茅台", depth="deep")
        assert result["samples_total"] == 400
        assert any("上限" in n for n in result["notes"])

    def test_invalid_depth_degrades_to_standard(self, fake_llm_agg,
                                                recording_collectors):
        """非法 depth 按 standard 处理并进 notes，绝不抛。"""
        calls, samples = recording_collectors
        samples["guba_posts"] = [_guba_post()]
        result = sm.get_sentiment_distribution(code="600519", depth="turbo")
        assert result["samples_total"] == 1
        assert calls["guba"][-1]["post_limit"] == 80
        assert any("深度档" in n for n in result["notes"])

    def test_since_days_passed_through(self, fake_llm_agg,
                                       recording_collectors):
        """since_days 透传两个采集路径。"""
        calls, samples = recording_collectors
        samples["guba_posts"] = [_guba_post()]
        samples["keyword_comments"] = [_bili_comment()]
        sm.get_sentiment_distribution(code="600519", keyword="茅台",
                                      since_days=3)
        assert calls["guba"][-1]["since_days"] == 3
        assert calls["keyword"][-1]["since_days"] == 3


class TestWindowOutput:
    def test_window_from_actual_samples(self, fake_llm_agg,
                                        recording_collectors):
        """window 从实际样本 published_at 计算 from/to。"""
        calls, samples = recording_collectors
        samples["guba_posts"] = [_guba_post("p1", "2026-08-01T08:00:00+00:00"),
                                 _guba_post("p2", "2026-08-05T09:00:00+00:00")]
        samples["keyword_comments"] = [
            _bili_comment("c1", "2026-08-12T10:00:00+00:00")]
        result = sm.get_sentiment_distribution(code="600519", keyword="茅台")
        assert result["window"] == {"from": "2026-08-01T08:00:00+00:00",
                                    "to": "2026-08-12T10:00:00+00:00"}

    def test_window_none_when_no_time_info(self, fake_llm_agg,
                                           recording_collectors):
        """样本全无时间信息：window 双 None 并进 notes。"""
        calls, samples = recording_collectors
        samples["guba_posts"] = [_guba_post("p1", "")]
        result = sm.get_sentiment_distribution(code="600519")
        assert result["window"] == {"from": None, "to": None}
        assert any("window" in n or "时间窗" in n for n in result["notes"])

    def test_window_none_on_empty_skeleton(self):
        """无样本空骨架：window 双 None，绝不抛。"""
        result = sm.get_sentiment_distribution()
        assert result["window"] == {"from": None, "to": None}


class TestBullBearPassthrough:
    def test_bull_bear_from_aggregate(self, fake_llm_agg,
                                      recording_collectors):
        """单平台路径 bull_bear 透传聚合层结果。"""
        calls, samples = recording_collectors
        samples["guba_posts"] = [_guba_post()]
        result = sm.get_sentiment_distribution(code="600519")
        assert result["bull_bear"] == {"乐观_pct": 100.0, "悲观_pct": 0.0}

    def test_bull_bear_none_variant_on_empty(self):
        result = sm.get_sentiment_distribution()
        bb = result["bull_bear"]
        assert bb["乐观_pct"] is None and bb["悲观_pct"] is None
        assert bb["note"] == "样本中无明确多空观点"

    def test_merged_bull_bear_recomputed(self, monkeypatch, fake_llm_agg,
                                         recording_collectors):
        """合并路径 bull_bear 按合并计数重算（乐观+悲观子集相对占比）。"""
        calls, samples = recording_collectors
        samples["guba_posts"] = [_guba_post()]
        samples["keyword_comments"] = [_bili_comment()]
        # 假 LLM 全乐观 → 合并后仍全乐观
        result = sm.get_sentiment_distribution(code="600519", keyword="茅台")
        assert result["bull_bear"] == {"乐观_pct": 100.0, "悲观_pct": 0.0}


# ════════════════════════════════════════════════════════════════
# 4. tools 层：depth schema 与透传
# ════════════════════════════════════════════════════════════════

class TestToolsDepth:
    def test_schema_depth_enum_both_tools(self):
        by_name = {t["function"]["name"]: t["function"]
                   for t in tools_mod.TOOL_REGISTRY}
        for name in ("get_stock_sentiment", "search_social_media"):
            props = by_name[name]["parameters"]["properties"]
            depth = props.get("depth")
            assert depth is not None, f"{name} 缺少 depth 参数"
            assert depth["type"] == "string"
            assert depth["enum"] == ["standard", "deep"]
            assert depth.get("description", "").strip()
        # depth 均为可选
        assert "depth" not in by_name["get_stock_sentiment"]["parameters"]["required"]
        assert by_name["search_social_media"]["parameters"]["required"] == ["keyword"]

    def _fake_social(self, monkeypatch):
        mod = types.ModuleType("social_media")
        mod.calls = []

        def get_sentiment_distribution(code=None, keyword=None, **kw):
            mod.calls.append({"code": code, "keyword": keyword, **kw})
            return {"target": {}, "samples_total": 1, "dist": {},
                    "weighted_dist": {}, "bull_bear": None,
                    "window": {"from": None, "to": None},
                    "confidence": {"level": "低", "reason": ""},
                    "trend": None, "representatives": [], "method": "llm",
                    "sources": [], "notes": []}

        mod.get_sentiment_distribution = get_sentiment_distribution
        monkeypatch.setattr(tools_mod, "_get_social_media_module", lambda: mod)
        return mod

    def _fake_sentiment(self, monkeypatch):
        mod = types.ModuleType("sentiment")
        mod.get_stock_sentiment = lambda code, days=30, news_items=None: {
            "code": code, "hot_rank": {"latest": 1},
            "news_sentiment": {"利好": 0, "利空": 0, "中性": 0},
            "sources": [], "notes": []}
        monkeypatch.setattr(tools_mod, "_get_sentiment_module", lambda: mod)
        monkeypatch.setattr(tools_mod, "_get_data_fetcher", lambda: None)

    def test_get_stock_sentiment_depth_passthrough(self, monkeypatch):
        self._fake_sentiment(monkeypatch)
        social = self._fake_social(monkeypatch)
        result = tools_mod.execute_tool(
            "get_stock_sentiment", {"stock_code": "600519", "depth": "deep"})
        assert result["ok"] is True
        assert social.calls[-1]["depth"] == "deep"

    def test_get_stock_sentiment_depth_default_standard(self, monkeypatch):
        self._fake_sentiment(monkeypatch)
        social = self._fake_social(monkeypatch)
        tools_mod.execute_tool("get_stock_sentiment", {"stock_code": "600519"})
        assert social.calls[-1]["depth"] == "standard"

    def test_search_social_media_depth_passthrough(self, monkeypatch):
        social = self._fake_social(monkeypatch)
        social.UNSUPPORTED_PLATFORMS = {}
        social.search_all = lambda keyword, platforms=None, limit=10, **kw: {
            "keyword": keyword, "date": "2026-08-12",
            "platforms": {"bilibili": 1},
            "posts": [{"platform": "bilibili", "post_id": "", "title": "视频",
                       "content": "", "metrics": {}, "url": "",
                       "published_at": "", "source": "bilibili_search"}],
            "sources": {"bilibili": "direct"}, "notes": []}
        social.aggregate_buzz = lambda items, scorer=None: {
            "total": 0, "sentiment": {"利好": 0, "利空": 0, "中性": 0},
            "by_platform": {}, "avg_score": 0.0}
        social.extract_stock_mentions = lambda posts, **kw: {}
        result = tools_mod.execute_tool(
            "search_social_media",
            {"keyword": "茅台", "with_comments": True, "depth": "deep"})
        assert result["ok"] is True
        dist_calls = [c for c in social.calls if c["keyword"] == "茅台"]
        assert dist_calls and dist_calls[-1]["depth"] == "deep"
