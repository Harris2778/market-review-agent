"""tests/test_social_media.py — 社媒门面测试（假模块注入 sys.modules，零网络）。

覆盖：惰性加载/能力探测、热榜合并去重、聚合器降级、缺席平台中文 note、
use_store best-effort 落盘、搜索能力分发、股票关联三类匹配与价格语境
误报控制、aggregate_buzz 注入 scorer 聚合。

零网络保证：autouse fixture 把五个社媒模块名（裸名 + agent. 前缀）在
sys.modules 中置 None（import 必失败），再按需 inject 假模块；兄弟
Worker 已交付的真实平台模块在本文件中绝不被 import/触网。
"""

import sys
import types

import pytest

from agent import social_media, social_store

ALL_MODULE_NAMES = ("social_weibo", "social_douyin", "social_bilibili",
                    "social_zhihu", "social_aggregator")


@pytest.fixture(autouse=True)
def block_real_social_modules(monkeypatch):
    """封锁真实社媒模块：sys.modules 置 None 使 import 必失败（零网络）。"""
    for name in ALL_MODULE_NAMES:
        monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.setitem(sys.modules, f"agent.{name}", None)


def inject(monkeypatch, name, **funcs):
    """构造假平台模块并以两种命名注入 sys.modules（后 set 覆盖封锁）。"""
    mod = types.ModuleType(name)
    for k, v in funcs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setitem(sys.modules, f"agent.{name}", mod)
    return mod


def make_post(platform="weibo", post_id="p1", title="标题", content="",
              author="", metrics=None, url="https://x", published_at=""):
    return {
        "platform": platform,
        "post_id": post_id,
        "title": title,
        "content": content,
        "author": author,
        "metrics": metrics or {},
        "url": url,
        "published_at": published_at,
        "source": f"{platform}_hot",
    }


@pytest.fixture(autouse=True)
def no_store_writes(monkeypatch):
    """默认拦截真实落盘；需要断言落盘的用例直接消费返回的 calls 列表。"""
    calls = []
    monkeypatch.setattr(social_store, "upsert_posts",
                        lambda posts, db_path=None: calls.append(posts) or len(posts))
    return calls


# ── 平台常数与惰性加载 ──


class TestModuleConstants:
    def test_platform_modules_four_platforms(self):
        assert set(social_media.PLATFORM_MODULES) == {
            "weibo", "douyin", "bilibili", "zhihu"}

    def test_aggregator_module_named(self):
        assert social_media.AGGREGATOR_MODULE == "social_aggregator"

    def test_xiaohongshu_marked_unsupported(self):
        assert "xiaohongshu" in social_media.UNSUPPORTED_PLATFORMS

    def test_load_module_from_sys_modules(self, monkeypatch):
        fake = inject(monkeypatch, "social_weibo")
        assert social_media._load_module("social_weibo") is fake

    def test_load_module_missing_returns_none(self):
        # 真实模块已被 autouse fixture 封锁 → 必 None
        assert social_media._load_module("social_weibo") is None

    def test_get_capability_missing_func(self, monkeypatch):
        mod = inject(monkeypatch, "social_weibo",
                     fetch_hot=lambda limit=20, sleep=None: [])
        assert social_media._get_capability(mod, "search") is None
        assert social_media._get_capability(None, "fetch_hot") is None
        assert callable(social_media._get_capability(mod, "fetch_hot"))


# ── get_hot_all ──


class TestGetHotAll:
    def test_merges_multiple_platforms(self, monkeypatch):
        inject(monkeypatch, "social_weibo",
               fetch_hot=lambda limit=20, sleep=None:
                   [make_post("weibo", "w1"), make_post("weibo", "w2")])
        inject(monkeypatch, "social_zhihu",
               fetch_hot=lambda limit=20, sleep=None: [make_post("zhihu", "z1")])
        out = social_media.get_hot_all(platforms=["weibo", "zhihu"])
        assert out["platforms"] == {"weibo": 2, "zhihu": 1}
        assert len(out["posts"]) == 3
        assert out["sources"] == {"weibo": "direct", "zhihu": "direct"}
        assert out["date"]

    def test_dedup_same_platform_post_id(self, monkeypatch):
        dup = make_post("weibo", "w1", title="同一条")
        inject(monkeypatch, "social_weibo",
               fetch_hot=lambda limit=20, sleep=None: [dup, dict(dup)])
        out = social_media.get_hot_all(platforms=["weibo"])
        assert len(out["posts"]) == 1

    def test_same_post_id_different_platforms_kept(self, monkeypatch):
        inject(monkeypatch, "social_weibo",
               fetch_hot=lambda limit=20, sleep=None: [make_post("weibo", "1")])
        inject(monkeypatch, "social_douyin",
               fetch_hot=lambda limit=20, sleep=None: [make_post("douyin", "1")])
        out = social_media.get_hot_all(platforms=["weibo", "douyin"])
        assert len(out["posts"]) == 2

    def test_fallback_to_aggregator_when_direct_empty(self, monkeypatch):
        inject(monkeypatch, "social_weibo",
               fetch_hot=lambda limit=20, sleep=None: [])
        seen = {}

        def agg_fetch(platform, limit=20, sleep=None):
            seen["platform"] = platform
            return [make_post("", "a1", title="聚合兜底")]  # platform 缺省

        inject(monkeypatch, "social_aggregator", fetch_hot=agg_fetch)
        out = social_media.get_hot_all(platforms=["weibo"])
        assert out["sources"]["weibo"] == "aggregator"
        assert out["posts"][0]["platform"] == "weibo"  # 门面补齐 platform
        assert seen["platform"] == "weibo"

    def test_fallback_when_fetch_raises(self, monkeypatch):
        def boom(limit=20, sleep=None):
            raise RuntimeError("network down")

        inject(monkeypatch, "social_weibo", fetch_hot=boom)
        inject(monkeypatch, "social_aggregator",
               fetch_hot=lambda platform, limit=20, sleep=None:
                   [make_post("weibo", "a1")])
        out = social_media.get_hot_all(platforms=["weibo"])
        assert out["sources"]["weibo"] == "aggregator"
        assert len(out["posts"]) == 1

    def test_both_empty_goes_to_notes(self, monkeypatch):
        inject(monkeypatch, "social_weibo",
               fetch_hot=lambda limit=20, sleep=None: [])
        inject(monkeypatch, "social_aggregator",
               fetch_hot=lambda platform, limit=20, sleep=None: [])
        out = social_media.get_hot_all(platforms=["weibo"])
        assert out["posts"] == []
        assert out["sources"]["weibo"] == "none"
        assert any("兜底均为空" in n for n in out["notes"])

    def test_module_absent_note_and_aggregator_attempt(self, monkeypatch):
        # social_weibo 模块被封锁（未交付）→ 直连跳过并尝试聚合器
        inject(monkeypatch, "social_aggregator",
               fetch_hot=lambda platform, limit=20, sleep=None:
                   [make_post("weibo", "a1")])
        out = social_media.get_hot_all(platforms=["weibo"])
        assert out["sources"]["weibo"] == "aggregator"
        assert any("未就绪" in n for n in out["notes"])

    def test_xiaohongshu_chinese_note(self):
        out = social_media.get_hot_all(platforms=["xiaohongshu"])
        assert out["posts"] == []
        assert any("小红书" in n and "暂不支持" in n for n in out["notes"])

    def test_unknown_platform_note(self):
        out = social_media.get_hot_all(platforms=["tiktok"])
        assert any("未知平台" in n for n in out["notes"])

    def test_sleep_passed_through(self, monkeypatch):
        captured = {}

        def fetch(limit=20, sleep=None):
            captured["sleep"] = sleep
            return [make_post("weibo", "w1")]

        sentinel = object()
        inject(monkeypatch, "social_weibo", fetch_hot=fetch)
        social_media.get_hot_all(platforms=["weibo"], sleep=sentinel)
        assert captured["sleep"] is sentinel

    def test_limit_passed_through(self, monkeypatch):
        captured = {}
        inject(monkeypatch, "social_weibo",
               fetch_hot=lambda limit=20, sleep=None:
                   captured.setdefault("limit", limit) or [])
        social_media.get_hot_all(platforms=["weibo"], limit=5)
        assert captured["limit"] == 5

    def test_use_store_true_upserts(self, monkeypatch, no_store_writes):
        inject(monkeypatch, "social_weibo",
               fetch_hot=lambda limit=20, sleep=None: [make_post("weibo", "w1")])
        social_media.get_hot_all(platforms=["weibo"], use_store=True)
        assert len(no_store_writes) == 1

    def test_use_store_false_skips_upsert(self, monkeypatch, no_store_writes):
        inject(monkeypatch, "social_weibo",
               fetch_hot=lambda limit=20, sleep=None: [make_post("weibo", "w1")])
        social_media.get_hot_all(platforms=["weibo"], use_store=False)
        assert no_store_writes == []

    def test_default_platforms_all_four(self):
        out = social_media.get_hot_all()
        # 平台模块全部被封锁（未交付态）→ 缺席但不抛，notes 非空
        assert out["posts"] == []
        assert out["notes"]

    def test_never_raises_on_garbage_platforms(self):
        out = social_media.get_hot_all(platforms=[None, 123, ""])
        assert isinstance(out, dict)


# ── search_all ──


class TestSearchAll:
    def test_search_dispatches_only_capable_platforms(self, monkeypatch):
        inject(monkeypatch, "social_bilibili",
               fetch_hot=lambda limit=20, sleep=None: [],
               search=lambda keyword, limit=20, sleep=None:
                   [make_post("bilibili", "b1", title=f"视频:{keyword}")])
        inject(monkeypatch, "social_weibo",
               fetch_hot=lambda limit=20, sleep=None: [])
        out = social_media.search_all("茅台", platforms=["bilibili", "weibo"])
        assert out["platforms"] == {"bilibili": 1}
        assert out["posts"][0]["title"] == "视频:茅台"
        assert any("weibo" in n and "搜索暂不支持" in n for n in out["notes"])

    def test_search_keyword_forwarded(self, monkeypatch):
        seen = {}
        inject(monkeypatch, "social_bilibili",
               search=lambda keyword, limit=20, sleep=None:
                   seen.setdefault("kw", keyword) or [])
        social_media.search_all("寒武纪", platforms=["bilibili"])
        assert seen["kw"] == "寒武纪"

    def test_search_empty_keyword_short_circuits(self):
        out = social_media.search_all("")
        assert out["posts"] == []
        assert any("关键词为空" in n for n in out["notes"])

    def test_search_dedup(self, monkeypatch):
        dup = make_post("bilibili", "b1")
        inject(monkeypatch, "social_bilibili",
               search=lambda keyword, limit=20, sleep=None: [dup, dict(dup)])
        out = social_media.search_all("x", platforms=["bilibili"])
        assert len(out["posts"]) == 1

    def test_search_xiaohongshu_note(self):
        out = social_media.search_all("茅台", platforms=["xiaohongshu"])
        assert any("小红书" in n for n in out["notes"])

    def test_search_exception_degrades_to_note(self, monkeypatch):
        def boom(keyword, limit=20, sleep=None):
            raise ValueError("412")

        inject(monkeypatch, "social_bilibili", search=boom)
        out = social_media.search_all("茅台", platforms=["bilibili"])
        assert out["posts"] == []
        assert out["sources"]["bilibili"] == "none"
        assert any("无结果" in n or "降级" in n for n in out["notes"])


# ── extract_stock_mentions ──


class TestExtractStockMentions:
    def test_code_regex_basic(self):
        out = social_media.extract_stock_mentions(
            [make_post(title="600519 创历史新高")], watchlist=[])
        assert out["600519"]["count"] == 1

    def test_code_regex_all_prefixes(self):
        codes = ["600519", "688001", "000001", "300750", "200011",
                 "830799", "430047"]
        posts = [make_post(post_id=c, title=f"{c} 异动") for c in codes]
        out = social_media.extract_stock_mentions(posts, watchlist=[])
        for c in codes:
            assert c in out, f"{c} 未识别"

    def test_code_regex_rejects_non_market_prefix(self):
        out = social_media.extract_stock_mentions(
            [make_post(title="尾号 123456 中奖号码 666666")], watchlist=[])
        assert "123456" not in out
        assert "666666" not in out

    def test_price_context_excluded(self):
        for text in ["600519元", "600519 元", "600519块",
                     "股价 300750 元/股", "募资 830799 万元"]:
            out = social_media.extract_stock_mentions(
                [make_post(title=text)], watchlist=[])
            assert not out, f"{text!r} 被误判为股票代码"

    def test_price_context_does_not_hurt_name_match(self):
        """「600519 元价格」语境：代码不计，但名称匹配不受影响。"""
        wl = [{"code": "sh600519", "name": "贵州茅台", "market": "cn"}]
        out = social_media.extract_stock_mentions(
            [make_post(title="600519 元价格的贵州茅台还能拿吗")],
            watchlist=wl)
        # 名称命中 → 归并到代码键；该键唯一来源是名称匹配
        assert out["600519"]["count"] == 1

    def test_watchlist_name_and_code_merge(self):
        wl = [{"code": "sz300750", "name": "宁德时代", "market": "cn"}]
        posts = [make_post(post_id="1", title="宁德时代发布新电池"),
                 make_post(post_id="2", title="300750 放量上涨")]
        out = social_media.extract_stock_mentions(posts, watchlist=wl)
        assert out["300750"]["count"] == 2

    def test_watchlist_lazy_load(self, monkeypatch):
        monkeypatch.setattr("agent.watchlist.list_stocks",
                            lambda: [{"code": "sh600519", "name": "贵州茅台",
                                      "market": "cn"}])
        out = social_media.extract_stock_mentions(
            [make_post(title="贵州茅台提价")])
        assert out["600519"]["count"] == 1

    def test_watchlist_load_failure_skips(self, monkeypatch):
        def boom():
            raise RuntimeError("io error")

        monkeypatch.setattr("agent.watchlist.list_stocks", boom)
        out = social_media.extract_stock_mentions(
            [make_post(title="贵州茅台提价")])
        assert out == {}

    def test_extra_names(self):
        out = social_media.extract_stock_mentions(
            [make_post(title="寒武纪人气爆棚")],
            watchlist=[], extra_names=["寒武纪"])
        assert out["寒武纪"]["count"] == 1

    def test_sample_titles_capped_at_three(self):
        posts = [make_post(post_id=str(i), title=f"600519 话题 {i}")
                 for i in range(5)]
        out = social_media.extract_stock_mentions(posts, watchlist=[])
        assert out["600519"]["count"] == 5
        assert len(out["600519"]["sample_titles"]) == 3

    def test_same_post_counts_once(self):
        out = social_media.extract_stock_mentions(
            [make_post(title="600519 又见 600519", content="600519")],
            watchlist=[])
        assert out["600519"]["count"] == 1

    def test_matches_content_field(self):
        out = social_media.extract_stock_mentions(
            [make_post(title="无关", content="正文提到 000001")], watchlist=[])
        assert out["000001"]["count"] == 1

    def test_empty_and_garbage_input(self):
        assert social_media.extract_stock_mentions([], watchlist=[]) == {}
        assert social_media.extract_stock_mentions(
            [None, "x", 42], watchlist=[]) == {}


# ── aggregate_buzz ──


class TestAggregateBuzz:
    def test_injected_scorer_distribution(self):
        posts = [make_post("weibo", "w1", title="利好贴"),
                 make_post("weibo", "w2", title="利空贴"),
                 make_post("zhihu", "z1", title="中性贴")]

        def scorer(item):
            if "利好" in item["title"]:
                return {"sentiment_score": 0.8}
            if "利空" in item["title"]:
                return {"sentiment_score": -0.6}
            return {"sentiment_score": 0.0}

        out = social_media.aggregate_buzz(posts, scorer=scorer)
        assert out["total"] == 3
        assert out["sentiment"] == {"利好": 1, "利空": 1, "中性": 1}
        assert out["by_platform"]["weibo"]["total"] == 2
        assert out["by_platform"]["zhihu"]["sentiment"]["中性"] == 1
        assert out["avg_score"] == round((0.8 - 0.6 + 0.0) / 3, 4)

    def test_scorer_exception_degrades_to_neutral(self):
        def boom(item):
            raise RuntimeError("llm down")

        out = social_media.aggregate_buzz([make_post(title="任意")],
                                          scorer=boom)
        assert out["total"] == 1
        assert out["sentiment"]["中性"] == 1

    def test_empty_posts(self):
        out = social_media.aggregate_buzz([])
        assert out == {"total": 0,
                       "sentiment": {"利好": 0, "利空": 0, "中性": 0},
                       "by_platform": {},
                       "avg_score": 0.0}

    def test_default_dict_scorer(self):
        """不注入 scorer 时走 sentiment 内置词典（'涨停' 权重 1.2 → 利好）。"""
        out = social_media.aggregate_buzz(
            [make_post(title="全板块涨停潮")])
        assert out["sentiment"]["利好"] == 1
        assert out["avg_score"] > 0

    def test_post_converted_to_title_content_shape(self):
        seen = {}

        def scorer(item):
            seen.update(item)
            return {"sentiment_score": 0.5}

        social_media.aggregate_buzz(
            [make_post(title="标题A", content="正文B")], scorer=scorer)
        assert seen["title"] == "标题A"
        assert seen["content"] == "正文B"
        # 只透传 title/content，不携带 platform 等杂项
        assert set(seen) <= {"title", "content"}
