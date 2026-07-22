"""tests/test_sentiment_distribution.py — 舆情「分布化」改造接线测试（新增 39）。

覆盖：
1. get_sentiment_distribution 主流程（假采集/假 LLM/假聚合全链）：code 路径、
   keyword 路径、code+keyword 合并路径（平台分别统计后归并）、快照落盘与
   趋势挂载、LLM 标签回填、打分文本截 100 字。
2. 打分双路径与 method 标注：llm / lexicon / mixed（合并路径双平台打分方式
   不一致）；LLM 模块缺席/能力缺失/异常/结构异常一律降级词典 + notes。
3. 降级路径绝不抛：采集缺席/异常/为空、聚合缺席/异常、代表样本/快照/趋势
   失败、主流程内部异常——全部降级进 notes。
4. 两工具 distribution 块：get_stock_sentiment 成功挂 sentiment_distribution
   键（code + post_limit=80），search_social_media with_comments=true 时挂
   （keyword 路径）；失败/能力缺失只进 notes，绝不影响主返回。
5. system_prompts「社媒舆情引用规范」新增第 7-10 条断言与既有条款保留。

零网络保证：假 sentiment_llm / sentiment_aggregate 模块注入 sys.modules
（裸名 + agent. 前缀），采集函数 monkeypatch 到真实 social_media 模块上；
工具用例把假 social_media / sentiment 模块注入工具层惰性解析点。
"""

import sys
import types

import pytest

import agent.social_media as sm
import agent.tools as tools_mod
from agent import system_prompts

# ════════════════════════════════════════════════════════════════
# 公共工具：假数据与假模块注入
# ════════════════════════════════════════════════════════════════

_LABEL_MAP = {"利好": "乐观", "利空": "悲观"}


def _guba_posts(n=3):
    return [{
        "platform": "guba",
        "post_id": f"p{i}",
        "title": f"茅台业绩真好创新高{i}" if i % 2 == 0 else f"恐慌大跌崩盘{i}",
        "content": "正文内容",
        "metrics": {"likes": i},
        "url": "https://guba.eastmoney.com/x",
        "published_at": "2026-08-12",
        "source": "贵州茅台吧",
    } for i in range(n)]


def _bili_comments(n=3):
    return [{
        "platform": "bilibili",
        "post_id": "av1001",
        "author": f"网友{i}",
        "content": f"看好后市大涨{i}" if i % 2 == 0 else f"崩盘暴跌预警{i}",
        "likes": i,
        "published_at": "2026-08-12",
    } for i in range(n)]


@pytest.fixture
def fake_collectors(monkeypatch):
    """假采集层：monkeypatch 到真实 social_media 模块（采集扩容 Worker 的函数位）。"""
    calls = {"guba": [], "keyword": []}

    def collect_guba_samples(code, post_limit=80, enrich=3, sleep=None):
        calls["guba"].append({"code": code, "post_limit": post_limit})
        return {"code": code, "posts": _guba_posts(), "notes": ["股吧采样内部note"]}

    def collect_keyword_samples(keyword, video_limit=5,
                                comments_per_video=24, sleep=None):
        calls["keyword"].append({"keyword": keyword})
        return {"keyword": keyword, "videos_used": 2,
                "comments": _bili_comments(), "notes": []}

    monkeypatch.setattr(sm, "collect_guba_samples",
                        collect_guba_samples, raising=False)
    monkeypatch.setattr(sm, "collect_keyword_samples",
                        collect_keyword_samples, raising=False)
    return calls


@pytest.fixture
def fake_llm(monkeypatch):
    """假 LLM 打分模块：按文本内容给 乐观/中性/悲观 标签。"""
    mod = types.ModuleType("sentiment_llm")
    mod.calls = []

    def score_texts_batch(texts, client=None, model=None, batch_size=20,
                          sleep=None):
        mod.calls.append(list(texts))
        out = []
        for i, t in enumerate(texts):
            if "好" in t or "看好" in t:
                label, score = "乐观", 0.6
            elif "跌" in t or "崩盘" in t:
                label, score = "悲观", -0.6
            else:
                label, score = "中性", 0.0
            out.append({"index": i, "label": label, "score": score,
                        "method": "llm"})
        return out

    mod.score_texts_batch = score_texts_batch
    monkeypatch.setitem(sys.modules, "sentiment_llm", mod)
    monkeypatch.setitem(sys.modules, "agent.sentiment_llm", mod)
    return mod


@pytest.fixture
def fake_agg(monkeypatch):
    """假聚合模块：aggregate_distribution / pick_representatives /
    save_snapshot / get_trend 四能力齐备，calls 记录调用。"""
    mod = types.ModuleType("sentiment_aggregate")
    mod.calls = {"aggregate": [], "reps": [], "snapshot": [], "trend": []}

    def aggregate_distribution(items):
        items = list(items)
        mod.calls["aggregate"].append(items)
        n = len(items)
        counts = {b: 0 for b in ("乐观", "中性", "悲观")}
        for it in items:
            label = it.get("sentiment")
            bucket = label if label in counts else _LABEL_MAP.get(label, "中性")
            counts[bucket] += 1
        dist = {b: {"count": c, "pct": round(c / n * 100, 1) if n else 0.0}
                for b, c in counts.items()}
        wsum = {b: 0.0 for b in counts}
        wtot = 0.0
        for it in items:
            metrics = it.get("metrics") if isinstance(it.get("metrics"), dict) else {}
            likes = metrics.get("likes", it.get("likes", 0)) or 0
            w = likes + 1
            label = it.get("sentiment")
            bucket = label if label in counts else _LABEL_MAP.get(label, "中性")
            wsum[bucket] += w
            wtot += w
        weighted = {b: round(wsum[b] / wtot * 100, 1) if wtot else 0.0
                    for b in counts}
        level = "低" if n < 30 else ("中" if n < 100 else "高")
        return {"n": n, "dist": dist, "weighted_dist": weighted,
                "confidence": {"level": level, "reason": f"n={n}"},
                "method": "fake"}

    def pick_representatives(items, per_bucket=2):
        mod.calls["reps"].append(per_bucket)
        return {"乐观": list(items)[:per_bucket]}

    def save_snapshot(snapshot, db_path=None):
        mod.calls["snapshot"].append(snapshot)

    def get_trend(platform, target, days=7, db_path=None):
        mod.calls["trend"].append({"platform": platform, "target": target,
                                   "days": days})
        return {"platform": platform, "target": target,
                "points": [{"date": "2026-08-12", "乐观": 60.0}]}

    mod.aggregate_distribution = aggregate_distribution
    mod.pick_representatives = pick_representatives
    mod.save_snapshot = save_snapshot
    mod.get_trend = get_trend
    monkeypatch.setitem(sys.modules, "sentiment_aggregate", mod)
    monkeypatch.setitem(sys.modules, "agent.sentiment_aggregate", mod)
    return mod


@pytest.fixture
def block_llm_module(monkeypatch):
    """确定性模拟 sentiment_llm 模块缺席（无论兄弟 Worker 是否已交付）。"""
    real_load = sm._load_module
    monkeypatch.setattr(
        sm, "_load_module",
        lambda name: None if name == sm.SENTIMENT_LLM_MODULE else real_load(name))


@pytest.fixture
def block_agg_module(monkeypatch):
    """确定性模拟 sentiment_aggregate 模块缺席。"""
    real_load = sm._load_module
    monkeypatch.setattr(
        sm, "_load_module",
        lambda name: None if name == sm.SENTIMENT_AGG_MODULE else real_load(name))


# ════════════════════════════════════════════════════════════════
# 1. 分布主流程（假模块全链）
# ════════════════════════════════════════════════════════════════

class TestDistributionMainFlow:

    def test_code_path_full_chain(self, fake_collectors, fake_llm, fake_agg):
        result = sm.get_sentiment_distribution(code="600519")
        assert result["target"] == {"code": "600519"}
        assert result["samples_total"] == 3
        assert set(result["dist"]) == {"乐观", "中性", "悲观"}
        total = sum(b["count"] for b in result["dist"].values())
        assert total == 3
        assert result["method"] == "llm"
        assert result["sources"] == ["eastmoney_guba"]
        assert result["confidence"]["level"] == "低"  # n=3 < 30
        assert result["trend"]["platform"] == "guba"
        assert result["representatives"], "代表性样本应挂载"
        # 采集入参：post_limit 默认 80
        assert fake_collectors["guba"][-1]["post_limit"] == 80

    def test_keyword_path_full_chain(self, fake_collectors, fake_llm, fake_agg):
        result = sm.get_sentiment_distribution(keyword="贵州茅台")
        assert result["target"] == {"keyword": "贵州茅台"}
        assert result["samples_total"] == 3
        assert result["method"] == "llm"
        assert result["sources"] == ["bilibili"]
        assert result["trend"]["platform"] == "bilibili"
        assert fake_collectors["keyword"][-1]["keyword"] == "贵州茅台"

    def test_merged_path_both_given(self, fake_collectors, fake_llm, fake_agg):
        result = sm.get_sentiment_distribution(code="600519", keyword="茅台")
        assert result["target"] == {"code": "600519", "keyword": "茅台"}
        assert result["samples_total"] == 6  # 平台分别统计后归并
        assert sum(b["count"] for b in result["dist"].values()) == 6
        assert result["method"] == "llm"  # 两平台同为 LLM 路径
        assert set(result["sources"]) == {"eastmoney_guba", "bilibili"}
        # 合并路径 trend / representatives 为 {平台: 值}
        assert set(result["trend"]) == {"guba", "bilibili"}
        assert set(result["representatives"]) == {"guba", "bilibili"}
        # 两平台各落一次快照
        assert len(fake_agg.calls["snapshot"]) == 2

    def test_snapshot_platform_and_target(self, fake_collectors, fake_llm,
                                          fake_agg):
        sm.get_sentiment_distribution(code="600519")
        snap = fake_agg.calls["snapshot"][-1]
        assert snap["platform"] == "guba"
        assert snap["target"] == "600519"
        assert snap["n"] == 3

    def test_trend_call_args(self, fake_collectors, fake_llm, fake_agg):
        sm.get_sentiment_distribution(keyword="茅台")
        call = fake_agg.calls["trend"][-1]
        assert call["platform"] == "bilibili"
        assert call["target"] == "茅台"
        assert call["days"] == 7

    def test_llm_backfills_labels_and_scores(self, fake_collectors, fake_llm,
                                             fake_agg):
        sm.get_sentiment_distribution(code="600519")
        items = fake_agg.calls["aggregate"][-1]
        labels = [it.get("sentiment") for it in items]
        assert set(labels) <= {"乐观", "中性", "悲观"}, labels
        assert "乐观" in labels and "悲观" in labels
        for it in items:
            assert -1.0 <= it.get("sentiment_score") <= 1.0

    def test_text_truncated_to_100(self, monkeypatch, fake_llm, fake_agg):
        def collect_guba_samples(code, post_limit=80, **kw):
            return {"code": code, "posts": [
                {"platform": "guba", "title": "好" * 200, "content": "正文",
                 "metrics": {}, "url": "", "published_at": "", "source": ""}],
                "notes": []}

        monkeypatch.setattr(sm, "collect_guba_samples",
                            collect_guba_samples, raising=False)
        sm.get_sentiment_distribution(code="600519")
        texts = fake_llm.calls[-1]
        assert texts and all(len(t) <= 100 for t in texts), (
            f"打分文本应截 100 字: {[len(t) for t in texts]}")

    def test_llm_internal_fallback_noted(self, monkeypatch, fake_collectors,
                                         fake_agg):
        mod = types.ModuleType("sentiment_llm")

        def score_texts_batch(texts, client=None, **kw):
            return [{"index": i, "label": "中性", "score": 0.0,
                     "method": "fallback"} for i in range(len(texts))]

        mod.score_texts_batch = score_texts_batch
        monkeypatch.setitem(sys.modules, "sentiment_llm", mod)
        monkeypatch.setitem(sys.modules, "agent.sentiment_llm", mod)
        result = sm.get_sentiment_distribution(code="600519")
        assert result["method"] == "llm"  # LLM 路径已用
        assert any("fallback" in n for n in result["notes"]), result["notes"]


# ════════════════════════════════════════════════════════════════
# 2. LLM / 词典双路径与 method 标注
# ════════════════════════════════════════════════════════════════

class TestScoringPaths:

    def test_method_lexicon_when_use_llm_false(self, fake_collectors, fake_agg):
        result = sm.get_sentiment_distribution(code="600519", use_llm=False)
        assert result["method"] == "lexicon"

    def test_lexicon_when_llm_module_absent(self, fake_collectors, fake_agg,
                                            block_llm_module):
        result = sm.get_sentiment_distribution(code="600519")
        assert result["method"] == "lexicon"
        assert any("sentiment_llm 未就绪" in n for n in result["notes"]), result["notes"]

    def test_lexicon_when_llm_capability_missing(self, monkeypatch,
                                                 fake_collectors, fake_agg):
        mod = types.ModuleType("sentiment_llm")  # 无 score_texts_batch
        monkeypatch.setitem(sys.modules, "sentiment_llm", mod)
        monkeypatch.setitem(sys.modules, "agent.sentiment_llm", mod)
        result = sm.get_sentiment_distribution(code="600519")
        assert result["method"] == "lexicon"
        assert any("score_texts_batch 缺失" in n for n in result["notes"])

    def test_lexicon_when_llm_raises(self, monkeypatch, fake_collectors,
                                     fake_agg):
        mod = types.ModuleType("sentiment_llm")

        def score_texts_batch(texts, client=None, **kw):
            raise RuntimeError("LLM 网关超时")

        mod.score_texts_batch = score_texts_batch
        monkeypatch.setitem(sys.modules, "sentiment_llm", mod)
        monkeypatch.setitem(sys.modules, "agent.sentiment_llm", mod)
        result = sm.get_sentiment_distribution(code="600519")  # 绝不抛
        assert result["method"] == "lexicon"
        assert any("LLM 批量打分失败" in n for n in result["notes"])
        assert result["samples_total"] == 3, "降级词典后分布仍应产出"

    def test_lexicon_when_llm_bad_structure(self, monkeypatch, fake_collectors,
                                            fake_agg):
        mod = types.ModuleType("sentiment_llm")
        mod.score_texts_batch = lambda texts, client=None, **kw: "not-a-list"
        monkeypatch.setitem(sys.modules, "sentiment_llm", mod)
        monkeypatch.setitem(sys.modules, "agent.sentiment_llm", mod)
        result = sm.get_sentiment_distribution(code="600519")
        assert result["method"] == "lexicon"
        assert any("返回结构异常" in n for n in result["notes"])

    def test_lexicon_labels_passed_to_aggregate(self, fake_collectors,
                                                fake_agg):
        """词典标签 利好/利空/中性 原样交给聚合层（聚合层负责归并）。"""
        result = sm.get_sentiment_distribution(code="600519", use_llm=False)
        items = fake_agg.calls["aggregate"][-1]
        labels = {it.get("sentiment") for it in items}
        assert labels <= {"利好", "利空", "中性"}, labels
        assert result["samples_total"] == 3  # 聚合层归并后仍出分布

    def test_mixed_method_when_paths_differ(self, monkeypatch, fake_collectors,
                                            fake_agg):
        """合并路径：股吧 LLM 成功、关键词 LLM 异常降级词典 → method='mixed'。"""
        mod = types.ModuleType("sentiment_llm")
        state = {"n": 0}

        def score_texts_batch(texts, client=None, **kw):
            state["n"] += 1
            if state["n"] > 1:
                raise RuntimeError("第二次调用炸了")
            return [{"index": i, "label": "乐观", "score": 0.5,
                     "method": "llm"} for i in range(len(texts))]

        mod.score_texts_batch = score_texts_batch
        monkeypatch.setitem(sys.modules, "sentiment_llm", mod)
        monkeypatch.setitem(sys.modules, "agent.sentiment_llm", mod)
        result = sm.get_sentiment_distribution(code="600519", keyword="茅台")
        assert result["method"] == "mixed"
        assert result["samples_total"] == 6


# ════════════════════════════════════════════════════════════════
# 3. 各降级路径绝不抛
# ════════════════════════════════════════════════════════════════

class TestDegradation:

    def test_no_code_no_keyword(self):
        result = sm.get_sentiment_distribution()  # 绝不抛
        assert result["samples_total"] == 0
        assert result["method"] == "none"
        assert result["trend"] is None
        assert result["representatives"] == []
        assert any("缺失/非法" in n for n in result["notes"])

    def test_collectors_absent(self, monkeypatch, fake_llm, fake_agg):
        monkeypatch.delattr(sm, "collect_guba_samples", raising=False)
        result = sm.get_sentiment_distribution(code="600519")
        assert result["samples_total"] == 0
        assert any("collect_guba_samples 未就绪" in n for n in result["notes"])

    def test_collector_raises(self, monkeypatch, fake_llm, fake_agg):
        def boom(code, **kw):
            raise RuntimeError("采集层炸了")

        monkeypatch.setattr(sm, "collect_guba_samples", boom, raising=False)
        result = sm.get_sentiment_distribution(code="600519")  # 绝不抛
        assert result["samples_total"] == 0
        assert any("股吧样本采集失败" in n for n in result["notes"])

    def test_collector_empty(self, monkeypatch, fake_llm, fake_agg):
        monkeypatch.setattr(
            sm, "collect_guba_samples",
            lambda code, **kw: {"code": code, "posts": [], "notes": []},
            raising=False)
        result = sm.get_sentiment_distribution(code="600519")
        assert result["samples_total"] == 0
        assert any("无有效帖子样本" in n for n in result["notes"])

    def test_aggregate_module_absent(self, fake_collectors, fake_llm,
                                     block_agg_module):
        result = sm.get_sentiment_distribution(code="600519")  # 绝不抛
        assert result["samples_total"] == 3  # 空骨架 n 兜底为条目数
        assert all(b["count"] == 0 for b in result["dist"].values())
        assert result["representatives"] == []
        assert result["trend"] is None
        assert any("aggregate_distribution 未就绪" in n for n in result["notes"])

    def test_aggregate_raises(self, fake_collectors, fake_llm, fake_agg):
        def boom(items):
            raise RuntimeError("聚合层炸了")

        fake_agg.aggregate_distribution = boom
        result = sm.get_sentiment_distribution(code="600519")  # 绝不抛
        assert all(b["count"] == 0 for b in result["dist"].values())
        assert any("分布聚合异常" in n for n in result["notes"])

    def test_representatives_raises(self, fake_collectors, fake_llm, fake_agg):
        def boom(items, per_bucket=2):
            raise RuntimeError("代表样本炸了")

        fake_agg.pick_representatives = boom
        result = sm.get_sentiment_distribution(code="600519")  # 绝不抛
        assert result["representatives"] == []
        assert any("代表样本选取失败" in n for n in result["notes"])
        assert result["samples_total"] == 3, "分布主结果不受影响"

    def test_snapshot_raises(self, fake_collectors, fake_llm, fake_agg):
        def boom(snapshot, db_path=None):
            raise RuntimeError("SQLite 锁死")

        fake_agg.save_snapshot = boom
        result = sm.get_sentiment_distribution(code="600519")  # 绝不抛
        assert result["samples_total"] == 3
        assert result["trend"]["platform"] == "guba", "趋势不受快照失败影响"
        assert any("快照落盘失败" in n for n in result["notes"])

    def test_trend_raises(self, fake_collectors, fake_llm, fake_agg):
        def boom(platform, target, days=7, db_path=None):
            raise RuntimeError("趋势查询炸了")

        fake_agg.get_trend = boom
        result = sm.get_sentiment_distribution(code="600519")  # 绝不抛
        assert result["trend"] is None
        assert any("趋势读取失败" in n for n in result["notes"])
        assert result["samples_total"] == 3

    def test_inner_exception_never_raises(self, monkeypatch, fake_collectors):
        def boom(code):
            raise ZeroDivisionError("内部炸了")

        monkeypatch.setattr(sm, "_normalize_guba_code", boom)
        result = sm.get_sentiment_distribution(code="600519")  # 绝不抛
        assert result["samples_total"] == 0
        assert any("内部异常" in n for n in result["notes"])

    def test_merged_confidence_recomputed(self, monkeypatch, fake_llm,
                                          fake_agg):
        """合并路径置信度按合并 n 重新分档：两平台各 20 条（各自低）→ 合并 40（中）。"""
        monkeypatch.setattr(
            sm, "collect_guba_samples",
            lambda code, **kw: {"code": code, "posts": _guba_posts(20),
                                "notes": []}, raising=False)
        monkeypatch.setattr(
            sm, "collect_keyword_samples",
            lambda kw, **kw2: {"keyword": kw, "videos_used": 2,
                               "comments": _bili_comments(20), "notes": []},
            raising=False)
        result = sm.get_sentiment_distribution(code="600519", keyword="茅台")
        assert result["samples_total"] == 40
        assert result["confidence"]["level"] == "中", result["confidence"]

    def test_merged_small_sample_declares_insufficient(self, fake_collectors,
                                                       fake_llm, fake_agg):
        result = sm.get_sentiment_distribution(code="600519", keyword="茅台")
        assert result["samples_total"] == 6
        assert result["confidence"]["level"] == "低"
        assert "样本不足" in result["confidence"]["reason"]


# ════════════════════════════════════════════════════════════════
# 4. 两工具 distribution 块挂载
# ════════════════════════════════════════════════════════════════

def _sentiment_result(code="600519"):
    return {
        "code": code,
        "hot_rank": {"latest": 3, "history_avg": 12.5, "trend": "上升"},
        "news_sentiment": {"利好": 1, "利空": 0, "中性": 2},
        "sources": ["eastmoney_hotrank"],
        "notes": [],
    }


def _dist_result(target_key, value):
    return {
        "target": {target_key: value},
        "samples_total": 3,
        "dist": {"乐观": {"count": 2, "pct": 66.7},
                 "中性": {"count": 1, "pct": 33.3},
                 "悲观": {"count": 0, "pct": 0.0}},
        "weighted_dist": {"乐观": 70.0, "中性": 20.0, "悲观": 10.0},
        "confidence": {"level": "低", "reason": "样本不足（n=3 < 30）"},
        "trend": {"points": [{"date": "2026-08-12", "乐观": 60.0}]},
        "representatives": {"乐观": []},
        "method": "llm",
        "sources": ["eastmoney_guba"],
        "notes": ["分布内部note"],
    }


@pytest.fixture
def fake_sentiment_tool(monkeypatch):
    """假情绪模块 + 数据层缺席（新闻注入降级，零网络）。"""
    mod = types.ModuleType("sentiment")
    mod.get_stock_sentiment = (
        lambda code, days=30, news_items=None: _sentiment_result(code))
    monkeypatch.setattr(tools_mod, "_get_sentiment_module", lambda: mod)
    monkeypatch.setattr(tools_mod, "_get_data_fetcher", lambda: None)
    return mod


@pytest.fixture
def fake_social_dist(monkeypatch):
    """假社媒门面：仅 get_sentiment_distribution 能力（记录调用）。"""
    mod = types.ModuleType("social_media")
    mod.calls = []

    def get_sentiment_distribution(code=None, keyword=None, post_limit=80,
                                   **kw):
        mod.calls.append({"code": code, "keyword": keyword,
                          "post_limit": post_limit})
        if code:
            return _dist_result("code", code)
        return _dist_result("keyword", keyword)

    mod.get_sentiment_distribution = get_sentiment_distribution
    monkeypatch.setattr(tools_mod, "_get_social_media_module", lambda: mod)
    return mod


class TestStockSentimentDistributionBlock:

    def test_distribution_attached(self, fake_sentiment_tool, fake_social_dist):
        result = tools_mod.execute_tool("get_stock_sentiment",
                                        {"stock_code": "600519"})
        assert result["ok"] is True, f"应成功: {result}"
        data = result["data"]
        # 主返回不受影响
        assert data["hot_rank"]["trend"] == "上升"
        assert data["news_sentiment"]["利好"] == 1
        # distribution 块挂载
        assert "sentiment_distribution" in data, data.keys()
        dist = data["sentiment_distribution"]
        assert dist["target"] == {"code": "600519"}
        assert dist["dist"]["乐观"]["count"] == 2
        assert dist["samples_total"] == 3
        # 内部降级说明透传进 notes
        assert any("情绪分布" in n for n in data.get("notes", [])), data["notes"]

    def test_distribution_call_args(self, fake_sentiment_tool,
                                    fake_social_dist):
        tools_mod.execute_tool("get_stock_sentiment",
                               {"stock_code": "sh600519"})
        call = fake_social_dist.calls[-1]
        assert call["code"] == "600519", f"应传归一后 6 位代码: {call}"
        assert call["post_limit"] == 80

    def test_distribution_failure_no_break(self, fake_sentiment_tool,
                                           fake_social_dist):
        def boom(code=None, keyword=None, **kw):
            raise RuntimeError("分布通道炸了")

        fake_social_dist.get_sentiment_distribution = boom
        result = tools_mod.execute_tool("get_stock_sentiment",
                                        {"stock_code": "600519"})
        assert result["ok"] is True, f"分布失败不应拖垮工具: {result}"
        data = result["data"]
        assert data["hot_rank"]["trend"] == "上升", "人气榜主返回应保持"
        assert data["news_sentiment"]["利好"] == 1, "新闻情感主返回应保持"
        assert "sentiment_distribution" not in data
        assert any("情绪分布" in n for n in data.get("notes", [])), data["notes"]

    def test_distribution_capability_missing(self, fake_sentiment_tool,
                                             monkeypatch):
        mod = types.ModuleType("social_media")  # 无 get_sentiment_distribution
        monkeypatch.setattr(tools_mod, "_get_social_media_module", lambda: mod)
        result = tools_mod.execute_tool("get_stock_sentiment",
                                        {"stock_code": "600519"})
        assert result["ok"] is True
        data = result["data"]
        assert data["hot_rank"]["trend"] == "上升"
        assert "sentiment_distribution" not in data
        assert any("情绪分布通道不可用" in n for n in data.get("notes", []))


@pytest.fixture
def fake_social_search(monkeypatch):
    """假社媒门面（搜索路径）：search_all + buzz + mentions + 分布能力。"""
    mod = types.ModuleType("social_media")
    mod.UNSUPPORTED_PLATFORMS = {}
    mod.dist_calls = []

    def search_all(keyword, platforms=None, limit=10, **kw):
        return {
            "keyword": keyword, "date": "2026-08-12",
            "platforms": {"bilibili": 1},
            "posts": [{"platform": "bilibili", "post_id": "", "title": "视频",
                       "content": "简介", "metrics": {}, "url": "u",
                       "published_at": "", "source": "bilibili_search"}],
            "sources": {"bilibili": "direct"},
            "notes": [],
        }

    mod.search_all = search_all
    mod.aggregate_buzz = lambda items, scorer=None: {
        "total": len(items),
        "sentiment": {"利好": 0, "利空": 0, "中性": len(items)},
        "by_platform": {}, "avg_score": 0.0}
    mod.extract_stock_mentions = lambda posts, **kw: {}

    def get_sentiment_distribution(code=None, keyword=None, **kw):
        mod.dist_calls.append({"code": code, "keyword": keyword})
        return _dist_result("keyword", keyword)

    mod.get_sentiment_distribution = get_sentiment_distribution
    monkeypatch.setattr(tools_mod, "_get_social_media_module", lambda: mod)
    return mod


class TestSearchSocialDistributionBlock:

    def test_distribution_attached_with_comments(self, fake_social_search):
        result = tools_mod.execute_tool(
            "search_social_media",
            {"keyword": "贵州茅台", "with_comments": True})
        assert result["ok"] is True, f"应成功: {result}"
        data = result["data"]
        assert "sentiment_distribution" in data, data.keys()
        dist = data["sentiment_distribution"]
        assert dist["target"] == {"keyword": "贵州茅台"}
        assert dist["dist"]["乐观"]["count"] == 2
        call = fake_social_search.dist_calls[-1]
        assert call["keyword"] == "贵州茅台"
        assert call["code"] is None, "关键词路径不应传 code"
        # 主返回不受影响
        assert data["buzz"]["total"] >= 0
        assert "comments" in data

    def test_no_distribution_without_comments(self, fake_social_search):
        result = tools_mod.execute_tool("search_social_media",
                                        {"keyword": "贵州茅台"})
        assert result["ok"] is True
        data = result["data"]
        assert "sentiment_distribution" not in data, (
            "with_comments=false 不应挂分布块")
        assert fake_social_search.dist_calls == []

    def test_distribution_failure_no_break(self, fake_social_search):
        def boom(code=None, keyword=None, **kw):
            raise RuntimeError("分布通道炸了")

        fake_social_search.get_sentiment_distribution = boom
        result = tools_mod.execute_tool(
            "search_social_media",
            {"keyword": "贵州茅台", "with_comments": True})
        assert result["ok"] is True, f"分布失败不应拖垮工具: {result}"
        data = result["data"]
        assert data["posts"], "搜索主返回应保持"
        assert "sentiment_distribution" not in data
        assert any("情绪分布" in n for n in data.get("notes", [])), data["notes"]


# ════════════════════════════════════════════════════════════════
# 5. system_prompts「社媒舆情引用规范」新增条款
# ════════════════════════════════════════════════════════════════

class TestDistributionPrompts:

    def test_distribution_as_main_body_clause(self):
        prompt = system_prompts.AGENT_QUERY_PROMPT
        assert "7. 呈现社媒情绪必须以整体情绪分布为主体" in prompt
        assert "样本量 n" in prompt
        assert "乐观/中性/悲观占比与置信度" in prompt

    def test_small_sample_declaration_clause(self):
        prompt = system_prompts.AGENT_QUERY_PROMPT
        assert "8. 样本量 n<30 时必须明确声明" in prompt
        assert "样本不足，分布仅供参考" in prompt

    def test_representative_sample_clause(self):
        prompt = system_prompts.AGENT_QUERY_PROMPT
        assert "9. 单条帖子/评论只能作为代表性样本点缀" in prompt
        assert "至多引用 1-2 条且注明点赞数" in prompt
        assert "个别言论概括整体情绪" in prompt

    def test_trend_clause(self):
        prompt = system_prompts.AGENT_QUERY_PROMPT
        assert "10. 工具返回趋势数据时必须说明情绪转向" in prompt
        assert "无趋势数据时不得编造历史对比" in prompt

    def test_existing_clauses_kept(self):
        prompt = system_prompts.AGENT_QUERY_PROMPT
        assert "## 社媒舆情引用规范" in prompt
        assert "1. 引用社媒舆情" in prompt
        assert "小红书暂未覆盖" in prompt
        assert "6. 引用股吧舆情必须标注" in prompt
