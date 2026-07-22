"""tests/test_guba_integration.py — 东财股吧舆情集成测试（v2-social，股吧通道）。

覆盖：
1. social_media.get_guba_buzz：成功路径（假 social_guba 模块注入
   sys.modules）、fetch/enrich 参数透传、enrich=0 跳过、富化失败/能力
   缺失降级、非法 code、模块缺席、空列表/异常/非 list 降级、帖子瘦身
   与 content 截 200 字、绝不抛异常、不进 PLATFORM_MODULES。
2. tools.get_stock_sentiment 的 guba 增强块：成功挂 guba={posts, buzz}
   键、limit=10/enrich=3 透传、股吧失败/模块缺席/能力缺失/结构异常
   均只进 notes 绝不影响人气榜+新闻情感主返回。
3. system_prompts：「社媒舆情引用规范」一节新增股吧两行断言，且新增
   内容位于该节内部、既有规则保留。

零网络保证：所有用例把假 social_guba / social_media / sentiment 模块
注入 sys.modules 或 monkeypatch 模块解析函数，真实爬取层与数据层绝不
被触达。buzz 打分复用真实 aggregate_buzz（关键词打分器，纯本地）。
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


def _guba_post(post_id="1748053530", title="茅台中报讨论", content="",
               likes=None):
    """统一 Post 契约（platform='guba'，metrics 只放拿到的键）。"""
    metrics = {"views": 1395, "comments": 8, "shares": 3}
    if likes is not None:
        metrics["likes"] = likes
    return {
        "platform": "guba",
        "post_id": post_id,
        "title": title,
        "content": content,
        "author": "股吧网友",
        "metrics": metrics,
        "url": f"https://guba.eastmoney.com/news,600519,{post_id}.html",
        "published_at": "2026-07-22T19:10:41+08:00",
        "source": "贵州茅台吧",
    }


@pytest.fixture
def fake_guba(monkeypatch):
    """假股吧模块：fetch_bar_posts 返 2 帖，enrich_posts 回填 content/likes。

    mod.calls 记录 fetch/enrich 调用参数。注入 sys.modules 两种命名。
    """
    mod = types.ModuleType("social_guba")
    mod.calls = {"fetch": [], "enrich": []}

    def fetch_bar_posts(code, limit=30, session=None, sleep=None):
        mod.calls["fetch"].append({"code": code, "limit": limit, "sleep": sleep})
        return [_guba_post("1748053530"), _guba_post("1748053531", title="利空出尽了吗")]

    def enrich_posts(posts, top_n=3, session=None, sleep=None):
        mod.calls["enrich"].append({"top_n": top_n, "n": len(posts), "sleep": sleep})
        out = []
        for i, p in enumerate(posts):
            q = dict(p)
            if i < top_n:
                q["content"] = f"正文摘要{i}"
                q["metrics"] = {**q.get("metrics", {}), "likes": 3 + i}
            out.append(q)
        return out

    mod.fetch_bar_posts = fetch_bar_posts
    mod.enrich_posts = enrich_posts
    monkeypatch.setitem(sys.modules, "agent.social_guba", mod)
    monkeypatch.setitem(sys.modules, "social_guba", mod)
    return mod


SLIM_GUBA_KEYS = {"platform", "post_id", "title", "content",
                  "metrics", "url", "published_at", "source"}


# ════════════════════════════════════════════════════════════════
# 1. get_guba_buzz：成功路径与参数透传
# ════════════════════════════════════════════════════════════════

class TestGetGubaBuzzSuccess:

    def test_success_structure(self, fake_guba):
        result = sm.get_guba_buzz("600519")
        assert set(result.keys()) == {"code", "posts", "buzz", "sources", "notes"}
        assert result["code"] == "600519"
        assert result["sources"] == ["eastmoney_guba"]
        assert len(result["posts"]) == 2
        buzz = result["buzz"]
        assert buzz["total"] == 2
        assert set(buzz["sentiment"]) == {"利好", "利空", "中性"}
        assert "guba" in buzz["by_platform"]
        assert buzz["by_platform"]["guba"]["total"] == 2

    def test_fetch_called_with_code_limit_sleep(self, fake_guba):
        sentinel_sleep = lambda s: None  # noqa: E731
        sm.get_guba_buzz("600519", limit=7, sleep=sentinel_sleep)
        call = fake_guba.calls["fetch"][-1]
        assert call["code"] == "600519"
        assert call["limit"] == 7
        assert call["sleep"] is sentinel_sleep

    def test_enrich_passthrough_top_n(self, fake_guba):
        result = sm.get_guba_buzz("600519", limit=5, enrich=1)
        call = fake_guba.calls["enrich"][-1]
        assert call["top_n"] == 1, f"enrich 应透传为 top_n: {call}"
        assert call["n"] == 2, "enrich 应收全量帖子列表"
        # 仅第 1 帖被富化出 likes 与正文
        assert result["posts"][0]["metrics"].get("likes") == 3
        assert result["posts"][0]["content"] == "正文摘要0"
        assert "likes" not in result["posts"][1]["metrics"]

    def test_enrich_zero_skips_enrich(self, fake_guba):
        result = sm.get_guba_buzz("600519", enrich=0)
        assert not fake_guba.calls["enrich"], "enrich=0 不应调 enrich_posts"
        assert len(result["posts"]) == 2
        assert any("跳过详情富化" in n for n in result["notes"])

    def test_invalid_limit_falls_back(self, fake_guba):
        sm.get_guba_buzz("600519", limit="abc")
        assert fake_guba.calls["fetch"][-1]["limit"] == 20
        sm.get_guba_buzz("600519", limit=0)
        assert fake_guba.calls["fetch"][-1]["limit"] == 1

    def test_prefixed_code_normalized(self, fake_guba):
        result = sm.get_guba_buzz("sh600519")
        assert result["code"] == "600519"
        assert fake_guba.calls["fetch"][-1]["code"] == "600519"

    def test_posts_slimmed_keys_and_truncation(self, fake_guba):
        fake_guba.enrich_posts = lambda posts, top_n=3, **kw: [
            {**p, "content": "长" * 500, "metrics": {**p["metrics"], "likes": 9}}
            for p in posts
        ]
        result = sm.get_guba_buzz("600519")
        p = result["posts"][0]
        assert set(p.keys()) == SLIM_GUBA_KEYS, f"瘦身字段应为 {SLIM_GUBA_KEYS}: {p.keys()}"
        assert len(p["content"]) == 200, "content 应截断为 200 字"
        assert "author" not in p, "瘦身不应保留 author"
        assert p["platform"] == "guba"
        assert isinstance(p["post_id"], str) and p["post_id"] == "1748053530"
        assert p["metrics"]["likes"] == 9

    def test_non_dict_posts_filtered(self, fake_guba):
        fake_guba.fetch_bar_posts = lambda code, limit=30, **kw: [
            _guba_post(), "junk", None, 42]
        result = sm.get_guba_buzz("600519")
        assert len(result["posts"]) == 1
        assert result["buzz"]["total"] == 1


# ════════════════════════════════════════════════════════════════
# 2. get_guba_buzz：降级路径（绝不抛）
# ════════════════════════════════════════════════════════════════

class TestGetGubaBuzzDegrade:

    def test_invalid_code_degrades(self, fake_guba):
        for bad in ("abc", "", "12345", "6005199", None):
            result = sm.get_guba_buzz(bad)
            assert result["posts"] == []
            assert result["buzz"]["total"] == 0
            assert result["sources"] == ["eastmoney_guba"]
            assert any("非法" in n for n in result["notes"]), result["notes"]
        assert not fake_guba.calls["fetch"], "非法 code 不应触发抓取"

    def test_module_missing_degrades(self, monkeypatch):
        # sys.modules 置 None 使 import 停机，确定性模拟模块缺席
        monkeypatch.setitem(sys.modules, "agent.social_guba", None)
        monkeypatch.setitem(sys.modules, "social_guba", None)
        result = sm.get_guba_buzz("600519")
        assert result["code"] == "600519"
        assert result["posts"] == []
        assert result["buzz"]["total"] == 0
        assert any("未就绪" in n for n in result["notes"]), result["notes"]

    def test_fetch_capability_missing_degrades(self, monkeypatch):
        mod = types.ModuleType("social_guba")  # 无 fetch_bar_posts
        monkeypatch.setitem(sys.modules, "agent.social_guba", mod)
        monkeypatch.setitem(sys.modules, "social_guba", mod)
        result = sm.get_guba_buzz("600519")
        assert result["posts"] == []
        assert any("fetch_bar_posts" in n for n in result["notes"])

    def test_empty_list_degrades(self, fake_guba):
        fake_guba.fetch_bar_posts = lambda code, limit=30, **kw: []
        result = sm.get_guba_buzz("600519")
        assert result["posts"] == []
        assert result["buzz"]["total"] == 0
        assert any("未抓到" in n for n in result["notes"]), result["notes"]
        assert not fake_guba.calls["enrich"], "空列表不应触发富化"

    def test_fetch_exception_degrades_no_raise(self, fake_guba):
        def _boom(code, limit=30, **kw):
            raise RuntimeError("WAF 403")

        fake_guba.fetch_bar_posts = _boom
        result = sm.get_guba_buzz("600519")  # 绝不抛
        assert result["posts"] == []
        assert result["notes"], "失败应有中文说明进 notes"

    def test_fetch_non_list_degrades(self, fake_guba):
        fake_guba.fetch_bar_posts = lambda code, limit=30, **kw: {"rc": 1}
        result = sm.get_guba_buzz("600519")
        assert result["posts"] == []
        assert any("未抓到" in n for n in result["notes"])

    def test_enrich_failure_degrades_keeps_posts(self, fake_guba):
        def _boom(posts, top_n=3, **kw):
            raise RuntimeError("详情页超时")

        fake_guba.enrich_posts = _boom
        result = sm.get_guba_buzz("600519")  # 绝不抛
        assert len(result["posts"]) == 2, "富化失败应保留列表原始帖子"
        assert result["buzz"]["total"] == 2
        assert any("富化失败" in n for n in result["notes"]), result["notes"]

    def test_enrich_capability_missing_degrades(self, fake_guba):
        del fake_guba.enrich_posts
        result = sm.get_guba_buzz("600519")
        assert len(result["posts"]) == 2
        assert any("enrich_posts" in n for n in result["notes"])

    def test_enrich_empty_return_keeps_list(self, fake_guba):
        fake_guba.enrich_posts = lambda posts, top_n=3, **kw: []
        result = sm.get_guba_buzz("600519")
        assert len(result["posts"]) == 2, "富化返空应保留列表原始帖子"
        assert any("富化返回为空" in n for n in result["notes"])

    def test_internal_exception_never_raises(self, fake_guba, monkeypatch):
        def _boom(posts, scorer=None):
            raise RuntimeError("打分器炸了")

        monkeypatch.setattr(sm, "aggregate_buzz", _boom)
        result = sm.get_guba_buzz("600519")  # 绝不抛
        assert result["posts"] == []
        assert any("内部异常" in n for n in result["notes"]), result["notes"]

    def test_guba_not_in_platform_modules(self):
        assert "guba" not in sm.PLATFORM_MODULES
        assert "social_guba" not in sm.PLATFORM_MODULES.values()
        assert "guba" not in sm.UNSUPPORTED_PLATFORMS


# ════════════════════════════════════════════════════════════════
# 3. get_stock_sentiment：guba 增强块
# ════════════════════════════════════════════════════════════════


def _sentiment_result(code="600519"):
    return {
        "code": code,
        "hot_rank": {"latest": 3, "history_avg": 12.5, "trend": "上升"},
        "news_sentiment": {"利好": 1, "利空": 0, "中性": 2},
        "sources": ["eastmoney_hotrank"],
        "notes": [],
    }


@pytest.fixture
def fake_sentiment(monkeypatch):
    """假情绪模块 + 数据层缺席（新闻注入降级，零网络）。"""
    mod = types.ModuleType("sentiment")
    mod.get_stock_sentiment = lambda code, days=30, news_items=None: _sentiment_result(code)
    monkeypatch.setattr(tools_mod, "_get_sentiment_module", lambda: mod)
    monkeypatch.setattr(tools_mod, "_get_data_fetcher", lambda: None)
    return mod


@pytest.fixture
def fake_social_guba(monkeypatch):
    """假社媒门面：get_guba_buzz 记录调用并返回股吧舆情块。"""
    mod = types.ModuleType("social_media")
    mod.calls = []

    def get_guba_buzz(code, limit=20, enrich=3, sleep=None):
        mod.calls.append({"code": code, "limit": limit, "enrich": enrich})
        return {
            "code": code,
            "posts": [{"platform": "guba", "post_id": "1748053530",
                       "title": "茅台中报讨论", "content": "正文摘要",
                       "metrics": {"views": 1395, "comments": 8,
                                   "shares": 3, "likes": 3},
                       "url": "https://guba.eastmoney.com/news,600519,1748053530.html",
                       "published_at": "2026-07-22T19:10:41+08:00",
                       "source": "贵州茅台吧"}],
            "buzz": {"total": 1,
                     "sentiment": {"利好": 1, "利空": 0, "中性": 0},
                     "by_platform": {"guba": {"total": 1, "sentiment":
                                              {"利好": 1, "利空": 0, "中性": 0}}},
                     "avg_score": 0.6},
            "sources": ["eastmoney_guba"],
            "notes": [],
        }

    mod.get_guba_buzz = get_guba_buzz
    monkeypatch.setattr(tools_mod, "_get_social_media_module", lambda: mod)
    return mod


class TestStockSentimentGuba:

    def test_guba_block_attached(self, fake_sentiment, fake_social_guba):
        result = tools_mod.execute_tool("get_stock_sentiment", {"stock_code": "600519"})
        assert result["ok"] is True, f"应成功: {result}"
        data = result["data"]
        # 既有返回不受影响
        assert data["hot_rank"]["trend"] == "上升"
        assert data["news_sentiment"]["利好"] == 1
        # guba 块挂载
        assert "guba" in data, f"应挂 guba 块: {data.keys()}"
        assert len(data["guba"]["posts"]) == 1
        assert data["guba"]["posts"][0]["metrics"]["likes"] == 3
        assert data["guba"]["buzz"]["total"] == 1

    def test_guba_call_args(self, fake_sentiment, fake_social_guba):
        tools_mod.execute_tool("get_stock_sentiment", {"stock_code": "sh600519"})
        call = fake_social_guba.calls[-1]
        assert call["code"] == "600519", f"应传归一后 6 位代码: {call}"
        assert call["limit"] == 10
        assert call["enrich"] == 3

    def test_guba_failure_does_not_break_main(self, fake_sentiment, fake_social_guba):
        def _boom(code, limit=20, enrich=3, sleep=None):
            raise RuntimeError("股吧端点炸了")

        fake_social_guba.get_guba_buzz = _boom
        result = tools_mod.execute_tool("get_stock_sentiment", {"stock_code": "600519"})
        assert result["ok"] is True, f"股吧失败不应拖垮工具: {result}"
        data = result["data"]
        assert data["hot_rank"]["trend"] == "上升", "人气榜主返回应保持"
        assert data["news_sentiment"]["利好"] == 1, "新闻情感主返回应保持"
        assert "guba" not in data
        assert any("股吧" in n for n in data.get("notes", [])), data.get("notes")

    def test_guba_module_missing_degrades(self, fake_sentiment, monkeypatch):
        monkeypatch.setattr(tools_mod, "_get_social_media_module", lambda: None)
        result = tools_mod.execute_tool("get_stock_sentiment", {"stock_code": "600519"})
        assert result["ok"] is True
        data = result["data"]
        assert data["hot_rank"]["trend"] == "上升"
        assert "guba" not in data
        assert any("股吧" in n for n in data.get("notes", [])), data.get("notes")

    def test_guba_capability_missing_degrades(self, fake_sentiment, monkeypatch):
        mod = types.ModuleType("social_media")  # 无 get_guba_buzz
        monkeypatch.setattr(tools_mod, "_get_social_media_module", lambda: mod)
        result = tools_mod.execute_tool("get_stock_sentiment", {"stock_code": "600519"})
        assert result["ok"] is True
        data = result["data"]
        assert "guba" not in data
        assert any("股吧舆情通道不可用" in n for n in data.get("notes", []))

    def test_guba_non_dict_return_degrades(self, fake_sentiment, fake_social_guba):
        fake_social_guba.get_guba_buzz = lambda code, limit=20, enrich=3, sleep=None: None
        result = tools_mod.execute_tool("get_stock_sentiment", {"stock_code": "600519"})
        assert result["ok"] is True
        data = result["data"]
        assert data["hot_rank"]["trend"] == "上升"
        assert "guba" not in data
        assert any("结构异常" in n for n in data.get("notes", []))

    def test_guba_internal_notes_propagated(self, fake_sentiment, fake_social_guba):
        def _empty(code, limit=20, enrich=3, sleep=None):
            return {"code": code, "posts": [],
                    "buzz": {"total": 0,
                             "sentiment": {"利好": 0, "利空": 0, "中性": 0},
                             "by_platform": {}, "avg_score": 0.0},
                    "sources": ["eastmoney_guba"],
                    "notes": ["股吧 600519 本轮未抓到帖子"]}

        fake_social_guba.get_guba_buzz = _empty
        result = tools_mod.execute_tool("get_stock_sentiment", {"stock_code": "600519"})
        assert result["ok"] is True
        data = result["data"]
        assert data["guba"]["posts"] == []
        assert any("股吧" in n and "未抓到" in n
                   for n in data.get("notes", [])), data.get("notes")

    def test_sentiment_missing_path_unchanged(self, monkeypatch):
        monkeypatch.setattr(tools_mod, "_get_sentiment_module", lambda: None)
        result = tools_mod.execute_tool("get_stock_sentiment", {"stock_code": "600519"})
        assert result["ok"] is True
        data = result["data"]
        assert data.get("ok") is False
        assert "情绪模块不可用" in data.get("note", "")
        assert "guba" not in data, "情绪模块缺席早退路径不应带 guba 块"


# ════════════════════════════════════════════════════════════════
# 4. system_prompts：社媒舆情引用规范新增股吧两行
# ════════════════════════════════════════════════════════════════


def _social_section() -> str:
    """截取 AGENT_QUERY_PROMPT 的「社媒舆情引用规范」一节文本。"""
    prompt = system_prompts.AGENT_QUERY_PROMPT
    start = prompt.index("## 社媒舆情引用规范")
    nxt = prompt.find("\n## ", start + 1)
    return prompt[start:] if nxt == -1 else prompt[start:nxt]


class TestGubaPrompts:

    def test_guba_coverage_line(self):
        section = _social_section()
        assert "东财股吧已覆盖" in section, "应声明东财股吧已覆盖"
        assert "无评论数据" in section, "应声明股吧无评论数据"

    def test_guba_citation_line(self):
        section = _social_section()
        assert "东方财富股吧" in section, "应要求标注东方财富股吧"
        assert "点赞/阅读/评论数" in section, "应要求互动数只用工具返回"

    def test_existing_rules_preserved(self):
        section = _social_section()
        for marker in ("标注平台与抓取日期", "不构成买卖依据", "小红书暂未覆盖",
                       "搜索与评论只有 B 站支持", "不得编造"):
            assert marker in section, f"既有规则 {marker!r} 丢失"
