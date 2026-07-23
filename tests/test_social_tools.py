"""tests/test_social_tools.py — 社媒舆情工具接线测试（v2-social，28→30）。

覆盖：
1. 2 个新工具（get_social_hot / search_social_media）的 schema 完整性、
   注册表规模（28→30）、工具名唯一性、目录展示。
2. execute_tool 分发正确性：平台参数分发（all/单平台/xiaohongshu 降级）、
   limit 夹取（1-30 默认 10）、posts 瘦身与 content 截 200 字、
   条数封顶 limit×平台数。
3. 富化挂载：buzz（aggregate_buzz）与 stock_mentions（extract_stock_mentions）
   挂载；门面失败/空结果/模块缺失降级，任何路径绝不抛异常。
4. with_comments 链路：前 3 条有 post_id 的 B 站结果各取 10 条评论（常量）、
   评论并入返回、buzz 对 posts+comments 合并打分；评论模块缺失/单帖失败降级。
5. system_prompts：AGENT_QUERY_PROMPT 新增「社媒舆情引用规范」一节存在性
   与既有节保留。

零网络保证：所有用例把假 social_media / social_bilibili 模块注入
sys.modules（裸名 + agent. 前缀），真实爬取层绝不被触达。
"""

import sys
import types

import pytest

import agent.tools as tools_mod
from agent import system_prompts

# ════════════════════════════════════════════════════════════════
# 公共工具：假数据与假模块注入
# ════════════════════════════════════════════════════════════════

_XHS_REASON = "小红书 v1 暂不支持：无可用无登录公开端点，v2 再评估接入。"


def _post(platform="bilibili", post_id="av1001", title="社媒热帖",
          content="正文内容", metrics=None):
    return {
        "platform": platform,
        "post_id": post_id,
        "title": title,
        "content": content,
        "author": "某作者",
        "metrics": metrics if metrics is not None else {"views": 123, "likes": 45},
        "url": "https://example.com/p",
        "published_at": "2026-08-12T10:00:00",
        "source": f"{platform}_hot",
    }


def _comment(post_id="av1001", idx=0):
    return {
        "platform": "bilibili",
        "post_id": post_id,
        "author": f"网友{idx}",
        "content": f"评论内容{idx}",
        "likes": 10 + idx,
        "published_at": "2026-08-12T11:00:00",
    }


def _hot_result(posts, notes=None):
    counts = {}
    for p in posts:
        counts[p["platform"]] = counts.get(p["platform"], 0) + 1
    return {
        "date": "2026-08-12",
        "platforms": counts,
        "posts": posts,
        "sources": {k: "direct" for k in counts},
        "notes": notes or [],
    }


def _search_result(keyword, posts, notes=None):
    out = _hot_result(posts, notes)
    out["keyword"] = keyword
    return out


def _buzz(items, scorer=None):
    return {
        "total": len(items),
        "sentiment": {"利好": 1, "利空": 0, "中性": max(0, len(items) - 1)},
        "by_platform": {"bilibili": {"total": len(items),
                                     "sentiment": {"利好": 1, "利空": 0,
                                                   "中性": max(0, len(items) - 1)}}},
        "avg_score": 0.5,
    }


def _mentions(posts, **kw):
    return {"600519": {"count": 1, "sample_titles": ["茅台热帖"]}}


@pytest.fixture
def fake_social(monkeypatch):
    """假社媒门面：注入 sys.modules 两种命名；可各用例覆盖函数。

    mod.calls 记录 get_hot_all / search_all / aggregate_buzz 调用参数。
    """
    mod = types.ModuleType("social_media")
    mod.calls = {"hot": [], "search": [], "buzz": []}
    mod.UNSUPPORTED_PLATFORMS = {"xiaohongshu": _XHS_REASON}

    def get_hot_all(platforms=None, limit=10, **kw):
        mod.calls["hot"].append({"platforms": platforms, "limit": limit})
        return _hot_result([_post()])

    def search_all(keyword, platforms=None, limit=10, **kw):
        mod.calls["search"].append({"keyword": keyword, "limit": limit})
        return _search_result(keyword, [_post()])

    def aggregate_buzz(items, scorer=None):
        mod.calls["buzz"].append(len(items))
        return _buzz(items)

    mod.get_hot_all = get_hot_all
    mod.search_all = search_all
    mod.aggregate_buzz = aggregate_buzz
    mod.extract_stock_mentions = _mentions
    monkeypatch.setitem(sys.modules, "agent.social_media", mod)
    monkeypatch.setitem(sys.modules, "social_media", mod)
    return mod


@pytest.fixture
def fake_bili(monkeypatch):
    """假 B 站模块：fetch_comments 记录调用，每条帖子返 2 条评论。"""
    mod = types.ModuleType("social_bilibili")
    mod.calls = []

    def fetch_comments(post_id, limit=20, **kw):
        mod.calls.append({"post_id": post_id, "limit": limit})
        return [_comment(post_id, i) for i in range(2)]

    mod.fetch_comments = fetch_comments
    monkeypatch.setitem(sys.modules, "agent.social_bilibili", mod)
    monkeypatch.setitem(sys.modules, "social_bilibili", mod)
    return mod


SLIM_POST_KEYS = {"platform", "title", "metrics", "url",
                  "published_at", "source", "content"}


# ════════════════════════════════════════════════════════════════
# 1. 注册表：规模 / schema 完整性 / 唯一性 / 目录
# ════════════════════════════════════════════════════════════════

class TestSocialToolRegistry:

    def test_registry_total_32(self):
        assert len(tools_mod.TOOL_REGISTRY) == 55, (
            f"TOOL_REGISTRY 应为 55 个工具（37 + 17 智研 MCP），"
            f"实际 {len(tools_mod.TOOL_REGISTRY)}"
        )

    def test_two_new_tools_schema_complete(self):
        by_name = {t["function"]["name"]: t["function"]
                   for t in tools_mod.TOOL_REGISTRY}
        for name in ("get_social_hot", "search_social_media"):
            assert name in by_name, f"注册表缺少新工具 {name}"
            fn = by_name[name]
            assert fn.get("description", "").strip(), f"{name} 缺少 description"
            params = fn.get("parameters")
            assert params and params.get("type") == "object"
            assert params.get("properties"), f"{name} 缺少 properties"
            import json
            json.dumps(params)  # 必须可 JSON 序列化

    def test_required_params(self):
        by_name = {t["function"]["name"]: t["function"]
                   for t in tools_mod.TOOL_REGISTRY}
        assert by_name["get_social_hot"]["parameters"].get("required", []) == []
        assert by_name["search_social_media"]["parameters"]["required"] == ["keyword"]

    def test_limit_and_with_comments_schema(self):
        by_name = {t["function"]["name"]: t["function"]
                   for t in tools_mod.TOOL_REGISTRY}
        for name in ("get_social_hot", "search_social_media"):
            limit = by_name[name]["parameters"]["properties"]["limit"]
            assert limit["type"] == "integer"
            assert limit["minimum"] == 1 and limit["maximum"] == 30
            assert limit["default"] == 10
        wc = by_name["search_social_media"]["parameters"]["properties"]["with_comments"]
        assert wc["type"] == "boolean" and wc["default"] is False

    def test_tool_names_unique(self):
        names = [t["function"]["name"] for t in tools_mod.TOOL_REGISTRY]
        assert len(names) == len(set(names)), f"存在重名工具: {names}"

    def test_catalog_and_short_desc(self):
        assert "get_social_hot" in tools_mod._SHORT_DESC
        assert "search_social_media" in tools_mod._SHORT_DESC
        catalog = tools_mod.get_tool_catalog()
        assert "get_social_hot" in catalog
        assert "search_social_media" in catalog
        assert "共 55 个" in catalog


# ════════════════════════════════════════════════════════════════
# 2. get_social_hot
# ════════════════════════════════════════════════════════════════

class TestGetSocialHot:

    def test_dispatch_success_structure(self, fake_social):
        result = tools_mod.execute_tool("get_social_hot", {})
        assert result["ok"] is True, f"应成功: {result}"
        data = result["data"]
        assert data["ok"] is True
        assert data["date"] == "2026-08-12"
        assert data["platforms"] == {"bilibili": 1}
        assert data["sources"] == {"bilibili": "direct"}
        # buzz / stock_mentions 挂载
        assert data["buzz"]["total"] == 1
        assert set(data["buzz"]["sentiment"]) == {"利好", "利空", "中性"}
        assert data["stock_mentions"]["600519"]["count"] == 1

    def test_posts_slimmed_and_content_truncated(self, fake_social):
        long_content = "长" * 500
        fake_social.get_hot_all = lambda platforms=None, limit=10, **kw: _hot_result(
            [_post(content=long_content)])
        result = tools_mod.execute_tool("get_social_hot", {})
        assert result["ok"] is True
        posts = result["data"]["posts"]
        assert len(posts) == 1
        p = posts[0]
        assert set(p.keys()) == SLIM_POST_KEYS, f"瘦身字段应为 {SLIM_POST_KEYS}: {p.keys()}"
        assert len(p["content"]) == 200, "content 应截断为 200 字"
        assert "author" not in p and "post_id" not in p

    def test_default_all_platforms_arg_none(self, fake_social):
        tools_mod.execute_tool("get_social_hot", {})
        call = fake_social.calls["hot"][-1]
        assert call["platforms"] is None, "platform 缺省/all 应传 None（全平台）"

    def test_explicit_all_platforms_arg_none(self, fake_social):
        tools_mod.execute_tool("get_social_hot", {"platform": "all"})
        assert fake_social.calls["hot"][-1]["platforms"] is None

    def test_single_platform_arg_list(self, fake_social):
        tools_mod.execute_tool("get_social_hot", {"platform": "Bilibili"})
        call = fake_social.calls["hot"][-1]
        assert call["platforms"] == ["bilibili"], (
            f"单平台应传 [platform] 且归一小写: {call}"
        )

    def test_xiaohongshu_degrades(self, fake_social):
        result = tools_mod.execute_tool("get_social_hot", {"platform": "xiaohongshu"})
        assert result["ok"] is True  # 分发层不失败
        data = result["data"]
        assert data["ok"] is False
        assert "小红书" in data.get("note", ""), f"应说明小红书缺席原因: {data}"
        assert not fake_social.calls["hot"], "缺席平台不应调门面 get_hot_all"

    def test_limit_clamped(self, fake_social):
        tools_mod.execute_tool("get_social_hot", {"limit": 999})
        assert fake_social.calls["hot"][-1]["limit"] == 30
        tools_mod.execute_tool("get_social_hot", {"limit": 0})
        assert fake_social.calls["hot"][-1]["limit"] == 1
        tools_mod.execute_tool("get_social_hot", {"limit": "abc"})
        assert fake_social.calls["hot"][-1]["limit"] == 10

    def test_posts_capped_limit_times_platforms(self, fake_social):
        posts = [_post("bilibili", f"av{i}") for i in range(6)]
        posts += [_post("weibo", f"w{i}") for i in range(6)]
        fake_social.get_hot_all = lambda platforms=None, limit=10, **kw: _hot_result(posts)
        result = tools_mod.execute_tool("get_social_hot", {"limit": 2})
        assert result["ok"] is True
        # 上限 = limit(2) × 平台数(2) = 4
        assert len(result["data"]["posts"]) == 4

    def test_facade_exception_never_raises(self, fake_social):
        def _boom(platforms=None, limit=10, **kw):
            raise RuntimeError("门面炸了")

        fake_social.get_hot_all = _boom
        result = tools_mod.execute_tool("get_social_hot", {})
        assert result["ok"] is False, f"门面异常应 ok=False 而非抛出: {result}"
        assert result.get("error")

    def test_facade_non_dict_degrades(self, fake_social):
        fake_social.get_hot_all = lambda platforms=None, limit=10, **kw: None
        result = tools_mod.execute_tool("get_social_hot", {})
        assert result["ok"] is True
        assert result["data"]["ok"] is False
        assert "结构异常" in result["data"].get("note", "")

    def test_empty_result_degrades(self, fake_social):
        fake_social.get_hot_all = lambda platforms=None, limit=10, **kw: _hot_result(
            [], notes=["weibo: 本轮缺席"])
        result = tools_mod.execute_tool("get_social_hot", {})
        assert result["ok"] is True
        data = result["data"]
        assert data["posts"] == []
        assert data["buzz"]["total"] == 0
        assert any("未抓到" in n for n in data["notes"]), data["notes"]

    def test_enrich_failure_degrades(self, fake_social):
        def _boom_buzz(items, scorer=None):
            raise RuntimeError("打分器炸了")

        fake_social.aggregate_buzz = _boom_buzz
        result = tools_mod.execute_tool("get_social_hot", {})
        assert result["ok"] is True
        data = result["data"]
        assert data["buzz"]["total"] == 0
        assert any("情感聚合失败" in n for n in data["notes"])

    def test_module_missing_degrades(self, monkeypatch):
        monkeypatch.setattr(tools_mod, "_get_social_media_module", lambda: None)
        result = tools_mod.execute_tool("get_social_hot", {})
        assert result["ok"] is True
        assert result["data"]["ok"] is False
        assert "社媒舆情模块不可用" in result["data"].get("note", "")


# ════════════════════════════════════════════════════════════════
# 3. search_social_media
# ════════════════════════════════════════════════════════════════

class TestSearchSocialMedia:

    def test_search_success_structure(self, fake_social):
        result = tools_mod.execute_tool("search_social_media", {"keyword": "茅台"})
        assert result["ok"] is True, f"应成功: {result}"
        data = result["data"]
        assert data["ok"] is True
        assert data["keyword"] == "茅台"
        assert data["buzz"]["total"] == 1
        assert data["stock_mentions"]["600519"]["count"] == 1
        assert set(data["posts"][0].keys()) == SLIM_POST_KEYS
        assert "comments" not in data, "with_comments 缺省 false 不应带 comments 键"

    def test_keyword_missing_ok_false(self, fake_social):
        result = tools_mod.execute_tool("search_social_media", {})
        assert result["ok"] is False
        assert "keyword" in result.get("error", "")

    def test_keyword_blank_ok_false(self, fake_social):
        for bad in ("", "   "):
            result = tools_mod.execute_tool("search_social_media", {"keyword": bad})
            assert result["ok"] is False, f"keyword={bad!r} 应 ok=False: {result}"
            assert not fake_social.calls["search"], "空关键词不应调门面 search_all"
            fake_social.calls["search"].clear()

    def test_search_limit_clamped(self, fake_social):
        tools_mod.execute_tool("search_social_media", {"keyword": "降息", "limit": 99})
        assert fake_social.calls["search"][-1]["limit"] == 30
        tools_mod.execute_tool("search_social_media", {"keyword": "降息", "limit": -5})
        assert fake_social.calls["search"][-1]["limit"] == 1

    def test_search_posts_capped(self, fake_social):
        posts = [_post("bilibili", f"av{i}") for i in range(20)]
        fake_social.search_all = lambda keyword, platforms=None, limit=10, **kw: (
            _search_result(keyword, posts))
        result = tools_mod.execute_tool(
            "search_social_media", {"keyword": "机器人", "limit": 5})
        assert result["ok"] is True
        # 上限 = limit(5) × 平台数(1) = 5
        assert len(result["data"]["posts"]) == 5

    def test_with_comments_top3_bilibili_only(self, fake_social, fake_bili):
        posts = [_post("bilibili", f"av{i}") for i in range(4)]
        posts.append(_post("weibo", "w1"))  # 非 B 站不取评论
        posts.append(_post("bilibili", ""))  # 无 post_id 不取评论
        fake_social.search_all = lambda keyword, platforms=None, limit=10, **kw: (
            _search_result(keyword, posts))
        result = tools_mod.execute_tool(
            "search_social_media", {"keyword": "茅台", "with_comments": True})
        assert result["ok"] is True, f"应成功: {result}"
        # 只对前 3 条有 post_id 的 B 站结果取评论，每条 limit=10
        assert len(fake_bili.calls) == 3
        assert [c["post_id"] for c in fake_bili.calls] == ["av0", "av1", "av2"]
        assert all(c["limit"] == 10 for c in fake_bili.calls)
        comments = result["data"]["comments"]
        assert len(comments) == 6, f"3 帖 × 2 评论 = 6: {len(comments)}"
        assert set(comments[0].keys()) == {"platform", "post_id", "author",
                                           "likes", "published_at", "content"}

    def test_buzz_scored_over_posts_plus_comments(self, fake_social, fake_bili):
        posts = [_post("bilibili", f"av{i}") for i in range(2)]
        fake_social.search_all = lambda keyword, platforms=None, limit=10, **kw: (
            _search_result(keyword, posts))
        result = tools_mod.execute_tool(
            "search_social_media", {"keyword": "茅台", "with_comments": True})
        assert result["ok"] is True
        # 2 帖 + 2 帖 × 2 评论 = 6 条合并打分
        assert fake_social.calls["buzz"][-1] == 6
        assert result["data"]["buzz"]["total"] == 6

    def test_comments_content_truncated(self, fake_social, fake_bili):
        fake_bili.fetch_comments = lambda post_id, limit=20, **kw: [
            {**_comment(post_id, 0), "content": "评" * 400}]
        result = tools_mod.execute_tool(
            "search_social_media", {"keyword": "茅台", "with_comments": True})
        assert result["ok"] is True
        assert len(result["data"]["comments"][0]["content"]) == 200

    def test_comments_fetch_failure_degrades(self, fake_social, fake_bili):
        posts = [_post("bilibili", "av1"), _post("bilibili", "av2")]
        fake_social.search_all = lambda keyword, platforms=None, limit=10, **kw: (
            _search_result(keyword, posts))

        def flaky(post_id, limit=20, **kw):
            if post_id == "av1":
                raise RuntimeError("评论接口 412 风控")
            return [_comment(post_id, 0)]

        fake_bili.fetch_comments = flaky
        result = tools_mod.execute_tool(
            "search_social_media", {"keyword": "茅台", "with_comments": True})
        assert result["ok"] is True, f"单帖评论失败不应拖垮工具: {result}"
        data = result["data"]
        assert len(data["comments"]) == 1, "av2 的评论仍应拿到"
        assert any("av1" in n and "失败" in n for n in data["notes"]), data["notes"]

    def test_comments_module_missing_degrades(self, fake_social, monkeypatch):
        monkeypatch.setattr(tools_mod, "_get_social_bilibili_module", lambda: None)
        result = tools_mod.execute_tool(
            "search_social_media", {"keyword": "茅台", "with_comments": True})
        assert result["ok"] is True
        data = result["data"]
        assert data["comments"] == []
        assert any("评论模块不可用" in n for n in data["notes"]), data["notes"]

    def test_search_facade_exception_never_raises(self, fake_social):
        def _boom(keyword, platforms=None, limit=10, **kw):
            raise RuntimeError("搜索门面炸了")

        fake_social.search_all = _boom
        result = tools_mod.execute_tool("search_social_media", {"keyword": "茅台"})
        assert result["ok"] is False
        assert result.get("error")

    def test_search_module_missing_degrades(self, monkeypatch):
        monkeypatch.setattr(tools_mod, "_get_social_media_module", lambda: None)
        result = tools_mod.execute_tool("search_social_media", {"keyword": "茅台"})
        assert result["ok"] is True
        assert result["data"]["ok"] is False
        assert "社媒舆情模块不可用" in result["data"].get("note", "")


# ════════════════════════════════════════════════════════════════
# 4. system_prompts：AGENT_QUERY_PROMPT 新增「社媒舆情引用规范」
# ════════════════════════════════════════════════════════════════

class TestSocialPromptSection:

    def test_section_exists(self):
        assert "社媒舆情引用规范" in system_prompts.AGENT_QUERY_PROMPT

    def test_section_rules_content(self):
        prompt = system_prompts.AGENT_QUERY_PROMPT
        assert "标注平台与抓取日期" in prompt, "应要求标注平台与抓取日期"
        assert "辅助参考" in prompt, "应声明社媒情绪仅作辅助参考"
        assert "不构成买卖依据" in prompt, "应声明不构成买卖依据"
        assert "小红书暂未覆盖" in prompt, "应诚实声明小红书未覆盖"
        assert "搜索与评论只有 B 站支持" in prompt, "应声明搜索/评论仅 B 站"
        assert "不得编造" in prompt, "应要求未返回内容不得编造"
        assert "get_social_hot" in prompt and "search_social_media" in prompt

    def test_existing_sections_preserved(self):
        prompt = system_prompts.AGENT_QUERY_PROMPT
        for marker in ("研报引用规范", "情绪数据引用规范", "技术分析纪律",
                       "投资人格框架", "输出结构", "输出格式"):
            assert marker in prompt, f"AGENT_QUERY_PROMPT 既有节 {marker!r} 丢失"
