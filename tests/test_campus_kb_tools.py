"""tests/test_campus_kb_tools.py — 校园知识库工具接线 + campus_kb 意图路由测试（全 mock 零网络）。

覆盖范围（接线工程师_F 交付）：
1. 工具注册表 30→32：两个新工具 schema 完整性（必填项/枚举/边界）与目录断言。
2. search_campus_knowledge：正常检索（content 截断 500 字、字段瘦身）、
   source 枚举校验（非法值按 None 处理）、limit 夹取、空库/未建库中文指引、
   非空库零命中提示、缺依赖（模块/接口缺失）降级、必填参数兜底。
3. get_course_review_summary：现成总结命中（不触发生成）、miss 时按
   metadata_json.course_sqid 分组取最多点评现场生成并回写、双 miss 中文提示、
   空库指引、总结模块缺失降级、必填参数兜底。
4. detect_intent：campus_kb 意图触发词矩阵 + 既有意图回归（复盘/社媒/人格/
   自选股/个股/数据查询/闲聊路由不变）+ 复盘护栏（含『复盘』不被校园词抢）。
5. hint 透传接线：campus_kb ∈ _AGENT_ROUTE_HINTS，process_message 走
   _agent_query 且 hint 透传（参照 social_sentiment 样板）。
6. system_prompts：AGENT_QUERY_PROMPT 含「校园知识库引用规范」一节及关键纪律。

规则（与项目其他测试一致）：
- 全 mock 零网络；campus_kb / review_summary 一律 monkeypatch 注入
  （tools 模块的 _get_campus_kb_module / _get_review_summary_module 惰性解析点）。
- 无 pytest-asyncio，异步用 asyncio.run 驱动。
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import agent.orchestrator as orchestrator
import agent.tools as tools_mod
from agent.orchestrator import MarketReviewAgent, detect_intent
from agent.system_prompts import AGENT_QUERY_PROMPT


# ════════════════════════════════════════════════════════════════
# 公共工具
# ════════════════════════════════════════════════════════════════

def _make_agent() -> MarketReviewAgent:
    agent = MarketReviewAgent()
    agent.client = MagicMock()
    return agent


def _kb_entry(source="thubook", title="测试条目", content="正文内容",
              url="https://example.com", score=1.5, metadata=None):
    entry = {
        "source": source,
        "source_id": f"{source}:test:1",
        "title": title,
        "content": content,
        "url": url,
        "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
        "updated_at": "2026-01-01T00:00:00",
        "score": score,
    }
    return entry


@pytest.fixture
def fake_kb(monkeypatch):
    """注入假的 campus_kb 模块（search_kb/stats/upsert_entries 全 mock）。"""
    kb = MagicMock()
    kb.search_kb = MagicMock(return_value=[])
    kb.stats = MagicMock(return_value={"total": 100, "by_source": {"thubook": 100}})
    kb.upsert_entries = MagicMock(return_value=1)
    monkeypatch.setattr(tools_mod, "_get_campus_kb_module", lambda: kb)
    return kb


@pytest.fixture
def fake_rs(monkeypatch):
    """注入假的 review_summary 模块（summarize/build_summary_entry 全 mock）。"""
    rs = MagicMock()
    rs.summarize_course_reviews = MagicMock(return_value={
        "summary_text": "基于 3 条点评的自动摘要：《数据结构》给分好。",
        "rating_avg": 4.3,
        "rating_dist": {"5": 2, "4": 1},
        "review_count": 3,
        "highlights": ["给分好", "作业多", "讲课清晰"],
        "method": "fallback",
    })
    rs.build_summary_entry = MagicMock(return_value={
        "source": "thucourse_summary",
        "source_id": "thucourse:summary:sq100",
        "title": "数据结构 · 点评综合总结",
        "content": "基于 3 条点评的自动摘要",
        "url": "thucourse:course:sq100",
        "metadata_json": "{}",
        "updated_at": "2026-01-01T00:00:00",
    })
    monkeypatch.setattr(tools_mod, "_get_review_summary_module", lambda: rs)
    return rs


# ════════════════════════════════════════════════════════════════
# 1. 注册表与 schema（30→32）
# ════════════════════════════════════════════════════════════════

class TestToolRegistry:
    def test_registry_has_32_tools_unique_names(self):
        assert len(tools_mod.TOOL_REGISTRY) == 32, (
            f"注册表应为 32 个工具（30→32），实际 {len(tools_mod.TOOL_REGISTRY)}"
        )
        names = [t["function"]["name"] for t in tools_mod.TOOL_REGISTRY]
        assert len(names) == len(set(names)), f"工具名不得重复: {names}"
        assert "search_campus_knowledge" in names
        assert "get_course_review_summary" in names

    def test_search_campus_knowledge_schema(self):
        tool = next(t for t in tools_mod.TOOL_REGISTRY
                    if t["function"]["name"] == "search_campus_knowledge")
        params = tool["function"]["parameters"]
        assert params["required"] == ["query"]
        src = params["properties"]["source"]
        assert set(src["enum"]) == {
            "sem_handbook", "thucourse_course", "thucourse_review",
            "thucourse_summary", "thubook",
        }
        limit = params["properties"]["limit"]
        assert limit["minimum"] == 1 and limit["maximum"] == 20
        assert limit["default"] == 5

    def test_get_course_review_summary_schema(self):
        tool = next(t for t in tools_mod.TOOL_REGISTRY
                    if t["function"]["name"] == "get_course_review_summary")
        params = tool["function"]["parameters"]
        assert params["required"] == ["course_query"]
        assert "course_query" in params["properties"]

    def test_tool_catalog_lists_new_tools(self):
        catalog = tools_mod.get_tool_catalog()
        assert "共 32 个" in catalog
        assert "search_campus_knowledge" in catalog
        assert "get_course_review_summary" in catalog


# ════════════════════════════════════════════════════════════════
# 2. search_campus_knowledge 工具
# ════════════════════════════════════════════════════════════════

class TestSearchCampusKnowledge:
    def test_success_path_fields_and_score(self, fake_kb):
        fake_kb.search_kb.return_value = [
            _kb_entry(title="选课手册第三章", content="选课规则正文", score=2.5),
            _kb_entry(title="宿舍分配", content="宿舍正文", score=1.0),
        ]
        result = tools_mod.execute_tool(
            "search_campus_knowledge", {"query": "选课 宿舍"}
        )
        assert result["ok"] is True
        data = result["data"]
        assert data["ok"] is True
        assert data["query"] == "选课 宿舍"
        assert len(data["results"]) == 2
        first = data["results"][0]
        assert set(first) == {"source", "title", "content", "url", "score"}
        assert first["title"] == "选课手册第三章"
        assert first["score"] == 2.5
        # 检索参数透传
        _, kwargs = fake_kb.search_kb.call_args
        assert kwargs["source"] is None and kwargs["limit"] == 5

    def test_content_truncated_to_500_chars(self, fake_kb):
        fake_kb.search_kb.return_value = [
            _kb_entry(content="字" * 800),
        ]
        result = tools_mod.execute_tool(
            "search_campus_knowledge", {"query": "手册"}
        )
        content = result["data"]["results"][0]["content"]
        assert len(content) == 500

    def test_valid_source_passthrough(self, fake_kb):
        fake_kb.search_kb.return_value = [_kb_entry(source="thubook")]
        tools_mod.execute_tool(
            "search_campus_knowledge", {"query": "笔记", "source": "thubook"}
        )
        _, kwargs = fake_kb.search_kb.call_args
        assert kwargs["source"] == "thubook"

    def test_invalid_source_treated_as_none(self, fake_kb):
        fake_kb.search_kb.return_value = [_kb_entry()]
        result = tools_mod.execute_tool(
            "search_campus_knowledge",
            {"query": "选课", "source": "not_a_real_source"},
        )
        assert result["ok"] is True
        _, kwargs = fake_kb.search_kb.call_args
        assert kwargs["source"] is None, "非法 source 应按 None（不限来源）处理"
        assert result["data"]["source"] is None

    def test_limit_clamped(self, fake_kb):
        fake_kb.search_kb.return_value = [_kb_entry()]
        tools_mod.execute_tool(
            "search_campus_knowledge", {"query": "选课", "limit": 999}
        )
        _, kwargs = fake_kb.search_kb.call_args
        assert kwargs["limit"] == 20

    def test_empty_db_returns_ingest_guidance(self, fake_kb):
        fake_kb.search_kb.return_value = []
        fake_kb.stats.return_value = {"total": 0, "by_source": {}}
        result = tools_mod.execute_tool(
            "search_campus_knowledge", {"query": "选课"}
        )
        assert result["ok"] is True  # 外层分发正常，降级语义在 data.ok
        data = result["data"]
        assert data["ok"] is False
        assert "sem_handbook_ingest" in data["note"], (
            f"空库指引应提示回填脚本: {data['note']!r}"
        )
        assert "回填" in data["note"]

    def test_no_hit_on_non_empty_db(self, fake_kb):
        fake_kb.search_kb.return_value = []
        fake_kb.stats.return_value = {"total": 100, "by_source": {"thubook": 100}}
        result = tools_mod.execute_tool(
            "search_campus_knowledge", {"query": "不存在的关键词"}
        )
        data = result["data"]
        assert data["ok"] is True
        assert data["results"] == []
        assert "未检索到" in data["note"]

    def test_module_missing_degrades(self, monkeypatch):
        monkeypatch.setattr(tools_mod, "_get_campus_kb_module", lambda: None)
        result = tools_mod.execute_tool(
            "search_campus_knowledge", {"query": "选课"}
        )
        data = result["data"]
        assert data["ok"] is False
        assert "campus_kb" in data["note"]

    def test_missing_required_query(self):
        result = tools_mod.execute_tool("search_campus_knowledge", {})
        assert result["ok"] is False
        assert "缺少必填参数" in result["error"]

    def test_blank_query_rejected(self, fake_kb):
        result = tools_mod.execute_tool(
            "search_campus_knowledge", {"query": "   "}
        )
        assert result["ok"] is False
        assert "query" in result["error"]


# ════════════════════════════════════════════════════════════════
# 3. get_course_review_summary 工具
# ════════════════════════════════════════════════════════════════

class TestGetCourseReviewSummary:
    def test_cached_summary_hit(self, fake_kb, fake_rs):
        cached = _kb_entry(
            source="thucourse_summary",
            title="数据结构 · 点评综合总结",
            content="基于 12 条点评的自动摘要：给分好， workload 适中。",
            metadata={
                "course_sqid": "sq100", "course_title": "数据结构",
                "rating_avg": 4.2, "rating_dist": {"5": 8, "4": 4},
                "review_count": 12, "highlights": ["给分好"],
                "method": "llm",
            },
        )
        fake_kb.search_kb.side_effect = (
            lambda q, source=None, limit=10, **kw:
            [cached] if source == "thucourse_summary" else []
        )
        result = tools_mod.execute_tool(
            "get_course_review_summary", {"course_query": "数据结构"}
        )
        data = result["data"]
        assert data["ok"] is True
        assert data["course_title"] == "数据结构"
        assert data["sqid"] == "sq100"
        assert data["summary_text"].startswith("基于 12 条点评")
        assert data["rating_avg"] == 4.2
        assert data["rating_dist"] == {"5": 8, "4": 4}
        assert data["review_count"] == 12
        assert data["highlights"] == ["给分好"]
        assert data["method"] == "llm"
        # 现成总结命中时不得触发现场生成与回写
        fake_rs.summarize_course_reviews.assert_not_called()
        fake_kb.upsert_entries.assert_not_called()

    def test_generate_from_reviews_grouped_by_sqid(self, fake_kb, fake_rs):
        def _review(sqid, title, n=1):
            return _kb_entry(
                source="thucourse_review", title=title, content=f"点评{n}",
                metadata={"course_sqid": sqid, "course_title": title},
            )

        reviews = (
            [_review("sq100", "数据结构", i) for i in range(3)]
            + [_review("sq200", "操作系统", 9)]
        )
        fake_kb.search_kb.side_effect = (
            lambda q, source=None, limit=10, **kw:
            reviews if source == "thucourse_review" else []
        )
        result = tools_mod.execute_tool(
            "get_course_review_summary", {"course_query": "数据结构"}
        )
        data = result["data"]
        assert data["ok"] is True
        assert data["sqid"] == "sq100", "应取点评最多的课程分组"
        assert data["course_title"] == "数据结构"
        assert data["review_count"] == 3
        assert data["method"] == "fallback"
        assert data["highlights"] == ["给分好", "作业多", "讲课清晰"]
        # summarize 收到最多点评分组的 3 条点评
        args, _ = fake_rs.summarize_course_reviews.call_args
        assert args[0] == "数据结构"
        assert len(args[1]) == 3
        # 回写：build_summary_entry + upsert 各调一次
        fake_rs.build_summary_entry.assert_called_once()
        fake_kb.upsert_entries.assert_called_once()
        assert any("回写" in n for n in data["notes"])

    def test_nothing_found_chinese_note(self, fake_kb, fake_rs):
        fake_kb.search_kb.return_value = []
        result = tools_mod.execute_tool(
            "get_course_review_summary", {"course_query": "量子神学"}
        )
        data = result["data"]
        assert data["ok"] is False
        assert "量子神学" in data["note"]
        assert "thucourse_crawler" in data["note"]
        fake_rs.summarize_course_reviews.assert_not_called()

    def test_empty_db_returns_ingest_guidance(self, fake_kb, fake_rs):
        fake_kb.search_kb.return_value = []
        fake_kb.stats.return_value = {"total": 0, "by_source": {}}
        result = tools_mod.execute_tool(
            "get_course_review_summary", {"course_query": "数据结构"}
        )
        data = result["data"]
        assert data["ok"] is False
        assert "回填" in data["note"]
        assert "sem_handbook_ingest" in data["note"]

    def test_review_summary_module_missing(self, fake_kb, monkeypatch):
        fake_kb.search_kb.side_effect = (
            lambda q, source=None, limit=10, **kw:
            [_kb_entry(source="thucourse_review",
                       metadata={"course_sqid": "sq100", "course_title": "数据结构"})]
            if source == "thucourse_review" else []
        )
        monkeypatch.setattr(tools_mod, "_get_review_summary_module", lambda: None)
        result = tools_mod.execute_tool(
            "get_course_review_summary", {"course_query": "数据结构"}
        )
        data = result["data"]
        assert data["ok"] is False
        assert "review_summary" in data["note"]

    def test_campus_kb_module_missing(self, monkeypatch):
        monkeypatch.setattr(tools_mod, "_get_campus_kb_module", lambda: None)
        result = tools_mod.execute_tool(
            "get_course_review_summary", {"course_query": "数据结构"}
        )
        data = result["data"]
        assert data["ok"] is False
        assert "campus_kb" in data["note"]

    def test_missing_required_course_query(self):
        result = tools_mod.execute_tool("get_course_review_summary", {})
        assert result["ok"] is False
        assert "缺少必填参数" in result["error"]

    def test_upsert_failure_does_not_break_result(self, fake_kb, fake_rs):
        fake_kb.search_kb.side_effect = (
            lambda q, source=None, limit=10, **kw:
            [_kb_entry(source="thucourse_review",
                       metadata={"course_sqid": "sq100", "course_title": "数据结构"})]
            if source == "thucourse_review" else []
        )
        fake_kb.upsert_entries.side_effect = Exception("db 写失败")
        result = tools_mod.execute_tool(
            "get_course_review_summary", {"course_query": "数据结构"}
        )
        data = result["data"]
        assert data["ok"] is True, "回写失败不得影响本次返回"
        assert data["summary_text"]
        assert any("回写" in n and "失败" in n for n in data["notes"])


# ════════════════════════════════════════════════════════════════
# 4. detect_intent：campus_kb 触发词 + 既有意图回归
# ════════════════════════════════════════════════════════════════

class TestCampusKbIntent:
    @pytest.mark.parametrize(
        "message, keyword",
        [
            ("下学期选课有什么建议", "选课"),
            ("数据结构这门课程怎么样", "课程"),
            ("这门课的点评都在说什么", "点评"),
            ("这门课老师教得好吗", "老师"),
            ("高等微积分给分怎么样", "给分"),
            ("保研需要什么条件", "保研"),
            ("大三出国交换怎么申请", "交换"),
            ("清华绩点怎么算", "绩点"),
            ("留学生公寓怎么申请", "留学"),
            ("转专业需要什么条件", "转专业"),
            ("计算机辅修怎么报名", "辅修"),
            ("奖学金评定标准是什么", "奖学金"),
            ("紫荆宿舍条件怎么样", "宿舍"),
            ("哪个食堂最好吃", "食堂"),
            ("校医院怎么挂号", "校医院"),
            ("军训一般多长时间", "军训"),
            ("校园卡丢了怎么补办", "校园"),
        ],
    )
    def test_campus_keyword_triggers(self, message, keyword):
        intent, sector = detect_intent(message)
        assert intent == "campus_kb", (
            f"校园触发词 {keyword!r} 应判 campus_kb，消息 {message!r} 实际 {intent}"
        )
        assert sector is None

    @pytest.mark.parametrize(
        "message, expected_intent",
        [
            # ── 复盘：含『复盘』二字一律优先复盘（护栏不被校园词抢）──
            ("今日复盘", "market_review"),
            ("复盘", "market_review"),
            ("课程复盘", "market_review"),
            # ── 社媒舆情（强/弱信号路由不变）──
            ("股吧里大家都在讨论什么股票", "social_sentiment"),
            ("今天市场情绪怎么样", "social_sentiment"),
            ("今天涨停多少家", "social_sentiment"),
            # ── 投资人格 ──
            ("用逆向投资的思路看看白酒板块能不能抄底", "persona"),
            # ── 板块深挖 / 个股 / 自选股 / 新闻 / 数据查询 ──
            ("半导体板块怎么样", "sector_deep_dive"),
            ("茅台怎么样", "stock_query"),
            ("我的自选股", "watchlist"),
            ("今天有什么新闻", "news_only"),
            ("今天涨停家数查询", "mcp_query"),
            # ── 纯闲聊 ──
            ("给我讲个笑话", "general_chat"),
            ("红烧肉怎么做", "general_chat"),
        ],
    )
    def test_legacy_intents_not_hijacked(self, message, expected_intent):
        intent, _ = detect_intent(message)
        assert intent == expected_intent, (
            f"消息 {message!r} 期望 {expected_intent}（不被 campus_kb 抢），实际 {intent}"
        )


# ════════════════════════════════════════════════════════════════
# 5. hint 透传接线（参照 social_sentiment 样板）
# ════════════════════════════════════════════════════════════════

class TestCampusKbRoutingHint:
    def test_hint_registered(self):
        assert "campus_kb" in orchestrator._AGENT_ROUTE_HINTS
        hint = orchestrator._AGENT_ROUTE_HINTS["campus_kb"]
        assert hint.startswith("【本问题路由提示】")
        assert "search_campus_knowledge" in hint
        assert "get_course_review_summary" in hint

    @pytest.mark.parametrize("stream", [False, True])
    def test_process_message_routes_with_hint(self, stream):
        agent = _make_agent()
        agent._agent_query = AsyncMock(
            return_value={"role": "assistant", "content": "校园答案"}
        )
        agent._chat = AsyncMock(return_value={"role": "assistant", "content": "闲聊答案"})
        agent._market_review = AsyncMock(
            return_value={"role": "assistant", "content": "复盘答案"}
        )

        result = asyncio.run(agent.process_message("保研需要什么条件", stream=stream))

        assert agent._agent_query.await_count == 1, (
            f"campus_kb 意图应路由到 _agent_query（stream={stream}）"
        )
        assert agent._chat.await_count == 0, "campus_kb 不应落入纯闲聊 _chat"
        assert agent._market_review.await_count == 0, "campus_kb 不应被劫持去复盘"
        _, kwargs = agent._agent_query.await_args
        hint = kwargs.get("hint")
        assert hint == orchestrator._AGENT_ROUTE_HINTS["campus_kb"], (
            f"hint 应为 _AGENT_ROUTE_HINTS['campus_kb'] 原文: {kwargs}"
        )
        assert result["content"] == "校园答案"


# ════════════════════════════════════════════════════════════════
# 6. system_prompts：校园知识库引用规范
# ════════════════════════════════════════════════════════════════

class TestCampusKbCitationPrompt:
    def test_citation_section_exists(self):
        assert "## 校园知识库引用规范" in AGENT_QUERY_PROMPT

    def test_key_rules_present(self):
        assert "基于 N 条学生点评的自动摘要" in AGENT_QUERY_PROMPT
        assert "官方政策" in AGENT_QUERY_PROMPT, (
            "必须含『不得把点评个例当作官方政策陈述』纪律"
        )
        assert "以最新官方通知为准" in AGENT_QUERY_PROMPT, "必须含信息时效提示"
        assert "校园知识库暂未覆盖" in AGENT_QUERY_PROMPT
        # 既有引用规范不被破坏
        assert "## 社媒舆情引用规范" in AGENT_QUERY_PROMPT
        assert "## 研报引用规范" in AGENT_QUERY_PROMPT
