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
        assert len(tools_mod.TOOL_REGISTRY) == 55, (
            f"注册表应为 55 个工具（37→54 智研 MCP 工具），实际 {len(tools_mod.TOOL_REGISTRY)}"
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
        assert limit["default"] == 10

    def test_get_course_review_summary_schema(self):
        tool = next(t for t in tools_mod.TOOL_REGISTRY
                    if t["function"]["name"] == "get_course_review_summary")
        params = tool["function"]["parameters"]
        assert params["required"] == ["course_query"]
        assert "course_query" in params["properties"]

    def test_tool_catalog_lists_new_tools(self):
        catalog = tools_mod.get_tool_catalog()
        assert "共 55 个" in catalog
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
        assert kwargs["source"] is None and kwargs["limit"] == 10

    def test_content_truncated_to_500_chars(self, fake_kb):
        fake_kb.search_kb.return_value = [
            _kb_entry(content="字" * 2000),
        ]
        result = tools_mod.execute_tool(
            "search_campus_knowledge", {"query": "手册"}
        )
        content = result["data"]["results"][0]["content"]
        assert len(content) == 1500  # 句对齐截断上限 1500（无标点时硬切）

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


# ════════════════════════════════════════════════════════════════
# QA 全题库质保第二轮系统性修复（2026-07-23）：
# 路由词表扩容 / 『排名』类指令词解封 / 金融信号让位 /
# general_chat 校园兜底探针 / 风险提示按意图分流
# ════════════════════════════════════════════════════════════════


class TestCampusRoutingExpansion:
    """QA 实测漏判问法回归：扩词后必须判 campus_kb。"""

    @pytest.mark.parametrize("q", [
        "成绩排名多少才能保研经管？",       # 『排名』指令词不得再封锁校园路由
        "想出国读研读博，手册里有什么建议？",
        "国际生实习加注怎么办理？",
        "经管本科有哪些必修课？",
        "经管的培养方案是怎样的？",
        "体育课选不上怎么办？",
        "商法学这门课怎么样？",
        "转化医学工程值得选吗？",
        "商法学难吗？期末怎么考？",
        "体测都测什么？不及格怎么办？",
        "清华的GPA怎么算？",
        "校内哪里可以自习？",
        "紫荆公寓是几人间？",
        "校园卡丢了怎么办？",
        "数据科学导论是谁开的课？",
    ])
    def test_expanded_keywords_route_campus(self, q):
        assert detect_intent(q)[0] == "campus_kb"

    @pytest.mark.parametrize("q", [
        "清华系持股的股票有哪些？",   # 金融信号共现让位，不得被『清华』劫持
        "紫金矿业行情怎么样？",       # 金融语境（『紫荆』谐音误差伤防护）
        "今天A股大盘怎么样？",         # 金融主线回归
        "帮我查一下茅台的估值",
        "今日复盘",
    ])
    def test_finance_signal_not_hijacked(self, q):
        assert detect_intent(q)[0] != "campus_kb"


class TestCampusFallbackProbe:
    """_campus_fallback_hit：top1 命中 ≥2 关键词才改道，异常静默 False。"""

    def _fake_kb(self, monkeypatch, results):
        import agent.campus_kb as campus_kb
        monkeypatch.setattr(campus_kb, "search_kb",
                            lambda *a, **kw: results)

    def test_hit_when_two_keywords_matched(self, monkeypatch):
        self._fake_kb(monkeypatch, [{
            "source": "thubook", "source_id": "thubook:dorm",
            "title": "宿舍介绍 - 紫荆公寓",
            "content": "紫荆公寓为4人间上床下桌，两间共用一个中厅。",
        }])
        assert orchestrator._campus_fallback_hit("紫荆公寓是几人间") is True

    def test_miss_when_no_results(self, monkeypatch):
        self._fake_kb(monkeypatch, [])
        assert orchestrator._campus_fallback_hit("茅台走势如何") is False

    def test_miss_when_single_keyword_hit(self, monkeypatch):
        self._fake_kb(monkeypatch, [{
            "source": "thubook", "source_id": "t:1",
            "title": "北京旅游推荐",
            "content": "故宫和长城。",
        }])
        assert orchestrator._campus_fallback_hit("北京哪里好玩") is False

    def test_exception_returns_false(self, monkeypatch):
        import agent.campus_kb as campus_kb
        def _boom(*a, **kw):
            raise RuntimeError("db gone")
        monkeypatch.setattr(campus_kb, "search_kb", _boom)
        assert orchestrator._campus_fallback_hit("宿舍怎么样") is False


class TestCampusFallbackWiring:
    """process_message 兜底接线：general_chat 命中探针 → campus_kb 工具链，
    且校园回答不附金融风险提示。"""

    def test_general_chat_rerouted_to_campus(self, monkeypatch):
        agent = _make_agent()
        agent._agent_query = AsyncMock(
            return_value={"role": "assistant", "content": "校园答案"}
        )
        agent._chat = AsyncMock(return_value={"role": "assistant", "content": "闲聊"})
        monkeypatch.setattr(orchestrator, "_campus_fallback_hit", lambda m: True)
        r = asyncio.run(agent.process_message("一个关键词表外的校园问题"))
        assert r["content"] == "校园答案"
        agent._chat.assert_not_called()
        _, kwargs = agent._agent_query.call_args
        assert kwargs.get("hint") == orchestrator._AGENT_ROUTE_HINTS["campus_kb"]
        assert kwargs.get("disclaimer") is False  # 校园回答不附金融风险提示

    def test_probe_miss_keeps_general_chat(self, monkeypatch):
        agent = _make_agent()
        agent._chat = AsyncMock(return_value={"role": "assistant", "content": "闲聊"})
        monkeypatch.setattr(orchestrator, "_campus_fallback_hit", lambda m: False)
        r = asyncio.run(agent.process_message("今天天气不错"))
        assert r["content"] == "闲聊"


class TestDisclaimerGating:
    """风险提示分流：闲聊与校园路径不追加，金融 Agent 路径保持追加。"""

    def test_chat_passes_disclaimer_false(self):
        agent = _make_agent()
        captured = {}
        async def fake_call(system, message, stream, history=None, **kw):
            captured.update(kw)
            return {"role": "assistant", "content": "ok"}
        agent._call_llm = fake_call
        asyncio.run(agent._chat("你好", False))
        assert captured.get("disclaimer") is False

    def test_agent_query_disclaimer_param_default_true(self):
        import inspect
        sig = inspect.signature(MarketReviewAgent._agent_query)
        assert sig.parameters["disclaimer"].default is True

    def test_call_llm_disclaimer_param_default_true(self):
        import inspect
        sig = inspect.signature(MarketReviewAgent._call_llm)
        assert sig.parameters["disclaimer"].default is True


class TestAgentLoopExhaustion:
    """工具循环超轮：基于已检索成果强制成文，不再丢弃上下文降级闲聊。"""

    def test_forced_answer_on_round_limit(self, monkeypatch):
        agent = _make_agent()
        calls = []

        tool_msg = MagicMock()
        tool_msg.tool_calls = [MagicMock(
            id="tc1",
            function=MagicMock(name="search_campus_knowledge",
                               arguments='{"query": "必修课"}'),
        )]
        tool_msg.content = None
        final_msg = MagicMock()
        final_msg.tool_calls = None
        final_msg.content = "基于已检索信息的回答"

        def make_completion(msg):
            comp = MagicMock()
            comp.choices = [MagicMock(message=msg, finish_reason="stop")]
            return comp

        async def fake_create(**kw):
            calls.append(kw)
            return make_completion(tool_msg if len(calls) <= 8 else final_msg)

        agent.client.chat.completions.create = fake_create
        monkeypatch.setattr(
            orchestrator, "execute_tool",
            lambda *a, **kw: {"ok": True, "results": []},
        )
        r = asyncio.run(agent._agent_query("经管本科有哪些必修课？", False))
        assert "基于已检索信息" in r["content"]
        assert len(calls) == 9  # 8 轮循环 + 1 次强制成文
        assert "tools" not in calls[-1]  # 强制成文禁止再调工具
        assert any(
            isinstance(m, dict) and "工具调用次数已用完" in str(m.get("content", ""))
            for m in calls[-1]["messages"]
        )


class TestContentCutAndMetaStrip:
    """第三轮修复：句对齐截断（硬事实不再拦腰切断）+ 元推理开头剥除。"""

    def test_cut_campus_content_sentence_aligned(self):
        text = "前提条件说明。" + "x" * 900 + "。材料递交后，需7个工作日可以拿到护照。" + "y" * 800
        out = tools_mod._cut_campus_content(text)
        # 截断点落在句末标点处，且长度不超上限
        assert len(out) <= 1500
        assert out.endswith(("。", "！", "？", "；", "\n"))
        # 句对齐窗口覆盖到 7 个工作日所在的完整句
        assert "7个工作日" in out

    def test_cut_campus_content_short_text_unchanged(self):
        assert tools_mod._cut_campus_content("短文本。") == "短文本。"
        assert tools_mod._cut_campus_content("") == ""

    def test_cut_campus_content_no_good_breakpoint_hard_cut(self):
        text = "一" * 2000  # 无句末标点 → 硬切
        out = tools_mod._cut_campus_content(text)
        assert len(out) == 1500

    @pytest.mark.parametrize("opening", [
        "数据已经够了。",
        "现有数据已经足够回答用户的问题了。",
        "信息已经足够了，",
        "下面整理回答。",
    ])
    def test_strip_meta_openings(self, opening):
        text = opening + "【结论】紫荆公寓为4人间。"
        assert orchestrator._strip_meta_openings(text) == "【结论】紫荆公寓为4人间。"

    def test_strip_meta_openings_preserves_body(self):
        text = "紫荆公寓数据已经够了4人间标准配置。"
        assert orchestrator._strip_meta_openings(text) == text  # 非开头不动
