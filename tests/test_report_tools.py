"""tests/test_report_tools.py — 研报库工具接线专项（全 mock 零网络）。

覆盖范围（研报库 v1，tools.py 接线）：
1. TOOL_REGISTRY 两新工具 schema 断言：名称/参数属性/类型/required 为空、
   days/limit 边界与默认值。
2. _REPORT_IMPL 映射契约：指向 report_library 公开函数名
   （search_reports / rating_summary，全局契约第 3 条）。
3. execute_tool 经 monkeypatch 替换 _get_report_library 为 fake 模块后的
   成功路径：验证函数名与 kwargs 适配正确（stock_code 前缀归一、
   days/limit 夹取）、ok/data 结构。
4. 异常降级：report_library 函数抛异常 → ok=False 不抛异常；
   _get_report_library 返回 None → ok=False「研报库模块不可用」；
   函数缺失 → ok=False「研报库接口暂不可用」。
5. stock_code 前缀归一（sh/sz/bj/大小写）与 days(1-365)/limit(1-50) 夹取。
6. search 三条件全空 → ok=False「至少提供一个检索条件」，且不调后端函数。
7. 未注册工具报错保持原行为（「未注册的工具」）。
8. prompt 断言：AGENT_QUERY_PROMPT 含「研报引用规范」一节及
   「研报库暂未覆盖」「评级」关键句，且位于「## 输出结构」之前。
9. get_tool_catalog() 行数断言（v1 落地时 23 行；v2 全文工具接入后由
   tests/test_report_content_tool.py 断言 24 行）。

规则（与项目其他测试一致）：
- report_library 模块由研报库工作线独立交付，本文件一律不 import 实体，
  只经 monkeypatch 注入 fake 模块——契约编程，零网络零文件读写。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import agent.tools as tools_mod
from agent import system_prompts


# ── fake 研报库模块：按全局契约返回结构，不触网 ──

_FAKE_SEARCH_RESULT = {
    "total": 2,
    "reports": [
        {
            "title": "贵州茅台：业绩稳健增长", "org": "中金公司", "author": "张三",
            "date": "2026-07-20", "rating": "买入", "rating_change": "维持",
            "target_price": "1800.0~2100.0", "eps_forecast": 68.5,
            "eps_next_year": 75.2, "stock_code": "600519",
            "stock_name": "贵州茅台", "industry": "食品饮料", "source": "eastmoney",
        },
        {
            "title": "白酒行业三季度跟踪", "org": "华泰证券", "author": "李四",
            "date": "2026-07-18", "rating": "增持", "rating_change": None,
            "target_price": None, "eps_forecast": None, "eps_next_year": None,
            "stock_code": "", "stock_name": "", "industry": "食品饮料",
            "source": "stockstar",
        },
    ],
}

_FAKE_SUMMARY_RESULT = {
    "total": 8,
    "rating_dist": {"买入": 6, "增持": 2},
    "target_price_range": [1800.0, 2100.0],
    "avg_eps_forecast": 68.5,
    "latest_reports": [
        {"title": "贵州茅台：业绩稳健增长", "org": "中金公司", "date": "2026-07-20"},
        {"title": "食品饮料板块配置价值分析", "org": "国泰君安", "date": "2026-07-19"},
        {"title": "白酒行业三季度跟踪", "org": "华泰证券", "date": "2026-07-18"},
    ],
}


def _fake_report_library(search_side_effect=None, summary_side_effect=None):
    """构造 fake report_library 模块（契约函数用 MagicMock 记录调用）。"""
    search = MagicMock(return_value=dict(_FAKE_SEARCH_RESULT))
    if search_side_effect is not None:
        search.side_effect = search_side_effect
    summary = MagicMock(return_value=dict(_FAKE_SUMMARY_RESULT))
    if summary_side_effect is not None:
        summary.side_effect = summary_side_effect
    return SimpleNamespace(search_reports=search, rating_summary=summary)


def _patch_report_library(monkeypatch, fake):
    """把 tools._get_report_library 替换为返回 fake 模块的桩。"""
    monkeypatch.setattr(tools_mod, "_get_report_library", lambda: fake)


def _registry_entry(name: str) -> dict:
    """按工具名从 TOOL_REGISTRY 取出 function 块。"""
    for tool in tools_mod.TOOL_REGISTRY:
        fn = tool.get("function", {})
        if fn.get("name") == name:
            return fn
    raise AssertionError(f"TOOL_REGISTRY 中找不到工具 {name!r}")


# ════════════════════════════════════════════════
# 1. TOOL_REGISTRY 两新工具 schema
# ════════════════════════════════════════════════

class TestReportToolRegistry:
    def test_search_tool_schema(self):
        fn = _registry_entry("search_research_reports")
        # description 写清何时用 + 三条件至少其一的运行时约束
        assert "研报" in fn["description"]
        assert "至少提供一个" in fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert params["required"] == []
        props = params["properties"]
        assert set(props) == {"query", "stock_code", "industry", "days", "limit"}
        assert props["query"]["type"] == "string"
        assert props["stock_code"]["type"] == "string"
        assert props["industry"]["type"] == "string"
        assert props["days"]["type"] == "integer"
        assert props["limit"]["type"] == "integer"
        # 边界与默认值照全局契约第 6 条
        assert props["days"]["minimum"] == 1
        assert props["days"]["maximum"] == 365
        assert props["days"]["default"] == 30
        assert props["limit"]["minimum"] == 1
        assert props["limit"]["maximum"] == 50
        assert props["limit"]["default"] == 10

    def test_summary_tool_schema(self):
        fn = _registry_entry("get_rating_summary")
        assert "评级" in fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert params["required"] == []
        props = params["properties"]
        assert set(props) == {"stock_code", "industry", "days"}
        assert props["stock_code"]["type"] == "string"
        assert props["industry"]["type"] == "string"
        assert props["days"]["type"] == "integer"
        assert props["days"]["minimum"] == 1
        assert props["days"]["maximum"] == 365
        assert props["days"]["default"] == 30


# ════════════════════════════════════════════════
# 2. _REPORT_IMPL 映射契约
# ════════════════════════════════════════════════

class TestReportImplMapping:
    def test_mapping_points_to_contract_functions(self):
        """映射值必须是全局契约第 3 条的 report_library 公开函数名。"""
        assert tools_mod._REPORT_IMPL["search_research_reports"][0] == "search_reports"
        assert tools_mod._REPORT_IMPL["get_rating_summary"][0] == "rating_summary"
        for name, entry in tools_mod._REPORT_IMPL.items():
            assert callable(entry[1]), f"{name} 缺少参数适配器"

    def test_report_tools_not_in_data_impl(self):
        """研报工具不进数据层映射表，避免误走 data_fetcher。"""
        assert "search_research_reports" not in tools_mod._IMPL
        assert "get_rating_summary" not in tools_mod._IMPL


# ════════════════════════════════════════════════
# 3. execute_tool 成功路径（fake 模块注入）
# ════════════════════════════════════════════════

class TestReportToolDispatch:
    def test_search_success_path(self, monkeypatch):
        fake = _fake_report_library()
        _patch_report_library(monkeypatch, fake)
        result = tools_mod.execute_tool(
            "search_research_reports", {"query": "茅台", "days": 15, "limit": 5}
        )
        assert result["ok"] is True
        assert result["data"] == _FAKE_SEARCH_RESULT
        fake.search_reports.assert_called_once_with(
            query="茅台", stock_code="", industry="", days=15, limit=5
        )
        fake.rating_summary.assert_not_called()

    def test_summary_success_path(self, monkeypatch):
        fake = _fake_report_library()
        _patch_report_library(monkeypatch, fake)
        result = tools_mod.execute_tool(
            "get_rating_summary", {"stock_code": "SH600519", "days": 90}
        )
        assert result["ok"] is True
        assert result["data"] == _FAKE_SUMMARY_RESULT
        fake.rating_summary.assert_called_once_with(
            stock_code="600519", industry="", days=90
        )
        fake.search_reports.assert_not_called()

    def test_stock_code_prefix_normalized(self, monkeypatch):
        """sh/sz/bj 前缀与大小写都归一为 6 位纯数字。"""
        fake = _fake_report_library()
        _patch_report_library(monkeypatch, fake)
        for raw, normalized in (
            ("sh600519", "600519"),
            ("SH600519", "600519"),
            ("sz000002", "000002"),
            ("SZ000002", "000002"),
            ("bj920001", "920001"),
            ("600519", "600519"),
        ):
            fake.search_reports.reset_mock()
            result = tools_mod.execute_tool(
                "search_research_reports", {"stock_code": raw}
            )
            assert result["ok"] is True, f"{raw}: {result}"
            assert fake.search_reports.call_args.kwargs["stock_code"] == normalized, (
                f"{raw} 应归一为 {normalized}，"
                f"实际 {fake.search_reports.call_args.kwargs['stock_code']!r}"
            )

    def test_days_limit_clamped(self, monkeypatch):
        """days 夹 1-365 默认 30；limit 夹 1-50 默认 10；非法值回默认。"""
        fake = _fake_report_library()
        _patch_report_library(monkeypatch, fake)
        cases = [
            # (args, 期望 days, 期望 limit)
            ({"query": "茅台"}, 30, 10),                      # 缺省
            ({"query": "茅台", "days": 0, "limit": 0}, 1, 1),  # 下界夹取
            ({"query": "茅台", "days": 999, "limit": 500}, 365, 50),  # 上界夹取
            ({"query": "茅台", "days": "abc", "limit": None}, 30, 10),  # 非法回默认
            ({"query": "茅台", "days": "7", "limit": "20"}, 7, 20),  # 数字字符串兼容
        ]
        for args, want_days, want_limit in cases:
            fake.search_reports.reset_mock()
            result = tools_mod.execute_tool("search_research_reports", args)
            assert result["ok"] is True, f"{args}: {result}"
            kwargs = fake.search_reports.call_args.kwargs
            assert kwargs["days"] == want_days, f"{args}: days={kwargs['days']}"
            assert kwargs["limit"] == want_limit, f"{args}: limit={kwargs['limit']}"

    def test_summary_defaults_and_clamp(self, monkeypatch):
        fake = _fake_report_library()
        _patch_report_library(monkeypatch, fake)
        result = tools_mod.execute_tool("get_rating_summary", {})
        assert result["ok"] is True
        fake.rating_summary.assert_called_once_with(stock_code="", industry="", days=30)
        fake.rating_summary.reset_mock()
        result = tools_mod.execute_tool("get_rating_summary", {"industry": "半导体", "days": 9999})
        assert result["ok"] is True
        fake.rating_summary.assert_called_once_with(
            stock_code="", industry="半导体", days=365
        )

    def test_search_args_as_json_string(self, monkeypatch):
        """模型把 arguments 序列化成字符串时照常分发。"""
        fake = _fake_report_library()
        _patch_report_library(monkeypatch, fake)
        result = tools_mod.execute_tool("search_research_reports", '{"query": "半导体"}')
        assert result["ok"] is True
        fake.search_reports.assert_called_once_with(
            query="半导体", stock_code="", industry="", days=30, limit=10
        )


# ════════════════════════════════════════════════
# 4/6. 参数校验与异常降级
# ════════════════════════════════════════════════

class TestReportToolValidationAndDegrade:
    def test_search_requires_at_least_one_condition(self, monkeypatch):
        fake = _fake_report_library()
        _patch_report_library(monkeypatch, fake)
        result = tools_mod.execute_tool("search_research_reports", {})
        assert result["ok"] is False
        assert "至少提供一个检索条件" in result["error"]
        fake.search_reports.assert_not_called()

    def test_search_blank_conditions_rejected(self, monkeypatch):
        """空白串/None/非法代码清洗后等同全空，同样拒绝。"""
        fake = _fake_report_library()
        _patch_report_library(monkeypatch, fake)
        result = tools_mod.execute_tool(
            "search_research_reports",
            {"query": "   ", "stock_code": "sh123", "industry": None},
        )
        assert result["ok"] is False
        assert "至少提供一个检索条件" in result["error"]
        fake.search_reports.assert_not_called()

    def test_report_module_unavailable(self, monkeypatch):
        """_get_report_library 返回 None（ImportError 降级）→ ok=False，不抛异常。"""
        monkeypatch.setattr(tools_mod, "_get_report_library", lambda: None)
        for name in ("search_research_reports", "get_rating_summary"):
            result = tools_mod.execute_tool(name, {"query": "茅台"})
            assert result["ok"] is False, f"{name}: {result}"
            assert result["error"] == "研报库模块不可用", f"{name}: {result}"

    def test_report_function_missing(self, monkeypatch):
        """fake 模块没有契约函数 → ok=False「研报库接口暂不可用」。"""
        _patch_report_library(monkeypatch, SimpleNamespace())
        result = tools_mod.execute_tool("search_research_reports", {"query": "茅台"})
        assert result["ok"] is False
        assert "研报库接口暂不可用" in result["error"]
        assert "search_reports" in result["error"]

    def test_report_function_raises_safe_degrade(self, monkeypatch):
        """后端函数抛异常 → ok=False + error，绝不向上抛。"""
        fake = _fake_report_library(search_side_effect=Exception("数据库炸了"))
        _patch_report_library(monkeypatch, fake)
        result = tools_mod.execute_tool("search_research_reports", {"query": "茅台"})
        assert result["ok"] is False
        assert "工具执行出错" in result["error"]
        fake_summary = _fake_report_library(summary_side_effect=ValueError("库损坏"))
        _patch_report_library(monkeypatch, fake_summary)
        result = tools_mod.execute_tool("get_rating_summary", {"industry": "电子"})
        assert result["ok"] is False
        assert "工具执行出错" in result["error"]

    def test_unknown_tool_keeps_original_behavior(self, monkeypatch):
        """未注册工具报错保持原行为（两表都不命中才报未注册）。"""
        fake = _fake_report_library()
        _patch_report_library(monkeypatch, fake)
        result = tools_mod.execute_tool("__definitely_not_a_tool__", {})
        assert result["ok"] is False
        assert "未注册的工具" in result["error"]
        fake.search_reports.assert_not_called()
        fake.rating_summary.assert_not_called()

    def test_data_tools_unaffected(self, monkeypatch):
        """既有数据工具仍走 _IMPL + data_fetcher，不受研报映射影响。"""
        fetch = MagicMock(return_value={"indices": []})
        monkeypatch.setattr("agent.data_fetcher.fetch_a_share_indices", fetch)
        result = tools_mod.execute_tool("get_market_indices", {"date": "20260722"})
        assert result["ok"] is True
        fetch.assert_called_once_with(date="20260722")


# ════════════════════════════════════════════════
# 8. prompt 断言
# ════════════════════════════════════════════════

class TestReportCitationPrompt:
    def test_citation_section_present(self):
        prompt = system_prompts.AGENT_QUERY_PROMPT
        assert "## 研报引用规范" in prompt
        # 图纸「Prompt 引用规范」4 条的语义落点
        assert "研报库暂未覆盖" in prompt
        assert "评级" in prompt
        assert "目标价" in prompt
        assert "盈利预测" in prompt
        assert "search_research_reports" in prompt
        assert "get_rating_summary" in prompt
        assert "不得编造券商观点" in prompt

    def test_citation_section_before_output_structure(self):
        """新增节必须插在「## 输出结构」之前，其余段落不动。"""
        prompt = system_prompts.AGENT_QUERY_PROMPT
        assert prompt.index("## 研报引用规范") < prompt.index("## 输出结构")
        # 原有结构标记一个不少（防误删）
        for marker in (
            "## 工作规则（按此流程走，不得跳步）",
            "## 数据真实性红线（最高优先级，违反任何一条都算失败）",
            "## 语言风格（像人写的，不像AI写的）",
            "## 合规边界",
            "## 输出结构",
            "## 输出格式",
        ):
            assert marker in prompt, f"AGENT_QUERY_PROMPT 原结构标记 {marker!r} 丢失"


# ════════════════════════════════════════════════
# 9. get_tool_catalog() 行数（v1 落地时 23；v2 全文工具接入后 24；
#    第十二波开源灵感模块工具接入后 28；社媒舆情接线后 30，
#    30 行专项断言见 tests/test_social_tools.py）
# ════════════════════════════════════════════════

class TestToolCatalog:
    def test_catalog_32_lines(self):
        assert len(tools_mod.TOOL_REGISTRY) == 32
        catalog = tools_mod.get_tool_catalog()
        assert "共 32 个" in catalog
        tool_lines = [l for l in catalog.splitlines() if l.startswith("- ")]
        assert len(tool_lines) == 32, (
            f"目录应有 32 行工具条目，实际 {len(tool_lines)} 行：\n{catalog}"
        )

    def test_new_tools_in_short_desc_and_catalog(self):
        assert "search_research_reports" in tools_mod._SHORT_DESC
        assert "get_rating_summary" in tools_mod._SHORT_DESC
        catalog = tools_mod.get_tool_catalog()
        assert "search_research_reports" in catalog
        assert "get_rating_summary" in catalog
        assert tools_mod._SHORT_DESC["search_research_reports"] in catalog
        assert tools_mod._SHORT_DESC["get_rating_summary"] in catalog
