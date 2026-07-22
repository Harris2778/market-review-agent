"""tests/test_report_content_tool.py — 研报全文向量检索工具接线专项（全 mock 零网络）。

覆盖范围（研报库 v2，tools.py 接线，全局契约第 5/6 条）：
1. TOOL_REGISTRY search_report_content schema 断言：query 必填、参数类型、
   days(1-365 默认 90)/top_k(1-10 默认 5) 边界与默认值、description 写清使用时机。
2. _REPORT_VEC_IMPL 映射契约：指向 report_vectors.search_vectors；
   新工具不进 _REPORT_IMPL / _IMPL。
3. execute_tool 经 monkeypatch 替换 _get_report_vectors 为 fake 后的成功路径：
   kwargs 适配正确（stock_code 前缀归一、query 透传）、search_vectors 结果
   原样包进 {"ok": True, "data": ...}。
4. days 夹 1-365 默认 90、top_k 夹 1-10 默认 5（非法值回默认）。
5. query 缺失 → ok=False「缺少必填参数」，且不调后端函数。
6. 三条失败路径：模块不可用 → ok=False「研报全文模块不可用」；
   空 hits + note → ok=False 原样透传 note；后端抛异常 → ok=False 不抛出。
7. prompt 断言：AGENT_QUERY_PROMPT 研报引用规范内含「券商+日期+研报标题」
   与「search_report_content」「一律视为编造」，且仍在「## 输出结构」之前。
8. get_tool_catalog() 变为 24 行；既有 23 个工具不受影响。

规则（与项目其他测试一致）：
- report_vectors 模块由向量检索工作线独立交付，本文件一律不 import 实体，
  只经 monkeypatch 注入 fake 模块——契约编程，零网络零文件读写。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import agent.tools as tools_mod
from agent import system_prompts


# ── fake 向量检索模块：按全局契约第 5 条返回结构，不触网 ──

_FAKE_HITS_RESULT = {
    "total_chunks": 42,
    "hits": [
        {
            "info_code": "AP202607201234567890",
            "title": "贵州茅台：业绩稳健增长",
            "org": "中金公司",
            "date": "2026-07-20",
            "rating": "买入",
            "section": "投资要点",
            "snippet": "公司渠道改革推进顺利，直营占比持续提升……",
            "score": 0.91,
        },
        {
            "info_code": "AP202607180987654321",
            "title": "白酒行业三季度跟踪",
            "org": "华泰证券",
            "date": "2026-07-18",
            "rating": "增持",
            "section": "正文",
            "snippet": "板块估值处于近一年低位，动销分化加剧……",
            "score": 0.83,
        },
    ],
}


def _fake_report_vectors(side_effect=None, return_value=None):
    """构造 fake report_vectors 模块（契约函数用 MagicMock 记录调用）。"""
    search = MagicMock(return_value=return_value or dict(_FAKE_HITS_RESULT))
    if side_effect is not None:
        search.side_effect = side_effect
    return SimpleNamespace(search_vectors=search)


def _patch_report_vectors(monkeypatch, fake):
    """把 tools._get_report_vectors 替换为返回 fake 模块的桩。"""
    monkeypatch.setattr(tools_mod, "_get_report_vectors", lambda: fake)


def _registry_entry(name: str) -> dict:
    """按工具名从 TOOL_REGISTRY 取出 function 块。"""
    for tool in tools_mod.TOOL_REGISTRY:
        fn = tool.get("function", {})
        if fn.get("name") == name:
            return fn
    raise AssertionError(f"TOOL_REGISTRY 中找不到工具 {name!r}")


# ════════════════════════════════════════════════
# 1. TOOL_REGISTRY schema
# ════════════════════════════════════════════════

class TestReportContentToolRegistry:
    def test_schema_query_required_and_types(self):
        fn = _registry_entry("search_report_content")
        # description 写清何时用（正文/观点细节/分歧/引用）
        assert "研报" in fn["description"]
        assert "正文" in fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert params["required"] == ["query"]
        props = params["properties"]
        assert set(props) == {"query", "stock_code", "industry", "days", "top_k"}
        assert props["query"]["type"] == "string"
        assert props["stock_code"]["type"] == "string"
        assert props["industry"]["type"] == "string"
        assert props["days"]["type"] == "integer"
        assert props["top_k"]["type"] == "integer"

    def test_schema_days_topk_bounds(self):
        """days 夹 1-365 默认 90；top_k 夹 1-10 默认 5（全局契约第 6 条）。"""
        props = _registry_entry("search_report_content")["parameters"]["properties"]
        assert props["days"]["minimum"] == 1
        assert props["days"]["maximum"] == 365
        assert props["days"]["default"] == 90
        assert props["top_k"]["minimum"] == 1
        assert props["top_k"]["maximum"] == 10
        assert props["top_k"]["default"] == 5


# ════════════════════════════════════════════════
# 2. _REPORT_VEC_IMPL 映射契约
# ════════════════════════════════════════════════

class TestReportVecImplMapping:
    def test_mapping_points_to_contract_function(self):
        """映射值必须是全局契约第 5 条的 report_vectors.search_vectors。"""
        entry = tools_mod._REPORT_VEC_IMPL["search_report_content"]
        assert entry[0] == "search_vectors"
        assert callable(entry[1]), "search_report_content 缺少参数适配器"

    def test_tool_not_in_other_impl_tables(self):
        """新工具不进研报元数据表与数据层表，查表顺序 _REPORT_VEC_IMPL 优先。"""
        assert "search_report_content" not in tools_mod._REPORT_IMPL
        assert "search_report_content" not in tools_mod._IMPL


# ════════════════════════════════════════════════
# 3. execute_tool 成功路径（fake 模块注入）
# ════════════════════════════════════════════════

class TestReportContentDispatch:
    def test_success_path_kwargs_and_passthrough(self, monkeypatch):
        fake = _fake_report_vectors()
        _patch_report_vectors(monkeypatch, fake)
        result = tools_mod.execute_tool(
            "search_report_content",
            {"query": "渠道改革进展", "stock_code": "SH600519",
             "industry": "食品饮料", "days": 60, "top_k": 3},
        )
        assert result["ok"] is True
        # search_vectors 结果原样透传
        assert result["data"] == _FAKE_HITS_RESULT
        fake.search_vectors.assert_called_once_with(
            query="渠道改革进展", stock_code="600519",
            industry="食品饮料", days=60, top_k=3,
        )

    def test_stock_code_prefix_normalized(self, monkeypatch):
        """sh/sz/bj 前缀与大小写都归一为 6 位纯数字；非法代码归空串。"""
        fake = _fake_report_vectors()
        _patch_report_vectors(monkeypatch, fake)
        for raw, normalized in (
            ("sh600519", "600519"),
            ("SZ000002", "000002"),
            ("bj920001", "920001"),
            ("600519", "600519"),
            ("sh123", ""),
        ):
            fake.search_vectors.reset_mock()
            result = tools_mod.execute_tool(
                "search_report_content", {"query": "茅台", "stock_code": raw}
            )
            assert result["ok"] is True, f"{raw}: {result}"
            assert fake.search_vectors.call_args.kwargs["stock_code"] == normalized

    def test_args_as_json_string(self, monkeypatch):
        """模型把 arguments 序列化成字符串时照常分发。"""
        fake = _fake_report_vectors()
        _patch_report_vectors(monkeypatch, fake)
        result = tools_mod.execute_tool(
            "search_report_content", '{"query": "国产替代逻辑"}'
        )
        assert result["ok"] is True
        fake.search_vectors.assert_called_once_with(
            query="国产替代逻辑", stock_code="", industry="", days=90, top_k=5
        )


# ════════════════════════════════════════════════
# 4. days / top_k 夹取
# ════════════════════════════════════════════════

class TestReportContentClamping:
    def test_days_topk_clamped(self, monkeypatch):
        fake = _fake_report_vectors()
        _patch_report_vectors(monkeypatch, fake)
        cases = [
            # (args, 期望 days, 期望 top_k)
            ({"query": "茅台"}, 90, 5),                          # 缺省
            ({"query": "茅台", "days": 0, "top_k": 0}, 1, 1),    # 下界夹取
            ({"query": "茅台", "days": 999, "top_k": 99}, 365, 10),  # 上界夹取
            ({"query": "茅台", "days": "abc", "top_k": None}, 90, 5),  # 非法回默认
            ({"query": "茅台", "days": "30", "top_k": "7"}, 30, 7),  # 数字字符串兼容
        ]
        for args, want_days, want_top_k in cases:
            fake.search_vectors.reset_mock()
            result = tools_mod.execute_tool("search_report_content", args)
            assert result["ok"] is True, f"{args}: {result}"
            kwargs = fake.search_vectors.call_args.kwargs
            assert kwargs["days"] == want_days, f"{args}: days={kwargs['days']}"
            assert kwargs["top_k"] == want_top_k, f"{args}: top_k={kwargs['top_k']}"


# ════════════════════════════════════════════════
# 5/6. 参数校验与三条失败路径
# ════════════════════════════════════════════════

class TestReportContentValidationAndDegrade:
    def test_query_missing_rejected(self, monkeypatch):
        """query 为必填：缺失/空串 → ok=False，且不调后端函数。"""
        fake = _fake_report_vectors()
        _patch_report_vectors(monkeypatch, fake)
        for args in ({}, {"query": ""}, {"query": None, "stock_code": "600519"}):
            result = tools_mod.execute_tool("search_report_content", args)
            assert result["ok"] is False, f"{args}: {result}"
            assert "缺少必填参数" in result["error"]
            assert "query" in result["error"]
        fake.search_vectors.assert_not_called()

    def test_module_unavailable(self, monkeypatch):
        """_get_report_vectors 返回 None（ImportError 降级）→ ok=False，不抛异常。"""
        monkeypatch.setattr(tools_mod, "_get_report_vectors", lambda: None)
        result = tools_mod.execute_tool("search_report_content", {"query": "茅台"})
        assert result["ok"] is False
        assert result["error"] == "研报全文模块不可用"

    def test_function_missing(self, monkeypatch):
        """fake 模块没有契约函数 → ok=False「研报全文接口暂不可用」。"""
        _patch_report_vectors(monkeypatch, SimpleNamespace())
        result = tools_mod.execute_tool("search_report_content", {"query": "茅台"})
        assert result["ok"] is False
        assert "研报全文接口暂不可用" in result["error"]
        assert "search_vectors" in result["error"]

    def test_empty_hits_with_note_passthrough(self, monkeypatch):
        """索引未建/依赖缺失信号：空 hits + note → ok=False 原样透传 note。"""
        note = "研报全文索引尚未建立，请先运行向量索引构建脚本"
        fake = _fake_report_vectors(
            return_value={"total_chunks": 0, "hits": [], "note": note}
        )
        _patch_report_vectors(monkeypatch, fake)
        result = tools_mod.execute_tool("search_report_content", {"query": "茅台"})
        assert result["ok"] is False
        assert result["error"] == note

    def test_empty_hits_without_note_stays_ok(self, monkeypatch):
        """空 hits 但无 note 是合法空结果（有索引无命中），保持 ok=True。"""
        fake = _fake_report_vectors(return_value={"total_chunks": 42, "hits": []})
        _patch_report_vectors(monkeypatch, fake)
        result = tools_mod.execute_tool("search_report_content", {"query": "冷门主题"})
        assert result["ok"] is True
        assert result["data"] == {"total_chunks": 42, "hits": []}

    def test_backend_raises_safe_degrade(self, monkeypatch):
        """后端抛异常 → ok=False + error，绝不向上抛。"""
        fake = _fake_report_vectors(side_effect=RuntimeError("向量库损坏"))
        _patch_report_vectors(monkeypatch, fake)
        result = tools_mod.execute_tool("search_report_content", {"query": "茅台"})
        assert result["ok"] is False
        assert "工具执行出错" in result["error"]

    def test_unknown_tool_keeps_original_behavior(self, monkeypatch):
        """未注册工具报错保持原行为（三表都不命中才报未注册）。"""
        fake = _fake_report_vectors()
        _patch_report_vectors(monkeypatch, fake)
        result = tools_mod.execute_tool("__definitely_not_a_tool__", {})
        assert result["ok"] is False
        assert "未注册的工具" in result["error"]
        fake.search_vectors.assert_not_called()


# ════════════════════════════════════════════════
# 7. prompt 断言
# ════════════════════════════════════════════════

class TestReportContentPrompt:
    def test_fulltext_citation_rules_present(self):
        prompt = system_prompts.AGENT_QUERY_PROMPT
        assert "## 研报引用规范" in prompt
        # 追加的两条：正文引用标注券商+日期+标题；正文只用 search_report_content 返回
        assert "券商+日期+研报标题" in prompt
        assert "search_report_content" in prompt
        assert "一律视为编造" in prompt

    def test_original_citation_rules_intact(self):
        """原有 4 条引用规范一字未动，且全文检索条款仍在「## 输出结构」之前。"""
        prompt = system_prompts.AGENT_QUERY_PROMPT
        for marker in (
            "中金公司 7 月 20 日研报（买入评级）认为",
            "search_research_reports / get_rating_summary",
            "研报库暂未覆盖",
            "不得编造券商观点",
        ):
            assert marker in prompt, f"研报引用规范原条款 {marker!r} 丢失"
        assert prompt.index("## 研报引用规范") < prompt.index("## 输出结构")
        assert prompt.index("search_report_content") < prompt.index("## 输出结构")


# ════════════════════════════════════════════════
# 8. get_tool_catalog() 28 行 + 既有工具回归
# ════════════════════════════════════════════════

class TestToolCatalog28:
    def test_catalog_28_lines(self):
        assert len(tools_mod.TOOL_REGISTRY) == 28
        catalog = tools_mod.get_tool_catalog()
        assert "共 28 个" in catalog
        tool_lines = [l for l in catalog.splitlines() if l.startswith("- ")]
        assert len(tool_lines) == 28, (
            f"目录应有 28 行工具条目，实际 {len(tool_lines)} 行：\n{catalog}"
        )

    def test_new_tool_in_short_desc_and_catalog(self):
        assert "search_report_content" in tools_mod._SHORT_DESC
        catalog = tools_mod.get_tool_catalog()
        assert "search_report_content" in catalog
        assert tools_mod._SHORT_DESC["search_report_content"] in catalog

    def test_existing_23_tools_unaffected(self, monkeypatch):
        """既有 23 个工具名一个不少；研报元数据工具仍走 _REPORT_IMPL。"""
        existing = {
            "get_market_indices", "get_sector_list", "get_sector_valuation",
            "get_sector_moneyflow", "get_sector_earnings", "get_sector_stocks",
            "get_fund_flows", "get_market_breadth", "get_limit_up_pool",
            "get_hot_stocks", "get_strong_sectors", "get_stock_quote",
            "get_stock_kline", "get_stock_news", "search_news",
            "get_futures", "get_us_sectors", "get_hk_sectors",
            "get_global_indices", "get_china_macro", "get_us_macro",
            "search_research_reports", "get_rating_summary",
        }
        registered = {t["function"]["name"] for t in tools_mod.TOOL_REGISTRY}
        assert existing <= registered, (
            f"既有工具丢失：{existing - registered}"
        )
        # 新增工具：研报全文（v2）+ 第十二波开源灵感模块 4 个
        assert registered - existing == {
            "search_report_content",
            "get_market_sentiment", "get_stock_sentiment",
            "get_technical_analysis", "analyze_with_persona",
        }

        # v1 研报工具分发路径不变（仍走 report_library）
        fake_rl = SimpleNamespace(
            search_reports=MagicMock(return_value={"total": 0, "reports": []}),
            rating_summary=MagicMock(return_value={
                "total": 0, "rating_dist": {}, "target_price_range": None,
                "avg_eps_forecast": None, "latest_reports": [],
            }),
        )
        monkeypatch.setattr(tools_mod, "_get_report_library", lambda: fake_rl)
        result = tools_mod.execute_tool("search_research_reports", {"query": "茅台"})
        assert result["ok"] is True
        fake_rl.search_reports.assert_called_once()

        # 数据层工具分发路径不变（仍走 data_fetcher）
        fetch = MagicMock(return_value={"indices": []})
        monkeypatch.setattr("agent.data_fetcher.fetch_a_share_indices", fetch)
        result = tools_mod.execute_tool("get_market_indices", {"date": "20260722"})
        assert result["ok"] is True
        fetch.assert_called_once_with(date="20260722")
