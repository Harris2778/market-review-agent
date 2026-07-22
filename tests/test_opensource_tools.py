"""第十二波『开源灵感模块接线』集成测试。

覆盖范围：
1. 4 个新工具的 schema 完整性与注册表规模（24→28，工具名唯一）。
2. execute_tool 分发正确性：get_market_sentiment / get_stock_sentiment /
   get_technical_analysis / analyze_with_persona 的成功路径结构断言。
3. 参数归一：sh600519 / 600519 / sz000002 等输入统一归一为 6 位代码，
   非法代码 ok=False 不抛异常。
4. 降级路径：模块缺失 / 新闻取数失败 / 日线取数失败 / 数据不足 /
   未知人格 / 交易日历不可用（启发式回退）——任何路径绝不抛异常。
5. 编排钩子：_agent_query 工具循环中 Scratchpad JSONL 落盘（tmp 隔离
   SCRATCHPAD_DIR）、ToolCallGuard 软警告注入工具结果尾部、microcompact
   每轮被调用且 cleared>0 时记入审计思考流；钩子异常不影响主循环。
6. system_prompts：AGENT_QUERY_PROMPT 新增三节存在性与关键纪律断言。

规则（与项目其他测试一致）：
- 所有外部调用全部 mock（DeepSeek 客户端 / 数据采集 / HTTP / Tushare），
  绝不发起真实网络请求。
- 无 pytest-asyncio，异步函数一律用 asyncio.run 驱动。
"""

import asyncio
import json
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agent.data_fetcher as data_fetcher
import agent.orchestrator as orchestrator
import agent.sentiment as sentiment
import agent.technical as technical
import agent.tools as tools_mod
from agent import personas, system_prompts
from agent.agent_audit import microcompact as real_microcompact
from agent.orchestrator import MarketReviewAgent

# ════════════════════════════════════════════════════════════════
# 公共工具
# ════════════════════════════════════════════════════════════════

NEW_TOOLS = {
    "get_market_sentiment": [],
    "get_stock_sentiment": ["stock_code"],
    "get_technical_analysis": ["stock_code"],
    "analyze_with_persona": ["persona"],
}

_TODAY = date.today()


def _ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _iso(d: date) -> str:
    return d.isoformat()


def _recent_weekday() -> date:
    """今天起往回找第一个周一~周五（对齐 is_trade_day 启发式）。"""
    d = _TODAY
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _daily_records(n: int, end_day: date):
    """生成 n 行 Tushare daily 风格记录（降序，最新在前），字段含 trade_date/OHLC/vol。"""
    rows = []
    price = 100.0
    for i in range(n):
        d = end_day - timedelta(days=i)
        close = price - i * 0.1  # 轻微上行趋势（往过去递减）
        rows.append({
            "trade_date": _ymd(d),
            "open": close - 0.3,
            "high": close + 0.6,
            "low": close - 0.8,
            "close": close,
            "vol": 10000.0 + i,
            "amount": 100000.0 + i * 10,
        })
    return rows  # 降序


def _fake_pro(records, cal_ok: bool = True):
    """伪造 Tushare pro 句柄：daily 返回给定记录的 DataFrame，trade_cal 可控。"""
    import pandas as pd
    pro = MagicMock()
    pro.daily = MagicMock(return_value=pd.DataFrame(records))
    if cal_ok:
        pro.trade_cal = MagicMock(return_value=pd.DataFrame(
            [{"cal_date": _ymd(_TODAY), "is_open": 1}]
        ))
    else:
        pro.trade_cal = MagicMock(side_effect=Exception("无 trade_cal 权限"))
    return pro


def _patch_pro(monkeypatch, pro):
    monkeypatch.setattr(data_fetcher, "_get_pro", lambda: pro)


def _make_agent() -> MarketReviewAgent:
    agent = MarketReviewAgent()
    agent.client = MagicMock()
    return agent


def _make_tool_call(name: str, arguments: dict, call_id: str):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
    )


def _completion_with_tool_calls(tool_calls):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=None, tool_calls=tool_calls),
            finish_reason="tool_calls",
        )]
    )


def _completion_text(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=None),
            finish_reason="stop",
        )]
    )


def _patch_execute_tool(exec_mock):
    return patch("agent.orchestrator.execute_tool", exec_mock)


def _read_jsonl(dir_path):
    """读取目录下全部 .jsonl 文件的解析后条目列表。"""
    entries = []
    files = sorted(dir_path.glob("*.jsonl"))
    for fh in files:
        for line in fh.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
    return files, entries


# ════════════════════════════════════════════════════════════════
# 1. 工具注册表：规模 / schema 完整性 / 唯一性
# ════════════════════════════════════════════════════════════════

class TestNewToolRegistry:

    def test_registry_total_32(self):
        assert len(tools_mod.TOOL_REGISTRY) == 32, (
            f"TOOL_REGISTRY 应为 32 个工具（24 既有 + 4 开源灵感 + 2 社媒舆情 "
            f"+ 2 校园知识库），实际 {len(tools_mod.TOOL_REGISTRY)}"
        )

    def test_new_tools_schema_complete(self):
        by_name = {t["function"]["name"]: t["function"] for t in tools_mod.TOOL_REGISTRY}
        for name, expected_required in NEW_TOOLS.items():
            assert name in by_name, f"注册表缺少新工具 {name}"
            fn = by_name[name]
            assert isinstance(fn.get("description"), str) and fn["description"].strip(), (
                f"{name} 缺少 description"
            )
            params = fn.get("parameters")
            assert isinstance(params, dict) and params.get("type") == "object", (
                f"{name} 的 parameters 不是合法 JSON Schema"
            )
            props = params.get("properties", {})
            assert isinstance(props, dict) and props, f"{name} 缺少 properties"
            assert params.get("required", []) == expected_required, (
                f"{name} 的 required 应为 {expected_required}，"
                f"实际 {params.get('required')}"
            )
            for r in expected_required:
                assert r in props, f"{name} 的必填参数 {r!r} 未在 properties 中定义"
            json.dumps(params)  # 必须可 JSON 序列化

    def test_tool_names_unique(self):
        names = [t["function"]["name"] for t in tools_mod.TOOL_REGISTRY]
        assert len(names) == len(set(names)), f"存在重名工具: {names}"

    def test_persona_param_enum(self):
        by_name = {t["function"]["name"]: t["function"] for t in tools_mod.TOOL_REGISTRY}
        enum = by_name["analyze_with_persona"]["parameters"]["properties"]["persona"]["enum"]
        assert enum == ["value_cn", "growth_cn", "trend_cn", "contrarian_cn"], (
            f"persona 枚举应为 4 个人格 key，实际 {enum}"
        )

    def test_catalog_mentions_new_tools(self):
        catalog = tools_mod.get_tool_catalog()
        for name in NEW_TOOLS:
            assert name in catalog, f"工具目录缺少 {name}"


# ════════════════════════════════════════════════════════════════
# 2. get_market_sentiment
# ════════════════════════════════════════════════════════════════

class TestMarketSentimentTool:

    def _fake_snapshot(self):
        return {
            "date": _iso(_TODAY),
            "temperature": 55.0,
            "temperature_label": "中性",
            "stats": {"zt_count": 56, "dt_count": 3, "zb_count": 10,
                      "炸板率": 15.15, "最高连板": 4},
            "hot_rank_top": [{"rank": 1, "code": "600519", "name": "贵州茅台",
                              "source": "eastmoney_hotrank"}],
            "news_sentiment": {"利好": 2, "利空": 1, "中性": 5},
            "sources": ["eastmoney_ztpool", "eastmoney_hotrank", "news_injected"],
            "notes": [],
        }

    def test_success_with_news_injected(self, monkeypatch):
        news = [{"title": "央行降准 利好市场", "time": "08:00", "source": "eastmoney"}]
        monkeypatch.setattr(data_fetcher, "fetch_eastmoney_news", lambda limit=25: news)
        captured = {}

        def fake_gms(**kwargs):
            captured.update(kwargs)
            return self._fake_snapshot()

        monkeypatch.setattr(sentiment, "get_market_sentiment", fake_gms)
        result = tools_mod.execute_tool("get_market_sentiment", {})
        assert result["ok"] is True, f"应成功: {result}"
        data = result["data"]
        assert data["temperature"] == 55.0
        assert data["temperature_label"] == "中性"
        assert data["stats"]["zt_count"] == 56
        assert data["hot_rank_top"][0]["code"] == "600519"
        # 新闻被注入底层函数
        assert captured.get("news_items") == news

    def test_date_passthrough(self, monkeypatch):
        monkeypatch.setattr(data_fetcher, "fetch_eastmoney_news", lambda limit=25: [])
        captured = {}
        monkeypatch.setattr(
            sentiment, "get_market_sentiment",
            lambda **kw: captured.update(kw) or self._fake_snapshot(),
        )
        result = tools_mod.execute_tool("get_market_sentiment", {"date": "2026-08-12"})
        assert result["ok"] is True
        assert captured.get("date") == "2026-08-12", f"date 应透传: {captured}"

    def test_news_fetch_failure_degrades(self, monkeypatch):
        def _boom(limit=25):
            raise RuntimeError("新闻源 502")

        monkeypatch.setattr(data_fetcher, "fetch_eastmoney_news", _boom)
        monkeypatch.setattr(
            sentiment, "get_market_sentiment",
            lambda **kw: self._fake_snapshot(),
        )
        result = tools_mod.execute_tool("get_market_sentiment", {})
        assert result["ok"] is True, f"新闻失败不应拖垮工具: {result}"
        notes = result["data"].get("notes", [])
        assert any("新闻" in n for n in notes), f"降级说明应进 notes: {notes}"

    def test_module_missing_degrades(self, monkeypatch):
        monkeypatch.setattr(tools_mod, "_get_sentiment_module", lambda: None)
        result = tools_mod.execute_tool("get_market_sentiment", {})
        assert result["ok"] is True  # 外层分发不失败
        assert result["data"].get("ok") is False
        assert "情绪模块不可用" in result["data"].get("note", "")


# ════════════════════════════════════════════════════════════════
# 3. get_stock_sentiment
# ════════════════════════════════════════════════════════════════

class TestStockSentimentTool:

    def _fake_result(self, code="600519"):
        return {
            "code": code,
            "hot_rank": {"latest": 3, "history_avg": 12.5, "trend": "上升"},
            "news_sentiment": {"利好": 1, "利空": 0, "中性": 2},
            "sources": ["eastmoney_hotrank", "news_injected"],
            "notes": [],
        }

    def test_success_prefixed_code_normalized(self, monkeypatch):
        news = [{"title": "茅台业绩增长 超预期", "url": "http://x"}]
        news_calls = {}
        monkeypatch.setattr(
            data_fetcher, "fetch_stock_news",
            lambda symbol, market="cn", limit=10: (
                news_calls.update(symbol=symbol, market=market) or news),
        )
        captured = {}
        monkeypatch.setattr(
            sentiment, "get_stock_sentiment",
            lambda code, days=30, **kw: (
                captured.update(code=code, days=days, news_items=kw.get("news_items"))
                or self._fake_result(code)),
        )
        result = tools_mod.execute_tool("get_stock_sentiment", {"stock_code": "sh600519"})
        assert result["ok"] is True, f"应成功: {result}"
        assert captured["code"] == "600519", f"sh 前缀应被归一: {captured}"
        assert captured["days"] == 30, f"days 默认应为 30: {captured}"
        assert captured["news_items"] == news, "个股新闻应注入 news_items"
        assert news_calls.get("symbol") == "sh600519", (
            f"个股新闻应用带前缀代码查询: {news_calls}"
        )
        data = result["data"]
        assert data["hot_rank"]["trend"] == "上升"
        assert data["news_sentiment"]["利好"] == 1

    def test_plain_code_and_days(self, monkeypatch):
        monkeypatch.setattr(data_fetcher, "fetch_stock_news", lambda **kw: [])
        captured = {}
        monkeypatch.setattr(
            sentiment, "get_stock_sentiment",
            lambda code, days=30, **kw: (
                captured.update(code=code, days=days) or self._fake_result(code)),
        )
        result = tools_mod.execute_tool(
            "get_stock_sentiment", {"stock_code": "000002", "days": 7})
        assert result["ok"] is True
        assert captured["code"] == "000002"
        assert captured["days"] == 7

    def test_news_failure_note_no_raise(self, monkeypatch):
        def _boom(**kw):
            raise RuntimeError("个股新闻源超时")

        monkeypatch.setattr(data_fetcher, "fetch_stock_news", _boom)
        monkeypatch.setattr(
            sentiment, "get_stock_sentiment",
            lambda code, days=30, **kw: self._fake_result(code),
        )
        result = tools_mod.execute_tool("get_stock_sentiment", {"stock_code": "600519"})
        assert result["ok"] is True
        notes = result["data"].get("notes", [])
        assert any("新闻" in n for n in notes), f"降级说明应进 notes: {notes}"

    def test_invalid_code_ok_false_no_raise(self):
        result = tools_mod.execute_tool("get_stock_sentiment", {"stock_code": "not-a-code"})
        assert result["ok"] is False, f"非法代码应 ok=False: {result}"
        assert "stock_code" in result.get("error", "")

    def test_missing_required_param(self):
        result = tools_mod.execute_tool("get_stock_sentiment", {})
        assert result["ok"] is False
        assert "stock_code" in result.get("error", "")

    def test_module_missing_degrades(self, monkeypatch):
        monkeypatch.setattr(tools_mod, "_get_sentiment_module", lambda: None)
        result = tools_mod.execute_tool("get_stock_sentiment", {"stock_code": "600519"})
        assert result["ok"] is True
        assert result["data"].get("ok") is False
        assert "情绪模块不可用" in result["data"].get("note", "")


# ════════════════════════════════════════════════════════════════
# 4. get_technical_analysis
# ════════════════════════════════════════════════════════════════

class TestTechnicalAnalysisTool:

    def test_success_structure(self, monkeypatch):
        pro = _fake_pro(_daily_records(150, _TODAY))
        _patch_pro(monkeypatch, pro)
        result = tools_mod.execute_tool(
            "get_technical_analysis", {"stock_code": "sh600519"})
        assert result["ok"] is True, f"应成功: {result}"
        data = result["data"]
        assert data["ok"] is True, f"取数与计算应成功: {data}"
        assert data["as_of"] == _iso(_TODAY), f"as_of 应为最后一根 K 线日期: {data['as_of']}"
        assert isinstance(data["close"], (int, float))
        # indicators 原样透传 compute_indicators 结构
        ind = data["indicators"]
        for key in ("ma", "ma_alignment", "bias", "volume_state", "macd",
                    "rsi", "support", "resistance", "score", "score_breakdown"):
            assert key in ind, f"indicators 缺少 {key}"
        assert 0 <= ind["score"] <= 100
        # verdict 原样透传 verdict_from_score 结构
        verdict = data["verdict"]
        assert verdict["band"] in ("强势", "偏多", "中性", "偏空", "弱势")
        assert 0 <= verdict["confidence_cap"] <= 1
        assert "guardrail_reason" in verdict
        assert f"Tushare daily（as_of {_iso(_TODAY)}）" in data["source"]
        assert "agent.technical 本地确定性计算" in data["source"]

    def test_ts_code_mapping(self, monkeypatch):
        pro = _fake_pro(_daily_records(150, _TODAY))
        _patch_pro(monkeypatch, pro)
        tools_mod.execute_tool("get_technical_analysis", {"stock_code": "sh600519"})
        assert pro.daily.call_args.kwargs["ts_code"] == "600519.SH"
        tools_mod.execute_tool("get_technical_analysis", {"stock_code": "000002"})
        assert pro.daily.call_args.kwargs["ts_code"] == "000002.SZ"

    def test_days_limit_applied(self, monkeypatch):
        # 150 行记录、days=60：计算应只用最近 60 根（as_of 仍为最新日期）
        pro = _fake_pro(_daily_records(150, _TODAY))
        _patch_pro(monkeypatch, pro)
        result = tools_mod.execute_tool(
            "get_technical_analysis", {"stock_code": "600519", "days": 60})
        assert result["ok"] is True
        assert result["data"]["ok"] is True
        assert result["data"]["as_of"] == _iso(_TODAY)

    def test_stale_guardrail(self, monkeypatch):
        # as_of 远旧于最近应有交易日 → stale 护栏压置信度上限
        old_day = date(2020, 1, 3)
        pro = _fake_pro(_daily_records(80, old_day))
        _patch_pro(monkeypatch, pro)
        result = tools_mod.execute_tool(
            "get_technical_analysis", {"stock_code": "600519"})
        assert result["ok"] is True
        data = result["data"]
        assert data["ok"] is True
        assert data["as_of"] == "2020-01-03"
        assert data["verdict"]["confidence_cap"] <= 0.5, (
            f"stale 数据应压 confidence_cap≤0.5: {data['verdict']}"
        )
        assert data["verdict"]["guardrail_reason"], "stale 应给护栏原因"

    def test_no_calendar_heuristic_fallback(self, monkeypatch):
        # trade_cal 不可用 → 启发式判定，notes 说明，绝不抛
        pro = _fake_pro(_daily_records(80, _recent_weekday()), cal_ok=False)
        _patch_pro(monkeypatch, pro)
        result = tools_mod.execute_tool(
            "get_technical_analysis", {"stock_code": "600519"})
        assert result["ok"] is True
        data = result["data"]
        assert data["ok"] is True
        assert any("启发式" in n for n in data.get("notes", [])), (
            f"日历不可用应在 notes 说明启发式回退: {data.get('notes')}"
        )
        # as_of 是最近工作日 → 不应误判 stale
        assert data["verdict"]["confidence_cap"] == 1.0

    def test_insufficient_rows_degrades(self, monkeypatch):
        pro = _fake_pro(_daily_records(3, _TODAY))
        _patch_pro(monkeypatch, pro)
        result = tools_mod.execute_tool(
            "get_technical_analysis", {"stock_code": "600519"})
        assert result["ok"] is True  # 分发层不失败
        data = result["data"]
        assert data["ok"] is False
        assert "数据不足" in data.get("note", ""), f"应说明数据不足: {data}"

    def test_fetch_failure_degrades(self, monkeypatch):
        def _boom():
            raise RuntimeError("Tushare 连接被拒")

        monkeypatch.setattr(data_fetcher, "_get_pro", _boom)
        result = tools_mod.execute_tool(
            "get_technical_analysis", {"stock_code": "600519"})
        assert result["ok"] is True
        data = result["data"]
        assert data["ok"] is False
        assert "Tushare" in data.get("note", "")

    def test_pro_unavailable_degrades(self, monkeypatch):
        monkeypatch.setattr(data_fetcher, "_get_pro", lambda: None)
        result = tools_mod.execute_tool(
            "get_technical_analysis", {"stock_code": "600519"})
        assert result["ok"] is True
        assert result["data"]["ok"] is False
        assert result["data"].get("note")

    def test_daily_exception_degrades(self, monkeypatch):
        pro = MagicMock()
        pro.daily = MagicMock(side_effect=RuntimeError("daily 接口 500"))
        _patch_pro(monkeypatch, pro)
        result = tools_mod.execute_tool(
            "get_technical_analysis", {"stock_code": "600519"})
        assert result["ok"] is True
        data = result["data"]
        assert data["ok"] is False
        assert "取数失败" in data.get("note", "")

    def test_module_missing_degrades(self, monkeypatch):
        monkeypatch.setattr(tools_mod, "_get_technical_module", lambda: None)
        result = tools_mod.execute_tool(
            "get_technical_analysis", {"stock_code": "600519"})
        assert result["ok"] is True
        assert result["data"].get("ok") is False
        assert "技术分析模块不可用" in result["data"].get("note", "")

    def test_invalid_code_ok_false_no_raise(self):
        result = tools_mod.execute_tool(
            "get_technical_analysis", {"stock_code": "??"})
        assert result["ok"] is False
        assert "stock_code" in result.get("error", "")


# ════════════════════════════════════════════════════════════════
# 5. analyze_with_persona
# ════════════════════════════════════════════════════════════════

class TestAnalyzeWithPersona:

    @pytest.mark.parametrize(
        "key", ["value_cn", "growth_cn", "trend_cn", "contrarian_cn"])
    def test_each_persona_framework(self, key):
        result = tools_mod.execute_tool(
            "analyze_with_persona", {"persona": key, "stock_code": "sh600519"})
        assert result["ok"] is True, f"{key} 应成功: {result}"
        data = result["data"]
        assert data["persona"] == key
        assert data["stock_code"] == "600519"
        fw = data["framework"]
        for field in ("name", "instructions", "scoring_weights", "thresholds",
                      "analysis_rules", "output_schema", "checklist", "disclaimer"):
            assert field in fw, f"{key} 框架缺少 {field}"
        assert fw["disclaimer"] == personas.DISCLAIMER
        assert "validate" in data["guidance"]
        assert "persona_defs.json" in data["source"]

    def test_unknown_persona_degrades(self):
        result = tools_mod.execute_tool("analyze_with_persona", {"persona": "wizard_cn"})
        assert result["ok"] is True  # 分发层不失败
        data = result["data"]
        assert "未知投资人格" in data.get("note", ""), f"应说明未知人格: {data}"
        available = data.get("available", [])
        assert len(available) == 4, f"应列出 4 个可用人格: {available}"
        keys = {p["key"] for p in available}
        assert keys == {"value_cn", "growth_cn", "trend_cn", "contrarian_cn"}

    def test_missing_persona_ok_false(self):
        result = tools_mod.execute_tool("analyze_with_persona", {})
        assert result["ok"] is False
        assert "persona" in result.get("error", "")

    def test_invalid_stock_code_tolerated(self):
        result = tools_mod.execute_tool(
            "analyze_with_persona", {"persona": "trend_cn", "stock_code": "bad"})
        assert result["ok"] is True
        assert result["data"]["stock_code"] is None, (
            "非法 stock_code 应归一为 None 而不是报错"
        )

    def test_module_missing_degrades(self, monkeypatch):
        monkeypatch.setattr(tools_mod, "_get_personas_module", lambda: None)
        result = tools_mod.execute_tool("analyze_with_persona", {"persona": "value_cn"})
        assert result["ok"] is True
        assert "不可用" in result["data"].get("note", "")
        assert result["data"].get("available") == []


# ════════════════════════════════════════════════════════════════
# 6. 编排钩子：Scratchpad / ToolCallGuard / microcompact
# ════════════════════════════════════════════════════════════════

class TestAgentLoopAuditHooks:

    def test_scratchpad_jsonl_written(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SCRATCHPAD_DIR", str(tmp_path))
        agent = _make_agent()
        tool_name = tools_mod.TOOL_REGISTRY[0]["function"]["name"]
        agent.client.chat.completions.create = AsyncMock(side_effect=[
            _completion_with_tool_calls([_make_tool_call(tool_name, {"date": "20260814"}, "call_1")]),
            _completion_text("最终分析文本"),
        ])
        exec_mock = MagicMock(return_value={"ok": True, "data": {"pe": 10}})
        with _patch_execute_tool(exec_mock):
            result = asyncio.run(agent._agent_query("比较白酒和半导体估值", stream=False))
        assert "最终分析文本" in result["content"]

        files, entries = _read_jsonl(tmp_path)
        assert files, "Scratchpad 应在 SCRATCHPAD_DIR 落盘 JSONL 文件"
        types = [e.get("type") for e in entries]
        assert "init" in types, f"缺少 init 条目: {types}"
        assert "tool_call" in types, f"缺少 tool_call 条目: {types}"
        assert "tool_result" in types, f"缺少 tool_result 条目: {types}"
        init_entry = next(e for e in entries if e["type"] == "init")
        assert "比较白酒和半导体估值" in init_entry.get("query", "")
        call_entry = next(e for e in entries if e["type"] == "tool_call")
        assert call_entry.get("tool") == tool_name
        result_entry = next(e for e in entries if e["type"] == "tool_result")
        assert result_entry.get("tool") == tool_name

    def test_guard_warning_injected_into_tool_result(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SCRATCHPAD_DIR", str(tmp_path))
        agent = _make_agent()
        tool_name = tools_mod.TOOL_REGISTRY[0]["function"]["name"]
        # 两轮完全相同的工具调用：第二轮命中相似度软警告
        agent.client.chat.completions.create = AsyncMock(side_effect=[
            _completion_with_tool_calls([_make_tool_call(tool_name, {"date": "20260814"}, "call_1")]),
            _completion_with_tool_calls([_make_tool_call(tool_name, {"date": "20260814"}, "call_2")]),
            _completion_text("最终分析文本"),
        ])
        exec_mock = MagicMock(return_value={"ok": True, "data": {"pe": 10}})
        with _patch_execute_tool(exec_mock):
            result = asyncio.run(agent._agent_query("重复查同一个数", stream=False))
        assert "最终分析文本" in result["content"]

        third_messages = agent.client.chat.completions.create.await_args_list[2].kwargs["messages"]
        tool_msgs = [m for m in third_messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2, f"应有两条工具结果: {tool_msgs}"
        assert any("软警告" in m["content"] for m in tool_msgs), (
            f"第二次重复调用的工具结果尾部应注入软警告: {[m['content'][-200:] for m in tool_msgs]}"
        )

    def test_microcompact_called_each_round(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SCRATCHPAD_DIR", str(tmp_path))
        mc_mock = MagicMock(side_effect=real_microcompact)
        monkeypatch.setattr(orchestrator, "microcompact", mc_mock)
        agent = _make_agent()
        tool_name = tools_mod.TOOL_REGISTRY[0]["function"]["name"]
        agent.client.chat.completions.create = AsyncMock(side_effect=[
            _completion_with_tool_calls([_make_tool_call(tool_name, {}, "call_1")]),
            _completion_text("最终分析文本"),
        ])
        exec_mock = MagicMock(return_value={"ok": True, "data": {"n": 1}})
        with _patch_execute_tool(exec_mock):
            asyncio.run(agent._agent_query("测试压缩钩子", stream=False))
        assert mc_mock.call_count == 2, (
            f"microcompact 应在每轮调 LLM 前各执行一次（共 2 轮），"
            f"实际 {mc_mock.call_count} 次"
        )

    def test_microcompact_cleared_logged_as_thinking(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SCRATCHPAD_DIR", str(tmp_path))

        def _fake_mc(messages):
            return messages, {"cleared": 3, "chars_before": 99999, "chars_after": 1234}

        monkeypatch.setattr(orchestrator, "microcompact", _fake_mc)
        agent = _make_agent()
        tool_name = tools_mod.TOOL_REGISTRY[0]["function"]["name"]
        agent.client.chat.completions.create = AsyncMock(side_effect=[
            _completion_with_tool_calls([_make_tool_call(tool_name, {}, "call_1")]),
            _completion_text("最终分析文本"),
        ])
        exec_mock = MagicMock(return_value={"ok": True, "data": {"n": 1}})
        with _patch_execute_tool(exec_mock):
            asyncio.run(agent._agent_query("测试压缩日志", stream=False))
        files, entries = _read_jsonl(tmp_path)
        thinking = [e for e in entries if e.get("type") == "thinking"]
        assert thinking, f"cleared>0 时应写入 thinking 审计条目: {entries}"
        assert any("microcompact" in e.get("text", "") for e in thinking)

    def test_audit_hooks_exception_never_breaks_loop(self, monkeypatch, tmp_path):
        """Scratchpad/ToolCallGuard 构造即炸：主循环照常完成。"""
        monkeypatch.setenv("SCRATCHPAD_DIR", str(tmp_path))
        monkeypatch.setattr(
            orchestrator, "Scratchpad",
            MagicMock(side_effect=RuntimeError("审计模块炸了")))
        monkeypatch.setattr(
            orchestrator, "ToolCallGuard",
            MagicMock(side_effect=RuntimeError("护栏模块炸了")))
        agent = _make_agent()
        tool_name = tools_mod.TOOL_REGISTRY[0]["function"]["name"]
        agent.client.chat.completions.create = AsyncMock(side_effect=[
            _completion_with_tool_calls([_make_tool_call(tool_name, {}, "call_1")]),
            _completion_text("最终分析文本"),
        ])
        exec_mock = MagicMock(return_value={"ok": True, "data": {"n": 1}})
        with _patch_execute_tool(exec_mock):
            result = asyncio.run(agent._agent_query("审计炸了也要回答", stream=False))
        assert "最终分析文本" in result["content"], (
            "审计钩子异常不得影响 Agent 工具循环主流程"
        )


# ════════════════════════════════════════════════════════════════
# 7. system_prompts：AGENT_QUERY_PROMPT 新增三节
# ════════════════════════════════════════════════════════════════

class TestAgentQueryPromptSections:

    def test_three_new_sections_exist(self):
        prompt = system_prompts.AGENT_QUERY_PROMPT
        for section in ("情绪数据引用规范", "技术分析纪律", "投资人格框架"):
            assert section in prompt, f"AGENT_QUERY_PROMPT 缺少「{section}」一节"

    def test_sentiment_section_rules(self):
        prompt = system_prompts.AGENT_QUERY_PROMPT
        assert "东方财富" in prompt, "情绪节应要求注明东方财富出处"
        assert "辅助信号" in prompt, "情绪节应声明情绪是辅助信号"
        assert "情绪数据未覆盖" in prompt, "情绪节应要求未覆盖时明说"

    def test_technical_section_rules(self):
        prompt = system_prompts.AGENT_QUERY_PROMPT
        assert "get_technical_analysis" in prompt
        assert "confidence_cap" in prompt, "技术节应约束置信度上限"
        assert "guardrail_reason" in prompt, "技术节应要求护栏原因原文透传"

    def test_persona_section_rules(self):
        prompt = system_prompts.AGENT_QUERY_PROMPT
        assert "analyze_with_persona" in prompt
        assert "checklist" in prompt, "人格节应要求按 checklist 逐项分析"
        assert "disclaimer" in prompt, "人格节应要求带免责声明"
        assert "0.9" in prompt, "人格节应约束 confidence 上限"
