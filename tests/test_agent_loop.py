"""Agent 工具循环 + 多 pass 生成（第二波）测试。

覆盖范围：
1. 工具注册表规范性：TOOL_REGISTRY 每项为 OpenAI function 格式
   （type=function、function.name/description/parameters），parameters 是
   合法 JSON Schema（type: object）；get_tool_catalog() 非空。
2. execute_tool 分发：mock 数据层函数后 ok=True 且带 data；未知工具名
   ok=False 不抛异常；数据层抛异常时 ok=False 安全降级。
3. Agent 循环完整流程：tool_calls → execute_tool → role="tool" 回灌 →
   纯文本收尾，最终输出为第二轮文本。
4. 轮次上限：模型永远返回 tool_calls 时，循环在 8 轮内停止并降级 _chat，
   不无限循环。
5. 异常降级：create 首轮抛异常 → 降级 _chat，不向上抛。
6. process_message 路由：复杂分析（比较/对比类）→ _agent_query；
   普通闲聊仍走 _chat，不进工具循环。
7. 多 pass：非流式 _sector_deep_dive 先草稿再 _critique_and_revise 审查
   修正，修正结果为最终输出；draft < 500 字时跳过审查（不再二次调 LLM）；
   审查调用失败时降级返回原 draft。
8. prompt 断言：AGENT_QUERY_PROMPT 含数据红线表述 + 至少 3 个禁用词；
   CRITIQUE_PROMPT 含数字出处 / 禁用词 / 越界三类检查项。

背景契约（生产代码由并行代理实现，本文件只按契约 mock）：
- agent/tools.py：TOOL_REGISTRY（OpenAI 格式工具列表）、
  execute_tool(name, args) -> {"ok": bool, "data"|"error"}（同步）、
  get_tool_catalog() -> str
- MarketReviewAgent._agent_query(user_message, stream, history=None)：
  工具循环最多 8 轮，tool_calls → execute_tool → role="tool" 消息回灌 →
  直到纯文本；失败降级 _chat
- process_message 路由：general_chat + 复杂分析特征（比较/对比/哪个更等）
  → _agent_query；普通闲聊仍走 _chat
- MarketReviewAgent._critique_and_revise(draft, context)：非流式
  _sector_deep_dive 输出先草稿再审查修正；draft<500 字或失败时返回原 draft
- system_prompts.AGENT_QUERY_PROMPT / CRITIQUE_PROMPT

规则（与项目其他测试一致）：
- 所有外部调用全部 mock（DeepSeek 客户端 / 数据采集 / execute_tool），
  绝不发起真实网络请求。
- 不用 create=True patch 不存在的函数——接线缺失必须表现为测试失败。
  agent.tools 尚未落地时，相关用例以 ImportError/AttributeError 如实失败，
  不用 skip 掩盖。
- 无 pytest-asyncio，异步函数一律用 asyncio.run 驱动。
"""

import asyncio
import itertools
import json
from contextlib import ExitStack
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agent.data_fetcher as data_fetcher
import agent.orchestrator as orchestrator
from agent import system_prompts
from agent.orchestrator import MarketReviewAgent

# 固定交易日，避免 _get_latest_trade_date 触达真实 tushare
FIXED_TRADE_DATE = datetime(2025, 1, 10)  # 周五
FIXED_DATE_STR = "20250110"


def _make_agent() -> MarketReviewAgent:
    """构造一个 agent，DeepSeek 客户端替换为 mock，防止任何真实 HTTP 调用。"""
    agent = MarketReviewAgent()
    agent.client = MagicMock()
    return agent


def _import_tools():
    """导入第二波新增的 agent.tools 模块。

    该模块由并行代理实现；尚未落地时 ImportError 直接让用例失败——
    如实暴露接线缺失，不用 skip / create=True 掩盖。
    """
    import agent.tools as tools_mod
    return tools_mod


def _make_tool_call(name: str, arguments: dict, call_id: str):
    """伪造 OpenAI 格式的 tool_call 对象。"""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
    )


def _completion_with_tool_calls(tool_calls):
    """伪造带 tool_calls 的 create 返回（模型要求调用工具）。"""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=None, tool_calls=tool_calls),
            finish_reason="tool_calls",
        )]
    )


def _completion_text(text: str):
    """伪造纯文本的 create 返回（工具循环终点 / 普通生成）。"""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=None),
            finish_reason="stop",
        )]
    )


# ════════════════════════════════════════════════════════════════
# 1. 工具注册表规范性
# ════════════════════════════════════════════════════════════════


class TestToolRegistry:
    """TOOL_REGISTRY 的 OpenAI function 格式 + get_tool_catalog()。"""

    def test_registry_nonempty_list(self):
        tools_mod = _import_tools()
        assert isinstance(tools_mod.TOOL_REGISTRY, list), (
            f"TOOL_REGISTRY 应为 list，实际 {type(tools_mod.TOOL_REGISTRY)}"
        )
        assert len(tools_mod.TOOL_REGISTRY) >= 1, "TOOL_REGISTRY 为空"

    def test_each_tool_openai_function_format(self):
        tools_mod = _import_tools()
        for tool in tools_mod.TOOL_REGISTRY:
            hint = json.dumps(tool, ensure_ascii=False, default=str)[:100]
            assert tool.get("type") == "function", (
                f"工具缺少 type=\"function\": {hint}"
            )
            fn = tool.get("function")
            assert isinstance(fn, dict), f"工具缺少 function 字段: {hint}"
            assert isinstance(fn.get("name"), str) and fn["name"].strip(), (
                f"工具缺少 function.name: {hint}"
            )
            assert isinstance(fn.get("description"), str) and fn["description"].strip(), (
                f"工具 {fn.get('name')} 缺少 function.description"
            )
            params = fn.get("parameters")
            assert isinstance(params, dict), (
                f"工具 {fn.get('name')} 缺少 function.parameters"
            )
            # parameters 必须是合法 JSON Schema：type=object + properties dict
            assert params.get("type") == "object", (
                f"工具 {fn.get('name')} 的 parameters.type 应为 \"object\"，"
                f"实际 {params.get('type')!r}"
            )
            props = params.get("properties", {})
            assert isinstance(props, dict), (
                f"工具 {fn.get('name')} 的 parameters.properties 应为 dict"
            )
            # 必须可 JSON 序列化（OpenAI API 要求）
            json.dumps(params)
            # required 若存在，每项都必须在 properties 中定义
            required = params.get("required")
            if required is not None:
                assert isinstance(required, list), (
                    f"工具 {fn.get('name')} 的 required 应为 list"
                )
                for r in required:
                    assert r in props, (
                        f"工具 {fn.get('name')} 的 required 参数 {r!r} "
                        f"未在 properties 中定义"
                    )

    def test_tool_names_unique(self):
        tools_mod = _import_tools()
        names = [t["function"]["name"] for t in tools_mod.TOOL_REGISTRY]
        assert len(names) == len(set(names)), (
            f"TOOL_REGISTRY 存在重名工具: {names}"
        )

    def test_get_tool_catalog_nonempty(self):
        tools_mod = _import_tools()
        catalog = tools_mod.get_tool_catalog()
        assert isinstance(catalog, str) and catalog.strip(), (
            f"get_tool_catalog() 应返回非空字符串，实际 {catalog!r}"
        )


# ════════════════════════════════════════════════════════════════
# 2. execute_tool 分发
# ════════════════════════════════════════════════════════════════


def _sample_args(tool: dict) -> dict:
    """按工具 schema 的 required 参数构造样例入参（底层数据函数已 mock，取值不重要）。"""
    params = tool["function"].get("parameters") or {}
    args = {}
    for p in params.get("required") or []:
        pl = p.lower()
        if "date" in pl:
            args[p] = FIXED_DATE_STR
        elif any(k in pl for k in ("sector", "industry", "board")):
            args[p] = "煤炭"
        elif any(k in pl for k in ("code", "symbol")):
            args[p] = "600519.SH"
        elif any(k in pl for k in ("limit", "count", "num", "top")):
            args[p] = 5
        else:
            args[p] = "测试参数"
    return args


def _patch_data_layer(stack: ExitStack, tool_name: str, mock_fn) -> int:
    """把数据层函数的所有可能引用点替换为 mock_fn，返回替换处数量。

    优先按 execute_tool 的工具名→数据层函数名映射（如 _IMPL）定位真实
    分发目标——execute_tool 运行时 getattr(data_fetcher, fn_name)，
    patch data_fetcher 命名空间即可拦截。同时兼容另外两种接线方式：
    工具名与数据层函数同名（动态 getattr / 模块顶层引用）、
    模块级 dispatch dict（{工具名: 函数}）。
    返回 0 表示找不到任何可 mock 的引用点——接线缺失，让用例如实失败。
    """
    tools_mod = _import_tools()
    patched = 0
    # 1. 工具名→数据层函数名映射：映射值可能是 (函数名, 参数适配器) 元组，
    #    也可能直接是函数名字符串
    impl_map = getattr(tools_mod, "_IMPL", None) or {}
    entry = impl_map.get(tool_name)
    if isinstance(entry, (list, tuple)) and entry:
        entry = entry[0]
    if isinstance(entry, str) and callable(getattr(data_fetcher, entry, None)):
        stack.enter_context(patch(f"agent.data_fetcher.{entry}", mock_fn))
        patched += 1
    # 2. 工具名即数据层函数名
    for mod, modname in ((data_fetcher, "agent.data_fetcher"),
                         (tools_mod, "agent.tools")):
        if callable(getattr(mod, tool_name, None)):
            stack.enter_context(patch(f"{modname}.{tool_name}", mock_fn))
            patched += 1
    # 3. 模块级 dispatch dict：直接换 entry，退出 with 时还原
    for attr in vars(tools_mod).values():
        if isinstance(attr, dict) and callable(attr.get(tool_name)):
            original = attr[tool_name]
            attr[tool_name] = mock_fn
            stack.callback(dict.__setitem__, attr, tool_name, original)
            patched += 1
    return patched


def _wire_first_dispatchable_tool(stack: ExitStack, mock_fn):
    """找到第一个能对上数据层函数的工具并完成 mock 接线。

    返回 (tool_dict, sample_args)；一个都对不上则返回 (None, None)。
    """
    tools_mod = _import_tools()
    for tool in tools_mod.TOOL_REGISTRY:
        name = tool["function"]["name"]
        if _patch_data_layer(stack, name, mock_fn):
            return tool, _sample_args(tool)
    return None, None


class TestExecuteToolDispatch:
    """execute_tool(name, args) -> {"ok": bool, "data"|"error"} 的分发行为。"""

    def test_known_tool_dispatch_ok(self):
        tools_mod = _import_tools()
        mock_fn = MagicMock(return_value={"pe": 10.0, "note": ""})
        with ExitStack() as stack:
            tool, args = _wire_first_dispatchable_tool(stack, mock_fn)
            assert tool is not None, (
                "TOOL_REGISTRY 中没有任何工具能对上数据层函数——"
                "execute_tool 的数据层接线缺失"
            )
            result = tools_mod.execute_tool(tool["function"]["name"], args)

        assert isinstance(result, dict), f"execute_tool 应返回 dict: {result!r}"
        assert result.get("ok") is True, (
            f"数据层正常返回时 execute_tool 应 ok=True: {result}"
        )
        assert "data" in result, f"ok=True 时应带 data 字段: {result}"
        assert mock_fn.call_count >= 1, "execute_tool 未真正调用数据层函数"

    def test_unknown_tool_returns_error_not_raise(self):
        tools_mod = _import_tools()
        # 不得抛异常
        result = tools_mod.execute_tool("__definitely_not_a_tool__", {})
        assert isinstance(result, dict), f"execute_tool 应返回 dict: {result!r}"
        assert result.get("ok") is False, f"未知工具应 ok=False: {result}"
        assert result.get("error"), f"未知工具应带 error 说明: {result}"

    def test_data_layer_exception_safe_degrade(self):
        """数据层函数抛异常 → ok=False + error，绝不向上抛。"""
        tools_mod = _import_tools()
        mock_fn = MagicMock(side_effect=Exception("数据层炸了"))
        with ExitStack() as stack:
            tool, args = _wire_first_dispatchable_tool(stack, mock_fn)
            assert tool is not None, (
                "TOOL_REGISTRY 中没有任何工具能对上数据层函数——"
                "execute_tool 的数据层接线缺失"
            )
            # 不得抛异常
            result = tools_mod.execute_tool(tool["function"]["name"], args)

        assert isinstance(result, dict), f"execute_tool 应返回 dict: {result!r}"
        assert result.get("ok") is False, (
            f"数据层抛异常时应 ok=False 安全降级: {result}"
        )
        assert result.get("error"), f"失败时应带 error 说明: {result}"


# ════════════════════════════════════════════════════════════════
# 3~5. Agent 工具循环：完整流程 / 轮次上限 / 异常降级
# ════════════════════════════════════════════════════════════════


def _patch_execute_tool(stack: ExitStack, exec_mock) -> None:
    """execute_tool 的 mock 接线。

    orchestrator 可能顶层 from import（patch orchestrator 命名空间），
    也可能持模块引用调用 tools.execute_tool（patch tools 命名空间）——
    两处都存在则都 patch；agent.tools 未落地则 ImportError 如实失败。
    """
    _import_tools()
    if hasattr(orchestrator, "execute_tool"):
        stack.enter_context(patch("agent.orchestrator.execute_tool", exec_mock))
    stack.enter_context(patch("agent.tools.execute_tool", exec_mock))


class TestAgentLoop:
    """_agent_query：tool_calls → execute_tool → role=tool 回灌 → 纯文本。"""

    def test_tool_call_then_text_full_loop(self):
        tools_mod = _import_tools()
        tool_name = tools_mod.TOOL_REGISTRY[0]["function"]["name"]
        agent = _make_agent()
        agent.client.chat.completions.create = AsyncMock(side_effect=[
            # 第 1 次：模型要求调工具
            _completion_with_tool_calls([_make_tool_call(tool_name, {}, "call_1")]),
            # 第 2 次：纯文本，循环结束
            _completion_text("最终分析文本"),
        ])
        # 工具结果用 ASCII marker，避免 json.dumps 转义后断言失真
        exec_mock = MagicMock(
            return_value={"ok": True, "data": {"pe_compare": "baijiu_lower"}}
        )

        with ExitStack() as stack:
            _patch_execute_tool(stack, exec_mock)
            result = asyncio.run(
                agent._agent_query("比较一下白酒和半导体的估值", stream=False)
            )

        create = agent.client.chat.completions.create
        assert create.await_count == 2, (
            f"一轮工具调用 + 一轮纯文本，create 应恰好 2 次，实际 {create.await_count}"
        )
        # execute_tool 被调用，且工具名正确
        assert exec_mock.call_count == 1, (
            f"execute_tool 应被调用 1 次，实际 {exec_mock.call_count}"
        )
        assert tool_name in str(exec_mock.call_args), (
            f"execute_tool 应以工具名 {tool_name!r} 调用: {exec_mock.call_args}"
        )
        # 工具结果以 role="tool" 消息回灌进第二轮 messages
        second_messages = create.await_args_list[1].kwargs["messages"]
        tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        assert tool_msgs, (
            f"第二轮 messages 缺少 role=\"tool\" 的工具结果回灌: {second_messages}"
        )
        assert any("baijiu_lower" in str(m.get("content")) for m in tool_msgs), (
            f"role=\"tool\" 消息未携带 execute_tool 的返回数据: {tool_msgs}"
        )
        # 最终输出为第 2 次（纯文本）的内容
        assert isinstance(result, dict), f"_agent_query 应返回 dict: {result!r}"
        assert "最终分析文本" in result["content"], (
            f"最终输出应为第 2 次纯文本: {result['content']!r}"
        )

    def test_round_cap_stops_and_degrades(self):
        """模型永远返回 tool_calls → 循环在 8 轮停止后基于已检索成果强制成文
        （第 9 次 create 不带 tools），仅在强制成文异常时才降级 _chat。"""
        tools_mod = _import_tools()
        tool_name = tools_mod.TOOL_REGISTRY[0]["function"]["name"]
        agent = _make_agent()
        counter = itertools.count(1)
        agent.client.chat.completions.create = AsyncMock(
            side_effect=lambda **kwargs: _completion_with_tool_calls(
                [_make_tool_call(tool_name, {}, f"call_{next(counter)}")]
            )
        )
        exec_mock = MagicMock(return_value={"ok": True, "data": {"n": 1}})
        agent._chat = AsyncMock(
            return_value={"role": "assistant", "content": "降级闲聊回答"}
        )

        with ExitStack() as stack:
            _patch_execute_tool(stack, exec_mock)
            # asyncio.run 能返回即证明没有无限循环
            result = asyncio.run(agent._agent_query("持续追问", stream=False))

        create = agent.client.chat.completions.create
        assert create.await_count == 9, (
            f"8 轮工具循环 + 1 次强制成文，create 实际被调 {create.await_count} 次"
        )
        # 强制成文调用禁止再调工具（不传 tools），并携带「工具用尽」指令
        last_kwargs = create.await_args_list[-1].kwargs
        assert "tools" not in last_kwargs
        assert any(
            "工具调用次数已用完" in str(m.get("content", ""))
            for m in last_kwargs["messages"] if isinstance(m, dict)
        )
        assert agent._chat.await_count == 0, (
            "强制成文成功时不应降级 _chat"
        )
        # 本 mock 下强制成文拿到的是空 draft，只验证未走 _chat 降级结果
        assert result["content"] != "降级闲聊回答", (
            f"强制成文路径不应返回 _chat 的结果: {result!r}"
        )

    def test_create_exception_degrades_to_chat(self):
        """create 首轮抛异常 → 降级 _chat，不向上抛。"""
        _import_tools()
        agent = _make_agent()
        agent.client.chat.completions.create = AsyncMock(
            side_effect=Exception("DeepSeek 500")
        )
        agent._chat = AsyncMock(
            return_value={"role": "assistant", "content": "降级闲聊回答"}
        )

        # 不得抛异常
        result = asyncio.run(agent._agent_query("随便问点复杂的", stream=False))

        assert agent._chat.await_count == 1, "create 抛异常后应降级 _chat"
        assert result["content"] == "降级闲聊回答", (
            f"降级后应返回 _chat 的结果: {result!r}"
        )


# ════════════════════════════════════════════════════════════════
# 6. process_message 路由：复杂分析 → _agent_query；闲聊 → _chat
# ════════════════════════════════════════════════════════════════


class TestProcessMessageRouting:
    """general_chat 里的复杂分析特征走工具循环，普通闲聊仍走 _chat。"""

    def test_complex_comparison_routes_to_agent_query(self):
        agent = _make_agent()
        agent._agent_query = AsyncMock(
            return_value={"role": "assistant", "content": "agent答案"}
        )
        agent._chat = AsyncMock(
            return_value={"role": "assistant", "content": "闲聊答案"}
        )
        agent._sector_deep_dive = AsyncMock(
            return_value={"role": "assistant", "content": "深挖答案"}
        )

        result = asyncio.run(
            agent.process_message("比较一下白酒和半导体的估值", stream=False)
        )

        assert agent._agent_query.await_count == 1, (
            "复杂分析（比较/对比类）应路由到 _agent_query 工具循环"
        )
        assert agent._chat.await_count == 0, "复杂分析不应落入普通闲聊 _chat"
        assert agent._sector_deep_dive.await_count == 0, (
            "复杂分析不应落入单板块深挖路径"
        )
        assert result["content"].startswith("agent答案")
        assert "风险提示" in result["content"]  # 出口统一兜底追加

    def test_small_talk_routes_to_chat(self):
        agent = _make_agent()
        agent._agent_query = AsyncMock(
            return_value={"role": "assistant", "content": "agent答案"}
        )
        agent._chat = AsyncMock(
            return_value={"role": "assistant", "content": "闲聊答案"}
        )

        result = asyncio.run(agent.process_message("你好呀", stream=False))

        assert agent._chat.await_count == 1, "普通闲聊应仍走 _chat"
        assert agent._agent_query.await_count == 0, (
            "普通闲聊不应进入 _agent_query 工具循环"
        )
        assert result["content"] == "闲聊答案"


# ════════════════════════════════════════════════════════════════
# 7. 多 pass：非流式 _sector_deep_dive 先草稿再审查修正
# ════════════════════════════════════════════════════════════════

# 长草稿（>500 字，去掉免责声明也超阈值）→ 必须走审查修正
_DRAFT_SENTENCE = "煤炭板块今日整体平稳，估值处于历史中枢附近，资金观望情绪较浓。"
LONG_DRAFT = "【草稿】" + _DRAFT_SENTENCE * 18  # ~560 字
# 修正后终稿（审查 pass 的 LLM 返回；需 ≥50 字，过短会被实现视为修正失败回退原稿）
REVISED = ("修正后终稿：煤炭板块估值处于历史中枢，资金观望情绪较浓，"
           "缺乏新增催化，短期维持中性观察，等待景气度信号进一步明确。")


def _sector_deep_dive_patches(stack: ExitStack) -> None:
    """板块深挖数据采集侧的全部 mock（对齐 test_orchestrator 的接线方式）。"""
    stack.enter_context(patch(
        "agent.orchestrator._get_latest_trade_date",
        return_value=FIXED_TRADE_DATE,
    ))
    stack.enter_context(patch(
        "agent.orchestrator.collect_market_snapshot",
        AsyncMock(return_value=MagicMock(name="snapshot")),
    ))
    stack.enter_context(patch(
        "agent.orchestrator.format_market_data_for_prompt", return_value="DATA"
    ))
    # 估值/资金/景气度附加数据：_fetch_sector_extras 内局部 import，
    # 因此 patch data_fetcher 命名空间（函数已存在，不用 create=True）
    stack.enter_context(patch(
        "agent.data_fetcher.fetch_sector_valuation",
        MagicMock(return_value={"pe": 10.0}),
    ))
    stack.enter_context(patch(
        "agent.data_fetcher.fetch_sector_moneyflow",
        MagicMock(return_value={"main_net": 1.0}),
    ))
    stack.enter_context(patch(
        "agent.data_fetcher.fetch_sector_earnings",
        MagicMock(return_value={"note": ""}),
    ))


class TestMultiPassCritiqueRevise:
    """_critique_and_revise：草稿 → 审查 → 修正终稿的多 pass 生成。"""

    def test_long_draft_goes_through_critique_and_revise(self):
        agent = _make_agent()
        state = itertools.count(1)
        # 第 1 次 create 返回长草稿；其后（审查/修正 pass）一律返回终稿
        agent.client.chat.completions.create = AsyncMock(
            side_effect=lambda **kwargs: (
                _completion_text(LONG_DRAFT) if next(state) == 1
                else _completion_text(REVISED)
            )
        )
        # wraps 原方法：既记录调用又执行真实审查逻辑；
        # 方法未接线时 AttributeError 如实失败
        agent._critique_and_revise = AsyncMock(wraps=agent._critique_and_revise)

        with ExitStack() as stack:
            _sector_deep_dive_patches(stack)
            result = asyncio.run(agent._sector_deep_dive("煤炭", stream=False))

        assert agent._critique_and_revise.await_count == 1, (
            "非流式 _sector_deep_dive 未经过 _critique_and_revise 审查修正路径"
        )
        create = agent.client.chat.completions.create
        assert create.await_count >= 2, (
            f"多 pass 至少应调用 2 次 LLM（草稿 + 审查修正），实际 {create.await_count}"
        )
        assert isinstance(result, dict), f"应返回 dict: {result!r}"
        assert REVISED in result["content"], (
            f"最终输出应为审查修正后的终稿: {result['content']!r}"
        )
        assert "【草稿】" not in result["content"], (
            "草稿原文不应出现在最终输出中"
        )

    def test_short_draft_skips_critique(self):
        """draft < 500 字（含免责声明也远低于阈值）→ 跳过审查，原样返回。"""
        agent = _make_agent()
        short = "板块数据暂缺，无法展开分析。"
        agent.client.chat.completions.create = AsyncMock(
            side_effect=lambda **kwargs: _completion_text(short)
        )
        agent._critique_and_revise = AsyncMock(wraps=agent._critique_and_revise)

        with ExitStack() as stack:
            _sector_deep_dive_patches(stack)
            result = asyncio.run(agent._sector_deep_dive("煤炭", stream=False))

        create = agent.client.chat.completions.create
        assert create.await_count == 1, (
            f"draft <500 字时应跳过审查（只调 1 次 LLM），实际 {create.await_count} 次"
        )
        assert isinstance(result, dict)
        assert short in result["content"], (
            f"跳过审查时应原样返回 draft: {result['content']!r}"
        )

    def test_critique_failure_returns_original_draft(self):
        """审查 pass 的 LLM 调用失败 → 降级返回原 draft，不向上抛。"""
        agent = _make_agent()
        state = itertools.count(1)

        def _side(**kwargs):
            if next(state) == 1:
                return _completion_text(LONG_DRAFT)
            raise Exception("审查模型调用失败")

        agent.client.chat.completions.create = AsyncMock(side_effect=_side)

        with ExitStack() as stack:
            _sector_deep_dive_patches(stack)
            # 不得抛异常
            result = asyncio.run(agent._sector_deep_dive("煤炭", stream=False))

        assert isinstance(result, dict), f"应返回 dict: {result!r}"
        assert "【草稿】" in result["content"], (
            f"审查失败时应降级返回原 draft: {result['content']!r}"
        )


# ════════════════════════════════════════════════════════════════
# 8. 第二波 prompt 内容断言
# ════════════════════════════════════════════════════════════════

# 与 test_sector_deep 保持一致的 AI 味禁用词候选清单
BANNED_WORD_CANDIDATES = ["护城河", "飞轮", "赋能", "抓手", "闭环",
                          "颗粒度", "生态化反", "降维打击"]


class TestAgentPrompts:
    """AGENT_QUERY_PROMPT / CRITIQUE_PROMPT 的红线与检查项断言。

    prompt 常量未接线时 AttributeError 如实失败。
    """

    def test_agent_query_prompt_data_redline(self):
        """AGENT_QUERY_PROMPT 必须含数据红线表述（禁止编造/虚构或 UNSOURCED 标注）。"""
        prompt = system_prompts.AGENT_QUERY_PROMPT
        assert any(w in prompt for w in
                   ("编造", "虚构", "捏造", "UNSOURCED", "数据暂缺")), (
            "AGENT_QUERY_PROMPT 缺少数据红线表述（禁止编造/虚构 或 缺失标注）"
        )

    def test_agent_query_prompt_banned_words(self):
        """AGENT_QUERY_PROMPT 的禁用词清单至少含 3 个 AI 味词汇。"""
        prompt = system_prompts.AGENT_QUERY_PROMPT
        hits = [w for w in BANNED_WORD_CANDIDATES if w in prompt]
        assert len(hits) >= 3, (
            f"AGENT_QUERY_PROMPT 至少应含 3 个禁用词，实际仅命中 {hits}"
        )

    def test_critique_prompt_number_source_check(self):
        """CRITIQUE_PROMPT 必须含数字出处核对检查项。"""
        prompt = system_prompts.CRITIQUE_PROMPT
        assert any(w in prompt for w in ("数字", "数据", "数值")), (
            "CRITIQUE_PROMPT 缺少数字/数据核对检查项"
        )
        assert any(w in prompt for w in ("出处", "来源", "依据", "数据块")), (
            "CRITIQUE_PROMPT 缺少数字出处（来源/依据）核对要求"
        )

    def test_critique_prompt_banned_words_check(self):
        """CRITIQUE_PROMPT 必须含禁用词检查项。"""
        prompt = system_prompts.CRITIQUE_PROMPT
        assert "禁用" in prompt, "CRITIQUE_PROMPT 缺少禁用词检查项"

    def test_critique_prompt_out_of_bounds_check(self):
        """CRITIQUE_PROMPT 必须含越界（投资建议/荐股/合规）检查项。"""
        prompt = system_prompts.CRITIQUE_PROMPT
        assert any(w in prompt for w in
                   ("越界", "投资建议", "荐股", "买卖建议", "合规")), (
            "CRITIQUE_PROMPT 缺少越界（投资建议/荐股/合规）检查项"
        )
