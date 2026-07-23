"""tests/test_output_hygiene.py — 输出卫生专项：去来源标签 + 全面禁 #/* + MCP 兜底综合 + 提示语分级。

覆盖范围：
1. _clean_markdown 强化：行内 *斜体*、残留单个 * 与 #、未配对 ** 全清除；
   正常中文财经文本零误伤；幂等。
2. _strip_md_symbols：符号级删除 * 与 #，无状态、空值安全。
3. _stream_response 流式清洗：脏 chunk（# 标题 / **加粗** / *斜体*）清洗后
   无 #/*；跨 chunk 拆分的 ** 拼接结果与整段清洗一致；干净 chunk 边界不变。
4. _news_only 展示行：条目格式「[时间] 标题」无【来源】标签；头部保留
   「来源：xxxN条」统计行；外部标题混入的 #/* 被清洗。
5. 板块新闻解读段：非流式与流式（_list_then_analysis）两条路径，
   LLM 解读文本中的 Markdown 符号都到不了用户。
6. _generic_mcp：循环跑满后不再 dump 原始 JSON，改走无 tools 最终综合；
   综合失败/为空时返回优雅提示；空数据早退回归；is_mcp_error/compact_mcp_result
   防御式集成与未就绪降级；正常 stop 回答清洗 markdown。
7. _stock_query / _futures_query / _fund_query / 自选股输出：数据里混入的
   #/* 被清洗。
8. main.py 流式加载提示语分级：仅 market_review / sector_deep_dive 发
   「正在采集…」提示，其他意图不发。
9. system_prompts：新闻解读等提示词明确要求纯文本、禁止 # 和 *。

规则（与项目其他测试一致）：
- 所有外部依赖全部 mock（fetch_* / _mcp_call / DeepSeek 客户端），零网络。
- _get_latest_trade_date 统一 patch 为固定交易日。
- 无 pytest-asyncio，异步函数一律用 asyncio.run 驱动。
"""

import asyncio
import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agent.orchestrator as orch_mod
import main
from agent import data_fetcher
from agent.orchestrator import MarketReviewAgent

# 固定交易日（周五），避免 _get_latest_trade_date 触达真实 tushare
FIXED_TRADE_DATE = datetime(2025, 1, 10)
DAY0 = "2025-01-10"

POOL_KEYS = {"mcp", "flash"}


def _make_agent() -> MarketReviewAgent:
    """构造 agent，DeepSeek 客户端替换为 mock，防止任何真实 HTTP 调用。"""
    agent = MarketReviewAgent()
    agent.client = MagicMock()
    return agent


def _empty_pool() -> dict:
    return {k: [] for k in POOL_KEYS}


def _item(title: str, time: str, source: str) -> dict:
    return {"title": title, "time": time, "source": source}


def _fake_completion(content: str = "正文", finish_reason: str = "stop", tool_calls=None):
    """伪造 chat.completions.create 的非流式返回对象。"""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=tool_calls),
            finish_reason=finish_reason,
        )]
    )


def _make_tool_call(name: str, arguments: dict, call_id: str):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
    )


def _chunk(content, finish_reason=None):
    """伪造一个流式 chunk。"""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content=content),
            finish_reason=finish_reason,
        )]
    )


# ════════════════════════════════════════════════════════════════
# 1. _clean_markdown 强化
# ════════════════════════════════════════════════════════════════


class TestCleanMarkdownStrengthened:
    """行内斜体、残留单 * 与 #、未配对 ** 全部清除；正常文本零误伤。"""

    def test_inline_italic_removed(self):
        assert orch_mod._clean_markdown("这是*斜体*文本") == "这是斜体文本"

    def test_bold_and_heading_removed(self):
        out = orch_mod._clean_markdown("## 标题\n**加粗**正文")
        assert out == "标题\n加粗正文"

    def test_residual_lone_star_removed(self):
        # 未配对 ** / 孤立 * 残留也必须清除
        assert orch_mod._clean_markdown("前半**未闭合") == "前半未闭合"
        assert orch_mod._clean_markdown("孤立*星号") == "孤立星号"
        assert orch_mod._clean_markdown("***三星***") == "三星"

    def test_residual_lone_hash_removed(self):
        assert orch_mod._clean_markdown("孤立#井号") == "孤立井号"
        assert orch_mod._clean_markdown("##### 五级标题") == "五级标题"
        assert orch_mod._clean_markdown("话题#财经#热议") == "话题财经热议"

    def test_mixed_markdown_fully_cleaned(self):
        dirty = "## 一、市场概览\n**上证指数** 涨 *1.2*%\n* 要点一\n* 要点二"
        out = orch_mod._clean_markdown(dirty)
        assert "#" not in out and "*" not in out
        assert "上证指数" in out and "要点一" in out
        # * 列表转为 - 列表（- 不在禁令内）
        assert "- 要点一" in out

    def test_pipe_table_still_converted(self):
        """回归：管道表格仍转缩进纯文本，不残留 | 表格线。"""
        table = "| 板块 | 涨跌幅 |\n|---|---|\n| 煤炭 | 1.2% |"
        out = orch_mod._clean_markdown(table)
        assert "|" not in out
        assert "煤炭" in out and "1.2%" in out

    @pytest.mark.parametrize("text", [
        "沪深两市成交额突破一万五千亿元",
        "上证指数 3891.25 点，涨跌幅 +1.25%；深证成指涨跌幅 -0.83%。",
        "【一、趋势位置】板块区间涨跌幅 2.35%，成交 999.99 亿。",
        "PE(TTM) 18.52 倍，PB 1.85 倍，ROE 12.40%。",
        "回复『复盘我的自选股』查看逐只复盘。",
        "- 列表项一：主力净流入 12.40 亿",
        "贵州茅台（600519）：现价 1500.00，涨跌幅 1.20%",
    ])
    def test_normal_chinese_text_untouched(self, text):
        """正常中文财经文本零误伤（【】、数字、%、『』、- 列表、括号保留）。"""
        assert orch_mod._clean_markdown(text) == text

    def test_idempotent(self):
        dirty = "## 标题\n**加粗**与*斜体*，残留#井号*星号"
        once = orch_mod._clean_markdown(dirty)
        assert orch_mod._clean_markdown(once) == once

    def test_empty_and_none_safe(self):
        assert orch_mod._clean_markdown("") == ""
        assert orch_mod._clean_markdown(None) == ""


# ════════════════════════════════════════════════════════════════
# 2. _strip_md_symbols 单元行为
# ════════════════════════════════════════════════════════════════


class TestStripMdSymbols:

    def test_removes_star_and_hash_only(self):
        assert orch_mod._strip_md_symbols("a*b#c") == "abc"
        assert orch_mod._strip_md_symbols("纯文本，不动。【标题】- 列表 | 管道") \
            == "纯文本，不动。【标题】- 列表 | 管道"

    def test_empty_and_none_safe(self):
        assert orch_mod._strip_md_symbols("") == ""
        assert orch_mod._strip_md_symbols(None) == ""


# ════════════════════════════════════════════════════════════════
# 3. _stream_response 流式逐 chunk 清洗
# ════════════════════════════════════════════════════════════════


class TestStreamResponseCleaning:

    def _drive(self, agent):
        async def _run():
            gen = agent._stream_response([{"role": "user", "content": "x"}])
            chunks = []
            async for piece in gen:
                chunks.append(piece)
            return chunks
        return asyncio.run(_run())

    def test_dirty_chunks_cleaned(self):
        agent = _make_agent()

        async def fake_stream():
            yield _chunk("## 标题\n")
            yield _chunk("**加粗**与*斜体*")
            yield _chunk("收尾", finish_reason="stop")

        agent.client.chat.completions.create = AsyncMock(return_value=fake_stream())
        chunks = self._drive(agent)

        full = "".join(chunks)
        assert "#" not in full and "*" not in full
        assert "标题" in full and "加粗与斜体" in full and "收尾" in full

    def test_cross_chunk_double_star_pairing_irrelevant(self):
        """** 拆在两个 chunk：无状态逐字符删除，拼接结果与整段清洗一致。"""
        agent = _make_agent()

        async def fake_stream():
            yield _chunk("你*")
            yield _chunk("*好**世界*", finish_reason="stop")

        agent.client.chat.completions.create = AsyncMock(return_value=fake_stream())
        chunks = self._drive(agent)

        assert "".join(chunks) == "你好世界"
        # 与整段清洗结果完全一致
        assert "".join(chunks) == orch_mod._strip_md_symbols("你**好**世界*")

    def test_clean_chunks_boundaries_preserved(self):
        """不含符号的 chunk 原样透传，边界不变（回归保护）。"""
        agent = _make_agent()

        async def fake_stream():
            yield _chunk("你好")
            yield _chunk("世界", finish_reason="stop")

        agent.client.chat.completions.create = AsyncMock(return_value=fake_stream())
        assert self._drive(agent) == ["你好", "世界"]

    def test_symbol_only_chunk_not_yielded(self):
        """整个 chunk 都是符号时清洗后为空，不向用户发空 chunk。"""
        agent = _make_agent()

        async def fake_stream():
            yield _chunk("正文")
            yield _chunk("**")
            yield _chunk("结尾", finish_reason="stop")

        agent.client.chat.completions.create = AsyncMock(return_value=fake_stream())
        assert self._drive(agent) == ["正文", "结尾"]


# ════════════════════════════════════════════════════════════════
# 4. _news_only 展示行：去【来源】标签 + 头部统计保留 + 标题符号清洗
# ════════════════════════════════════════════════════════════════


class TestNewsOnlySourceLabelRemoval:

    def _run(self, pool, message="今天有什么新闻", stream=False):
        agent = _make_agent()
        agent._call_llm = AsyncMock(
            return_value={"role": "assistant", "content": "占位"}
        )
        with patch("agent.data_fetcher.fetch_news_pool", MagicMock(return_value=pool)), \
             patch("agent.orchestrator._get_latest_trade_date",
                   return_value=FIXED_TRADE_DATE):
            return asyncio.run(agent.process_message(message, stream=stream))

    def test_item_lines_have_no_source_label(self):
        pool = _empty_pool()
        pool["mcp"] = [
            _item("沪深两市成交额突破一万五千亿元", f"{DAY0} 09:30:00", "智研"),
            _item("央行开展5000亿元MLF操作", f"{DAY0} 11:00:00", "智研"),
        ]
        pool["flash"] = [
            _item("北向资金单日净流入超百亿元", f"{DAY0} 10:00:00", "智研快讯"),
        ]
        content = self._run(pool)["content"]

        # 任何来源展示名（含已弃用源）都不得以【】标签形式出现在条目前缀
        for label in ("【智研】", "【智研快讯】",
                      "【新浪】", "【东方财富】", "【新浪智研】", "【财联社】", "【Tushare】"):
            assert label not in content, f"条目不应再带来源标签 {label}:\n{content}"
        # 条目行格式：[时间] 标题
        assert f"[01-10 09:30] 沪深两市成交额突破一万五千亿元" in content
        assert f"[01-10 10:00] 北向资金单日净流入超百亿元" in content

    def test_header_source_stats_retained(self):
        """头部「来源：xxxN条」统计行保留（用户靠它验证多源出货）。"""
        pool = _empty_pool()
        pool["mcp"] = [
            _item("新闻条目标题甲内容", f"{DAY0} 09:30:00", "智研"),
            _item("新闻条目标题乙内容", f"{DAY0} 10:00:00", "智研"),
        ]
        pool["flash"] = [
            _item("北向资金单日净流入超百亿元", f"{DAY0} 11:00:00", "智研快讯"),
        ]
        content = self._run(pool)["content"]
        assert "来源：" in content
        assert "智研2条" in content and "智研快讯1条" in content

    def test_external_title_symbols_stripped(self):
        """外部新闻标题混入 #/* 时，展示清单不带这两个符号。"""
        pool = _empty_pool()
        pool["mcp"] = [
            _item("央行开展5000亿元MLF操作#重磅*解读", f"{DAY0} 09:30:00", "智研"),
        ]
        content = self._run(pool)["content"]
        assert "#" not in content and "*" not in content
        assert "央行开展5000亿元MLF操作重磅解读" in content


# ════════════════════════════════════════════════════════════════
# 5. 板块新闻解读段：非流式 + 流式两条路径的 LLM 解读清洗
# ════════════════════════════════════════════════════════════════


class TestSectorNewsAnalysisCleaning:

    BANK_TITLE = "央行宣布降准释放长期流动性支持银行体系"

    def _pool(self):
        pool = _empty_pool()
        pool["mcp"] = [_item(self.BANK_TITLE, f"{DAY0} 09:30:00", "智研")]
        return pool

    def test_nonstream_analysis_markdown_cleaned(self):
        """非流式：LLM 解读段含 Markdown 符号时，最终 content 无 #/*。"""
        agent = _make_agent()
        agent.client.chat.completions.create = AsyncMock(
            return_value=_fake_completion("## 一、主题归纳\n**偏多**：降准释放流动性，*利好*银行体系。")
        )
        with patch("agent.data_fetcher.fetch_news_pool",
                   MagicMock(return_value=self._pool())), \
             patch("agent.orchestrator._get_latest_trade_date",
                   return_value=FIXED_TRADE_DATE):
            result = asyncio.run(agent.process_message("银行板块的新闻", stream=False))

        content = result["content"]
        assert "#" not in content and "*" not in content, (
            f"非流式解读段仍含 Markdown 符号:\n{content}"
        )
        assert self.BANK_TITLE in content  # 确定性清单原样保留
        assert "偏多" in content and "降准释放流动性" in content

    def test_stream_analysis_markdown_cleaned(self):
        """流式（_list_then_analysis）：清单 chunk 与解读 chunk 都无 #/*。"""
        agent = _make_agent()

        async def fake_stream():
            yield _chunk("## 一、主题归纳\n")
            yield _chunk("**偏多**：")
            yield _chunk("降准*利好*银行", finish_reason="stop")

        agent.client.chat.completions.create = AsyncMock(return_value=fake_stream())

        async def _drive():
            gen = await agent.process_message("银行板块的新闻", stream=True)
            chunks = []
            async for c in gen:
                chunks.append(c)
            return chunks

        with patch("agent.data_fetcher.fetch_news_pool",
                   MagicMock(return_value=self._pool())), \
             patch("agent.orchestrator._get_latest_trade_date",
                   return_value=FIXED_TRADE_DATE):
            chunks = asyncio.run(_drive())

        assert chunks, "流式板块新闻应产出 chunk"
        full = "".join(chunks)
        assert "#" not in full and "*" not in full, (
            f"流式输出仍含 Markdown 符号:\n{full}"
        )
        # 第一块是确定性清单（不经过 LLM），且不带来源标签
        assert chunks[0].startswith("银行板块新闻汇总")
        assert self.BANK_TITLE in chunks[0]
        assert "【智研】" not in chunks[0]
        # 解读正文在清单之后流出
        assert "偏多" in full and "降准利好银行" in full
        assert full.index(self.BANK_TITLE) < full.index("偏多")


# ════════════════════════════════════════════════════════════════
# 6. _generic_mcp：兜底综合 + 结果压缩 + 错误简述
# ════════════════════════════════════════════════════════════════


class TestGenericMcp:

    TOOLS = [{"name": "globalStockSearchSymbols", "desc": "搜索股票代码",
              "params": ["keyword"]}]

    def _patch_mcp(self, mcp_return):
        return (
            patch("agent.data_fetcher.get_mcp_tools", MagicMock(return_value=self.TOOLS)),
            patch("agent.data_fetcher._mcp_call", MagicMock(return_value=mcp_return)),
        )

    def _loop_forever_create(self, tool_name="globalStockSearchSymbols"):
        """模型每轮都要求调工具的 create mock。"""
        counter = {"n": 0}

        def _side(**kwargs):
            counter["n"] += 1
            return _fake_completion(
                content=None, finish_reason="tool_calls",
                tool_calls=[_make_tool_call(tool_name, {"keyword": "鲟龙科技"},
                                            f"call_{counter['n']}")],
            )
        return AsyncMock(side_effect=_side)

    def test_max_rounds_exhausted_uses_final_synthesis_not_raw_json(self):
        """循环跑满 → 无 tools 最终综合，用户拿到人话总结而非原始 JSON dump。"""
        agent = _make_agent()
        synthesis_text = "鲟龙科技财务数据暂未完整获取：营收与净利数据可用，其余指标数据源返回异常。"
        p_tools, p_call = self._patch_mcp({"result": {"data": [{"name": "鲟龙科技", "code": "835342"}]}})
        with p_tools, p_call:
            agent.client.chat.completions.create = AsyncMock(side_effect=[
                _fake_completion(content=None, finish_reason="tool_calls",
                                 tool_calls=[_make_tool_call(
                                     "globalStockSearchSymbols", {"keyword": "鲟龙科技"}, "call_1")]),
                _fake_completion(content=None, finish_reason="tool_calls",
                                 tool_calls=[_make_tool_call(
                                     "globalStockSearchSymbols", {"keyword": "鲟龙科技"}, "call_2")]),
                _fake_completion(synthesis_text),  # 最终综合
            ])
            result = asyncio.run(
                agent._generic_mcp("查询鲟龙科技近期的财务指标", False, max_rounds=2)
            )

        create = agent.client.chat.completions.create
        assert create.await_count == 3, "两轮工具循环 + 一次最终综合"
        # 最终综合调用不传 tools（模型只能写正文）
        synth_kwargs = create.await_args_list[-1].kwargs
        assert "tools" not in synth_kwargs or synth_kwargs.get("tools") is None
        # 综合上下文带用户问题与已收集工具结果
        synth_messages = synth_kwargs["messages"]
        assert any("鲟龙科技" in str(m.get("content")) for m in synth_messages)
        # 用户拿到综合文本，绝不是原始 JSON dump
        content = result["content"]
        assert synthesis_text in content
        assert "查询结果" not in content
        assert "globalStockSearchSymbols" not in content
        assert "835342" not in content, f"原始字段值不应透给用户: {content}"

    def test_final_synthesis_failure_returns_graceful_message(self):
        """最终综合调用抛异常 → 优雅「数据暂未获取成功」类提示，仍无 JSON。"""
        agent = _make_agent()
        p_tools, p_call = self._patch_mcp({"result": {"data": [{"name": "x"}]}})
        with p_tools, p_call:
            agent.client.chat.completions.create = AsyncMock(side_effect=[
                _fake_completion(content=None, finish_reason="tool_calls",
                                 tool_calls=[_make_tool_call(
                                     "globalStockSearchSymbols", {}, "call_1")]),
                RuntimeError("DeepSeek 综合调用超时"),
            ])
            result = asyncio.run(agent._generic_mcp("查询财务指标", False, max_rounds=1))

        content = result["content"]
        assert "暂未获取成功" in content or "暂不可用" in content or "重试" in content, (
            f"综合失败应返回优雅降级提示: {content!r}"
        )
        assert "{" not in content and "globalStockSearchSymbols" not in content

    def test_final_synthesis_empty_returns_graceful_message(self):
        """最终综合返回空串 → 优雅降级提示。"""
        agent = _make_agent()
        p_tools, p_call = self._patch_mcp({"result": {"data": [{"name": "x"}]}})
        with p_tools, p_call:
            agent.client.chat.completions.create = AsyncMock(side_effect=[
                _fake_completion(content=None, finish_reason="tool_calls",
                                 tool_calls=[_make_tool_call(
                                     "globalStockSearchSymbols", {}, "call_1")]),
                _fake_completion(""),
            ])
            result = asyncio.run(agent._generic_mcp("查询财务指标", False, max_rounds=1))

        assert result["content"] == orch_mod._MCP_DATA_UNAVAILABLE

    def test_empty_tool_result_fallback_early_exit(self):
        """回归：工程师B 函数未就绪（None）时，工具返回空（{}）→ 维持既有
        「暂不支持查询」早退（防御式降级=现状行为）。"""
        agent = _make_agent()
        p_tools, p_call = self._patch_mcp({})
        with p_tools, p_call, \
             patch.object(orch_mod, "is_mcp_error", None), \
             patch.object(orch_mod, "mcp_error_brief", None), \
             patch.object(orch_mod, "compact_mcp_result", None):
            agent.client.chat.completions.create = AsyncMock(
                return_value=_fake_completion(
                    content=None, finish_reason="tool_calls",
                    tool_calls=[_make_tool_call("globalStockSearchSymbols", {}, "call_1")],
                )
            )
            result = asyncio.run(agent._generic_mcp("查询财务指标", False, max_rounds=1))

        assert "暂不支持查询" in result["content"]

    def test_empty_result_with_error_helpers_feeds_brief(self):
        """工程师B 错误识别就绪时：空结果不再早退，而是压成简短说明喂回模型，
        由模型基于错误说明给出优雅回答（不反复重试、不 dump 原文）。"""
        agent = _make_agent()
        p_tools, p_call = self._patch_mcp({})
        with p_tools, p_call, \
             patch.object(orch_mod, "is_mcp_error", lambda d: True), \
             patch.object(orch_mod, "mcp_error_brief", lambda d: "无有效数据"), \
             patch.object(orch_mod, "compact_mcp_result", MagicMock()):
            agent.client.chat.completions.create = AsyncMock(side_effect=[
                _fake_completion(content=None, finish_reason="tool_calls",
                                 tool_calls=[_make_tool_call(
                                     "globalStockSearchSymbols", {}, "call_1")]),
                _fake_completion("该数据暂未获取成功，请换个问法再试。"),
            ])
            result = asyncio.run(agent._generic_mcp("查询财务指标", False))

        second_messages = agent.client.chat.completions.create.await_args_list[1].kwargs["messages"]
        tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        assert tool_msgs[0]["content"].startswith("[工具 globalStockSearchSymbols 返回错误]")
        assert "无有效数据" in tool_msgs[0]["content"]
        assert "暂未获取成功" in result["content"]

    def test_normal_stop_answer_markdown_cleaned(self):
        """模型轮次内直接回答：回答中的 Markdown 符号被清洗。"""
        agent = _make_agent()
        p_tools, p_call = self._patch_mcp({"result": {"data": [{"name": "x"}]}})
        with p_tools, p_call:
            agent.client.chat.completions.create = AsyncMock(
                return_value=_fake_completion("## 结论\n**上涨** 1.2%")
            )
            result = asyncio.run(agent._generic_mcp("今天涨跌如何", False))

        content = result["content"]
        assert "#" not in content and "*" not in content
        assert "上涨" in content

    def test_error_result_fed_back_as_brief(self):
        """工程师B 的 is_mcp_error/mcp_error_brief 就绪时：错误返回压成简短说明喂回。"""
        agent = _make_agent()
        p_tools, p_call = self._patch_mcp({"error": "HTTP_500"})
        with p_tools, p_call, \
             patch.object(orch_mod, "is_mcp_error", lambda d: True), \
             patch.object(orch_mod, "mcp_error_brief", lambda d: "HTTP_500 服务错误"), \
             patch.object(orch_mod, "compact_mcp_result", MagicMock()) as compact_mock:
            agent.client.chat.completions.create = AsyncMock(side_effect=[
                _fake_completion(content=None, finish_reason="tool_calls",
                                 tool_calls=[_make_tool_call(
                                     "globalStockSearchSymbols", {}, "call_1")]),
                _fake_completion("数据源返回异常，该指标暂未获取成功。"),
            ])
            result = asyncio.run(agent._generic_mcp("查询财务指标", False))

        # 喂回模型的是简短错误说明而非大段原文
        second_messages = agent.client.chat.completions.create.await_args_list[1].kwargs["messages"]
        tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        assert tool_msgs, "应有 role=tool 的工具结果回灌"
        assert tool_msgs[0]["content"].startswith("[工具 globalStockSearchSymbols 返回错误]")
        assert "HTTP_500 服务错误" in tool_msgs[0]["content"]
        # 错误路径不走 compact
        assert compact_mock.call_count == 0
        assert "暂未获取成功" in result["content"]

    def test_compact_applied_to_normal_result(self):
        """工程师B 的 compact_mcp_result 就绪时：正常结果先压缩再喂回。"""
        agent = _make_agent()
        raw = {"result": {"data": [{"name": "鲟龙科技", "filler": "长" * 100}]}}
        compacted = {"result": {"data": [{"name": "鲟龙科技"}]}}
        p_tools, p_call = self._patch_mcp(raw)
        with p_tools, p_call, \
             patch.object(orch_mod, "is_mcp_error", lambda d: False), \
             patch.object(orch_mod, "mcp_error_brief", lambda d: ""), \
             patch.object(orch_mod, "compact_mcp_result",
                          MagicMock(return_value=compacted)) as compact_mock:
            agent.client.chat.completions.create = AsyncMock(side_effect=[
                _fake_completion(content=None, finish_reason="tool_calls",
                                 tool_calls=[_make_tool_call(
                                     "globalStockSearchSymbols", {}, "call_1")]),
                _fake_completion("已查到鲟龙科技。"),
            ])
            asyncio.run(agent._generic_mcp("查询财务指标", False))

        assert compact_mock.call_count == 1
        second_messages = agent.client.chat.completions.create.await_args_list[1].kwargs["messages"]
        tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        assert "鲟龙科技" in tool_msgs[0]["content"]
        assert "filler" not in tool_msgs[0]["content"], "应喂压缩后的结果"

    def test_helpers_not_ready_falls_back_to_json_dumps(self):
        """工程师B 函数未就绪（None）时：退回 json.dumps 截断的现状行为。"""
        agent = _make_agent()
        raw = {"result": {"data": [{"name": "鲟龙科技"}]}}
        p_tools, p_call = self._patch_mcp(raw)
        with p_tools, p_call, \
             patch.object(orch_mod, "is_mcp_error", None), \
             patch.object(orch_mod, "mcp_error_brief", None), \
             patch.object(orch_mod, "compact_mcp_result", None):
            agent.client.chat.completions.create = AsyncMock(side_effect=[
                _fake_completion(content=None, finish_reason="tool_calls",
                                 tool_calls=[_make_tool_call(
                                     "globalStockSearchSymbols", {}, "call_1")]),
                _fake_completion("已查到。"),
            ])
            asyncio.run(agent._generic_mcp("查询财务指标", False))

        second_messages = agent.client.chat.completions.create.await_args_list[1].kwargs["messages"]
        tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        assert json.dumps(raw, ensure_ascii=False)[:3000] == tool_msgs[0]["content"]


class TestMcpResultForPrompt:
    """_mcp_result_for_prompt 单元行为：空判定 / 错误简述 / 压缩 / 异常降级。"""

    def test_empty_dict_fallback_returns_none(self):
        """防御式降级（工程师B 函数未就绪）：空 dict 走既有空判定 → None。"""
        with patch.object(orch_mod, "is_mcp_error", None), \
             patch.object(orch_mod, "mcp_error_brief", None), \
             patch.object(orch_mod, "compact_mcp_result", None):
            assert MarketReviewAgent._mcp_result_for_prompt("t", {}) is None

    def test_empty_data_list_fallback_legacy_quirk(self):
        """防御式降级下的历史怪癖如实保留：json.dumps 在冒号后带空格，
        既有 '\"data\":[]'（无空格）子串检查实际不命中，{"data": []} 退回
        json 原文（与改动前生产行为逐字一致，不在本次顺手「修复」）。"""
        with patch.object(orch_mod, "is_mcp_error", None), \
             patch.object(orch_mod, "mcp_error_brief", None), \
             patch.object(orch_mod, "compact_mcp_result", None):
            out = MarketReviewAgent._mcp_result_for_prompt("t", {"data": []})
        assert out == json.dumps({"data": []}, ensure_ascii=False)[:3000]

    def test_s_list_exception_kept(self):
        out = MarketReviewAgent._mcp_result_for_prompt(
            "t", {"data": [], "s_list": [{"a": 1}]}
        )
        assert out is not None and "s_list" in out

    def test_normal_result_json_truncated(self):
        big = {"data": [{"v": "长" * 4000}]}
        out = MarketReviewAgent._mcp_result_for_prompt("t", big)
        assert out is not None and len(out) <= 3000

    def test_is_mcp_error_exception_falls_back_to_normal(self):
        """is_mcp_error 自身抛异常：按正常数据处理，绝不向上抛。"""
        with patch.object(orch_mod, "is_mcp_error",
                          MagicMock(side_effect=RuntimeError("判定炸了"))):
            out = MarketReviewAgent._mcp_result_for_prompt("t", {"data": [{"a": 1}]})
        assert out is not None and "a" in out

    def test_compact_exception_falls_back_to_raw(self):
        """compact_mcp_result 抛异常：退回原始数据 json.dumps。"""
        raw = {"data": [{"a": 1}]}
        with patch.object(orch_mod, "is_mcp_error", lambda d: False), \
             patch.object(orch_mod, "compact_mcp_result",
                          MagicMock(side_effect=RuntimeError("压缩炸了"))):
            out = MarketReviewAgent._mcp_result_for_prompt("t", raw)
        assert out == json.dumps(raw, ensure_ascii=False)[:3000]


# ════════════════════════════════════════════════════════════════
# 7. _stock_query / _futures_query / _fund_query / 自选股输出清洗
# ════════════════════════════════════════════════════════════════


class TestDeterministicOutputsCleaning:

    def test_stock_query_news_symbols_cleaned(self):
        agent = _make_agent()
        with patch("agent.data_fetcher.fetch_stock_quote",
                   MagicMock(return_value={"price": 1500.0, "pct": 1.2})), \
             patch("agent.data_fetcher.fetch_stock_kline",
                   MagicMock(return_value=[{"date": "2025-01-10", "close": 1500.0}])), \
             patch("agent.data_fetcher.fetch_stock_news",
                   MagicMock(return_value=[{"title": "茅台年报#超预期*发布"}])):
            result = asyncio.run(agent._stock_query("茅台怎么样", False))

        content = result["content"]
        assert "#" not in content and "*" not in content
        assert "茅台年报超预期发布" in content

    def test_futures_query_symbols_cleaned(self):
        agent = _make_agent()
        with patch("agent.data_fetcher.fetch_futures_quote",
                   MagicMock(return_value={"price": "560.5#*", "pct": 0.8, "volume": 100})):
            result = asyncio.run(agent._futures_query("黄金期货价格", False))

        content = result["content"]
        assert "#" not in content and "*" not in content
        assert "560.5" in content

    def test_fund_query_hot_list_symbols_cleaned(self):
        agent = _make_agent()
        with patch("agent.data_fetcher.fetch_hot_stocks",
                   MagicMock(return_value=[{"name": "贵州茅台#*", "code": "600519",
                                            "heat": 99}])):
            result = asyncio.run(agent._fund_query("A股热搜榜", False))

        content = result["content"]
        assert "#" not in content and "*" not in content
        assert "贵州茅台" in content

    def test_watchlist_add_text_cleaned(self):
        agent = _make_agent()
        fake_watchlist = MagicMock()
        fake_watchlist.add_stock.return_value = (True, "已添加自选：**贵州茅台**（sh600519）#1")
        with patch.object(orch_mod, "watchlist", fake_watchlist):
            result = asyncio.run(agent._watchlist("加自选 茅台", False))

        content = result["content"]
        assert "#" not in content and "*" not in content
        assert "已添加自选：贵州茅台（sh600519）1" in content

    def test_watchlist_list_text_cleaned(self):
        agent = _make_agent()
        fake_watchlist = MagicMock()
        fake_watchlist.list_stocks.return_value = [
            {"name": "贵州茅台#*", "code": "sh600519"},
        ]
        with patch.object(orch_mod, "watchlist", fake_watchlist):
            result = asyncio.run(agent._watchlist("我的自选股", False))

        content = result["content"]
        assert "#" not in content and "*" not in content
        assert "贵州茅台" in content


# ════════════════════════════════════════════════════════════════
# 8. main.py 流式加载提示语分级
# ════════════════════════════════════════════════════════════════


class TestStreamHintTiering:

    def _drive_sse(self, user_message: str) -> list:
        """驱动 main._stream_chat_completion，解码全部 SSE chunk 的 delta.content
        列表（hint/正文/免责 chunk 的 json 默认 ensure_ascii，必须解码后断言）。"""
        agent = MagicMock()
        agent.cache_warm = True

        async def fake_answer():
            yield "答案正文"

        agent.process_message = AsyncMock(return_value=fake_answer())

        async def _collect():
            contents = []
            async for sse in main._stream_chat_completion(
                agent, user_message, "market-review-agent", []
            ):
                for line in sse.splitlines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    payload = json.loads(line[len("data: "):])
                    for choice in payload.get("choices", []):
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            contents.append(text)
            return contents

        return asyncio.run(_collect())

    @pytest.mark.parametrize("message", [
        "今日复盘",          # market_review
        "煤炭板块复盘",       # sector_deep_dive
        "电子行业怎么样",     # sector_deep_dive
    ])
    def test_heavy_intents_get_loading_hint(self, message):
        contents = self._drive_sse(message)
        full = "".join(contents)
        assert "正在采集" in full, (
            f"重数据采集意图 {message!r} 应发加载提示语"
        )
        assert "约需" in full, "提示语应带预计等待秒数"
        # 提示语在正文之前发出
        assert full.index("正在采集") < full.index("答案正文")
        assert "答案正文" in full  # 正常内容不受影响

    @pytest.mark.parametrize("message", [
        "给我讲个笑话",       # general_chat
        "今天有什么新闻",     # news_only
        "今天涨停家数查询",   # mcp_query
        "茅台怎么样",         # stock_query
        "黄金期货价格",       # futures_query
    ])
    def test_other_intents_no_loading_hint(self, message):
        contents = self._drive_sse(message)
        full = "".join(contents)
        assert "正在采集" not in full, (
            f"非重数据采集意图 {message!r} 不应发加载提示语"
        )
        assert "答案正文" in full, "不发提示语也应正常输出答案"
        assert "风险提示" in full  # 免责声明仍在


# ════════════════════════════════════════════════════════════════
# 9. system_prompts：纯文本 + 禁止 # 和 * 的明示要求
# ════════════════════════════════════════════════════════════════


class TestPromptPlainTextRequirements:

    @pytest.mark.parametrize("name", [
        "NEWS_ANALYSIS_PROMPT",
        "MARKET_REVIEW_PROMPT",
        "STOCK_ANALYSIS_PROMPT",
        "SECTOR_DEEP_DIVE_PROMPT",
        "AGENT_QUERY_PROMPT",
        "WATCHLIST_REVIEW_PROMPT",
    ])
    def test_prompt_requires_plain_text_no_md_symbols(self, name):
        from agent import system_prompts
        prompt = getattr(system_prompts, name)
        assert "纯文本" in prompt, f"{name} 缺少纯文本要求"
        assert "#和*号" in prompt, f"{name} 缺少禁止 # 和 * 的明示要求"

    def test_news_analysis_framework_intact(self):
        """新闻解读提示词强化后，原解读框架与红线结构不漂移。"""
        from agent.system_prompts import NEWS_ANALYSIS_PROMPT
        for marker in ("【一、主题归纳】", "【二、方向判断】",
                       "【三、与板块的关联】", "【四、关注清单】",
                       "数据真实性红线"):
            assert marker in NEWS_ANALYSIS_PROMPT, f"新闻解读框架标记 {marker!r} 丢失"
