"""tests/test_news.py — 新闻系统升级测试（48小时新闻池 + 重要性截断 + LLM 分析解读）。

覆盖范围：
1. data_fetcher.fetch_news_pool：五源（sina/eastmoney/mcp/cls/tushare）聚合结构、
   统一字段（title/time/source）、跨源标题去重、单源失败安全降级。
2. fetch_mcp_news：解析 _mcp_call 返回的原始数据后，每条 time 字段非空（回归修复）。
3. orchestrator._news_only 透传模式：按天分组、头部含总条数与各源统计、条目带来源标注。
4. 行业过滤：查『半导体新闻』时输出不含无关行业标题。
5. 重要性截断：全市场每天上限 30 条，高分新闻（业绩/政策/异动/公司行动）优先保留，
   即使它们在新闻池列表的末尾。
6. 分析模式：消息含『解读/分析/影响』时走 LLM 分析（NEWS_ANALYSIS_PROMPT），而非透传。
7. NEWS_ANALYSIS_PROMPT 本身：主题归纳/方向判断结构、数据红线表述、至少 3 个禁用词。

规则（与 tests/test_orchestrator.py 一致）：
- 所有外部依赖全部 mock（fetch_*_news / fetch_news_pool / _mcp_call / DeepSeek 客户端），
  绝不发起真实网络请求。
- _get_latest_trade_date 统一 patch 为固定交易日，避免触达真实 tushare 交易日历。
- 无 pytest-asyncio，异步函数一律用 asyncio.run 驱动。
"""

import asyncio
import contextlib
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agent.orchestrator as orch_mod
from agent import data_fetcher
from agent.orchestrator import MarketReviewAgent

# 固定交易日（周五），避免 _get_latest_trade_date 触达真实 tushare
FIXED_TRADE_DATE = datetime(2025, 1, 10)
DAY0 = "2025-01-10"
DAY1 = "2025-01-09"

# 五个新闻源的 pool key（契约）
POOL_KEYS = {"sina", "eastmoney", "mcp", "cls", "tushare"}


def _make_agent() -> MarketReviewAgent:
    """构造 agent，DeepSeek 客户端替换为 mock，防止任何真实 HTTP 调用。"""
    agent = MarketReviewAgent()
    agent.client = MagicMock()
    return agent


def _item(title: str, time: str, source: str) -> dict:
    """构造一条契约格式的新闻条目。"""
    return {"title": title, "time": time, "source": source}


def _empty_pool() -> dict:
    return {k: [] for k in POOL_KEYS}


@contextlib.contextmanager
def _mock_news_pool(return_value=None, side_effect=None):
    """patch fetch_news_pool。

    同时覆盖两种引用方式：orchestrator 内部 ''from agent.data_fetcher import
    fetch_news_pool''（函数内延迟 import，patch data_fetcher 即可生效）与
    模块顶层 import（需额外 patch orchestrator 上的引用，存在才 patch，不用 create=True）。
    """
    mock = MagicMock(side_effect=side_effect) if side_effect is not None \
        else MagicMock(return_value=return_value)
    with patch("agent.data_fetcher.fetch_news_pool", mock):
        if hasattr(orch_mod, "fetch_news_pool"):
            with patch("agent.orchestrator.fetch_news_pool", mock):
                yield mock
        else:
            yield mock


def _run_message(agent, message: str) -> dict:
    """以固定交易日驱动 process_message（非流式），返回 result dict。"""
    with patch(
        "agent.orchestrator._get_latest_trade_date",
        return_value=FIXED_TRADE_DATE,
    ):
        return asyncio.run(agent.process_message(message, stream=False))


# ════════════════════════════════════════════════════════════════
# 1. fetch_news_pool：五源聚合 + 统一字段 + 跨源去重 + 失败降级
# ════════════════════════════════════════════════════════════════


class TestFetchNewsPool:
    """fetch_news_pool(sector_keywords=None, days=3) -> dict 五源聚合。"""

    def _patch_sources(self, sina=None, eastmoney=None, mcp=None, cls=None, tushare=None):
        """返回一组 patcher，替换五个源的 fetch 函数。"""
        return (
            patch.object(data_fetcher, "fetch_sina_news",
                         MagicMock(return_value=sina or [])),
            patch.object(data_fetcher, "fetch_eastmoney_news",
                         MagicMock(return_value=eastmoney or [])),
            patch.object(data_fetcher, "fetch_mcp_news",
                         MagicMock(return_value=mcp or [])),
            patch.object(data_fetcher, "fetch_cls_telegraph",
                         MagicMock(return_value=cls or [])),
            patch.object(data_fetcher, "fetch_tushare_news",
                         MagicMock(return_value=tushare or [])),
        )

    def test_aggregates_five_sources_with_unified_fields(self):
        """聚合结果含五个源 key，每条都有非空 title/time/source。"""
        sina = [_item("沪深两市成交额突破一万五千亿元", f"{DAY0} 09:30:00", "新浪财经")]
        em = [_item("北向资金单日净流入超百亿元", f"{DAY0} 10:00:00", "东方财富")]
        mcp = [_item("央行开展5000亿元MLF操作", f"{DAY0} 11:00:00", "智研")]
        cls = [_item("证监会召开系统工作会议", f"{DAY0} 12:00:00", "财联社电报")]
        ts = [_item("多家公司披露年度业绩预告", f"{DAY0} 13:00:00", "tushare")]

        with contextlib.ExitStack() as stack:
            for p in self._patch_sources(sina, em, mcp, cls, ts):
                stack.enter_context(p)
            pool = data_fetcher.fetch_news_pool()

        assert isinstance(pool, dict), "fetch_news_pool 应返回 dict"
        assert POOL_KEYS <= set(pool.keys()), (
            f"pool 应含五源 key {POOL_KEYS}，实际 {set(pool.keys())}"
        )
        for name in POOL_KEYS:
            assert isinstance(pool[name], list), f"pool[{name}] 应为 list"
            for it in pool[name]:
                assert {"title", "time", "source"} <= set(it.keys()), (
                    f"pool[{name}] 条目缺统一字段: {it}"
                )
                assert it["title"] and it["time"] and it["source"], (
                    f"pool[{name}] 条目字段为空: {it}"
                )
        # 五个源各自的内容都进了池子
        assert any("成交额" in it["title"] for it in pool["sina"])
        assert any("北向资金" in it["title"] for it in pool["eastmoney"])
        assert any("MLF" in it["title"] for it in pool["mcp"])
        assert any("证监会" in it["title"] for it in pool["cls"])
        assert any("业绩预告" in it["title"] for it in pool["tushare"])

    def test_cross_source_title_dedup(self):
        """同一标题出现在多个源时，跨源去重后全池只保留一条。"""
        dup = "央行开展5000亿元中期借贷便利操作"
        sina = [
            _item(dup, f"{DAY0} 09:30:00", "新浪财经"),
            _item("沪深两市成交额突破一万五千亿元", f"{DAY0} 10:00:00", "新浪财经"),
        ]
        em = [
            _item(dup, f"{DAY0} 09:31:00", "东方财富"),  # 跨源重复标题
            _item("北向资金单日净流入超百亿元", f"{DAY0} 10:05:00", "东方财富"),
        ]
        mcp = [_item(dup, f"{DAY0} 09:32:00", "智研")]  # 再次跨源重复

        with contextlib.ExitStack() as stack:
            for p in self._patch_sources(sina, em, mcp, [], []):
                stack.enter_context(p)
            pool = data_fetcher.fetch_news_pool()

        all_titles = [it["title"] for items in pool.values() for it in items]
        assert all_titles.count(dup) == 1, (
            f"跨源重复标题未被去重，出现 {all_titles.count(dup)} 次"
        )

    def test_single_source_failure_does_not_affect_others(self):
        """单源抛异常时失败安全降级：该源为空列表，其他源不受影响。"""
        sina = [_item("沪深两市成交额突破一万五千亿元", f"{DAY0} 09:30:00", "新浪财经")]
        em = [_item("北向资金单日净流入超百亿元", f"{DAY0} 10:00:00", "东方财富")]
        mcp = [_item("央行开展5000亿元MLF操作", f"{DAY0} 11:00:00", "智研")]
        ts = [_item("多家公司披露年度业绩预告", f"{DAY0} 13:00:00", "tushare")]

        with contextlib.ExitStack() as stack:
            for p in self._patch_sources(sina, em, mcp, None, ts):
                stack.enter_context(p)
            # 财联社源直接抛异常
            stack.enter_context(patch.object(
                data_fetcher, "fetch_cls_telegraph",
                MagicMock(side_effect=RuntimeError("财联社接口超时")),
            ))
            pool = data_fetcher.fetch_news_pool()  # 不应抛异常

        assert pool["cls"] == [], "失败源应降级为空列表"
        assert len(pool["sina"]) == 1
        assert len(pool["eastmoney"]) == 1
        assert len(pool["mcp"]) == 1
        assert len(pool["tushare"]) == 1


# ════════════════════════════════════════════════════════════════
# 2. fetch_mcp_news：time 字段修复回归
# ════════════════════════════════════════════════════════════════


class TestMcpNewsTime:
    """mock requests.post 模拟智研 MCP 两步 JSON-RPC（initialize + tools/call），
    返回带时间字段的原始数据，解析出的条目 time 必须非空（回归修复）。"""

    def test_mcp_news_time_not_empty(self):
        import json as _json

        news_payload = {
            "result": {"data": {"data": [
                {"title": "央行开展5000亿元MLF操作", "ctime": f"{DAY0} 09:30:00"},
                # 故意用另一个时间字段名，覆盖多字段兜底解析
                {"title": "证监会就市值管理指引公开征求意见", "pub_date": f"{DAY0} 10:05:00"},
            ]}}
        }

        def _fake_post(url, **kwargs):
            resp = MagicMock()
            if kwargs.get("json", {}).get("method") == "initialize":
                resp.headers = {"Mcp-Session-Id": "fake-session-id"}
            else:
                resp.headers = {}
                resp.json.return_value = {
                    "result": {"content": [{"text": _json.dumps(news_payload)}]}
                }
            return resp

        with patch.object(data_fetcher, "requests") as mock_requests:
            mock_requests.post = MagicMock(side_effect=_fake_post)
            items = data_fetcher.fetch_mcp_news("A股", 30)

        assert items, "fetch_mcp_news 应从 MCP 原始数据中解析出条目"
        for it in items:
            assert it.get("time"), f"解析出的条目 time 为空（回归）: {it}"
            assert it.get("title"), f"解析出的条目 title 为空: {it}"


# ════════════════════════════════════════════════════════════════
# 3. _news_only 透传模式：按天分组 + 总条数 + 来源标注
# ════════════════════════════════════════════════════════════════


class TestNewsOnlyPassthrough:
    """普通新闻查询（不含解读/分析/影响）走透传：结构化罗列新闻池内容。"""

    def test_output_grouped_by_day_with_total_and_sources(self):
        pool = _empty_pool()
        pool["sina"] = [
            _item("沪深两市成交额突破一万五千亿元", f"{DAY0} 09:30:00", "新浪财经"),
            _item("央行开展5000亿元MLF操作", f"{DAY1} 10:00:00", "新浪财经"),
        ]
        pool["eastmoney"] = [
            _item("北向资金单日净流入超百亿元", f"{DAY0} 11:00:00", "东方财富"),
        ]

        agent = _make_agent()
        agent._call_llm = AsyncMock(
            return_value={"role": "assistant", "content": "占位"}
        )

        with _mock_news_pool(pool):
            result = _run_message(agent, "今天有什么新闻")

        content = result["content"]
        # 按天分组：两个日期都应出现在输出中
        assert DAY0 in content, f"输出缺少按天分组日期 {DAY0}:\n{content}"
        assert DAY1 in content, f"输出缺少按天分组日期 {DAY1}:\n{content}"
        # 三条新闻标题全部展示（48小时尽量多展示）
        for title in ["沪深两市成交额突破一万五千亿元",
                      "央行开展5000亿元MLF操作",
                      "北向资金单日净流入超百亿元"]:
            assert title in content, f"输出缺少新闻标题 {title!r}:\n{content}"
        # 头部含总条数（共 3 条）
        assert "共" in content and "3" in content and "条" in content, (
            f"输出头部缺少总条数统计:\n{content[:200]}"
        )
        # 来源标注 / 各源统计（展示名见 orchestrator._NEWS_SOURCE_NAMES）
        assert "新浪" in content, f"输出缺少新浪来源标注:\n{content[:300]}"
        assert "东方财富" in content, f"输出缺少东方财富来源标注:\n{content[:300]}"


# ════════════════════════════════════════════════════════════════
# 4. 行业过滤：『半导体新闻』不含无关行业标题
# ════════════════════════════════════════════════════════════════


class TestNewsOnlySectorFilter:
    """行业新闻查询只输出本行业相关标题。"""

    ELEC = "电子行业半导体设备龙头发布新一代光刻机"
    LIQUOR = "贵州茅台白酒春节销量超市场预期"
    BANK = "央行宣布降准释放长期流动性"

    def _pool(self):
        pool = _empty_pool()
        pool["sina"] = [
            _item(self.ELEC, f"{DAY0} 09:30:00", "新浪财经"),
            _item(self.LIQUOR, f"{DAY0} 10:00:00", "新浪财经"),
            _item(self.BANK, f"{DAY0} 11:00:00", "新浪财经"),
        ]
        return pool

    def _fake_pool(self, sector_keywords=None, days=3):
        """若 orchestrator 把行业关键词传给 pool，则 mock 侧也按关键词过滤；
        若 orchestrator 自己过滤（不传关键词），则返回全量由其过滤。"""
        pool = self._pool()
        if sector_keywords:
            return {
                k: [it for it in v
                    if any(kw in it["title"] for kw in sector_keywords)]
                for k, v in pool.items()
            }
        return pool

    def test_unrelated_sector_titles_excluded(self):
        agent = _make_agent()
        agent._call_llm = AsyncMock(
            return_value={"role": "assistant", "content": "占位"}
        )

        with _mock_news_pool(side_effect=self._fake_pool):
            result = _run_message(agent, "半导体新闻")

        content = result["content"]
        assert self.ELEC in content, (
            f"电子（半导体）相关标题应保留:\n{content}"
        )
        assert self.LIQUOR not in content, (
            f"白酒（食品饮料）标题不应出现在半导体新闻中:\n{content}"
        )
        assert self.BANK not in content, (
            f"银行（降准）标题不应出现在半导体新闻中:\n{content}"
        )


# ════════════════════════════════════════════════════════════════
# 5. 重要性截断：全市场每天上限 30 条，高分新闻优先保留
# ════════════════════════════════════════════════════════════════


class TestNewsImportanceTruncation:
    """单日 40 条全市场新闻 → 输出 ≤30 条；业绩/政策类高分新闻即使排在池子末尾也保留。"""

    IMPORTANT = [
        "A上市公司业绩预告净利润同比预增300%",      # 业绩词（预增/净利）+3
        "证监会发布重磅政策稳定资本市场预期",        # 政策词（政策/证监会）+3
        "B上市公司股价异动午后直线涨停",            # 异动词（涨停）+2
        "C上市公司公告拟回购股份不超过10亿元",       # 公司行动词（回购）+2
    ]

    def test_truncation_keeps_high_score_news(self):
        # 36 条不含任何重要性关键词的普通快讯打底，高分新闻故意放最后
        fillers = [f"市场日常资金动态快讯第{i:02d}期" for i in range(1, 37)]
        titles = fillers + self.IMPORTANT
        assert len(titles) == 40

        pool = _empty_pool()
        pool["sina"] = [
            _item(t, f"{DAY0} {9 + (i % 8):02d}:{i % 60:02d}:00", "新浪财经")
            for i, t in enumerate(titles)
        ]

        agent = _make_agent()
        agent._call_llm = AsyncMock(
            return_value={"role": "assistant", "content": "占位"}
        )

        with _mock_news_pool(pool):
            result = _run_message(agent, "今天有什么新闻")

        content = result["content"]
        shown = [t for t in titles if t in content]
        assert len(shown) <= 30, (
            f"全市场单日应截断到 30 条以内，实际展示 {len(shown)} 条"
        )
        for t in self.IMPORTANT:
            assert t in shown, (
                f"高分新闻 {t!r} 应优先保留却未出现在输出中:\n{content[:500]}"
            )


# ════════════════════════════════════════════════════════════════
# 6. 分析模式：消息含『分析/影响』→ 走 LLM 分析而非透传
# ════════════════════════════════════════════════════════════════


class TestNewsAnalysisMode:
    """『分析一下今天的新闻有什么影响』应触发 NEWS_ANALYSIS_PROMPT 的 LLM 分析。"""

    NEWS_TITLE = "人形机器人产业链订单爆发式增长"

    def test_analysis_message_calls_llm_with_news_prompt(self):
        pool = _empty_pool()
        pool["sina"] = [
            _item(self.NEWS_TITLE, f"{DAY0} 09:30:00", "新浪财经"),
            _item("沪深两市成交额突破一万五千亿元", f"{DAY0} 10:00:00", "新浪财经"),
        ]

        agent = _make_agent()
        agent._call_llm = AsyncMock(
            return_value={"role": "assistant", "content": "新闻解读分析结果"}
        )

        with _mock_news_pool(pool):
            result = _run_message(agent, "分析一下今天的新闻有什么影响")

        # _call_llm 被调用 → 走了 LLM 分析，而非直接透传
        assert agent._call_llm.await_count >= 1, (
            "分析类新闻消息应调用 _call_llm 走 LLM 分析模式"
        )
        args, kwargs = agent._call_llm.await_args
        system_arg = args[0] if len(args) >= 1 else kwargs.get("system", "")
        user_arg = args[1] if len(args) >= 2 else kwargs.get("user_prompt", "")
        # system prompt 是新闻分析 prompt（主题归纳结构 + 数据红线）
        assert "主题" in system_arg or "主线" in system_arg, (
            f"分析模式 system prompt 应含主题归纳结构，实际:\n{system_arg[:300]}"
        )
        # user prompt 中带有新闻条目（LLM 基于新闻池内容做解读）
        assert self.NEWS_TITLE in user_arg, (
            f"分析模式 user prompt 应包含新闻条目，实际:\n{user_arg[:300]}"
        )
        # 最终输出是 LLM 的分析结果，而不是透传的新闻清单
        assert "新闻解读分析结果" in result["content"]


# ════════════════════════════════════════════════════════════════
# 7. NEWS_ANALYSIS_PROMPT 结构断言
# ════════════════════════════════════════════════════════════════


class TestNewsAnalysisPrompt:
    """NEWS_ANALYSIS_PROMPT：主题归纳/方向判断结构 + 数据红线 + ≥3 个禁用词。"""

    def test_prompt_structure(self):
        from agent.system_prompts import NEWS_ANALYSIS_PROMPT

        # 主题归纳 / 方向判断结构
        assert "主题" in NEWS_ANALYSIS_PROMPT, "prompt 缺少主题归纳结构"
        assert "方向" in NEWS_ANALYSIS_PROMPT or "判断" in NEWS_ANALYSIS_PROMPT, (
            "prompt 缺少方向判断结构"
        )
        # 数据红线表述
        assert "红线" in NEWS_ANALYSIS_PROMPT, "prompt 缺少数据红线表述"
        # 去 AI 味禁用词清单：至少 3 个
        banned = [
            "护城河", "飞轮", "赋能", "格局", "至关重要", "值得注意的是",
            "综上所述", "深度", "全方位", "拥抱", "长期主义", "黄金坑", "戴维斯双击",
        ]
        hits = [w for w in banned if w in NEWS_ANALYSIS_PROMPT]
        assert len(hits) >= 3, (
            f"prompt 禁用词清单应至少含 3 个禁用词，实际命中 {hits}"
        )
