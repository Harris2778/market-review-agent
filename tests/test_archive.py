"""agent/archive.py 存档层 + orchestrator 存档接入（第四波·自我问责）测试。

覆盖范围：
1. 存档往返：save_analysis → load_records 读回，schema 字段逐项核对
   （id/ts/trade_date/mode/sector/content/context_excerpt/numbers/score 三 null）。
2. load_records：date_str=None 跨天加载；非法日期 / 目录不存在 → []。
3. update_record：读改写命中更新；未命中 / 非法入参 → False；id 主键不可覆盖。
4. 并发写不串行：多线程并发追加，读回行数完整、每行均为合法 JSON、id 唯一。
5. fail-safe：存档目录不可写 / save_analysis 内部异常 → 返回 None 不抛出；
   orchestrator 侧 archive 抛异常或 archive=None 时主流程产出不受影响。
6. orchestrator 集成：_market_review / _sector_deep_dive / _agent_query 三条
   非流式路径产出后确实落档（mock archive 断言 mode/sector/trade_date/content/context），
   以及 _market_review 流式路径在流结束后经回调落档。

规则（与项目其他测试一致）：
- 所有外部依赖全部 mock（collect_market_snapshot / DeepSeek 客户端 / archive 模块），
  绝不发起真实网络请求。
- ARCHIVE_DIR 一律 monkeypatch 到 tmp_path，绝不写真实 data/archive/。
- _get_latest_trade_date 统一 patch 为固定日期，避免触发真实 tushare 交易日历请求。
- 无 pytest-asyncio，异步函数一律用 asyncio.run 驱动。
"""

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agent.archive as archive
import agent.orchestrator as orchestrator
from agent.orchestrator import MarketReviewAgent

# 固定交易日，避免 _get_latest_trade_date 触达真实 tushare
FIXED_TRADE_DATE = datetime(2025, 1, 10)  # 周五
FIXED_DATE_STR = "20250110"

RECORD_SCHEMA_FIELDS = {
    "id", "ts", "trade_date", "mode", "sector", "content",
    "context_excerpt", "numbers", "score", "scored_at", "score_note",
}


@pytest.fixture()
def archive_dir(tmp_path, monkeypatch):
    """把存档目录隔离到 tmp_path，绝不触碰真实 data/archive/。"""
    d = tmp_path / "archive"
    monkeypatch.setenv(archive.ARCHIVE_DIR_ENV, str(d))
    return d


def _make_agent() -> MarketReviewAgent:
    """构造一个 agent，DeepSeek 客户端替换为 mock，防止任何真实 HTTP 调用。"""
    agent = MarketReviewAgent()
    agent.client = MagicMock()
    return agent


# ════════════════════════════════════════════════════════════════
# 1. 存档往返：schema 契约逐项核对
# ════════════════════════════════════════════════════════════════


class TestSaveLoadRoundTrip:
    def test_round_trip_schema(self, archive_dir):
        content = "沪指涨2.35%，全市场成交1.2万亿，北向净流出15.6亿元。"
        context = "【数据上下文】\n" + "上证指数 3251.85点 涨2.35%\n" * 3
        rid = archive.save_analysis(
            mode="market_review", sector=None, content=content,
            context=context, trade_date=FIXED_DATE_STR,
        )
        assert isinstance(rid, str) and len(rid) == 32, "save_analysis 应返回 uuid4hex"

        records = archive.load_records(FIXED_DATE_STR)
        assert len(records) == 1
        rec = records[0]
        assert set(rec.keys()) == RECORD_SCHEMA_FIELDS, (
            f"存档字段集合与契约不一致: {sorted(rec.keys())}"
        )
        assert rec["id"] == rid
        assert rec["trade_date"] == FIXED_DATE_STR
        assert rec["mode"] == "market_review"
        assert rec["sector"] is None
        assert rec["content"] == content
        assert rec["context_excerpt"] == context
        # score 三字段初始一律为 null，留给打分层写回
        assert rec["score"] is None
        assert rec["scored_at"] is None
        assert rec["score_note"] is None
        # ts 为可读 ISO 时间戳
        datetime.fromisoformat(rec["ts"])

    def test_numbers_extracted_slim_and_serializable(self, archive_dir):
        rid = archive.save_analysis(
            mode="market_review", sector=None,
            content="沪指涨2.35%，成交1.2万亿",
            context="ctx", trade_date=FIXED_DATE_STR,
        )
        assert rid is not None
        rec = archive.load_records(FIXED_DATE_STR)[0]
        assert isinstance(rec["numbers"], list) and len(rec["numbers"]) > 0, (
            "含数字的正文应抽取出 numbers 清单"
        )
        for item in rec["numbers"]:
            assert set(item.keys()) == {"value", "normalized", "raw", "unit"}, (
                f"numbers 精简 dict 字段不符: {sorted(item.keys())}"
            )
        # 整体必须可 JSON 序列化（JSONL 落盘前提）
        json.dumps(rec, ensure_ascii=False)

    def test_sector_and_mode_preserved(self, archive_dir):
        rid = archive.save_analysis(
            mode="sector_deep_dive", sector="煤炭",
            content="煤炭板块分析", context="ctx", trade_date=FIXED_DATE_STR,
        )
        assert rid is not None
        rec = archive.load_records(FIXED_DATE_STR)[0]
        assert rec["mode"] == "sector_deep_dive"
        assert rec["sector"] == "煤炭"

    def test_context_excerpt_truncated_to_4000(self, archive_dir):
        long_context = "数" * 5000
        rid = archive.save_analysis(
            mode="agent_query", sector=None, content="正文",
            context=long_context, trade_date=FIXED_DATE_STR,
        )
        assert rid is not None
        rec = archive.load_records(FIXED_DATE_STR)[0]
        assert len(rec["context_excerpt"]) == 4000, "context_excerpt 应截断为前 4000 字符"

    def test_trade_date_fallback_to_today(self, archive_dir):
        """trade_date 非法/缺失时回退当前日期文件，不抛出。"""
        rid = archive.save_analysis(
            mode="market_review", sector=None, content="正文",
            context="ctx", trade_date="not-a-date",
        )
        assert rid is not None
        today_str = datetime.now().strftime("%Y%m%d")
        rec = archive.load_records(today_str)[0]
        assert rec["trade_date"] == today_str

    def test_file_layout_one_file_per_day(self, archive_dir):
        archive.save_analysis("market_review", None, "A", "c", "20250109")
        archive.save_analysis("market_review", None, "B", "c", FIXED_DATE_STR)
        archive.save_analysis("market_review", None, "C", "c", FIXED_DATE_STR)
        files = sorted(os.listdir(archive_dir))
        assert files == ["archive_20250109.jsonl", "archive_20250110.jsonl"]
        assert len(archive.load_records("20250109")) == 1
        assert len(archive.load_records(FIXED_DATE_STR)) == 2

    def test_load_all_across_days(self, archive_dir):
        archive.save_analysis("market_review", None, "A", "c", "20250109")
        archive.save_analysis("sector_deep_dive", "煤炭", "B", "c", FIXED_DATE_STR)
        all_records = archive.load_records()
        assert len(all_records) == 2
        assert {r["content"] for r in all_records} == {"A", "B"}


# ════════════════════════════════════════════════════════════════
# 2. load_records 容错
# ════════════════════════════════════════════════════════════════


class TestLoadRecordsFailSafe:
    def test_missing_dir_returns_empty(self, archive_dir):
        assert archive.load_records() == []
        assert archive.load_records(FIXED_DATE_STR) == []

    def test_invalid_date_returns_empty(self, archive_dir):
        assert archive.load_records("2025-01-10") == []
        assert archive.load_records("../../etc") == []

    def test_bad_line_skipped(self, archive_dir):
        archive_dir.mkdir(parents=True)
        path = archive_dir / f"archive_{FIXED_DATE_STR}.jsonl"
        path.write_text(
            '{"id": "ok1", "trade_date": "20250110"}\n'
            "这不是JSON\n"
            '{"id": "ok2", "trade_date": "20250110"}\n',
            encoding="utf-8",
        )
        records = archive.load_records(FIXED_DATE_STR)
        assert [r["id"] for r in records] == ["ok1", "ok2"], "坏行应跳过、好行保留"


# ════════════════════════════════════════════════════════════════
# 3. update_record 读改写
# ════════════════════════════════════════════════════════════════


class TestUpdateRecord:
    def test_update_score_fields_round_trip(self, archive_dir):
        rid = archive.save_analysis(
            "market_review", None, "看多正文", "ctx", FIXED_DATE_STR
        )
        assert rid is not None
        ok = archive.update_record(rid, {
            "score": "hit",
            "scored_at": "2025-01-17T09:00:00",
            "score_note": "方向判断=bullish；实际区间涨跌幅=+2.10%",
        })
        assert ok is True
        rec = archive.load_records(FIXED_DATE_STR)[0]
        assert rec["score"] == "hit"
        assert rec["scored_at"] == "2025-01-17T09:00:00"
        assert rec["score_note"].startswith("方向判断=bullish")
        # 未更新字段保持原值
        assert rec["content"] == "看多正文"
        assert rec["id"] == rid

    def test_update_preserves_other_records_and_bad_lines(self, archive_dir):
        rid1 = archive.save_analysis("market_review", None, "A", "c", FIXED_DATE_STR)
        rid2 = archive.save_analysis("market_review", None, "B", "c", FIXED_DATE_STR)
        # 手工塞入一行坏行，验证回写不丢数据
        path = archive_dir / f"archive_{FIXED_DATE_STR}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write("坏行不是JSON\n")
        assert archive.update_record(rid2, {"score": "miss"}) is True
        raw = path.read_text(encoding="utf-8")
        assert "坏行不是JSON" in raw, "回写后坏行应原样保留"
        records = archive.load_records(FIXED_DATE_STR)
        by_id = {r["id"]: r for r in records}
        assert by_id[rid1]["score"] is None, "未命中的记录不应被改动"
        assert by_id[rid2]["score"] == "miss"

    def test_update_not_found_returns_false(self, archive_dir):
        archive.save_analysis("market_review", None, "A", "c", FIXED_DATE_STR)
        assert archive.update_record("0" * 32, {"score": "hit"}) is False

    def test_update_invalid_args_returns_false(self, archive_dir):
        rid = archive.save_analysis("market_review", None, "A", "c", FIXED_DATE_STR)
        assert archive.update_record(None, {"score": "hit"}) is False
        assert archive.update_record(rid, None) is False
        assert archive.update_record(rid, {}) is False

    def test_update_missing_dir_returns_false(self, archive_dir):
        assert archive.update_record("0" * 32, {"score": "hit"}) is False

    def test_id_primary_key_immutable(self, archive_dir):
        rid = archive.save_analysis("market_review", None, "A", "c", FIXED_DATE_STR)
        assert archive.update_record(rid, {"id": "f" * 32, "score": "hit"}) is True
        rec = archive.load_records(FIXED_DATE_STR)[0]
        assert rec["id"] == rid, "id 主键不应被 fields 覆盖"
        assert rec["score"] == "hit"

    def test_update_record_in_other_day_file(self, archive_dir):
        """记录不在当天文件时，update_record 应跨文件扫描命中。"""
        archive.save_analysis("market_review", None, "今天", "c", FIXED_DATE_STR)
        rid_old = archive.save_analysis("market_review", None, "昨天", "c", "20250109")
        assert archive.update_record(rid_old, {"score": "neutral"}) is True
        assert archive.load_records("20250109")[0]["score"] == "neutral"
        assert archive.load_records(FIXED_DATE_STR)[0]["score"] is None


# ════════════════════════════════════════════════════════════════
# 4. 并发写不串行
# ════════════════════════════════════════════════════════════════


class TestConcurrentWrites:
    def test_concurrent_appends_no_interleaving(self, archive_dir):
        """多线程并发追加：行数完整、每行合法 JSON、id 唯一。"""
        n_threads, n_per_thread = 8, 10

        def _worker(tid):
            ids = []
            for i in range(n_per_thread):
                rid = archive.save_analysis(
                    "agent_query", None,
                    f"线程{tid}第{i}条，涨1.5%", "ctx", FIXED_DATE_STR,
                )
                ids.append(rid)
            return ids

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            results = list(pool.map(_worker, range(n_threads)))

        all_ids = [rid for ids in results for rid in ids]
        assert all(rid is not None for rid in all_ids), "并发下 save_analysis 不应失败"
        assert len(set(all_ids)) == n_threads * n_per_thread, "id 应全部唯一"

        # 逐行校验：文件里每一行都是完整合法的 JSON（无串行/交错）
        path = archive_dir / f"archive_{FIXED_DATE_STR}.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == n_threads * n_per_thread
        for line in lines:
            obj = json.loads(line)
            assert obj["id"] in set(all_ids)

        records = archive.load_records(FIXED_DATE_STR)
        assert len(records) == n_threads * n_per_thread

    def test_concurrent_update_and_append(self, archive_dir):
        """追加与读改写并发：update 命中同一文件时不丢行、不错乱。"""
        base_ids = [
            archive.save_analysis("market_review", None, f"基线{i}", "c", FIXED_DATE_STR)
            for i in range(5)
        ]

        def _append(i):
            return archive.save_analysis("agent_query", None, f"追加{i}", "c", FIXED_DATE_STR)

        def _update(rid):
            return archive.update_record(rid, {"score": "hit"})

        with ThreadPoolExecutor(max_workers=8) as pool:
            f1 = pool.map(_append, range(10))
            f2 = pool.map(_update, base_ids * 2)  # 重复更新同一批 id
        assert all(r is not None for r in f1)
        assert any(ok for ok in f2)

        records = archive.load_records(FIXED_DATE_STR)
        assert len(records) == 15, "并发追加+更新后总行数应完整"
        scored = [r for r in records if r.get("score") == "hit"]
        assert len(scored) == 5, "5 条基线记录应被打分，追加记录不受影响"


# ════════════════════════════════════════════════════════════════
# 5. fail-safe：任何异常只记 log，绝不炸主流程
# ════════════════════════════════════════════════════════════════


class TestFailSafe:
    def test_save_returns_none_on_write_failure(self, archive_dir, monkeypatch):
        monkeypatch.setattr(archive.os, "makedirs", MagicMock(side_effect=OSError("只读文件系统")))
        assert archive.save_analysis("market_review", None, "正文", "ctx", FIXED_DATE_STR) is None

    def test_save_tolerates_non_string_inputs(self, archive_dir):
        rid = archive.save_analysis("market_review", None, None, None, FIXED_DATE_STR)
        assert rid is not None
        rec = archive.load_records(FIXED_DATE_STR)[0]
        assert rec["content"] == ""
        assert rec["context_excerpt"] == ""
        assert rec["numbers"] == []

    def test_update_returns_false_on_io_failure(self, archive_dir, monkeypatch):
        rid = archive.save_analysis("market_review", None, "A", "c", FIXED_DATE_STR)
        monkeypatch.setattr(archive, "_write_jsonl", MagicMock(side_effect=OSError("磁盘满")))
        assert archive.update_record(rid, {"score": "hit"}) is False

    def test_orchestrator_survives_archive_exception(self):
        """archive.save_analysis 抛异常时，_market_review 主流程产出不受影响。"""
        agent = _make_agent()
        agent._call_llm = AsyncMock(
            return_value={"role": "assistant", "content": "复盘正文"}
        )
        broken_archive = MagicMock()
        broken_archive.save_analysis.side_effect = RuntimeError("存档层爆炸")

        with patch(
            "agent.orchestrator._get_latest_trade_date", return_value=FIXED_TRADE_DATE
        ), patch(
            "agent.orchestrator.collect_market_snapshot",
            AsyncMock(return_value=SimpleNamespace(tag="snapshot")),
        ), patch(
            "agent.orchestrator.format_market_data_for_prompt", return_value="DATA"
        ), patch.object(orchestrator, "archive", broken_archive):
            result = asyncio.run(agent._market_review(stream=False))

        assert result["content"] == "复盘正文", "存档异常不应影响主流程产出"
        assert broken_archive.save_analysis.call_count == 1

    def test_orchestrator_survives_archive_none(self):
        """archive 模块未就绪（置 None）时，_market_review 正常工作。"""
        agent = _make_agent()
        agent._call_llm = AsyncMock(
            return_value={"role": "assistant", "content": "复盘正文"}
        )
        with patch(
            "agent.orchestrator._get_latest_trade_date", return_value=FIXED_TRADE_DATE
        ), patch(
            "agent.orchestrator.collect_market_snapshot",
            AsyncMock(return_value=SimpleNamespace(tag="snapshot")),
        ), patch(
            "agent.orchestrator.format_market_data_for_prompt", return_value="DATA"
        ), patch.object(orchestrator, "archive", None):
            result = asyncio.run(agent._market_review(stream=False))

        assert result["content"] == "复盘正文"


# ════════════════════════════════════════════════════════════════
# 6. orchestrator 集成：三条非流式路径产出后落档
# ════════════════════════════════════════════════════════════════


def _sector_extras_patches(stack):
    """板块深挖附加数据 mock（对齐 test_agent_loop 的接线方式）。"""
    from contextlib import ExitStack
    assert isinstance(stack, ExitStack)
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


class TestOrchestratorArchiveIntegration:
    def test_market_review_archived(self):
        agent = _make_agent()
        agent._call_llm = AsyncMock(
            return_value={"role": "assistant", "content": "复盘正文，沪指涨1.2%"}
        )
        mock_archive = MagicMock()
        mock_archive.save_analysis.return_value = "a" * 32

        with patch(
            "agent.orchestrator._get_latest_trade_date", return_value=FIXED_TRADE_DATE
        ), patch(
            "agent.orchestrator.collect_market_snapshot",
            AsyncMock(return_value=SimpleNamespace(tag="snapshot")),
        ), patch(
            "agent.orchestrator.format_market_data_for_prompt", return_value="DATA"
        ), patch.object(orchestrator, "archive", mock_archive):
            result = asyncio.run(agent._market_review(stream=False))

        assert result["content"].startswith("复盘正文")
        assert mock_archive.save_analysis.call_count == 1, "复盘产出后应落档一次"
        kwargs = mock_archive.save_analysis.call_args.kwargs
        assert kwargs["mode"] == "market_review"
        assert kwargs["sector"] is None
        assert kwargs["trade_date"] == FIXED_DATE_STR
        assert kwargs["content"].startswith("复盘正文"), "落档内容应为最终产出全文"
        assert "DATA" in kwargs["context"], "落档上下文应含数据上下文"

    def test_sector_deep_dive_archived(self):
        from contextlib import ExitStack
        agent = _make_agent()
        # 短草稿（<500 字）→ 多 pass 审查跳过，最终产出即草稿清理版
        agent._call_llm = AsyncMock(
            return_value={"role": "assistant", "content": "煤炭板块终稿"}
        )
        mock_archive = MagicMock()
        mock_archive.save_analysis.return_value = "b" * 32

        with ExitStack() as stack:
            stack.enter_context(patch(
                "agent.orchestrator._get_latest_trade_date",
                return_value=FIXED_TRADE_DATE,
            ))
            stack.enter_context(patch(
                "agent.orchestrator.collect_market_snapshot",
                AsyncMock(return_value=SimpleNamespace(tag="snapshot")),
            ))
            stack.enter_context(patch(
                "agent.orchestrator.format_market_data_for_prompt", return_value="DATA"
            ))
            _sector_extras_patches(stack)
            stack.enter_context(patch.object(orchestrator, "archive", mock_archive))
            result = asyncio.run(agent._sector_deep_dive("煤炭", stream=False))

        assert "煤炭板块终稿" in result["content"]
        assert mock_archive.save_analysis.call_count == 1, "板块深挖产出后应落档一次"
        kwargs = mock_archive.save_analysis.call_args.kwargs
        assert kwargs["mode"] == "sector_deep_dive"
        assert kwargs["sector"] == "煤炭"
        assert kwargs["trade_date"] == FIXED_DATE_STR
        assert "煤炭板块终稿" in kwargs["content"]
        assert "DATA" in kwargs["context"]

    def test_agent_query_archived(self):
        agent = _make_agent()
        # 首轮即纯文本收尾（无 tool_calls），draft <500 字跳过审查
        completion = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="白酒更强，茅台更稳", tool_calls=None),
                finish_reason="stop",
            )]
        )
        agent.client.chat.completions.create = AsyncMock(return_value=completion)
        mock_archive = MagicMock()
        mock_archive.save_analysis.return_value = "c" * 32

        with patch.object(orchestrator, "archive", mock_archive):
            result = asyncio.run(
                agent._agent_query("比较白酒和半导体哪个更值得关注", stream=False, history=None)
            )

        assert "白酒更强" in result["content"]
        assert mock_archive.save_analysis.call_count == 1, "Agent 查询产出后应落档一次"
        kwargs = mock_archive.save_analysis.call_args.kwargs
        assert kwargs["mode"] == "agent_query"
        assert kwargs["sector"] is None
        assert kwargs["trade_date"] == datetime.now().strftime("%Y%m%d"), (
            "agent_query 的 trade_date 应为当前日期"
        )
        assert "白酒更强" in kwargs["content"], "落档内容应为清理+免责声明后的最终产出"
        assert "比较白酒和半导体" in kwargs["context"], "落档上下文应为 user_message+工具上下文拼接"

    def test_market_review_stream_archived_on_finish(self):
        """流式路径：生成器完整消费后，经 archive 回调以最终全文落档。"""
        agent = _make_agent()

        async def fake_chunk_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="复盘"), finish_reason=None,
                )]
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="全文"), finish_reason="stop",
                )]
            )

        agent.client.chat.completions.create = AsyncMock(
            return_value=fake_chunk_stream()
        )
        mock_archive = MagicMock()
        mock_archive.save_analysis.return_value = "d" * 32

        async def run_stream():
            with patch(
                "agent.orchestrator._get_latest_trade_date", return_value=FIXED_TRADE_DATE
            ), patch(
                "agent.orchestrator.collect_market_snapshot",
                AsyncMock(return_value=SimpleNamespace(tag="snapshot")),
            ), patch(
                "agent.orchestrator.format_market_data_for_prompt", return_value="DATA"
            ), patch.object(orchestrator, "archive", mock_archive):
                gen = await agent._market_review(stream=True)
                chunks = []
                async for piece in gen:
                    chunks.append(piece)
                return chunks

        chunks = asyncio.run(run_stream())
        assert "".join(chunks) == "复盘全文"
        assert mock_archive.save_analysis.call_count == 1, "流结束后应落档一次"
        kwargs = mock_archive.save_analysis.call_args.kwargs
        assert kwargs["mode"] == "market_review"
        assert kwargs["trade_date"] == FIXED_DATE_STR
        assert kwargs["content"] == "复盘全文", "流式落档内容应为流累积全文"

    def test_stream_survives_archive_callback_exception(self):
        """流式存档回调抛异常：chunk 完整送达、不向上抛。"""
        agent = _make_agent()

        async def fake_chunk_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="正文"), finish_reason="stop",
                )]
            )

        agent.client.chat.completions.create = AsyncMock(
            return_value=fake_chunk_stream()
        )
        broken_archive = MagicMock()
        broken_archive.save_analysis.side_effect = RuntimeError("存档层爆炸")

        async def run_stream():
            with patch(
                "agent.orchestrator._get_latest_trade_date", return_value=FIXED_TRADE_DATE
            ), patch(
                "agent.orchestrator.collect_market_snapshot",
                AsyncMock(return_value=SimpleNamespace(tag="snapshot")),
            ), patch(
                "agent.orchestrator.format_market_data_for_prompt", return_value="DATA"
            ), patch.object(orchestrator, "archive", broken_archive):
                gen = await agent._market_review(stream=True)
                return [piece async for piece in gen]

        chunks = asyncio.run(run_stream())
        assert "".join(chunks) == "正文", "存档异常不应影响流式产出"
        assert broken_archive.save_analysis.call_count == 1



# ════════════════════════════════════════════════════════════════
# 7. DATA_DIR 统一约定（存储持久化波次）
# ════════════════════════════════════════════════════════════════


class TestDataDirConvention:
    """ARCHIVE_DIR 显式优先；缺省推导为 ${DATA_DIR:-data}/archive。"""

    def test_default_derives_from_data_dir(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ARCHIVE_DIR", raising=False)
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "mydata"))
        assert archive._archive_dir() == str(tmp_path / "mydata" / "archive")

    def test_default_without_data_dir_is_data_archive(self, monkeypatch):
        monkeypatch.delenv("ARCHIVE_DIR", raising=False)
        monkeypatch.delenv("DATA_DIR", raising=False)
        assert archive._archive_dir() == os.path.join("data", "archive")

    def test_empty_data_dir_falls_back(self, monkeypatch):
        monkeypatch.delenv("ARCHIVE_DIR", raising=False)
        monkeypatch.setenv("DATA_DIR", "")
        assert archive._archive_dir() == os.path.join("data", "archive")

    def test_explicit_archive_dir_wins_over_data_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "mydata"))
        monkeypatch.setenv("ARCHIVE_DIR", str(tmp_path / "explicit"))
        assert archive._archive_dir() == str(tmp_path / "explicit")

    def test_save_analysis_uses_data_dir_derived_path(self, tmp_path, monkeypatch):
        """不设 ARCHIVE_DIR 时，存档实际落到 ${DATA_DIR}/archive 下。"""
        monkeypatch.delenv("ARCHIVE_DIR", raising=False)
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "mydata"))
        rid = archive.save_analysis(
            "market_review", None, "正文涨1.2%", "ctx", FIXED_DATE_STR
        )
        assert rid is not None
        day_file = tmp_path / "mydata" / "archive" / f"archive_{FIXED_DATE_STR}.jsonl"
        assert day_file.exists(), "存档应落到 DATA_DIR 推导目录"
        rec = archive.load_records(FIXED_DATE_STR)[0]
        assert rec["id"] == rid


# ════════════════════════════════════════════════════════════════
# 8. Railway 临时存储警告（每模块仅警告一次）
# ════════════════════════════════════════════════════════════════

_WARN_KEYWORD = "运行在 Railway 但未挂卷"


class TestEphemeralStorageWarning:
    def test_warns_on_railway_without_volume(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(archive, "_EPHEMERAL_WARNED", False)
        monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
        monkeypatch.setenv("ARCHIVE_DIR", str(tmp_path / "archive"))  # 非 /data 开头
        with caplog.at_level(logging.WARNING, logger="agent.archive"):
            archive._archive_dir()
        assert any(_WARN_KEYWORD in r.getMessage() for r in caplog.records)

    def test_no_warning_when_dir_under_volume(self, monkeypatch, caplog):
        monkeypatch.setattr(archive, "_EPHEMERAL_WARNED", False)
        monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
        monkeypatch.setenv("ARCHIVE_DIR", "/data/archive")  # 挂载卷路径下
        with caplog.at_level(logging.WARNING, logger="agent.archive"):
            archive._archive_dir()
        assert not any(_WARN_KEYWORD in r.getMessage() for r in caplog.records)

    def test_no_warning_off_railway(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(archive, "_EPHEMERAL_WARNED", False)
        monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
        monkeypatch.setenv("ARCHIVE_DIR", str(tmp_path / "archive"))
        with caplog.at_level(logging.WARNING, logger="agent.archive"):
            archive._archive_dir()
        assert not any(_WARN_KEYWORD in r.getMessage() for r in caplog.records)

    def test_warns_only_once_per_module(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(archive, "_EPHEMERAL_WARNED", False)
        monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
        monkeypatch.setenv("ARCHIVE_DIR", str(tmp_path / "archive"))
        with caplog.at_level(logging.WARNING, logger="agent.archive"):
            archive._archive_dir()
            archive._archive_dir()
            archive._archive_dir()
        warns = [r for r in caplog.records if _WARN_KEYWORD in r.getMessage()]
        assert len(warns) == 1, "同一模块重复解析目录只应警告一次"

    def test_data_dir_derived_ephemeral_path_warns(self, tmp_path, monkeypatch, caplog):
        """DATA_DIR 推导出的临时目录同样触发警告（未挂卷场景）。"""
        monkeypatch.setattr(archive, "_EPHEMERAL_WARNED", False)
        monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
        monkeypatch.delenv("ARCHIVE_DIR", raising=False)
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "mydata"))
        with caplog.at_level(logging.WARNING, logger="agent.archive"):
            archive._archive_dir()
        assert any(_WARN_KEYWORD in r.getMessage() for r in caplog.records)

    def test_data_dir_on_volume_no_warning(self, monkeypatch, caplog):
        """DATA_DIR=/data 时推导目录在挂载卷下，不警告（生产正确配置）。"""
        monkeypatch.setattr(archive, "_EPHEMERAL_WARNED", False)
        monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
        monkeypatch.delenv("ARCHIVE_DIR", raising=False)
        monkeypatch.setenv("DATA_DIR", "/data")
        with caplog.at_level(logging.WARNING, logger="agent.archive"):
            assert archive._archive_dir() == "/data/archive"
        assert not any(_WARN_KEYWORD in r.getMessage() for r in caplog.records)
