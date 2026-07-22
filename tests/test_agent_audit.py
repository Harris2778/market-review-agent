"""agent/agent_audit.py 审计工程三件套（第五波）测试。

覆盖范围：
1. Scratchpad：四类日志写入与 JSONL 格式、文件名/目录惰性解析三级回退、
   path 属性（目录未初始化也可返回）、坏目录/不可写目录降级（不抛异常 +
   warning）、不可序列化对象兜底、超长截断、时间注入、并发写不串行、
   default_scratchpad 惰性单例。
2. ToolCallGuard：计数警告（边界：恰好上限不警告、超上限警告含已调用次数
   与三条出路）、相似度警告（≥阈值触发、<阈值不触发、键序无关、跨工具不
   误判）、check 不记账、reset 清空、不可序列化 args 不崩。
3. microcompact：双触发条件（tool 消息数 / 总字符数）、keep_recent 保护、
   不触发时原样返回、入参不可变、tool_calls 与 tool_call_id 配对完整性、
   system/user/assistant 消息不动、统计字段正确。

规则（与项目其他测试一致）：
- 全 mock 零网络；SCRATCHPAD_DIR / DATA_DIR 一律 monkeypatch 到 tmp_path，
  绝不写真实 data/scratchpad/。
- 时间经 monkeypatch agent_audit._now_iso / _now_stamp 注入固定值。
"""

import difflib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest

import agent.agent_audit as agent_audit
from agent.agent_audit import (
    CLEARED_PLACEHOLDER,
    MAX_RESULT_CHARS,
    MAX_TEXT_CHARS,
    Scratchpad,
    ToolCallGuard,
    default_scratchpad,
    microcompact,
)

FIXED_ISO = "2025-01-10T15:30:00"
FIXED_STAMP = "20250110_153000"


@pytest.fixture(autouse=True)
def clean_scratchpad_env(monkeypatch):
    """清理 SCRATCHPAD_DIR / DATA_DIR，保证目录解析测试互不污染。"""
    monkeypatch.delenv("SCRATCHPAD_DIR", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)


@pytest.fixture()
def fixed_time(monkeypatch):
    """时间注入：固定 ISO 时间戳与文件名时间戳。"""
    monkeypatch.setattr(agent_audit, "_now_iso", lambda: FIXED_ISO)
    monkeypatch.setattr(agent_audit, "_now_stamp", lambda: FIXED_STAMP)


def _read_lines(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ══════════════════════════ Scratchpad ══════════════════════════


class TestScratchpadLogging:
    """四类日志写入与 JSONL 格式。"""

    def test_log_init_fields(self, tmp_path, fixed_time):
        sp = Scratchpad(dir_path=str(tmp_path), session_id="s1")
        sp.log_init("今天大盘怎么样？")
        (entry,) = _read_lines(sp.path)
        assert entry["ts"] == FIXED_ISO
        assert entry["type"] == "init"
        assert entry["session_id"] == "s1"
        assert json.loads(entry["query"]) == "今天大盘怎么样？"

    def test_log_tool_call_fields(self, tmp_path, fixed_time):
        sp = Scratchpad(dir_path=str(tmp_path))
        sp.log_tool_call("get_stock_price", {"symbol": "600519.SH"})
        (entry,) = _read_lines(sp.path)
        assert entry["ts"] == FIXED_ISO
        assert entry["type"] == "tool_call"
        assert entry["tool"] == "get_stock_price"
        assert json.loads(entry["args"]) == {"symbol": "600519.SH"}

    def test_log_tool_result_with_summary(self, tmp_path):
        sp = Scratchpad(dir_path=str(tmp_path))
        sp.log_tool_result("get_stock_price", {"price": 1680.5}, llm_summary="茅台现价 1680.5")
        (entry,) = _read_lines(sp.path)
        assert entry["type"] == "tool_result"
        assert entry["tool"] == "get_stock_price"
        assert json.loads(entry["result"]) == {"price": 1680.5}
        assert json.loads(entry["llm_summary"]) == "茅台现价 1680.5"

    def test_log_tool_result_without_summary_is_none(self, tmp_path):
        sp = Scratchpad(dir_path=str(tmp_path))
        sp.log_tool_result("t", {"ok": True})
        (entry,) = _read_lines(sp.path)
        assert entry["llm_summary"] is None

    def test_log_thinking(self, tmp_path, fixed_time):
        sp = Scratchpad(dir_path=str(tmp_path))
        sp.log_thinking("先查指数再查板块")
        (entry,) = _read_lines(sp.path)
        assert entry["type"] == "thinking"
        assert entry["ts"] == FIXED_ISO
        assert json.loads(entry["text"]) == "先查指数再查板块"

    def test_multiple_logs_append_in_order(self, tmp_path):
        sp = Scratchpad(dir_path=str(tmp_path))
        sp.log_init("q")
        sp.log_tool_call("t1", {})
        sp.log_tool_result("t1", {"r": 1})
        sp.log_thinking("think")
        entries = _read_lines(sp.path)
        assert [e["type"] for e in entries] == ["init", "tool_call", "tool_result", "thinking"]
        assert all(isinstance(e["ts"], str) for e in entries)

    def test_timestamp_is_iso8601_by_default(self, tmp_path):
        """不打时间补丁时，ts 也应是合法 ISO8601。"""
        sp = Scratchpad(dir_path=str(tmp_path))
        sp.log_init("q")
        (entry,) = _read_lines(sp.path)
        datetime.fromisoformat(entry["ts"])  # 不抛异常即合法

    def test_concurrent_writes_no_interleave(self, tmp_path):
        """多线程并发写：行数完整、每行均为合法 JSON。"""
        sp = Scratchpad(dir_path=str(tmp_path))

        def worker(i):
            for j in range(20):
                sp.log_thinking(f"t{i}-{j}")

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, range(8)))
        lines = _read_lines(sp.path)
        assert len(lines) == 160
        assert all(e["type"] == "thinking" for e in lines)


class TestScratchpadPathAndDir:
    """文件名规则、目录三级惰性解析、path 属性。"""

    def test_filename_pattern_with_session_id(self, tmp_path, fixed_time):
        sp = Scratchpad(dir_path=str(tmp_path), session_id="abc123")
        assert os.path.basename(sp.path) == f"{FIXED_STAMP}_abc123.jsonl"

    def test_filename_pattern_with_short_uuid(self, tmp_path, fixed_time):
        sp = Scratchpad(dir_path=str(tmp_path))
        name = os.path.basename(sp.path)
        assert re.fullmatch(rf"{FIXED_STAMP}_[0-9a-f]{{8}}\.jsonl", name)
        assert re.fullmatch(r"[0-9a-f]{8}", sp.session_id)

    def test_path_available_before_dir_initialized(self, tmp_path):
        """目录尚未创建时 path 也可返回预期路径，且不触碰文件系统。"""
        target = tmp_path / "not_yet_created"
        sp = Scratchpad(dir_path=str(target), session_id="s")
        expected = os.path.join(str(target), os.path.basename(sp.path))
        assert sp.path == expected
        assert not target.exists()  # 仅读取 path 不创建目录

    def test_dir_param_wins_over_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCRATCHPAD_DIR", str(tmp_path / "env_dir"))
        sp = Scratchpad(dir_path=str(tmp_path / "param_dir"), session_id="s")
        assert sp.path.startswith(str(tmp_path / "param_dir"))

    def test_dir_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCRATCHPAD_DIR", str(tmp_path / "env_dir"))
        sp = Scratchpad(session_id="s")
        assert sp.path.startswith(str(tmp_path / "env_dir"))
        sp.log_init("q")
        assert os.path.exists(sp.path)

    def test_dir_fallback_data_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "data_root"))
        sp = Scratchpad(session_id="s")
        assert sp.path == os.path.join(str(tmp_path / "data_root"), "scratchpad", os.path.basename(sp.path))

    def test_dir_fallback_default(self, monkeypatch, tmp_path):
        """两级环境变量都缺失时回退 data/scratchpad（相对 cwd）。"""
        monkeypatch.chdir(tmp_path)
        sp = Scratchpad(session_id="s")
        assert sp.path == os.path.join("data", "scratchpad", os.path.basename(sp.path))


class TestScratchpadFailSafe:
    """坏目录降级：任何 OSError 吞掉 + warning，绝不抛异常。"""

    def test_dir_path_is_a_file(self, tmp_path, caplog):
        """dir_path 指向已存在的文件 → makedirs 抛 FileExistsError（OSError 子类）。"""
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file", encoding="utf-8")
        sp = Scratchpad(dir_path=str(blocker / "sub"))
        with caplog.at_level(logging.WARNING, logger="agent.agent_audit"):
            sp.log_init("q")  # 不得抛异常
            sp.log_tool_call("t", {})
        assert any("Scratchpad 写入失败" in r.message for r in caplog.records)

    def test_unwritable_dir(self, tmp_path, caplog):
        """目录只读 → open 抛 PermissionError，同样吞掉。"""
        ro_dir = tmp_path / "readonly"
        ro_dir.mkdir()
        os.chmod(ro_dir, 0o555)
        try:
            if os.access(ro_dir, os.W_OK):  # root 等场景下跳过
                pytest.skip("当前用户可绕过目录写权限")
            sp = Scratchpad(dir_path=str(ro_dir))
            with caplog.at_level(logging.WARNING, logger="agent.agent_audit"):
                sp.log_init("q")
            assert any("Scratchpad 写入失败" in r.message for r in caplog.records)
        finally:
            os.chmod(ro_dir, 0o755)

    def test_failure_does_not_block_subsequent_success(self, tmp_path, monkeypatch, caplog):
        """一次失败后，目录恢复可用时后续写入照常成功。"""
        sp = Scratchpad(dir_path=str(tmp_path / "d"), session_id="s")
        real_makedirs = os.makedirs

        def boom(*a, **kw):
            raise OSError("模拟磁盘故障")

        monkeypatch.setattr(agent_audit.os, "makedirs", boom)
        with caplog.at_level(logging.WARNING, logger="agent.agent_audit"):
            sp.log_init("第一次（失败）")
        monkeypatch.setattr(agent_audit.os, "makedirs", real_makedirs)
        sp.log_init("第二次（成功）")
        entries = _read_lines(sp.path)
        assert len(entries) == 1
        assert "第二次" in entries[0]["query"]


class TestScratchpadJsonSafe:
    """不可序列化对象兜底与超长截断。"""

    def test_non_serializable_result_as_str(self, tmp_path):
        sp = Scratchpad(dir_path=str(tmp_path))
        sp.log_tool_result("t", object())
        (entry,) = _read_lines(sp.path)  # 每行必须是合法 JSON
        assert "<object object at" in entry["result"]  # default=str 兜底（外层带 JSON 引号）

    def test_nested_non_serializable_result(self, tmp_path):
        sp = Scratchpad(dir_path=str(tmp_path))
        sp.log_tool_result("t", {"data": {1, 2, 3}, "obj": object()})
        (entry,) = _read_lines(sp.path)
        assert "data" in entry["result"]  # default=str 兜底后仍可解析出结构

    def test_non_serializable_args(self, tmp_path):
        sp = Scratchpad(dir_path=str(tmp_path))
        sp.log_tool_call("t", {"cb": lambda x: x})
        (entry,) = _read_lines(sp.path)
        assert "function" in entry["args"]

    def test_overlong_result_truncated(self, tmp_path):
        sp = Scratchpad(dir_path=str(tmp_path))
        sp.log_tool_result("t", "x" * (MAX_RESULT_CHARS * 3))
        (entry,) = _read_lines(sp.path)
        assert "已截断" in entry["result"]
        assert len(entry["result"]) < MAX_RESULT_CHARS * 2

    def test_overlong_thinking_truncated(self, tmp_path):
        sp = Scratchpad(dir_path=str(tmp_path))
        sp.log_thinking("想" * (MAX_TEXT_CHARS * 3))
        (entry,) = _read_lines(sp.path)
        assert "已截断" in entry["text"]


class TestDefaultScratchpad:
    """模块级惰性单例。"""

    def test_singleton_same_instance(self, monkeypatch, tmp_path):
        monkeypatch.setattr(agent_audit, "_default_scratchpad", None)
        monkeypatch.setenv("SCRATCHPAD_DIR", str(tmp_path))
        sp1 = default_scratchpad()
        sp2 = default_scratchpad()
        assert sp1 is sp2
        assert isinstance(sp1, Scratchpad)

    def test_singleton_lazy(self, monkeypatch):
        monkeypatch.setattr(agent_audit, "_default_scratchpad", None)
        assert agent_audit._default_scratchpad is None  # 未调用前不创建


# ══════════════════════════ ToolCallGuard ══════════════════════════


class TestGuardCountWarning:
    """计数软警告：边界恰好上限不警告，超上限警告。"""

    def test_no_warning_under_limit(self):
        g = ToolCallGuard(max_calls_per_tool=3)
        assert g.check("t", {"a": 1}) is None

    def test_boundary_exactly_at_limit_no_warning(self):
        # 计数边界测试须避开相似度规则干扰：阈值拉满且参数与历史差异明显
        g = ToolCallGuard(max_calls_per_tool=3, similarity_threshold=1.0)
        g.record("t", {"q": "第一次查询内容AAAA"})
        g.record("t", {"q": "第二次查询内容BBBB"})
        # 第 3 次调用（含本次=上限）不警告
        assert g.check("t", {"q": "完全不同的第三次zzzz"}) is None

    def test_over_limit_warning_content(self):
        g = ToolCallGuard(max_calls_per_tool=3)
        for _ in range(3):
            g.record("t", {"a": 1})
        warn = g.check("t", {"a": 2})  # 第 4 次
        assert warn is not None
        assert "已调用 3 次" in warn
        assert "第 4 次" in warn
        # 三条出路齐备
        assert "换一个工具" in warn
        assert "换一组参数" in warn
        assert "承认数据缺口" in warn
        assert "收尾作答" in warn

    def test_counts_independent_per_tool(self):
        # 参数各异，排除相似度规则干扰，专注验证计数按工具独立
        g = ToolCallGuard(max_calls_per_tool=2, similarity_threshold=1.0)
        g.record("t1", {"q": "t1第一次AAAA"})
        g.record("t1", {"q": "t1第二次BBBB"})
        g.record("t2", {"q": "t2第一次CCCC"})
        assert g.check("t1", {"q": "t1第三次zzzz"}) is not None  # t1 第 3 次
        assert g.check("t2", {"q": "t2第二次yyyy"}) is None  # t2 才第 2 次

    def test_check_does_not_record(self):
        """check 是纯查询，不影响计数；重复 check 同一调用不累计。"""
        g = ToolCallGuard(max_calls_per_tool=1)
        assert g.check("t", {"a": 1}) is None
        assert g.check("t", {"a": 1}) is None  # 仍未 record，不应触发计数警告


class TestGuardSimilarityWarning:
    """重复查询软警告：difflib 相似度 ≥ 阈值触发。"""

    def test_identical_args_trigger(self):
        g = ToolCallGuard(similarity_threshold=0.7)
        g.record("t", {"symbol": "600519.SH", "days": 30})
        warn = g.check("t", {"symbol": "600519.SH", "days": 30})
        assert warn is not None
        assert "相似" in warn
        assert "重复查询" in warn

    def test_key_order_independent(self):
        """args 规范化 sorted keys：键序不同视为同一查询。"""
        g = ToolCallGuard(similarity_threshold=0.99)
        g.record("t", {"a": 1, "b": 2})
        assert g.check("t", {"b": 2, "a": 1}) is not None

    def test_below_threshold_no_warning(self):
        g = ToolCallGuard(similarity_threshold=0.7)
        args1 = {"symbol": "600519.SH", "window": "2024Q1"}
        args2 = {"keyword": "新能源板块政策补贴退坡", "page": 3, "lang": "zh"}
        # 自验证：两者规范化 JSON 相似度确实低于阈值
        n1 = g._normalize_args(args1)
        n2 = g._normalize_args(args2)
        assert difflib.SequenceMatcher(None, n1, n2).ratio() < 0.7
        g.record("t", args1)
        assert g.check("t", args2) is None

    def test_at_threshold_boundary_triggers(self):
        """阈值边界：相似度恰好等于阈值（≥）也触发。"""
        g = ToolCallGuard(similarity_threshold=1.0)
        g.record("t", {"a": 1})
        # 相似度 1.0 ≥ 1.0
        assert g.check("t", {"a": 1}) is not None

    def test_different_tool_same_args_no_warning(self):
        """重复查询仅与同工具历史比较，跨工具不误判。"""
        g = ToolCallGuard(similarity_threshold=0.7)
        g.record("tool_a", {"q": "茅台"})
        assert g.check("tool_b", {"q": "茅台"}) is None

    def test_similarity_warning_suggests_alternatives(self):
        g = ToolCallGuard(similarity_threshold=0.5)
        g.record("t", {"symbol": "600519.SH"})
        warn = g.check("t", {"symbol": "600519.SH"})
        assert "换一组参数" in warn
        assert "收尾作答" in warn

    def test_non_serializable_args_no_crash(self):
        g = ToolCallGuard()
        g.record("t", {"cb": object()})
        # 不抛异常；结果可以是警告或 None，但绝不能崩
        g.check("t", {"cb": object()})


class TestGuardReset:
    def test_reset_clears_count_and_history(self):
        g = ToolCallGuard(max_calls_per_tool=1, similarity_threshold=0.9)
        g.record("t", {"a": 1})
        assert g.check("t", {"a": 1}) is not None  # 超上限
        g.reset()
        assert g.check("t", {"a": 1}) is None  # 计数与历史均已清空


# ══════════════════════════ microcompact ══════════════════════════


def _make_tool_loop(n_rounds, content_size=100):
    """构造 n_rounds 轮 OpenAI 工具循环消息：每轮 assistant(tool_calls) + 对应 tool 消息。"""
    messages = [
        {"role": "system", "content": "你是 A 股分析助手"},
        {"role": "user", "content": "分析一下茅台"},
    ]
    for i in range(n_rounds):
        call_id = f"call_{i}"
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": "get_price", "arguments": "{}"},
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": f"r{i}-" + "x" * content_size,
        })
    return messages


class TestMicrocompactNoTrigger:
    def test_no_trigger_returns_equal_content(self):
        messages = _make_tool_loop(3)
        result, stats = microcompact(messages)
        assert stats["cleared"] == 0
        assert [m.get("content") for m in result] == [m.get("content") for m in messages]
        assert stats["chars_after"] == stats["chars_before"]

    def test_no_trigger_returns_new_list(self):
        messages = _make_tool_loop(2)
        result, _ = microcompact(messages)
        assert result is not messages
        assert result == messages

    def test_boundary_exactly_at_limits_no_trigger(self):
        """tool 数 == max_tool_msgs 且字符数 == max_chars 时不触发（严格大于才触发）。"""
        messages = _make_tool_loop(2, content_size=10)
        total = sum(len(str(m.get("content") or "")) for m in messages)
        result, stats = microcompact(messages, max_tool_msgs=2, max_chars=total)
        assert stats["cleared"] == 0


class TestMicrocompactTrigger:
    def test_trigger_by_tool_count(self):
        messages = _make_tool_loop(9)  # 9 > max_tool_msgs=8
        result, stats = microcompact(messages, max_tool_msgs=8, keep_recent=4)
        tool_msgs = [m for m in result if m["role"] == "tool"]
        assert stats["cleared"] == 5
        # 最旧 5 条被清理，最近 4 条原样保留
        assert all(m["content"] == CLEARED_PLACEHOLDER for m in tool_msgs[:5])
        assert all(m["content"].startswith("r") for m in tool_msgs[5:])

    def test_trigger_by_chars_only(self):
        messages = _make_tool_loop(6, content_size=5000)
        total = sum(len(str(m.get("content") or "")) for m in messages)
        result, stats = microcompact(messages, max_tool_msgs=100, max_chars=total - 1, keep_recent=2)
        assert stats["cleared"] == 4  # 6 条 tool，保护最近 2 条

    def test_trigger_both_conditions(self):
        messages = _make_tool_loop(10, content_size=5000)
        result, stats = microcompact(messages, max_tool_msgs=5, max_chars=100, keep_recent=3)
        assert stats["cleared"] == 7

    def test_keep_recent_protects_all_when_few_tools(self):
        """tool 消息数 ≤ keep_recent 时，即使字符超限也不清理任何一条。"""
        messages = _make_tool_loop(3, content_size=10000)
        total = sum(len(str(m.get("content") or "")) for m in messages)
        result, stats = microcompact(messages, max_tool_msgs=100, max_chars=total - 1, keep_recent=4)
        assert stats["cleared"] == 0
        assert stats["chars_after"] == stats["chars_before"]

    def test_keep_recent_zero_clears_all(self):
        messages = _make_tool_loop(9)
        result, stats = microcompact(messages, max_tool_msgs=8, keep_recent=0)
        assert stats["cleared"] == 9
        assert all(m["content"] == CLEARED_PLACEHOLDER for m in result if m["role"] == "tool")

    def test_chars_stats_correct(self):
        messages = _make_tool_loop(9, content_size=100)
        result, stats = microcompact(messages, max_tool_msgs=8, keep_recent=4)
        orig_tool = [m for m in messages if m["role"] == "tool"]
        saved = sum(len(m["content"]) - len(CLEARED_PLACEHOLDER) for m in orig_tool[:5])
        assert stats["chars_after"] == stats["chars_before"] - saved

    def test_none_and_nonstring_tool_content(self):
        """content 为 None 计 0 字符；触发后同样被替换为占位符。"""
        messages = [
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": None},
        ]
        # 构造 9 条 tool 触发
        messages = messages + _make_tool_loop(8)[2:]
        result, stats = microcompact(messages, max_tool_msgs=8, keep_recent=1)
        assert result[1]["content"] == CLEARED_PLACEHOLDER
        assert stats["cleared"] == 8


class TestMicrocompactSafety:
    def test_input_not_mutated(self):
        messages = _make_tool_loop(9)
        snapshot = [dict(m) for m in messages]
        microcompact(messages, max_tool_msgs=8, keep_recent=4)
        assert messages == snapshot  # 入参原封不动

    def test_tool_call_pairing_intact(self):
        """tool_calls 结构与 tool_call_id 配对关系绝不改变。"""
        messages = _make_tool_loop(9)
        result, _ = microcompact(messages, max_tool_msgs=8, keep_recent=4)
        # assistant 消息完全未动（含 tool_calls）
        for orig, new in zip(messages, result):
            if orig["role"] == "assistant":
                assert new == orig
        # 每条 tool 消息的 tool_call_id 与角色保留，配对顺序不变
        orig_ids = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
        new_ids = [m["tool_call_id"] for m in result if m["role"] == "tool"]
        assert new_ids == orig_ids
        # 每个 assistant tool_calls 的 id 仍能找到对应 tool 消息
        for m in result:
            if m["role"] == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    assert tc["id"] in new_ids

    def test_system_user_assistant_untouched(self):
        messages = _make_tool_loop(9)
        result, _ = microcompact(messages, max_tool_msgs=8, keep_recent=4)
        for orig, new in zip(messages, result):
            if orig["role"] in ("system", "user", "assistant"):
                assert new == orig

    def test_result_length_and_order_preserved(self):
        messages = _make_tool_loop(9)
        result, _ = microcompact(messages, max_tool_msgs=8, keep_recent=4)
        assert len(result) == len(messages)
        assert [m["role"] for m in result] == [m["role"] for m in messages]
