"""
事后打分层测试（tests/test_scorer.py）。

覆盖 agent/scorer.py 全部核心逻辑 + scripts/score_accountability.py 的
pct_fn 生产实现（pro 全 mock）。绝不发起真实网络请求。

运行：/usr/local/bin/python3 -m pytest tests/test_scorer.py -v
"""

import importlib.util
import json
import logging
import os
import sys
import threading
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# 保证无论 conftest.py 是否就绪，都能从项目根导入 agent 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import scorer  # noqa: E402

# ── 以文件路径加载 CLI 脚本（scripts/ 不是包）──
_CLI_PATH = Path(__file__).resolve().parent.parent / "scripts" / "score_accountability.py"
_spec = importlib.util.spec_from_file_location("score_accountability", _CLI_PATH)
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)

TODAY = date(2024, 2, 1)  # 固定"今天"，保证 find_pending 断言稳定


# ─────────────────────────────────────────────
# 测试数据构造辅助
# ─────────────────────────────────────────────

def make_record(rec_id="r1", trade_date="20240115", mode="market_review",
                sector=None, content="综合判断：市场偏多，建议积极关注。",
                score=None):
    return {
        "id": rec_id,
        "ts": f"{trade_date}T20:00:00",
        "trade_date": trade_date,
        "mode": mode,
        "sector": sector,
        "content": content,
        "context_excerpt": " excerpt ",
        "numbers": [1.23],
        "score": score,
        "scored_at": None,
        "score_note": None,
    }


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_index_daily_df(closes, start="20240115"):
    """构造 mock 的 pro.index_daily 返回值：从 start 起连续交易日的收盘价。"""
    base = date(int(start[:4]), int(start[4:6]), int(start[6:8]))
    rows = []
    for i, close in enumerate(closes):
        td = (base + timedelta(days=i)).strftime("%Y%m%d")
        rows.append({"trade_date": td, "close": float(close)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# extract_direction：各分支
# ─────────────────────────────────────────────

class TestExtractDirection:
    @pytest.mark.parametrize("word", ["偏多", "乐观", "强势", "看好"])
    def test_bullish_words_in_priority_scope(self, word):
        assert scorer.extract_direction(f"综合判断：{word}。") == "bullish"

    @pytest.mark.parametrize("word", ["偏空", "谨慎", "弱势", "回避"])
    def test_bearish_words_in_priority_scope(self, word):
        assert scorer.extract_direction(f"综合判断：{word}。") == "bearish"

    def test_zongti_marker_also_priority(self):
        assert scorer.extract_direction("总体来看，后市偏空。") == "bearish"

    def test_no_direction_words_is_neutral(self):
        assert scorer.extract_direction("今天市场成交额有所放大，板块轮动明显。") == "neutral"

    def test_empty_content_is_neutral(self):
        assert scorer.extract_direction("") == "neutral"
        assert scorer.extract_direction(None) == "neutral"

    def test_priority_scope_beats_full_text(self):
        # 全文其他位置有"看好"，但『综合判断』行说谨慎 → 以优先范围为准
        content = "资金看好题材股炒作。\n综合判断：短线谨慎，控制仓位。"
        assert scorer.extract_direction(content) == "bearish"

    def test_direction_on_line_after_marker(self):
        # 标题与结论分两行：标记行本身无方向词，下一行有
        content = "综合判断\n偏多思路，逢低布局。"
        assert scorer.extract_direction(content) == "bullish"

    def test_tie_in_priority_scope_is_neutral(self):
        # 多空词频打平 → 保守记 neutral（如『谨慎乐观』）
        assert scorer.extract_direction("综合判断：谨慎乐观。") == "neutral"

    def test_fallback_to_full_text_when_priority_has_no_words(self):
        # 有『综合判断』标记但该行及下一行无方向词 → 退而对全文裁决
        content = "综合判断：见下。\n市场结构分化。\n操作上看好低估值蓝筹。"
        assert scorer.extract_direction(content) == "bullish"

    def test_word_count_majority_wins(self):
        # 优先范围内 2 个牛市词 vs 1 个熊市词 → bullish
        content = "综合判断：偏多。整体强势，但局部需谨慎。"
        assert scorer.extract_direction(content) == "bullish"


# ─────────────────────────────────────────────
# score_record：hit/miss/neutral 与 ±1% 边界
# ─────────────────────────────────────────────

class TestScoreRecord:
    def test_hit_bullish_up(self):
        score, note = scorer.score_record(make_record(content="综合判断：偏多。"), 2.5)
        assert score == "hit"
        assert "bullish" in note and "+2.50%" in note

    def test_hit_bearish_down(self):
        score, _ = scorer.score_record(make_record(content="综合判断：偏空。"), -3.0)
        assert score == "hit"

    def test_miss_bullish_but_down(self):
        score, _ = scorer.score_record(make_record(content="综合判断：偏多。"), -1.5)
        assert score == "miss"

    def test_miss_bearish_but_up(self):
        score, _ = scorer.score_record(make_record(content="综合判断：谨慎。"), 1.5)
        assert score == "miss"

    def test_boundary_plus_1pct_is_not_hit(self):
        # 恰为 +1.0%：阈值是严格大于，不计 hit
        score, _ = scorer.score_record(make_record(content="综合判断：看多，偏多。"), 1.0)
        assert score == "neutral"

    def test_boundary_minus_1pct_is_not_hit(self):
        score, _ = scorer.score_record(make_record(content="综合判断：偏空。"), -1.0)
        assert score == "neutral"

    def test_just_beyond_boundary_hits(self):
        score, _ = scorer.score_record(make_record(content="综合判断：偏多。"), 1.01)
        assert score == "hit"
        score, _ = scorer.score_record(make_record(content="综合判断：偏空。"), -1.01)
        assert score == "hit"

    def test_neutral_direction_always_neutral(self):
        # 方向 neutral 时，无论涨跌幅多大都不计 hit/miss
        score, note = scorer.score_record(make_record(content="市场今日成交活跃。"), 5.0)
        assert score == "neutral"
        assert "neutral" in note

    def test_small_move_is_neutral(self):
        score, _ = scorer.score_record(make_record(content="综合判断：偏多。"), 0.5)
        assert score == "neutral"

    def test_note_contains_basis(self):
        _, note = scorer.score_record(make_record(content="综合判断：谨慎。"), -2.0)
        assert "方向判断=bearish" in note
        assert "实际区间涨跌幅=-2.00%" in note
        assert "hit" in note


# ─────────────────────────────────────────────
# find_pending：过滤规则
# ─────────────────────────────────────────────

class TestFindPending:
    def test_unscored_old_record_is_pending(self):
        rec = make_record(trade_date="20240120")  # 距今 12 天
        assert scorer.find_pending([rec], days=5, today=TODAY) == [rec]

    def test_scored_record_excluded(self):
        rec = make_record(trade_date="20240120", score="hit")
        assert scorer.find_pending([rec], days=5, today=TODAY) == []

    def test_recent_record_excluded(self):
        rec = make_record(trade_date="20240130")  # 距今 2 天 < 5
        assert scorer.find_pending([rec], days=5, today=TODAY) == []

    def test_exactly_days_old_included(self):
        rec = make_record(trade_date="20240127")  # 距今恰好 5 天，>= 5 应入选
        assert scorer.find_pending([rec], days=5, today=TODAY) == [rec]

    def test_invalid_trade_date_excluded(self):
        for bad in ["", "2024-01-20", "20241301", None]:
            rec = make_record(trade_date=bad)
            assert scorer.find_pending([rec], days=5, today=TODAY) == []

    def test_days_parameter_respected(self):
        rec = make_record(trade_date="20240129")  # 距今 3 天
        assert scorer.find_pending([rec], days=3, today=TODAY) == [rec]
        assert scorer.find_pending([rec], days=5, today=TODAY) == []


# ─────────────────────────────────────────────
# apply_scores：端到端写回
# ─────────────────────────────────────────────

class TestApplyScores:
    def _setup_archive(self, tmp_path):
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        old = make_record("old-bull", trade_date="20240120",
                          content="综合判断：偏多。")
        old_bear = make_record("old-bear", trade_date="20240120",
                               content="综合判断：偏空。")
        recent = make_record("recent", trade_date="20240131",
                             content="综合判断：偏多。")
        done = make_record("done", trade_date="20240120", score="hit")
        write_jsonl(archive_dir / "archive_20240120.jsonl", [old, done])
        write_jsonl(archive_dir / "archive_20240131.jsonl", [old_bear, recent])
        return archive_dir, {"old-bull": old, "old-bear": old_bear,
                             "recent": recent, "done": done}

    def test_end_to_end_writeback(self, tmp_path):
        archive_dir, _ = self._setup_archive(tmp_path)
        pct_map = {"old-bull": 2.0, "old-bear": -3.0}
        pct_fn = lambda rec: pct_map.get(rec["id"])

        result = scorer.apply_scores(archive_dir, pct_fn, days=5,
                                     writer_lock=threading.Lock(), today=TODAY)

        assert len(result["scored"]) == 2
        assert result["skipped"] == []
        assert result["files_rewritten"] == 2

        f1 = {r["id"]: r for r in read_jsonl(archive_dir / "archive_20240120.jsonl")}
        f2 = {r["id"]: r for r in read_jsonl(archive_dir / "archive_20240131.jsonl")}
        assert f1["old-bull"]["score"] == "hit"
        assert f2["old-bear"]["score"] == "hit"
        for rec in (f1["old-bull"], f2["old-bear"]):
            assert rec["scored_at"]  # 已写回打分时间
            assert "方向判断=" in rec["score_note"]

        # 未到期与已打分的记录保持原样
        assert f2["recent"]["score"] is None
        assert f2["recent"]["scored_at"] is None
        assert f1["done"]["score"] == "hit"
        assert f1["done"]["score_note"] is None

    def test_pct_fn_none_skips_without_writeback(self, tmp_path):
        archive_dir, _ = self._setup_archive(tmp_path)
        # old-bull 数据取不到 → 跳过；old-bear 正常打分
        pct_map = {"old-bull": None, "old-bear": -2.0}
        result = scorer.apply_scores(archive_dir, lambda rec: pct_map.get(rec["id"]),
                                     days=5, today=TODAY)

        assert [s["id"] for s in result["skipped"]] == ["old-bull"]
        assert [s["id"] for s in result["scored"]] == ["old-bear"]
        # 20240120 文件里唯一 pending 的记录被跳过 → 文件不应回写
        assert result["files_rewritten"] == 1

        f1 = {r["id"]: r for r in read_jsonl(archive_dir / "archive_20240120.jsonl")}
        assert f1["old-bull"]["score"] is None
        assert f1["old-bull"]["scored_at"] is None

    def test_miss_and_neutral_written(self, tmp_path):
        archive_dir, _ = self._setup_archive(tmp_path)
        pct_map = {"old-bull": -2.0, "old-bear": 0.3}  # miss / neutral
        result = scorer.apply_scores(archive_dir, lambda rec: pct_map.get(rec["id"]),
                                     days=5, today=TODAY)
        scores = {s["id"]: s["score"] for s in result["scored"]}
        assert scores == {"old-bull": "miss", "old-bear": "neutral"}

    def test_no_pending_no_writeback(self, tmp_path):
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        write_jsonl(archive_dir / "archive_20240131.jsonl",
                    [make_record("recent", trade_date="20240131")])
        called = []
        result = scorer.apply_scores(
            archive_dir, lambda rec: called.append(rec) or 1.0, days=5, today=TODAY)
        assert result["scored"] == [] and result["skipped"] == []
        assert result["files_rewritten"] == 0
        assert called == []  # 无 pending 时不应调用 pct_fn

    def test_bad_jsonl_lines_preserved(self, tmp_path):
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        path = archive_dir / "archive_20240120.jsonl"
        rec = make_record("old", trade_date="20240120", content="综合判断：偏多。")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.write("{这不是合法JSON\n")

        scorer.apply_scores(archive_dir, lambda r: 2.0, days=5, today=TODAY)
        with open(path, encoding="utf-8") as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        assert len(lines) == 2
        assert json.loads(lines[0])["score"] == "hit"
        assert lines[1] == "{这不是合法JSON"  # 坏行原样保留

    def test_days_parameter_filters(self, tmp_path):
        archive_dir, _ = self._setup_archive(tmp_path)
        pct_fn = lambda rec: 2.0
        result = scorer.apply_scores(archive_dir, pct_fn, days=20, today=TODAY)
        assert result["scored"] == []  # 距今最多 12 天 < 20，无待打分


# ─────────────────────────────────────────────
# CLI pct_fn 生产实现（pro 全 mock）
# ─────────────────────────────────────────────

class TestCliPctFn:
    def test_resolve_sw_code_exact(self):
        assert cli.resolve_sw_code("银行") == "801780.SI"

    def test_resolve_sw_code_alias(self):
        assert cli.resolve_sw_code("半导体") == "801080.SI"  # 半导体 → 电子

    def test_resolve_sw_code_unknown(self):
        assert cli.resolve_sw_code("不存在的板块") is None
        assert cli.resolve_sw_code("") is None
        assert cli.resolve_sw_code(None) is None

    def test_pct_change_enough_rows(self):
        pro = MagicMock()
        # 6 个交易日：100 → 105.5（第 5 个交易日后），涨 +5.5%
        pro.index_daily.return_value = make_index_daily_df(
            [100, 101, 102, 103, 104, 105.5])
        pct = cli.pct_change_n_trading_days(pro, "000001.SH", "20240115")
        assert pct == pytest.approx(5.5)
        pro.index_daily.assert_called_once()
        _, kwargs = pro.index_daily.call_args
        assert kwargs["ts_code"] == "000001.SH"
        assert kwargs["start_date"] == "20240115"

    def test_pct_change_insufficient_rows_returns_none(self):
        pro = MagicMock()
        pro.index_daily.return_value = make_index_daily_df([100, 101, 102])  # 不足 6 行
        assert cli.pct_change_n_trading_days(pro, "000001.SH", "20240115") is None

    def test_pct_change_empty_df_returns_none(self):
        pro = MagicMock()
        pro.index_daily.return_value = pd.DataFrame()
        assert cli.pct_change_n_trading_days(pro, "000001.SH", "20240115") is None
        pro.index_daily.return_value = None
        assert cli.pct_change_n_trading_days(pro, "000001.SH", "20240115") is None

    def test_pct_fn_routes_sector_to_sw_index(self):
        pro = MagicMock()
        pro.index_daily.return_value = make_index_daily_df([100, 100, 100, 100, 100, 102])
        pct_fn = cli.make_pct_fn(pro)
        rec = make_record(mode="sector_deep_dive", sector="银行")
        assert pct_fn(rec) == pytest.approx(2.0)
        _, kwargs = pro.index_daily.call_args
        assert kwargs["ts_code"] == "801780.SI"

    def test_pct_fn_routes_market_modes_to_sh_index(self):
        pro = MagicMock()
        pro.index_daily.return_value = make_index_daily_df([100, 100, 100, 100, 100, 98])
        pct_fn = cli.make_pct_fn(pro)
        for mode in ("market_review", "agent_query"):
            rec = make_record(mode=mode)
            assert pct_fn(rec) == pytest.approx(-2.0)
            _, kwargs = pro.index_daily.call_args
            assert kwargs["ts_code"] == "000001.SH"

    def test_pct_fn_unknown_sector_returns_none(self):
        pct_fn = cli.make_pct_fn(MagicMock())
        rec = make_record(mode="sector_deep_dive", sector="不存在的板块")
        assert pct_fn(rec) is None

    def test_pct_fn_exception_returns_none(self):
        pro = MagicMock()
        pro.index_daily.side_effect = RuntimeError("网络错误")
        assert cli.make_pct_fn(pro)(make_record()) is None


# ─────────────────────────────────────────────
# CLI main：端到端（_get_pro 打桩，存档用 tmp 目录）
# ─────────────────────────────────────────────

class TestCliMain:
    def test_main_scores_and_prints_summary(self, tmp_path, monkeypatch, capsys):
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        write_jsonl(archive_dir / "archive_20240120.jsonl", [
            make_record("r1", trade_date="20240120", content="综合判断：偏多。"),
            make_record("r2", trade_date="20240120",
                        mode="sector_deep_dive", sector="银行",
                        content="综合判断：谨慎。"),
        ])

        pro = MagicMock()
        # r1 看多且大盘上涨 → hit；r2 看空且板块下跌 → hit
        pro.index_daily.side_effect = [
            make_index_daily_df([100, 101, 102, 103, 104, 106]),
            make_index_daily_df([100, 99, 98, 97, 96, 94]),
        ]
        monkeypatch.setattr("agent.data_fetcher._get_pro", lambda: pro)

        rc = cli.main(["--days", "5", "--archive-dir", str(archive_dir)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "hit=2" in out and "miss=0" in out and "neutral=0" in out
        assert "r1" in out and "r2" in out

        scored = {r["id"]: r for r in read_jsonl(archive_dir / "archive_20240120.jsonl")}
        assert scored["r1"]["score"] == "hit"
        assert scored["r2"]["score"] == "hit"

    def test_main_missing_archive_dir(self, tmp_path, capsys):
        rc = cli.main(["--archive-dir", str(tmp_path / "不存在")])
        assert rc == 0
        assert "无待打分记录" in capsys.readouterr().out

    def test_main_no_pro_aborts(self, tmp_path, monkeypatch, capsys):
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        monkeypatch.setattr("agent.data_fetcher._get_pro", lambda: None)
        rc = cli.main(["--archive-dir", str(archive_dir)])
        assert rc == 1
        assert "Tushare 连接不可用" in capsys.readouterr().out


# ─────────────────────────────────────────────
# DATA_DIR 统一约定 + Railway 临时存储警告（存储持久化波次）
# ─────────────────────────────────────────────

_WARN_KEYWORD = "运行在 Railway 但未挂卷"


class TestDataDirConvention:
    """ARCHIVE_DIR 显式优先；缺省推导为 ${DATA_DIR:-data}/archive。"""

    def test_default_derives_from_data_dir(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ARCHIVE_DIR", raising=False)
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "mydata"))
        assert scorer.default_archive_dir() == str(tmp_path / "mydata" / "archive")

    def test_default_without_data_dir_is_data_archive(self, monkeypatch):
        monkeypatch.delenv("ARCHIVE_DIR", raising=False)
        monkeypatch.delenv("DATA_DIR", raising=False)
        assert scorer.default_archive_dir() == os.path.join("data", "archive")

    def test_explicit_archive_dir_wins_over_data_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "mydata"))
        monkeypatch.setenv("ARCHIVE_DIR", str(tmp_path / "explicit"))
        assert scorer.default_archive_dir() == str(tmp_path / "explicit")


class TestEphemeralStorageWarning:
    def test_warns_on_railway_without_volume(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(scorer, "_EPHEMERAL_WARNED", False)
        monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
        monkeypatch.setenv("ARCHIVE_DIR", str(tmp_path / "archive"))
        with caplog.at_level(logging.WARNING, logger="agent.scorer"):
            scorer.default_archive_dir()
        assert any(_WARN_KEYWORD in r.getMessage() for r in caplog.records)

    def test_no_warning_when_dir_under_volume(self, monkeypatch, caplog):
        monkeypatch.setattr(scorer, "_EPHEMERAL_WARNED", False)
        monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
        monkeypatch.setenv("ARCHIVE_DIR", "/data/archive")
        with caplog.at_level(logging.WARNING, logger="agent.scorer"):
            scorer.default_archive_dir()
        assert not any(_WARN_KEYWORD in r.getMessage() for r in caplog.records)

    def test_no_warning_off_railway(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(scorer, "_EPHEMERAL_WARNED", False)
        monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
        monkeypatch.setenv("ARCHIVE_DIR", str(tmp_path / "archive"))
        with caplog.at_level(logging.WARNING, logger="agent.scorer"):
            scorer.default_archive_dir()
        assert not any(_WARN_KEYWORD in r.getMessage() for r in caplog.records)

    def test_warns_only_once_per_module(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(scorer, "_EPHEMERAL_WARNED", False)
        monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
        monkeypatch.setenv("ARCHIVE_DIR", str(tmp_path / "archive"))
        with caplog.at_level(logging.WARNING, logger="agent.scorer"):
            scorer.default_archive_dir()
            scorer.default_archive_dir()
        warns = [r for r in caplog.records if _WARN_KEYWORD in r.getMessage()]
        assert len(warns) == 1

    def test_data_dir_on_volume_no_warning(self, monkeypatch, caplog):
        monkeypatch.setattr(scorer, "_EPHEMERAL_WARNED", False)
        monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
        monkeypatch.delenv("ARCHIVE_DIR", raising=False)
        monkeypatch.setenv("DATA_DIR", "/data")
        with caplog.at_level(logging.WARNING, logger="agent.scorer"):
            assert scorer.default_archive_dir() == "/data/archive"
        assert not any(_WARN_KEYWORD in r.getMessage() for r in caplog.records)


class TestWriteJsonlMakedirs:
    """_write_jsonl 目录补强：目录不存在时自动创建（挂载卷首次写入场景）。"""

    def test_write_jsonl_creates_missing_dir(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "archive_20240120.jsonl"
        assert not path.parent.exists()
        rec = make_record("r1")
        scorer._write_jsonl(str(path), [rec], ["坏行"])
        assert path.exists()
        lines = path.read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[0])["id"] == "r1"
        assert lines[1] == "坏行"
