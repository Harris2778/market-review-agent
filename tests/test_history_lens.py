"""
以史为鉴模块测试（tests/test_history_lens.py）。

覆盖 agent/history_lens.py 全部核心逻辑：多文件扫描、sector/mode 过滤、
limit 与排序、方向提取三分支、打分映射、依据截断、300 字上限、
空目录/缺失目录返回 None、损坏行跳过、DATA_DIR 回退、accuracy 汇总。
绝不发起真实网络请求（模块为纯 stdlib 本地文件读取）。

运行：/usr/local/bin/python3 -m pytest tests/test_history_lens.py -v
"""

import json
import os
import sys
from pathlib import Path

import pytest

# 保证无论 conftest.py 是否就绪，都能从项目根导入 agent 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import history_lens  # noqa: E402


# ─────────────────────────────────────────────
# 测试数据构造辅助
# ─────────────────────────────────────────────

def make_record(rec_id="r1", trade_date="20240115", mode="market_review",
                sector=None, content="综合判断：市场偏多，建议积极关注。",
                score=None, score_note=None):
    return {
        "id": rec_id,
        "ts": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}T20:00:00",
        "trade_date": trade_date,
        "mode": mode,
        "sector": sector,
        "content": content,
        "context_excerpt": " excerpt ",
        "numbers": [],
        "score": score,
        "scored_at": None if score is None else "2024-01-20T09:00:00",
        "score_note": score_note,
    }


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@pytest.fixture
def archive_dir(tmp_path, monkeypatch):
    """把 ARCHIVE_DIR 隔离到 tmp_path（覆盖 conftest 的全局默认值）。"""
    d = tmp_path / "archive"
    d.mkdir()
    monkeypatch.setenv("ARCHIVE_DIR", str(d))
    return d


# ─────────────────────────────────────────────
# get_history_note：扫描、过滤、排序、limit
# ─────────────────────────────────────────────

class TestHistoryNoteScan:
    def test_multi_file_scan_and_desc_order(self, archive_dir):
        write_jsonl(archive_dir / "archive_20240115.jsonl",
                    [make_record("r-old", trade_date="20240115")])
        write_jsonl(archive_dir / "archive_20240120.jsonl",
                    [make_record("r-new", trade_date="20240120")])

        note = history_lens.get_history_note()
        lines = note.splitlines()
        assert lines[0] == "【以史为鉴：本智能体历史判断回顾】"
        # trade_date 倒序：新记录在前
        assert lines[1].startswith("01-20 ")
        assert lines[2].startswith("01-15 ")
        assert len(lines) == 3

    def test_ignores_non_archive_files(self, archive_dir):
        write_jsonl(archive_dir / "archive_20240115.jsonl",
                    [make_record("r1", trade_date="20240115")])
        (archive_dir / "notes.txt").write_text("archive_99999999 假记录", encoding="utf-8")
        write_jsonl(archive_dir / "other_20240120.jsonl",
                    [make_record("r-fake", trade_date="20240120")])

        note = history_lens.get_history_note()
        assert "01-15" in note
        assert "01-20" not in note  # 非 archive_ 前缀文件被忽略

    def test_empty_dir_returns_none(self, archive_dir):
        assert history_lens.get_history_note() is None

    def test_missing_dir_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ARCHIVE_DIR", str(tmp_path / "不存在"))
        assert history_lens.get_history_note() is None

    def test_sector_filter(self, archive_dir):
        write_jsonl(archive_dir / "archive_20240115.jsonl", [
            make_record("bank1", trade_date="20240115", mode="sector_deep_dive",
                        sector="银行", content="综合判断：偏多。"),
            make_record("semi", trade_date="20240116", mode="sector_deep_dive",
                        sector="半导体", content="综合判断：偏空。"),
            make_record("bank2", trade_date="20240117", mode="sector_deep_dive",
                        sector="银行", content="综合判断：谨慎。"),
            make_record("market", trade_date="20240118", sector=None),
        ])
        note = history_lens.get_history_note(sector="银行")
        lines = note.splitlines()[1:]
        assert len(lines) == 2
        assert lines[0].startswith("01-17 ")  # bank2（较新）
        assert lines[1].startswith("01-15 ")  # bank1

    def test_mode_filter(self, archive_dir):
        write_jsonl(archive_dir / "archive_20240115.jsonl", [
            make_record("m1", trade_date="20240115", mode="market_review"),
            make_record("s1", trade_date="20240116", mode="sector_deep_dive",
                        sector="银行"),
            make_record("q1", trade_date="20240117", mode="agent_query"),
        ])
        note = history_lens.get_history_note(mode="market_review")
        lines = note.splitlines()[1:]
        assert len(lines) == 1
        assert lines[0].startswith("01-15 ")

    def test_sector_and_mode_combined(self, archive_dir):
        write_jsonl(archive_dir / "archive_20240115.jsonl", [
            make_record("a", trade_date="20240115", mode="sector_deep_dive",
                        sector="银行", content="综合判断：偏多。"),
            make_record("b", trade_date="20240116", mode="market_review",
                        sector="银行", content="综合判断：偏多。"),
            make_record("c", trade_date="20240117", mode="sector_deep_dive",
                        sector="半导体", content="综合判断：偏多。"),
        ])
        note = history_lens.get_history_note(sector="银行", mode="sector_deep_dive")
        lines = note.splitlines()[1:]
        assert len(lines) == 1
        assert lines[0].startswith("01-15 ")

    def test_none_filters_match_all(self, archive_dir):
        write_jsonl(archive_dir / "archive_20240115.jsonl", [
            make_record("a", trade_date="20240115", sector="银行"),
            make_record("b", trade_date="20240116", sector=None),
        ])
        note = history_lens.get_history_note()
        assert len(note.splitlines()) == 3  # 头 + 2 条

    def test_limit_respected(self, archive_dir):
        recs = [make_record(f"r{i}", trade_date=f"202401{i:02d}") for i in range(10, 16)]
        write_jsonl(archive_dir / "archive_20240110.jsonl", recs[:3])
        write_jsonl(archive_dir / "archive_20240113.jsonl", recs[3:])

        note = history_lens.get_history_note(limit=2)
        lines = note.splitlines()[1:]
        assert len(lines) == 2
        assert lines[0].startswith("01-15 ")  # 最新两条
        assert lines[1].startswith("01-14 ")

    def test_limit_default_is_5(self, archive_dir):
        recs = [make_record(f"r{i}", trade_date=f"202401{i:02d}") for i in range(10, 18)]
        write_jsonl(archive_dir / "archive_20240110.jsonl", recs)
        note = history_lens.get_history_note()
        assert len(note.splitlines()) == 6  # 头 + 5 条

    def test_invalid_limit_falls_back_to_default(self, archive_dir):
        recs = [make_record(f"r{i}", trade_date=f"202401{i:02d}") for i in range(10, 18)]
        write_jsonl(archive_dir / "archive_20240110.jsonl", recs)
        for bad in (0, -3, "abc", None):
            note = history_lens.get_history_note(limit=bad)
            assert len(note.splitlines()) == 6  # 回退默认 5 条

    def test_same_trade_date_newer_ts_first(self, archive_dir):
        r1 = make_record("r1", trade_date="20240115", score="hit",
                         score_note="较早记录")
        r1["ts"] = "2024-01-15T20:00:00"
        r2 = make_record("r2", trade_date="20240115", score="hit",
                         score_note="较晚记录")
        r2["ts"] = "2024-01-15T22:00:00"
        write_jsonl(archive_dir / "archive_20240115.jsonl", [r1, r2])
        lines = history_lens.get_history_note().splitlines()[1:]
        # 同 trade_date 时 ts 较晚者排前
        assert len(lines) == 2
        assert "较晚记录" in lines[0]
        assert "较早记录" in lines[1]


# ─────────────────────────────────────────────
# 方向提取：三分支 + 优先范围/兜底/平局
# ─────────────────────────────────────────────

class TestDirectionExtraction:
    def test_bullish_branch(self):
        assert history_lens._extract_direction_label("综合判断：偏多。") == "偏多"

    def test_bearish_branch(self):
        assert history_lens._extract_direction_label("综合判断：偏空。") == "偏空"

    def test_neutral_branch_no_direction_words(self):
        assert history_lens._extract_direction_label("成交额放大，板块轮动。") == "中性"

    def test_empty_content_is_neutral(self):
        assert history_lens._extract_direction_label("") == "中性"
        assert history_lens._extract_direction_label(None) == "中性"

    def test_priority_marker_beats_full_text(self):
        content = "资金看好题材股。\n综合判断：短线谨慎，控制仓位。"
        assert history_lens._extract_direction_label(content) == "偏空"

    def test_direction_on_line_after_marker(self):
        assert history_lens._extract_direction_label("综合判断\n偏多思路。") == "偏多"

    def test_tie_is_neutral(self):
        assert history_lens._extract_direction_label("综合判断：谨慎乐观。") == "中性"

    def test_fallback_to_full_text(self):
        content = "综合判断：见下。\n市场结构分化。\n操作上看好低估值蓝筹。"
        assert history_lens._extract_direction_label(content) == "偏多"

    def test_direction_shown_in_note(self, archive_dir):
        write_jsonl(archive_dir / "archive_20240115.jsonl", [
            make_record("bull", trade_date="20240115", content="综合判断：偏多。"),
            make_record("bear", trade_date="20240116", content="综合判断：偏空。"),
            make_record("flat", trade_date="20240117", content="板块轮动明显。"),
        ])
        lines = history_lens.get_history_note().splitlines()[1:]
        # trade_date 倒序：flat(01-17) → bear(01-16) → bull(01-15)
        assert "中性" in lines[0]
        assert "偏空" in lines[1]
        assert "偏多" in lines[2]


# ─────────────────────────────────────────────
# 打分映射与依据截断
# ─────────────────────────────────────────────

class TestScoreMappingAndBasis:
    def test_score_labels(self, archive_dir):
        write_jsonl(archive_dir / "archive_20240115.jsonl", [
            make_record("h", trade_date="20240115", score="hit",
                        score_note="方向对"),
            make_record("m", trade_date="20240116", score="miss",
                        score_note="方向错"),
            make_record("n", trade_date="20240117", score="neutral",
                        score_note="震荡"),
            make_record("p", trade_date="20240118", score=None),
        ])
        lines = history_lens.get_history_note().splitlines()[1:]
        assert " 待评分" in lines[0]  # 01-18 score=None
        assert " 中性 " in lines[1]   # 01-17 neutral
        assert " 偏差 " in lines[2]   # 01-16 miss
        assert " 命中 " in lines[3]   # 01-15 hit

    def test_unknown_score_value_shows_pending(self, archive_dir):
        write_jsonl(archive_dir / "archive_20240115.jsonl",
                    [make_record("x", trade_date="20240115", score="weird")])
        assert "待评分" in history_lens.get_history_note()

    def test_basis_from_score_note(self, archive_dir):
        write_jsonl(archive_dir / "archive_20240115.jsonl", [
            make_record("h", trade_date="20240115", score="hit",
                        score_note="方向判断=bullish；实际区间涨跌幅=+2.50%"),
        ])
        line = history_lens.get_history_note().splitlines()[1]
        assert line == "01-15 偏多 命中 方向判断=bullish；实际区间涨跌幅=+2.50%"

    def test_basis_truncated_to_40_chars(self, archive_dir):
        long_note = "核对说明：" + "涨" * 60  # 远超 40 字
        write_jsonl(archive_dir / "archive_20240115.jsonl", [
            make_record("h", trade_date="20240115", score="hit",
                        score_note=long_note),
        ])
        line = history_lens.get_history_note().splitlines()[1]
        basis = line.split(" ", 3)[3]
        assert basis == long_note[:40]
        assert len(basis) == 40

    def test_no_score_note_omits_basis(self, archive_dir):
        write_jsonl(archive_dir / "archive_20240115.jsonl", [
            make_record("p", trade_date="20240115", score=None, score_note=None),
        ])
        line = history_lens.get_history_note().splitlines()[1]
        assert line == "01-15 偏多 待评分"

    def test_multiline_score_note_flattened(self, archive_dir):
        write_jsonl(archive_dir / "archive_20240115.jsonl", [
            make_record("h", trade_date="20240115", score="hit",
                        score_note="第一行\n第二行"),
        ])
        note = history_lens.get_history_note()
        assert len(note.splitlines()) == 2  # 注入块仍为两行，依据被单行化
        assert "第一行 第二行" in note.splitlines()[1]

    def test_bad_trade_date_shows_placeholder(self, archive_dir):
        rec = make_record("bad", trade_date="2024-01-15")
        write_jsonl(archive_dir / "archive_20240115.jsonl", [rec])
        line = history_lens.get_history_note().splitlines()[1]
        assert line.startswith("??-?? ")


# ─────────────────────────────────────────────
# 损坏行容错
# ─────────────────────────────────────────────

class TestCorruptedLines:
    def test_bad_json_lines_skipped(self, archive_dir):
        path = archive_dir / "archive_20240115.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(make_record("ok", trade_date="20240115"),
                               ensure_ascii=False) + "\n")
            f.write("{这不是合法JSON\n")
            f.write("[1, 2, 3]\n")  # 合法 JSON 但非对象
            f.write("\n")

        note = history_lens.get_history_note()
        assert note is not None
        assert len(note.splitlines()) == 2  # 头 + 唯一一条好记录

    def test_all_lines_bad_returns_none(self, archive_dir):
        (archive_dir / "archive_20240115.jsonl").write_text(
            "{坏行\n又一条坏行\n", encoding="utf-8")
        assert history_lens.get_history_note() is None


# ─────────────────────────────────────────────
# 300 字上限
# ─────────────────────────────────────────────

class TestMaxChars:
    def test_note_capped_at_300_chars(self, archive_dir):
        # 8 条记录，每条依据 40 字（截断后行长约 52 字）
        recs = [
            make_record(f"r{i}", trade_date=f"202401{i:02d}", score="hit",
                        score_note="核对说明" + "涨" * 50)
            for i in range(10, 18)
        ]
        write_jsonl(archive_dir / "archive_20240110.jsonl", recs)

        note = history_lens.get_history_note(limit=8)
        assert len(note) <= 300
        assert note.startswith("【以史为鉴：本智能体历史判断回顾】\n")
        # 头部 16 字 + 每行约 53 字 → 最多放 5 行
        assert len(note.splitlines()) == 6
        # 截断发生在整行边界：最后一行是完整记录行
        last = note.splitlines()[-1]
        assert last.startswith("01-")

    def test_short_notes_allow_more_lines_within_cap(self, archive_dir):
        recs = [make_record(f"r{i}", trade_date=f"202401{i:02d}") for i in range(10, 18)]
        write_jsonl(archive_dir / "archive_20240110.jsonl", recs)
        note = history_lens.get_history_note(limit=8)
        # 无依据时行很短，但 limit=8 封顶（且 300 字内 8 行可放下）
        assert len(note) <= 300
        assert len(note.splitlines()) == 9  # 头 + 8 条


# ─────────────────────────────────────────────
# 存档目录解析：ARCHIVE_DIR 优先，DATA_DIR 回退
# ─────────────────────────────────────────────

class TestArchiveDirResolution:
    def test_data_dir_fallback(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ARCHIVE_DIR", raising=False)
        data_dir = tmp_path / "mydata"
        (data_dir / "archive").mkdir(parents=True)
        monkeypatch.setenv("DATA_DIR", str(data_dir))
        write_jsonl(data_dir / "archive" / "archive_20240115.jsonl",
                    [make_record("r1", trade_date="20240115")])
        assert history_lens.get_history_note() is not None

    def test_archive_dir_env_wins(self, tmp_path, monkeypatch, archive_dir):
        # archive_dir fixture 已设 ARCHIVE_DIR；同时设 DATA_DIR 指空目录
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "空数据目录"))
        write_jsonl(archive_dir / "archive_20240115.jsonl",
                    [make_record("r1", trade_date="20240115")])
        assert history_lens.get_history_note() is not None

    def test_default_data_dir_used_when_env_missing(self, monkeypatch):
        monkeypatch.delenv("ARCHIVE_DIR", raising=False)
        monkeypatch.delenv("DATA_DIR", raising=False)
        assert history_lens._archive_dir() == os.path.join("data", "archive")


# ─────────────────────────────────────────────
# get_accuracy_summary：按 mode 分组汇总
# ─────────────────────────────────────────────

class TestAccuracySummary:
    def test_grouped_counts_and_hit_rate(self, archive_dir):
        write_jsonl(archive_dir / "archive_20240115.jsonl", [
            make_record("m1", trade_date="20240110", mode="market_review",
                        score="hit"),
            make_record("m2", trade_date="20240111", mode="market_review",
                        score="miss"),
            make_record("m3", trade_date="20240112", mode="market_review",
                        score="hit"),
            make_record("s1", trade_date="20240113", mode="sector_deep_dive",
                        sector="银行", score="neutral"),
            make_record("s2", trade_date="20240114", mode="sector_deep_dive",
                        sector="银行", score="hit"),
            make_record("p", trade_date="20240115", mode="market_review",
                        score=None),  # 待评分不计入
        ])
        summary = history_lens.get_accuracy_summary()

        total = summary["total"]
        assert total["hit"] == 3 and total["miss"] == 1 and total["neutral"] == 1
        assert total["scored"] == 5
        assert total["hit_rate"] == pytest.approx(0.6)

        by_mode = summary["by_mode"]
        assert set(by_mode) == {"market_review", "sector_deep_dive"}
        mr = by_mode["market_review"]
        assert mr["hit"] == 2 and mr["miss"] == 1 and mr["neutral"] == 0
        assert mr["scored"] == 3
        assert mr["hit_rate"] == pytest.approx(round(2 / 3, 4))
        sd = by_mode["sector_deep_dive"]
        assert sd["hit"] == 1 and sd["neutral"] == 1 and sd["miss"] == 0
        assert sd["scored"] == 2
        assert sd["hit_rate"] == pytest.approx(0.5)

    def test_unscored_records_excluded(self, archive_dir):
        write_jsonl(archive_dir / "archive_20240115.jsonl", [
            make_record("p1", trade_date="20240115", score=None),
            make_record("p2", trade_date="20240116", score=None),
        ])
        summary = history_lens.get_accuracy_summary()
        assert summary["total"]["scored"] == 0
        assert summary["total"]["hit_rate"] is None
        assert summary["by_mode"] == {}

    def test_empty_dir_returns_zero_structure(self, archive_dir):
        summary = history_lens.get_accuracy_summary()
        assert summary == {
            "total": {"hit": 0, "miss": 0, "neutral": 0,
                      "scored": 0, "hit_rate": None},
            "by_mode": {},
        }

    def test_missing_dir_returns_zero_structure(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ARCHIVE_DIR", str(tmp_path / "不存在"))
        summary = history_lens.get_accuracy_summary()
        assert summary["total"]["scored"] == 0
        assert summary["by_mode"] == {}

    def test_missing_mode_grouped_as_unknown(self, archive_dir):
        rec = make_record("x", trade_date="20240115", score="hit")
        del rec["mode"]
        write_jsonl(archive_dir / "archive_20240115.jsonl", [rec])
        summary = history_lens.get_accuracy_summary()
        assert summary["by_mode"]["unknown"]["hit"] == 1
        assert summary["by_mode"]["unknown"]["hit_rate"] == 1.0

    def test_bad_lines_do_not_break_summary(self, archive_dir):
        path = archive_dir / "archive_20240115.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(make_record("ok", trade_date="20240115",
                                           score="miss"), ensure_ascii=False) + "\n")
            f.write("{坏行\n")
        summary = history_lens.get_accuracy_summary()
        assert summary["total"]["miss"] == 1
        assert summary["total"]["hit_rate"] == 0.0

    def test_unknown_score_value_not_counted(self, archive_dir):
        write_jsonl(archive_dir / "archive_20240115.jsonl",
                    [make_record("x", trade_date="20240115", score="weird")])
        summary = history_lens.get_accuracy_summary()
        assert summary["total"]["scored"] == 0
