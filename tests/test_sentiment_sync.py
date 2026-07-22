"""agent.sentiment_sync 单元测试（全 mock，零网络）。

- collect_snapshots_for_date：用 tmp sqlite 真库验证读取与容错。
- push_snapshots / sync_today：requests.post 全 mock，覆盖成功、分批、
  重试、4xx 不重试、网络异常、缺 key、空快照等路径。
"""

import os
import sqlite3
from unittest.mock import MagicMock, patch

import pytest
import requests

from agent import sentiment_sync
from agent.sentiment_sync import (
    collect_snapshots_for_date,
    push_snapshots,
    sync_today,
)

COLUMNS = ("platform", "target", "date", "n", "pos", "neu", "neg",
           "w_pos", "w_neu", "w_neg", "created_at")


def _make_db(path, rows):
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE sentiment_snapshots ("
        "platform TEXT, target TEXT, date TEXT, n INTEGER, "
        "pos REAL, neu REAL, neg REAL, "
        "w_pos REAL, w_neu REAL, w_neg REAL, created_at TEXT, "
        "PRIMARY KEY (platform, target, date))"
    )
    conn.executemany(
        "INSERT INTO sentiment_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


def _row(platform="weibo", target="600519", date="2025-01-15", n=10,
         pos=0.5, neu=0.3, neg=0.2, w_pos=0.4, w_neu=0.35, w_neg=0.25,
         created_at="2025-01-15T20:00:00"):
    return (platform, target, date, n, pos, neu, neg,
            w_pos, w_neu, w_neg, created_at)


def _ok_response(saved):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"ok": True, "saved": saved, "failed": []}
    return resp


def _err_response(status, body=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body if body is not None else {"ok": False,
                                                            "failed": [{}]}
    return resp


# ── collect_snapshots_for_date ─────────────────────────────────────────

class TestCollect:
    def test_reads_rows_for_date(self, tmp_path):
        db = tmp_path / "social.db"
        _make_db(db, [_row(date="2025-01-15"),
                      _row(platform="zhihu", date="2025-01-15"),
                      _row(date="2025-01-14")])  # 其他日期不应读出
        result = collect_snapshots_for_date("2025-01-15", str(db))
        assert len(result) == 2
        for item in result:
            assert set(item.keys()) == set(COLUMNS) - {"created_at"}
            assert item["date"] == "2025-01-15"
            assert item["n"] == 10

    def test_db_file_missing_returns_empty(self, tmp_path):
        assert collect_snapshots_for_date(
            "2025-01-15", str(tmp_path / "nope.db")) == []

    def test_table_missing_returns_empty(self, tmp_path):
        db = tmp_path / "social.db"
        sqlite3.connect(str(db)).close()  # 空库，无表
        assert collect_snapshots_for_date("2025-01-15", str(db)) == []

    def test_db_path_env_fallback(self, tmp_path, monkeypatch):
        db = tmp_path / "env.db"
        _make_db(db, [_row(date="2025-01-15")])
        monkeypatch.setenv("SOCIAL_DB_PATH", str(db))
        assert len(collect_snapshots_for_date("2025-01-15")) == 1

    def test_db_path_data_dir_fallback(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "datadir"
        data_dir.mkdir()
        _make_db(data_dir / "social.db", [_row(date="2025-01-15")])
        monkeypatch.delenv("SOCIAL_DB_PATH", raising=False)
        monkeypatch.setenv("DATA_DIR", str(data_dir))
        assert len(collect_snapshots_for_date("2025-01-15")) == 1


# ── push_snapshots ─────────────────────────────────────────────────────

class TestPush:
    def test_success_single_batch(self):
        snaps = [{"platform": "weibo", "target": "t", "date": "d"}] * 3
        with patch("agent.sentiment_sync.requests.post",
                   return_value=_ok_response(3)) as post:
            result = push_snapshots("https://x.example/", "KEY", snaps)
        assert result == {"ok": True, "saved": 3, "attempted": 3,
                          "note": "全部批次推送成功"}
        # 尾斜杠容错 + Bearer 头
        args, kwargs = post.call_args
        assert args[0] == "https://x.example/v1/admin/sentiment/snapshots"
        assert kwargs["headers"]["Authorization"] == "Bearer KEY"
        assert kwargs["json"] == {"snapshots": snaps}

    def test_empty_snapshots(self):
        with patch("agent.sentiment_sync.requests.post") as post:
            result = push_snapshots("https://x.example", "KEY", [])
        assert result["ok"] is True
        assert result["attempted"] == 0
        post.assert_not_called()

    def test_missing_api_key(self):
        with patch("agent.sentiment_sync.requests.post") as post:
            result = push_snapshots("https://x.example", "", [{"a": 1}])
        assert result["ok"] is False
        assert "api_key" in result["note"]
        post.assert_not_called()

    def test_batching_over_100(self):
        snaps = [{"i": i} for i in range(250)]
        with patch("agent.sentiment_sync.requests.post",
                   return_value=_ok_response(100)) as post:
            result = push_snapshots("https://x.example", "KEY", snaps)
        assert post.call_count == 3  # 100 + 100 + 50
        sizes = [len(c.kwargs["json"]["snapshots"])
                 for c in post.call_args_list]
        assert sizes == [100, 100, 50]
        assert result["ok"] is True
        assert result["saved"] == 300
        assert result["attempted"] == 250

    def test_5xx_retried_once(self):
        snaps = [{"a": 1}]
        with patch("agent.sentiment_sync.requests.post",
                   side_effect=[_err_response(500), _ok_response(1)]) as post:
            result = push_snapshots("https://x.example", "KEY", snaps)
        assert post.call_count == 2
        assert result["ok"] is True
        assert result["saved"] == 1

    def test_5xx_persistent_fails_after_one_retry(self):
        with patch("agent.sentiment_sync.requests.post",
                   side_effect=[_err_response(500), _err_response(502)]) as post:
            result = push_snapshots("https://x.example", "KEY", [{"a": 1}])
        assert post.call_count == 2  # 只重试一次
        assert result["ok"] is False
        assert result["saved"] == 0
        assert "502" in result["note"]

    def test_4xx_not_retried(self):
        with patch("agent.sentiment_sync.requests.post",
                   return_value=_err_response(401)) as post:
            result = push_snapshots("https://x.example", "KEY", [{"a": 1}])
        assert post.call_count == 1
        assert result["ok"] is False
        assert "401" in result["note"]

    def test_network_exception_retried_once(self):
        with patch("agent.sentiment_sync.requests.post",
                   side_effect=[requests.ConnectionError("boom"),
                                _ok_response(2)]) as post:
            result = push_snapshots("https://x.example", "KEY",
                                    [{"a": 1}, {"a": 2}])
        assert post.call_count == 2
        assert result["ok"] is True
        assert result["saved"] == 2

    def test_network_exception_persistent_no_raise(self):
        with patch("agent.sentiment_sync.requests.post",
                   side_effect=requests.Timeout("slow")) as post:
            result = push_snapshots("https://x.example", "KEY", [{"a": 1}])
        assert post.call_count == 2
        assert result["ok"] is False
        assert "网络异常" in result["note"]


# ── sync_today ─────────────────────────────────────────────────────────

class TestSyncToday:
    def test_no_snapshots_returns_ok(self, tmp_path):
        db = tmp_path / "empty.db"
        with patch("agent.sentiment_sync.requests.post") as post:
            result = sync_today(api_key="KEY", db_path=str(db),
                                date_str="2025-01-15")
        assert result["ok"] is True
        assert result["note"] == "无快照无需同步"
        post.assert_not_called()

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("AGENT_API_KEY", raising=False)
        result = sync_today(base_url="https://x.example", api_key=None,
                            db_path="/nonexistent.db", date_str="2025-01-15")
        assert result["ok"] is False
        assert "AGENT_API_KEY" in result["note"]

    def test_orchestrates_collect_and_push(self, tmp_path):
        db = tmp_path / "social.db"
        _make_db(db, [_row(date="2025-01-15"),
                      _row(platform="zhihu", date="2025-01-15")])
        with patch("agent.sentiment_sync.requests.post",
                   return_value=_ok_response(2)) as post:
            result = sync_today(base_url="https://x.example", api_key="KEY",
                                db_path=str(db), date_str="2025-01-15")
        assert result["ok"] is True
        assert result["saved"] == 2
        payload = post.call_args.kwargs["json"]["snapshots"]
        assert len(payload) == 2
        assert all("created_at" not in s for s in payload)

    def test_env_fallbacks(self, tmp_path, monkeypatch):
        db = tmp_path / "social.db"
        _make_db(db, [_row(date="2025-01-15")])
        monkeypatch.setenv("SENTIMENT_SYNC_URL", "https://env.example/")
        monkeypatch.setenv("AGENT_API_KEY", "ENV_KEY")
        with patch("agent.sentiment_sync.requests.post",
                   return_value=_ok_response(1)) as post:
            result = sync_today(db_path=str(db), date_str="2025-01-15")
        assert result["ok"] is True
        args, kwargs = post.call_args
        assert args[0].startswith("https://env.example/v1/admin/")
        assert kwargs["headers"]["Authorization"] == "Bearer ENV_KEY"

    def test_default_base_url(self, tmp_path, monkeypatch):
        db = tmp_path / "social.db"
        _make_db(db, [_row(date="2025-01-15")])
        monkeypatch.delenv("SENTIMENT_SYNC_URL", raising=False)
        with patch("agent.sentiment_sync.requests.post",
                   return_value=_ok_response(1)) as post:
            result = sync_today(api_key="KEY", db_path=str(db),
                                date_str="2025-01-15")
        assert result["ok"] is True
        assert post.call_args.args[0].startswith(
            "https://market-review-agent-production.up.railway.app")

    def test_date_defaults_to_today(self, tmp_path):
        db = tmp_path / "social.db"
        _make_db(db, [])
        with patch("agent.sentiment_sync._date") as mock_date:
            mock_date.today.return_value.isoformat.return_value = "2025-01-15"
            result = sync_today(api_key="KEY", db_path=str(db))
        assert result["ok"] is True
        assert result["note"] == "无快照无需同步"
