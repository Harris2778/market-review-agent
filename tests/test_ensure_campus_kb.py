"""scripts/ensure_campus_kb.py 单测：生产启动期校园知识库幂等落位。"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import ensure_campus_kb as m  # noqa: E402

PAYLOAD = b"SQLite format 3\x00" + b"campus-kb" * 4096  # 任意非空内容


@pytest.fixture()
def asset(tmp_path):
    p = tmp_path / "campus_kb.db.gz"
    with gzip.open(p, "wb") as f:
        f.write(PAYLOAD)
    return p


def test_restores_db_when_missing(tmp_path, asset):
    data_dir = tmp_path / "data"
    status, detail = m.ensure_campus_kb(data_dir, asset)
    assert status == "restored"
    target = data_dir / "campus_kb.db"
    assert target.read_bytes() == PAYLOAD
    assert str(target) == detail
    # 解压临时文件无残留
    assert list(data_dir.glob(".campus_kb.*")) == []


def test_idempotent_keeps_existing_db(tmp_path, asset):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    target = data_dir / "campus_kb.db"
    target.write_bytes(b"user-upgraded-db")
    status, _ = m.ensure_campus_kb(data_dir, asset)
    assert status == "ready"
    assert target.read_bytes() == b"user-upgraded-db"  # 不被快照覆盖


def test_skipped_when_asset_missing(tmp_path):
    status, detail = m.ensure_campus_kb(tmp_path / "data", tmp_path / "nope.gz")
    assert status == "skipped"
    assert "asset missing" in detail
    assert not (tmp_path / "data" / "campus_kb.db").exists()


def test_main_returns_zero_on_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CAMPUS_KB_ASSET", str(tmp_path / "missing.gz"))
    assert m.main() == 0
    assert "skipped" in capsys.readouterr().out
