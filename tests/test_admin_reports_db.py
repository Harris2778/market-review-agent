"""研报库管理端点测试（POST /v1/admin/reports-db + GET /v1/admin/reports-db/info）。"""
import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from agent.report_library import init_db, upsert_reports

ADMIN = {"Authorization": "Bearer test-agent-api-key"}


def _make_db_bytes(tmp_path):
    db = tmp_path / "upload_src.db"
    init_db(str(db))
    upsert_reports([{
        "info_code": "X_1", "org": "东吴证券", "author": "张三",
        "publish_date": "2026-07-20", "stock_code": "000001", "stock_name": "平安银行",
        "industry": "银行", "title": "测试研报", "rating": "买入",
        "target_price_low": 12.0, "target_price_high": 15.0,
        "encode_url": "http://x", "source": "eastmoney",
    }], db_path=str(db))
    return db.read_bytes()


@pytest.fixture()
def client():
    import main
    return TestClient(main.app)


@pytest.fixture()
def db_env(tmp_path, monkeypatch):
    target = tmp_path / "live" / "reports.db"
    monkeypatch.setenv("REPORTS_DB_PATH", str(target))
    return target


# ---------- POST /v1/admin/reports-db ----------

def test_upload_requires_auth(client):
    r = client.post("/v1/admin/reports-db", content=b"x")
    assert r.status_code == 401
    r = client.post("/v1/admin/reports-db", content=b"x",
                    headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_upload_rejects_empty(client, db_env):
    r = client.post("/v1/admin/reports-db", content=b"", headers=ADMIN)
    assert r.status_code == 400
    assert "空" in r.json()["detail"]


def test_upload_rejects_non_sqlite(client, db_env):
    r = client.post("/v1/admin/reports-db", content=b"not a sqlite file at all",
                    headers=ADMIN)
    assert r.status_code == 400
    assert "SQLite" in r.json()["detail"]
    assert not db_env.exists()


def test_upload_rejects_oversize(client, db_env, monkeypatch):
    import main
    monkeypatch.setattr(main, "_REPORTS_DB_MAX_BYTES", 10)
    r = client.post("/v1/admin/reports-db", content=b"x" * 64, headers=ADMIN)
    assert r.status_code == 413


def test_upload_writes_db_and_replaces_old(client, db_env, tmp_path):
    payload = _make_db_bytes(tmp_path)
    r = client.post("/v1/admin/reports-db", content=payload, headers=ADMIN)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and body["size_bytes"] == len(payload)
    assert db_env.read_bytes() == payload
    # 再次上传应原子替换
    r2 = client.post("/v1/admin/reports-db", content=payload, headers=ADMIN)
    assert r2.status_code == 200
    assert db_env.read_bytes() == payload


def test_upload_then_stats_readable(client, db_env, tmp_path):
    import sqlite3
    payload = _make_db_bytes(tmp_path)
    client.post("/v1/admin/reports-db", content=payload, headers=ADMIN)
    conn = sqlite3.connect(f"file:{db_env}?mode=ro", uri=True)
    try:
        n = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        latest = conn.execute("SELECT MAX(publish_date) FROM reports").fetchone()[0]
    finally:
        conn.close()
    assert n == 1
    assert latest == "2026-07-20"


# ---------- GET /v1/admin/reports-db/info ----------

def test_info_requires_auth(client):
    assert client.get("/v1/admin/reports-db/info").status_code == 401


def test_info_when_missing(client, db_env):
    r = client.get("/v1/admin/reports-db/info", headers=ADMIN)
    assert r.status_code == 200
    body = r.json()
    assert body["exists"] is False
    assert body.get("total_reports", 0) == 0
    assert body["path"] == str(db_env)


def test_info_after_upload(client, db_env, tmp_path):
    payload = _make_db_bytes(tmp_path)
    client.post("/v1/admin/reports-db", content=payload, headers=ADMIN)
    r = client.get("/v1/admin/reports-db/info", headers=ADMIN)
    assert r.status_code == 200
    body = r.json()
    assert body["exists"] is True
    assert body["total_reports"] == 1
    assert body["latest_publish_date"] == "2026-07-20"
    assert body["size_bytes"] == len(payload)
