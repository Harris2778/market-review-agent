"""舆情快照接收端点测试（POST /v1/admin/sentiment/snapshots）。

SOCIAL_DB_PATH 指向 tmp 路径真写库，验证幂等 REPLACE 与落盘数值；
全程本地 SQLite，断网可跑。
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

ADMIN = {"Authorization": "Bearer test-agent-api-key"}
URL = "/v1/admin/sentiment/snapshots"


def _snap(platform="xueqiu", target="sh600519", date="2026-07-21", n=120):
    return {
        "platform": platform,
        "target": target,
        "date": date,
        "n": n,
        "pos": 55.0,
        "neu": 25.0,
        "neg": 20.0,
        "w_pos": 61.5,
        "w_neu": 20.5,
        "w_neg": 18.0,
    }


def _rows(db_path):
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(
            "SELECT * FROM sentiment_snapshots ORDER BY platform, target, date"
        ).fetchall()]


@pytest.fixture()
def client():
    import main
    return TestClient(main.app)


@pytest.fixture()
def db_env(tmp_path, monkeypatch):
    target = tmp_path / "live" / "social.db"
    monkeypatch.setenv("SOCIAL_DB_PATH", str(target))
    return target


# ---------- 鉴权 ----------

def test_requires_auth(client):
    assert client.post(URL, json={"snapshots": []}).status_code == 401
    r = client.post(URL, json={"snapshots": []},
                    headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


# ---------- 正常批量写入 ----------

def test_batch_insert_writes_rows(client, db_env):
    snaps = [_snap(target="sh600519"), _snap(target="sz300750", date="2026-07-21"),
             _snap(platform="weibo", target="sh600519")]
    r = client.post(URL, json={"snapshots": snaps}, headers=ADMIN)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["saved"] == 3
    assert body["failed"] == []
    rows = _rows(db_env)
    assert len(rows) == 3
    row = next(x for x in rows if x["platform"] == "xueqiu" and x["target"] == "sh600519")
    assert row["n"] == 120
    assert row["pos"] == 55.0
    assert row["w_pos"] == 61.5
    assert row["date"] == "2026-07-21"


def test_numeric_strings_coerced(client, db_env):
    snap = _snap()
    snap["n"] = "88"
    snap["pos"] = "55.5"
    r = client.post(URL, json={"snapshots": [snap]}, headers=ADMIN)
    assert r.status_code == 200
    assert r.json()["saved"] == 1
    row = _rows(db_env)[0]
    assert row["n"] == 88
    assert row["pos"] == 55.5


# ---------- 幂等重复推送（REPLACE） ----------

def test_idempotent_replace(client, db_env):
    r1 = client.post(URL, json={"snapshots": [_snap()]}, headers=ADMIN)
    assert r1.status_code == 200 and r1.json()["saved"] == 1
    updated = _snap()
    updated.update({"n": 200, "pos": 70.0})
    r2 = client.post(URL, json={"snapshots": [updated]}, headers=ADMIN)
    assert r2.status_code == 200 and r2.json()["saved"] == 1
    rows = _rows(db_env)
    assert len(rows) == 1  # 主键 platform+target+date 幂等 REPLACE，不产生重复行
    assert rows[0]["n"] == 200
    assert rows[0]["pos"] == 70.0


# ---------- 非法条目混入不中断 ----------

def test_invalid_items_mixed(client, db_env):
    snaps = [
        _snap(target="ok1"),                                    # 合法
        {"target": "x", "date": "2026-07-21"},                  # 缺 platform
        dict(_snap(target="bad_date"), date=20260721),          # date 非字符串
        dict(_snap(target="bad_float"), pos="not-a-number"),    # pos 无法转 float
        dict(_snap(target="bad_bool"), neg=True),               # bool 判非法
        "not-a-dict",                                           # 非对象
        _snap(target="ok2"),                                    # 合法
    ]
    r = client.post(URL, json={"snapshots": snaps}, headers=ADMIN)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["saved"] == 2
    assert [f["index"] for f in body["failed"]] == [1, 2, 3, 4, 5]
    assert all(f["reason"] for f in body["failed"])
    assert "platform" in body["failed"][0]["reason"]
    rows = _rows(db_env)
    assert sorted(x["target"] for x in rows) == ["ok1", "ok2"]


# ---------- 空列表合法 ----------

def test_empty_list_ok(client, db_env):
    r = client.post(URL, json={"snapshots": []}, headers=ADMIN)
    assert r.status_code == 200
    body = r.json()
    assert body == {"ok": True, "saved": 0, "failed": []}
    assert not db_env.exists()  # 无任何写入


# ---------- 超上限整批拒绝 ----------

def test_over_limit_rejected(client, db_env):
    snaps = [_snap(target=f"t{i}") for i in range(501)]
    r = client.post(URL, json={"snapshots": snaps}, headers=ADMIN)
    assert r.status_code == 400
    assert "500" in r.json()["detail"]
    assert not db_env.exists()  # 一条都不写


def test_at_limit_accepted(client, db_env):
    snaps = [_snap(target=f"t{i}") for i in range(500)]
    r = client.post(URL, json={"snapshots": snaps}, headers=ADMIN)
    assert r.status_code == 200
    assert r.json()["saved"] == 500
    assert len(_rows(db_env)) == 500


# ---------- 请求体结构错误 ----------

def test_missing_snapshots_key(client, db_env):
    r = client.post(URL, json={"foo": []}, headers=ADMIN)
    assert r.status_code == 400


# ---------- 未预期异常兜底 200 + failed，绝不 500 ----------

def test_unexpected_exception_falls_back_to_200(client, db_env, monkeypatch):
    import main

    def _boom(snapshot):
        raise RuntimeError("simulated disk meltdown")

    monkeypatch.setattr(main, "_save_snapshot", _boom)
    r = client.post(URL, json={"snapshots": [_snap()]}, headers=ADMIN)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["saved"] == 0
    assert body["failed"][0]["index"] == 0
    assert "disk meltdown" in body["failed"][0]["reason"]


def test_module_not_loaded_falls_back_to_200(client, db_env, monkeypatch):
    import main
    monkeypatch.setattr(main, "_save_snapshot", None)
    r = client.post(URL, json={"snapshots": [_snap()]}, headers=ADMIN)
    assert r.status_code == 200
    body = r.json()
    assert body["saved"] == 0
    assert len(body["failed"]) == 1
