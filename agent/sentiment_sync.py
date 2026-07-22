"""本地 → Railway 舆情快照同步客户端。

本地 Mac 每日抓取脚本跑完舆情分布后，快照落在本地 social.db 的
sentiment_snapshots 表（主键 platform+target+date）。本模块负责把当天
快照批量 POST 到 Railway 端 /v1/admin/sentiment/snapshots。

设计原则：三个公开函数（collect_snapshots_for_date / push_snapshots /
sync_today）在任何异常路径下都不抛出，统一返回带 note 说明的 dict，
保证每日脚本在 cron 环境里静默容错、只留日志。
"""

from __future__ import annotations

import os
import sqlite3
from datetime import date as _date

import requests

DEFAULT_BASE_URL = "https://market-review-agent-production.up.railway.app"
SNAPSHOTS_ENDPOINT = "/v1/admin/sentiment/snapshots"
BATCH_SIZE = 100
SNAPSHOT_COLUMNS = ("platform", "target", "date", "n", "pos", "neu", "neg",
                    "w_pos", "w_neu", "w_neg")


def _resolve_db_path(db_path: str | None) -> str:
    """db_path 缺省走 env SOCIAL_DB_PATH > ${DATA_DIR:-data}/social.db。"""
    if db_path:
        return db_path
    env_path = os.environ.get("SOCIAL_DB_PATH")
    if env_path:
        return env_path
    data_dir = os.environ.get("DATA_DIR") or "data"
    return os.path.join(data_dir, "social.db")


def collect_snapshots_for_date(date_str: str, db_path: str | None = None) -> list[dict]:
    """读取指定日期的全部快照行，转为 dict 列表（剔除 created_at）。

    库文件不存在、表不存在或任何读取异常均返回 []，绝不抛异常。
    """
    try:
        path = _resolve_db_path(db_path)
        if not os.path.exists(path):
            return []
        conn = sqlite3.connect(path)
        try:
            cols = ",".join(SNAPSHOT_COLUMNS)
            rows = conn.execute(
                f"SELECT {cols} FROM sentiment_snapshots WHERE date=?",
                (date_str,),
            ).fetchall()
        finally:
            conn.close()
        return [dict(zip(SNAPSHOT_COLUMNS, row)) for row in rows]
    except Exception:
        return []


def _post_batch(url: str, headers: dict, batch: list[dict], timeout: int) -> dict:
    """推送单个批次，返回该批的标准化结果 dict（内部函数，不对外）。"""
    resp = requests.post(
        url,
        json={"snapshots": batch},
        headers=headers,
        timeout=timeout,
    )
    body: dict = {}
    try:
        parsed = resp.json()
        if isinstance(parsed, dict):
            body = parsed
    except Exception:
        body = {}
    if 200 <= resp.status_code < 300 and body.get("ok"):
        return {"ok": True, "saved": int(body.get("saved") or len(batch)),
                "note": ""}
    failed = body.get("failed")
    note = f"HTTP {resp.status_code}"
    if failed:
        note += f" failed={len(failed) if isinstance(failed, list) else failed}"
    return {"ok": False, "saved": int(body.get("saved") or 0), "note": note}


def push_snapshots(base_url: str, api_key: str, snapshots: list[dict],
                   timeout: int = 30) -> dict:
    """批量推送快照（每批最多 100 条），5xx/网络异常自动重试一次。

    返回 {"ok": bool, "saved": int, "attempted": int, "note": str}，
    任何异常路径都不抛出。
    """
    attempted = len(snapshots)
    if not snapshots:
        return {"ok": True, "saved": 0, "attempted": 0, "note": "无快照无需推送"}
    if not base_url:
        return {"ok": False, "saved": 0, "attempted": attempted,
                "note": "缺少 base_url"}
    if not api_key:
        return {"ok": False, "saved": 0, "attempted": attempted,
                "note": "缺少 api_key"}

    url = base_url.rstrip("/") + SNAPSHOTS_ENDPOINT
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}

    total_saved = 0
    notes: list[str] = []
    all_ok = True
    for start in range(0, attempted, BATCH_SIZE):
        batch = snapshots[start:start + BATCH_SIZE]
        result = None
        for attempt in range(2):  # 首次 + 5xx/网络异常重试一次
            try:
                result = _post_batch(url, headers, batch, timeout)
                if result["ok"]:
                    break
                # 4xx 等客户端错误不重试；5xx 重试一次
                if "HTTP 5" not in result["note"] or attempt == 1:
                    break
            except requests.RequestException as exc:
                result = {"ok": False, "saved": 0,
                          "note": f"网络异常: {type(exc).__name__}"}
                if attempt == 1:
                    break
            except Exception as exc:  # 兜底，绝不抛
                result = {"ok": False, "saved": 0,
                          "note": f"未知异常: {type(exc).__name__}"}
                break
        if result is None:  # 理论不可达，兜底
            result = {"ok": False, "saved": 0, "note": "未知异常"}
        if result["ok"]:
            total_saved += result["saved"]
        else:
            all_ok = False
            notes.append(f"批次{start // BATCH_SIZE + 1}: {result['note']}")

    note = "; ".join(notes) if notes else "全部批次推送成功"
    return {"ok": all_ok, "saved": total_saved, "attempted": attempted,
            "note": note}


def sync_today(base_url: str | None = None, api_key: str | None = None,
               db_path: str | None = None, date_str: str | None = None) -> dict:
    """编排：读取当天本地快照 → 推送到 Railway。任何路径都不抛异常。"""
    base_url = base_url or os.environ.get("SENTIMENT_SYNC_URL") or DEFAULT_BASE_URL
    api_key = api_key or os.environ.get("AGENT_API_KEY") or ""
    date_str = date_str or _date.today().isoformat()

    if not api_key:
        return {"ok": False, "saved": 0, "attempted": 0,
                "note": "缺少 AGENT_API_KEY，无法推送快照"}

    snapshots = collect_snapshots_for_date(date_str, db_path)
    if not snapshots:
        return {"ok": True, "saved": 0, "attempted": 0,
                "note": "无快照无需同步"}
    return push_snapshots(base_url, api_key, snapshots)
