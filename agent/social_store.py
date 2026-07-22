"""agent/social_store.py 社媒舆情帖子 SQLite 持久化（微舆 InsightEngine 轻量版）。

职责：
1. 统一 Post 字典（见 agent/social_media.py 模块 docstring 的全局契约）的
   本地落盘：表 social_posts，主键 (platform, post_id)。
2. 幂等 upsert：重复命中时刷新正文快照 + last_seen 更新 + hit_count 自增，
   用于衡量同一热点的持续在榜时长/复现频次。
3. 多维度查询：按平台 / 关键词（title+content LIKE）/ 最近 N 天（last_seen）
   / 条数上限过滤。

路径解析（惰性，调用时才求值）：
    db_path 参数 > 环境变量 SOCIAL_DB_PATH > ${DATA_DIR:-data}/social.db

合规注记（与社媒爬取层整体一致，端点 2026-07-22 实测定案）：
    本模块只做本地存储，不触网；入库数据应来自公开端点、低频访问、
    无登录自动化的采集链路。平台条款变动可能导致上游失效，需按
    warning 日志观测。

铁律：所有公开函数任何异常记 warning 并返回安全值（False/0/[]），绝不抛出。
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)

DB_PATH_ENV = "SOCIAL_DB_PATH"
DATA_DIR_ENV = "DATA_DIR"
DEFAULT_DATA_DIR = "data"
DB_FILENAME = "social.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS social_posts (
    platform      TEXT NOT NULL,
    post_id       TEXT NOT NULL,
    title         TEXT NOT NULL DEFAULT '',
    content       TEXT NOT NULL DEFAULT '',
    author        TEXT NOT NULL DEFAULT '',
    metrics_json  TEXT NOT NULL DEFAULT '{}',
    url           TEXT NOT NULL DEFAULT '',
    published_at  TEXT NOT NULL DEFAULT '',
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    hit_count     INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (platform, post_id)
)
"""


def _resolve_db_path(db_path: Optional[str] = None) -> str:
    """惰性解析数据库路径：参数 > SOCIAL_DB_PATH > ${DATA_DIR:-data}/social.db。"""
    if isinstance(db_path, str) and db_path.strip():
        return db_path.strip()
    env = os.getenv(DB_PATH_ENV)
    if env and env.strip():
        return env.strip()
    data_dir = os.getenv(DATA_DIR_ENV) or DEFAULT_DATA_DIR
    return os.path.join(data_dir, DB_FILENAME)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db(db_path: Optional[str] = None) -> bool:
    """建库建表（幂等）。成功 True；任何异常记 warning 返回 False。"""
    try:
        path = _resolve_db_path(db_path)
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.execute(_SCHEMA_SQL)
        return True
    except Exception as e:
        logger.warning("social_store 初始化失败 path=%r（降级，不落盘）: %s",
                       db_path, e, exc_info=True)
        return False


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    try:
        metrics = json.loads(d.get("metrics_json") or "{}")
        d["metrics"] = metrics if isinstance(metrics, dict) else {}
    except Exception:
        d["metrics"] = {}
    d.pop("metrics_json", None)
    return d


def upsert_posts(posts: List[dict], db_path: Optional[str] = None) -> int:
    """批量幂等写入 Post 字典，返回成功处理条数（失败 0，绝不抛）。

    - 缺 platform/post_id 或非 dict 的条目跳过（不计数）。
    - 已存在的 (platform, post_id)：刷新 title/content/author/metrics/url/
      published_at 快照，last_seen=now，hit_count+1；first_seen 保持不变。
    - metrics 只存 dict（其他类型按 {} 落盘），JSON 序列化失败按 {} 落盘。
    """
    try:
        path = _resolve_db_path(db_path)
        if not init_db(path):
            return 0
        now = _now_iso()
        n = 0
        with sqlite3.connect(path) as conn:
            for p in posts or []:
                if not isinstance(p, dict):
                    continue
                platform = str(p.get("platform") or "").strip()
                post_id = str(p.get("post_id") or "").strip()
                if not platform or not post_id:
                    continue
                metrics = p.get("metrics")
                try:
                    metrics_json = (json.dumps(metrics, ensure_ascii=False)
                                    if isinstance(metrics, dict) else "{}")
                except Exception:
                    metrics_json = "{}"
                title = str(p.get("title") or "")
                content = str(p.get("content") or "")
                author = str(p.get("author") or "")
                url = str(p.get("url") or "")
                published_at = str(p.get("published_at") or "")
                row = conn.execute(
                    "SELECT hit_count FROM social_posts "
                    "WHERE platform = ? AND post_id = ?",
                    (platform, post_id),
                ).fetchone()
                if row:
                    conn.execute(
                        "UPDATE social_posts SET title = ?, content = ?, "
                        "author = ?, metrics_json = ?, url = ?, "
                        "published_at = ?, last_seen = ?, "
                        "hit_count = hit_count + 1 "
                        "WHERE platform = ? AND post_id = ?",
                        (title, content, author, metrics_json, url,
                         published_at, now, platform, post_id),
                    )
                else:
                    conn.execute(
                        "INSERT INTO social_posts (platform, post_id, title, "
                        "content, author, metrics_json, url, published_at, "
                        "first_seen, last_seen, hit_count) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                        (platform, post_id, title, content, author,
                         metrics_json, url, published_at, now, now),
                    )
                n += 1
        return n
    except Exception as e:
        logger.warning("social_store upsert 失败（返回 0）: %s", e, exc_info=True)
        return 0


def query_posts(platform: Optional[str] = None,
                keyword: Optional[str] = None,
                days: Optional[int] = 7,
                limit: int = 100,
                db_path: Optional[str] = None) -> List[dict]:
    """查询已存帖子，返回 dict 列表（metrics 已解析回 dict）。失败 []，绝不抛。

    - platform：精确匹配；None/空串不过滤。
    - keyword：对 title+content 做 LIKE %kw% 过滤；None/空串不过滤。
    - days：仅保留 last_seen 在最近 N 天内的记录；<=0 或 None 不做时间过滤。
    - limit：返回上限（按 last_seen 倒序），非法值回退 100。
    """
    try:
        path = _resolve_db_path(db_path)
        if not init_db(path):
            return []
        clauses, params = [], []
        if platform and str(platform).strip():
            clauses.append("platform = ?")
            params.append(str(platform).strip())
        if keyword and str(keyword).strip():
            clauses.append("(title LIKE ? OR content LIKE ?)")
            like = f"%{str(keyword).strip()}%"
            params.extend([like, like])
        try:
            days_int = int(days) if days is not None else 0
        except (TypeError, ValueError):
            days_int = 0
        if days_int > 0:
            cutoff = (datetime.now(timezone.utc)
                      - timedelta(days=days_int)).isoformat(timespec="seconds")
            clauses.append("last_seen >= ?")
            params.append(cutoff)
        try:
            limit_int = int(limit)
            if limit_int <= 0:
                limit_int = 100
        except (TypeError, ValueError):
            limit_int = 100
        sql = "SELECT * FROM social_posts"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY last_seen DESC LIMIT ?"
        params.append(limit_int)
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]
    except Exception as e:
        logger.warning("social_store 查询失败（返回 []）: %s", e, exc_info=True)
        return []
