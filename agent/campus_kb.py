"""agent/campus_kb.py 校园知识库存储与中文检索层（校园知识库 v1）。

职责：
1. 清华校园知识条目（选课手册/课程信息/课程评价/课程总结/书籍笔记）的
   SQLite 持久化（纯 stdlib sqlite3，单文件库）。
2. 中文全文检索：优先 FTS5 虚表（trigram tokenizer，支持中文子串匹配，
   运行时探测编译支持，探测结果模块级缓存）；不支持或关键词过短时
   自动降级 LIKE '%kw%' 多关键词（AND 语义）检索，对调用方透明。
3. 检索排序：FTS5 用 bm25（标题列加权），LIKE 用命中次数启发式
   （标题命中权重高于正文），每条结果附 score 字段（越大越相关）。

存储契约（与全局契约一致）：
- 路径解析：显式 db_path 参数 > CAMPUS_KB_DB_PATH 环境变量 >
  <项目根>/data/campus_kb.db；env 一律调用时惰性读取，禁止 import 时定死。
- 表 kb_entries：source TEXT、source_id TEXT、title TEXT、content TEXT、
  url TEXT、metadata_json TEXT、updated_at TEXT，
  PRIMARY KEY(source, source_id)，重复 upsert 全字段覆盖。
- source ∈ {sem_handbook, thucourse_course, thucourse_review,
  thucourse_summary, thubook}（入库侧不强校验枚举，采集侧自律）。

容错纪律（照 agent/report_library.py 防御风格）：
- 读取路径（search_kb / get_entry / stats）在库文件不存在、无表、
  FTS 损坏或任何异常时，分别返回 [] / None / 带 error 字段的字典，
  绝不向调用方抛异常；读取路径绝不创建目录/文件。
- upsert_entries 仅对参数校验错误抛 ValueError；数据库故障按 0 返回，
  绝不抛出。
"""

import json
import logging
import os
import re
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CAMPUS_KB_DB_PATH_ENV = "CAMPUS_KB_DB_PATH"
DB_FILENAME = "campus_kb.db"

# 项目根 = agent/ 的上一级（本文件位于 <项目根>/agent/campus_kb.py）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TABLE_NAME = "kb_entries"
FTS_TABLE_NAME = "kb_entries_fts"

LIMIT_DEFAULT, LIMIT_MIN, LIMIT_MAX = 10, 1, 100

# trigram tokenizer 只能索引/匹配 ≥3 字符的连续片段；
# 短于此的关键词走 FTS 必然零命中，需透明降级 LIKE。
FTS_MIN_KEYWORD_CHARS = 3

# LIKE 启发式评分：标题命中权重高于正文
_LIKE_TITLE_WEIGHT = 3

# 入库列（顺序与 INSERT 参数一致；主键列在前）
_COLUMNS: Tuple[str, ...] = (
    "source", "source_id", "title", "content",
    "url", "metadata_json", "updated_at",
)

_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
  source TEXT NOT NULL,
  source_id TEXT NOT NULL,
  title TEXT,
  content TEXT,
  url TEXT,
  metadata_json TEXT,
  updated_at TEXT,
  PRIMARY KEY (source, source_id)
);
"""

# FTS5 虚表：title/content 建索引，source/source_id 仅随行存储（UNINDEXED）
# 用于同步删除与结果回联；trigram tokenizer 提供中文子串匹配。
_FTS_SCHEMA_SQL = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE_NAME} USING fts5(
  source UNINDEXED,
  source_id UNINDEXED,
  title,
  content,
  tokenize='trigram'
);
"""

_UPSERT_SQL = (
    f"INSERT INTO {TABLE_NAME} ({', '.join(_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in _COLUMNS)}) "
    f"ON CONFLICT(source, source_id) DO UPDATE SET "
    + ", ".join(
        f"{col} = excluded.{col}" for col in _COLUMNS[2:]
    )
)

_FTS_DELETE_SQL = (
    f"DELETE FROM {FTS_TABLE_NAME} WHERE source = ? AND source_id = ?"
)
_FTS_INSERT_SQL = (
    f"INSERT INTO {FTS_TABLE_NAME}(source, source_id, title, content) "
    f"SELECT source, source_id, title, content FROM {TABLE_NAME} "
    f"WHERE source = ? AND source_id = ?"
)

# 模块级写锁：init/upsert 写路径共用（SQLite 文件级并发之外的双保险）。
_LOCK = threading.Lock()

# FTS5+trigram 编译支持探测缓存：None 未探测 / True / False。
_FTS_SUPPORTED: Optional[bool] = None


# ── 路径解析与连接 ──


def _db_path(db_path: Optional[str] = None) -> str:
    """库文件路径解析（调用时惰性读 env）：

    显式参数 > CAMPUS_KB_DB_PATH 环境变量 > <项目根>/data/campus_kb.db。
    """
    if isinstance(db_path, str) and db_path.strip():
        return db_path.strip()
    env_path = os.getenv(CAMPUS_KB_DB_PATH_ENV)
    if env_path and env_path.strip():
        return env_path.strip()
    return os.path.join(_PROJECT_ROOT, "data", DB_FILENAME)


def _connect_write(path: str) -> sqlite3.Connection:
    """写路径连接：父目录不存在自动创建。"""
    dir_name = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_name, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _connect_read(path: str) -> Optional[sqlite3.Connection]:
    """读路径连接：库文件不存在或为目录返回 None（绝不创建文件）。"""
    if not os.path.isfile(path):
        return None
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


# ── FTS 支持探测与表存在性 ──


def _fts_supported() -> bool:
    """探测当前 sqlite3 是否编译支持 FTS5 + trigram tokenizer（结果缓存）。

    用 :memory: 库实际建表探测，一次进程只探测一次；
    任何异常按不支持处理（自动降级 LIKE，调用方无感）。
    """
    global _FTS_SUPPORTED
    if _FTS_SUPPORTED is not None:
        return _FTS_SUPPORTED
    try:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE _probe USING fts5(x, tokenize='trigram')"
            )
        finally:
            conn.close()
        _FTS_SUPPORTED = True
    except Exception as e:
        logger.info("FTS5+trigram 不可用，降级 LIKE 检索: %s", e)
        _FTS_SUPPORTED = False
    return _FTS_SUPPORTED


def _fts_table_exists(conn: sqlite3.Connection) -> bool:
    """目标库中 FTS 虚表是否已建。"""
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (FTS_TABLE_NAME,),
        ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False


# ── 条目归一化与校验 ──


def _to_text(value: Any) -> str:
    """文本字段归一化：None → 空串，其余 str 化并去首尾空白。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _coerce_entry(entry: Any, index: int) -> tuple:
    """把条目字典归一化为入库参数元组。

    参数校验错误抛 ValueError（唯一允许外抛的异常类别）：
    非 dict、缺 source / source_id 或其为空串。
    metadata_json 允许传 dict/list，自动 json.dumps 成字符串。
    updated_at 缺省补当前 ISO 时间戳。
    """
    if not isinstance(entry, dict):
        raise ValueError(f"条目 #{index} 非 dict: {type(entry).__name__}")
    source = _to_text(entry.get("source"))
    source_id = _to_text(entry.get("source_id"))
    if not source:
        raise ValueError(f"条目 #{index} 缺少 source")
    if not source_id:
        raise ValueError(f"条目 #{index} (source={source!r}) 缺少 source_id")
    metadata = entry.get("metadata_json")
    if isinstance(metadata, (dict, list)):
        metadata_json = json.dumps(metadata, ensure_ascii=False)
    else:
        metadata_json = _to_text(metadata)
    updated_at = _to_text(entry.get("updated_at")) or datetime.now().isoformat(
        timespec="seconds"
    )
    return (
        source,
        source_id,
        _to_text(entry.get("title")),
        _to_text(entry.get("content")),
        _to_text(entry.get("url")),
        metadata_json,
        updated_at,
    )


# ── 检索辅助 ──


def _clamp_limit(limit: Any) -> int:
    """limit 夹取 1-100，非法值回退默认 10。"""
    try:
        n = int(limit)
    except (TypeError, ValueError):
        n = LIMIT_DEFAULT
    return max(LIMIT_MIN, min(LIMIT_MAX, n))


def _keywords(query: Any) -> List[str]:
    """查询分词：按空白切多关键词（AND 语义），去空去重保序。"""
    text = _to_text(query)
    if not text:
        return []
    seen, out = set(), []
    for kw in re.split(r"\s+", text):
        if kw and kw not in seen:
            seen.add(kw)
            out.append(kw)
    return out


def _escape_like(text: str) -> str:
    """LIKE 通配符转义（配合 ESCAPE '\\'），防止用户输入注入 %/_ 改变语义。"""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _build_fts_match(keywords: List[str]) -> str:
    """构造 FTS5 MATCH 表达式：各关键词双引号短语（子串语义）AND 连接。"""
    return " AND ".join(f'"{kw.replace(chr(34), chr(34) * 2)}"' for kw in keywords)


def _like_score(title: str, content: str, keywords: List[str]) -> float:
    """LIKE 命中次数启发式评分：标题命中 ×权重 + 正文命中次数（大小写不敏感）。"""
    t, c = title.lower(), content.lower()
    score = 0.0
    for kw in keywords:
        k = kw.lower()
        if not k:
            continue
        score += _LIKE_TITLE_WEIGHT * t.count(k) + c.count(k)
    return score


def _row_to_entry(row: sqlite3.Row, score: Optional[float] = None) -> dict:
    """结果行 → 条目字典（8 字段契约；score 仅检索路径附带）。"""
    entry = {
        "source": row["source"] or "",
        "source_id": row["source_id"] or "",
        "title": row["title"] or "",
        "content": row["content"] or "",
        "url": row["url"] or "",
        "metadata_json": row["metadata_json"] or "",
        "updated_at": row["updated_at"] or "",
    }
    if score is not None:
        entry["score"] = round(float(score), 6)
    return entry


def _search_fts(
    conn: sqlite3.Connection, keywords: List[str], source: str, limit: int
) -> List[dict]:
    """FTS5 检索：bm25 相关度排序（标题列加权 ×5），score = -bm25 越大越相关。"""
    match = _build_fts_match(keywords)
    where = f"{FTS_TABLE_NAME} MATCH ?"
    params: list = [match]
    if source:
        where += f" AND {FTS_TABLE_NAME}.source = ?"
        params.append(source)
    rows = conn.execute(
        f"SELECT e.*, -bm25({FTS_TABLE_NAME}, 0.0, 0.0, 5.0, 1.0) AS score "
        f"FROM {FTS_TABLE_NAME} JOIN {TABLE_NAME} e "
        f"ON e.source = {FTS_TABLE_NAME}.source "
        f"AND e.source_id = {FTS_TABLE_NAME}.source_id "
        f"WHERE {where} ORDER BY bm25({FTS_TABLE_NAME}, 0.0, 0.0, 5.0, 1.0) "
        f"LIMIT ?",
        (*params, limit),
    ).fetchall()
    return [_row_to_entry(r, r["score"]) for r in rows]


def _search_like(
    conn: sqlite3.Connection, keywords: List[str], source: str, limit: int
) -> List[dict]:
    """LIKE 检索：多关键词 AND（title OR content 子串），命中次数启发式排序。"""
    where_parts: List[str] = []
    params: list = []
    if source:
        where_parts.append("source = ?")
        params.append(source)
    for kw in keywords:
        like = f"%{_escape_like(kw)}%"
        where_parts.append(
            "(title LIKE ? ESCAPE '\\' OR content LIKE ? ESCAPE '\\')"
        )
        params.extend([like, like])
    where = " AND ".join(where_parts) if where_parts else "1=1"
    rows = conn.execute(
        f"SELECT * FROM {TABLE_NAME} WHERE {where}", params
    ).fetchall()
    scored = [
        (_like_score(r["title"] or "", r["content"] or "", keywords), r)
        for r in rows
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [_row_to_entry(r, s) for s, r in scored[:limit]]


# ── 公开 API ──


def init_db(db_path: Optional[str] = None) -> None:
    """建库建表（幂等：CREATE TABLE/VIRTUAL TABLE IF NOT EXISTS）。

    FTS5+trigram 编译支持时同步建 FTS 虚表；不支持则跳过（检索自动降级）。
    任何异常仅记日志，绝不抛出。
    """
    path = _db_path(db_path)
    try:
        with _LOCK:
            with _connect_write(path) as conn:
                conn.executescript(_SCHEMA_SQL)
                if _fts_supported():
                    try:
                        conn.execute(_FTS_SCHEMA_SQL)
                    except sqlite3.Error as e:
                        logger.warning("FTS 虚表创建失败（降级 LIKE 检索）: %s", e)
                conn.commit()
    except Exception as e:
        logger.warning("init_db 异常（fail-safe）path=%s: %s", path, e, exc_info=True)


def upsert_entries(entries: List[dict], db_path: Optional[str] = None) -> int:
    """批量 upsert 知识条目（executemany，(source, source_id) 主键重复全字段覆盖），
    返回实际写入条数。

    参数校验错误抛 ValueError（entries 非 list / 条目非 dict /
    缺 source 或 source_id）；数据库故障按已写入条数返回，绝不抛出。
    写库同时同步 FTS 索引（同事务；FTS 不可用时跳过）。
    """
    if not isinstance(entries, (list, tuple)):
        raise ValueError(f"entries 须为 list，收到 {type(entries).__name__}")
    rows = [_coerce_entry(e, i) for i, e in enumerate(entries)]
    if not rows:
        return 0
    try:
        path = _db_path(db_path)
        init_db(path)
        with _LOCK:
            with _connect_write(path) as conn:
                conn.executemany(_UPSERT_SQL, rows)
                if _fts_supported() and _fts_table_exists(conn):
                    for row in rows:
                        conn.execute(_FTS_DELETE_SQL, (row[0], row[1]))
                        conn.execute(_FTS_INSERT_SQL, (row[0], row[1]))
                conn.commit()
        return len(rows)
    except Exception as e:
        logger.warning("upsert_entries 异常（fail-safe）: %s", e, exc_info=True)
        return 0


def search_kb(
    query: str,
    source: Optional[str] = None,
    limit: int = LIMIT_DEFAULT,
    db_path: Optional[str] = None,
) -> List[dict]:
    """检索知识条目：返回按相关度降序的条目字典列表（含 score 字段）。

    - 多关键词空白分隔，AND 语义；中文子串匹配。
    - FTS5+trigram 可用且关键词均 ≥3 字符时走 bm25 排序；
      否则自动降级 LIKE 命中次数启发式排序（对调用方透明）。
    - source 非空时精确过滤来源；limit 夹取 1-100。
    库不存在/无表/任何异常返回 []，绝不抛出。
    """
    try:
        keywords = _keywords(query)
        if not keywords:
            return []
        src = _to_text(source)
        path = _db_path(db_path)
        conn = _connect_read(path)
        if conn is None:
            return []
        try:
            use_fts = (
                _fts_supported()
                and _fts_table_exists(conn)
                and all(len(kw) >= FTS_MIN_KEYWORD_CHARS for kw in keywords)
            )
            if use_fts:
                try:
                    return _search_fts(conn, keywords, src, _clamp_limit(limit))
                except sqlite3.Error as e:
                    logger.warning("FTS 检索失败（降级 LIKE）: %s", e)
            return _search_like(conn, keywords, src, _clamp_limit(limit))
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.warning("search_kb 查询失败（按空结果处理）: %s", e)
        return []
    except Exception as e:
        logger.warning("search_kb 异常（fail-safe）: %s", e, exc_info=True)
        return []


def get_entry(
    source: str, source_id: str, db_path: Optional[str] = None
) -> Optional[dict]:
    """按 (source, source_id) 主键取单条完整条目；不存在或异常返回 None。"""
    try:
        src, sid = _to_text(source), _to_text(source_id)
        if not src or not sid:
            return None
        path = _db_path(db_path)
        conn = _connect_read(path)
        if conn is None:
            return None
        try:
            row = conn.execute(
                f"SELECT * FROM {TABLE_NAME} WHERE source = ? AND source_id = ?",
                (src, sid),
            ).fetchone()
            return _row_to_entry(row) if row is not None else None
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.warning("get_entry 查询失败（按未命中处理）: %s", e)
        return None
    except Exception as e:
        logger.warning("get_entry 异常（fail-safe）: %s", e, exc_info=True)
        return None


def list_entries(
    source: Optional[str] = None,
    limit: Optional[int] = None,
    db_path: Optional[str] = None,
) -> List[dict]:
    """按来源批量列出条目（不走检索评分，供回填/批处理管道全量取数）。

    - source 非空时精确过滤来源；为空/非法时列出全部来源。
    - limit 为正整数时截断；缺省或非法值不截断（调用方自控内存）。
    - 按 source_id 升序返回（确定性顺序），score 字段为 None。
    库不存在/无表/任何异常返回 []，绝不抛出。
    """
    try:
        path = _db_path(db_path)
        conn = _connect_read(path)
        if conn is None:
            return []
        try:
            src = _to_text(source)
            sql = f"SELECT * FROM {TABLE_NAME}"
            params: list = []
            if src:
                sql += " WHERE source = ?"
                params.append(src)
            sql += " ORDER BY source_id"
            if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
                sql += " LIMIT ?"
                params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            return [_row_to_entry(row) for row in rows]
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.warning("list_entries 查询失败（按空结果处理）: %s", e)
        return []
    except Exception as e:
        logger.warning("list_entries 异常（fail-safe）: %s", e, exc_info=True)
        return []


def stats(db_path: Optional[str] = None) -> dict:
    """库统计：{"total": 总条目数, "by_source": {source: 条数},
    "fts_mode": "fts5"|"like", "db_path": 解析路径}。

    by_source 按条数降序；库不存在/异常时返回 total=0 结构并附 error 字段
    （库不存在属正常初始态，不附 error），绝不抛出。
    """
    path = _db_path(db_path)
    result = {
        "total": 0,
        "by_source": {},
        "fts_mode": "fts5" if _fts_supported() else "like",
        "db_path": path,
    }
    try:
        conn = _connect_read(path)
        if conn is None:
            return result
        try:
            result["fts_mode"] = (
                "fts5"
                if (_fts_supported() and _fts_table_exists(conn))
                else "like"
            )
            rows = conn.execute(
                f"SELECT source, COUNT(*) AS n FROM {TABLE_NAME} "
                f"GROUP BY source ORDER BY n DESC, source"
            ).fetchall()
            result["by_source"] = {(r["source"] or ""): r["n"] for r in rows}
            result["total"] = sum(result["by_source"].values())
            return result
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.warning("stats 查询失败: %s", e)
        result["error"] = str(e)
        return result
    except Exception as e:
        logger.warning("stats 异常（fail-safe）: %s", e, exc_info=True)
        result["error"] = str(e)
        return result
