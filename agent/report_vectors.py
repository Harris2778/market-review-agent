"""agent/report_vectors.py 研报全文分块 + 向量化 + 检索层（研报库 v2 向量侧）。

职责：
1. 向量索引表管理（契约 3，与 reports 同库）：report_chunks / report_embeddings /
   vector_meta，init_vector_tables 幂等。
2. 分块：chunk_report 按节顺序切块（默认 ≤500 字/块、重叠 50），空节跳过，
   chunk_id = f"{info_code}#{chunk_idx}"（chunk_idx 为全报告跨节连续序号）。
3. Embedder 协议（契约 4）：.name / .dim / .embed(texts) -> list[list[float]]
   （L2 归一）。FakeEmbedder 为确定性哈希实现（测试与缺省降级用，零网络）；
   BgeEmbedder 为生产实现（bge-small-zh-v1.5），sentence_transformers/torch
   只在构造时惰性导入——模块级 import 绝不引入重依赖。
4. build_index：report_fulltext JOIN reports → 分块 → embed → 写
   report_chunks / report_embeddings（numpy float32 tobytes），vector_meta 记录
   embedder name/dim；已索引的 info_code 默认跳过，force=True 重建。
5. search_vectors：query embed → numpy 余弦暴力检索，元数据与过滤器走
   JOIN reports（stock_code 去前缀精确匹配 / industry LIKE / publish_date 窗口）。

纪律（与全局契约一致）：
- 路径解析一律复用 agent.report_library._db_path（函数内惰性导入，不重写路径逻辑）。
- 公开函数 fail-safe：依赖缺失/未建索引/维度不符/任何异常均降级为带 note 的
  合法结构，绝不向调用方抛异常。
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

CHUNKS_TABLE = "report_chunks"
EMBEDDINGS_TABLE = "report_embeddings"
META_TABLE = "vector_meta"

CHUNK_MAX_CHARS = 500
CHUNK_OVERLAP = 50

DAYS_DEFAULT, DAYS_MIN, DAYS_MAX = 90, 1, 365
TOP_K_DEFAULT, TOP_K_MIN, TOP_K_MAX = 5, 1, 50

# 契约 3 表结构（与 reports 同库；report_fulltext 由全文采集侧负责）
_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {CHUNKS_TABLE} (
  chunk_id TEXT PRIMARY KEY,
  info_code TEXT,
  section TEXT,
  text TEXT
);
CREATE TABLE IF NOT EXISTS {EMBEDDINGS_TABLE} (
  chunk_id TEXT PRIMARY KEY,
  vector BLOB
);
CREATE TABLE IF NOT EXISTS {META_TABLE} (
  k TEXT PRIMARY KEY,
  v TEXT
);
"""

# 模块级写锁：建表/建索引写路径共用（对齐 report_library 的双保险风格）。
_LOCK = threading.Lock()


# ── 参数夹取 ──


def _clamp_days(days: Any) -> int:
    """days 夹取 1-365，非法值回退默认 90。"""
    try:
        d = int(days)
    except (TypeError, ValueError):
        d = DAYS_DEFAULT
    return max(DAYS_MIN, min(DAYS_MAX, d))


def _clamp_top_k(top_k: Any) -> int:
    """top_k 夹取 1-50，非法值回退默认 5。"""
    try:
        k = int(top_k)
    except (TypeError, ValueError):
        k = TOP_K_DEFAULT
    return max(TOP_K_MIN, min(TOP_K_MAX, k))


def _clamp_positive_int(value: Any) -> Optional[int]:
    """可选正整数参数（days/limit）：非法或 None 返回 None（表示不限制）。"""
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


# ── 分块 ──


def chunk_report(
    sections: Any,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """按节顺序把研报全文切成重叠块。

    入参 sections 为 [{"name": 节名, "text": 节文本}...]（契约 2）；
    返回 [{"chunk_idx": 全报告连续序号, "section": 节名, "text": 块文本}...]。

    规则：
    - 每块 ≤ max_chars 字，相邻块重叠 overlap 字（overlap 夹取到 [0, max_chars-1]，
      保证窗口必前进不死循环）；
    - 空节（text 空白/非字符串）与非 dict 项跳过；节内切不完再进下一节，
      chunk_idx 跨节连续递增；
    - 非法入参（非 list/非法数值）防御式回退，绝不抛出。
    """
    try:
        max_chars = max(1, int(max_chars))
    except (TypeError, ValueError):
        max_chars = CHUNK_MAX_CHARS
    try:
        overlap = int(overlap)
    except (TypeError, ValueError):
        overlap = CHUNK_OVERLAP
    overlap = max(0, min(overlap, max_chars - 1))
    step = max_chars - overlap

    chunks: List[Dict[str, Any]] = []
    if not isinstance(sections, (list, tuple)):
        return chunks
    idx = 0
    for item in sections:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue  # 空节跳过
        name = str(item.get("name") or "正文")
        n = len(text)
        start = 0
        while start < n:
            chunks.append(
                {"chunk_idx": idx, "section": name, "text": text[start:start + max_chars]}
            )
            idx += 1
            if start + max_chars >= n:
                break
            start += step
    return chunks


def _parse_sections(sections_json: Any, fulltext: Any) -> List[Dict[str, str]]:
    """sections_json → [{"name","text"}...]；解析失败/为空时退化为单节「正文」。

    （契约 2：东财 PDF 解析不出结构时退化为单节正文；这里对任意源做同样兜底。）
    """
    data: Any = None
    if isinstance(sections_json, str) and sections_json.strip():
        try:
            data = json.loads(sections_json)
        except (json.JSONDecodeError, ValueError):
            data = None
    sections: List[Dict[str, str]] = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                sections.append({"name": str(item.get("name") or "正文"), "text": text})
    if sections:
        return sections
    if isinstance(fulltext, str) and fulltext.strip():
        return [{"name": "正文", "text": fulltext}]
    return []


# ── Embedder（契约 4）──


class FakeEmbedder:
    """确定性哈希 Embedder：同文本恒同向量，零网络零依赖。

    每个维度由 sha256(f"{text}#{i}") 派生 [-1, 1] 浮点，再整体 L2 归一。
    用于测试与缺省降级；无语义相似度，仅相同/近似文本可得高余弦分。
    """

    name = "fake"

    def __init__(self, dim: int = 32):
        try:
            self.dim = max(1, int(dim))
        except (TypeError, ValueError):
            self.dim = 32

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts or []:
            s = t if isinstance(t, str) else str(t)
            comps = []
            for i in range(self.dim):
                h = hashlib.sha256(f"{s}#{i}".encode("utf-8")).digest()
                comps.append(int.from_bytes(h[:8], "big") / float(2**64) * 2.0 - 1.0)
            arr = np.asarray(comps, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm > 0:
                arr = arr / np.float32(norm)
            out.append([float(x) for x in arr])
        return out


class BgeEmbedder:
    """生产 Embedder：BAAI/bge-small-zh-v1.5（512 维中文向量）。

    sentence_transformers（及其 torch 依赖）只在构造时惰性导入——ImportError
    只在构造时抛出，模块级 import 绝不受影响。HF_ENDPOINT 环境变量由
    huggingface_hub 原生读取（镜像加速用），这里不覆盖不改写。
    """

    def __init__(self, model: str = "", dim: int = 512):
        from sentence_transformers import SentenceTransformer  # 惰性导入：仅构造时

        # 模型路径/ID 解析：显式参数 > REPORT_EMBED_MODEL 环境变量 > HF 默认 ID。
        # 国内环境 huggingface.co 不可达时（实测 2026-07-22）改用 ModelScope：
        #   pip install modelscope
        #   python -c "from modelscope import snapshot_download; \
        #       print(snapshot_download('BAAI/bge-small-zh-v1.5'))"
        # 再把 REPORT_EMBED_MODEL 设为打印出的本地路径。
        model = (
            model
            or os.environ.get("REPORT_EMBED_MODEL")
            or "BAAI/bge-small-zh-v1.5"
        )
        self.name = f"bge:{model}"
        self.dim = int(dim)
        self._model = SentenceTransformer(model)

    def embed(self, texts: List[str]) -> List[List[float]]:
        arr = self._model.encode(
            list(texts or []),
            normalize_embeddings=True,  # 批量 L2 归一（契约 4）
            convert_to_numpy=True,
        )
        return [[float(x) for x in row] for row in np.asarray(arr, dtype=np.float32)]


def _default_embedder() -> Optional[Any]:
    """惰性构造生产 Embedder；依赖缺失/模型不可下载等任何异常返回 None。"""
    try:
        return BgeEmbedder()
    except Exception as e:
        logger.warning(
            "BgeEmbedder 构造失败（缺依赖或模型不可下载，按不可用降级）: %s", e
        )
        return None


def _degraded(note: str) -> Dict[str, Any]:
    """search_vectors 降级结构：total_chunks=0 + 空 hits + 中文说明。"""
    return {"total_chunks": 0, "hits": [], "note": note}


# ── 建表 ──


def init_vector_tables(db_path: Optional[str] = None) -> str:
    """建向量索引三表（幂等：CREATE TABLE IF NOT EXISTS），返回解析后的库路径。

    路径解析复用 agent.report_library._db_path（惰性导入）；目录不存在自动创建。
    任何异常仅记日志，仍返回解析路径，绝不抛出。
    """
    from . import report_library

    path = report_library._db_path(db_path)
    try:
        with _LOCK:
            with report_library._connect_write(path) as conn:
                conn.executescript(_SCHEMA_SQL)
                conn.commit()
    except Exception as e:
        logger.warning(
            "init_vector_tables 异常（fail-safe）path=%s: %s", path, e, exc_info=True
        )
    return path


def _write_meta(conn: sqlite3.Connection, embedder: Any) -> None:
    """vector_meta 记录 embedder name/dim 与最近索引时间。"""
    conn.execute(
        f"INSERT OR REPLACE INTO {META_TABLE}(k, v) VALUES('embedder_name', ?)",
        (str(getattr(embedder, "name", "?")),),
    )
    conn.execute(
        f"INSERT OR REPLACE INTO {META_TABLE}(k, v) VALUES('embedder_dim', ?)",
        (str(int(getattr(embedder, "dim", 0))),),
    )
    conn.execute(
        f"INSERT OR REPLACE INTO {META_TABLE}(k, v) VALUES('indexed_at', ?)",
        (datetime.now().isoformat(timespec="seconds"),),
    )


# ── 索引构建 ──


def build_index(
    db_path: Optional[str] = None,
    embedder: Optional[Any] = None,
    days: Optional[int] = None,
    limit: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """把 report_fulltext 的全文分块向量化写入索引。

    返回 {"indexed_reports": n, "indexed_chunks": m, "skipped": k}：
    - 数据源：report_fulltext JOIN reports（取 publish_date 排序/过滤用）；
      days/limit 可选限定最近 N 天/最多 N 篇；
    - 已有索引的 info_code 默认跳过（计入 skipped）；force=True 时先清旧块
      旧向量再重建；空内容（分不出任何块）也计入 skipped；
    - embedder=None 时惰性构造 BgeEmbedder；构造失败返回带 note 的零统计
      结构，绝不抛出；全文表不存在/任何异常同样降级为带 note 的零统计。
    """
    stats: Dict[str, Any] = {"indexed_reports": 0, "indexed_chunks": 0, "skipped": 0}
    try:
        from . import report_library

        if embedder is None:
            embedder = _default_embedder()
            if embedder is None:
                return {
                    **stats,
                    "note": "向量模型不可用：请安装 sentence-transformers 后重试，"
                    "或显式传入 embedder（如 FakeEmbedder）",
                }

        path = report_library.init_db(db_path)  # 确保 reports 表存在（幂等）
        init_vector_tables(path)

        days_n = _clamp_positive_int(days)
        limit_n = _clamp_positive_int(limit)
        where = ""
        params: List[Any] = []
        if days_n is not None:
            where = "AND r.publish_date >= date('now', ?)"
            params.append(f"-{days_n} days")
        sql = (
            "SELECT f.info_code, f.source, f.fulltext, f.sections_json, r.publish_date "
            "FROM report_fulltext f "
            "JOIN reports r ON r.info_code = f.info_code "
            f"{where} ORDER BY r.publish_date DESC, f.info_code"
        )
        if limit_n is not None:
            sql += " LIMIT ?"
            params.append(limit_n)

        with _LOCK:
            with report_library._connect_write(path) as conn:
                try:
                    rows = conn.execute(sql, params).fetchall()
                except sqlite3.Error as e:
                    logger.info("build_index：全文表不可用（按零统计降级）: %s", e)
                    return {
                        **stats,
                        "note": "全文表 report_fulltext 尚不可用：请先运行全文采集 "
                        "（scripts/report_fulltext.py）再建索引",
                    }
                if not rows:
                    return stats

                existing = set()
                if not force:
                    existing = {
                        r["info_code"]
                        for r in conn.execute(
                            f"SELECT DISTINCT info_code FROM {CHUNKS_TABLE}"
                        )
                    }

                for row in rows:
                    code = row["info_code"]
                    if code in existing:
                        stats["skipped"] += 1
                        continue
                    if force:
                        # 重建：先清该报告的旧块与旧向量，避免残留过期块
                        old_ids = [
                            r["chunk_id"]
                            for r in conn.execute(
                                f"SELECT chunk_id FROM {CHUNKS_TABLE} "
                                "WHERE info_code = ?",
                                (code,),
                            )
                        ]
                        conn.execute(
                            f"DELETE FROM {CHUNKS_TABLE} WHERE info_code = ?", (code,)
                        )
                        if old_ids:
                            conn.executemany(
                                f"DELETE FROM {EMBEDDINGS_TABLE} WHERE chunk_id = ?",
                                [(cid,) for cid in old_ids],
                            )

                    sections = _parse_sections(row["sections_json"], row["fulltext"])
                    chunks = chunk_report(sections)
                    if not chunks:
                        logger.info("build_index：空内容跳过 info_code=%r", code)
                        stats["skipped"] += 1
                        continue

                    vectors = embedder.embed([c["text"] for c in chunks])
                    for c, vec in zip(chunks, vectors):
                        cid = f"{code}#{c['chunk_idx']}"
                        conn.execute(
                            f"INSERT OR REPLACE INTO {CHUNKS_TABLE}"
                            "(chunk_id, info_code, section, text) VALUES(?, ?, ?, ?)",
                            (cid, code, c["section"], c["text"]),
                        )
                        blob = np.asarray(vec, dtype=np.float32).tobytes()
                        conn.execute(
                            f"INSERT OR REPLACE INTO {EMBEDDINGS_TABLE}"
                            "(chunk_id, vector) VALUES(?, ?)",
                            (cid, blob),
                        )
                    stats["indexed_reports"] += 1
                    stats["indexed_chunks"] += len(chunks)

                _write_meta(conn, embedder)
                conn.commit()
        return stats
    except Exception as e:
        logger.warning("build_index 异常（fail-safe）: %s", e, exc_info=True)
        return {**stats, "note": f"索引构建失败：{e}"}


# ── 向量检索 ──


def search_vectors(
    query: str,
    stock_code: str = "",
    industry: str = "",
    days: int = DAYS_DEFAULT,
    top_k: int = TOP_K_DEFAULT,
    db_path: Optional[str] = None,
    embedder: Optional[Any] = None,
) -> Dict[str, Any]:
    """研报全文向量检索（契约 5）。

    返回 {"total_chunks": 索引总块数, "hits": [{"info_code","title","org","date",
    "rating","section","snippet","score"}...]}：
    - query embed 后对全部候选块做 numpy 余弦暴力检索，score 降序取 top_k；
    - 过滤与元数据走 JOIN reports：stock_code 去 sh/sz/bj 前缀精确匹配、
      industry LIKE %..%（通配符转义）、publish_date >= date('now','-N days')；
    - snippet 为块文本原样（≤500 字），score round 4；
    - 维度不符/未建索引/依赖缺失/任何异常 → {"total_chunks":0,"hits":[],
      "note":"中文说明"}，绝不抛出。
    """
    try:
        from . import report_library

        q = query if isinstance(query, str) else str(query or "")
        if not q.strip():
            return _degraded("查询词为空：请传入非空 query")

        if embedder is None:
            embedder = _default_embedder()
            if embedder is None:
                return _degraded(
                    "向量模型不可用：请安装 sentence-transformers，"
                    "或显式传入 embedder（如 FakeEmbedder）"
                )

        path = report_library._db_path(db_path)
        conn = report_library._connect_read(path)
        if conn is None:
            return _degraded("研报向量索引尚未建立：请先运行全文采集与 build_index")
        try:
            # 维度校验：索引 embedder 维度必须与当前 embedder 一致
            meta: Dict[str, str] = {}
            try:
                for r in conn.execute(f"SELECT k, v FROM {META_TABLE}"):
                    meta[r["k"]] = r["v"]
            except sqlite3.Error as e:
                logger.info("search_vectors：向量元信息不可用（按未建索引降级）: %s", e)
                return _degraded(
                    "研报向量索引尚未建立：请先运行全文采集与 build_index"
                )
            try:
                stored_dim = int(meta.get("embedder_dim") or 0)
            except (TypeError, ValueError):
                stored_dim = 0
            if stored_dim <= 0:
                return _degraded(
                    "向量索引缺少 embedder 元信息：请先运行 build_index 建索引"
                )
            if stored_dim != int(getattr(embedder, "dim", 0)):
                return _degraded(
                    f"向量维度不匹配：索引为 {stored_dim} 维"
                    f"（{meta.get('embedder_name') or '未知模型'}），当前 embedder 为 "
                    f"{getattr(embedder, 'dim', '?')} 维，请用同一模型重建索引"
                )

            # 过滤器与元数据 JOIN reports（契约 5）
            where = ["r.publish_date >= date('now', ?)"]
            params: List[Any] = [f"-{_clamp_days(days)} days"]
            code = report_library._normalize_stock_code(stock_code)
            if code:
                where.append("r.stock_code = ?")
                params.append(code)
            ind = industry.strip() if isinstance(industry, str) else ""
            if ind:
                where.append("r.industry LIKE ? ESCAPE '\\'")
                params.append(f"%{report_library._escape_like(ind)}%")
            sql = (
                "SELECT c.chunk_id, c.info_code, c.section, c.text, e.vector, "
                "r.title, r.org, r.publish_date, r.rating "
                f"FROM {CHUNKS_TABLE} c "
                f"JOIN {EMBEDDINGS_TABLE} e ON e.chunk_id = c.chunk_id "
                "JOIN reports r ON r.info_code = c.info_code "
                f"WHERE {' AND '.join(where)}"
            )
            try:
                rows = conn.execute(sql, params).fetchall()
                total_chunks = conn.execute(
                    f"SELECT COUNT(*) AS n FROM {CHUNKS_TABLE}"
                ).fetchone()["n"]
            except sqlite3.Error as e:
                logger.info("search_vectors：向量索引未就绪（按未建索引降级）: %s", e)
                return _degraded(
                    "研报向量索引尚未建立：请先运行全文采集与 build_index"
                )
        finally:
            conn.close()

        if not rows:
            return {"total_chunks": total_chunks, "hits": []}

        qvec = np.asarray(embedder.embed([q])[0], dtype=np.float32)
        qnorm = float(np.linalg.norm(qvec))
        scored = []
        for row in rows:
            blob = row["vector"]
            if not blob or len(blob) % 4 != 0:
                continue
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.shape[0] != qvec.shape[0]:
                continue  # 单块维度异常跳过，不拖垮整体
            denom = float(np.linalg.norm(vec)) * qnorm
            score = float(np.dot(vec, qvec) / denom) if denom > 0 else 0.0
            scored.append((score, row))
        # score 降序；同分按日期新→旧
        scored.sort(key=lambda x: (x[0], x[1]["publish_date"] or ""), reverse=True)

        k = _clamp_top_k(top_k)
        hits = []
        for score, row in scored[:k]:
            hits.append(
                {
                    "info_code": row["info_code"] or "",
                    "title": row["title"] or "",
                    "org": row["org"] or "",
                    "date": row["publish_date"] or "",
                    "rating": row["rating"] or "",
                    "section": row["section"] or "",
                    "snippet": (row["text"] or "")[:CHUNK_MAX_CHARS],
                    "score": round(score, 4),
                }
            )
        return {"total_chunks": total_chunks, "hits": hits}
    except Exception as e:
        logger.warning("search_vectors 异常（fail-safe）: %s", e, exc_info=True)
        return _degraded(f"向量检索失败：{e}")
