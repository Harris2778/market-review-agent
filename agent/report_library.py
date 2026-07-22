"""agent/report_library.py 研报库存储与检索层（研报库 v1）。

职责：
1. 券商研报元数据的 SQLite 持久化（纯 stdlib sqlite3，单文件库）。
2. 多源去重合并：以 info_code 为主键 upsert，冲突字段按源优先级裁决。
3. 检索与聚合：search_reports（关键词/个股/行业/天数过滤）与
   rating_summary（评级分布/目标价区间/EPS 均值/最新研报），
   供 agent/tools.py 的两个新工具接线调用。

存储契约（与全局契约一致）：
- 路径解析：显式 db_path 参数 > REPORTS_DB_PATH 环境变量 >
  ${DATA_DIR:-data}/reports.db；一律调用时惰性读 env，禁止 import 时定死。
- 表结构：docs/RESEARCH_LIB_DESIGN.md「数据库 Schema」节 + source 列
  （TEXT DEFAULT 'eastmoney'）；两个索引照图纸；init_db 幂等。
- 记录字段（16 + source）：info_code, title, org, author,
  publish_date(YYYY-MM-DD), stock_code(6位纯数字或空串), stock_name,
  industry, rating, rating_change, eps_this_year, eps_next_year,
  target_price_high, target_price_low, encode_url, source。
- upsert 合并规则（INSERT ... ON CONFLICT(info_code) DO UPDATE）：
  某字段旧值为空（NULL/空串）用新值回填；双方非空冲突时以源优先级高者为准
  （eastmoney > stockstar > djyanbao > sina > hibor）；
  同优先级（如同源重抓）视为刷新，以新值为准；created_at 保留首次写入时间。

读取纪律：
- search_reports / rating_summary 在库文件不存在、无表、空库或任何异常时，
  返回 total=0 的合法结构，绝不向调用方抛异常。
- 读取路径绝不创建目录/文件（不存在即返回空结构）；写入路径自动建目录。
"""

import logging
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

REPORTS_DB_PATH_ENV = "REPORTS_DB_PATH"
DATA_DIR_ENV = "DATA_DIR"
DEFAULT_DATA_DIR = "data"
DB_FILENAME = "reports.db"

TABLE_NAME = "reports"
DEFAULT_SOURCE = "eastmoney"

# 源优先级（数值大者胜）：docs/RESEARCH_LIB_DESIGN.md「通用采集纪律」
# 东财 > 证券之星 > 洞见 > 新浪 > 慧博；未知源按最低优先级处理。
SOURCE_PRIORITY: Dict[str, int] = {
    "eastmoney": 50,
    "stockstar": 40,
    "djyanbao": 30,
    "sina": 20,
    "hibor": 10,
}
UNKNOWN_SOURCE_PRIORITY = 0

DAYS_DEFAULT, DAYS_MIN, DAYS_MAX = 30, 1, 365
LIMIT_DEFAULT, LIMIT_MIN, LIMIT_MAX = 10, 1, 50

# 入库列（不含 created_at；顺序与 INSERT 参数一致）
_COLUMNS: Tuple[str, ...] = (
    "info_code",
    "title", "org", "author", "publish_date",
    "stock_code", "stock_name", "industry",
    "rating", "rating_change",
    "eps_this_year", "eps_next_year",
    "target_price_high", "target_price_low",
    "encode_url", "source",
)

# 冲突时可合并的列（info_code 主键除外；created_at 保留首次写入）
_MERGE_COLUMNS: Tuple[str, ...] = _COLUMNS[1:]

_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
  info_code TEXT PRIMARY KEY,
  title TEXT, org TEXT, author TEXT, publish_date TEXT,
  stock_code TEXT, stock_name TEXT, industry TEXT,
  rating TEXT, rating_change TEXT,
  eps_this_year REAL, eps_next_year REAL,
  target_price_high REAL, target_price_low REAL,
  encode_url TEXT,
  source TEXT DEFAULT '{DEFAULT_SOURCE}',
  created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_reports_stock
  ON {TABLE_NAME}(stock_code, publish_date);
CREATE INDEX IF NOT EXISTS idx_reports_industry
  ON {TABLE_NAME}(industry, publish_date);
"""

# 模块级写锁：init/upsert 的写路径共用（SQLite 文件级并发之外的双保险）。
_LOCK = threading.Lock()


# ── 路径解析与连接 ──


def _db_path(db_path: Optional[str] = None) -> str:
    """库文件路径解析（调用时惰性读 env）：

    显式参数 > REPORTS_DB_PATH 环境变量 > ${DATA_DIR:-data}/reports.db。
    """
    if isinstance(db_path, str) and db_path.strip():
        return db_path.strip()
    env_path = os.getenv(REPORTS_DB_PATH_ENV)
    if env_path and env_path.strip():
        return env_path.strip()
    data_dir = os.getenv(DATA_DIR_ENV) or DEFAULT_DATA_DIR
    return os.path.join(data_dir, DB_FILENAME)


def _connect_write(path: str) -> sqlite3.Connection:
    """写路径连接：父目录不存在自动创建。"""
    dir_name = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_name, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _connect_read(path: str) -> Optional[sqlite3.Connection]:
    """读路径连接：库文件不存在返回 None（绝不创建文件）。"""
    if not os.path.exists(path):
        return None
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


# ── 记录归一化 ──


def _normalize_stock_code(code: Any) -> str:
    """股票代码归一化：去 sh/sz/bj 交易所前缀并小写（sh600519/SH600519 → 600519）。"""
    if not isinstance(code, str):
        return ""
    c = code.strip().lower()
    for prefix in ("sh", "sz", "bj"):
        if c.startswith(prefix):
            return c[len(prefix):]
    return c


def _to_text(value: Any) -> str:
    """文本字段归一化：None → 空串，其余 str 化并去首尾空白。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _to_float(value: Any) -> Optional[float]:
    """数值字段归一化：可转 float 则转，否则 None（空串/非法值/布尔均 None）。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _coerce_record(record: Any) -> Optional[tuple]:
    """把爬虫记录字典归一化为入库参数元组；缺 info_code 的非法记录返回 None。"""
    if not isinstance(record, dict):
        return None
    info_code = _to_text(record.get("info_code"))
    if not info_code:
        return None
    source = _to_text(record.get("source")).lower() or DEFAULT_SOURCE
    return (
        info_code,
        _to_text(record.get("title")),
        _to_text(record.get("org")),
        _to_text(record.get("author")),
        _to_text(record.get("publish_date")),
        _normalize_stock_code(record.get("stock_code")),
        _to_text(record.get("stock_name")),
        _to_text(record.get("industry")),
        _to_text(record.get("rating")),
        _to_text(record.get("rating_change")),
        _to_float(record.get("eps_this_year")),
        _to_float(record.get("eps_next_year")),
        _to_float(record.get("target_price_high")),
        _to_float(record.get("target_price_low")),
        _to_text(record.get("encode_url")),
        source,
    )


# ── upsert SQL 构造（合并规则见模块 docstring）──


def _priority_case(col_ref: str) -> str:
    """生成源优先级 CASE 表达式：按 source 列取值映射优先级数值。"""
    branches = " ".join(
        f"WHEN '{src}' THEN {prio}" for src, prio in SOURCE_PRIORITY.items()
    )
    return f"CASE {col_ref} {branches} ELSE {UNKNOWN_SOURCE_PRIORITY} END"


def _merge_set_clause() -> str:
    """ON CONFLICT DO UPDATE 的 SET 子句：

    每列规则：旧值为空(NULL/空串) → 新值回填；新值为空 → 保留旧值；
    双方非空 → 源优先级高者为准（同优先级视为刷新取新值）。
    """
    prio_new = _priority_case("excluded.source")
    prio_old = _priority_case(f"{TABLE_NAME}.source")
    parts = []
    for col in _MERGE_COLUMNS:
        parts.append(
            f"{col} = CASE "
            f"WHEN {TABLE_NAME}.{col} IS NULL OR {TABLE_NAME}.{col} = '' "
            f"THEN excluded.{col} "
            f"WHEN excluded.{col} IS NULL OR excluded.{col} = '' "
            f"THEN {TABLE_NAME}.{col} "
            f"WHEN {prio_new} >= {prio_old} THEN excluded.{col} "
            f"ELSE {TABLE_NAME}.{col} END"
        )
    return ", ".join(parts)


_UPSERT_SQL = (
    f"INSERT INTO {TABLE_NAME} ({', '.join(_COLUMNS)}, created_at) "
    f"VALUES ({', '.join('?' for _ in _COLUMNS)}, ?) "
    f"ON CONFLICT(info_code) DO UPDATE SET {_merge_set_clause()}"
)


# ── 检索辅助 ──


def _clamp_days(days: Any) -> int:
    """days 夹取 1-365，非法值回退默认 30。"""
    try:
        d = int(days)
    except (TypeError, ValueError):
        d = DAYS_DEFAULT
    return max(DAYS_MIN, min(DAYS_MAX, d))


def _clamp_limit(limit: Any) -> int:
    """limit 夹取 1-50，非法值回退默认 10。"""
    try:
        n = int(limit)
    except (TypeError, ValueError):
        n = LIMIT_DEFAULT
    return max(LIMIT_MIN, min(LIMIT_MAX, n))


def _escape_like(text: str) -> str:
    """LIKE 通配符转义（配合 ESCAPE '\\'），防止用户输入注入 %/_ 改变语义。"""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _build_filters(
    stock_code: Any = "",
    industry: Any = "",
    query: Any = "",
    days: Any = DAYS_DEFAULT,
) -> Tuple[str, list]:
    """构造 WHERE 子句与参数。

    - days：publish_date >= date('now', '-N days')（publish_date 为空的记录天然排除）
    - stock_code：归一化后精确匹配
    - industry：LIKE %industry%
    - query：非空时对 title/stock_name/org 做 LIKE %query% OR 匹配
    """
    where = ["publish_date >= date('now', ?)"]
    params: list = [f"-{_clamp_days(days)} days"]

    code = _normalize_stock_code(stock_code)
    if code:
        where.append("stock_code = ?")
        params.append(code)

    ind = _to_text(industry)
    if ind:
        where.append("industry LIKE ? ESCAPE '\\'")
        params.append(f"%{_escape_like(ind)}%")

    q = _to_text(query)
    if q:
        like = f"%{_escape_like(q)}%"
        where.append(
            "(title LIKE ? ESCAPE '\\' OR stock_name LIKE ? ESCAPE '\\' "
            "OR org LIKE ? ESCAPE '\\')"
        )
        params.extend([like, like, like])

    return " AND ".join(where), params


def _fmt_number(value: Optional[float]) -> str:
    """目标价展示格式化：整数值去小数点（1800.0 → "1800"），
    其余最多 2 位小数并去尾零（1999.5 → "1999.5"，61.2 → "61.2"）。"""
    if value is None:
        return ""
    f = float(value)
    if f == int(f):
        return str(int(f))
    return f"{f:.2f}".rstrip("0").rstrip(".")


def _num_out(value: Optional[float]) -> Optional[float]:
    """聚合数值输出：整数值转 int，其余 round 2 位；None 原样。"""
    if value is None:
        return None
    f = float(value)
    if f == int(f):
        return int(f)
    return round(f, 2)


def _fmt_target_price(low: Optional[float], high: Optional[float]) -> Optional[str]:
    """target_price 展示字段："低~高" 字符串；高低同值（零宽区间）给单值；
    仅单侧有值给单侧；全空 None。"""
    if low is None and high is None:
        return None
    if low is not None and high is not None:
        if low == high:
            return _fmt_number(low)
        return f"{_fmt_number(low)}~{_fmt_number(high)}"
    return _fmt_number(low if low is not None else high)


def _row_to_item(row: sqlite3.Row) -> dict:
    """检索结果行 → 报告项（字段契约见全局契约 4）。"""
    return {
        "title": row["title"] or "",
        "org": row["org"] or "",
        "author": row["author"] or "",
        "date": row["publish_date"] or "",
        "rating": row["rating"] or "",
        "rating_change": row["rating_change"] or "",
        "target_price": _fmt_target_price(
            row["target_price_low"], row["target_price_high"]
        ),
        "eps_forecast": row["eps_this_year"],
        "eps_next_year": row["eps_next_year"],
        "stock_code": row["stock_code"] or "",
        "stock_name": row["stock_name"] or "",
        "industry": row["industry"] or "",
        "source": row["source"] or DEFAULT_SOURCE,
    }


def _empty_search() -> dict:
    return {"total": 0, "reports": []}


def _empty_summary() -> dict:
    return {
        "total": 0,
        "rating_dist": {},
        "target_price_range": None,
        "avg_eps_forecast": None,
        "latest_reports": [],
    }


# ── 公开 API ──


def init_db(db_path: Optional[str] = None) -> str:
    """建库建表（幂等：CREATE TABLE/INDEX IF NOT EXISTS），返回解析后的库路径。

    目录不存在自动创建；老库缺 source 列时自动 ALTER TABLE 补齐（平滑迁移）。
    任何异常仅记日志，仍返回解析路径，绝不抛出。
    """
    path = _db_path(db_path)
    try:
        with _LOCK:
            with _connect_write(path) as conn:
                conn.executescript(_SCHEMA_SQL)
                # 平滑迁移：历史库（无 source 列）补列
                cols = {
                    r["name"]
                    for r in conn.execute(f"PRAGMA table_info({TABLE_NAME})")
                }
                if "source" not in cols:
                    conn.execute(
                        f"ALTER TABLE {TABLE_NAME} "
                        f"ADD COLUMN source TEXT DEFAULT '{DEFAULT_SOURCE}'"
                    )
                conn.commit()
    except Exception as e:
        logger.warning("init_db 异常（fail-safe）path=%s: %s", path, e, exc_info=True)
    return path


def upsert_reports(records: List[dict], db_path: Optional[str] = None) -> int:
    """批量写入研报记录，返回实际写入/更新条数。

    以 info_code 为主键去重合并（规则见模块 docstring）；缺 info_code 的
    非法记录跳过不计数；任何异常按已写入条数返回，绝不抛出。
    """
    try:
        if not records:
            return 0
        rows = [r for r in (_coerce_record(rec) for rec in records) if r is not None]
        if not rows:
            return 0
        path = init_db(db_path)
        now = datetime.now().isoformat(timespec="seconds")
        written = 0
        with _LOCK:
            with _connect_write(path) as conn:
                for row in rows:
                    try:
                        cur = conn.execute(_UPSERT_SQL, (*row, now))
                        # INSERT 与 ON CONFLICT DO UPDATE 均报 1 行受影响
                        written += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 1
                    except sqlite3.Error as e:
                        logger.warning("研报写入跳过 info_code=%r: %s", row[0], e)
                conn.commit()
        return written
    except Exception as e:
        logger.warning("upsert_reports 异常（fail-safe）: %s", e, exc_info=True)
        return 0


def search_reports(
    query: str = "",
    stock_code: str = "",
    industry: str = "",
    days: int = DAYS_DEFAULT,
    limit: int = LIMIT_DEFAULT,
    db_path: Optional[str] = None,
) -> dict:
    """检索研报：{"total": 命中总数, "reports": [报告项...]}。

    按 publish_date DESC、created_at DESC 排序；limit 夹取 1-50；
    total 为过滤条件命中总数（不受 limit 截断）。
    库文件不存在/无表/空库/异常时返回 total=0 的合法结构，绝不抛出。
    """
    try:
        path = _db_path(db_path)
        conn = _connect_read(path)
        if conn is None:
            return _empty_search()
        try:
            where, params = _build_filters(stock_code, industry, query, days)
            lim = _clamp_limit(limit)
            total = conn.execute(
                f"SELECT COUNT(*) AS n FROM {TABLE_NAME} WHERE {where}", params
            ).fetchone()["n"]
            rows = conn.execute(
                f"SELECT * FROM {TABLE_NAME} WHERE {where} "
                f"ORDER BY publish_date DESC, created_at DESC LIMIT ?",
                (*params, lim),
            ).fetchall()
            return {"total": total, "reports": [_row_to_item(r) for r in rows]}
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.warning("search_reports 查询失败（按空结果处理）: %s", e)
        return _empty_search()
    except Exception as e:
        logger.warning("search_reports 异常（fail-safe）: %s", e, exc_info=True)
        return _empty_search()


def rating_summary(
    stock_code: str = "",
    industry: str = "",
    days: int = DAYS_DEFAULT,
    db_path: Optional[str] = None,
) -> dict:
    """评级聚合统计：

    {"total": 命中数,
     "rating_dist": {"买入": n, ...}（rating 非空计数，按数量降序）,
     "target_price_range": [min(target_price_low), max(target_price_high)]
                           （忽略空值，全空 None）,
     "avg_eps_forecast": eps_this_year 非空均值（保留 2 位，全空 None）,
     "latest_reports": [{"title","org","date"} 最多 3 篇，日期最新在前]}

    库文件不存在/无表/空库/异常时返回 total=0 的合法结构，绝不抛出。
    """
    try:
        path = _db_path(db_path)
        conn = _connect_read(path)
        if conn is None:
            return _empty_summary()
        try:
            where, params = _build_filters(stock_code, industry, "", days)
            result = _empty_summary()
            total = conn.execute(
                f"SELECT COUNT(*) AS n FROM {TABLE_NAME} WHERE {where}", params
            ).fetchone()["n"]
            result["total"] = total
            if total == 0:
                return result

            dist_rows = conn.execute(
                f"SELECT rating, COUNT(*) AS n FROM {TABLE_NAME} WHERE {where} "
                f"AND rating IS NOT NULL AND rating != '' "
                f"GROUP BY rating ORDER BY n DESC, rating",
                params,
            ).fetchall()
            result["rating_dist"] = {r["rating"]: r["n"] for r in dist_rows}

            agg = conn.execute(
                f"SELECT MIN(target_price_low) AS lo, MAX(target_price_high) AS hi, "
                f"AVG(eps_this_year) AS eps FROM {TABLE_NAME} WHERE {where}",
                params,
            ).fetchone()
            if agg["lo"] is not None or agg["hi"] is not None:
                result["target_price_range"] = [
                    _num_out(agg["lo"]),
                    _num_out(agg["hi"]),
                ]
            if agg["eps"] is not None:
                result["avg_eps_forecast"] = round(float(agg["eps"]), 2)

            latest = conn.execute(
                f"SELECT title, org, publish_date FROM {TABLE_NAME} WHERE {where} "
                f"ORDER BY publish_date DESC, created_at DESC LIMIT 3",
                params,
            ).fetchall()
            result["latest_reports"] = [
                {
                    "title": r["title"] or "",
                    "org": r["org"] or "",
                    "date": r["publish_date"] or "",
                }
                for r in latest
            ]
            return result
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.warning("rating_summary 查询失败（按空结果处理）: %s", e)
        return _empty_summary()
    except Exception as e:
        logger.warning("rating_summary 异常（fail-safe）: %s", e, exc_info=True)
        return _empty_summary()
