"""
第四波『自我问责系统』存档层：分析产出 JSONL 存档（纯 stdlib，零网络）。

组成：
- 存档层（本模块）：把每次分析产出追加为 JSONL，按天一个文件。
- 打分层（agent/scorer.py，另一工程师并行开发）：事后核对实际行情，
  通过 update_record（或其自实现的同格式读写）写回 score / scored_at / score_note。

存储布局：
- 目录解析（每次调用动态读取，便于测试注入与部署切换）：
  ARCHIVE_DIR 显式设置时优先；缺省推导为 ${DATA_DIR:-data}/archive
  （DATA_DIR 为全项目统一的数据根目录约定，Railway 挂卷后设 DATA_DIR=/data
  即可让存档与图表全部落到挂载卷，见 DEPLOY.md）；
- 文件 archive_YYYYMMDD.jsonl（日期以 trade_date 为准，缺失/非法时回退当前日期），
  每行一个 JSON 对象，追加写入；
- 并发安全：模块级 threading.Lock 保护所有写入与读改写操作，多线程并发追加
  不会出现行交错/串行。

记录 schema（与打分工程师的契约，字段名一字不能改）：
{
  "id":              uuid4hex 字符串,
  "ts":              ISO8601 时间戳（秒级）,
  "trade_date":      "YYYYMMDD",
  "mode":            "market_review" | "sector_deep_dive" | "agent_query",
  "sector":          字符串 或 null,
  "content":         分析全文,
  "context_excerpt": 数据上下文前 4000 字符,
  "numbers":         用 agent.validators.extract_numbers 抽取的数字清单
                     （精简 dict：value / normalized / raw / unit，可 JSON 序列化）,
  "score":           null（由打分层事后写回 "hit" | "miss" | "neutral"）,
  "scored_at":       null（由打分层写回 ISO8601 时间戳）,
  "score_note":      null（由打分层写回核对说明）
}

设计说明：
- 全部 fail-safe：任何异常只记 log。save_analysis 失败返回 None 不抛出；
  load_records 失败返回 []；update_record 失败返回 False。
- 流式路径存档：orchestrator 通过 _stream_response 的可选 archive 回调，
  在流结束（生成器被完整消费）后以最终全文存档。若客户端中途断开导致流
  未被完整消费，该次不存档——属于可接受的 best-effort 语义，不影响主流程。
- update_record 为读改写：同一把锁内定位记录所在日文件，原子替换回写；
  无法解析的坏行原样保留，不丢数据；"id" 主键字段不可被 fields 覆盖。
- 临时存储警告：运行在 Railway（RAILWAY_ENVIRONMENT 存在）但最终目录不在
  挂载卷路径前缀（RAILWAY_VOLUME_PREFIX，默认 /data）下时，logger.warning
  提醒数据将在重启后丢失；每模块仅警告一次（模块级标志位）。
"""

import json
import logging
import os
import tempfile
import threading
import uuid
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# 数字抽取依赖第三波 validators；未就绪时 numbers 记为空清单，不影响存档主流程
try:
    from . import validators
except Exception as e:  # ImportError 及其他异常一律兜底
    logger.warning("agent/validators.py 未就绪，存档 numbers 字段将记为空清单: %s", e)
    validators = None

# 模块级写锁：save_analysis 追加与 update_record 读改写共用同一把锁；
# 打分层（scorer.apply_scores）可通过 writer_lock 参数共用此锁
WRITE_LOCK = threading.Lock()

ARCHIVE_DIR_ENV = "ARCHIVE_DIR"
DATA_DIR_ENV = "DATA_DIR"
DEFAULT_DATA_DIR = "data"
# 旧缺省常量（= os.path.join(DEFAULT_DATA_DIR, "archive")），保留以兼容外部引用；
# 新代码请用 _default_archive_dir() 动态推导
DEFAULT_ARCHIVE_DIR = os.path.join("data", "archive")
RAILWAY_ENV_ENV = "RAILWAY_ENVIRONMENT"
# Railway 挂载卷路径前缀判定：最终目录以此开头才视为已挂卷持久化。
# 判定规则集中在这一处常量，如需调整（例如更换挂载路径）改这里即可。
RAILWAY_VOLUME_PREFIX = "/data"
ARCHIVE_FILE_PREFIX = "archive_"
ARCHIVE_FILE_SUFFIX = ".jsonl"

# context_excerpt 截断长度（契约：数据上下文前 4000 字符）
CONTEXT_EXCERPT_MAX_CHARS = 4000

# 临时存储警告只发一次的模块级标志位（测试可用 monkeypatch 重置）
_EPHEMERAL_WARNED = False


def _data_dir() -> str:
    """数据根目录：环境变量 DATA_DIR，缺省 "data"（空串回退缺省）。"""
    return os.getenv(DATA_DIR_ENV) or DEFAULT_DATA_DIR


def _default_archive_dir() -> str:
    """存档目录缺省值：${DATA_DIR:-data}/archive。"""
    return os.path.join(_data_dir(), "archive")


def _warn_if_ephemeral_storage(path) -> None:
    """Railway 临时存储警告：未挂卷时提醒数据将在重启后丢失（每模块仅警告一次）。

    判定规则：环境变量 RAILWAY_ENVIRONMENT 存在（Railway 运行时自动注入），
    且最终目录不以 RAILWAY_VOLUME_PREFIX（默认 "/data"，挂载卷路径前缀）开头。
    挂载路径前缀的判定集中在 RAILWAY_VOLUME_PREFIX 常量，如需调整改该常量。
    本函数自身 fail-safe：任何异常静默吞掉，绝不影响目录解析主流程。
    """
    global _EPHEMERAL_WARNED
    if _EPHEMERAL_WARNED:
        return
    try:
        if not os.getenv(RAILWAY_ENV_ENV):
            return
        if str(path).startswith(RAILWAY_VOLUME_PREFIX):
            return
        logger.warning(
            "运行在 Railway 但未挂卷，重启后数据将丢失"
            "（当前存档目录=%r，不在挂载卷路径 %s 下；"
            "请挂载 Volume 到 %s 并设置 %s=%s，详见 DEPLOY.md）",
            path, RAILWAY_VOLUME_PREFIX,
            RAILWAY_VOLUME_PREFIX, DATA_DIR_ENV, RAILWAY_VOLUME_PREFIX,
        )
        _EPHEMERAL_WARNED = True
    except Exception:
        pass


def _archive_dir() -> str:
    """存档目录：ARCHIVE_DIR 显式设置优先，缺省推导为 ${DATA_DIR:-data}/archive。"""
    path = os.getenv(ARCHIVE_DIR_ENV) or _default_archive_dir()
    _warn_if_ephemeral_storage(path)
    return path


def _is_valid_trade_date(value) -> bool:
    """trade_date 合法性：8 位数字字符串（防路径穿越与文件名污染）。"""
    try:
        s = str(value).strip()
        return len(s) == 8 and s.isdigit()
    except Exception:
        return False


def _normalize_trade_date(trade_date) -> str:
    """trade_date 兜底：非法/缺失时回退当前日期。"""
    if _is_valid_trade_date(trade_date):
        return str(trade_date).strip()
    return datetime.now().strftime("%Y%m%d")


def _day_file(trade_date: str) -> str:
    return os.path.join(
        _archive_dir(), f"{ARCHIVE_FILE_PREFIX}{trade_date}{ARCHIVE_FILE_SUFFIX}"
    )


def _list_archive_files(archive_dir: str) -> list:
    """列出存档目录下全部日文件（按文件名排序）。目录不存在返回 []。"""
    try:
        if not os.path.isdir(archive_dir):
            return []
        return [
            os.path.join(archive_dir, name)
            for name in sorted(os.listdir(archive_dir))
            if name.startswith(ARCHIVE_FILE_PREFIX) and name.endswith(ARCHIVE_FILE_SUFFIX)
        ]
    except Exception as e:
        logger.warning("存档目录列举失败（返回空清单）: %s", e, exc_info=True)
        return []


def _extract_numbers_slim(content: str) -> list:
    """用 validators.extract_numbers 抽取数字并精简为可序列化 dict；失败返回 []。"""
    if validators is None:
        return []
    try:
        tokens = validators.extract_numbers(content or "")
        slim = []
        for tok in tokens or []:
            try:
                slim.append({
                    "value": tok.get("value"),
                    "normalized": tok.get("normalized"),
                    "raw": tok.get("raw"),
                    "unit": tok.get("unit"),
                })
            except Exception:
                continue
        return slim
    except Exception as e:
        logger.warning("存档数字抽取异常（记为空清单）: %s", e, exc_info=True)
        return []


def _read_jsonl(path: str) -> tuple:
    """读 JSONL 文件，返回 (records, bad_lines)。

    坏行（JSON 解析失败）原样保留在 bad_lines 中，供回写时不丢数据。
    文件不存在或读取失败返回 ([], [])。
    """
    records = []
    bad_lines = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                    if isinstance(obj, dict):
                        records.append(obj)
                    else:
                        bad_lines.append(stripped)
                except json.JSONDecodeError:
                    logger.warning("存档文件 %s 第 %d 行 JSON 解析失败（按坏行保留）", path, lineno)
                    bad_lines.append(stripped)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("存档文件读取失败 %s（返回空）: %s", path, e, exc_info=True)
    return records, bad_lines


def _write_jsonl(path: str, records: list, bad_lines: list) -> None:
    """原子回写：先写临时文件再 os.replace，避免中途崩溃截断存档。"""
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=".archive_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for line in bad_lines:
                f.write(line + "\n")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_analysis(mode, sector, content, context, trade_date) -> Optional[str]:
    """追加一条分析存档，返回记录 id；任何失败返回 None，绝不抛出。

    mode:       "market_review" | "sector_deep_dive" | "agent_query"
    sector:     板块名字符串或 None
    content:    分析全文（非字符串输入强转为字符串）
    context:    数据上下文（仅前 4000 字符入库为 context_excerpt）
    trade_date: "YYYYMMDD"，非法/缺失回退当前日期
    """
    try:
        td = _normalize_trade_date(trade_date)
        text = content if isinstance(content, str) else str(content or "")
        ctx = context if isinstance(context, str) else str(context or "")
        record = {
            "id": uuid.uuid4().hex,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "trade_date": td,
            "mode": mode if isinstance(mode, str) else str(mode),
            "sector": sector if isinstance(sector, str) and sector else None,
            "content": text,
            "context_excerpt": ctx[:CONTEXT_EXCERPT_MAX_CHARS],
            "numbers": _extract_numbers_slim(text),
            "score": None,
            "scored_at": None,
            "score_note": None,
        }
        line = json.dumps(record, ensure_ascii=False)
        archive_dir = _archive_dir()
        path = _day_file(td)
        with WRITE_LOCK:
            os.makedirs(archive_dir, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return record["id"]
    except Exception as e:
        logger.warning("分析存档写入失败（忽略，不影响主流程）: %s", e, exc_info=True)
        return None


def load_records(date_str=None) -> list:
    """加载存档记录。date_str=None 加载全部日期（按文件名排序拼接），
    否则仅加载该日文件。任何失败返回 []，绝不抛出。"""
    try:
        if date_str is not None:
            if not _is_valid_trade_date(date_str):
                logger.warning("load_records 收到非法日期 %r（返回空清单）", date_str)
                return []
            return _read_jsonl(_day_file(str(date_str).strip()))[0]
        records = []
        for path in _list_archive_files(_archive_dir()):
            records.extend(_read_jsonl(path)[0])
        return records
    except Exception as e:
        logger.warning("存档加载失败（返回空清单）: %s", e, exc_info=True)
        return []


def update_record(record_id, fields) -> bool:
    """读改写更新记录（与追加写共用同一把锁）。

    按文件名排序扫描全部日文件，定位 id 匹配的记录后用 fields 更新
    （"id" 主键不可覆盖），原子替换回写该文件；坏行原样保留。
    找到并写回返回 True；未找到或任何异常返回 False，绝不抛出。
    """
    try:
        if not record_id or not isinstance(record_id, str):
            return False
        if not isinstance(fields, dict) or not fields:
            return False
        with WRITE_LOCK:
            for path in _list_archive_files(_archive_dir()):
                records, bad_lines = _read_jsonl(path)
                hit = None
                for rec in records:
                    if rec.get("id") == record_id:
                        hit = rec
                        break
                if hit is None:
                    continue
                for key, value in fields.items():
                    if key == "id":
                        continue
                    hit[key] = value
                _write_jsonl(path, records, bad_lines)
                return True
        return False
    except Exception as e:
        logger.warning("存档更新失败（id=%s，返回 False）: %s", record_id, e, exc_info=True)
        return False
