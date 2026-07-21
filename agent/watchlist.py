"""agent/watchlist.py 自选股存储与解析（第七波·个性化）。

职责：
1. 自选股清单的持久化存储（纯 stdlib，单 JSON 文件）。
2. 名称/代码 → (code, name, market) 解析：resolver 可注入；
   缺省实现内部调 agent.data_fetcher.search_stock（触网！测试必须注入 mock resolver）。
3. prompt 注入块格式化（format_watchlist_block），供 orchestrator 拼进系统提示。

存储契约：
- 路径：环境变量 WATCHLIST_PATH 优先；缺省 ${DATA_DIR:-data}/watchlist.json。
- 结构：list[dict]，每条 {"code": "sh600519", "name": "贵州茅台",
  "market": "cn", "added_at": ISO 时间戳（秒级）}。
- 写入：模块级 threading.Lock + 临时文件 os.replace 原子写。
- 自愈：文件损坏（JSON 解析失败 / 顶层不是 list）时，备份为 <path>.bak
  并原子重写为空表 []，随后按空表继续，绝不抛出。

API 契约（供集成者）：
- add_stock(name_or_code, resolver=None) -> (bool, str)
    resolver: keyword -> (code, name, market) | None，可注入；缺省触网。
    幂等：已存在返回 (False, "已在自选股中")；上限 MAX_WATCHLIST_SIZE = 50。
- remove_stock(name_or_code) -> (bool, str)：按 code（大小写不敏感）或 name 精确匹配。
- list_stocks() -> list[dict]（副本）；is_empty() -> bool；clear() -> bool（仅测试用）。
- format_watchlist_block() -> str | None：空清单返回 None。
全部 fail-safe：任何异常吞掉并记日志，绝不抛出。
"""

import json
import logging
import os
import tempfile
import threading
from datetime import datetime
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

WATCHLIST_PATH_ENV = "WATCHLIST_PATH"
DATA_DIR_ENV = "DATA_DIR"
DEFAULT_DATA_DIR = "data"
WATCHLIST_FILENAME = "watchlist.json"

MAX_WATCHLIST_SIZE = 50
DEFAULT_MARKET = "cn"
KNOWN_MARKETS = ("cn", "hk", "us")

# resolver 契约：name_or_code -> (code, name, market) | None
Resolver = Callable[[str], Optional[Tuple[str, str, str]]]

# 模块级读写锁：add/remove/clear 的读改写与 list/format 的读共用同一把锁。
# 注意：resolver（可能触网）永远在锁外调用，锁内只做读-校验-改-写。
_LOCK = threading.Lock()


# ── 路径与解析 ──


def _watchlist_path() -> str:
    """自选股文件路径：WATCHLIST_PATH 环境变量优先，缺省 ${DATA_DIR:-data}/watchlist.json。"""
    p = os.getenv(WATCHLIST_PATH_ENV)
    if p and p.strip():
        return p.strip()
    data_dir = os.getenv(DATA_DIR_ENV) or DEFAULT_DATA_DIR
    return os.path.join(data_dir, WATCHLIST_FILENAME)


def _normalize_market(market) -> str:
    """market 归一化：cn/hk/us 原样保留；其余（如新浪数字类型码）回退 cn。

    缺省 resolver 用的 search_stock 固定搜 A 股（type=11），其 market 字段
    为上游数字类型码而非 cn/hk/us，统一归一为 cn。
    """
    if isinstance(market, str):
        m = market.strip().lower()
        if m in KNOWN_MARKETS:
            return m
    return DEFAULT_MARKET


def _default_resolver(keyword: str) -> Optional[Tuple[str, str, str]]:
    """缺省解析器：调 agent.data_fetcher.search_stock 取首条结果（触网！）。

    full_code（如 sh600519）作为 code；任何异常/无结果返回 None。
    延迟 import，避免模块加载期触达 data_fetcher 的依赖。
    """
    try:
        from . import data_fetcher

        items = data_fetcher.search_stock(keyword)
    except Exception as e:
        logger.warning("search_stock 调用失败 keyword=%r（按未找到处理）: %s", keyword, e)
        return None
    if not items or not isinstance(items[0], dict):
        return None
    top = items[0]
    code = (top.get("full_code") or "").strip()
    name = (top.get("name") or "").strip()
    if not code or not name:
        return None
    return code, name, _normalize_market(top.get("market"))


def _normalize_resolved(resolved) -> Optional[Tuple[str, str, str]]:
    """校验 resolver 返回值：可解包为三元组且 code/name 非空，否则 None。"""
    try:
        if not resolved:
            return None
        code, name, market = resolved
        code = code.strip() if isinstance(code, str) else ""
        name = name.strip() if isinstance(name, str) else ""
        if not code or not name:
            return None
        return code, name, _normalize_market(market)
    except Exception:
        return None


# ── 底层读写（调用方须已持有 _LOCK）──


def _is_valid_entry(item) -> bool:
    return (
        isinstance(item, dict)
        and isinstance(item.get("code"), str)
        and bool(item.get("code").strip())
        and isinstance(item.get("name"), str)
        and bool(item.get("name").strip())
    )


def _sanitize_entry(item: dict) -> dict:
    """条目字段补齐/清洗：market 缺省 cn，added_at 缺省空串。"""
    added_at = item.get("added_at")
    return {
        "code": item["code"].strip(),
        "name": item["name"].strip(),
        "market": _normalize_market(item.get("market")),
        "added_at": added_at if isinstance(added_at, str) else "",
    }


def _read_stocks(path: str) -> list:
    """读取并校验自选股清单。文件不存在返回 []；损坏时自愈后返回 []。绝不抛出。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        logger.warning("自选股文件损坏 %s（自愈：备份并重写空表）: %s", path, e)
        _heal_corrupted(path)
        return []
    except Exception as e:
        logger.warning("自选股文件读取失败 %s（返回空清单）: %s", path, e, exc_info=True)
        return []
    if not isinstance(data, list):
        logger.warning("自选股文件顶层结构非法 %s（自愈：备份并重写空表）", path)
        _heal_corrupted(path)
        return []
    return [_sanitize_entry(item) for item in data if _is_valid_entry(item)]


def _heal_corrupted(path: str) -> None:
    """损坏文件自愈：备份为 <path>.bak 并原子重写空表。失败仅记日志。"""
    try:
        os.replace(path, path + ".bak")
    except OSError as e:
        logger.warning("损坏自选股文件备份失败 %s（继续重写空表）: %s", path, e)
    _write_stocks(path, [])


def _write_stocks(path: str, stocks: list) -> bool:
    """原子写：先写临时文件再 os.replace，避免中途崩溃截断。失败返回 False。"""
    try:
        dir_name = os.path.dirname(path) or "."
        os.makedirs(dir_name, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".watchlist_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(stocks, f, ensure_ascii=False, indent=2)
                f.write("\n")
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return True
    except Exception as e:
        logger.warning("自选股文件写入失败 %s: %s", path, e, exc_info=True)
        return False


def _find_index(stocks: list, key: str) -> int:
    """按 code（大小写不敏感）或 name 精确匹配，返回下标或 -1。"""
    k = (key or "").strip()
    if not k:
        return -1
    k_lower = k.lower()
    for i, s in enumerate(stocks):
        if s.get("code", "").lower() == k_lower or s.get("name") == k:
            return i
    return -1


def _clean_keyword(name_or_code) -> str:
    if isinstance(name_or_code, str):
        return name_or_code.strip()
    if name_or_code is None:
        return ""
    return str(name_or_code).strip()


# ── 公开 API ──


def add_stock(name_or_code, resolver: Optional[Resolver] = None) -> Tuple[bool, str]:
    """添加自选股。resolver 缺省触网（search_stock），测试必须注入 mock。

    返回 (是否添加成功, 用户可读文案)：
    - 成功:            (True,  "已添加自选股：{name}（{code}）")
    - 已存在（幂等）:   (False, "已在自选股中")
    - 超限:            (False, "自选股已达上限（50 只）")
    - 解析失败:        (False, "未找到匹配的股票：{keyword}")
    - 入参为空:        (False, "股票名称或代码不能为空")
    - 落盘失败:        (False, "自选股保存失败，请稍后重试")
    """
    try:
        keyword = _clean_keyword(name_or_code)
        if not keyword:
            return False, "股票名称或代码不能为空"
        path = _watchlist_path()
        # 第一段锁：幂等/上限快速短路，避免无谓的 resolver（可能触网）调用
        with _LOCK:
            stocks = _read_stocks(path)
            if _find_index(stocks, keyword) >= 0:
                return False, "已在自选股中"
            if len(stocks) >= MAX_WATCHLIST_SIZE:
                return False, f"自选股已达上限（{MAX_WATCHLIST_SIZE} 只）"
        # resolver 在锁外执行（可能触网，绝不阻塞其他操作）
        resolve = resolver if callable(resolver) else _default_resolver
        try:
            resolved = resolve(keyword)
        except Exception as e:
            logger.warning("自选股 resolver 异常 keyword=%r: %s", keyword, e)
            resolved = None
        norm = _normalize_resolved(resolved)
        if norm is None:
            return False, f"未找到匹配的股票：{keyword}"
        code, name, market = norm
        # 第二段锁：重读-复检-追加-原子写（并发下防重复/防超限）
        with _LOCK:
            stocks = _read_stocks(path)
            if (
                _find_index(stocks, keyword) >= 0
                or _find_index(stocks, code) >= 0
                or _find_index(stocks, name) >= 0
            ):
                return False, "已在自选股中"
            if len(stocks) >= MAX_WATCHLIST_SIZE:
                return False, f"自选股已达上限（{MAX_WATCHLIST_SIZE} 只）"
            stocks.append(
                {
                    "code": code,
                    "name": name,
                    "market": market,
                    "added_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            if not _write_stocks(path, stocks):
                return False, "自选股保存失败，请稍后重试"
            return True, f"已添加自选股：{name}（{code}）"
    except Exception as e:
        logger.warning("add_stock 异常（fail-safe）: %s", e, exc_info=True)
        return False, "自选股添加失败"


def remove_stock(name_or_code) -> Tuple[bool, str]:
    """删除自选股：按 code（大小写不敏感）或 name 精确匹配。

    返回 (是否删除成功, 文案)：未找到返回 (False, "自选股中未找到：{keyword}")。
    """
    try:
        keyword = _clean_keyword(name_or_code)
        if not keyword:
            return False, "股票名称或代码不能为空"
        path = _watchlist_path()
        with _LOCK:
            stocks = _read_stocks(path)
            idx = _find_index(stocks, keyword)
            if idx < 0:
                return False, f"自选股中未找到：{keyword}"
            removed = stocks.pop(idx)
            if not _write_stocks(path, stocks):
                return False, "自选股保存失败，请稍后重试"
            return True, f"已删除自选股：{removed['name']}（{removed['code']}）"
    except Exception as e:
        logger.warning("remove_stock 异常（fail-safe）: %s", e, exc_info=True)
        return False, "自选股删除失败"


def list_stocks() -> List[dict]:
    """当前自选股清单（dict 副本，调用方改动不影响存储）。失败返回 []。"""
    try:
        path = _watchlist_path()
        with _LOCK:
            return [dict(s) for s in _read_stocks(path)]
    except Exception as e:
        logger.warning("list_stocks 异常（fail-safe）: %s", e, exc_info=True)
        return []


def is_empty() -> bool:
    """自选股是否为空。"""
    return len(list_stocks()) == 0


def clear() -> bool:
    """清空自选股（仅供测试/管理用途）。落盘成功 True，失败 False。"""
    try:
        path = _watchlist_path()
        with _LOCK:
            return _write_stocks(path, [])
    except Exception as e:
        logger.warning("clear 异常（fail-safe）: %s", e, exc_info=True)
        return False


def format_watchlist_block() -> Optional[str]:
    """把当前自选股格式化为 prompt 注入块；空清单返回 None。

    输出形如：
        【用户自选股】共 2 只（复盘时请优先纳入分析）：
        1. 贵州茅台（sh600519）
        2. 宁德时代（sz300750）
    """
    try:
        stocks = list_stocks()
        if not stocks:
            return None
        lines = [f"【用户自选股】共 {len(stocks)} 只（复盘时请优先纳入分析）："]
        for i, s in enumerate(stocks, 1):
            lines.append(f"{i}. {s['name']}（{s['code']}）")
        return "\n".join(lines)
    except Exception as e:
        logger.warning("format_watchlist_block 异常（fail-safe）: %s", e, exc_info=True)
        return None
