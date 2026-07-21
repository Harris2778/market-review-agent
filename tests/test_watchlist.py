"""agent/watchlist.py 自选股模块（第七波·个性化）测试。

覆盖范围：
1. CRUD 全流程：add → list → remove → is_empty；落盘 JSON 结构逐项核对。
2. 幂等：同关键词重复添加、别名解析到同一只，均返回 (False, "已在自选股中")。
3. 上限：MAX_WATCHLIST_SIZE = 50，第 51 只拒绝且清单不变。
4. 解析失败文案：resolver 返回 None / 抛异常 / 返回非法值 → "未找到匹配的股票"；
   空入参 → "股票名称或代码不能为空"。
5. 并发安全：多线程并发添加不同股票不丢不重；并发添加同一只仅一个成功。
6. 损坏 JSON 自愈：垃圾内容 / 顶层非 list → 备份 <path>.bak + 原子重写空表，后续可用。
7. format_watchlist_block：空清单返回 None；非空输出编号+名称+代码。
8. 路径解析：WATCHLIST_PATH 优先，DATA_DIR 兜底，缺省 data/watchlist.json。
9. 缺省 resolver：mock agent.data_fetcher.search_stock，验证 full_code/market 归一化；
   search_stock 抛异常 → 按未找到处理（绝不真实触网）。
10. fail-safe：落盘失败返回 (False, "自选股保存失败，请稍后重试") 不抛出。

规则（与项目其他测试一致）：所有 resolver/search_stock 全部 mock，零网络；
WATCHLIST_PATH 一律 monkeypatch 到 tmp_path，绝不写真实 data/。
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest

import agent.data_fetcher as data_fetcher
import agent.watchlist as watchlist


# ── 工具 ──


def make_resolver(mapping):
    """构造 mock resolver：dict 查找，命中返回 (code, name, market)，未命中 None。"""
    def _resolver(keyword):
        return mapping.get(keyword)
    return _resolver


MOUTAI = ("sh600519", "贵州茅台", "cn")
CATL = ("sz300750", "宁德时代", "cn")
PING_AN = ("sh601318", "中国平安", "cn")


@pytest.fixture()
def wl_path(tmp_path, monkeypatch):
    """把自选股文件隔离到 tmp_path，绝不触碰真实 data/。"""
    p = tmp_path / "watchlist.json"
    monkeypatch.setenv(watchlist.WATCHLIST_PATH_ENV, str(p))
    return p


# ── 1. CRUD 全流程 ──


def test_crud_roundtrip(wl_path):
    resolver = make_resolver({"贵州茅台": MOUTAI, "宁德时代": CATL})

    assert watchlist.is_empty() is True

    ok, msg = watchlist.add_stock("贵州茅台", resolver)
    assert ok is True
    assert "贵州茅台" in msg and "sh600519" in msg

    ok, msg = watchlist.add_stock("宁德时代", resolver)
    assert ok is True

    stocks = watchlist.list_stocks()
    assert len(stocks) == 2
    assert stocks[0]["code"] == "sh600519"
    assert stocks[0]["name"] == "贵州茅台"
    assert stocks[0]["market"] == "cn"
    # added_at 为合法 ISO 时间戳
    datetime.fromisoformat(stocks[0]["added_at"])
    assert stocks[1]["code"] == "sz300750"
    assert watchlist.is_empty() is False

    # 落盘文件为合法 JSON list，字段齐全
    with open(wl_path, "r", encoding="utf-8") as f:
        on_disk = json.load(f)
    assert isinstance(on_disk, list) and len(on_disk) == 2
    assert set(on_disk[0].keys()) == {"code", "name", "market", "added_at"}

    # list_stocks 返回副本：调用方篡改不影响存储
    stocks[0]["code"] = "tampered"
    assert watchlist.list_stocks()[0]["code"] == "sh600519"

    ok, msg = watchlist.remove_stock("sh600519")
    assert ok is True
    assert "贵州茅台" in msg
    assert [s["code"] for s in watchlist.list_stocks()] == ["sz300750"]

    ok, msg = watchlist.remove_stock("宁德时代")  # 按名称删除
    assert ok is True
    assert watchlist.is_empty() is True


def test_remove_not_found(wl_path):
    ok, msg = watchlist.remove_stock("不存在的股票")
    assert ok is False
    assert "未找到" in msg


# ── 2. 幂等 ──


def test_add_idempotent_same_keyword(wl_path):
    resolver = make_resolver({"贵州茅台": MOUTAI})
    assert watchlist.add_stock("贵州茅台", resolver)[0] is True
    ok, msg = watchlist.add_stock("贵州茅台", resolver)
    assert ok is False
    assert msg == "已在自选股中"
    assert len(watchlist.list_stocks()) == 1


def test_add_idempotent_alias_resolves_to_same_code(wl_path):
    """别名（如代码 600519）解析到已持有的同一只股票 → 幂等拒绝。"""
    resolver = make_resolver({"贵州茅台": MOUTAI, "600519": MOUTAI})
    assert watchlist.add_stock("贵州茅台", resolver)[0] is True
    ok, msg = watchlist.add_stock("600519", resolver)
    assert ok is False
    assert msg == "已在自选股中"
    assert len(watchlist.list_stocks()) == 1


# ── 3. 上限 ──


def test_add_limit_50(wl_path):
    mapping = {f"股票{i:02d}": (f"sh60{i:04d}", f"股票{i:02d}", "cn") for i in range(60)}
    resolver = make_resolver(mapping)
    for i in range(watchlist.MAX_WATCHLIST_SIZE):
        ok, _ = watchlist.add_stock(f"股票{i:02d}", resolver)
        assert ok is True
    assert len(watchlist.list_stocks()) == watchlist.MAX_WATCHLIST_SIZE == 50

    ok, msg = watchlist.add_stock("股票50", resolver)
    assert ok is False
    assert "上限" in msg
    assert len(watchlist.list_stocks()) == 50


# ── 4. 解析失败文案 ──


def test_resolve_returns_none(wl_path):
    ok, msg = watchlist.add_stock("查无此股", make_resolver({}))
    assert ok is False
    assert msg == "未找到匹配的股票：查无此股"
    assert watchlist.is_empty() is True


def test_resolver_raises(wl_path):
    def boom(_kw):
        raise RuntimeError("network down")

    ok, msg = watchlist.add_stock("贵州茅台", boom)
    assert ok is False
    assert "未找到匹配的股票" in msg
    assert watchlist.is_empty() is True


def test_resolver_returns_malformed(wl_path):
    assert watchlist.add_stock("x", lambda _kw: ("", "", ""))[0] is False
    assert watchlist.add_stock("x", lambda _kw: ("sh600519", "", "cn"))[0] is False
    assert watchlist.add_stock("x", lambda _kw: ("only_code",))[0] is False
    assert watchlist.add_stock("x", lambda _kw: "not-a-tuple")[0] is False
    assert watchlist.is_empty() is True


def test_add_blank_input(wl_path):
    ok, msg = watchlist.add_stock("   ", make_resolver({}))
    assert ok is False
    assert "不能为空" in msg


def test_market_default_when_resolver_omits(wl_path):
    resolver = make_resolver({"贵州茅台": ("sh600519", "贵州茅台", "")})
    assert watchlist.add_stock("贵州茅台", resolver)[0] is True
    assert watchlist.list_stocks()[0]["market"] == "cn"


# ── 5. 并发安全 ──


def test_concurrent_adds_no_loss_no_dup(wl_path):
    n = 20
    mapping = {f"股票{i:02d}": (f"sh60{i:04d}", f"股票{i:02d}", "cn") for i in range(n)}
    resolver = make_resolver(mapping)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda kw: watchlist.add_stock(kw, resolver), mapping.keys()))

    assert all(ok for ok, _ in results)
    stocks = watchlist.list_stocks()
    assert len(stocks) == n
    assert len({s["code"] for s in stocks}) == n  # 无重复
    # 落盘文件仍为合法 JSON
    with open(wl_path, "r", encoding="utf-8") as f:
        assert len(json.load(f)) == n


def test_concurrent_add_same_stock_only_one_wins(wl_path):
    resolver = make_resolver({"贵州茅台": MOUTAI})

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: watchlist.add_stock("贵州茅台", resolver), range(10)))

    successes = [r for r in results if r[0]]
    duplicates = [r for r in results if not r[0]]
    assert len(successes) == 1
    assert all(msg == "已在自选股中" for _, msg in duplicates)
    assert len(watchlist.list_stocks()) == 1


def test_concurrent_add_remove_consistent(wl_path):
    mapping = {f"股票{i:02d}": (f"sh60{i:04d}", f"股票{i:02d}", "cn") for i in range(10)}
    resolver = make_resolver(mapping)
    for kw in mapping:
        watchlist.add_stock(kw, resolver)

    def remove_one(i):
        return watchlist.remove_stock(f"sh60{i:04d}")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(remove_one, range(10)))

    assert all(ok for ok, _ in results)
    assert watchlist.is_empty() is True
    with open(wl_path, "r", encoding="utf-8") as f:
        assert json.load(f) == []


# ── 6. 损坏 JSON 自愈 ──


def test_corrupted_json_self_heal(wl_path):
    wl_path.write_text("not json at all{{{", encoding="utf-8")

    assert watchlist.list_stocks() == []  # 触发自愈，按空表处理
    assert (wl_path.parent / "watchlist.json.bak").read_text(encoding="utf-8") == "not json at all{{{"  # 原文件已备份
    assert json.loads(wl_path.read_text(encoding="utf-8")) == []  # 已重写为空表

    # 自愈后正常可用
    ok, _ = watchlist.add_stock("贵州茅台", make_resolver({"贵州茅台": MOUTAI}))
    assert ok is True
    assert len(watchlist.list_stocks()) == 1


def test_top_level_non_list_self_heal(wl_path):
    wl_path.write_text('{"code": "sh600519"}', encoding="utf-8")

    assert watchlist.list_stocks() == []
    assert (wl_path.parent / "watchlist.json.bak").exists()
    assert json.loads(wl_path.read_text(encoding="utf-8")) == []


def test_invalid_entries_filtered_on_load(wl_path):
    """JSON 合法但含非法条目：合法条目保留，非法条目被过滤。"""
    wl_path.write_text(
        json.dumps([
            {"code": "sh600519", "name": "贵州茅台", "market": "cn", "added_at": "2026-01-01T09:00:00"},
            {"code": "", "name": "坏条目"},
            "garbage",
            {"name": "缺代码"},
        ], ensure_ascii=False),
        encoding="utf-8",
    )
    stocks = watchlist.list_stocks()
    assert len(stocks) == 1
    assert stocks[0]["code"] == "sh600519"


# ── 7. format 输出 ──


def test_format_block_empty_returns_none(wl_path):
    assert watchlist.format_watchlist_block() is None


def test_format_block_output(wl_path):
    resolver = make_resolver({"贵州茅台": MOUTAI, "宁德时代": CATL})
    watchlist.add_stock("贵州茅台", resolver)
    watchlist.add_stock("宁德时代", resolver)

    block = watchlist.format_watchlist_block()
    assert block is not None
    lines = block.split("\n")
    assert lines[0].startswith("【用户自选股】")
    assert "2" in lines[0]
    assert lines[1] == "1. 贵州茅台（sh600519）"
    assert lines[2] == "2. 宁德时代（sz300750）"


# ── 8. 路径解析 ──


def test_path_env_priority(tmp_path, monkeypatch):
    custom = tmp_path / "custom.json"
    monkeypatch.setenv(watchlist.WATCHLIST_PATH_ENV, str(custom))
    monkeypatch.setenv(watchlist.DATA_DIR_ENV, str(tmp_path / "ignored"))
    assert watchlist._watchlist_path() == str(custom)


def test_path_data_dir_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv(watchlist.WATCHLIST_PATH_ENV, raising=False)
    monkeypatch.setenv(watchlist.DATA_DIR_ENV, str(tmp_path / "mydata"))
    assert watchlist._watchlist_path() == os.path.join(str(tmp_path / "mydata"), "watchlist.json")

    # DATA_DIR 下的文件确实可读写
    assert watchlist.add_stock("贵州茅台", make_resolver({"贵州茅台": MOUTAI}))[0] is True
    assert (tmp_path / "mydata" / "watchlist.json").exists()


def test_path_default(monkeypatch):
    monkeypatch.delenv(watchlist.WATCHLIST_PATH_ENV, raising=False)
    monkeypatch.delenv(watchlist.DATA_DIR_ENV, raising=False)
    assert watchlist._watchlist_path() == os.path.join("data", "watchlist.json")


# ── 9. 缺省 resolver（mock search_stock，零网络）──


def test_default_resolver_uses_search_stock(wl_path, monkeypatch):
    def fake_search(keyword):
        assert keyword == "贵州茅台"
        return [{"name": "贵州茅台", "market": "11", "code": "600519", "full_code": "sh600519"}]

    monkeypatch.setattr(data_fetcher, "search_stock", fake_search)

    ok, msg = watchlist.add_stock("贵州茅台")  # 不注入 resolver → 走缺省实现
    assert ok is True
    stock = watchlist.list_stocks()[0]
    assert stock["code"] == "sh600519"
    assert stock["name"] == "贵州茅台"
    assert stock["market"] == "cn"  # 上游数字类型码归一化为 cn


def test_default_resolver_search_raises(wl_path, monkeypatch):
    def boom(_keyword):
        raise RuntimeError("network down")

    monkeypatch.setattr(data_fetcher, "search_stock", boom)
    ok, msg = watchlist.add_stock("贵州茅台")
    assert ok is False
    assert "未找到匹配的股票" in msg
    assert watchlist.is_empty() is True


def test_default_resolver_empty_result(wl_path, monkeypatch):
    monkeypatch.setattr(data_fetcher, "search_stock", lambda _kw: [])
    ok, _ = watchlist.add_stock("查无此股")
    assert ok is False


# ── 10. 落盘失败 fail-safe ──


def test_write_failure_returns_friendly_message(tmp_path, monkeypatch):
    """WATCHLIST_PATH 指向一个父路径是普通文件的位置 → 落盘失败但不抛出。"""
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file", encoding="utf-8")
    monkeypatch.setenv(watchlist.WATCHLIST_PATH_ENV, str(blocker / "watchlist.json"))

    ok, msg = watchlist.add_stock("贵州茅台", make_resolver({"贵州茅台": MOUTAI}))
    assert ok is False
    assert "保存失败" in msg

    ok, msg = watchlist.remove_stock("贵州茅台")
    assert ok is False  # 未找到或保存失败，绝不抛出

    assert watchlist.list_stocks() == []
    assert watchlist.clear() is False


def test_clear(wl_path):
    resolver = make_resolver({"贵州茅台": MOUTAI, "宁德时代": CATL})
    watchlist.add_stock("贵州茅台", resolver)
    watchlist.add_stock("宁德时代", resolver)
    assert len(watchlist.list_stocks()) == 2

    assert watchlist.clear() is True
    assert watchlist.is_empty() is True
    assert watchlist.format_watchlist_block() is None
    with open(wl_path, "r", encoding="utf-8") as f:
        assert json.load(f) == []
