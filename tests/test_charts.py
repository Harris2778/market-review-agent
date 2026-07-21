"""第五波『可视化』agent/charts.py 单元测试。

全部本地构造数据、写入 pytest tmp_path，零网络、零第三方服务。
覆盖契约：合法 XML、标题与条目齐全、正负双色分支、空数据不抛、
out_dir 自动创建、默认路径含日期子目录、字段缺失跳过并记 log。
"""

import logging
import os
import xml.etree.ElementTree as ET
from datetime import datetime

import pytest

from agent.charts import (
    DOWN_COLOR,
    FLAT_COLOR,
    UP_COLOR,
    _bar_chart_svg,
    generate_daily_charts,
)
from agent.data_fetcher import MarketSnapshot

# ── 测试数据构造 ──

SECTOR_NAMES = [
    "农林牧渔", "采掘", "化工", "钢铁", "有色金属", "电子", "家用电器",
    "食品饮料", "纺织服装", "轻工制造", "医药生物", "公用事业", "交通运输",
    "房地产", "商业贸易", "休闲服务", "综合", "建筑材料", "建筑装饰",
    "电气设备", "国防军工", "计算机", "传媒", "通信", "银行", "非银金融",
    "汽车", "机械设备", "煤炭", "石油石化", "环保",
]


def _make_indices():
    return {
        "上证综指": {"close": 3900.12, "pct_chg": 0.85},
        "深证成指": {"close": 13000.5, "pct_chg": 1.2345},
        "创业板指": {"close": 3100.0, "pct_chg": -2.5},
        "科创50": {"close": 1400.0, "pct_chg": -0.42},
        "沪深300": {"close": 4600.0, "pct_chg": 0.0},
        "中证500": {"close": 7000.0, "pct_chg": 0.66},
        "中证1000": {"close": 8000.0, "pct_chg": -1.01},
    }


def _make_sectors():
    """31 个申万行业，故意乱序，验证模块内部按涨跌幅降序重排。"""
    sectors = []
    for i, name in enumerate(SECTOR_NAMES):
        # 电子 +3.5 最高，银行 -2.1 最低，其余穿插正负
        pct = round(3.5 - i * 0.183, 3)
        sectors.append({"name": name, "pct_chg": pct, "tag": "中性"})
    sectors[24]["pct_chg"] = -2.1   # 银行垫底
    return sectors


def _make_breadth():
    return {
        "date": "2026-06-09", "涨停": "58", "涨7-10": "40", "涨5-7": "90",
        "涨2-5": "500", "涨0-2": "1500", "平": "180", "跌0-2": "1400",
        "跌2-5": "450", "跌5-7": "80", "跌7-10": "30", "跌停": "12",
        "total_up": "2188", "total_down": "1972",
    }


def _full_snapshot():
    snap = MarketSnapshot(date="20260609")
    snap.indices = _make_indices()
    snap.sectors = _make_sectors()
    snap._breadth = _make_breadth()  # 动态属性，对齐 collect_market_snapshot
    return snap


def _parse(path_or_text):
    if os.path.exists(str(path_or_text)):
        return ET.parse(str(path_or_text)).getroot()
    return ET.fromstring(path_or_text)


# ── _bar_chart_svg ──


def test_bar_chart_svg_valid_xml_and_contains_all_text():
    svg = _bar_chart_svg(
        [("上证综指", 1.2345), ("创业板指", -2.5), ("沪深300", 0.0)],
        "主要指数涨跌幅（2026-06-09）", 720, 300)
    root = ET.fromstring(svg)  # 合法 XML，不抛即过
    assert root.tag.endswith("svg")
    assert "主要指数涨跌幅（2026-06-09）" in svg
    for name in ("上证综指", "创业板指", "沪深300"):
        assert name in svg
    # 数值保留两位小数
    assert "+1.23" in svg
    assert "-2.50" in svg
    assert "+0.00" in svg


def test_bar_chart_svg_positive_negative_flat_color_branches():
    svg = _bar_chart_svg(
        [("涨", 1.0), ("跌", -1.0), ("平", 0.0)], "三色", 400, 200)
    assert UP_COLOR in svg      # 正值走红
    assert DOWN_COLOR in svg    # 负值走绿
    assert FLAT_COLOR in svg    # 零值走灰


def test_bar_chart_svg_long_name_truncated():
    long_name = "这是一个非常非常长的行业名称测试"
    svg = _bar_chart_svg([(long_name, 1.0)], "截断", 400, 200)
    assert "…" in svg
    assert long_name not in svg


def test_bar_chart_svg_empty_items_returns_valid_xml():
    svg = _bar_chart_svg([], "空图", 400, 200)
    ET.fromstring(svg)
    assert "空图" in svg
    assert "暂无数据" in svg


def test_bar_chart_svg_custom_display_and_color():
    svg = _bar_chart_svg(
        [("涨停", 58, "58", UP_COLOR), ("跌停", 12, "12", DOWN_COLOR)],
        "涨跌分布", 400, 200)
    ET.fromstring(svg)
    assert ">58</text>" in svg
    assert ">12</text>" in svg


def test_bar_chart_svg_skips_illegal_items():
    svg = _bar_chart_svg(
        [("好", 1.0), (None, None), ("坏", "abc"), ("非有限", float("nan"))],
        "容错", 400, 200)
    ET.fromstring(svg)
    assert "好" in svg
    assert "abc" not in svg


# ── generate_daily_charts ──


def test_generate_daily_charts_full_snapshot(tmp_path):
    out = str(tmp_path / "c")
    paths = generate_daily_charts(_full_snapshot(), out_dir=out)
    assert len(paths) == 3
    assert all(isinstance(p, str) for p in paths)
    names = sorted(os.path.basename(p) for p in paths)
    assert names == ["breadth.svg", "indices.svg", "sectors.svg"]
    for p in paths:
        assert os.path.exists(p)
        _parse(p)  # 每张图都是合法 XML

    by_name = {os.path.basename(p): p for p in paths}

    idx_svg = open(by_name["indices.svg"], encoding="utf-8").read()
    for name in _make_indices():
        assert name in idx_svg
    assert UP_COLOR in idx_svg and DOWN_COLOR in idx_svg

    sec_svg = open(by_name["sectors.svg"], encoding="utf-8").read()
    for name in SECTOR_NAMES:  # 31 行业全画
        assert name in sec_svg
    # 按涨跌幅降序：电子(+3.5) 必须先于 银行(-2.1) 出现
    assert sec_svg.find("电子") < sec_svg.find("银行")

    br_svg = open(by_name["breadth.svg"], encoding="utf-8").read()
    for key in ("涨停", "涨0-2", "平", "跌0-2", "跌停"):
        assert key in br_svg
    assert ">58</text>" in br_svg and ">12</text>" in br_svg


def test_generate_daily_charts_empty_snapshot_returns_empty_list(tmp_path):
    paths = generate_daily_charts(MarketSnapshot(date="20260609"),
                                  out_dir=str(tmp_path / "c"))
    assert paths == []  # 空数据返回空列表，不抛异常


def test_generate_daily_charts_missing_fields_skip_and_log(tmp_path, caplog):
    snap = MarketSnapshot(date="20260609")
    snap.sectors = _make_sectors()  # 只有行业，无指数、无涨跌分布
    with caplog.at_level(logging.INFO, logger="agent.charts"):
        paths = generate_daily_charts(snap, out_dir=str(tmp_path / "c"))
    assert [os.path.basename(p) for p in paths] == ["sectors.svg"]
    skip_logs = [r.getMessage() for r in caplog.records if "跳过" in r.getMessage()]
    assert any("indices.svg" in m for m in skip_logs)
    assert any("breadth.svg" in m for m in skip_logs)


def test_generate_daily_charts_out_dir_auto_created(tmp_path):
    out = str(tmp_path / "deep" / "nested" / "dir")  # 不存在的多级目录
    paths = generate_daily_charts(_full_snapshot(), out_dir=out)
    assert os.path.isdir(out)
    assert len(paths) == 3


def test_generate_daily_charts_default_dir_uses_env_and_date_subdir(tmp_path, monkeypatch):
    monkeypatch.setenv("CHART_DIR", str(tmp_path / "mycharts"))
    snap = _full_snapshot()
    snap.date = "2026-06-09"  # 带连字符也应归一化为 20260609
    paths = generate_daily_charts(snap)
    assert len(paths) == 3
    for p in paths:
        assert os.path.basename(os.path.dirname(p)) == "20260609"  # 日期子目录
        assert str(tmp_path / "mycharts") in p
        assert os.path.exists(p)


def test_generate_daily_charts_default_dir_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("CHART_DIR", raising=False)
    monkeypatch.chdir(tmp_path)  # 相对 charts/ 落进临时目录，不污染仓库
    snap = _full_snapshot()
    snap.date = ""  # 缺失日期回退为今天
    paths = generate_daily_charts(snap)
    today = datetime.now().strftime("%Y%m%d")
    assert len(paths) == 3
    for p in paths:
        assert os.path.basename(os.path.dirname(p)) == today
        assert os.path.exists(tmp_path / p)


def test_generate_daily_charts_bad_values_never_raises(tmp_path):
    snap = MarketSnapshot(date="20260609")
    snap.indices = {
        "缺值": {"pct_chg": None},
        "坏值": {"pct_chg": "abc"},
        "异形": "not-a-dict",
        "正常": {"pct_chg": 1.5},
    }
    snap.sectors = [{"name": "缺pct"}, {"pct_chg": "bad"}, None,
                    {"name": "电子", "pct_chg": 0.5}]
    snap._breadth = {"涨停": "xx", "平": 100}
    paths = generate_daily_charts(snap, out_dir=str(tmp_path / "c"))
    assert len(paths) == 3  # 每组均有至少一条有效数据 → 三张图都出
    by_name = {os.path.basename(p): p for p in paths}
    idx = open(by_name["indices.svg"], encoding="utf-8").read()
    assert "正常" in idx and "坏值" not in idx
    br = open(by_name["breadth.svg"], encoding="utf-8").read()
    assert "平" in br and "涨停" not in br  # 解析失败的桶被跳过


def test_generate_daily_charts_all_bad_indices_skips_chart(tmp_path):
    snap = MarketSnapshot(date="20260609")
    snap.indices = {"坏值": {"pct_chg": "abc"}}
    paths = generate_daily_charts(snap, out_dir=str(tmp_path / "c"))
    assert paths == []  # 全部条目无效 → 跳过该图且不抛
