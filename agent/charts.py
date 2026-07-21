"""
第五波『可视化』核心模块：零依赖 SVG 图表生成。

设计原则：
- 纯 stdlib，禁止引入 matplotlib 等第三方依赖（Railway 部署保持轻量）。
- 中文字体由查看端渲染，SVG 只携带文本与 font-family 提示，不内嵌字体。
- A 股配色惯例：涨红跌绿、零值灰，低饱和朴素风格。
- 契约函数 generate_daily_charts(snapshot, out_dir=None) 供推送工程师
  try/except import 调用：snapshot 字段缺失时跳过对应图并记 log，绝不抛异常。

数据结构（对齐 data_fetcher.MarketSnapshot）：
- snapshot.indices: dict {指数名: {"pct_chg": float, ...}}
- snapshot.sectors: list [{"name": str, "pct_chg": float, ...}]
- snapshot._breadth: dict {"涨停": "123", "涨7-10": ..., "平": ..., "跌0-2": ...,
                           "跌停": ..., "total_up": ..., "total_down": ...}（值多为字符串）
"""

import logging
import math
import os
from datetime import datetime
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)

# ── 朴素低饱和配色（涨红跌绿）──
UP_COLOR = "#cf4b44"      # 涨- muted 红
DOWN_COLOR = "#3d9a6c"    # 跌- muted 绿
FLAT_COLOR = "#9aa0a6"    # 平/零- 灰
TEXT_COLOR = "#3a3a3a"
AXIS_COLOR = "#8f8a80"
GRID_COLOR = "#e8e5de"
BG_COLOR = "#ffffff"

_FONT_FAMILY = "PingFang SC, Microsoft YaHei, Noto Sans CJK SC, Hiragino Sans GB, sans-serif"

# 涨跌分布桶顺序（自上而下）与配色
_BREADTH_BUCKETS = [
    ("涨停", UP_COLOR), ("涨7-10", UP_COLOR), ("涨5-7", UP_COLOR),
    ("涨2-5", UP_COLOR), ("涨0-2", UP_COLOR),
    ("平", FLAT_COLOR),
    ("跌0-2", DOWN_COLOR), ("跌2-5", DOWN_COLOR), ("跌5-7", DOWN_COLOR),
    ("跌7-10", DOWN_COLOR), ("跌停", DOWN_COLOR),
]

_NAME_MAX_CHARS = 8  # 左侧名称标签最大字符数，超出截断加省略号


def _snap_get(snapshot, key, default=None):
    """兼容 dataclass 对象与 dict 两种 snapshot 形态。"""
    if isinstance(snapshot, dict):
        return snapshot.get(key, default)
    return getattr(snapshot, key, default)


def _to_float(value):
    """宽松数值解析：失败/非有限值返回 None。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _fmt_pct(v: float) -> str:
    """涨跌幅数值标签：带符号、两位小数、百分号。"""
    return f"{v:+.2f}%"


def _norm_date(raw) -> str:
    """日期归一化为 YYYYMMDD；缺失/异常时回退为今天。"""
    digits = "".join(ch for ch in str(raw or "") if ch.isdigit())
    if len(digits) == 8:
        return digits
    return datetime.now().strftime("%Y%m%d")


def _disp_date(yyyymmdd: str) -> str:
    """YYYYMMDD → YYYY-MM-DD（仅用于标题展示）。"""
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def _truncate(name: str, max_chars: int = _NAME_MAX_CHARS) -> str:
    """长名称截断，防止溢出左侧标签区。"""
    name = str(name)
    if len(name) <= max_chars:
        return name
    return name[: max_chars - 1] + "…"


def _chart_height(n: int) -> int:
    """按条目数估算画布高度（行业图条目多、行高压窄）。"""
    row = 30 if n <= 12 else 24
    return max(180, 74 + n * row)


def _bar_chart_svg(items, title, width, height) -> str:
    """手写横向条形图 SVG（坐标轴/柱体/数值标签/标题）。

    items 元素支持三种形态：
      (label, value)                    —— 数值标签默认 "+X.XX"，颜色按正负自动
      (label, value, display)           —— 自定义数值标签文本
      (label, value, display, color)    —— 再自定义柱体颜色（用于涨跌分布桶）
    非法条目静默跳过；空数据返回仅含标题的合法 SVG。
    """
    # ── 条目归一化 ──
    rows = []
    for it in items or []:
        try:
            label = str(it[0])
            value = float(it[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not math.isfinite(value):
            continue
        display = str(it[2]) if len(it) > 2 else f"{value:+.2f}"
        color = str(it[3]) if len(it) > 3 else None
        rows.append((label, value, display, color))

    width = max(int(width), 240)
    height = max(int(height), 120)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{_FONT_FAMILY}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{BG_COLOR}"/>',
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" font-size="15" '
        f'font-weight="600" fill="{TEXT_COLOR}">{escape(str(title))}</text>',
    ]

    if not rows:
        parts.append(
            f'<text x="{width / 2:.1f}" y="{height / 2:.1f}" text-anchor="middle" '
            f'font-size="12" fill="{FLAT_COLOR}">暂无数据</text>'
        )
        parts.append("</svg>")
        return "".join(parts)

    # ── 布局 ──
    margin_top, margin_bottom, margin_left, margin_right = 44, 34, 100, 76
    plot_w = max(width - margin_left - margin_right, 40)
    plot_h = max(height - margin_top - margin_bottom, 20)
    n = len(rows)
    row_h = plot_h / n
    bar_h = round(min(row_h * 0.62, 20), 1)

    lo = min(0.0, min(r[1] for r in rows))
    hi = max(0.0, max(r[1] for r in rows))
    if hi == lo:  # 全零数据，避免除零
        hi = lo + 1.0

    def x_of(v: float) -> float:
        return margin_left + (v - lo) / (hi - lo) * plot_w

    zero_x = x_of(0.0)
    axis_y = height - margin_bottom

    # ── 参考线（lo / 0 / hi，过近去重）＋ 底部坐标轴 ──
    ticks = [(lo, x_of(lo)), (0.0, zero_x), (hi, x_of(hi))]
    placed_x = []
    for val, tx in ticks:
        if any(abs(tx - px) < 30 for px in placed_x):
            continue
        placed_x.append(tx)
        line_color = AXIS_COLOR if val == 0 else GRID_COLOR
        parts.append(
            f'<line x1="{tx:.1f}" y1="{margin_top - 6}" x2="{tx:.1f}" y2="{axis_y}" '
            f'stroke="{line_color}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{tx:.1f}" y="{axis_y + 16}" text-anchor="middle" font-size="10" '
            f'fill="{FLAT_COLOR}">{val:.2f}</text>'
        )
    parts.append(
        f'<line x1="{margin_left}" y1="{axis_y}" x2="{width - margin_right}" y2="{axis_y}" '
        f'stroke="{AXIS_COLOR}" stroke-width="1"/>'
    )

    # ── 柱体 + 名称标签 + 数值标签 ──
    for i, (label, value, display, color) in enumerate(rows):
        cy = margin_top + row_h * i + row_h / 2
        if value > 0:
            fill = color or UP_COLOR
        elif value < 0:
            fill = color or DOWN_COLOR
        else:
            fill = color or FLAT_COLOR

        # 名称标签（右对齐，长名截断）
        name = _truncate(label)
        parts.append(
            f'<text x="{margin_left - 8}" y="{cy + 4:.1f}" text-anchor="end" '
            f'font-size="12" fill="{TEXT_COLOR}">{escape(name)}</text>'
        )

        if value != 0:
            x0, x1 = sorted((zero_x, x_of(value)))
            bw = max(x1 - x0, 1.0)
            parts.append(
                f'<rect x="{x0:.1f}" y="{cy - bar_h / 2:.1f}" width="{bw:.1f}" '
                f'height="{bar_h}" fill="{fill}" rx="1.5"/>'
            )

        # 数值标签：正值在柱右、负值在柱左，越界时翻转锚点防溢出
        est_w = len(display) * 6.4
        if value >= 0:
            lx, anchor = x_of(value) + 5, "start"
            if lx + est_w > width - 4:
                lx, anchor = width - 4, "end"
        else:
            lx, anchor = x_of(value) - 5, "end"
            if lx - est_w < 4:
                lx, anchor = x_of(value) + 5, "start"
        parts.append(
            f'<text x="{lx:.1f}" y="{cy + 3.8:.1f}" text-anchor="{anchor}" '
            f'font-size="11" fill="{fill}">{escape(display)}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _write_svg(path: str, svg: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)


def generate_daily_charts(snapshot, out_dir=None) -> list[str]:
    """生成当日常用行情图，返回生成的文件路径列表。

    - out_dir 为 None 时取环境变量 CHART_DIR（缺省 "charts"）+ 当天日期子目录
      （charts/YYYYMMDD/）；显式传入时按原样使用。
    - 产出：indices.svg（主要指数涨跌）、sectors.svg（申万 31 行业横向条形图，
      正红负绿按涨跌幅降序）、breadth.svg（涨跌分布，需 snapshot 带 _breadth）。
    - snapshot 字段缺失时跳过对应图并记 log，绝不抛异常。
    """
    paths: list[str] = []

    try:
        date_str = _norm_date(_snap_get(snapshot, "date", ""))
        if out_dir is None:
            base = os.getenv("CHART_DIR", "charts") or "charts"
            out_dir = os.path.join(base, date_str)
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        logger.warning("charts: 输出目录准备失败，放弃生成图表", exc_info=True)
        return paths

    disp = _disp_date(date_str)

    # ── (a) 主要指数涨跌条形图 ──
    try:
        indices = _snap_get(snapshot, "indices") or {}
        items = []
        if isinstance(indices, dict):
            for name, d in indices.items():
                v = _to_float(d.get("pct_chg") if isinstance(d, dict) else None)
                if v is not None:
                    items.append((str(name), v, _fmt_pct(v)))
        if items:
            svg = _bar_chart_svg(items, f"主要指数涨跌幅（{disp}）",
                                 720, _chart_height(len(items)))
            p = os.path.join(out_dir, "indices.svg")
            _write_svg(p, svg)
            paths.append(p)
        else:
            logger.info("charts: snapshot 缺少有效指数数据，跳过 indices.svg")
    except Exception:
        logger.warning("charts: 生成 indices.svg 失败", exc_info=True)

    # ── (b) 申万行业涨跌幅横向条形图（正红负绿，按涨跌幅降序）──
    try:
        sectors = _snap_get(snapshot, "sectors") or []
        items = []
        for s in sectors:
            if not isinstance(s, dict):
                continue
            v = _to_float(s.get("pct_chg"))
            if v is None:
                continue
            items.append((str(s.get("name", "?")), v))
        if items:
            items.sort(key=lambda t: t[1], reverse=True)
            items = [(name, v, _fmt_pct(v)) for name, v in items]
            svg = _bar_chart_svg(items, f"申万一级行业涨跌幅（{disp}）",
                                 760, _chart_height(len(items)))
            p = os.path.join(out_dir, "sectors.svg")
            _write_svg(p, svg)
            paths.append(p)
        else:
            logger.info("charts: snapshot 缺少有效行业数据，跳过 sectors.svg")
    except Exception:
        logger.warning("charts: 生成 sectors.svg 失败", exc_info=True)

    # ── (c) 全市场涨跌分布（可选，依赖 snapshot._breadth）──
    try:
        breadth = _snap_get(snapshot, "_breadth") or _snap_get(snapshot, "breadth") or {}
        items = []
        if isinstance(breadth, dict):
            for key, color in _BREADTH_BUCKETS:
                v = _to_float(breadth.get(key))
                if v is None:
                    continue
                items.append((key, v, str(int(v)), color))
        if items:
            svg = _bar_chart_svg(items, f"A股全市场涨跌分布（{disp}）",
                                 720, _chart_height(len(items)))
            p = os.path.join(out_dir, "breadth.svg")
            _write_svg(p, svg)
            paths.append(p)
        else:
            logger.info("charts: snapshot 缺少涨跌分布数据，跳过 breadth.svg")
    except Exception:
        logger.warning("charts: 生成 breadth.svg 失败", exc_info=True)

    return paths
