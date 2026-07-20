"""
图表生成：行业热力图、指数走势、资金流向。用 matplotlib，支持中文。
"""

import io
import os
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime

# 中文字体路径（Docker 中通过 apt 安装，macOS 通过系统字体）
def _get_chinese_font():
    """查找可用的中文字体。"""
    font_names = [
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "Noto Sans CJK SC", "Noto Sans SC",
        "SimHei", "STHeiti", "PingFang SC",
        "Heiti SC", "Arial Unicode MS",
    ]
    for name in font_names:
        for f in fm.fontManager.ttflist:
            if name in f.name:
                return f.name
    return None


FONT_NAME = _get_chinese_font()

if FONT_NAME:
    plt.rcParams["font.family"] = FONT_NAME
plt.rcParams["axes.unicode_minus"] = False


def _fig_to_base64(fig) -> str:
    """将 matplotlib figure 转为 base64 PNG。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def sector_heatmap(sectors: list, date: str = "") -> str:
    """
    行业涨跌热力图 — 横条图，红涨绿跌。
    sectors: [{"name": "煤炭", "pct_chg": 5.47, "tag": "强势"}, ...]
    返回 base64 PNG 字符串。
    """
    if not sectors:
        return ""

    names = [s["name"] for s in sectors]
    values = [s["pct_chg"] for s in sectors]
    tags = [s.get("tag", "") for s in sectors]

    # 按涨幅排序
    sorted_data = sorted(zip(names, values, tags), key=lambda x: x[1], reverse=True)
    names, values, tags = zip(*sorted_data)

    # 颜色：红涨绿跌，中性灰色
    colors = []
    for v in values:
        if v > 1:
            colors.append("#e74c3c")  # 红色 - 强势
        elif v > 0:
            colors.append("#f39c12")  # 橙色 - 偏强
        elif v >= -1:
            colors.append("#95a5a6")  # 灰色 - 中性
        elif v >= -2:
            colors.append("#3498db")  # 蓝色 - 偏弱
        else:
            colors.append("#2ecc71")  # 绿色 - 弱势

    fig, ax = plt.subplots(figsize=(8, 10))
    y_pos = range(len(names))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel("涨跌幅 (%)", fontsize=10)
    title = f"申万一级行业涨跌幅 — {date}" if date else "申万一级行业涨跌幅"
    ax.set_title(title, fontsize=13, fontweight="bold")

    # 在条形末端标注数值
    for i, (v, bar) in enumerate(zip(values, bars)):
        x_pos = bar.get_width()
        offset = 0.3 if v >= 0 else -0.3
        ha = "left" if v >= 0 else "right"
        ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                f"{v:+.1f}%", va="center", ha=ha, fontsize=7, color="#333")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    return _fig_to_base64(fig)


def index_comparison(indices: dict, global_indices: dict = None) -> str:
    """
    指数对比图 — A股指数 + 全球指数。
    返回 base64 PNG 字符串。
    """
    if not indices:
        return ""

    # A股指数
    names = list(indices.keys())
    values = [d["pct_chg"] for d in indices.values()]
    colors_a = ["#e74c3c" if v > 0 else "#2ecc71" if v < 0 else "#95a5a6" for v in values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左：A股
    y_pos = range(len(names))
    ax1.barh(y_pos, values, color=colors_a, edgecolor="white", height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.invert_yaxis()
    ax1.axvline(x=0, color="black", linewidth=0.5)
    ax1.set_title("A股主要指数", fontsize=11, fontweight="bold")
    for i, v in enumerate(values):
        ax1.text(v + (0.3 if v >= 0 else -0.3), i, f"{v:+.2f}%",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)

    # 右：全球
    if global_indices:
        g_names = list(global_indices.keys())
        g_values = [d["pct_chg"] for d in global_indices.values()]
        g_colors = ["#e74c3c" if v > 0 else "#2ecc71" if v < 0 else "#95a5a6" for v in g_values]
        g_y = range(len(g_names))
        ax2.barh(g_y, g_values, color=g_colors, edgecolor="white", height=0.6)
        ax2.set_yticks(g_y)
        ax2.set_yticklabels(g_names, fontsize=8)
        ax2.invert_yaxis()
        ax2.axvline(x=0, color="black", linewidth=0.5)
        ax2.set_title("全球主要指数", fontsize=11, fontweight="bold")
        for i, v in enumerate(g_values):
            ax2.text(v + (0.3 if v >= 0 else -0.3), i, f"{v:+.2f}%",
                    va="center", ha="left" if v >= 0 else "right", fontsize=8)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return _fig_to_base64(fig)
