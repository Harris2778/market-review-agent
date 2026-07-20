"""
图表生成 v2 — 用 QuickChart.io 免费 API，无需 matplotlib，无需中文字体。
"""

import json
import urllib.parse


def sector_heatmap_url(sectors: list, date: str = "") -> str:
    """
    行业涨跌热力图 URL（QuickChart.io）。
    红涨绿跌，横条图，31 行业一目了然。
    """
    if not sectors:
        return ""

    sorted_data = sorted(sectors, key=lambda x: x["pct_chg"], reverse=True)
    names = [s["name"] for s in sorted_data]
    values = [s["pct_chg"] for s in sorted_data]

    # 颜色映射
    colors = []
    for v in values:
        if v > 1:
            colors.append("#e74c3c")
        elif v > 0:
            colors.append("#f39c12")
        elif v >= -1:
            colors.append("#95a5a6")
        elif v >= -2:
            colors.append("#3498db")
        else:
            colors.append("#27ae60")

    title = f"申万一级行业涨跌幅 — {date}" if date else "申万一级行业涨跌幅"

    chart_config = {
        "type": "horizontalBar",
        "data": {
            "labels": names,
            "datasets": [{
                "data": values,
                "backgroundColor": colors,
                "borderColor": colors,
                "borderWidth": 0,
            }],
        },
        "options": {
            "title": {
                "display": True,
                "text": title,
                "fontSize": 16,
                "fontColor": "#333",
            },
            "legend": {"display": False},
            "scales": {
                "xAxes": [{
                    "ticks": {
                        "callback": "function(v) { return v.toFixed(1) + '%'; }",
                        "fontSize": 10,
                    },
                    "gridLines": {"display": True, "color": "#eee"},
                }],
                "yAxes": [{
                    "ticks": {"fontSize": 11, "fontColor": "#333"},
                    "gridLines": {"display": False},
                }],
            },
            "plugins": {
                "datalabels": {
                    "display": True,
                    "anchor": "right" if values[0] >= 0 else "left",
                    "align": "right" if values[0] >= 0 else "left",
                    "color": "#333",
                    "font": {"size": 9},
                    "formatter": "function(v) { return v.toFixed(1) + '%'; }",
                },
            },
            "layout": {
                "padding": {"left": 10, "right": 40, "top": 20, "bottom": 10},
            },
        },
    }

    config_str = json.dumps(chart_config, ensure_ascii=False)
    encoded = urllib.parse.quote(config_str, safe="")
    return f"https://quickchart.io/chart?w=700&h=900&c={encoded}"


def index_bar_url(indices: dict, title: str = "A股主要指数涨跌幅") -> str:
    """指数涨跌柱状图 URL。"""
    if not indices:
        return ""

    names = list(indices.keys())
    values = [d["pct_chg"] for d in indices.values()]
    colors = ["#e74c3c" if v > 0 else "#27ae60" if v < 0 else "#95a5a6" for v in values]

    chart_config = {
        "type": "bar",
        "data": {
            "labels": names,
            "datasets": [{
                "label": "涨跌幅 (%)",
                "data": values,
                "backgroundColor": colors,
            }],
        },
        "options": {
            "title": {"display": True, "text": title, "fontSize": 15},
            "legend": {"display": False},
            "scales": {
                "yAxes": [{
                    "ticks": {
                        "callback": "function(v) { return v.toFixed(1) + '%'; }",
                    },
                }],
            },
            "plugins": {
                "datalabels": {
                    "display": True,
                    "anchor": "end",
                    "align": "end",
                    "color": "#333",
                    "font": {"size": 11},
                    "formatter": "function(v) { return v.toFixed(2) + '%'; }",
                },
            },
        },
    }

    config_str = json.dumps(chart_config, ensure_ascii=False)
    encoded = urllib.parse.quote(config_str, safe="")
    return f"https://quickchart.io/chart?w=600&h=400&c={encoded}"
