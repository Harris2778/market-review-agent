"""
数据采集器：市场行情 + 全球指数 + 新闻 + 宏观数据。

设计原则：
- 每个数据源独立封装，失败时返回 None 而不是抛异常
- 支持降级：API → 公开数据 → 标记不可用
- 所有时间戳统一为北京时间
"""

import os
import json
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Any
from dataclasses import dataclass, field

import requests
import pandas as pd

# ── 配置（运行时读取，非导入时） ──

def _token(key: str, default: str = "") -> str:
    return os.getenv(key, default)


@dataclass
class MarketSnapshot:
    """全市场快照。"""
    date: str = ""
    indices: dict = field(default_factory=dict)     # {指数名: {close, pct_chg, vol}}
    global_indices: dict = field(default_factory=dict)
    sectors: list = field(default_factory=list)       # [{name, pct_chg, tag}]
    fund_flows: dict = field(default_factory=dict)     # {north_bound, south_bound, margin_bal, turnover}
    news_items: list = field(default_factory=list)    # [{tier, time, source, title, impact, sector}]
    calendar: list = field(default_factory=list)       # [{date, event, importance}]
    macro_data: dict = field(default_factory=dict)     # {指标名: 数值}
    errors: list = field(default_factory=list)         # 各数据源错误信息


# ── Tushare: A股数据 ──

def _get_tushare_pro():
    """获取 Tushare Pro 连接。"""
    if not _token("TUSHARE_TOKEN"):
        return None
    try:
        import tushare as ts
        ts.set_token(_token("TUSHARE_TOKEN"))
        return ts.pro_api()
    except Exception:
        return None


def fetch_a_share_indices(date: str) -> dict:
    """获取A股主要指数行情。"""
    pro = _get_tushare_pro()
    if not pro:
        return {}

    index_codes = {
        "000001.SH": "上证综指", "399001.SZ": "深证成指",
        "399006.SZ": "创业板指", "000688.SH": "科创50",
        "000300.SH": "沪深300", "000905.SH": "中证500",
        "000852.SH": "中证1000",
    }
    result = {}
    try:
        df = pro.index_daily(ts_code=",".join(index_codes.keys()),
                             start_date=date, end_date=date)
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                name = index_codes.get(row["ts_code"], row["ts_code"])
                result[name] = {
                    "close": round(float(row["close"]), 2),
                    "pct_chg": round(float(row["pct_chg"]), 2),
                    "vol": round(float(row.get("vol", 0)) / 10000, 2),
                }
    except Exception:
        pass
    return result


# 申万一级31行业指数代码
SW_SECTOR_CODES = {
    "801010.SI": "农林牧渔", "801020.SI": "采掘", "801030.SI": "化工",
    "801040.SI": "钢铁", "801050.SI": "有色金属", "801080.SI": "电子",
    "801110.SI": "家用电器", "801120.SI": "食品饮料", "801130.SI": "纺织服装",
    "801140.SI": "轻工制造", "801150.SI": "医药生物", "801160.SI": "公用事业",
    "801170.SI": "交通运输", "801180.SI": "房地产", "801200.SI": "商业贸易",
    "801210.SI": "休闲服务", "801230.SI": "综合", "801710.SI": "建筑材料",
    "801720.SI": "建筑装饰", "801730.SI": "电气设备", "801740.SI": "国防军工",
    "801750.SI": "计算机", "801760.SI": "传媒", "801770.SI": "通信",
    "801780.SI": "银行", "801790.SI": "非银金融", "801880.SI": "汽车",
    "801890.SI": "机械设备", "801950.SI": "煤炭", "801960.SI": "石油石化",
    "801970.SI": "环保",
}


def fetch_shenwan_sectors(date: str) -> list:
    """获取申万一级31行业涨跌幅。用 index_daily 接口（免费版也可用）。"""
    pro = _get_tushare_pro()
    if not pro:
        return []

    # 方式1: 用 index_daily 批量查申万行业指数（单次调用）
    try:
        all_codes = ",".join(SW_SECTOR_CODES.keys())
        df = pro.index_daily(ts_code=all_codes, start_date=date, end_date=date)
        if df is not None and not df.empty:
            sectors = []
            for _, row in df.iterrows():
                name = SW_SECTOR_CODES.get(row["ts_code"], row["ts_code"])
                pct = round(float(row["pct_chg"]), 2)
                if pct > 2:
                    tag = "强势"
                elif pct > 1:
                    tag = "偏强"
                elif pct >= -1:
                    tag = "中性"
                elif pct >= -2:
                    tag = "偏弱"
                else:
                    tag = "弱势"
                sectors.append({"name": name, "pct_chg": pct, "tag": tag})
            sectors.sort(key=lambda x: x["pct_chg"], reverse=True)
            return sectors
    except Exception:
        pass

    # 方式2: 尝试 sw_daily（需要更高权限）
    try:
        df = pro.sw_daily(trade_date=date)
        if df is not None and not df.empty:
            sectors = []
            for _, row in df.iterrows():
                pct = round(float(row["pct_chg"]), 2)
                tag = "强势" if pct > 2 else "偏强" if pct > 1 else "中性" if pct >= -1 else "偏弱" if pct >= -2 else "弱势"
                sectors.append({"name": row["sw_name"], "pct_chg": pct, "tag": tag})
            sectors.sort(key=lambda x: x["pct_chg"], reverse=True)
            return sectors
    except Exception:
        pass

    return []


def fetch_fund_flows(date: str) -> dict:
    """获取资金流向：北向/南向/融资融券。"""
    pro = _get_tushare_pro()
    if not pro:
        return {}

    result = {}
    try:
        df = pro.moneyflow_hsgt(start_date=date, end_date=date)
        if df is not None and not df.empty:
            row = df.iloc[0]
            result["north_bound"] = round(float(row.get("north_net_inflow", 0)), 2)
            result["south_bound"] = round(float(row.get("south_net_inflow", 0)), 2)
    except Exception:
        pass

    try:
        df_m = pro.margin(trade_date=date)
        if df_m is not None and not df_m.empty:
            result["margin_bal"] = round(float(df_m["rzye"].sum()) / 1e8, 2)
    except Exception:
        pass

    # 成交额直接从指数数据取，这里设个默认
    if "turnover" not in result:
        result["turnover"] = None
    return result


def fetch_sector_detail(sector_name: str, date: str) -> dict:
    """获取单板块深度数据：成分股表现。"""
    pro = _get_tushare_pro()
    if not pro:
        return {}

    result = {"name": sector_name}
    try:
        df_member = pro.sw_member(sw_name=sector_name)
        if df_member is not None and not df_member.empty:
            stocks = df_member["ts_code"].tolist()[:30]
            df_stock = pro.daily(ts_code=",".join(stocks),
                                start_date=date, end_date=date)
            if df_stock is not None and not df_stock.empty:
                df_sorted = df_stock.sort_values("pct_chg", ascending=False)
                result["top_gainers"] = [
                    {"name": r["ts_code"], "pct_chg": round(float(r["pct_chg"]), 2)}
                    for _, r in df_sorted.head(5).iterrows()
                ]
                result["top_losers"] = [
                    {"name": r["ts_code"], "pct_chg": round(float(r["pct_chg"]), 2)}
                    for _, r in df_sorted.tail(5).iterrows()
                ]
                result["up_count"] = int((df_stock["pct_chg"] > 0).sum())
                result["down_count"] = int((df_stock["pct_chg"] < 0).sum())
                result["flat_count"] = int((df_stock["pct_chg"] == 0).sum())
    except Exception:
        pass
    return result


# ── Finnhub: 全球指数 ──

def fetch_global_indices() -> dict:
    """获取全球主要指数行情。"""
    if not _token("FINNHUB_API_KEY"):
        return {}

    result = {}
    indices = {
        "^GSPC": "标普500", "^DJI": "道琼斯工业", "^IXIC": "纳斯达克",
        "^HSI": "恒生指数", "^N225": "日经225", "^FTSE": "富时100",
    }
    try:
        import finnhub
        client = finnhub.Client(api_key=_token("FINNHUB_API_KEY"))
        for symbol, name in indices.items():
            try:
                quote = client.quote(symbol)
                if quote and quote.get("c"):
                    prev = quote.get("pc", quote["c"])
                    pct = round((quote["c"] - prev) / prev * 100, 2) if prev else 0
                    result[name] = {"close": round(quote["c"], 2), "pct_chg": pct}
            except Exception:
                pass
    except Exception:
        pass
    return result


# ── FRED: 美国宏观 ──

def fetch_us_macro() -> dict:
    """获取美国关键宏观数据。"""
    if not _token("FRED_API_KEY"):
        return {}

    result = {}
    series = {
        "DFF": "联邦基金利率", "DGS10": "10Y美债收益率",
        "DGS2": "2Y美债收益率", "T10Y2Y": "10Y-2Y利差",
        "VIXCLS": "VIX恐慌指数",
    }
    try:
        from fredapi import Fred
        fred = Fred(api_key=_token("FRED_API_KEY"))
        today = datetime.now().strftime("%Y-%m-%d")
        for sid, name in series.items():
            try:
                data = fred.get_series(sid, observation_start="2026-01-01")
                if data is not None and len(data) > 0:
                    result[name] = round(float(data.iloc[-1]), 4)
            except Exception:
                pass
    except Exception:
        pass
    return result


# ── 新闻采集 ──

def fetch_cls_news(limit: int = 20) -> list:
    """从财联社采集实时电报新闻。免费，无需API Key。"""
    items = []
    try:
        # 财联社公开接口
        url = "https://www.cls.cn/api/sw?app=CailianpressWeb&os=web&sv=8.4.6"
        resp = requests.post(url, json={
            "type": "telegram", "page": 1, "limit": limit,
            "last_time": int(datetime.now().timestamp()),
        }, headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.cls.cn/telegraph",
        }, timeout=15)
        data = resp.json()
        for item in data.get("data", {}).get("roll_data", [])[:limit]:
            items.append({
                "source": "财联社",
                "time": item.get("ctime", ""),
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "brief": item.get("brief", ""),
            })
    except Exception:
        pass
    return items


def fetch_finnhub_news(limit: int = 10) -> list:
    """从 Finnhub 获取全球财经新闻。免费层60次/分钟。"""
    if not _token("FINNHUB_API_KEY"):
        return []

    items = []
    try:
        import finnhub
        client = finnhub.Client(api_key=_token("FINNHUB_API_KEY"))
        news = client.general_news("general", min_id=0)
        for item in news[:limit]:
            items.append({
                "source": item.get("source", "Finnhub"),
                "time": datetime.fromtimestamp(item.get("datetime", 0)).strftime("%H:%M"),
                "title": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
            })
    except Exception:
        pass
    return items


def search_financial_news(query: str, limit: int = 8) -> list:
    """搜索金融新闻。用 Brave Search API（免费2000次/月）或降级到公开源。"""
    items = []
    if _token("BRAVE_SEARCH_API_KEY"):
        try:
            resp = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": f"{query} 今日", "count": limit, "freshness": "pd"},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": _token("BRAVE_SEARCH_API_KEY"),
                },
                timeout=15,
            )
            data = resp.json()
            for r in data.get("web", {}).get("results", [])[:limit]:
                items.append({
                    "source": "Brave Search",
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "description": r.get("description", ""),
                })
        except Exception:
            pass
    return items


# ── 经济日历 ──

def fetch_economic_calendar() -> list:
    """获取近期经济事件日历。从公开源获取。"""
    items = []
    try:
        # 金十数据公开接口
        resp = requests.get(
            "https://cdn-rili.jin10.com/web_data/2026/daily/00/en.json",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            today = datetime.now().strftime("%Y%m%d")
            tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
            for date_key in [today, tomorrow]:
                for event in data.get(date_key, [])[:8]:
                    items.append({
                        "date": date_key,
                        "time": event.get("time", ""),
                        "event": event.get("name", event.get("title", "")),
                        "country": event.get("country", ""),
                        "importance": event.get("star", event.get("importance", 1)),
                        "previous": event.get("previous", ""),
                        "consensus": event.get("consensus", ""),
                    })
    except Exception:
        pass
    return items


# ── 综合采集 ──

async def collect_market_snapshot(
    date: Optional[str] = None,
    sector_focus: Optional[str] = None,
) -> MarketSnapshot:
    """并行采集全市场数据，返回聚合快照。"""
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    snapshot = MarketSnapshot(date=date)

    # 并行执行所有数据采集（用线程池，因为都是IO操作）
    loop = asyncio.get_event_loop()

    # A股数据
    indices = await loop.run_in_executor(None, fetch_a_share_indices, date)
    snapshot.indices = indices

    # 申万行业
    sectors = await loop.run_in_executor(None, fetch_shenwan_sectors, date)
    snapshot.sectors = sectors

    # 资金流向
    flows = await loop.run_in_executor(None, fetch_fund_flows, date)
    snapshot.fund_flows = flows

    # 全球指数
    global_idx = await loop.run_in_executor(None, fetch_global_indices)
    snapshot.global_indices = global_idx

    # 美国宏观
    macro = await loop.run_in_executor(None, fetch_us_macro)
    snapshot.macro_data = macro

    # 新闻
    cls_news = await loop.run_in_executor(None, fetch_cls_news, 15)
    fh_news = await loop.run_in_executor(None, fetch_finnhub_news, 8)
    snapshot.news_items = {
        "cls": cls_news,
        "global": fh_news,
    }

    # 经济日历
    calendar = await loop.run_in_executor(None, fetch_economic_calendar)
    snapshot.calendar = calendar

    # 单板块深度数据
    if sector_focus:
        sector_detail = await loop.run_in_executor(
            None, fetch_sector_detail, sector_focus, date
        )
        snapshot._sector_detail = sector_detail

    return snapshot


def format_market_data_for_prompt(snapshot: MarketSnapshot) -> str:
    """将市场数据格式化为注入 LLM prompt 的文本。"""
    lines = []
    lines.append(f"## 实时市场数据（采集时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}）\n")

    # 指数
    if snapshot.indices:
        lines.append("### A股主要指数")
        for name, data in snapshot.indices.items():
            lines.append(f"- {name}: {data['close']} | {data['pct_chg']:+.2f}% | 成交额{data['vol']}亿")
    else:
        lines.append("### A股主要指数\n[Tushare未配置，数据不可用]")

    lines.append("")

    # 申万行业
    if snapshot.sectors:
        lines.append("### 申万一级行业涨跌幅（共{}个行业）".format(len(snapshot.sectors)))
        strong = [s for s in snapshot.sectors if s["tag"] in ("强势", "偏强")]
        weak = [s for s in snapshot.sectors if s["tag"] in ("弱势", "偏弱")]
        if strong:
            strong_str = ", ".join(f"{s['name']}({s['pct_chg']:+.2f}%)" for s in strong[:8])
            lines.append(f"领涨：{strong_str}")
        if weak:
            weak_str = ", ".join(f"{s['name']}({s['pct_chg']:+.2f}%)" for s in weak[:8])
            lines.append(f"领跌：{weak_str}")
        lines.append("完整31行业：")
        for s in snapshot.sectors:
            lines.append(f"  {s['name']}: {s['pct_chg']:+.2f}% [{s['tag']}]")
    else:
        lines.append("### 申万行业\n[Tushare未配置，数据不可用]")

    lines.append("")

    # 全球指数
    if snapshot.global_indices:
        lines.append("### 全球主要指数")
        for name, data in snapshot.global_indices.items():
            lines.append(f"- {name}: {data['close']} | {data['pct_chg']:+.2f}%")

    lines.append("")

    # 资金流向
    if snapshot.fund_flows:
        lines.append("### 资金面")
        if "north_bound" in snapshot.fund_flows:
            lines.append(f"- 北向资金: {snapshot.fund_flows['north_bound']:+.2f}亿")
        if "south_bound" in snapshot.fund_flows:
            lines.append(f"- 南向资金: {snapshot.fund_flows['south_bound']:+.2f}亿")
        if "margin_bal" in snapshot.fund_flows:
            lines.append(f"- 融资余额: {snapshot.fund_flows['margin_bal']:.2f}亿")

    lines.append("")

    # 财联社新闻
    if snapshot.news_items.get("cls"):
        lines.append("### 财联社今日电报（最新15条）")
        for item in snapshot.news_items["cls"][:15]:
            lines.append(f"- [{item['time']}] {item['title']}")
            if item.get("brief"):
                lines.append(f"  {item['brief'][:150]}")

    lines.append("")

    # 全球新闻
    if snapshot.news_items.get("global"):
        lines.append("### 全球财经新闻")
        for item in snapshot.news_items["global"][:8]:
            lines.append(f"- [{item['source']} {item['time']}] {item['title']}")

    lines.append("")

    # 经济日历
    if snapshot.calendar:
        lines.append("### 近期经济事件")
        for ev in snapshot.calendar[:10]:
            stars = "*" * ev.get("importance", 1)
            lines.append(f"- {ev['date']} {ev['time']} [{stars}] {ev['event']} ({ev.get('country','')})")

    lines.append("")

    # 美国宏观
    if snapshot.macro_data:
        lines.append("### 美国宏观数据")
        for name, val in snapshot.macro_data.items():
            lines.append(f"- {name}: {val}")

    return "\n".join(lines)
