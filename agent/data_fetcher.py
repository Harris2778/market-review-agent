"""
数据采集器 v2.0：行情 + 趋势 + 新闻 + 宏观 + 量价分析。

设计原则：
- 每个数据源独立封装，失败返回空而非抛异常
- 支持降级：API → 公开数据 → 标记不可用
- 所有时间戳统一为北京时间
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


# ── 数据结构 ──

@dataclass
class MarketSnapshot:
    date: str = ""
    indices: dict = field(default_factory=dict)
    sectors: list = field(default_factory=list)
    fund_flows: dict = field(default_factory=dict)
    global_indices: dict = field(default_factory=dict)
    macro_data: dict = field(default_factory=dict)
    news_items: list = field(default_factory=list)
    calendar: list = field(default_factory=list)


# ── Tushare 连接 ──

def _get_pro():
    token = _env("TUSHARE_TOKEN")
    if not token:
        return None
    try:
        import tushare as ts
        ts.set_token(token)
        return ts.pro_api()
    except Exception:
        return None


# ═══════════════════════════════════════════
# A 股行情 + 历史趋势 + 均线 + 量价
# ═══════════════════════════════════════════

A_INDEX_CODES = {
    "000001.SH": "上证综指", "399001.SZ": "深证成指",
    "399006.SZ": "创业板指", "000688.SH": "科创50",
    "000300.SH": "沪深300", "000905.SH": "中证500",
    "000852.SH": "中证1000",
}

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


def fetch_a_share_indices(date: str) -> dict:
    """
    获取A股7大指数行情 + 5日/20日趋势 + 均线位置 + 量比。
    拉取近60天数据用于计算均线和历史趋势。
    """
    pro = _get_pro()
    if not pro:
        return {}

    start_60d = (datetime.strptime(date, "%Y%m%d") - timedelta(days=90)).strftime("%Y%m%d")
    result = {}

    for code, name in A_INDEX_CODES.items():
        try:
            df = pro.index_daily(ts_code=code, start_date=start_60d, end_date=date)
            if df is None or df.empty:
                continue
            df = df.sort_values("trade_date")
            latest = df.iloc[-1]
            close = float(latest["close"])
            pct_chg = float(latest["pct_chg"])
            vol = float(latest.get("vol", 0)) / 10000

            closes = df["close"].astype(float)
            vols = df["vol"].astype(float)

            # 均线
            ma5 = round(float(closes.tail(5).mean()), 2) if len(closes) >= 5 else None
            ma10 = round(float(closes.tail(10).mean()), 2) if len(closes) >= 10 else None
            ma20 = round(float(closes.tail(20).mean()), 2) if len(closes) >= 20 else None
            ma60 = round(float(closes.tail(60).mean()), 2) if len(closes) >= 60 else None

            # 历史涨跌
            def _chg_from(days_ago):
                if len(closes) > days_ago:
                    prev = float(closes.iloc[-(days_ago + 1)])
                    return round((close - prev) / prev * 100, 2)
                return None

            chg_5d = _chg_from(5)
            chg_20d = _chg_from(20)

            # 量比（今日量 / 5日均量）
            avg_vol_5d = float(vols.tail(6).head(5).mean()) if len(vols) >= 6 else vol
            vol_ratio = round(vol / (avg_vol_5d / 10000), 2) if avg_vol_5d > 0 else 1.0

            # 趋势方向
            trend_5d = "上涨" if chg_5d and chg_5d > 0 else "下跌" if chg_5d else "—"
            trend_20d = "上涨" if chg_20d and chg_20d > 0 else "下跌" if chg_20d else "—"

            # 均线排列
            mas = [m for m in [ma5, ma10, ma20, ma60] if m is not None]
            above_mas = sum(1 for m in mas if close > m)
            total_mas = len(mas)

            result[name] = {
                "close": round(close, 2),
                "pct_chg": round(pct_chg, 2),
                "vol": round(vol, 2),
                "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60,
                "chg_5d": chg_5d, "chg_20d": chg_20d,
                "trend_5d": trend_5d, "trend_20d": trend_20d,
                "vol_ratio": vol_ratio,
                "mas_above": f"{above_mas}/{total_mas}" if total_mas else "—",
            }
        except Exception:
            pass
    return result


def fetch_shenwan_sectors(date: str) -> list:
    """获取31行业涨跌幅 + 5日趋势。"""
    pro = _get_pro()
    if not pro:
        return []

    start_10d = (datetime.strptime(date, "%Y%m%d") - timedelta(days=15)).strftime("%Y%m%d")
    sectors = []

    for code, name in SW_SECTOR_CODES.items():
        try:
            df = pro.index_daily(ts_code=code, start_date=start_10d, end_date=date)
            if df is None or df.empty:
                continue
            df = df.sort_values("trade_date")
            row = df.iloc[-1]
            pct = round(float(row["pct_chg"]), 2)

            # 标签
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

            # 5日趋势
            closes = df["close"].astype(float)
            chg_5d = None
            if len(closes) > 5:
                prev_5 = float(closes.iloc[-6])
                chg_5d = round((float(closes.iloc[-1]) - prev_5) / prev_5 * 100, 2)

            # 连续涨跌天数
            streak = 0
            streak_dir = ""
            for i in range(len(df) - 1, 0, -1):
                if float(df.iloc[i]["pct_chg"]) > 0:
                    if streak_dir == "":
                        streak_dir = "涨"
                    if streak_dir == "涨":
                        streak += 1
                    else:
                        break
                elif float(df.iloc[i]["pct_chg"]) < 0:
                    if streak_dir == "":
                        streak_dir = "跌"
                    if streak_dir == "跌":
                        streak += 1
                    else:
                        break
                else:
                    break

            streak_str = f"连{streak_dir}{streak}天" if streak >= 2 else "—"

            sectors.append({
                "name": name, "pct_chg": pct, "tag": tag,
                "chg_5d": chg_5d, "streak": streak_str,
            })
        except Exception:
            pass

    sectors.sort(key=lambda x: x["pct_chg"], reverse=True)
    return sectors


def fetch_fund_flows(date: str) -> dict:
    """资金流向：北向/南向/两融。"""
    pro = _get_pro()
    if not pro:
        return {}

    result = {}
    # 北向/南向（万元 → 亿）
    try:
        df = pro.moneyflow_hsgt(start_date=date, end_date=date)
        if df is not None and not df.empty:
            row = df.iloc[0]
            nm = float(row.get("north_money", 0))
            sm = float(row.get("south_money", 0))
            if nm > 0:
                result["north_bound"] = round(nm / 10000, 2)
            if sm > 0:
                result["south_bound"] = round(sm / 10000, 2)
    except Exception:
        try:
            df = pro.moneyflow_hsgt(trade_date=date)
            if df is not None and not df.empty:
                row = df.iloc[0]
                nm = float(row.get("north_money", 0))
                sm = float(row.get("south_money", 0))
                if nm > 0:
                    result["north_bound"] = round(nm / 10000, 2)
                if sm > 0:
                    result["south_bound"] = round(sm / 10000, 2)
        except Exception:
            pass

    # 融资融券
    try:
        df_m = pro.margin(trade_date=date)
        if df_m is not None and not df_m.empty:
            result["margin_bal"] = round(float(df_m["rzye"].sum()) / 1e8, 2)
    except Exception:
        pass

    return result


# ═══════════════════════════════════════════
# 新闻采集（Tushare 新闻 + Finnhub）
# ═══════════════════════════════════════════

def fetch_tushare_news(date: str, limit: int = 25) -> list:
    """从 Tushare 获取主流财经新闻。"""
    pro = _get_pro()
    if not pro:
        return []

    items = []
    try:
        df = pro.news(src="cls", start_date=date, end_date=date)
        if df is not None and not df.empty:
            for _, row in df.head(limit).iterrows():
                items.append({
                    "source": "财联社",
                    "time": str(row.get("datetime", "")),
                    "title": str(row.get("title", "")),
                    "content": str(row.get("content", ""))[:300],
                })
    except Exception:
        pass

    # 如果 CLS 源为空，尝试其他源
    if not items:
        try:
            df = pro.news(src="sina", start_date=date, end_date=date)
            if df is not None and not df.empty:
                for _, row in df.head(limit).iterrows():
                    items.append({
                        "source": "新浪财经",
                        "time": str(row.get("datetime", "")),
                        "title": str(row.get("title", "")),
                        "content": str(row.get("content", ""))[:300],
                    })
        except Exception:
            pass

    return items


def fetch_finnhub_news(limit: int = 10) -> list:
    """Finnhub 全球财经新闻。"""
    key = _env("FINNHUB_API_KEY")
    if not key:
        return []

    items = []
    try:
        import finnhub
        client = finnhub.Client(api_key=key)
        news = client.general_news("general", min_id=0)
        for item in news[:limit]:
            ts = item.get("datetime", 0)
            time_str = datetime.fromtimestamp(ts).strftime("%H:%M") if ts else ""
            items.append({
                "source": item.get("source", "Finnhub"),
                "time": time_str,
                "title": item.get("headline", ""),
                "summary": item.get("summary", ""),
            })
    except Exception:
        pass
    return items


def fetch_eastmoney_news(limit: int = 25) -> list:
    """东方财富 7x24 实时快讯（免费，质量高，带完整摘要）。"""
    items = []
    try:
        import uuid as _uuid
        url = "https://np-weblist.eastmoney.com/comm/web/getFastNewsList"
        params = {
            "client": "web", "biz": "fastnews", "fastColumn": "102",
            "sortEnd": "", "pageSize": limit, "pageIndex": 1,
            "req_trace": str(_uuid.uuid4()).replace("-", "")[:32],
        }
        resp = requests.get(url, params=params, timeout=15,
                          headers={"User-Agent": "Mozilla/5.0",
                                   "Referer": "https://www.eastmoney.com/"})
        data = resp.json()
        news_list = data.get("data", {}).get("fastNewsList", [])
        for item in news_list[:limit]:
            items.append({
                "source": "东方财富",
                "time": item.get("showTime", ""),
                "title": item.get("title", "")[:150],
                "summary": item.get("summary", "")[:300] if item.get("summary") else "",
            })
    except Exception:
        pass
    return items


def fetch_sina_news(limit: int = 20) -> list:
    """新浪财经滚动新闻（备用）。"""
    items = []
    try:
        import re
        url = "https://feed.mix.sina.com.cn/api/roll/get"
        params = {"pageid": 153, "lid": 2509, "k": "", "num": limit, "page": 1}
        resp = requests.get(url, params=params, timeout=15,
                          headers={"User-Agent": "Mozilla/5.0"})
        data = resp.json()
        for item in data.get("result", {}).get("data", [])[:limit]:
            title = re.sub(r'<[^>]+>', '', item.get("title", ""))
            ctime_raw = item.get("ctime", "")
            try:
                ts = int(ctime_raw)
                time_str = datetime.fromtimestamp(ts).strftime("%H:%M")
            except Exception:
                time_str = str(ctime_raw)
            items.append({
                "source": "新浪财经",
                "time": time_str,
                "title": title[:150],
            })
    except Exception:
        pass
    return items


def fetch_cls_telegraph(limit: int = 20) -> list:
    """财联社电报（备用）。"""
    items = []
    try:
        resp = requests.post(
            "https://www.cls.cn/api/sw",
            json={"type": "telegram", "page": 1, "limit": limit,
                  "last_time": int(datetime.now().timestamp())},
            headers={"Content-Type": "application/json", "Referer": "https://www.cls.cn/telegraph"},
            timeout=15,
        )
        data = resp.json()
        for item in data.get("data", {}).get("roll_data", [])[:limit]:
            items.append({
                "source": "财联社电报",
                "time": item.get("ctime", ""),
                "title": item.get("title", ""),
                "brief": str(item.get("brief", ""))[:200],
            })
    except Exception:
        pass
    return items


# ═══════════════════════════════════════════
# 中国宏观数据
# ═══════════════════════════════════════════

def fetch_china_macro() -> dict:
    """中国宏观数据：CPI/PPI/PMI/M2/GDP/社融（Tushare）。"""
    pro = _get_pro()
    if not pro:
        return {}

    result = {}
    this_month = datetime.now().strftime("%Y%m")

    # CPI
    try:
        df = pro.cn_cpi(start_m="202601", end_m=this_month)
        if df is not None and not df.empty:
            row = df.iloc[-1]
            result["CPI同比"] = f"{float(row['nt_yoy']):.2f}%"
            result["CPI环比"] = f"{float(row['nt_mom']):.2f}%"
    except Exception:
        pass

    # PPI
    try:
        df = pro.cn_ppi(start_m="202601", end_m=this_month)
        if df is not None and not df.empty:
            row = df.iloc[-1]
            result["PPI同比"] = f"{float(row['ppi_yoy']):.2f}%"
    except Exception:
        pass

    # PMI
    try:
        df = pro.cn_pmi(start_m="202601", end_m=this_month)
        if df is not None and not df.empty:
            row = df.iloc[-1]
            col0 = df.columns[0]
            result["制造业PMI"] = f"{float(row[col0]):.1f}"
    except Exception:
        pass

    # M2
    try:
        df = pro.cn_m(start_m="202601", end_m=this_month)
        if df is not None and not df.empty:
            row = df.iloc[-1]
            result["M2同比"] = f"{float(row['m2_yoy']):.2f}%"
            result["M2余额"] = f"{float(row['m2']):.2f}万亿"  # M2绝对值作为背景
    except Exception:
        pass

    # GDP
    try:
        df = pro.cn_gdp(start_q="2025Q1", end_q="2026Q2")
        if df is not None and not df.empty:
            row = df.iloc[-1]
            result["GDP同比"] = f"{float(row['gdp_yoy']):.2f}%"
            result["GDP季度"] = str(row.get("quarter", ""))
    except Exception:
        pass

    # 社融
    try:
        df = pro.sf_month(start_m="202601", end_m=this_month)
        if df is not None and not df.empty:
            row = df.iloc[-1]
            val = float(row.get("inc_month", 0))
            result["社融当月新增"] = f"{val / 10000:.2f}万亿" if val > 10000 else f"{val:.2f}亿"
    except Exception:
        pass

    return result


# ═══════════════════════════════════════════
# 全球指数 + 美国宏观
# ═══════════════════════════════════════════

def fetch_global_indices() -> dict:
    """全球主要指数（yfinance，免费）。"""
    result = {}
    indices = [
        ("^GSPC", "标普500"), ("^DJI", "道琼斯工业"), ("^IXIC", "纳斯达克"),
        ("^HSI", "恒生指数"), ("^N225", "日经225"), ("^FTSE", "富时100"),
        ("EURUSD=X", "欧元/美元"),
    ]
    try:
        import yfinance as yf
        for symbol, name in indices:
            try:
                t = yf.Ticker(symbol)
                hist = t.history(period="2d")
                if hist is not None and len(hist) >= 1:
                    close = round(float(hist["Close"].iloc[-1]), 2)
                    prev = round(float(hist["Close"].iloc[-2]), 2) if len(hist) >= 2 else close
                    pct = round((close - prev) / prev * 100, 2) if prev else 0.0
                    result[name] = {"close": close, "pct_chg": pct}
            except Exception:
                pass
    except Exception:
        pass
    return result


def fetch_us_macro() -> dict:
    """FRED 美国宏观数据。"""
    key = _env("FRED_API_KEY")
    if not key:
        return {}

    result = {}
    series = {
        "DFF": "联邦基金利率", "DGS10": "10Y美债收益率",
        "DGS2": "2Y美债收益率", "T10Y2Y": "10Y-2Y利差",
        "VIXCLS": "VIX恐慌指数",
    }
    try:
        from fredapi import Fred
        fred = Fred(api_key=key)
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


# ═══════════════════════════════════════════
# 经济日历
# ═══════════════════════════════════════════

def fetch_economic_calendar() -> list:
    """近期经济事件。"""
    items = []
    try:
        resp = requests.get(
            "https://cdn-rili.jin10.com/web_data/2026/daily/00/en.json",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            today = datetime.now().strftime("%Y%m%d")
            tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
            for dk in [today, tomorrow]:
                for ev in data.get(dk, [])[:8]:
                    items.append({
                        "date": dk, "time": ev.get("time", ""),
                        "event": ev.get("name", ev.get("title", "")),
                        "country": ev.get("country", ""),
                        "importance": ev.get("star", 1),
                    })
    except Exception:
        pass
    return items


# ═══════════════════════════════════════════
# 综合采集 + 格式化
# ═══════════════════════════════════════════

async def collect_market_snapshot(
    date: Optional[str] = None,
    sector_focus: Optional[str] = None,
) -> MarketSnapshot:
    """并行采集全市场数据。"""
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    import asyncio
    loop = asyncio.get_event_loop()

    snapshot = MarketSnapshot(date=date)

    # 并行执行
    indices = await loop.run_in_executor(None, fetch_a_share_indices, date)
    snapshot.indices = indices

    sectors = await loop.run_in_executor(None, fetch_shenwan_sectors, date)
    snapshot.sectors = sectors

    flows = await loop.run_in_executor(None, fetch_fund_flows, date)
    snapshot.fund_flows = flows

    gidx = await loop.run_in_executor(None, fetch_global_indices)
    snapshot.global_indices = gidx

    china_macro = await loop.run_in_executor(None, fetch_china_macro)
    us_macro = await loop.run_in_executor(None, fetch_us_macro)
    snapshot.macro_data = {
        "china": china_macro,
        "us": us_macro,
    }

    # 新闻：东方财富（主力）→ 新浪（备用）→ Tushare → Finnhub
    em_news = await loop.run_in_executor(None, fetch_eastmoney_news, 25)
    sina = await loop.run_in_executor(None, fetch_sina_news, 20)
    ts_news = await loop.run_in_executor(None, fetch_tushare_news, date)
    fh_news = await loop.run_in_executor(None, fetch_finnhub_news, 8)
    snapshot.news_items = {
        "eastmoney": em_news,
        "sina": sina,
        "ts_news": ts_news,
        "global": fh_news,
    }

    calendar = await loop.run_in_executor(None, fetch_economic_calendar)
    snapshot.calendar = calendar

    return snapshot


def format_market_data_for_prompt(snapshot: MarketSnapshot) -> str:
    """格式化数据为 LLM prompt。"""
    lines = []
    ts = datetime.now().strftime("%H:%M")
    lines.append(f"## 实时市场数据（采集时间：{ts}）\n")

    # ── A 股指数 ──
    if snapshot.indices:
        lines.append("### A股主要指数（含趋势分析）")
        for name, d in snapshot.indices.items():
            extras = []
            if d.get("chg_5d") is not None:
                extras.append(f"5日{d['chg_5d']:+.2f}%")
            if d.get("chg_20d") is not None:
                extras.append(f"20日{d['chg_20d']:+.2f}%")
            if d.get("vol_ratio"):
                extras.append(f"量比{d['vol_ratio']}x")
            if d.get("mas_above") and d["mas_above"] != "—":
                extras.append(f"站上{d['mas_above']}条均线")
            extra_str = f" | {' | '.join(extras)}" if extras else ""
            lines.append(
                f"- {name}: {d['close']} | {d['pct_chg']:+.2f}%"
                f" | 成交额{d['vol']}亿{extra_str}"
            )
    else:
        lines.append("### A股主要指数\n[Tushare未配置]")

    lines.append("")

    # ── 申万行业 ──
    if snapshot.sectors:
        lines.append(f"### 申万一级行业（共{len(snapshot.sectors)}个）")
        strong = [s for s in snapshot.sectors if s["tag"] in ("强势", "偏强")]
        weak = [s for s in snapshot.sectors if s["tag"] in ("弱势", "偏弱")]
        if strong:
            names = ", ".join(f"{s['name']}({s['pct_chg']:+.2f}%)" for s in strong[:8])
            lines.append(f"偏强：{names}")
        if weak:
            names = ", ".join(f"{s['name']}({s['pct_chg']:+.2f}%)" for s in weak[:8])
            lines.append(f"偏弱：{names}")
        lines.append("完整列表：")
        for s in snapshot.sectors:
            streak = f" [{s['streak']}]" if s.get("streak") and s["streak"] != "—" else ""
            chg5 = f" 5日{s['chg_5d']:+.2f}%" if s.get("chg_5d") is not None else ""
            lines.append(f"  {s['name']}: {s['pct_chg']:+.2f}% [{s['tag']}]{streak}{chg5}")
    else:
        lines.append("### 申万行业\n[Tushare未配置]")

    lines.append("")

    # ── 全球指数 ──
    if snapshot.global_indices:
        lines.append("### 全球主要指数")
        for name, d in snapshot.global_indices.items():
            lines.append(f"- {name}: {d['close']} | {d['pct_chg']:+.2f}%")
    lines.append("")

    # ── 资金面 ──
    if snapshot.fund_flows:
        lines.append("### 资金面")
        if snapshot.fund_flows.get("north_bound"):
            lines.append(f"- 北向资金: {snapshot.fund_flows['north_bound']:+.2f}亿")
        if snapshot.fund_flows.get("south_bound"):
            lines.append(f"- 南向资金: {snapshot.fund_flows['south_bound']:+.2f}亿")
        if snapshot.fund_flows.get("margin_bal"):
            lines.append(f"- 融资余额: {snapshot.fund_flows['margin_bal']:.2f}亿")
    lines.append("")

    # ── 东方财富新闻 ──
    em = snapshot.news_items.get("eastmoney", [])
    if em:
        lines.append(f"### 东方财富 7x24 实时快讯（共{len(em)}条）")
        for item in em[:20]:
            lines.append(f"- [{item['time']}] {item['title']}")
            if item.get("summary"):
                lines.append(f"  {item['summary'][:250]}")

    # ── 新浪新闻（备用）──
    sina = snapshot.news_items.get("sina", [])
    if sina and not em:
        lines.append(f"### 新浪财经新闻（共{len(sina)}条）")
        for item in sina[:15]:
            lines.append(f"- [{item['time']}] {item['title']}")

    # ── Tushare 新闻 ──
    ts_news = snapshot.news_items.get("ts_news", [])
    if ts_news:
        lines.append(f"\n### 财联社新闻（Tushare，共{len(ts_news)}条）")
        for item in ts_news[:20]:
            lines.append(f"- [{item['time']}] {item['title']}")
            if item.get("content"):
                lines.append(f"  {item['content'][:200]}")

    # ── 财联社电报 ──
    cls = snapshot.news_items.get("cls_telegraph", [])
    if cls:
        lines.append(f"\n### 财联社电报（实时，共{len(cls)}条）")
        for item in cls[:15]:
            lines.append(f"- [{item['time']}] {item['title']}")
            if item.get("brief"):
                lines.append(f"  {item['brief'][:150]}")

    lines.append("")

    # ── 全球新闻 ──
    global_news = snapshot.news_items.get("global", [])
    if global_news:
        lines.append("### 全球财经新闻（Finnhub）")
        for item in global_news[:8]:
            lines.append(f"- [{item['source']} {item['time']}] {item['title']}")

    lines.append("")

    # ── 中国宏观 ──
    china = snapshot.macro_data.get("china", {})
    if china:
        lines.append("### 中国宏观数据（Tushare）")
        for name, val in china.items():
            lines.append(f"- {name}: {val}")
        lines.append("")

    # ── 美国宏观 ──
    us = snapshot.macro_data.get("us", {})
    if us:
        lines.append("### 美国宏观数据（FRED）")
        for name, val in us.items():
            lines.append(f"- {name}: {val}")
    lines.append("")

    # ── 日历 ──
    if snapshot.calendar:
        lines.append("### 近期经济事件")
        for ev in snapshot.calendar[:8]:
            stars = "*" * ev.get("importance", 1)
            lines.append(f"- {ev['date']} {ev['time']} [{stars}] {ev['event']}")

    return "\n".join(lines)
