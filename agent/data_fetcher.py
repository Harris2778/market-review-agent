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

    start_30d = (datetime.strptime(date, "%Y%m%d") - timedelta(days=45)).strftime("%Y%m%d")
    sectors = []

    for code, name in SW_SECTOR_CODES.items():
        try:
            df = pro.index_daily(ts_code=code, start_date=start_30d, end_date=date)
            if df is None or df.empty:
                continue
            df = df.sort_values("trade_date")
            row = df.iloc[-1]
            pct = round(float(row["pct_chg"]), 2)

            # 标签
            if pct > 2: tag = "强势"
            elif pct > 1: tag = "偏强"
            elif pct >= -1: tag = "中性"
            elif pct >= -2: tag = "偏弱"
            else: tag = "弱势"

            closes = df["close"].astype(float)

            def _chg(days):
                if len(closes) > days:
                    prev = float(closes.iloc[-(days+1)])
                    return round((float(closes.iloc[-1]) - prev) / prev * 100, 2)
                return None

            chg_5d = _chg(5)
            chg_10d = _chg(10)
            chg_20d = _chg(20)

            # 均线
            ma5 = round(float(closes.tail(5).mean()), 2) if len(closes) >= 5 else None
            ma10 = round(float(closes.tail(10).mean()), 2) if len(closes) >= 10 else None
            ma20 = round(float(closes.tail(20).mean()), 2) if len(closes) >= 20 else None

            # 连续涨跌
            streak = 0
            streak_dir = ""
            for i in range(len(df) - 1, 0, -1):
                if float(df.iloc[i]["pct_chg"]) > 0:
                    if streak_dir == "": streak_dir = "涨"
                    if streak_dir == "涨": streak += 1
                    else: break
                elif float(df.iloc[i]["pct_chg"]) < 0:
                    if streak_dir == "": streak_dir = "跌"
                    if streak_dir == "跌": streak += 1
                    else: break
                else: break

            streak_str = f"连{streak_dir}{streak}天" if streak >= 2 else "—"

            sectors.append({
                "name": name, "pct_chg": pct, "tag": tag,
                "chg_5d": chg_5d, "chg_10d": chg_10d, "chg_20d": chg_20d,
                "ma5": ma5, "ma10": ma10, "ma20": ma20,
                "close": round(float(row["close"]), 2),
                "open": round(float(row.get("open", 0)), 2),
                "high": round(float(row.get("high", 0)), 2),
                "low": round(float(row.get("low", 0)), 2),
                "vol": round(float(row.get("vol", 0)) / 10000, 2),
                "amount": round(float(row.get("amount", 0)) / 1e8, 2),
                "streak": streak_str,
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


# 行业关键词映射（用于新闻板块匹配）
SECTOR_NEWS_KEYWORDS = {
    "食品饮料": ["茅台", "白酒", "食品", "饮料", "乳业", "乳制品", "猪肉", "啤酒", "调味品", "餐饮", "消费", "零食", "预制菜", "五粮液", "伊利", "蒙牛", "海天", "农夫山泉", "山西汾酒", "泸州老窖", "洋河", "古井贡酒", "提价"],
    "电子": ["芯片", "半导体", "存储", "晶圆", "光刻", "华为", "中芯国际", "电子", "电路板", "PCB", "封测", "英伟达", "NVIDIA", "AMD", "英特尔", "海思", "算力", "GPU", "CPU", "服务器", "HBM", "先进封装"],
    "计算机": ["软件", "AI", "人工智能", "大模型", "ChatGPT", "自动驾驶", "算法", "数据", "云计算", "信创", "国产替代", "科大讯飞", "商汤"],
    "电气设备": ["新能源", "光伏", "锂电", "锂电池", "储能", "宁德时代", "比亚迪", "隆基", "通威", "阳光电源", "风电", "硅料", "硅片", "组件", "逆变器", "充电桩", "固态电池"],
    "医药生物": ["医药", "创新药", "CRO", "生物医药", "疫苗", "医疗器械", "恒瑞医药", "迈瑞", "药明", "百济", "PD-1", "基因", "细胞治疗", "GLP-1", "减肥药"],
    "汽车": ["汽车", "新能源车", "整车", "比亚迪", "特斯拉", "蔚来", "小鹏", "理想", "自动驾驶", "智能驾驶", "锂电", "充电", "车市", "乘用车", "商用车"],
    "银行": ["银行", "贷款利率", "存款利率", "息差", "工商银行", "招商银行", "建设银行", "农业银行", "中国银行", "净息差", "降准", "降息", "MLF", "LPR"],
    "非银金融": ["券商", "保险", "证券", "中信证券", "华泰证券", "中国平安", "中国人寿", "投行", "IPO", "再融资", "融资融券"],
    "房地产": ["房地产", "地产", "万科", "保利", "碧桂园", "恒大", "融创", "楼市", "房价", "商品房", "土地出让", "房贷", "公积金", "限购", "城中村"],
    "煤炭": ["煤炭", "煤价", "中国神华", "陕西煤业", "动力煤", "焦煤", "煤矿"],
    "石油石化": ["石油", "石化", "原油", "成品油", "中国石油", "中国石化", "中海油", "油价", "OPEC", "钻井"],
    "有色金属": ["有色", "铜价", "铝价", "黄金", "稀土", "锂矿", "镍", "锌", "紫金矿业", "赣锋锂业", "天齐锂业", "洛阳钼业"],
    "国防军工": ["军工", "导弹", "战斗机", "航母", "航天", "军机", "航发", "中航", "兵器", "国防"],
    "传媒": ["游戏", "电影", "院线", "短剧", "直播", "广告", "出版", "媒体", "互联网", "视频", "影视", "票房", "综艺", "网剧", "抖音", "快手"],
    "公用事业": ["电力", "水务", "燃气", "环保", "长江电力", "华能国际", "新能源发电", "电价", "绿电", "碳排放"],
    "交通运输": ["航运", "物流", "快递", "港口", "铁路", "高速", "中远海控", "顺丰", "航空", "机场", "集装箱", "运价", "波罗的海"],
    "机械设备": ["机械", "工程机械", "机器人", "三一重工", "挖掘机", "机床", "工业母机", "自动化", "人形机器人"],
    "钢铁": ["钢铁", "钢价", "宝钢", "铁矿石", "螺纹钢", "热卷", "冷轧", "钢厂"],
    "化工": ["化工", "万华化学", "MDI", "化肥", "农药", "化纤", "聚酯", "乙烯", "丙烯", "甲醇"],
    "建筑材料": ["水泥", "玻璃", "建材", "海螺水泥", "东方雨虹", "石膏板", "防水"],
    "农林牧渔": ["农业", "种业", "养殖", "猪肉", "粮食", "转基因", "饲料", "渔业", "牧原", "温氏", "新希望"],
    "通信": ["5G", "6G", "通信", "光模块", "光纤", "中兴通讯", "运营商", "中国移动", "卫星通信", "光通信"],
    "纺织服装": ["纺织", "服装", "鞋帽", "运动品牌", "安踏", "李宁", "耐克", "代工"],
    "轻工制造": ["造纸", "家居", "包装", "印刷", "太阳纸业", "欧派", "顾家"],
    "家用电器": ["家电", "空调", "冰箱", "洗衣机", "美的", "格力", "海尔", "扫地机", "黑电", "白电"],
    "商业贸易": ["零售", "超市", "免税", "电商", "跨境电商", "百货", "中国中免", "王府井", "拼多多", "京东", "阿里"],
    "休闲服务": ["旅游", "酒店", "免税", "景区", "出境游", "锦江", "华住"],
    "建筑装饰": ["建筑", "基建", "中国建筑", "中国中铁", "城投", "专项债", "PPP", "一带一路"],
    "综合": [],
    "环保": ["环保", "碳中和", "碳达峰", "污水处理", "垃圾焚烧"],
}


def filter_news_by_sector(news_items: list, sector: str) -> list:
    """按行业关键词过滤新闻。直接匹配的排前面，间接的排后面。"""
    keywords = SECTOR_NEWS_KEYWORDS.get(sector, [])
    if not keywords:
        return news_items[:30]  # 无关键词的行业返回前30条

    direct, indirect = [], []
    for item in news_items:
        title = item.get("title", "")
        summary = item.get("summary", "")
        text = title + summary
        # 检查是否匹配该行业关键词
        matched = [kw for kw in keywords if kw in text]
        if matched:
            direct.append(item)
        else:
            # 宏观/政策类仍作为间接相关
            is_macro = any(kw in text for kw in ["央行", "证监会", "国务院", "政策", "利率", "PMI", "GDP", "降准", "降息", "财政", "LPR", "MLF"])
            if is_macro:
                indirect.append(item)

    # 直接相关全取 + 重要宏观只取5条
    result = direct + indirect[:5]
    return result


def fetch_eastmoney_news_page2(limit: int = 80) -> list:
    """东方财富第2页新闻（覆盖前24-48小时）。"""
    items = []
    try:
        import uuid as _uuid
        url = "https://np-weblist.eastmoney.com/comm/web/getFastNewsList"
        params = {
            "client": "web", "biz": "fastnews", "fastColumn": "102",
            "sortEnd": "", "pageSize": limit, "pageIndex": 2,
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
# 龙虎榜 + 北向持仓 + SHIBOR
# ═══════════════════════════════════════════

def fetch_top_list(date: str) -> list:
    """龙虎榜数据：机构/游资在哪些板块活跃。"""
    pro = _get_pro()
    if not pro:
        return []
    items = []
    try:
        df = pro.top_list(trade_date=date)
        if df is not None and not df.empty:
            for _, r in df.iterrows():
                items.append({
                    "name": str(r.get("name", "")),
                    "code": str(r.get("ts_code", "")),
                    "pct_chg": round(float(r.get("pct_change", 0)), 2),
                    "net_amount": round(float(r.get("net_amount", 0)) / 1e8, 2),
                    "l_buy": round(float(r.get("l_buy", 0)) / 1e8, 2),
                    "l_sell": round(float(r.get("l_sell", 0)) / 1e8, 2),
                    "reason": str(r.get("reason", ""))[:100],
                })
    except Exception:
        pass
    return items


def fetch_north_holdings() -> list:
    """北向资金持仓TOP20（按持股市值排序）。"""
    pro = _get_pro()
    if not pro:
        return []
    items = []
    try:
        today = datetime.now().strftime("%Y%m%d")
        df = pro.hk_hold(trade_date=today)
        if df is not None and not df.empty:
            df = df.sort_values("vol", ascending=False)
            for _, r in df.head(20).iterrows():
                items.append({
                    "name": str(r.get("name", "")),
                    "code": str(r.get("ts_code", "")),
                    "vol": round(float(r.get("vol", 0)) / 1e8, 2),
                    "ratio": round(float(r.get("ratio", 0)), 2),
                })
    except Exception:
        pass
    return items


def fetch_shibor() -> dict:
    """SHIBOR利率（国内流动性指标）。"""
    pro = _get_pro()
    if not pro:
        return {}
    try:
        today = datetime.now().strftime("%Y%m%d")
        df = pro.shibor(start_date=today, end_date=today)
        if df is not None and not df.empty:
            r = df.iloc[0]
            return {
                "隔夜": f"{float(r['on']):.4f}%",
                "1周": f"{float(r['1w']):.4f}%",
                "1月": f"{float(r['1m']):.4f}%",
                "3月": f"{float(r['3m']):.4f}%",
            }
    except Exception:
        pass
    return {}


# ═══════════════════════════════════════════
# 机构数据
# ═══════════════════════════════════════════

def fetch_broker_recommendations() -> list:
    """获取本月券商推荐热度排名（前20只股票）。"""
    pro = _get_pro()
    if not pro:
        return []

    this_month = datetime.now().strftime("%Y%m")
    try:
        df = pro.broker_recommend(month=this_month)
        if df is not None and not df.empty:
            # 统计每只股票被推荐次数
            counts = df.groupby(["ts_code", "name"]).size().reset_index(name="count")
            counts = counts.sort_values("count", ascending=False)
            top = counts.head(20)
            return [
                {"code": r["ts_code"], "name": r["name"], "brokers": int(r["count"])}
                for _, r in top.iterrows()
            ]
    except Exception:
        pass
    return []


# ═══════════════════════════════════════════
# 个股级别数据（板块聚焦专用）
# ═══════════════════════════════════════════

# 申万行业 → 指数代码
SW_INDEX_MAP = {
    "农林牧渔": "801010.SI", "采掘": "801020.SI", "化工": "801030.SI",
    "钢铁": "801040.SI", "有色金属": "801050.SI", "电子": "801080.SI",
    "家用电器": "801110.SI", "食品饮料": "801120.SI", "纺织服装": "801130.SI",
    "轻工制造": "801140.SI", "医药生物": "801150.SI", "公用事业": "801160.SI",
    "交通运输": "801170.SI", "房地产": "801180.SI", "商业贸易": "801200.SI",
    "休闲服务": "801210.SI", "综合": "801230.SI", "建筑材料": "801710.SI",
    "建筑装饰": "801720.SI", "电气设备": "801730.SI", "国防军工": "801740.SI",
    "计算机": "801750.SI", "传媒": "801760.SI", "通信": "801770.SI",
    "银行": "801780.SI", "非银金融": "801790.SI", "汽车": "801880.SI",
    "机械设备": "801890.SI", "煤炭": "801950.SI", "石油石化": "801960.SI",
    "环保": "801970.SI",
}


# 股票代码→名称缓存（一次加载，全局复用）
_stock_name_cache: dict = {}

def _load_stock_names():
    """加载全部A股代码→名称映射。"""
    global _stock_name_cache
    if _stock_name_cache:
        return
    pro = _get_pro()
    if not pro:
        return
    try:
        df = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,industry")
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                _stock_name_cache[row["ts_code"]] = row["name"]
    except Exception:
        pass


def _stock_name(code: str) -> str:
    """获取股票名称，LLM 兜底。"""
    if code in _stock_name_cache:
        return _stock_name_cache[code]
    return code


def fetch_sector_stock_detail(sector_name: str, date: str) -> dict:
    """
    获取板块成分股当日行情：领涨/领跌、主力资金、盘中节奏。
    返回给 LLM 的格式化文本。
    """
    pro = _get_pro()
    if not pro:
        return {}

    idx_code = SW_INDEX_MAP.get(sector_name)
    if not idx_code:
        return {}

    # 确保股票名称缓存已加载
    _load_stock_names()

    result = {"sector": sector_name, "stocks": [], "top_gainers": [], "top_losers": []}

    try:
        # 1. 获取成分股列表
        df_member = pro.index_member(index_code=idx_code)
        if df_member is None or df_member.empty:
            return result
        stocks = df_member["con_code"].tolist()

        # 2. 批量获取全部成分股今日行情（O/H/L/C/涨跌幅/成交量）
        all_daily = []
        for i in range(0, len(stocks), 50):
            batch = stocks[i:i + 50]
            try:
                df = pro.daily(ts_code=",".join(batch), trade_date=date)
                if df is not None and not df.empty:
                    all_daily.append(df)
            except Exception:
                pass

        if not all_daily:
            return result

        import pandas as pd
        daily_df = pd.concat(all_daily, ignore_index=True)
        # 按涨幅排序
        daily_df = daily_df.sort_values("pct_chg", ascending=False)

        top5 = daily_df.head(5)
        bottom5 = daily_df.tail(5)

        result["total_stocks"] = len(daily_df)
        result["up_count"] = int((daily_df["pct_chg"] > 0).sum())
        result["down_count"] = int((daily_df["pct_chg"] < 0).sum())
        result["flat_count"] = int((daily_df["pct_chg"] == 0).sum())

        result["top_gainers"] = [
            {"code": r["ts_code"], "name": _stock_name(r["ts_code"]),
             "pct_chg": round(float(r["pct_chg"]), 2),
             "close": round(float(r["close"]), 2),
             "open": round(float(r.get("open", 0)), 2),
             "high": round(float(r.get("high", 0)), 2),
             "low": round(float(r.get("low", 0)), 2),
             "vol": round(float(r.get("vol", 0)) / 10000, 2)}
            for _, r in top5.iterrows()
        ]
        result["top_losers"] = [
            {"code": r["ts_code"], "name": _stock_name(r["ts_code"]),
             "pct_chg": round(float(r["pct_chg"]), 2),
             "close": round(float(r["close"]), 2),
             "open": round(float(r.get("open", 0)), 2),
             "high": round(float(r.get("high", 0)), 2),
             "low": round(float(r.get("low", 0)), 2),
             "vol": round(float(r.get("vol", 0)) / 10000, 2)}
            for _, r in bottom5.iterrows()
        ]

        # 日内振幅统计
        daily_df["amplitude"] = (daily_df["high"] - daily_df["low"]) / daily_df["pre_close"] * 100
        high_amp = daily_df[daily_df["amplitude"] > 5]
        if len(high_amp) > 0:
            result["high_amplitude"] = [
                {"code": r["ts_code"], "name": _stock_name(r["ts_code"]),
                 "amplitude": round(float(r["amplitude"]), 1),
                 "pct_chg": round(float(r["pct_chg"]), 2)}
                for _, r in high_amp.head(5).iterrows()
            ]

        # 3. 个股资金流向（top10，非阻塞：失败不影响主数据）
        try:
            top10 = daily_df.head(10)["ts_code"].tolist()
            df_flow = pro.moneyflow(ts_code=",".join(top10), trade_date=date)
            if df_flow is not None and not df_flow.empty:
                buy_lg = float(df_flow["buy_lg_amount"].sum()) if "buy_lg_amount" in df_flow.columns else 0
                sell_lg = float(df_flow["sell_lg_amount"].sum()) if "sell_lg_amount" in df_flow.columns else 0
                buy_md = float(df_flow["buy_md_amount"].sum()) if "buy_md_amount" in df_flow.columns else 0
                sell_md = float(df_flow["sell_md_amount"].sum()) if "sell_md_amount" in df_flow.columns else 0
                buy_sm = float(df_flow["buy_sm_amount"].sum()) if "buy_sm_amount" in df_flow.columns else 0
                sell_sm = float(df_flow["sell_sm_amount"].sum()) if "sell_sm_amount" in df_flow.columns else 0
                result["fund_flow"] = {
                    "lg_net": round((buy_lg - sell_lg) / 1e8, 2),
                    "md_net": round((buy_md - sell_md) / 1e8, 2),
                    "sm_net": round((buy_sm - sell_sm) / 1e8, 2),
                }
        except Exception:
            pass

        return result

    except Exception:
        return result


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

    # PPI产业链子项（上游→中游→下游传导）
    try:
        df = pro.cn_ppi(start_m="202601", end_m=this_month)
        if df is not None and not df.empty:
            row = df.iloc[-1]
            result["PPI_采掘上游"] = f"{float(row['ppi_mp_qm_yoy']):.1f}%"
            result["PPI_原材料中游"] = f"{float(row['ppi_mp_rm_yoy']):.1f}%"
            result["PPI_加工中游"] = f"{float(row['ppi_mp_p_yoy']):.1f}%"
            result["PPI_生活资料下游"] = f"{float(row['ppi_cg_yoy']):.1f}%"
            result["PPI_食品终端"] = f"{float(row['ppi_cg_f_yoy']):.1f}%"
    except Exception:
        pass

    return result


# ═══════════════════════════════════════════
# 商品期货 + 汇率（yfinance）
# ═══════════════════════════════════════════

def fetch_commodities_and_fx() -> dict:
    """商品期货 + 人民币汇率（yfinance 免费）。"""
    result = {}
    tickers = {
        "CL=F": "WTI原油", "GC=F": "黄金", "HG=F": "铜",
        "USDCNY=X": "在岸人民币", "CNH=X": "离岸人民币",
    }
    try:
        import yfinance as yf
        for sym, name in tickers.items():
            try:
                t = yf.Ticker(sym)
                hist = t.history(period="3d")
                if hist is not None and len(hist) >= 2:
                    close = round(float(hist["Close"].iloc[-1]), 2)
                    prev = round(float(hist["Close"].iloc[-2]), 2)
                    pct = round((close - prev) / prev * 100, 2) if prev else 0.0
                    result[name] = {"close": close, "pct_chg": pct}
            except Exception:
                pass
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

    # 全部并行执行（从串行 15s → 并行 3s）
    tasks = {
        "indices": loop.run_in_executor(None, fetch_a_share_indices, date),
        "sectors": loop.run_in_executor(None, fetch_shenwan_sectors, date),
        "flows": loop.run_in_executor(None, fetch_fund_flows, date),
        "gidx": loop.run_in_executor(None, fetch_global_indices),
        "broker": loop.run_in_executor(None, fetch_broker_recommendations),
        "comm": loop.run_in_executor(None, fetch_commodities_and_fx),
        "shibor": loop.run_in_executor(None, fetch_shibor),
        "north": loop.run_in_executor(None, fetch_north_holdings),
        "toplist": loop.run_in_executor(None, fetch_top_list, date),
        "cn_macro": loop.run_in_executor(None, fetch_china_macro),
        "us_macro": loop.run_in_executor(None, fetch_us_macro),
        "em_p1": loop.run_in_executor(None, fetch_eastmoney_news, 80),
        "em_p2": loop.run_in_executor(None, fetch_eastmoney_news_page2, 80),
        "fh_news": loop.run_in_executor(None, fetch_finnhub_news, 20),
        "calendar": loop.run_in_executor(None, fetch_economic_calendar),
    }
    if sector_focus:
        tasks["stock"] = loop.run_in_executor(None, fetch_sector_stock_detail, sector_focus, date)

    gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
    results_raw = dict(zip(tasks.keys(), gathered))

    def safe(v, default):
        """如果结果是异常或None，返回默认值。"""
        if isinstance(v, Exception) or v is None:
            return default
        return v

    snapshot.indices = safe(results_raw.get("indices"), {})
    snapshot.sectors = safe(results_raw.get("sectors"), [])
    snapshot.fund_flows = safe(results_raw.get("flows"), {})
    snapshot.global_indices = safe(results_raw.get("gidx"), {})
    snapshot._broker_recs = safe(results_raw.get("broker"), [])
    snapshot._commodities = safe(results_raw.get("comm"), {})
    snapshot._shibor = safe(results_raw.get("shibor"), {})
    snapshot._north_hold = safe(results_raw.get("north"), [])
    snapshot._top_list = safe(results_raw.get("toplist"), [])
    snapshot.macro_data = {
        "china": safe(results_raw.get("cn_macro"), {}),
        "us": safe(results_raw.get("us_macro"), {}),
    }
    all_em = safe(results_raw.get("em_p1"), []) + safe(results_raw.get("em_p2"), [])
    if sector_focus:
        all_em = filter_news_by_sector(all_em, sector_focus)
    snapshot.news_items = {
        "eastmoney": all_em,
        "global": safe(results_raw.get("fh_news"), []),
    }
    snapshot.calendar = safe(results_raw.get("calendar"), [])
    if sector_focus:
        snapshot._stock_detail = safe(results_raw.get("stock"), None)

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
            parts = [f"  {s['name']}: {s['pct_chg']:+.2f}% [{s['tag']}]{streak}"]
            if s.get("chg_5d") is not None: parts.append(f"5日{s['chg_5d']:+.2f}%")
            if s.get("chg_10d") is not None: parts.append(f"10日{s['chg_10d']:+.2f}%")
            if s.get("chg_20d") is not None: parts.append(f"20日{s['chg_20d']:+.2f}%")
            if s.get("ma5") and s.get("close"):
                above = sum(1 for m in [s.get("ma5"), s.get("ma10"), s.get("ma20")] if m and s["close"] > m)
                parts.append(f"站上{above}/3均线")
            if s.get("open") and s["open"] > 0:
                parts.append(f"开{s['open']:.2f}高{s.get('high',0):.2f}低{s.get('low',0):.2f}")
            if s.get("vol") and s["vol"] > 0:
                parts.append(f"量{s['vol']:.0f}万手")
            lines.append("  ".join(parts))
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
        for item in em[:120]:
            summary = item.get("summary", "")[:120]
            lines.append(f"- [{item['time']}] {item['title']}")
            if summary:
                lines.append(f"  {summary}")

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

    # ── SHIBOR ──
    shibor = getattr(snapshot, "_shibor", {})
    if shibor:
        lines.append("### 国内流动性（SHIBOR）")
        lines.append(f"隔夜{shibor['隔夜']} 1周{shibor['1周']} 1月{shibor['1月']} 3月{shibor['3月']}")
        lines.append("")

    # ── 商品 + 汇率 ──
    comm = getattr(snapshot, "_commodities", {})
    if comm:
        lines.append("### 商品期货与汇率")
        for name, d in comm.items():
            lines.append(f"- {name}: {d['close']} {d['pct_chg']:+.2f}%")
        lines.append("")

    # ── 龙虎榜 ──
    top_list = getattr(snapshot, "_top_list", [])
    if top_list:
        lines.append(f"### 龙虎榜活跃股（共{len(top_list)}只上榜）")
        inst = [t for t in top_list if t['l_buy'] > t['l_sell']]
        retail = [t for t in top_list if t['l_buy'] < t['l_sell']]
        if inst:
            names = ", ".join(f"{t['name']}(净买入{t['net_amount']:+.1f}亿)" for t in inst[:10])
            lines.append(f"机构净买入: {names}")
        if retail:
            names = ", ".join(f"{t['name']}(净卖出{abs(t['net_amount']):.1f}亿)" for t in retail[:10])
            lines.append(f"机构净卖出: {names}")
        lines.append("")

    # ── 北向持仓TOP ──
    north_h = getattr(snapshot, "_north_hold", [])
    if north_h:
        lines.append("### 北向资金持仓TOP20")
        for h in north_h[:15]:
            lines.append(f"- {h['name']}({h['code']}) 持仓{h['vol']}亿 占比{h['ratio']}%")
        lines.append("")

    # ── 美国宏观 ──
    us = snapshot.macro_data.get("us", {})
    if us:
        lines.append("### 美国宏观数据（FRED）")
        for name, val in us.items():
            lines.append(f"- {name}: {val}")
    lines.append("")

    # ── 板块个股详情 ──
    stock_detail = getattr(snapshot, "_stock_detail", None)
    if stock_detail:
        lines.append("### 板块成分股数据（个股级别）")
        total = stock_detail.get("total_stocks", 0)
        up = stock_detail.get("up_count", 0)
        down = stock_detail.get("down_count", 0)
        flat = stock_detail.get("flat_count", 0)
        lines.append(f"成分股总数: {total}只  上涨: {up}只  下跌: {down}只  平盘: {flat}只")

        if stock_detail.get("top_gainers"):
            lines.append("领涨前5（含日内四价）：")
            for s in stock_detail["top_gainers"]:
                name = s.get("name", s["code"])
                o, h, l = s.get("open", 0), s.get("high", 0), s.get("low", 0)
                lines.append(f"  {name}({s['code']}) 收{s['close']} {s['pct_chg']:+.2f}% 开{o} 高{h} 低{l} 振幅{abs(s.get('high',0)-s.get('low',0)):.2f} 量{s['vol']}万手")

        if stock_detail.get("top_losers"):
            lines.append("领跌前5（含日内四价）：")
            for s in stock_detail["top_losers"]:
                name = s.get("name", s["code"])
                o, h, l = s.get("open", 0), s.get("high", 0), s.get("low", 0)
                lines.append(f"  {name}({s['code']}) 收{s['close']} {s['pct_chg']:+.2f}% 开{o} 高{h} 低{l} 振幅{abs(s.get('high',0)-s.get('low',0)):.2f} 量{s['vol']}万手")

        if stock_detail.get("high_amplitude"):
            lines.append("高振幅异动（振幅>5%）：")
            for s in stock_detail["high_amplitude"]:
                name = s.get("name", s["code"])
                lines.append(f"  {name}({s['code']}) 振幅{s['amplitude']}% 涨跌{s['pct_chg']:+.2f}%")

        if stock_detail.get("fund_flow"):
            ff = stock_detail["fund_flow"]
            lines.append(f"资金拆解（前10权重股）：大单净额{ff['lg_net']:+.2f}亿（机构） 中单净额{ff['md_net']:+.2f}亿（游资） 小单净额{ff['sm_net']:+.2f}亿（散户）")
        lines.append("")

    # ── 券商推荐 ──
    recs = getattr(snapshot, "_broker_recs", [])
    if recs:
        lines.append(f"### 本月券商推荐热度TOP20（共32家券商，{len(recs)}只最热股票）")
        for r in recs:
            lines.append(f"- {r['name']}({r['code']}) {r['brokers']}家券商推荐")
        lines.append("")

    # ── 日历 ──
    if snapshot.calendar:
        lines.append("### 近期经济事件")
        for ev in snapshot.calendar[:8]:
            stars = "*" * ev.get("importance", 1)
            lines.append(f"- {ev['date']} {ev['time']} [{stars}] {ev['event']}")

    return "\n".join(lines)
