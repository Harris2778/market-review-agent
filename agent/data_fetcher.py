"""
数据采集器 v2.0：行情 + 趋势 + 新闻 + 宏观 + 量价分析。

设计原则：
- 每个数据源独立封装，失败返回空而非抛异常
- 支持降级：API → 公开数据 → 标记不可用
- 所有时间戳统一为北京时间
"""

import os
import json
import logging
import time
import requests
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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
    news_items: dict = field(default_factory=dict)
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
                "amount": round(float(latest.get("amount", 0) or 0) / 1e5, 2),  # 千元→亿
                "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60,
                "chg_5d": chg_5d, "chg_20d": chg_20d,
                "trend_5d": trend_5d, "trend_20d": trend_20d,
                "vol_ratio": vol_ratio,
                "mas_above": f"{above_mas}/{total_mas}" if total_mas else "—",
            }
        except Exception as e:
            logger.warning("获取指数行情失败 code=%s", code, exc_info=True)
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
                "amount": round(float(row.get("amount", 0)) / 1e5, 2),
                "streak": streak_str,
            })
        except Exception as e:
            logger.warning("获取申万行业行情失败 code=%s", code, exc_info=True)

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
            if nm != 0:
                result["north_bound"] = round(nm / 10000, 2)
            if sm != 0:
                result["south_bound"] = round(sm / 10000, 2)
    except Exception:
        try:
            df = pro.moneyflow_hsgt(trade_date=date)
            if df is not None and not df.empty:
                row = df.iloc[0]
                nm = float(row.get("north_money", 0))
                sm = float(row.get("south_money", 0))
                if nm != 0:
                    result["north_bound"] = round(nm / 10000, 2)
                if sm != 0:
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
    except Exception as e:
        # 常见原因：token 无 news 接口权限，降级尝试 sina 源
        logger.warning("Tushare 新闻(cls源)获取失败: %s", str(e)[:120])

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
        except Exception as e:
            logger.warning("Tushare 新闻(sina源)获取失败: %s", str(e)[:120])

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
    "医药生物": ["医药", "医疗", "药", "疫苗", "生物", "基因", "细胞", "病毒", "疫情", "流感", "新冠", "创新药", "CRO", "医疗器械", "恒瑞医药", "迈瑞", "药明康德", "药明", "百济神州", "君实", "信达", "康希诺", "智飞", "沃森", "康泰", "PD-1", "GLP-1", "减肥药", "抗癌", "肿瘤", "临床", "FDA", "审批", "医保", "集采"],
    "汽车": ["汽车", "车", "新能源车", "电动车", "整车", "乘用车", "商用车", "比亚迪", "特斯拉", "蔚来", "小鹏", "理想", "小米", "华为", "自动驾驶", "智能驾驶", "锂电", "充电桩", "车市", "销量", "出口", "SUV", "轿车", "卡车", "客车", "轮胎", "4S", "经销商", "上汽", "广汽", "吉利", "长城", "长安"],
    "银行": ["银行", "金融", "贷款", "存款", "利率", "息差", "工商银行", "招商银行", "建设银行", "农业银行", "中国银行", "交通银行", "邮储银行", "兴业银行", "浦发银行", "中信银行", "民生银行", "光大银行", "平安银行", "净息差", "降准", "降息", "MLF", "LPR", "信贷", "社融", "M2", "货币政策", "央行", "银保监", "金监", "商业银行", "城商行", "农商行"],
    "非银金融": ["券商", "保险", "证券", "中信证券", "华泰证券", "中国平安", "中国人寿", "投行", "IPO", "再融资", "融资融券"],
    "房地产": ["房地产", "地产", "万科", "保利", "碧桂园", "恒大", "融创", "楼市", "房价", "商品房", "土地出让", "房贷", "公积金", "限购", "城中村"],
    "煤炭": ["煤炭", "煤价", "煤", "矿", "中国神华", "陕西煤业", "中煤能源", "兖矿", "动力煤", "焦煤", "煤矿", "能源", "采掘"],
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

    # 直接相关全取 + 重要宏观取15条
    result = direct + indirect[:15]
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


def fetch_eastmoney_news_page3(limit: int = 100) -> list:
    """东方财富第3页新闻。"""
    items = []
    try:
        import uuid as _uuid
        url = "https://np-weblist.eastmoney.com/comm/web/getFastNewsList"
        params = {
            "client": "web", "biz": "fastnews", "fastColumn": "102",
            "sortEnd": "", "pageSize": limit, "pageIndex": 3,
            "req_trace": str(_uuid.uuid4()).replace("-", "")[:32],
        }
        resp = requests.get(url, params=params, timeout=15,
                          headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.eastmoney.com/"})
        data = resp.json()
        for item in data.get("data", {}).get("fastNewsList", []):
            items.append({
                "source": "东方财富", "time": item.get("showTime", ""),
                "title": item.get("title", "")[:150],
            })
    except Exception:
        pass
    return items


# 缓存 MCP 工具列表
_mcp_tools_cache = None

def get_mcp_tools() -> list:
    """获取所有可用MCP工具列表。"""
    global _mcp_tools_cache
    if _mcp_tools_cache:
        return _mcp_tools_cache
    token = _env("SINA_MCP_TOKEN", "")
    if not token:
        return []
    try:
        base = "https://mcp.finance.sina.com.cn/mcp-http"
        r = requests.post(f"{base}?token={token}", json={
            "jsonrpc":"2.0","method":"initialize","id":1,
            "params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"a","version":"1"}}
        }, timeout=15)
        sid = r.headers.get("Mcp-Session-Id","")
        if sid:
            r2 = requests.post(f"{base}?token={token}", json={
                "jsonrpc":"2.0","method":"tools/list","id":2
            }, headers={"Mcp-Session-Id":sid}, timeout=15)
            tools = r2.json().get("result",{}).get("tools",[])
            _mcp_tools_cache = [{"name":t["name"],"desc":t.get("description","")[:200],
                                "params":list(t.get("inputSchema",{}).get("properties",{}).keys())}
                              for t in tools]
            return _mcp_tools_cache
    except Exception:
        pass
    return []


def _mcp_call(tool_name: str, args: dict) -> dict:
    """通用MCP工具调用。自动兼容JSON和分号分隔字符串两种响应格式。"""
    token = _env("SINA_MCP_TOKEN", "")
    if not token:
        return {}
    try:
        base = "https://mcp.finance.sina.com.cn/mcp-http"
        r = requests.post(f"{base}?token={token}", json={
            "jsonrpc":"2.0","method":"initialize","id":1,
            "params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"a","version":"1"}}
        }, timeout=15)
        sid = r.headers.get("Mcp-Session-Id","")
        if not sid:
            return {}
        r2 = requests.post(f"{base}?token={token}", json={
            "jsonrpc":"2.0","method":"tools/call","id":2,
            "params":{"name": tool_name, "arguments": args}
        }, headers={"Mcp-Session-Id":sid}, timeout=30)
        d = r2.json()
        content = d.get("result",{}).get("content",[])
        if content and isinstance(content,list):
            text = content[0].get("text","")
            if not text:
                return {}
            # 格式1: JSON → 尝试解析
            try:
                return json.loads(text)
            except:
                pass
            # 格式2: var xxx="csv,data" 或 HTML/纯文本
            if "var " in text and "=\"" in text:
                csv = text.split('="')[1].split('"')[0] if '="' in text else text
                return {"raw": csv, "type": "csv"}
            # 格式3: 纯文本 → 尝试当JSON，失败则包起来
            if text.strip().startswith("{") or text.strip().startswith("["):
                try:
                    return json.loads(text)
                except:
                    return {"raw": text, "type": "text"}
            return {"raw": text, "type": "text"}
    except Exception:
        pass
    return {}


def fetch_market_breadth() -> dict:
    """A股全市场涨跌分布。"""
    d = _mcp_call("cnMarketUpdownDistribution", {})
    raw = d.get("raw","")
    if not raw or "," not in raw:
        return {}
    parts = raw.split(",")
    if len(parts) < 13:
        logger.warning("涨跌分布CSV字段数异常（%d < 13），原始数据: %s", len(parts), raw[:200])
        return {}
    try:
        return {"date": parts[1], "跌停": parts[2], "跌7-10": parts[3],
                "跌5-7": parts[4], "跌2-5": parts[5], "跌0-2": parts[6], "平": parts[7],
                "涨0-2": parts[8], "涨2-5": parts[9], "涨5-7": parts[10], "涨7-10": parts[11], "涨停": parts[12],
                "total_up": str(int(parts[8])+int(parts[9])+int(parts[10])+int(parts[11])+int(parts[12])),
                "total_down": str(int(parts[2])+int(parts[3])+int(parts[4])+int(parts[5])+int(parts[6]))}
    except (ValueError, IndexError) as e:
        logger.warning("解析涨跌分布CSV失败: %s", raw[:200], exc_info=True)
        return {}


def fetch_hot_stocks() -> list:
    """A股股票热搜榜。type: d, num: 数量, page: 页码"""
    d = _mcp_call("globalStockHotBoard", {"type": "d", "num": 10, "page": 1})
    items = []
    data = d.get("result",{}).get("data",[]) or []
    for it in data:
        items.append({"name": it.get("name",""), "code": it.get("symbol",""), "heat": it.get("pv","")})
    return items[:15]


def fetch_us_breadth() -> dict:
    """美股涨跌分布。数据格式: result.data.{rise,fall,ping,up_0_2,...}"""
    d = _mcp_call("usMarketStatisticsUpdown", {})
    data = d.get("result",{}).get("data",{}) or d.get("data",{})
    if data:
        return {"涨": data.get("rise","?"), "跌": data.get("fall","?"), "平": data.get("ping","?"),
                "涨停": data.get("up_10","?"), "跌停": data.get("down_10","?")}
    return {}


def fetch_strong_sectors() -> list:
    """A股强势板块。type=all/ck/other, bk=gn概念/hy行业/dy地域"""
    d = _mcp_call("cnMarketStrongSectors", {"type": "all", "isNotSt": "1", "bk": "gn"})
    items = []
    data = d.get("result",{}).get("data",[]) or []
    for it in data[:10]:
        items.append({"name": it.get("name",""), "pct": it.get("percent","")})
    return items


def fetch_valuation(symbol: str, rank: str = "y1", val_type: str = "syl") -> dict:
    """个股估值明细。rank: y1/y3/y5/y10/all, type: syl市盈率/sjl市净率/sxl市现率/gxl股息率/zsz总市值"""
    d = _mcp_call("cnStockValuationDetail", {"symbol": symbol, "rank": rank, "type": val_type})
    data = d.get("result",{}).get("data",{}) or d.get("data",{}) or {}
    return {"个股": data.get("gg",[]), "行业": data.get("hy",[]), "大盘": data.get("dp",[])}


def fetch_limit_up_pool() -> list:
    """A股涨停池。"""
    d = _mcp_call("cnMarketLimitUpPool", {})
    items = []
    data = d.get("result",{}).get("data",[]) or []
    for it in data[:15]:
        items.append({"name": it.get("name",""), "code": it.get("symbol",""), "pct": it.get("change_pct",""),
                      "reason": it.get("reason","")})
    return items


def fetch_lian_ban() -> list:
    """连板个股。"""
    d = _mcp_call("cnStockLianBC", {})
    items = []
    data = d.get("result",{}).get("data",[]) or []
    for it in data[:10]:
        items.append({"name": it.get("name",""), "code": it.get("symbol",""), "count": it.get("limit_count","")})
    return items


def fetch_us_fund_flow(symbol: str = "aapl") -> dict:
    """美股今日资金流向。symbol小写如aapl"""
    d = _mcp_call("usTradingFundFlow1Day", {"symbol": symbol.lower()})
    data = d.get("data",{}) or {}
    return {"超大单": data.get("r0","?"), "大单": data.get("r1","?"), "中单": data.get("r2","?"), "小单": data.get("r3","?"), "成交额": data.get("amount","?")}


def fetch_stock_quote(market: str, symbol: str) -> dict:
    """实时股票行情。market: cn/hk/us, symbol需带前缀如sh688001、sz000002"""
    d = _mcp_call("globalStockQuoteRealtime", {"market": market, "symbol": symbol})
    data = d.get("data",{}) or {}
    return {"name": data.get("name",""), "price": data.get("price",""), "pct": data.get("percent",""),
            "high": data.get("high",""), "low": data.get("low",""), "open": data.get("openPrice",""),
            "volume": data.get("volume",""), "preClose": data.get("preClose","")}


def fetch_stock_kline(market: str, symbol: str, days: int = 10) -> list:
    """股票日K线。market: cn/hk/us"""
    d = _mcp_call("globalStockKlineDaily", {"market": market, "symbol": symbol, "num": str(days)})
    data = d.get("data",[]) or []
    items = []
    for it in (data or [])[-days:]:
        items.append({"date": it.get("date",""), "close": it.get("close",""), "pct": it.get("change_pct","")})
    return items


def fetch_stock_news(symbol: str, market: str = "cn", limit: int = 10) -> list:
    """个股新闻搜索。symbol需带sh/sz前缀"""
    d = _mcp_call("stockNewsSearch", {"market": market, "symbol": symbol, "num": str(min(20,limit)), "page": "1"})
    items = []
    data = d.get("result",{}).get("data",[]) or []
    for it in data[:limit]:
        items.append({"title": it.get("title",""), "url": it.get("url","")})
    return items


def fetch_revenue_composition(paper_code: str, fr_date: str = "") -> dict:
    """A股主营构成。paperCode如sz000002，frDate如20231231(可选)"""
    args = {"paperCode": paper_code}
    if fr_date:
        args["frDate"] = fr_date
    d = _mcp_call("cnFinanceRevenueComposition", args)
    data = d.get("result",{}).get("data",{}) or d.get("data",{}) or {}
    return {"股票": data.get("sname",""), "按产品": data.get("by_product",[]), "按行业": data.get("by_business",[]), "按地区": data.get("by_region",[])}


def fetch_sector_components(node: str, sort: str = "percent", num: int = 20) -> list:
    """指数/行业成分股排行。node如sh000001(上证指数)"""
    d = _mcp_call("cnSectorComponentsRanking", {"node": node, "sort": sort, "asc": "0", "num": str(num), "page": "1"})
    items = []
    data = d.get("data",[]) or []
    for it in data:
        items.append({"name": it.get("name",""), "code": it.get("symbol",""), "pct": it.get("percent",""),
                      "price": it.get("price",""), "pe": it.get("pe",""), "mcap": it.get("totalShare","")})
    return items


def fetch_sw_classify(symbol: str) -> dict:
    """申万行业分类。symbol不带前缀如600519"""
    d = _mcp_call("swSymbolList", {"symbol": symbol})
    data = d.get("data",{}) or {}
    return {"一级": data.get("sw1",""), "二级": data.get("sw2",""), "三级": data.get("sw3","")}


def search_stock(keyword: str) -> list:
    """股票代码搜索。返回CSV格式：结果：名称,市场,代码,完整代码,..."""
    d = _mcp_call("globalStockSearchSymbols", {"type": "11", "key": keyword, "format": "text", "num": "5"})
    items = []
    raw = d.get("raw","") or ""
    # 格式: "结果：贵州茅台,11,600519,sh600519,...  结果组成定义：{...}"
    if "结果：" in raw:
        data_part = raw.split("结果：")[1].split("结果组成定义")[0].strip()
        for block in data_part.split("\n"):
            parts = [p.strip() for p in block.split(",")]
            if len(parts) >= 4 and parts[0]:
                items.append({"name": parts[0], "market": parts[1], "code": parts[2], "full_code": parts[3]})
    return items[:5]


def fetch_futures_quote(market: str, symbol: str) -> dict:
    """期货行情。market: dce/shfe/czce/gfex"""
    d = _mcp_call("future_quotes", {"market": market, "symbol": symbol})
    data = d.get("result",{}).get("data",{}) or {}
    return {"price": data.get("price",""), "pct": data.get("change_pct",""), "volume": data.get("volume","")}


def fetch_financials(code: str) -> dict:
    """个股三大报表。"""
    pro = _get_pro()
    if not pro:
        return {}
    result = {}
    for name, fn in [("利润表", pro.income), ("资产负债表", pro.balancesheet), ("现金流量表", pro.cashflow)]:
        try:
            df = fn(ts_code=code, start_date="20260101", end_date="20260630")
            if df is not None and not df.empty:
                result[name] = df.iloc[0].to_dict()
        except Exception:
            pass
    return result


def fetch_forecast(date: str = "") -> list:
    """业绩预告。"""
    pro = _get_pro()
    if not pro:
        return []
    d = date or datetime.now().strftime("%Y%m%d")
    try:
        df = pro.forecast(ann_date=d)
        if df is not None and not df.empty:
            return [{"code": r["ts_code"], "type": r.get("type",""), "p_min": r.get("p_change_min",""), "p_max": r.get("p_change_max","")} for _, r in df.head(20).iterrows()]
    except Exception:
        pass
    return []


def fetch_express(date: str = "") -> list:
    """业绩快报。"""
    pro = _get_pro()
    if not pro:
        return []
    d = date or datetime.now().strftime("%Y%m%d")
    try:
        df = pro.express(ann_date=d)
        if df is not None and not df.empty:
            return [{"code": r["ts_code"], "revenue": r.get("revenue",""), "profit": r.get("operate_profit","")} for _, r in df.head(20).iterrows()]
    except Exception:
        pass
    return []


def fetch_block_trades(code: str = "", date: str = "") -> list:
    """大宗交易。"""
    pro = _get_pro()
    if not pro:
        return []
    d = date or datetime.now().strftime("%Y%m%d")[:6] + "01"
    try:
        df = pro.block_trade(ts_code=code, start_date=d, end_date=date or datetime.now().strftime("%Y%m%d")) if code else pro.block_trade(start_date=d, end_date=date or datetime.now().strftime("%Y%m%d"))
        if df is not None and not df.empty:
            return [{"code": r["ts_code"], "date": r.get("trade_date",""), "price": r.get("price",""), "amount": r.get("amount","")} for _, r in df.head(20).iterrows()]
    except Exception:
        pass
    return []


def fetch_fund_list(market: str = "E") -> list:
    """基金列表。market: E(ETF)/O(开放式)/F(封闭式)"""
    pro = _get_pro()
    if not pro:
        return []
    try:
        df = pro.fund_basic(market=market)
        if df is not None and not df.empty:
            return [{"code": r["ts_code"], "name": r["name"], "type": r.get("fund_type",""), "company": r.get("management","")} for _, r in df.head(50).iterrows()]
    except Exception:
        pass
    return []


def fetch_ggt_daily() -> list:
    """港股通每日资金流向。"""
    pro = _get_pro()
    if not pro:
        return []
    try:
        today = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        df = pro.ggt_daily(start_date=start, end_date=today)
        if df is not None and not df.empty:
            return [{"date": r["trade_date"], "buy": r.get("buy_amount",""), "sell": r.get("sell_amount","")} for _, r in df.iterrows()]
    except Exception:
        pass
    return []


def fetch_repurchase(date: str = "") -> list:
    """股票回购。"""
    pro = _get_pro()
    if not pro:
        return []
    d = date or datetime.now().strftime("%Y%m%d")
    try:
        df = pro.repurchase(ann_date=d)
        if df is not None and not df.empty:
            return [{"code": r["ts_code"], "vol": r.get("vol",""), "proc": r.get("proc","")} for _, r in df.head(20).iterrows()]
    except Exception:
        pass
    return []


def fetch_share_float(date: str = "") -> list:
    """限售解禁。"""
    pro = _get_pro()
    if not pro:
        return []
    d = date or datetime.now().strftime("%Y%m%d")
    try:
        df = pro.share_float(ann_date=d)
        if df is not None and not df.empty:
            return [{"code": r["ts_code"], "date": r.get("float_date",""), "share": r.get("float_share",""), "ratio": r.get("float_ratio","")} for _, r in df.head(20).iterrows()]
    except Exception:
        pass
    return []


def fetch_fund_info(symbol: str) -> dict:
    """基金档案。"""
    d = _mcp_call("fund_info", {"symbol": symbol})
    data = d.get("result",{}).get("data",{}) or {}
    return {"name": data.get("name",""), "type": data.get("type",""), "nav": data.get("nav","")}


def fetch_hk_sectors() -> list:
    """港股板块行情。type: hk_plate_rise领涨/hk_plate_drop领跌/ahg/ggt"""
    d = _mcp_call("hkSectorQuotesList", {"type": "hk_plate_rise", "num": 10, "page": 1})
    items = []
    data = d.get("result",{}).get("data",{}).get("data",[]) or d.get("result",{}).get("data",[]) or []
    for it in data[:10]:
        items.append({"name": it.get("name",""), "pct": it.get("change",""), "lead": it.get("symbol_name","")})
    return items


def fetch_us_sectors() -> list:
    """美股板块排行。page/num/sort/asc全部必填"""
    d = _mcp_call("usSectorRanking", {"page": "1", "num": "10", "sort": "percent", "asc": "0"})
    items = []
    data = d.get("data",[]) or d.get("result",{}).get("data",{}).get("data",[]) or []
    for it in data[:10]:
        items.append({"name": it.get("category_cn",""), "pct": it.get("percent",""), "lead": it.get("lead_cname","")})
    return items


def fetch_northbound_flow() -> list:
    """沪深港通实时资金流向。"""
    items = []
    d = _mcp_call("cnStockConnectHoldings", {"type": "sh", "sort": "hold_market", "asc": 0, "num": 10, "page": 1})
    slist = d.get("result",{}).get("data",{}).get("s_list",[])
    for it in slist[:10]:
        items.append({"name": it.get("name",""), "code": it.get("symbol",""),
                      "hold": it.get("cur_capital",""), "chg": it.get("day1_capital_chg","")})
    return items


def _fmt_news_time(raw, fallback: str = "") -> str:
    """新闻时间归一化：epoch 秒/毫秒 → 'YYYY-MM-DD HH:MM:SS'，其余原样截断。

    raw 为空时返回 fallback（通常为当天日期），保证前端不出现空时间括号。
    """
    if raw is None or raw == "":
        return fallback
    s = str(raw).strip()
    if not s:
        return fallback
    # 纯数字按 epoch 处理（新浪/财联社接口返回 unix 时间戳）
    if s.isdigit():
        try:
            ts = int(s)
            if ts > 1e12:  # 毫秒时间戳
                ts = ts / 1000
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, OverflowError, OSError) as e:
            logger.warning("新闻时间戳解析失败: %s (%s)", s, e)
            return fallback
    return s[:19]


def fetch_mcp_news(keyword: str, limit: int = 30) -> list:
    """新浪智研MCP新闻搜索。复用通用 _mcp_call 通道。"""
    items = []
    if not _env("SINA_MCP_TOKEN", ""):
        return items
    try:
        parsed = _mcp_call("qNewsSearch", {"keyword": keyword, "num": min(20, limit), "page": 1})
        if not parsed:
            return items
        news_list = parsed.get("result",{}).get("data",{}).get("data",[])
        if not news_list:
            news_list = parsed.get("data",{}).get("data",[])
        # 该接口实测只返回 title/content，不带时间字段；
        # 先尽力解析常见时间字段，全部缺失时降级为当天日期，避免前端显示空括号 []
        today_str = datetime.now().strftime("%Y-%m-%d")
        for nd in news_list:
            title = nd.get("title","") or nd.get("content","")[:80]
            if not title:
                continue
            raw_time = ""
            for tf in ("ctime", "mtime", "pubtime", "time", "create_time", "pub_date", "date"):
                if nd.get(tf):
                    raw_time = nd.get(tf)
                    break
            items.append({"source":"智研","time":_fmt_news_time(raw_time, today_str),"title":title})
    except Exception as e:
        logger.warning("智研MCP新闻解析失败: %s", e)
    return items


def fetch_sina_news(limit: int = 20, date_str: str = "") -> list:
    """新浪财经历史新闻（支持按日期查询）。

    接口单页上限 50 条；limit > 50 时自动翻第 2 页（页间 sleep 防限流），
    单日最多可取 100 条。第 2 页失败时降级保留第 1 页结果。
    """
    import re

    def _fetch_page(page: int, num: int) -> list:
        """拉取单页新闻，失败返回空列表。"""
        page_items = []
        try:
            url = "https://feed.mix.sina.com.cn/api/roll/get"
            params = {"pageid": 153, "lid": 2509, "k": "", "num": num, "page": page}
            if date_str:
                params["date"] = date_str
            resp = requests.get(url, params=params, timeout=15,
                                headers={"User-Agent": "Mozilla/5.0"})
            data = resp.json()
            for item in data.get("result", {}).get("data", [])[:num]:
                title = re.sub(r'<[^>]+>', '', item.get("title", ""))
                page_items.append({
                    "source": "新浪财经",
                    # 按日期查询时直接用日期；否则把 epoch ctime 格式化为可读时间
                    "time": date_str if date_str else _fmt_news_time(item.get("ctime", "")),
                    "title": title[:150],
                })
        except Exception as e:
            logger.warning("新浪新闻第%d页获取失败: %s", page, e)
        return page_items

    # 第 1 页（接口 num 上限 50，超出无效）
    items = _fetch_page(1, min(limit, 50))

    # limit > 50 时翻第 2 页补齐（最多再取 50 条）
    if limit > 50 and items:
        time.sleep(0.3)  # 轻微限速，避免触发新浪接口限流
        page2 = _fetch_page(2, min(limit - 50, 50))
        items.extend(page2)

    return items[:limit]


def _cls_sign(params: dict) -> str:
    """财联社接口签名：参数按 key 排序 → key=value& 拼接 → SHA-1 → MD5。"""
    import hashlib
    raw = "&".join(f"{k}={params[k]}" for k in sorted(params))
    return hashlib.md5(hashlib.sha1(raw.encode()).hexdigest().encode()).hexdigest()


def fetch_cls_telegraph(limit: int = 20) -> list:
    """财联社电报（7x24 快讯）。

    旧接口 /api/sw 已下线（404），现行接口为 /api/cache?name=telegraphList
    （签名算法对照 RSSHub lib/routes/cls，财联社再改版时照它更新）。
    单页固定返回约 20 条，通过 last_time 翻页直到凑满 limit（最多 5 页）。
    """
    items = []
    try:
        last_time = int(datetime.now().timestamp())
        max_pages = min(5, (limit + 19) // 20)  # 每页约 20 条
        for _ in range(max_pages):
            params = {
                "app": "CailianpressWeb", "name": "telegraphList",
                "last_time": str(last_time), "os": "web", "rn": "20", "sv": "8.7.9",
            }
            params["sign"] = _cls_sign(params)
            resp = requests.get(
                "https://www.cls.cn/api/cache", params=params,
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.cls.cn/"},
                timeout=15,
            )
            data = resp.json()
            if data.get("errno") != 0:
                logger.warning("财联社电报接口返回错误: errno=%s msg=%s",
                               data.get("errno"), str(data.get("msg"))[:100])
                break
            roll = data.get("data", {}).get("roll_data", [])
            if not roll:
                break
            for item in roll:
                # 快讯类电报 title 常为空，降级用 brief 摘要前 80 字
                title = item.get("title", "") or str(item.get("brief", ""))[:80]
                if not title:
                    continue
                items.append({
                    "source": "财联社电报",
                    # ctime 为 epoch 秒，统一格式化为可读时间
                    "time": _fmt_news_time(item.get("ctime", "")),
                    "title": title,
                    "brief": str(item.get("brief", ""))[:200],
                })
            if len(items) >= limit:
                break
            # 用本页最旧一条的 ctime 作为下一页翻页锚点，轻微限速防限流
            last_time = min(int(it.get("ctime", last_time)) for it in roll)
            time.sleep(0.3)
    except Exception as e:
        logger.warning("财联社电报获取失败: %s", e)
    return items[:limit]


def fetch_news_pool(sector_keywords: list = None, days: int = 3) -> dict:
    """统一新闻聚合池：并行拉取 5 个新闻源，跨源按标题去重。

    参数：
        sector_keywords: 行业关键词列表，首个关键词用作智研 MCP 搜索词（缺省 "A股"）
        days: 新浪/Tushare 回溯天数（每天各拉取一次，覆盖 48 小时以上）

    返回：
        {'sina': [...], 'eastmoney': [...], 'mcp': [...], 'cls': [...], 'tushare': [...]}
        每条统一为 {'title': str, 'time': str, 'source': str}；
        单源失败只降级为该源空列表，不影响其他源。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    search_kw = sector_keywords[0] if sector_keywords else "A股"

    # 回溯日期列表：今天起往前 days 天（新浪用 YYYY-MM-DD，Tushare 用 YYYYMMDD）
    today = datetime.now()
    date_list = [(today - timedelta(days=i)) for i in range(days)]

    def _job_sina():
        """新浪：每天最多 100 条（内部自动翻页），逐日串行避免触发限流。"""
        pool = []
        for d in date_list:
            pool.extend(fetch_sina_news(100, d.strftime("%Y-%m-%d")))
        return pool

    def _job_eastmoney():
        """东方财富：7x24 实时快讯第 1 页。"""
        return fetch_eastmoney_news(50)

    def _job_mcp():
        """智研 MCP：按关键词搜索快讯。"""
        return fetch_mcp_news(search_kw, 60)

    def _job_cls():
        """财联社电报：实时滚动。"""
        return fetch_cls_telegraph(50)

    def _job_tushare():
        """Tushare 新闻：逐日拉取。"""
        pool = []
        for d in date_list:
            pool.extend(fetch_tushare_news(d.strftime("%Y%m%d"), 40))
        return pool

    jobs = {
        "sina": _job_sina,
        "eastmoney": _job_eastmoney,
        "mcp": _job_mcp,
        "cls": _job_cls,
        "tushare": _job_tushare,
    }

    # 并行拉取；单源异常只记日志并降级为空列表
    raw = {name: [] for name in jobs}
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        future_map = {executor.submit(fn): name for name, fn in jobs.items()}
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                raw[name] = future.result() or []
            except Exception as e:
                logger.warning("新闻池源 %s 拉取失败，已降级为空: %s", name, e)
                raw[name] = []

    # 统一为 {'title','time','source'} 并跨源按标题去重
    # 优先级：实时源（东财/财联社）在前，保证重复新闻保留时间更准的一条
    today_str = today.strftime("%Y-%m-%d")
    seen_titles = set()
    pool = {}
    for name in ("eastmoney", "cls", "mcp", "sina", "tushare"):
        unified = []
        for it in raw[name]:
            title = (it.get("title", "") or "").strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            unified.append({
                "title": title,
                "time": _fmt_news_time(it.get("time", ""), today_str),
                "source": it.get("source", "") or name,
            })
        pool[name] = unified

    total = sum(len(v) for v in pool.values())
    logger.info("新闻池聚合完成: %s，共 %d 条（去重后）",
                {k: len(v) for k, v in pool.items()}, total)
    return pool


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


def aggregate_northbound_by_sector(holdings: list) -> dict:
    """将北向持仓按行业聚合，返回各行业持股数量(亿股)排名。"""
    _load_stock_names()
    sector_holdings = {}
    for h in holdings:
        sector = _stock_sector_cache.get(h["code"], "")
        if not sector:
            continue
        sector_holdings[sector] = sector_holdings.get(sector, 0) + h["vol"]
    # 按持股数量排序
    sorted_sectors = sorted(sector_holdings.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_sectors[:15])


def fetch_sector_volume_all(date: str) -> dict:
    """全市场31行业成交额（成分股合计口径）。一次API拉全部个股。"""
    pro = _get_pro()
    if not pro:
        return {}

    _load_stock_names()
    result = {}

    try:
        # 一次拉全部个股日线
        df = pro.daily(trade_date=date)
        if df is None or df.empty:
            return result

        # 按行业聚合
        sector_amount = {}
        sector_vol = {}
        for _, row in df.iterrows():
            code = row["ts_code"]
            sector = _stock_sector_cache.get(code, "")
            if not sector:
                continue
            amt = float(row.get("amount", 0) or 0) / 1e5  # 千元→亿
            vol = float(row.get("vol", 0) or 0) / 10000    # 手→万手
            sector_amount[sector] = sector_amount.get(sector, 0) + amt
            sector_vol[sector] = sector_vol.get(sector, 0) + vol

        for sector in sector_amount:
            result[sector] = {
                "amount": round(sector_amount[sector], 2),
                "vol": round(sector_vol[sector], 2),
            }
    except Exception:
        pass
    return result


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


# 股票代码→名称+行业缓存
_stock_name_cache: dict = {}
_stock_sector_cache: dict = {}

def _load_stock_names():
    """加载全部A股代码→名称+行业映射。"""
    global _stock_name_cache, _stock_sector_cache
    if _stock_name_cache:
        return
    pro = _get_pro()
    if not pro:
        return
    try:
        df = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,industry")
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                _stock_name_cache[row["ts_code"]] = row.get("name", "")
                _stock_sector_cache[row["ts_code"]] = row.get("industry", "")
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
        # 板块合计成交额（成分股口径）
        total_amount = float(daily_df["amount"].sum()) if "amount" in daily_df.columns else 0
        total_vol = float(daily_df["vol"].sum()) if "vol" in daily_df.columns else 0
        result["total_amount"] = round(total_amount / 1e5, 2)  # 千元→亿
        result["total_vol"] = round(total_vol / 10000, 2)      # 手→万手

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
                    "lg_net": round((buy_lg - sell_lg) / 10000, 2),  # 万元→亿
                    "md_net": round((buy_md - sell_md) / 10000, 2),
                    "sm_net": round((buy_sm - sell_sm) / 10000, 2),
                }
        except Exception:
            pass

        return result

    except Exception:
        return result


# ═══════════════════════════════════════════
# 板块深度分析（估值水位 + 资金博弈）
# ═══════════════════════════════════════════

# 常见板块俗称/二级行业名 → 申万一级行业（用于板块名兜底映射）
SW_SECTOR_ALIAS = {
    "半导体": "电子", "芯片": "电子", "白酒": "食品饮料",
    "券商": "非银金融", "保险": "非银金融", "医药": "医药生物",
    "医疗": "医药生物", "军工": "国防军工", "地产": "房地产",
    "光伏": "电气设备", "锂电": "电气设备", "新能源": "电气设备",
    "新能源车": "汽车", "AI": "计算机", "人工智能": "计算机",
    "游戏": "传媒", "影视": "传媒", "5G": "通信",
}


def _get_sector_member_codes(pro, sector_name: str):
    """获取申万行业成分股 ts_code 列表，失败返回 None。"""
    name = sector_name.strip()
    idx_code = SW_INDEX_MAP.get(name)
    if not idx_code:
        # 兜底：俗称别名 → 包含匹配
        alias = SW_SECTOR_ALIAS.get(name)
        if not alias:
            for key in SW_INDEX_MAP:
                if key in name or name in key:
                    alias = key
                    break
        if alias:
            idx_code = SW_INDEX_MAP.get(alias)
    if not idx_code:
        logger.warning("未找到板块对应的申万指数 sector=%s", sector_name)
        return None
    try:
        df_member = pro.index_member(index_code=idx_code)
        if df_member is None or df_member.empty:
            return None
        return df_member["con_code"].tolist()
    except Exception:
        logger.warning("获取板块成分股失败 sector=%s", sector_name, exc_info=True)
        return None


def _api_with_date_fallback(pro, api_name: str, trade_date: str, max_back: int = 5):
    """
    调用 tushare 全市场接口（如 daily_basic/moneyflow）；
    当日无数据（非交易日）时逐日回退，最多回退 max_back 天。
    返回 (df, 实际交易日)，失败返回 (None, None)。
    """
    dt = datetime.strptime(trade_date, "%Y%m%d")
    for i in range(max_back + 1):
        d = (dt - timedelta(days=i)).strftime("%Y%m%d")
        try:
            df = getattr(pro, api_name)(trade_date=d)
            if df is not None and not df.empty:
                return df, d
        except Exception:
            logger.warning("%s 调用失败 date=%s", api_name, d, exc_info=True)
    return None, None


def _weighted_pe_pb(df):
    """
    按总市值（total_mv，单位万元，加权时单位可约掉）计算板块加权 PE/PB。
    PE 剔除 pe<=0 的亏损股；PB 用全部有值股。
    """
    pe = pb = None
    if df is None or df.empty:
        return pe, pb
    df = df.dropna(subset=["total_mv"])
    df = df[df["total_mv"] > 0]
    if df.empty:
        return pe, pb
    # PE：剔除亏损股（pe<=0）后按市值加权
    df_pe = df.dropna(subset=["pe"])
    df_pe = df_pe[df_pe["pe"] > 0]
    if not df_pe.empty:
        pe = round(float((df_pe["pe"] * df_pe["total_mv"]).sum() / df_pe["total_mv"].sum()), 2)
    # PB：全部有值股按市值加权
    df_pb = df.dropna(subset=["pb"])
    if not df_pb.empty:
        pb = round(float((df_pb["pb"] * df_pb["total_mv"]).sum() / df_pb["total_mv"].sum()), 2)
    return pe, pb


def fetch_sector_valuation(sector_name: str, trade_date: str) -> dict:
    """
    板块估值水位：成分股 daily_basic 按总市值加权 PE/PB + 近 1 年历史分位。
    申万行业指数（801xx0.SI）本身无 daily_basic 数据，必须用成分股聚合。
    trade_date 格式 YYYYMMDD；任何一步失败对应字段为 None，note 说明原因。
    """
    result = {"pe": None, "pb": None, "pe_percentile": None, "pb_percentile": None,
              "sample_count": 0, "note": ""}
    notes = []

    pro = _get_pro()
    if not pro:
        result["note"] = "Tushare 未配置或初始化失败"
        return result

    # 1. 成分股列表
    stocks = _get_sector_member_codes(pro, sector_name)
    if not stocks:
        result["note"] = f"未找到板块「{sector_name}」或成分股为空"
        return result

    # 2. 当日全市场 daily_basic（非交易日最多回退 5 天），过滤成分股
    df_all, actual_date = _api_with_date_fallback(pro, "daily_basic", trade_date)
    if df_all is None:
        result["note"] = f"daily_basic 自 {trade_date} 回退 5 天均无数据"
        return result
    df = df_all[df_all["ts_code"].isin(stocks)]
    if df.empty:
        result["note"] = "成分股当日无 daily_basic 数据"
        return result
    result["sample_count"] = len(df)
    result["pe"], result["pb"] = _weighted_pe_pb(df)
    if result["pe"] is None and result["pb"] is None:
        result["note"] = "成分股 pe/pb 字段全为空"
        return result
    if result["pe"] is None:
        notes.append("成分股 pe 全为空或全为亏损股")
    if result["pb"] is None:
        notes.append("成分股 pb 全为空")

    # 3. 历史分位：近 1 年每月月末交易日采样约 12 个点
    #    优先用 trade_cal 取真实月末交易日；失败则退化为自然月末+逐日回退
    sample_dates = []
    try:
        dt = datetime.strptime(actual_date, "%Y%m%d")
        start_1y = (dt - timedelta(days=370)).strftime("%Y%m%d")
        cal = pro.trade_cal(exchange="SSE", start_date=start_1y,
                            end_date=actual_date, is_open="1")
        if cal is None or cal.empty:
            raise ValueError("trade_cal 返回为空")
        by_month = {}
        for d in sorted(cal["cal_date"].tolist()):
            by_month[str(d)[:6]] = str(d)  # 每月最后一个交易日
        sample_dates = [d for d in sorted(by_month.values()) if d != actual_date][-12:]
        if not sample_dates:
            raise ValueError("trade_cal 无可用历史月末交易日")
    except Exception:
        logger.warning("trade_cal 获取月末交易日失败，改用自然月末兜底", exc_info=True)
        dt = datetime.strptime(actual_date, "%Y%m%d")
        year, month = dt.year, dt.month
        for _ in range(12):
            month -= 1
            if month == 0:
                month = 12
                year -= 1
            last_day = (datetime(year, 12, 31) if month == 12
                        else datetime(year, month + 1, 1) - timedelta(days=1))
            sample_dates.append(last_day.strftime("%Y%m%d"))
        sample_dates.sort()

    hist_pe, hist_pb = [], []
    for d in sample_dates:
        hdf, _ = _api_with_date_fallback(pro, "daily_basic", d)
        if hdf is not None:
            hdf = hdf[hdf["ts_code"].isin(stocks)]
            hpe, hpb = _weighted_pe_pb(hdf)
            if hpe is not None:
                hist_pe.append(hpe)
            if hpb is not None:
                hist_pb.append(hpb)
        time.sleep(0.3)  # 防限流

    # 分位 = 历史样本中 ≤ 当前值的比例（0-100 整数）；样本过少则不给分位
    if result["pe"] is not None and len(hist_pe) >= 6:
        result["pe_percentile"] = int(round(
            sum(1 for v in hist_pe if v <= result["pe"]) / len(hist_pe) * 100))
    else:
        notes.append(f"PE 历史样本不足（{len(hist_pe)} 点），分位缺失")
    if result["pb"] is not None and len(hist_pb) >= 6:
        result["pb_percentile"] = int(round(
            sum(1 for v in hist_pb if v <= result["pb"]) / len(hist_pb) * 100))
    else:
        notes.append(f"PB 历史样本不足（{len(hist_pb)} 点），分位缺失")

    result["note"] = "；".join(notes)
    return result


def fetch_sector_moneyflow(sector_name: str, trade_date: str) -> dict:
    """
    板块资金博弈：成分股 moneyflow 聚合。
    主力 = 特大单 + 大单；中小单 = 中单 + 小单。
    金额单位万元，÷10000 得亿元。trade_date 格式 YYYYMMDD；
    任何一步失败对应字段为 None/空列表，note 说明原因。
    """
    result = {"main_net": None, "retail_net": None, "top_inflow": [],
              "top_outflow": [], "stock_count": 0, "note": ""}

    pro = _get_pro()
    if not pro:
        result["note"] = "Tushare 未配置或初始化失败"
        return result

    # 1. 成分股列表
    stocks = _get_sector_member_codes(pro, sector_name)
    if not stocks:
        result["note"] = f"未找到板块「{sector_name}」或成分股为空"
        return result

    # 确保股票名称缓存已加载
    _load_stock_names()

    # 2. 全市场当日资金流（非交易日最多回退 5 天），过滤成分股
    df_all, actual_date = _api_with_date_fallback(pro, "moneyflow", trade_date)
    if df_all is None:
        result["note"] = f"moneyflow 自 {trade_date} 回退 5 天均无数据"
        return result
    import pandas as pd
    df = df_all[df_all["ts_code"].isin(stocks)].copy()
    if df.empty:
        result["note"] = "成分股当日无资金流数据"
        return result

    # 3. 聚合计算（金额单位：万元 → 亿元，÷10000）
    try:
        flow_cols = ["buy_elg_amount", "sell_elg_amount",
                     "buy_lg_amount", "sell_lg_amount",
                     "buy_md_amount", "sell_md_amount",
                     "buy_sm_amount", "sell_sm_amount"]
        for col in flow_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        main_net_wy = (df["buy_elg_amount"].sum() + df["buy_lg_amount"].sum()
                       - df["sell_elg_amount"].sum() - df["sell_lg_amount"].sum())
        retail_net_wy = (df["buy_md_amount"].sum() + df["buy_sm_amount"].sum()
                         - df["sell_md_amount"].sum() - df["sell_sm_amount"].sum())
        result["main_net"] = round(float(main_net_wy) / 10000, 2)    # 万元→亿
        result["retail_net"] = round(float(retail_net_wy) / 10000, 2)  # 万元→亿
        result["stock_count"] = len(df)

        # 个股主力净流入排行（万元→亿），取净流入/净流出前 5
        df["main_net_ind"] = (df["buy_elg_amount"] + df["buy_lg_amount"]
                              - df["sell_elg_amount"] - df["sell_lg_amount"]) / 10000
        df = df.sort_values("main_net_ind", ascending=False)
        result["top_inflow"] = [
            (_stock_name(r["ts_code"]), round(float(r["main_net_ind"]), 2))
            for _, r in df.head(5).iterrows() if r["main_net_ind"] > 0
        ]
        result["top_outflow"] = [
            (_stock_name(r["ts_code"]), round(float(r["main_net_ind"]), 2))
            for _, r in df.tail(5).iloc[::-1].iterrows() if r["main_net_ind"] < 0
        ]
    except Exception:
        logger.warning("板块资金流聚合失败 sector=%s", sector_name, exc_info=True)
        result["note"] = "资金流聚合计算失败"

    return result


# 业绩预告类型 → 预喜/预忧分类
_FORECAST_POSITIVE_TYPES = {"预增", "略增", "扭亏", "减亏"}
_FORECAST_NEGATIVE_TYPES = {"预减", "略减", "首亏", "续亏", "增亏"}

# 报告期 MMDD → 中文名称
_PERIOD_NAME_MAP = {"0331": "一季报", "0630": "中报", "0930": "三季报", "1231": "年报"}


def _period_label(end_date: str) -> str:
    """报告期 YYYYMMDD → 「2026中报」样式。"""
    if not end_date or len(end_date) < 8:
        return ""
    return f"{end_date[:4]}{_PERIOD_NAME_MAP.get(end_date[4:8], end_date[4:8])}"


def fetch_sector_earnings(sector_name: str, trade_date: str) -> dict:
    """
    板块景气度：成分股业绩预告聚合 + 业绩快报增强。
    从 trade_date 往前回溯约 120 天、按周采样 ann_date 调 pro.forecast，
    过滤成分股后每只股票只保留最新一条预告；
    p_change_min/max 本身是百分数数值，取二者中值代表变动幅度。
    trade_date 格式 YYYYMMDD；任何一步失败对应字段为安全默认值，note 说明原因。
    """
    result = {"total_forecast": 0, "positive_count": 0, "negative_count": 0,
              "positive_ratio": 0.0, "median_change": None,
              "top_improvers": [], "top_decliners": [],
              "express_count": 0, "period": "", "note": ""}
    notes = []

    pro = _get_pro()
    if not pro:
        result["note"] = "Tushare 未配置或初始化失败"
        return result

    # 1. 成分股列表
    stocks = _get_sector_member_codes(pro, sector_name)
    if not stocks:
        result["note"] = f"未找到板块「{sector_name}」或成分股为空"
        return result
    stock_set = set(stocks)

    # 确保股票名称缓存已加载
    _load_stock_names()

    # 2. 业绩预告：往前 120 天按周采样 ann_date（约 17 次调用），
    #    过滤成分股，每只股票只保留公告日期最新的一条
    latest = {}  # ts_code -> (ann_date, 记录dict)
    dt = datetime.strptime(trade_date, "%Y%m%d")
    api_fail = 0
    for offset in range(0, 119, 7):
        d = (dt - timedelta(days=offset)).strftime("%Y%m%d")
        try:
            df = pro.forecast(ann_date=d)
            if df is not None and not df.empty:
                df = df[df["ts_code"].isin(stock_set)]
                for _, r in df.iterrows():
                    code = r["ts_code"]
                    ann = str(r.get("ann_date", ""))
                    if code not in latest or ann >= latest[code][0]:
                        latest[code] = (ann, r.to_dict())
        except Exception:
            api_fail += 1
            logger.warning("forecast 调用失败 ann_date=%s", d, exc_info=True)
        time.sleep(0.3)  # 防限流

    if api_fail:
        notes.append(f"业绩预告采样有 {api_fail} 次调用失败")

    # 3. 聚合统计
    records = [rec for _, rec in latest.values()]
    result["total_forecast"] = len(records)

    if records:
        import statistics
        for rec in records:
            ftype = str(rec.get("type", "")).strip()
            if ftype in _FORECAST_POSITIVE_TYPES:
                result["positive_count"] += 1
            elif ftype in _FORECAST_NEGATIVE_TYPES:
                result["negative_count"] += 1
        classified = result["positive_count"] + result["negative_count"]
        if classified:
            result["positive_ratio"] = round(
                result["positive_count"] / classified * 100, 1)

        # 变动幅度：取 p_change_min/max 中值（百分数数值）
        def _mid_change(rec):
            try:
                lo = float(rec.get("p_change_min"))
                hi = float(rec.get("p_change_max"))
                return (lo + hi) / 2
            except (TypeError, ValueError):
                return None

        with_chg = [(rec, _mid_change(rec)) for rec in records]
        with_chg = [(rec, c) for rec, c in with_chg if c is not None]
        if with_chg:
            result["median_change"] = round(
                float(statistics.median(c for _, c in with_chg)), 1)
            by_chg = sorted(with_chg, key=lambda x: x[1], reverse=True)
            result["top_improvers"] = [
                (_stock_name(r["ts_code"]), str(r.get("type", "")).strip(),
                 round(c, 1))
                for r, c in by_chg[:3] if c > 0
            ]
            result["top_decliners"] = [
                (_stock_name(r["ts_code"]), str(r.get("type", "")).strip(),
                 round(c, 1))
                for r, c in by_chg[::-1][:3] if c < 0
            ]
        else:
            notes.append("预告变动幅度字段缺失，无法统计中位数")

        # 报告期：取最新预告中出现最多的 end_date
        periods = [str(rec.get("end_date", "")) for rec in records
                   if rec.get("end_date")]
        if periods:
            common = max(set(periods), key=periods.count)
            result["period"] = _period_label(common)

        if len(records) < 5:
            notes.append("处于财报披露真空期，样本有限")
    else:
        notes.append("近120天内成分股无业绩预告，处于财报披露真空期，样本有限")

    # 4. 业绩快报（可选增强）：拉最近一个报告期数据，失败就跳过
    try:
        if result["period"]:
            # 由预告报告期反推 end_date；没有预告则按 trade_date 推最近季度末
            year = result["period"][:4]
            mmdd = next(k for k, v in _PERIOD_NAME_MAP.items()
                        if v in result["period"])
            express_period = f"{year}{mmdd}"
        else:
            q_ends = [(dt.year, md) for md in ("1231", "0930", "0630", "0331")]
            q_ends += [(dt.year - 1, "1231")]
            express_period = next(
                f"{y}{md}" for y, md in q_ends
                if datetime.strptime(f"{y}{md}", "%Y%m%d") < dt)
        df_exp = pro.express(period=express_period)
        if df_exp is not None and not df_exp.empty:
            df_exp = df_exp[df_exp["ts_code"].isin(stock_set)]
            result["express_count"] = len(df_exp)
            if not df_exp.empty and "yoy_net_profit" in df_exp.columns:
                import pandas as pd
                import statistics
                yoy = pd.to_numeric(df_exp["yoy_net_profit"],
                                    errors="coerce").dropna()
                # 剔除负基数导致的极端值（|同比|>1000% 无分析意义）
                yoy = yoy[yoy.abs() <= 1000]
                if len(yoy) >= 3:
                    notes.append(
                        f"快报样本 {result['express_count']} 家，"
                        f"净利润同比增速中位数 {round(float(statistics.median(yoy)), 1)}%")
                elif result["express_count"] > 0:
                    notes.append(
                        f"快报样本仅 {result['express_count']} 家，"
                        "样本过少，同比增速中位数不具参考价值")
    except Exception:
        logger.warning("业绩快报获取失败 sector=%s", sector_name, exc_info=True)
        notes.append("业绩快报获取失败，已跳过")

    result["note"] = "；".join(notes)
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

def fetch_forex() -> dict:
    """最新汇率。symbol: USDCNY"""
    d = _mcp_call("forexQuoteLatest", {"symbol": "USDCNY"})
    data = d.get("data",{}) or {}
    return {"在岸人民币": f"{data.get('price','?')}", "涨跌": f"{data.get('pctChg','?')}%"}


def fetch_forex_batch() -> list:
    """批量汇率。"""
    d = _mcp_call("forexQuotesBatch", {"from": "USD", "to": "CNY,JPY,EUR"})
    items = []
    data = d.get("status",{}).get("data",[]) or d.get("data",[]) or []
    for it in data:
        items.append(f"{it.get('symbol','?')}:{it.get('price','?')}")
    return items[:5]


def fetch_futures(market: str = "gn", symbol: str = "AU0") -> dict:
    """期货行情。market: gn内盘/global外盘/cff股指。
    主路径：新浪智研 MCP；主路径失败时回退 yfinance 商品期货 + 人民币汇率。"""
    d = _mcp_call("future_quotes", {"market": market, "symbol": symbol})
    data = d.get("data",{}) or {}
    if data.get("price"):
        return {"name": data.get("name",""), "price": data.get("price",""),
                "pct": data.get("percent",""), "vol": data.get("volume",""),
                "open": data.get("openPrice",""), "high": data.get("high",""), "low": data.get("low","")}

    # fallback：商品期货 + 人民币汇率（yfinance 免费）
    logger.warning("MCP future_quotes 无数据（market=%s symbol=%s），回退 yfinance", market, symbol)
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
            except Exception as e:
                logger.warning("yfinance 获取 %s 失败", sym, exc_info=True)
    except Exception as e:
        logger.warning("yfinance 兜底整体失败", exc_info=True)
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

    # 新浪历史日期
    trade_dt = datetime.strptime(date, "%Y%m%d")
    d1 = trade_dt.strftime("%Y-%m-%d")
    d2 = (trade_dt - timedelta(days=1)).strftime("%Y-%m-%d")

    # 全部并行执行
    tasks = {
        "indices": loop.run_in_executor(None, fetch_a_share_indices, date),
        "sectors": loop.run_in_executor(None, fetch_shenwan_sectors, date),
        "flows": loop.run_in_executor(None, fetch_fund_flows, date),
        "gidx": loop.run_in_executor(None, fetch_global_indices),
        "broker": loop.run_in_executor(None, fetch_broker_recommendations),
        "forex": loop.run_in_executor(None, fetch_forex),
        "shibor": loop.run_in_executor(None, fetch_shibor),
        "north": loop.run_in_executor(None, fetch_north_holdings),
        "toplist": loop.run_in_executor(None, fetch_top_list, date),
        "cn_macro": loop.run_in_executor(None, fetch_china_macro),
        "us_macro": loop.run_in_executor(None, fetch_us_macro),
        "em_p1": loop.run_in_executor(None, fetch_eastmoney_news, 100),
        "em_p2": loop.run_in_executor(None, fetch_eastmoney_news_page2, 100),
        "em_p3": loop.run_in_executor(None, fetch_eastmoney_news_page3, 100),
        "sina": loop.run_in_executor(None, fetch_sina_news, 30, d1),
        "sina2": loop.run_in_executor(None, fetch_sina_news, 30, d2),
        "fh_news": loop.run_in_executor(None, fetch_finnhub_news, 20),
        "calendar": loop.run_in_executor(None, fetch_economic_calendar),
        "breadth": loop.run_in_executor(None, fetch_market_breadth),
        "hot": loop.run_in_executor(None, fetch_hot_stocks),
        "us_breadth": loop.run_in_executor(None, fetch_us_breadth),
        "limit_up": loop.run_in_executor(None, fetch_limit_up_pool),
        "lian_ban": loop.run_in_executor(None, fetch_lian_ban),
        "forecast": loop.run_in_executor(None, fetch_forecast, date),
        "express": loop.run_in_executor(None, fetch_express, date),
        "block": loop.run_in_executor(None, fetch_block_trades, "", date),
        "ggt": loop.run_in_executor(None, fetch_ggt_daily),
        "repurchase": loop.run_in_executor(None, fetch_repurchase, date),
        "share_float": loop.run_in_executor(None, fetch_share_float, date),
        "funds": loop.run_in_executor(None, fetch_fund_list, "E"),
        "strong_sec": loop.run_in_executor(None, fetch_strong_sectors),
        "north_flow": loop.run_in_executor(None, fetch_northbound_flow),
        "us_sec": loop.run_in_executor(None, fetch_us_sectors),
        "hk_sec": loop.run_in_executor(None, fetch_hk_sectors),
        "ts_news": loop.run_in_executor(None, fetch_tushare_news, date),
        "cls_telegraph": loop.run_in_executor(None, fetch_cls_telegraph, 30),
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
    snapshot._forex = safe(results_raw.get("forex"), {})
    snapshot._shibor = safe(results_raw.get("shibor"), {})
    snapshot._north_hold = safe(results_raw.get("north"), [])
    snapshot._north_sector = aggregate_northbound_by_sector(snapshot._north_hold)
    snapshot._sector_volumes = await loop.run_in_executor(None, fetch_sector_volume_all, date)
    snapshot._breadth = safe(results_raw.get("breadth"), {})
    snapshot._hot = safe(results_raw.get("hot"), [])
    snapshot._us_breadth = safe(results_raw.get("us_breadth"), {})
    snapshot._strong_sec = safe(results_raw.get("strong_sec"), [])
    snapshot._north_flow = safe(results_raw.get("north_flow"), [])
    snapshot._us_sec = safe(results_raw.get("us_sec"), [])
    snapshot._hk_sec = safe(results_raw.get("hk_sec"), [])
    snapshot._limit_up = safe(results_raw.get("limit_up"), [])
    snapshot._lian_ban = safe(results_raw.get("lian_ban"), [])
    snapshot._forecast = safe(results_raw.get("forecast"), [])
    snapshot._express = safe(results_raw.get("express"), [])
    snapshot._block = safe(results_raw.get("block"), [])
    snapshot._ggt = safe(results_raw.get("ggt"), [])
    snapshot._repurchase = safe(results_raw.get("repurchase"), [])
    snapshot._share_float = safe(results_raw.get("share_float"), [])
    snapshot._funds = safe(results_raw.get("funds"), [])
    snapshot._top_list = safe(results_raw.get("toplist"), [])
    snapshot.macro_data = {
        "china": safe(results_raw.get("cn_macro"), {}),
        "us": safe(results_raw.get("us_macro"), {}),
    }
    # 去重（用标题前60字做key）
    def _dedup(items):
        seen = set()
        out = []
        for it in items:
            key = it.get("title", "")[:60]
            if key not in seen:
                seen.add(key)
                out.append(it)
        return out

    all_em = _dedup(safe(results_raw.get("em_p1"), []) + safe(results_raw.get("em_p2"), []) + safe(results_raw.get("em_p3"), []))
    all_sina = _dedup(safe(results_raw.get("sina"), []) + safe(results_raw.get("sina2"), []))
    if sector_focus:
        all_em = filter_news_by_sector(all_em, sector_focus)
        # Sina已按日期过滤，不再按行业过滤，保留全部
    snapshot.news_items = {
        "eastmoney": all_em,
        "sina": all_sina,
        "ts_news": safe(results_raw.get("ts_news"), []),
        "cls_telegraph": safe(results_raw.get("cls_telegraph"), []),
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
            amt = d.get("amount")
            vol_str = f"成交额{amt}亿" if amt else f"成交量{d['vol']}万手"
            lines.append(
                f"- {name}: {d['close']} | {d['pct_chg']:+.2f}%"
                f" | {vol_str}{extra_str}"
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
            # 成分股合计成交额
            sec_vols = getattr(snapshot, "_sector_volumes", {})
            if s["name"] in sec_vols:
                parts.append(f"成交{sec_vols[s['name']]['amount']:.0f}亿")
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

    # ── 新浪历史新闻（放前面，覆盖48h）──
    sina = snapshot.news_items.get("sina", [])
    if sina:
        lines.append(f"### 新浪财经历史新闻（交易日+前日，共{len(sina)}条，覆盖48小时）")
        for item in sina[:10]:
            lines.append(f"- [{item['time']}] {item['title']}")

    # ── 东方财富实时（补充最新）──
    em = snapshot.news_items.get("eastmoney", [])
    if em:
        lines.append(f"### 东方财富7x24实时快讯（共{len(em)}条）")
        for item in em[:10]:
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
    fx = getattr(snapshot, "_forex", {})
    if fx:
        lines.append(f"### 汇率 在岸人民币 {fx.get('在岸人民币','?')} {fx.get('涨跌','?')}")

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

    # ── 涨跌分布 + 热搜 ──
    breadth = getattr(snapshot, "_breadth", {})
    if breadth:
        lines.append(f"### A股全市场涨跌分布（{breadth.get('date','')}）")
        lines.append(f"涨停{breadth.get('涨停','?')}家 跌停{breadth.get('跌停','?')}家 上涨{breadth.get('total_up','?')}家 下跌{breadth.get('total_down','?')}家 平盘{breadth.get('平','?')}家")
    us_b = getattr(snapshot, "_us_breadth", {})
    if us_b:
        lines.append(f"### 美股涨跌分布 涨{us_b.get('涨','?')}家 跌{us_b.get('跌','?')}家 平{us_b.get('平','?')}家")
    north_f = getattr(snapshot, "_north_flow", [])
    if north_f:
        lines.append("### 北向资金持仓TOP10（MCP实时）")
        for h in north_f[:10]:
            lines.append(f"- {h['name']}({h['code']}) 持仓{h.get('hold','?')} 日变动{h.get('chg','?')}")
    hot = getattr(snapshot, "_hot", [])
    if hot:
        lines.append("### 股票热搜榜TOP10")
        for h in hot[:10]:
            lines.append(f"- {h['name']}({h['code']}) 热度{h.get('heat','?')}")
    strong = getattr(snapshot, "_strong_sec", [])
    if strong:
        lines.append("### 强势行业板块（MCP）")
        names = ", ".join(f"{s['name']}({s['pct']})" for s in strong[:8])
        lines.append(names)
    us_sec = getattr(snapshot, "_us_sec", [])
    if us_sec:
        lines.append("### 美股板块表现TOP10")
        for s in us_sec[:8]:
            lines.append(f"- {s['name']}: {s['pct']}")
    hk_sec = getattr(snapshot, "_hk_sec", [])
    if hk_sec:
        lines.append("### 港股板块表现TOP10")
        for s in hk_sec[:8]:
            lines.append(f"- {s['name']}: {s['pct']}")
    limit_up = getattr(snapshot, "_limit_up", [])
    if limit_up:
        lines.append(f"### A股涨停池（{len(limit_up)}只）")
        for s in limit_up[:10]:
            lines.append(f"- {s['name']}({s['code']}) {s['pct']} {s.get('reason','')}")
    lian_ban = getattr(snapshot, "_lian_ban", [])
    if lian_ban:
        lines.append("### 连板个股")
        for s in lian_ban[:8]:
            lines.append(f"- {s['name']}({s['code']}) {s.get('count','')}连板")
    ggt = getattr(snapshot, "_ggt", [])
    if ggt:
        latest = ggt[-1]
        lines.append(f"### 港股通资金流向 买入{latest.get('buy','?')} 卖出{latest.get('sell','?')}")
    rep = getattr(snapshot, "_repurchase", [])
    if rep:
        lines.append(f"### 今日回购（{len(rep)}只）")
        for r in rep[:8]:
            lines.append(f"- {r['code']} 回购{r.get('vol','?')}万股")
    sf = getattr(snapshot, "_share_float", [])
    if sf:
        lines.append(f"### 近期限售解禁（{len(sf)}只）")
        for r in sf[:8]:
            lines.append(f"- {r['code']} {r.get('date','?')} 解禁{r.get('share','?')}万股")
    fc = getattr(snapshot, "_forecast", [])
    if fc:
        lines.append(f"### 今日业绩预告（{len(fc)}只）")
        for r in fc[:8]:
            lines.append(f"- {r['code']} {r.get('type','?')} 变动{r.get('p_min','?')}%~{r.get('p_max','?')}%")
    ex = getattr(snapshot, "_express", [])
    if ex:
        lines.append(f"### 今日业绩快报（{len(ex)}只）")
        for r in ex[:8]:
            lines.append(f"- {r['code']} 营收{r.get('revenue','?')} 利润{r.get('profit','?')}")
    bt = getattr(snapshot, "_block", [])
    if bt:
        lines.append(f"### 今日大宗交易（{len(bt)}笔）")
        for r in bt[:8]:
            lines.append(f"- {r['code']} {r.get('date','?')} 成交{r.get('amount','?')}万元")

    # ── 北向持仓TOP + 行业分布 ──
    north_h = getattr(snapshot, "_north_hold", [])
    if north_h:
        lines.append("### 北向资金持仓TOP20")
        for h in north_h[:15]:
            lines.append(f"- {h['name']}({h['code']}) 持仓{h['vol']}亿股 占比{h['ratio']}%")
        # 北向行业分布
        north_sec = getattr(snapshot, "_north_sector", {})
        if north_sec:
            lines.append("北向持仓行业分布（持股数量排名）:")
            for sec, val in list(north_sec.items())[:10]:
                lines.append(f"  {sec}: {val:.0f}亿股")
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

        if stock_detail.get("total_amount"):
            lines.append(f"板块合计成交额: {stock_detail['total_amount']}亿（成分股口径） 成交量: {stock_detail['total_vol']}万手")

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
            if ff['lg_net'] == 0 and ff['md_net'] == 0 and ff['sm_net'] == 0:
                lines.append("资金流向：数据暂缺（接口返回为空）")
            else:
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
