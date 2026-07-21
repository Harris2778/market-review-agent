"""
Agent 编排器：意图识别 → 数据采集 → LLM生成 → 响应格式化。
"""

import re
import json
import os
import time
import asyncio
from datetime import datetime, timedelta
from typing import AsyncGenerator, Optional

from openai import AsyncOpenAI
from .data_fetcher import collect_market_snapshot, format_market_data_for_prompt
from .system_prompts import get_system_prompt


def _get_latest_trade_date(ref_date: datetime) -> datetime:
    """获取最近一个交易日。用 Tushare 交易日历，失败则用周末规则兜底。"""
    token = os.getenv("TUSHARE_TOKEN", "")
    if token:
        try:
            import tushare as ts
            ts.set_token(token)
            pro = ts.pro_api()
            start = (ref_date - timedelta(days=10)).strftime("%Y%m%d")
            end = ref_date.strftime("%Y%m%d")
            df = pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
            if df is not None and not df.empty:
                trading_days = df[df["is_open"] == 1]["cal_date"].sort_values(ascending=False)
                if len(trading_days) > 0:
                    return datetime.strptime(str(trading_days.iloc[0]), "%Y%m%d")
        except Exception:
            pass

    # 兜底：按周末规则
    d = ref_date
    while d.weekday() >= 5:  # 周六=5, 周日=6
        d = d - timedelta(days=1)
    return d

# ── 意图关键词 ──

MARKET_REVIEW_KEYWORDS = [
    # 直接指令
    "复盘", "每日复盘", "今日复盘", "昨天复盘", "收盘复盘", "盘后复盘",
    "市场回顾", "市场日报", "市场简报", "市场概览", "市场综述",
    "大盘分析", "大盘走势", "大盘总结", "大盘回顾",
    "行情分析", "行情总结", "行情回顾", "行情复盘",
    "市场分析", "市场总结", "市场复盘",
    "盘面分析", "盘面总结", "盘后总结", "收盘分析", "收盘总结",
    "A股复盘", "A股分析", "股市复盘", "股市分析",
    # 英文
    "market review", "market update", "daily review", "market recap",
    # 口语化
    "复盘一下", "分析一下市场", "总结一下今天", "梳理一下行情",
    # 时间+市场组合（单一词不够，需要组合命中）
    "今天市场", "今日大盘", "今天A股", "今天股市", "今天行情", "今天盘面",
    "昨天市场", "昨天大盘", "昨天A股", "昨天股市", "昨天行情", "昨天盘面",
    "今日市场", "今日A股", "今日股市", "今日行情", "今日盘面",
    "昨日市场", "昨日大盘", "昨日A股", "昨日股市", "昨日行情", "昨日盘面",
    # 市场提问
    "市场怎么样", "大盘怎么样", "行情怎么样", "A股怎么样",
    "市场如何", "大盘如何", "行情如何",
    "市场表现", "大盘表现", "A股表现", "股市表现",
    "今天怎么样了", "今天什么情况", "今天发生了什么",
    "红了没", "绿了没", "今天涨了没", "今天跌了没",
]

MARKET_REVIEW_PATTERNS = [
    # 时间词 + 市场词 + 可选提问
    r"(今天|今日|昨天|昨日|最近|近期|当下|当前).{0,6}(市场|大盘|行情|A股|股市|盘面|指数).{0,8}(怎么样|如何|什么情况|发生了什么|表现|走势|分析|复盘|总结|回顾|梳理)",
    # 动作词 + 市场词
    r"(回顾|复盘|分析|总结|梳理|看下|看看|聊|说说|讲).{0,4}(今天|今日|昨天|昨日|最近)?(市场|大盘|行情|A股|股市|盘面|指数)",
    r"(回顾|复盘|分析|总结|梳理|看下|看看).{0,10}(市场|大盘|行情|A股|股市)",
    # 市场词 + 提问词
    r"(市场|大盘|行情|A股|股市).{0,5}(怎么样|如何|什么|怎么|表现|情况|走势)",
    # 时间 + 怎么样（上下文明确）
    r"(今天|今日|昨天|昨日).*(涨|跌|红|绿).*(了|没)",
    r"(今天|今日|昨天|昨日).{0,3}(怎么样|如何)",
    # 盘后/收盘相关
    r"(收盘|盘后|收市).{0,4}(复盘|分析|总结|回顾|走势|情况)",
    # 日报/简报/概览类
    r"(市场|大盘|行情|A股).{0,4}(日报|简报|概览|综述|总结|回顾)",
    r"^(复盘|市场|大盘|行情).*",
]

SECTOR_KEYWORDS = [
    "聚焦", "深入看", "重点看", "只看", "重点关注",
    "板块复盘", "行业分析", "板块深度",
]

SECTOR_FOCUS_PATTERNS = [
    r"(聚焦|深入看|重点看|只看|重点关注).+",
    r".+(板块|行业).*(怎么样|分析|复盘|深度|聚焦)",
]

# 申万一级行业名称 → 标准化名称
SECTOR_NAME_MAP = {
    "半导体": "电子", "芯片": "电子", "集成电路": "电子",
    "新能源": "电气设备", "光伏": "电气设备", "锂电": "电气设备", "储能": "电气设备",
    "AI": "计算机", "人工智能": "计算机", "软件": "计算机",
    "白酒": "食品饮料", "食品": "食品饮料", "消费": "食品饮料",
    "医药": "医药生物", "创新药": "医药生物", "CRO": "医药生物", "生物医药": "医药生物",
    "军工": "国防军工", "航天": "国防军工",
    "新能源车": "汽车", "整车": "汽车",
    "银行": "银行", "大金融": "银行",
    "券商": "非银金融", "保险": "非银金融",
    "地产": "房地产",
    "通信": "通信", "5G": "通信", "6G": "通信",
    "传媒": "传媒", "游戏": "传媒", "互联网": "传媒",
    "电力": "公用事业", "公用事业": "公用事业",
    "石油": "石油石化", "石化": "石油石化",
    "建筑": "建筑装饰", "基建": "建筑装饰",
    "建材": "建筑材料", "水泥": "建筑材料",
    "家电": "家用电器", "白电": "家用电器",
    "农业": "农林牧渔", "养殖": "农林牧渔",
    "航运": "交通运输", "航空": "交通运输",
    "机械": "机械设备", "机器人": "机械设备",
    "零售": "商业贸易", "免税": "商业贸易",
    "旅游": "休闲服务", "酒店": "休闲服务",
    "化工": "化工", "钢铁": "钢铁", "煤炭": "煤炭",
    "有色": "有色金属", "稀土": "有色金属", "黄金": "有色金属",
    "电子": "电子", "计算机": "计算机", "环保": "环保",
    "纺织": "纺织服装", "服装": "纺织服装",
    "轻工": "轻工制造", "造纸": "轻工制造",
    "综合": "综合",
}


def detect_intent(message: str) -> tuple[str, Optional[str]]:
    """
    意图识别。
    返回 (intent_type, sector_name_or_None)
    intent_type: "market_review" | "sector_deep_dive" | "general_chat"
    """
    msg = message.strip()

    # 先提取行业名（如果有的话，优先判定为板块聚焦）
    sector = _extract_sector(msg)

    # 如果明确有聚焦关键词 + 行业 → 板块聚焦
    for sk in SECTOR_KEYWORDS:
        if sk in msg and sector:
            return ("sector_deep_dive", sector)

    # 如果提到行业名 + 复盘/回顾类关键词 → 板块聚焦（行业名优先级高于全市场）
    if sector:
        for kw in MARKET_REVIEW_KEYWORDS:
            if kw in msg:
                return ("sector_deep_dive", sector)
        for pattern in MARKET_REVIEW_PATTERNS:
            if re.search(pattern, msg):
                return ("sector_deep_dive", sector)

    # 全市场复盘
    for kw in MARKET_REVIEW_KEYWORDS:
        if kw in msg:
            return ("market_review", None)

    for pattern in MARKET_REVIEW_PATTERNS:
        if re.search(pattern, msg):
            return ("market_review", None)

    # 检查是不是单板块聚焦
    for pattern in SECTOR_FOCUS_PATTERNS:
        if re.search(pattern, msg):
            sector = _extract_sector(msg)
            if sector:
                return ("sector_deep_dive", sector)

    # 新闻专属模式：用户明确要新闻（含板块名+新闻、全市场新闻）
    news_patterns = [
        r".*(新闻|快讯|资讯).*(汇总|总结|盘点|梳理|报告)",
        r"有什么.*新闻", r"新闻.*怎么样",
        r".+板块.+新闻",  # "银行板块新闻"
        r".+行业.+新闻",  # "银行行业新闻"
        r"全市场.*新闻", r"新闻.*全市场",
        r"^(新闻|快讯|资讯)$",
    ]
    for pattern in news_patterns:
        if re.search(pattern, msg):
            sector = _extract_sector(msg)
            return ("news_only", sector)  # sector may be None for full market news

    # 期货查询（优先于行业匹配）
    if any(kw in msg for kw in ["期货", "黄金", "原油", "铜价", "螺纹钢", "铁矿石", "白银", "焦煤"]):
        return ("futures_query", msg)

    # 基金查询
    if any(kw in msg for kw in ["基金", "ETF", "净值", "申赎"]) and not any(kw in msg for kw in ["板块", "行业", "复盘"]):
        return ("fund_query", msg)

    # 直接提行业名——但如果含"新闻"则走新闻模式
    for kw, sw_name in SECTOR_NAME_MAP.items():
        if kw in msg:
            if "新闻" in msg:
                return ("news_only", sw_name)
            return ("sector_deep_dive", sw_name)

    # 提板块/行业 + 任何疑问词 → 板块聚焦
    for pattern in [r"(板块|行业|赛道).*(怎么样|如何|分析|复盘|表现|走势|情况|回顾)", r"(怎么看|分析一下|回顾一下).*(板块|行业|市场)"]:
        if re.search(pattern, msg):
            sector = _extract_sector(msg)
            if sector:
                return ("sector_deep_dive", sector)

    # 股票代码/名称查询
    stock_patterns = [r"分析.{0,4}[A-Z]{1,5}$", r"[A-Z]{1,5}.*(股价|行情|分析|怎么样)", r"(茅台|五粮液|宁德|比亚迪|中芯)"]
    for p in stock_patterns:
        if re.search(p, msg, re.IGNORECASE):
            return ("stock_query", msg)

    return ("general_chat", None)


def _extract_sector(msg: str) -> Optional[str]:
    """从消息中提取行业名并映射到申万一级。"""
    # 按长度降序匹配（先匹配长词如"新能源车"，再短词如"汽车"）
    sorted_kws = sorted(SECTOR_NAME_MAP.keys(), key=len, reverse=True)
    for kw in sorted_kws:
        if kw in msg:
            return SECTOR_NAME_MAP[kw]
    return None


# ── Markdown 清理 ──

def _clean_markdown(text: str) -> str:
    """后处理：强制清除所有markdown格式。"""
    import re

    # 1. 管道表格 → 缩进纯文本
    lines = text.split("\n")
    cleaned = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        # 检测管道表格行
        if stripped.startswith("|") and stripped.endswith("|"):
            if "---" in stripped or ":--" in stripped:
                in_table = True
                continue
            if in_table:
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                cleaned.append("  " + "  ".join(cells))
            continue
        in_table = False
        cleaned.append(line)
    text = "\n".join(cleaned)

    # 2. **加粗** → 去掉星号
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)

    # 3. # 标题 → 空行分隔
    text = re.sub(r"^#{1,4}\s*", "", text, flags=re.MULTILINE)

    # 4. * 列表 → - 列表
    text = re.sub(r"^[\*\+]\s+", "- ", text, flags=re.MULTILINE)

    # 5. 残留的单个 | 替换为空格
    text = text.replace(" | ", "  ")

    return text


def _format_all_news(snapshot, date_display: str) -> str:
    """新闻专属模式：全量48h，不过滤。"""
    lines = [f"48小时新闻全覆盖 — {date_display}及前一天"]
    lines.append("")

    # 新浪历史（交易日+前日，各30条，共60条）
    sina = snapshot.news_items.get("sina", [])
    if sina:
        # 按日期分组
        from collections import defaultdict
        by_date = defaultdict(list)
        for item in sina:
            t = item.get("time", "")[:10]
            by_date[t].append(item.get("title", ""))
        for d in sorted(by_date.keys(), reverse=True):
            lines.append(f"--- {d} ---")
            for title in by_date[d]:
                lines.append(f"[{d}] {title}")
        lines.append("")

    # 东方财富实时（最新补全）
    em = snapshot.news_items.get("eastmoney", [])
    if em:
        em_dedup = []
        seen = set()
        for item in sorted(em, key=lambda x: x.get("time", "")):
            key = item.get("title", "")[:60]
            if key not in seen:
                seen.add(key)
                em_dedup.append(item)
        lines.append(f"--- 实时快讯 ---")
        for item in em_dedup[:40]:
            lines.append(f"[{item['time']}] {item['title']}")

    return "\n".join(lines)


def _format_multi_day_news(snapshot, sector, date_str) -> str:
    """预格式化多日新闻，LLM无法跳过，直接注入prompt。"""
    from datetime import datetime, timedelta
    lines = ["【预格式化新闻（48小时覆盖，请全部列出）】"]

    # 新浪历史（交易日+前日）
    sina = snapshot.news_items.get("sina", [])
    if sina:
        lines.append(f"新浪历史({len(sina)}条):")
        for item in sina[:15]:
            lines.append(f"- [{item['time']}] {item['title']}")

    # 东方财富实时
    em = snapshot.news_items.get("eastmoney", [])
    if em:
        em_sorted = sorted(em, key=lambda x: x.get("time", ""), reverse=True)[:10]
        lines.append(f"东方财富实时({len(em_sorted)}条):")
        for item in em_sorted:
            lines.append(f"- [{item['time']}] {item['title']}")

    return "\n".join(lines)


# ── Agent ──

class MarketReviewAgent:
    """市场复盘智能体。"""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )
        self.model = "deepseek-chat"
        self._cache: dict = {}  # 行情数据缓存
        self._pending: dict = {}  # 待续写状态：{session_key: {system, user}}

    @property
    def cache_warm(self) -> bool:
        """检查今日数据是否已缓存。"""
        today = datetime.now().strftime("%Y%m%d")
        return f"snapshot_{today}" in self._cache

    async def process_message(
        self, message: str, stream: bool = False
    ) -> dict | AsyncGenerator:
        """处理用户消息。"""
        # "继续"：续写上次截断内容
        if message.strip() in ["继续", "继续输出", "接着来", "继续分析", "go on", "continue"]:
            if self._pending:
                key = list(self._pending.keys())[-1]
                state = self._pending.pop(key)
                return await self._continue_output(state, stream)
            return {"role": "assistant", "content": "没有待续写的内容。"}

        intent, sector = detect_intent(message)

        if intent == "general_chat":
            return await self._chat(message, stream)
        elif intent == "market_review":
            return await self._market_review(stream)
        elif intent == "news_only":
            return await self._news_only(sector, stream)
        elif intent == "stock_query":
            return await self._stock_query(message, stream)
        elif intent == "futures_query":
            return await self._futures_query(message, stream)
        elif intent == "fund_query":
            return await self._fund_query(message, stream)
        else:
            return await self._sector_deep_dive(sector, stream)

    async def _chat(self, message: str, stream: bool):
        """通用对话。"""
        system = get_system_prompt("general_chat")
        return await self._call_llm(system, message, stream)

    async def _market_review(self, stream: bool):
        """全市场复盘。"""
        # 1. 确定有效的交易日期
        today = datetime.now()
        trade_date = _get_latest_trade_date(today)

        date_str = trade_date.strftime("%Y%m%d")
        weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][trade_date.weekday()]
        date_display = trade_date.strftime("%Y年%m月%d日")

        is_today = trade_date.strftime("%Y%m%d") == today.strftime("%Y%m%d")
        date_note = "今日" if is_today else f"（今日为{today.strftime('%Y年%m月%d日')}，最新可用交易日数据为{date_display}）"

        # 2. 采集数据（带缓存：同一交易日只采集一次）
        cache_key = f"snapshot_{date_str}"
        if cache_key in self._cache:
            snapshot = self._cache[cache_key]
        else:
            snapshot = await collect_market_snapshot(date=date_str)
            self._cache = {cache_key: snapshot}  # 每天只保留最新
        market_data = format_market_data_for_prompt(snapshot)

        # 3. 构建 prompt（日期注入系统提示词）
        system = get_system_prompt("market_review").replace("[日期]", date_display)

        user_prompt = f"""交易日：{date_display} {weekday}{date_note}

{market_data}

生成A股市场复盘。不列出新闻（如需新闻请说\"今天市场新闻\"）。31行业全部列出。数据缺失标[UNSOURCED]。"""

        return await self._call_llm(system, user_prompt, stream)

    async def _sector_deep_dive(self, sector: str, stream: bool):
        """单板块深度聚焦。"""
        today = datetime.now()
        trade_date = _get_latest_trade_date(today)
        date_str = trade_date.strftime("%Y%m%d")
        weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][trade_date.weekday()]
        date_display = trade_date.strftime("%Y年%m月%d日")

        # 采集全市场数据 + 板块深度数据（带缓存）
        cache_key = f"snapshot_{date_str}_{sector}"
        if cache_key in self._cache:
            snapshot = self._cache[cache_key]
        else:
            snapshot = await collect_market_snapshot(date=date_str, sector_focus=sector)
            self._cache = {cache_key: snapshot}
        market_data = format_market_data_for_prompt(snapshot)

        system = get_system_prompt("sector_deep_dive", sector)
        system = system.replace("[日期]", date_display).replace("[板块名]", sector)

        user_prompt = f"""交易日：{date_display} {weekday} | 行业：{sector}

{market_data}

深度分析{sector}板块。不列出新闻。如需新闻请用户说\"{sector}板块新闻\"。"""

        return await self._call_llm(system, user_prompt, stream)

    async def _news_only(self, sector: str, stream: bool):
        """新闻专属模式：代码层按行业关键字精准过滤，48小时全覆盖。"""
        today = datetime.now()
        trade_date = _get_latest_trade_date(today)
        date_str = trade_date.strftime("%Y%m%d")
        date_display = trade_date.strftime("%Y年%m月%d日")
        weekday = ["周一","周二","周三","周四","周五","周六","周日"][trade_date.weekday()]

        # 新闻模式：Sina每天150条(3页×50)，3天覆盖 + EM补最新
        from agent.data_fetcher import fetch_sina_news as _sina, fetch_eastmoney_news as _em
        import asyncio
        loop = asyncio.get_event_loop()

        # 每日期3页、每页50条、重试2次 = 150条/天 × 3天 = 450条
        d0 = trade_date.strftime("%Y-%m-%d")
        d1 = (trade_date - timedelta(days=1)).strftime("%Y-%m-%d")
        d2 = (trade_date - timedelta(days=2)).strftime("%Y-%m-%d")

        # 新闻数据并行拉取
        from agent.data_fetcher import fetch_mcp_news as _mcp, fetch_sina_news as _s, fetch_eastmoney_news as _e
        search_kw = sector if sector else "A股"
        # 直接同步调用，不用线程池
        mcp_items = _mcp(search_kw, 60)
        em1 = _e(50)
        all_sina = []
        for date_str in [d0, d1, d2]:
            for page in [1, 2]:
                items = _s(50, date_str)
                if items:
                    all_sina.extend(items)

        # 申万31行业关键词（每个行业名+简称，用于新闻自动归类）
        SW_ALL_KEYWORDS = {
            "农林牧渔": ["农", "牧", "渔", "养殖", "种业", "粮食", "猪肉", "饲料", "转基因", "大豆", "玉米", "棉花"],
            "采掘": ["采掘", "矿业", "矿山"],
            "化工": ["化工", "化学", "化肥", "农药", "化纤", "万华", "MDI", "乙烯", "丙烯"],
            "钢铁": ["钢铁", "钢价", "宝钢", "螺纹钢", "热卷", "铁矿石"],
            "有色金属": ["有色", "铜", "铝", "黄金", "稀土", "锂矿", "镍", "锌", "钴", "紫金矿业", "赣锋", "天齐"],
            "电子": ["电子", "芯片", "半导体", "存储", "晶圆", "光刻", "海思", "算力", "GPU", "CPU", "HBM", "封装", "PCB", "英伟达", "NVIDIA", "AMD", "英特尔", "中芯国际", "韦尔"],
            "家用电器": ["家电", "空调", "冰箱", "洗衣机", "美的", "格力", "海尔", "扫地机"],
            "食品饮料": ["食品", "饮料", "白酒", "茅台", "五粮液", "乳业", "啤酒", "调味品", "零食", "预制菜", "餐饮", "猪肉", "糖"],
            "纺织服装": ["纺织", "服装", "鞋", "安踏", "李宁", "耐克", "代工"],
            "轻工制造": ["造纸", "家居", "包装", "印刷", "太阳纸业", "欧派", "顾家"],
            "医药生物": ["医药", "医疗", "药", "疫苗", "生物", "基因", "细胞", "病毒", "疫情", "流感", "创新药", "CRO", "医疗器械", "恒瑞", "迈瑞", "药明", "百济", "PD-1", "GLP-1", "减肥药", "医保", "集采", "FDA"],
            "公用事业": ["电力", "水务", "燃气", "碳排放", "绿电", "长江电力", "华能", "新能源发电"],
            "交通运输": ["航运", "物流", "快递", "港口", "铁路", "高速", "航空", "机场", "集装箱", "中远海控", "顺丰", "波罗的海"],
            "房地产": ["房地产", "地产", "楼市", "房价", "商品房", "土地出让", "房贷", "公积金", "限购", "万科", "保利", "碧桂园", "城中村"],
            "商业贸易": ["零售", "超市", "免税", "电商", "跨境电商", "百货", "中国中免", "王府井", "拼多多", "京东", "阿里"],
            "休闲服务": ["旅游", "酒店", "景区", "出境游", "锦江", "华住"],
            "建筑材料": ["水泥", "玻璃", "建材", "海螺水泥", "东方雨虹", "石膏板"],
            "建筑装饰": ["建筑", "基建", "中国建筑", "中国中铁", "城投", "专项债", "PPP", "一带一路"],
            "电气设备": ["新能源", "光伏", "锂电", "锂电池", "储能", "宁德时代", "比亚迪", "隆基", "通威", "阳光电源", "风电", "硅料", "硅片", "组件", "逆变器", "充电桩", "固态电池", "动力电池"],
            "国防军工": ["军工", "导弹", "战斗机", "航母", "航天", "军机", "航发", "中航", "兵器"],
            "计算机": ["软件", "AI", "人工智能", "大模型", "ChatGPT", "自动驾驶", "算法", "云计算", "信创", "IT", "数据", "科大讯飞", "商汤"],
            "传媒": ["游戏", "电影", "院线", "短剧", "直播", "广告", "出版", "媒体", "互联网", "视频", "影视", "票房", "抖音", "快手", "综艺"],
            "通信": ["通信", "5G", "6G", "光模块", "光纤", "中兴通讯", "运营商", "中国移动", "卫星通信", "光通信"],
            "银行": ["银行", "金融", "贷款", "存款", "利率", "息差", "工商银行", "招商银行", "建设银行", "农业银行", "中国银行", "交通银行", "邮储银行", "兴业银行", "浦发银行", "中信银行", "民生银行", "光大银行", "平安银行", "净息差", "降准", "降息", "MLF", "LPR", "信贷", "社融", "M2", "货币政策", "央行", "银保监", "金监", "商业银行", "城商行", "农商行"],
            "非银金融": ["券商", "保险", "证券", "中信证券", "华泰证券", "中国平安", "中国人寿", "投行", "IPO", "再融资"],
            "汽车": ["汽车", "车", "新能源车", "电动车", "整车", "乘用车", "商用车", "比亚迪", "特斯拉", "蔚来", "小鹏", "理想", "小米汽车", "华为汽车", "自动驾驶", "智能驾驶", "锂电", "充电桩", "车市", "销量", "出口", "SUV", "轿车", "卡车", "客车", "轮胎", "4S", "经销商", "上汽", "广汽", "吉利", "长城", "长安", "东风"],
            "机械设备": ["机械", "工程机械", "机器人", "三一重工", "挖掘机", "机床", "工业母机", "自动化", "人形机器人"],
            "煤炭": ["煤炭", "煤价", "煤", "矿", "中国神华", "陕西煤业", "中煤能源", "兖矿", "动力煤", "焦煤", "煤矿"],
            "石油石化": ["石油", "石化", "原油", "成品油", "中国石油", "中国石化", "中海油", "油价", "OPEC", "钻井", "天然气"],
            "环保": ["环保", "碳中和", "碳达峰", "污水处理", "垃圾焚烧"],
        }

        from collections import defaultdict
        by_date = defaultdict(list)
        label = f"{sector}板块" if sector else "全市场"

        all_items = mcp_items + all_sina + (em1 or [])
        for item in all_items:
            t = (item.get("time", "") or "")[:10]
            title = (item.get("title", "") or "").strip()
            if not t or not title or len(title) < 4:
                continue
            if sector:
                keywords = SW_ALL_KEYWORDS.get(sector, [sector])
                if not any(kw in title for kw in keywords):
                    continue
            by_date[t].append(title)

        total = sum(len(v) for v in by_date.values())
        if total == 0:
            news_text = f"{label}新闻汇总 — {date_display} {weekday}\n未找到与该行业相关的新闻。请尝试更宽泛的关键词，或查询\"全市场新闻\"。"
        # MCP独家新闻块
        mcp_block = ""
        if mcp_items:
            mcp_block = f"\n\n新浪智研快讯（关键字精准匹配，共{len(mcp_items)}条）:\n"
            for it in mcp_items[:30]:
                t = (it.get("time","") or "")[:16]
                mcp_block += f"[{t}] {it.get('title','')}\n"

        if total == 0 and not mcp_items:
            news_text = f"{label}新闻汇总 — {date_display} {weekday}\n未找到相关新闻。"
        else:
            news_text = f"{label}新闻汇总 — {date_display} {weekday}（48小时覆盖，共{total}+{len(mcp_items)}条）\n"
        for d in sorted(by_date.keys(), reverse=True):
            items = sorted(by_date[d])
            news_text += f"\n--- {d}（{len(items)}条）---\n"
            for title in items:
                news_text += f"[{d}] {title}\n"
        news_text += mcp_block

        system = "你是财经新闻编辑。将以下新闻汇总原样输出。不删减、不分析、不改格式。"
        user_prompt = f"{news_text}\n\n以上{label}48小时新闻汇总。原样输出。"

        result = await self._call_llm(system, user_prompt, stream)
        if not stream and isinstance(result, dict):
            result["content"] = news_text
        return result

    async def _stock_query(self, message: str, stream: bool):
        """个股查询。"""
        from agent.data_fetcher import fetch_stock_quote, fetch_stock_kline, fetch_stock_news, search_stock
        # 先搜索股票
        results = search_stock(message[:20])
        if not results:
            return {"role": "assistant", "content": "未找到该股票，请尝试输入完整代码如 600519.SH 或公司名如 贵州茅台"}
        s = results[0]
        market = "cn" if s.get("market","") in ["11","cn"] else "us"
        quote = fetch_stock_quote(market, s["code"])
        kline = fetch_stock_kline(market, s["code"], 5)
        news = fetch_stock_news(s["code"], market, 5)
        info = f"""{s['name']}({s['code']})
实时行情: 价格{quote.get('price','?')} 涨跌{quote.get('pct','?')}% 成交量{quote.get('vol','?')}
开盘{quote.get('open','?')} 最高{quote.get('high','?')} 最低{quote.get('low','?')}
近5日K线: {', '.join(f"{k['date'][-5:]}:{k['close']}({k['pct']}%)" for k in kline)}
相关新闻({len(news)}条): {'; '.join(f"{n['title'][:40]} {n['time']}" for n in news[:3])}
"""
        system = "你是股票分析师。根据数据简要分析该股票。禁止markdown格式。"
        result = await self._call_llm(system, f"股票数据:\n{info}\n\n请简要分析这只股票。", stream)
        if not stream and isinstance(result, dict):
            result["content"] = info + "\n" + result["content"]
        return result

    async def _futures_query(self, message: str, stream: bool):
        """期货查询。"""
        from agent.data_fetcher import fetch_futures_quote
        # 简单关键词映射
        kw_map = {"黄金": ("shfe", "AU0"), "原油": ("dce", "SC0"), "铜": ("shfe", "CU0"),
                  "螺纹钢": ("shfe", "RB0"), "铁矿石": ("dce", "I0"), "白银": ("shfe", "AG0"), "焦煤": ("dce", "JM0")}
        matched = None
        for kw, (mkt, sym) in kw_map.items():
            if kw in message:
                matched = (kw, mkt, sym)
                break
        if not matched:
            return {"role": "assistant", "content": "请指定期货品种：黄金/原油/铜/螺纹钢/铁矿石/白银/焦煤"}
        kw, mkt, sym = matched
        q = fetch_futures_quote(mkt, sym)
        info = f"{kw}期货: 价格{q.get('price','?')} 涨跌{q.get('pct','?')}% 成交量{q.get('volume','?')}"
        return {"role": "assistant", "content": info}

    async def _continue_output(self, state: dict, stream: bool):
        """续写上次截断的内容。"""
        system = state["system"]
        user_prompt = f"上一段末尾内容:\n{state['prev']}\n\n请从上一段末尾继续输出，不要重复已输出的内容。数据如下:\n{state['data']}"
        result = await self._call_llm(system, user_prompt, stream)
        return result

    async def _fund_query(self, message: str, stream: bool):
        """基金/期货/股票/汇率等通用MCP查询——LLM自动选择工具。"""
        from agent.data_fetcher import get_mcp_tools, _mcp_call
        tools = get_mcp_tools()
        if not tools:
            return {"role": "assistant", "content": "MCP服务暂不可用"}

        tool_desc = "\n".join(f"- {t['name']}: {t['desc']} (参数:{','.join(t['params'])})" for t in tools[:60])
        system = f"""你是金融数据助手。根据用户问题，从以下MCP工具中选择最合适的并生成调用参数（JSON格式）。
可用工具:
{tool_desc}

输出格式: {{"tool":"工具名","args":{{"参数":"值"}}}}。只输出JSON，不要解释。"""
        result = await self._call_llm(system, f"用户问题: {message}\n请选择合适的工具和参数。", stream)
        if stream or not isinstance(result, dict):
            return result
        try:
            import json as _json
            content = result["content"].strip()
            if "```" in content:
                content = content.split("```")[1].split("```")[0].replace("json","").strip()
            call = _json.loads(content)
            data = _mcp_call(call["tool"], call["args"])
            return {"role": "assistant", "content": f"查询结果({call['tool']}):\n{_json.dumps(data,ensure_ascii=False,indent=2)[:2000]}"}
        except Exception:
            return {"role": "assistant", "content": f"无法解析工具调用: {result['content'][:300]}"}

    async def _call_llm(
        self, system_prompt: str, user_message: str, stream: bool = False
    ):
        """调用 DeepSeek API。"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        if stream:
            return self._stream_response(messages)
        else:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=8192,
            )
            raw = completion.choices[0].message.content
            disclaimer = "\n\n风险提示：以上内容仅为行情数据复盘，不构成任何投资建议。本智能体由AI驱动，市场数据来源于公开信息，分析结论仅供参考。智能体开发同学与以上内容无任何责任关系。市场有风险，投资需谨慎。"
            clean = _clean_markdown(raw)
            # 检测是否被截断（内容>5000字且末尾无句号）
            finish = completion.choices[0].finish_reason
            if finish == "length" or (len(clean) > 5000 and not clean.rstrip().endswith(("。","）",")","\"","\n"))):
                hint = '\n\n受模型token限制，以上为部分内容。回复"继续"查看剩余部分'
                self._pending[str(len(self._pending))] = {"system": system_prompt, "data": user_message[:2000], "prev": clean[-1000:]}
                return {"role": "assistant", "content": clean + disclaimer + hint}
            return {"role": "assistant", "content": clean + disclaimer}

    async def _stream_response(self, messages: list) -> AsyncGenerator:
        """流式响应生成器。注意：流式不清理markdown，但非流式会清理。"""
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=8192,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ── 全局单例 ──

_agent: Optional[MarketReviewAgent] = None


def get_agent() -> MarketReviewAgent:
    global _agent
    if _agent is None:
        _agent = MarketReviewAgent()
    return _agent
