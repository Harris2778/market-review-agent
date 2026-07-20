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
    "每日复盘", "今日复盘", "今天市场", "今日大盘", "市场回顾", "复盘",
    "market review", "今天行情", "今天A股", "今天股市", "大盘分析",
    "收盘复盘", "今天怎么样", "市场日报", "昨天A股", "昨天市场",
    "回顾", "行情分析", "市场表现", "A股复盘", "股市复盘",
    "今天盘面", "昨天盘面", "复盘一下", "市场分析", "大盘走势",
]

MARKET_REVIEW_PATTERNS = [
    r"^(今天|今日|昨天|昨日).*(市场|大盘|行情|复盘|A股|股市|盘面)",
    r".*(复盘|市场回顾|市场日报|市场表现|行情分析|大盘走势|大盘分析).*",
    r".*(回顾|分析|看看|看下).*(A股|市场|大盘|行情|股市|盘面).*",
    r".*(最近|近期).*(市场|大盘|行情|A股|股市).*(怎么样|如何).*",
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

    # 先检查是不是全市场复盘
    for kw in MARKET_REVIEW_KEYWORDS:
        if kw in msg:
            # 但如果同时有行业关键词，则可能是聚焦模式
            for sector_keyword in SECTOR_KEYWORDS:
                if sector_keyword in msg:
                    # 尝试提取行业名
                    sector = _extract_sector(msg)
                    if sector:
                        return ("sector_deep_dive", sector)
            return ("market_review", None)

    for pattern in MARKET_REVIEW_PATTERNS:
        if re.search(pattern, msg):
            for sector_keyword in SECTOR_KEYWORDS:
                if sector_keyword in msg:
                    sector = _extract_sector(msg)
                    if sector:
                        return ("sector_deep_dive", sector)
            return ("market_review", None)

    # 检查是不是单板块聚焦
    for pattern in SECTOR_FOCUS_PATTERNS:
        if re.search(pattern, msg):
            sector = _extract_sector(msg)
            if sector:
                return ("sector_deep_dive", sector)

    # 直接提行业名 + 问怎么样/复盘
    for kw, sw_name in SECTOR_NAME_MAP.items():
        if kw in msg and any(w in msg for w in ["怎么样", "复盘", "分析", "走势", "行情", "表现"]):
            return ("sector_deep_dive", sw_name)

    return ("general_chat", None)


def _extract_sector(msg: str) -> Optional[str]:
    """从消息中提取行业名并映射到申万一级。"""
    # 按长度降序匹配（先匹配长词如"新能源车"，再短词如"汽车"）
    sorted_kws = sorted(SECTOR_NAME_MAP.keys(), key=len, reverse=True)
    for kw in sorted_kws:
        if kw in msg:
            return SECTOR_NAME_MAP[kw]
    return None


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

    async def process_message(
        self, message: str, stream: bool = False
    ) -> dict | AsyncGenerator:
        """
        处理用户消息，返回 OpenAI 兼容格式的响应。
        """
        intent, sector = detect_intent(message)

        if intent == "general_chat":
            return await self._chat(message, stream)
        elif intent == "market_review":
            return await self._market_review(stream)
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

        # 3. 构建 prompt
        system = get_system_prompt("market_review")

        user_prompt = f"""今日日期：{today.strftime('%Y年%m月%d日')} {['周一','周二','周三','周四','周五','周六','周日'][today.weekday()]}
数据日期：{date_display} {weekday}（最近可用交易日）{date_note}

{market_data}

请根据以上实时市场数据，生成今日A股市场每日复盘报告。严格按照系统提示词中指定的格式输出。

注意事项：
- 如果某项数据标记为"不可用"，在报告中标注[UNSOURCED]而不是编造
- 今天是{'交易日（正常复盘）' if weekday not in ['周六', '周日'] else '周末，请告知用户今日休市'}
- 如果今天是周一，需要说明使用的是上周五的数据
- 新闻按S/A/B/C四级分类，级别标注要准确
- 31个行业全部列出，不要只列前几名"""

        return await self._call_llm(system, user_prompt, stream)

    async def _sector_deep_dive(self, sector: str, stream: bool):
        """单板块深度聚焦。"""
        today = datetime.now()
        trade_date = _get_latest_trade_date(today)
        date_str = trade_date.strftime("%Y%m%d")
        weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][trade_date.weekday()]
        date_display = trade_date.strftime("%Y年%m月%d日")

        # 采集全市场数据 + 板块深度数据（带缓存）
        cache_key = f"snapshot_{date_str}"
        if cache_key in self._cache:
            snapshot = self._cache[cache_key]
        else:
            snapshot = await collect_market_snapshot(date=date_str, sector_focus=sector)
            self._cache = {cache_key: snapshot}
        market_data = format_market_data_for_prompt(snapshot)

        # 额外板块数据
        sector_extra = ""
        if hasattr(snapshot, "_sector_detail") and snapshot._sector_detail:
            sd = snapshot._sector_detail
            sector_extra = f"""
## 板块深度数据：{sector}

### 成分股表现
- 板块内上涨: {sd.get('up_count', 'N/A')}家
- 板块内下跌: {sd.get('down_count', 'N/A')}家
- 板块内平盘: {sd.get('flat_count', 'N/A')}家

### 领涨个股
"""
            for s in sd.get("top_gainers", []):
                sector_extra += f"- {s['name']}: {s['pct_chg']:+.2f}%\n"

            sector_extra += "\n### 领跌个股\n"
            for s in sd.get("top_losers", []):
                sector_extra += f"- {s['name']}: {s['pct_chg']:+.2f}%\n"

        system = get_system_prompt("sector_deep_dive", sector)

        user_prompt = f"""数据日期：{date_display} {weekday}（最近可用交易日）。
用户要求聚焦分析：**{sector}**板块。

{market_data}

{sector_extra}

请对{sector}板块进行7维度深度分析。先给出全市场概览（1-2句话+行业热力图），再按A-G七个维度逐一展开。"""

        return await self._call_llm(system, user_prompt, stream)

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
                temperature=0.3,
                max_tokens=4096,
            )
            return {
                "role": "assistant",
                "content": completion.choices[0].message.content,
            }

    async def _stream_response(self, messages: list) -> AsyncGenerator:
        """流式响应生成器。"""
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
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
