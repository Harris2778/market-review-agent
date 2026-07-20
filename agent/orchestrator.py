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

    # 直接提行业名 → 板块聚焦（不强制要求问句）
    for kw, sw_name in SECTOR_NAME_MAP.items():
        if kw in msg:
            return ("sector_deep_dive", sw_name)

    # 提板块/行业 + 任何疑问词 → 板块聚焦
    for pattern in [r"(板块|行业|赛道).*(怎么样|如何|分析|复盘|表现|走势|情况|新闻|回顾)", r"(怎么看|分析一下|回顾一下).*(板块|行业|市场)"]:
        if re.search(pattern, msg):
            sector = _extract_sector(msg)
            if sector:
                return ("sector_deep_dive", sector)

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
    """后处理：强制清除 LLM 输出的所有 markdown 格式标记。"""
    import re

    # 1. 管道表格 → 缩进纯文本
    lines = text.split("\n")
    cleaned = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            if "---" in stripped:
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

    # 3. # → 空行替代
    text = re.sub(r"^###?\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n###?\s+", "\n", text)

    # 4. * 列表 → - 列表
    text = re.sub(r"^\* ", "- ", text, flags=re.MULTILINE)

    return text


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

    @property
    def cache_warm(self) -> bool:
        """检查今日数据是否已缓存。"""
        today = datetime.now().strftime("%Y%m%d")
        return f"snapshot_{today}" in self._cache

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

        # 3. 构建 prompt（日期注入系统提示词）
        system = get_system_prompt("market_review").replace("[日期]", date_display)

        user_prompt = f"""交易日：{date_display} {weekday}{date_note}

{market_data}

生成A股市场复盘。所有新闻全部列出。31行业全部列出。数据缺失标[UNSOURCED]。"""

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
        # 注入日期和板块名
        system = system.replace("[日期]", date_display).replace("[板块名]", sector)

        user_prompt = f"""交易日：{date_display} {weekday} | 行业：{sector}

{market_data}

深度分析{sector}板块。按系统提示词框架展开。所有新闻必须列出。"""

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
                temperature=0.2,
                max_tokens=8192,
            )
            raw = completion.choices[0].message.content
            disclaimer = "\n\n风险提示：以上内容仅为行情数据复盘，不构成任何投资建议。本智能体由AI驱动，市场数据来源于公开信息，分析结论仅供参考。智能体开发同学与以上内容无任何责任关系。市场有风险，投资需谨慎。"
            return {
                "role": "assistant",
                "content": _clean_markdown(raw) + disclaimer,
            }

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
