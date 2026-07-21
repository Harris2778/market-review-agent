"""
Agent 编排器：意图识别 → 数据采集 → LLM生成 → 响应格式化。
"""

import re
import json
import logging
import os
import time
import asyncio
from datetime import datetime, timedelta
from typing import AsyncGenerator, Optional

from openai import AsyncOpenAI
from .data_fetcher import collect_market_snapshot, format_market_data_for_prompt
from .system_prompts import get_system_prompt

logger = logging.getLogger(__name__)

# 第二波：Agent 工具注册表（并行开发的 agent/tools.py 未落地时降级为 None，走普通对话）
try:
    from .tools import TOOL_REGISTRY, execute_tool
except ImportError as e:
    logger.warning("agent/tools.py 未就绪，Agent 工具调用路径将降级为普通对话: %s", e)
    TOOL_REGISTRY = None
    execute_tool = None

# 第二波：Agent 循环提示词与自我审查提示词（未就绪时对应能力降级，不影响主流程）
try:
    from .system_prompts import AGENT_QUERY_PROMPT, CRITIQUE_PROMPT
except ImportError as e:
    logger.warning("AGENT_QUERY_PROMPT/CRITIQUE_PROMPT 未就绪，Agent 循环与自我审查将降级: %s", e)
    AGENT_QUERY_PROMPT = None
    CRITIQUE_PROMPT = None

# 第三波：输出后数字溯源校验层（agent/validators.py 未就绪时校验能力整体降级，不影响主流程）
try:
    from . import validators
except ImportError as e:
    logger.warning("agent/validators.py 未就绪，数字溯源校验将整体降级跳过: %s", e)
    validators = None

# 第四波：分析存档层（自我问责系统，agent/archive.py 未就绪时存档能力整体降级，不影响主流程）
try:
    from . import archive
except ImportError as e:
    logger.warning("agent/archive.py 未就绪，分析存档将整体降级跳过: %s", e)
    archive = None


def _log_stream_violations(label: str, violations: list) -> None:
    """流式路径 log-only：记录疑似无出处数字的数量与前几条原文，绝不抛出、不拦截。"""
    try:
        preview = "；".join(
            str(v.get("raw", "?")) for v in violations[:5] if isinstance(v, dict)
        )
        logger.warning(
            "流式%s提示文本含 %d 个疑似无出处数字（log-only，不拦截）: %s",
            label, len(violations), preview,
        )
    except Exception as e:
        logger.warning("流式数字溯源校验日志记录异常（忽略）: %s", e, exc_info=True)


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
        except Exception as e:
            logger.warning("Tushare 交易日历获取失败，使用周末规则兜底: %s", e, exc_info=True)

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
    # 简单数据查询 → MCP而非完整复盘
    simple_data_patterns = [
        r"(多少|几家|哪些|排名|前\d|列表|列出|查询|查一下|帮我查)",
        r"^(今天|今日|昨天|当前).{0,10}(涨跌|上涨|下跌|涨停|跌停|热搜|北向)",
    ]
    is_simple = any(re.search(p, msg) for p in simple_data_patterns)
    is_not_review = not any(kw in msg for kw in ["复盘","分析","总结","回顾","报告","日报","怎么样","如何","走势","行情"])

    for kw in MARKET_REVIEW_KEYWORDS:
        if kw in msg:
            if is_simple and is_not_review:
                return ("mcp_query", msg)  # 简单查询走MCP
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
        # 解读类新闻请求（如「解读一下今天的新闻」「这些新闻怎么看」）
        r"(解读|分析|影响|怎么看|意味着什么|说明什么).{0,10}(新闻|快讯|资讯)",
        r"(新闻|快讯|资讯).{0,10}(解读|怎么看|意味着什么|说明什么)",
    ]
    for pattern in news_patterns:
        if re.search(pattern, msg):
            sector = _extract_sector(msg)
            return ("news_only", sector)  # sector may be None for full market news

    # 纯数据查询（热搜/榜单/搜索等）→ MCP
    pure_data_kw = ["热搜", "榜单", "搜索", "排名前", "前10", "前5", "列表", "列出", "查询", "涨跌分布", "涨了多少", "跌了多少", "哪些股票涨停", "哪些股票跌停", "北向资金多少", "汇率", "人民币", "美元兑"]
    if any(kw in msg for kw in pure_data_kw):
        return ("mcp_query", msg)

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


# ── 多轮对话：历史截断 + 上下文感知意图继承 ──

MAX_HISTORY_MESSAGES = 20  # 最多保留最近 10 轮（user+assistant 各一条为一轮），防 token 膨胀

# 追问/对比类表达（保守清单：只在上文是数据型意图时用于继承判定）
_FOLLOWUP_PATTERNS = [
    # 对比：跟昨天比 / 和上周相比 / 对比之前
    r"(跟|和|与|对比|相比|比).{0,6}(昨天|昨日|前天|上周|之前|上次|早盘|上午)",
    # 归因：为什么跌 / 怎么回事 / 怎么涨了
    r"(为什么|为啥|怎么回事|怎么).{0,8}(跌|涨|砸|拉|崩|跳水|异动|杀跌|走强|走弱)",
    # 走势预判：还会跌吗 / 会不会反弹 / 能不能追
    r"(还会|还能|会不会|要不要|能不能|可否).{0,6}(跌|涨|反弹|继续|新高|新低|抄底|追|加仓|减仓)",
    # 维度追问：资金呢 / 主力怎么样 / 北向情况 / 估值呢 / 业绩如何
    r"(资金|主力|北向|散户|量能|成交|换手|龙虎榜)(呢|怎么样|如何|情况|流向|数据)?$",
    r"(估值|景气|业绩|新闻|催化|风险|逻辑|基本面|技术面)(呢|怎么样|如何|情况)?$",
    # 展开追问：继续 / 再说说 / 详细说 / 那怎么办
    r"^(继续|再说说|详细说|展开说|具体说|深入说)",
    r"^(那|那么|所以)(呢|怎么样|如何|怎么办|意味着)",
]

# 意图类型 → 中文描述（用于追问提示行）
_INTENT_LABELS = {
    "market_review": "全市场复盘",
    "sector_deep_dive": "板块深挖",
}


def _trim_history(history: Optional[list]) -> list:
    """
    清洗并截断对话历史：只保留 user/assistant 且 content 为非空字符串的条目，
    最多保留最近 10 轮（20 条）。
    """
    if not history:
        return []
    cleaned = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content})
    return cleaned[-MAX_HISTORY_MESSAGES:]


def _resolve_contextual_intent(message: str, history: list) -> tuple[str, Optional[str], Optional[str]]:
    """
    上下文感知意图识别（detect_intent 的保守包装，detect_intent 本身不变）。

    返回 (intent, sector, context_note_label_or_None)。
    仅当以下条件全部满足时才继承上文意图，否则原样返回 detect_intent 的结果：
      1. history 非空，且能找到最近一轮 user 消息；
      2. 最近一轮 user 消息的意图（复用 detect_intent）是 sector_deep_dive 或 market_review；
      3a. 上文是板块深挖 且 当前消息含新行业名（裸行业名或 general_chat）→ 继承深挖、切换行业；
      3b. 上文是板块深挖 且 当前消息识别为 general_chat 且命中追问/对比模式 → 继承原板块；
      3c. 上文是全市场复盘 且 当前消息识别为 general_chat 且命中追问/对比模式 → 继承复盘。
    拿不准就落回 detect_intent 原始结果——宁可不继承，也不能错继承。
    """
    intent, sector = detect_intent(message)
    if not history:
        return intent, sector, None

    prev_user_msg = None
    for msg in reversed(history):
        if msg.get("role") == "user":
            prev_user_msg = msg.get("content")
            break
    if not prev_user_msg:
        return intent, sector, None

    prev_intent, prev_sector = detect_intent(prev_user_msg)
    if prev_intent not in ("market_review", "sector_deep_dive"):
        return intent, sector, None

    msg = message.strip()
    new_sector = _extract_sector(msg)
    # 裸行业名：消息很短且整体就是在说一个行业（如"那半导体呢"、"电子"）
    bare_sector = bool(new_sector) and len(msg) <= 12
    # 追问：当前消息识别为 general_chat（规则意图识别没接住）且命中追问模式
    is_followup = intent == "general_chat" and any(
        re.search(p, msg) for p in _FOLLOWUP_PATTERNS
    )

    if prev_intent == "sector_deep_dive":
        label = f"板块深挖（{prev_sector}板块）"
        # 切换到新行业：上文在深挖某板块，本条直接点了另一个行业
        if new_sector and (intent == "general_chat" or bare_sector):
            return "sector_deep_dive", new_sector, label
        # 同板块追问：继承原板块
        if is_followup:
            return "sector_deep_dive", prev_sector, label
    elif prev_intent == "market_review" and is_followup:
        return "market_review", None, _INTENT_LABELS["market_review"]

    return intent, sector, None


# ── 复杂分析路由（general_chat → Agent 工具循环） ──

# 跨实体比较/归因类表达：命中且带市场语境时，认为问题超出闲聊范畴
_AGENT_QUERY_PATTERNS = [
    r"(比较|对比|对照)",
    r"哪个更",
    r"谁更",
    r"和.{1,12}哪个",
    r"值不值得",
    r"怎么看.{1,12}和",
]

# 市场语境词：比较模式命中时要求至少出现一个，防止『我比较喜欢吃辣』误入 Agent 循环
_MARKET_CONTEXT_KEYWORDS = [
    "股", "板块", "行业", "基金", "估值", "大盘", "A股", "市场",
    "行情", "指数", "ETF", "期货", "涨", "跌", "资金", "业绩",
]

# 常见个股实体（用于多实体计数，与 _stock_query 的 NAME_MAP 口径一致）
_STOCK_ENTITY_KEYWORDS = [
    "茅台", "五粮液", "宁德", "比亚迪", "中芯", "招商银行",
    "平安", "苹果", "特斯拉", "英伟达", "腾讯", "阿里",
]


def _count_entities(msg: str) -> int:
    """统计消息中提到的不同行业/股票实体数量。"""
    entities = set()
    for kw, sw_name in SECTOR_NAME_MAP.items():
        if kw in msg:
            entities.add(f"sector:{sw_name}")
    for kw in _STOCK_ENTITY_KEYWORDS:
        if kw in msg:
            entities.add(f"stock:{kw}")
    return len(entities)


def _should_use_agent_query(message: str) -> bool:
    """
    保守判定：general_chat 消息是否属于复杂分析（应走 Agent 工具循环）。
    放行条件（满足其一）：
      1. 命中跨实体分析模式，且消息带市场语境（实体或市场词）；
      2. 同时提到 2 个以上行业/股票实体。
    拿不准（短消息、纯闲聊如『你好』）一律回 False 走 _chat。
    """
    msg = message.strip()
    if len(msg) < 6:  # 过短消息不可能是复杂分析
        return False
    entity_count = _count_entities(msg)
    if entity_count >= 2:
        return True
    if any(re.search(p, msg) for p in _AGENT_QUERY_PATTERNS):
        has_market_context = entity_count >= 1 or any(kw in msg for kw in _MARKET_CONTEXT_KEYWORDS)
        if has_market_context:
            return True
    return False


def _assistant_tool_message(message) -> dict:
    """把 SDK 返回的 assistant 消息（含 tool_calls）转成可回传给 API 的 dict。"""
    tool_calls = []
    for tc in (getattr(message, "tool_calls", None) or []):
        try:
            tool_calls.append(tc.model_dump())
        except AttributeError:  # 兼容无 model_dump 的轻量 mock 对象
            tool_calls.append({
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            })
    return {"role": "assistant", "content": message.content, "tool_calls": tool_calls}


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
                lines.append(f"[{_fmt_news_time(d, fallback=date_display)}] {title}")
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
            lines.append(f"[{_fmt_news_time(item.get('time', ''), fallback=date_display)}] {item['title']}")

    return "\n".join(lines)


# ── 板块深挖附加数据（估值 / 资金流向）格式化 ──

def _format_flow_items(items, limit: int = 5) -> str:
    """防御式格式化资金流入/流出居前个股列表，兼容 dict 或字符串条目。"""
    out = []
    for it in list(items)[:limit]:
        if isinstance(it, dict):
            name = it.get("name") or it.get("code") or "?"
            amt = it.get("amount")
            if amt is None:
                amt = it.get("net")
            if isinstance(amt, (int, float)):
                out.append(f"{name} {amt:+.2f}亿")
            else:
                out.append(str(name))
        else:
            out.append(str(it))
    return "、".join(out)


def _format_sector_valuation_block(valuation: Optional[dict]) -> str:
    """板块估值水位数据块。字段为 None 时跳过；整体缺失时标注数据未获取。"""
    header = "【二、板块估值水位】（Tushare 成分股市值加权，分位为历史百分位）"
    missing = "板块估值数据未获取，分析时请标注『估值数据暂缺』，禁止编造。"
    if not isinstance(valuation, dict):
        return f"{header}\n{missing}"
    parts = []
    pe = valuation.get("pe")
    if isinstance(pe, (int, float)):
        parts.append(f"PE(TTM) {pe:.1f}倍")
    pb = valuation.get("pb")
    if isinstance(pb, (int, float)):
        parts.append(f"PB {pb:.2f}倍")
    pe_pct = valuation.get("pe_percentile")
    if isinstance(pe_pct, (int, float)):
        parts.append(f"PE历史分位 {pe_pct:.1f}%")
    pb_pct = valuation.get("pb_percentile")
    if isinstance(pb_pct, (int, float)):
        parts.append(f"PB历史分位 {pb_pct:.1f}%")
    lines = [header, "  ".join(parts) if parts else missing]
    sample = valuation.get("sample_count")
    if isinstance(sample, (int, float)) and sample:
        lines.append(f"样本：{int(sample)}只成分股")
    note = valuation.get("note")
    if note:
        lines.append(f"备注：{note}")
    return "\n".join(lines)


def _format_sector_moneyflow_block(moneyflow: Optional[dict]) -> str:
    """板块资金博弈数据块。金额单位：亿元。字段为 None 时跳过。"""
    header = "【三、板块资金博弈】（Tushare 资金流向，单位：亿元）"
    missing = "板块资金流向数据未获取，分析时请标注『资金数据暂缺』，禁止编造。"
    if not isinstance(moneyflow, dict):
        return f"{header}\n{missing}"
    lines = [header]
    parts = []
    main = moneyflow.get("main_net")
    if isinstance(main, (int, float)):
        direction = "净流入" if main >= 0 else "净流出"
        parts.append(f"主力资金当日{direction} {abs(main):.2f}亿元")
    retail = moneyflow.get("retail_net")
    if isinstance(retail, (int, float)):
        direction = "净流入" if retail >= 0 else "净流出"
        parts.append(f"散户资金当日{direction} {abs(retail):.2f}亿元")
    lines.append("；".join(parts) if parts else "主力/散户资金数据未获取。")
    for key, label in (("top_inflow", "资金流入居前"), ("top_outflow", "资金流出居前")):
        items = moneyflow.get(key)
        if items:
            lines.append(f"{label}：{_format_flow_items(items)}")
    count = moneyflow.get("stock_count")
    if isinstance(count, (int, float)) and count:
        lines.append(f"统计样本：{int(count)}只成分股")
    note = moneyflow.get("note")
    if note:
        lines.append(f"备注：{note}")
    return "\n".join(lines)


def _format_earnings_items(items) -> str:
    """改善/恶化居前个股列表，宽容处理 dict/str 两种条目形态。"""
    out = []
    for it in items[:5]:
        if isinstance(it, dict):
            name = it.get("name") or it.get("ts_code") or "?"
            change = it.get("p_change")
            if change is None:
                change = it.get("change")
            if isinstance(change, (int, float)):
                out.append(f"{name} {change:+.1f}%")
            else:
                out.append(str(name))
        else:
            out.append(str(it))
    return "、".join(out)


def _format_sector_earnings_block(earnings: Optional[dict]) -> str:
    """板块景气度（业绩预告/快报）数据块。字段为 None 时跳过；整体缺失时标注数据未获取。"""
    header = "【四、板块景气度（业绩预告）】（Tushare 业绩预告/快报，percent 为百分数数值）"
    missing = "板块景气度数据未获取，分析时请标注『景气度数据暂缺』，禁止编造。"
    if not isinstance(earnings, dict):
        return f"{header}\n{missing}"
    lines = [header]
    period = earnings.get("period")
    if period:
        lines.append(f"披露期：{period}")
    parts = []
    total = earnings.get("total_forecast")
    if isinstance(total, (int, float)):
        parts.append(f"披露业绩预告 {int(total)} 家")
    pos = earnings.get("positive_count")
    if isinstance(pos, (int, float)):
        parts.append(f"预喜 {int(pos)} 家")
    neg = earnings.get("negative_count")
    if isinstance(neg, (int, float)):
        parts.append(f"预忧 {int(neg)} 家")
    ratio = earnings.get("positive_ratio")
    if isinstance(ratio, (int, float)):
        parts.append(f"预喜比例 {ratio:.1f}%")
    if parts:
        lines.append("；".join(parts))
    median = earnings.get("median_change")
    if isinstance(median, (int, float)):
        lines.append(f"净利润变动幅度中位数 {median:+.1f}%")
    express = earnings.get("express_count")
    if isinstance(express, (int, float)) and express:
        lines.append(f"另有业绩快报 {int(express)} 家")
    for key, label in (("top_improvers", "改善居前"), ("top_decliners", "恶化居前")):
        items = earnings.get(key)
        if items:
            lines.append(f"{label}：{_format_earnings_items(items)}")
    note = earnings.get("note")
    if note:
        lines.append(f"备注：{note}")
    if len(lines) == 1:
        lines.append(missing)
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
            lines.append(f"- [{_fmt_news_time(item.get('time', ''), fallback=date_str)}] {item['title']}")

    # 东方财富实时
    em = snapshot.news_items.get("eastmoney", [])
    if em:
        em_sorted = sorted(em, key=lambda x: x.get("time", ""), reverse=True)[:10]
        lines.append(f"东方财富实时({len(em_sorted)}条):")
        for item in em_sorted:
            lines.append(f"- [{_fmt_news_time(item.get('time', ''), fallback=date_str)}] {item['title']}")

    return "\n".join(lines)


# ── 新闻模式：重要性评分词表、触发词与展示辅助 ──

# 消息含这些词时，新闻模式走 LLM 解读而非原文透传
_NEWS_ANALYSIS_TRIGGERS = ("解读", "分析", "影响", "怎么看", "意味着什么", "说明什么")

# 新闻源内部名 → 展示名
_NEWS_SOURCE_NAMES = {
    "sina": "新浪", "eastmoney": "东方财富", "mcp": "新浪智研",
    "cls": "财联社", "tushare": "Tushare",
}

# 重要性评分词表（词组, 分值）：超展示上限时按评分截断
_NEWS_SCORE_WORDS = (
    (("预增", "扭亏", "净利", "营收", "中标", "签约"), 3),            # 业绩词
    (("政策", "国务院", "央行", "证监会", "工信部", "规划", "补贴"), 3),  # 政策词
    (("涨停", "跌停", "大涨", "大跌", "创新高", "创新低"), 2),          # 异动词
    (("增持", "回购", "减持", "并购", "重组", "上市"), 2),              # 公司行动词
)


def _news_importance(title: str) -> int:
    """新闻重要性评分：命中一组词加对应分值，用于超限时截断排序。"""
    score = 0
    for words, pts in _NEWS_SCORE_WORDS:
        if any(w in title for w in words):
            score += pts
    return score


def _fmt_news_time(t: str, fallback: str = "") -> str:
    """新闻时间统一显示为 MM-DD HH:mm；格式异常时原样截断返回。

    t 为空时返回 fallback（通常为所在日期分组的日期），避免出现空括号 []。
    """
    t = (t or "").strip()
    if not t:
        return fallback
    if len(t) >= 16 and t[4] == "-" and t[7] == "-":
        return f"{t[5:10]} {t[11:16]}"
    return t[:16]


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
        """检查最新交易日数据是否已缓存（与实际缓存使用的交易日 key 保持一致）。"""
        trade_date = _get_latest_trade_date(datetime.now())
        return f"snapshot_{trade_date.strftime('%Y%m%d')}" in self._cache

    # ── 第四波：自我问责存档接入（fail-safe，绝不影响主流程）──

    def _archive_safe(self, mode: str, sector, content: str, context: str, trade_date: str) -> None:
        """非流式路径存档：最终 content 产出后调用。任何异常只记 log。"""
        if archive is None:
            return
        try:
            archive.save_analysis(
                mode=mode, sector=sector, content=content,
                context=context, trade_date=trade_date,
            )
        except Exception as e:
            logger.warning("分析存档异常（忽略，不影响主流程）: %s", e, exc_info=True)

    def _stream_archive_callback(self, mode: str, sector, context: str, trade_date: str):
        """流式路径存档回调：流结束后以最终全文存档；archive 未就绪时返回 None。

        回调在 _stream_response 生成器被完整消费后触发；客户端中途断开导致
        流未完整消费时该次不存档（best-effort，见 agent/archive.py docstring）。
        """
        if archive is None:
            return None

        def _callback(final_text: str) -> None:
            self._archive_safe(mode, sector, final_text or "", context, trade_date)

        return _callback

    async def process_message(
        self, user_message: str, stream: bool = False, history: list = None
    ) -> dict | AsyncGenerator:
        """
        处理用户消息。流式模式下自动将dict包装为生成器。

        history: 之前的对话轮次 [{"role": "user"|"assistant", "content": str}]，
        仅保留最近 10 轮（20 条），用于追问场景的意图继承与闲聊上下文。
        """
        history = _trim_history(history)
        intent, sector, inherited_from = _resolve_contextual_intent(user_message, history)
        # 继承意图的追问：在数据路径的 user prompt 开头加一行上下文说明
        context_note = (
            f"用户此前在询问{inherited_from}，本条为追问，请结合该语境组织回答。"
            if inherited_from else None
        )

        # 第二波：复杂分析（跨实体比较/多实体问题）优先升级为 Agent 工具循环。
        # 裸行业名命中板块意图时，若消息实为比较/多实体分析（如『比较白酒和半导体』），
        # 也让位给 Agent 循环；规则保守，拿不准走原路径
        if intent in ("general_chat", "sector_deep_dive") and _should_use_agent_query(user_message):
            return await self._agent_query(user_message, stream, history=history)

        if intent == "general_chat":
            return await self._chat(user_message, stream, history=history)
        elif intent == "market_review":
            return await self._market_review(stream, context_note=context_note)
        elif intent == "news_only":
            return await self._news_only(sector, stream, message=user_message)
        elif intent == "stock_query":
            result = await self._stock_query(user_message, False)
        elif intent == "futures_query":
            result = await self._futures_query(user_message, False)
        elif intent == "fund_query" or intent == "mcp_query":
            result = await self._fund_query(user_message, False)
        else:
            return await self._sector_deep_dive(sector, stream, context_note=context_note)

        # 简单查询返回dict → 流式模式包装为生成器
        if stream and isinstance(result, dict):
            async def _wrap():
                text = result.get("content", "")
                for i in range(0, len(text), 80):
                    yield text[i:i+80]
                    await asyncio.sleep(0.01)
            return _wrap()
        return result

    async def _chat(self, message: str, stream: bool, history: list = None):
        """通用对话。带对话历史，让闲聊也有上下文。"""
        system = get_system_prompt("general_chat")
        return await self._call_llm(system, message, stream, history=history)

    async def _market_review(self, stream: bool, context_note: str = None):
        """全市场复盘。context_note 非空时（继承意图的追问）在 user prompt 开头加一行说明。"""
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
            self._cache[cache_key] = snapshot  # 按 key 更新，避免冲掉其他快照
        market_data = format_market_data_for_prompt(snapshot)

        # 3. 构建 prompt（日期注入系统提示词）
        system = get_system_prompt("market_review").replace("[日期]", date_display)

        note_line = f"{context_note}\n\n" if context_note else ""
        user_prompt = f"""{note_line}交易日：{date_display} {weekday}{date_note}

{market_data}

生成A股市场复盘。不列出新闻（如需新闻请说\"今天市场新闻\"）。31行业全部列出。数据缺失标[UNSOURCED]。"""

        # 第四波：存档接入——流式走 _stream_response 结束回调，非流式在最终 content 产出后落档
        if stream:
            return await self._call_llm(
                system, user_prompt, stream,
                archive_callback=self._stream_archive_callback(
                    "market_review", None, user_prompt, date_str
                ),
            )
        result = await self._call_llm(system, user_prompt, stream)
        if isinstance(result, dict):
            self._archive_safe(
                "market_review", None, result.get("content", ""), user_prompt, date_str
            )
        return result

    async def _fetch_sector_extras(
        self, sector: str, trade_date: str
    ) -> tuple[Optional[dict], Optional[dict], Optional[dict]]:
        """
        并行采集板块估值、资金流向与景气度（业绩预告/快报，Tushare）。
        单点失败（import 缺失 / 抛异常 / 返回非dict）降级为 None，绝不向上抛出。
        trade_date 格式 %Y%m%d。
        """
        try:
            from .data_fetcher import fetch_sector_valuation, fetch_sector_moneyflow
        except ImportError as e:
            logger.warning("板块深挖附加数据函数未就绪（data_fetcher 未提供）: %s", e)
            return None, None, None

        # 景气度函数独立 import，缺失时仅景气度降级为 None，不影响估值/资金
        try:
            from .data_fetcher import fetch_sector_earnings
        except ImportError as e:
            logger.warning("板块景气度函数未就绪（data_fetcher 未提供）: %s", e)
            fetch_sector_earnings = None

        loop = asyncio.get_running_loop()

        async def _safe(fn, label: str) -> Optional[dict]:
            if fn is None:
                return None
            try:
                result = await loop.run_in_executor(None, fn, sector, trade_date)
                if not isinstance(result, dict):
                    logger.warning("%s 返回非dict（%s板块 %s）: %r", label, sector, trade_date, result)
                    return None
                return result
            except Exception as e:
                logger.warning("%s 获取失败（%s板块 %s）: %s", label, sector, trade_date, e, exc_info=True)
                return None

        valuation, moneyflow, earnings = await asyncio.gather(
            _safe(fetch_sector_valuation, "板块估值"),
            _safe(fetch_sector_moneyflow, "板块资金流向"),
            _safe(fetch_sector_earnings, "板块景气度"),
        )
        return valuation, moneyflow, earnings

    async def _sector_deep_dive(self, sector: str, stream: bool, context_note: str = None):
        """单板块深度聚焦。context_note 非空时（继承意图的追问）在 user prompt 开头加一行说明。"""
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
            self._cache[cache_key] = snapshot
        market_data = format_market_data_for_prompt(snapshot)

        # 板块估值 + 资金流向 + 景气度（独立采集，单点失败降级为『数据未获取』，不影响主流程）
        valuation, moneyflow, earnings = await self._fetch_sector_extras(sector, date_str)
        valuation_block = _format_sector_valuation_block(valuation)
        moneyflow_block = _format_sector_moneyflow_block(moneyflow)
        earnings_block = _format_sector_earnings_block(earnings)

        system = get_system_prompt("sector_deep_dive", sector)
        system = system.replace("[日期]", date_display).replace("[板块名]", sector)

        note_line = f"{context_note}\n\n" if context_note else ""
        user_prompt = f"""{note_line}交易日：{date_display} {weekday} | 行业：{sector}

【一、行情与趋势数据】
{market_data}

{valuation_block}

{moneyflow_block}

{earnings_block}

【五、新闻与景气背景】
上方"行情与趋势数据"中的东方财富实时快讯与新浪历史新闻已按{sector}行业过滤，请作为催化/风险判断的素材引用，不要原样罗列。

深度分析{sector}板块。输出标题需包含日期{date_display}。新闻条目仅作为第五维催化与风险的引用素材，不要单独罗列新闻清单。数据缺失处标注数据暂缺，禁止编造。"""

        if stream:
            # 流式路径跳过自我审查以保延迟（审查+修正需两轮非流式调用，会显著抬高首字延迟）
            # 第三波：数字溯源校验 log-only 接入——仅记录疑似无出处数字，不修改输出、不阻塞
            if validators is not None:
                try:
                    stream_violations = validators.find_unsourced_numbers(
                        system + "\n\n" + user_prompt, user_prompt
                    )
                    if stream_violations:
                        _log_stream_violations(f"板块深挖({sector})", stream_violations)
                except Exception as e:
                    logger.warning("流式板块深挖数字溯源校验异常（log-only，忽略）: %s", e, exc_info=True)
            # 第四波：流结束后以最终全文存档（best-effort 回调）
            return await self._call_llm(
                system, user_prompt, stream,
                archive_callback=self._stream_archive_callback(
                    "sector_deep_dive", sector, user_prompt, date_str
                ),
            )
        result = await self._call_llm(system, user_prompt, stream)
        if isinstance(result, dict):
            # 第二波：多 pass 自我审查修正；数据上下文用完整 user_prompt 供数字出处核对
            draft = result.get("content", "")
            result["content"] = _clean_markdown(await self._critique_and_revise(draft, user_prompt))
            # 第四波：存档最终产出（审查修正后的终稿）
            self._archive_safe(
                "sector_deep_dive", sector, result.get("content", ""), user_prompt, date_str
            )
        return result

    async def _news_only(self, sector: str, stream: bool, message: str = ""):
        """新闻专属模式：五源新闻池 + 行业关键词过滤，48小时尽量多展示。

        - 数据层优先用 fetch_news_pool（五源聚合、跨源去重、时间已修复），
          未就绪或调用失败时降级到旧三源直采逻辑。
        - 行业过滤除标题外，同时匹配 time/content/summary 字段（有的话）。
        - 展示策略：按天分组、按时间倒序；全市场查询每天上限30条（超限按重要性
          评分截断），行业查询每天全量展示。
        - message 含解读类触发词（解读/分析/影响/怎么看/意味着什么/说明什么）时
          走 LLM 分析模式（NEWS_ANALYSIS_PROMPT）；否则非流式直接透传原文，
          不经过 LLM 改写。
        """
        today = datetime.now()
        trade_date = _get_latest_trade_date(today)
        date_display = trade_date.strftime("%Y年%m月%d日")
        weekday = ["周一","周二","周三","周四","周五","周六","周日"][trade_date.weekday()]

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

        # ── 1. 数据采集：五源新闻池优先，未就绪/失败时降级到旧三源直采 ──
        sources = self._collect_news_sources(sector, trade_date, SW_ALL_KEYWORDS)

        # ── 2. 跨源去重 + 行业过滤 + 按天分组 ──
        sector_kws = SW_ALL_KEYWORDS.get(sector, [sector]) if sector else None
        seen_titles = set()
        source_counts = {}  # 源 → 有效条数（过滤后）
        for src in ("sina", "eastmoney", "mcp", "cls", "tushare"):
            for item in sources.get(src) or []:
                title = (item.get("title", "") or "").strip()
                t = (item.get("time", "") or "")[:10]
                if not t:
                    # 无时间条目不再丢弃：归入交易日分组，照常参与去重与展示
                    t = trade_date.strftime("%Y-%m-%d")
                if not title or len(title) < 4:
                    continue
                if title in seen_titles:
                    continue
                if sector_kws:
                    # 标题之外，content/summary/time 字段（有的话）也参与关键词匹配
                    match_text = title + (item.get("content", "") or "") \
                        + (item.get("summary", "") or "") + (item.get("time", "") or "")
                    if not any(kw in match_text for kw in sector_kws):
                        continue
                seen_titles.add(title)
                by_date[t].append({
                    "title": title,
                    "time": (item.get("time", "") or ""),
                    "source": src,
                })
                source_counts[src] = source_counts.get(src, 0) + 1

        total_raw = sum(len(v) for v in by_date.values())

        # ── 3. 组装展示文本：按天分组、时间倒序、全市场每天上限30条 ──
        if total_raw == 0:
            if sector:
                news_text = f"{label}新闻汇总 — {date_display} {weekday}\n未找到与该行业相关的新闻。请尝试更宽泛的关键词，或查询\"全市场新闻\"。"
            else:
                news_text = f"{label}新闻汇总 — {date_display} {weekday}\n未找到相关新闻。"
        else:
            day_cap = None if sector else 30  # 全市场每天上限30条，行业查询全量展示
            truncated = False
            display_total = 0
            blocks = []
            for d in sorted(by_date.keys(), reverse=True):
                items = by_date[d]
                raw_cnt = len(items)
                if day_cap and raw_cnt > day_cap:
                    # 超限：按重要性评分排序截断（同分按时间倒序）
                    items = sorted(
                        items,
                        key=lambda x: (_news_importance(x["title"]), x["time"]),
                        reverse=True,
                    )[:day_cap]
                    truncated = True
                else:
                    items = sorted(items, key=lambda x: x["time"], reverse=True)
                display_total += len(items)
                block = f"\n--- {d}（{len(items)}条"
                if raw_cnt > len(items):
                    block += f"，原始{raw_cnt}条按重要性截断"
                block += "）---\n"
                for it in items:
                    src_name = _NEWS_SOURCE_NAMES.get(it["source"], it["source"])
                    block += f"[{_fmt_news_time(it['time'], fallback=d)}] 【{src_name}】{it['title']}\n"
                blocks.append(block)

            # 头部：总条数、覆盖时间段、各源条数
            all_times = [it["time"] for v in by_date.values() for it in v if it["time"]]
            span = ""
            if all_times:
                span = f"覆盖{_fmt_news_time(min(all_times))}~{_fmt_news_time(max(all_times))}"
            src_stat = " / ".join(
                f"{_NEWS_SOURCE_NAMES.get(s, s)}{source_counts[s]}条"
                for s in ("sina", "eastmoney", "mcp", "cls", "tushare")
                if source_counts.get(s)
            )
            news_text = f"{label}新闻汇总 — {date_display} {weekday}（48小时覆盖，共{display_total}条）\n"
            meta_parts = [p for p in (span, f"来源：{src_stat}" if src_stat else "") if p]
            if truncated:
                meta_parts.append("单日超30条已按重要性截断")
            if meta_parts:
                news_text += " | ".join(meta_parts) + "\n"
            news_text += "".join(blocks)

        # ── 4. 分析模式：消息含解读类触发词且有新闻时，走 LLM 解读而非透传 ──
        analysis_mode = (
            bool(message) and total_raw > 0
            and any(w in message for w in _NEWS_ANALYSIS_TRIGGERS)
        )
        if analysis_mode:
            system = get_system_prompt("news_analysis")
            user_prompt = (
                f"{news_text}\n\n以上是{label}近48小时新闻清单。请按系统提示的框架输出解读，"
                "只分析清单内的新闻条目，清单未覆盖的写「本期新闻未覆盖」。"
            )
            return await self._call_llm(system, user_prompt, stream)

        # 透传模式：非流式直接返回原文，不经过 LLM 改写
        system = "你是财经新闻编辑。将以下新闻汇总原样输出。不删减、不分析、不改格式。"
        user_prompt = f"{news_text}\n\n以上{label}48小时新闻汇总。原样输出。"

        result = await self._call_llm(system, user_prompt, stream)
        if not stream and isinstance(result, dict):
            result["content"] = news_text
        return result

    def _collect_news_sources(self, sector: str, trade_date: datetime, sw_keywords: dict) -> dict:
        """新闻采集：优先 fetch_news_pool 五源聚合，未就绪/失败时降级到旧三源直采。

        返回统一结构 {'sina','eastmoney','mcp','cls','tushare'} → 条目列表，
        每条至少含 {'title','time'}，部分源可能带 content/summary 字段。
        """
        sector_kws = sw_keywords.get(sector, [sector]) if sector else None
        try:
            from agent.data_fetcher import fetch_news_pool as _pool
        except ImportError as e:
            logger.warning("fetch_news_pool 未就绪，新闻模式降级到旧三源逻辑: %s", e)
            _pool = None
        if _pool is not None:
            try:
                pool = _pool(sector_keywords=sector_kws, days=3)
                if isinstance(pool, dict):
                    return {k: (pool.get(k) or []) for k in ("sina", "eastmoney", "mcp", "cls", "tushare")}
                logger.warning("fetch_news_pool 返回非dict，降级到旧三源逻辑: %r", pool)
            except Exception as e:
                logger.warning("fetch_news_pool 调用失败，降级到旧三源逻辑: %s", e, exc_info=True)

        # 旧三源直采：Sina 每天50条×3天覆盖 + EM补最新 + 智研快讯
        from agent.data_fetcher import (
            fetch_sina_news as _s,
            fetch_eastmoney_news as _e,
            fetch_mcp_news as _m,
        )
        search_kw = sector if sector else "A股"
        mcp_items = _m(search_kw, 60)
        em_items = _e(50)
        d0 = trade_date.strftime("%Y-%m-%d")
        d1 = (trade_date - timedelta(days=1)).strftime("%Y-%m-%d")
        d2 = (trade_date - timedelta(days=2)).strftime("%Y-%m-%d")
        all_sina = []
        for news_date in [d0, d1, d2]:
            # fetch_sina_news 的 page 参数内部硬编码为 1，每个日期只需调用一次
            items = _s(50, news_date)
            if items:
                all_sina.extend(items)
        return {
            "sina": all_sina,
            "eastmoney": em_items or [],
            "mcp": mcp_items or [],
            "cls": [],
            "tushare": [],
        }


    async def _stock_query(self, message: str, stream: bool):
        """个股查询——中文名本地映射，不走搜索API。"""
        from agent.data_fetcher import fetch_stock_quote, fetch_stock_kline, fetch_stock_news
        # 常见中文名→代码映射
        NAME_MAP = {
            "茅台": ("贵州茅台", "cn", "sh600519"), "五粮液": ("五粮液", "cn", "sz000858"),
            "宁德": ("宁德时代", "cn", "sz300750"), "比亚迪": ("比亚迪", "cn", "sz002594"),
            "中芯": ("中芯国际", "cn", "sh688981"), "招商银行": ("招商银行", "cn", "sh600036"),
            "平安": ("中国平安", "cn", "sh601318"), "苹果": ("Apple", "us", "AAPL"),
            "特斯拉": ("Tesla", "us", "TSLA"), "英伟达": ("NVIDIA", "us", "NVDA"),
            "腾讯": ("腾讯", "hk", "00700"), "阿里": ("阿里巴巴", "us", "BABA"),
        }
        matched = None
        for kw, (name, mkt, code) in NAME_MAP.items():
            if kw in message:
                matched = (name, mkt, code)
                break
        if not matched:
            return {"role": "assistant", "content": "请使用常见股票名称查询，如：茅台、宁德、比亚迪、苹果、特斯拉。或输入完整代码。"}
        name, market, code = matched
        try:
            quote = fetch_stock_quote(market, code) or {}
            kline = fetch_stock_kline(market, code, 5) or []
            news = fetch_stock_news(code, market, 5) or []
            kline_str = ", ".join(k.get("date","")[-5:] + ":" + str(k.get("close","?")) for k in kline[:5])
            news_str = " | ".join(n.get("title","")[:40] for n in news[:3])
            info = f"{name}({code})\n行情：价格{quote.get('price','?')} 涨跌{quote.get('pct','?')}%\n开盘{quote.get('open','?')} 最高{quote.get('high','?')} 最低{quote.get('low','?')}\n近5日K线：{kline_str}\n相关新闻：{news_str}\n"
        except Exception as e:
            logger.warning("个股数据获取失败(%s %s): %s", market, code, e, exc_info=True)
            info = f"{name}({code})\n数据暂不可用，请稍后再试。"
        return {"role": "assistant", "content": info}

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

    async def _generic_mcp(self, message: str, stream: bool, max_rounds: int = 5):
        """通用MCP查询——使用DeepSeek原生function calling，100%可靠。"""
        from agent.data_fetcher import get_mcp_tools, _mcp_call
        tools = get_mcp_tools()
        if not tools:
            return {"role": "assistant", "content": "MCP服务暂不可用"}

        # 转成OpenAI function calling格式
        ds_tools = []
        for t in tools[:80]:
            props = {}
            for p in t.get("params", [])[:5]:
                props[p] = {"type": "string", "description": p}
            ds_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("desc", "")[:200],
                    "parameters": {"type": "object", "properties": props}
                }
            })

        messages = [
            {"role": "system", "content": "你是数据查询助手。用纯文本回答，禁止Markdown表格（禁止|和-组成的表格线）。数据用缩进对齐或列表呈现。"},
            {"role": "user", "content": message}
        ]
        all_results = []

        for _ in range(max_rounds):
            completion = await self.client.chat.completions.create(
                model=self.model, messages=messages, tools=ds_tools,
                temperature=0.1, max_tokens=4096,
            )
            choice = completion.choices[0]
            if choice.finish_reason == "stop" and not choice.message.tool_calls:
                # 无需工具，直接回答
                answer = choice.message.content or ""
                if all_results:
                    answer += f"\n[数据来源: {len(all_results)}次MCP调用]"
                return {"role": "assistant", "content": answer}

            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    fn = tc.function
                    try:
                        args = json.loads(fn.arguments) if fn.arguments else {}
                    except Exception as e:
                        logger.warning("MCP工具参数解析失败(%s): %s", fn.name, e, exc_info=True)
                        args = {}
                    data = _mcp_call(fn.name, args)
                    data_str = json.dumps(data, ensure_ascii=False)[:3000]
                    if not data_str or data_str == "{}" or ('"data":[]' in data_str and '"s_list":[]' not in data_str):
                        return {"role": "assistant", "content": "抱歉，该数据暂不支持查询。新浪智研API返回为空，可能原因：非交易时段数据未更新、该接口暂不可用、或查询参数不支持。请尝试其他问题。"}
                    all_results.append({"tool": fn.name, "result": data_str[:300]})
                    messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": data_str})
            else:
                break

        # 汇总
        if all_results:
            summary = "\n".join(f"- {r['tool']}: {r['result']}" for r in all_results)
            return {"role": "assistant", "content": f"查询结果:\n{summary}\n[共{len(all_results)}次工具调用]"}
        return {"role": "assistant", "content": "未获取到数据，请尝试更具体的查询。"}

    async def _fund_query(self, message: str, stream: bool):
        """通用MCP查询。简单数据用预封装函数直接返回，复杂问题走function calling。"""
        from agent.data_fetcher import fetch_market_breadth, fetch_hot_stocks, fetch_forex, fetch_futures
        msg = message
        # 涨跌分布 → 直接用预封装函数，不绕function calling
        if any(kw in msg for kw in ["涨跌","上涨","下跌","涨停","跌停","涨了","跌了","多少家"]):
            try:
                b = fetch_market_breadth()
            except Exception as e:
                logger.warning("fetch_market_breadth 失败: %s", e, exc_info=True)
                return {"role": "assistant", "content": "涨跌分布数据暂不可用，数据源解析失败。请稍后再试。"}
            if b:
                return {"role": "assistant", "content": f"今日A股涨跌分布：上涨{b.get('total_up','?')}家 下跌{b.get('total_down','?')}家 平盘{b.get('平','?')}家。涨停{b.get('涨停','?')}家 跌停{b.get('跌停','?')}家。"}
        if any(kw in msg for kw in ["热搜","热榜"]):
            h = fetch_hot_stocks()
            if h:
                items = "\n".join(f"{i+1}. {s['name']}({s['code']}) 热度{s.get('heat','?')}" for i, s in enumerate(h[:10]))
                return {"role": "assistant", "content": f"A股热搜榜Top10：\n{items}\n"}
            return {"role": "assistant", "content": "热搜数据暂不可用，新浪智研API返回为空。请稍后再试。"}
        if any(kw in msg for kw in ["汇率","人民币","美元"]):
            f = fetch_forex()
            if f.get("在岸人民币","?") != "?":
                return {"role": "assistant", "content": f"最新汇率：在岸人民币 {f['在岸人民币']} 涨跌{f.get('涨跌','?')} "}
            return {"role": "assistant", "content": "汇率数据暂不可用，新浪智研API返回为空。请稍后再试。"}
        if any(kw in msg for kw in ["期货","黄金","原油","铜"]):
            kw_map = {"黄金":("gn","AU0"),"原油":("gn","SC0"),"铜":("gn","CU0")}
            for kw, (mkt, sym) in kw_map.items():
                if kw in msg:
                    d = fetch_futures(mkt, sym)
                    if d.get("price"):
                        return {"role": "assistant", "content": f"{kw}期货：价格{d['price']} 涨跌{d.get('pct','?')}% 成交量{d.get('vol','?')} "}
                    return {"role": "assistant", "content": f"{kw}期货数据暂不可用，请稍后再试。"}
        return await self._generic_mcp(message, stream)

    # 第二波：Agent 工具循环与多 pass 生成配置
    _AGENT_MAX_ROUNDS = 8        # 工具调用循环上限，超轮降级 _chat
    _TOOL_RESULT_MAX_CHARS = 3000  # 单条工具结果截断长度，防 context 膨胀
    _CRITIQUE_MIN_CHARS = 500    # 草稿不长于此长度时不做自我审查（成本护栏）

    async def _agent_query(self, user_message: str, stream: bool, history: list = None):
        """
        Agent 工具调用循环（第二波）：模型自主决定调用哪些数据工具，多轮拿数后再成文。
        仅由 process_message 在复杂分析场景路由进入。
        兜底原则：工具注册表未就绪 / 任一轮异常 / 循环超轮 → 降级 _chat，绝不向上抛异常。
        """
        if not TOOL_REGISTRY or execute_tool is None or AGENT_QUERY_PROMPT is None:
            logger.warning("Agent 工具注册表或提示词未就绪，降级为普通对话路径")
            return await self._chat(user_message, stream, history=history)

        date_display = datetime.now().strftime("%Y年%m月%d日")
        date_str = datetime.now().strftime("%Y%m%d")  # 第四波存档用：agent_query 的 trade_date 取当前日期
        system = AGENT_QUERY_PROMPT + f"\n\n当前日期：{date_display}。"
        messages = [{"role": "system", "content": system}]
        messages.extend(_trim_history(history))  # 最多保留最近 10 轮（20 条）
        messages.append({"role": "user", "content": user_message})

        tool_context_parts = []  # 工具返回摘要，供自我审查核对数字出处
        loop = asyncio.get_running_loop()

        try:
            for _ in range(self._AGENT_MAX_ROUNDS):
                # 工具循环阶段一律非流式（tool calling 无法流式）
                completion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_REGISTRY,
                    tool_choice="auto",
                    temperature=0.2,
                    max_tokens=8192,
                )
                choice = completion.choices[0]
                tool_calls = getattr(choice.message, "tool_calls", None)

                if not tool_calls:
                    # 模型返回纯文本 → 工具循环收敛，最终成文
                    # 数据上下文（用户问题 + 工具返回摘要拼接）：供数字校验与第四波存档共用
                    query_context = user_message + "\n\n" + "\n".join(tool_context_parts)
                    if stream:
                        # 最终纯文本生成那一轮改用流式输出（不传 tools，模型只能写正文）。
                        # 流式路径同时跳过自我审查以保延迟，见 _critique_and_revise 注释。
                        # 第三波：数字溯源校验 log-only 接入——上下文用 tool_context_parts 拼接，
                        # 仅记录不拦截
                        if validators is not None:
                            try:
                                stream_violations = validators.find_unsourced_numbers(
                                    system + "\n\n" + user_message, query_context
                                )
                                if stream_violations:
                                    _log_stream_violations("Agent 查询", stream_violations)
                            except Exception as e:
                                logger.warning("流式 Agent 查询数字溯源校验异常（log-only，忽略）: %s", e, exc_info=True)
                        # 第四波：流结束后以最终全文存档（trade_date 用当前日期）
                        return self._stream_response(
                            messages,
                            archive_callback=self._stream_archive_callback(
                                "agent_query", None, query_context, date_str
                            ),
                        )
                    draft = choice.message.content or ""
                    final = await self._critique_and_revise(draft, query_context)
                    disclaimer = "\n\n风险提示：以上内容仅为客观数据整理与公开信息分析，不构成任何投资建议。市场有风险，投资需谨慎。"
                    content = _clean_markdown(final) + disclaimer
                    # 第四波：存档最终产出（trade_date 用当前日期）
                    self._archive_safe("agent_query", None, content, query_context, date_str)
                    return {"role": "assistant", "content": content}

                # 有工具调用：回显 assistant 消息，逐个在线程池执行并追加 tool 结果
                messages.append(_assistant_tool_message(choice.message))
                for tc in tool_calls:
                    fn_name = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except Exception as e:
                        logger.warning("Agent 工具参数解析失败(%s): %s", fn_name, e, exc_info=True)
                        fn_args = {}
                    try:
                        result = await loop.run_in_executor(None, execute_tool, fn_name, fn_args)
                    except Exception as e:
                        logger.warning("Agent 工具执行异常(%s): %s", fn_name, e, exc_info=True)
                        result = {"ok": False, "error": f"工具执行异常: {e}"}
                    content = json.dumps(result, ensure_ascii=False)[:self._TOOL_RESULT_MAX_CHARS]
                    tool_context_parts.append(f"[{fn_name}] {content[:800]}")
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
        except Exception as e:
            logger.warning("Agent 工具循环异常，降级为普通对话路径: %s", e, exc_info=True)
            return await self._chat(user_message, stream, history=history)

        # 循环超轮：模型仍要调工具 → 降级
        logger.warning("Agent 工具循环达到 %d 轮上限仍未收敛，降级为普通对话路径", self._AGENT_MAX_ROUNDS)
        return await self._chat(user_message, stream, history=history)

    async def _critique_and_revise(self, draft: str, context: str) -> str:
        """
        多 pass 生成：先用 CRITIQUE_PROMPT 审查草稿（数字出处/禁用词/过度推断/合规边界/AI腔），
        再基于审查意见生成修正版。两次均为非流式调用。
        成本护栏：草稿不超过 500 字直接返回原文；审查结论为「通过」时跳过修正。
        失败安全：任何一步失败（或修正版异常偏短）都返回原 draft，绝不抛出。
        """
        if CRITIQUE_PROMPT is None:
            return draft
        if not draft or len(draft) <= self._CRITIQUE_MIN_CHARS:
            return draft
        context_part = (context or "")[:4000]
        # 第三波：确定性数字溯源校验——违规清单追加到 critique user 消息末尾；
        # validators 未就绪或执行异常时仅记 log，不影响审查主流程
        validator_appendix = ""
        if validators is not None:
            try:
                violations = validators.find_unsourced_numbers(draft, context_part)
                if violations:
                    validator_appendix = (
                        "\n\n【确定性校验结果】以下数字在数据上下文中未找到出处，审查时必须要求删除或改写：\n"
                        + validators.format_violations_for_critique(violations)
                    )
            except Exception as e:
                logger.warning("数字溯源校验执行异常，跳过校验结果追加: %s", e, exc_info=True)
        try:
            critique_completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CRITIQUE_PROMPT},
                    {"role": "user", "content": f"【数据上下文】\n{context_part}\n\n【待审查初稿】\n{draft}{validator_appendix}"},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            critique = critique_completion.choices[0].message.content or ""
        except Exception as e:
            logger.warning("自我审查（critique）失败，返回原草稿: %s", e, exc_info=True)
            return draft

        # 五条清单全部通过 → 无需修正，省一次调用
        if critique.strip().startswith("通过"):
            return draft

        try:
            revise_completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是修订编辑。根据审查意见修正初稿：删除或改写无出处的数字、禁用词、无据断言、合规越界和 AI 腔内容，保留其余内容与原结构。只输出修正后的完整正文，不要解释。"},
                    {"role": "user", "content": f"【数据上下文】\n{context_part}\n\n【初稿】\n{draft}\n\n【审查意见】\n{critique}"},
                ],
                temperature=0.2,
                max_tokens=8192,
            )
            revised = revise_completion.choices[0].message.content or ""
            if len(revised.strip()) < 50:  # 修正版异常偏短（疑似失败），回退原稿
                logger.warning("自我审查修正版过短（%d 字），返回原草稿", len(revised.strip()))
                return draft
            return revised
        except Exception as e:
            logger.warning("自我审查修正（revise）失败，返回原草稿: %s", e, exc_info=True)
            return draft

    async def _call_llm(
        self, system_prompt: str, user_message: str, stream: bool = False,
        history: list = None, archive_callback=None,
    ):
        """
        调用 DeepSeek API。
        history 非空时按 system + history + 当前 user 组装 messages（闲聊上下文）。
        archive_callback：第四波存档回调，仅流式路径有效，透传给 _stream_response，
        在流结束后以最终全文调用（签名为 callback(accumulated_text)）。
        """
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(_trim_history(history))
        messages.append({"role": "user", "content": user_message})

        if stream:
            return self._stream_response(messages, archive_callback=archive_callback)
        else:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=8192,
            )
            raw = completion.choices[0].message.content
            # 输出截断检测：finish_reason="length" 表示撞了 max_tokens，自动续写一次
            if getattr(completion.choices[0], "finish_reason", None) == "length":
                continuation = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages + [
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": "请从上次中断处继续，不要重复已输出内容。"},
                    ],
                    temperature=0.2,
                    max_tokens=8192,
                )
                more = continuation.choices[0].message.content or ""
                if getattr(continuation.choices[0], "finish_reason", None) == "length":
                    logger.warning("LLM 输出截断，续写一次后仍为 length，放弃继续续写")
                else:
                    logger.info("检测到 LLM 输出截断（finish_reason=length），已自动续写一次")
                raw = (raw or "") + more
            disclaimer = "\n\n风险提示：以上内容仅为客观数据整理与公开信息分析，不构成任何投资建议。市场有风险，投资需谨慎。"
            clean = _clean_markdown(raw)
            return {"role": "assistant", "content": clean + disclaimer}

    async def _stream_response(self, messages: list, archive_callback=None) -> AsyncGenerator:
        """流式响应生成器。注意：流式不清理markdown，但非流式会清理。

        archive_callback：可选存档回调（第四波），在流结束（生成器被完整消费、
        含截断续写完成）后以最终全文调用一次；客户端中途断开时不触发。
        """
        accumulated = ""
        continued = False
        current_messages = messages
        while True:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=current_messages,
                temperature=0.2,
                max_tokens=8192,
                stream=True,
            )
            finish_reason = None
            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.delta.content:
                    accumulated += choice.delta.content
                    yield choice.delta.content
                # 只有最后一个 chunk 的 finish_reason 有值（中间 chunk 为 None）
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
            # 输出截断检测：撞了 max_tokens 则无缝续写一轮（最多一次），对用户透明
            if finish_reason == "length" and not continued:
                continued = True
                logger.warning("流式输出被截断（finish_reason=length），自动续写一次")
                current_messages = messages + [
                    {"role": "assistant", "content": accumulated},
                    {"role": "user", "content": "请从上次中断处继续，不要重复已输出内容。"},
                ]
                continue
            break

        # 第四波：流正常结束后以最终全文存档（fail-safe，回调异常只记 log）
        if archive_callback is not None:
            try:
                archive_callback(accumulated)
            except Exception as e:
                logger.warning("流式存档回调异常（忽略）: %s", e, exc_info=True)


# ── 全局单例 ──

_agent: Optional[MarketReviewAgent] = None


def get_agent() -> MarketReviewAgent:
    global _agent
    if _agent is None:
        _agent = MarketReviewAgent()
    return _agent
