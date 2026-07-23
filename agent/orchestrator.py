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
from types import SimpleNamespace
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

# 第七波：自选股（agent/watchlist.py 未就绪时自选股能力整体降级，不影响主流程）
try:
    from . import watchlist
except ImportError as e:
    logger.warning("agent/watchlist.py 未就绪，自选股能力将整体降级: %s", e)
    watchlist = None

# 第七波：行业知识库（agent/industry_kb.py 未就绪时知识库注入降级跳过，不影响主流程）
try:
    from . import industry_kb
except ImportError as e:
    logger.warning("agent/industry_kb.py 未就绪，行业知识库注入将降级跳过: %s", e)
    industry_kb = None

# 第七波：以史为鉴（agent/history_lens.py 未就绪时历史回顾注入降级跳过，不影响主流程）
try:
    from . import history_lens
except ImportError as e:
    logger.warning("agent/history_lens.py 未就绪，以史为鉴注入将降级跳过: %s", e)
    history_lens = None

# 第八波：MCP 结果压缩与错误识别（工程师B 在 data_fetcher.py 提供）。
# 跨层接口契约：防御式导入，未就绪时退回现状行为（json.dumps 截断 + 原文喂回），
# 绝不能硬依赖。
try:
    from .data_fetcher import compact_mcp_result, is_mcp_error, mcp_error_brief
except ImportError as e:
    logger.warning(
        "data_fetcher 的 compact_mcp_result/is_mcp_error/mcp_error_brief 未就绪，"
        "MCP 结果压缩与错误简述将降级为现状行为: %s", e,
    )
    compact_mcp_result = None
    is_mcp_error = None
    mcp_error_brief = None

# 第十二波：Agent 工程三件套（agent/agent_audit.py，纯 stdlib 无重依赖，模块顶层导入）。
# Scratchpad 审计日志 / ToolCallGuard 工具调用软护栏 / microcompact 上下文压缩。
# 导入失败时三件套整体降级，不影响主流程。
try:
    from .agent_audit import Scratchpad, ToolCallGuard, microcompact
except ImportError as e:
    logger.warning("agent/agent_audit.py 未就绪，审计三件套将整体降级: %s", e)
    Scratchpad = None
    ToolCallGuard = None
    microcompact = None


def _audit_safe(func, *args, **kwargs):
    """审计三件套钩子的统一防御调用：任何异常只记 log 并返回 None，绝不影响主流程。"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning("Agent 审计钩子异常（忽略，不影响主流程）: %s", e, exc_info=True)
        return None


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

# ── 第七波：自选股 / 行业知识库 / 以史为鉴 集成 ──

# 自选股动作词（detect_intent 与 _watchlist handler 共用同一口径）
_WATCHLIST_ADD_VERBS = ("添加", "加入", "加")
_WATCHLIST_REMOVE_VERBS = ("删除", "移除", "删")
_WATCHLIST_REVIEW_WORDS = ("复盘", "看看", "怎么样")

_WATCHLIST_USAGE = (
    "自选股用法：\n"
    "- 添加：『加自选 茅台』或『添加自选 600519』\n"
    "- 删除：『删自选 茅台』\n"
    "- 查看清单：『我的自选股』\n"
    "- 逐只复盘：『复盘我的自选股』"
)

_WATCHLIST_EMPTY_GUIDE = (
    "您还没有添加自选股。回复『加自选 茅台』即可添加；"
    "添加后可回复『我的自选股』查看清单，或『复盘我的自选股』获取逐只复盘。"
)

# 内联兜底提示词：get_system_prompt("watchlist") 取不到或返回空时使用
_DEFAULT_WATCHLIST_SYSTEM_PROMPT = (
    "你是严谨的自选股复盘助手。基于用户自选股清单与实时行情数据逐只复盘："
    "每只给出当日表现、关键价位与一句话简评，最后给一段整体小结。"
    "仅使用上方提供的数据，数据缺失处标注数据暂缺，禁止编造；"
    "纯文本输出，不使用 Markdown 表格；内容仅为客观数据整理，不构成投资建议。"
)


def _industry_kb_block(sector: str) -> str:
    """第七波：行业知识库注入块（fail-safe）。模块未就绪/未知行业/异常返回空串。"""
    if industry_kb is None:
        return ""
    try:
        block = industry_kb.format_kb_block(sector)
        if block:
            return f"\n\n{block}"
    except Exception as e:
        logger.warning("行业知识库注入失败（%s板块，跳过）: %s", sector, e, exc_info=True)
    return ""


def _history_note_block(sector=None, mode=None) -> str:
    """第七波：以史为鉴注入块（fail-safe）。模块自带头部，此处不加任何引导语。"""
    if history_lens is None:
        return ""
    try:
        note = history_lens.get_history_note(sector=sector, mode=mode)
        if note:
            return f"\n\n{note}"
    except Exception as e:
        logger.warning(
            "以史为鉴注入失败（sector=%s mode=%s，跳过）: %s", sector, mode, e, exc_info=True
        )
    return ""


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


# ── 社媒舆情 / 投资人格 意图关键词（高优先级，判定逻辑见 detect_intent）──

# 强信号：单独命中即判 social_sentiment，无需市场语境
_SOCIAL_STRONG_KEYWORDS = [
    "股吧", "微博", "知乎", "抖音", "B站", "b站", "哔哩", "小红书",
    "热搜", "热榜", "舆情", "舆论",
]

# 弱信号：需同时带市场语境（市场语境词或个股/行业实体）才判 social_sentiment
_SOCIAL_WEAK_KEYWORDS = [
    "情绪", "人气", "热度", "关注", "讨论", "评论",
    "涨停", "跌停", "炸板", "打板", "连板",
]

# 投资人格/框架：需市场语境才判 persona
_PERSONA_KEYWORDS = [
    "价值投资", "成长投资", "趋势交易", "逆向投资",
    "巴菲特", "格雷厄姆", "投资框架", "投资风格",
]

# 校园知识库：选课/课程评价/保研交换/校园生活类，单独命中即判 campus_kb
# （与金融语境正交，无需市场语境；插入位置与 social_sentiment 同块，
#  含『复盘』或金融信号词的消息让位金融路由——宁可漏判，不可误判）
_CAMPUS_KB_KEYWORDS = [
    # 学业与课程
    "选课", "课程", "这门课", "点评", "老师", "给分", "绩点", "GPA", "gpa",
    "必修", "培养方案", "考核", "学分", "考试", "期末", "作业", "难吗",
    "开课", "开的课", "开课单位", "值得选", "成绩", "课表",
    # 升学与发展
    "保研", "考研", "读研", "读博", "深造", "PhD", "phd", "出国",
    "交换", "留学", "留学生", "国际生", "国际学生", "转专业", "辅修",
    "奖学金", "实习", "加注", "导师", "四六级", "勤工",
    # 校园生活
    "宿舍", "紫荆", "几人间", "食堂", "校医院", "军训", "校园", "校内",
    "体测", "自习", "校园卡", "快递", "新生", "入学", "报到", "图书馆",
    "校历", "放假", "学费", "住宿费", "社团", "志愿者", "体育馆", "游泳",
    "体育课", "清华",
]

# 金融强信号词：与校园关键词共现时让位金融路由
# （防『清华系持股』『某基金会捐赠』类金融消息被校园路由劫持）
_FINANCE_STRONG_KEYWORDS = [
    "股", "板块", "大盘", "基金", "期货", "债", "行情", "涨停", "跌停",
    "资金流", "市值", "估值", "K线", "北向", "指数", "ETF",
]

# 显式数据查询指令词：命中时保持旧 mcp_query 路由，不判 social_sentiment/persona
# （防止抢走『今天涨停家数查询』等既有简单数据查询——宁可漏判，不可误判）
_DATA_QUERY_COMMANDS = ("查询", "查一下", "帮我查", "列出", "列表", "排名")

# ── 海外市场（美股/港股）识别：优先于 A 股复盘/板块/个股模板 ──
# 防『纳斯达克指数』被 A 股复盘模板劫持、『00700』前导零代码无人识别、
# 『美联储联邦基金利率』被 fund_query 的『基金』二字碰撞
_US_MARKET_KEYWORDS = [
    "美股", "纳斯达克", "纳指", "标普", "道琼斯", "道指", "中概",
    "美联储", "联邦基金", "非农", "苹果", "英伟达", "特斯拉", "微软",
    "谷歌", "亚马逊",
]
_HK_MARKET_KEYWORDS = [
    "港股", "恒生", "恒指", "港交所", "港股通", "南向", "H股",
    "腾讯", "美团", "小米集团",
]
_HK_CODE_RE = re.compile(r"(?<!\d)0\d{4}(?!\d)")  # 港股 5 位前导零代码（00700/06715）
_US_TICKER_RE = re.compile(
    r"(?<![A-Za-z])(AAPL|NVDA|TSLA|MSFT|GOOG|GOOGL|AMZN|META|NFLX|AMD|INTC|BABA|JD|PDD|NIO|XPEV|LI)(?![A-Za-z])"
)
_CN_CODE_RE = re.compile(r"(?<!\d)(?:60|68|00|30)\d{4}(?!\d)|(?:sh|sz|bj)\d{6}(?!\d)", re.IGNORECASE)

# 宏观关键词：命中即走 Agent 工具循环（宏观取数工具），
# 同时防止大写缩写撞个股 stock_patterns 的 [A-Z] 模式（『GDP增速』≠ 个股）
_MACRO_KEYWORDS = [
    "CPI", "cpi", "PPI", "ppi", "PMI", "pmi", "GDP", "gdp", "LPR", "lpr",
    "非农", "利率决议", "降准", "降息", "存款利率", "社融", "M2",
]

# 分析型个股/公司诉求：带实体命中时走 Agent 工具循环，
# 不被确定性行情卡短路（模板卡只有价格快照，无技术分析/估值/财报能力）
_ANALYTIC_STOCK_KEYWORDS = [
    "技术分析", "MACD", "macd", "KDJ", "kdj", "均线", "支撑", "压力位",
    "估值", "市盈率", "市净率", "贵不贵", "便不便宜",
    "财报", "年报", "季报", "业绩", "财务", "基本面", "股息", "分红", "换手率",
    "量价", "趋势", "对比", "比较", "公告", "介绍", "情况", "新闻",
]

# 两意图的路由提示：process_message 接线时作为 hint 透传给 _agent_query，
# 非空时以『【本问题路由提示】...』追加到 AGENT_QUERY_PROMPT 系统消息尾部
_AGENT_ROUTE_HINTS = {
    "social_sentiment": (
        "【本问题路由提示】用户在问市场/个股的社媒舆情与情绪面，请优先使用舆情情绪类"
        "工具取数：全市场或板块层面的情绪温度与涨跌停/连板结构用 get_market_sentiment；"
        "个股人气与舆情用 get_stock_sentiment；社媒热榜/热搜用 get_social_hot；"
        "指定平台（股吧/微博/知乎/抖音/B站）的内容搜索用 search_social_media。"
        "按问题涉及的范围（全市场/板块/个股/指定平台）选择合适工具组合，"
        "拿到数据后再组织回答，不要凭空描述舆情。"
    ),
    "persona": (
        "【本问题路由提示】用户希望以特定投资人格/框架分析，请先调用 analyze_with_persona "
        "获取对应投资框架，再按框架逐项调用数据工具取数分析，最后以该人格的视角成文。"
    ),
    "campus_kb": (
        "【本问题路由提示】用户在问清华校园相关问题（选课/课程评价/保研交换/校园生活等），"
        "请优先使用校园知识库工具取数：选课手册、课程信息、书籍笔记等泛检索用 "
        "search_campus_knowledge；某门课程的点评综合总结（评分分布/代表性观点）用 "
        "get_course_review_summary。引用时遵守「校园知识库引用规范」：标注来源，"
        "点评总结注明基于 N 条学生点评的自动摘要并提示信息时效，"
        "不得把学生点评个例当作官方政策陈述。拿到数据后再组织回答，不要凭空描述校园信息。"
        "检索结果不理想时，可用 source 参数按来源分源重试"
        "（sem_handbook/thubook/thucourse_course/thucourse_summary），"
        "或换同义关键词多次检索，不要一次检索无果就放弃。"
    ),
    "us_hk_query": (
        "【本问题路由提示】用户在问美股/港股/海外市场问题，数据源为新浪智研，"
        "港美股数据覆盖齐全，务必优先调用工具取数："
        "个股实时行情 get_stock_quote（market=hk/us，港股传裸 5 位代码如 00700、"
        "美股传裸 ticker 如 AAPL，均不要加 hk/us 前缀）、"
        "日K线 get_stock_kline、港股财报 get_hk_finance_report、"
        "港股资金流 get_hk_fund_flow、美股资金流 get_us_fund_flow、"
        "美股大盘涨跌分布 get_us_market_breadth、全球指数 get_global_indices、"
        "个股新闻 get_stock_news（market=hk/us）、重大事项 get_stock_major_events、"
        "公司公告/新闻检索 search_news。"
        "接口返回空或数据未覆盖时如实说明「数据未覆盖」，绝不凭训练记忆报行情数字；"
        "客观数据查询不得套用合规拒答话术。"
    ),
    "agent_analyze": (
        "【本问题路由提示】用户在问金融数据/分析型问题（个股技术分析/估值/财报/公告新闻/"
        "多标的对比/宏观数据/资金流向等），请按需组合工具取数：行情K线、技术指标、"
        "财务基本面、宏观数据、新闻舆情、多标的分别取数后对比。数据未覆盖的如实说明，"
        "不得凭记忆编造任何数字、日期与来源；客观数据查询不得套用合规拒答话术。"
    ),
}


def _campus_fallback_hit(message: str) -> bool:
    """general_chat 校园兜底探针：用原始消息检索校园知识库，
    top1 条目命中 ≥2 个不同关键词（含长词二字组扩词）则改道 campus_kb。

    关键词路由白名单无法穷尽自然问法（『紫荆公寓是几人间』『想出国读研读博』），
    漏判会让 LLM 在无工具路径凭空编造。此探针以检索命中做二次确认：
    金融/闲聊消息在校园库中几乎零命中，不会误判。
    任何异常返回 False——绝不抛出，闲聊路径零影响。
    """
    try:
        from agent import campus_kb
        kws = campus_kb._expand_relaxed_keywords(campus_kb._keywords(message))
        if not kws:
            return False
        results = campus_kb.search_kb(message, limit=1)
        if not results:
            return False
        top = results[0]
        text = ((top.get("title") or "") + (top.get("content") or "")).lower()
        hits = sum(1 for kw in kws if kw.lower() in text)
        return hits >= 2
    except Exception:
        return False


# general_chat 金融兜底探针词表：闲聊路由里混入的金融数据/分析信号。
# 命中即改道 Agent 工具循环（有工具、有合规、有风险提示），
# 防无工具闲聊路径凭训练记忆编造行情/宏观数字（QA 实锤：北向/沪指/成交额/
# CPI/PMI/LPR/财报类问题漏路由后硬编数字、虚构来源）。
_FINANCE_FALLBACK_KEYWORDS = [
    "北向", "南向", "沪指", "上证", "深成指", "创业板", "科创板", "成交额",
    "涨停", "跌停", "炸板", "连板", "龙虎榜",
    "CPI", "cpi", "PPI", "ppi", "PMI", "pmi", "GDP", "gdp", "LPR", "lpr",
    "非农", "美联储", "加息", "降息", "降准", "利率", "汇率",
    "财报", "年报", "季报", "业绩预告", "市盈率", "市净率", "估值",
    "换手率", "股息", "分红", "美股", "港股", "A股", "大盘", "股市",
    "纳斯达克", "恒生", "中概", "股价", "股票", "板块", "指数", "基金",
    "债券", "国债", "期货", "资金流向", "资金流", "主力",
    "财经", "金融", "宏观", "利好", "利空", "新闻", "资讯", "快讯",
    "行情", "走势", "涨跌", "市值", "营收", "净利润",
]
# 闲聊豁免：含这些词的消息不触发金融兜底（『讲个笑话』『涨知识了』类）
_CHITCHAT_EXEMPTIONS = ("笑话", "段子", "故事", "绕口令", "诗", "歌词", "游戏")


def _finance_fallback_hit(message: str) -> bool:
    """general_chat 金融兜底探针：消息含金融数据/分析信号词则改道 Agent 工具循环。

    只在 detect_intent 已判 general_chat（其他路由都没接住）后运行，
    词表全是金融信号词，命中即说明问题实质是金融数据/分析诉求。
    """
    msg = (message or "").strip()
    if len(msg) < 4:
        return False
    if any(w in msg for w in _CHITCHAT_EXEMPTIONS):
        return False
    return any(w in msg for w in _FINANCE_FALLBACK_KEYWORDS)


def _has_stock_entity(msg: str) -> bool:
    """公司级实体探测（分析型路由用）：个股关键词、6 位 A 股代码、
    5 位港股代码、常见美股 ticker、『公司』。
    行业词不算——『半导体新闻』要保持板块新闻既有路由不被抢走。"""
    if any(kw in msg for kw in _STOCK_ENTITY_KEYWORDS):
        return True
    if _CN_CODE_RE.search(msg) or _HK_CODE_RE.search(msg) or _US_TICKER_RE.search(msg):
        return True
    return "公司" in msg


# ── 风险提示统一兜底 ──
# QA 实锤：风险提示按路由模板挂载，stock_query/news_only/mcp_query 等确定性
# 拼接路径与 general_chat 下的金融内容全部漏挂。改为 dict 出口集中追加。
_DISCLAIMER_TEXT = (
    "\n\n风险提示：以上内容仅为客观数据整理与公开信息分析，不构成任何投资建议。"
    "市场有风险，投资需谨慎。"
)
# 内容级金融信号：general_chat 回答里出现这些词说明实际输出了金融内容
_FINANCE_CONTENT_SIGNALS = [
    "股", "基金", "涨", "跌", "板块", "指数", "行情", "估值", "资金",
    "期货", "债", "汇率", "利率", "CPI", "GDP", "营收", "净利润",
    "市值", "成交", "财经", "宏观", "大盘",
]


def _ensure_disclaimer(result, intent: str):
    """风险提示统一兜底（dict 出口，判重幂等）。

    - campus_kb：永不追加（校园回答与投资风险无关）；
    - general_chat：仅当回答内容实际含金融信号时追加（闲聊不打扰）；
    - 其他金融意图：一律追加（已含『风险提示』的跳过）；
    - 非 dict（流式生成器）：原样透传。
    """
    if not isinstance(result, dict):
        return result
    content = result.get("content") or ""
    if intent == "campus_kb" or "风险提示" in content:
        return result
    if intent != "general_chat" or any(w in content for w in _FINANCE_CONTENT_SIGNALS):
        result = dict(result)
        result["content"] = content + _DISCLAIMER_TEXT
    return result


def detect_intent(message: str) -> tuple[str, Optional[str]]:
    """
    意图识别。
    返回 (intent_type, sector_name_or_None)
    intent_type: "market_review" | "sector_deep_dive" | "social_sentiment"
                 | "persona" | "campus_kb" | "watchlist" | "news_only" | "stock_query"
                 | "futures_query" | "fund_query" | "mcp_query" | "general_chat"
    """
    msg = message.strip()

    # 第七波：自选股意图（保守规则：必须含『自选』二字才判；裸『自选/自选股』按列表处理）
    if "自选" in msg:
        wl_action_words = (
            _WATCHLIST_ADD_VERBS + _WATCHLIST_REMOVE_VERBS
            + _WATCHLIST_REVIEW_WORDS + ("我的", "列表")
        )
        if msg in ("自选", "自选股") or any(w in msg for w in wl_action_words):
            return ("watchlist", None)

    # 社媒舆情 / 投资人格 高优先级意图（插在自选股判定之后、行业提取与复盘
    # 关键词判定之前）。防误判红线：
    #   1. 不含任何上述关键词的消息零影响（直接跳过本块，旧判定顺序语义不变）；
    #   2. 含『复盘』二字的消息优先复盘，跳过本块（『今日复盘』必须仍走 market_review）；
    #   3. 含显式数据查询指令词的消息保持旧 mcp_query 路由（『今天涨停家数查询』不被抢）；
    #   4. 弱信号与 persona 关键词必须同时带市场语境（市场语境词或个股/行业实体）才判。
    # 校园强信号单独前置：不受数据查询指令词护栏约束（『成绩排名多少才能保研』
    # 含『排名』仍须判 campus_kb），但金融信号词共现时让位（『清华系持股』不被劫持）。
    has_finance_signal = any(w in msg for w in _FINANCE_STRONG_KEYWORDS)
    if "复盘" not in msg and not has_finance_signal:
        if any(kw in msg for kw in _CAMPUS_KB_KEYWORDS):
            return ("campus_kb", None)
    if "复盘" not in msg and not any(w in msg for w in _DATA_QUERY_COMMANDS):
        if any(kw in msg for kw in _SOCIAL_STRONG_KEYWORDS):
            return ("social_sentiment", None)
        has_market_context = _count_entities(msg) >= 1 or any(
            kw in msg for kw in _MARKET_CONTEXT_KEYWORDS
        )
        if has_market_context:
            if any(kw in msg for kw in _SOCIAL_WEAK_KEYWORDS):
                return ("social_sentiment", None)
            if any(kw in msg for kw in _PERSONA_KEYWORDS):
                return ("persona", None)

    # 海外市场（美股/港股）：优先于 A 股复盘/板块/个股模板判定，
    # 防『纳斯达克指数』被 A 股复盘模板劫持、『美联储加息』被 fund_query 碰撞、
    # 『00700』前导零代码无人识别。复盘诉求（『美股复盘』暂不支持）让位旧逻辑。
    if "复盘" not in msg:
        if (any(kw in msg for kw in _US_MARKET_KEYWORDS)
                or any(kw in msg for kw in _HK_MARKET_KEYWORDS)
                or _HK_CODE_RE.search(msg) or _US_TICKER_RE.search(msg)):
            return ("us_hk_query", msg)

    # 分析型个股/公司诉求（技术分析/估值/财报/公告新闻/多实体对比）：
    # 走 Agent 工具循环（有取数与分析能力），不被确定性行情卡短路
    # （模板卡只有价格快照；分析型问题走模板会零分析、无风险提示）。
    # 含『板块/行业』的消息让位板块路由（板块新闻/板块深挖既有路径不动）。
    if "复盘" not in msg and "板块" not in msg and "行业" not in msg and _has_stock_entity(msg):
        if _count_entities(msg) >= 2:
            return ("agent_analyze", msg)
        if any(kw in msg for kw in _ANALYTIC_STOCK_KEYWORDS):
            return ("agent_analyze", msg)

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

    # 宏观关键词护栏：CPI/GDP 等大写缩写会撞 stock_patterns 的 [A-Z] 模式
    # （『GDP增速怎么样』被误判个股查询——QA 实锤），个股判定前拦截
    if any(kw in msg for kw in _MACRO_KEYWORDS):
        return ("agent_analyze", msg)

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
    # 指代追问：那它的财报呢 / 那这家公司估值怎么样（上文个股查询）
    r"^(那|那么)(它|他|她|这家|这个|该公司|公司|该股)?(的)?"
    r"(财报|业绩|财务|估值|新闻|走势|股价|资金流|基本面|技术面|公告|舆情|分红)",
    # 跨市场/跨标的短追问：那港股呢 / 那茅台呢
    r"^那.{1,8}呢[？?]?$",
]

# 意图类型 → 中文描述（用于追问提示行）
_INTENT_LABELS = {
    "market_review": "全市场复盘",
    "sector_deep_dive": "板块深挖",
    "stock_query": "个股查询",
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
    if prev_intent not in ("market_review", "sector_deep_dive", "stock_query"):
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
    elif prev_intent == "stock_query" and is_followup:
        # 个股追问（『那它的财报怎么样』）：升级为 Agent 工具循环，
        # 上文个股语境经 history 透传，防跌回无工具闲聊路径凭记忆背数字
        return "agent_analyze", None, _INTENT_LABELS["stock_query"]

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
            logger.warning("tool_call 对象无 model_dump，使用属性兜底序列化: %r", tc, exc_info=True)
            tool_calls.append({
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            })
    return {"role": "assistant", "content": message.content, "tool_calls": tool_calls}


# ── Markdown 清理 ──

# 符号级删除表：流式 chunk 与确定性拼装文本共用（无状态、跨 chunk 安全）
_MD_SYMBOLS_TRANSLATE = str.maketrans("", "", "*#")


def _strip_md_symbols(text: str) -> str:
    """轻量符号级清洗：删除文本中所有 * 与 # 字符，其余一律不动。

    无状态、逐 chunk 安全：因为目标是「* 与 # 零到达用户」，跨 chunk 的
    ** 配对无需识别——无论 **加粗** 完整到达还是拆成两个 chunk，每个 *
    都被删除，拼接结果与整段清洗完全一致。用于流式 chunk 清洗与
    确定性拼装文本（新闻清单/个股/期货/基金/自选股输出）的兜底清洗。
    """
    if not text:
        return text or ""
    return text.translate(_MD_SYMBOLS_TRANSLATE)


# ── [UNSOURCED] 泄漏双保险（复盘出口专属，_clean_markdown 被多处复用不动它）──

_UNSOURCED_TOKEN = "[UNSOURCED]"
_UNSOURCED_REPLACEMENT = "（数据未覆盖）"


def _replace_unsourced(text: str) -> str:
    """把残留的 [UNSOURCED] 标记替换为「（数据未覆盖）」。无标记时原样返回。"""
    if not text:
        return text or ""
    return text.replace(_UNSOURCED_TOKEN, _UNSOURCED_REPLACEMENT)


async def _replace_unsourced_stream(agen):
    """流式 chunk 跨边界安全替换 [UNSOURCED]：每段与上一段保留的尾部拼接后替换，
    并保留末尾 (len(token)-1) 个字符以防标记被拆到两个 chunk；流结束时冲刷尾部。
    仅包在全市场复盘流式出口，其他流式路径不受影响。
    """
    hold = len(_UNSOURCED_TOKEN) - 1
    tail = ""
    async for piece in agen:
        buf = _replace_unsourced(tail + piece)
        if len(buf) > hold:
            emit, tail = buf[:-hold], buf[-hold:]
        else:
            emit, tail = "", buf
        if emit:
            yield emit
    if tail:
        yield _replace_unsourced(tail)


# 元推理开头清洗：LLM 偶发把「数据已经够了」类思考句写进正文开头，
# 校园路径出口做确定性剥除（只剥开头连续元推理句，不动正文）
_META_OPENING_RE = re.compile(
    r"^\s*(?:"
    r"数据已经(?:比较)?(?:充分|够)?了?[。，,]?"
    r"|现有数据已经足够回答(?:用户的)?问题了?[。，,]?"
    r"|现在数据(?:已经)?足够回答(?:用户的)?问题了?[。，,]?"
    r"|信息已经(?:足够|够)(?:回答(?:用户的)?问题)?了?[。，,]?"
    r"|数据已经足够回答用户的问题了?[。，,]?"
    r"|下面整理回答[。，,]?"
    r"|我来整理回答[。，,]?"
    r"|(?:现在)?数据已经够了[。，,]?(?:可以给出结论[。，,]?)?"
    r"|以下(?:是|为)?修正后的(?:完整)?(?:正文|回答|内容)?[:：]?"
    r"|修正后的(?:完整)?正文[:：]?"
    r"|根据(?:以上|上述)(?:审查|审核)(?:意见|建议)?[，,]?"
    r")+"
)


def _strip_meta_openings(text: str) -> str:
    """剥除正文开头的连续元推理句（校园问答出口卫生）。"""
    return _META_OPENING_RE.sub("", text or "", count=1)


# 工具函数名软泄漏：LLM 偶发把 snake_case 工具名写进自然语言正文
# （『通过 get_market_sentiment 获取…』）。确定性剥除标记本身，保留句子其余部分。
_FUNC_NAME_RE = re.compile(
    r"\b(?:get|fetch|search|analyze|collect|format|compact|execute)_[a-z0-9_]+\b"
)
# 内部结构化字段泄漏：persona 框架等场景的 signal:/confidence:/disclaimer: 字段
# 被 LLM 原文誊进正文（QA 实锤），连字段名带取值一并剥除
_INTERNAL_FIELD_RE = re.compile(
    r"\b(?:signal|confidence|disclaimer|output_schema|checklist)\s*[:：]\s*[^\n，。；]*",
    re.IGNORECASE,
)


def _strip_function_names(text: str) -> str:
    """剥除正文中泄漏的工具函数名与内部结构化字段标记，保留句子其余文字。"""
    if not text:
        return text or ""
    return _INTERNAL_FIELD_RE.sub("", _FUNC_NAME_RE.sub("", text))


# DeepSeek DSML 文本工具调用：模型偶发把工具调用以纯文本标记输出
# （<｜｜DSML｜｜tool_calls>…</｜｜DSML｜｜tool_calls>，｜ 为全角竖线 U+FF5C），
# 而非结构化 message.tool_calls。这里做确定性解析与剥除，杜绝原始标记泄漏给用户。
_DSML_TOOL_CALLS_RE = re.compile(
    r"<[｜|]{2}DSML[｜|]{2}tool_calls>.*?(?:</[｜|]{2}DSML[｜|]{2}tool_calls>|\Z)",
    re.DOTALL,
)
_DSML_INVOKE_RE = re.compile(
    r"<[｜|]{2}DSML[｜|]{2}invoke\s+name=\"([^\"]+)\"[^>]*>"
    r"(.*?)(?:</[｜|]{2}DSML[｜|]{2}invoke>|\Z)",
    re.DOTALL,
)
_DSML_PARAM_RE = re.compile(
    r"<[｜|]{2}DSML[｜|]{2}parameter\s+name=\"([^\"]+)\""
    r"(?:\s+string=\"(true|false)\")?[^>]*>"
    r"(.*?)(?:</[｜|]{2}DSML[｜|]{2}parameter>|\Z)",
    re.DOTALL,
)


def _strip_dsml(text: str) -> str:
    """剥除 DSML 工具调用标记：完整块整体删除，未闭合/孤立的标记行逐行剔除。"""
    if not text or "DSML" not in text:
        return text or ""
    text = _DSML_TOOL_CALLS_RE.sub("", text)
    lines = [
        ln for ln in text.split("\n")
        if "DSML｜｜" not in ln and "DSML||" not in ln
    ]
    return "\n".join(lines).strip()


def _parse_dsml_tool_calls(text: str):
    """解析 DSML 纯文本工具调用块。

    返回 (calls, remainder)：calls 为 [(工具名, 参数dict)]；remainder 为剔除
    DSML 标记后的剩余文本。无 DSML 内容时 calls 为空、remainder 为原文。
    参数类型按 string 属性还原：string="true" 保留字符串，否则按 JSON 解析
    （数字/布尔），解析失败回退为字符串。
    """
    if not text or "DSML" not in text:
        return [], text or ""
    calls = []
    for inv in _DSML_INVOKE_RE.finditer(text):
        name = inv.group(1).strip()
        if not name:
            continue
        args = {}
        for pm in _DSML_PARAM_RE.finditer(inv.group(2)):
            pname = pm.group(1).strip()
            is_string = (pm.group(2) or "true").strip().lower() != "false"
            raw = pm.group(3).strip()
            if is_string:
                args[pname] = raw
            else:
                try:
                    args[pname] = json.loads(raw)
                except Exception:
                    args[pname] = raw
        calls.append((name, args))
    return calls, _strip_dsml(text)


def _clean_markdown(text: str) -> str:
    """后处理：强制清除所有 markdown 格式。

    结构性清洗（管道表格/加粗/斜体/标题/列表符号/管道符）之后，
    末尾用 _strip_md_symbols 兜底删除一切残留的 * 与 #（含行内 *斜体*
    未配对残留、孤立 #），保证任何 markdown 符号都到不了用户。
    """
    if not text:
        return text or ""

    # 0. 水平分隔线（---/———）整行删除
    text = re.sub(r"^\s*-{3,}\s*$", "", text, flags=re.MULTILINE)

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

    # 2. **加粗** / __加粗__ / *斜体* → 去掉星号保正文
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"\*([^*\n]+)\*", r"\1", text)

    # 3. # 标题 → 去掉行首标记（1-6 个 #）
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

    # 4. * 列表 → - 列表
    text = re.sub(r"^[\*\+]\s+", "- ", text, flags=re.MULTILINE)

    # 5. 残留的单个 | 替换为空格
    text = text.replace(" | ", "  ")

    # 6. 兜底：删除一切残留的 * 与 #（未配对星号、孤立井号、行内残余标记）
    text = _strip_md_symbols(text)

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


# 摘要展示上限（字）：展示抓取层传入的 content/summary/brief 时使用
_NEWS_SUMMARY_CAP = 200

# 句末标点：句子边界截断的候选边界（换行在快讯正文中也是天然分句）
_SENTENCE_ENDINGS = "。！？!?；;\n"


def _truncate_at_sentence(text: str, limit: int = _NEWS_SUMMARY_CAP) -> str:
    """长文本防御性截断：展示层绝不拦腰截断句子。

    优先在 limit 内最后一个句末标点处截断并加省略号；实在找不到句末
    标点时按 limit 硬截，同样补省略号，明确告知用户「后面还有内容」。
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    boundary = -1
    for i, ch in enumerate(text):
        if i >= limit:
            break
        if ch in _SENTENCE_ENDINGS:
            boundary = i + 1
    if boundary > 0:
        return text[:boundary].rstrip() + "……"
    return text[:limit].rstrip() + "……"


# ── 通用 MCP 查询（_generic_mcp）──

# system 提示词：纯文本无 Markdown；候选代码精确匹配；错误/空数据不反复重试
_GENERIC_MCP_SYSTEM_PROMPT = (
    "你是数据查询助手。用纯文本回答，禁止Markdown表格（禁止|和-组成的表格线），"
    "禁止使用#和*号。数据用缩进对齐或列表呈现。\n"
    "搜索类工具返回多个候选代码时，必须选用与用户公司名完全匹配的那一条，"
    "不要用名称近似的其他公司凑合。\n"
    "调用财报类工具时严格按参数枚举值填参，不要编造枚举外的取值。"
    "查询A股财务指标的推荐链路：先用 cnFinanceReportDateList 拿报告期，"
    "再用 cnFinanceReportsFull（source 填 gjzb/lrb/fzb/llb/zxzb 等枚举值）取数。\n"
    "报告期选择：cnFinanceReportDateList 返回的报告期按新到旧排列。用户问近期/最新"
    "财务指标时，必须覆盖列表最前面的最新年报和最新季报（至少各取一期），"
    "不要只取早期年度；回答时优先呈现最新年报与最新季报的数据，已取各期都要讲到。\n"
    "工具返回错误或空数据时，不要反复重试同一个工具；最多换参数重试一次，"
    "仍拿不到就基于已经拿到的数据回答，或如实说明该项数据暂缺。\n"
    "诚实约束：只有实际调用过且工具确实返回错误/占位数据的报告期，才可以说"
    "该期数据暂缺；没有查询过的报告期，不得声称其无数据，直接不提即可。\n"
    "查询港股财务指标推荐用 hk_finance_all（symbol 填港股代码如 06715，"
    "frType 填 gjzb/lrb/fzb/llb/yh，currencyType 2=港币，yearNum 填年数）；"
    "不要用 A 股专用工具（cnCompanyBasicInfo、cnFinanceReportsFull 等）查港股代码。"
    "财报数据里 item_display、item_display_type 等是界面展示样式标记（如灰底），"
    "不代表数据缺失；数值看 item_value，同比涨跌看 item_tongbi。\n"
    "数字保真约束：回答里的每个数值必须逐项抄自工具返回的 item_value/"
    "item_tongbi，禁止估算、约算或编造；工具结果里看不到的指标就不要写，"
    "宁少勿假。"
)

def _generic_mcp_system_prompt() -> str:
    """通用 MCP 查询 system 提示词 + 当天日期注入。

    注入日期是为了防止模型按训练记忆误判报告期是否已发布
    （例如把已披露的最新年报当未来数据跳过，转而去取早期年度）。
    """
    return (
        _GENERIC_MCP_SYSTEM_PROMPT
        + f"\n今天是{datetime.now().strftime('%Y年%m月%d日')}，以此判断哪些报告期已经发布、可以查询。"
    )


# 单次 MCP 工具结果喂回模型的字符上限。财报类载荷（hk_finance_all 等）
# 压缩后仍需约 10K 字符才能装下 3 年 × 57 项指标；3000 的历史取值曾把
# 真实数值截掉，导致模型编造财务数字。12000 ≈ 6K tokens，5 轮循环内可控。
_MCP_TOOL_FEED_MAX_CHARS = 12000
# all_results 单次结果存档上限（供循环跑满后的最终综合调用使用）
_MCP_SYNTH_RESULT_MAX_CHARS = 4000


# 最终综合也失败时的优雅降级提示（任何情况下都不把原始 JSON 给用户）
_MCP_DATA_UNAVAILABLE = (
    "抱歉，相关数据暂未获取成功（数据源返回异常或为空）。"
    "请稍后重试，或换个更具体的问法（如指明股票代码、市场与具体指标）。"
)


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
        # general_chat 金融兜底（先于校园探针）：金融数据/分析诉求改道 Agent 工具循环。
        # 校园库含证券投资类书籍笔记，『推荐几只股票』『沪指走势』会在库中命中
        # ≥2 关键词被校园探针误判（QA 实锤），金融探针必须先跑
        if intent == "general_chat" and _finance_fallback_hit(user_message):
            intent = "agent_analyze"
        # general_chat 校园兜底：关键词白名单漏判的校园问题用知识库检索二次确认，
        # 命中即改道 campus_kb（走工具链），避免无工具闲聊路径凭空编造；
        # 金融强信号词共现时让位（『清华系持股』类不被校园路由劫持）
        if (intent == "general_chat"
                and not any(w in user_message for w in _FINANCE_STRONG_KEYWORDS)
                and _campus_fallback_hit(user_message)):
            intent = "campus_kb"
        # 继承意图的追问：在数据路径的 user prompt 开头加一行上下文说明
        context_note = (
            f"用户此前在询问{inherited_from}，本条为追问，请结合该语境组织回答。"
            if inherited_from else None
        )

        # 第二波：复杂分析（跨实体比较/多实体问题）优先升级为 Agent 工具循环。
        # 裸行业名命中板块意图时，若消息实为比较/多实体分析（如『比较白酒和半导体』），
        # 也让位给 Agent 循环；规则保守，拿不准走原路径
        if intent in ("general_chat", "sector_deep_dive") and _should_use_agent_query(user_message):
            return _ensure_disclaimer(
                await self._agent_query(user_message, stream, history=history), intent
            )

        # 社媒舆情 / 投资人格高优先级意图：直接走 Agent 工具循环，并透传路由提示
        # hint（追加到 AGENT_QUERY_PROMPT 系统消息尾部）。工具注册表未就绪或循环
        # 异常时 _agent_query 内部仍降级 _chat，兜底链不变。
        if intent in _AGENT_ROUTE_HINTS:
            return _ensure_disclaimer(
                await self._agent_query(
                    user_message, stream, history=history,
                    hint=_AGENT_ROUTE_HINTS[intent],
                    disclaimer=(intent != "campus_kb"),  # 校园回答不附金融风险提示
                ),
                intent,
            )

        if intent == "general_chat":
            return _ensure_disclaimer(
                await self._chat(user_message, stream, history=history), intent
            )
        elif intent == "market_review":
            return _ensure_disclaimer(
                await self._market_review(stream, context_note=context_note), intent
            )
        elif intent == "watchlist":
            result = await self._watchlist(user_message, stream)
        elif intent == "news_only":
            return await self._news_only(sector, stream, message=user_message)
        elif intent == "stock_query":
            result = await self._stock_query(user_message, False)
        elif intent == "futures_query":
            result = await self._futures_query(user_message, False)
        elif intent == "fund_query" or intent == "mcp_query":
            result = await self._fund_query(user_message, False)
        else:
            return _ensure_disclaimer(
                await self._sector_deep_dive(sector, stream, context_note=context_note), intent
            )

        # 简单查询返回dict → 流式模式包装为生成器
        if stream and isinstance(result, dict):
            result = _ensure_disclaimer(result, intent)

            async def _wrap():
                text = result.get("content", "")
                for i in range(0, len(text), 80):
                    yield text[i:i+80]
                    await asyncio.sleep(0.01)
            return _wrap()
        return _ensure_disclaimer(result, intent)

    async def _chat(self, message: str, stream: bool, history: list = None):
        """通用对话。带对话历史，让闲聊也有上下文。闲聊不附金融风险提示。"""
        system = get_system_prompt("general_chat")
        return await self._call_llm(system, message, stream, history=history, disclaimer=False)

    # ── 第七波：自选股管理（加 / 删 / 列表 / 复盘）──

    async def _watchlist(self, message: str, stream: bool):
        """自选股意图入口：按动作词分发到加/删/列表/复盘。

        模块未就绪或任一环节异常都降级为文案提示，绝不向上抛出。
        """
        if watchlist is None:
            return {"role": "assistant", "content": "自选股功能暂不可用（模块未就绪），请稍后再试。"}
        msg = (message or "").strip()
        has_add = any(v in msg for v in _WATCHLIST_ADD_VERBS)
        has_remove = any(v in msg for v in _WATCHLIST_REMOVE_VERBS)
        has_review = any(w in msg for w in _WATCHLIST_REVIEW_WORDS)

        if has_add and not has_remove:
            return self._watchlist_add(msg)
        if has_remove and not has_add:
            return self._watchlist_remove(msg)
        if has_review:
            return await self._watchlist_review(stream)
        return self._watchlist_list()

    @staticmethod
    def _clean_watchlist_text(text: str) -> str:
        """剥掉『自选股/自选』字样与首尾标点空白，得到候选股票名/代码。"""
        t = (text or "").replace("自选股", " ").replace("自选", " ")
        return t.strip(" \t　，。,.、：:！!？?；;")

    @classmethod
    def _extract_watchlist_keyword(cls, msg: str, verbs) -> str:
        """提取动作词后的股票名；为空时兜底取动作词前的文本（如『把茅台加入自选股』）。"""
        after, before = "", ""
        for v in verbs:
            if v in msg:
                after, before = msg.split(v, 1)[1], msg.split(v, 1)[0]
                break
        keyword = cls._clean_watchlist_text(after)
        if keyword:
            return keyword
        keyword = cls._clean_watchlist_text(before)
        for prefix in ("帮我", "把", "将", "请"):
            if keyword.startswith(prefix):
                keyword = keyword[len(prefix):]
        return keyword

    def _watchlist_add(self, msg: str) -> dict:
        """添加自选股：resolver 用模块缺省实现（内部触网搜索），回显 (code, name) 供用户确认。"""
        keyword = self._extract_watchlist_keyword(msg, _WATCHLIST_ADD_VERBS)
        if not keyword:
            return {"role": "assistant", "content": _WATCHLIST_USAGE}
        try:
            ok, text = watchlist.add_stock(keyword)
        except Exception as e:
            logger.warning("自选股添加调用异常（fail-safe）: %s", e, exc_info=True)
            return {"role": "assistant", "content": "自选股添加失败，请稍后再试。"}
        content = text
        if ok:
            content += f"\n请核对上方股票名称与代码；如不正确，回复『删自选 {keyword}』后重新添加。"
        return {"role": "assistant", "content": _clean_markdown(content)}

    def _watchlist_remove(self, msg: str) -> dict:
        """删除自选股：回显删除结果。"""
        keyword = self._extract_watchlist_keyword(msg, _WATCHLIST_REMOVE_VERBS)
        if not keyword:
            return {"role": "assistant", "content": _WATCHLIST_USAGE}
        try:
            ok, text = watchlist.remove_stock(keyword)
        except Exception as e:
            logger.warning("自选股删除调用异常（fail-safe）: %s", e, exc_info=True)
            return {"role": "assistant", "content": "自选股删除失败，请稍后再试。"}
        return {"role": "assistant", "content": _clean_markdown(text)}

    def _watchlist_list(self) -> dict:
        """自选股清单：纯文本直接输出，无需 LLM。"""
        try:
            stocks = watchlist.list_stocks()
        except Exception as e:
            logger.warning("自选股清单获取异常（fail-safe）: %s", e, exc_info=True)
            stocks = []
        if not stocks:
            return {"role": "assistant", "content": _WATCHLIST_EMPTY_GUIDE}
        lines = [f"我的自选股（共 {len(stocks)} 只）："]
        for i, s in enumerate(stocks, 1):
            name = s.get("name", "?") if isinstance(s, dict) else "?"
            code = s.get("code", "?") if isinstance(s, dict) else "?"
            lines.append(f"{i}. {name}（{code}）")
        lines.append("")
        lines.append("回复『复盘我的自选股』查看逐只复盘；『加自选 名称』/『删自选 名称』管理清单。")
        return {"role": "assistant", "content": _clean_markdown("\n".join(lines))}

    def _watchlist_system_prompt(self) -> str:
        """自选股复盘系统提示词：优先 get_system_prompt("watchlist")，取不到/为空用内联兜底。"""
        try:
            prompt = get_system_prompt("watchlist")
            if isinstance(prompt, str) and prompt.strip():
                return prompt
            logger.warning("get_system_prompt(\"watchlist\") 返回空，使用内联默认提示词")
        except Exception as e:
            logger.warning(
                "get_system_prompt(\"watchlist\") 获取失败，使用内联默认提示词: %s",
                e, exc_info=True,
            )
        return _DEFAULT_WATCHLIST_SYSTEM_PROMPT

    async def _fetch_watchlist_quotes(self, stocks: list) -> str:
        """逐只取实时行情（线程池并行，单只失败降级为『数据暂不可用』，绝不抛出）。"""
        try:
            from .data_fetcher import fetch_stock_quote
        except ImportError as e:
            logger.warning("fetch_stock_quote 未就绪，自选股行情整体降级: %s", e)
            fetch_stock_quote = None
        loop = asyncio.get_running_loop()

        async def _one(stock) -> str:
            name = stock.get("name", "?") if isinstance(stock, dict) else "?"
            code = stock.get("code", "") if isinstance(stock, dict) else ""
            market = (stock.get("market") or "cn") if isinstance(stock, dict) else "cn"
            if fetch_stock_quote is None or not code:
                return f"{name}（{code or '?'}）：行情数据暂不可用"
            try:
                quote = await loop.run_in_executor(None, fetch_stock_quote, market, code)
            except Exception as e:
                logger.warning(
                    "自选股行情获取失败（%s %s，单只降级）: %s", market, code, e, exc_info=True
                )
                quote = None
            if not isinstance(quote, dict) or not quote:
                return f"{name}（{code}）：行情数据暂不可用"
            return (
                f"{name}（{code}）：现价 {quote.get('price', '?')}，"
                f"涨跌幅 {quote.get('pct', '?')}%，"
                f"最高 {quote.get('high', '?')}，最低 {quote.get('low', '?')}"
            )

        lines = await asyncio.gather(*(_one(s) for s in stocks))
        return "\n".join(lines)

    async def _watchlist_review(self, stream: bool):
        """自选股复盘：format_watchlist_block + 行情块走 LLM；空清单返回引导文案。"""
        try:
            stocks = watchlist.list_stocks()
        except Exception as e:
            logger.warning("自选股清单获取异常（fail-safe）: %s", e, exc_info=True)
            stocks = []
        if not stocks:
            return {"role": "assistant", "content": _WATCHLIST_EMPTY_GUIDE}

        quotes_text = await self._fetch_watchlist_quotes(stocks)

        try:
            block = watchlist.format_watchlist_block()
        except Exception as e:
            logger.warning("format_watchlist_block 异常（fail-safe，手工拼装兜底）: %s", e, exc_info=True)
            block = None
        if not block:
            lines = [f"【用户自选股】共 {len(stocks)} 只："]
            for i, s in enumerate(stocks, 1):
                name = s.get("name", "?") if isinstance(s, dict) else "?"
                code = s.get("code", "?") if isinstance(s, dict) else "?"
                lines.append(f"{i}. {name}（{code}）")
            block = "\n".join(lines)

        user_prompt = (
            f"{block}\n\n【自选股实时行情】\n{quotes_text}\n\n"
            "请逐只复盘以上自选股：当日表现、关键价位、一句话简评；最后给一段整体小结。"
            "仅使用上方数据，数据缺失处标注数据暂缺，禁止编造。"
        )
        system = self._watchlist_system_prompt()
        return await self._call_llm(system, user_prompt, stream)

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
        # 第七波：以史为鉴注入（fail-safe；模块自带头部，不加引导语）
        history_note = _history_note_block(sector=None, mode="market_review")
        user_prompt = f"""{note_line}交易日：{date_display} {weekday}{date_note}

{market_data}

生成A股市场复盘。不列出新闻（如需新闻请说\"今天市场新闻\"）。31行业全部列出。数据缺失写「数据未覆盖」。{history_note}"""

        # 第四波：存档接入——流式走 _stream_response 结束回调，非流式在最终 content 产出后落档
        if stream:
            gen = await self._call_llm(
                system, user_prompt, stream,
                archive_callback=self._stream_archive_callback(
                    "market_review", None, user_prompt, date_str
                ),
            )
            # 双保险：复盘出口专属的 [UNSOURCED] 残留清洗（流式跨 chunk 安全）
            return _replace_unsourced_stream(gen)
        result = await self._call_llm(system, user_prompt, stream)
        if isinstance(result, dict):
            # 双保险：复盘出口专属的 [UNSOURCED] 残留清洗（非流式）
            result["content"] = _replace_unsourced(result.get("content", ""))
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
        # 第七波：行业知识库注入（追加在【五、新闻与景气背景】之后；fail-safe）
        kb_block = _industry_kb_block(sector)
        # 第七波：以史为鉴注入（user_prompt 末尾；fail-safe；模块自带头部，不加引导语）
        history_note = _history_note_block(sector=sector, mode="sector_deep_dive")
        user_prompt = f"""{note_line}交易日：{date_display} {weekday} | 行业：{sector}

【一、行情与趋势数据】
{market_data}

{valuation_block}

{moneyflow_block}

{earnings_block}

【五、新闻与景气背景】
上方"行情与趋势数据"中的东方财富实时快讯与新浪历史新闻已按{sector}行业过滤，请作为催化/风险判断的素材引用，不要原样罗列。{kb_block}

深度分析{sector}板块。输出标题需包含日期{date_display}。新闻条目仅作为第五维催化与风险的引用素材，不要单独罗列新闻清单。数据缺失处标注数据暂缺，禁止编造。{history_note}"""

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
        """新闻专属模式：五源新闻池 + 行业关键词过滤，按实际覆盖尽量多展示。

        - 数据层优先用 fetch_news_pool（五源聚合、跨源去重、时间已修复），
          未就绪或调用失败时降级到旧三源直采逻辑。
        - 行业过滤除标题外，同时匹配 time/content/summary 字段（有的话）。
        - 展示策略：按天分组、按时间倒序；全市场查询每天上限30条（超限按重要性
          评分截断），行业查询每天全量展示。条目格式为「[时间] 标题」，不再带
          【来源】标签（省出字符给新闻本身，来源归属由头部「来源：xxxN条」统计行
          统一承载）。标题一律完整输出，绝不拦腰截断；抓取层把正文拦腰截断当标题
          时（title 是 content/summary/brief 的裸前缀），用摘要字段按句子边界截断
          （上限 _NEWS_SUMMARY_CAP 字）修复展示。
        - 头部覆盖描述按实际数据生成：单日写「当日」/实际日期，多日写起止日期，
          不照抄「48小时」模板；来源统计只列实际有贡献的源。
        - 解读段：板块查询（sector 非 None）默认在确定性清单后追加 LLM 解读
          （news_analysis 系统提示词），清单本体绝不经过 LLM 改写；全市场查询
          （sector 为 None）保持触发词逻辑——message 含解读类触发词
          （解读/分析/影响/怎么看/意味着什么/说明什么）时走纯 LLM 分析，
          否则非流式直接透传原文，不经过 LLM 改写。
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
                    # 抓取层可能带摘要字段（cls 的 brief、部分源的 content/summary），
                    # 展示循环用它防御性修复「被拦腰截断的标题」
                    "summary": (item.get("summary", "") or item.get("content", "")
                                or item.get("brief", "") or ""),
                })
                source_counts[src] = source_counts.get(src, 0) + 1

        total_raw = sum(len(v) for v in by_date.values())

        # 头部覆盖描述按实际数据生成，不照抄「48小时」模板：
        # 单日且就是交易日写「当日」，单日非交易日给出实际日期，多日给出起止日期
        days_sorted = sorted(by_date.keys())
        trade_day = trade_date.strftime("%Y-%m-%d")
        if not days_sorted:
            coverage_desc = "当日"
        elif len(days_sorted) == 1:
            coverage_desc = "当日" if days_sorted[0] == trade_day else f"{days_sorted[0]}单日"
        else:
            coverage_desc = f"{days_sorted[0]}至{days_sorted[-1]}"

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
                    text = it["title"]
                    summary = (it.get("summary") or "").strip()
                    # 防御：抓取层把正文拦腰截断当标题（如 cls brief[:80]、智研
                    # content[:80]）时，title 是摘要的裸前缀。改用摘要字段按句子
                    # 边界截断（上限 _NEWS_SUMMARY_CAP 字），展示完整句子而非片段。
                    # 摘要缺失或并非标题前缀时，标题原样完整输出，绝不拦腰截断。
                    if summary and len(summary) > len(text) and summary.startswith(text):
                        text = _truncate_at_sentence(summary)
                    # 展示行格式：[时间] 标题。条目不再带【来源】标签（省出字符给
                    # 新闻本身）；来源归属由头部「来源：xxxN条」统计行统一承载。
                    block += f"[{_fmt_news_time(it['time'], fallback=d)}] {text}\n"
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
            news_text = f"{label}新闻汇总 — {date_display} {weekday}（{coverage_desc}，共{display_total}条）\n"
            meta_parts = [p for p in (span, f"来源：{src_stat}" if src_stat else "") if p]
            if truncated:
                meta_parts.append("单日超30条已按重要性截断")
            if meta_parts:
                news_text += " | ".join(meta_parts) + "\n"
            news_text += "".join(blocks)

        # 输出卫生兜底：新闻标题为外部文本，可能混入 # 或 * 字符。清单进入
        # 解读 prompt 与最终展示前统一做符号级清洗（只删 #/*，不动结构、
        # 不动头部「来源：xxxN条」统计行与其余字符）。
        news_text = _strip_md_symbols(news_text)

        # ── 4. 解读段：板块查询默认附带 LLM 解读；全市场保持触发词逻辑 ──
        triggered = bool(message) and any(w in message for w in _NEWS_ANALYSIS_TRIGGERS)
        if total_raw > 0 and (sector or triggered):
            system = get_system_prompt("news_analysis")
            user_prompt = (
                f"{news_text}\n\n以上是{label}{coverage_desc}新闻清单。请按系统提示的框架输出解读，"
                "只分析清单内的新闻条目，清单未覆盖的写「本期新闻未覆盖」。"
            )
            if sector:
                # 板块：确定性清单原样保留（绝不经过 LLM 改写），LLM 解读段追加在后。
                # 非流式 content = 清单 + 解读；流式先输出清单，再流式输出解读。
                # 此前板块查询无触发词时不依赖 LLM，解读调用失败必须降级为只返回
                # 清单，不能让 LLM 故障拖垮整个板块新闻查询。
                analysis_header = f"\n\n—— {label}新闻解读 ——\n\n"
                if stream:
                    try:
                        analysis_stream = await self._call_llm(system, user_prompt, True)
                    except Exception as e:
                        logger.warning("板块新闻解读 LLM 调用失败，降级为只输出清单: %s", e, exc_info=True)
                        analysis_stream = None

                    async def _list_then_analysis():
                        yield news_text + analysis_header
                        if analysis_stream is None:
                            yield "（解读暂时不可用，以上为确定性新闻清单。）"
                            return
                        try:
                            async for chunk in analysis_stream:
                                yield chunk
                        except Exception as e:
                            logger.warning("板块新闻解读流中断，保留已输出清单: %s", e, exc_info=True)
                            yield "\n（解读中断，以上为确定性新闻清单。）"

                    return _list_then_analysis()
                try:
                    result = await self._call_llm(system, user_prompt, False)
                    analysis = result.get("content", "") if isinstance(result, dict) else ""
                except Exception as e:
                    logger.warning("板块新闻解读 LLM 调用失败，降级为只输出清单: %s", e, exc_info=True)
                    analysis = ""
                if not analysis:
                    return {"role": "assistant", "content": news_text}
                return {"role": "assistant", "content": news_text + analysis_header + analysis}
            # 全市场 + 触发词：维持原有纯 LLM 分析行为
            return await self._call_llm(system, user_prompt, stream)

        # 透传模式：非流式直接返回原文，不经过 LLM 改写
        system = "你是财经新闻编辑。将以下新闻汇总原样输出。不删减、不分析、不改格式。"
        user_prompt = f"{news_text}\n\n以上{label}{coverage_desc}新闻汇总。原样输出。"

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
            # K线拼接防御：date/trade_date 双键兜底，取不到日期不输出裸冒号
            kline_parts = []
            for k in kline[:5]:
                d = str(k.get("date") or k.get("trade_date") or "")[-5:]
                c = k.get("close", "?")
                kline_parts.append(f"{d}:{c}" if d else f"收盘{c}")
            kline_str = ", ".join(kline_parts) if kline_parts else "数据未覆盖"
            news_items = [n.get("title", "")[:40] for n in news[:3] if n.get("title")]
            news_str = "；".join(news_items) if news_items else "数据未覆盖"
            if not quote and not kline and not news:
                # 取数全失败：如实声明，不输出空白模板
                info = f"{name}({code})\n行情数据暂不可用（接口未返回数据），请稍后再试。"
            else:
                quote_str = (
                    f"价格{quote.get('price','?')} 涨跌{quote.get('pct','?')}%"
                    if quote else "数据未覆盖"
                )
                ohl_str = (
                    f"开盘{quote.get('open','?')} 最高{quote.get('high','?')} 最低{quote.get('low','?')}"
                    if quote else ""
                )
                info = f"{name}({code})\n行情：{quote_str}\n{ohl_str}\n近5日K线：{kline_str}\n相关新闻：{news_str}\n"
        except Exception as e:
            logger.warning("个股数据获取失败(%s %s): %s", market, code, e, exc_info=True)
            info = f"{name}({code})\n数据暂不可用，请稍后再试。"
        return {"role": "assistant", "content": _clean_markdown(info)}

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
        return {"role": "assistant", "content": _clean_markdown(info)}

    async def _generic_mcp(self, message: str, stream: bool, max_rounds: int = 5):
        """通用MCP查询——使用DeepSeek原生function calling，100%可靠。

        输出卫生（任何路径都不把工具原始 JSON dump 给用户）：
        - 模型在轮次内给出纯文本回答：_clean_markdown 清洗后返回；
        - 循环跑满 max_rounds 仍要调工具：把所有已收集工具结果拼入上下文，
          再发起一次无 tools 的最终综合调用（_mcp_final_synthesis），让 LLM
          用人话总结；综合也失败时返回优雅的「数据暂未获取成功」类提示；
        - 工具结果喂回模型前经 _mcp_result_for_prompt 压缩：错误返回压成
          简短错误说明（工程师B 的 is_mcp_error/mcp_error_brief，未就绪时
          退回现状原文），正常返回经 compact_mcp_result 压缩（未就绪时退回
          json.dumps 截断 3000），防止大段原文挤爆上下文并诱导反复重试。
        """
        from agent.data_fetcher import get_mcp_tools, _mcp_call
        tools = get_mcp_tools()
        if not tools:
            return {"role": "assistant", "content": "MCP服务暂不可用"}

        # 转成OpenAI function calling格式
        ds_tools = []
        for t in tools[:80]:
            schema = t.get("schema")
            if isinstance(schema, dict) and schema.get("properties"):
                # 新缓存格式：真实 inputSchema（含参数描述/枚举/required）
                parameters = {"type": "object", "properties": schema["properties"]}
                if schema.get("required"):
                    parameters["required"] = schema["required"]
            else:
                # 旧缓存格式降级：只有参数名列表，描述退回参数名本身
                props = {}
                for p in t.get("params", [])[:5]:
                    props[p] = {"type": "string", "description": p}
                parameters = {"type": "object", "properties": props}
            ds_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("desc", "")[:200],
                    "parameters": parameters
                }
            })

        system_prompt = _generic_mcp_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
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
                # 无需工具，直接回答（清洗 markdown 符号后返回）
                answer = _clean_markdown(choice.message.content or "")
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
                    data_str = self._mcp_result_for_prompt(fn.name, data)
                    if data_str is None:
                        return {"role": "assistant", "content": "抱歉，该数据暂不支持查询。新浪智研API返回为空，可能原因：非交易时段数据未更新、该接口暂不可用、或查询参数不支持。请尝试其他问题。"}
                    all_results.append({"tool": fn.name, "result": data_str[:_MCP_SYNTH_RESULT_MAX_CHARS]})
                    messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": data_str})
            else:
                break

        # 循环跑满：禁止返回原始 JSON dump，改走无 tools 的最终综合
        return await self._mcp_final_synthesis(message, all_results)

    @staticmethod
    def _mcp_result_for_prompt(tool_name: str, data) -> Optional[str]:
        """把一次 MCP 工具返回压缩为喂回模型的文本；返回 None 表示数据为空
        （走既有的「暂不支持查询」早退路径）。

        - 错误返回（第八波 is_mcp_error 识别）→ 简短错误说明，不喂大段原文；
        - 正常返回 → compact_mcp_result 压缩（未就绪时退回 json.dumps 截断）；
        - 空判定口径与历史行为一致（{} / "data":[] 且非 s_list 场景）。
        """
        # 第八波：错误识别优先（工程师B 提供；导入未就绪时本段整体跳过，退回现状）
        if is_mcp_error is not None:
            try:
                if is_mcp_error(data):
                    brief = mcp_error_brief(data) if mcp_error_brief is not None else "工具返回错误"
                    logger.info("MCP工具返回错误(%s): %s", tool_name, brief)
                    return f"[工具 {tool_name} 返回错误] {brief}"
            except Exception as e:
                logger.warning(
                    "is_mcp_error/mcp_error_brief 判定异常（%s，按正常数据处理）: %s",
                    tool_name, e, exc_info=True,
                )
        payload = data
        if compact_mcp_result is not None:
            try:
                compacted = compact_mcp_result(data)
                if compacted is not None:
                    payload = compacted
            except Exception as e:
                logger.warning(
                    "compact_mcp_result 压缩异常（%s，退回原始数据）: %s",
                    tool_name, e, exc_info=True,
                )
        data_str = json.dumps(payload, ensure_ascii=False)[:_MCP_TOOL_FEED_MAX_CHARS]
        if not data_str or data_str == "{}" or ('"data":[]' in data_str and '"s_list":[]' not in data_str):
            return None
        return data_str

    async def _mcp_final_synthesis(self, message: str, all_results: list) -> dict:
        """工具循环跑满 max_rounds 后的收尾：把所有已收集工具结果拼入上下文，
        发起一次无 tools 的最终综合调用，让 LLM 用人话总结。

        综合调用失败或返回为空时，返回优雅的「数据暂未获取成功」类提示；
        任何情况下都不把原始 JSON 给用户。
        """
        if not all_results:
            return {"role": "assistant", "content": "未获取到数据，请尝试更具体的查询。"}
        summary = "\n".join(f"[{r['tool']}] {r['result']}" for r in all_results)
        synth_messages = [
            {"role": "system", "content": _generic_mcp_system_prompt()},
            {"role": "user", "content": (
                f"用户问题：{message}\n\n"
                f"以下是通过数据工具查询到的结果（可能包含错误说明或不完整数据）：\n"
                f"{summary}\n\n"
                "请基于以上结果，用通顺的中文纯文本直接回答用户问题："
                "数据可用就正常解读；结果是错误说明、空数据或明显占位（如字段全为--）"
                "的部分，如实告知用户该项数据暂未获取成功，并给出可行的追问建议。"
                "禁止输出JSON或字段名罗列，禁止描述工具调用过程，禁止使用#和*号。"
            )},
        ]
        try:
            completion = await self.client.chat.completions.create(
                model=self.model, messages=synth_messages,
                temperature=0.2, max_tokens=4096,
            )
            answer = (completion.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning("MCP 最终综合调用失败，返回优雅降级提示: %s", e, exc_info=True)
            answer = ""
        answer = _clean_markdown(answer).strip()
        if answer:
            return {"role": "assistant", "content": answer}
        return {"role": "assistant", "content": _MCP_DATA_UNAVAILABLE}

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
                return {"role": "assistant", "content": _clean_markdown(f"今日A股涨跌分布：上涨{b.get('total_up','?')}家 下跌{b.get('total_down','?')}家 平盘{b.get('平','?')}家。涨停{b.get('涨停','?')}家 跌停{b.get('跌停','?')}家。")}
        if any(kw in msg for kw in ["热搜","热榜"]):
            h = fetch_hot_stocks()
            if h:
                items = "\n".join(f"{i+1}. {s['name']}({s['code']}) 热度{s.get('heat','?')}" for i, s in enumerate(h[:10]))
                return {"role": "assistant", "content": _clean_markdown(f"A股热搜榜Top10：\n{items}\n")}
            return {"role": "assistant", "content": "热搜数据暂不可用，新浪智研API返回为空。请稍后再试。"}
        if any(kw in msg for kw in ["汇率","人民币","美元"]):
            f = fetch_forex()
            if f.get("在岸人民币","?") != "?":
                return {"role": "assistant", "content": _clean_markdown(f"最新汇率：在岸人民币 {f['在岸人民币']} 涨跌{f.get('涨跌','?')} ")}
            return {"role": "assistant", "content": "汇率数据暂不可用，新浪智研API返回为空。请稍后再试。"}
        if any(kw in msg for kw in ["期货","黄金","原油","铜"]):
            kw_map = {"黄金":("gn","AU0"),"原油":("gn","SC0"),"铜":("gn","CU0")}
            for kw, (mkt, sym) in kw_map.items():
                if kw in msg:
                    d = fetch_futures(mkt, sym)
                    if d.get("price"):
                        return {"role": "assistant", "content": _clean_markdown(f"{kw}期货：价格{d['price']} 涨跌{d.get('pct','?')}% 成交量{d.get('vol','?')} ")}
                    return {"role": "assistant", "content": f"{kw}期货数据暂不可用，请稍后再试。"}
        return await self._generic_mcp(message, stream)

    # 第二波：Agent 工具循环与多 pass 生成配置
    _AGENT_MAX_ROUNDS = 8        # 工具调用循环上限，超轮降级 _chat
    _TOOL_RESULT_MAX_CHARS = 3000  # 单条工具结果截断长度，防 context 膨胀
    _CRITIQUE_MIN_CHARS = 500    # 草稿不长于此长度时不做自我审查（成本护栏）

    async def _agent_query(self, user_message: str, stream: bool, history: list = None, hint: str = None, disclaimer: bool = True):
        """
        Agent 工具调用循环（第二波）：模型自主决定调用哪些数据工具，多轮拿数后再成文。
        仅由 process_message 在复杂分析场景路由进入。
        hint：可选路由提示（社媒舆情/投资人格等高优先级意图传入），非空时以
        『【本问题路由提示】...』追加到 AGENT_QUERY_PROMPT 系统消息尾部，
        引导模型优先选用对应工具；None 时系统消息与原来完全一致。
        兜底原则：工具注册表未就绪 / 任一轮异常 / 循环超轮 → 降级 _chat，绝不向上抛异常。
        """
        if not TOOL_REGISTRY or execute_tool is None or AGENT_QUERY_PROMPT is None:
            logger.warning("Agent 工具注册表或提示词未就绪，降级为普通对话路径")
            return await self._chat(user_message, stream, history=history)

        date_display = datetime.now().strftime("%Y年%m月%d日")
        date_str = datetime.now().strftime("%Y%m%d")  # 第四波存档用：agent_query 的 trade_date 取当前日期
        system = AGENT_QUERY_PROMPT + f"\n\n当前日期：{date_display}。"
        if hint:
            system = f"{system}\n\n{hint}"
        messages = [{"role": "system", "content": system}]
        messages.extend(_trim_history(history))  # 最多保留最近 10 轮（20 条）
        messages.append({"role": "user", "content": user_message})

        tool_context_parts = []  # 工具返回摘要，供自我审查核对数字出处
        loop = asyncio.get_running_loop()

        # 第十二波：Agent 工程三件套接入（全部 fail-safe，任何异常不影响主循环）
        # Scratchpad：本次会话的 JSONL 审计日志（session_id 缺省短 uuid）
        scratchpad = _audit_safe(Scratchpad) if Scratchpad is not None else None
        if scratchpad is not None:
            _audit_safe(scratchpad.log_init, user_message)
        # ToolCallGuard：工具调用软护栏（只警告不阻断，警告注入工具结果尾部）
        guard = _audit_safe(ToolCallGuard) if ToolCallGuard is not None else None

        try:
            for _ in range(self._AGENT_MAX_ROUNDS):
                # microcompact 上下文压缩：每轮调 LLM 前执行；
                # cleared>0 时把压缩事实记入审计思考流
                if microcompact is not None:
                    compacted = _audit_safe(microcompact, messages)
                    if isinstance(compacted, tuple) and len(compacted) == 2:
                        messages, compact_stats = compacted
                        if (scratchpad is not None and isinstance(compact_stats, dict)
                                and compact_stats.get("cleared")):
                            _audit_safe(
                                scratchpad.log_thinking,
                                f"microcompact 清理 {compact_stats['cleared']} 条历史工具结果"
                                f"（{compact_stats.get('chars_before')}"
                                f"→{compact_stats.get('chars_after')} 字符）",
                            )
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
                assistant_msg_override = None

                # DeepSeek 偶发把工具调用以 DSML 纯文本输出而非结构化 tool_calls：
                # 解析成合成调用继续工具循环，杜绝原始标记泄漏给用户
                if not tool_calls:
                    dsml_calls, dsml_remainder = _parse_dsml_tool_calls(
                        choice.message.content or ""
                    )
                    if dsml_calls:
                        logger.warning(
                            "检测到 DSML 文本工具调用 %d 个，解析后继续工具循环",
                            len(dsml_calls),
                        )
                        tool_calls = [
                            SimpleNamespace(
                                id=f"dsml-{_}-{i}",
                                function=SimpleNamespace(
                                    name=n,
                                    arguments=json.dumps(a, ensure_ascii=False),
                                ),
                            )
                            for i, (n, a) in enumerate(dsml_calls)
                        ]
                        assistant_msg_override = {
                            "role": "assistant",
                            "content": dsml_remainder or None,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": n,
                                        "arguments": json.dumps(a, ensure_ascii=False),
                                    },
                                }
                                for tc, (n, a) in zip(tool_calls, dsml_calls)
                            ],
                        }

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
                    draft = _strip_dsml(choice.message.content or "")
                    final = await self._critique_and_revise(draft, query_context)
                    # 出口卫生（全路径）：剥除开头元推理句 + 工具函数名软泄漏
                    final = _strip_function_names(_strip_meta_openings(final))
                    disclaimer = (
                        "\n\n风险提示：以上内容仅为客观数据整理与公开信息分析，不构成任何投资建议。市场有风险，投资需谨慎。"
                        if disclaimer else ""
                    )
                    content = _clean_markdown(final) + disclaimer
                    # 第四波：存档最终产出（trade_date 用当前日期）
                    self._archive_safe("agent_query", None, content, query_context, date_str)
                    return {"role": "assistant", "content": content}

                # 有工具调用：回显 assistant 消息，逐个在线程池执行并追加 tool 结果
                messages.append(assistant_msg_override or _assistant_tool_message(choice.message))
                for tc in tool_calls:
                    fn_name = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except Exception as e:
                        logger.warning("Agent 工具参数解析失败(%s): %s", fn_name, e, exc_info=True)
                        fn_args = {}
                    # 审计钩子：调用落盘 + 软护栏检查（警告不阻断，注入工具结果尾部）
                    if scratchpad is not None:
                        _audit_safe(scratchpad.log_tool_call, fn_name, fn_args)
                    guard_warning = (
                        _audit_safe(guard.check, fn_name, fn_args)
                        if guard is not None else None
                    )
                    try:
                        result = await loop.run_in_executor(None, execute_tool, fn_name, fn_args)
                    except Exception as e:
                        logger.warning("Agent 工具执行异常(%s): %s", fn_name, e, exc_info=True)
                        result = {"ok": False, "error": f"工具执行异常: {e}"}
                    # 审计钩子：实际调用计数 + 结果落盘
                    if guard is not None:
                        _audit_safe(guard.record, fn_name, fn_args)
                    if scratchpad is not None:
                        _audit_safe(scratchpad.log_tool_result, fn_name, result)
                    content = json.dumps(result, ensure_ascii=False)[:self._TOOL_RESULT_MAX_CHARS]
                    if guard_warning:
                        content = f"{content}\n{guard_warning}"
                    tool_context_parts.append(f"[{fn_name}] {content[:800]}")
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
        except Exception as e:
            logger.warning("Agent 工具循环异常，降级为普通对话路径: %s", e, exc_info=True)
            return await self._chat(user_message, stream, history=history)

        # 循环超轮：模型仍要调工具 → 不丢弃已检索成果，
        # 追加「工具用尽」指令后做最后一次无工具强制成文；异常才降级 _chat
        logger.warning("Agent 工具循环达到 %d 轮上限，基于已检索成果强制成文", self._AGENT_MAX_ROUNDS)
        try:
            messages.append({
                "role": "user",
                "content": "工具调用次数已用完，禁止再调用任何工具。请基于以上已经检索到的"
                           "信息直接回答最初的问题：已查证的内容如实组织成文并按来源规范"
                           "标注出处；未能查证的部分明确说明「未能查证」，不得编造。",
            })
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=8192,
            )
            draft = _strip_dsml(completion.choices[0].message.content or "")
            query_context = user_message + "\n\n" + "\n".join(tool_context_parts)
            final = await self._critique_and_revise(draft, query_context)
            # 出口卫生（全路径）：剥除开头元推理句 + 工具函数名软泄漏
            final = _strip_function_names(_strip_meta_openings(final))
            disclaimer_text = (
                "\n\n风险提示：以上内容仅为客观数据整理与公开信息分析，不构成任何投资建议。市场有风险，投资需谨慎。"
                if disclaimer else ""
            )
            content = _clean_markdown(final) + disclaimer_text
            self._archive_safe("agent_query", None, content, query_context, date_str)
            return {"role": "assistant", "content": content}
        except Exception as e:
            logger.warning("超轮强制成文异常，降级为普通对话路径: %s", e, exc_info=True)
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
        history: list = None, archive_callback=None, disclaimer: bool = True,
    ):
        """
        调用 DeepSeek API。
        history 非空时按 system + history + 当前 user 组装 messages（闲聊上下文）。
        archive_callback：第四波存档回调，仅流式路径有效，透传给 _stream_response，
        在流结束后以最终全文调用（签名为 callback(accumulated_text)）。
        disclaimer：非流式出口是否追加金融风险提示（闲聊/校园路径传 False）。
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
            disclaimer = (
                "\n\n风险提示：以上内容仅为客观数据整理与公开信息分析，不构成任何投资建议。市场有风险，投资需谨慎。"
                if disclaimer else ""
            )
            clean = _clean_markdown(_strip_function_names(_strip_meta_openings(_strip_dsml(raw))))
            return {"role": "assistant", "content": clean + disclaimer}

    async def _stream_response(self, messages: list, archive_callback=None) -> AsyncGenerator:
        """流式响应生成器。

        输出卫生：流式对每个 chunk 做符号级清洗（_strip_md_symbols，删除 *
        与 #），非流式则做完整 _clean_markdown 结构性清洗。符号级方案无
        状态、跨 chunk 安全——** 配对无需识别：目标是 * 与 # 零到达用户，
        无论 ** 完整到达还是拆在两个 chunk，每个 * 都被删除，拼接结果与
        整段清洗一致；不含符号的 chunk 原样透传，边界不受影响。

        archive_callback：可选存档回调（第四波），在流结束（生成器被完整消费、
        含截断续写完成）后以最终全文（清洗后）调用一次；客户端中途断开时不触发。
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
                    piece = _strip_md_symbols(choice.delta.content)
                    accumulated += piece
                    if piece:  # 整个 chunk 都是 #/* 时清洗后为空，跳过不发送
                        yield piece
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
