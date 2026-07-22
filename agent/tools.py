"""
工具注册表 v1.0：把 data_fetcher 数据层函数包装成 OpenAI function calling 标准格式。

设计原则：
- TOOL_REGISTRY 使用规范 JSON Schema（description/parameters 完整展开），
  供 DeepSeek function calling 自主选择调用
- execute_tool 统一分发执行，绝不抛异常：
  成功返回 {"ok": True, "data": ...}，失败返回 {"ok": False, "error": "中文错误说明"}
- 分发三映射表：研报全文向量检索工具先查 _REPORT_VEC_IMPL（report_vectors 向量检索层），
  研报库工具查 _REPORT_IMPL（report_library 存储/检索层），其余数据工具查 _IMPL（data_fetcher 数据层）
- 数据层/研报库函数缺失（改名/未实现）时返回 ok=False，而不是 ImportError
- 内部同步调用，编排层负责放线程池
"""

import json
import logging

logger = logging.getLogger(__name__)


# ── 共享 JSON Schema 片段 ──

_DATE_PARAM = {
    "type": "string",
    "description": "交易日期，格式 YYYYMMDD（如 20260612）。须为最近交易日；"
                   "非交易日时数据层会自动回退最多 5 天取最近有数据的一天。",
}

_SECTOR_PARAM = {
    "type": "string",
    "description": "申万一级行业中文名（如 食品饮料、电子、医药生物）。"
                   "支持常见俗称/别名（如 白酒→食品饮料、半导体→电子、券商→非银金融），"
                   "数据层已做映射。",
}

_MARKET_PARAM = {
    "type": "string",
    "enum": ["cn", "hk", "us"],
    "description": "市场代码：cn=A股，hk=港股，us=美股。",
}

_SYMBOL_PARAM = {
    "type": "string",
    "description": "股票代码，A股需带交易所前缀小写（如 sh600519、sz000002），"
                   "港股如 hk00700，美股小写如 aapl。",
}

# ── 研报库工具共享片段 ──

_REPORT_STOCK_CODE_PARAM = {
    "type": "string",
    "description": "A 股股票代码，支持带交易所前缀（如 sh600519、sz000002）"
                   "或纯 6 位代码（如 600519），系统会自动去前缀归一。",
}

_REPORT_INDUSTRY_PARAM = {
    "type": "string",
    "description": "行业名（如 食品饮料、电子、半导体），按行业检索/聚合研报时使用。",
}

_REPORT_DAYS_PARAM = {
    "type": "integer",
    "minimum": 1,
    "maximum": 365,
    "default": 30,
    "description": "回溯最近多少天的研报，默认 30，最大 365。",
}

# ── 研报全文向量检索工具共享片段（研报库 v2）──

_REPORT_VEC_DAYS_PARAM = {
    "type": "integer",
    "minimum": 1,
    "maximum": 365,
    "default": 90,
    "description": "回溯最近多少天的研报正文，默认 90，最大 365。",
}

_REPORT_VEC_TOP_K_PARAM = {
    "type": "integer",
    "minimum": 1,
    "maximum": 10,
    "default": 5,
    "description": "返回最相关的正文段落条数，默认 5，最多 10。",
}


# ═══════════════════════════════════════════
# 工具注册表（OpenAI function calling 格式）
# ═══════════════════════════════════════════

TOOL_REGISTRY: list = [
    {
        "type": "function",
        "function": {
            "name": "get_market_indices",
            "description": "获取 A 股 7 大指数（上证综指/深证成指/创业板指/科创50/沪深300/中证500/中证1000）"
                           "当日行情、5日/20日趋势、均线位置和量比。"
                           "分析大盘整体走势、判断指数强弱时使用。",
            "parameters": {
                "type": "object",
                "properties": {"date": _DATE_PARAM},
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sector_list",
            "description": "获取 31 个申万一级行业当日涨跌幅排行 + 5日/10日/20日趋势 + 强弱标签。"
                           "了解行业轮动、找领涨/领跌板块时使用。",
            "parameters": {
                "type": "object",
                "properties": {"date": _DATE_PARAM},
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sector_valuation",
            "description": "获取指定板块估值水位：成分股按总市值加权 PE/PB + 近 1 年历史分位。"
                           "判断板块贵不贵、估值处于历史什么位置时使用（板块深挖第一步）。",
            "parameters": {
                "type": "object",
                "properties": {"sector": _SECTOR_PARAM, "date": _DATE_PARAM},
                "required": ["sector", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sector_moneyflow",
            "description": "获取指定板块资金博弈：主力（特大单+大单）/中小单净流入、"
                           "个股主力净流入/流出 TOP5。分析板块资金面、主力动向时使用。",
            "parameters": {
                "type": "object",
                "properties": {"sector": _SECTOR_PARAM, "date": _DATE_PARAM},
                "required": ["sector", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sector_earnings",
            "description": "获取指定板块景气度：成分股业绩预告聚合（预喜/预忧占比、变动幅度中值、"
                           "改善/恶化个股）+ 业绩快报增强。分析板块基本面景气方向时使用。",
            "parameters": {
                "type": "object",
                "properties": {"sector": _SECTOR_PARAM, "date": _DATE_PARAM},
                "required": ["sector", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sector_stocks",
            "description": "获取指定板块成分股当日行情明细：领涨/领跌 TOP5、涨跌家数、"
                           "板块成交额、高振幅个股、头部个股资金流。"
                           "深入板块内部结构、找领涨龙头时使用。",
            "parameters": {
                "type": "object",
                "properties": {"sector": _SECTOR_PARAM, "date": _DATE_PARAM},
                "required": ["sector", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fund_flows",
            "description": "获取全市场资金流向：北向/南向资金净买入（亿元）+ 融资融券余额。"
                           "分析增量资金、杠杆资金动向时使用。",
            "parameters": {
                "type": "object",
                "properties": {"date": _DATE_PARAM},
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_breadth",
            "description": "获取 A 股全市场涨跌分布（涨停/跌停数、各涨幅区间家数、上涨/下跌总数）。"
                           "判断市场情绪温度、赚钱效应时使用。数据为当日实时/最新。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_limit_up_pool",
            "description": "获取 A 股当日涨停池（涨停个股名称、涨幅、涨停原因）。"
                           "分析短线题材热点、连板情绪时使用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hot_stocks",
            "description": "获取 A 股股票热搜榜 TOP15（按关注度排序）。"
                           "了解散户关注焦点、市场人气方向时使用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_strong_sectors",
            "description": "获取 A 股当日强势概念板块 TOP10（剔除 ST）。"
                           "找题材主线、概念炒作方向时使用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": "获取单只股票实时行情（现价、涨跌幅、开高低、成交量、昨收）。"
                           "查询具体个股当前价格表现时使用。",
            "parameters": {
                "type": "object",
                "properties": {"market": _MARKET_PARAM, "symbol": _SYMBOL_PARAM},
                "required": ["market", "symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_kline",
            "description": "获取单只股票近期日 K 线（日期、收盘价、涨跌幅）。"
                           "分析个股近期走势形态时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "market": _MARKET_PARAM,
                    "symbol": _SYMBOL_PARAM,
                    "days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 60,
                        "default": 10,
                        "description": "返回最近多少个交易日的 K 线，默认 10，最多 60。",
                    },
                },
                "required": ["market", "symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_news",
            "description": "搜索单只股票相关新闻（标题 + 链接）。"
                           "了解个股消息面、公告舆情时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": _SYMBOL_PARAM,
                    "market": _MARKET_PARAM,
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 10,
                        "description": "返回新闻条数，默认 10，最多 20。",
                    },
                },
                "required": ["symbol", "market"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "按关键词搜索财经新闻（新浪智研）。"
                           "追踪某个主题/事件/政策的相关报道时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词，如 降准、人工智能、宁德时代。",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 15,
                        "description": "返回新闻条数，默认 15，最多 20。",
                    },
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_futures",
            "description": "获取期货行情（内盘商品/外盘商品/股指期货）。"
                           "分析大宗商品、黄金原油、期股联动时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "market": {
                        "type": "string",
                        "enum": ["gn", "global", "cff"],
                        "default": "gn",
                        "description": "期货市场：gn=内盘商品，global=外盘商品，cff=股指期货。默认 gn。",
                    },
                    "symbol": {
                        "type": "string",
                        "default": "AU0",
                        "description": "期货合约代码，如 AU0=沪金主力、SC0=原油主力、CU0=沪铜主力。默认 AU0。",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_us_sectors",
            "description": "获取美股板块涨幅排行 TOP10（含领涨股）。"
                           "分析隔夜美股结构、映射 A 股相关板块时使用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hk_sectors",
            "description": "获取港股领涨板块 TOP10（含领涨股）。"
                           "分析港股热点、AH 联动方向时使用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_global_indices",
            "description": "获取全球主要指数最新行情（标普500/道琼斯/纳斯达克/恒生/日经/富时100/欧元美元）。"
                           "分析隔夜外围市场表现时使用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_china_macro",
            "description": "获取中国宏观数据：CPI/PPI/PMI/M2/GDP/社融 + PPI 产业链子项。"
                           "分析国内经济周期位置、政策环境时使用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_us_macro",
            "description": "获取美国宏观数据：联邦基金利率、10Y/2Y 美债收益率、利差、VIX 恐慌指数。"
                           "分析海外流动性、风险偏好时使用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_research_reports",
            "description": "检索券商研报库：按关键词、个股或行业查询近期券商研究报告，"
                           "返回研报标题、券商、日期、评级、目标价、盈利预测（EPS）等。"
                           "用户询问券商研报观点、目标价、盈利预测、某只股票/某个行业有哪些研报覆盖时使用。"
                           "注意：query、stock_code、industry 三个检索条件至少提供一个，"
                           "全部留空会返回参数错误。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索关键词，匹配研报标题/股票名称/券商名，如 茅台、半导体、中金。",
                    },
                    "stock_code": _REPORT_STOCK_CODE_PARAM,
                    "industry": _REPORT_INDUSTRY_PARAM,
                    "days": _REPORT_DAYS_PARAM,
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                        "description": "返回研报条数，按日期倒序，默认 10，最多 50。",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_rating_summary",
            "description": "聚合券商评级观点：统计指定个股或行业近期研报的评级分布"
                           "（买入/增持/中性等各多少篇）、目标价区间、平均盈利预测与最新研报列表。"
                           "用户问「券商最近怎么看某只股票/某个行业」、比较多家券商的共识与分歧时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": _REPORT_STOCK_CODE_PARAM,
                    "industry": _REPORT_INDUSTRY_PARAM,
                    "days": _REPORT_DAYS_PARAM,
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_report_content",
            "description": "研报正文全文检索：按问题/主题对券商研报正文做向量语义检索，"
                           "返回最相关的正文段落（含券商、日期、研报标题、章节、相关度）。"
                           "用户问研报观点细节、券商的论证逻辑、多家券商观点分歧的原因，"
                           "或需要引用研报正文原文时使用。"
                           "只要评级/目标价等元数据时用 search_research_reports，"
                           "本工具只在需要正文内容时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索问题或主题（必填），如 "
                                       "「茅台的渠道改革进展」「半导体国产替代的逻辑」。",
                    },
                    "stock_code": _REPORT_STOCK_CODE_PARAM,
                    "industry": _REPORT_INDUSTRY_PARAM,
                    "days": _REPORT_VEC_DAYS_PARAM,
                    "top_k": _REPORT_VEC_TOP_K_PARAM,
                },
                "required": ["query"],
            },
        },
    },
]


# ═══════════════════════════════════════════
# 工具分发执行
# ═══════════════════════════════════════════

def _get_data_fetcher():
    """延迟导入数据层，导入失败返回 None（不让 ImportError 冒到编排层）。"""
    try:
        from . import data_fetcher
        return data_fetcher
    except ImportError:
        try:
            from agent import data_fetcher
            return data_fetcher
        except ImportError:
            try:
                import data_fetcher  # 同目录已加入 sys.path 的场景
                return data_fetcher
            except ImportError:
                logger.warning("data_fetcher 模块导入失败", exc_info=True)
                return None


def _get_report_library():
    """延迟导入研报库存储/检索层，导入失败返回 None（不让 ImportError 冒到编排层）。

    report_library 由研报库工作线独立交付，tools 只按公开函数契约
    （search_reports / rating_summary）接线，模块缺失时优雅降级。
    """
    try:
        from . import report_library
        return report_library
    except ImportError:
        try:
            from agent import report_library
            return report_library
        except ImportError:
            try:
                import report_library  # 同目录已加入 sys.path 的场景
                return report_library
            except ImportError:
                logger.warning("report_library 模块导入失败", exc_info=True)
                return None


def _get_report_vectors():
    """延迟导入研报全文向量检索层，导入失败返回 None（不让 ImportError 冒到编排层）。

    report_vectors 由研报库 v2 向量检索工作线独立交付，tools 只按公开函数契约
    （search_vectors，全局契约第 5 条）接线；模块缺失或重依赖
    （sentence_transformers/torch）未安装时优雅降级。
    """
    try:
        from . import report_vectors
        return report_vectors
    except ImportError:
        try:
            from agent import report_vectors
            return report_vectors
        except ImportError:
            try:
                import report_vectors  # 同目录已加入 sys.path 的场景
                return report_vectors
            except ImportError:
                logger.warning("report_vectors 模块导入失败", exc_info=True)
                return None


def _apply_sector_alias(df, sector: str) -> str:
    """板块俗称 → 申万一级行业标准名（数据层常量缺失时原样返回）。"""
    alias_map = getattr(df, "SW_SECTOR_ALIAS", None) or {}
    return alias_map.get(sector.strip(), sector)


# 工具名 → (数据层函数名, 参数适配器(module, args) -> kwargs)
# 适配器把工具参数名翻译成数据层函数真实签名
_IMPL = {
    "get_market_indices":   ("fetch_a_share_indices", lambda df, a: {"date": a["date"]}),
    "get_sector_list":      ("fetch_shenwan_sectors", lambda df, a: {"date": a["date"]}),
    "get_sector_valuation": ("fetch_sector_valuation",
                             lambda df, a: {"sector_name": a["sector"], "trade_date": a["date"]}),
    "get_sector_moneyflow": ("fetch_sector_moneyflow",
                             lambda df, a: {"sector_name": a["sector"], "trade_date": a["date"]}),
    "get_sector_earnings":  ("fetch_sector_earnings",
                             lambda df, a: {"sector_name": a["sector"], "trade_date": a["date"]}),
    "get_sector_stocks":    ("fetch_sector_stock_detail",
                             lambda df, a: {"sector_name": _apply_sector_alias(df, a["sector"]),
                                            "date": a["date"]}),
    "get_fund_flows":       ("fetch_fund_flows", lambda df, a: {"date": a["date"]}),
    "get_market_breadth":   ("fetch_market_breadth", lambda df, a: {}),
    "get_limit_up_pool":    ("fetch_limit_up_pool", lambda df, a: {}),
    "get_hot_stocks":       ("fetch_hot_stocks", lambda df, a: {}),
    "get_strong_sectors":   ("fetch_strong_sectors", lambda df, a: {}),
    "get_stock_quote":      ("fetch_stock_quote",
                             lambda df, a: {"market": a["market"], "symbol": a["symbol"]}),
    "get_stock_kline":      ("fetch_stock_kline",
                             lambda df, a: {"market": a["market"], "symbol": a["symbol"],
                                            "days": _clamp_int(a.get("days"), 10, 1, 60)}),
    "get_stock_news":       ("fetch_stock_news",
                             lambda df, a: {"symbol": a["symbol"], "market": a["market"],
                                            "limit": _clamp_int(a.get("limit"), 10, 1, 20)}),
    "search_news":          ("fetch_mcp_news",
                             lambda df, a: {"keyword": a["keyword"],
                                            "limit": _clamp_int(a.get("limit"), 15, 1, 20)}),
    "get_futures":          ("fetch_futures",
                             lambda df, a: {"market": a.get("market") or "gn",
                                            "symbol": a.get("symbol") or "AU0"}),
    "get_us_sectors":       ("fetch_us_sectors", lambda df, a: {}),
    "get_hk_sectors":       ("fetch_hk_sectors", lambda df, a: {}),
    "get_global_indices":   ("fetch_global_indices", lambda df, a: {}),
    "get_china_macro":      ("fetch_china_macro", lambda df, a: {}),
    "get_us_macro":         ("fetch_us_macro", lambda df, a: {}),
}


# ── 研报库工具分发（report_library 存储/检索层，研报库工作线交付）──

class _ParamError(Exception):
    """工具参数适配器抛出的参数错误：dispatch 统一转成 ok=False 中文提示。"""


def _clean_str(value) -> str:
    """字符串参数清洗：None/非字符串归空串，去首尾空白。"""
    return value.strip() if isinstance(value, str) else ""


def _normalize_stock_code(code) -> str:
    """个股代码归一：去交易所前缀（sh/sz/bj，大小写不敏感）。

    合法输入（sh600519 / SZ000002 / 600519）返回 6 位纯数字代码；
    空值或清洗后非 6 位纯数字时返回空串（交由检索层按空条件处理，
    绝不抛异常）。
    """
    c = _clean_str(code).lower()
    for prefix in ("sh", "sz", "bj"):
        if c.startswith(prefix):
            c = c[len(prefix):]
            break
    return c if len(c) == 6 and c.isdigit() else ""


def _search_reports_kwargs(_rl, a: dict) -> dict:
    """search_research_reports 参数适配：归一个股代码、夹取 days(1-365)/limit(1-50)。

    query / stock_code / industry 三个检索条件全空时抛 _ParamError，
    dispatch 转成 ok=False 中文提示（schema 无必填项，此处做运行时校验）。
    """
    query = _clean_str(a.get("query"))
    stock_code = _normalize_stock_code(a.get("stock_code"))
    industry = _clean_str(a.get("industry"))
    if not (query or stock_code or industry):
        raise _ParamError("检索条件不足：query / stock_code / industry 至少提供一个检索条件")
    return {
        "query": query,
        "stock_code": stock_code,
        "industry": industry,
        "days": _clamp_int(a.get("days"), 30, 1, 365),
        "limit": _clamp_int(a.get("limit"), 10, 1, 50),
    }


def _rating_summary_kwargs(_rl, a: dict) -> dict:
    """get_rating_summary 参数适配：归一个股代码、夹取 days(1-365)。"""
    return {
        "stock_code": _normalize_stock_code(a.get("stock_code")),
        "industry": _clean_str(a.get("industry")),
        "days": _clamp_int(a.get("days"), 30, 1, 365),
    }


# 研报库工具名 → (report_library 函数名, 参数适配器(module, args) -> kwargs)
# 结构与 _IMPL 一致；execute_tool 先查本表再查 _IMPL
_REPORT_IMPL = {
    "search_research_reports": ("search_reports", _search_reports_kwargs),
    "get_rating_summary":      ("rating_summary", _rating_summary_kwargs),
}


# ── 研报全文向量检索工具分发（report_vectors 向量检索层，研报库 v2 交付）──

def _search_vectors_kwargs(_rv, a: dict) -> dict:
    """search_report_content 参数适配：归一个股代码、夹取 days(1-365 默认 90)、
    top_k(1-10 默认 5)。query 必填由 schema required + 运行时必填校验兜底。
    db_path / embedder 走检索层默认（惰性路径解析 / FakeEmbedder 降级）。"""
    return {
        "query": _clean_str(a.get("query")),
        "stock_code": _normalize_stock_code(a.get("stock_code")),
        "industry": _clean_str(a.get("industry")),
        "days": _clamp_int(a.get("days"), 90, 1, 365),
        "top_k": _clamp_int(a.get("top_k"), 5, 1, 10),
    }


# 研报全文工具名 → (report_vectors 函数名, 参数适配器(module, args) -> kwargs)
# 结构与 _REPORT_IMPL 一致；execute_tool 查表顺序：本表 → _REPORT_IMPL → _IMPL
_REPORT_VEC_IMPL = {
    "search_report_content": ("search_vectors", _search_vectors_kwargs),
}


def _clamp_int(value, default: int, lo: int, hi: int) -> int:
    """整数参数兜底：非法值用默认值，并夹在 [lo, hi] 区间。"""
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def _required_params(name: str) -> list:
    """从注册表 schema 里取该工具的必填参数列表。"""
    for tool in TOOL_REGISTRY:
        fn = tool.get("function", {})
        if fn.get("name") == name:
            return fn.get("parameters", {}).get("required", []) or []
    return []


def _json_safe(data):
    """保证返回值可被 json.dumps 序列化（numpy/pandas 类型转字符串兜底）。"""
    try:
        json.dumps(data)
        return data
    except (TypeError, ValueError):
        logger.warning("工具返回含不可 JSON 序列化字段，已做字符串化兜底")
        return json.loads(json.dumps(data, default=str, ensure_ascii=False))


def execute_tool(name: str, args: dict) -> dict:
    """
    统一分发执行工具，绝不抛异常。

    返回 {"ok": True, "data": ...} 或 {"ok": False, "error": "中文错误说明"}。
    内部同步调用数据层/研报库，编排层如需并发请自行放线程池。
    """
    # 模型有时把 arguments 序列化成字符串，兼容一下
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except (json.JSONDecodeError, ValueError):
            return {"ok": False, "error": f"工具参数不是合法 JSON：{args[:100]}"}
    if not isinstance(args, dict):
        return {"ok": False, "error": f"工具参数必须是对象，收到 {type(args).__name__}"}

    # 查表顺序：研报全文向量 _REPORT_VEC_IMPL → 研报库 _REPORT_IMPL → 数据层 _IMPL；
    # 三表都不命中才报未注册
    vec_impl = _REPORT_VEC_IMPL.get(name)
    report_impl = _REPORT_IMPL.get(name)
    impl = vec_impl or report_impl or _IMPL.get(name)
    if not impl:
        return {"ok": False, "error": f"未注册的工具：{name}"}
    fetcher_name, arg_builder = impl

    # 必填参数校验（schema 已声明，这里做运行时兜底）
    missing = [k for k in _required_params(name)
               if k not in args or args[k] in (None, "")]
    if missing:
        return {"ok": False, "error": f"缺少必填参数：{', '.join(missing)}"}

    # 后端函数可能改名/未实现：getattr 保护，不抛 ImportError/AttributeError
    if vec_impl:
        backend = _get_report_vectors()
        if backend is None:
            return {"ok": False, "error": "研报全文模块不可用"}
    elif report_impl:
        backend = _get_report_library()
        if backend is None:
            return {"ok": False, "error": "研报库模块不可用"}
    else:
        backend = _get_data_fetcher()
        if backend is None:
            return {"ok": False, "error": "数据采集模块不可用"}
    fetcher = getattr(backend, fetcher_name, None)
    if not callable(fetcher):
        logger.warning("后端模块函数不存在或不可调用：%s（工具 %s）", fetcher_name, name)
        if vec_impl:
            kind = "研报全文接口"
        else:
            kind = "研报库接口" if report_impl else "数据接口"
        return {"ok": False, "error": f"{kind}暂不可用（{fetcher_name} 未实现）"}

    try:
        kwargs = arg_builder(backend, args)
        data = fetcher(**kwargs)
        # 研报全文检索降级信号：hits 为空且带 note 说明索引未建/依赖缺失等，
        # 按全局契约第 6 条转成 ok=False 并原样透传 note
        if vec_impl and isinstance(data, dict) and not data.get("hits") and data.get("note"):
            return {"ok": False, "error": _clean_str(data.get("note"))}
        return {"ok": True, "data": _json_safe(data)}
    except _ParamError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        logger.warning("工具执行失败 name=%s fetcher=%s", name, fetcher_name, exc_info=True)
        return {"ok": False, "error": f"工具执行出错：{e}"}


# ── 工具目录（调试 / 测试断言用）──

_SHORT_DESC = {
    "get_market_indices": "A 股 7 大指数行情与趋势",
    "get_sector_list": "31 个申万一级行业涨跌幅排行",
    "get_sector_valuation": "板块估值水位（加权 PE/PB + 历史分位）",
    "get_sector_moneyflow": "板块资金博弈（主力/中小单净流入）",
    "get_sector_earnings": "板块景气度（业绩预告聚合）",
    "get_sector_stocks": "板块成分股行情明细（领涨/领跌）",
    "get_fund_flows": "全市场资金流向（北向/两融）",
    "get_market_breadth": "A 股全市场涨跌分布",
    "get_limit_up_pool": "当日涨停池",
    "get_hot_stocks": "A 股热搜榜 TOP15",
    "get_strong_sectors": "当日强势概念板块 TOP10",
    "get_stock_quote": "个股实时行情",
    "get_stock_kline": "个股近期日 K 线",
    "get_stock_news": "个股新闻搜索",
    "search_news": "财经新闻关键词搜索",
    "get_futures": "期货行情（内盘/外盘/股指）",
    "get_us_sectors": "美股板块涨幅排行",
    "get_hk_sectors": "港股领涨板块",
    "get_global_indices": "全球主要指数行情",
    "get_china_macro": "中国宏观数据（CPI/PMI/M2 等）",
    "get_us_macro": "美国宏观数据（美债/VIX 等）",
    "search_research_reports": "券商研报检索（评级/目标价/盈利预测）",
    "get_rating_summary": "券商评级聚合（评级分布/目标价区间）",
    "search_report_content": "研报正文全文检索（观点细节/论证逻辑）",
}


def get_tool_catalog() -> str:
    """把工具名 + 一句话描述拼成文本，供调试和测试断言。"""
    lines = [f"- {name}：{_SHORT_DESC.get(name, tool['function']['description'].split('。')[0])}"
             for name, tool in
             ((t["function"]["name"], t) for t in TOOL_REGISTRY)]
    return "可用工具目录（共 %d 个）：\n%s" % (len(lines), "\n".join(lines))
