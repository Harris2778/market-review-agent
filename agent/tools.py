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

import importlib
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
    "description": "股票代码。A股需带交易所前缀小写（如 sh600519、sz000002）；"
                   "港股为裸 5 位代码（如 00700，不要加 hk 前缀）；"
                   "美股为裸 ticker（如 AAPL，大小写均可，不要加 us 前缀）。",
}

# ── 智研扩展数据工具共享片段（公司基本面/基金/外汇/商品期货/港美榜单）──

_A_SHARE_SYMBOL_PARAM = {
    "type": "string",
    "description": "A 股股票代码，需带交易所前缀小写（如 sh600519、sz000002）。",
}

_FUND_CODE_PARAM = {
    "type": "string",
    "description": "基金代码，纯 6 位数字（如 110022），不带任何前缀。",
}

# ── 研报库工具共享片段 ──

_REPORT_STOCK_CODE_PARAM = {
    "type": "string",
    "description": "A 股股票代码，支持带交易所前缀（如 sh600519、sz000002）"
                   "或纯 6 位代码（如 600519），系统会自动去前缀归一。",
}

_REPORT_INDUSTRY_PARAM = {
    "type": "string",
    "description": (
        "行业名过滤。库内为申万风格行业名（如 白酒Ⅱ、非白酒、电池、半导体、"
        "汽车零部件、证券Ⅱ、IT服务Ⅱ），通俗叫法（如 食品饮料、白酒）常无命中。"
        "不确定时建议留空；检索类工具在行业过滤无命中时会自动回退为不限行业"
        "并在 note 字段说明。"
    ),
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
            "name": "get_hk_finance_report",
            "description": "获取港股公司财报指标（净利润/营业收入等，智研港股财报库）。"
                           "查询港股公司基本面、财报表现时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "港股裸 5 位代码（如 00700，不要加 hk 前缀）。",
                    },
                    "indicator": {
                        "type": "string",
                        "default": "净利润",
                        "description": "财报指标名，如 净利润、营业收入、毛利率。默认 净利润。",
                    },
                    "years": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 1,
                        "description": "回看年度数，默认 1。",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hk_fund_flow",
            "description": "获取港股个股主力资金流向历史（近 N 日净流入）。"
                           "分析港股资金面时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "港股裸 5 位代码（如 00700，不要加 hk 前缀）。",
                    },
                    "days": {
                        "type": "integer",
                        "minimum": 3,
                        "maximum": 60,
                        "default": 10,
                        "description": "回看交易日数（3/5/10/20/60 档），默认 10。",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_us_fund_flow",
            "description": "获取美股个股当日资金流向（超大单/大单/中单/小单净流入）。"
                           "分析美股资金面时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "美股裸 ticker（如 AAPL，大小写均可，不要加 us 前缀）。",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_us_market_breadth",
            "description": "获取美股全市场涨跌分布（各涨跌幅区间股票数量）。"
                           "回答美股大盘整体表现、市场热度时使用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_major_events",
            "description": "获取上市公司重大事项（公告级事件：分红送转/并购重组/停复牌等）。"
                           "查询公司最新公告、重大事件时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "market": _MARKET_PARAM,
                    "symbol": _SYMBOL_PARAM,
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 10,
                        "description": "返回事件条数，默认 10，最多 20。",
                    },
                },
                "required": ["market", "symbol"],
            },
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
    # ── 智研扩展数据工具（公司基本面/基金/外汇/商品期货/港美榜单）──
    {
        "type": "function",
        "function": {
            "name": "get_company_profile",
            "description": "获取 A 股上市公司基本资料：公司简介、所属行业、上市信息、注册地址等。"
                           "用户问公司是做什么的、主营业务、所属行业、上市时间等基本面背景时使用。"
                           "symbol 为 A 股代码，需带交易所前缀小写（如 sh600519、sz000002）。",
            "parameters": {
                "type": "object",
                "properties": {"symbol": _A_SHARE_SYMBOL_PARAM},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_managers",
            "description": "获取 A 股上市公司高管名单与履历：董事长/总经理/董秘等姓名、职务、简介。"
                           "用户问公司管理层、高管背景、人事变动时使用。"
                           "symbol 为 A 股代码，需带交易所前缀小写（如 sh600519）。",
            "parameters": {
                "type": "object",
                "properties": {"symbol": _A_SHARE_SYMBOL_PARAM},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_shareholder_count",
            "description": "获取 A 股股东户数及历史变化：最新股东户数、较上期增减、筹码集中/分散趋势。"
                           "分析筹码集中度、散户进出动向时使用。"
                           "code 为 A 股代码，带交易所前缀（sh600519）或纯 6 位（600519）均可，"
                           "系统会自动去前缀。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "A 股股票代码，带交易所前缀（如 sh600519）或纯 6 位"
                                       "（如 600519）均可，系统会自动去前缀。",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_report_full",
            "description": "获取 A 股完整财务报表（资产负债表/利润表/现金流量表明细，默认国际准则 gjzb）。"
                           "需要财报科目级明细、做深度财务分析时使用；只要关键指标快报时不必调用本工具。"
                           "paper_code 为 A 股代码，需带交易所前缀小写（如 sh600519）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_code": _A_SHARE_SYMBOL_PARAM,
                    "source": {
                        "type": "string",
                        "default": "gjzb",
                        "description": "报表口径数据源，缺省 gjzb（国际准则报表）；一般无需指定。",
                    },
                    "r_date": {
                        "type": "string",
                        "default": "",
                        "description": "报告期（可选，如 2024-12-31）；留空返回最新报告期。",
                    },
                },
                "required": ["paper_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_valuation",
            "description": "获取 A 股个股估值历史：PE/PB 时间序列与当前估值水平。"
                           "判断个股贵贱、估值分位、做历史估值对比时使用。"
                           "symbol 为 A 股代码，需带交易所前缀小写（如 sh600519）。",
            "parameters": {
                "type": "object",
                "properties": {"symbol": _A_SHARE_SYMBOL_PARAM},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lockup_schedule",
            "description": "获取 A 股限售解禁日程：解禁日期、解禁数量/市值、股东类型。"
                           "评估解禁抛压、回答某股近期有无解禁时使用。"
                           "symbol 为 A 股代码，需带交易所前缀小写（如 sh600519）。",
            "parameters": {
                "type": "object",
                "properties": {"symbol": _A_SHARE_SYMBOL_PARAM},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_margin_detail",
            "description": "获取 A 股个股融资融券明细：融资余额/融资买入额、融券余量、两融余额变化。"
                           "分析个股杠杆资金动向、多空力量时使用。"
                           "symbol 为 A 股代码，需带交易所前缀小写（如 sh600519）。",
            "parameters": {
                "type": "object",
                "properties": {"symbol": _A_SHARE_SYMBOL_PARAM},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_block_trades",
            "description": "获取 A 股个股大宗交易记录：成交日期、成交价/成交量/成交额、买卖营业部、折溢价率。"
                           "跟踪机构大宗调仓、异常折价成交时使用。"
                           "symbol 为 A 股代码，需带交易所前缀小写（如 sh600519）。",
            "parameters": {
                "type": "object",
                "properties": {"symbol": _A_SHARE_SYMBOL_PARAM},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_connect_holdings",
            "description": "获取沪深港通持股榜：北向资金（沪股通/深股通/港股通）个股持股数量/比例排行。"
                           "分析外资持仓动向、北向增持/减持个股排行时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["sh", "sz", "hk"],
                        "default": "sh",
                        "description": "通道类型：sh=沪股通（默认），sz=深股通，hk=港股通。",
                    },
                    "sort": {
                        "type": "string",
                        "enum": ["hold_ratio", "hold_num", "hold_date"],
                        "default": "hold_ratio",
                        "description": "排序字段：hold_ratio=持股比例（默认），hold_num=持股数量，"
                                       "hold_date=持股日期。",
                    },
                    "num": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 20,
                        "description": "返回条数，默认 20，最大 100。",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fund_info",
            "description": "获取公募基金档案：基金名称、类型、成立日期、规模、管理人等基本信息。"
                           "用户问某只基金是什么类型、规模多大、哪家公司管理时使用。"
                           "symbol 为 6 位基金代码（如 110022），不带任何前缀。",
            "parameters": {
                "type": "object",
                "properties": {"symbol": _FUND_CODE_PARAM},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fund_networth",
            "description": "获取公募基金净值：单位净值/累计净值及近期走势。"
                           "用户问基金净值、近期涨跌表现时使用。"
                           "symbol 为 6 位基金代码（如 110022），不带任何前缀。",
            "parameters": {
                "type": "object",
                "properties": {"symbol": _FUND_CODE_PARAM},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fund_holdings",
            "description": "获取公募基金重仓持股：前十大重仓股、占净值比例。"
                           "分析基金持仓结构、跟踪明星基金经理调仓时使用。"
                           "symbol 为 6 位基金代码（如 110022），不带任何前缀。",
            "parameters": {
                "type": "object",
                "properties": {"symbol": _FUND_CODE_PARAM},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fund_dividend",
            "description": "获取公募基金分红记录：分红公告日、每份分红金额、累计分红。"
                           "用户问基金分红历史、分红频率时使用。"
                           "symbol 为 6 位基金代码（如 110022），不带任何前缀。",
            "parameters": {
                "type": "object",
                "properties": {"symbol": _FUND_CODE_PARAM},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forex_quote",
            "description": "获取外汇汇率行情：指定货币对的最新汇率与涨跌。"
                           "回答人民币汇率、美元欧元等货币对汇率问题时使用。"
                           "symbol 为全大写货币对代码（如 USDCNY、EURUSD）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "外汇货币对代码，全大写（如 USDCNY、EURUSD、USDJPY）。",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_commodity_futures_list",
            "description": "获取商品期货合约列表：指定交易所（大商所/上期所/郑商所/广期所）全部上市合约。"
                           "回答某交易所有哪些期货品种合约、查具体合约代码时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "market": {
                        "type": "string",
                        "enum": ["dce", "shfe", "czce", "gfex"],
                        "default": "dce",
                        "description": "交易所代码：dce=大连商品交易所（默认），shfe=上海期货交易所，"
                                       "czce=郑州商品交易所，gfex=广州期货交易所。",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hk_special_ranking",
            "description": "获取港股特色榜单：国企股/蓝筹股/红筹股行情榜（涨跌幅、成交额排名）。"
                           "浏览港股国企蓝筹红筹表现、找港股大盘蓝筹异动时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "node": {
                        "type": "string",
                        "enum": ["gqg_hk", "lcg_hk", "hcg_hk"],
                        "default": "gqg_hk",
                        "description": "榜单类型：gqg_hk=国企股榜（默认），lcg_hk=蓝筹股榜，"
                                       "hcg_hk=红筹股榜。",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_us_fund_flow_history",
            "description": "获取美股个股主力资金流向历史：最近 N 日（默认 60 日）主力净流入/流出序列。"
                           "分析美股个股资金趋势、主力持续进出时使用；只要当日资金流用 get_us_fund_flow。"
                           "symbol 为美股裸 ticker（如 AAPL，不加 us 前缀）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "美股裸 ticker（如 AAPL、TSLA），不要加 us 前缀。",
                    },
                    "days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 60,
                        "default": 60,
                        "description": "回溯天数，默认 60，最大 60（底层接口为 60 日资金流历史）。",
                    },
                },
                "required": ["symbol"],
            },
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
    {
        "type": "function",
        "function": {
            "name": "get_market_sentiment",
            "description": "获取 A 股市场级社交情绪快照：情绪温度（0-100，含冰点/低迷/中性/活跃/亢奋标签）"
                           "+ 涨跌停/炸板统计（涨停数、跌停数、炸板率、最高连板）+ 东方财富人气榜 TOP10"
                           "+ 注入新闻的情感分布（利好/利空/中性）。"
                           "判断市场情绪温度、短线赚钱效应时使用。情绪是辅助信号，不是买卖依据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "交易日期，格式 YYYY-MM-DD 或 YYYYMMDD；缺省为当日。",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_sentiment",
            "description": "获取个股社交情绪：东方财富人气榜当前排名、近 N 天排名均值与变化趋势"
                           "（上升/下降/平稳/未知）+ 该股相关新闻的情感分布（利好/利空/中性）。"
                           "分析个股散户关注度、人气异动时使用。情绪是辅助信号，不是买卖依据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": _REPORT_STOCK_CODE_PARAM,
                    "days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 120,
                        "default": 30,
                        "description": "人气历史回溯天数，默认 30，最大 120。",
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["standard", "deep"],
                        "default": "standard",
                        "description": "情绪分布采样深度档：standard=标准采样（默认），"
                                       "deep=扩大采样量（更多帖子/评论，耗时更长）。",
                    },
                },
                "required": ["stock_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_analysis",
            "description": "获取个股技术分析仪表盘：均线排列（七态）、乖离率、MACD、RSI、量能状态、"
                           "支撑/压力位与 0-100 综合评分，并给出评分带结论（强势/偏多/中性/偏空/弱势）"
                           "与置信度上限（confidence_cap）及数据质量护栏说明（guardrail_reason）。"
                           "指标与评分为本地确定性计算，原样引用；回答技术分析、趋势形态类问题时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": _REPORT_STOCK_CODE_PARAM,
                    "days": {
                        "type": "integer",
                        "minimum": 30,
                        "maximum": 250,
                        "default": 120,
                        "description": "参与计算的最近日线根数，默认 120，最大 250。",
                    },
                },
                "required": ["stock_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_with_persona",
            "description": "获取指定投资人格的方法论分析框架（打分权重、阈值、分析规则、输出 schema、"
                           "分析清单与免责声明）。用户要求用某种投资风格（价值/成长/趋势/逆向）"
                           "分析标的时先调本工具拿框架，再按框架 checklist 逐项调用其他数据工具取数分析。"
                           "框架结论的 signal 只能取枚举值，confidence 不超过 0.9，末尾须带免责声明。",
            "parameters": {
                "type": "object",
                "properties": {
                    "persona": {
                        "type": "string",
                        "enum": ["value_cn", "growth_cn", "trend_cn", "contrarian_cn"],
                        "description": "投资人格：value_cn=价值、growth_cn=成长、"
                                       "trend_cn=趋势、contrarian_cn=逆向。",
                    },
                    "stock_code": {
                        "type": "string",
                        "description": "可选，待分析的 A 股股票代码（如 sh600519、600519），"
                                       "仅作上下文标注，不影响框架内容。",
                    },
                },
                "required": ["persona"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_social_hot",
            "description": "获取社媒平台热榜舆情：微博/知乎/抖音/B站热榜聚合（逐平台直连、聚合器兜底、"
                           "去重）+ 情感分布（利好/利空/中性）+ 股票关联提取（代码/名称命中统计）。"
                           "了解社媒热议焦点、散户舆情方向、热榜里被讨论的个股时使用。"
                           "社媒情绪噪声大，仅作辅助参考，不构成买卖依据。"
                           "小红书 v1 暂未覆盖，指定 platform=xiaohongshu 会返回缺席原因说明。",
            "parameters": {
                "type": "object",
                "properties": {
                    "platform": {
                        "type": "string",
                        "default": "all",
                        "description": "平台选择：all=全部支持平台（微博/知乎/抖音/B站，默认），"
                                       "或单个平台 weibo / zhihu / douyin / bilibili。"
                                       "xiaohongshu（小红书）暂未覆盖，传入会返回中文缺席说明。",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 30,
                        "default": 10,
                        "description": "每个平台返回热榜条数，默认 10，最多 30。",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_social_media",
            "description": "按关键词搜索社媒内容：v1 仅 B 站具备搜索能力（视频+专栏），"
                           "可选附带前 3 条结果的热门评论；返回结果附带情感分布"
                           "（利好/利空/中性）与股票关联提取（代码/名称命中统计）。"
                           "追踪某个主题、事件或个股在社媒的讨论时使用。"
                           "微博/知乎/抖音仅有热榜无搜索能力，会在 notes 中说明。"
                           "社媒情绪噪声大，仅作辅助参考，不构成买卖依据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词（必填），如 茅台、降息、人形机器人。",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 30,
                        "default": 10,
                        "description": "返回搜索结果条数，默认 10，最多 30。",
                    },
                    "with_comments": {
                        "type": "boolean",
                        "default": False,
                        "description": "是否附带评论：true 时对前 3 条有 post_id 的 B 站结果"
                                       "各取最多 10 条热门评论并入返回。默认 false。",
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["standard", "deep"],
                        "default": "standard",
                        "description": "情绪分布采样深度档（with_comments=true 时生效）："
                                       "standard=标准采样（默认），deep=扩大采样量。",
                    },
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_campus_knowledge",
            "description": "检索清华校园知识库：选课手册、课程信息、学生课程点评/总结、"
                           "thubook 书籍笔记的中文全文检索，返回带来源/标题/正文片段/"
                           "链接/相关度的条目列表。用户问选课建议、课程信息、保研交换、"
                           "转专业辅修、奖学金、宿舍食堂、校医院军训等校园问题时使用。"
                           "某门课程的点评综合总结请用 get_course_review_summary。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索关键词（必填），如 选课、保研、转专业、"
                                       "数据结构、宿舍。多关键词用空格分隔（AND 语义）。",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["sem_handbook", "thucourse_course",
                                 "thucourse_review", "thucourse_summary", "thubook"],
                        "description": "来源过滤：sem_handbook=选课手册，"
                                       "thucourse_course=课程信息，"
                                       "thucourse_review=学生课程点评，"
                                       "thucourse_summary=课程点评综合总结，"
                                       "thubook=书籍笔记。缺省为全部来源。",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 10,
                        "description": "返回条目数，默认 10，最多 20。",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_course_review_summary",
            "description": "获取指定课程的学生点评综合总结：平均评分、评分分布、点评条数、"
                           "代表性观点与总结文本。优先取库内现成总结；没有时按课程聚合"
                           "最多点评现场生成并回写知识库。用户问某门课怎么样、给分如何、"
                           "老师教得好不好、值不值得选时使用。总结是往届学生经验的自动"
                           "摘要，引用时须标注点评条数并提示时效。",
            "parameters": {
                "type": "object",
                "properties": {
                    "course_query": {
                        "type": "string",
                        "description": "课程名或课程关键词（必填），如 数据结构、"
                                       "高等微积分、某教师姓名。",
                    },
                },
                "required": ["course_query"],
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
    "get_hk_finance_report": ("fetch_hk_finance_report",
                              lambda df, a: {"symbol": a["symbol"],
                                             "indicator": a.get("indicator") or "净利润",
                                             "years": _clamp_int(a.get("years"), 1, 1, 10)}),
    "get_hk_fund_flow":     ("fetch_hk_fund_flow",
                             lambda df, a: {"symbol": a["symbol"],
                                            "days": _clamp_int(a.get("days"), 10, 3, 60)}),
    "get_us_fund_flow":     ("fetch_us_fund_flow",
                             lambda df, a: {"symbol": a["symbol"]}),
    "get_us_market_breadth": ("fetch_us_market_breadth", lambda df, a: {}),
    "get_stock_major_events": ("fetch_stock_major_events",
                               lambda df, a: {"market": a["market"], "symbol": a["symbol"],
                                              "limit": _clamp_int(a.get("limit"), 10, 1, 20)}),
    "get_china_macro":      ("fetch_china_macro", lambda df, a: {}),
    "get_us_macro":         ("fetch_us_macro", lambda df, a: {}),
    # ── 智研扩展数据工具（公司基本面/基金/外汇/商品期货/港美榜单）──
    "get_company_profile":  ("fetch_company_profile",
                             lambda df, a: {"symbol": a["symbol"]}),
    "get_company_managers": ("fetch_company_managers",
                             lambda df, a: {"symbol": a["symbol"]}),
    "get_shareholder_count": ("fetch_shareholder_count",
                              lambda df, a: {"code": a["code"]}),
    "get_financial_report_full": ("fetch_financial_report_full",
                                  lambda df, a: {"paper_code": a["paper_code"],
                                                 "source": a.get("source") or "gjzb",
                                                 "r_date": a.get("r_date") or ""}),
    "get_stock_valuation":  ("fetch_stock_valuation",
                             lambda df, a: {"symbol": a["symbol"]}),
    "get_lockup_schedule":  ("fetch_lockup_schedule",
                             lambda df, a: {"symbol": a["symbol"]}),
    "get_margin_detail":    ("fetch_margin_detail",
                             lambda df, a: {"symbol": a["symbol"]}),
    "get_block_trades":     ("fetch_block_trades",
                             lambda df, a: {"symbol": a["symbol"]}),
    "get_connect_holdings": ("fetch_connect_holdings",
                             lambda df, a: {"type": a.get("type") or "sh",
                                            "sort": a.get("sort") or "hold_ratio",
                                            "num": _clamp_int(a.get("num"), 20, 1, 100)}),
    "get_fund_info":        ("fetch_fund_info",
                             lambda df, a: {"symbol": a["symbol"]}),
    "get_fund_networth":    ("fetch_fund_networth",
                             lambda df, a: {"symbol": a["symbol"]}),
    "get_fund_holdings":    ("fetch_fund_holdings",
                             lambda df, a: {"symbol": a["symbol"]}),
    "get_fund_dividend":    ("fetch_fund_dividend",
                             lambda df, a: {"symbol": a["symbol"]}),
    "get_forex_quote":      ("fetch_forex_quote",
                             lambda df, a: {"symbol": a["symbol"]}),
    "get_commodity_futures_list": ("fetch_commodity_futures_list",
                                   lambda df, a: {"market": a.get("market") or "dce"}),
    "get_hk_special_ranking": ("fetch_hk_special_ranking",
                               lambda df, a: {"node": a.get("node") or "gqg_hk"}),
    "get_us_fund_flow_history": ("fetch_us_fund_flow_history",
                                 lambda df, a: {"symbol": a["symbol"],
                                                "days": _clamp_int(a.get("days"), 60, 1, 60)}),
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


# ── 开源灵感模块工具分发（sentiment / technical / personas 本地分析层）──
# 与 _IMPL 查表模式一致：工具名 → 本地处理器(args) -> dict。这三个工具的
# 底层不是单一数据函数，而是 Stage 1 交付的本地分析模块（BettaFish 灵感
# 社交情绪层 / daily_stock_analysis 灵感确定性技术分析 / Fincept 灵感
# 投资人格框架库），处理器负责多步编排与降级；execute_tool 查表顺序在
# 三表之后追加 _ANALYSIS_IMPL，不改旧条目。


def _lazy_import_module(mod_name: str):
    """惰性导入 agent 包内分析模块，导入失败返回 None（不让 ImportError 冒到编排层）。

    依次尝试：包内相对导入（agent.xxx）→ 绝对导入（agent.xxx）→ 裸名导入
    （同目录已加入 sys.path 的场景）；任何候选失败都继续尝试下一个，
    全部失败记 warning 返回 None。
    """
    candidates = []
    package = __package__ or "agent"
    candidates.append((f".{mod_name}", package))
    candidates.append((f"agent.{mod_name}", None))
    candidates.append((mod_name, None))
    for name, pkg in candidates:
        try:
            if pkg:
                return importlib.import_module(name, package=pkg)
            return importlib.import_module(name)
        except Exception:  # noqa: BLE001 - 惰性解析绝不抛出
            continue
    logger.warning("%s 模块导入失败（分析类工具将优雅降级）", mod_name)
    return None


def _get_sentiment_module():
    """惰性解析 agent/sentiment.py（BettaFish 灵感社交情绪层），失败返回 None。"""
    return _lazy_import_module("sentiment")


def _get_technical_module():
    """惰性解析 agent/technical.py（确定性技术分析层），失败返回 None。"""
    return _lazy_import_module("technical")


def _get_personas_module():
    """惰性解析 agent/personas.py（投资人格框架库），失败返回 None。"""
    return _lazy_import_module("personas")


def _prefixed_symbol(code: str) -> str:
    """6 位 A 股代码 → 带交易所前缀小写（600519→sh600519）；归一失败原样返回。"""
    c = _normalize_stock_code(code)
    if not c:
        return _clean_str(code)
    if c[0] == "6":
        return "sh" + c
    if c[0] in "489":
        return "bj" + c
    return "sz" + c


def _to_ts_code(code: str) -> str:
    """6 位 A 股代码 → Tushare ts_code（600519→600519.SH）；归一失败原样返回。"""
    c = _normalize_stock_code(code)
    if not c:
        return _clean_str(code)
    if c[0] == "6":
        return f"{c}.SH"
    if c[0] in "489":
        return f"{c}.BJ"
    return f"{c}.SZ"


def _inject_market_news(df):
    """市场情绪的可选新闻增强：取低代价的东财实时快讯单源注入情感分布。

    返回 (news_items, note)；取数不可用/失败/为空时 news_items=None 且
    note 为中文说明，绝不抛异常。
    """
    fetch_news = getattr(df, "fetch_eastmoney_news", None) if df is not None else None
    if not callable(fetch_news):
        return None, "未注入市场新闻（数据层无低代价当日新闻函数），情感分布仅含人气/涨跌停信号"
    try:
        items = fetch_news(limit=30)
    except Exception as e:
        logger.warning("市场情绪新闻注入失败（降级不注入）: %s", e)
        return None, "市场新闻获取失败，情感分布未计入新闻"
    if isinstance(items, list) and items:
        return items, None
    return None, "市场新闻为空，情感分布未计入新闻"


def _inject_stock_news(df, code: str):
    """个股情绪的新闻增强：复用 data_fetcher.fetch_stock_news 取该股新闻注入情感分布。

    返回 (news_items, note)；取数不可用/失败/为空时 news_items=None 且
    note 为中文说明，绝不抛异常。
    """
    fetch_news = getattr(df, "fetch_stock_news", None) if df is not None else None
    if not callable(fetch_news):
        return None, "未注入个股新闻（数据层个股新闻函数不可用），情感分布未计入"
    try:
        items = fetch_news(symbol=_prefixed_symbol(code), market="cn", limit=10)
    except Exception as e:
        logger.warning("个股新闻注入失败（降级不注入）: %s", e)
        return None, "个股新闻获取失败，情感分布未计入新闻"
    if isinstance(items, list) and items:
        return items, None
    return None, "个股新闻为空，情感分布未计入新闻"


def _append_note(result, note: str):
    """把一条降级说明追加进底层返回的 notes 列表（防御式，结构异常则忽略）。"""
    if note and isinstance(result, dict):
        notes = result.setdefault("notes", [])
        if isinstance(notes, list):
            notes.append(note)
    return result


_GUBA_BUZZ_LIMIT = 10   # get_stock_sentiment 附加股吧舆情的帖子条数
_GUBA_BUZZ_ENRICH = 3   # 股吧详情富化（回填正文+点赞）的帖子数


def _append_guba_block(result, code: str):
    """get_stock_sentiment 的股吧舆情增强：惰性调 social_media.get_guba_buzz。

    股吧端点 2026-07-22 实测定案仅覆盖个股舆情（帖子+正文+点赞，无评论
    数据）。成功且结构正常时在返回上追加 guba={posts, buzz} 键，股吧内部
    降级说明透传进 notes；模块缺席/能力缺失/异常/结构异常一律只进
    notes，绝不影响人气榜与新闻情感的既有返回。绝不抛。
    """
    if not isinstance(result, dict):
        return result
    try:
        social = _get_social_media_module()
        get_buzz = getattr(social, "get_guba_buzz", None) if social is not None else None
        if not callable(get_buzz):
            return _append_note(result, "股吧舆情通道不可用（social_media.get_guba_buzz 未就绪），个股舆情未含股吧数据")
        guba = get_buzz(code, limit=_GUBA_BUZZ_LIMIT, enrich=_GUBA_BUZZ_ENRICH)
        if not isinstance(guba, dict):
            return _append_note(result, "股吧舆情返回结构异常，个股舆情未含股吧数据")
        result["guba"] = {
            "posts": guba.get("posts") if isinstance(guba.get("posts"), list) else [],
            "buzz": guba.get("buzz") if isinstance(guba.get("buzz"), dict) else {},
        }
        for note in guba.get("notes") or []:
            _append_note(result, f"股吧：{note}")
    except Exception as e:  # noqa: BLE001 - 股吧失败绝不影响主返回
        logger.warning("股吧舆情增强失败（不影响人气榜与新闻情感）: %s", e)
        _append_note(result, f"股吧舆情获取失败（已跳过，不影响人气榜与新闻情感）: {e}")
    return result


_GUBA_DIST_POST_LIMIT = 80  # get_stock_sentiment 情绪分布块的股吧帖子采样条数


def _append_distribution_block(result, code: str = None, keyword: str = None,
                               depth: str = "standard"):
    """情绪分布增强块：惰性调 social_media.get_sentiment_distribution（分布化改造）。

    舆情呈现从「引用个别帖子/评论」升级为「整体情绪分布」为主体。成功且
    结构正常时在返回上追加 sentiment_distribution={target, samples_total,
    dist, weighted_dist, bull_bear, window, confidence, trend,
    representatives, method, sources, notes} 键；depth（standard/deep）
    透传采样深度档；模块缺席/能力缺失/异常/结构异常一律只进 notes，
    绝不影响主返回。绝不抛。
    """
    if not isinstance(result, dict):
        return result
    try:
        social = _get_social_media_module()
        get_dist = getattr(social, "get_sentiment_distribution", None) \
            if social is not None else None
        if not callable(get_dist):
            return _append_note(result, "情绪分布通道不可用（social_media.get_sentiment_distribution 未就绪），未挂情绪分布")
        if depth not in ("standard", "deep"):
            depth = "standard"
        if code:
            dist = get_dist(code=code, post_limit=_GUBA_DIST_POST_LIMIT,
                            depth=depth)
        else:
            dist = get_dist(keyword=keyword, depth=depth)
        if not isinstance(dist, dict):
            return _append_note(result, "情绪分布返回结构异常，未挂情绪分布")
        result["sentiment_distribution"] = dist
        for note in dist.get("notes") or []:
            _append_note(result, f"情绪分布：{note}")
    except Exception as e:  # noqa: BLE001 - 分布块失败绝不影响主返回
        logger.warning("情绪分布增强失败（不影响主返回）: %s", e)
        _append_note(result, f"情绪分布获取失败（已跳过，不影响主返回）: {e}")
    return result


def _handle_get_market_sentiment(args: dict) -> dict:
    """get_market_sentiment 处理器：市场情绪快照 + 可选新闻情感增强。"""
    sent = _get_sentiment_module()
    if sent is None:
        return {"ok": False, "note": "情绪模块不可用（agent/sentiment.py 未就绪）"}
    date_arg = _clean_str(args.get("date")) or None
    news_items, news_note = _inject_market_news(_get_data_fetcher())
    snapshot = sent.get_market_sentiment(date=date_arg, news_items=news_items)
    return _append_note(snapshot, news_note)


def _handle_get_stock_sentiment(args: dict) -> dict:
    """get_stock_sentiment 处理器：人气排名/趋势 + 新闻情感 + 股吧舆情 + 情绪分布。"""
    code = _normalize_stock_code(args.get("stock_code"))
    if not code:
        raise _ParamError(f"stock_code 无法归一为 6 位 A 股代码：{args.get('stock_code')!r}")
    days = _clamp_int(args.get("days"), 30, 1, 120)
    depth = _clean_str(args.get("depth")) or "standard"
    sent = _get_sentiment_module()
    if sent is None:
        return {"ok": False, "code": code, "note": "情绪模块不可用（agent/sentiment.py 未就绪）"}
    news_items, news_note = _inject_stock_news(_get_data_fetcher(), code)
    result = sent.get_stock_sentiment(code, days=days, news_items=news_items)
    result = _append_note(result, news_note)
    result = _append_guba_block(result, code)
    return _append_distribution_block(result, code=code, depth=depth)


def _fetch_stock_daily_rows(df, code: str, days: int):
    """经 data_fetcher 的 Tushare 连接取个股日线（升序 rows，限近 days 行）。

    data_fetcher 现有个股 K 线函数（fetch_stock_kline）只含日期/收盘/涨跌幅，
    不足以支撑技术指标，这里直接复用其 Tushare 连接（_get_pro）调 pro.daily。
    返回 (rows, note)；任何失败 rows=None 且 note 为中文说明，绝不抛异常。
    """
    if df is None:
        return None, "数据采集模块不可用，技术分析取数失败"
    get_pro = getattr(df, "_get_pro", None)
    if not callable(get_pro):
        return None, "data_fetcher 未提供 Tushare 连接（_get_pro 缺失），技术分析取数失败"
    try:
        pro = get_pro()
    except Exception as e:
        return None, f"Tushare 连接失败：{e}"
    if pro is None or not callable(getattr(pro, "daily", None)):
        return None, "Tushare 连接不可用（TUSHARE_TOKEN 未配置或连接失败）"
    from datetime import datetime as _dt, timedelta as _td
    end = _dt.now().strftime("%Y%m%d")
    start = (_dt.now() - _td(days=days * 2 + 30)).strftime("%Y%m%d")
    try:
        daily_df = pro.daily(ts_code=_to_ts_code(code), start_date=start, end_date=end)
    except Exception as e:
        return None, f"Tushare daily 取数失败：{e}"
    try:
        records = daily_df.to_dict("records")
    except Exception:  # noqa: BLE001 - 绝不抛出
        return None, "Tushare daily 返回结构异常，技术分析取数失败"
    rows = [r for r in records if isinstance(r, dict)]
    rows.sort(key=lambda r: str(r.get("trade_date", "")))
    if not rows:
        return None, f"Tushare daily 无 {code} 近期日线数据"
    return rows[-days:], None


def _latest_expected_trade_day(tech, df):
    """最近的应有交易日（'YYYY-MM-DD'）+ 判定方式说明（note，无补充说明为 None）。

    优先用 data_fetcher 的 Tushare 交易日历（trade_cal）；日历不可用时回退
    agent.technical.is_trade_day 的周一~周五启发式。绝不抛异常。
    """
    from datetime import date as _date, timedelta as _td
    today = _date.today()
    get_pro = getattr(df, "_get_pro", None) if df is not None else None
    if callable(get_pro):
        try:
            pro = get_pro()
            cal_df = pro.trade_cal(
                exchange="SSE",
                start_date=(today - _td(days=14)).strftime("%Y%m%d"),
                end_date=today.strftime("%Y%m%d"),
            )
            records = cal_df.to_dict("records")
            open_days = sorted({
                f"{str(r.get('cal_date'))[:4]}-{str(r.get('cal_date'))[4:6]}-{str(r.get('cal_date'))[6:8]}"
                for r in records
                if isinstance(r, dict) and str(r.get("is_open")) == "1"
                and len(str(r.get("cal_date"))) == 8
            })
            open_days = [d for d in open_days if d <= today.isoformat()]
            if open_days:
                return open_days[-1], None
        except Exception as e:
            logger.warning("交易日历获取失败，stale 判定改用启发式: %s", e)
    d = today
    for _ in range(14):
        try:
            if tech.is_trade_day(d.isoformat()):
                return d.isoformat(), "交易日历不可用，stale 判定用周一~周五启发式"
        except Exception:  # noqa: BLE001 - 绝不抛出
            break
        d -= _td(days=1)
    return None, "交易日历不可用且启发式判定失败，未做 stale 检查"


_TECH_SOURCE = "Tushare daily + agent.technical 本地确定性计算"


def _handle_get_technical_analysis(args: dict) -> dict:
    """get_technical_analysis 处理器：取日线 → compute_indicators → verdict 护栏。"""
    code = _normalize_stock_code(args.get("stock_code"))
    if not code:
        raise _ParamError(f"stock_code 无法归一为 6 位 A 股代码：{args.get('stock_code')!r}")
    days = _clamp_int(args.get("days"), 120, 30, 250)
    tech = _get_technical_module()
    if tech is None:
        return {"ok": False, "note": "技术分析模块不可用（agent/technical.py 未就绪）"}
    df = _get_data_fetcher()
    rows, fetch_note = _fetch_stock_daily_rows(df, code, days)
    if rows is None:
        return {"ok": False, "note": fetch_note, "source": _TECH_SOURCE}
    indicators = tech.compute_indicators(rows)
    if not isinstance(indicators, dict) or not indicators.get("ok"):
        note = indicators.get("note") if isinstance(indicators, dict) else None
        return {"ok": False, "note": note or "技术指标计算失败", "source": _TECH_SOURCE}
    as_of = indicators.get("as_of")
    notes = []
    expected, cal_note = _latest_expected_trade_day(tech, df)
    if cal_note:
        notes.append(cal_note)
    stale = bool(expected and isinstance(as_of, str) and as_of < expected)
    data_quality = {
        "stale": stale,
        "no_volume": indicators.get("volume_state") == "missing",
        "insufficient": False,
    }
    verdict = tech.verdict_from_score(indicators.get("score"), data_quality)
    return {
        "ok": True,
        "as_of": as_of,
        "close": indicators.get("close"),
        "indicators": indicators,
        "verdict": verdict,
        "source": f"Tushare daily（as_of {as_of}）+ agent.technical 本地确定性计算",
        "notes": notes,
    }


def _handle_analyze_with_persona(args: dict) -> dict:
    """analyze_with_persona 处理器：渲染人格方法论框架；未知人格降级为清单。"""
    key = _clean_str(args.get("persona"))
    pers = _get_personas_module()
    if pers is None:
        return {"note": "投资人格模块不可用（agent/personas.py 未就绪）", "available": []}
    framework = pers.render_persona_framework(key)
    if framework is None:
        return {"note": f"未知投资人格「{key}」，请从可用人格中选择",
                "available": pers.list_personas()}
    return {
        "persona": key,
        "stock_code": _normalize_stock_code(args.get("stock_code")) or None,
        "framework": framework,
        "guidance": "请按该框架调用其他数据工具取数后逐项分析，结论需过 validate 纪律",
        "source": "agent/persona_defs.json（本地方法论框架库）",
    }


# 情绪层工具名 → 处理器（agent/sentiment.py，BettaFish 灵感社交情绪层）
_SENTIMENT_IMPL = {
    "get_market_sentiment": _handle_get_market_sentiment,
    "get_stock_sentiment": _handle_get_stock_sentiment,
}

# 技术分析工具名 → 处理器（agent/technical.py，确定性技术分析层）
_TECH_IMPL = {
    "get_technical_analysis": _handle_get_technical_analysis,
}

# 投资人格工具名 → 处理器（agent/personas.py，Fincept 灵感人格框架库）
_PERSONA_IMPL = {
    "analyze_with_persona": _handle_analyze_with_persona,
}


# ── 社媒舆情工具分发（social_media 门面，微舆 BettaFish 式爬取层）──

def _get_social_media_module():
    """惰性解析 agent/social_media.py（社媒舆情门面），失败返回 None。"""
    return _lazy_import_module("social_media")


def _get_social_bilibili_module():
    """惰性解析 agent/social_bilibili.py（B 站评论能力），失败返回 None。"""
    return _lazy_import_module("social_bilibili")


_SOCIAL_COMMENT_TOP_N = 3   # with_comments=true 时只对前 3 条 B 站结果取评论
_SOCIAL_COMMENT_LIMIT = 10  # 每条 B 站帖子取评论条数（常量）
_SOCIAL_CONTENT_MAX = 200   # 帖子/评论正文截断长度（防上下文爆炸）


def _slim_social_post(post) -> dict:
    """帖子瘦身：只保留 platform/title/metrics/url/published_at/source，
    content 截 200 字；非 dict 输入返回 None（调用方过滤）。绝不抛。"""
    if not isinstance(post, dict):
        return None
    metrics = post.get("metrics")
    return {
        "platform": str(post.get("platform") or ""),
        "title": str(post.get("title") or ""),
        "metrics": metrics if isinstance(metrics, dict) else {},
        "url": str(post.get("url") or ""),
        "published_at": str(post.get("published_at") or ""),
        "source": str(post.get("source") or ""),
        "content": str(post.get("content") or "")[:_SOCIAL_CONTENT_MAX],
    }


def _slim_social_comment(comment) -> dict:
    """评论瘦身：保留 platform/post_id/author/likes/published_at，content 截 200 字。"""
    if not isinstance(comment, dict):
        return None
    return {
        "platform": str(comment.get("platform") or ""),
        "post_id": str(comment.get("post_id") or ""),
        "author": str(comment.get("author") or ""),
        "likes": comment.get("likes") if isinstance(comment.get("likes"), int) else 0,
        "published_at": str(comment.get("published_at") or ""),
        "content": str(comment.get("content") or "")[:_SOCIAL_CONTENT_MAX],
    }


def _social_enrich(social, items):
    """对帖子（可含评论）做情感聚合 + 股票关联提取；任一步失败降级为空骨架，
    并把说明追加进 notes。返回 (buzz, stock_mentions, notes)。绝不抛。"""
    notes: list = []
    buzz = {"total": 0, "sentiment": {"利好": 0, "利空": 0, "中性": 0},
            "by_platform": {}, "avg_score": 0.0}
    mentions: dict = {}
    try:
        buzz = social.aggregate_buzz(items)
        if not isinstance(buzz, dict):
            raise TypeError("aggregate_buzz 返回非 dict")
    except Exception as e:  # noqa: BLE001 - 富化步骤绝不抛出
        logger.warning("社媒情感聚合失败（降级空骨架）: %s", e)
        notes.append("情感聚合失败，buzz 为空骨架")
    try:
        mentions = social.extract_stock_mentions(items)
        if not isinstance(mentions, dict):
            raise TypeError("extract_stock_mentions 返回非 dict")
    except Exception as e:  # noqa: BLE001 - 富化步骤绝不抛出
        logger.warning("社媒股票关联提取失败（降级空结果）: %s", e)
        notes.append("股票关联提取失败，stock_mentions 为空")
    return buzz, mentions, notes


def _social_post_cap(result: dict, limit: int) -> int:
    """posts 条数上限 = limit × 平台数（platforms 计数缺失时按 1 个平台兜底）。"""
    platforms = result.get("platforms") if isinstance(result, dict) else None
    n = len(platforms) if isinstance(platforms, dict) and platforms else 1
    return limit * max(1, n)


def _handle_get_social_hot(args: dict) -> dict:
    """get_social_hot 处理器：热榜门面 + buzz 情感聚合 + 股票关联提取。"""
    social = _get_social_media_module()
    if social is None:
        return {"ok": False, "note": "社媒舆情模块不可用（agent/social_media.py 未就绪）"}
    limit = _clamp_int(args.get("limit"), 10, 1, 30)
    platform = _clean_str(args.get("platform")).lower() or "all"
    unsupported = getattr(social, "UNSUPPORTED_PLATFORMS", None) or {}
    if platform in unsupported:
        return {"ok": False, "platform": platform, "note": unsupported[platform]}
    platforms_arg = None if platform == "all" else [platform]
    hot = social.get_hot_all(platforms=platforms_arg, limit=limit)
    if not isinstance(hot, dict):
        return {"ok": False, "note": "社媒热榜门面返回结构异常，本轮无结果"}
    posts = [p for p in (hot.get("posts") or []) if isinstance(p, dict)]
    capped = posts[:_social_post_cap(hot, limit)]
    buzz, mentions, enrich_notes = _social_enrich(social, capped)
    slim_posts = [s for s in (_slim_social_post(p) for p in capped) if s]
    notes = list(hot.get("notes") or []) + enrich_notes
    if not slim_posts:
        notes.append("本轮未抓到社媒热榜内容")
    return {
        "ok": True,
        "date": hot.get("date"),
        "platforms": hot.get("platforms") or {},
        "sources": hot.get("sources") or {},
        "posts": slim_posts,
        "buzz": buzz,
        "stock_mentions": mentions,
        "notes": notes,
    }


def _fetch_bili_comments(posts) -> tuple:
    """对前 3 条有 post_id 的 B 站帖子各取最多 _SOCIAL_COMMENT_LIMIT 条评论。

    返回 (comments, notes)；评论模块缺失/单帖失败均降级记 note，绝不抛。
    """
    notes: list = []
    bili = _get_social_bilibili_module()
    fetch_comments = getattr(bili, "fetch_comments", None) if bili is not None else None
    if not callable(fetch_comments):
        notes.append("B 站评论模块不可用，未取评论")
        return [], notes
    targets = [p for p in posts
               if str(p.get("platform") or "").strip().lower() == "bilibili"
               and str(p.get("post_id") or "").strip()][:_SOCIAL_COMMENT_TOP_N]
    comments: list = []
    for post in targets:
        try:
            batch = fetch_comments(post["post_id"], limit=_SOCIAL_COMMENT_LIMIT)
        except Exception as e:  # noqa: BLE001 - 单帖评论失败降级
            logger.warning("B 站评论抓取失败（跳过该帖）: %s", e)
            notes.append(f"B 站帖子 {post['post_id']} 评论抓取失败，已跳过")
            continue
        if isinstance(batch, list):
            comments.extend(c for c in batch if isinstance(c, dict))
    if targets and not comments:
        notes.append("B 站评论均为空或抓取失败")
    return comments, notes


def _handle_search_social_media(args: dict) -> dict:
    """search_social_media 处理器：关键词搜索 + 可选评论 + buzz + 股票关联。"""
    social = _get_social_media_module()
    if social is None:
        return {"ok": False, "note": "社媒舆情模块不可用（agent/social_media.py 未就绪）"}
    keyword = _clean_str(args.get("keyword"))
    if not keyword:
        raise _ParamError("keyword 不能为空：请输入要搜索的社媒关键词")
    limit = _clamp_int(args.get("limit"), 10, 1, 30)
    result = social.search_all(keyword, limit=limit)
    if not isinstance(result, dict):
        return {"ok": False, "note": "社媒搜索门面返回结构异常，本轮无结果",
                "keyword": keyword}
    posts = [p for p in (result.get("posts") or []) if isinstance(p, dict)]
    capped = posts[:_social_post_cap(result, limit)]
    notes = list(result.get("notes") or [])
    comments: list = []
    if args.get("with_comments") is True:
        comments, comment_notes = _fetch_bili_comments(capped)
        notes.extend(comment_notes)
    scored_items = capped + comments  # buzz 与股票关联对帖子+评论合并计算
    buzz, mentions, enrich_notes = _social_enrich(social, scored_items)
    notes.extend(enrich_notes)
    slim_posts = [s for s in (_slim_social_post(p) for p in capped) if s]
    if not slim_posts:
        notes.append("本次搜索无社媒结果")
    out = {
        "ok": True,
        "keyword": result.get("keyword") or keyword,
        "date": result.get("date"),
        "platforms": result.get("platforms") or {},
        "sources": result.get("sources") or {},
        "posts": slim_posts,
        "buzz": buzz,
        "stock_mentions": mentions,
        "notes": notes,
    }
    if args.get("with_comments") is True:
        out["comments"] = [s for s in (_slim_social_comment(c) for c in comments) if s]
        # 分布化改造：有评论时追加整体情绪分布块（失败只进 notes，不影响主返回）
        depth = _clean_str(args.get("depth")) or "standard"
        out = _append_distribution_block(out, keyword=keyword, depth=depth)
    return out


# 社媒舆情工具名 → 处理器（agent/social_media.py 门面，v2-social 爬取层）
_SOCIAL_IMPL = {
    "get_social_hot": _handle_get_social_hot,
    "search_social_media": _handle_search_social_media,
}


# ── 校园知识库工具分发（campus_kb 检索层 / review_summary 总结层）──
# 与 _SENTIMENT_IMPL 等查表模式一致：惰性解析、绝不抛异常、ok=False 中文降级。

def _get_campus_kb_module():
    """惰性解析 agent/campus_kb.py（校园知识库存储/检索层），失败返回 None。"""
    return _lazy_import_module("campus_kb")


def _get_review_summary_module():
    """惰性解析 agent/review_summary.py（课程点评综合总结层），失败返回 None。"""
    return _lazy_import_module("review_summary")


# 知识库合法来源枚举（与 agent/campus_kb.py 全局契约一致）；
# 非法 source 参数按 None 处理（不过滤来源），绝不报错
_CAMPUS_KB_SOURCES = (
    "sem_handbook", "thucourse_course", "thucourse_review",
    "thucourse_summary", "thubook",
)

# 检索结果正文截断长度（防上下文爆炸）
_CAMPUS_CONTENT_MAX = 1500


def _cut_campus_content(text: str, limit: int = _CAMPUS_CONTENT_MAX) -> str:
    """校园正文截断：句对齐切割——在 limit 内最后一个句末标点（。！？；\n）处截，
    避免把硬事实（时限/费用/比例等）从句子中间拦腰切断。"""
    text = str(text or "")
    if len(text) <= limit:
        return text
    window = text[:limit]
    cut = max(window.rfind(p) for p in ("。", "！", "？", "；", "\n"))
    if cut >= int(limit * 0.5):  # 句末点太靠前才用，否则硬切
        return window[: cut + 1]
    return window

# 库为空/未建库时的中文指引（提示先运行回填脚本）
_CAMPUS_EMPTY_GUIDANCE = (
    "校园知识库为空或尚未建库：请先运行数据回填脚本——"
    "scripts/sem_handbook_ingest.py（选课手册）、"
    "scripts/thucourse_crawler.py（课程信息与学生点评）、"
    "scripts/thubook_ingest.py（书籍笔记），"
    "再运行 scripts/generate_course_summaries.py 生成课程点评总结，完成后重试。"
)


def _normalize_campus_source(value):
    """source 参数归一：命中枚举原样返回，非法值/空值按 None（不限来源）处理。"""
    src = _clean_str(value)
    return src if src in _CAMPUS_KB_SOURCES else None


def _parse_kb_metadata(entry) -> dict:
    """条目 metadata_json 容错解析；任何失败返回 {}。绝不抛。"""
    raw = entry.get("metadata_json") if isinstance(entry, dict) else None
    if isinstance(raw, dict):  # 容忍直接给 dict 的调用方
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _campus_db_empty(kb) -> bool:
    """库为空/未建库判定：stats().total == 0。stats 不可用或异常按非空处理
    （空检索结果另行提示），绝不抛。"""
    stats_fn = getattr(kb, "stats", None)
    if not callable(stats_fn):
        return False
    try:
        st = stats_fn()
    except Exception:  # noqa: BLE001 - 探测失败绝不抛出
        return False
    return isinstance(st, dict) and st.get("total") == 0


def _handle_search_campus_knowledge(args: dict) -> dict:
    """search_campus_knowledge 处理器：关键词检索 + 正文截断 + 空库指引。"""
    query = _clean_str(args.get("query"))
    if not query:
        raise _ParamError("query 不能为空：请输入要检索的校园知识关键词")
    kb = _get_campus_kb_module()
    if kb is None:
        return {"ok": False, "query": query,
                "note": "校园知识库模块不可用（agent/campus_kb.py 未就绪）"}
    search_fn = getattr(kb, "search_kb", None)
    if not callable(search_fn):
        return {"ok": False, "query": query,
                "note": "校园知识库检索接口不可用（search_kb 未实现）"}
    source = _normalize_campus_source(args.get("source"))
    limit = _clamp_int(args.get("limit"), 10, 1, 20)
    results = search_fn(query, source=source, limit=limit)
    if not isinstance(results, list):
        results = []
    if not results:
        if _campus_db_empty(kb):
            return {"ok": False, "query": query, "source": source,
                    "note": _CAMPUS_EMPTY_GUIDANCE}
        return {"ok": True, "query": query, "source": source, "results": [],
                "note": "校园知识库中未检索到相关条目，可换关键词或去掉来源过滤重试"}
    slim = []
    for entry in results:
        if not isinstance(entry, dict):
            continue
        slim.append({
            "source": str(entry.get("source") or ""),
            "title": str(entry.get("title") or ""),
            "content": _cut_campus_content(entry.get("content")),
            "url": str(entry.get("url") or ""),
            "score": entry.get("score"),
        })
    note = "正文为检索片段（截断至 1500 字），引用时请按来源规范标注出处"
    if len(slim) >= limit:
        note += f"；本次命中已达到返回上限（前 {len(slim)} 条），" \
                "枚举类问题如需更全清单请加大 limit 或缩小检索范围"
    return {
        "ok": True,
        "query": query,
        "source": source,
        "results": slim,
        "note": note,
    }


def _summary_from_cached_entry(entry) -> dict:
    """现成 thucourse_summary 条目 → 工具返回结构（结构化字段取 metadata_json）。"""
    meta = _parse_kb_metadata(entry)
    return {
        "ok": True,
        "course_title": str(meta.get("course_title") or entry.get("title") or ""),
        "sqid": str(meta.get("course_sqid") or ""),
        "summary_text": str(entry.get("content") or ""),
        "rating_avg": meta.get("rating_avg"),
        "rating_dist": meta.get("rating_dist")
                       if isinstance(meta.get("rating_dist"), dict) else {},
        "review_count": meta.get("review_count", 0),
        "highlights": meta.get("highlights")
                      if isinstance(meta.get("highlights"), list) else [],
        "method": meta.get("method") or "cached",
        "note": "命中库内现成的点评综合总结",
    }


def _handle_get_course_review_summary(args: dict) -> dict:
    """get_course_review_summary 处理器：现成总结优先，miss 时按课程聚合
    最多点评现场生成并回写知识库，双 miss 给中文提示。"""
    course_query = _clean_str(args.get("course_query"))
    if not course_query:
        raise _ParamError("course_query 不能为空：请输入课程名或课程关键词")
    kb = _get_campus_kb_module()
    if kb is None:
        return {"ok": False, "course_query": course_query,
                "note": "校园知识库模块不可用（agent/campus_kb.py 未就绪）"}
    search_fn = getattr(kb, "search_kb", None)
    if not callable(search_fn):
        return {"ok": False, "course_query": course_query,
                "note": "校园知识库检索接口不可用（search_kb 未实现）"}

    # 1) 现成总结优先
    summaries = search_fn(course_query, source="thucourse_summary", limit=3)
    if isinstance(summaries, list) and summaries:
        top = summaries[0]
        if isinstance(top, dict):
            return _summary_from_cached_entry(top)

    # 2) 无现成总结：取点评按 course_sqid 分组，现场生成
    reviews = search_fn(course_query, source="thucourse_review", limit=50)
    reviews = ([r for r in reviews if isinstance(r, dict)]
               if isinstance(reviews, list) else [])
    if not reviews:
        if _campus_db_empty(kb):
            return {"ok": False, "course_query": course_query,
                    "note": _CAMPUS_EMPTY_GUIDANCE}
        return {
            "ok": False,
            "course_query": course_query,
            "note": f"校园知识库中未找到「{course_query}」相关的课程点评或总结，"
                    "可换课程名/教师名重试；若库尚未回填课程数据，"
                    "请先运行 scripts/thucourse_crawler.py 与 "
                    "scripts/generate_course_summaries.py。",
        }
    groups: dict = {}
    for r in reviews:
        sqid = str(_parse_kb_metadata(r).get("course_sqid") or "").strip() or "unknown"
        groups.setdefault(sqid, []).append(r)
    sqid, group = max(groups.items(), key=lambda kv: len(kv[1]))
    meta0 = _parse_kb_metadata(group[0])
    course_title = str(meta0.get("course_title")
                       or group[0].get("title") or course_query)

    rs = _get_review_summary_module()
    summarize = getattr(rs, "summarize_course_reviews", None) if rs is not None else None
    if not callable(summarize):
        return {"ok": False, "course_title": course_title, "sqid": sqid,
                "note": "点评总结模块不可用（agent/review_summary.py 未就绪）；"
                        f"库内共有 {len(group)} 条该课程原始点评，"
                        "可改用 search_campus_knowledge 检索原文"}
    summary = summarize(course_title, group)
    if not isinstance(summary, dict):
        return {"ok": False, "course_title": course_title, "sqid": sqid,
                "note": "点评总结生成失败（返回结构异常）"}

    # 回写知识库（best-effort：失败不影响本次返回，仅记 note）
    notes = [f"库内无现成总结，基于 {len(group)} 条点评现场生成"]
    build_entry = getattr(rs, "build_summary_entry", None)
    upsert_fn = getattr(kb, "upsert_entries", None)
    if callable(build_entry) and callable(upsert_fn):
        try:
            entry = build_entry(sqid, course_title, summary)
            written = upsert_fn([entry])
            if isinstance(written, int) and written > 0:
                notes[0] += "并已回写知识库"
            else:
                notes.append("总结回写知识库未生效（不影响本次返回）")
        except Exception as e:  # noqa: BLE001 - 回写失败绝不影响返回
            logger.warning("课程总结回写知识库失败: %s", e)
            notes.append(f"总结回写知识库失败（不影响本次返回）: {e}")
    else:
        notes.append("回写接口不可用，总结未落库（不影响本次返回）")

    return {
        "ok": True,
        "course_title": course_title,
        "sqid": sqid,
        "summary_text": str(summary.get("summary_text") or ""),
        "rating_avg": summary.get("rating_avg"),
        "rating_dist": summary.get("rating_dist")
                       if isinstance(summary.get("rating_dist"), dict) else {},
        "review_count": summary.get("review_count", len(group)),
        "highlights": summary.get("highlights")
                      if isinstance(summary.get("highlights"), list) else [],
        "method": summary.get("method") or "fallback",
        "notes": notes,
    }


# 校园知识库工具名 → 处理器（agent/campus_kb.py 检索层 + agent/review_summary.py 总结层）
_CAMPUS_KB_IMPL = {
    "search_campus_knowledge": _handle_search_campus_knowledge,
    "get_course_review_summary": _handle_get_course_review_summary,
}

# 开源灵感模块工具汇总表：execute_tool 查表顺序追加在最后
# （_REPORT_VEC_IMPL → _REPORT_IMPL → _IMPL → _ANALYSIS_IMPL），不改旧条目
_ANALYSIS_IMPL = {**_SENTIMENT_IMPL, **_TECH_IMPL, **_PERSONA_IMPL, **_SOCIAL_IMPL,
                  **_CAMPUS_KB_IMPL}


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

    # 查表顺序：研报全文向量 _REPORT_VEC_IMPL → 研报库 _REPORT_IMPL → 数据层 _IMPL
    # → 开源灵感模块 _ANALYSIS_IMPL；四表都不命中才报未注册
    vec_impl = _REPORT_VEC_IMPL.get(name)
    report_impl = _REPORT_IMPL.get(name)
    impl = vec_impl or report_impl or _IMPL.get(name)
    if not impl:
        handler = _ANALYSIS_IMPL.get(name)
        if handler is not None:
            return _execute_analysis_tool(name, handler, args)
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


def _execute_analysis_tool(name: str, handler, args: dict) -> dict:
    """开源灵感模块工具（_ANALYSIS_IMPL）的执行入口：必填校验 + 绝不抛异常包装。

    handler 返回的 dict 作为 data 原样返回（其内部自带 ok/note 降级语义）；
    handler 抛 _ParamError 转成 ok=False 中文提示。
    """
    missing = [k for k in _required_params(name)
               if k not in args or args[k] in (None, "")]
    if missing:
        return {"ok": False, "error": f"缺少必填参数：{', '.join(missing)}"}
    try:
        data = handler(args)
        return {"ok": True, "data": _json_safe(data)}
    except _ParamError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        logger.warning("分析类工具执行失败 name=%s", name, exc_info=True)
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
    "get_hk_finance_report": "港股公司财报指标（智研港股财报库）",
    "get_hk_fund_flow": "港股个股主力资金流向",
    "get_us_fund_flow": "美股个股当日资金流向",
    "get_us_market_breadth": "美股全市场涨跌分布",
    "get_stock_major_events": "上市公司重大事项（沪深/港股/美股）",
    "get_china_macro": "中国宏观数据（CPI/PMI/M2 等）",
    "get_us_macro": "美国宏观数据（美债/VIX 等）",
    "get_company_profile": "A 股公司基本资料（简介/行业/上市信息）",
    "get_company_managers": "A 股公司高管名单与履历",
    "get_shareholder_count": "股东户数及筹码集中度变化",
    "get_financial_report_full": "完整财报科目明细（默认国际准则，可按报告期）",
    "get_stock_valuation": "个股估值历史（PE/PB 时间序列）",
    "get_lockup_schedule": "限售解禁日程与解禁规模",
    "get_margin_detail": "个股融资融券明细（两融资金动向）",
    "get_block_trades": "个股大宗交易记录（折溢价/营业部）",
    "get_connect_holdings": "沪深港通持股榜（北向持仓排行）",
    "get_fund_info": "基金档案（类型/规模/管理人）",
    "get_fund_networth": "基金净值走势",
    "get_fund_holdings": "基金重仓持股（前十大重仓）",
    "get_fund_dividend": "基金分红记录",
    "get_forex_quote": "外汇汇率行情（货币对如 USDCNY）",
    "get_commodity_futures_list": "商品期货合约列表（四大交易所）",
    "get_hk_special_ranking": "港股特色榜（国企/蓝筹/红筹）",
    "get_us_fund_flow_history": "美股主力资金流向历史（默认 60 日）",
    "search_research_reports": "券商研报检索（评级/目标价/盈利预测）",
    "get_rating_summary": "券商评级聚合（评级分布/目标价区间）",
    "search_report_content": "研报正文全文检索（观点细节/论证逻辑）",
    "get_market_sentiment": "市场社交情绪快照（温度+涨跌停统计+人气榜）",
    "get_stock_sentiment": "个股社交情绪（人气排名趋势+新闻情感分布）",
    "get_technical_analysis": "个股技术分析仪表盘（指标+评分带+护栏）",
    "analyze_with_persona": "投资人格方法论框架（价值/成长/趋势/逆向）",
    "get_social_hot": "社媒热榜舆情（微博/知乎/抖音/B站+情感分布+股票关联）",
    "search_social_media": "社媒关键词搜索（B站+可选评论+情感分布+股票关联）",
    "search_campus_knowledge": "清华校园知识库检索（手册/课程/点评/书籍笔记）",
    "get_course_review_summary": "课程点评综合总结（评分分布+代表性观点）",
}


def get_tool_catalog() -> str:
    """把工具名 + 一句话描述拼成文本，供调试和测试断言。"""
    lines = [f"- {name}：{_SHORT_DESC.get(name, tool['function']['description'].split('。')[0])}"
             for name, tool in
             ((t["function"]["name"], t) for t in TOOL_REGISTRY)]
    return "可用工具目录（共 %d 个）：\n%s" % (len(lines), "\n".join(lines))
