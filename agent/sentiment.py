"""A 股社交情绪采集与打分层（BettaFish 启发的「情绪工程」子系统 · 纯原创实现）。

职责：从免登录公开源采集「人气/资金博弈」类社交情绪信号，并提供确定性的
中文金融新闻情感打分与市场级「情绪温度」合成，供编排层作为工具调用。

数据源（均为东方财富公开接口，代码与文案全部原创）：

1. 人气榜 getAllCurrentList：POST emappdata.eastmoney.com/stockrank/...，
   appId=appId01、globalId=786e4c21-70dc-435a-93bb-38、marketType 空、
   pageNo/pageSize 分页。2026-07-22 实测定案：payload["data"] 直接为 list
   （无嵌套 data.data、无 total 字段），条目 {"sc":"SZ000938","rk":1,
   "hisRc":1}——sc 带 SH/SZ 前缀（正则提取 6 位数字）、无 name 字段、
   hisRc 为排名变化；name 由 fetch_stock_names 批量回填。解析层同时兼容
   嵌套 dict 形态（旧 akshare 公开文档结构）。
2. 个股人气历史 getHisList：同域名 POST。2026-07-22 实测定案：参数必须为
   srcSecurityCode 且带市场前缀（"SH600519" 正常，"600519"/stockCode/
   entityId 均返回 status=-1「srcSecurityCode 股票代码不能为空」）；
   响应 payload["data"] 直接为 list，条目 {"calcTime":"2026-03-25",
   "rank":103}（日期键 calcTime，兼容 d/date；rank 键兼容 rk/rank）。
3. 涨跌停池 push2ex：GET getTopicZTPool（涨停）/ getTopicDTPool（跌停）/
   getTopicZBPool（炸板），公共参数 ut=7eea3edcaed734bea9cbfc24409ed989、
   dpt=wz.ztzt、pagesize、date=YYYYMMDD；条目字段 c=代码、n=名称、
   p=最新价（千分之一）、zdp=涨跌幅、fbt=首次封板时间、lbt=最后封板时间、
   fund=封板资金、hybk=所属行业、lbc=连板数。
   2026-07-22 实测定案：三池结构与解析一致（实测 47 涨停/36 炸板/最高连板 5）。
4. 名称批量回填 push2 ulist：GET push2.eastmoney.com/api/qt/ulist.np/get，
   fields=f12,f14、secids=市场.代码（SH→1.，SZ/BJ→0.，逗号分隔，单批 50），
   data.diff 兼容 list/dict 两种形态。

字段缺失时全部走防御式降级（跳过该条 / 子池置空 + note），绝不抛异常。

工程纪律（对齐 scripts/report_crawler.py）：
- 所有 HTTP 收口于可注入的 http_get / http_post（默认 requests 实现）；
- 限速 ≤1 req/s + 随机抖动（sleep 可注入，测试用 fake）；
- 单源/单分支失败记 warning 并降级返回 note，公开函数绝不向调用方抛异常；
- 所有数据条目带 source 字段标明出处（项目合规：每个数字须有数据块出处）；
- 时钟可注入：测试可 monkeypatch 模块级 _today_ymd / _today_iso；
- 新闻情感默认打分为确定性词典法（原创词典，见 BULL_LEXICON / BEAR_LEXICON），
  不引入 torch/transformers 等重依赖；scorer 可注入自定义函数。
"""

import json
import logging
import random
import re
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:  # pragma: no cover - 依赖缺失时仅默认 HTTP 不可用
    requests = None

logger = logging.getLogger(__name__)

# ── 端点与公共参数 ──

EM_HOT_RANK_URL = "https://emappdata.eastmoney.com/stockrank/getAllCurrentList"
EM_HOT_RANK_HIS_URL = "https://emappdata.eastmoney.com/stockrank/getHisList"
EM_ZT_POOL_URL = "https://push2ex.eastmoney.com/getTopicZTPool"   # 涨停池
EM_DT_POOL_URL = "https://push2ex.eastmoney.com/getTopicDTPool"   # 跌停池
EM_ZB_POOL_URL = "https://push2ex.eastmoney.com/getTopicZBPool"   # 炸板池

EM_APP_ID = "appId01"
EM_GLOBAL_ID = "786e4c21-70dc-435a-93bb-38"
EM_POOL_UT = "7eea3edcaed734bea9cbfc24409ed989"
EM_POOL_DPT = "wz.ztzt"
EM_POOL_PAGE_SIZE = 500          # 涨跌停池单页容量（实测远小于该值，一页拉全）
EM_HOT_RANK_PAGE_SIZE = 100      # 人气榜单页容量（与 akshare 一致）
EM_ULIST_URL = "https://push2.eastmoney.com/api/qt/ulist.np/get"  # 名称批量回填
EM_ULIST_BATCH = 50              # 名称回填单批 secids 上限

DEFAULT_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
DEFAULT_TIMEOUT = 15
DEFAULT_RATE = 1.0    # 同一轮内连续请求间隔下限（秒）：≤1 req/s
DEFAULT_JITTER = 0.3  # 随机抖动上限（秒）

# ── 情绪温度公式权重（确定性公式，见 _calc_temperature）──
# 设计意图：温度 = 基线 + 涨停强度 - 跌停压制 + 封板质量 + 连板高度。
TEMP_BASE = 20.0        # 基线分：市场天然具备一定活跃度
TEMP_W_ZT = 35.0        # 涨停强度权重：涨停家数是最直接的多头情绪
TEMP_ZT_SATURATE = 80   # 涨停家数饱和点（≥80 家视为满分，极端日 ~100+）
TEMP_W_DT = 25.0        # 跌停压制权重（减分项）：恐慌情绪强度
TEMP_DT_SATURATE = 30   # 跌停家数饱和点（≥30 家视为满分压制）
TEMP_W_ZB = 20.0        # 封板质量权重：炸板率越低加分越多（1 - 炸板率）
TEMP_W_LB = 20.0        # 连板高度权重：最高连板代表赚钱效应/接力情绪
TEMP_LB_SATURATE = 6    # 连板高度饱和点（6 板及以上视为满分）

# 温度标签分档边界（含下限）
TEMP_LABELS: Tuple[Tuple[float, str], ...] = (
    (80.0, "亢奋"),
    (60.0, "活跃"),
    (40.0, "中性"),
    (20.0, "低迷"),
    (0.0, "冰点"),
)

# ── 新闻情感词典（原创编撰，词 → 权重；权重 0.5~1.5 表征强度）──
# 利好词典：业绩/订单/资金/政策/价格行为五类多头信号
BULL_LEXICON: Dict[str, float] = {
    "涨停": 1.2, "大涨": 0.8, "利好": 0.8, "突破": 0.7, "创新高": 1.0,
    "业绩增长": 1.0, "扭亏": 0.9, "预增": 1.0, "超预期": 1.2, "翻倍": 0.9,
    "中标": 0.9, "签约": 0.7, "大单": 0.8, "订单饱满": 1.0, "供不应求": 1.0,
    "回购": 0.9, "增持": 0.8, "分红": 0.6, "派息": 0.6, "高送转": 0.7,
    "获批": 0.8, "上市": 0.6, "并购": 0.7, "重组": 0.8, "扩产": 0.7,
    "满产": 0.8, "提价": 0.8, "涨价": 0.7, "放量上涨": 0.8, "净流入": 0.8,
    "反弹": 0.6, "回暖": 0.6, "复苏": 0.7, "景气": 0.7, "上调评级": 1.0,
    "目标价上调": 1.0, "政策扶持": 0.9, "降准": 1.0, "降息": 1.0, "补贴": 0.7,
}
# 利空词典：业绩/监管/资金/风险事件四类空头信号
BEAR_LEXICON: Dict[str, float] = {
    "跌停": 1.2, "大跌": 0.8, "利空": 0.8, "破发": 0.9, "创新低": 1.0,
    "业绩下滑": 1.0, "亏损": 0.9, "预减": 1.0, "低于预期": 1.0, "商誉减值": 1.1,
    "减持": 0.9, "解禁": 0.7, "清仓": 1.0, "抛售": 0.9, "净流出": 0.8,
    "退市": 1.3, "立案": 1.2, "处罚": 1.0, "违规": 0.9, "问询函": 0.7,
    "诉讼": 0.8, "爆雷": 1.3, "违约": 1.2, "债务危机": 1.2, "质押": 0.6,
    "冻结": 0.9, "跳水": 0.9, "闪崩": 1.2, "崩盘": 1.3, "萎缩": 0.7,
    "衰退": 0.8, "产能过剩": 0.8, "下调评级": 1.0, "目标价下调": 1.0,
    "政策收紧": 0.9, "加息": 0.9, "加税": 0.8, "罚款": 0.9, "计提": 0.8,
}

# 情感分数归一化常数：多空权重差达到该值即饱和到 ±1
SENTI_NORM = 3.0
# 情感分类阈值：|score| ≤ 该值视为中性
SENTI_NEUTRAL_THRESHOLD = 0.1

_CODE_RE = re.compile(r"(\d{6})")
_DATE_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ═══════════════════════════════════════════
# 纯函数工具
# ═══════════════════════════════════════════

def _today_ymd() -> str:
    """今日日期 YYYYMMDD（时钟注入点：测试可 monkeypatch 本函数）。"""
    return time.strftime("%Y%m%d")


def _today_iso() -> str:
    """今日日期 YYYY-MM-DD（时钟注入点：测试可 monkeypatch 本函数）。"""
    return time.strftime("%Y-%m-%d")


def _extract_code(value) -> str:
    """从 sc/c 等字段（形如 '1.600519'）正则提取 6 位纯数字代码；失败返回 ''。"""
    if value is None:
        return ""
    m = _CODE_RE.search(str(value))
    return m.group(1) if m else ""


def _to_int(value) -> Optional[int]:
    """宽松 int 转换；非法返回 None，绝不抛出。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _to_float(value) -> Optional[float]:
    """宽松 float 转换；空串、'-'、'--'、非法值返回 None，绝不抛出。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text in ("-", "--", "—"):
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _clean_str(value) -> str:
    """归一字符串字段：None/占位符 → ''，其余 strip。"""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text in ("-", "--") else text


def _clamp(value: float, lo: float, hi: float) -> float:
    """数值截断到 [lo, hi]。"""
    return max(lo, min(hi, value))


# ═══════════════════════════════════════════
# HTTP 基础设施（可注入，对齐 report_crawler 风格）
# ═══════════════════════════════════════════

def _default_http_get(url: str, **kw):
    """生产 GET 实现（requests）；kw 原样透传（params/timeout 等）。"""
    if requests is None:
        raise RuntimeError("requests 库不可用")
    kw.setdefault("timeout", DEFAULT_TIMEOUT)
    headers = dict(kw.pop("headers", None) or {})
    headers.setdefault("User-Agent", DEFAULT_UA)
    return requests.get(url, headers=headers, **kw)


def _default_http_post(url: str, **kw):
    """生产 POST 实现（requests）；kw 原样透传（json/data/timeout 等）。"""
    if requests is None:
        raise RuntimeError("requests 库不可用")
    kw.setdefault("timeout", DEFAULT_TIMEOUT)
    headers = dict(kw.pop("headers", None) or {})
    headers.setdefault("User-Agent", DEFAULT_UA)
    return requests.post(url, headers=headers, **kw)


class _RateGate:
    """单轮限速门：同一轮内首个请求不限速，其后间隔 RATE + random(0, JITTER)。"""

    def __init__(self, sleep: Callable[[float], None],
                 rate: float = DEFAULT_RATE, jitter: float = DEFAULT_JITTER):
        self._sleep = sleep
        self._rate = max(0.0, rate)
        self._jitter = max(0.0, jitter)
        self._used = False

    def wait(self) -> None:
        if not self._used:
            self._used = True
            return
        self._sleep(self._rate + random.uniform(0, self._jitter))


def _resp_json(resp) -> Optional[dict]:
    """从 response 提取 JSON 对象：优先 resp.json()，回退 text/content 解析；
    顶层非 dict 或解析失败返回 None，绝不抛出。"""
    payload = None
    try:
        payload = resp.json()
    except Exception:
        content = getattr(resp, "text", None)
        if content is None:
            raw = getattr(resp, "content", b"")
            content = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
    return payload if isinstance(payload, dict) else None


def _request_json(method: str, url: str, *, gate: _RateGate,
                  http_get=None, http_post=None, tag: str = "", **kw) -> Optional[dict]:
    """统一请求入口：限速 + 状态码检查 + JSON 解析。失败记 warning 返回 None。"""
    gate.wait()
    http = http_post if method == "POST" else http_get
    http = http or (_default_http_post if method == "POST" else _default_http_get)
    try:
        resp = http(url, **kw)
    except Exception as e:
        logger.warning("%s 请求失败 %s: %s", tag, url, e)
        return None
    sc = getattr(resp, "status_code", None)
    if isinstance(sc, int) and sc >= 400:
        logger.warning("%s HTTP %s %s", tag, sc, url)
        return None
    payload = _resp_json(resp)
    if payload is None:
        logger.warning("%s 响应不是合法 JSON 对象：%s", tag, url)
    return payload


# ═══════════════════════════════════════════
# 1. 东方财富人气榜
# ═══════════════════════════════════════════

def _market_prefix(code: str) -> str:
    """6 位代码 → 市场前缀（SH/SZ/BJ）。

    规则（2026-07-22 实测常用段）：60/68/90 开头→SH，00/30/20 开头→SZ，
    4/8/920 开头→BJ；其余拿不准的默认 SH（沪市为存量主体，注释注明）。
    """
    if code.startswith(("60", "68", "90")):
        return "SH"
    if code.startswith(("00", "30", "20")):
        return "SZ"
    if code.startswith(("4", "8", "920")):
        return "BJ"
    return "SH"


def _payload_rows(payload: dict, dict_key: str) -> list:
    """兼容取列表：payload['data'] 直接为 list（实测形态优先），
    或 payload['data'][dict_key] 嵌套 dict 形态（旧公开文档结构）。"""
    data = payload.get("data")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        rows = data.get(dict_key)
        if isinstance(rows, list):
            return rows
    return []


def _parse_hot_rank_items(payload: dict) -> List[dict]:
    """防御式解析人气榜一页条目：字段缺失/代码无法提取/rank 非法的条目跳过。
    实测条目 {"sc":"SZ000938","rk":1,"hisRc":1}：无 name（留空串待回填），
    hisRc（排名变化）有则解析为 rank_change，没有则省略该键。"""
    items: List[dict] = []
    for row in _payload_rows(payload, "data"):
        if not isinstance(row, dict):
            continue
        rank = _to_int(row.get("rk", row.get("rank")))
        code = _extract_code(row.get("sc", row.get("c", row.get("code"))))
        if rank is None or not code:
            continue  # 防御：核心字段缺失跳过该条
        item = {
            "rank": rank,
            "code": code,
            "name": _clean_str(row.get("n", row.get("name"))),
            "source": "eastmoney_hotrank",
        }
        rank_change = _to_int(row.get("hisRc", row.get("rankChange")))
        if rank_change is not None:
            item["rank_change"] = rank_change
        items.append(item)
    return items


def fetch_hot_rank(limit: int = 100, http_post=None, sleep=None) -> List[dict]:
    """抓取东方财富人气榜（POST 分页聚合）。

    返回 [{rank:int, code:str(6位数字), name:str, source:'eastmoney_hotrank',
    (rank_change:int)}]，按 rank 升序截断到 limit。实测无 total 字段，翻页
    终止条件：空页或本页不足 pageSize；嵌套 dict 形态下仍兼容 total 终止。
    抓完后调用 fetch_stock_names 批量回填 name（失败 name 留空串不报错）。
    任何失败记 warning 并返回（可能为空的）列表，绝不抛异常。
    """
    limit = max(1, _to_int(limit) or 100)
    gate = _RateGate(sleep or time.sleep)
    items: List[dict] = []
    page_no = 1
    while len(items) < limit:
        page_size = min(EM_HOT_RANK_PAGE_SIZE, limit - len(items))
        payload = {
            "appId": EM_APP_ID,
            "globalId": EM_GLOBAL_ID,
            "marketType": "",
            "pageNo": page_no,
            "pageSize": page_size,
        }
        resp_payload = _request_json(
            "POST", EM_HOT_RANK_URL, gate=gate, http_post=http_post,
            tag="hotrank", json=payload)
        if resp_payload is None:
            break  # 请求失败：保留已抓到的部分（防御式降级）
        page_items = _parse_hot_rank_items(resp_payload)
        if not page_items:
            break  # 空页终止
        items.extend(page_items)
        # 翻页终止：嵌套形态有 total 时按总数；实测形态无 total，
        # 本页不足 pageSize 即视为最后一页
        data = resp_payload.get("data")
        total = _to_int(data.get("total")) if isinstance(data, dict) else None
        if total is not None:
            if page_no * page_size >= total:
                break
        elif len(page_items) < page_size:
            break
        page_no += 1
    items.sort(key=lambda x: x["rank"])
    items = items[:limit]

    # 名称回填：实测人气榜条目无 name，批量取 f14 补齐（失败留空串不报错）
    missing = [it["code"] for it in items if not it["name"]]
    if missing:
        try:
            names = fetch_stock_names(missing, sleep=sleep)
        except Exception as e:  # 防御：回填异常不影响排名数据
            logger.warning("hotrank 名称回填失败: %s", e)
            names = {}
        for it in items:
            if not it["name"] and names.get(it["code"]):
                it["name"] = names[it["code"]]
    return items


def fetch_stock_names(codes: List[str], http_get=None, sleep=None) -> Dict[str, str]:
    """批量回填股票名称（GET 东财 ulist 行情接口，fields=f12,f14）。

    secid 规则：SH→1.代码，SZ/BJ→0.代码，逗号分隔，单批 EM_ULIST_BATCH(50)；
    data.diff 兼容 list / dict（按 code 索引）两种形态。失败返回已解析部分，
    绝不抛异常。返回 {code: name}。
    """
    names: Dict[str, str] = {}
    norm_codes: List[str] = []
    for c in codes or []:
        code = _extract_code(c)
        if code and code not in norm_codes:
            norm_codes.append(code)
    if not norm_codes:
        return names
    gate = _RateGate(sleep or time.sleep)
    for i in range(0, len(norm_codes), EM_ULIST_BATCH):
        batch = norm_codes[i:i + EM_ULIST_BATCH]
        secids = ",".join(
            f"{'1' if _market_prefix(c) == 'SH' else '0'}.{c}" for c in batch)
        payload = _request_json(
            "GET", EM_ULIST_URL, gate=gate, http_get=http_get,
            tag="stock_names",
            params={"fltt": "2", "invt": "2", "fields": "f12,f14",
                    "secids": secids})
        if payload is None:
            continue  # 单批失败：保留已解析部分继续下一批
        diff = payload.get("data")
        diff = diff.get("diff") if isinstance(diff, dict) else None
        rows = diff if isinstance(diff, list) else (
            list(diff.values()) if isinstance(diff, dict) else [])
        for row in rows:
            if not isinstance(row, dict):
                continue
            code = _extract_code(row.get("f12"))
            name = _clean_str(row.get("f14"))
            if code and name:
                names[code] = name
    return names


# ═══════════════════════════════════════════
# 2. 个股人气历史
# ═══════════════════════════════════════════

def _parse_hot_rank_history(payload: dict, code: str) -> List[dict]:
    """防御式解析人气历史条目：缺日期/rank 的行跳过。
    实测条目 {"calcTime":"2026-03-25","rank":103}：日期键兼容 calcTime/d/date，
    rank 键兼容 rk/rank；data 兼容直 list 与嵌套 dict 两种形态。"""
    items: List[dict] = []
    for row in _payload_rows(payload, "data"):
        if not isinstance(row, dict):
            continue
        date_text = _clean_str(
            row.get("calcTime", row.get("d", row.get("date"))))[:10]
        rank = _to_int(row.get("rk", row.get("rank")))
        if not _DATE_ISO_RE.match(date_text) or rank is None:
            continue  # 防御：核心字段缺失跳过该条
        items.append({"date": date_text, "rank": rank, "code": code})
    return items


def fetch_hot_rank_history(code: str, days: int = 30,
                           http_post=None, sleep=None) -> List[dict]:
    """抓取个股人气历史排名（POST getHisList）。

    2026-07-22 实测：参数必须为 srcSecurityCode 且带市场前缀（SH600519），
    不带前缀或用 stockCode/entityId 均返回 status=-1。
    返回 [{date:'YYYY-MM-DD', rank:int, code:str}]，按日期升序、保留近 days 条；
    失败返回空列表并记 warning，绝不抛异常。
    """
    norm_code = _extract_code(code)
    if not norm_code:
        logger.warning("hotrank_history 非法代码入参: %r", code)
        return []
    days = max(1, _to_int(days) or 30)
    gate = _RateGate(sleep or time.sleep)
    payload = {
        "appId": EM_APP_ID,
        "globalId": EM_GLOBAL_ID,
        "marketType": "",
        "srcSecurityCode": f"{_market_prefix(norm_code)}{norm_code}",
    }
    resp_payload = _request_json(
        "POST", EM_HOT_RANK_HIS_URL, gate=gate, http_post=http_post,
        tag="hotrank_history", json=payload)
    if resp_payload is None:
        return []
    items = _parse_hot_rank_history(resp_payload, norm_code)
    items.sort(key=lambda x: x["date"])
    return items[-days:]


# ═══════════════════════════════════════════
# 3. 涨跌停池（涨停/跌停/炸板）
# ═══════════════════════════════════════════

def _parse_pool_items(payload: dict, *, lianban_default: int = 0) -> List[dict]:
    """防御式解析涨跌停池条目：缺代码的行跳过，数值字段缺失给默认值。"""
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    rows = data.get("pool")
    if not isinstance(rows, list):
        return []
    items: List[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = _extract_code(row.get("c"))
        if not code:
            continue  # 防御：无代码条目跳过
        price_raw = _to_float(row.get("p"))      # 最新价（千分之一）
        items.append({
            "code": code,
            "name": _clean_str(row.get("n")),
            "price": round(price_raw / 1000.0, 3) if price_raw is not None else None,
            "pct": _to_float(row.get("zdp")) or 0.0,
            "seal_amount": _to_float(row.get("fund")) or 0.0,
            "industry": _clean_str(row.get("hybk")),
            "lianban": _to_int(row.get("lbc"))
            if _to_int(row.get("lbc")) is not None else lianban_default,
            "first_seal": _clean_str(row.get("fbt")),
            "last_seal": _clean_str(row.get("lbt")),
        })
    return items


def _fetch_pool(url: str, date_ymd: str, *, gate: _RateGate, http_get,
                tag: str, lianban_default: int = 0) -> Tuple[List[dict], Optional[str]]:
    """抓取单个池子；返回 (items, note)。失败 items=[] 且 note 非空。"""
    params = {
        "ut": EM_POOL_UT,
        "dpt": EM_POOL_DPT,
        "Pageindex": "0",
        "pagesize": str(EM_POOL_PAGE_SIZE),
        "sort": "fbt:asc",
        "date": date_ymd,
    }
    payload = _request_json("GET", url, gate=gate, http_get=http_get,
                            tag=tag, params=params)
    if payload is None:
        return [], f"{tag} 池抓取失败，已降级为空"
    return _parse_pool_items(payload, lianban_default=lianban_default), None


def fetch_limit_up_pools(date: Optional[str] = None, http_get=None, sleep=None) -> dict:
    """抓取东财涨跌停池（涨停/跌停/炸板三池）。

    date 支持 'YYYY-MM-DD' / 'YYYYMMDD'，为空用今日（时钟可注入，
    monkeypatch _today_ymd）。任一子池失败降级为该池空 + notes 说明，绝不抛。
    返回 {date, zt, dt, zb, stats, source, notes}。
    """
    if date:
        digits = re.sub(r"\D", "", str(date))
        date_ymd = digits[:8] if len(digits) >= 8 else _today_ymd()
    else:
        date_ymd = _today_ymd()
    date_iso = f"{date_ymd[:4]}-{date_ymd[4:6]}-{date_ymd[6:8]}"

    gate = _RateGate(sleep or time.sleep)
    notes: List[str] = []
    zt, note = _fetch_pool(EM_ZT_POOL_URL, date_ymd, gate=gate,
                           http_get=http_get, tag="ztpool")
    if note:
        notes.append(note)
    dt, note = _fetch_pool(EM_DT_POOL_URL, date_ymd, gate=gate,
                           http_get=http_get, tag="dtpool")
    if note:
        notes.append(note)
    zb, note = _fetch_pool(EM_ZB_POOL_URL, date_ymd, gate=gate,
                           http_get=http_get, tag="zbpool")
    if note:
        notes.append(note)

    zt_count, dt_count, zb_count = len(zt), len(dt), len(zb)
    denominator = zt_count + zb_count
    broken_rate = round(zb_count / denominator * 100.0, 2) if denominator > 0 else 0.0
    max_lianban = max((it["lianban"] for it in zt), default=0)
    stats = {
        "zt_count": zt_count,
        "dt_count": dt_count,
        "zb_count": zb_count,
        "炸板率": broken_rate,
        "最高连板": max_lianban,
    }
    return {
        "date": date_iso,
        "zt": zt,
        "dt": dt,
        "zb": zb,
        "stats": stats,
        "source": "eastmoney_ztpool",
        "notes": notes,
    }


# ═══════════════════════════════════════════
# 4. 新闻情感打分（确定性词典法）
# ═══════════════════════════════════════════

def _item_text(item: dict) -> str:
    """拼接新闻条目可打分文本：title + summary/content。"""
    parts = [_clean_str(item.get("title")),
             _clean_str(item.get("summary")),
             _clean_str(item.get("content"))]
    return " ".join(p for p in parts if p)


def _default_score_item(item: dict) -> dict:
    """内置词典打分：同一词命中只计一次；score = clamp((多-空)/NORM, -1, 1)。"""
    text = _item_text(item)
    hits: List[str] = []
    bull = bear = 0.0
    for word, weight in BULL_LEXICON.items():
        if word in text:
            hits.append(word)
            bull += weight
    for word, weight in BEAR_LEXICON.items():
        if word in text:
            hits.append(word)
            bear += weight
    score = round(_clamp((bull - bear) / SENTI_NORM, -1.0, 1.0), 4)
    if score > SENTI_NEUTRAL_THRESHOLD:
        label = "利好"
    elif score < -SENTI_NEUTRAL_THRESHOLD:
        label = "利空"
    else:
        label = "中性"
    return {"sentiment": label, "sentiment_score": score, "hits": hits}


def score_news_sentiment(items: List[dict], scorer: Optional[Callable] = None) -> List[dict]:
    """对新闻条目逐条情感打分，返回原条目 + sentiment/sentiment_score/hits。

    scorer 可注入自定义打分函数：scorer(item) -> dict，至少含
    sentiment_score 或 sentiment 之一；缺省字段自动补齐。scorer 抛异常时
    该条降级为中性（绝不抛）。非 dict 条目跳过。
    """
    results: List[dict] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue  # 防御：非字典条目跳过
        scored = dict(item)
        if scorer is not None:
            try:
                custom = scorer(item)
            except Exception as e:
                logger.warning("自定义 scorer 打分失败，降级中性: %s", e)
                custom = None
            if isinstance(custom, dict):
                score = _to_float(custom.get("sentiment_score"))
                score = round(_clamp(score, -1.0, 1.0), 4) if score is not None else 0.0
                label = _clean_str(custom.get("sentiment"))
                if label not in ("利好", "利空", "中性"):
                    label = ("利好" if score > SENTI_NEUTRAL_THRESHOLD
                             else "利空" if score < -SENTI_NEUTRAL_THRESHOLD
                             else "中性")
                hits = custom.get("hits")
                scored.update({
                    "sentiment": label,
                    "sentiment_score": score,
                    "hits": list(hits) if isinstance(hits, (list, tuple)) else [],
                })
            else:
                scored.update({"sentiment": "中性", "sentiment_score": 0.0, "hits": []})
        else:
            scored.update(_default_score_item(item))
        results.append(scored)
    return results


def _news_distribution(items: List[dict]) -> Dict[str, int]:
    """统计已打分条目的情感分布 {利好, 利空, 中性}。"""
    dist = {"利好": 0, "利空": 0, "中性": 0}
    for it in items:
        label = it.get("sentiment")
        if label in dist:
            dist[label] += 1
    return dist


# ═══════════════════════════════════════════
# 5. 市场级情绪快照（含当日进程内缓存）
# ═══════════════════════════════════════════

_MARKET_CACHE: Dict[str, dict] = {}
_CACHE_LOCK = threading.Lock()


def _clear_market_cache() -> None:
    """清空市场情绪缓存（测试/调试用私有助手）。"""
    with _CACHE_LOCK:
        _MARKET_CACHE.clear()


def _calc_temperature(stats: dict) -> float:
    """情绪温度确定性公式（0-100，权重为模块常量）：

    temperature = clamp(
        TEMP_BASE                                   # 基线 20
        + TEMP_W_ZT  * min(zt_count / 80, 1)        # 涨停强度，至多 +35
        - TEMP_W_DT  * min(dt_count / 30, 1)        # 跌停压制，至多 -25
        + TEMP_W_ZB  * (1 - 炸板率 / 100)            # 封板质量，至多 +20
        + TEMP_W_LB  * min(最高连板 / 6, 1),         # 连板高度，至多 +20
        0, 100)
    """
    zt = max(0, _to_int(stats.get("zt_count")) or 0)
    dt = max(0, _to_int(stats.get("dt_count")) or 0)
    broken = _clamp(_to_float(stats.get("炸板率")) or 0.0, 0.0, 100.0)
    lianban = max(0, _to_int(stats.get("最高连板")) or 0)
    temp = (TEMP_BASE
            + TEMP_W_ZT * min(zt / TEMP_ZT_SATURATE, 1.0)
            - TEMP_W_DT * min(dt / TEMP_DT_SATURATE, 1.0)
            + TEMP_W_ZB * (1.0 - broken / 100.0)
            + TEMP_W_LB * min(lianban / TEMP_LB_SATURATE, 1.0))
    return round(_clamp(temp, 0.0, 100.0), 1)


def _temperature_label(temperature: float) -> str:
    """温度 → 标签分档（冰点/低迷/中性/活跃/亢奋）。"""
    for threshold, label in TEMP_LABELS:
        if temperature >= threshold:
            return label
    return "冰点"


def get_market_sentiment(date: Optional[str] = None, http_get=None, http_post=None,
                         sleep=None, scorer: Optional[Callable] = None,
                         news_items: Optional[List[dict]] = None) -> dict:
    """市场级情绪快照：涨跌停池统计 + 人气榜 top + 注入新闻情感分布。

    情绪温度 0-100 由 _calc_temperature 的确定性加权公式合成。进程内当日
    缓存（dict + 锁，时钟可注入 _today_iso）：同日重复调用直接命中缓存，
    不再发起 HTTP。任何子源失败降级并在 notes 说明，绝不抛异常。
    """
    if date:
        digits = re.sub(r"\D", "", str(date))
        date_iso = (f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
                    if len(digits) >= 8 else _today_iso())
    else:
        date_iso = _today_iso()

    with _CACHE_LOCK:
        cached = _MARKET_CACHE.get(date_iso)
    if cached is not None:
        return dict(cached)

    notes: List[str] = []
    sources: List[str] = []

    # 子源 1：涨跌停池
    try:
        pools = fetch_limit_up_pools(date_iso, http_get=http_get, sleep=sleep)
        stats = pools.get("stats", {})
        notes.extend(pools.get("notes", []))
        sources.append("eastmoney_ztpool")
    except Exception as e:  # 防御：子源异常降级
        logger.warning("get_market_sentiment 涨跌停池失败: %s", e)
        stats = {"zt_count": 0, "dt_count": 0, "zb_count": 0,
                 "炸板率": 0.0, "最高连板": 0}
        notes.append("涨跌停池异常，统计降级为空")

    # 子源 2：人气榜 top10 异动
    hot_rank_top: List[dict] = []
    try:
        hot_rank_top = fetch_hot_rank(limit=10, http_post=http_post, sleep=sleep)
        if hot_rank_top:
            sources.append("eastmoney_hotrank")
        else:
            notes.append("人气榜为空或抓取失败，已降级")
    except Exception as e:
        logger.warning("get_market_sentiment 人气榜失败: %s", e)
        notes.append("人气榜异常，已降级为空")

    # 子源 3：注入新闻情感分布
    news_dist = {"利好": 0, "利空": 0, "中性": 0}
    if news_items:
        try:
            scored = score_news_sentiment(news_items, scorer=scorer)
            news_dist = _news_distribution(scored)
            sources.append("news_injected")
        except Exception as e:
            logger.warning("get_market_sentiment 新闻打分失败: %s", e)
            notes.append("新闻情感打分异常，分布降级为空")

    temperature = _calc_temperature(stats)
    snapshot = {
        "date": date_iso,
        "temperature": temperature,
        "temperature_label": _temperature_label(temperature),
        "stats": stats,
        "hot_rank_top": hot_rank_top,
        "news_sentiment": news_dist,
        "sources": sources,
        "notes": notes,
    }
    with _CACHE_LOCK:
        _MARKET_CACHE[date_iso] = dict(snapshot)
    return snapshot


# ═══════════════════════════════════════════
# 6. 个股情绪
# ═══════════════════════════════════════════

def get_stock_sentiment(code: str, days: int = 30, http_post=None,
                        scorer: Optional[Callable] = None,
                        news_items: Optional[List[dict]] = None) -> dict:
    """个股情绪：人气当前排名 + 历史排名变化 + 注入新闻情感分布。

    trend 判定：latest < history_avg → '上升'（排名数值变小=人气走高），
    latest > history_avg → '下降'，相等 → '平稳'；数据不足 → '未知'。
    任何子源失败降级并在 notes 说明，绝不抛异常。
    """
    norm_code = _extract_code(code)
    notes: List[str] = []
    sources: List[str] = []
    if not norm_code:
        return {
            "code": _clean_str(code),
            "hot_rank": {"latest": None, "history_avg": None, "trend": "未知"},
            "news_sentiment": {"利好": 0, "利空": 0, "中性": 0},
            "sources": [],
            "notes": ["非法股票代码入参，已降级"],
        }

    # 当前人气排名（人气榜前 100 内查找）
    latest_rank: Optional[int] = None
    try:
        board = fetch_hot_rank(limit=100, http_post=http_post)
        for it in board:
            if it["code"] == norm_code:
                latest_rank = it["rank"]
                break
        if board:
            sources.append("eastmoney_hotrank")
        if latest_rank is None:
            notes.append("个股未进入人气榜前 100 或榜单抓取失败")
    except Exception as e:
        logger.warning("get_stock_sentiment 人气榜失败: %s", e)
        notes.append("人气榜异常，当前排名降级为空")

    # 历史排名变化
    history: List[dict] = []
    history_avg: Optional[float] = None
    try:
        history = fetch_hot_rank_history(norm_code, days=days, http_post=http_post)
        if history:
            history_avg = round(sum(h["rank"] for h in history) / len(history), 2)
            if "eastmoney_hotrank" not in sources:
                sources.append("eastmoney_hotrank")
            if latest_rank is None:
                latest_rank = history[-1]["rank"]  # 回退：用历史最新一条
        else:
            notes.append("人气历史为空或抓取失败，已降级")
    except Exception as e:
        logger.warning("get_stock_sentiment 人气历史失败: %s", e)
        notes.append("人气历史异常，已降级为空")

    if latest_rank is not None and history_avg is not None:
        if latest_rank < history_avg:
            trend = "上升"
        elif latest_rank > history_avg:
            trend = "下降"
        else:
            trend = "平稳"
    else:
        trend = "未知"

    # 新闻情感分布
    news_dist = {"利好": 0, "利空": 0, "中性": 0}
    if news_items:
        try:
            scored = score_news_sentiment(news_items, scorer=scorer)
            news_dist = _news_distribution(scored)
            sources.append("news_injected")
        except Exception as e:
            logger.warning("get_stock_sentiment 新闻打分失败: %s", e)
            notes.append("新闻情感打分异常，分布降级为空")

    return {
        "code": norm_code,
        "hot_rank": {"latest": latest_rank, "history_avg": history_avg, "trend": trend},
        "news_sentiment": news_dist,
        "sources": sources,
        "notes": notes,
    }
