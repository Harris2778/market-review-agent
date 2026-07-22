#!/usr/bin/env python3
"""研报多源爬虫（研报库 v1 · 发现/元数据层）。

职责：按日期范围从四个免登录公开源抓取券商研报元数据，产出统一 16 字段
记录字典（全局契约 1，见 RECORD_FIELDS / make_record），可注入 upsert 直接
落库，也可离线收集记录。

数据源（全部为列表/元数据级抓取，不碰详情页与 PDF；实测细节以
docs/RESEARCH_LIB_DESIGN.md「数据源全景」节为准，2026-07-22 当日复验）：

1. 东方财富研报中心 reportapi（主，结构化最完整）：
   - GET reportapi.eastmoney.com/report/list，qType=0 个股 / 1 行业 / 2 策略；
     qType=3 实测近 30 天恒 hits=0（疑似下线），仍保留遍历、自然空跑；
   - 翻页终止：data 为空 或 pageNo >= TotalPage（实测 pageNo 越界返回空 data）；
   - ratingChange 实测映射：2=首次覆盖（lastEmRating 恒为空）、3=维持（前后评级
     恒相同）；1/4 按上调/下调处理，未知码按 emRatingValue 数值比较推导
     （东财评级数值越小越看多：买入=1 … 卖出=5）。
2. 慧博投研列表页（补头部券商/大白马覆盖）：
   - /microns_{id}.html 各分类 SSR 表格（#tableList），分页 /microns_{id}_{N}.html；
   - 标题自带完整元数据「券商-个股-代码-标题-YYMMDD」，容错解析，解析不出
     个股则 stock_code/stock_name 置空；只做列表页，不碰详情页与 PDF；
   - /report/{md5}.html 链接为公司公告（非券商研报），一律排除。
3. 洞见研报 api.djyanbao.com（个股元数据索引）：
   - GET /api/report/?page=N&limit=20，信封 {data:{data:[...], meta:{itemCount}}}；
   - 匿名分页上限约 250 条/查询：实测 limit=20 时 page=13 返回 HTTP 401
     {"code":401,"message":"登录以访问更多数据"}，被拒（401/403 或空页）即停；
   - 列表非日期序，日期过滤在本地做，按页抓到上限为止。
4. 证券之星研报频道（增量监控 + 结构化评级）：
   - 五栏目 SSR 列表 /report_list/report1..5.htm（GBK，仅时间戳+标题）；
   - 结构化栏目 /report/data_all.htm（指标速递）/data_ih.htm（评级调高）
     /report/data_fn.htm（首次关注）：代码/简称/机构/评级/目标价/EPS 表格，
     数据行（13 格=无评级变动 / 14 格=含评级变动）+ 摘要行（1 格，前缀为标题）。

通用纪律：
- 默认 ≤1 req/s + 0~0.5s 随机抖动（sleep 可注入，测试用 fake）；
- 证券之星显式 GBK 解码（GB2312 为 GBK 子集，同链路兼容）；慧博 UTF-8 优先；
- 单源/单分支失败记 warning 并继续其他源，核心函数绝不向调用方抛异常；
- 禁止硬编码 3mbang.com 与 data.10jqka.com.cn/report/（图纸已否决的失效域名）；
- User-Agent 使用常见浏览器串。

可测试性：所有 HTTP 访问收口在可注入的 http_get(url, **kw)（默认 requests
实现）；解析函数全部纯函数化（输入 html/json 文本或字节，输出记录列表）。

CLI 用法：
    /usr/local/bin/python3 scripts/report_crawler.py --days 1              # 每日增量
    /usr/local/bin/python3 scripts/report_crawler.py --days 90 -v          # 90 天回填
    /usr/local/bin/python3 scripts/report_crawler.py --sources eastmoney,stockstar
"""

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import time
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

try:
    import requests
except ImportError:  # pragma: no cover - 依赖缺失时仅默认 HTTP 不可用
    requests = None

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - 解析函数会判空降级
    BeautifulSoup = None

logger = logging.getLogger(__name__)

# 保证从项目根可导入 agent 包（脚本可被任意 cwd 调用；仅 CLI 落库路径使用）。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── 全局常量 ──

DEFAULT_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
DEFAULT_RATE = 1.0    # 请求间隔下限（秒）：≤1 req/s
DEFAULT_JITTER = 0.5  # 随机抖动上限（秒）
DEFAULT_TIMEOUT = 15

# 记录字典 16 字段（爬虫 → 存储层，全局契约 1，不得擅自改动）
RECORD_FIELDS: Tuple[str, ...] = (
    "info_code", "title", "org", "author", "publish_date",
    "stock_code", "stock_name", "industry", "rating", "rating_change",
    "eps_this_year", "eps_next_year", "target_price_high", "target_price_low",
    "encode_url", "source",
)

# 字符串字段（缺省补空串）；其余数值字段缺省 None
_STR_FIELDS = ("title", "org", "author", "publish_date", "stock_code",
               "stock_name", "industry", "rating", "rating_change", "encode_url")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# upsert 回调契约：records -> 实际写入/更新条数（int），由存储层/测试注入
UpsertFn = Callable[[List[dict]], int]


# ═══════════════════════════════════════════
# 纯函数工具
# ═══════════════════════════════════════════

def synth_info_code(source: str, title: str, org: str, publish_date: str) -> str:
    """非东财源 info_code 合成规则（全局契约 2）：
    f"{source}:" + sha1(title + org + publish_date)[:16]。"""
    digest = hashlib.sha1((title + org + publish_date).encode("utf-8")).hexdigest()[:16]
    return f"{source}:{digest}"


def make_record(source: str, **fields) -> dict:
    """构造 16 字段记录字典：缺省字符串字段补空串、数值字段补 None；
    info_code 缺省时按契约 2 合成。保证产出记录契约完整。"""
    record: Dict[str, object] = {name: None for name in RECORD_FIELDS}
    for name in _STR_FIELDS:
        record[name] = ""
    record.update(fields)
    record["source"] = source
    if not record.get("info_code"):
        record["info_code"] = synth_info_code(
            source, str(record["title"]), str(record["org"]), str(record["publish_date"]))
    return record


def _to_float(value) -> Optional[float]:
    """字符串/数字 → float；空串、'-'、'--'、None、非法值 → None。绝不抛出。"""
    if value is None:
        return None
    if isinstance(value, bool):
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
    """归一字符串字段：None/占位符（'-'、'--'）→ ''，其余 strip。"""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text in ("-", "--") else text


def _norm_date(value) -> str:
    """入参日期归一为 'YYYY-MM-DD'；支持 date/datetime/字符串前缀。非法返回 ''。"""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    text = (str(value) if value is not None else "").strip()[:10]
    return text if _DATE_RE.match(text) else ""


def _in_date_range(publish_date: str, start: str, end: str) -> bool:
    """publish_date 是否落在 [start, end]（ISO 字符串比较）；空日期一律不通过。"""
    if not publish_date:
        return False
    if start and publish_date < start:
        return False
    if end and publish_date > end:
        return False
    return True


def _decode(content, encodings: Tuple[str, ...]) -> str:
    """按候选编码显式解码字节（GB2312 为 GBK 子集，GBK 链路兼容）；
    str 输入原样返回，None 返回 ''，全部失败用首个编码 errors=replace。"""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    for enc in encodings:
        try:
            return content.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return content.decode(encodings[0], errors="replace")


def _default_http_get(url: str, **kw):
    """生产 HTTP 实现（requests）；kw 原样透传（params/headers/timeout 等）。"""
    if requests is None:
        raise RuntimeError("requests 库不可用")
    kw.setdefault("timeout", DEFAULT_TIMEOUT)
    headers = dict(kw.pop("headers", None) or {})
    headers.setdefault("User-Agent", DEFAULT_UA)
    headers.setdefault("Accept-Language", "zh-CN,zh;q=0.9")
    return requests.get(url, headers=headers, **kw)


# ═══════════════════════════════════════════
# 源抽象基类
# ═══════════════════════════════════════════

class ReportSource(ABC):
    """研报源抽象基类：name 属性 + iter_records(start_date, end_date, upsert=None)
    生成器，产出契约 1 的 16 字段记录字典。

    可注入点（测试全 mock 零网络）：
    - http_get(url, **kw) -> response（需有 status_code / content 或 text）
    - sleep(seconds)：限速休眠；rate/jitter 控制节奏
    iter_records 注入 upsert 可调用时按页批直接落库（并统计写入条数），
    记录照常产出；任何分支失败记 warning 并继续，绝不抛出。
    """

    name = "base"

    def __init__(self, *, http_get=None, sleep=None,
                 rate: float = DEFAULT_RATE, jitter: float = DEFAULT_JITTER):
        self._http_get = http_get or _default_http_get
        self._sleep = sleep or time.sleep
        self._rate = max(0.0, rate)
        self._jitter = max(0.0, jitter)
        self.stats = {"pages": 0, "fetched": 0, "upserted": 0}
        self._gate_used = False

    @abstractmethod
    def iter_records(self, start_date, end_date,
                     upsert: Optional[UpsertFn] = None) -> Iterator[dict]:
        """按 [start_date, end_date]（YYYY-MM-DD，闭区间）产出记录字典。"""
        raise NotImplementedError

    # ── 内部基础设施 ──

    def _reset_run(self) -> None:
        """每轮 iter_records 重置统计与限速门（本轮首个请求不限速）。"""
        self.stats = {"pages": 0, "fetched": 0, "upserted": 0}
        self._gate_used = False

    def _throttle(self) -> None:
        """限速门：同一轮内连续请求间隔 rate + random.uniform(0, jitter) 秒。"""
        if not self._gate_used:
            self._gate_used = True
            return
        delay = self._rate + random.uniform(0, self._jitter)
        logger.debug("%s 限速休眠 %.2fs", self.name, delay)
        self._sleep(delay)

    def _get(self, url: str, **kw):
        """统一 HTTP 入口：限速 + 状态码检查。失败返回 None，绝不抛出。
        status_code 非真实 int 时不做非 200 判定（mock 兼容）。"""
        self._throttle()
        try:
            resp = self._http_get(url, **kw)
        except Exception as e:
            logger.warning("%s 请求失败 %s: %s", self.name, url, e)
            return None
        self.stats["pages"] += 1
        sc = getattr(resp, "status_code", None)
        if isinstance(sc, int) and sc >= 400:
            logger.warning("%s HTTP %s %s（该分支停止）", self.name, sc, url)
            return None
        return resp

    def _get_text(self, url: str, encodings: Tuple[str, ...], **kw) -> Optional[str]:
        """GET 并按源显式解码为文本；失败返回 None。"""
        resp = self._get(url, **kw)
        if resp is None:
            return None
        content = getattr(resp, "content", None)
        if content is None:
            content = getattr(resp, "text", "")
        return _decode(content, encodings)

    def _get_json(self, url: str, **kw) -> Optional[dict]:
        """GET 并解析 JSON 对象；非 2xx/非法 JSON/顶层非对象均返回 None。"""
        resp = self._get(url, **kw)
        if resp is None:
            return None
        content = getattr(resp, "content", None)
        if content is None:
            content = getattr(resp, "text", "")
        text = _decode(content, ("utf-8", "gbk"))
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            logger.warning("%s 响应不是合法 JSON：%s", self.name, url)
            return None
        if not isinstance(payload, dict):
            logger.warning("%s 响应 JSON 顶层非对象：%s", self.name, url)
            return None
        return payload

    def _emit(self, records: List[dict], upsert: Optional[UpsertFn]) -> None:
        """一页记录落库（注入 upsert 时）+ 统计；upsert 异常吞掉并记 warning。"""
        if upsert is not None and records:
            try:
                n = upsert(records)
                if isinstance(n, int) and not isinstance(n, bool):
                    self.stats["upserted"] += n
            except Exception:
                logger.warning("%s upsert 落库失败（%d 条），继续",
                               self.name, len(records), exc_info=True)
        self.stats["fetched"] += len(records)


# ═══════════════════════════════════════════
# 源 1：东方财富研报中心
# ═══════════════════════════════════════════

EM_API = "https://reportapi.eastmoney.com/report/list"
EM_PAGE_SIZE = 50
EM_QTYPES = (0, 1, 2, 3)   # 0 个股 / 1 行业 / 2 策略 / 3 宏观（实测近 30 天恒空）
EM_MAX_PAGES = 200         # 单分类安全上限（90 天回填约 160 页，留足余量）

# 实测映射（2026-07-22，200 篇样本）：2 恒伴随 lastEmRating 为空=首次覆盖，
# 3 恒前后评级一致=维持；1/4 未见样本，按上调/下调处理
_EM_RATING_CHANGE = {1: "上调", 2: "首次覆盖", 3: "维持", 4: "下调"}


def _em_rating_change(item: dict) -> str:
    """东财评级变动：数字码映射优先，缺失/未知码按前后评级推导。无信息返回 ''。"""
    code = item.get("ratingChange")
    cur = _clean_str(item.get("emRatingName"))
    last = _clean_str(item.get("lastEmRatingName"))
    try:
        c = int(code)
    except (TypeError, ValueError):
        c = None
    if c in _EM_RATING_CHANGE:
        return _EM_RATING_CHANGE[c]
    if cur and not last:
        return "首次覆盖"
    if cur and last and cur == last:
        return "维持"
    if cur and last:
        cur_v = _to_float(item.get("emRatingValue"))
        last_v = _to_float(item.get("lastEmRatingValue"))
        if cur_v is not None and last_v is not None and cur_v != last_v:
            return "上调" if cur_v < last_v else "下调"  # 数值越小越看多
    return ""


def _em_authors(item: dict) -> str:
    """author 形如 ['11000408132.曾帅', ...] → '曾帅,彭海兰'；兜底 researcher 字段。"""
    raw = item.get("author")
    names: List[str] = []
    if isinstance(raw, list):
        for a in raw:
            name = str(a).split(".", 1)[-1].strip()
            if name:
                names.append(name)
    elif isinstance(raw, str) and raw.strip():
        names.append(raw.strip())
    if not names:
        researcher = _clean_str(item.get("researcher"))
        if researcher:
            names.append(researcher)
    return ",".join(names)


def parse_eastmoney_payload(payload) -> List[dict]:
    """东财 reportapi 响应 → 记录列表（纯函数）。非法输入返回 []。
    EPS/目标价字符串转 float（空串 → None）；publishDate 截断为 YYYY-MM-DD。"""
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    records = []
    for item in data:
        if not isinstance(item, dict):
            continue
        record = make_record(
            "eastmoney",
            info_code=_clean_str(item.get("infoCode")),  # 契约 2：东财用 infoCode 原值
            title=_clean_str(item.get("title")),
            org=_clean_str(item.get("orgSName")) or _clean_str(item.get("orgName")),
            author=_em_authors(item),
            publish_date=_norm_date(item.get("publishDate")),
            stock_code=_clean_str(item.get("stockCode")),
            stock_name=_clean_str(item.get("stockName")),
            industry=_clean_str(item.get("indvInduName")) or _clean_str(item.get("industryName")),
            rating=_clean_str(item.get("emRatingName")),
            rating_change=_em_rating_change(item),
            eps_this_year=_to_float(item.get("predictThisYearEps")),
            eps_next_year=_to_float(item.get("predictNextYearEps")),
            target_price_high=_to_float(item.get("indvAimPriceT")),
            target_price_low=_to_float(item.get("indvAimPriceL")),
            encode_url=_clean_str(item.get("encodeUrl")),
        )
        if not record["title"]:
            continue  # 无标题记录无法溯源，丢弃
        records.append(record)
    return records


class EastmoneySource(ReportSource):
    """东方财富研报中心：全分类（qType 0-3）翻页抓取。"""

    name = "eastmoney"

    def __init__(self, qtypes=EM_QTYPES, max_pages: int = EM_MAX_PAGES, **kw):
        super().__init__(**kw)
        self._qtypes = tuple(qtypes)
        self._max_pages = max(1, max_pages)

    def iter_records(self, start_date, end_date,
                     upsert: Optional[UpsertFn] = None) -> Iterator[dict]:
        start, end = _norm_date(start_date), _norm_date(end_date)
        self._reset_run()
        for qtype in self._qtypes:
            page = 1
            while page <= self._max_pages:
                payload = self._get_json(EM_API, params={
                    "industryCode": "*", "pageSize": EM_PAGE_SIZE,
                    "pageNo": page, "qType": qtype,
                    "beginTime": start, "endTime": end,
                })
                if payload is None:
                    break  # 请求/解析失败已记 warning，换下一个分类
                parsed = parse_eastmoney_payload(payload)
                if not parsed:
                    break  # 空页（含 pageNo 越界、分类无数据）即停
                batch = [r for r in parsed
                         if _in_date_range(r["publish_date"], start, end)]
                if batch:
                    self._emit(batch, upsert)
                    yield from batch
                total_page = payload.get("TotalPage")
                if not isinstance(total_page, int) or page >= total_page:
                    break
                page += 1


# ═══════════════════════════════════════════
# 源 2：慧博投研列表页
# ═══════════════════════════════════════════

HIBOR_BASE = "https://www.hibor.com.cn"
# 主导航实测分类：公司调研（个股）/ 行业分析 / 投资策略 / 宏观经济 / 晨会早刊
HIBOR_CATEGORIES = ((1, "公司调研"), (2, "行业分析"), (4, "投资策略"),
                    (13, "宏观经济"), (14, "晨会早刊"))
HIBOR_MAX_PAGES = 2  # 每分类抓取页数（列表页即足够覆盖每日增量）
HIBOR_ENCODINGS = ("utf-8", "gbk")

_HIBOR_DATA_HREF_RE = re.compile(r"^/data/[0-9a-f]{32}\.html$")
_HIBOR_META_RE = re.compile(r"^(?P<org>[^-]+)-(?P<body>.+)-(?P<d>\d{6})$")
_HIBOR_STOCK_RE = re.compile(r"^(?P<name>[^-]+)-(?P<code>\d{6})-(?P<title>.+)$")
_HIBOR_AUTHOR_RE = re.compile(r"作者：(.*?)评级")
_A_SHARE_PREFIXES = "02345689"  # 沪深 0/3/6、北交所 4/8/92、及其他 6 位代码段


def _yymmdd_to_date(yymmdd: str) -> str:
    """'260721' → '2026-07-21'；非法（如 261399）返回 ''。"""
    try:
        return datetime.strptime("20" + yymmdd, "%Y%m%d").strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return ""


def parse_hibor_metadata(raw_title: str) -> Optional[dict]:
    """慧博标题元数据容错解析：
    「券商-个股-代码-标题-YYMMDD」（如 东吴证券-璞泰来-603659-…-260720）或
    「券商-标题-YYMMDD」（解析不出个股则 stock_code/stock_name 置空）。
    缺日期尾缀/日期非法/缺机构或标题时返回 None。"""
    text = (raw_title or "").strip()
    m = _HIBOR_META_RE.match(text)
    if not m:
        return None
    publish_date = _yymmdd_to_date(m.group("d"))
    if not publish_date:
        return None
    org = m.group("org").strip()
    body = m.group("body").strip()
    if not org or not body:
        return None
    stock_name, stock_code, title = "", "", body
    ms = _HIBOR_STOCK_RE.match(body)
    if ms and ms.group("code")[0] in _A_SHARE_PREFIXES:
        stock_name = ms.group("name").strip()
        stock_code = ms.group("code")
        title = ms.group("title").strip()
    if not title:
        return None
    return {"org": org, "stock_name": stock_name, "stock_code": stock_code,
            "title": title, "publish_date": publish_date}


def parse_hibor_list(content) -> List[dict]:
    """慧博分类/首页列表 HTML（str 或字节）→ 记录列表（纯函数）。
    只取 /data/{md5}.html 研报链接（title 属性或锚文本自带元数据）；
    /report/{md5}.html 为公司公告一律排除；[详细] 摘要锚不重复产出。
    表格（#tableList）内按行序回填「作者：」元数据。"""
    if BeautifulSoup is None:
        logger.warning("beautifulsoup4 未安装，慧博解析不可用")
        return []
    html = _decode(content, HIBOR_ENCODINGS)
    if not html.strip():
        return []
    soup = BeautifulSoup(html, "html.parser")
    records: List[dict] = []
    seen = set()
    rec_by_anchor: Dict[int, dict] = {}
    for a in soup.find_all("a", href=True):
        if not _HIBOR_DATA_HREF_RE.match(a["href"]):
            continue
        raw = (a.get("title") or a.get_text(strip=True) or "").strip()
        if not raw or raw in ("[详细]", "详细"):
            continue
        meta = parse_hibor_metadata(raw)
        if meta is None:
            continue
        rec = make_record("hibor", **meta)
        if rec["info_code"] in seen:
            continue
        seen.add(rec["info_code"])
        records.append(rec)
        rec_by_anchor[id(a)] = rec
    # 行序回填作者：标题行之后的「…作者：XXX评级：…」元数据行归属于当前记录
    for table in soup.find_all("table"):
        current: Optional[dict] = None
        for tr in table.find_all("tr"):
            title_anchor = None
            for a in tr.find_all("a", href=True):
                if not _HIBOR_DATA_HREF_RE.match(a["href"]):
                    continue
                txt = (a.get("title") or a.get_text(strip=True) or "").strip()
                if txt and txt not in ("[详细]", "详细"):
                    title_anchor = a
                    break
            if title_anchor is not None:
                current = rec_by_anchor.get(id(title_anchor))
                continue
            if current is not None and not current["author"]:
                m = _HIBOR_AUTHOR_RE.search(tr.get_text(" ", strip=True))
                if m:
                    current["author"] = m.group(1).strip()
    return records


class HiborSource(ReportSource):
    """慧博投研：各分类 SSR 列表页抓取（只做列表页，不碰详情页与 PDF）。"""

    name = "hibor"

    def __init__(self, categories=HIBOR_CATEGORIES, max_pages: int = HIBOR_MAX_PAGES, **kw):
        super().__init__(**kw)
        self._categories = tuple(categories)
        self._max_pages = max(1, max_pages)

    def iter_records(self, start_date, end_date,
                     upsert: Optional[UpsertFn] = None) -> Iterator[dict]:
        start, end = _norm_date(start_date), _norm_date(end_date)
        self._reset_run()
        for cat_id, _cat_name in self._categories:
            for page in range(self._max_pages):
                if page == 0:
                    url = f"{HIBOR_BASE}/microns_{cat_id}.html"
                else:
                    url = f"{HIBOR_BASE}/microns_{cat_id}_{page}.html"
                html = self._get_text(url, HIBOR_ENCODINGS)
                if html is None:
                    break  # 请求失败已记 warning，换下一个分类
                parsed = parse_hibor_list(html)
                if not parsed:
                    break  # 空页到底
                batch = [r for r in parsed
                         if _in_date_range(r["publish_date"], start, end)]
                if batch:
                    self._emit(batch, upsert)
                    yield from batch
                # 列表按日期倒序：本页最新一篇已早于 start → 更老的页不必再抓
                newest = max(r["publish_date"] for r in parsed)
                if start and newest < start:
                    break


# ═══════════════════════════════════════════
# 源 3：洞见研报 API
# ═══════════════════════════════════════════

DJY_API = "https://api.djyanbao.com/api/report/"
DJY_PAGE_LIMIT = 20
DJY_MAX_PAGES = 13  # 匿名分页上限约 250 条（limit=20 时 page=13 实测 401）


def parse_djyanbao_payload(payload) -> List[dict]:
    """洞见研报 API 响应 → 记录列表（纯函数）。信封 {data:{data:[...],meta}}。
    列表无评级/EPS/目标价字段，相应字段留空；publishAt 截断为 YYYY-MM-DD。"""
    if not isinstance(payload, dict):
        return []
    inner = payload.get("data")
    if not isinstance(inner, dict):
        return []
    items = inner.get("data")
    if not isinstance(items, list):
        return []
    records = []
    for item in items:
        if not isinstance(item, dict):
            continue
        authors = item.get("authors")
        if isinstance(authors, list):
            authors = ",".join(str(a).strip() for a in authors if str(a).strip())
        record = make_record(
            "djyanbao",
            title=_clean_str(item.get("title")),
            org=_clean_str(item.get("orgName")),
            author=_clean_str(authors),
            publish_date=_norm_date(item.get("publishAt")),
            stock_name=_clean_str(item.get("stockName")),  # 个股结构化字段
            encode_url=_clean_str(item.get("fileUrl")),  # 匿名 403 私有桶，仅作 v2 标识
        )
        if not record["title"]:
            continue
        records.append(record)
    return records


class DjyanbaoSource(ReportSource):
    """洞见研报：匿名 JSON 分页抓取，被拒（401/403 或空页）即停。"""

    name = "djyanbao"

    def __init__(self, query: Optional[str] = None, max_pages: int = DJY_MAX_PAGES, **kw):
        super().__init__(**kw)
        self._query = (query or "").strip() or None
        self._max_pages = max(1, max_pages)

    def iter_records(self, start_date, end_date,
                     upsert: Optional[UpsertFn] = None) -> Iterator[dict]:
        start, end = _norm_date(start_date), _norm_date(end_date)
        self._reset_run()
        for page in range(1, self._max_pages + 1):
            params: Dict[str, object] = {"page": page, "limit": DJY_PAGE_LIMIT}
            if self._query:
                params["q"] = self._query
            payload = self._get_json(DJY_API, params=params)
            if payload is None:
                break  # HTTP 401/403/网络失败：被拒即停（已记 warning）
            code = payload.get("code")
            if isinstance(code, int) and code in (401, 403):
                logger.warning("%s 匿名访问被拒（code=%s %s），停止翻页",
                               self.name, code, payload.get("message"))
                break
            parsed = parse_djyanbao_payload(payload)
            if not parsed:
                break  # 空页到底
            # 列表非日期序，无法按日期提前终止；本地过滤后按页抓到上限
            batch = [r for r in parsed
                     if _in_date_range(r["publish_date"], start, end)]
            if batch:
                self._emit(batch, upsert)
                yield from batch


# ═══════════════════════════════════════════
# 源 4：证券之星研报频道
# ═══════════════════════════════════════════

STOCKSTAR_BASE = "https://stock.stockstar.com"
STOCKSTAR_ENCODINGS = ("gbk", "gb18030", "utf-8")  # 页面为 GBK（GB2312 子集兼容）
# 五栏目 SSR 列表：公司研究 / 行业研究 / 策略趋势 / 券商晨会 / 宏观研究
STOCKSTAR_COLUMNS = (("report1", "公司研究"), ("report2", "行业研究"),
                     ("report3", "策略趋势"), ("report4", "券商晨会"),
                     ("report5", "宏观研究"))
# 结构化栏目：指标速递 / 评级调高 / 首次关注（评级/目标价/EPS 全字段）
STOCKSTAR_DATA_PAGES = ("data_all", "data_ih", "data_fn")

_SS_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})(?:\s+\d{2}:\d{2}(?::\d{2})?)?")


def parse_stockstar_columns(content) -> List[dict]:
    """证券之星五栏目列表 HTML（str 或 GBK 字节）→ 记录列表（纯函数）。
    条目形态 <li><span>2026-07-21 21:53:00</span><a href=...shtml>标题</a></li>；
    列表无机构/个股字段，相应字段留空；时间戳截断为日期。"""
    if BeautifulSoup is None:
        logger.warning("beautifulsoup4 未安装，证券之星解析不可用")
        return []
    html = _decode(content, STOCKSTAR_ENCODINGS)
    if not html.strip():
        return []
    soup = BeautifulSoup(html, "html.parser")
    records: List[dict] = []
    seen = set()
    for li in soup.find_all("li"):
        a = li.find("a", href=True)
        if a is None or ".shtml" not in a["href"]:
            continue
        m = _SS_TS_RE.search(li.get_text(" ", strip=True))
        if not m:
            continue
        title = a.get_text(strip=True)
        if not title:
            continue
        rec = make_record("stockstar", title=title, publish_date=m.group(1))
        if rec["info_code"] in seen:
            continue
        seen.add(rec["info_code"])
        records.append(rec)
    return records


def parse_stockstar_data_table(content) -> List[dict]:
    """证券之星结构化栏目（指标速递/评级调高/首次关注）HTML → 记录列表（纯函数）。
    表格行对：数据行（13 格=无评级变动 / 14 格=含评级变动：序号/代码/简称/机构/
    评级/[评级变动]/目标价/收盘价/预期涨幅/本年EPS/次年EPS/后年EPS/日期/摘要）
    + 摘要行（1 格，文本前缀为报告标题，后接 '简称(代码)'）。
    '-' 占位 → 空串/None；目标价为单值时高低价同值（零宽度区间）。"""
    if BeautifulSoup is None:
        logger.warning("beautifulsoup4 未安装，证券之星解析不可用")
        return []
    html = _decode(content, STOCKSTAR_ENCODINGS)
    if not html.strip():
        return []
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return []
    rows = table.find_all("tr")
    records: List[dict] = []
    i = 0
    while i < len(rows):
        cells = [c.get_text(strip=True) for c in rows[i].find_all(["td", "th"])]
        n = len(cells)
        if n in (13, 14) and cells[0].isdigit() and re.fullmatch(r"\d{6}", cells[1]):
            off = 1 if n == 14 else 0  # 14 格含「评级变动」列，后续列右移 1
            publish_date = _norm_date(cells[11 + off])
            if not publish_date:
                i += 1
                continue
            title = ""
            consumed = 1
            if i + 1 < len(rows):
                nxt = rows[i + 1].find_all(["td", "th"])
                if len(nxt) == 1:
                    consumed = 2
                    summary = nxt[0].get_text(strip=True)
                    marker = f"{cells[2]}({cells[1]})"
                    if marker in summary:
                        title = summary.split(marker, 1)[0].strip()
                    else:
                        title = summary[:120].strip()  # 标记缺失时截断兜底
            target = _to_float(cells[5 + off])
            rec = make_record(
                "stockstar",
                title=title,
                org=_clean_str(cells[3]),
                publish_date=publish_date,
                stock_code=cells[1],
                stock_name=_clean_str(cells[2]),
                rating=_clean_str(cells[4]),
                rating_change=_clean_str(cells[5]) if off else "",
                target_price_high=target,
                target_price_low=target,
                eps_this_year=_to_float(cells[8 + off]),   # 表头动态年份，前两列依次对应本年/次年
                eps_next_year=_to_float(cells[9 + off]),
            )
            records.append(rec)
            i += consumed
            continue
        i += 1
    return records


class StockstarSource(ReportSource):
    """证券之星：五栏目增量列表 + 三个结构化栏目（评级/目标价/EPS）。"""

    name = "stockstar"

    def __init__(self, columns=STOCKSTAR_COLUMNS, data_pages=STOCKSTAR_DATA_PAGES, **kw):
        super().__init__(**kw)
        self._columns = tuple(columns)
        self._data_pages = tuple(data_pages)

    def iter_records(self, start_date, end_date,
                     upsert: Optional[UpsertFn] = None) -> Iterator[dict]:
        start, end = _norm_date(start_date), _norm_date(end_date)
        self._reset_run()
        for col_id, _col_name in self._columns:
            url = f"{STOCKSTAR_BASE}/report_list/{col_id}.htm"
            html = self._get_text(url, STOCKSTAR_ENCODINGS)
            if html is None:
                continue  # 单栏目失败记 warning（_get 内），继续其他栏目
            batch = [r for r in parse_stockstar_columns(html)
                     if _in_date_range(r["publish_date"], start, end)]
            if batch:
                self._emit(batch, upsert)
                yield from batch
        for page_id in self._data_pages:
            url = f"{STOCKSTAR_BASE}/report/{page_id}.htm"
            html = self._get_text(url, STOCKSTAR_ENCODINGS)
            if html is None:
                continue
            batch = [r for r in parse_stockstar_data_table(html)
                     if _in_date_range(r["publish_date"], start, end)]
            if batch:
                self._emit(batch, upsert)
                yield from batch


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

SOURCES: Dict[str, type] = {
    "eastmoney": EastmoneySource,
    "hibor": HiborSource,
    "djyanbao": DjyanbaoSource,
    "stockstar": StockstarSource,
}


def _load_report_library():
    """惰性导入存储层（仅 CLI 落库路径；测试 monkeypatch 本函数注入 fake，
    不对 agent.report_library 实体形成硬依赖）。失败返回 None。"""
    try:
        from agent import report_library
        return report_library
    except Exception:
        logger.warning("agent.report_library 导入失败", exc_info=True)
        return None


def main(argv=None, *, http_get=None, sleep=None) -> int:
    """研报爬虫 CLI：按 --days 回溯窗口抓取各源并落库，结尾打印各源统计。
    http_get/sleep 仅供测试注入；生产走默认 requests 与 time.sleep。"""
    parser = argparse.ArgumentParser(
        description="研报多源爬虫：抓取券商研报元数据入库（SQLite）")
    parser.add_argument("--days", type=int, default=1,
                        help="回溯天数（默认 1=仅当日；回填用 90）")
    parser.add_argument("--sources", default=",".join(SOURCES),
                        help="逗号分隔源列表（默认全部：eastmoney,hibor,djyanbao,stockstar）")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE,
                        help="请求间隔下限秒数（默认 1.0，另加 0~0.5s 随机抖动）")
    parser.add_argument("--db-path", default=None,
                        help="SQLite 路径（缺省走存储层解析：显式 > REPORTS_DB_PATH "
                             "> ${DATA_DIR:-data}/reports.db）")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出 DEBUG 日志")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    names = [n.strip() for n in (args.sources or "").split(",") if n.strip()]
    for n in [n for n in names if n not in SOURCES]:
        logger.warning("未知数据源：%s（可用：%s）", n, ",".join(SOURCES))
    names = [n for n in names if n in SOURCES]
    if not names:
        print("无有效数据源，终止。")
        return 2

    lib = _load_report_library()
    if lib is None:
        print("存储层 agent.report_library 不可用（详见日志），终止。")
        return 1
    try:
        db_path = lib.init_db(args.db_path)
    except Exception:
        logger.warning("研报库初始化失败", exc_info=True)
        print("研报库初始化失败（详见日志），终止。")
        return 1

    days = max(1, args.days)
    end_d = date.today()
    start_d = end_d - timedelta(days=days - 1)
    start, end = start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d")
    print(f"抓取区间：{start} ~ {end}（{days} 天），源：{','.join(names)}，库：{db_path}")

    total_fetched = total_written = 0
    for name in names:
        source = SOURCES[name](http_get=http_get, sleep=sleep, rate=args.rate)
        try:
            def _upsert(records, _lib=lib, _db=db_path):
                return _lib.upsert_reports(records, _db)

            for _ in source.iter_records(start, end, upsert=_upsert):
                pass
            st = source.stats
            total_fetched += st["fetched"]
            total_written += st["upserted"]
            print(f"[{name}] 抓取 {st['fetched']} 篇，入库 {st['upserted']} 篇，"
                  f"请求 {st['pages']} 次")
        except Exception:
            # 源实现按 fail-safe 设计，正常到不了这里；双保险继续其他源
            logger.warning("源 %s 抓取异常（继续其他源）", name, exc_info=True)
            print(f"[{name}] 抓取失败（详见日志），已跳过")
    print(f"合计：抓取 {total_fetched} 篇，入库 {total_written} 篇。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
