#!/usr/bin/env python3
"""研报全文获取层（研报库 v2 · 全文层，Worker D）。

职责：对 reports 表（v1 元数据层）中尚未有全文的研报，从两个免登录源
抓取全文、解析分节，写入 report_fulltext 表（全局契约 1/2），供
agent/report_vectors.py 建向量索引消费。

两个全文源（实测细节见 docs/RESEARCH_LIB_DESIGN.md「数据源全景」节）：

1. 东财 PDF（主）：直链用 **info_code**（reports 表主键，纯字母数字）拼接
   ——akshare stock_research_report_em 源码实测拼法（2026-07-22 复验，
   直接 200 拿到 492KB 真实 PDF，无挑战）：
       https://pdf.dfcfw.com/pdf/H3_{info_code}_1.pdf
   - ⚠️ 切勿用 encode_url 拼路径（P0 修复 2026-07-22）：encode_url 是
     base64 风格串常含 /，原样拼入路径被 Tomcat 当多段路径 404，
     quote(%2F) 又被 Tomcat 安全策略（allowEncodedSlash=false）直接
     400 Bad Request（实测 400 响应体为 Tomcat 默认错误页，非 WAF）——
     两条路都是死路；info_code 无特殊字符，天然免疫；
   - 服务端（Tencent EdgeOne）对异常路径/高频访问会触发 EO_Bot JS 挑战
     （~1KB <script> 页，Content-Type 伪装 application/pdf）：JS 内嵌动态
     常量，浏览器执行后设置两个 cookie 并重载：
         __tst_status = WTKkN + bOYDu + wyeCN（三数之和，实测恒 4011260683）
         EO_Bot_Ssid = n() 内 iTyzs(t, N) 的 N（每期动态）
     本模块保留正则求解 + cookie/Referer 重取作为兜底
     （solve_eo_bot_challenge），配合 EM_FETCH_MAX_ATTEMPTS 轮有限重试；
     持续失败记 warning 跳过，由新浪通道兜底，绝不抛出。
   - PDF 用 pymupdf 解析（惰性导入，模块级不引入；缺失时 PDF 通道整体降级），
     全程内存解析（fitz.open(stream=...)），解析完即弃不落地。

2. 新浪研报网页全文（辅）：列表页（GB2312 SSR）拿 rptid 与标题/机构/日期，
   详情页 vReport_Show/kind/lastest/rptid/{rptid}/index.phtml 正文在
   <div class="blk_container"> 内（<br> 分段、&nbsp;/全角空格缩进），
   按研报惯用节名（投资要点/盈利预测/风险提示等）分节；
   与 reports 表候选按归一化标题匹配（精确相等 → 互相包含兜底）。

通用纪律：
- 两源各自独立限速门：≤1 req/s + 0~0.5s 随机抖动（sleep 可注入，
  每源本轮首个请求不限速）；
- 新浪显式 GB2312 解码（GB2312 为 GBK 子集，解码链 GBK 兼容）；
- 单篇失败记 warning 跳过，不拖垮批次；公开函数绝不向调用方抛异常；
- 路径解析一律复用 agent.report_library._db_path（惰性导入，契约 7）。

输出契约（全局契约 1/2）：
- 表：report_fulltext(info_code TEXT PRIMARY KEY, source TEXT,
  fulltext TEXT, sections_json TEXT, fetched_at TEXT)，与 reports 同库；
- 记录：{"info_code","source","fulltext","sections":[{"name","text"}...],
  "fetched_at"}；sections_json = json.dumps(sections, ensure_ascii=False)；
- source 取全文来源（"eastmoney" / "sina"），与 reports 行源可能不同；
- 分节解析不出结构时退化为单节「正文」。

CLI 用法：
    /usr/local/bin/python3 scripts/report_fulltext.py --days 30
    /usr/local/bin/python3 scripts/report_fulltext.py --days 7 --limit 20 -v
"""

import argparse
import json
import logging
import os
import random
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import quote

try:
    import requests
except ImportError:  # pragma: no cover - 依赖缺失时仅默认 HTTP 不可用
    requests = None

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - 新浪列表解析会判空降级
    BeautifulSoup = None

logger = logging.getLogger(__name__)

# 保证从项目根可导入 agent 包（脚本可被任意 cwd 调用）。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── 全局常量 ──

DEFAULT_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
DEFAULT_RATE = 1.0    # 请求间隔下限（秒）：≤1 req/s
DEFAULT_JITTER = 0.5  # 随机抖动上限（秒）
DEFAULT_TIMEOUT = 20
DEFAULT_DAYS = 30

FULLTEXT_TABLE = "report_fulltext"
DEFAULT_SOURCE = "eastmoney"

_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {FULLTEXT_TABLE} (
  info_code TEXT PRIMARY KEY,
  source TEXT,
  fulltext TEXT,
  sections_json TEXT,
  fetched_at TEXT
);
"""

# 东财 PDF 直链（akshare 源码实测拼法：用 infoCode 而非 encodeUrl，2026-07-22
# 复验直接 200 拿真实 PDF；encodeUrl 含 / 时 Tomcat 404/400 双死路，勿用）
EM_PDF_URL = "https://pdf.dfcfw.com/pdf/H3_{info_code}_1.pdf"
# 单篇东财抓取最大尝试轮数（WAF 多节点行为随机，重试换节点提高命中；
# 每轮 = 裸 GET + 可能的挑战重取一次，全部经限速门 ≤1 req/s）
EM_FETCH_MAX_ATTEMPTS = 3

# 新浪研报中心（GB2312 SSR）
SINA_LIST_URL = ("https://vip.stock.finance.sina.com.cn/q/go.php/"
                 "vReport_List/kind/lastest/index.phtml?p={page}")
SINA_DETAIL_URL = ("https://stock.finance.sina.com.cn/stock/go.php/"
                   "vReport_Show/kind/lastest/rptid/{rptid}/index.phtml")
SINA_ENCODINGS = ("gb2312", "gbk", "gb18030", "utf-8")
SINA_LIST_MAX_PAGES = 5  # 列表翻页上限（40 条/页 ≈ 200 篇，覆盖数日增量）

# 研报惯用节名（长名在前避免短名抢先匹配）；行首匹配，可带冒号
_SECTION_NAMES = (
    "盈利预测与投资建议", "盈利预测及投资建议", "盈利预测与估值",
    "投资要点", "核心观点", "核心要点", "投资摘要", "报告摘要",
    "盈利预测", "投资建议", "风险提示", "事件点评", "事件", "摘要", "正文",
)
_SECTION_ONLY_RE = re.compile(
    r"^(" + "|".join(_SECTION_NAMES) + r")\s*[:：]?\s*$")
_SECTION_INLINE_RE = re.compile(
    r"^(" + "|".join(_SECTION_NAMES) + r")\s*[:：]\s*(.+)$")

_SINA_RPTID_RE = re.compile(r"/vReport_Show/kind/(\w+)/rptid/(\d+)/")
_SINA_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# EO_Bot 挑战页常量提取（2026-07-22 实测三份样本字段名稳定）
_EO_SUM_KEYS = ("WTKkN", "bOYDu", "wyeCN")
_EO_SSID_RE = re.compile(r"\(t,(\d+)\)")


# ═══════════════════════════════════════════
# 纯函数工具
# ═══════════════════════════════════════════

def _decode(content, encodings: Tuple[str, ...]) -> str:
    """按候选编码显式解码字节；str 原样返回，None 返回 ''，
    全部失败用首个编码 errors=replace。绝不抛出。"""
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


def _clean_line(text: str) -> str:
    """行归一化：去全角空格/&nbsp; 残留、压缩连续空白、去首尾空白。"""
    t = (text or "").replace("　", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", t).strip()


def split_sections(text) -> List[Dict[str, str]]:
    """把研报全文按惯用节名分节 → [{"name","text"}...]（契约 2）。

    规则：整行恰为节名（可带冒号）→ 开新节；「节名：内容」同行 →
    开新节且内容入节；其余行归入当前节（首节默认为「正文」）；
    空行忽略；解析不出任何节结构时自然退化为单节「正文」。绝不抛出。"""
    if not isinstance(text, str) or not text.strip():
        return []
    sections: List[Dict[str, str]] = []
    name, buf = "正文", []

    def _flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append({"name": name, "text": body})

    for raw in text.splitlines():
        line = _clean_line(raw)
        if not line:
            continue
        m = _SECTION_ONLY_RE.match(line)
        if m:
            _flush()
            name, buf = m.group(1), []
            continue
        m = _SECTION_INLINE_RE.match(line)
        if m:
            _flush()
            name, buf = m.group(1), [m.group(2).strip()]
            continue
        buf.append(line)
    _flush()
    return sections


def _normalize_title(title: str) -> str:
    """标题归一化（跨源匹配用）：去全部空白与常见标点差异。"""
    t = re.sub(r"\s+", "", title or "")
    return t.replace("：", ":").replace("(", "(").replace(")", ")")


# ═══════════════════════════════════════════
# 源 1：东财 PDF
# ═══════════════════════════════════════════

def eastmoney_pdf_url(info_code: str) -> str:
    """info_code → pdf.dfcfw.com 直链（akshare stock_research_report_em 源码拼法）。

    用 info_code（纯字母数字）而非 encode_url：encode_url 含 / 时原样拼路径
    404、quote(%2F) 被 Tomcat 拒 400（2026-07-22 实测双死路）；info_code
    无特殊字符，直链 200 拿真实 PDF。quote(safe='') 为零成本防御。"""
    return EM_PDF_URL.format(info_code=quote((info_code or "").strip(), safe=""))


def solve_eo_bot_challenge(content) -> Optional[str]:
    """从 EO_Bot JS 挑战页提取动态常量，计算重载所需 Cookie 头。

    挑战页 JS 内嵌（2026-07-22 实测）：
    - __tst_status = WTKkN + bOYDu + wyeCN（三个整数字面量之和，尾缀 '#'）；
    - EO_Bot_Ssid = n() 内 iTyzs(t, N) 的 N。
    返回 "k1=v1#; k2=v2" 形式 Cookie 头；非挑战页/常量缺失返回 None。"""
    js = _decode(content, ("utf-8",))
    if "EO_Bot_Ssid" not in js or "__tst_status" not in js:
        return None
    try:
        parts = []
        for key in _EO_SUM_KEYS:
            m = re.search(key + r":(\d+)", js)
            if not m:
                return None
            parts.append(int(m.group(1)))
        m = _EO_SSID_RE.search(js)
        if not m:
            return None
        return f"__tst_status={sum(parts)}#; EO_Bot_Ssid={m.group(1)}"
    except Exception:
        logger.warning("EO_Bot 挑战页解析失败", exc_info=True)
        return None


def parse_pdf_fulltext(pdf_bytes) -> Optional[Dict[str, object]]:
    """PDF 字节 → {"fulltext","sections"}；内存解析用完即弃（不落地）。

    pymupdf 惰性导入（缺装时 PDF 通道整体降级返回 None）；
    非 PDF 字节/解析异常/全文为空均返回 None，绝不抛出。"""
    if not isinstance(pdf_bytes, (bytes, bytearray)) or not pdf_bytes:
        return None
    if not bytes(pdf_bytes[:5]) == b"%PDF-":
        return None
    try:
        import fitz  # 惰性导入：模块级不引入重依赖
    except ImportError:
        logger.warning("pymupdf 未安装，东财 PDF 全文通道不可用")
        return None
    try:
        doc = fitz.open(stream=bytes(pdf_bytes), filetype="pdf")
        try:
            text = "\n".join(page.get_text() for page in doc)
        finally:
            doc.close()
    except Exception as e:
        logger.warning("PDF 解析失败（按单篇失败处理）: %s", e)
        return None
    sections = split_sections(text)
    if not sections:
        return None
    fulltext = "\n".join(s["text"] for s in sections)
    return {"fulltext": fulltext, "sections": sections}


# ═══════════════════════════════════════════
# 源 2：新浪研报网页全文
# ═══════════════════════════════════════════

def sina_detail_url(rptid: str) -> str:
    """rptid → 新浪研报详情页 URL。"""
    return SINA_DETAIL_URL.format(rptid=str(rptid).strip())


def parse_sina_list(content) -> List[Dict[str, str]]:
    """新浪研报列表页（str 或 GB2312 字节）→
    [{"rptid","title","org","date","category"}...]（纯函数）。

    行结构（2026-07-22 实测）：tr > td(序号) / td(标题 a.rptid) /
    td(类别) / td(日期) / td(机构) / td(研究员)；按 rptid 去重。"""
    if BeautifulSoup is None:
        logger.warning("beautifulsoup4 未安装，新浪列表解析不可用")
        return []
    html = _decode(content, SINA_ENCODINGS)
    if not html.strip():
        return []
    soup = BeautifulSoup(html, "html.parser")
    entries: List[Dict[str, str]] = []
    seen = set()
    for a in soup.find_all("a", href=True):
        m = _SINA_RPTID_RE.search(a["href"])
        if not m:
            continue
        rptid = m.group(2)
        if rptid in seen:
            continue
        title = (a.get("title") or a.get_text(strip=True) or "").strip()
        if not title:
            continue
        category, date, org = "", "", ""
        tr = a.find_parent("tr")
        if tr is not None:
            cells = [c.get_text(strip=True) for c in tr.find_all("td")]
            if len(cells) >= 3:
                category = cells[2]
            for c in cells:
                if _SINA_DATE_RE.match(c):
                    date = c
                    break
            if len(cells) >= 5:
                org = cells[4]
        seen.add(rptid)
        entries.append({"rptid": rptid, "title": title, "org": org,
                        "date": date, "category": category})
    return entries


def parse_sina_detail(content) -> Optional[Dict[str, object]]:
    """新浪研报详情页（str 或 GB2312 字节）→
    {"title","fulltext","sections"}；解析失败/正文为空返回 None（纯函数）。

    正文容器 <div class="blk_container">（<br> 分段、&nbsp;/全角空格缩进，
    2026-07-22 实测两篇样本结构一致）；标题取 <h1>。"""
    html = _decode(content, SINA_ENCODINGS)
    if not html.strip():
        return None
    title = ""
    body_html = ""
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        h1 = soup.find("h1")
        if h1 is not None:
            title = h1.get_text(strip=True)
        box = soup.find("div", class_="blk_container")
        if box is not None:
            for br in box.find_all("br"):
                br.replace_with("\n")
            body_html = box.get_text("\n")
    else:  # pragma: no cover - bs4 缺失时的正则兜底
        mh = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.S)
        if mh:
            title = re.sub(r"<[^>]+>", "", mh.group(1)).strip()
        mb = re.search(r'<div class="blk_container">(.*?)</div>', html, re.S)
        if mb:
            body_html = re.sub(r"<br\s*/?>", "\n", mb.group(1))
            body_html = re.sub(r"<[^>]+>", "", body_html)
    if not body_html.strip():
        return None
    text = body_html.replace("&nbsp;", " ")
    sections = split_sections(text)
    if not sections:
        return None
    fulltext = "\n".join(s["text"] for s in sections)
    return {"title": title, "fulltext": fulltext, "sections": sections}


# ═══════════════════════════════════════════
# 存储层（契约 1/2；路径解析复用 agent.report_library._db_path）
# ═══════════════════════════════════════════

def _resolve_db_path(db_path: Optional[str] = None) -> str:
    """路径解析：惰性复用 agent.report_library._db_path（契约 7）。"""
    try:
        from agent.report_library import _db_path
        return _db_path(db_path)
    except Exception:  # pragma: no cover - 防御：agent 包不可用时本地等价
        logger.warning("agent.report_library 导入失败，使用本地路径解析", exc_info=True)
        if isinstance(db_path, str) and db_path.strip():
            return db_path.strip()
        env_path = os.getenv("REPORTS_DB_PATH")
        if env_path and env_path.strip():
            return env_path.strip()
        return os.path.join(os.getenv("DATA_DIR") or "data", "reports.db")


def _connect_write(path: str) -> sqlite3.Connection:
    """写路径连接：父目录不存在自动创建。"""
    dir_name = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_name, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_fulltext_db(db_path: Optional[str] = None) -> str:
    """建 report_fulltext 表（幂等），返回解析后的库路径。

    任何异常仅记日志仍返回解析路径，绝不抛出。"""
    path = _resolve_db_path(db_path)
    try:
        with _connect_write(path) as conn:
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
    except Exception as e:
        logger.warning("init_fulltext_db 异常（fail-safe）path=%s: %s",
                       path, e, exc_info=True)
    return path


def _coerce_fulltext_record(record) -> Optional[tuple]:
    """记录字典 → 入库参数元组；缺 info_code/全文为空返回 None。"""
    if not isinstance(record, dict):
        return None
    info_code = str(record.get("info_code") or "").strip()
    if not info_code:
        return None
    fulltext = str(record.get("fulltext") or "").strip()
    if not fulltext:
        return None
    source = str(record.get("source") or DEFAULT_SOURCE).strip() or DEFAULT_SOURCE
    sections = record.get("sections")
    if not isinstance(sections, list):
        sections = []
    clean_sections = []
    for item in sections:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if text:
            clean_sections.append(
                {"name": str(item.get("name") or "正文"), "text": text})
    if not clean_sections:
        clean_sections = [{"name": "正文", "text": fulltext}]
    fetched_at = str(record.get("fetched_at") or "").strip() or \
        datetime.now().isoformat(timespec="seconds")
    return (info_code, source, fulltext,
            json.dumps(clean_sections, ensure_ascii=False), fetched_at)


def upsert_fulltext(records: List[dict], db_path: Optional[str] = None) -> int:
    """批量写入全文记录，返回实际写入/更新条数（幂等：info_code 主键覆盖）。

    缺 info_code/全文为空的非法记录跳过不计数；任何异常按已写入条数
    返回，绝不抛出。"""
    try:
        if not records:
            return 0
        rows = [r for r in (_coerce_fulltext_record(rec) for rec in records)
                if r is not None]
        if not rows:
            return 0
        path = init_fulltext_db(db_path)
        written = 0
        with _connect_write(path) as conn:
            for row in rows:
                try:
                    cur = conn.execute(
                        f"INSERT INTO {FULLTEXT_TABLE} "
                        "(info_code, source, fulltext, sections_json, fetched_at) "
                        "VALUES (?, ?, ?, ?, ?) "
                        "ON CONFLICT(info_code) DO UPDATE SET "
                        "source=excluded.source, fulltext=excluded.fulltext, "
                        "sections_json=excluded.sections_json, "
                        "fetched_at=excluded.fetched_at",
                        row,
                    )
                    written += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 1
                except sqlite3.Error as e:
                    logger.warning("全文写入跳过 info_code=%r: %s", row[0], e)
            conn.commit()
        return written
    except Exception as e:
        logger.warning("upsert_fulltext 异常（fail-safe）: %s", e, exc_info=True)
        return 0


def _select_candidates(path: str, days: Optional[int], ids: Optional[List[str]],
                       limit: Optional[int]) -> List[dict]:
    """挑出 report_fulltext 里还没有的研报（新近优先）。

    - days 给定：publish_date >= date('now', '-N days')；
    - ids 给定：追加 info_code IN (...) 过滤；
    - limit 给定：LIMIT 截断；
    reports 表不存在/任何异常返回 []，绝不抛出。"""
    try:
        where = ["f.info_code IS NULL"]
        params: list = []
        try:
            d = int(days) if days is not None else None
        except (TypeError, ValueError):
            d = None
        if d is not None and d > 0:
            where.append("r.publish_date >= date('now', ?)")
            params.append(f"-{d} days")
        if ids:
            marks = ",".join("?" for _ in ids)
            where.append(f"r.info_code IN ({marks})")
            params.extend(str(i) for i in ids)
        sql = (
            "SELECT r.info_code, r.title, r.org, r.source, r.encode_url, "
            "r.publish_date "
            "FROM reports r "
            f"LEFT JOIN {FULLTEXT_TABLE} f ON f.info_code = r.info_code "
            f"WHERE {' AND '.join(where)} "
            "ORDER BY r.publish_date DESC, r.info_code"
        )
        if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
            sql += " LIMIT ?"
            params.append(limit)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("候选研报查询失败（按空列表处理）: %s", e)
        return []


# ═══════════════════════════════════════════
# 抓取管道（http_get / sleep 可注入，测试全 mock 零网络）
# ═══════════════════════════════════════════

def _default_http_get(url: str, **kw):
    """生产 HTTP 实现（requests）；kw 原样透传（headers/timeout 等）。"""
    if requests is None:
        raise RuntimeError("requests 库不可用")
    kw.setdefault("timeout", DEFAULT_TIMEOUT)
    headers = dict(kw.pop("headers", None) or {})
    headers.setdefault("User-Agent", DEFAULT_UA)
    headers.setdefault("Accept-Language", "zh-CN,zh;q=0.9")
    return requests.get(url, headers=headers, **kw)


class RateGate:
    """单源限速门：同一轮内连续请求间隔 rate + uniform(0, jitter) 秒；
    每源独立实例（东财与新浪互不影响），本轮首个请求不限速。"""

    def __init__(self, *, sleep: Optional[Callable[[float], None]] = None,
                 rate: float = DEFAULT_RATE, jitter: float = DEFAULT_JITTER):
        self._sleep = sleep or time.sleep
        self._rate = max(0.0, rate)
        self._jitter = max(0.0, jitter)
        self._used = False

    def wait(self) -> None:
        if not self._used:
            self._used = True
            return
        delay = self._rate + random.uniform(0, self._jitter)
        logger.debug("限速休眠 %.2fs", delay)
        self._sleep(delay)


def _get_bytes(url: str, http_get, gate: RateGate, **kw) -> Optional[bytes]:
    """统一 HTTP 入口：限速 + 状态码检查 → 响应字节；失败返回 None，绝不抛出。"""
    gate.wait()
    try:
        resp = http_get(url, **kw)
    except Exception as e:
        logger.warning("请求失败 %s: %s", url, e)
        return None
    sc = getattr(resp, "status_code", None)
    if isinstance(sc, int) and sc >= 400:
        logger.warning("HTTP %s %s", sc, url)
        return None
    content = getattr(resp, "content", None)
    if content is None:
        content = getattr(resp, "text", "")
    if isinstance(content, str):
        return content.encode("utf-8")
    return bytes(content) if content else None


def _fetch_one_eastmoney(row: dict, http_get, gate: RateGate,
                         max_attempts: int = EM_FETCH_MAX_ATTEMPTS) -> Optional[dict]:
    """单篇东财 PDF：info_code 直链下载（含 EO_Bot 挑战兜底重试）→ 契约 2 记录。

    每轮尝试：裸 GET info_code 直链 →
    - 命中 %PDF-（常态，实测直链无挑战）直接成功；
    - 得 EO_Bot 挑战页 → 解 cookie 带 Referer 重取一次；
    - 4xx/5xx/空响应 → 进入下一轮。
    最多 max_attempts 轮，全部失败返回 None（调用方记 failed 跳过）。"""
    info_code = (row.get("info_code") or "").strip()
    if not info_code:
        return None
    url = eastmoney_pdf_url(info_code)
    body: Optional[bytes] = None
    for attempt in range(1, max(1, max_attempts) + 1):
        body = _get_bytes(url, http_get, gate)
        if body is not None and body.startswith(b"%PDF-"):
            break  # 直接命中 PDF
        if body is not None:
            cookie = solve_eo_bot_challenge(body)
            if cookie:
                body = _get_bytes(url, http_get, gate,
                                  headers={"Cookie": cookie, "Referer": url})
                if body is not None and body.startswith(b"%PDF-"):
                    break  # 挑战重载命中 PDF
        logger.debug("东财 PDF 第 %d/%d 轮未命中 info_code=%r",
                     attempt, max_attempts, info_code)
    if body is None or not body.startswith(b"%PDF-"):
        logger.warning("东财 PDF 抓取失败（%d 轮尝试用尽）info_code=%r",
                       max_attempts, info_code)
        return None
    parsed = parse_pdf_fulltext(body)
    if parsed is None:
        return None
    return {"info_code": info_code, "source": "eastmoney",
            "fulltext": parsed["fulltext"], "sections": parsed["sections"],
            "fetched_at": datetime.now().isoformat(timespec="seconds")}


def _match_sina_entry(row: dict, entries: List[Dict[str, str]]) -> Optional[dict]:
    """按归一化标题把 reports 行匹配到新浪列表条目。

    - 主路径：标题精确相等 → 互相包含（短者 ≥8 字）兜底；
    - 短标题（<8 字，如慧博晨会早刊的泛化标题「晨会纪要」）单独标题
      无法安全匹配，追加约束：标题精确相等 + 发布日期相同 + 机构非空
      且互相包含（慧博「东吴证券」对新浪「东吴证券股份有限公司」），
      三者齐备才认配，否则放弃（误配比漏配更糟）。"""
    target = _normalize_title(row.get("title") or "")
    row_date = (row.get("publish_date") or "")[:10]
    row_org = re.sub(r"\s+", "", row.get("org") or "")
    if len(target) < 8:
        if not target:
            return None
        for e in entries:
            if _normalize_title(e["title"]) != target:
                continue
            if row_date and e.get("date") and e["date"] != row_date:
                continue
            e_org = re.sub(r"\s+", "", e.get("org") or "")
            if row_org and e_org and (row_org in e_org or e_org in row_org):
                return e
        return None
    for e in entries:
        if _normalize_title(e["title"]) == target:
            return e
    for e in entries:
        cand = _normalize_title(e["title"])
        if len(cand) >= 8 and (cand in target or target in cand):
            return e
    return None


def _fetch_one_sina(row: dict, entry: dict, http_get, gate: RateGate) -> Optional[dict]:
    """单篇新浪网页全文：详情页 → 分节解析 → 契约 2 记录。"""
    info_code = row.get("info_code") or ""
    if not info_code:
        return None
    url = sina_detail_url(entry["rptid"])
    body = _get_bytes(url, http_get, gate)
    if body is None:
        return None
    parsed = parse_sina_detail(body)
    if parsed is None:
        logger.warning("新浪详情页解析失败 info_code=%r rptid=%s",
                       info_code, entry["rptid"])
        return None
    return {"info_code": info_code, "source": "sina",
            "fulltext": parsed["fulltext"], "sections": parsed["sections"],
            "fetched_at": datetime.now().isoformat(timespec="seconds")}


def fetch_fulltext(db_path: Optional[str] = None,
                   days: Optional[int] = None,
                   limit: Optional[int] = None,
                   ids: Optional[List[str]] = None,
                   http_get=None,
                   sleep=None,
                   rate: float = DEFAULT_RATE,
                   jitter: float = DEFAULT_JITTER,
                   sina_max_pages: int = SINA_LIST_MAX_PAGES) -> Dict[str, int]:
    """全文抓取管道：候选挑选 → 东财 PDF / 新浪网页双源抓取 → 入库。

    返回统计 {"candidates","fetched","upserted","failed","skipped"}：
    - candidates：reports 表内尚无全文的候选数（days/ids/limit 过滤后）；
    - 东财通道：source=='eastmoney' 且 encode_url 非空的候选走 PDF 直链；
    - 新浪通道：其余候选按标题匹配新浪列表（近 start_date 起翻页，
      最多 sina_max_pages 页）；匹配不到计 skipped；
    - 单篇失败记 warning 计入 failed 并跳过，不拖垮批次；绝不抛出。"""
    stats = {"candidates": 0, "fetched": 0, "upserted": 0,
             "failed": 0, "skipped": 0}
    try:
        http_get = http_get or _default_http_get
        path = init_fulltext_db(db_path)
        candidates = _select_candidates(path, days, ids, limit)
        stats["candidates"] = len(candidates)
        if not candidates:
            return stats

        em_gate = RateGate(sleep=sleep, rate=rate, jitter=jitter)
        sina_gate = RateGate(sleep=sleep, rate=rate, jitter=jitter)

        em_rows, sina_rows = [], []
        for row in candidates:
            # 东财 PDF 直链用 info_code 拼接，不再要求 encode_url
            if (row.get("source") or "") == "eastmoney":
                em_rows.append(row)
            else:
                sina_rows.append(row)

        # ── 东财 PDF 通道 ──
        for row in em_rows:
            try:
                record = _fetch_one_eastmoney(row, http_get, em_gate)
            except Exception:
                logger.warning("东财单篇抓取异常 info_code=%r",
                               row.get("info_code"), exc_info=True)
                record = None
            if record is None:
                stats["failed"] += 1
                continue
            stats["fetched"] += 1
            stats["upserted"] += upsert_fulltext([record], path)

        # ── 新浪网页全文通道 ──
        if sina_rows:
            start_date = min(
                (r.get("publish_date") or "" for r in sina_rows), default="")
            entries: List[Dict[str, str]] = []
            for page in range(1, max(1, sina_max_pages) + 1):
                body = _get_bytes(SINA_LIST_URL.format(page=page),
                                  http_get, sina_gate)
                if body is None:
                    break  # 请求失败已记 warning，停止翻页
                batch = parse_sina_list(body)
                if not batch:
                    break  # 空页到底
                entries.extend(batch)
                oldest = min((e["date"] for e in batch if e["date"]), default="")
                if start_date and oldest and oldest < start_date:
                    break  # 本页最老一篇已早于候选最早日期，更老的页不必再抓
            for row in sina_rows:
                try:
                    entry = _match_sina_entry(row, entries)
                    if entry is None:
                        stats["skipped"] += 1
                        logger.debug("新浪未匹配 info_code=%r title=%r",
                                     row.get("info_code"), row.get("title"))
                        continue
                    record = _fetch_one_sina(row, entry, http_get, sina_gate)
                except Exception:
                    logger.warning("新浪单篇抓取异常 info_code=%r",
                                   row.get("info_code"), exc_info=True)
                    record = None
                if record is None:
                    stats["failed"] += 1  # 已匹配但抓取/解析失败
                    continue
                stats["fetched"] += 1
                stats["upserted"] += upsert_fulltext([record], path)
        return stats
    except Exception as e:
        logger.warning("fetch_fulltext 异常（fail-safe）: %s", e, exc_info=True)
        return stats


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

def main(argv=None, *, http_get=None, sleep=None) -> int:
    """全文抓取 CLI：--days N（默认 30）/--limit/--db-path/--verbose，
    结尾打印抓取/入库/失败统计。http_get/sleep 仅供测试注入。"""
    parser = argparse.ArgumentParser(
        description="研报全文获取：东财 PDF + 新浪网页全文 → report_fulltext 表")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"只抓近 N 天研报的候选（默认 {DEFAULT_DAYS}）")
    parser.add_argument("--limit", type=int, default=None,
                        help="候选数上限（默认不限）")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE,
                        help="请求间隔下限秒数（默认 1.0，另加 0~0.5s 随机抖动）")
    parser.add_argument("--db-path", default=None,
                        help="SQLite 路径（缺省走存储层解析：显式 > REPORTS_DB_PATH "
                             "> ${DATA_DIR:-data}/reports.db）")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出 DEBUG 日志")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    days = max(1, args.days)
    print(f"全文抓取：近 {days} 天候选"
          + (f"，上限 {args.limit} 篇" if args.limit else "")
          + f"，库：{_resolve_db_path(args.db_path)}")

    stats = fetch_fulltext(db_path=args.db_path, days=days, limit=args.limit,
                           http_get=http_get, sleep=sleep, rate=args.rate)
    print(f"全文抓取完成：候选 {stats['candidates']} 篇，"
          f"抓取 {stats['fetched']} 篇，入库 {stats['upserted']} 篇，"
          f"失败 {stats['failed']} 篇，跳过（新浪未匹配）{stats['skipped']} 篇")
    return 0


if __name__ == "__main__":
    sys.exit(main())
