#!/usr/bin/env python3
"""THU 选课社区（thucourse）全量爬虫（校园知识库 · 选课数据层）。

数据源：https://yourschool.cc.cd（纯静态 JSON 站，免登录，2026-09 实测连通）：
- GET /data/manifest.json           → {total_courses, total_reviews}
- GET /data/full_index.json         → 全部课程元数据（实测 37804 门），
  结构 {courses: {'课程名(教师名)': {kcm, sqid, jsm, tid, kkdw}}}
- GET /data/with_comment_index.json → 有点评课程（实测 1325 门），
  课程字典多 count/avg 两个字段
- GET /data/courses/{sqid}.json     → 单课程点评，
  结构 {count, next, previous, results: [{id, rating, comment, created_at, score}]}
  next 为分页 URL（契约上可能是完整 URL 或相对路径，实测单页恒 null），
  跟随翻页直到 next 为 null。

入库条目（全局契约 8 字段，upsert_entries 契约见 agent/campus_kb.py）：
1. 课程条目 source='thucourse_course'（全量）：
   - source_id = 'thucourse:course:{sqid}'
   - title     = '{kcm}（{jsm}）'
   - content   = 课程基本信息自然语言描述（课程名/教师/开课单位/点评数/平均评分）
   - url       = 有点评课程为课程页链接，无点评课程为空串
   - metadata_json 保留全部原始字段 + has_reviews/reviews_done 标记
2. 点评条目 source='thucourse_review'（每条点评一条）：
   - source_id = 'thucourse:review:{review_id}'
   - title     = '{kcm}（{jsm}）点评 - {rating}星'
   - content   = 点评原文（OCR 式空格/换行噪声入库前做基础清洗：
                 连续空白折叠为单个空格 + strip，语义不动）
   - metadata_json 含 course_sqid、kcm、jsm、rating、score、created_at

工程纪律：
- http_get/sleep 全可注入，测试全 mock 零网络；默认实现用 urllib；
- 限速默认 0.5s/req（≤2 req/s），sleep 注入后测试零等待；
- 断点续爬：本地 progress json（默认 data/thucourse_progress.json）记录已完成
  点评抓取的 sqid；同时课程条目 metadata_json 写入 reviews_done 标记，
  可用 get_entry('thucourse_course', source_id) 无文件续爬。已完成 sqid 一律跳过；
- 单课程点评抓取失败记 warning 并继续下一门，绝不整体抛出；
- 本模块不在 import 时依赖 agent.campus_kb（仅 CLI 落库路径惰性导入，
  且可被 _load_campus_kb monkeypatch 替换）。

CLI 用法：
    /usr/local/bin/python3 scripts/thucourse_crawler.py --only-index      # 只入库课程元数据
    /usr/local/bin/python3 scripts/thucourse_crawler.py --max-courses 50  # 调试小批量
    /usr/local/bin/python3 scripts/thucourse_crawler.py --db data/campus_kb.db -v
"""

import argparse
import json
import logging
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 保证从项目根可导入 agent 包（脚本可被任意 cwd 调用；仅 CLI 落库路径使用）。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── 全局常量 ──

BASE_URL = "https://yourschool.cc.cd"
MANIFEST_URL = f"{BASE_URL}/data/manifest.json"
FULL_INDEX_URL = f"{BASE_URL}/data/full_index.json"
COMMENT_INDEX_URL = f"{BASE_URL}/data/with_comment_index.json"
COURSE_URL_TMPL = f"{BASE_URL}/data/courses/{{sqid}}.json"
COURSE_PAGE_TMPL = f"{BASE_URL}/thucourse/course.html?sqid={{sqid}}"

SOURCE_COURSE = "thucourse_course"
SOURCE_REVIEW = "thucourse_review"

DEFAULT_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
DEFAULT_RATE = 0.5        # 请求间隔（秒）：≤2 req/s
DEFAULT_TIMEOUT = 20
DEFAULT_BATCH_SIZE = 500  # 课程条目单批 upsert 条数
DEFAULT_MAX_PAGES = 50    # 单课程点评翻页安全上限（实测单页恒 null，双保险）
DEFAULT_PROGRESS_PATH = PROJECT_ROOT / "data" / "thucourse_progress.json"

_WS_RE = re.compile(r"\s+")

# upsert 回调契约：entries -> 实际写入/更新条数（int），由存储层/测试注入
UpsertFn = Callable[[List[dict]], int]
# get_entry 回调契约：(source, source_id) -> 条目 dict 或 None
GetEntryFn = Callable[[str, str], Optional[dict]]


# ═══════════════════════════════════════════
# 纯函数工具
# ═══════════════════════════════════════════

def _now_iso() -> str:
    """本地时区 ISO 时间戳（秒级），用于条目 updated_at。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def clean_comment(text) -> str:
    """点评原文基础清洗：连续空白（空格/制表/换行等 OCR 噪声）折叠为单个空格，
    首尾 strip。只做排版归一，不删改任何文字内容，语义保持原样。
    None/非字符串输入安全降级为空串。"""
    if text is None:
        return ""
    return _WS_RE.sub(" ", str(text)).strip()


def _to_int(value) -> Optional[int]:
    """宽松 int 转换；失败返回 None，绝不抛出。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _to_float(value) -> Optional[float]:
    """宽松 float 转换；失败返回 None，绝不抛出。"""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


# ═══════════════════════════════════════════
# 解析纯函数（输入 JSON payload，输出规范化结构）
# ═══════════════════════════════════════════

def parse_index(payload) -> List[dict]:
    """full_index / with_comment_index 响应 → 规范化课程字典列表（纯函数）。

    输出字段：kcm, jsm, kkdw（字符串，缺省空串）、sqid, tid（int）、
    count（int，缺省 0）、avg（float 或 None）、index_key（索引原始键）。
    sqid 缺失或无法转 int 的课程丢弃；非法输入返回 []。"""
    if not isinstance(payload, dict):
        return []
    courses = payload.get("courses")
    if not isinstance(courses, dict):
        return []
    out: List[dict] = []
    for key, raw in courses.items():
        if not isinstance(raw, dict):
            continue
        sqid = _to_int(raw.get("sqid"))
        if sqid is None:
            logger.warning("课程缺少合法 sqid，丢弃：%s", key)
            continue
        out.append({
            "sqid": sqid,
            "kcm": str(raw.get("kcm") or "").strip(),
            "jsm": str(raw.get("jsm") or "").strip(),
            "kkdw": str(raw.get("kkdw") or "").strip(),
            "tid": _to_int(raw.get("tid")),
            "count": _to_int(raw.get("count")) or 0,
            "avg": _to_float(raw.get("avg")),
            "index_key": str(key),
        })
    return out


def merge_courses(full: List[dict], commented: List[dict]) -> List[dict]:
    """用有点评索引的 count/avg 回填全量索引（按 sqid 匹配，纯函数）。
    有点评课程 has_reviews=True；full 中已自带的 count/avg 会被有点评索引覆盖。"""
    by_sqid: Dict[int, dict] = {c["sqid"]: c for c in commented}
    merged: List[dict] = []
    for course in full:
        extra = by_sqid.get(course["sqid"])
        item = dict(course)
        if extra is not None:
            item["count"] = extra["count"]
            item["avg"] = extra["avg"]
        item["has_reviews"] = item["count"] > 0
        merged.append(item)
    return merged


def parse_course_page(payload) -> Tuple[List[dict], Optional[str]]:
    """单课程点评页响应 → (点评字典列表, 下一页 URL 或 None)（纯函数）。

    点评保留原始字段 id/rating/comment/created_at/score；缺合法 id 的条目丢弃。
    next 非字符串或空串一律归一为 None。非法输入返回 ([], None)。"""
    if not isinstance(payload, dict):
        return [], None
    results = payload.get("results")
    reviews: List[dict] = []
    if isinstance(results, list):
        for raw in results:
            if not isinstance(raw, dict):
                continue
            rid = _to_int(raw.get("id"))
            if rid is None:
                logger.warning("点评缺少合法 id，丢弃：%r", raw)
                continue
            reviews.append({
                "id": rid,
                "rating": _to_int(raw.get("rating")),
                "comment": raw.get("comment"),
                "created_at": str(raw.get("created_at") or "").strip(),
                "score": _to_float(raw.get("score")),
            })
    nxt = payload.get("next")
    if not isinstance(nxt, str) or not nxt.strip():
        nxt = None
    return reviews, nxt


# ═══════════════════════════════════════════
# 条目构造（全局契约 8 字段）
# ═══════════════════════════════════════════

def make_course_entry(course: dict, now: Optional[str] = None) -> dict:
    """课程元数据 → thucourse_course 条目（8 字段契约，纯函数）。
    metadata_json 保留全部原始字段 + has_reviews/reviews_done 断点标记。"""
    sqid = course["sqid"]
    kcm, jsm, kkdw = course["kcm"], course["jsm"], course["kkdw"]
    count, avg = course.get("count") or 0, course.get("avg")
    has_reviews = bool(course.get("has_reviews", count > 0))
    lines = [f"《{kcm}》（教师：{jsm}）", f"开课单位：{kkdw}。"]
    if has_reviews:
        avg_text = f"{avg:g}" if isinstance(avg, (int, float)) else "未知"
        lines.append(f"THU 选课社区共有 {count} 条学生点评，平均评分 {avg_text} 星（满分 5 星）。")
    else:
        lines.append("THU 选课社区已收录该课程，暂无学生点评。")
    metadata = {
        "sqid": sqid, "kcm": kcm, "jsm": jsm, "kkdw": kkdw,
        "tid": course.get("tid"), "count": count, "avg": avg,
        "index_key": course.get("index_key", ""),
        "has_reviews": has_reviews,
        "reviews_done": bool(course.get("reviews_done", not has_reviews)),
    }
    return {
        "source": SOURCE_COURSE,
        "source_id": f"thucourse:course:{sqid}",
        "title": f"{kcm}（{jsm}）",
        "content": "\n".join(lines),
        "url": COURSE_PAGE_TMPL.format(sqid=sqid) if has_reviews else "",
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
        "updated_at": now or _now_iso(),
    }


def make_review_entry(review: dict, course: dict,
                      now: Optional[str] = None) -> Optional[dict]:
    """单条点评 → thucourse_review 条目（8 字段契约，纯函数）。
    content 为清洗后的点评原文；清洗后为空且无任何评分信息的条目返回 None（不入库）。"""
    rid = _to_int(review.get("id"))
    if rid is None:
        return None
    kcm, jsm = course["kcm"], course["jsm"]
    rating = _to_int(review.get("rating"))
    content = clean_comment(review.get("comment"))
    if not content and rating is None and review.get("score") is None:
        return None  # 空点评且无评分：无可检索内容，丢弃
    metadata = {
        "course_sqid": course["sqid"], "kcm": kcm, "jsm": jsm,
        "rating": rating, "score": _to_float(review.get("score")),
        "created_at": str(review.get("created_at") or "").strip(),
    }
    return {
        "source": SOURCE_REVIEW,
        "source_id": f"thucourse:review:{rid}",
        "title": f"{kcm}（{jsm}）点评 - {rating if rating is not None else '?'}星",
        "content": content,
        "url": COURSE_PAGE_TMPL.format(sqid=course["sqid"]),
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
        "updated_at": now or _now_iso(),
    }


# ═══════════════════════════════════════════
# 默认 HTTP（urllib，生产用；测试注入 fake）
# ═══════════════════════════════════════════

def _default_http_get(url: str, **kw):
    """生产 HTTP 实现（urllib），返回响应字节。kw 支持 timeout。"""
    req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_UA})
    with urllib.request.urlopen(req, timeout=kw.get("timeout", DEFAULT_TIMEOUT)) as resp:
        return resp.read()


# ═══════════════════════════════════════════
# 爬虫主体
# ═══════════════════════════════════════════

class ThucourseCrawler:
    """THU 选课社区全量爬虫：索引抓取 + 点评分页抓取 + 契约条目产出。

    可注入点（测试全 mock 零网络）：
    - http_get(url, **kw)：可返回 dict（已解析 JSON）/ str / 字节 /
      带 .json()/.text/.content 的响应对象
    - sleep(seconds)：限速休眠；rate 控制节奏（默认 0.5s/req，≤2 req/s）
    所有分支失败记 warning 并降级，公开方法绝不向调用方抛异常。
    """

    name = "thucourse"

    def __init__(self, *, http_get=None, sleep=None,
                 rate: float = DEFAULT_RATE, timeout: int = DEFAULT_TIMEOUT):
        self._http_get = http_get or _default_http_get
        self._sleep = sleep or time.sleep
        self._rate = max(0.0, rate)
        self._timeout = timeout
        self.requests = 0
        self._requested = False

    # ── 内部基础设施 ──

    def _reset_run(self) -> None:
        """每轮 run 重置请求计数与限速门（本轮首个请求不限速）。"""
        self.requests = 0
        self._requested = False

    def _throttle(self) -> None:
        """限速门：同一轮内连续请求间隔 rate 秒。"""
        if not self._requested:
            self._requested = True
            return
        self._sleep(self._rate)

    def _get_json(self, url: str) -> Optional[dict]:
        """GET 并解析 JSON 对象；请求异常/非法 JSON/顶层非对象均记 warning 返回 None。"""
        self._throttle()
        try:
            raw = self._http_get(url, timeout=self._timeout)
        except Exception as e:
            logger.warning("%s 请求失败 %s: %s", self.name, url, e)
            return None
        self.requests += 1
        payload = raw
        if not isinstance(payload, dict):
            if hasattr(raw, "json"):
                try:
                    payload = raw.json()
                except Exception:
                    payload = None
            else:
                content = getattr(raw, "content", None)
                if content is None:
                    content = getattr(raw, "text", raw)
                try:
                    if isinstance(content, bytes):
                        content = content.decode("utf-8")
                    payload = json.loads(content)
                except (json.JSONDecodeError, ValueError, TypeError, UnicodeDecodeError):
                    payload = None
        if not isinstance(payload, dict):
            logger.warning("%s 响应不是合法 JSON 对象：%s", self.name, url)
            return None
        return payload

    # ── 抓取 ──

    def fetch_courses(self) -> List[dict]:
        """抓取并合并两份索引 → 规范化课程列表（含 has_reviews 标记）。
        全量索引失败返回 []；有点评索引失败降级为全量无点评标记。"""
        full_payload = self._get_json(FULL_INDEX_URL)
        if full_payload is None:
            logger.warning("%s 全量课程索引抓取失败，本轮终止", self.name)
            return []
        full = parse_index(full_payload)
        comment_payload = self._get_json(COMMENT_INDEX_URL)
        commented = parse_index(comment_payload) if comment_payload is not None else []
        if comment_payload is None:
            logger.warning("%s 有点评索引抓取失败，降级为仅课程元数据", self.name)
        return merge_courses(full, commented)

    def fetch_reviews(self, course: dict,
                      max_pages: int = DEFAULT_MAX_PAGES) -> Optional[List[dict]]:
        """抓取单课程全部点评（跟随 next 翻页直到 null）。
        任一页面失败返回 None（整门课标记失败，交由断点续爬下轮重试）；
        成功返回点评列表（可为空列表）。"""
        url: Optional[str] = COURSE_URL_TMPL.format(sqid=course["sqid"])
        reviews: List[dict] = []
        seen_urls = set()
        for _ in range(max(1, max_pages)):
            if url is None or url in seen_urls:
                break
            seen_urls.add(url)
            payload = self._get_json(url)
            if payload is None:
                logger.warning("%s 课程 sqid=%s 点评页抓取失败：%s",
                               self.name, course["sqid"], url)
                return None
            page_reviews, nxt = parse_course_page(payload)
            reviews.extend(page_reviews)
            if nxt is None:
                break
            url = urllib.parse.urljoin(url, nxt)  # next 可能是完整 URL 或相对路径
        return reviews

    # ── 断点续爬 ──

    @staticmethod
    def _load_progress(progress_path) -> set:
        """读取本地进度文件 → 已完成点评抓取的 sqid 集合。失败/缺失返回空集。"""
        if not progress_path:
            return set()
        try:
            payload = json.loads(Path(progress_path).read_text(encoding="utf-8"))
            sqids = payload.get("done_sqids")
            if isinstance(sqids, list):
                return {s for s in (_to_int(x) for x in sqids) if s is not None}
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        return set()

    @staticmethod
    def _save_progress(progress_path, done: set) -> None:
        """原子写进度文件（tmp + replace）；失败仅记 warning，绝不抛出。"""
        if not progress_path:
            return
        try:
            path = Path(progress_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps({
                "done_sqids": sorted(done),
                "updated_at": _now_iso(),
            }, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            logger.warning("%s 进度文件写入失败：%s", ThucourseCrawler.name,
                           progress_path, exc_info=True)

    @staticmethod
    def _entry_reviews_done(get_entry: Optional[GetEntryFn], sqid: int) -> bool:
        """get_entry 续爬判定：课程条目 metadata_json 中 reviews_done 为真即已完成。
        get_entry 异常/条目缺失/metadata 非法一律按未完成处理，绝不抛出。"""
        if get_entry is None:
            return False
        try:
            entry = get_entry(SOURCE_COURSE, f"thucourse:course:{sqid}")
        except Exception:
            logger.warning("get_entry 查询失败（sqid=%s），按未完成处理",
                           sqid, exc_info=True)
            return False
        if not isinstance(entry, dict):
            return False
        try:
            meta = json.loads(entry.get("metadata_json") or "{}")
        except (json.JSONDecodeError, ValueError, TypeError):
            return False
        return bool(meta.get("reviews_done"))

    def _emit(self, entries: List[dict], upsert: Optional[UpsertFn]) -> int:
        """一批条目落库（注入 upsert 时）；upsert 异常吞掉记 warning，返回写入条数。"""
        if upsert is None or not entries:
            return 0
        try:
            n = upsert(entries)
            return n if isinstance(n, int) and not isinstance(n, bool) else len(entries)
        except Exception:
            logger.warning("%s upsert 落库失败（%d 条），继续",
                           self.name, len(entries), exc_info=True)
            return 0

    # ── 主流程 ──

    def run(self, *, upsert: Optional[UpsertFn] = None,
            get_entry: Optional[GetEntryFn] = None,
            only_index: bool = False,
            max_courses: Optional[int] = None,
            progress_path=None,
            resume: bool = True,
            batch_size: int = DEFAULT_BATCH_SIZE) -> dict:
        """全量抓取主流程：课程索引入库 →（可选）逐课程点评入库。

        返回统计字典：
        {courses_indexed, courses_upserted, reviews_fetched, reviews_upserted,
         review_courses_done, skipped_resume, failed_sqids, requests}
        任何单课程失败只记入 failed_sqids 并继续，绝不整体抛出。"""
        self._reset_run()
        stats = {"courses_indexed": 0, "courses_upserted": 0,
                 "reviews_fetched": 0, "reviews_upserted": 0,
                 "review_courses_done": 0, "skipped_resume": 0,
                 "failed_sqids": [], "requests": 0}
        courses = self.fetch_courses()
        if isinstance(max_courses, int) and max_courses >= 0:
            courses = courses[:max_courses]
        stats["courses_indexed"] = len(courses)
        if not courses:
            stats["requests"] = self.requests
            return stats

        done = self._load_progress(progress_path) if resume else set()
        run_now = _now_iso()

        # 1) 课程条目全量入库（批 upsert）；reviews_done 标记反映既有进度
        batch: List[dict] = []
        for course in courses:
            if not only_index and course.get("has_reviews"):
                course["reviews_done"] = (
                    course["sqid"] in done
                    or (resume and self._entry_reviews_done(get_entry, course["sqid"])))
            else:
                course["reviews_done"] = not course.get("has_reviews")
            batch.append(make_course_entry(course, now=run_now))
            if len(batch) >= max(1, batch_size):
                stats["courses_upserted"] += self._emit(batch, upsert)
                batch = []
        if batch:
            stats["courses_upserted"] += self._emit(batch, upsert)

        if only_index:
            stats["requests"] = self.requests
            return stats

        # 2) 有点评课程逐门抓取点评（断点续爬：已完成 sqid 跳过）
        for course in sorted((c for c in courses if c.get("has_reviews")),
                             key=lambda c: c["sqid"]):
            sqid = course["sqid"]
            if course.get("reviews_done"):
                stats["skipped_resume"] += 1
                done.add(sqid)
                continue
            reviews = self.fetch_reviews(course)
            if reviews is None:
                stats["failed_sqids"].append(sqid)
                continue  # 单课程失败记 warning（fetch_reviews 内），继续下一门
            entries = [e for e in (make_review_entry(r, course, now=run_now)
                                   for r in reviews) if e is not None]
            stats["reviews_fetched"] += len(reviews)
            stats["reviews_upserted"] += self._emit(entries, upsert)
            # 完成标记：课程条目 reviews_done=True 重入库 + 本地进度文件
            stats["courses_upserted"] += self._emit(
                [make_course_entry({**course, "reviews_done": True}, now=run_now)],
                upsert)
            done.add(sqid)
            stats["review_courses_done"] += 1
            self._save_progress(progress_path, done)
        self._save_progress(progress_path, done)
        stats["requests"] = self.requests
        return stats


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

def _load_campus_kb():
    """惰性导入知识库存储层（仅 CLI 落库路径；测试 monkeypatch 本函数注入
    fake，不对 agent.campus_kb 实体形成硬依赖）。失败返回 None。"""
    try:
        from agent import campus_kb
        return campus_kb
    except Exception:
        logger.warning("agent.campus_kb 导入失败", exc_info=True)
        return None


def main(argv=None, *, http_get=None, sleep=None) -> int:
    """thucourse 爬虫 CLI：抓索引入库课程元数据，按需抓点评，结尾打印统计。
    http_get/sleep 仅供测试注入；生产走默认 urllib 与 time.sleep。"""
    parser = argparse.ArgumentParser(
        description="THU 选课社区爬虫：课程元数据 + 学生点评入校园知识库（SQLite）")
    parser.add_argument("--only-index", action="store_true",
                        help="只入库课程元数据，不抓点评")
    parser.add_argument("--max-courses", type=int, default=None,
                        help="最多处理课程数（调试用，默认全量）")
    parser.add_argument("--db", default=None,
                        help="SQLite 路径（缺省走存储层默认解析）")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE,
                        help="请求间隔秒数（默认 0.5，≤2 req/s）")
    parser.add_argument("--progress-path", default=str(DEFAULT_PROGRESS_PATH),
                        help="断点续爬进度文件（默认 data/thucourse_progress.json）")
    parser.add_argument("--no-resume", action="store_true",
                        help="忽略既有进度，全量重抓（幂等覆盖）")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出 DEBUG 日志")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    kb = _load_campus_kb()
    if kb is None:
        print("存储层 agent.campus_kb 不可用（详见日志），终止。")
        return 1
    try:
        kb.init_db(args.db)
    except Exception:
        logger.warning("校园知识库初始化失败", exc_info=True)
        print("校园知识库初始化失败（详见日志），终止。")
        return 1

    print(f"模式：{'仅课程索引' if args.only_index else '课程索引 + 点评'}，"
          f"进度文件：{args.progress_path}，库：{args.db or '（存储层默认）'}")

    crawler = ThucourseCrawler(http_get=http_get, sleep=sleep, rate=args.rate)

    def _upsert(entries, _kb=kb, _db=args.db):
        return _kb.upsert_entries(entries, db_path=_db)

    def _get_entry(source, source_id, _kb=kb, _db=args.db):
        return _kb.get_entry(source, source_id, db_path=_db)

    stats = crawler.run(
        upsert=_upsert, get_entry=_get_entry,
        only_index=args.only_index, max_courses=args.max_courses,
        progress_path=args.progress_path, resume=not args.no_resume)
    print(f"课程：索引 {stats['courses_indexed']} 门，入库 {stats['courses_upserted']} 条；")
    print(f"点评：抓取 {stats['reviews_fetched']} 条，入库 {stats['reviews_upserted']} 条，"
          f"完成课程 {stats['review_courses_done']} 门，续爬跳过 {stats['skipped_resume']} 门；")
    print(f"请求 {stats['requests']} 次，失败课程 {len(stats['failed_sqids'])} 门"
          f"{('：' + ','.join(map(str, stats['failed_sqids'][:20]))) if stats['failed_sqids'] else ''}。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
