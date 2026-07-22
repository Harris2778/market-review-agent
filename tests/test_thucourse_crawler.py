"""scripts/thucourse_crawler.py 测试（全 mock http_get，零网络、零真实外部文件）。

fixture 结构按 2026-09 对 https://yourschool.cc.cd 的实测侦察构造：
- full_index.json         → {courses: {'课程名(教师名)': {kcm, sqid, jsm, tid, kkdw}}}
- with_comment_index.json → 同上 + count/avg
- courses/{sqid}.json     → {count, next, previous, results: [{id, rating, comment,
                              created_at, score}]}，next 实测恒 null（分页逻辑仍覆盖）
存储层 campus_kb 一律以 fake 注入（monkeypatch _load_campus_kb 或直接传
upsert/get_entry 回调），绝不 import 实体模块、绝不触碰真实 db。
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import thucourse_crawler as tc  # noqa: E402


# ── 实测结构 fixture ──

FULL_INDEX = {
    "courses": {
        "高等统计选讲(I)(李东)": {
            "kcm": "高等统计选讲(I)", "sqid": 1, "jsm": "李东",
            "tid": 0, "kkdw": "工业工程系",
        },
        "商法学(梁上上)": {
            "kcm": "商法学", "sqid": 219, "jsm": "梁上上",
            "tid": 197, "kkdw": "法学院",
        },
        "线性代数(1)(曾惠慧)": {
            "kcm": "线性代数(1)", "sqid": 25437, "jsm": "曾惠慧",
            "tid": 20000, "kkdw": "数学科学系",
        },
    }
}

COMMENT_INDEX = {
    "courses": {
        "商法学(梁上上)": {
            "kcm": "商法学", "sqid": 219, "jsm": "梁上上",
            "tid": 197, "kkdw": "法学院", "count": 1, "avg": 4.0,
        },
        "线性代数(1)(曾惠慧)": {
            "kcm": "线性代数(1)", "sqid": 25437, "jsm": "曾惠慧",
            "tid": 20000, "kkdw": "数学科学系", "count": 65, "avg": 3.8,
        },
    }
}

COURSE_219_PAGE = {
    "count": 1, "next": None, "previous": None,
    "results": [{
        "id": 1006, "rating": 4,
        "comment": "老师不会点名   但是会点人回答问题\n\n以及老师会记录谁发言了  会加分",
        "created_at": "2024/07/21 18:44", "score": None,
    }],
}

COURSE_25437_PAGE1 = {
    "count": 65, "next": "/data/courses/25437_2.json", "previous": None,
    "results": [{
        "id": 2483, "rating": 1,
        "comment": "考核方式：\n\n授课质量与给分：-1",
        "created_at": "2025/09/19 09:46", "score": None,
    }],
}

COURSE_25437_PAGE2 = {
    "count": 65, "next": None, "previous": "/data/courses/25437.json",
    "results": [{
        "id": 2484, "rating": 5, "comment": "讲得很好，给分友善",
        "created_at": "2025/09/20 10:00", "score": 95,
    }],
}


class FakeHTTP:
    """按 URL 路由的 fake http_get：payload 可以是 dict / Exception / 可调用。"""

    def __init__(self, routes):
        self.routes = dict(routes)
        self.calls = []

    def __call__(self, url, **kw):
        self.calls.append(url)
        payload = self.routes.get(url)
        if payload is None:
            raise ConnectionError(f"fake 未收录 URL：{url}")
        if isinstance(payload, Exception):
            raise payload
        if callable(payload):
            return payload(url, **kw)
        return payload


def _full_routes(**extra):
    """覆盖两份索引 + 两门有点评课程详情页的基础路由。"""
    routes = {
        tc.FULL_INDEX_URL: FULL_INDEX,
        tc.COMMENT_INDEX_URL: COMMENT_INDEX,
        tc.COURSE_URL_TMPL.format(sqid=219): COURSE_219_PAGE,
        tc.COURSE_URL_TMPL.format(sqid=25437): COURSE_25437_PAGE1,
        tc.BASE_URL + "/data/courses/25437_2.json": COURSE_25437_PAGE2,
    }
    routes.update(extra)
    return routes


class UpsertCollector:
    """收集 upsert 批次并模拟返回写入条数。"""

    def __init__(self):
        self.batches = []

    def __call__(self, entries):
        self.batches.append(list(entries))
        return len(entries)

    @property
    def entries(self):
        return [e for batch in self.batches for e in batch]


def _crawler(http, sleeps=None):
    sleeps = sleeps if sleeps is not None else []
    return tc.ThucourseCrawler(http_get=http, sleep=sleeps.append, rate=0.5)


# ═══════════════════════════════════════════
# 清洗逻辑
# ═══════════════════════════════════════════

def test_clean_comment_collapses_whitespace():
    raw = "老师不会点名   但是会点人回答问题\n\n以及老师会记录谁发言了  会加分"
    assert tc.clean_comment(raw) == "老师不会点名 但是会点人回答问题 以及老师会记录谁发言了 会加分"


def test_clean_comment_strips_and_preserves_semantics():
    assert tc.clean_comment("  给分 很 好 \t\n") == "给分 很 好"
    # 不删改任何文字内容，只归一排版
    assert tc.clean_comment("考核方式：p/f") == "考核方式：p/f"


def test_clean_comment_none_and_non_str():
    assert tc.clean_comment(None) == ""
    assert tc.clean_comment(123) == "123"


# ═══════════════════════════════════════════
# 索引解析
# ═══════════════════════════════════════════

def test_parse_index_normalizes_fields():
    courses = tc.parse_index(FULL_INDEX)
    by_sqid = {c["sqid"]: c for c in courses}
    assert set(by_sqid) == {1, 219, 25437}
    c = by_sqid[219]
    assert c["kcm"] == "商法学" and c["jsm"] == "梁上上"
    assert c["kkdw"] == "法学院" and c["tid"] == 197
    assert c["count"] == 0 and c["avg"] is None
    assert c["index_key"] == "商法学(梁上上)"


def test_parse_index_comment_fields():
    courses = tc.parse_index(COMMENT_INDEX)
    c = {c["sqid"]: c for c in courses}[25437]
    assert c["count"] == 65 and c["avg"] == pytest.approx(3.8)


def test_parse_index_invalid_payload():
    assert tc.parse_index(None) == []
    assert tc.parse_index([1, 2]) == []
    assert tc.parse_index({"courses": [1]}) == []


def test_parse_index_skips_bad_sqid(caplog):
    payload = {"courses": {
        "坏课程": {"kcm": "坏课程", "sqid": "abc", "jsm": "x", "tid": 0, "kkdw": "y"},
        "好课程(甲)": {"kcm": "好课程", "sqid": "7", "jsm": "甲", "tid": 1, "kkdw": "z"},
    }}
    courses = tc.parse_index(payload)
    assert [c["sqid"] for c in courses] == [7]  # 字符串 sqid 宽松转 int


def test_merge_courses_adds_count_avg_and_flag():
    merged = tc.merge_courses(tc.parse_index(FULL_INDEX), tc.parse_index(COMMENT_INDEX))
    by_sqid = {c["sqid"]: c for c in merged}
    assert by_sqid[1]["has_reviews"] is False and by_sqid[1]["count"] == 0
    assert by_sqid[219]["has_reviews"] is True and by_sqid[219]["count"] == 1
    assert by_sqid[25437]["avg"] == pytest.approx(3.8)
    assert len(merged) == 3  # 全量索引课程一门不少


# ═══════════════════════════════════════════
# 点评页解析与分页
# ═══════════════════════════════════════════

def test_parse_course_page_results_and_next():
    reviews, nxt = tc.parse_course_page(COURSE_25437_PAGE1)
    assert len(reviews) == 1 and reviews[0]["id"] == 2483
    assert reviews[0]["rating"] == 1
    assert nxt == "/data/courses/25437_2.json"
    reviews2, nxt2 = tc.parse_course_page(COURSE_25437_PAGE2)
    assert nxt2 is None and reviews2[0]["score"] == pytest.approx(95.0)


def test_parse_course_page_invalid():
    assert tc.parse_course_page(None) == ([], None)
    assert tc.parse_course_page({"results": "oops", "next": 123}) == ([], None)


def test_parse_course_page_skips_missing_id(caplog):
    payload = {"count": 2, "next": None, "previous": None, "results": [
        {"rating": 3, "comment": "无 id 点评"},
        {"id": 9, "rating": 5, "comment": "正常"},
    ]}
    reviews, _ = tc.parse_course_page(payload)
    assert [r["id"] for r in reviews] == [9]


# ═══════════════════════════════════════════
# 条目契约
# ═══════════════════════════════════════════

ENTRY_KEYS = {"source", "source_id", "title", "content", "url",
              "metadata_json", "updated_at"}


def _course(sqid=219):
    merged = tc.merge_courses(tc.parse_index(FULL_INDEX), tc.parse_index(COMMENT_INDEX))
    return {c["sqid"]: c for c in merged}[sqid]


def test_make_course_entry_contract_with_reviews():
    entry = tc.make_course_entry(_course(219), now="2026-09-01T00:00:00+08:00")
    assert set(entry) == ENTRY_KEYS
    assert entry["source"] == "thucourse_course"
    assert entry["source_id"] == "thucourse:course:219"
    assert entry["title"] == "商法学（梁上上）"
    assert entry["url"] == "https://yourschool.cc.cd/thucourse/course.html?sqid=219"
    assert "商法学" in entry["content"] and "梁上上" in entry["content"]
    assert "法学院" in entry["content"] and "1 条学生点评" in entry["content"]
    meta = json.loads(entry["metadata_json"])
    assert meta["sqid"] == 219 and meta["count"] == 1 and meta["avg"] == pytest.approx(4.0)
    assert meta["has_reviews"] is True and meta["tid"] == 197
    assert entry["updated_at"] == "2026-09-01T00:00:00+08:00"


def test_make_course_entry_no_reviews_url_empty():
    entry = tc.make_course_entry(_course(1), now="2026-09-01T00:00:00+08:00")
    assert entry["url"] == ""
    assert "暂无学生点评" in entry["content"]
    meta = json.loads(entry["metadata_json"])
    assert meta["has_reviews"] is False
    assert meta["reviews_done"] is True  # 无点评课程视为已完成


def test_make_review_entry_contract_and_cleaning():
    course = _course(219)
    review = tc.parse_course_page(COURSE_219_PAGE)[0][0]
    entry = tc.make_review_entry(review, course, now="2026-09-01T00:00:00+08:00")
    assert set(entry) == ENTRY_KEYS
    assert entry["source"] == "thucourse_review"
    assert entry["source_id"] == "thucourse:review:1006"
    assert entry["title"] == "商法学（梁上上）点评 - 4星"
    # OCR 噪声已清洗：连续空白折叠、无换行
    assert entry["content"] == "老师不会点名 但是会点人回答问题 以及老师会记录谁发言了 会加分"
    assert entry["url"] == "https://yourschool.cc.cd/thucourse/course.html?sqid=219"
    meta = json.loads(entry["metadata_json"])
    assert meta == {"course_sqid": 219, "kcm": "商法学", "jsm": "梁上上",
                    "rating": 4, "score": None, "created_at": "2024/07/21 18:44"}


def test_make_review_entry_missing_id_returns_none():
    assert tc.make_review_entry({"rating": 3, "comment": "x"}, _course(219)) is None


def test_make_review_entry_empty_comment_no_rating_dropped():
    entry = tc.make_review_entry({"id": 5, "comment": "  \n ", "rating": None,
                                  "score": None}, _course(219))
    assert entry is None
    # 空点评但有评分仍保留（评分本身是可检索信息）
    kept = tc.make_review_entry({"id": 6, "comment": "", "rating": 5,
                                 "score": None}, _course(219))
    assert kept is not None and kept["content"] == ""


# ═══════════════════════════════════════════
# 抓取流程（fake http_get）
# ═══════════════════════════════════════════

def test_fetch_courses_merges_indexes():
    crawler = _crawler(FakeHTTP(_full_routes()))
    courses = crawler.fetch_courses()
    assert len(courses) == 3
    assert sum(1 for c in courses if c["has_reviews"]) == 2
    assert crawler.requests == 2


def test_fetch_courses_full_index_failure_returns_empty():
    http = FakeHTTP({tc.FULL_INDEX_URL: ConnectionError("boom")})
    crawler = _crawler(http)
    assert crawler.fetch_courses() == []  # 失败降级，绝不抛出
    # 全量索引失败即终止，不再请求有点评索引
    assert http.calls == [tc.FULL_INDEX_URL]


def test_fetch_courses_comment_index_failure_degrades():
    http = FakeHTTP(_full_routes(**{tc.COMMENT_INDEX_URL: ConnectionError("boom")}))
    crawler = _crawler(http)
    courses = crawler.fetch_courses()
    assert len(courses) == 3
    assert all(not c["has_reviews"] for c in courses)  # 降级为纯元数据


def test_fetch_reviews_follows_relative_pagination():
    crawler = _crawler(FakeHTTP(_full_routes()))
    reviews = crawler.fetch_reviews(_course(25437))
    assert [r["id"] for r in reviews] == [2483, 2484]  # 相对路径 next 已跟随
    assert crawler.requests == 2


def test_fetch_reviews_follows_absolute_next_url():
    page1 = dict(COURSE_25437_PAGE1)
    page1["next"] = tc.BASE_URL + "/data/courses/25437_2.json"  # 完整 URL 形态
    http = FakeHTTP(_full_routes(**{tc.COURSE_URL_TMPL.format(sqid=25437): page1}))
    crawler = _crawler(http)
    reviews = crawler.fetch_reviews(_course(25437))
    assert [r["id"] for r in reviews] == [2483, 2484]


def test_fetch_reviews_page_failure_returns_none():
    http = FakeHTTP(_full_routes(**{
        tc.BASE_URL + "/data/courses/25437_2.json": ConnectionError("boom")}))
    crawler = _crawler(http)
    assert crawler.fetch_reviews(_course(25437)) is None  # 整门课标记失败，不抛出


def test_throttle_sleeps_between_requests():
    sleeps = []
    crawler = _crawler(FakeHTTP(_full_routes()), sleeps=sleeps)
    crawler.fetch_courses()
    # 每轮首个请求不限速，第二次请求前 sleep(rate)
    assert sleeps == [0.5]


# ═══════════════════════════════════════════
# run 主流程：入库 / 断点续爬 / 失败降级
# ═══════════════════════════════════════════

def test_run_only_index_no_review_requests():
    http = FakeHTTP(_full_routes())
    crawler = _crawler(http)
    upsert = UpsertCollector()
    stats = crawler.run(upsert=upsert, only_index=True)
    assert stats["courses_indexed"] == 3
    assert stats["courses_upserted"] == 3
    assert stats["reviews_fetched"] == 0
    assert http.calls == [tc.FULL_INDEX_URL, tc.COMMENT_INDEX_URL]  # 不碰点评页
    assert all(e["source"] == "thucourse_course" for e in upsert.entries)


def test_run_full_upserts_reviews_and_marks_done(tmp_path):
    http = FakeHTTP(_full_routes())
    crawler = _crawler(http)
    upsert = UpsertCollector()
    progress = tmp_path / "progress.json"
    stats = crawler.run(upsert=upsert, progress_path=str(progress))
    assert stats["reviews_fetched"] == 3
    assert stats["reviews_upserted"] == 3
    assert stats["review_courses_done"] == 2
    assert stats["skipped_resume"] == 0
    reviews = [e for e in upsert.entries if e["source"] == "thucourse_review"]
    assert {e["source_id"] for e in reviews} == {
        "thucourse:review:1006", "thucourse:review:2483", "thucourse:review:2484"}
    # 完成后课程条目 reviews_done=True 重新入库
    done_flags = [json.loads(e["metadata_json"])["reviews_done"]
                  for e in upsert.entries
                  if e["source_id"] == "thucourse:course:219"]
    assert done_flags[-1] is True
    # 进度文件已落盘
    saved = json.loads(progress.read_text(encoding="utf-8"))
    assert saved["done_sqids"] == [219, 25437]


def test_run_resume_skips_completed_via_progress(tmp_path):
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps({"done_sqids": [219, 25437]}), encoding="utf-8")
    http = FakeHTTP(_full_routes())
    crawler = _crawler(http)
    upsert = UpsertCollector()
    stats = crawler.run(upsert=upsert, progress_path=str(progress))
    assert stats["skipped_resume"] == 2
    assert stats["reviews_fetched"] == 0
    # 已完成 sqid 不再请求点评页
    assert http.calls == [tc.FULL_INDEX_URL, tc.COMMENT_INDEX_URL]
    assert all(e["source"] == "thucourse_course" for e in upsert.entries)


def test_run_resume_skips_via_get_entry_no_progress_file(tmp_path):
    def fake_get_entry(source, source_id):
        assert source == "thucourse_course"
        if source_id == "thucourse:course:219":
            return {"metadata_json": json.dumps({"reviews_done": True})}
        return None

    http = FakeHTTP(_full_routes())
    crawler = _crawler(http)
    upsert = UpsertCollector()
    stats = crawler.run(upsert=upsert, get_entry=fake_get_entry,
                        progress_path=str(tmp_path / "p.json"))
    assert stats["skipped_resume"] == 1  # 219 被 get_entry 判定完成
    assert stats["review_courses_done"] == 1  # 25437 正常抓取
    assert {e["source_id"] for e in upsert.entries
            if e["source"] == "thucourse_review"} == {
        "thucourse:review:2483", "thucourse:review:2484"}


def test_run_course_failure_continues_others(tmp_path):
    http = FakeHTTP(_full_routes(**{
        tc.COURSE_URL_TMPL.format(sqid=219): ConnectionError("boom")}))
    crawler = _crawler(http)
    upsert = UpsertCollector()
    stats = crawler.run(upsert=upsert, progress_path=str(tmp_path / "p.json"))
    assert stats["failed_sqids"] == [219]
    assert stats["review_courses_done"] == 1  # 25437 不受影响
    assert stats["reviews_upserted"] == 2
    # 失败课程不进进度文件
    saved = json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))
    assert saved["done_sqids"] == [25437]


def test_run_max_courses_limits_index(tmp_path):
    http = FakeHTTP(_full_routes())
    crawler = _crawler(http)
    upsert = UpsertCollector()
    stats = crawler.run(upsert=upsert, max_courses=1,
                        progress_path=str(tmp_path / "p.json"))
    assert stats["courses_indexed"] == 1
    assert stats["reviews_fetched"] == 0  # 截断到 sqid=1（无点评）


def test_run_upsert_exception_degrades(tmp_path):
    http = FakeHTTP(_full_routes())
    crawler = _crawler(http)

    def bad_upsert(entries):
        raise RuntimeError("db locked")

    stats = crawler.run(upsert=bad_upsert, progress_path=str(tmp_path / "p.json"))
    assert stats["reviews_fetched"] == 3  # 抓取照常，落库失败不炸
    assert stats["reviews_upserted"] == 0


def test_run_without_upsert_is_dry_run(tmp_path):
    http = FakeHTTP(_full_routes())
    crawler = _crawler(http)
    stats = crawler.run(upsert=None, progress_path=str(tmp_path / "p.json"))
    assert stats["reviews_fetched"] == 3 and stats["reviews_upserted"] == 0
    assert stats["courses_upserted"] == 0


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

class FakeKB:
    """campus_kb 公开 API 的 fake 实现（init_db/upsert_entries/get_entry）。"""

    def __init__(self):
        self.inited = []
        self.entries = []

    def init_db(self, db_path=None):
        self.inited.append(db_path)

    def upsert_entries(self, entries, db_path=None):
        self.entries.extend(entries)
        return len(entries)

    def get_entry(self, source, source_id, db_path=None):
        return None


def test_main_cli_only_index(monkeypatch, tmp_path, capsys):
    kb = FakeKB()
    monkeypatch.setattr(tc, "_load_campus_kb", lambda: kb)
    http = FakeHTTP(_full_routes())
    rc = tc.main(["--only-index", "--db", str(tmp_path / "kb.db"),
                  "--progress-path", str(tmp_path / "p.json")],
                 http_get=http, sleep=lambda s: None)
    assert rc == 0
    assert kb.inited == [str(tmp_path / "kb.db")]
    assert len(kb.entries) == 3
    assert all(e["source"] == "thucourse_course" for e in kb.entries)
    out = capsys.readouterr().out
    assert "仅课程索引" in out and "索引 3 门" in out


def test_main_cli_full_run(monkeypatch, tmp_path):
    kb = FakeKB()
    monkeypatch.setattr(tc, "_load_campus_kb", lambda: kb)
    http = FakeHTTP(_full_routes())
    rc = tc.main(["--progress-path", str(tmp_path / "p.json")],
                 http_get=http, sleep=lambda s: None)
    assert rc == 0
    assert sum(1 for e in kb.entries if e["source"] == "thucourse_review") == 3


def test_main_cli_kb_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(tc, "_load_campus_kb", lambda: None)
    rc = tc.main(["--only-index"], http_get=FakeHTTP({}), sleep=lambda s: None)
    assert rc == 1
    assert "不可用" in capsys.readouterr().out
