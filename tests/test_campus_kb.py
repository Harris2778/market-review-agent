"""agent/campus_kb.py 校园知识库存储与中文检索层（校园知识库 v1）测试。

覆盖范围：
1. 建库：init_db 幂等；kb_entries 7 列齐全 + (source, source_id) 主键；
   FTS5+trigram 可用时建 kb_entries_fts 虚表，强制降级时不建。
2. 路径解析：显式参数 > CAMPUS_KB_DB_PATH > <项目根>/data/campus_kb.db；
   env 调用时惰性读取。
3. upsert：新增计数；(source, source_id) 重复全字段覆盖且行数不膨胀；
   参数校验（非 list / 非 dict / 缺 source / 缺 source_id）抛 ValueError；
   updated_at 缺省自动补时间戳；metadata_json 传 dict 自动序列化；
   数据库故障返回 0 不抛出。
4. search_kb：中文子串（短词 LIKE 路径 / 长词 FTS 路径）、英文大小写不敏感、
   多关键词 AND 语义、source 过滤、limit 夹取、相关度排序（标题命中优先）、
   score 字段存在且降序；空查询/库不存在/坏库返回 []。
5. 双模式：FTS 与强制 LIKE（monkeypatch _FTS_SUPPORTED=False）都验证检索。
6. 故障降级：db 路径指向目录 → upsert 0 / search [] / get None；
   坏库文件 → search []、stats 带 error 字段。
7. get_entry / stats：单条往返、未命中 None、按 source 分组计数、
   fts_mode 标识、库不存在时不带 error。

规则（与项目其他测试一致）：全 mock 零网络；库一律落在 tmp_path，
绝不写真实 data/。
"""

import os
import sqlite3

import pytest

import agent.campus_kb as kb


# ── 工具 ──


def make_entry(**kw) -> dict:
    """构造一条合法知识条目（8 字段契约），按需覆盖。"""
    entry = {
        "source": "sem_handbook",
        "source_id": "sem:default",
        "title": "清华选课手册总则",
        "content": "本科生选课分为正选、补退选两个阶段，请在选课系统开放时间内操作。",
        "url": "https://example.edu/handbook",
        "metadata_json": '{"year": 2024}',
        "updated_at": "2025-01-01T00:00:00",
    }
    entry.update(kw)
    return entry


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """把知识库隔离到 tmp_path（env 惰性解析路径），绝不触碰真实 data/。"""
    p = tmp_path / "campus_kb.db"
    monkeypatch.setenv(kb.CAMPUS_KB_DB_PATH_ENV, str(p))
    return str(p)


@pytest.fixture()
def seeded_db(db):
    """预置中英混合五条语料的库路径。"""
    kb.upsert_entries(
        [
            make_entry(),
            make_entry(
                source="thucourse_course",
                source_id="thucourse:course:219",
                title="Linear Algebra 线性代数",
                content="matrix determinant eigenvalue，考试难度大，给分一般。",
                url="https://thucourse.example/course/219",
            ),
            make_entry(
                source="thucourse_review",
                source_id="thucourse:review:9",
                title="某课评价一则",
                content="这门课的选课攻略很实用，绩点计算规则也讲清楚了。",
            ),
            make_entry(
                source="thucourse_summary",
                source_id="thucourse:summary:3",
                title="数据结构课程总结",
                content="期末考试重点是树与图，作业量适中，选课攻略见附录。",
            ),
            make_entry(
                source="thubook",
                source_id="thubook:专题/srt",
                title="SRT 专题笔记",
                content="大学生研究训练计划（SRT）报名指南与选题建议。",
            ),
        ],
        db_path=db,
    )
    return db


def _table_names(path: str) -> set:
    conn = sqlite3.connect(path)
    try:
        return {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        conn.close()


# ── 建库与路径解析 ──


def test_init_db_creates_table_columns_and_pk(db):
    kb.init_db(db)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        cols = {
            r["name"]: r for r in conn.execute("PRAGMA table_info(kb_entries)")
        }
    finally:
        conn.close()
    assert set(cols) == {
        "source", "source_id", "title", "content",
        "url", "metadata_json", "updated_at",
    }
    pk_cols = {name for name, r in cols.items() if r["pk"]}
    assert pk_cols == {"source", "source_id"}


def test_init_db_idempotent(db):
    kb.init_db(db)
    kb.init_db(db)  # 第二次不抛异常
    assert os.path.isfile(db)


@pytest.mark.skipif(not kb._fts_supported(), reason="sqlite3 未编译 FTS5+trigram")
def test_init_db_creates_fts_table_when_supported(db):
    kb.init_db(db)
    assert "kb_entries_fts" in _table_names(db)


def test_init_db_skips_fts_table_when_forced_like(db, monkeypatch):
    monkeypatch.setattr(kb, "_FTS_SUPPORTED", False)
    kb.init_db(db)
    names = _table_names(db)
    assert "kb_entries" in names
    assert "kb_entries_fts" not in names


def test_db_path_explicit_param_beats_env(db, tmp_path):
    explicit = str(tmp_path / "explicit.db")
    assert kb._db_path(explicit) == explicit
    kb.init_db(explicit)
    assert os.path.isfile(explicit)


def test_db_path_env_used_when_no_param(db):
    assert kb._db_path() == db
    kb.init_db()  # 走 env 惰性解析
    assert os.path.isfile(db)


def test_db_path_default_under_project_data(monkeypatch):
    monkeypatch.delenv(kb.CAMPUS_KB_DB_PATH_ENV, raising=False)
    monkeypatch.delenv(kb.DATA_DIR_ENV, raising=False)
    path = kb._db_path()
    assert path.endswith(os.path.join("data", "campus_kb.db"))
    assert os.path.dirname(os.path.dirname(path)) == os.path.dirname(
        os.path.dirname(os.path.abspath(kb.__file__))
    )


def test_db_path_data_dir_absolute_used(monkeypatch, tmp_path):
    """DATA_DIR 为绝对路径（如 Railway 卷 /data）时直接使用。"""
    monkeypatch.delenv(kb.CAMPUS_KB_DB_PATH_ENV, raising=False)
    monkeypatch.setenv(kb.DATA_DIR_ENV, str(tmp_path))
    path = kb._db_path()
    assert path == str(tmp_path / "campus_kb.db")
    kb.init_db()  # 惰性解析落库到 DATA_DIR
    assert os.path.isfile(path)


def test_db_path_data_dir_relative_resolves_under_project(monkeypatch):
    monkeypatch.delenv(kb.CAMPUS_KB_DB_PATH_ENV, raising=False)
    monkeypatch.setenv(kb.DATA_DIR_ENV, "custom_data")
    path = kb._db_path()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(kb.__file__)))
    assert path == os.path.join(project_root, "custom_data", "campus_kb.db")


def test_db_path_env_beats_data_dir(monkeypatch, tmp_path):
    """CAMPUS_KB_DB_PATH 优先级高于 DATA_DIR。"""
    monkeypatch.setenv(kb.CAMPUS_KB_DB_PATH_ENV, str(tmp_path / "env.db"))
    monkeypatch.setenv(kb.DATA_DIR_ENV, str(tmp_path / "other"))
    assert kb._db_path() == str(tmp_path / "env.db")


# ── upsert ──


def test_upsert_returns_count_and_roundtrip(db):
    n = kb.upsert_entries([make_entry(), make_entry(source_id="sem:2")], db_path=db)
    assert n == 2
    entry = kb.get_entry("sem_handbook", "sem:default", db_path=db)
    assert entry is not None
    assert entry["title"] == "清华选课手册总则"
    assert entry["url"] == "https://example.edu/handbook"
    assert entry["metadata_json"] == '{"year": 2024}'
    assert entry["updated_at"] == "2025-01-01T00:00:00"
    assert "score" not in entry  # get_entry 不附 score


def test_upsert_duplicate_overwrites_without_growth(db):
    kb.upsert_entries([make_entry(content="旧版本内容")], db_path=db)
    n = kb.upsert_entries([make_entry(content="新版本内容")], db_path=db)
    assert n == 1
    assert kb.get_entry("sem_handbook", "sem:default", db_path=db)["content"] == (
        "新版本内容"
    )
    assert kb.stats(db_path=db)["total"] == 1


def test_upsert_fills_updated_at_when_missing(db):
    entry = make_entry()
    del entry["updated_at"]
    kb.upsert_entries([entry], db_path=db)
    got = kb.get_entry("sem_handbook", "sem:default", db_path=db)
    assert got["updated_at"]  # 自动补非空 ISO 时间戳


def test_upsert_metadata_dict_auto_serialized(db):
    kb.upsert_entries(
        [make_entry(metadata_json={"course_no": 219, "tags": ["硬课"]})],
        db_path=db,
    )
    got = kb.get_entry("sem_handbook", "sem:default", db_path=db)
    assert '"course_no": 219' in got["metadata_json"]


def test_upsert_valueerror_non_list(db):
    with pytest.raises(ValueError):
        kb.upsert_entries({"source": "x"}, db_path=db)


def test_upsert_valueerror_entry_not_dict(db):
    with pytest.raises(ValueError):
        kb.upsert_entries(["not-a-dict"], db_path=db)


def test_upsert_valueerror_missing_source(db):
    with pytest.raises(ValueError):
        kb.upsert_entries([make_entry(source="  ")], db_path=db)


def test_upsert_valueerror_missing_source_id(db):
    with pytest.raises(ValueError):
        kb.upsert_entries([make_entry(source_id="")], db_path=db)


def test_upsert_db_failure_returns_zero(tmp_path):
    # db 路径指向目录：sqlite3.connect 必失败 → 返回 0 不抛出
    assert kb.upsert_entries([make_entry()], db_path=str(tmp_path)) == 0


# ── search_kb：命中与语义 ──


def test_search_chinese_short_keyword_substring(seeded_db):
    """2 字中文短词（trigram 无法索引）自动走 LIKE 路径仍命中。"""
    hits = kb.search_kb("选课", db_path=seeded_db)
    ids = {h["source_id"] for h in hits}
    assert "sem:default" in ids
    assert "thucourse:review:9" in ids
    assert "thubook:专题/srt" not in ids  # 不含「选课」


def test_search_chinese_long_keyword_fts_path(seeded_db):
    hits = kb.search_kb("选课攻略", db_path=seeded_db)
    ids = {h["source_id"] for h in hits}
    assert ids == {"thucourse:review:9", "thucourse:summary:3"}


def test_search_english_case_insensitive(seeded_db):
    lower = {h["source_id"] for h in kb.search_kb("matrix", db_path=seeded_db)}
    upper = {h["source_id"] for h in kb.search_kb("MATRIX", db_path=seeded_db)}
    assert lower == upper == {"thucourse:course:219"}


def test_search_multi_keyword_and_semantics(seeded_db):
    hits = kb.search_kb("选课 绩点", db_path=seeded_db)
    # 严格 AND 命中排最前；宽松兜底可在其后补全部分命中条目
    assert hits[0]["source_id"] == "thucourse:review:9"


def test_search_source_filter(seeded_db):
    hits = kb.search_kb("选课", source="thucourse_review", db_path=seeded_db)
    assert {h["source_id"] for h in hits} == {"thucourse:review:9"}
    assert all(h["source"] == "thucourse_review" for h in hits)


def test_search_limit_and_clamp(seeded_db):
    hits = kb.search_kb("选课", limit=2, db_path=seeded_db)
    assert len(hits) == 2
    clamped_low = kb.search_kb("选课", limit=0, db_path=seeded_db)
    assert len(clamped_low) == 1  # 0 夹取到最小 1
    clamped_high = kb.search_kb("选课", limit=999, db_path=seeded_db)
    assert len(clamped_high) == 3  # 上限不放大真实命中数


def test_search_empty_query_returns_empty(seeded_db):
    assert kb.search_kb("   ", db_path=seeded_db) == []
    assert kb.search_kb("", db_path=seeded_db) == []


def test_search_no_hit_returns_empty(seeded_db):
    assert kb.search_kb("量子引力波", db_path=seeded_db) == []


def test_search_missing_db_returns_empty(tmp_path):
    assert kb.search_kb("选课", db_path=str(tmp_path / "nope.db")) == []


# ── search_kb：排序与 score ──


def test_search_relevance_title_beats_content(db):
    kb.upsert_entries(
        [
            make_entry(source_id="a:1", title="绩点计算规则详解", content="正文无关键词。"),
            make_entry(
                source_id="b:1", title="学生手册",
                content="这里介绍绩点计算规则的细则与历史沿革，绩点计算规则如下。",
            ),
        ],
        db_path=db,
    )
    # 长关键词走 FTS（bm25 标题加权），短/常规走 LIKE（标题命中 ×3），均标题优先
    for query in ("绩点计算规则", "绩点"):
        hits = kb.search_kb(query, db_path=db)
        assert hits[0]["source_id"] == "a:1", query


def test_search_score_field_present_and_descending(seeded_db):
    hits = kb.search_kb("选课", db_path=seeded_db)
    assert hits
    assert all(isinstance(h["score"], (int, float)) for h in hits)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_result_has_full_entry_fields(seeded_db):
    hit = kb.search_kb("SRT", db_path=seeded_db)[0]
    assert set(hit) == {
        "source", "source_id", "title", "content",
        "url", "metadata_json", "updated_at", "score",
    }


# ── 强制 LIKE 降级模式 ──


@pytest.fixture()
def like_db(tmp_path, monkeypatch):
    """强制禁用 FTS 的隔离库：建库前 monkeypatch，全程 LIKE 路径。"""
    monkeypatch.setattr(kb, "_FTS_SUPPORTED", False)
    p = str(tmp_path / "like.db")
    kb.init_db(p)
    kb.upsert_entries(
        [
            make_entry(),
            make_entry(
                source="thucourse_review", source_id="thucourse:review:9",
                title="某课评价", content="选课攻略与绩点计算。",
            ),
        ],
        db_path=p,
    )
    return p


def test_like_mode_search_hits(like_db):
    hits = kb.search_kb("选课攻略", db_path=like_db)
    assert {h["source_id"] for h in hits} == {"thucourse:review:9"}


def test_like_mode_stats_reports_like(like_db):
    s = kb.stats(db_path=like_db)
    assert s["fts_mode"] == "like"
    assert s["total"] == 2


def test_fts_mode_stats_reports_fts5(seeded_db):
    if not kb._fts_supported():
        pytest.skip("sqlite3 未编译 FTS5+trigram")
    assert kb.stats(db_path=seeded_db)["fts_mode"] == "fts5"


# ── 故障降级 ──


def test_search_db_path_is_directory_returns_empty(tmp_path):
    assert kb.search_kb("选课", db_path=str(tmp_path)) == []
    assert kb.get_entry("sem_handbook", "sem:default", db_path=str(tmp_path)) is None


def test_corrupt_db_search_empty_and_stats_error(tmp_path):
    bad = tmp_path / "corrupt.db"
    bad.write_bytes(b"this is not a sqlite database at all")
    path = str(bad)
    assert kb.search_kb("选课", db_path=path) == []
    s = kb.stats(db_path=path)
    assert s["total"] == 0
    assert s["by_source"] == {}
    assert "error" in s  # 坏库必须带 error 字段


def test_init_db_failure_does_not_raise(tmp_path):
    kb.init_db(str(tmp_path))  # 目录路径：内部记日志，不抛出


# ── get_entry / stats ──


def test_get_entry_not_found_and_missing_db(seeded_db, tmp_path):
    assert kb.get_entry("sem_handbook", "sem:nope", db_path=seeded_db) is None
    assert kb.get_entry("sem_handbook", "x", db_path=str(tmp_path / "no.db")) is None
    assert kb.get_entry("", "x", db_path=seeded_db) is None  # 空主键直接 None


def test_stats_grouping_and_total(seeded_db):
    s = kb.stats(db_path=seeded_db)
    assert s["total"] == 5
    assert s["by_source"] == {
        "sem_handbook": 1,
        "thucourse_course": 1,
        "thucourse_review": 1,
        "thucourse_summary": 1,
        "thubook": 1,
    }
    assert s["db_path"] == seeded_db


def test_stats_missing_db_is_zero_without_error(tmp_path):
    s = kb.stats(db_path=str(tmp_path / "nope.db"))
    assert s["total"] == 0
    assert s["by_source"] == {}
    assert "error" not in s  # 库不存在属正常初始态


def test_stats_uses_env_lazy_resolution(seeded_db):
    s = kb.stats()  # 不传 db_path，走 env 惰性解析
    assert s["total"] == 5


# ── 检索质量：停用词规范化 + 宽松兜底 + 粘连长词扩词（2026-07-23 生产反馈修复）──


def _seed_dorm_corpus(db):
    """宿舍场景语料：快递页（含校名+宿舍楼字样）与宿舍介绍页（无校名字样）。"""
    kb.upsert_entries([
        make_entry(
            source="thubook", source_id="thubook:express",
            title="快递 - 大件物品采买与快递信息",
            content="清华大学学生公寓快递地址：南区宿舍楼填清华大学学生公寓xx楼，紫荆公寓填清华大学紫荆公寓xx楼。",
        ),
        make_entry(
            source="thubook", source_id="thubook:dorm",
            title="宿舍介绍 - 南区",
            content="南区宿舍为4人间上床下桌，无中厅，楼内有淋浴间与公共洗衣房。",
        ),
    ])


def test_keywords_stopword_substring_strip():
    """粘连问句的停用词子串剥离：『清华大学宿舍怎么样』→『宿舍』。"""
    assert kb._keywords("清华大学宿舍怎么样") == ["宿舍"]
    assert kb._keywords("保研怎么办？") == ["保研"]
    assert kb._keywords("清华大学 宿舍") == ["宿舍"]


def test_keywords_all_stopwords_fallback():
    """查询全是停用词时回退未过滤词表，避免空查询。"""
    assert kb._keywords("清华大学") == ["清华大学"]


def test_keywords_punctuation_stripped():
    assert kb._keywords("宿舍，食堂？") == ["宿舍", "食堂"]


def test_search_noise_school_name_still_finds_dorm_page(db):
    """生产事故复现：『清华大学 宿舍』曾只命中快递页，
    宿舍介绍页因不含校名被严格 AND 误杀；修复后应排第一。"""
    _seed_dorm_corpus(db)
    rs = kb.search_kb("清华大学 宿舍", limit=5)
    ids = [r["source_id"] for r in rs]
    assert "thubook:dorm" in ids
    assert ids[0] == "thubook:dorm"


def test_search_fused_question_finds_dorm_page(db):
    """无空格自然问句整串检索：『清华大学宿舍怎么样』。"""
    _seed_dorm_corpus(db)
    rs = kb.search_kb("清华大学宿舍怎么样", limit=5)
    assert rs and rs[0]["source_id"] == "thubook:dorm"


def test_relaxed_partial_match_fills_results(db):
    """严格 AND 不足额时宽松 OR 补全，且多命中词文档排前。"""
    kb.upsert_entries([
        make_entry(
            source="sem_handbook", source_id="h:both",
            title="保研加分细则",
            content="科研竞赛保研加分政策说明。",
        ),
        make_entry(
            source="sem_handbook", source_id="h:one",
            title="保研流程",
            content="推免流程与时间安排。",
        ),
    ])
    rs = kb.search_kb("保研 加分 不存在词xyz", limit=5)
    ids = [r["source_id"] for r in rs]
    assert ids[:2] == ["h:both", "h:one"]  # 命中 2 词的排前，1 词的兜底入选


def test_relaxed_fused_long_word_bigram_expansion(db):
    """粘连长词（≥5字纯中文）在宽松兜底中按二字组扩词召回。"""
    kb.upsert_entries([
        make_entry(
            source="sem_handbook", source_id="h:baoyan",
            title="保研手册 - 申请资格",
            content="总评成绩排名与加分政策：科研竞赛加分计入综合成绩。",
        ),
    ])
    rs = kb.search_kb("经管保研加分政策", limit=5)
    assert any(r["source_id"] == "h:baoyan" for r in rs)


def test_expand_relaxed_keywords_only_long_cjk():
    assert "经管" in kb._expand_relaxed_keywords(["经管保研加分政策"])
    # 短词与非纯中文词不扩
    assert kb._expand_relaxed_keywords(["宿舍"]) == ["宿舍"]
    assert kb._expand_relaxed_keywords(["GPA计算"]) == ["GPA计算"]


def test_strict_and_still_preferred_over_partial(db):
    """严格 AND 命中足够时不触发兜底排序变化。"""
    kb.upsert_entries([
        make_entry(source="thubook", source_id="t:both",
                   title="选课与绩点", content="选课规则与绩点计算方式。"),
        make_entry(source="thubook", source_id="t:one",
                   title="选课简介", content="本科生选课阶段安排。"),
    ])
    rs = kb.search_kb("选课 绩点", limit=5)
    assert rs[0]["source_id"] == "t:both"
