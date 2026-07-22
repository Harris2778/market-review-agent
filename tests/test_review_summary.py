"""agent/review_summary.py + scripts/generate_course_summaries.py 测试。

覆盖范围（全 mock、零网络、零真实外部文件/模块依赖）：
1. summarize_course_reviews 公开契约：返回结构六字段完整性、
   空列表/None/畸形条目输入返回 review_count=0 空总结、绝不抛异常。
2. 评分提取与归一：metadata_json 多键（rating/score/评分）、字符串数值、
   十分制折半、布尔/非法值忽略、无评分时 avg=None 且 dist={}。
3. fallback 确定性路径：平均评分与分布统计正确、总结文本标注
   「基于 N 条点评的自动摘要」、高频关键词词频正确性、highlights
   3~5 条且按评分分层覆盖、同输入重复调用结果一致。
4. LLM 路径：fake llm_fn 三态（正常 JSON / 返回垃圾 / 抛异常）——
   正常走 method='llm' 且 prompt 含课程名/评分分布/点评原文，
   垃圾与异常均自动降级 fallback；markdown 围栏包裹的 JSON 可解析；
   超长点评列表的 prompt 按预算截断。
5. build_summary_entry：8 字段契约完整、source/source_id/url 规则、
   metadata_json 可解析且字段齐全、畸形输入不抛异常。
6. 分组脚本：course_key 多路径（metadata 优先 → source_id → title）、
   run 主流程（fake kb 分组/逐课程 upsert/统计）、--limit、
   sleep 注入断言、--use-llm 未接线时降级 fallback、单课程失败继续、
   main CLI 全链路（monkeypatch _load_campus_kb）。
"""

import json

import pytest

from agent import review_summary as rs
import scripts.generate_course_summaries as gcs


# ── 公共工具 ──

def make_review(content, rating=None, sqid="219", title="操作系统",
                source_id=None):
    """构造符合全局契约的 thucourse_review 条目。"""
    meta = {"course_sqid": sqid, "course_title": title}
    if rating is not None:
        meta["rating"] = rating
    return {
        "source": "thucourse_review",
        "source_id": source_id or f"thucourse:review:{sqid}:{abs(hash(content)) % 99999}",
        "title": title,
        "content": content,
        "url": f"https://thucourse.example/review/{sqid}",
        "metadata_json": json.dumps(meta, ensure_ascii=False),
        "updated_at": "2026-08-01T10:00:00+08:00",
    }


SUMMARY_KEYS = {"summary_text", "rating_avg", "rating_dist",
                "review_count", "highlights", "method"}
ENTRY_KEYS = {"source", "source_id", "title", "content", "url",
              "metadata_json", "updated_at"}


# ═══════════════════════════════════════════
# 1. summarize_course_reviews 公开契约
# ═══════════════════════════════════════════

def test_empty_reviews_returns_zero_count_structure():
    result = rs.summarize_course_reviews("操作系统", [])
    assert set(result) == SUMMARY_KEYS
    assert result["review_count"] == 0
    assert result["rating_avg"] is None
    assert result["rating_dist"] == {}
    assert result["highlights"] == []
    assert result["method"] == "fallback"
    assert "0 条点评" in result["summary_text"]


def test_none_and_garbage_input_never_raise():
    for bad in (None, "not-a-list", 123):
        result = rs.summarize_course_reviews("操作系统", bad)
        assert result["review_count"] == 0
    # 列表内混入非 dict 畸形条目被过滤
    result = rs.summarize_course_reviews("操作系统", [None, "x", 42])
    assert result["review_count"] == 0
    # 课程名为 None 也不抛异常
    result = rs.summarize_course_reviews(None, [make_review("这门课作业很多，每周都要写代码。")])
    assert result["review_count"] == 1


# ═══════════════════════════════════════════
# 2. 评分提取与归一
# ═══════════════════════════════════════════

def test_rating_stats_average_and_distribution():
    reviews = [
        make_review("老师讲得很好，给分也不错，推荐选课。", rating=5),
        make_review("工作量偏大但收获很多，期末考核合理。", rating=4),
        make_review("课程内容一般，考核方式比较随意。", rating=2),
    ]
    result = rs.summarize_course_reviews("操作系统", reviews)
    assert result["rating_avg"] == pytest.approx(3.67)
    assert result["rating_dist"] == {"5": 1, "4": 1, "2": 1}
    assert result["review_count"] == 3
    assert result["method"] == "fallback"


def test_rating_metadata_key_variants_and_string_numbers():
    r1 = make_review("这门课挺不错的，值得推荐给大家。", rating=None)
    r1["metadata_json"] = json.dumps({"course_sqid": "1", "score": "4.5"})
    r2 = make_review("这门课挺不错的，值得推荐给大家。", rating=None)
    r2["metadata_json"] = json.dumps({"course_sqid": "1", "评分": 3})
    result = rs.summarize_course_reviews("课", [r1, r2])
    assert result["rating_avg"] == pytest.approx(3.75)
    # round() 为银行家舍入：4.5 → 4 分桶，3 → 3 分桶
    assert result["rating_dist"] == {"4": 1, "3": 1}


def test_ten_point_scale_is_halved():
    reviews = [make_review("十分制打分场景下这门课非常好。", rating=9)]
    result = rs.summarize_course_reviews("课", reviews)
    assert result["rating_avg"] == pytest.approx(4.5)
    assert result["rating_dist"] == {"5": 1} or result["rating_dist"] == {"4": 1}


def test_invalid_ratings_are_ignored():
    r1 = make_review("无评分点评，内容还算具体，有一定参考价值。")
    r2 = make_review("布尔评分不应生效，内容还算具体，有参考价值。")
    r2["metadata_json"] = json.dumps({"course_sqid": "1", "rating": True})
    r3 = make_review("越界评分不应生效，内容还算具体，有参考价值。")
    r3["metadata_json"] = json.dumps({"course_sqid": "1", "rating": 99})
    result = rs.summarize_course_reviews("课", [r1, r2, r3])
    assert result["rating_avg"] is None
    assert result["rating_dist"] == {}
    assert result["review_count"] == 3


# ═══════════════════════════════════════════
# 3. fallback 确定性路径
# ═══════════════════════════════════════════

def test_fallback_marks_auto_summary_and_count():
    reviews = [make_review("这门课作业量很大，每周都有编程作业。", rating=4),
               make_review("老师讲课清晰，考核以项目为主。", rating=5)]
    result = rs.summarize_course_reviews("操作系统", reviews)
    assert "基于 2 条点评的自动摘要" in result["summary_text"]
    assert "《操作系统》" in result["summary_text"]
    assert "平均评分 4.5/5" in result["summary_text"]


def test_fallback_keyword_frequency_extraction():
    # 「作业量很大」在 3 条点评中重复出现，应进入高频关键词/代表性观点
    repeated = "这门课作业量很大，每周都要花十几个小时写作业"
    reviews = [make_review(f"{repeated}，第{i}学期修读。", rating=4)
               for i in range(3)]
    reviews.append(make_review("老师人很好，课堂气氛轻松活跃。", rating=5))
    result = rs.summarize_course_reviews("操作系统", reviews)
    text = result["summary_text"] + "".join(result["highlights"])
    assert "作业量很大" in text or "作业" in result["summary_text"]
    assert "高频提及" in result["summary_text"]


def test_fallback_highlights_count_and_content():
    reviews = [
        make_review("这门课工作量适中，作业每周一次，难度可以接受。", rating=4),
        make_review("老师讲课非常清晰，课件质量高，推荐认真听讲。", rating=5),
        make_review("期末考试难度大，给分比较严格，需要提前复习。", rating=2),
        make_review("考核方式为期末考试加课程项目，项目占比四成。", rating=3),
    ]
    result = rs.summarize_course_reviews("操作系统", reviews)
    assert 3 <= len(result["highlights"]) <= 5
    for h in result["highlights"]:
        assert isinstance(h, str) and len(h) >= 6
    # 代表性句子来自点评原文
    corpus = "".join(r["content"] for r in reviews)
    assert any(h[:20] in corpus for h in result["highlights"])


def test_fallback_highlights_stratified_by_rating():
    # 高分与低分点评的代表性观点都应被覆盖
    positive = "这门课老师讲得极其精彩，案例丰富，收获特别大。"
    negative = "这门课给分极差，工作量爆炸，期末考试完全不考上课内容。"
    reviews = [make_review(positive, rating=5),
               make_review(negative, rating=1)]
    result = rs.summarize_course_reviews("课", reviews)
    joined = "".join(result["highlights"])
    assert "精彩" in joined and "给分极差" in joined


def test_fallback_is_deterministic():
    reviews = [
        make_review("作业量很大但很有收获，助教研讨课质量高。", rating=4),
        make_review("作业量很大，考试难度也不小，需要投入时间。", rating=3),
        make_review("老师讲课节奏快，适合有基础的同学选修。", rating=4),
    ]
    r1 = rs.summarize_course_reviews("操作系统", reviews)
    r2 = rs.summarize_course_reviews("操作系统", list(reviews))
    assert r1 == r2


# ═══════════════════════════════════════════
# 4. LLM 路径
# ═══════════════════════════════════════════

def _good_llm(prompt):
    return json.dumps({
        "summary_text": "该课工作量适中、给分宽松、教学质量高、"
                        "考核以项目为主，适合对系统方向感兴趣的同学。",
        "highlights": ["工作量适中", "给分宽松", "适合系统方向"],
    }, ensure_ascii=False)


def test_llm_path_success():
    reviews = [make_review("这门课质量不错，项目考核，推荐选修。", rating=4)]
    result = rs.summarize_course_reviews("操作系统", reviews, llm_fn=_good_llm)
    assert result["method"] == "llm"
    assert "工作量" in result["summary_text"]
    assert result["highlights"] == ["工作量适中", "给分宽松", "适合系统方向"]
    assert result["review_count"] == 1
    assert result["rating_avg"] == pytest.approx(4.0)


def test_llm_prompt_contains_context():
    captured = {}

    def spy_llm(prompt):
        captured["prompt"] = prompt
        return _good_llm(prompt)

    reviews = [
        make_review("作业很多每周都要交，考试是开卷形式。", rating=3, sqid="219"),
        make_review("给分非常好老师也很负责，强烈推荐。", rating=5, sqid="219"),
    ]
    rs.summarize_course_reviews("操作系统", reviews, llm_fn=spy_llm)
    prompt = captured["prompt"]
    assert "《操作系统》" in prompt
    assert "点评总数：2 条" in prompt
    assert "分布" in prompt and "5分×1" in prompt and "3分×1" in prompt
    assert "作业很多每周都要交" in prompt
    assert "给分非常好老师也很负责" in prompt
    for dim in ("工作量", "给分", "教学质量", "考核方式", "适合人群"):
        assert dim in prompt


def test_llm_prompt_budget_truncation():
    captured = {}

    def spy_llm(prompt):
        captured["prompt"] = prompt
        return _good_llm(prompt)

    # 30 条 × 800 字截断后远超 6000 字预算，应触发分层抽样
    long_text = "这门课内容非常充实。" * 100
    reviews = [make_review(f"{long_text}编号{i}。", rating=3 + i % 3)
               for i in range(30)]
    result = rs.summarize_course_reviews("课", reviews, llm_fn=spy_llm)
    assert result["method"] == "llm"
    assert "分层抽样展示" in captured["prompt"]
    assert len(captured["prompt"]) < 12000


def test_llm_garbage_output_falls_back():
    def garbage_llm(prompt):
        return "我是一株植物，不会输出 JSON。🌱🌱🌱"

    reviews = [make_review("这门课作业量很大，每周都有编程作业。", rating=4)]
    result = rs.summarize_course_reviews("操作系统", reviews, llm_fn=garbage_llm)
    assert result["method"] == "fallback"
    assert "基于 1 条点评的自动摘要" in result["summary_text"]


def test_llm_raising_falls_back():
    def boom_llm(prompt):
        raise RuntimeError("LLM 服务挂了")

    reviews = [make_review("这门课作业量很大，每周都有编程作业。", rating=4)]
    result = rs.summarize_course_reviews("操作系统", reviews, llm_fn=boom_llm)
    assert result["method"] == "fallback"
    assert result["review_count"] == 1


def test_llm_fenced_json_is_parsed():
    def fenced_llm(prompt):
        return ("好的，以下是总结：\n```json\n"
                '{"summary_text": "工作量饱满，给分中等，教学质量优秀，'
                '考核为闭卷考试，适合数理基础好的同学。", '
                '"highlights": ["教学质量优秀"]}\n```')

    reviews = [make_review("闭卷考试难度大，但老师讲得很好。", rating=4)]
    result = rs.summarize_course_reviews("课", reviews, llm_fn=fenced_llm)
    assert result["method"] == "llm"
    assert "闭卷考试" in result["summary_text"]


def test_llm_empty_summary_text_falls_back():
    def empty_llm(prompt):
        return '{"summary_text": "", "highlights": []}'

    reviews = [make_review("这门课作业量很大，每周都有编程作业。", rating=4)]
    result = rs.summarize_course_reviews("课", reviews, llm_fn=empty_llm)
    assert result["method"] == "fallback"


# ═══════════════════════════════════════════
# 5. build_summary_entry
# ═══════════════════════════════════════════

def test_build_summary_entry_contract_fields():
    reviews = [make_review("这门课作业量很大，每周都有编程作业。", rating=4,
                           sqid="219")]
    summary = rs.summarize_course_reviews("操作系统", reviews)
    entry = rs.build_summary_entry("219", "操作系统", summary)
    assert set(entry) == ENTRY_KEYS
    assert entry["source"] == "thucourse_summary"
    assert entry["source_id"] == "thucourse:summary:219"
    assert entry["title"] == "操作系统 · 点评综合总结"
    assert entry["content"] == summary["summary_text"]
    assert entry["url"] == "thucourse:course:219"
    meta = json.loads(entry["metadata_json"])
    assert meta["course_sqid"] == "219"
    assert meta["course_title"] == "操作系统"
    assert meta["review_count"] == 1
    assert meta["rating_avg"] == pytest.approx(4.0)
    assert meta["method"] == "fallback"
    assert isinstance(meta["highlights"], list)
    # updated_at 是合法 ISO 时间戳
    assert "T" in entry["updated_at"]


def test_build_summary_entry_garbage_input_never_raise():
    entry = rs.build_summary_entry(None, None, None)
    assert entry["source"] == "thucourse_summary"
    assert entry["source_id"] == "thucourse:summary:unknown"
    assert json.loads(entry["metadata_json"]) == {} or True  # 可解析即可
    entry2 = rs.build_summary_entry("x", "课", {"summary_text": 123})
    assert entry2["content"] == "123"


# ═══════════════════════════════════════════
# 6. 分组脚本
# ═══════════════════════════════════════════

def test_course_key_metadata_priority():
    entry = make_review("内容足够长的点评文本。", sqid="777", title="软件工程")
    sqid, title = gcs.course_key(entry)
    assert sqid == "777" and title == "软件工程"


def test_course_key_fallbacks():
    # metadata 无 sqid → 从 source_id 解析
    entry = make_review("内容足够长的点评文本。")
    entry["metadata_json"] = "{}"
    entry["source_id"] = "thucourse:review:555:42"
    entry["title"] = "编译原理"
    assert gcs.course_key(entry) == ("555", "编译原理")
    # source_id 也不可用 → title 兜底
    entry["source_id"] = "garbage"
    sqid, title = gcs.course_key(entry)
    assert sqid == "title:编译原理" and title == "编译原理"
    # metadata_json 非法 JSON 不抛异常
    entry["metadata_json"] = "{not json"
    assert gcs.course_key(entry)[1] == "编译原理"


def test_group_reviews_by_course():
    reviews = [
        make_review("点评一内容足够长。", sqid="219"),
        make_review("点评二内容足够长。", sqid="219"),
        make_review("点评三内容足够长。", sqid="100"),
    ]
    groups = gcs.group_reviews_by_course(reviews)
    assert set(groups) == {"219", "100"}
    assert len(groups["219"]["reviews"]) == 2
    assert groups["219"]["title"] == "操作系统"


class FakeCampusKB:
    """campus_kb 假实现：内存存储 + 调用记录。"""

    def __init__(self, reviews):
        self._reviews = reviews
        self.upserted = []
        self.init_called_with = "unset"

    def init_db(self, db_path=None):
        self.init_called_with = db_path

    def search_kb(self, query, source=None, limit=10, db_path=None):
        assert source == "thucourse_review"
        return list(self._reviews)

    def upsert_entries(self, entries, db_path=None):
        self.upserted.extend(entries)
        return len(entries)


def _three_course_reviews():
    return [
        make_review("这门课作业量很大，每周都有编程作业。", rating=4, sqid="219"),
        make_review("给分很好，老师讲课清晰，推荐选修。", rating=5, sqid="219"),
        make_review("考核方式为期末考试，难度适中。", rating=3, sqid="100"),
        make_review("项目驱动教学，收获非常大。", rating=5, sqid="777"),
    ]


def test_run_main_flow_groups_and_upserts():
    kb = FakeCampusKB(_three_course_reviews())
    stats = gcs.run(kb, sleep=lambda s: None, interval=0)
    assert stats["reviews"] == 4
    assert stats["courses"] == 3
    assert stats["upserted"] == 3
    assert stats["failed"] == 0
    assert stats["llm_used"] is False
    sqids = {e["source_id"] for e in kb.upserted}
    assert sqids == {"thucourse:summary:219", "thucourse:summary:100",
                     "thucourse:summary:777"}
    for e in kb.upserted:
        assert e["source"] == "thucourse_summary"
        assert set(e) == ENTRY_KEYS
    # 219 号课程合并了 2 条点评
    e219 = next(e for e in kb.upserted if e["source_id"].endswith(":219"))
    assert json.loads(e219["metadata_json"])["review_count"] == 2


def test_run_limit_caps_courses():
    kb = FakeCampusKB(_three_course_reviews())
    stats = gcs.run(kb, limit=1, sleep=lambda s: None, interval=0)
    assert stats["courses"] == 1
    assert len(kb.upserted) == 1


def test_run_sleep_injected_between_courses():
    kb = FakeCampusKB(_three_course_reviews())
    calls = []
    gcs.run(kb, interval=0.25, sleep=calls.append)
    assert calls == [0.25, 0.25]  # 3 门课程，首个不限速


def test_run_use_llm_unwired_falls_back(monkeypatch):
    # --use-llm 开启但工厂未接线（Stage 3 TODO）→ 降级 fallback，不抛异常
    monkeypatch.setattr(gcs, "_make_llm_fn", lambda: None)
    kb = FakeCampusKB(_three_course_reviews())
    stats = gcs.run(kb, use_llm=True, sleep=lambda s: None, interval=0)
    assert stats["llm_used"] is False
    assert stats["courses"] == 3
    methods = {json.loads(e["metadata_json"])["method"] for e in kb.upserted}
    assert methods == {"fallback"}


def test_run_with_injected_llm_fn():
    kb = FakeCampusKB([make_review("这门课作业量很大，每周都有编程作业。",
                                   rating=4, sqid="219")])
    stats = gcs.run(kb, llm_fn=_good_llm, sleep=lambda s: None, interval=0)
    assert stats["llm_used"] is True
    assert json.loads(kb.upserted[0]["metadata_json"])["method"] == "llm"


def test_run_single_course_failure_continues():
    kb = FakeCampusKB(_three_course_reviews())
    original = kb.upsert_entries
    state = {"n": 0}

    def flaky_upsert(entries, db_path=None):
        state["n"] += 1
        if state["n"] == 2:
            raise RuntimeError("db 写入失败")
        return original(entries, db_path=db_path)

    kb.upsert_entries = flaky_upsert
    stats = gcs.run(kb, sleep=lambda s: None, interval=0)
    assert stats["failed"] == 1
    assert stats["courses"] == 2
    assert stats["upserted"] == 2


def test_run_search_failure_returns_zero_stats():
    class BrokenKB(FakeCampusKB):
        def search_kb(self, query, source=None, limit=10, db_path=None):
            raise RuntimeError("db 读取失败")

    stats = gcs.run(BrokenKB([]), sleep=lambda s: None, interval=0)
    assert stats == {"courses": 0, "reviews": 0, "upserted": 0,
                     "llm_used": False, "failed": 0}


def test_main_cli_end_to_end(monkeypatch, capsys):
    kb = FakeCampusKB(_three_course_reviews())
    monkeypatch.setattr(gcs, "_load_campus_kb", lambda: kb)
    rc = gcs.main(["--limit", "2", "--interval", "0"], sleep=lambda s: None)
    assert rc == 0
    assert kb.init_called_with is None
    out = capsys.readouterr().out
    assert "处理课程 2 门" in out and "入库总结 2 条" in out
    assert "fallback" in out


def test_main_cli_kb_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(gcs, "_load_campus_kb", lambda: None)
    rc = gcs.main([], sleep=lambda s: None)
    assert rc == 1
    assert "不可用" in capsys.readouterr().out
