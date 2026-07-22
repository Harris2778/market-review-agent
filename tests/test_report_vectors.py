"""agent/report_vectors.py 研报全文分块 + 向量化 + 检索层（研报库 v2 向量侧）测试。

覆盖范围：
1. 建表：init_vector_tables 幂等；三表（report_chunks/report_embeddings/vector_meta）
   契约 3 结构齐全；路径解析复用 report_library._db_path（env 惰性生效）。
2. 分块：≤500 字/块、重叠 50 正确性；空节/非 dict 项跳过；chunk_idx 跨节连续；
   短文本单块；非法入参不抛出。
3. FakeEmbedder：同文本恒同向量（跨实例）、L2 归一、name='fake' dim=32。
4. build_index：端到端（tmp_path 隔离库，手工插 reports+report_fulltext 行）、
   幂等跳过、force 重建不残留、embedder 构造失败/无全文表降级为带 note 零统计。
5. search_vectors：精确命中排首位（score≈1.0）、stock_code 带前缀过滤、
   industry/days 过滤、维度不符降级、未建索引降级、top_k 夹取、hits 字段契约。

规则（与项目其他测试一致）：全 mock 零网络，全程 FakeEmbedder；
REPORTS_DB_PATH 一律 monkeypatch 到 tmp_path，绝不写真实 data/。
"""

import json
import sqlite3
from datetime import date, timedelta

import numpy as np
import pytest

import agent.report_library as rl
import agent.report_vectors as rv


# ── 工具 ──


def _iso(days_ago: int) -> str:
    """N 天前的 YYYY-MM-DD。"""
    return (date.today() - timedelta(days=days_ago)).isoformat()


def make_record(**kw) -> dict:
    """构造一条合法研报元数据记录（对齐 test_report_library 范式）。"""
    rec = {
        "info_code": "IC-DEFAULT",
        "title": "贵州茅台：稳健增长",
        "org": "中金公司",
        "author": "张三",
        "publish_date": _iso(1),
        "stock_code": "600519",
        "stock_name": "贵州茅台",
        "industry": "食品饮料",
        "rating": "买入",
        "rating_change": "维持",
        "eps_this_year": 68.5,
        "eps_next_year": 75.0,
        "target_price_high": 2100.0,
        "target_price_low": 1800.0,
        "encode_url": "enc123",
        "source": "eastmoney",
    }
    rec.update(kw)
    return rec


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """把研报库隔离到 tmp_path，绝不触碰真实 data/。返回库路径字符串。"""
    p = tmp_path / "reports.db"
    monkeypatch.setenv(rl.REPORTS_DB_PATH_ENV, str(p))
    return str(p)


def _seed_fulltext(db_path: str, info_code: str, fulltext: str,
                   sections=None, source: str = "eastmoney") -> None:
    """手工建 report_fulltext 表（契约 1，Worker D 侧）并插一行全文记录。"""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS report_fulltext ("
            "info_code TEXT PRIMARY KEY, source TEXT, fulltext TEXT, "
            "sections_json TEXT, fetched_at TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO report_fulltext"
            "(info_code, source, fulltext, sections_json, fetched_at) "
            "VALUES(?, ?, ?, ?, ?)",
            (
                info_code,
                source,
                fulltext,
                json.dumps(sections, ensure_ascii=False) if sections else None,
                "2026-07-22T00:00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _seed_report(db_path: str, info_code: str, fulltext: str,
                 sections=None, **report_kw) -> None:
    """插一条 reports 元数据 + 一条 report_fulltext 全文（同 info_code）。"""
    rl.upsert_reports([make_record(info_code=info_code, **report_kw)], db_path=db_path)
    _seed_fulltext(db_path, info_code, fulltext, sections=sections)


def _table_names(db_path: str) -> set:
    conn = sqlite3.connect(db_path)
    try:
        return {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        conn.close()


def _count(db_path: str, table: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


# ── 1. 建表 ──


def test_init_vector_tables_idempotent(db):
    """两次调用幂等；契约 3 三表齐全；返回 env 解析路径。"""
    p1 = rv.init_vector_tables()
    p2 = rv.init_vector_tables()
    assert p1 == p2 == db
    names = _table_names(db)
    assert {"report_chunks", "report_embeddings", "vector_meta"} <= names


def test_init_vector_tables_explicit_path(tmp_path):
    """显式 db_path 优先于 env；父目录自动创建。"""
    p = str(tmp_path / "sub" / "v.db")
    assert rv.init_vector_tables(p) == p
    assert "report_chunks" in _table_names(p)


# ── 2. 分块 ──


def test_chunk_boundaries_and_overlap():
    """超长节切块：每块 ≤500 字；相邻块重叠 50 字且窗口步进 450。"""
    text = "".join(str(i % 10) for i in range(1200))
    chunks = rv.chunk_report([{"name": "正文", "text": text}])
    assert len(chunks) == 3
    assert all(len(c["text"]) <= 500 for c in chunks)
    assert chunks[0]["text"] == text[:500]
    assert chunks[1]["text"] == text[450:950]
    assert chunks[2]["text"] == text[900:]
    # 重叠正确性：前块尾部 50 字 == 后块头部 50 字
    assert chunks[0]["text"][-50:] == chunks[1]["text"][:50]
    assert chunks[1]["text"][-50:] == chunks[2]["text"][:50]
    assert [c["chunk_idx"] for c in chunks] == [0, 1, 2]
    assert all(c["section"] == "正文" for c in chunks)


def test_chunk_empty_sections_skipped_and_idx_continuous():
    """空节（空白/非字符串）与非 dict 项跳过；chunk_idx 跨节连续。"""
    sections = [
        {"name": "投资要点", "text": "要点" * 600},   # 1200 字 → 3 块
        {"name": "空节", "text": "   "},
        {"name": "坏节", "text": None},
        "非字典项",
        {"name": "风险提示", "text": "风险" * 600},   # 1200 字 → 3 块
    ]
    chunks = rv.chunk_report(sections)
    assert len(chunks) == 6
    assert [c["chunk_idx"] for c in chunks] == list(range(6))
    assert {c["section"] for c in chunks[:3]} == {"投资要点"}
    assert {c["section"] for c in chunks[3:]} == {"风险提示"}


def test_chunk_short_text_single_chunk_and_bad_input():
    """短文本单块；非 list 入参返回空表不抛出；overlap 越界自动夹取。"""
    chunks = rv.chunk_report([{"name": "正文", "text": "短短一句"}])
    assert len(chunks) == 1 and chunks[0]["text"] == "短短一句"
    assert rv.chunk_report(None) == []
    assert rv.chunk_report("不是列表") == []
    # overlap >= max_chars 时夹取保证窗口前进，不死循环
    chunks = rv.chunk_report(
        [{"name": "正文", "text": "x" * 1000}], max_chars=100, overlap=999
    )
    assert len(chunks) > 1 and all(len(c["text"]) <= 100 for c in chunks)


# ── 3. FakeEmbedder ──


def test_fake_embedder_deterministic_and_normalized():
    """同文本恒同向量（跨实例）；L2 归一；不同文本不同向量；契约属性。"""
    e1, e2 = rv.FakeEmbedder(), rv.FakeEmbedder(dim=32)
    assert e1.name == "fake" and e1.dim == 32
    v1 = e1.embed(["贵州茅台稳健增长"])[0]
    v2 = e2.embed(["贵州茅台稳健增长"])[0]
    assert len(v1) == 32
    assert np.allclose(v1, v2, atol=1e-6)
    assert abs(float(np.linalg.norm(np.asarray(v1))) - 1.0) < 1e-4
    v3 = e1.embed(["完全不同的另一篇研报"])[0]
    assert not np.allclose(v1, v3, atol=1e-3)


# ── 4. build_index ──


def test_build_index_end_to_end(db):
    """端到端：手工插 reports+report_fulltext → 分块 embed 写库；meta 记录。"""
    long_text = "业绩增长" * 400  # 1600 字 → 4 块
    _seed_report(db, "IC-1", long_text)
    _seed_report(db, "IC-2", "短全文一句",
                 title="半导体行业点评", stock_code="688981",
                 stock_name="中芯国际", industry="电子", rating="增持")

    stats = rv.build_index(db_path=db, embedder=rv.FakeEmbedder())
    assert stats == {"indexed_reports": 2, "indexed_chunks": 5, "skipped": 0}

    assert _count(db, "report_chunks") == 5
    assert _count(db, "report_embeddings") == 5
    # chunk_id 格式 f"{info_code}#{chunk_idx}"；embedding 为 float32 32 维 blob
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM report_chunks WHERE info_code = 'IC-1' ORDER BY chunk_id"
        ).fetchone()
        assert row["chunk_id"].startswith("IC-1#")
        assert row["section"] == "正文"
        blob = conn.execute(
            "SELECT vector FROM report_embeddings WHERE chunk_id = ?",
            (row["chunk_id"],),
        ).fetchone()["vector"]
        assert len(blob) == 32 * 4  # float32 × dim
        meta = {
            r["k"]: r["v"]
            for r in conn.execute("SELECT k, v FROM vector_meta")
        }
        assert meta["embedder_name"] == "fake"
        assert meta["embedder_dim"] == "32"
    finally:
        conn.close()


def test_build_index_sections_json_multi_section(db):
    """sections_json 多节结构按节切块并保留节名；坏 JSON 退化为单节正文。"""
    sections = [
        {"name": "投资要点", "text": "要点内容" * 100},
        {"name": "风险提示", "text": "风险内容"},
    ]
    _seed_report(db, "IC-S1", "全文忽略", sections=sections)
    _seed_report(db, "IC-S2", "坏JSON的正文全文", sections=None)
    # 手工塞一条坏 sections_json
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "UPDATE report_fulltext SET sections_json = '{不是合法JSON' "
            "WHERE info_code = 'IC-S2'"
        )
        conn.commit()
    finally:
        conn.close()

    stats = rv.build_index(db_path=db, embedder=rv.FakeEmbedder())
    assert stats["indexed_reports"] == 2

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        secs = {
            r["section"]
            for r in conn.execute(
                "SELECT section FROM report_chunks WHERE info_code = 'IC-S1'"
            )
        }
        assert secs == {"投资要点", "风险提示"}
        row2 = conn.execute(
            "SELECT section, text FROM report_chunks WHERE info_code = 'IC-S2'"
        ).fetchone()
        assert row2["section"] == "正文"
        assert row2["text"] == "坏JSON的正文全文"
    finally:
        conn.close()


def test_build_index_skips_already_indexed(db):
    """幂等：二次运行默认跳过已索引报告，块数不膨胀。"""
    _seed_report(db, "IC-1", "内容" * 300)
    s1 = rv.build_index(db_path=db, embedder=rv.FakeEmbedder())
    assert s1["indexed_reports"] == 1
    n1 = _count(db, "report_chunks")

    s2 = rv.build_index(db_path=db, embedder=rv.FakeEmbedder())
    assert s2 == {"indexed_reports": 0, "indexed_chunks": 0, "skipped": 1}
    assert _count(db, "report_chunks") == n1


def test_build_index_force_rebuild(db):
    """force=True 重建：重新索引且旧块不残留（行数稳定）。"""
    _seed_report(db, "IC-1", "旧内容" * 400)  # 1600 字 → 4 块
    s1 = rv.build_index(db_path=db, embedder=rv.FakeEmbedder())
    n1 = s1["indexed_chunks"]

    # 全文变短后 force 重建：旧的长块必须被清掉
    _seed_fulltext(db, "IC-1", "新内容很短")
    s2 = rv.build_index(db_path=db, embedder=rv.FakeEmbedder(), force=True)
    assert s2["indexed_reports"] == 1
    assert s2["skipped"] == 0
    assert _count(db, "report_chunks") == 1
    assert _count(db, "report_embeddings") == 1
    assert n1 > 1  # 首次确实是多块


def test_build_index_embedder_failure_returns_note(db, monkeypatch):
    """embedder=None 且构造失败：返回带 note 的零统计结构，绝不抛出。"""
    _seed_report(db, "IC-1", "内容")
    monkeypatch.setattr(rv, "_default_embedder", lambda: None)
    stats = rv.build_index(db_path=db, embedder=None)
    assert stats["indexed_reports"] == 0
    assert stats["indexed_chunks"] == 0
    assert "note" in stats
    assert _count(db, "report_chunks") if "report_chunks" in _table_names(db) else True


def test_build_index_no_fulltext_table_degrades(db):
    """全文表不存在（未跑全文采集）：带 note 零统计，绝不抛出。"""
    rl.init_db(db)  # 只有 reports 表
    stats = rv.build_index(db_path=db, embedder=rv.FakeEmbedder())
    assert stats["indexed_reports"] == 0
    assert "note" in stats


# ── 5. search_vectors ──


def _seed_search_corpus(db_path: str) -> None:
    """三篇可区分语料（各单块），供检索排序/过滤用。"""
    _seed_report(db_path, "IC-A", "茅台三季度业绩超预期 批价企稳回升")
    _seed_report(db_path, "IC-B", "半导体设备国产替代加速 订单饱满",
                 title="半导体设备行业点评", stock_code="688981",
                 stock_name="中芯国际", industry="电子", rating="增持")
    _seed_report(db_path, "IC-C", "白酒渠道库存去化 动销改善",
                 title="白酒行业跟踪", stock_code="000858",
                 stock_name="五粮液", industry="食品饮料", rating="买入")
    rv.build_index(db_path=db_path, embedder=rv.FakeEmbedder())


def test_search_exact_match_ranks_first(db):
    """相关度排序：query 与某块文本一致时该块排首位且 score≈1.0，其余更低。"""
    _seed_search_corpus(db)
    target = "半导体设备国产替代加速 订单饱满"
    res = rv.search_vectors(target, db_path=db, embedder=rv.FakeEmbedder())
    assert res["total_chunks"] == 3
    assert len(res["hits"]) == 3
    assert res["hits"][0]["info_code"] == "IC-B"
    assert res["hits"][0]["snippet"] == target
    assert res["hits"][0]["score"] == pytest.approx(1.0, abs=1e-4)
    assert all(h["score"] < res["hits"][0]["score"] for h in res["hits"][1:])


def test_search_stock_code_prefix_filter(db):
    """stock_code 带 sh/sz 前缀精确过滤（归一化后匹配）。"""
    _seed_search_corpus(db)
    res = rv.search_vectors(
        "研报", stock_code="SZ000858", db_path=db, embedder=rv.FakeEmbedder()
    )
    assert res["total_chunks"] == 3  # 索引总块数不受过滤影响
    assert len(res["hits"]) == 1
    assert res["hits"][0]["info_code"] == "IC-C"


def test_search_industry_filter(db):
    """industry LIKE %..% 过滤；无命中自动回退为不限行业并带 note。"""
    _seed_search_corpus(db)
    res = rv.search_vectors(
        "研报", industry="电子", db_path=db, embedder=rv.FakeEmbedder()
    )
    assert {h["info_code"] for h in res["hits"]} == {"IC-B"}
    assert "note" not in res  # 正常命中不带回退说明


def test_search_industry_fallback_on_zero_hit(db):
    """行业过滤零命中：回退为不限行业检索，note 说明回退（LIKE 转义仍生效）。"""
    _seed_search_corpus(db)
    # 库内不存在的行业名 → 直接过滤为空 → 回退后命中全部
    res = rv.search_vectors(
        "研报", industry="白酒Ⅱ", db_path=db, embedder=rv.FakeEmbedder()
    )
    assert len(res["hits"]) == 3
    assert "行业过滤「白酒Ⅱ」无命中" in res["note"]
    # 注入 % 通配符：按字面匹配直接命中为零（转义生效），随后回退
    res2 = rv.search_vectors(
        "研报", industry="%", db_path=db, embedder=rv.FakeEmbedder()
    )
    assert len(res2["hits"]) == 3
    assert "无命中" in res2["note"]  # note 存在即证明直接过滤是零命中（转义成功）


def test_search_days_filter(db):
    """days 窗口过滤：超窗报告被排除。"""
    _seed_report(db, "IC-NEW", "近期研报内容", publish_date=_iso(5))
    _seed_report(db, "IC-OLD", "陈旧研报内容", publish_date=_iso(200))
    rv.build_index(db_path=db, embedder=rv.FakeEmbedder())

    res = rv.search_vectors("研报", days=30, db_path=db, embedder=rv.FakeEmbedder())
    assert {h["info_code"] for h in res["hits"]} == {"IC-NEW"}
    res2 = rv.search_vectors("研报", days=365, db_path=db, embedder=rv.FakeEmbedder())
    assert {h["info_code"] for h in res2["hits"]} == {"IC-NEW", "IC-OLD"}


def test_search_dimension_mismatch_degrades(db):
    """维度不符：索引 32 维 vs 查询 16 维 embedder → 降级带 note，绝不抛出。"""
    _seed_search_corpus(db)
    res = rv.search_vectors("研报", db_path=db, embedder=rv.FakeEmbedder(dim=16))
    assert res["total_chunks"] == 0
    assert res["hits"] == []
    assert "维度" in res["note"]


def test_search_not_indexed_degrades(db, tmp_path, monkeypatch):
    """未建索引降级：库文件不存在 / 有 reports 无向量表，均带 note 不抛出。"""
    # 库文件不存在
    missing = str(tmp_path / "nonexistent.db")
    res = rv.search_vectors("研报", db_path=missing, embedder=rv.FakeEmbedder())
    assert res == {"total_chunks": 0, "hits": [], "note": res["note"]}
    assert "note" in res

    # 有 reports 表但无向量表/无 meta
    rl.init_db(db)
    res2 = rv.search_vectors("研报", db_path=db, embedder=rv.FakeEmbedder())
    assert res2["total_chunks"] == 0
    assert res2["hits"] == []
    assert "note" in res2


def test_search_embedder_failure_degrades(db, monkeypatch):
    """embedder=None 且构造失败：降级带 note，绝不抛出。"""
    _seed_search_corpus(db)
    monkeypatch.setattr(rv, "_default_embedder", lambda: None)
    res = rv.search_vectors("研报", db_path=db, embedder=None)
    assert res["total_chunks"] == 0
    assert res["hits"] == []
    assert "note" in res


def test_search_top_k_clamp(db):
    """top_k 夹取：显式 2 只回 2 条；0 夹到最小 1；非数值回退默认 5；超大夹到 50。"""
    _seed_search_corpus(db)
    res = rv.search_vectors("研报", top_k=2, db_path=db, embedder=rv.FakeEmbedder())
    assert len(res["hits"]) == 2
    res0 = rv.search_vectors("研报", top_k=0, db_path=db, embedder=rv.FakeEmbedder())
    assert len(res0["hits"]) == 1  # 夹到最小值 1（对齐 report_library._clamp_limit）
    res_bad = rv.search_vectors(
        "研报", top_k="不是数字", db_path=db, embedder=rv.FakeEmbedder()
    )
    assert len(res_bad["hits"]) == 3  # 回退默认 5，实际只有 3 块
    res_big = rv.search_vectors(
        "研报", top_k=9999, db_path=db, embedder=rv.FakeEmbedder()
    )
    assert len(res_big["hits"]) == 3  # 夹到 50，实际只有 3 块


def test_search_hit_fields_contract(db):
    """hits 字段契约：八字段齐全、snippet ≤500 字原样、score round 4。"""
    long_text = "业绩" * 400  # 800 字 → 2 块，块长 500/350
    _seed_report(db, "IC-F", long_text)
    rv.build_index(db_path=db, embedder=rv.FakeEmbedder())

    res = rv.search_vectors(long_text[:500], db_path=db, embedder=rv.FakeEmbedder())
    assert res["hits"], "应有命中"
    hit = res["hits"][0]
    assert set(hit.keys()) == {
        "info_code", "title", "org", "date", "rating",
        "section", "snippet", "score",
    }
    assert hit["info_code"] == "IC-F"
    assert hit["title"] == "贵州茅台：稳健增长"
    assert hit["org"] == "中金公司"
    assert hit["date"] == _iso(1)
    assert hit["rating"] == "买入"
    assert hit["section"] == "正文"
    assert len(hit["snippet"]) <= 500
    assert hit["score"] == round(hit["score"], 4)
