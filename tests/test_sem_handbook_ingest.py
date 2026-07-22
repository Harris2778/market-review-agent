"""scripts/sem_handbook_ingest.py 测试（手册解析工程师_A）。

红线：全 mock、零网络、零真实外部文件依赖——
- PDF fixture 用 pymupdf 现场生成带中文文字的小型 PDF（tmp_path）；
- 知识库存储层一律 monkeypatch _load_campus_kb 注入假实现，绝不 import
  agent.campus_kb 实体模块、绝不触碰真实 db。
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import sem_handbook_ingest as ing  # noqa: E402

fitz = pytest.importorskip("fitz", reason="pymupdf 未安装，跳过手册解析测试")


# ═══════════════════════════════════════════
# fixture：现场生成小型 PDF
# ═══════════════════════════════════════════

def make_pdf(path: Path, pages) -> Path:
    """用 pymupdf 现场生成带中文文字的小型 PDF。

    pages: [[每页行文本, ...], ...]，内置 CJK 字体 china-s。"""
    doc = fitz.open()
    for lines in pages:
        page = doc.new_page()
        page.insert_text((72, 72), "\n".join(lines), fontname="china-s",
                         fontsize=11)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def sample_pdf(tmp_path):
    """多页、含标题行（第一章/一、/1.1）与年份文件名的样例手册。"""
    return make_pdf(tmp_path / "2026年版测试保研手册.pdf", [
        ["前言部分", "这本手册介绍保研政策。" * 10],          # 第1页：无标题前言
        ["第一章 总则", "一、基本条件", "1.1 学分要求",
         "学生应修满规定学分。" * 20],                        # 第2页：三个标题
        ["1.2 成绩要求", "平均学分绩点不低于标准线。" * 20],  # 第3页：跨页续章
    ])


class FakeCampusKB:
    """假知识库存储层：记录 init_db / upsert_entries 调用，模拟主键覆盖。"""

    def __init__(self):
        self.init_calls = []
        self.upsert_calls = []
        self.store = {}

    def init_db(self, db_path=None):
        self.init_calls.append(db_path)

    def upsert_entries(self, entries, db_path=None):
        self.upsert_calls.append((list(entries), db_path))
        for e in entries:
            self.store[(e["source"], e["source_id"])] = e
        return len(entries)

    def install(self, monkeypatch):
        monkeypatch.setattr(ing, "_load_campus_kb", lambda: (self.init_db,
                                                             self.upsert_entries))
        return self


# ═══════════════════════════════════════════
# 纯函数：年份提取 / 标题判定
# ═══════════════════════════════════════════

def test_extract_publish_hint_variants():
    assert ing.extract_publish_hint("【清华经管科协】2026年版保研手册.pdf") == "2026"
    assert ing.extract_publish_hint("经管学院留学手册-2023年.pdf") == "2023"
    assert ing.extract_publish_hint("国际学生在华实习加注办理手册 (2026.5).pdf") == "2026"
    assert ing.extract_publish_hint("2025经管学院院级交换手册.pdf") == "2025"
    assert ing.extract_publish_hint("经管学院本科生学术手册_第十一版.pdf") == ""
    assert ing.extract_publish_hint("") == ""


def test_is_heading_patterns():
    assert ing._is_heading("第一章 总则")
    assert ing._is_heading("第12节 申请流程")
    assert ing._is_heading("一、基本条件")
    assert ing._is_heading("1.1 学分要求")
    assert ing._is_heading("2.3.1 细则")
    assert ing._is_heading("（一）组织领导")
    assert ing._is_heading("附录 常见问题")
    # 非标题：正文长行 / 无编号 / 年份开头不带点号
    assert not ing._is_heading("学生应当按时完成培养方案规定的全部课程学习任务。")
    assert not ing._is_heading("2026年版保研政策说明")
    assert not ing._is_heading("这是一条没有编号但长度明显超过四十字的普通正文行"
                               "内容需要被判定为非标题行")
    assert not ing._is_heading("")


# ═══════════════════════════════════════════
# 纯函数：分块流水线
# ═══════════════════════════════════════════

def _pages(*page_lines):
    return [(i + 1, "\n".join(lines)) for i, lines in enumerate(page_lines)]


def test_split_into_segments_by_headings():
    pages = _pages(["前言", "第一章 总则", "正文一"], ["1.1 细则", "正文二"])
    segs = ing.split_into_segments(pages)
    assert len(segs) == 3
    assert segs[0]["title"] == ""            # 首个标题前的内容无标题
    assert segs[1]["title"] == "第一章 总则"
    assert segs[2]["title"] == "1.1 细则"
    assert segs[2]["page_start"] == segs[2]["page_end"] == 2
    # 标题行保留在 lines 内，块正文自带章节上下文
    assert segs[1]["lines"][0] == (1, "第一章 总则")


def test_chunk_cross_page_merge():
    """跨页段落合并为一块，page_start/page_end 跨页。"""
    body1 = "甲" * 200
    body2 = "乙" * 200
    pages = _pages(["第一章 总则", body1], [body2])
    chunks = ing.chunk_handbook(pages)
    assert len(chunks) == 1
    assert chunks[0]["page_start"] == 1
    assert chunks[0]["page_end"] == 2
    assert body1 in chunks[0]["content"] and body2 in chunks[0]["content"]


def test_merge_small_segments_up_to_min():
    """多个小章节段（均 <min_chars）贪婪合并，不超过 max_chars。"""
    pages = _pages(["一、甲", "短内容甲" * 10, "二、乙", "短内容乙" * 10,
                    "三、丙", "短内容丙" * 10])
    chunks = ing.chunk_handbook(pages, min_chars=300, max_chars=1500)
    assert len(chunks) == 1
    assert chunks[0]["title"] == "一、甲"     # 取首个非空标题
    assert "三、丙" in chunks[0]["content"]


def test_split_oversize_chunk():
    """超长章节按行切分为 ≤max_chars 多片，标题带（i/n）序号。"""
    line = "长" * 400
    pages = _pages(["第一章 超长章"] + [line] * 10)  # 约 4000+ 字
    chunks = ing.chunk_handbook(pages, min_chars=300, max_chars=1500)
    assert len(chunks) >= 3
    for c in chunks:
        assert len(c["content"]) <= 1500
    assert chunks[0]["title"] == "第一章 超长章（1/%d）" % len(chunks)
    assert chunks[-1]["title"].endswith("（%d/%d）" % (len(chunks), len(chunks)))


def test_split_single_oversize_line_hard_slice():
    """单行超 max_chars 硬切片，不产出超限块。"""
    pages = _pages(["第一章", "字" * 3200])
    chunks = ing.chunk_handbook(pages, min_chars=300, max_chars=1500)
    assert len(chunks) >= 2
    assert all(len(c["content"]) <= 1500 for c in chunks)


# ═══════════════════════════════════════════
# 条目构造（全局契约 8 字段）
# ═══════════════════════════════════════════

def test_build_entries_contract_fields():
    chunks = [
        {"title": "第一章 总则", "content": "正文" * 200,
         "page_start": 2, "page_end": 3},
        {"title": "", "content": "前言" * 200, "page_start": 1, "page_end": 1},
    ]
    entries = ing.build_entries(Path("/x/2026年版保研手册.pdf"), chunks,
                                now_iso="2026-07-23T00:00:00+00:00")
    assert len(entries) == 2
    e = entries[0]
    assert set(e.keys()) == {"source", "source_id", "title", "content", "url",
                             "metadata_json", "updated_at"}
    assert e["source"] == "sem_handbook"
    assert e["source_id"] == "sem_handbook:2026年版保研手册:001"
    assert entries[1]["source_id"] == "sem_handbook:2026年版保研手册:002"
    assert e["title"] == "2026年版保研手册 - 第一章 总则"
    # 无标题块回退页码标识
    assert entries[1]["title"] == "2026年版保研手册 - 第1页"
    assert e["url"] == "local://清华经管/2026年版保研手册.pdf"
    assert e["updated_at"] == "2026-07-23T00:00:00+00:00"
    meta = json.loads(e["metadata_json"])
    assert meta == {"handbook_name": "2026年版保研手册", "page_start": 2,
                    "page_end": 3, "publish_hint": "2026"}


def test_build_entries_skips_empty_content():
    chunks = [{"title": "空块", "content": "  ", "page_start": 1, "page_end": 1},
              {"title": "实块", "content": "有内容", "page_start": 1,
               "page_end": 1}]
    entries = ing.build_entries(Path("/x/手册.pdf"), chunks)
    assert len(entries) == 1
    assert entries[0]["source_id"] == "sem_handbook:手册:001"  # 序号连续不跳空


# ═══════════════════════════════════════════
# 端到端：现场 PDF → 解析 → 条目 → 入库
# ═══════════════════════════════════════════

def test_extract_pages_from_generated_pdf(sample_pdf):
    pages = ing.extract_pages(sample_pdf)
    assert [p for p, _ in pages] == [1, 2, 3]
    assert "第一章 总则" in pages[1][1]
    assert "1.2 成绩要求" in pages[2][1]


def test_parse_pdf_end_to_end(sample_pdf):
    chunks = ing.parse_pdf(sample_pdf)
    assert chunks
    assert all(c["content"].strip() for c in chunks)
    titles = " ".join(c["title"] for c in chunks)
    assert "第一章 总则" in titles
    # 页码均在手册页数范围内且 start<=end
    assert all(1 <= c["page_start"] <= c["page_end"] <= 3 for c in chunks)


def test_iter_pdf_files_sorted(tmp_path):
    make_pdf(tmp_path / "乙手册.pdf", [["乙"]])
    make_pdf(tmp_path / "甲手册.pdf", [["甲"]])
    (tmp_path / "说明.txt").write_text("不是 PDF", encoding="utf-8")
    pdfs = ing.iter_pdf_files(tmp_path)
    assert [p.name for p in pdfs] == sorted(["乙手册.pdf", "甲手册.pdf"])


def test_main_dry_run_no_db_touch(sample_pdf, monkeypatch, capsys):
    """--dry-run 只打印清单，绝不加载知识库、绝不入库。"""
    def _forbidden():
        raise AssertionError("dry-run 不得加载 campus_kb")

    monkeypatch.setattr(ing, "_load_campus_kb", _forbidden)
    rc = ing.main(["--pdf-dir", str(sample_pdf.parent), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "【dry-run】" in out
    assert "2026年版测试保研手册.pdf" in out
    assert "合计将入库" in out
    assert "sem_handbook" not in out  # dry-run 不打印入库统计


def test_main_upsert_via_fake_kb(sample_pdf, monkeypatch):
    fake = FakeCampusKB().install(monkeypatch)
    rc = ing.main(["--pdf-dir", str(sample_pdf.parent), "--db", "/tmp/fake.db"])
    assert rc == 0
    assert fake.init_calls == ["/tmp/fake.db"]
    assert len(fake.upsert_calls) == 1
    entries, db_path = fake.upsert_calls[0]
    assert db_path == "/tmp/fake.db"
    assert entries
    assert all(e["source"] == "sem_handbook" for e in entries)
    assert all(e["source_id"].startswith("sem_handbook:2026年版测试保研手册:")
               for e in entries)
    assert [e["source_id"] for e in entries] == sorted(e["source_id"]
                                                       for e in entries)


def test_main_upsert_idempotent(sample_pdf, monkeypatch):
    """重复运行：source_id 稳定，(source, source_id) 主键覆盖，条数不膨胀。"""
    fake = FakeCampusKB().install(monkeypatch)
    assert ing.main(["--pdf-dir", str(sample_pdf.parent)]) == 0
    first_ids = sorted(fake.store)
    n_first = len(first_ids)
    assert ing.main(["--pdf-dir", str(sample_pdf.parent)]) == 0
    assert sorted(fake.store) == first_ids
    assert len(fake.store) == n_first
    assert len(fake.upsert_calls) == 2


def test_main_corrupt_pdf_skipped(tmp_path, sample_pdf, monkeypatch, capsys):
    """坏 PDF 记 warning 跳过，其余手册正常入库，整体返回 0。"""
    (tmp_path / "0坏手册.pdf").write_text("这不是真的 PDF 内容", encoding="utf-8")
    fake = FakeCampusKB().install(monkeypatch)
    rc = ing.main(["--pdf-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert fake.upsert_calls  # 好手册照常入库
    assert "1/2 份 PDF" in out


def test_main_pdf_dir_not_found(capsys):
    rc = ing.main(["--pdf-dir", "/tmp/绝不存在的目录_清小搭测试"])
    assert rc == 2
    assert "--pdf-dir" in capsys.readouterr().out


def test_main_empty_dir(tmp_path, capsys):
    rc = ing.main(["--pdf-dir", str(tmp_path)])
    assert rc == 1
    assert "未找到 PDF" in capsys.readouterr().out


def test_main_missing_pymupdf_clean_exit(sample_pdf, monkeypatch, capsys):
    """缺 pymupdf 依赖：打印中文指引干净退出，不抛栈。"""
    def _boom():
        raise ing.DependencyError("缺少 PDF 解析依赖 pymupdf，无法解析手册。")

    monkeypatch.setattr(ing, "_require_fitz", _boom)
    rc = ing.main(["--pdf-dir", str(sample_pdf.parent)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "pymupdf" in out


def test_main_campus_kb_unavailable(sample_pdf, monkeypatch, capsys):
    """知识库模块缺失：打印中文提示返回 1，不抛栈。"""
    def _boom():
        raise ImportError("No module named 'agent.campus_kb'")

    monkeypatch.setattr(ing, "_load_campus_kb", _boom)
    rc = ing.main(["--pdf-dir", str(sample_pdf.parent)])
    assert rc == 1
    assert "campus_kb" in capsys.readouterr().out
