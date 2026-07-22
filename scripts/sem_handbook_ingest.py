#!/usr/bin/env python3
"""经管学院 PDF 手册解析入库脚本（校园知识库 · sem_handbook 源）。

职责：扫描指定目录下的经管学院 PDF 手册，按页提取文本，做章节级分块
（优先按标题行/编号模式切分，单块 300~1500 字，跨页合并），每块构造一条
全局契约 8 字段 kb 条目（source='sem_handbook'）并 upsert 入库。

真实手册默认位于 research/清华经管手册/（保研手册、院级交换手册、留学手册、
本科生学术手册、体育课选课手册、国际生实习指南、国际学生在华实习加注手册
共 7 份），但目录一律通过 --pdf-dir 指定（默认该路径），绝不硬编码为唯一来源。

条目规则（全局契约，不得偏离）：
- 条目字典固定 8 字段：source / source_id / title / content / url /
  metadata_json / updated_at（外加检索用正文 content 均为纯文本）；
- source_id = 'sem_handbook:{文件名去扩展名}:{块序号:03d}'；
- title = '{手册名} - {章节标题或页码标识}'；
- metadata_json = JSON 字符串，含 handbook_name / page_start / page_end /
  publish_hint（从文件名提取年份，无则空串）；
- url = 'local://清华经管/{文件名}'。

可测试性纪律：
- pymupdf（import fitz）惰性导入，缺依赖时给出中文安装指引而非抛栈；
- 知识库存储层收口在 _load_campus_kb()（运行期真实
  from agent.campus_kb import init_db, upsert_entries），测试 monkeypatch
  本函数注入假实现，绝不 import 实体模块、绝不触碰真实 db；
- 解析/分块/条目构造全部纯函数化，测试用 pymupdf 现场生成小型 PDF 做 fixture；
- 单文件解析失败记 warning 并继续其他文件，绝不因一份坏 PDF 中断全量入库。

CLI 用法：
    /usr/local/bin/python3 scripts/sem_handbook_ingest.py --dry-run
    /usr/local/bin/python3 scripts/sem_handbook_ingest.py --pdf-dir /path/to/pdfs
    /usr/local/bin/python3 scripts/sem_handbook_ingest.py --db data/campus_kb.db -v
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 保证从项目根可导入 agent 包（脚本可被任意 cwd 调用；仅 CLI 落库路径使用）。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── 全局常量 ──

SOURCE = "sem_handbook"
DEFAULT_PDF_DIR = "/Users/harriszhang/Documents/kimi/workspace/research/清华经管手册"
URL_PREFIX = "local://清华经管/"

MIN_CHUNK_CHARS = 300   # 单块建议下限（低于则尝试与后续块合并）
MAX_CHUNK_CHARS = 1500  # 单块硬上限（超过则按行切分）
HEADING_MAX_LEN = 40    # 标题行长度上限（超过视为正文行）
TITLE_LABEL_MAX_LEN = 30  # 条目标题中章节标签的截断长度

# 章节标题/编号模式：'第X章/节/篇/编/部'、'一、'、'1.1'（可多级）、'（一）'、'附录'
_HEADING_RES: Tuple[re.Pattern, ...] = (
    re.compile(r"^第[0-9零一二三四五六七八九十百]+[章节篇编部][\s　:：]?\S*"),
    re.compile(r"^[一二三四五六七八九十]+、\S+"),
    re.compile(r"^\d+(?:\.\d+)+[\s　]?\S*"),
    re.compile(r"^（[一二三四五六七八九十\d]+）\S+"),
    re.compile(r"^附录[\s　:：]?\S*"),
)

_YEAR_RE = re.compile(r"(20\d{2})")


class DependencyError(RuntimeError):
    """可选依赖缺失（pymupdf）：message 为中文安装指引，CLI 打印后干净退出。"""


# ═══════════════════════════════════════════
# 依赖与基础工具
# ═══════════════════════════════════════════

def _require_fitz():
    """惰性导入 pymupdf（import fitz）。缺依赖抛 DependencyError（中文指引），
    绝不让原始 ImportError 栈泄漏给 CLI 用户。"""
    try:
        import fitz
        return fitz
    except ImportError as e:  # pragma: no cover - 本机已装依赖，仅 CI/裸机触发
        raise DependencyError(
            "缺少 PDF 解析依赖 pymupdf，无法解析手册。\n"
            "请先安装：/usr/local/bin/python3 -m pip install pymupdf\n"
            "（或安装项目依赖：/usr/local/bin/python3 -m pip install "
            "-r requirements-rag.txt）") from e


def _now_iso() -> str:
    """当前 UTC 时间 ISO 时间戳（条目 updated_at 字段）。"""
    return datetime.now(timezone.utc).isoformat()


def extract_pages(pdf_path) -> List[Tuple[int, str]]:
    """按页提取 PDF 文本 → [(页码_1起, 文本)]。坏 PDF/加密文件异常向上抛，
    由调用方记 warning 跳过。"""
    fitz = _require_fitz()
    pages: List[Tuple[int, str]] = []
    with fitz.open(str(pdf_path)) as doc:
        for i in range(doc.page_count):
            pages.append((i + 1, doc[i].get_text()))
    return pages


def _is_heading(line: str) -> bool:
    """判断一行是否为章节标题行：命中编号模式且长度不超过 HEADING_MAX_LEN。"""
    if not line or len(line) > HEADING_MAX_LEN:
        return False
    return any(p.match(line) for p in _HEADING_RES)


def extract_publish_hint(filename: str) -> str:
    """从文件名提取 4 位年份（20xx）作为 publish_hint；无年份返回空串。"""
    m = _YEAR_RE.search(filename or "")
    return m.group(1) if m else ""


# ═══════════════════════════════════════════
# 分块流水线（纯函数）
# ═══════════════════════════════════════════

def split_into_segments(pages: List[Tuple[int, str]]) -> List[dict]:
    """页文本流 → 章节段列表（纯函数，天然跨页合并）。

    每段 dict：title（标题行原文，首个标题前的内容为空串）/ lines
    ([(页码, 行文本)]) / page_start / page_end。标题行保留在 lines 内，
    使块正文自带章节上下文。"""
    segments: List[dict] = []
    current = {"title": "", "lines": [], "page_start": None, "page_end": None}

    def _close():
        if current["lines"]:
            segments.append(dict(current, lines=list(current["lines"])))

    for page_no, text in pages:
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            if _is_heading(line):
                _close()
                current = {"title": line, "lines": [(page_no, line)],
                           "page_start": page_no, "page_end": page_no}
            else:
                current["lines"].append((page_no, line))
                if current["page_start"] is None:
                    current["page_start"] = page_no
                current["page_end"] = page_no
    _close()
    return segments


def _chunk_len(chunk: dict) -> int:
    """块正文字符数（行间换行符计入）。"""
    lines = chunk["lines"]
    if not lines:
        return 0
    return sum(len(t) for _, t in lines) + len(lines) - 1


def split_chunk(chunk: dict, max_chars: int = MAX_CHUNK_CHARS) -> List[dict]:
    """超大块按行切分为 ≤max_chars 的若干片（纯函数）。

    单行超长先按 max_chars 硬切片（保持页码归属）；多片标题追加
    '（i/n）' 序号；每片记录自身 page_start/page_end。"""
    lines: List[Tuple[int, str]] = []
    for page, text in chunk["lines"]:
        while len(text) > max_chars:
            lines.append((page, text[:max_chars]))
            text = text[max_chars:]
        lines.append((page, text))
    pieces: List[List[Tuple[int, str]]] = []
    cur: List[Tuple[int, str]] = []
    cur_len = 0
    for item in lines:
        add = len(item[1]) + (1 if cur else 0)
        if cur and cur_len + add > max_chars:
            pieces.append(cur)
            cur, cur_len = [], 0
            add = len(item[1])
        cur.append(item)
        cur_len += add
    if cur:
        pieces.append(cur)
    total = len(pieces)
    result = []
    for i, piece in enumerate(pieces):
        title = chunk["title"]
        if total > 1:
            title = f"{title}（{i + 1}/{total}）" if title else f"（{i + 1}/{total}）"
        result.append({
            "title": title,
            "content": "\n".join(t for _, t in piece),
            "page_start": piece[0][0],
            "page_end": piece[-1][0],
        })
    return result


def merge_segments(segments: List[dict], min_chars: int = MIN_CHUNK_CHARS,
                   max_chars: int = MAX_CHUNK_CHARS) -> List[dict]:
    """章节段 → 最终块列表（纯函数）。

    贪婪合并：当前块不足 min_chars 且并入下一段后不超过 max_chars 时合并
    （标题取首个非空标题）；随后对仍超 max_chars 的块按行切分。
    产出块 dict：title / content / page_start / page_end。"""
    chunks: List[dict] = []
    acc: Optional[dict] = None
    for seg in segments:
        if acc is None:
            acc = dict(seg, lines=list(seg["lines"]))
            continue
        if _chunk_len(acc) < min_chars and _chunk_len(acc) + 1 + _chunk_len(seg) <= max_chars:
            if not acc["title"]:
                acc["title"] = seg["title"]
            acc["lines"].extend(seg["lines"])
            acc["page_end"] = seg["page_end"]
        else:
            chunks.append(acc)
            acc = dict(seg, lines=list(seg["lines"]))
    if acc is not None:
        chunks.append(acc)
    final: List[dict] = []
    for c in chunks:
        final.extend(split_chunk(c, max_chars))
    return final


def chunk_handbook(pages: List[Tuple[int, str]], min_chars: int = MIN_CHUNK_CHARS,
                   max_chars: int = MAX_CHUNK_CHARS) -> List[dict]:
    """页文本流 → 最终块列表（分节 → 合并 → 切分的完整流水线，纯函数）。"""
    return merge_segments(split_into_segments(pages), min_chars, max_chars)


def parse_pdf(pdf_path, min_chars: int = MIN_CHUNK_CHARS,
              max_chars: int = MAX_CHUNK_CHARS) -> List[dict]:
    """单份 PDF → 最终块列表（提取 + 分块流水线）。"""
    return chunk_handbook(extract_pages(pdf_path), min_chars, max_chars)


# ═══════════════════════════════════════════
# 条目构造（全局契约 8 字段）
# ═══════════════════════════════════════════

def build_entries(pdf_path, chunks: List[dict],
                  now_iso: Optional[str] = None) -> List[dict]:
    """单份手册的分块结果 → kb 条目列表（纯函数，全局契约 8 字段）。

    空正文块跳过；块序号 1 起、03d 零填充；同一次调用内所有条目共享
    同一 updated_at 时间戳（可注入 now_iso 便于测试断言）。"""
    path = Path(pdf_path)
    name, fname = path.stem, path.name
    publish_hint = extract_publish_hint(fname)
    ts = now_iso or _now_iso()
    entries: List[dict] = []
    seq = 0
    for chunk in chunks:
        content = (chunk.get("content") or "").strip()
        if not content:
            continue
        seq += 1
        heading = (chunk.get("title") or "").strip()[:TITLE_LABEL_MAX_LEN]
        label = heading or f"第{chunk['page_start']}页"
        meta = {
            "handbook_name": name,
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "publish_hint": publish_hint,
        }
        entries.append({
            "source": SOURCE,
            "source_id": f"{SOURCE}:{name}:{seq:03d}",
            "title": f"{name} - {label}",
            "content": content,
            "url": f"{URL_PREFIX}{fname}",
            "metadata_json": json.dumps(meta, ensure_ascii=False),
            "updated_at": ts,
        })
    return entries


def iter_pdf_files(pdf_dir) -> List[Path]:
    """目录下的 PDF 文件清单（按文件名排序，非递归）。"""
    return sorted(Path(pdf_dir).glob("*.pdf"))


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

def _load_campus_kb():
    """惰性导入知识库存储层（仅 CLI 真实落库路径）。

    测试 monkeypatch 本函数注入假实现，绝不 import 实体模块、绝不触碰真实 db。
    返回 (init_db, upsert_entries) 函数对。"""
    from agent.campus_kb import init_db, upsert_entries
    return init_db, upsert_entries


def main(argv=None) -> int:
    """手册入库 CLI：解析 --pdf-dir 下全部 PDF 并 upsert 入库；
    --dry-run 只打印将入库的块数与标题清单（不触碰知识库）。"""
    parser = argparse.ArgumentParser(
        description="经管学院 PDF 手册解析入库（校园知识库 sem_handbook 源）")
    parser.add_argument("--pdf-dir", default=DEFAULT_PDF_DIR,
                        help="PDF 手册目录（默认：%(default)s）")
    parser.add_argument("--db", default=None,
                        help="知识库 SQLite 路径（缺省走 campus_kb 惰性路径解析）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将入库的块数与标题清单，不入库")
    parser.add_argument("--min-chars", type=int, default=MIN_CHUNK_CHARS,
                        help="单块建议下限字符数（默认 %(default)s）")
    parser.add_argument("--max-chars", type=int, default=MAX_CHUNK_CHARS,
                        help="单块硬上限字符数（默认 %(default)s）")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出 DEBUG 日志")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.is_dir():
        print(f"PDF 目录不存在：{pdf_dir}（请通过 --pdf-dir 指定有效目录）")
        return 2
    pdfs = iter_pdf_files(pdf_dir)
    if not pdfs:
        print(f"目录下未找到 PDF 文件：{pdf_dir}")
        return 1

    try:
        _require_fitz()  # 预检依赖，缺 pymupdf 时干净退出而非抛栈
    except DependencyError as e:
        print(e)
        return 1

    min_chars = max(1, args.min_chars)
    max_chars = max(min_chars, args.max_chars)

    per_file: List[Tuple[str, Optional[List[dict]]]] = []
    all_entries: List[dict] = []
    for pdf in pdfs:
        try:
            chunks = parse_pdf(pdf, min_chars, max_chars)
        except DependencyError:
            raise
        except Exception:
            logger.warning("PDF 解析失败，已跳过：%s", pdf, exc_info=True)
            per_file.append((pdf.name, None))
            continue
        entries = build_entries(pdf, chunks)
        per_file.append((pdf.name, entries))
        all_entries.extend(entries)

    if args.dry_run:
        print(f"【dry-run】目录：{pdf_dir}（共 {len(pdfs)} 份 PDF，不入库）")
        for fname, entries in per_file:
            if entries is None:
                print(f"  {fname}：解析失败（详见日志），已跳过")
                continue
            print(f"  {fname}：{len(entries)} 块")
            for e in entries:
                meta = json.loads(e["metadata_json"])
                print(f"    - {e['title']}"
                      f"（{len(e['content'])} 字，"
                      f"p{meta['page_start']}-{meta['page_end']}）")
        print(f"合计将入库：{len(all_entries)} 块。")
        return 0

    try:
        init_db, upsert_entries = _load_campus_kb()
    except Exception:
        logger.warning("知识库模块 agent.campus_kb 导入失败", exc_info=True)
        print("知识库模块 agent.campus_kb 不可用（详见日志），终止。")
        return 1
    try:
        init_db(args.db)
        written = upsert_entries(all_entries, args.db)
    except Exception:
        logger.warning("知识库入库失败", exc_info=True)
        print("知识库入库失败（详见日志），终止。")
        return 1

    ok = sum(1 for _, entries in per_file if entries is not None)
    print(f"解析完成：{ok}/{len(pdfs)} 份 PDF，共 {len(all_entries)} 块；"
          f"入库 {written} 条（source={SOURCE}）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
