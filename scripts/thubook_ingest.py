#!/usr/bin/env python3
"""thubook 清华校内百科全量入库脚本（校园知识库 · source='thubook'）。

职责：解析 thubook（清华校内生活学习百科，VuePress 开源站）的 markdown 源
文件，清洗 VuePress/组件语法后按二级/三级标题分块，产出全局契约 8 字段条目
字典，经 agent/campus_kb API 全量 upsert 入库。

解析规则：
- 剥离 YAML front matter（简单 key: value 解析，不依赖 pyyaml）；
- 剥离/降级 VuePress 语法：::: tip 等容器（保留容器标题文本）、
  Badge/VPCard 等组件标签（提取可读属性文本）、[[...]] 双链（保留可读文本）、
  图片降级为 [图片: alt]、链接降级为 文本（url）、粗体/删除线标记去壳；
- markdown 表格转为可读文本行（有表头时「列名：值；…」，无表头用 / 连接）；
- 按 ##/### 标题分块；无标题的短文件整块一条；单块建议 300~1500 字，
  过短小节与相邻小节合并，超长块按段落再切（单段超长硬切）。

条目规则（全局契约，不得偏离）：
- source='thubook'；
- source_id='thubook:{md 相对路径去扩展名}[:{块序号:02d}]'（单块文件不带后缀）；
- title='{页面标题} - {小节标题}'（页面标题取一级标题，缺失取文件名；
  无小节标题时只用页面标题）；
- url='https://yourschool.cc.cd/thubook/{对应路由}.html'
  （VuePress 目录路由惯例：a/b.md → a/b.html，README.md → index.html）；
- metadata_json 含 file_path / section / category（顶级目录名，无则 'root'）
  及 page_title / chunk / chunks；
- updated_at 为 ISO 时间戳（可注入 now 便于测试）。

可测试性：解析函数全部纯函数化；campus_kb 延迟导入且 upsert 回调可注入；
--dry-run 完全不触达 campus_kb 与真实 db。

CLI 用法：
    /usr/local/bin/python3 scripts/thubook_ingest.py --dry-run
    /usr/local/bin/python3 scripts/thubook_ingest.py --src-dir <repo>/src --db data/campus_kb.sqlite
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 保证从项目根可导入 agent 包（脚本可被任意 cwd 调用；仅真实落库路径使用）。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── 全局常量 ──

DEFAULT_SRC_DIR = Path(
    "/Users/harriszhang/Documents/kimi/workspace/research/thubook_repo/src"
)
BASE_URL = "https://yourschool.cc.cd/thubook/"
SOURCE = "thubook"

MIN_BLOCK_CHARS = 300   # 单块建议下限（过短小节与相邻小节合并）
MAX_BLOCK_CHARS = 1500  # 单块建议上限（超长块按段落再切）

ENTRY_FIELDS: Tuple[str, ...] = (
    "source", "source_id", "title", "content", "url", "metadata_json",
    "updated_at",
)

# upsert 回调契约：entries -> 实际写入/更新条数（int），由 campus_kb/测试注入
UpsertFn = Callable[[List[dict]], int]


# ═══════════════════════════════════════════
# 纯函数：清洗
# ═══════════════════════════════════════════

def strip_front_matter(text: str) -> Tuple[str, Dict[str, str]]:
    """剥离文件开头 YAML front matter；返回 (正文, 简单解析的键值字典)。
    仅解析顶层 key: value 行，不依赖 pyyaml；无 front matter 时原样返回。"""
    meta: Dict[str, str] = {}
    m = re.match(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", text, re.S)
    if not m:
        return text, meta
    for line in m.group(1).splitlines():
        kv = re.match(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$", line)
        if kv:
            meta[kv.group(1)] = kv.group(2).strip().strip("\"'")
    return text[m.end():], meta


def _strip_html_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def _strip_containers(text: str) -> str:
    """剥离 ::: tip 等 VuePress 容器标记；容器自带标题（::: tip 标题）降级为
    普通文本行保留可读性。"""
    out = []
    for line in text.splitlines():
        m = re.match(r"^\s*:::\s*[\w-]*\s*(.*?)\s*$", line)
        if m:
            if m.group(1):
                out.append(m.group(1))
            continue
        out.append(line)
    return "\n".join(out)


def _strip_wikilinks(text: str) -> str:
    """[[...]] 双链保留可读文本。"""
    return re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)


def _vpcard_repl(m: re.Match) -> str:
    tag = m.group(0)

    def attr(name: str) -> str:
        mm = re.search(r'\b%s="([^"]*)"' % re.escape(name), tag)
        return mm.group(1) if mm else ""

    title, desc, link = attr("title"), attr("desc"), attr("link")
    parts = title or ""
    if desc:
        parts += ("：" if parts else "") + desc
    if link:
        parts += ("（" if parts else "") + link + ("）" if parts else "")
    return parts


def _strip_components(text: str) -> str:
    """Badge/VPCard 等组件标签降级为可读文本；其余 HTML 标签去壳保留内文。"""
    text = re.sub(r'<Badge\b[^>]*\btext="([^"]*)"[^>]*/?>', r"\1", text)
    text = re.sub(r"<VPCard\b[^>]*/?>", _vpcard_repl, text)
    # 成对标签（<VPLink>..</VPLink>、<span>..</span> 等）去壳
    text = re.sub(r"</?[A-Za-z][^>]*>", "", text)
    return text


def _strip_images_links(text: str) -> str:
    """图片降级为 [图片: alt]（无 alt 则移除）；链接降级为 文本（url）。"""
    text = re.sub(
        r"!\[([^\]]*)\]\([^)]*\)",
        lambda m: "[图片: %s]" % m.group(1) if m.group(1).strip() else "",
        text,
    )
    text = re.sub(
        r"\[([^\]]*)\]\(([^)]*)\)",
        lambda m: "%s（%s）" % (m.group(1), m.group(2)),
        text,
    )
    return text


def _split_table_row(line: str) -> List[str]:
    cells = line.strip().strip("|").split("|")
    return [c.strip() for c in cells]


def _convert_tables(text: str) -> str:
    """markdown 表格转可读文本行：有表头时每行「列名：值；…」，
    无法识别表头时各单元格用 / 连接。"""
    lines = text.splitlines()
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        is_row = line.strip().startswith("|") and line.strip().endswith("|")
        has_sep = (
            i + 1 < len(lines)
            and re.match(r"^\s*\|?[\s:\-|]+\|?\s*$", lines[i + 1]) is not None
            and "-" in lines[i + 1]
        )
        if is_row and has_sep:
            header = _split_table_row(line)
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = _split_table_row(lines[i])
                pairs = [
                    "%s：%s" % (h, c)
                    for h, c in zip(header, cells)
                    if h and c
                ]
                out.append("；".join(pairs) if pairs else " / ".join(cells))
                i += 1
        else:
            out.append(line)
            i += 1
    return "\n".join(out)


def _strip_emphasis(text: str) -> str:
    """粗体/斜体/删除线标记去壳，保留可读文本。"""
    return re.sub(r"(\*\*|__|~~)", "", text)


def clean_markdown(text: str) -> str:
    """整体清洗流水线（front matter 已由 strip_front_matter 先行剥离）。
    输出纯可读文本，保留 #/##/### 标题行供后续分块。"""
    text = _strip_html_comments(text)
    text = _strip_containers(text)
    text = _strip_wikilinks(text)
    text = _strip_components(text)
    text = _strip_images_links(text)
    text = _convert_tables(text)
    text = _strip_emphasis(text)
    # 收敛空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ═══════════════════════════════════════════
# 纯函数：分块
# ═══════════════════════════════════════════

_HEADING_RE = re.compile(r"^(#{2,3})\s+(.*?)\s*$")
_H1_RE = re.compile(r"^#\s+(.*?)\s*$", re.M)


def page_title_of(cleaned: str, path: Path) -> str:
    """页面标题取一级标题文本，缺失时取文件名（去扩展名）。"""
    m = _H1_RE.search(cleaned)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return path.stem


def split_sections(cleaned: str) -> List[Tuple[str, str]]:
    """按 ##/### 标题分块；# 一级标题行剔除（已用于页面标题）。
    返回 [(小节标题, 正文)]；前言部分小节标题为空串。"""
    sections: List[Tuple[str, List[str]]] = []
    cur_title = ""
    cur: List[str] = []
    for line in cleaned.splitlines():
        if _H1_RE.match(line):
            continue
        m = _HEADING_RE.match(line)
        if m:
            sections.append((cur_title, cur))
            cur_title, cur = m.group(2).strip(), []
        else:
            cur.append(line)
    sections.append((cur_title, cur))
    return [
        (t, "\n".join(b).strip())
        for t, b in sections
        if t or "\n".join(b).strip()
    ]


def _join_titles(titles: List[str], max_len: int = 60) -> str:
    """合并组内各小节标题；过长时截断避免撑爆块标题与正文前缀。"""
    joined = " / ".join(titles)
    if len(joined) > max_len:
        return joined[:max_len].rstrip(" /") + "…"
    return joined


def _group_sections(sections: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """贪心合并过短小节：当前组不足 MIN 或下一节不足 MIN 且合并后不超 MAX
    时并入当前组；组标题为各小节标题以「 / 」连接（超长截断）。"""
    groups: List[Tuple[str, str]] = []
    cur_titles: List[str] = []
    cur_body = ""
    for title, body in sections:
        candidate = (cur_body + "\n\n" + body).strip() if cur_body else body
        can_merge = (
            bool(cur_body)
            and len(candidate) <= MAX_BLOCK_CHARS
            and (len(cur_body) < MIN_BLOCK_CHARS or (body and len(body) < MIN_BLOCK_CHARS))
        )
        if can_merge:
            cur_body = candidate
            if title:
                cur_titles.append(title)
        else:
            if cur_body or cur_titles:
                groups.append((_join_titles(cur_titles), cur_body))
            cur_titles = [title] if title else []
            cur_body = body
    if cur_body or cur_titles:
        groups.append((_join_titles(cur_titles), cur_body))
    return groups


def _split_oversized(text: str, max_chars: int = MAX_BLOCK_CHARS) -> List[str]:
    """超长块按段落（空行分隔）贪心再切；单段超长时硬切。"""
    if len(text) <= max_chars:
        return [text]
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    cur = ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = (cur + "\n\n" + p).strip() if cur else p
        while len(cur) > max_chars:
            chunks.append(cur[:max_chars])
            cur = cur[max_chars:]
    if cur:
        chunks.append(cur)
    return chunks


def chunk_sections(sections: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """合并过短 + 切分超长，返回 [(小节标题, 块文本)] 最终块序列。"""
    chunks: List[Tuple[str, str]] = []
    for title, body in _group_sections(sections):
        for piece in _split_oversized(body):
            chunks.append((title, piece))
    return chunks


# ═══════════════════════════════════════════
# 纯函数：路由与条目构造
# ═══════════════════════════════════════════

def route_for(rel_path: str) -> str:
    """VuePress 目录路由惯例：a/b.md → a/b.html，README.md → index.html。"""
    p = rel_path[:-3] if rel_path.endswith(".md") else rel_path
    if p == "README":
        return "index.html"
    if p.endswith("/README"):
        return p[: -len("/README")] + "/index.html"
    return p + ".html"


def make_entry(
    rel_path: str,
    page_title: str,
    section: str,
    chunk_text: str,
    chunk_index: int,
    chunk_count: int,
    now: Optional[str] = None,
) -> dict:
    """构造全局契约 8 字段条目字典。"""
    stem = rel_path[:-3] if rel_path.endswith(".md") else rel_path
    if chunk_count > 1:
        source_id = "%s:%s:%02d" % (SOURCE, stem, chunk_index)
    else:
        source_id = "%s:%s" % (SOURCE, stem)
    title = "%s - %s" % (page_title, section) if section else page_title
    content = ("%s\n%s" % (section, chunk_text)).strip() if section else chunk_text
    category = rel_path.split("/")[0] if "/" in rel_path else "root"
    metadata = {
        "file_path": rel_path,
        "section": section,
        "category": category,
        "page_title": page_title,
        "chunk": chunk_index,
        "chunks": chunk_count,
    }
    return {
        "source": SOURCE,
        "source_id": source_id,
        "title": title,
        "content": content,
        "url": BASE_URL + route_for(rel_path),
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
        "updated_at": now or datetime.now().isoformat(timespec="seconds"),
    }


def parse_file(path: Path, src_dir: Path, now: Optional[str] = None) -> List[dict]:
    """解析单个 markdown 文件为条目列表（纯解析，不写库）。"""
    rel = path.relative_to(src_dir).as_posix()
    raw = path.read_text(encoding="utf-8")
    body, _fm = strip_front_matter(raw)
    cleaned = clean_markdown(body)
    if not cleaned:
        return []
    page_title = page_title_of(cleaned, path)
    chunks = chunk_sections(split_sections(cleaned))
    chunks = [(t, c) for t, c in chunks if c.strip()]
    if not chunks:
        return []
    n = len(chunks)
    return [
        make_entry(rel, page_title, title, text, i + 1, n, now=now)
        for i, (title, text) in enumerate(chunks)
    ]


def collect_entries(src_dir, now: Optional[str] = None) -> List[dict]:
    """遍历 src_dir 全部 .md（含子目录），产出全量条目列表。"""
    src = Path(src_dir)
    entries: List[dict] = []
    for path in sorted(src.rglob("*.md")):
        try:
            entries.extend(parse_file(path, src, now=now))
        except Exception as exc:  # 单文件失败记 warning 并继续，绝不中断全量
            logger.warning("解析失败 %s: %s", path, exc)
    return entries


# ═══════════════════════════════════════════
# 入库与 CLI
# ═══════════════════════════════════════════

def _default_upsert(db_path=None) -> UpsertFn:
    """延迟导入 campus_kb（测试期由 monkeypatch/注入替代，绝不触真实 db）。"""
    from agent import campus_kb

    campus_kb.init_db(db_path=db_path)
    return lambda entries: campus_kb.upsert_entries(entries, db_path=db_path)


def ingest(
    src_dir=None,
    db_path=None,
    dry_run: bool = False,
    upsert_fn: Optional[UpsertFn] = None,
    now: Optional[str] = None,
) -> dict:
    """全量入库主流程。返回统计字典 {pages, chunks, upserted, files}。
    dry_run 或注入 upsert_fn 时绝不导入 campus_kb。"""
    src = Path(src_dir) if src_dir else DEFAULT_SRC_DIR
    entries = collect_entries(src, now=now)
    pages = sorted({json.loads(e["metadata_json"])["file_path"] for e in entries})
    upserted: Optional[int] = None
    if not dry_run:
        fn = upsert_fn or _default_upsert(db_path)
        upserted = fn(entries)
    return {
        "pages": len(pages),
        "chunks": len(entries),
        "upserted": upserted,
        "files": pages,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="thubook 清华校内百科 markdown 全量解析入库（source='thubook'）"
    )
    parser.add_argument("--src-dir", default=str(DEFAULT_SRC_DIR),
                        help="thubook 仓库 src 目录（默认已下载路径）")
    parser.add_argument("--db", default=None, help="campus_kb 数据库路径（默认 None）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只解析打印将入库块数与页面清单，不写库")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    result = ingest(src_dir=args.src_dir, db_path=args.db, dry_run=args.dry_run)

    if args.dry_run:
        print("[dry-run] 将入库 %d 个页面、%d 个条目块：" % (result["pages"], result["chunks"]))
        for f in result["files"]:
            print("  - %s" % f)
        print("[dry-run] 未写库。")
    else:
        print("入库完成：%d 个页面、%d 个条目块，upsert %d 条。"
              % (result["pages"], result["chunks"], result["upserted"] or 0))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
