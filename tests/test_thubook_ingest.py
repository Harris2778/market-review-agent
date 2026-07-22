"""scripts/thubook_ingest.py thubook 百科入库脚本测试（全 mock 零网络零真实文件）。

覆盖范围：
1. 清洗：front matter 剥离、::: 容器降级、Badge/VPCard 组件提取、
   [[...]] 双链保留文本、表格转可读行、图片/链接/粗体删除线降级。
2. 分块：##/### 标题分块、无标题短文件整块一条、超长块按段落再切、
   过短小节合并、#### 四级标题不触发分块。
3. 条目字段：契约 7 键齐全、source_id 规则（单块无后缀/多块 :NN）、
   title 组合、url 路由（含 README→index.html）、metadata category
   （子目录名/root）、updated_at ISO。
4. 入库：fake upsert 注入（绝不 import agent.campus_kb）、幂等覆盖、
   dry-run 不写库、CLI dry-run 输出页面清单。

所有 fixture 均在 tmp_path 现场书写，零真实外部文件依赖。
"""

import json

import pytest

import scripts.thubook_ingest as ti


# ── 公共工具 ──

NOW = "2026-07-22T12:00:00"


def write_md(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def parse_one(tmp_path, rel, text):
    """写入单文件并解析，返回条目列表。"""
    write_md(tmp_path, rel, text)
    return ti.parse_file(tmp_path / rel, tmp_path, now=NOW)


class FakeStore:
    """模拟 campus_kb 的 upsert 语义：(source, source_id) 主键覆盖。"""

    def __init__(self):
        self.rows = {}
        self.calls = 0

    def upsert(self, entries):
        self.calls += 1
        for e in entries:
            self.rows[(e["source"], e["source_id"])] = e
        return len(entries)


# ── 清洗 ──

class TestClean:
    def test_strip_front_matter(self):
        text, meta = ti.strip_front_matter(
            "---\ntitle: 测试页\nauthor: 张三\n---\n# 正文标题\n内容\n"
        )
        assert meta == {"title": "测试页", "author": "张三"}
        assert not text.startswith("---")
        assert "正文标题" in text

    def test_front_matter_keys_not_in_content(self, tmp_path):
        entries = parse_one(
            tmp_path, "a.md",
            "---\ntitle: 页面\n---\n# 页面\n\n这是正文内容。\n",
        )
        assert entries and "author" not in entries[0]["content"]

    def test_container_tip_stripped(self):
        cleaned = ti.clean_markdown(
            "::: tip 重要提示\n记得带校园卡。\n:::\n\n::: warning\n注意安全\n:::\n"
        )
        assert ":::" not in cleaned
        assert "重要提示" in cleaned  # 容器标题降级为可读文本
        assert "记得带校园卡。" in cleaned
        assert "注意安全" in cleaned

    def test_badge_and_vpcard_components(self):
        cleaned = ti.clean_markdown(
            '<Badge text="必读" type="tip"/> 新生指南\n'
            '<VPCard title="info 门户" desc="校内信息门户" '
            'link="https://info.tsinghua.edu.cn" />\n'
        )
        assert "<Badge" not in cleaned and "<VPCard" not in cleaned
        assert "必读" in cleaned
        assert "info 门户" in cleaned and "校内信息门户" in cleaned
        assert "https://info.tsinghua.edu.cn" in cleaned

    def test_wikilink_kept_readable(self):
        cleaned = ti.clean_markdown("详见 [[常用网站]] 与 [[新生指南|指南]]。\n")
        assert "[[" not in cleaned and "]]" not in cleaned
        assert "常用网站" in cleaned

    def test_table_to_readable_lines(self):
        cleaned = ti.clean_markdown(
            "| 等级 | 绩点 |\n|:-:|:-:|\n|A+|4.0|\n|B-|3.0|\n"
        )
        assert "|" not in cleaned
        assert "等级：A+；绩点：4.0" in cleaned
        assert "等级：B-；绩点：3.0" in cleaned

    def test_image_link_emphasis(self):
        cleaned = ti.clean_markdown(
            "![xiuche](/assets/xiuche.jpg)\n"
            "登录 [info](http://info.tsinghua.edu.cn/) 查询，"
            "**非常重要**，~~已废弃~~。\n"
        )
        assert "![" not in cleaned
        assert "[图片: xiuche]" in cleaned
        assert "info（http://info.tsinghua.edu.cn/）" in cleaned
        assert "**" not in cleaned and "~~" not in cleaned
        assert "非常重要" in cleaned and "已废弃" in cleaned


# ── 分块 ──

class TestChunking:
    def test_split_by_h2_h3_and_title(self, tmp_path):
        entries = parse_one(
            tmp_path, "专题/srt.md",
            "# SRT 项目指南\n\n" + "介绍文字。" * 80 + "\n\n"
            "## 立项申请\n\n" + "第一步填写申请书，提交并确认。" * 30 + "\n\n"
            "### 学生报名\n\n" + "查询项目并填写报名申请书。" * 30 + "\n",
        )
        assert len(entries) == 3  # 前言 + ## 立项申请 + ### 学生报名（各 ≥300 字不合并）
        assert all(e["title"].startswith("SRT 项目指南") for e in entries)
        sections = [json.loads(e["metadata_json"])["section"] for e in entries]
        assert "立项申请" in sections and "学生报名" in sections

    def test_no_heading_short_file_single_block(self, tmp_path):
        entries = parse_one(
            tmp_path, "words.md",
            "# 清华黑话\n\n" + "这是一条没有小节标题的解释内容。" * 5 + "\n",
        )
        assert len(entries) == 1
        e = entries[0]
        assert e["source_id"] == "thubook:words"  # 单块不带 :NN 后缀
        assert e["title"] == "清华黑话"

    def test_long_section_split_by_paragraph(self, tmp_path):
        para = "清华大学于 1911 年建校，前身为清华学堂，历史沿革内容详实。"
        body = "\n\n".join(para * 10 for _ in range(12))  # 远超 1500 字
        entries = parse_one(tmp_path, "thustory.md", "# 校史\n\n## 沿革\n\n" + body)
        assert len(entries) > 1
        assert entries[0]["source_id"] == "thubook:thustory:01"
        assert entries[1]["source_id"] == "thubook:thustory:02"
        for e in entries:
            # 块正文 ≤1500，加小节标题前缀后仍受控
            assert len(e["content"]) <= ti.MAX_BLOCK_CHARS + 100
        # 块序号 zero-padded 且与 metadata 一致
        metas = [json.loads(e["metadata_json"]) for e in entries]
        assert [m["chunk"] for m in metas] == list(range(1, len(entries) + 1))
        assert all(m["chunks"] == len(entries) for m in metas)

    def test_small_sections_merged(self, tmp_path):
        text = "# 修车铺\n\n"
        for name in ["紫3", "紫5", "紫9", "小桥"]:
            text += "## %s\n\n地点：某处，营业时间 08:30-21:30。\n\n" % name
        entries = parse_one(tmp_path, "校内生活设施/xiuche.md", text)
        # 每个小节远不足 300 字，应合并为少量条目
        assert len(entries) <= 2
        body = "".join(e["content"] for e in entries)
        for name in ["紫3", "紫5", "紫9", "小桥"]:
            assert name in body

    def test_h4_not_split(self):
        sections = ti.split_sections("# 页\n\n## 节\n\n#### 四级标题\n\n内容。\n")
        assert len(sections) == 1  # #### 不触发分块，留在 ## 节正文
        assert "#### 四级标题" in sections[0][1]


# ── 条目字段契约 ──

class TestEntryFields:
    def test_contract_keys_and_types(self, tmp_path):
        entries = parse_one(
            tmp_path, "专题/srt.md", "# SRT\n\n" + "内容。" * 100 + "\n"
        )
        assert len(entries) == 1
        e = entries[0]
        assert set(e.keys()) == set(ti.ENTRY_FIELDS)
        assert e["source"] == "thubook"
        assert isinstance(e["content"], str) and len(e["content"]) >= 100
        assert e["updated_at"] == NOW

    def test_metadata_category_and_route(self, tmp_path):
        e = parse_one(tmp_path, "专题/srt.md", "# SRT\n\n" + "内容。" * 100)[0]
        meta = json.loads(e["metadata_json"])
        assert meta["file_path"] == "专题/srt.md"
        assert meta["category"] == "专题"
        assert e["url"] == "https://yourschool.cc.cd/thubook/专题/srt.html"

        e2 = parse_one(tmp_path, "phones.md", "# 电话\n\n" + "内容。" * 100)[0]
        meta2 = json.loads(e2["metadata_json"])
        assert meta2["category"] == "root"
        assert e2["url"] == "https://yourschool.cc.cd/thubook/phones.html"

    def test_readme_route_index(self, tmp_path):
        e = parse_one(tmp_path, "README.md", "# 手册首页\n\n" + "欢迎。" * 100)[0]
        assert e["source_id"] == "thubook:README"
        assert e["url"] == "https://yourschool.cc.cd/thubook/index.html"

    def test_page_title_fallback_to_filename(self, tmp_path):
        e = parse_one(tmp_path, "jiqiao.md", "没有一级标题的正文。" * 50)[0]
        assert e["title"].startswith("jiqiao")

    def test_section_title_in_entry_title(self, tmp_path):
        entries = parse_one(
            tmp_path, "专题/tice.md",
            "# 体测指南\n\n## 免测申请\n\n" + "去医院开证明并提交。" * 30 + "\n",
        )
        assert entries[0]["title"] == "体测指南 - 免测申请"

    def test_empty_file_skipped(self, tmp_path):
        assert parse_one(tmp_path, "empty.md", "") == []
        assert parse_one(tmp_path, "blank.md", "---\ntitle: x\n---\n\n  \n") == []


# ── 入库与 CLI ──

class TestIngest:
    def _build_tree(self, root):
        write_md(root, "phones.md", "# 常用电话\n\n" + "校医院 62782082。" * 30)
        write_md(root, "专题/srt.md",
                 "# SRT\n\n## 报名\n\n" + "登录 info 报名。" * 40)
        write_md(root, "专题/shitang.md",
                 "# 食堂\n\n## 紫荆园\n\n" + "四层川菜。" * 40)
        write_md(root, "校内生活设施/xiuche.md",
                 "# 修车铺\n\n## 紫3\n\n地点：桃李园西侧。" * 20)

    def test_collect_entries_multi_files(self, tmp_path):
        self._build_tree(tmp_path)
        entries = ti.collect_entries(tmp_path, now=NOW)
        assert len(entries) >= 4
        assert all(e["source"] == "thubook" for e in entries)
        sids = [e["source_id"] for e in entries]
        assert len(sids) == len(set(sids))  # 全量 source_id 唯一
        assert "thubook:phones" in sids

    def test_ingest_fake_upsert_idempotent(self, tmp_path):
        self._build_tree(tmp_path)
        store = FakeStore()
        r1 = ti.ingest(src_dir=tmp_path, upsert_fn=store.upsert, now=NOW)
        n1 = len(store.rows)
        assert r1["upserted"] == r1["chunks"] == n1
        assert r1["pages"] == 4
        # 幂等：重复全量跑，(source, source_id) 主键覆盖，行数不变
        store2_count = ti.ingest(src_dir=tmp_path, upsert_fn=store.upsert, now=NOW)
        assert store2_count["chunks"] == n1
        assert len(store.rows) == n1
        assert store.calls == 2

    def test_dry_run_never_imports_campus_kb(self, tmp_path, monkeypatch):
        self._build_tree(tmp_path)
        monkeypatch.setitem(
            __import__("sys").modules, "agent.campus_kb", None
        )  # 若被导入会立刻炸
        result = ti.ingest(src_dir=tmp_path, dry_run=True, now=NOW)
        assert result["upserted"] is None
        assert result["chunks"] > 0

    def test_cli_dry_run_output(self, tmp_path, capsys):
        self._build_tree(tmp_path)
        rc = ti.main(["--src-dir", str(tmp_path), "--dry-run"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "dry-run" in out
        assert "专题/srt.md" in out and "phones.md" in out
        assert "未写库" in out
