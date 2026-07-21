"""行业知识库（agent.industry_kb）契约测试 — 全 mock 零网络。

覆盖：31 行业数据完整性、字段齐全与条数约束、别名解析、未知行业返回 None、
format 长度上限与固定头部、坏数据 / 坏文件静默容错。
"""

import json
from pathlib import Path

import pytest

from agent import industry_kb

# 任务给定 30 个申万行业 + 第 31 个「综合」（申万一级传统第 31 席）。
EXPECTED_INDUSTRIES = {
    "农林牧渔", "采掘", "化工", "钢铁", "有色金属", "电子", "家用电器",
    "食品饮料", "纺织服装", "轻工制造", "医药生物", "公用事业", "交通运输",
    "房地产", "商业贸易", "休闲服务", "建筑材料", "建筑装饰", "电气设备",
    "国防军工", "计算机", "传媒", "通信", "银行", "非银金融", "汽车",
    "机械设备", "煤炭", "石油石化", "环保", "综合",
}

DATA_PATH = Path(industry_kb.__file__).with_name("industry_kb_data.json")


@pytest.fixture()
def raw_data():
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


# ── 数据文件完整性 ──────────────────────────────────────────────

def test_data_file_is_valid_json(raw_data):
    assert isinstance(raw_data, dict) and raw_data


def test_covers_all_31_industries(raw_data):
    assert len(raw_data) == 31
    assert set(raw_data.keys()) == EXPECTED_INDUSTRIES


def test_list_industries_matches_data(raw_data):
    names = industry_kb.list_industries()
    assert isinstance(names, list)
    assert len(names) == 31
    assert set(names) == EXPECTED_INDUSTRIES
    assert all(isinstance(n, str) for n in names)


def test_list_industries_returns_fresh_copy():
    a = industry_kb.list_industries()
    a.append("污染")
    assert "污染" not in industry_kb.list_industries()


@pytest.mark.parametrize("sector", sorted(EXPECTED_INDUSTRIES))
def test_profile_fields_complete(sector, raw_data):
    profile = raw_data[sector]
    assert set(profile.keys()) == {"chain", "drivers", "indicators", "leaders"}
    assert isinstance(profile["chain"], str) and profile["chain"].strip()
    assert isinstance(profile["drivers"], list)
    assert 2 <= len(profile["drivers"]) <= 4
    assert isinstance(profile["indicators"], list)
    assert 2 <= len(profile["indicators"]) <= 3
    assert isinstance(profile["leaders"], list)
    assert 3 <= len(profile["leaders"]) <= 5
    for field in ("drivers", "indicators", "leaders"):
        assert all(isinstance(x, str) and x.strip() for x in profile[field])


# ── get_industry_profile ────────────────────────────────────────

def test_get_profile_exact_match(raw_data):
    profile = industry_kb.get_industry_profile("电子")
    assert profile is not None
    assert profile == raw_data["电子"]


@pytest.mark.parametrize("alias,canonical", [
    ("半导体", "电子"),
    ("芯片", "电子"),
    ("白酒", "食品饮料"),
    ("光伏", "电气设备"),
    ("新能源", "电气设备"),
    ("券商", "非银金融"),
    ("保险", "非银金融"),
    ("医疗", "医药生物"),
    ("养殖", "农林牧渔"),
    ("水泥", "建筑材料"),
])
def test_get_profile_alias(alias, canonical, raw_data):
    assert industry_kb.get_industry_profile(alias) == raw_data[canonical]


@pytest.mark.parametrize("loose,canonical", [
    ("军工", "国防军工"),
    ("地产", "房地产"),
    ("石化", "石油石化"),
    ("医药", "医药生物"),
    ("电子行业", "电子"),
    ("白酒板块 ", "食品饮料"),
])
def test_get_profile_loose_tolerance(loose, canonical, raw_data):
    assert industry_kb.get_industry_profile(loose) == raw_data[canonical]


@pytest.mark.parametrize("bad", [
    "不存在的行业", "", "   ", None, 123, ["电子"], "煤",
])
def test_get_profile_unknown_returns_none(bad):
    assert industry_kb.get_industry_profile(bad) is None


def test_get_profile_returns_copy_not_cache():
    profile = industry_kb.get_industry_profile("电子")
    profile["drivers"].append("污染")
    profile["chain"] = "污染"
    fresh = industry_kb.get_industry_profile("电子")
    assert "污染" not in fresh["drivers"]
    assert fresh["chain"] != "污染"


# ── format_kb_block ─────────────────────────────────────────────

def test_format_header_and_structure():
    block = industry_kb.format_kb_block("电子")
    assert block is not None
    assert block.startswith(industry_kb.KB_HEADER)
    assert block.startswith("【六、行业知识库（背景知识，数据以数据块为准）】")
    assert "行业：电子（申万一级）" in block
    for label in ("产业链：", "核心驱动：", "关键指标：", "代表公司："):
        assert label in block


@pytest.mark.parametrize("sector", sorted(EXPECTED_INDUSTRIES))
def test_format_length_limit_all_industries(sector):
    block = industry_kb.format_kb_block(sector)
    assert block is not None
    assert len(block) <= 400
    assert block.startswith(industry_kb.KB_HEADER)


def test_format_alias_resolves():
    block = industry_kb.format_kb_block("白酒")
    assert block is not None
    assert "行业：食品饮料（申万一级）" in block
    assert "贵州茅台" in block


@pytest.mark.parametrize("bad", ["不存在的行业", "", None, 123])
def test_format_unknown_returns_none(bad):
    assert industry_kb.format_kb_block(bad) is None


# ── 容错：坏数据 / 坏文件不抛异常 ────────────────────────────────

def test_missing_fields_return_none(monkeypatch):
    monkeypatch.setattr(industry_kb, "_cache", {
        "电子": {"chain": "有 chain 但缺列表字段"},
        "银行": {"chain": "x", "drivers": [], "indicators": ["a"], "leaders": ["b"]},
    })
    assert industry_kb.get_industry_profile("电子") is None
    assert industry_kb.format_kb_block("电子") is None
    assert industry_kb.get_industry_profile("银行") is None
    assert industry_kb.list_industries() == ["电子", "银行"]


def test_data_file_missing_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(industry_kb, "_cache", None)
    monkeypatch.setattr(industry_kb, "_DATA_PATH", tmp_path / "不存在.json")
    assert industry_kb.get_industry_profile("电子") is None
    assert industry_kb.format_kb_block("电子") is None
    assert industry_kb.list_industries() == []


def test_data_file_corrupted_returns_none(monkeypatch, tmp_path):
    bad = tmp_path / "industry_kb_data.json"
    bad.write_text("{ 这不是合法 JSON", encoding="utf-8")
    monkeypatch.setattr(industry_kb, "_cache", None)
    monkeypatch.setattr(industry_kb, "_DATA_PATH", bad)
    assert industry_kb.get_industry_profile("电子") is None
    assert industry_kb.list_industries() == []


# ── 模块约定 ────────────────────────────────────────────────────

def test_alias_table_within_limit():
    assert len(industry_kb._ALIASES) <= 10
    for alias, target in industry_kb._ALIASES.items():
        assert target in EXPECTED_INDUSTRIES
        assert alias != target


def test_module_is_pure_stdlib():
    import ast

    tree = ast.parse(Path(industry_kb.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert imported <= {"json", "pathlib", "__future__"}
