"""投资人格体系（agent.personas）契约测试 — 全 mock 零网络。

覆盖：JSON 加载 / 惰性缓存 / 环境变量路径覆盖 / 坏文件优雅降级、
四个人格字段完整性遍历断言、framework 渲染结构与拷贝隔离、
validate_persona_output 各违规分支（signal 非法 / 缺失、confidence
超限 / 缺失 / 非数值、非 dict 输入）、模块纯 stdlib 与零网络约束。
"""

import json
import logging
import threading
from pathlib import Path

import pytest

from agent import personas

EXPECTED_KEYS = {"value_cn", "growth_cn", "trend_cn", "contrarian_cn"}

DATA_PATH = Path(personas.__file__).parent / "persona_defs.json"

REQUIRED_FIELDS = (
    "key",
    "name",
    "description",
    "instructions",
    "scoring_weights",
    "thresholds",
    "analysis_rules",
    "data_requirements",
    "output_schema",
    "checklist",
)


@pytest.fixture(autouse=True)
def reset_cache(monkeypatch):
    """每个用例前清缓存并剥掉环境变量，保证用例间隔离。"""
    monkeypatch.delenv(personas.ENV_PATH_KEY, raising=False)
    monkeypatch.setattr(personas, "_cache", None)
    yield
    monkeypatch.setattr(personas, "_cache", None)


@pytest.fixture()
def raw_data():
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


# ── 数据文件与人格字段完整性 ─────────────────────────────────────

def test_data_file_is_valid_json(raw_data):
    assert isinstance(raw_data, dict) and raw_data


def test_covers_exactly_four_personas(raw_data):
    assert set(raw_data.keys()) == EXPECTED_KEYS


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
def test_persona_required_fields_complete(key, raw_data):
    persona = raw_data[key]
    for field in REQUIRED_FIELDS:
        assert field in persona, f"{key} 缺字段 {field}"
    assert persona["key"] == key
    assert isinstance(persona["name"], str) and persona["name"].strip()
    assert isinstance(persona["description"], str) and persona["description"].strip()
    assert isinstance(persona["scoring_weights"], dict) and persona["scoring_weights"]
    assert isinstance(persona["thresholds"], dict) and persona["thresholds"]
    assert isinstance(persona["analysis_rules"], list) and persona["analysis_rules"]
    assert isinstance(persona["data_requirements"], list) and persona["data_requirements"]
    assert isinstance(persona["checklist"], list) and persona["checklist"]
    assert all(isinstance(r, str) and r.strip() for r in persona["analysis_rules"])
    assert all(isinstance(d, str) and d.strip() for d in persona["data_requirements"])
    assert all(isinstance(c, str) and c.strip() for c in persona["checklist"])


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
def test_instructions_length_within_200_400(key, raw_data):
    instructions = raw_data[key]["instructions"]
    assert isinstance(instructions, str)
    assert 200 <= len(instructions) <= 400


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
def test_scoring_weights_sum_to_one(key, raw_data):
    weights = raw_data[key]["scoring_weights"]
    assert all(isinstance(v, (int, float)) and v > 0 for v in weights.values())
    assert abs(sum(weights.values()) - 1.0) < 1e-9


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
def test_output_schema_contract(key, raw_data):
    schema = raw_data[key]["output_schema"]
    assert schema["signal"] == list(personas.SIGNAL_ENUM)
    assert schema["confidence_max"] == 0.9
    assert set(schema["分项"].keys()) == set(raw_data[key]["scoring_weights"].keys())


# ── 加载 / 缓存 / 路径覆盖 / 降级 ────────────────────────────────

def test_list_personas_returns_four_summaries():
    items = personas.list_personas()
    assert len(items) == 4
    assert {item["key"] for item in items} == EXPECTED_KEYS
    for item in items:
        assert set(item.keys()) == {"key", "name", "description"}
        assert all(isinstance(v, str) and v for v in item.values())


def test_list_personas_returns_fresh_copies():
    a = personas.list_personas()
    a[0]["name"] = "污染"
    a.append({"key": "污染", "name": "污染", "description": "污染"})
    fresh = personas.list_personas()
    assert len(fresh) == 4
    assert all(item["name"] != "污染" for item in fresh)


def test_lazy_load_caches_after_first_call(monkeypatch):
    assert personas._cache is None
    personas.list_personas()
    assert personas._cache is not None
    # 缓存后即使路径失效也仍返回缓存数据，不重复读盘、不告警。
    monkeypatch.setenv(personas.ENV_PATH_KEY, "/不存在/路径.json")
    assert len(personas.list_personas()) == 4


def test_env_path_override(monkeypatch, tmp_path):
    custom = tmp_path / "custom_personas.json"
    custom.write_text(json.dumps({
        "solo_cn": {"key": "solo_cn", "name": "单人", "description": "测试"}
    }), encoding="utf-8")
    monkeypatch.setenv(personas.ENV_PATH_KEY, str(custom))
    assert personas._cache is None
    items = personas.list_personas()
    assert [item["key"] for item in items] == ["solo_cn"]


def test_missing_file_degrades_to_empty(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv(personas.ENV_PATH_KEY, str(tmp_path / "不存在.json"))
    with caplog.at_level(logging.WARNING):
        assert personas.list_personas() == []
        assert personas.get_persona("value_cn") is None
        assert personas.render_persona_framework("value_cn") is None
    assert "人格定义加载失败" in caplog.text


def test_corrupted_json_degrades_to_empty(monkeypatch, tmp_path, caplog):
    bad = tmp_path / "bad.json"
    bad.write_text("{ 这不是合法 JSON", encoding="utf-8")
    monkeypatch.setenv(personas.ENV_PATH_KEY, str(bad))
    with caplog.at_level(logging.WARNING):
        assert personas.list_personas() == []
    assert "人格定义加载失败" in caplog.text


def test_non_dict_top_level_degrades_to_empty(monkeypatch, tmp_path, caplog):
    bad = tmp_path / "list.json"
    bad.write_text(json.dumps(["不是对象"]), encoding="utf-8")
    monkeypatch.setenv(personas.ENV_PATH_KEY, str(bad))
    with caplog.at_level(logging.WARNING):
        assert personas.list_personas() == []
    assert "顶层不是 JSON 对象" in caplog.text


def test_concurrent_first_load_thread_safe():
    results: list[list] = []

    def worker():
        results.append(personas.list_personas())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 8
    assert all(len(r) == 4 for r in results)


# ── get_persona ─────────────────────────────────────────────────

def test_get_persona_returns_full_definition(raw_data):
    persona = personas.get_persona("value_cn")
    assert persona is not None
    assert persona == raw_data["value_cn"]


@pytest.mark.parametrize("bad", ["不存在", "", "   ", None, 123, ["value_cn"]])
def test_get_persona_bad_input_returns_none(bad):
    assert personas.get_persona(bad) is None


def test_get_persona_returns_deep_copy():
    persona = personas.get_persona("trend_cn")
    persona["thresholds"]["bias_max"] = 999
    persona["analysis_rules"].append("污染")
    fresh = personas.get_persona("trend_cn")
    assert fresh["thresholds"]["bias_max"] == 10
    assert "污染" not in fresh["analysis_rules"]


# ── render_persona_framework ────────────────────────────────────

FRAMEWORK_KEYS = {
    "name", "instructions", "scoring_weights", "thresholds",
    "analysis_rules", "output_schema", "checklist", "disclaimer",
}


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
def test_render_framework_structure(key):
    framework = personas.render_persona_framework(key)
    assert framework is not None
    assert set(framework.keys()) == FRAMEWORK_KEYS
    assert framework["disclaimer"] == personas.DISCLAIMER
    assert framework["disclaimer"] == "方法论框架仅供参考，不构成投资建议"
    assert framework["output_schema"]["signal"] == list(personas.SIGNAL_ENUM)


def test_render_framework_unknown_returns_none():
    assert personas.render_persona_framework("不存在") is None
    assert personas.render_persona_framework(None) is None


def test_render_framework_returns_copy():
    framework = personas.render_persona_framework("growth_cn")
    framework["checklist"].append("污染")
    fresh = personas.render_persona_framework("growth_cn")
    assert "污染" not in fresh["checklist"]


# ── validate_persona_output ─────────────────────────────────────

def test_validate_clean_output_ok():
    data = {"signal": "增持", "confidence": 0.7, "分项": {"财务质量": 80}}
    result = personas.validate_persona_output(data, persona_key="value_cn")
    assert result["ok"] is True
    assert result["violations"] == []
    assert result["normalized"]["signal"] == "增持"
    assert result["normalized"]["confidence"] == 0.7
    assert result["normalized"]["分项"] == {"财务质量": 80}


def test_validate_does_not_mutate_input():
    data = {"signal": "买入", "confidence": 0.95}
    personas.validate_persona_output(data)
    assert data == {"signal": "买入", "confidence": 0.95}


def test_validate_confidence_over_max_clamped():
    result = personas.validate_persona_output({"signal": "买入", "confidence": 0.95})
    assert result["ok"] is False
    assert result["normalized"]["confidence"] == 0.9
    assert any("超过上限" in v for v in result["violations"])


def test_validate_confidence_negative_clamped():
    result = personas.validate_persona_output({"signal": "观望", "confidence": -0.3})
    assert result["ok"] is False
    assert result["normalized"]["confidence"] == 0.0
    assert any("低于 0" in v for v in result["violations"])


def test_validate_confidence_non_numeric_defaults():
    result = personas.validate_persona_output({"signal": "观望", "confidence": "很高"})
    assert result["ok"] is False
    assert result["normalized"]["confidence"] == 0.5
    assert any("不是数值" in v for v in result["violations"])


def test_validate_confidence_bool_treated_as_non_numeric():
    result = personas.validate_persona_output({"signal": "观望", "confidence": True})
    assert result["ok"] is False
    assert result["normalized"]["confidence"] == 0.5


def test_validate_invalid_signal_forced_to_observe():
    result = personas.validate_persona_output({"signal": "梭哈", "confidence": 0.6})
    assert result["ok"] is False
    assert result["normalized"]["signal"] == "观望"
    assert any("不在允许枚举" in v for v in result["violations"])


def test_validate_missing_signal_defaults():
    result = personas.validate_persona_output({"confidence": 0.6})
    assert result["ok"] is False
    assert result["normalized"]["signal"] == "观望"
    assert any("缺少 signal" in v for v in result["violations"])


def test_validate_missing_confidence_defaults():
    result = personas.validate_persona_output({"signal": "买入"})
    assert result["ok"] is False
    assert result["normalized"]["confidence"] == 0.5
    assert any("缺少 confidence" in v for v in result["violations"])


def test_validate_non_dict_input():
    result = personas.validate_persona_output("随便一段文本")
    assert result["ok"] is False
    assert result["normalized"]["signal"] == "观望"
    assert result["normalized"]["confidence"] == 0.5
    assert any("不是 JSON 对象" in v for v in result["violations"])


def test_validate_multiple_violations_accumulate():
    result = personas.validate_persona_output({"signal": "满仓", "confidence": 1.5})
    assert result["ok"] is False
    assert len(result["violations"]) == 2
    assert result["normalized"]["signal"] == "观望"
    assert result["normalized"]["confidence"] == 0.9


def test_validate_persona_key_uses_schema_contract():
    # 人格 schema 的 signal 枚举与缺省一致时合法信号应通过。
    result = personas.validate_persona_output(
        {"signal": "回避", "confidence": 0.4}, persona_key="contrarian_cn"
    )
    assert result["ok"] is True
    assert result["normalized"]["signal"] == "回避"


def test_validate_unknown_persona_key_falls_back(caplog):
    with caplog.at_level(logging.WARNING):
        result = personas.validate_persona_output(
            {"signal": "买入", "confidence": 0.8}, persona_key="不存在的人格"
        )
    assert result["ok"] is True
    assert result["normalized"]["signal"] == "买入"
    assert "未知人格" in caplog.text


def test_validate_custom_schema_confidence_max(monkeypatch):
    monkeypatch.setattr(personas, "_cache", {
        "strict_cn": {
            "name": "严格",
            "output_schema": {"signal": ["买入", "观望"], "confidence_max": 0.5},
        }
    })
    result = personas.validate_persona_output(
        {"signal": "买入", "confidence": 0.7}, persona_key="strict_cn"
    )
    assert result["ok"] is False
    assert result["normalized"]["confidence"] == 0.5
    assert any("超过上限 0.5" in v for v in result["violations"])


# ── 模块约定 ────────────────────────────────────────────────────

def test_module_is_pure_stdlib():
    import ast

    tree = ast.parse(Path(personas.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert imported <= {"copy", "json", "logging", "os", "threading", "pathlib", "__future__"}
