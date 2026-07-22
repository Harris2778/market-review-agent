"""投资人格体系（第十波）— FinceptTerminal 启发的「配置即提示词」方法论框架。

不做人物扮演，只做方法论纪律：每个人格 = 一套可配置的分析纪律
（打分权重 + 阈值 + 分析规则 + 结构化输出 schema + 分析清单）。
人格定义存于同目录 ``persona_defs.json``，模块级惰性加载（首次调用读盘，
进程内缓存 + 锁保证并发安全，后续调用零 I/O）。纯 stdlib，零 LLM 调用、零网络。

路径解析优先级：环境变量 ``PERSONA_DEFS_PATH`` > 同目录 ``persona_defs.json``。

对外契约：
- ``list_personas() -> list[dict]``
    返回 ``[{"key", "name", "description"}]``；数据不可用返回空列表并 warning。
- ``get_persona(key) -> dict | None``
    返回人格完整定义的深拷贝；未知 key / 坏输入返回 None。
- ``render_persona_framework(key) -> dict | None``
    渲染供 Stage 2 拼进工具返回 / 提示词的框架字典：
    ``{name, instructions, scoring_weights, thresholds, analysis_rules,
    output_schema, checklist, disclaimer}``；未知人格返回 None。
- ``validate_persona_output(data, persona_key=None) -> dict``
    归一化 LLM 产出：confidence 钳制到 confidence_max（默认 0.9）、
    signal 非法改为「观望」、缺字段补默认值，逐项记入 violations。
    返回 ``{"ok": bool, "normalized": dict, "violations": list[str]}``。

所有接口对坏输入 / 坏数据一律优雅降级（返回 None / 空列表 / 带 violations
的归一化结果），绝不抛异常。
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

#: 人格定义文件路径覆盖用的环境变量名（契约，集成者依赖，勿改）。
ENV_PATH_KEY = "PERSONA_DEFS_PATH"

_DEFAULT_PATH = Path(__file__).parent / "persona_defs.json"

#: 信号枚举（契约，与 persona_defs.json 中 output_schema.signal 保持一致）。
SIGNAL_ENUM = ("买入", "增持", "观望", "减持", "回避")

#: 信号非法 / 缺失时的兜底档位。
DEFAULT_SIGNAL = "观望"

#: 置信度上限缺省值（人格 output_schema.confidence_max 可覆盖）。
DEFAULT_CONFIDENCE_MAX = 0.9

#: 置信度缺失 / 非法时的兜底值。
DEFAULT_CONFIDENCE = 0.5

#: 框架渲染固定免责声明（契约，集成者依赖，勿改）。
DISCLAIMER = "方法论框架仅供参考，不构成投资建议"

#: render_persona_framework 透传的字段（顺序即渲染字典键序）。
_FRAMEWORK_FIELDS = (
    "instructions",
    "scoring_weights",
    "thresholds",
    "analysis_rules",
    "output_schema",
    "checklist",
)

# 模块级懒加载缓存：None = 尚未读盘；锁保证并发首次加载只读一次盘。
_cache: dict | None = None
_cache_lock = threading.Lock()


def _resolve_path() -> Path:
    """解析人格定义文件路径：环境变量 PERSONA_DEFS_PATH 优先，缺省用同目录文件。"""
    env = os.environ.get(ENV_PATH_KEY)
    if isinstance(env, str) and env.strip():
        return Path(env.strip())
    return _DEFAULT_PATH


def _load() -> dict:
    """读盘并缓存人格定义；文件缺失 / 损坏 / 顶层非对象时返回空 dict，不抛异常。"""
    global _cache
    if _cache is not None:
        return _cache
    with _cache_lock:
        if _cache is not None:
            return _cache
        path = _resolve_path()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("人格定义加载失败（%s）：%s", path, exc)
            raw = {}
        if not isinstance(raw, dict):
            logger.warning("人格定义文件顶层不是 JSON 对象（%s），按空数据降级", path)
            raw = {}
        _cache = raw
        return _cache


def list_personas() -> list[dict]:
    """返回全部人格的 [{key, name, description}] 列表；数据不可用返回空列表。"""
    data = _load()
    if not data:
        return []
    out: list[dict] = []
    for key, persona in data.items():
        if not isinstance(key, str) or not isinstance(persona, dict):
            continue
        name = persona.get("name")
        description = persona.get("description")
        if not isinstance(name, str) or not isinstance(description, str):
            continue
        out.append({"key": key, "name": name, "description": description})
    return out


def get_persona(key: object) -> dict | None:
    """返回人格完整定义的深拷贝；未知 key / 坏输入 / 坏数据返回 None。"""
    if not isinstance(key, str) or not key.strip():
        return None
    persona = _load().get(key.strip())
    if not isinstance(persona, dict):
        return None
    return copy.deepcopy(persona)


def render_persona_framework(key: object) -> dict | None:
    """渲染人格方法论框架字典（含免责声明），供 Stage 2 拼进工具返回 / 提示词。

    返回 ``{name, instructions, scoring_weights, thresholds, analysis_rules,
    output_schema, checklist, disclaimer}``；未知人格或缺 name 返回 None。
    返回值为深拷贝，调用方可安全修改。
    """
    persona = get_persona(key)
    if persona is None:
        return None
    name = persona.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    framework: dict = {"name": name}
    for field in _FRAMEWORK_FIELDS:
        framework[field] = persona.get(field)
    framework["disclaimer"] = DISCLAIMER
    return framework


def _resolve_output_contract(persona_key: object) -> tuple[tuple[str, ...], float]:
    """从人格 output_schema 解析信号枚举与置信度上限；人格缺失 / 字段坏时回退缺省值。"""
    signals: tuple[str, ...] = SIGNAL_ENUM
    conf_max = DEFAULT_CONFIDENCE_MAX
    persona = get_persona(persona_key) if isinstance(persona_key, str) else None
    if persona is None:
        if persona_key is not None:
            logger.warning("validate_persona_output：未知人格 %r，回退缺省输出契约", persona_key)
        return signals, conf_max
    schema = persona.get("output_schema")
    if isinstance(schema, dict):
        raw_signals = schema.get("signal")
        if (
            isinstance(raw_signals, list)
            and raw_signals
            and all(isinstance(s, str) for s in raw_signals)
        ):
            signals = tuple(raw_signals)
        raw_max = schema.get("confidence_max")
        if isinstance(raw_max, (int, float)) and not isinstance(raw_max, bool) and raw_max > 0:
            conf_max = float(raw_max)
    return signals, conf_max


def validate_persona_output(data: object, persona_key: object = None) -> dict:
    """归一化 LLM 人格分析产出，返回 {ok, normalized, violations}。

    规则：
    - data 非 dict：按空对象归一化并记 violation；
    - signal 缺失：补默认值「观望」并记 violation；
    - signal 不在枚举内：改为「观望」并记 violation；
    - confidence 缺失 / 非数值：补默认值并记 violation；
    - confidence 超出 [0, confidence_max]：钳制到边界并记 violation。

    ``ok`` 为 True 当且仅当 violations 为空。不修改入参，normalized 为新对象。
    """
    signals, conf_max = _resolve_output_contract(persona_key)
    violations: list[str] = []

    if isinstance(data, dict):
        normalized = copy.deepcopy(data)
    else:
        normalized = {}
        violations.append("输出不是 JSON 对象，已按空结果归一化")

    signal = normalized.get("signal")
    if signal is None:
        normalized["signal"] = DEFAULT_SIGNAL
        violations.append("缺少 signal 字段，已补默认值「观望」")
    elif signal not in signals:
        violations.append(f"signal「{signal}」不在允许枚举 {list(signals)} 内，已改为「观望」")
        normalized["signal"] = DEFAULT_SIGNAL

    confidence = normalized.get("confidence")
    if confidence is None:
        normalized["confidence"] = DEFAULT_CONFIDENCE
        violations.append(f"缺少 confidence 字段，已补默认值 {DEFAULT_CONFIDENCE}")
    elif not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        normalized["confidence"] = DEFAULT_CONFIDENCE
        violations.append(f"confidence「{confidence}」不是数值，已补默认值 {DEFAULT_CONFIDENCE}")
    else:
        confidence = float(confidence)
        if confidence > conf_max:
            normalized["confidence"] = conf_max
            violations.append(f"confidence {confidence} 超过上限 {conf_max}，已钳制到 {conf_max}")
        elif confidence < 0:
            normalized["confidence"] = 0.0
            violations.append(f"confidence {confidence} 低于 0，已钳制到 0.0")
        else:
            normalized["confidence"] = confidence

    return {"ok": not violations, "normalized": normalized, "violations": violations}
