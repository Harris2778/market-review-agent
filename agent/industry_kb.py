"""行业知识库（第七波）— 申万一级行业精炼档案。

数据存于同目录 ``industry_kb_data.json``，模块级懒加载（首次调用时读盘，
进程内缓存，后续调用零 I/O）。纯 stdlib，无网络、无第三方依赖。

对外契约：
- ``get_industry_profile(sector) -> dict | None``
    返回 ``{"chain": str, "drivers": [...], "indicators": [...], "leaders": [...]}``
    的拷贝；精确匹配 + 别名表容错 + 保守子串兜底；缺行业 / 缺字段返回 None。
- ``format_kb_block(sector) -> str | None``
    格式化为 ≤400 字的 prompt 注入块，固定头部
    「【六、行业知识库（背景知识，数据以数据块为准）】」；未知行业返回 None。
- ``list_industries() -> list[str]``
    返回全部行业名（数据文件键序）；数据不可用返回空列表。

所有接口对坏输入 / 坏数据一律静默返回 None（或空列表），不抛异常。
"""

from __future__ import annotations

import json
from pathlib import Path

_DATA_PATH = Path(__file__).with_name("industry_kb_data.json")

_REQUIRED_LIST_FIELDS = ("drivers", "indicators", "leaders")

#: 别名表（≤10 条）：口语 / 细分叫法 → 申万一级行业名。
#: 能被保守子串兜底覆盖的叫法（如「军工」「地产」「医药」）不占用名额。
_ALIASES = {
    "半导体": "电子",
    "芯片": "电子",
    "白酒": "食品饮料",
    "光伏": "电气设备",
    "新能源": "电气设备",
    "券商": "非银金融",
    "保险": "非银金融",
    "医疗": "医药生物",
    "养殖": "农林牧渔",
    "水泥": "建筑材料",
}

#: prompt 注入块固定头部（契约，集成者依赖，勿改）。
KB_HEADER = "【六、行业知识库（背景知识，数据以数据块为准）】"

#: prompt 注入块最大字符数（契约）。
FORMAT_MAX_LEN = 400

# 模块级懒加载缓存：None = 尚未读盘。
_cache: dict | None = None


def _load() -> dict:
    """读盘并缓存行业数据；文件缺失 / 损坏时返回空 dict，不抛异常。"""
    global _cache
    if _cache is None:
        try:
            raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        _cache = raw if isinstance(raw, dict) else {}
    return _cache


def _normalize(sector: object) -> str | None:
    """去空白并剥掉「行业」「板块」尾巴；非字符串 / 空串返回 None。"""
    if not isinstance(sector, str):
        return None
    name = sector.strip()
    for suffix in ("行业", "板块"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name or None


def _resolve_name(sector: object) -> str | None:
    """把用户输入解析为数据文件里的行业名；解析失败返回 None。

    顺序：精确命中 → 别名表 → 保守子串兜底（输入≥2 字，双向包含）。
    """
    name = _normalize(sector)
    if not name:
        return None
    data = _load()
    if name in data:
        return name
    alias_target = _ALIASES.get(name)
    if alias_target is not None and alias_target in data:
        return alias_target
    if len(name) >= 2:
        for key in data:
            if isinstance(key, str) and (name in key or key in name):
                return key
    return None


def _valid_profile(profile: object) -> bool:
    """校验档案结构完整性：chain 非空字符串，三个列表字段非空且元素为非空字符串。"""
    if not isinstance(profile, dict):
        return False
    chain = profile.get("chain")
    if not isinstance(chain, str) or not chain.strip():
        return False
    for field in _REQUIRED_LIST_FIELDS:
        value = profile.get(field)
        if not isinstance(value, list) or not value:
            return False
        if not all(isinstance(item, str) and item.strip() for item in value):
            return False
    return True


def get_industry_profile(sector: object) -> dict | None:
    """返回行业档案拷贝（chain / drivers / indicators / leaders），缺行业或缺字段返回 None。"""
    canonical = _resolve_name(sector)
    if canonical is None:
        return None
    profile = _load().get(canonical)
    if not _valid_profile(profile):
        return None
    return {
        "chain": profile["chain"].strip(),
        "drivers": [item.strip() for item in profile["drivers"]],
        "indicators": [item.strip() for item in profile["indicators"]],
        "leaders": [item.strip() for item in profile["leaders"]],
    }


def format_kb_block(sector: object) -> str | None:
    """格式化为 ≤400 字的 prompt 注入块（固定头部 KB_HEADER）；未知行业返回 None。"""
    canonical = _resolve_name(sector)
    if canonical is None:
        return None
    profile = get_industry_profile(canonical)
    if profile is None:
        return None
    lines = [
        KB_HEADER,
        f"行业：{canonical}（申万一级）",
        f"产业链：{profile['chain']}",
        f"核心驱动：{'、'.join(profile['drivers'])}",
        f"关键指标：{'、'.join(profile['indicators'])}",
        f"代表公司：{'、'.join(profile['leaders'])}",
    ]
    block = "\n".join(lines)
    if len(block) > FORMAT_MAX_LEN:  # 安全兜底：当前档案远达不到上限
        block = block[: FORMAT_MAX_LEN - 1].rstrip("、，,。；;：: \n") + "…"
    return block


def list_industries() -> list[str]:
    """返回全部行业名（数据文件键序）；数据不可用返回空列表。"""
    return [key for key in _load() if isinstance(key, str)]
