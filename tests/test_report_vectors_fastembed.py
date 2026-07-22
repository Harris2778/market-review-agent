"""agent/report_vectors.py FastembedEmbedder 与后端回退链测试。

覆盖范围：
1. FastembedEmbedder：惰性导入 fastembed（sys.modules 注入假模块，零网络）；
   name/dim 契约；embed 输出 float 列表；REPORT_FASTEMBED_MODEL /
   REPORT_FASTEMBED_CACHE 环境变量生效（cache_dir 透传）。
2. _default_embedder 后端链：auto 默认 ST 优先；ST 失败回退 fastembed；
   REPORT_EMBED_BACKEND=st / fastembed 强制单后端；全部失败返回 None；
   未知 backend 值按 auto 处理。

规则（与项目其他测试一致）：全 mock 零网络，绝不构造真实模型。
"""

import sys
import types

import numpy as np
import pytest

import agent.report_vectors as rv


# ── 假 fastembed 模块 ──


class _FakeTextEmbedding:
    """记录构造参数，embed 返回确定性 512 维向量。"""

    instances = []

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        _FakeTextEmbedding.instances.append(self)

    def embed(self, texts):
        for i, _t in enumerate(texts):
            yield np.full(512, 0.5 + i, dtype=np.float32)


@pytest.fixture(autouse=True)
def _clear_embedder_cache():
    """每个用例前后清单例缓存，避免跨用例串实例。"""
    rv._DEFAULT_EMBEDDER_CACHE.clear()
    yield
    rv._DEFAULT_EMBEDDER_CACHE.clear()


@pytest.fixture()
def fake_fastembed(monkeypatch):
    """注入假 fastembed 模块，返回模块与记录器。"""
    _FakeTextEmbedding.instances.clear()
    mod = types.ModuleType("fastembed")
    mod.TextEmbedding = _FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", mod)
    return mod


@pytest.fixture()
def no_st(monkeypatch):
    """让 sentence_transformers 导入失败（模拟生产容器未安装）。"""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)


# ── FastembedEmbedder ──


def test_fastembed_name_dim_and_embed(fake_fastembed):
    e = rv.FastembedEmbedder()
    assert e.name == "fastembed:BAAI/bge-small-zh-v1.5"
    assert e.dim == 512
    out = e.embed(["甲", "乙"])
    assert len(out) == 2
    assert all(isinstance(x, float) for x in out[0])
    assert len(out[0]) == 512
    # 默认不传 cache_dir
    assert _FakeTextEmbedding.instances[0].kwargs == {}


def test_fastembed_env_overrides(fake_fastembed, monkeypatch):
    monkeypatch.setenv("REPORT_FASTEMBED_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("REPORT_FASTEMBED_CACHE", "/opt/fx")
    e = rv.FastembedEmbedder()
    inst = _FakeTextEmbedding.instances[0]
    assert inst.model == "BAAI/bge-m3"
    assert inst.kwargs == {"cache_dir": "/opt/fx"}
    assert e.name == "fastembed:BAAI/bge-m3"


def test_fastembed_offline_mode(fake_fastembed, monkeypatch):
    """REPORT_FASTEMBED_OFFLINE=1 → local_files_only=True 透传（生产零网络）。"""
    monkeypatch.setenv("REPORT_FASTEMBED_CACHE", "/opt/fx")
    monkeypatch.setenv("REPORT_FASTEMBED_OFFLINE", "1")
    rv.FastembedEmbedder()
    inst = _FakeTextEmbedding.instances[0]
    assert inst.kwargs == {"cache_dir": "/opt/fx", "local_files_only": True}


def test_fastembed_offline_default_off(fake_fastembed, monkeypatch):
    monkeypatch.delenv("REPORT_FASTEMBED_OFFLINE", raising=False)
    rv.FastembedEmbedder()
    assert "local_files_only" not in _FakeTextEmbedding.instances[0].kwargs


def test_fastembed_import_error_propagates(monkeypatch):
    """未安装 fastembed 时构造抛 ImportError（由回退链捕获）。"""
    monkeypatch.setitem(sys.modules, "fastembed", None)
    with pytest.raises(ImportError):
        rv.FastembedEmbedder()


# ── _default_embedder 后端链 ──


def test_default_auto_prefers_st(monkeypatch):
    sentinel = object()
    monkeypatch.delenv("REPORT_EMBED_BACKEND", raising=False)
    monkeypatch.setattr(rv, "BgeEmbedder", lambda: sentinel)
    assert rv._default_embedder() is sentinel


def test_default_auto_falls_back_to_fastembed(monkeypatch, fake_fastembed, no_st):
    monkeypatch.delenv("REPORT_EMBED_BACKEND", raising=False)
    e = rv._default_embedder()
    assert isinstance(e, rv.FastembedEmbedder)
    assert e.dim == 512


def test_default_backend_forced_fastembed(monkeypatch, fake_fastembed):
    monkeypatch.setenv("REPORT_EMBED_BACKEND", "fastembed")
    # ST 可用也不应被选中
    monkeypatch.setattr(rv, "BgeEmbedder", lambda: pytest.fail("不应构造 ST"))
    e = rv._default_embedder()
    assert isinstance(e, rv.FastembedEmbedder)


def test_default_backend_forced_st_no_fallback(monkeypatch, fake_fastembed, no_st):
    """强制 st 且 ST 不可用时直接 None，不回退 fastembed。"""
    monkeypatch.setenv("REPORT_EMBED_BACKEND", "st")
    assert rv._default_embedder() is None


def test_default_all_backends_fail_returns_none(monkeypatch, no_st):
    monkeypatch.delenv("REPORT_EMBED_BACKEND", raising=False)
    monkeypatch.setitem(sys.modules, "fastembed", None)
    assert rv._default_embedder() is None


def test_default_unknown_backend_treated_as_auto(monkeypatch, fake_fastembed, no_st):
    monkeypatch.setenv("REPORT_EMBED_BACKEND", "weird")
    e = rv._default_embedder()
    assert isinstance(e, rv.FastembedEmbedder)


def test_default_embedder_singleton_cached(monkeypatch, fake_fastembed, no_st):
    """构造成功后按 backend 缓存：第二次调用不再重建模型。"""
    monkeypatch.setenv("REPORT_EMBED_BACKEND", "fastembed")
    e1 = rv._default_embedder()
    e2 = rv._default_embedder()
    assert e1 is e2
    assert len(_FakeTextEmbedding.instances) == 1


def test_default_embedder_failure_not_cached(monkeypatch, no_st):
    """构造失败不缓存：依赖恢复后再次调用可自愈。"""
    monkeypatch.setenv("REPORT_EMBED_BACKEND", "fastembed")
    monkeypatch.setitem(sys.modules, "fastembed", None)
    assert rv._default_embedder() is None
    assert "fastembed" not in rv._DEFAULT_EMBEDDER_CACHE
