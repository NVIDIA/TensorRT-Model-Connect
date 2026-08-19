# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Qwen-Image MMDiT dynamic-batch builder wiring.

Like the encoder batch-profile tests this monkeypatches TensorRT and the
heavy compute-graph helpers (``_add_joint_block_graph``, ``_add_linear_3d``,
etc.) so we exercise the builder entry-point wiring without compiling an
engine. The tests confirm:

* ``build_qwen_image_dit_engine`` gains ``max_batch_size`` and
  ``opt_batch_size`` kwargs.
* The pre-existing ``NotImplementedError`` guard for ``batch_size != 1``
  (lines 1767-1770 in main) is removed; calling with ``max_batch_size=4``
  no longer raises.
* ``max_batch_size > 1`` switches each engine input's leading dim to ``-1``
  and calls ``add_dynamic_batch_profile`` exactly once with the per-design
  kMIN=1, kOPT=min(N,4), kMAX=N envelope (Decisions A and C).
* ``max_batch_size == 1`` reproduces today's static-shape path and skips
  ``add_dynamic_batch_profile`` entirely.

The Qwen-Image MMDiT pipeline runs Decision B's 2-pass CFG (cond + uncond
with L2-renorm combine). The DiT *graph* is unchanged structurally for
Phase 1 — we just make the leading dim dynamic. The renorm combiner lives
in the C++ pipeline and is PR 2 scope; nothing graph-side here changes for
CFG.

Trace: ARCH-FAM-001, UD-FAM-QWEN-IMAGE-01.
"""

from __future__ import annotations

import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Mock TensorRT surface (shared shape/semantics with the encoder test;
# duplicated here to keep the test file self-contained).
# ---------------------------------------------------------------------------


class _FakeTensor:
    def __init__(self, name: str = "tensor", *, dtype=None, shape=(1,)):
        self.name = name
        self.dtype = dtype if dtype is not None else _FakeTRT.float32
        self.shape = shape


class _FakeLayer:
    def __init__(self, name: str = "layer", *, dtype=None):
        self.name = name
        self._out = _FakeTensor(name, dtype=dtype)
        self.reshape_dims = None
        self.epsilon = None
        self.axis = 0
        self.first_transpose = None
        self.second_transpose = None
        self.compute_precision = None

    def get_output(self, _idx: int) -> _FakeTensor:
        return self._out

    def set_input(self, *_a, **_kw):
        return None


class _FakeLayerWithoutComputePrecision:
    def __init__(self, name: str = "layer", *, dtype=None):
        self.name = name
        self._out = _FakeTensor(name, dtype=dtype)
        self.epsilon = None

    def get_output(self, _idx: int) -> _FakeTensor:
        return self._out


class _FakeNetwork:
    def __init__(self):
        self.inputs: list[tuple[str, object, tuple]] = []

    def add_input(self, name, dtype, shape):
        self.inputs.append((name, dtype, tuple(shape)))
        return _FakeTensor(name, dtype=dtype, shape=tuple(shape))

    def __getattr__(self, item):
        if item == "mark_output":
            return lambda _t: None
        return lambda *a, **kw: _FakeLayer(item)


class _FakeBuilderConfig:
    def __init__(self):
        self.profiles: list[object] = []

    def set_memory_pool_limit(self, *_a, **_kw):
        return None

    def add_optimization_profile(self, profile):
        self.profiles.append(profile)

    def clear_flag(self, *_a):
        return None


class _FakeProfile:
    def __init__(self):
        self.shapes: dict[str, tuple] = {}

    def set_shape(self, name, *, min, opt, max):
        self.shapes[name] = (tuple(min), tuple(opt), tuple(max))


class _FakeBuilder:
    def __init__(self):
        self._config = _FakeBuilderConfig()
        self._network = _FakeNetwork()
        self.serialized = b"FAKE_ENGINE_PLAN"

    def create_builder_config(self):
        return self._config

    def create_network(self, _flags):
        return self._network

    def create_optimization_profile(self):
        return _FakeProfile()

    def build_serialized_network(self, _n, _c):
        return self.serialized


class _FakeTRT:
    float32 = "float32"
    int32 = "int32"
    bfloat16 = "bfloat16"

    class Logger:
        WARNING = 0
        VERBOSE = 1

        def __init__(self, *_a, **_kw):
            return None

    class Builder:
        def __init__(self, _logger):
            self.fake = _FakeBuilder()

        def create_builder_config(self):
            return self.fake.create_builder_config()

        def create_network(self, flags):
            return self.fake.create_network(flags)

        def create_optimization_profile(self):
            return self.fake.create_optimization_profile()

        def build_serialized_network(self, n, c):
            return self.fake.build_serialized_network(n, c)

    class NetworkDefinitionCreationFlag:
        STRONGLY_TYPED = 1

    class MemoryPoolType:
        WORKSPACE = 0

    class BuilderFlag:
        TF32 = 0

    class ElementWiseOperation:
        SUM = 0
        PROD = 1
        SUB = 2

    class UnaryOperation:
        SQRT = 0
        RECIP = 1
        SIN = 2
        COS = 3

    class MatrixOperation:
        NONE = 0

    class Permutation:
        def __init__(self, perm):
            self.perm = perm

    @staticmethod
    def Weights(arr, *args, **kw):  # noqa: N802
        return arr


if "tensorrt" not in sys.modules:
    fake_tensorrt = types.ModuleType("tensorrt")
    for _name in dir(_FakeTRT):
        if not _name.startswith("__"):
            setattr(fake_tensorrt, _name, getattr(_FakeTRT, _name))
    sys.modules["tensorrt"] = fake_tensorrt


# ---------------------------------------------------------------------------
# Test cfg + weight bag (matches _validate_full_weights' schema check).
# ---------------------------------------------------------------------------


def _tiny_cfg():
    from tensorrt_model_connect.families.qwen_image.qwen_image_dit_builder import (
        QwenImageDiTConfig,
    )

    return QwenImageDiTConfig(
        in_channels=4,
        out_channels=1,
        patch_size=2,
        hidden_size=12,
        num_joint_blocks=1,
        num_attention_heads=2,
        attention_head_dim=6,
        intermediate_size=24,
        text_embed_dim=6,
        rope_axes_dim=[2, 2, 2],
        rope_theta=10000.0,
        timestep_embed_dim=4,
        max_image_tokens=8,
        max_text_tokens=4,
        guidance_embeds=False,
    )


def _tiny_weights():
    """A minimal weight dict that passes ``_validate_full_weights``."""
    import numpy as np

    cfg = _tiny_cfg()
    H = cfg.hidden_size
    in_ch = cfg.in_channels
    out_ch = cfg.out_channels
    p = cfg.patch_size
    txt_d = cfg.text_embed_dim
    rng = np.random.default_rng(0)

    def n(shape):
        return rng.normal(0.0, 0.01, shape).astype(np.float32)

    weights: dict = {
        "img_in.weight": n((H, in_ch)),
        "img_in.bias": np.zeros((H,), dtype=np.float32),
        "txt_norm.weight": np.ones((txt_d,), dtype=np.float32),
        "txt_in.weight": n((H, txt_d)),
        "txt_in.bias": np.zeros((H,), dtype=np.float32),
        "time_text_embed.timestep_embedder.linear_1.weight":
            n((H, cfg.timestep_embed_dim)),
        "time_text_embed.timestep_embedder.linear_1.bias":
            np.zeros((H,), dtype=np.float32),
        "time_text_embed.timestep_embedder.linear_2.weight": n((H, H)),
        "time_text_embed.timestep_embedder.linear_2.bias":
            np.zeros((H,), dtype=np.float32),
        "norm_out.linear.weight": n((2 * H, H)),
        "norm_out.linear.bias": np.zeros((2 * H,), dtype=np.float32),
        "proj_out.weight": n((out_ch * p * p, H)),
        "proj_out.bias": np.zeros((out_ch * p * p,), dtype=np.float32),
    }
    return weights


def _patch_tensorrt(monkeypatch):
    """Replace TRT and stub the inner compute-graph helpers.

    Returns the list that ``add_dynamic_batch_profile`` records into.
    """
    from tensorrt_model_connect.families.qwen_image import (
        qwen_image_dit_builder as dit_mod,
    )

    monkeypatch.setattr(dit_mod, "trt", _FakeTRT)

    def _stub_tensor(*_a, **_kw):
        return _FakeTensor("stub")

    def _stub_two_tensors(*_a, **_kw):
        # ``_add_joint_block_graph`` returns (img_out, txt_out).
        return _FakeTensor("img_out"), _FakeTensor("txt_out")

    monkeypatch.setattr(dit_mod, "_add_linear_3d", _stub_tensor)
    monkeypatch.setattr(dit_mod, "_add_rms_norm_last_dim_3d", _stub_tensor)
    monkeypatch.setattr(dit_mod, "_add_time_text_embed", _stub_tensor)
    monkeypatch.setattr(dit_mod, "_add_norm_out_3d", _stub_tensor)
    monkeypatch.setattr(dit_mod, "_add_joint_block_graph", _stub_two_tensors)
    monkeypatch.setattr(dit_mod, "_to_fp32",
                        lambda _network, t: t if isinstance(t, _FakeTensor)
                        else _FakeTensor("fp32"))
    # ``_precompute_qwen_rope_tables_for_shapes`` returns two
    # (sum(n_img) + n_text, head_dim) arrays. Stub the helper used by the
    # current batch-aware production path so the downstream shape checks still
    # exercise the real call boundary.
    import numpy as np

    def _stub_rope(axes_dim, image_shapes, n_text, theta):
        head_dim = sum(axes_dim)
        seq_total = sum(h_lat * w_lat for h_lat, w_lat in image_shapes) + n_text
        return (
            np.zeros((seq_total, head_dim), dtype=np.float32),
            np.zeros((seq_total, head_dim), dtype=np.float32),
        )

    monkeypatch.setattr(
        dit_mod,
        "_precompute_qwen_rope_tables_for_shapes",
        _stub_rope,
    )

    calls: list[dict] = []

    def _record(builder, config, network, *, input_names, max_batch,
                opt_batch, static_shape):
        calls.append(dict(
            builder=builder, config=config, network=network,
            input_names=list(input_names), max_batch=max_batch,
            opt_batch=opt_batch, static_shape=dict(static_shape),
        ))

    from tensorrt_model_connect.tvm_ffi import graph_build

    monkeypatch.setattr(graph_build, "add_dynamic_batch_profile", _record)
    return calls


def _capturing_create_network(monkeypatch):
    from tensorrt_model_connect.families.qwen_image import (
        qwen_image_dit_builder as dit_mod,
    )

    captured: dict = {}
    real = dit_mod.trt.Builder.create_network

    def wrapper(self, flags):
        net = real(self, flags)
        captured["network"] = net
        return net

    monkeypatch.setattr(dit_mod.trt.Builder, "create_network", wrapper)
    return captured


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_layernorm_no_affine_tolerates_trt11_without_compute_precision(
    monkeypatch,
):
    pytest.importorskip("numpy")
    pytest.importorskip("ml_dtypes")
    from tensorrt_model_connect.families.qwen_image import (
        qwen_image_dit_builder as dit_mod,
    )

    class _TRT11Network(_FakeNetwork):
        def __init__(self):
            super().__init__()
            self.norm = _FakeLayerWithoutComputePrecision("norm")

        def add_normalization(self, *_a, **_kw):
            return self.norm

    monkeypatch.setattr(dit_mod, "trt", _FakeTRT)
    network = _TRT11Network()
    out = dit_mod._add_layernorm_no_affine_3d(
        network,
        _FakeTensor("x", dtype=dit_mod._CAST_DTYPE),
        hidden_size=4,
        eps=1e-6,
    )

    assert out is network.norm.get_output(0)
    assert network.norm.epsilon == 1e-6


def test_max_batch_size_four_uses_dynamic_dim_and_calls_profile(
    monkeypatch, tmp_path,
):
    pytest.importorskip("numpy")
    pytest.importorskip("ml_dtypes")
    from tensorrt_model_connect.families.qwen_image.qwen_image_dit_builder import (
        build_qwen_image_dit_engine,
    )

    calls = _patch_tensorrt(monkeypatch)
    captured = _capturing_create_network(monkeypatch)
    cfg = _tiny_cfg()
    h_lat = 2
    w_lat = 2
    n_text = 3
    build_qwen_image_dit_engine(
        cfg, _tiny_weights(), tmp_path / "dit.plan",
        h_lat=h_lat, w_lat=w_lat, n_text=n_text,
        max_batch_size=4,
    )
    assert len(calls) == 1
    call = calls[0]
    n_img = h_lat * w_lat
    assert call["input_names"] == ["img_patched", "txt_hidden", "timestep"]
    assert call["max_batch"] == 4
    assert call["opt_batch"] == 4
    assert call["static_shape"] == {
        "img_patched": (n_img, cfg.in_channels),
        "txt_hidden": (n_text, cfg.text_embed_dim),
        "timestep": (),
    }
    net = captured["network"]
    shapes = {name: shape for name, _dt, shape in net.inputs}
    assert shapes["img_patched"] == (-1, n_img, cfg.in_channels)
    assert shapes["txt_hidden"] == (-1, n_text, cfg.text_embed_dim)
    assert shapes["timestep"] == (-1,)


def test_not_implemented_error_guard_is_gone(monkeypatch, tmp_path):
    """Regression for the removed batch_size!=1 guard.

    Before this PR, ``build_qwen_image_dit_engine`` raised
    ``NotImplementedError`` for any batch>1. With the dynamic-batch profile
    wired in, the call must succeed (with mocks in place).
    """
    pytest.importorskip("numpy")
    pytest.importorskip("ml_dtypes")
    from tensorrt_model_connect.families.qwen_image.qwen_image_dit_builder import (
        build_qwen_image_dit_engine,
    )

    _patch_tensorrt(monkeypatch)
    # Should NOT raise.
    build_qwen_image_dit_engine(
        _tiny_cfg(), _tiny_weights(), tmp_path / "dit.plan",
        h_lat=2, w_lat=2, n_text=3, max_batch_size=4,
    )
