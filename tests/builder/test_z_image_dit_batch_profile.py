"""Unit tests for the Z-Image DiT dynamic-batch profile.

Verifies the contract added in PR 1 of the diffusion batch-inference series:

* ``max_batch_size=1`` (the default) reproduces today's static-shape DiT
  build byte-for-byte — ``hidden_states`` / ``encoder_hidden_states`` keep
  their 2-D ``(S, D)`` shapes, ``timestep_embedding`` keeps its singleton
  ``(1, adaln_embed_dim)`` shape, and :func:`add_dynamic_batch_profile`
  is not invoked.
* ``max_batch_size > 1`` makes the leading dim of ``hidden_states``,
  ``encoder_hidden_states``, and ``timestep_embedding`` runtime-dynamic
  (``-1``) and attaches exactly one dynamic-batch profile with the
  expected ``input_names``, ``max_batch``, ``opt_batch``, and
  ``static_shape``. RoPE caches stay non-batched.

TRT and graph_ops are monkeypatched so we never compile a real engine.
(Reference: design doc Decisions A and C — kMIN=1, kOPT=min(max,4),
kMAX=max; DiT default cap = 4.)
"""

from __future__ import annotations

import types

import numpy as np
import pytest

try:
    from tensorrt_model_connect.families.z_image import z_image_dit_builder
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "tensorrt_model_connect requires tensorrt", allow_module_level=True)


# ---------------------------------------------------------------------------
# Reusable lightweight TRT/graph_ops fakes (mirrors the encoder test).
# ---------------------------------------------------------------------------


class _Tensor:
    def __init__(self, name: str = "tensor", shape: tuple = ()):
        self.name = name
        self.shape = shape
        self.dtype = None


class _Layer:
    def __init__(self, *_args, **_kwargs):
        self._out = _Tensor()

    def get_output(self, _idx: int = 0):
        return self._out

    def set_input(self, _idx: int, _tensor):
        # Used by dynamic-batch slice ops to bind a runtime-resolved shape.
        pass

    def __setattr__(self, name: str, value):
        object.__setattr__(self, name, value)


class _Network:
    def __init__(self):
        self.inputs: list[tuple[str, object, tuple]] = []
        self.outputs: list[_Tensor] = []

    def add_input(self, name: str, dtype, shape):
        self.inputs.append((name, dtype, tuple(shape)))
        return _Tensor(name=name, shape=tuple(shape))

    def mark_output(self, tensor):
        self.outputs.append(tensor)

    def __getattr__(self, attr: str):
        if attr.startswith("add_"):
            def _factory(*_args, **_kwargs):
                return _Layer()
            return _factory
        raise AttributeError(attr)


class _Config:
    def __init__(self):
        self.profiles: list = []

    def set_memory_pool_limit(self, *_args, **_kwargs):
        pass

    def add_optimization_profile(self, profile):
        self.profiles.append(profile)


class _Profile:
    def __init__(self):
        self.shapes: dict = {}

    def set_shape(self, name, *, min, opt, max):
        self.shapes[name] = (tuple(min), tuple(opt), tuple(max))


class _Builder:
    def __init__(self, *_args, **_kwargs):
        self._network = _Network()
        self._config = _Config()
        self._profile = _Profile()

    def create_network(self, _flag):
        return self._network

    def create_builder_config(self):
        return self._config

    def create_optimization_profile(self):
        return self._profile

    def build_serialized_network(self, _network, _config):
        return b"z-image-dit-plan"


class _FakeTRT(types.SimpleNamespace):
    class _Dtype:
        def __init__(self, name):
            self.name = name

    int32 = _Dtype("int32")
    float32 = _Dtype("float32")
    Builder = _Builder

    @staticmethod
    def Permutation(perm):
        return tuple(perm)

    class Logger:
        def __init__(self, *_a, **_kw):
            pass
        WARNING = 1
        VERBOSE = 2

    class MemoryPoolType:
        WORKSPACE = "workspace"

    class NetworkDefinitionCreationFlag:
        STRONGLY_TYPED = 0
        EXPLICIT_BATCH = 1

    class ActivationType:
        SIGMOID = "sigmoid"
        TANH = "tanh"

    class ElementWiseOperation:
        SUM = "sum"
        PROD = "prod"
        SUB = "sub"
        MAX = "max"

    class ReduceOperation:
        AVG = "avg"
        SUM = "sum"

    class UnaryOperation:
        SQRT = "sqrt"
        RECIP = "recip"


def _make_tensor(*_a, **_kw) -> _Tensor:
    return _Tensor()


def _patch_graph_ops(monkeypatch):
    """Replace heavy graph_ops calls with tensor-returning stubs."""
    import tensorrt_model_connect.graph_ops as gops
    for name in (
        "add_constant",
        "add_matmul_rhs_constant",
        "add_bias_sum",
        "add_rms_norm",
        "add_rms_norm_last_dim",
        "add_rms_norm_per_head",
        "add_rms_norm_per_head_batched",
        "add_apply_rope_native",
        "add_apply_rope_native_sequence",
        "add_apply_rope_native_from_full_cache",
        "add_attention_core",
        "add_attention_from_rows",
        "validate_native_rope_dim",
        "reshape_rows_to_heads_4d",
        "reshape_heads_4d_to_rows",
    ):
        if hasattr(gops, name):
            if name == "validate_native_rope_dim":
                monkeypatch.setattr(gops, name, lambda v, **kw: v)
            else:
                monkeypatch.setattr(gops, name, _make_tensor)


# ---------------------------------------------------------------------------
# Tiny synthetic DiT weights — only the keys looked up by the builder.
# ---------------------------------------------------------------------------


def _tiny_dit_weights(
    *,
    dim: int = 8,
    head_dim: int = 2,
    num_heads: int = 4,
    ffn_dim: int = 16,
    adaln_embed_dim: int = 6,
    num_layers: int = 1,
    num_refiner_layers: int = 1,
    out_channels: int = 8,
) -> dict[str, np.ndarray]:
    z = np.zeros

    def _block(prefix: str, *, has_adaln: bool) -> dict[str, np.ndarray]:
        w: dict[str, np.ndarray] = {
            f"{prefix}.to_q": z((dim, dim), dtype=np.float32),
            f"{prefix}.to_k": z((dim, dim), dtype=np.float32),
            f"{prefix}.to_v": z((dim, dim), dtype=np.float32),
            f"{prefix}.to_out": z((dim, dim), dtype=np.float32),
            f"{prefix}.norm_q": z((head_dim,), dtype=np.float32),
            f"{prefix}.norm_k": z((head_dim,), dtype=np.float32),
            f"{prefix}.attn_norm1": z((dim,), dtype=np.float32),
            f"{prefix}.attn_norm2": z((dim,), dtype=np.float32),
            f"{prefix}.ff_w1": z((dim, ffn_dim), dtype=np.float32),
            f"{prefix}.ff_w2": z((ffn_dim, dim), dtype=np.float32),
            f"{prefix}.ff_w3": z((dim, ffn_dim), dtype=np.float32),
            f"{prefix}.ffn_norm1": z((dim,), dtype=np.float32),
            f"{prefix}.ffn_norm2": z((dim,), dtype=np.float32),
        }
        if has_adaln:
            w[f"{prefix}.adaln.weight"] = z(
                (adaln_embed_dim, 4 * dim), dtype=np.float32)
            w[f"{prefix}.adaln.bias"] = z((4 * dim,), dtype=np.float32)
        return w

    weights: dict[str, np.ndarray] = {}
    for i in range(num_layers):
        weights.update(_block(f"main.{i}", has_adaln=True))
    for i in range(num_refiner_layers):
        weights.update(_block(f"noise_refiner.{i}", has_adaln=True))
        weights.update(_block(f"context_refiner.{i}", has_adaln=False))

    weights["final_adaLN.weight"] = z((adaln_embed_dim, dim), dtype=np.float32)
    weights["final_adaLN.bias"] = z((dim,), dtype=np.float32)
    weights["final_linear.weight"] = z((dim, out_channels), dtype=np.float32)
    weights["final_linear.bias"] = z((out_channels,), dtype=np.float32)
    return weights


def _call_builder(monkeypatch, *, max_batch_size: int = 1, opt_batch_size=None):
    monkeypatch.setattr(z_image_dit_builder, "trt", _FakeTRT)
    _patch_graph_ops(monkeypatch)

    profile_calls: list[dict] = []

    def _record_profile(builder, config, network, *, input_names,
                        max_batch, opt_batch, static_shape):
        profile_calls.append({
            "input_names": list(input_names),
            "max_batch": max_batch,
            "opt_batch": opt_batch,
            "static_shape": dict(static_shape),
        })

    monkeypatch.setattr(
        z_image_dit_builder, "add_dynamic_batch_profile", _record_profile)

    holder: dict = {}
    real_builder_cls = _FakeTRT.Builder

    def _builder_factory(*args, **kwargs):
        inst = real_builder_cls(*args, **kwargs)
        holder["builder"] = inst
        return inst

    monkeypatch.setattr(_FakeTRT, "Builder", _builder_factory)

    plan = z_image_dit_builder.build_z_image_dit_engine(
        _tiny_dit_weights(),
        dim=8,
        num_heads=4,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=16,
        num_patches=12,
        text_seq_len=5,
        head_dim=2,
        adaln_embed_dim=6,
        eps=1e-5,
        verbose=False,
        max_batch_size=max_batch_size,
        opt_batch_size=opt_batch_size,
    )
    assert plan == b"z-image-dit-plan"
    return holder["builder"]._network, profile_calls


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_dynamic_batch_adds_leading_minus_one_to_all_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``max_batch_size > 1`` makes every engine input dynamic-batched and
    attaches exactly one dynamic-batch profile.

    Z-Image's per-sample RoPE depends on the caption's padded length, so the
    runtime supplies a distinct RoPE per sample stacked along the batch
    axis — ``rotary_cos`` / ``rotary_sin`` grow a leading ``-1`` alongside
    ``hidden_states`` / ``encoder_hidden_states`` / ``timestep_embedding``.
    """
    network, profile_calls = _call_builder(monkeypatch, max_batch_size=4)

    inputs = {name: shape for name, _dtype, shape in network.inputs}
    assert inputs["hidden_states"] == (-1, 12, 8)
    assert inputs["encoder_hidden_states"] == (-1, 5, 8)
    assert inputs["timestep_embedding"] == (-1, 6)
    assert inputs["rotary_cos"] == (-1, 17, 2)
    assert inputs["rotary_sin"] == (-1, 17, 2)

    assert len(profile_calls) == 1
    call = profile_calls[0]
    assert sorted(call["input_names"]) == [
        "encoder_hidden_states", "hidden_states", "rotary_cos",
        "rotary_sin", "timestep_embedding"]
    assert call["max_batch"] == 4
    assert call["opt_batch"] == 4
    assert call["static_shape"] == {
        "hidden_states": (12, 8),
        "encoder_hidden_states": (5, 8),
        "timestep_embedding": (6,),
        "rotary_cos": (17, 2),
        "rotary_sin": (17, 2),
    }
