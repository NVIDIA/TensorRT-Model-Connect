# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Z-Image DiT attention, batching, and precision contracts."""

from __future__ import annotations

import types

import numpy as np
import pytest

from families.z_image import graph_ops
from families.z_image import z_image_dit_builder as dit


class _Tensor:
    def __init__(self, name: str = "tensor", shape: tuple = ()):
        self.name = name
        self.shape = shape
        self.dtype = None


class _Layer:
    def __init__(self, *_args, **_kwargs):
        self._out = _Tensor()

    def get_output(self, _index: int = 0):
        return self._out

    def set_input(self, _index: int, _tensor):
        pass


class _Network:
    def __init__(self):
        self.inputs: list[tuple[str, object, tuple]] = []
        self.outputs: list[_Tensor] = []

    def add_input(self, name: str, dtype, shape):
        self.inputs.append((name, dtype, tuple(shape)))
        return _Tensor(name=name, shape=tuple(shape))

    def mark_output(self, tensor):
        self.outputs.append(tensor)

    def add_cast(self, _tensor, dtype):
        layer = _Layer()
        layer.get_output(0).dtype = dtype
        return layer

    def __getattr__(self, attribute: str):
        if attribute.startswith("add_"):
            return lambda *_args, **_kwargs: _Layer()
        raise AttributeError(attribute)


class _Config:
    def __init__(self):
        self.profiles: list = []
        self.builder_optimization_level = 0

    def set_memory_pool_limit(self, *_args, **_kwargs):
        pass

    def add_optimization_profile(self, profile):
        self.profiles.append(profile)

    def clear_flag(self, *_args):
        pass


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
    class Dtype:
        def __init__(self, name):
            self.name = name

    int32 = Dtype("int32")
    float32 = Dtype("float32")
    float16 = Dtype("float16")
    Builder = _Builder

    @staticmethod
    def Permutation(permutation):
        return tuple(permutation)

    class Logger:
        WARNING = 1
        VERBOSE = 2

        def __init__(self, *_args, **_kwargs):
            pass

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


def _make_tensor(*_args, **_kwargs) -> _Tensor:
    return _Tensor()


def _patch_graph_ops(monkeypatch):
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
        if hasattr(graph_ops, name):
            replacement = (
                (lambda value, **_kwargs: value)
                if name == "validate_native_rope_dim"
                else _make_tensor
            )
            monkeypatch.setattr(graph_ops, name, replacement)


def _tiny_dit_weights(
    *,
    dim: int = 8,
    head_dim: int = 2,
    ffn_dim: int = 16,
    adaln_embed_dim: int = 6,
    num_layers: int = 1,
    num_refiner_layers: int = 1,
    out_channels: int = 8,
) -> dict[str, np.ndarray]:
    zeros = np.zeros

    def block(prefix: str, *, has_adaln: bool) -> dict[str, np.ndarray]:
        weights = {
            f"{prefix}.to_q": zeros((dim, dim), dtype=np.float32),
            f"{prefix}.to_k": zeros((dim, dim), dtype=np.float32),
            f"{prefix}.to_v": zeros((dim, dim), dtype=np.float32),
            f"{prefix}.to_out": zeros((dim, dim), dtype=np.float32),
            f"{prefix}.norm_q": zeros((head_dim,), dtype=np.float32),
            f"{prefix}.norm_k": zeros((head_dim,), dtype=np.float32),
            f"{prefix}.attn_norm1": zeros((dim,), dtype=np.float32),
            f"{prefix}.attn_norm2": zeros((dim,), dtype=np.float32),
            f"{prefix}.ff_w1": zeros((dim, ffn_dim), dtype=np.float32),
            f"{prefix}.ff_w2": zeros((ffn_dim, dim), dtype=np.float32),
            f"{prefix}.ff_w3": zeros((dim, ffn_dim), dtype=np.float32),
            f"{prefix}.ffn_norm1": zeros((dim,), dtype=np.float32),
            f"{prefix}.ffn_norm2": zeros((dim,), dtype=np.float32),
        }
        if has_adaln:
            weights[f"{prefix}.adaln.weight"] = zeros((adaln_embed_dim, 4 * dim), dtype=np.float32)
            weights[f"{prefix}.adaln.bias"] = zeros((4 * dim,), dtype=np.float32)
        return weights

    weights: dict[str, np.ndarray] = {}
    for index in range(num_layers):
        weights.update(block(f"main.{index}", has_adaln=True))
    for index in range(num_refiner_layers):
        weights.update(block(f"noise_refiner.{index}", has_adaln=True))
        weights.update(block(f"context_refiner.{index}", has_adaln=False))
    weights["final_adaLN.weight"] = zeros((adaln_embed_dim, dim), dtype=np.float32)
    weights["final_adaLN.bias"] = zeros((dim,), dtype=np.float32)
    weights["final_linear.weight"] = zeros((dim, out_channels), dtype=np.float32)
    weights["final_linear.bias"] = zeros((out_channels,), dtype=np.float32)
    return weights


def _call_builder(
    monkeypatch,
    *,
    max_batch_size: int = 1,
    precision: str = "fp32",
    fp32_layers: tuple[int, ...] = (),
):
    monkeypatch.setattr(dit, "trt", _FakeTRT)
    _patch_graph_ops(monkeypatch)
    profile_calls: list[dict] = []

    def record_profile(builder, config, *, input_names, max_batch, opt_batch, static_shape):
        del builder, config
        profile_calls.append(
            {
                "input_names": list(input_names),
                "max_batch": max_batch,
                "opt_batch": opt_batch,
                "static_shape": dict(static_shape),
            }
        )

    monkeypatch.setattr(dit, "add_dynamic_batch_profile", record_profile)
    holder = {}
    real_builder = _FakeTRT.Builder

    def builder_factory(*args, **kwargs):
        instance = real_builder(*args, **kwargs)
        holder["builder"] = instance
        return instance

    monkeypatch.setattr(_FakeTRT, "Builder", builder_factory)
    plan = dit.build_z_image_dit_engine(
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
        precision=precision,
        fp32_layers=fp32_layers,
        verbose=False,
        max_batch_size=max_batch_size,
    )
    assert plan == b"z-image-dit-plan"
    return holder["builder"]._network, profile_calls


def test_static_fp16_attention_uses_native_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def attention(*_args, **kwargs):
        calls.append(kwargs)
        return _Tensor()

    monkeypatch.setattr(graph_ops, "add_attention_from_rows", attention)
    network = _Network()
    query = _Tensor()
    query.dtype = dit.trt.float16
    mask = _Tensor()
    mask.dtype = dit.trt.float32
    dit._multi_head_attention(
        network,
        query,
        _Tensor(),
        _Tensor(),
        num_heads=2,
        head_dim=4,
        q_seq=8,
        kv_seq=8,
        scale_t=0.5,
        mask=mask,
        dtype=np.float16,
    )

    assert calls[0]["explicit_attention"] is False
    assert calls[0]["fp32_accumulation"] is False
    assert calls[0]["mask"].dtype == dit.trt.float16
    assert calls[0]["scale"] == 0.5


def test_static_dit_accepts_runtime_caption_attention_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    network, profile_calls = _call_builder(monkeypatch, max_batch_size=1)
    inputs = {name: shape for name, _dtype, shape in network.inputs}
    assert inputs["attention_mask"] == (17,)
    assert profile_calls == []


def test_dynamic_batch_adds_leading_minus_one_to_all_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network, profile_calls = _call_builder(monkeypatch, max_batch_size=4)
    inputs = {name: shape for name, _dtype, shape in network.inputs}
    assert inputs == {
        "hidden_states": (-1, 12, 8),
        "encoder_hidden_states": (-1, 5, 8),
        "timestep_embedding": (-1, 6),
        "rotary_cos": (-1, 17, 2),
        "rotary_sin": (-1, 17, 2),
        "attention_mask": (-1, 17),
    }
    assert len(profile_calls) == 1
    call = profile_calls[0]
    assert sorted(call["input_names"]) == [
        "attention_mask",
        "encoder_hidden_states",
        "hidden_states",
        "rotary_cos",
        "rotary_sin",
        "timestep_embedding",
    ]
    assert call["max_batch"] == 4
    assert call["opt_batch"] == 4
    assert call["static_shape"] == {
        "hidden_states": (12, 8),
        "encoder_hidden_states": (5, 8),
        "timestep_embedding": (6,),
        "rotary_cos": (17, 2),
        "rotary_sin": (17, 2),
        "attention_mask": (17,),
    }


def test_static_fp16_accepts_selective_fp32_dit_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    network, profile_calls = _call_builder(
        monkeypatch,
        precision="fp16",
        fp32_layers=(0, 2, 3),
    )
    assert profile_calls == []
    assert [name for name, _dtype, _shape in network.inputs] == [
        "hidden_states",
        "encoder_hidden_states",
        "timestep_embedding",
        "rotary_cos",
        "rotary_sin",
        "attention_mask",
    ]


def test_sandwich_prescale_params_fp16_scales_and_swaps_eps() -> None:
    weights = {
        "b.to_out": np.ones((4, 4), dtype=np.float32),
        "b.ff_w3": np.full((4, 4), 2.0, dtype=np.float32),
    }
    eps_t, sandwich_eps_t = object(), object()
    w_out, w_ff3, norm_eps = dit._sandwich_prescale_params(
        weights, "b", eps_t, sandwich_eps_t, np.float16
    )
    alpha = dit._SANDWICH_PRESCALE
    np.testing.assert_allclose(w_out, np.full((4, 4), alpha))
    np.testing.assert_allclose(w_ff3, np.full((4, 4), 2.0 * alpha))
    assert norm_eps is sandwich_eps_t

    w_out, w_ff3, norm_eps = dit._sandwich_prescale_params(
        weights, "b", eps_t, sandwich_eps_t, np.float32
    )
    assert w_out is weights["b.to_out"]
    assert w_ff3 is weights["b.ff_w3"]
    assert norm_eps is eps_t
    w_out, _w_ff3, norm_eps = dit._sandwich_prescale_params(weights, "b", eps_t, None, np.float16)
    assert w_out is weights["b.to_out"]
    assert norm_eps is eps_t


def test_sandwich_prescale_epsilon_identity() -> None:
    rng = np.random.default_rng(3)
    values = rng.standard_normal((5, 64)).astype(np.float64) * 37.0
    gamma = rng.standard_normal(64)
    epsilon = 1e-5
    alpha = dit._SANDWICH_PRESCALE

    def rms_norm(array, eps):
        return array / np.sqrt((array**2).mean(axis=-1, keepdims=True) + eps) * gamma

    np.testing.assert_allclose(
        rms_norm(values, epsilon),
        rms_norm(alpha * values, epsilon * alpha * alpha),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    ("tp_size", "max_batch_size", "precision", "fp32_layers"),
    [
        (1, 1, "fp16", (2, 3, 4, 7, 8)),
        (2, 1, "fp16", (2,)),
        (1, 4, "fp32", ()),
    ],
)
def test_component_weights_expire_before_the_next_component(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    max_batch_size: int,
    precision: str,
    fp32_layers: tuple[int, ...],
) -> None:
    import struct
    import weakref

    from families.z_image import model
    from families.z_image import qwen3_encoder_builder as encoder
    from families.z_image import vae_2d_builder as vae
    from families.z_image import z_image_dit_tp_builder as tp_dit

    references: dict[str, weakref.ReferenceType] = {}
    phases: list[str] = []

    def load_text(_directory, **_kwargs):
        values = np.arange(8, dtype=np.float32)
        references["text"] = weakref.ref(values)
        return model.WeightDict(text=values)

    def compile_text(weights, **kwargs):
        assert weights["text"] is references["text"]()
        assert kwargs["precision"] == precision
        assert kwargs["max_batch_size"] == min(max_batch_size * 2, 8)
        phases.append("text")
        return b"unchanged-text-plan"

    def load_dit(_directory, **_kwargs):
        assert references["text"]() is None
        values = {
            "t_emb.0.weight": np.arange(6, dtype=np.float32).reshape(2, 3).copy(),
            "cap_norm.weight": np.array([7, 8, 9], dtype=np.float32),
            "main.0.to_q": np.ones((4, 4), dtype=np.float32),
        }
        references.update({name: weakref.ref(value) for name, value in values.items()})
        return model.WeightDict(values)

    def require_live_dit(weights):
        assert references["text"]() is None
        for name in ("t_emb.0.weight", "cap_norm.weight", "main.0.to_q"):
            assert weights[name] is references[name]()

    def compile_dit(weights, **kwargs):
        require_live_dit(weights)
        assert kwargs["precision"] == precision
        assert kwargs["max_batch_size"] == max_batch_size
        assert kwargs["fp32_layers"] == tuple(
            selector - 3 for selector in fp32_layers if selector >= 3
        )
        phases.append("dit")
        return b"unchanged-dit-plan"

    def compile_rank(weights, **kwargs):
        require_live_dit(weights)
        rank = kwargs["parallel_config"].rank
        phases.append(f"dit-rank-{rank}")
        return f"unchanged-dit-rank-{rank}".encode()

    original_serialize = model._serialize_preprocessor_weights

    def serialize(weights):
        require_live_dit(weights)
        phases.append("preprocessor")
        return original_serialize(weights)

    def compile_vae(_directory, **kwargs):
        assert all(reference() is None for reference in references.values())
        assert phases[-1] == "preprocessor"
        assert kwargs["precision"] == ("fp32" if 2 in fp32_layers else precision)
        phases.append("vae")
        return b"unchanged-vae-plan"

    monkeypatch.setattr(encoder, "load_qwen3_encoder_weights", load_text)
    monkeypatch.setattr(encoder, "build_qwen3_encoder_engine", compile_text)
    monkeypatch.setattr(dit, "load_z_image_dit_weights", load_dit)
    monkeypatch.setattr(dit, "build_z_image_dit_engine", compile_dit)
    monkeypatch.setattr(tp_dit, "build_z_image_dit_engine", compile_rank)
    monkeypatch.setattr(vae, "build_vae_2d_decoder_engine", compile_vae)
    monkeypatch.setattr(model, "_serialize_preprocessor_weights", serialize)
    result = model._ZImageModel().build_components(
        "/model",
        model.ModelConfig(
            raw={"image_height": 512, "image_width": 512, "_fp32_layers": fp32_layers}
        ),
        model.WeightDict(_text_encoder_dir="/text", _transformer_dir="/dit", _vae_dir="/vae"),
        precision=precision,
        parallel_config=model.ParallelConfig(tp_size=tp_size),
        max_batch_size=max_batch_size,
    )

    index = (
        b'{"t_embedder.mlp.0.weight": {"offset": 0, "shape": [2, 3]}, '
        b'"cap_embedder.norm.weight": {"offset": 24, "shape": [3]}}'
    )
    expected_bytes = (
        struct.pack("<I", len(index)) + index + struct.pack("<9f", 0, 1, 2, 3, 4, 5, 7, 8, 9)
    )
    expected = {
        "text_encoders": [("qwen3", b"unchanged-text-plan")],
        "vae_decoder": b"unchanged-vae-plan",
        "preprocessor_weights": expected_bytes,
    }
    if tp_size == 1:
        expected["denoiser"] = b"unchanged-dit-plan"
        assert phases == ["text", "dit", "preprocessor", "vae"]
    else:
        expected["denoiser_ranks"] = {
            rank: f"unchanged-dit-rank-{rank}".encode() for rank in range(tp_size)
        }
        assert phases == ["text", "dit-rank-0", "dit-rank-1", "preprocessor", "vae"]
    if max_batch_size > 1:
        expected["max_batch_size_envelope"] = {
            "dit": max_batch_size,
            "text_encoder": min(max_batch_size * 2, 8),
            "vae": 1,
        }
    assert result == expected
