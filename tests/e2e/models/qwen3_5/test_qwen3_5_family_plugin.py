# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Branch-focused tests for the Qwen3.5 family plugin.

Trace: ARCH-FAM-001, UD-FAM-QWEN3-5
Intent: Validate Qwen3.5 DeltaNet/attention layer routing normalization and weight loading branches
Preconditions: Mixed DeltaNet/attention layer types and synthetic tensors with partial keys are provided
Postconditions: Layer type aliases are normalized correctly and branch-specific weights load with fallback behavior
"""

from __future__ import annotations

import numpy as np
import pytest

trt = pytest.importorskip(
    "tensorrt", reason="TensorRT is required for family builder tests"
)


try:
    from tensorrt_model_connect.config import ModelConfig
    import tensorrt_model_connect.families.qwen3_5 as qwen3_5
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _seq(*shape: int, start: int = 0) -> np.ndarray:
    size = int(np.prod(shape))
    return np.arange(start, start + size, dtype=np.float32).reshape(shape)


def _patch_tensor_io(monkeypatch: pytest.MonkeyPatch,
                     tensor_map: dict[str, np.ndarray]) -> None:
    monkeypatch.setattr(qwen3_5, "_open_safetensors", lambda _: ["reader"])
    monkeypatch.setattr(
        qwen3_5, "_has_tensor", lambda _readers, name: name in tensor_map)

    def _load(_readers, name: str):
        if name not in tensor_map:
            raise KeyError(name)
        return tensor_map[name]

    monkeypatch.setattr(qwen3_5, "_load_tensor", _load)


def test_parse_layer_types_normalizes_aliases():
    """Intent: validate routing normalization for mixed aliases.
    Preconditions: layer type strings include canonical, alias, and unknown values.
    Postconditions: aliases map to deltanet/attention and unknowns are lower-cased.
    """
    parsed = qwen3_5._parse_layer_types(
        ["linear", "FULL", "linear_attention", "full_attention", "Custom"])
    assert parsed == ["deltanet", "attention", "deltanet", "attention", "custom"]


def test_fp16_runtime_inputs_preserve_fp32_recurrent_state():
    """FP16 storage must not quantize the persistent DeltaNet state."""

    class _Tensor:
        def __init__(self, name: str, dtype):
            self.name = name
            self.dtype = dtype

    class _Layer:
        def __init__(self, output: _Tensor):
            self.output = output

        def get_output(self, index: int) -> _Tensor:
            assert index == 0
            return self.output

    class _Network:
        def __init__(self):
            self.cast_inputs: list[_Tensor] = []

        def add_cast(self, tensor: _Tensor, dtype):
            self.cast_inputs.append(tensor)
            return _Layer(_Tensor(f"{tensor.name}_cast", dtype))

    network = _Network()
    attention_mask = _Tensor("attention_mask", trt.float32)
    conv_state = _Tensor("conv_state", trt.float32)
    ssm_state = _Tensor("ssm_state", trt.float32)
    cache_k = _Tensor("cache_k", trt.float16)
    cache_v = _Tensor("cache_v", trt.float16)

    (
        prepared_mask,
        prepared_conv,
        prepared_ssm,
        prepared_cache_k,
        prepared_cache_v,
    ) = qwen3_5._prepare_runtime_inputs(
        network,
        trt.float16,
        attention_mask,
        [conv_state],
        [ssm_state],
        [cache_k],
        [cache_v],
    )

    assert prepared_mask.dtype == trt.float16
    assert prepared_conv[0].dtype == trt.float16
    assert prepared_cache_k[0].dtype == trt.float16
    assert prepared_cache_v[0].dtype == trt.float16
    assert prepared_ssm == [ssm_state]
    assert prepared_ssm[0].dtype == trt.float32
    assert ssm_state not in network.cast_inputs


def test_load_weights_mixed_branches_and_fallbacks(monkeypatch: pytest.MonkeyPatch):
    """Intent: execute DeltaNet + attention branches and optional-key fallbacks.
    Preconditions: one layer is DeltaNet, one layer is attention, with partial tensors missing.
    Postconditions: normalized weights/metadata are emitted with correct fallback behavior.
    """
    raw = {
        "text_config": {
            "layer_types": ["linear", "FULL"],
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_value_head_dim": 2,
            "linear_conv_kernel_dim": 3,
            "rope_parameters": {
                "partial_rotary_factor": 0.5,
                "rope_theta": 321.0,
            },
        }
    }
    cfg = ModelConfig(
        model_type="qwen3_5",
        vocab_size=5,
        hidden_size=8,
        intermediate_size=6,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        rope_theta=9999.0,
        raw=raw,
    )

    tensors: dict[str, np.ndarray] = {
        # Embedding only under fallback key.
        "model.embed_tokens.weight": _seq(5, 8, start=0),
        # Layer 0 DeltaNet path.
        "model.language_model.layers.0.input_layernorm.weight": _seq(8, start=100),
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": _seq(
            8, 8, start=200
        ),
        "model.language_model.layers.0.linear_attn.in_proj_z.weight": _seq(
            4, 8, start=400
        ),
        "model.language_model.layers.0.linear_attn.in_proj_a.weight": _seq(
            2, 8, start=500
        ),
        "model.language_model.layers.0.linear_attn.in_proj_b.weight": _seq(
            2, 8, start=600
        ),
        "model.language_model.layers.0.linear_attn.A_log": np.log(
            np.array([1.0, 2.0], dtype=np.float32)
        ),
        "model.language_model.layers.0.linear_attn.dt_bias": np.array(
            [0.25, -0.5], dtype=np.float32
        ),
        "model.language_model.layers.0.linear_attn.conv1d.weight": _seq(
            8, 1, 3, start=700
        ),
        # conv1d.bias intentionally omitted to exercise zero fallback.
        "model.language_model.layers.0.linear_attn.norm.weight": np.array(
            [0.5, 1.5], dtype=np.float32
        ),
        "model.language_model.layers.0.linear_attn.out_proj.weight": _seq(
            8, 4, start=800
        ),
        "model.language_model.layers.0.mlp.gate_proj.weight": _seq(6, 8, start=900),
        "model.language_model.layers.0.mlp.up_proj.weight": _seq(6, 8, start=1000),
        "model.language_model.layers.0.mlp.down_proj.weight": _seq(8, 6, start=1100),
        # Layer 1 attention path.
        # input_layernorm intentionally omitted to exercise ones fallback.
        "model.language_model.layers.1.post_attention_layernorm.weight": _seq(
            8, start=1200
        ),
        "model.language_model.layers.1.self_attn.q_proj.weight": _seq(
            16, 8, start=1300
        ),
        "model.language_model.layers.1.self_attn.k_proj.weight": _seq(4, 8, start=1500),
        "model.language_model.layers.1.self_attn.v_proj.weight": _seq(4, 8, start=1600),
        "model.language_model.layers.1.self_attn.o_proj.weight": _seq(8, 8, start=1700),
        "model.language_model.layers.1.self_attn.q_norm.weight": np.array(
            [0.1, 0.2, 0.3, 0.4], dtype=np.float32
        ),
        # k_norm intentionally omitted.
        # Layer 1 MLP intentionally omitted (gate check should skip all MLP loads).
    }
    _patch_tensor_io(monkeypatch, tensors)

    weights = qwen3_5.plugin.load_weights("/unused", cfg)

    np.testing.assert_allclose(weights["embedding"], tensors["model.embed_tokens.weight"])

    np.testing.assert_allclose(
        weights["layer.0.input_norm"],
        1.0 + tensors["model.language_model.layers.0.input_layernorm.weight"],
    )
    np.testing.assert_allclose(weights["layer.1.input_norm"], np.ones(8, dtype=np.float32))

    np.testing.assert_allclose(
        weights["layer.0.post_attn_norm"], np.ones(8, dtype=np.float32))
    np.testing.assert_allclose(
        weights["layer.1.post_attn_norm"],
        1.0 + tensors["model.language_model.layers.1.post_attention_layernorm.weight"],
    )

    np.testing.assert_allclose(
        weights["layer.0.conv1d_bias"], np.zeros(8, dtype=np.float32))
    np.testing.assert_allclose(
        weights["layer.0.deltanet_norm"],
        np.array([0.5, 1.5, 0.5, 1.5], dtype=np.float32),
    )
    np.testing.assert_allclose(
        weights["layer.0.A"], np.array([-1.0, -2.0], dtype=np.float32))

    q_raw = tensors["model.language_model.layers.1.self_attn.q_proj.weight"]
    q_reshaped = q_raw.reshape(2, 8, 8)
    expected_q = q_reshaped[:, :4, :].reshape(8, 8).T.astype(np.float32)
    expected_gate = q_reshaped[:, 4:, :].reshape(8, 8).T.astype(np.float32)
    np.testing.assert_allclose(weights["layer.1.w_q"], expected_q)
    np.testing.assert_allclose(weights["layer.1.w_gate_attn"], expected_gate)
    np.testing.assert_allclose(
        weights["layer.1.q_norm"],
        np.tile(
            1.0 + tensors["model.language_model.layers.1.self_attn.q_norm.weight"],
            2,
        ),
    )
    assert "layer.1.k_norm" not in weights

    assert "layer.0.w_gate" in weights
    assert "layer.0.w_up" in weights
    assert "layer.0.w_down" in weights
    assert "layer.1.w_gate" not in weights
    assert "layer.1.w_up" not in weights
    assert "layer.1.w_down" not in weights

    np.testing.assert_allclose(weights["final_norm"], np.ones(8, dtype=np.float32))
    np.testing.assert_allclose(
        weights["w_lm_head"], tensors["model.embed_tokens.weight"].T.astype(np.float32)
    )

    assert weights["_layer_types"] == ["deltanet", "attention"]
    assert weights["_num_mamba_layers"] == 1
    assert weights["_num_attention_layers"] == 1
    assert weights["_partial_rotary_factor"] == 0.5
    assert weights["_rope_theta"] == 321.0


def test_get_bundle_config_overrides_normalizes_hybrid_fields():
    """Intent: validate bundle-config normalization for hybrid runtime fields.
    Preconditions: text_config provides aliased layer types and linear dimensions.
    Postconditions: output reports normalized layer types, counts, and derived dims.
    """
    raw = {
        "text_config": {
            "layer_types": ["linear_attention", "full_attention", "unknown"],
            "linear_num_value_heads": 3,
            "linear_num_key_heads": 1,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 5,
        }
    }
    cfg = ModelConfig(
        model_type="qwen3_5",
        vocab_size=5,
        hidden_size=12,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=3,
        num_key_value_heads=1,
        raw=raw,
    )

    overrides = qwen3_5.plugin.get_bundle_config_overrides(cfg)
    assert overrides["layer_types"] == ["deltanet", "attention", "unknown"]
    assert overrides["num_mamba_layers"] == 1
    assert overrides["num_attention_layers"] == 1
    assert overrides["d_inner"] == 12
    assert overrides["mamba_d_state"] == 4
    assert overrides["mamba_d_conv"] == 5
    assert overrides["mamba_nheads"] == 3
    assert overrides["mamba_head_dim"] == 4
    assert overrides["conv_dim"] == 20
