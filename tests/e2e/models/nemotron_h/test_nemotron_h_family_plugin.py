# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Branch-focused tests for the Nemotron-H family plugin.

Trace: ARCH-FAM-001, UD-FAM-NEMOTRON-H
Intent: Validate Nemotron-H hybrid mamba/attention layer routing and weight loading branches
Preconditions: Layer type pattern string and synthetic tensors for mamba2/mlp/attention layers are provided
Postconditions: Layer types are correctly parsed and branch-specific weights load with correct fallback behavior
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from tensorrt_model_connect.config import ModelConfig
    nemotron_h = importlib.import_module(
        "tensorrt_model_connect.families.nemotron_h.plugin")
    plugin = nemotron_h.plugin
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _seq(*shape: int, start: int = 0) -> np.ndarray:
    size = int(np.prod(shape))
    return np.arange(start, start + size, dtype=np.float32).reshape(shape)


def _patch_tensor_io(monkeypatch: pytest.MonkeyPatch,
                     tensor_map: dict[str, np.ndarray]) -> None:
    monkeypatch.setattr(nemotron_h, "_open_safetensors", lambda _: ["reader"])
    monkeypatch.setattr(
        nemotron_h, "_has_tensor", lambda _readers, name: name in tensor_map)

    def _load(_readers, name: str):
        if name not in tensor_map:
            raise KeyError(name)
        return tensor_map[name]

    monkeypatch.setattr(nemotron_h, "_load_tensor", _load)


def test_parse_layer_types_maps_and_filters_pattern_chars():
    """Intent: validate pattern-to-layer-type conversion.
    Preconditions: pattern includes valid markers and unrelated characters.
    Postconditions: only valid markers are retained and mapped to canonical layer names.
    """
    parsed = nemotron_h._parse_layer_types("M-x*-M")
    assert parsed == ["mamba2", "mlp", "attention", "mlp", "mamba2"]


def test_load_weights_mixed_layer_routing_and_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
):
    """Intent: execute mamba2/mlp/attention branches plus fallback code paths.
    Preconditions: three-layer hybrid pattern with missing optional mamba norm/final/lm_head keys.
    Postconditions: branch-specific keys are loaded and fallback values are synthesized correctly.
    """
    raw = {
        "hybrid_override_pattern": "M-*",
        "mamba_num_heads": 2,
        "mamba_head_dim": 3,
        "n_groups": 1,
        "ssm_state_size": 2,
        "conv_kernel": 2,
    }
    cfg = ModelConfig(
        model_type="nemotron_h",
        vocab_size=5,
        hidden_size=8,
        intermediate_size=10,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        raw=raw,
    )

    tensors: dict[str, np.ndarray] = {
        "backbone.embeddings.weight": _seq(5, 8, start=0),
        "backbone.layers.0.norm.weight": _seq(8, start=100),
        "backbone.layers.1.norm.weight": _seq(8, start=200),
        "backbone.layers.2.norm.weight": _seq(8, start=300),
        # Layer 0 (mamba2)
        "backbone.layers.0.mixer.in_proj.weight": _seq(18, 8, start=400),
        "backbone.layers.0.mixer.conv1d.weight": _seq(10, 1, 2, start=600),
        "backbone.layers.0.mixer.conv1d.bias": _seq(10, start=700),
        "backbone.layers.0.mixer.out_proj.weight": _seq(8, 6, start=800),
        "backbone.layers.0.mixer.A_log": np.log(np.array([1.0, 3.0], dtype=np.float32)),
        "backbone.layers.0.mixer.D": np.array([0.25, 0.5], dtype=np.float32),
        "backbone.layers.0.mixer.dt_bias": np.array([-0.1, 0.2], dtype=np.float32),
        # mixer.norm.weight intentionally omitted to exercise ones fallback.
        # Layer 1 (mlp)
        "backbone.layers.1.mixer.up_proj.weight": _seq(10, 8, start=900),
        "backbone.layers.1.mixer.down_proj.weight": _seq(8, 10, start=1000),
        # Layer 2 (attention)
        "backbone.layers.2.mixer.q_proj.weight": _seq(8, 8, start=1100),
        "backbone.layers.2.mixer.k_proj.weight": _seq(4, 8, start=1200),
        "backbone.layers.2.mixer.v_proj.weight": _seq(4, 8, start=1300),
        "backbone.layers.2.mixer.o_proj.weight": _seq(8, 8, start=1400),
    }
    _patch_tensor_io(monkeypatch, tensors)

    weights = plugin.load_weights("/unused", cfg)

    np.testing.assert_allclose(
        weights["embedding"], tensors["backbone.embeddings.weight"])
    np.testing.assert_allclose(
        weights["layer.0.input_norm"], tensors["backbone.layers.0.norm.weight"])
    np.testing.assert_allclose(
        weights["layer.1.input_norm"], tensors["backbone.layers.1.norm.weight"])
    np.testing.assert_allclose(
        weights["layer.2.input_norm"], tensors["backbone.layers.2.norm.weight"])

    np.testing.assert_allclose(
        weights["layer.0.mamba_norm"], np.ones(6, dtype=np.float32))
    np.testing.assert_allclose(
        weights["layer.0.A"], np.array([-1.0, -3.0], dtype=np.float32))
    np.testing.assert_allclose(
        weights["layer.0.conv1d_weight"],
        tensors["backbone.layers.0.mixer.conv1d.weight"].reshape(10, 2),
    )

    assert "layer.1.w_up" in weights
    assert "layer.1.w_down" in weights
    assert "layer.1.w_q" not in weights

    assert weights["layer.2.w_q"].shape == (8, 8)
    assert weights["layer.2.w_k"].shape == (8, 4)
    assert weights["layer.2.w_v"].shape == (8, 4)
    assert weights["layer.2.w_o"].shape == (8, 8)

    np.testing.assert_allclose(weights["final_norm"], np.ones(8, dtype=np.float32))
    np.testing.assert_allclose(
        weights["w_lm_head"], tensors["backbone.embeddings.weight"].T.astype(np.float32)
    )

    assert weights["_layer_types"] == ["mamba2", "mlp", "attention"]
    assert weights["_num_mamba_layers"] == 1
    assert weights["_num_attention_layers"] == 1
    assert weights["_d_inner"] == 6
    assert weights["_conv_dim"] == 10
    assert weights["_d_conv"] == 2


def test_load_weights_raises_for_pattern_length_mismatch(
    monkeypatch: pytest.MonkeyPatch,
):
    """Intent: ensure malformed hybrid patterns fail fast.
    Preconditions: pattern maps to fewer layer markers than num_hidden_layers.
    Postconditions: load_weights raises AssertionError before tensor mapping proceeds.
    """
    cfg = ModelConfig(
        model_type="nemotron_h",
        vocab_size=5,
        hidden_size=8,
        intermediate_size=10,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        raw={"hybrid_override_pattern": "M-"},
    )
    monkeypatch.setattr(nemotron_h, "_open_safetensors", lambda _: ["reader"])

    with pytest.raises(AssertionError, match="Pattern length"):
        plugin.load_weights("/unused", cfg)


def test_get_bundle_config_overrides_derives_runtime_fields():
    """Intent: validate derived runtime fields for bundle config emission.
    Preconditions: raw config includes custom mamba dimensions and mixed layer pattern.
    Postconditions: overrides include normalized layer routing and derived shape metadata.
    """
    raw = {
        "hybrid_override_pattern": "M-*-x",
        "mamba_num_heads": 3,
        "mamba_head_dim": 2,
        "n_groups": 2,
        "mamba_state_dim": 4,
        "conv_kernel": 5,
    }
    cfg = ModelConfig(
        model_type="nemotron_h",
        vocab_size=5,
        hidden_size=8,
        intermediate_size=10,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        raw=raw,
    )

    overrides = plugin.get_bundle_config_overrides(cfg)
    assert overrides["layer_types"] == ["mamba2", "mlp", "attention", "mlp"]
    assert overrides["num_mamba_layers"] == 1
    assert overrides["num_attention_layers"] == 1
    assert overrides["d_inner"] == 6
    assert overrides["mamba_d_state"] == 4
    assert overrides["mamba_d_conv"] == 5
    assert overrides["mamba_nheads"] == 3
    assert overrides["mamba_head_dim"] == 2
    assert overrides["conv_dim"] == 22
    assert overrides["n_groups"] == 2
