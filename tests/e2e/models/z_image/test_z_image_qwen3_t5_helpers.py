# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for owned Qwen3/T5 helper and weight-loader logic.

These tests avoid TensorRT runtime by importing modules with a fake trt stub.

Trace: ARCH-FAM-001, UD-FAM-QWEN3-T5
Intent: Validate Qwen3 and T5 helper functions and weight-loader logic with fake TRT stub
Preconditions: Fake tensorrt module is injected; no real TRT runtime
Postconditions: Helper functions produce correct weight transformations and config derivations
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# Ensure imports resolve to this workspace's Python package.
_PKG_ROOT = Path(__file__).resolve().parents[4] / "python"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))


def _make_fake_trt() -> types.SimpleNamespace:
    class _Logger:
        VERBOSE = 2
        WARNING = 1

        def __init__(self, _level):
            pass

    return types.SimpleNamespace(
        Logger=_Logger,
        ElementWiseOperation=types.SimpleNamespace(SUM="sum", SUB="sub", PROD="prod"),
        ReduceOperation=types.SimpleNamespace(AVG="avg"),
        UnaryOperation=types.SimpleNamespace(SQRT="sqrt", RECIP="recip"),
        MatrixOperation=types.SimpleNamespace(NONE="none", TRANSPOSE="transpose"),
        ActivationType=types.SimpleNamespace(SIGMOID="sigmoid"),
        MemoryPoolType=types.SimpleNamespace(WORKSPACE="workspace"),
        BuilderFlag=types.SimpleNamespace(TF32="tf32"),
        NetworkDefinitionCreationFlag=types.SimpleNamespace(EXPLICIT_BATCH=0),
        Permutation=lambda dims: tuple(dims),
        float32="float32",
        int32="int32",
    )


def _drop_imported_module(module_name: str) -> None:
    sys.modules.pop(module_name, None)
    package_name, _, attribute_name = module_name.rpartition(".")
    package = sys.modules.get(package_name)
    if package is not None and hasattr(package, attribute_name):
        delattr(package, attribute_name)


def _import_with_fake_trt(module_name: str):
    """Import a tensorrt_model_connect submodule while tensorrt is mocked."""
    sentinel = object()
    old_trt = sys.modules.get("tensorrt", sentinel)
    _drop_imported_module(module_name)
    sys.modules["tensorrt"] = _make_fake_trt()
    try:
        return importlib.import_module(module_name)
    finally:
        if old_trt is sentinel:
            sys.modules.pop("tensorrt", None)
        else:
            sys.modules["tensorrt"] = old_trt


@pytest.mark.unit
def test_qwen3_native_rope_table_has_expected_identities() -> None:
    """Intent: validate deterministic shared native RoPE table construction math.

    Preconditions: Qwen3 helper module is importable with fake trt.
    Postconditions: half-dim cos/sin tables satisfy position-0 identity and trig invariants.
    """
    mod = _import_with_fake_trt("tensorrt_model_connect.families.z_image.qwen3_encoder_builder")

    cos = mod.graph_ops.make_rope_table_half_dim(
        max_cache_length=3,
        head_dim=4,
        rope_theta=10000.0,
        cosine=True,
    )
    sin = mod.graph_ops.make_rope_table_half_dim(
        max_cache_length=3,
        head_dim=4,
        rope_theta=10000.0,
        cosine=False,
    )

    assert cos.shape == (3, 2)
    assert sin.shape == (3, 2)
    np.testing.assert_allclose(cos[0], np.ones((2,), dtype=np.float32))
    np.testing.assert_allclose(sin[0], np.zeros((2,), dtype=np.float32))
    np.testing.assert_allclose(cos[1] ** 2 + sin[1] ** 2, np.ones((2,), dtype=np.float32), atol=1e-5)
    assert cos[1, 0] == pytest.approx(np.cos(1.0), abs=1e-7)
    assert sin[1, 1] == pytest.approx(np.sin(0.01), abs=1e-7)


@pytest.mark.unit
def test_load_qwen3_encoder_weights_transposes_and_optional_norm() -> None:
    """Intent: verify Qwen3 loader key mapping, transpose logic, and optional final norm.

    Preconditions: Safetensors reader helpers are mocked with deterministic arrays.
    Postconditions: Returned WeightDict has expected keys and transformed values.
    """
    mod = _import_with_fake_trt("tensorrt_model_connect.families.z_image.qwen3_encoder_builder")

    tensors = {
        "model.embed_tokens.weight": np.arange(20, dtype=np.float32).reshape(5, 4),
        "model.layers.0.self_attn.q_proj.weight": np.arange(16, dtype=np.float32).reshape(4, 4),
        "model.layers.0.self_attn.k_proj.weight": np.arange(16, dtype=np.float32).reshape(4, 4) + 10,
        "model.layers.0.self_attn.v_proj.weight": np.arange(16, dtype=np.float32).reshape(4, 4) + 20,
        "model.layers.0.self_attn.o_proj.weight": np.arange(16, dtype=np.float32).reshape(4, 4) + 30,
        "model.layers.0.self_attn.q_norm.weight": np.array([1.0, 2.0], dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.array([3.0, 4.0], dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.array(
            [9.0, 10.0, 11.0, 12.0], dtype=np.float32
        ),
        "model.layers.0.mlp.gate_proj.weight": np.arange(32, dtype=np.float32).reshape(8, 4),
        "model.layers.0.mlp.up_proj.weight": np.arange(32, dtype=np.float32).reshape(8, 4) + 1,
        "model.layers.0.mlp.down_proj.weight": np.arange(32, dtype=np.float32).reshape(4, 8) + 2,
        "model.norm.weight": np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32),
    }

    with patch.object(mod, "_open_safetensors", lambda _path: tensors), patch.object(
        mod, "_load_tensor", lambda readers, name: readers[name]
    ), patch.object(mod, "_has_tensor", lambda readers, name: name in readers):
        weights = mod.load_qwen3_encoder_weights(
            model_dir="unused",
            hidden_size=4,
            num_layers=1,
            num_heads=2,
            num_kv_heads=2,
            intermediate_size=8,
            vocab_size=5,
        )

    np.testing.assert_allclose(weights["embed_tokens"], tensors["model.embed_tokens.weight"])
    np.testing.assert_allclose(
        weights["layer.0.q_proj"],
        tensors["model.layers.0.self_attn.q_proj.weight"].T.astype(np.float32),
    )
    np.testing.assert_allclose(
        weights["layer.0.down_proj"],
        tensors["model.layers.0.mlp.down_proj.weight"].T.astype(np.float32),
    )
    np.testing.assert_allclose(weights["final_norm"], tensors["model.norm.weight"])

    tensors_no_final = dict(tensors)
    tensors_no_final.pop("model.norm.weight")
    with patch.object(mod, "_open_safetensors", lambda _path: tensors_no_final), patch.object(
        mod, "_load_tensor", lambda readers, name: readers[name]
    ), patch.object(mod, "_has_tensor", lambda readers, name: name in readers):
        weights_no_final = mod.load_qwen3_encoder_weights(
            model_dir="unused",
            hidden_size=4,
            num_layers=1,
            num_heads=2,
            num_kv_heads=2,
            intermediate_size=8,
            vocab_size=5,
        )
    assert "final_norm" not in weights_no_final


@pytest.mark.unit
def test_load_t5_weights_transposes_and_bias_fallback() -> None:
    """Intent: verify T5 loader transposes linear weights and handles per-layer bias fallback.

    Preconditions: checkpoint_mapper helpers are replaced by a fake deterministic module.
    Postconditions: Returned weights are float32 and include expected optional keys.
    """
    mod = _import_with_fake_trt("tensorrt_model_connect.families.flux.t5_encoder_builder")

    tensors: dict[str, np.ndarray] = {
        "shared.weight": np.arange(28, dtype=np.float32).reshape(7, 4),
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": np.arange(
            64, dtype=np.float32
        ).reshape(32, 2),
        "encoder.final_layer_norm.weight": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
    }

    for layer in range(2):
        prefix = f"encoder.block.{layer}"
        tensors[f"{prefix}.layer.0.SelfAttention.q.weight"] = np.arange(16, dtype=np.float32).reshape(4, 4) + layer
        tensors[f"{prefix}.layer.0.SelfAttention.k.weight"] = np.arange(16, dtype=np.float32).reshape(4, 4) + layer + 1
        tensors[f"{prefix}.layer.0.SelfAttention.v.weight"] = np.arange(16, dtype=np.float32).reshape(4, 4) + layer + 2
        tensors[f"{prefix}.layer.0.SelfAttention.o.weight"] = np.arange(16, dtype=np.float32).reshape(4, 4) + layer + 3
        tensors[f"{prefix}.layer.0.layer_norm.weight"] = np.arange(4, dtype=np.float32) + layer
        tensors[f"{prefix}.layer.1.DenseReluDense.wi_0.weight"] = np.arange(24, dtype=np.float32).reshape(6, 4) + layer
        tensors[f"{prefix}.layer.1.DenseReluDense.wi_1.weight"] = np.arange(24, dtype=np.float32).reshape(6, 4) + layer + 1
        tensors[f"{prefix}.layer.1.DenseReluDense.wo.weight"] = np.arange(24, dtype=np.float32).reshape(4, 6) + layer + 2
        tensors[f"{prefix}.layer.1.layer_norm.weight"] = np.arange(4, dtype=np.float32) + layer + 10

    cm = importlib.import_module("tensorrt_model_connect.families.flux.checkpoint_mapper")

    with patch.object(cm, "_open_safetensors", lambda _path: tensors), patch.object(
        cm, "_load_tensor", lambda readers, name: readers[name]
    ), patch.object(cm, "_has_tensor", lambda readers, name: name in readers), patch.object(
        cm, "_target_np_dtype",
        lambda precision: np.float16 if precision == "fp16" else np.float32,
    ):
        weights = mod.load_t5_weights(
            model_dir="unused",
            d_model=4,
            num_heads=2,
            d_kv=2,
            d_ff=6,
            num_layers=2,
            vocab_size=7,
        )

    np.testing.assert_allclose(
        weights["encoder.block.0.layer.0.SelfAttention.q.weight"],
        tensors["encoder.block.0.layer.0.SelfAttention.q.weight"].T.astype(np.float32),
    )
    np.testing.assert_allclose(
        weights["encoder.block.1.layer.1.DenseReluDense.wi_0.weight"],
        tensors["encoder.block.1.layer.1.DenseReluDense.wi_0.weight"].T.astype(np.float32),
    )
    assert "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight" in weights
    assert "encoder.block.1.layer.0.SelfAttention.relative_attention_bias.weight" not in weights
    assert weights["encoder.final_layer_norm.weight"].dtype == np.float32

    with patch.object(cm, "_open_safetensors", lambda _path: tensors), patch.object(
        cm, "_load_tensor", lambda readers, name: readers[name]
    ), patch.object(cm, "_has_tensor", lambda readers, name: name in readers), patch.object(
        cm, "_target_np_dtype",
        lambda precision: np.float16 if precision == "fp16" else np.float32,
    ):
        fp16_weights = mod.load_t5_weights(
            model_dir="unused",
            d_model=4,
            num_heads=2,
            d_kv=2,
            d_ff=6,
            num_layers=2,
            vocab_size=7,
            precision="fp16",
        )

    assert fp16_weights["shared.weight"].dtype == np.float16
    assert (
        fp16_weights["encoder.block.0.layer.0.SelfAttention.q.weight"].dtype
        == np.float16
    )
    assert fp16_weights["encoder.block.0.layer.0.layer_norm.weight"].dtype == np.float32
