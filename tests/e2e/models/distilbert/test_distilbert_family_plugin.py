# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the DistilBERT family plugin.

These tests are deterministic and avoid TRT/GPU dependencies by mocking
checkpoint I/O and engine-builder calls.

Trace: ARCH-FAM-001, UD-FAM-DISTILBERT
Intent: Validate DistilBERT family plugin weight loading and config parsing with mocked I/O
Preconditions: Mocked tensor I/O returns synthetic DistilBERT weight tensors
Postconditions: Plugin correctly maps HF weight keys and produces expected canonical WeightDict
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from tensorrt_model_connect.config import ModelConfig
    import tensorrt_model_connect.families.distilbert as distilbert_mod
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _cfg(**raw_overrides: object) -> ModelConfig:
    payload = {
        "model_type": "distilbert",
        "vocab_size": 8,
        "hidden_size": 4,
        "intermediate_size": 6,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "max_position_embeddings": 7,
        "type_vocab_size": 3,
    }
    payload.update(raw_overrides)
    return ModelConfig.from_json(json.dumps(payload))


def _tensor_maker() -> callable:
    cursor = {"v": 1.0}

    def make(*shape: int) -> np.ndarray:
        n = int(np.prod(shape))
        start = cursor["v"]
        arr = np.arange(start, start + n, dtype=np.float32).reshape(shape)
        cursor["v"] += n
        return arr

    return make


def _distil_tensors() -> dict[str, np.ndarray]:
    m = _tensor_maker()
    t: dict[str, np.ndarray] = {}

    t["distilbert.embeddings.word_embeddings.weight"] = m(8, 4)
    t["distilbert.embeddings.position_embeddings.weight"] = m(7, 4)
    t["distilbert.embeddings.LayerNorm.weight"] = m(4)
    t["distilbert.embeddings.LayerNorm.bias"] = m(4)

    p = "distilbert.transformer.layer.0"
    t[f"{p}.attention.q_lin.weight"] = m(4, 4)
    t[f"{p}.attention.k_lin.weight"] = m(4, 4)
    t[f"{p}.attention.v_lin.weight"] = m(4, 4)
    t[f"{p}.attention.q_lin.bias"] = m(4)
    t[f"{p}.attention.k_lin.bias"] = m(4)
    t[f"{p}.attention.v_lin.bias"] = m(4)
    t[f"{p}.attention.out_lin.weight"] = m(4, 4)
    t[f"{p}.attention.out_lin.bias"] = m(4)

    t[f"{p}.sa_layer_norm.weight"] = m(4)
    t[f"{p}.sa_layer_norm.bias"] = m(4)

    t[f"{p}.ffn.lin1.weight"] = m(6, 4)
    t[f"{p}.ffn.lin1.bias"] = m(6)
    t[f"{p}.ffn.lin2.weight"] = m(4, 6)
    t[f"{p}.ffn.lin2.bias"] = m(4)

    t[f"{p}.output_layer_norm.weight"] = m(4)
    t[f"{p}.output_layer_norm.bias"] = m(4)
    return t


def _install_tensor_stubs(monkeypatch: pytest.MonkeyPatch,
                          tensors: dict[str, np.ndarray]) -> None:
    monkeypatch.setattr(distilbert_mod, "_open_safetensors", lambda _: ["reader"])
    monkeypatch.setattr(distilbert_mod, "_load_tensor", lambda _r, name: tensors[name])


def test_matches() -> None:
    """Intent: verify model_type routing for DistilBERT.

    Preconditions: plugin instance is imported.
    Postconditions: only "distilbert" (case-insensitive) matches.
    """
    plugin = distilbert_mod.plugin
    assert plugin.matches("distilbert")
    assert plugin.matches("DistilBERT")
    assert not plugin.matches("bert")


def test_load_weights_maps_and_transposes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: validate key mapping, zero token-type synthesis, and transposes.

    Preconditions: checkpoint loading helpers are mocked with deterministic tensors.
    Postconditions: output WeightDict has expected transformed arrays.
    """
    tensors = _distil_tensors()
    _install_tensor_stubs(monkeypatch, tensors)

    cfg = _cfg(type_vocab_size=3)
    weights = distilbert_mod.plugin.load_weights("/unused", cfg)

    np.testing.assert_allclose(
        weights["embedding"],
        tensors["distilbert.embeddings.word_embeddings.weight"],
    )
    np.testing.assert_allclose(
        weights["position_embedding"],
        tensors["distilbert.embeddings.position_embeddings.weight"],
    )

    assert weights["token_type_embedding"].shape == (3, 4)
    np.testing.assert_allclose(weights["token_type_embedding"], np.zeros((3, 4), dtype=np.float32))

    np.testing.assert_allclose(
        weights["layer.0.w_q"],
        tensors["distilbert.transformer.layer.0.attention.q_lin.weight"].T,
    )
    np.testing.assert_allclose(
        weights["layer.0.w_fc1"],
        tensors["distilbert.transformer.layer.0.ffn.lin1.weight"].T,
    )
    np.testing.assert_allclose(
        weights["layer.0.w_fc2"],
        tensors["distilbert.transformer.layer.0.ffn.lin2.weight"].T,
    )


def test_load_weights_rejects_bad_embedding_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intent: cover the embedding shape-guard failure path.

    Preconditions: mocked embedding tensor has wrong first dimension.
    Postconditions: load_weights raises AssertionError with shape context.
    """
    tensors = _distil_tensors()
    tensors["distilbert.embeddings.word_embeddings.weight"] = np.zeros((9, 4), dtype=np.float32)
    _install_tensor_stubs(monkeypatch, tensors)

    with pytest.raises(AssertionError, match="Embedding shape"):
        distilbert_mod.plugin.load_weights("/unused", _cfg())


def test_build_engine_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: verify build_engine forwards arguments to encoder builder.

    Preconditions: encoder builder function is monkeypatched with a recorder stub.
    Postconditions: plugin returns stub bytes and forwards max_seq_length/verbose.
    """
    calls: dict[str, object] = {}

    def fake_builder(config, weights, *, max_seq_length, precision, verbose):
        calls["config"] = config
        calls["weights"] = weights
        calls["max_seq_length"] = max_seq_length
        calls["precision"] = precision
        calls["verbose"] = verbose
        return b"distil-plan"

    monkeypatch.setattr(distilbert_mod, "build_encoder_engine", fake_builder)

    cfg = _cfg()
    raw_weights = {"embedding": np.zeros((8, 4), dtype=np.float32)}
    out = distilbert_mod.plugin.build_engine(cfg, raw_weights, 19, verbose=True)

    assert out == b"distil-plan"
    assert calls["config"] is cfg
    assert calls["weights"] is raw_weights
    assert calls["max_seq_length"] == 19
    assert calls["precision"] == "fp32"
    assert calls["verbose"] is True
