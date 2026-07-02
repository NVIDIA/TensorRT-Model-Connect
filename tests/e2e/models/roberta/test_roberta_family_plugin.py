# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the RoBERTa/XLM-RoBERTa family plugin.

Tests are deterministic and isolate filesystem/TRT dependencies via monkeypatch.

Trace: ARCH-FAM-001, UD-FAM-ROBERTA
Intent: Validate RoBERTa/XLM-RoBERTa family plugin weight key mapping and config parsing
Preconditions: Mocked tensor I/O returns synthetic RoBERTa weight tensors with correct naming
Postconditions: Plugin correctly maps HF weight keys to canonical encoder-only WeightDict format
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from tensorrt_model_connect.config import ModelConfig
    import tensorrt_model_connect.families.roberta as roberta_mod
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _cfg(**raw_overrides: object) -> ModelConfig:
    payload = {
        "model_type": "roberta",
        "vocab_size": 7,
        "hidden_size": 4,
        "intermediate_size": 6,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "max_position_embeddings": 10,
        "type_vocab_size": 2,
        "pad_token_id": 1,
    }
    payload.update(raw_overrides)
    return ModelConfig.from_json(json.dumps(payload))


def _tensor_maker() -> callable:
    cursor = {"v": 10.0}

    def make(*shape: int) -> np.ndarray:
        n = int(np.prod(shape))
        start = cursor["v"]
        arr = np.arange(start, start + n, dtype=np.float32).reshape(shape)
        cursor["v"] += n
        return arr

    return make


def _make_roberta_tensors(
    root: str,
    *,
    include_token_type: bool,
    include_pooler: bool,
    embed_ln_style: str,
    attn_ln_style: str,
    out_ln_style: str,
) -> dict[str, np.ndarray]:
    m = _tensor_maker()
    t: dict[str, np.ndarray] = {}

    t[f"{root}.embeddings.word_embeddings.weight"] = m(7, 4)
    t[f"{root}.embeddings.position_embeddings.weight"] = m(10, 4)
    if include_token_type:
        t[f"{root}.embeddings.token_type_embeddings.weight"] = m(2, 4)

    if embed_ln_style == "weight":
        t[f"{root}.embeddings.LayerNorm.weight"] = m(4)
        t[f"{root}.embeddings.LayerNorm.bias"] = m(4)
    else:
        t[f"{root}.embeddings.LayerNorm.gamma"] = m(4)
        t[f"{root}.embeddings.LayerNorm.beta"] = m(4)

    p = f"{root}.encoder.layer.0"
    t[f"{p}.attention.self.query.weight"] = m(4, 4)
    t[f"{p}.attention.self.query.bias"] = m(4)
    t[f"{p}.attention.self.key.weight"] = m(4, 4)
    t[f"{p}.attention.self.key.bias"] = m(4)
    t[f"{p}.attention.self.value.weight"] = m(4, 4)
    t[f"{p}.attention.self.value.bias"] = m(4)

    t[f"{p}.attention.output.dense.weight"] = m(4, 4)
    t[f"{p}.attention.output.dense.bias"] = m(4)

    if attn_ln_style == "weight":
        t[f"{p}.attention.output.LayerNorm.weight"] = m(4)
        t[f"{p}.attention.output.LayerNorm.bias"] = m(4)
    else:
        t[f"{p}.attention.output.LayerNorm.gamma"] = m(4)
        t[f"{p}.attention.output.LayerNorm.beta"] = m(4)

    t[f"{p}.intermediate.dense.weight"] = m(6, 4)
    t[f"{p}.intermediate.dense.bias"] = m(6)
    t[f"{p}.output.dense.weight"] = m(4, 6)
    t[f"{p}.output.dense.bias"] = m(4)

    if out_ln_style == "weight":
        t[f"{p}.output.LayerNorm.weight"] = m(4)
        t[f"{p}.output.LayerNorm.bias"] = m(4)
    else:
        t[f"{p}.output.LayerNorm.gamma"] = m(4)
        t[f"{p}.output.LayerNorm.beta"] = m(4)

    if include_pooler:
        t[f"{root}.pooler.dense.weight"] = m(4, 4)
        t[f"{root}.pooler.dense.bias"] = m(4)

    return t


def _install_tensor_stubs(monkeypatch: pytest.MonkeyPatch,
                          tensors: dict[str, np.ndarray]) -> None:
    monkeypatch.setattr(roberta_mod, "_open_safetensors", lambda _p: ["reader"])
    monkeypatch.setattr(roberta_mod, "_has_tensor", lambda _r, name: name in tensors)
    monkeypatch.setattr(roberta_mod, "_load_tensor", lambda _r, name: tensors[name])


def test_matches() -> None:
    """Intent: validate model_type matching aliases.

    Preconditions: plugin instance is imported.
    Postconditions: roberta/xlm-roberta match, unrelated types do not.
    """
    plugin = roberta_mod.plugin
    assert plugin.matches("roberta")
    assert plugin.matches("xlm-roberta")
    assert plugin.matches("XLM-RoBERTa")
    assert not plugin.matches("bert")


def test_detect_prefix_prefers_model_roberta(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: cover prefix detection when checkpoints use model.roberta.* keys.

    Preconditions: _has_tensor is patched to expose only model.roberta key.
    Postconditions: helper returns model.roberta prefix.
    """
    monkeypatch.setattr(
        roberta_mod,
        "_has_tensor",
        lambda _r, name: name == "model.roberta.embeddings.word_embeddings.weight",
    )
    assert roberta_mod._detect_prefix(["reader"]) == "model.roberta"


def test_load_ln_falls_back_to_gamma_beta(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: exercise legacy LayerNorm gamma/beta loading path.

    Preconditions: weight/bias keys are absent while gamma/beta are present.
    Postconditions: returned tensors come from gamma/beta keys.
    """
    tensors = {
        "x.LayerNorm.gamma": np.array([1, 2, 3], dtype=np.float32),
        "x.LayerNorm.beta": np.array([4, 5, 6], dtype=np.float32),
    }
    monkeypatch.setattr(roberta_mod, "_has_tensor", lambda _r, name: name in tensors)
    monkeypatch.setattr(roberta_mod, "_load_tensor", lambda _r, name: tensors[name])

    w, b = roberta_mod._load_ln(["reader"], "x.LayerNorm")
    np.testing.assert_allclose(w, tensors["x.LayerNorm.gamma"])
    np.testing.assert_allclose(b, tensors["x.LayerNorm.beta"])


def test_load_weights_with_model_prefix_and_pooler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intent: validate mapping for model.roberta prefix and optional pooler path.

    Preconditions: token-type and pooler tensors are present; embed LN uses gamma/beta.
    Postconditions: position slicing, transposes, and optional keys are populated.
    """
    tensors = _make_roberta_tensors(
        "model.roberta",
        include_token_type=True,
        include_pooler=True,
        embed_ln_style="gamma",
        attn_ln_style="gamma",
        out_ln_style="weight",
    )
    _install_tensor_stubs(monkeypatch, tensors)

    cfg = _cfg(type_vocab_size=2, pad_token_id=1)
    weights = roberta_mod.plugin.load_weights("/unused", cfg)

    raw_pos = tensors["model.roberta.embeddings.position_embeddings.weight"]
    np.testing.assert_allclose(weights["position_embedding"], raw_pos[2:])
    np.testing.assert_allclose(
        weights["token_type_embedding"],
        tensors["model.roberta.embeddings.token_type_embeddings.weight"],
    )

    np.testing.assert_allclose(
        weights["layer.0.w_q"],
        tensors["model.roberta.encoder.layer.0.attention.self.query.weight"].T,
    )
    assert "pooler_w" in weights
    assert "pooler_bias" in weights


def test_load_weights_without_token_type_or_pooler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intent: cover synthetic token-type table and missing-pooler branch.

    Preconditions: token-type and pooler tensors are absent in checkpoint map.
    Postconditions: zero token-type table is synthesized and pooler keys are absent.
    """
    tensors = _make_roberta_tensors(
        "roberta",
        include_token_type=False,
        include_pooler=False,
        embed_ln_style="weight",
        attn_ln_style="weight",
        out_ln_style="gamma",
    )
    _install_tensor_stubs(monkeypatch, tensors)

    cfg = _cfg(type_vocab_size=5, pad_token_id=3)
    weights = roberta_mod.plugin.load_weights("/unused", cfg)

    raw_pos = tensors["roberta.embeddings.position_embeddings.weight"]
    np.testing.assert_allclose(weights["position_embedding"], raw_pos[4:])

    assert weights["token_type_embedding"].shape == (5, 4)
    np.testing.assert_allclose(weights["token_type_embedding"], np.zeros((5, 4), dtype=np.float32))

    assert "pooler_w" not in weights
    assert "pooler_bias" not in weights


def test_build_engine_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: verify build_engine delegates with expected keyword mapping.

    Preconditions: build_encoder_engine is replaced by a recording stub.
    Postconditions: plugin return bytes and forwards max_cache_length as max_seq_length.
    """
    calls: dict[str, object] = {}

    def fake_builder(config, weights, *, max_seq_length, verbose):
        calls["config"] = config
        calls["weights"] = weights
        calls["max_seq_length"] = max_seq_length
        calls["verbose"] = verbose
        return b"roberta-plan"

    monkeypatch.setattr(roberta_mod, "build_encoder_engine", fake_builder)

    cfg = _cfg()
    raw_weights = {"embedding": np.zeros((7, 4), dtype=np.float32)}
    out = roberta_mod.plugin.build_engine(cfg, raw_weights, 33, verbose=False)

    assert out == b"roberta-plan"
    assert calls["config"] is cfg
    assert calls["weights"] is raw_weights
    assert calls["max_seq_length"] == 33
    assert calls["verbose"] is False
