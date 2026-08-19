# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MPNet family plugin helpers and plugin behavior.

Trace: ARCH-FAM-001, UD-FAM-MPNET
Intent: Validate MPNet family plugin weight key mapping, relative position bias handling, and config parsing
Preconditions: Synthetic MPNet weight tensors with optional relative position bias are provided
Postconditions: Plugin produces correct canonical weight keys including optional bias terms
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from tensorrt_model_connect.config import ModelConfig
    import tensorrt_model_connect.models.mpnet.model as mpnet_mod
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _cfg(**raw_overrides: object) -> ModelConfig:
    payload = {
        "model_type": "mpnet",
        "vocab_size": 9,
        "hidden_size": 4,
        "intermediate_size": 6,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "max_position_embeddings": 9,
        "type_vocab_size": 3,
        "pad_token_id": 1,
    }
    payload.update(raw_overrides)
    return ModelConfig.from_json(json.dumps(payload))


def _tensor_maker() -> callable:
    cursor = {"v": 100.0}

    def make(*shape: int) -> np.ndarray:
        n = int(np.prod(shape))
        start = cursor["v"]
        arr = np.arange(start, start + n, dtype=np.float32).reshape(shape)
        cursor["v"] += n
        return arr

    return make


def _make_mpnet_tensors(root: str, *, include_rel_bias: bool) -> dict[str, np.ndarray]:
    m = _tensor_maker()
    prefix = f"{root}." if root else ""
    t: dict[str, np.ndarray] = {}

    t[f"{prefix}embeddings.word_embeddings.weight"] = m(9, 4)
    t[f"{prefix}embeddings.position_embeddings.weight"] = m(9, 4)
    t[f"{prefix}embeddings.LayerNorm.weight"] = m(4)
    t[f"{prefix}embeddings.LayerNorm.bias"] = m(4)

    p = f"{prefix}encoder.layer.0"
    t[f"{p}.attention.attn.q.weight"] = m(4, 4)
    t[f"{p}.attention.attn.q.bias"] = m(4)
    t[f"{p}.attention.attn.k.weight"] = m(4, 4)
    t[f"{p}.attention.attn.k.bias"] = m(4)
    t[f"{p}.attention.attn.v.weight"] = m(4, 4)
    t[f"{p}.attention.attn.v.bias"] = m(4)
    t[f"{p}.attention.attn.o.weight"] = m(4, 4)
    t[f"{p}.attention.attn.o.bias"] = m(4)

    t[f"{p}.attention.LayerNorm.weight"] = m(4)
    t[f"{p}.attention.LayerNorm.bias"] = m(4)

    t[f"{p}.intermediate.dense.weight"] = m(6, 4)
    t[f"{p}.intermediate.dense.bias"] = m(6)
    t[f"{p}.output.dense.weight"] = m(4, 6)
    t[f"{p}.output.dense.bias"] = m(4)
    t[f"{p}.output.LayerNorm.weight"] = m(4)
    t[f"{p}.output.LayerNorm.bias"] = m(4)

    if include_rel_bias:
        t[f"{prefix}encoder.relative_attention_bias.weight"] = m(8, 2)

    return t


def _install_tensor_stubs(monkeypatch: pytest.MonkeyPatch,
                          tensors: dict[str, np.ndarray]) -> None:
    monkeypatch.setattr(mpnet_mod, "_open_safetensors", lambda _p: ["reader"])
    monkeypatch.setattr(mpnet_mod, "_has_tensor", lambda _r, name: name in tensors)
    monkeypatch.setattr(mpnet_mod, "_load_tensor", lambda _r, name: tensors[name])


def test_matches_and_prefix_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: cover matching behavior and prefix utility branches.

    Preconditions: _has_tensor is monkeypatched for each scenario.
    Postconditions: detect_prefix and _pfx return expected values.
    """
    assert mpnet_mod.matches("mpnet")
    assert mpnet_mod.matches("MPNet")
    assert not mpnet_mod.matches("bert")

    monkeypatch.setattr(
        mpnet_mod,
        "_has_tensor",
        lambda _r, name: name == "mpnet.embeddings.word_embeddings.weight",
    )
    assert mpnet_mod._detect_prefix(["reader"]) == "mpnet"

    monkeypatch.setattr(
        mpnet_mod,
        "_has_tensor",
        lambda _r, name: name == "embeddings.word_embeddings.weight",
    )
    assert mpnet_mod._detect_prefix(["reader"]) == ""

    monkeypatch.setattr(mpnet_mod, "_has_tensor", lambda _r, _name: False)
    assert mpnet_mod._detect_prefix(["reader"]) == "mpnet"

    assert mpnet_mod._pfx("mpnet", "x.y") == "mpnet.x.y"
    assert mpnet_mod._pfx("", "x.y") == "x.y"


def test_compute_relative_position_bias_shape_and_sign_buckets() -> None:
    """Intent: validate deterministic bias bucketing math.

    Preconditions: deterministic bias table and short sequence length are used.
    Postconditions: output shape is correct and opposite relative signs map to different buckets.
    """
    bias_table = np.arange(16, dtype=np.float32).reshape(8, 2)
    out = mpnet_mod._compute_relative_position_bias(
        seq_length=4,
        num_buckets=8,
        num_heads=2,
        bias_table=bias_table,
    )

    assert out.shape == (2, 4, 4)
    assert out.dtype == np.float32

    # Diagonal uses bucket 0.
    np.testing.assert_allclose(out[:, 0, 0], bias_table[0])

    # q=0,k=1 has positive relative position -> half-bucket offset path.
    np.testing.assert_allclose(out[0, 0, 1], bias_table[5, 0])
    # q=1,k=0 has negative relative position -> no offset.
    np.testing.assert_allclose(out[0, 1, 0], bias_table[1, 0])


def test_load_weights_no_prefix_with_relative_bias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intent: cover root='' loading and optional relative-bias capture.

    Preconditions: tensor map contains bare-key MPNet tensors plus relative bias.
    Postconditions: standard keys load, position embeddings are offset-sliced, and metadata keys are emitted.
    """
    tensors = _make_mpnet_tensors("", include_rel_bias=True)
    _install_tensor_stubs(monkeypatch, tensors)

    cfg = _cfg(pad_token_id=1, type_vocab_size=3)
    weights = mpnet_mod.load_weights("/unused", cfg)

    np.testing.assert_allclose(weights["embedding"], tensors["embeddings.word_embeddings.weight"])
    np.testing.assert_allclose(weights["position_embedding"], tensors["embeddings.position_embeddings.weight"][2:])
    np.testing.assert_allclose(weights["layer.0.w_q"], tensors["encoder.layer.0.attention.attn.q.weight"].T)

    assert weights["token_type_embedding"].shape == (3, 4)
    np.testing.assert_allclose(weights["token_type_embedding"], np.zeros((3, 4), dtype=np.float32))

    assert "_relative_attention_bias" in weights
    assert "_relative_attention_num_buckets" in weights
    assert weights["_relative_attention_num_buckets"] == 8


def test_load_weights_mpnet_prefix_without_relative_bias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intent: cover root='mpnet' branch and missing relative-bias path.

    Preconditions: tensor map only includes mpnet.* keys and no relative bias tensor.
    Postconditions: weights load successfully and private relative keys are absent.
    """
    tensors = _make_mpnet_tensors("mpnet", include_rel_bias=False)
    _install_tensor_stubs(monkeypatch, tensors)

    weights = mpnet_mod.load_weights("/unused", _cfg())

    np.testing.assert_allclose(
        weights["embedding"],
        tensors["mpnet.embeddings.word_embeddings.weight"],
    )
    assert "_relative_attention_bias" not in weights
    assert "_relative_attention_num_buckets" not in weights


def test_build_engine_with_relative_bias_precompute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intent: verify build_engine precomputes and rewrites relative bias.

    Preconditions: input weights include private relative-bias entries.
    Postconditions: private keys are consumed, computed matrix is inserted, and builder is called.
    """
    calls: dict[str, object] = {}
    fake_bias_matrix = np.ones((2, 5, 5), dtype=np.float32)

    def fake_compute(seq_length, num_buckets, num_heads, bias_table):
        calls["compute"] = {
            "seq_length": seq_length,
            "num_buckets": num_buckets,
            "num_heads": num_heads,
            "bias_table": bias_table,
        }
        return fake_bias_matrix

    def fake_builder(config, weights, *, max_seq_length, precision, verbose):
        calls["builder"] = {
            "config": config,
            "weights": dict(weights),
            "max_seq_length": max_seq_length,
            "precision": precision,
            "verbose": verbose,
        }
        return b"mpnet-plan"

    monkeypatch.setattr(mpnet_mod, "_compute_relative_position_bias", fake_compute)
    monkeypatch.setattr(mpnet_mod, "build_encoder_engine", fake_builder)

    cfg = _cfg()
    bias_table = np.arange(16, dtype=np.float32).reshape(8, 2)
    weights = {
        "embedding": np.zeros((9, 4), dtype=np.float32),
        "_relative_attention_bias": bias_table,
        "_relative_attention_num_buckets": 8,
    }

    out = mpnet_mod.build_engine(cfg, weights, 5, verbose=True)

    assert out == b"mpnet-plan"
    assert calls["compute"]["seq_length"] == 5
    assert calls["compute"]["num_buckets"] == 8
    assert calls["builder"]["precision"] == "fp32"
    np.testing.assert_allclose(calls["compute"]["bias_table"], bias_table)

    builder_weights = calls["builder"]["weights"]
    assert "_relative_attention_bias" not in builder_weights
    assert "_relative_attention_num_buckets" not in builder_weights
    np.testing.assert_allclose(builder_weights["relative_position_bias"], fake_bias_matrix)


def test_build_engine_without_relative_bias_does_not_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intent: cover build path where no relative-bias metadata exists.

    Preconditions: _compute_relative_position_bias is patched to fail if called.
    Postconditions: build delegates directly without relative bias insertion.
    """
    def fail_compute(*_args, **_kwargs):
        raise AssertionError("_compute_relative_position_bias should not be called")

    def fake_builder(
        _config, weights, *, max_seq_length, precision, verbose,
    ):
        assert max_seq_length == 7
        assert precision == "fp32"
        assert verbose is False
        assert "relative_position_bias" not in weights
        return b"mpnet-no-rel-bias"

    monkeypatch.setattr(mpnet_mod, "_compute_relative_position_bias", fail_compute)
    monkeypatch.setattr(mpnet_mod, "build_encoder_engine", fake_builder)

    out = mpnet_mod.build_engine(
        _cfg(),
        {"embedding": np.zeros((9, 4), dtype=np.float32)},
        7,
        verbose=False,
    )
    assert out == b"mpnet-no-rel-bias"
