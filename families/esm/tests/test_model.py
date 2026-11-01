# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-side unit tests for ESM-2.

Two subjects: ``config.resolve``, which has to refuse the ESM variants that
load every tensor and still run, and ``graph.rotary_tables``, where a wrong
answer is an array of the right shape full of the wrong numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from families.esm import config as config_module, graph, model as model_module
from tensorrt_model_connect import BuildRequest

_HIDDEN = 64
_INTERMEDIATE = 128
_LAYERS = 2
_HEADS = 4
_VOCAB = 33


def _weights(prefix: str = "esm.", **overrides) -> dict:
    weights = {
        f"{prefix}embeddings.word_embeddings.weight": np.zeros((_VOCAB, _HIDDEN), np.float32),
        f"{prefix}encoder.emb_layer_norm_after.weight": np.zeros((_HIDDEN,), np.float32),
        f"{prefix}encoder.emb_layer_norm_after.bias": np.zeros((_HIDDEN,), np.float32),
        "lm_head.dense.weight": np.zeros((_HIDDEN, _HIDDEN), np.float32),
        "lm_head.dense.bias": np.zeros((_HIDDEN,), np.float32),
        "lm_head.layer_norm.weight": np.zeros((_HIDDEN,), np.float32),
        "lm_head.layer_norm.bias": np.zeros((_HIDDEN,), np.float32),
        "lm_head.decoder.weight": np.zeros((_VOCAB, _HIDDEN), np.float32),
        "lm_head.bias": np.zeros((_VOCAB,), np.float32),
    }
    for index in range(_LAYERS):
        layer = f"{prefix}encoder.layer.{index}."
        weights[f"{layer}intermediate.dense.weight"] = np.zeros((_INTERMEDIATE, _HIDDEN),
                                                                np.float32)
    weights.update(overrides)
    return weights


def _config(tmp_path: Path, **overrides) -> Path:
    body = {
        "model_type": "esm",
        "hidden_size": _HIDDEN,
        "intermediate_size": _INTERMEDIATE,
        "num_hidden_layers": _LAYERS,
        "num_attention_heads": _HEADS,
        "vocab_size": _VOCAB,
        "position_embedding_type": "rotary",
        "emb_layer_norm_before": False,
        "token_dropout": True,
        "mask_token_id": 32,
        "pad_token_id": 1,
        "layer_norm_eps": 1e-5,
        "max_position_embeddings": 1026,
    }
    body.update(overrides)
    (tmp_path / "config.json").write_text(json.dumps(body), encoding="utf-8")
    return tmp_path


def test_geometry_comes_from_the_tensors(tmp_path: Path) -> None:
    cfg = config_module.resolve(_config(tmp_path), _weights(), "esm.")
    assert cfg["hidden_size"] == _HIDDEN
    assert cfg["intermediate_size"] == _INTERMEDIATE
    assert cfg["num_layers"] == _LAYERS
    assert cfg["num_heads"] == _HEADS
    assert cfg["head_dim"] == _HIDDEN // _HEADS
    assert cfg["token_dropout"] is True


def test_a_config_that_contradicts_the_tensors_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        config_module.resolve(_config(tmp_path, hidden_size=_HIDDEN + 8), _weights(), "esm.")


def test_absolute_positions_are_refused(tmp_path: Path) -> None:
    # ESM-1b and ESM-1v are "esm" too, and carry a learned position table.
    # Building them as rotary would place every residue wrongly.
    with pytest.raises(NotImplementedError, match="rotary"):
        config_module.resolve(_config(tmp_path, position_embedding_type="absolute"),
                              _weights(), "esm.")


def test_a_pre_embedding_layer_norm_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="emb_layer_norm_before"):
        config_module.resolve(_config(tmp_path, emb_layer_norm_before=True), _weights(), "esm.")


def test_the_folding_variant_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="ESMFold"):
        config_module.resolve(_config(tmp_path, is_folding_model=True), _weights(), "esm.")


def test_an_odd_head_width_is_refused(tmp_path: Path) -> None:
    # The rotation pairs the two halves of a head, so an odd width has no
    # pairing at all. 64 heads over 64 columns divides evenly and still
    # leaves one column per head.
    with pytest.raises(NotImplementedError, match="even head width"):
        config_module.resolve(_config(tmp_path, num_attention_heads=_HIDDEN),
                              _weights(), "esm.")


def test_a_mask_token_outside_the_vocabulary_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mask_token_id"):
        config_module.resolve(_config(tmp_path, mask_token_id=_VOCAB + 1), _weights(), "esm.")


def test_non_contiguous_layer_indices_are_refused(tmp_path: Path) -> None:
    weights = _weights()
    weights["esm.encoder.layer.7.intermediate.dense.weight"] = np.zeros(
        (_INTERMEDIATE, _HIDDEN), np.float32)
    with pytest.raises(ValueError, match="contiguous"):
        config_module.resolve(_config(tmp_path), weights, "esm.")


def test_the_rotary_tables_start_at_the_identity() -> None:
    cos, sin = graph.rotary_tables(8, 16)
    assert cos.shape == (1, 8, 16) and sin.shape == (1, 8, 16)
    # Position 0 rotates by nothing at all: every cosine is one and every
    # sine is zero. A table built from positions 1..n would fail here.
    assert np.allclose(cos[0, 0], np.ones(16))
    assert np.allclose(sin[0, 0], np.zeros(16))


def test_the_rotary_tables_mirror_their_halves() -> None:
    # emb = cat(freqs, freqs): the second half repeats the first, which is
    # what pairs the table with the rotate_half convention. A table built
    # from interleaved frequencies has the same shape and the wrong values.
    cos, sin = graph.rotary_tables(5, 12)
    assert np.allclose(cos[0, :, :6], cos[0, :, 6:])
    assert np.allclose(sin[0, :, :6], sin[0, :, 6:])


def test_the_rotary_schedule_is_the_roformer_one() -> None:
    head_dim = 8
    cos, _ = graph.rotary_tables(3, head_dim)
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2) / head_dim))
    assert np.allclose(cos[0, 2, :4], np.cos(2.0 * inv_freq), atol=1e-6)


def test_fp16_is_refused(tmp_path: Path) -> None:
    request = BuildRequest(model_dir=tmp_path, output_path=tmp_path / "b.bundle", family="esm",
                           task="masked_text_to_token_scores", precision="fp16",
                           max_sequence_length=32)
    with pytest.raises(NotImplementedError, match="fp32"):
        model_module.build(request, object())


def test_a_sequence_length_is_required(tmp_path: Path) -> None:
    request = BuildRequest(model_dir=tmp_path, output_path=tmp_path / "b.bundle", family="esm",
                           task="masked_text_to_token_scores", precision="fp32")
    with pytest.raises(ValueError, match="max_sequence_length"):
        model_module.build(request, object())
