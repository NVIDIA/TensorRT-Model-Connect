# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-side unit tests for GTE-v1.5.

Two subjects: ``config.resolve``, which reads a config whose switches change
the arithmetic without changing any shape, and ``rope.tables``, where the
difference between a correct and an incorrect table is values rather than
shape - and where the reference's float32 arithmetic has to be reproduced
rather than improved on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from families.gte import config as config_module, model as model_module, rope
from tensorrt_model_connect import BuildRequest

_HIDDEN = 128
_INTERMEDIATE = 256
_LAYERS = 2
_HEADS = 4
_VOCAB = 64


def _weights(**overrides) -> dict:
    weights = {
        "embeddings.word_embeddings.weight": np.zeros((_VOCAB, _HIDDEN), np.float32),
        "embeddings.LayerNorm.weight": np.zeros((_HIDDEN,), np.float32),
        "embeddings.LayerNorm.bias": np.zeros((_HIDDEN,), np.float32),
    }
    for index in range(_LAYERS):
        prefix = f"encoder.layer.{index}."
        weights[f"{prefix}attention.qkv_proj.weight"] = np.zeros((3 * _HIDDEN, _HIDDEN), np.float32)
        weights[f"{prefix}attention.qkv_proj.bias"] = np.zeros((3 * _HIDDEN,), np.float32)
        weights[f"{prefix}attention.o_proj.weight"] = np.zeros((_HIDDEN, _HIDDEN), np.float32)
        weights[f"{prefix}attention.o_proj.bias"] = np.zeros((_HIDDEN,), np.float32)
        weights[f"{prefix}mlp.up_gate_proj.weight"] = np.zeros(
            (2 * _INTERMEDIATE, _HIDDEN), np.float32)
        weights[f"{prefix}mlp.down_proj.weight"] = np.zeros((_HIDDEN, _INTERMEDIATE), np.float32)
        weights[f"{prefix}mlp.down_proj.bias"] = np.zeros((_HIDDEN,), np.float32)
        weights[f"{prefix}attn_ln.weight"] = np.zeros((_HIDDEN,), np.float32)
        weights[f"{prefix}attn_ln.bias"] = np.zeros((_HIDDEN,), np.float32)
        weights[f"{prefix}mlp_ln.weight"] = np.zeros((_HIDDEN,), np.float32)
        weights[f"{prefix}mlp_ln.bias"] = np.zeros((_HIDDEN,), np.float32)
    weights.update(overrides)
    return weights


_BASE = {
    "model_type": "new",
    "hidden_act": "gelu",
    "layer_norm_type": "layer_norm",
    "position_embedding_type": "rope",
    "num_attention_heads": _HEADS,
    "type_vocab_size": 0,
    "max_position_embeddings": 8192,
    "rope_theta": 500000,
    "rope_scaling": {"type": "ntk", "factor": 2.0},
    "pack_qkv": True,
}


def _model_dir(tmp_path: Path, config: dict | None = None) -> Path:
    merged = dict(_BASE)
    merged.update(config or {})
    (tmp_path / "config.json").write_text(json.dumps(merged))
    return tmp_path


def test_geometry_comes_from_the_tensors(tmp_path: Path) -> None:
    cfg = config_module.resolve(_model_dir(tmp_path), _weights())
    assert cfg["hidden_size"] == _HIDDEN
    # The intermediate size is half of up_gate_proj, not the whole of it.
    assert cfg["intermediate_size"] == _INTERMEDIATE
    assert cfg["num_layers"] == _LAYERS
    assert cfg["vocab_size"] == _VOCAB
    assert cfg["head_dim"] == _HIDDEN // _HEADS
    assert cfg["rope_theta"] == 500000.0
    assert cfg["rope_scaling_factor"] == 2.0


def test_a_config_that_contradicts_the_tensors_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tensors say"):
        config_module.resolve(_model_dir(tmp_path, {"hidden_size": _HIDDEN * 2}), _weights())


def test_a_stated_intermediate_size_that_doubles_the_stack_is_refused(tmp_path: Path) -> None:
    # up_gate_proj is two matrices stacked, so a reader who took its row
    # count as the intermediate size would be out by exactly 2x. The config
    # states the real one, and the two have to agree.
    with pytest.raises(ValueError, match="tensors say"):
        config_module.resolve(
            _model_dir(tmp_path, {"intermediate_size": 2 * _INTERMEDIATE}), _weights())


def test_a_head_count_that_does_not_divide_the_width_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="divisible"):
        config_module.resolve(_model_dir(tmp_path, {"num_attention_heads": 7}), _weights())


def test_a_tanh_gelu_checkpoint_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="exact gelu"):
        config_module.resolve(_model_dir(tmp_path, {"hidden_act": "gelu_new"}), _weights())


def test_an_rms_norm_checkpoint_is_refused(tmp_path: Path) -> None:
    # RMSNorm has no bias and does not centre, so reading these weights as
    # one would be a different function of the same numbers.
    with pytest.raises(NotImplementedError, match="layer_norm"):
        config_module.resolve(_model_dir(tmp_path, {"layer_norm_type": "rms_norm"}), _weights())


def test_an_absolute_position_checkpoint_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="rotary"):
        config_module.resolve(
            _model_dir(tmp_path, {"position_embedding_type": "absolute"}), _weights())


def test_a_position_table_is_refused(tmp_path: Path) -> None:
    weights = _weights(**{"embeddings.position_embeddings.weight":
                          np.zeros((512, _HIDDEN), np.float32)})
    with pytest.raises(NotImplementedError, match="rotary positions"):
        config_module.resolve(_model_dir(tmp_path), weights)


def test_a_token_type_vocabulary_is_refused(tmp_path: Path) -> None:
    # type_vocab_size is 0 on this checkpoint, so the reference never builds
    # a token type table. A checkpoint that had one would need the extra add
    # this family does not make.
    with pytest.raises(NotImplementedError, match="token-type-free"):
        config_module.resolve(_model_dir(tmp_path, {"type_vocab_size": 2}), _weights())


def test_a_token_type_table_the_config_denies_is_refused(tmp_path: Path) -> None:
    weights = _weights(**{"embeddings.token_type_embeddings.weight":
                          np.zeros((2, _HIDDEN), np.float32)})
    with pytest.raises(ValueError, match="token_type_embeddings"):
        config_module.resolve(_model_dir(tmp_path), weights)


def test_the_logn_attention_scale_is_refused(tmp_path: Path) -> None:
    # Its query multiplier depends on the input length, which a fixed-length
    # engine with build-time constants cannot carry.
    with pytest.raises(NotImplementedError, match="logn"):
        config_module.resolve(_model_dir(tmp_path, {"logn_attention_scale": True}), _weights())


def test_an_unpacked_qkv_claim_against_a_packed_tensor_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pack_qkv"):
        config_module.resolve(_model_dir(tmp_path, {"pack_qkv": False}), _weights())


def test_an_up_gate_bias_is_refused(tmp_path: Path) -> None:
    weights = _weights(**{"encoder.layer.0.mlp.up_gate_proj.bias":
                          np.zeros((2 * _INTERMEDIATE,), np.float32)})
    with pytest.raises(ValueError, match="up_gate_proj.bias"):
        config_module.resolve(_model_dir(tmp_path), weights)


def test_a_missing_projection_bias_is_refused(tmp_path: Path) -> None:
    # Every projection but up_gate_proj has a bias here. A checkpoint
    # without one would build and would quietly drop a term per layer.
    weights = _weights()
    del weights["encoder.layer.0.attention.o_proj.bias"]
    with pytest.raises(KeyError, match="o_proj.bias"):
        config_module.resolve(_model_dir(tmp_path), weights)


def test_non_contiguous_layer_indices_are_refused(tmp_path: Path) -> None:
    weights = _weights()
    weights["encoder.layer.7.mlp.down_proj.weight"] = np.zeros(
        (_HIDDEN, _INTERMEDIATE), np.float32)
    with pytest.raises(ValueError, match="contiguous"):
        config_module.resolve(_model_dir(tmp_path), weights)


def test_an_unknown_rope_scaling_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="ntk"):
        config_module.resolve(
            _model_dir(tmp_path, {"rope_scaling": {"type": "linear", "factor": 2.0}}), _weights())


def test_the_mixed_b_ntk_rotary_is_refused(tmp_path: Path) -> None:
    # Mixed NTK divides each frequency by its own lambda rather than by one
    # factor; same shape, different table.
    with pytest.raises(NotImplementedError, match="mixed-b"):
        config_module.resolve(
            _model_dir(tmp_path, {"rope_scaling": {"type": "ntk", "factor": 2.0, "mixed_b": 0.3}}),
            _weights())


def test_the_rotary_table_is_tiled_not_interleaved() -> None:
    """The split-half convention, which is a layout rather than a shape.

    ``cos`` repeats each frequency once in the second half of the head, so
    column ``d`` and column ``d + head_dim/2`` carry the same value. The
    interleaved layout puts them side by side instead. Both tables are
    ``[seq, head_dim]``, so nothing downstream would complain.
    """
    cos, sin = rope.tables(8, 16, 500000.0, 2.0)
    assert cos.shape == (8, 16) and sin.shape == (8, 16)
    assert np.array_equal(cos[:, :8], cos[:, 8:])
    assert np.array_equal(sin[:, :8], sin[:, 8:])
    assert not np.allclose(cos[:, 0], cos[:, 1])


def test_the_ntk_scaling_actually_moves_the_table() -> None:
    """factor 2.0 is not a no-op, and it is not the same as doubling theta.

    The reference's NTK branch multiplies the base *and* divides every
    frequency by ``factor ** (2 / head_dim)``. Doing only the first is the
    mistake that looks right.
    """
    head_dim = 64
    scaled = rope.inverse_frequencies(head_dim, 500000.0, 2.0)
    unscaled = rope.inverse_frequencies(head_dim, 500000.0, 1.0)
    base_only = rope.inverse_frequencies(head_dim, 1000000.0, 1.0)
    assert not np.allclose(scaled, unscaled)
    assert not np.allclose(scaled, base_only)
    assert np.allclose(scaled, base_only / np.float32(2.0 ** (2.0 / head_dim)), atol=0, rtol=1e-6)
    # Frequency 0 has exponent 0, so the base cancels and only the division
    # by the factor is left - the one term a base-only reading would lose.
    assert scaled[0] == pytest.approx(1.0 / 2.0 ** (2.0 / head_dim), rel=1e-6)


def test_the_rotary_table_is_built_in_float32_like_the_reference() -> None:
    """Not float64, which would be a more accurate table and a different one.

    Measured against the reference module's own ``cos_cached``, the float32
    construction agrees to 5.96e-08 over all 16384 cached rows while a
    float64 construction differs by 3.9e-05 at 512 positions. The reference
    ships no ``inv_freq`` buffer - it is registered ``persistent=False`` -
    so there is nothing to load instead of computing, and the computation
    is made the reference's.

    This is the one rule in this family whose break the E2E gate does not
    catch: building the float64 table on purpose moved the pooled embedding
    by 2.86e-06, which is the correct build's own noise. The assertion
    below is therefore on the table, where the difference is real, and not
    dressed up as an accuracy claim about the model.
    """
    head_dim = 64
    cos, sin = rope.tables(512, head_dim, 500000.0, 2.0)
    assert cos.dtype == np.float32 and sin.dtype == np.float32
    assert rope.inverse_frequencies(head_dim, 500000.0, 2.0).dtype == np.float32

    inv_freq = rope.inverse_frequencies(head_dim, 500000.0, 2.0)
    wide = np.outer(np.arange(512, dtype=np.float64), inv_freq.astype(np.float64))
    wide_cos = np.concatenate([np.cos(wide), np.cos(wide)], axis=-1)
    # The two tables differ by far more than float32 noise: they are two
    # different answers, and only one of them is what the reference rotates
    # the queries by.
    assert np.abs(wide_cos - cos).max() > 1e-6


def _request(tmp_path: Path, **changes) -> BuildRequest:
    base = {
        "model_dir": tmp_path,
        "output_path": tmp_path / "gte.bundle",
        "family": "gte",
        "task": "text_to_embedding",
        "precision": "fp32",
        "max_sequence_length": 512,
    }
    base.update(changes)
    return BuildRequest(**base)


def test_fp16_is_refused(tmp_path: Path) -> None:
    # No fp16 number has been measured against the reference, so fp16 is
    # refused rather than built into a bundle nothing checks.
    with pytest.raises(NotImplementedError, match="fp32 only"):
        model_module.build(_request(tmp_path, precision="fp16"), writer=None)


def test_a_sequence_length_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_sequence_length"):
        model_module.build(_request(tmp_path, max_sequence_length=None), writer=None)
