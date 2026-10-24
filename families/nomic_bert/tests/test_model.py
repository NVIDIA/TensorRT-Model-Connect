# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-side unit tests for nomic-embed-text.

Two subjects: ``config.resolve``, which reads a GPT-2-shaped config that
states most values twice, and ``rope.tables``, where the difference between
a correct and an incorrect table is values rather than shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from families.nomic_bert import config as config_module, model as model_module, rope
from tensorrt_model_connect import BuildRequest

_HIDDEN = 128
_INTERMEDIATE = 256
_LAYERS = 2
_HEADS = 4
_VOCAB = 64


def _weights(**overrides) -> dict:
    weights = {
        "embeddings.word_embeddings.weight": np.zeros((_VOCAB, _HIDDEN), np.float32),
        "embeddings.token_type_embeddings.weight": np.zeros((2, _HIDDEN), np.float32),
        "emb_ln.weight": np.zeros((_HIDDEN,), np.float32),
        "emb_ln.bias": np.zeros((_HIDDEN,), np.float32),
    }
    for index in range(_LAYERS):
        prefix = f"encoder.layers.{index}."
        weights[f"{prefix}attn.Wqkv.weight"] = np.zeros((3 * _HIDDEN, _HIDDEN), np.float32)
        weights[f"{prefix}attn.out_proj.weight"] = np.zeros((_HIDDEN, _HIDDEN), np.float32)
        weights[f"{prefix}mlp.fc11.weight"] = np.zeros((_INTERMEDIATE, _HIDDEN), np.float32)
        weights[f"{prefix}mlp.fc12.weight"] = np.zeros((_INTERMEDIATE, _HIDDEN), np.float32)
        weights[f"{prefix}mlp.fc2.weight"] = np.zeros((_HIDDEN, _INTERMEDIATE), np.float32)
        weights[f"{prefix}norm1.weight"] = np.zeros((_HIDDEN,), np.float32)
        weights[f"{prefix}norm1.bias"] = np.zeros((_HIDDEN,), np.float32)
        weights[f"{prefix}norm2.weight"] = np.zeros((_HIDDEN,), np.float32)
        weights[f"{prefix}norm2.bias"] = np.zeros((_HIDDEN,), np.float32)
    weights.update(overrides)
    return weights


_BASE = {
    "activation_function": "swiglu",
    "num_attention_heads": _HEADS,
    "type_vocab_size": 2,
    "max_trained_positions": 2048,
    "rotary_emb_base": 1000,
}


def _model_dir(tmp_path: Path, config: dict | None = None) -> Path:
    merged = dict(_BASE)
    merged.update(config or {})
    (tmp_path / "config.json").write_text(json.dumps(merged))
    return tmp_path


def test_geometry_comes_from_the_tensors(tmp_path: Path) -> None:
    cfg = config_module.resolve(_model_dir(tmp_path), _weights())
    assert cfg["hidden_size"] == _HIDDEN
    assert cfg["intermediate_size"] == _INTERMEDIATE
    assert cfg["num_layers"] == _LAYERS
    assert cfg["vocab_size"] == _VOCAB
    assert cfg["head_dim"] == _HIDDEN // _HEADS


def test_a_config_that_contradicts_the_tensors_is_refused(tmp_path: Path) -> None:
    # n_embd and hidden_size say the same thing here; either one disagreeing
    # with the tensors is a reason to stop rather than to pick a winner.
    with pytest.raises(ValueError, match="tensors say"):
        config_module.resolve(_model_dir(tmp_path, {"n_embd": _HIDDEN * 2}), _weights())


def test_a_head_count_that_does_not_divide_the_width_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="divisible"):
        config_module.resolve(_model_dir(tmp_path, {"num_attention_heads": 7}), _weights())


def test_a_gelu_mlp_is_refused(tmp_path: Path) -> None:
    # A gelu block has one matrix where this has two, so the weights would
    # not even be read the same way.
    with pytest.raises(NotImplementedError, match="swiglu"):
        config_module.resolve(_model_dir(tmp_path, {"activation_function": "gelu"}), _weights())


def test_a_prenorm_checkpoint_is_refused(tmp_path: Path) -> None:
    # Measured, reading these weights pre-norm moves the features by 184.
    with pytest.raises(NotImplementedError, match="post-norm"):
        config_module.resolve(_model_dir(tmp_path, {"prenorm": True}), _weights())


def test_the_interleaved_rotary_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="non-interleaved"):
        config_module.resolve(_model_dir(tmp_path, {"rotary_emb_interleaved": True}), _weights())


def test_the_dynamic_ntk_rotary_is_refused(tmp_path: Path) -> None:
    # Its base depends on the input length, and this family bakes the table
    # at build time, so there is no length to bake it for.
    with pytest.raises(NotImplementedError, match="dynamic-NTK"):
        config_module.resolve(_model_dir(tmp_path, {"rotary_scaling_factor": 2.0}), _weights())


def test_a_partial_rotary_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="whole head"):
        config_module.resolve(_model_dir(tmp_path, {"rotary_emb_fraction": 0.5}), _weights())


def test_a_stated_bias_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="biasless"):
        config_module.resolve(_model_dir(tmp_path, {"qkv_proj_bias": True}), _weights())


def test_a_bias_tensor_the_config_denies_is_refused(tmp_path: Path) -> None:
    # The flag and the tensors are two separate claims, and a checkpoint that
    # carries a bias while saying it has none is not one this family can read.
    weights = _weights(**{"encoder.layers.0.mlp.fc2.bias": np.zeros((_HIDDEN,), np.float32)})
    with pytest.raises(ValueError, match="should not exist"):
        config_module.resolve(_model_dir(tmp_path), weights)


def test_a_position_table_is_refused(tmp_path: Path) -> None:
    # Positions come from the rotary here. A checkpoint with a learned table
    # is a different model wearing the same config.
    weights = _weights(**{"embeddings.position_embeddings.weight":
                          np.zeros((512, _HIDDEN), np.float32)})
    with pytest.raises(NotImplementedError, match="rotary positions"):
        config_module.resolve(_model_dir(tmp_path), weights)


def test_non_contiguous_layer_indices_are_refused(tmp_path: Path) -> None:
    weights = _weights()
    weights["encoder.layers.7.mlp.fc11.weight"] = np.zeros((_INTERMEDIATE, _HIDDEN), np.float32)
    with pytest.raises(ValueError, match="contiguous"):
        config_module.resolve(_model_dir(tmp_path), weights)


def test_the_mixture_of_experts_variant_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="mixture-of-experts"):
        config_module.resolve(_model_dir(tmp_path, {"moe_every_n_layers": 2}), _weights())


def test_the_rotary_table_is_tiled_not_interleaved() -> None:
    """The split-half convention, which is a layout rather than a shape.

    ``cos`` repeats each frequency once in the second half of the head, so
    column ``d`` and column ``d + head_dim/2`` carry the same value. The
    interleaved layout puts them side by side instead. Both tables are
    ``[seq, head_dim]``, so nothing downstream would complain.
    """
    cos, sin = rope.tables(8, 16, 1000.0)
    assert cos.shape == (8, 16) and sin.shape == (8, 16)
    assert np.array_equal(cos[:, :8], cos[:, 8:])
    assert np.array_equal(sin[:, :8], sin[:, 8:])
    assert not np.allclose(cos[:, 0], cos[:, 1])


def test_the_rotary_table_uses_the_checkpoints_base() -> None:
    # Position 1, frequency 0 is cos(1 / base**0) = cos(1) whatever the base,
    # so the base has to be read where it actually shows: a later frequency.
    for base in (1000.0, 10000.0):
        cos, _ = rope.tables(4, 16, base)
        expected = np.cos(1.0 / base ** (2.0 / 16.0))
        assert np.isclose(cos[1, 1], expected, atol=1e-6)
    assert not np.isclose(rope.tables(4, 16, 1000.0)[0][1, 1],
                          rope.tables(4, 16, 10000.0)[0][1, 1])


def _request(tmp_path: Path, **changes) -> BuildRequest:
    base = {
        "model_dir": tmp_path,
        "output_path": tmp_path / "nomic.bundle",
        "family": "nomic_bert",
        "task": "text_to_embedding",
        "precision": "fp32",
        "max_sequence_length": 512,
    }
    base.update(changes)
    return BuildRequest(**base)


def test_fp16_is_refused(tmp_path: Path) -> None:
    # Same rule as the vision families next door: no fp16 number has been
    # measured against the reference, so fp16 is refused rather than built
    # into a bundle nothing checks.
    with pytest.raises(NotImplementedError, match="fp32 only"):
        model_module.build(_request(tmp_path, precision="fp16"), writer=None)


def test_a_sequence_length_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_sequence_length"):
        model_module.build(_request(tmp_path, max_sequence_length=None), writer=None)
