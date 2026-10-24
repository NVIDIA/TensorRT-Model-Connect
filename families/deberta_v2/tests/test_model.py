# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-side unit tests for DeBERTa-v2/v3.

Two subjects: ``config.resolve``, which has to refuse several variants that
load every tensor and still run, and ``rel_position``, where a wrong answer
is a matrix of the right shape full of the wrong integers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from families.deberta_v2 import config as config_module, model as model_module, rel_position
from tensorrt_model_connect import BuildRequest

_HIDDEN = 128
_INTERMEDIATE = 256
_LAYERS = 2
_HEADS = 4
_VOCAB = 64
_SPAN = 16


def _weights(prefix: str = "deberta.", **overrides) -> dict:
    weights = {
        f"{prefix}embeddings.word_embeddings.weight": np.zeros((_VOCAB, _HIDDEN), np.float32),
        f"{prefix}embeddings.LayerNorm.weight": np.zeros((_HIDDEN,), np.float32),
        f"{prefix}embeddings.LayerNorm.bias": np.zeros((_HIDDEN,), np.float32),
        f"{prefix}encoder.rel_embeddings.weight": np.zeros((2 * _SPAN, _HIDDEN), np.float32),
        f"{prefix}encoder.LayerNorm.weight": np.zeros((_HIDDEN,), np.float32),
        f"{prefix}encoder.LayerNorm.bias": np.zeros((_HIDDEN,), np.float32),
    }
    for index in range(_LAYERS):
        layer = f"{prefix}encoder.layer.{index}."
        weights[f"{layer}intermediate.dense.weight"] = np.zeros((_INTERMEDIATE, _HIDDEN),
                                                                np.float32)
    weights.update(overrides)
    return weights


_BASE = {
    "num_attention_heads": _HEADS,
    "relative_attention": True,
    "position_biased_input": False,
    "pos_att_type": "p2c|c2p",
    "share_att_key": True,
    "norm_rel_ebd": "layer_norm",
    "position_buckets": _SPAN,
    "max_position_embeddings": 512,
    "type_vocab_size": 0,
}


def _model_dir(tmp_path: Path, config: dict | None = None) -> Path:
    merged = dict(_BASE)
    merged.update(config or {})
    (tmp_path / "config.json").write_text(json.dumps(merged))
    return tmp_path


def test_geometry_comes_from_the_tensors(tmp_path: Path) -> None:
    cfg = config_module.resolve(_model_dir(tmp_path), _weights(), "deberta.")
    assert cfg["hidden_size"] == _HIDDEN
    assert cfg["intermediate_size"] == _INTERMEDIATE
    assert cfg["num_layers"] == _LAYERS
    assert cfg["head_dim"] == _HIDDEN // _HEADS
    assert cfg["span"] == _SPAN
    # One content term plus one per positional term, and the attention scale
    # is built from it rather than from the head width alone.
    assert cfg["scale_factor"] == 3


def test_a_config_that_contradicts_the_tensors_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tensors say"):
        config_module.resolve(_model_dir(tmp_path, {"hidden_size": _HIDDEN * 2}),
                              _weights(), "deberta.")


def test_absolute_position_embeddings_are_refused(tmp_path: Path) -> None:
    # The table is in every v3 checkpoint whether it is used or not, so this
    # has to be read from the config. Measured, adding it costs 4.80.
    with pytest.raises(NotImplementedError, match="absolute position"):
        config_module.resolve(_model_dir(tmp_path, {"position_biased_input": True}),
                              _weights(), "deberta.")


def test_a_single_positional_term_is_refused(tmp_path: Path) -> None:
    # Not a subset of this model: the term count divides every score.
    with pytest.raises(NotImplementedError, match="c2p and p2c"):
        config_module.resolve(_model_dir(tmp_path, {"pos_att_type": "c2p"}),
                              _weights(), "deberta.")


def test_unshared_positional_projections_are_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="share_att_key"):
        config_module.resolve(_model_dir(tmp_path, {"share_att_key": False}),
                              _weights(), "deberta.")


def test_an_unnormalised_relative_table_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="layer-normalised"):
        config_module.resolve(_model_dir(tmp_path, {"norm_rel_ebd": "none"}),
                              _weights(), "deberta.")


def test_the_convolution_variant_is_refused(tmp_path: Path) -> None:
    # DeBERTa-v2-xlarge puts a convolution beside the first layer.
    with pytest.raises(NotImplementedError, match="convolution"):
        config_module.resolve(_model_dir(tmp_path, {"conv_kernel_size": 3}),
                              _weights(), "deberta.")


def test_token_type_embeddings_are_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="token type"):
        config_module.resolve(_model_dir(tmp_path, {"type_vocab_size": 2}),
                              _weights(), "deberta.")


def test_a_projected_embedding_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="embed_proj"):
        config_module.resolve(_model_dir(tmp_path, {"embedding_size": _HIDDEN // 2}),
                              _weights(), "deberta.")


def test_a_relative_table_of_the_wrong_height_is_refused(tmp_path: Path) -> None:
    weights = _weights(**{"deberta.encoder.rel_embeddings.weight":
                          np.zeros((2 * _SPAN + 1, _HIDDEN), np.float32)})
    with pytest.raises(ValueError, match="relative table"):
        config_module.resolve(_model_dir(tmp_path), weights, "deberta.")


def test_non_contiguous_layer_indices_are_refused(tmp_path: Path) -> None:
    weights = _weights()
    weights["deberta.encoder.layer.7.intermediate.dense.weight"] = np.zeros(
        (_INTERMEDIATE, _HIDDEN), np.float32)
    with pytest.raises(ValueError, match="contiguous"):
        config_module.resolve(_model_dir(tmp_path), weights, "deberta.")


def test_the_bucketing_is_the_identity_only_near_the_diagonal() -> None:
    """Why the shipped case is long.

    Below ``bucket_size / 2`` the bucket of a distance is the distance, so a
    short passage cannot tell a correct implementation from a plain clamp.
    Past it the two part company, and that is where the gate gets its grip.
    """
    buckets, maximum = 256, 512
    near = rel_position.relative_matrix(64, buckets, maximum)
    positions = np.arange(64)
    assert np.array_equal(near, positions[:, None] - positions[None, :])

    far = rel_position.relative_matrix(300, buckets, maximum)
    linear = np.arange(300)[:, None] - np.arange(300)[None, :]
    assert not np.array_equal(far, linear)
    # Compressed, never expanded: a bucket index is never further from the
    # diagonal than the distance it stands for.
    assert np.all(np.abs(far) <= np.abs(linear))


def test_the_gather_indices_are_mirror_images() -> None:
    """c2p reads the distance forwards; p2c reads it backwards."""
    content_to_position, position_to_content = rel_position.gather_indices(32, 256, 512, 256)
    assert content_to_position.shape == (32, 32)
    assert np.array_equal(content_to_position, position_to_content.T)
    assert content_to_position.min() >= 0
    assert content_to_position.max() <= 2 * 256 - 1


def _request(tmp_path: Path, **changes) -> BuildRequest:
    base = {
        "model_dir": tmp_path,
        "output_path": tmp_path / "deberta.bundle",
        "family": "deberta_v2",
        "task": "text_to_token_features",
        "precision": "fp32",
        "max_sequence_length": 256,
    }
    base.update(changes)
    return BuildRequest(**base)


def test_fp16_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="fp32 only"):
        model_module.build(_request(tmp_path, precision="fp16"), writer=None)


def test_a_sequence_length_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_sequence_length"):
        model_module.build(_request(tmp_path, max_sequence_length=None), writer=None)
