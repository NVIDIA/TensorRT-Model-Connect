# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the diffusion transformer's geometry."""

from __future__ import annotations

import importlib
import math

import pytest

np = pytest.importorskip("numpy")

dit = importlib.import_module("tensorrt_model_connect.families.minimax_music3.dit")


def test_widths_match_the_published_tensor_shapes() -> None:
    """proj_in.weight is [2048, 2304] and preprocess_conv is [2304, 2304, 1]."""

    assert dit.INNER_DIM == 2048
    assert dit.CONCAT_CHANNELS == 2304
    assert dit.CONCAT_CHANNELS == 2 * dit.IN_CHANNELS + dit.CONDITION_DIM


def test_the_zero_block_is_what_makes_the_input_2304_wide() -> None:
    """Latents, a zero block of the same width, then the condition."""

    assert dit.CONCAT_CHANNELS - dit.CONDITION_DIM == 2 * dit.IN_CHANNELS


def test_attention_length_includes_the_timestep_prefix() -> None:
    assert dit.sequence_length(689) == 690
    with pytest.raises(ValueError, match="must be positive"):
        dit.sequence_length(0)


def test_fourier_projection_doubles_into_the_embedding_width() -> None:
    """time_embed.linear_1.weight is [2048, 256] and time_proj.weight is [128, 1]."""

    assert dit.fourier_weight_rows() == 128
    assert 2 * dit.fourier_weight_rows() == dit.FOURIER_EMBEDDING_DIM


def test_fourier_features_are_cosine_then_sine() -> None:
    weight = np.array([[0.5], [1.0]], dtype=np.float32)

    features = dit.fourier_features(np.array([0.25]), weight)

    assert features.shape == (1, 4)
    angles = 2.0 * math.pi * 0.25 * np.array([0.5, 1.0])
    assert np.allclose(features[0, :2], np.cos(angles), atol=1e-6)
    assert np.allclose(features[0, 2:], np.sin(angles), atol=1e-6)


def test_fourier_features_are_unit_norm_per_pair() -> None:
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((8, 1)).astype(np.float32)

    features = dit.fourier_features(np.array([0.7]), weight)
    cos, sin = np.split(features[0], 2)

    assert np.allclose(cos ** 2 + sin ** 2, 1.0, atol=1e-6)


def test_rotary_tables_are_as_wide_as_the_rotated_slice() -> None:
    cos, sin = dit.rotary_tables(690)

    assert cos.shape == (690, dit.ROTARY_DIM)
    assert sin.shape == (690, dit.ROTARY_DIM)
    # Duplicated halves: the table repeats after rotary_dim / 2.
    assert np.allclose(cos[:, : dit.ROTARY_DIM // 2], cos[:, dit.ROTARY_DIM // 2:])


def test_rotary_position_zero_is_the_identity() -> None:
    cos, sin = dit.rotary_tables(4)

    assert np.allclose(cos[0], 1.0, atol=1e-6)
    assert np.allclose(sin[0], 0.0, atol=1e-6)


def test_partial_rope_leaves_the_tail_untouched() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, dit.NUM_ATTENTION_HEADS,
                             dit.ATTENTION_HEAD_DIM)).astype(np.float32)
    cos, sin = dit.rotary_tables(6)

    out = dit.apply_partial_rope(x, cos, sin)

    assert out.shape == x.shape
    assert np.allclose(out[..., dit.ROTARY_DIM:], x[..., dit.ROTARY_DIM:])
    assert not np.allclose(out[..., : dit.ROTARY_DIM], x[..., : dit.ROTARY_DIM])


def test_partial_rope_preserves_the_norm_of_the_rotated_slice() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5, 2, dit.ATTENTION_HEAD_DIM)).astype(np.float32)
    cos, sin = dit.rotary_tables(5)

    out = dit.apply_partial_rope(x, cos, sin)

    before = np.linalg.norm(x[..., : dit.ROTARY_DIM], axis=-1)
    after = np.linalg.norm(out[..., : dit.ROTARY_DIM], axis=-1)
    assert np.allclose(before, after, atol=1e-4)


def test_position_zero_rope_is_a_no_op() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((1, 2, dit.ATTENTION_HEAD_DIM)).astype(np.float32)
    cos, sin = dit.rotary_tables(1)

    assert np.allclose(dit.apply_partial_rope(x, cos, sin), x, atol=1e-6)


def test_attention_scale() -> None:
    assert dit.attention_scale() == pytest.approx(64 ** -0.5)


def test_layer_and_head_counts_match_the_checkpoint() -> None:
    """The published transformer shard carries 19 blocks of a 36-layer stack."""

    assert dit.NUM_LAYERS == 36
    assert dit.NUM_ATTENTION_HEADS * dit.ATTENTION_HEAD_DIM == dit.INNER_DIM
    # ff_in.weight is [16384, 2048]: twice the inner feed-forward width.
    assert 2 * dit.FF_INNER_DIM == 16384
