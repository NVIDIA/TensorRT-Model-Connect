# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the vocoder geometry."""

from __future__ import annotations

import importlib

import pytest

np = pytest.importorskip("numpy")

voc = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.vocoder"
)


def test_blocks_halve_the_width_each_step() -> None:
    widths = [(b.input_dim, b.output_dim, b.stride) for b in voc.blocks()]

    assert widths == [
        (1536, 768, 8),
        (768, 384, 8),
        (384, 192, 4),
        (192, 96, 2),
    ]


def test_block_kernels_and_padding_match_the_checkpoint() -> None:
    first, *_, last = voc.blocks()

    # conv_t1.weight_v of block 0 is [1536, 768, 16]: kernel 2 * stride.
    assert (first.kernel_size, first.padding) == (16, 4)
    assert (last.kernel_size, last.padding) == (4, 1)


def test_conv_out_consumes_the_last_block_width() -> None:
    """conv_out.weight_v is [1, 96, 7]."""

    assert voc.blocks()[-1].output_dim == 96
    assert voc.OUTPUT_CHANNELS == 1


def test_stereo_is_two_folded_streams() -> None:
    """dec_in_proj.weight is [1024, 64, 1] while the config declares 128."""

    assert voc.STREAM_CHANNELS == 64
    assert voc.STREAMS * voc.STREAM_CHANNELS == voc.LATENT_CHANNELS


def test_each_block_is_an_exact_integer_upsample() -> None:
    for block in voc.blocks():
        for length in (1, 7, 100, 689):
            assert voc.block_output_length(length, block.stride) == length * block.stride


def test_upsample_factor_matches_the_condition_hop() -> None:
    condition = importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.condition_encoder"
    )

    assert voc.upsample_factor() == 512 == condition.OUTPUT_HOP_LENGTH


def test_recorded_window_decodes_to_a_whole_number_of_samples() -> None:
    parity = importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.parity"
    )

    assert voc.waveform_samples(parity.BASELINE_LATENT_SHAPE[2]) == 689 * 512


def test_the_stitched_windows_reproduce_the_recorded_sample_count() -> None:
    """The crop arithmetic has to land on 882688 samples, and it does."""

    spec = importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.pipeline_spec"
    )
    parity = importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.parity"
    )

    window = parity.BASELINE_LATENT_SHAPE[2]
    count = len(parity.BASELINE_CHUNK_STARTS)
    total = 0
    for index in range(count):
        left = 0 if index == 0 else spec.CROP_LEFT_LATENT
        right = 0 if index == count - 1 else spec.CROP_RIGHT_LATENT
        total += window - left - right

    assert total == 1724
    assert total * voc.upsample_factor() == parity.BASELINE_WAVEFORM_SAMPLES


def test_residual_padding_preserves_length() -> None:
    assert [voc.residual_padding(d) for d in voc.RESIDUAL_DILATIONS] == [3, 9, 27]


def test_fuse_weight_norm_reproduces_the_torch_parameterisation() -> None:
    rng = np.random.default_rng(0)
    v = rng.standard_normal((5, 3, 7)).astype(np.float32)
    g = rng.standard_normal((5, 1, 1)).astype(np.float32)

    fused = voc.fuse_weight_norm(g, v)

    expected = g.reshape(5, 1, 1) * v / np.sqrt((v * v).sum(axis=(1, 2), keepdims=True))
    assert fused.shape == v.shape
    assert np.allclose(fused, expected, atol=1e-6)


def test_fused_weight_rows_have_the_declared_magnitude() -> None:
    rng = np.random.default_rng(1)
    v = rng.standard_normal((4, 2, 3)).astype(np.float32)
    g = np.abs(rng.standard_normal((4, 1, 1)).astype(np.float32)) + 0.5

    fused = voc.fuse_weight_norm(g, v)
    norms = np.sqrt((fused * fused).sum(axis=(1, 2)))

    assert np.allclose(norms, g.reshape(-1), atol=1e-5)


def test_snake_is_identity_at_zero_and_grows_with_alpha() -> None:
    x = np.zeros((1, 2, 3), dtype=np.float32)
    alpha = np.ones((1, 2, 1), dtype=np.float32)

    assert np.allclose(voc.snake(x, alpha), 0.0, atol=1e-6)

    x = np.full((1, 2, 3), 0.5, dtype=np.float32)
    out = voc.snake(x, alpha)
    expected = 0.5 + np.sin(0.5) ** 2 / (1.0 + 1e-9)
    assert np.allclose(out, expected, atol=1e-6)
