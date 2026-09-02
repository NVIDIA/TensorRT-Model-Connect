# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the condition-encoder oracle.

`forward` was checked against the reference module on an RTX 4090 with the
published weights: two 200-frame windows and one 500-frame window from the
recorded `frame_hiddens`, agreeing to a maximum relative error of 4e-06, which
is float32 accumulation noise. The resampling index map was checked against
torch over 220 input lengths with no disagreement.

The fixtures here stand in for that so the tests need neither torch nor the
checkpoint.
"""

from __future__ import annotations

import importlib

import pytest

np = pytest.importorskip("numpy")

ce = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.condition_encoder"
)


def test_resample_ratio_is_the_two_clocks() -> None:
    # 25 Hz in, 44100/512 Hz out.
    assert ce.resample_ratio() == pytest.approx(3.4453125)
    assert ce.INPUT_SAMPLING_RATE / ce.INPUT_HOP_LENGTH == pytest.approx(25.0)
    assert ce.OUTPUT_SAMPLING_RATE / ce.OUTPUT_HOP_LENGTH == pytest.approx(86.13, abs=0.01)


def test_a_window_becomes_the_recorded_latent_length() -> None:
    """A 200-frame window is 689 latent frames, which is what the run produced."""

    spec = importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.pipeline_spec"
    )
    parity = importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.parity"
    )

    assert ce.latent_length(spec.CHUNK_FRAMES) == 689
    assert parity.BASELINE_LATENT_SHAPE[2] == 689


def test_truncation_not_rounding() -> None:
    # 200 * 3.4453125 = 689.0625; rounding would give 689 too, so use a case
    # where they differ: 60 * 3.4453125 = 206.71875.
    assert ce.latent_length(60) == 206


def test_latent_length_rejects_an_empty_window() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ce.latent_length(0)


def test_nearest_indices_are_monotonic_and_in_range() -> None:
    frames = 200
    indices = ce.nearest_indices(frames)

    assert len(indices) == ce.latent_length(frames)
    assert indices[0] == 0
    assert all(0 <= i < frames for i in indices)
    assert all(b >= a for a, b in zip(indices, indices[1:]))


def test_exact_integer_boundary_follows_float32_not_arithmetic() -> None:
    """The reference computes the source index in single precision.

    For 500 input frames the output is 1722 long, and at position 861 the
    quotient 861 * 500 / 1722 is exactly 250. Exact integer arithmetic returns
    250; float32 lands a hair below and floors to 249, which is what the
    reference does and therefore what the engine must do. Checked against
    torch over 220 input lengths, with no disagreement.
    """

    indices = ce.nearest_indices(500)

    assert len(indices) == 1722
    assert indices[861] == 249
    assert (861 * 500) // 1722 == 250  # what exact arithmetic would have given


def test_nearest_indices_upsample_each_source_frame() -> None:
    """Every input frame is used, since the ratio is above one."""

    frames = 50
    assert set(ce.nearest_indices(frames)) == set(range(frames))


def _weights(rng, layers=ce.NUM_CONDITION_LAYERS):
    return (
        rng.standard_normal(layers).astype(np.float32),
        np.float32(1.0),
        rng.standard_normal(
            (ce.OUT_DIM, ce.CONDITION_HIDDEN_DIM, ce.PROJ_KERNEL_SIZE)
        ).astype(np.float32) * 0.01,
        rng.standard_normal(ce.OUT_DIM).astype(np.float32) * 0.01,
    )


def test_forward_shapes(monkeypatch) -> None:
    # Keep the test cheap: shrink the widths the oracle reads from constants.
    monkeypatch.setattr(ce, "CONDITION_HIDDEN_DIM", 8)
    monkeypatch.setattr(ce, "OUT_DIM", 4)
    rng = np.random.default_rng(0)
    frames = 20
    hidden = rng.standard_normal(
        (1, frames, ce.NUM_CONDITION_LAYERS * 8)
    ).astype(np.float32)
    logits = rng.standard_normal(ce.NUM_CONDITION_LAYERS).astype(np.float32)
    weight = rng.standard_normal((4, 8, 3)).astype(np.float32)
    bias = rng.standard_normal(4).astype(np.float32)

    out = ce.forward(hidden, logits, np.float32(1.0), weight, bias)

    assert out.shape == (1, ce.latent_length(frames), 4)


def test_forward_rejects_a_mismatched_width() -> None:
    hidden = np.zeros((1, 4, 123), dtype=np.float32)

    with pytest.raises(ValueError, match="expected"):
        ce.forward(hidden, np.zeros(8), np.float32(1.0), np.zeros((1, 1, 3)), np.zeros(1))


def test_uniform_logits_average_the_streams(monkeypatch) -> None:
    """Equal logits make the mixture the plain mean over the eight streams."""

    monkeypatch.setattr(ce, "CONDITION_HIDDEN_DIM", 2)
    monkeypatch.setattr(ce, "OUT_DIM", 2)
    frames = 4
    # Stream l is filled with the constant l, so the mean is 3.5.
    hidden = np.zeros((1, frames, 8 * 2), dtype=np.float32)
    for layer in range(8):
        hidden[:, :, layer * 2 : (layer + 1) * 2] = layer
    # Identity-ish projection: pick the centre tap only.
    weight = np.zeros((2, 2, 3), dtype=np.float32)
    weight[0, 0, 1] = 1.0
    weight[1, 1, 1] = 1.0

    out = ce.forward(
        hidden, np.zeros(8, dtype=np.float32), np.float32(1.0), weight,
        np.zeros(2, dtype=np.float32),
    )

    assert np.allclose(out, 3.5, atol=1e-5)


def test_layer_scale_multiplies_the_mixture(monkeypatch) -> None:
    monkeypatch.setattr(ce, "CONDITION_HIDDEN_DIM", 2)
    monkeypatch.setattr(ce, "OUT_DIM", 2)
    hidden = np.ones((1, 4, 16), dtype=np.float32)
    weight = np.zeros((2, 2, 3), dtype=np.float32)
    weight[0, 0, 1] = weight[1, 1, 1] = 1.0
    bias = np.zeros(2, dtype=np.float32)
    logits = np.zeros(8, dtype=np.float32)

    one = ce.forward(hidden, logits, np.float32(1.0), weight, bias)
    three = ce.forward(hidden, logits, np.float32(3.0), weight, bias)

    assert np.allclose(three, 3.0 * one, atol=1e-5)


def test_engine_io_shapes_match_the_baseline() -> None:
    parity = importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.parity"
    )
    shapes = ce.engine_io_shapes(200)

    assert shapes["hidden_states"] == (1, 200, parity.BASELINE_FRAME_HIDDENS_SHAPE[2])
    assert shapes["condition"] == (1, parity.BASELINE_LATENT_SHAPE[2], ce.OUT_DIM)
