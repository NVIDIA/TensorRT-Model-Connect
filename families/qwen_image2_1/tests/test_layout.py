# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the Qwen-Image 2.1 position, resampling and geometry helpers.

Each helper is compared with a NumPy transcription of the reference module it replaces
(``QwenImage21Rope``, ``QwenImage21DupUp3D``), written the way the reference computes it rather
than the way the family does.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from families.qwen_image2_1 import rope
from families.qwen_image2_1.bundle_config import (
    DEFAULT_TEXT_TOKENS,
    resolve_geometry,
    scheduler_settings,
)
from families.qwen_image2_1.resample import shortcut_channels

AXES = (16, 56, 56)


def _reference_rope_angles(text_length: int, height: int, width: int) -> np.ndarray:
    """``QwenImage21Rope.forward`` for text followed by one image block, as angles."""
    positive = np.arange(8192)
    negative = np.arange(1024)[::-1] * -1 - 1
    tables = []
    for axis_dim in AXES:
        inverse = np.float32(1.0) / np.power(
            np.float32(10000), np.arange(0, axis_dim, 2).astype(np.float32) / np.float32(axis_dim)
        )
        tables.append(
            np.concatenate(
                [
                    np.outer(positive.astype(np.float32), inverse),
                    np.outer(negative.astype(np.float32), inverse),
                ]
            ).astype(np.float32)
        )
    frame_index = list(range(text_length)) + [text_length] * (height * width)
    height_index = list(frame_index)
    width_index = list(frame_index)
    height_index[text_length:] = [
        h for h in range(-(height - height // 2), height // 2) for _ in range(width)
    ]
    width_index[text_length:] = [
        w for _ in range(height) for w in range(-(width - width // 2), width // 2)
    ]
    # Negative indices wrap to the table's negative-position rows, as in the reference.
    return np.concatenate(
        [tables[0][frame_index], tables[1][height_index], tables[2][width_index]], axis=-1
    )


@pytest.mark.parametrize(("text_length", "height", "width"), [(7, 4, 6), (33, 5, 3), (1, 64, 64)])
def test_rotary_angles_match_the_reference_layout(text_length: int, height: int, width: int) -> None:
    expected = _reference_rope_angles(text_length, height, width)
    text_slots = text_length + 5  # the engine's text budget exceeds the prompt
    text = rope.text_angles(text_slots, AXES)
    grid = rope.image_grid_angles(height, width, AXES)
    frame = np.float32(text_length) * rope.axis_inverse_frequencies(AXES[0])
    image = np.concatenate([np.broadcast_to(frame, (height * width, frame.size)), grid], axis=1)

    assert text.shape == (text_slots, sum(AXES) // 2) and text.dtype == np.float32
    assert image.dtype == np.float32
    np.testing.assert_array_equal(text[:text_length], expected[:text_length])
    np.testing.assert_array_equal(image, expected[text_length:])


def test_rotary_positions_outside_the_model_range_are_rejected() -> None:
    with pytest.raises(ValueError, match="outside the range"):
        rope.text_angles(rope.MAX_POSITIVE_POSITION + 1, AXES)
    with pytest.raises(ValueError, match="outside the range"):
        rope.image_grid_angles(2 * rope.MAX_NEGATIVE_POSITION + 2, 2, AXES)
    with pytest.raises(ValueError, match="positive and even"):
        rope.axis_inverse_frequencies(15)


def test_text_encoder_rotary_tables() -> None:
    cos, sin = rope.rotate_half_tables(9, 8, 5_000_000.0)
    assert cos.shape == sin.shape == (9, 4)
    inverse = 1.0 / np.power(5_000_000.0, np.arange(0, 8, 2) / 8.0)
    np.testing.assert_allclose(cos, np.cos(np.outer(np.arange(9), inverse)), atol=1e-6)
    np.testing.assert_allclose(sin, np.sin(np.outer(np.arange(9), inverse)), atol=1e-6)
    np.testing.assert_array_equal(cos[0], np.ones(4, dtype=np.float32))


def _reference_shortcut(values: np.ndarray, out_dim: int, factor_t: int, factor_s: int = 2):
    """``QwenImage21DupUp3D.forward(x, first_chunk=True)`` for one frame ``[1, C, 1, H, W]``."""
    _, in_dim, frames, height, width = values.shape
    repeats = out_dim * factor_t * factor_s * factor_s // in_dim
    values = np.repeat(values, repeats, axis=1)
    values = values.reshape(1, out_dim, factor_t, factor_s, factor_s, frames, height, width)
    values = values.transpose(0, 1, 5, 2, 6, 3, 7, 4)
    values = values.reshape(1, out_dim, frames * factor_t, height * factor_s, width * factor_s)
    return values[:, :, factor_t - 1 :]


@pytest.mark.parametrize(
    ("in_dim", "out_dim", "factor_t"),
    # The decoder's four upsampling stages (scaled-down channel counts), then a generic case.
    [(8, 8, 2), (8, 4, 2), (4, 2, 1), (16, 8, 2), (12, 6, 1)],
)
def test_shortcut_channels_match_the_reference_upsampling(in_dim, out_dim, factor_t) -> None:
    height, width = 3, 5
    values = np.random.default_rng(0).standard_normal((1, in_dim, 1, height, width))
    expected = _reference_shortcut(values, out_dim, factor_t)[:, :, 0]

    spread = values[:, shortcut_channels(in_dim, out_dim, factor_t), 0]
    spread = spread.reshape(1, out_dim, 2, 2, height, width).transpose(0, 1, 4, 2, 5, 3)
    np.testing.assert_array_equal(spread.reshape(1, out_dim, height * 2, width * 2), expected)


def test_shortcut_rejects_incompatible_channel_counts() -> None:
    with pytest.raises(ValueError, match="not compatible"):
        shortcut_channels(7, 3, 1)


def test_geometry_defaults_and_alignment() -> None:
    geometry = resolve_geometry(
        None, None, None, vae_scale=16, max_image_tokens=8192, frame_tokens=8
    )
    assert (geometry.image_height, geometry.image_width) == (1024, 1024)
    assert (geometry.latent_height, geometry.latent_width) == (64, 64)
    assert geometry.image_tokens == 4096 and geometry.text_tokens == DEFAULT_TEXT_TOKENS

    wide = resolve_geometry(512, 768, 128, vae_scale=16, max_image_tokens=8192, frame_tokens=8)
    assert (wide.latent_height, wide.latent_width, wide.text_tokens) == (32, 48, 128)


@pytest.mark.parametrize(
    ("height", "width", "tokens", "message"),
    [
        (1000, 1024, None, "image_height must be a multiple of 32"),
        (1024, 1040, None, "image_width must be a multiple of 32"),
        (224, 1024, None, "at least 256"),
        (2048, 2048, None, "at most 8192"),
        (1024, 1024, 0, "max_sequence_length"),
        (1024, 1024, 8, "8 tokens that frame the prompt"),
        (1024, 1024, 5000, "max_sequence_length"),
    ],
)
def test_geometry_rejects_unbuildable_requests(height, width, tokens, message) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_geometry(
            height, width, tokens, vae_scale=16, max_image_tokens=8192, frame_tokens=8
        )


def _write_checkpoint(root, *, pipeline="QwenImage21Pipeline", index_extra=None, **scheduler):
    settings = {
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "base_image_seq_len": 256,
        "base_shift": 0.5,
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": 0.9,
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": 0.02,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
        **scheduler,
    }
    (root / "scheduler").mkdir(parents=True, exist_ok=True)
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(settings))
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": pipeline, **(index_extra or {})})
    )
    return root


def test_scheduler_settings_come_from_the_checkpoint(tmp_path) -> None:
    assert scheduler_settings(_write_checkpoint(tmp_path)) == {
        "num_train_timesteps": 1000,
        "base_shift": 0.5,
        "max_shift": 0.9,
        "base_image_seq_len": 256,
        "max_image_seq_len": 8192,
        "shift_terminal": 0.02,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"pipeline": "QwenImagePipeline"}, "requires a QwenImage21Pipeline"),
        ({"index_extra": {"sample_sigmas": [1.0, 0.5]}}, "sample_sigmas"),
        ({"use_dynamic_shifting": False}, "exponential dynamic-shift"),
        ({"time_shift_type": "linear"}, "exponential dynamic-shift"),
        ({"use_karras_sigmas": True}, "exponential dynamic-shift"),
        ({"stochastic_sampling": True}, "exponential dynamic-shift"),
        ({"num_train_timesteps": 500}, "num_train_timesteps = 1000"),
        ({"_class_name": "EulerDiscreteScheduler"}, "unsupported scheduler class"),
    ],
)
def test_unsupported_schedules_are_rejected(tmp_path, overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        scheduler_settings(_write_checkpoint(tmp_path, **overrides))
