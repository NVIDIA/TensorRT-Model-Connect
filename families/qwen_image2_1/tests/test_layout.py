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
    TOKENIZER_NORMALIZER,
    TOKENIZER_PRE_TOKENIZER,
    require_embedded_token_ids,
    require_supported_tokenizer,
    resolve_geometry,
    scheduler_settings,
)
from families.qwen_image2_1.resample import shortcut_channels
from families.qwen_image2_1.tests.image_checks import png_levels, spatial_std, to_png_bytes

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


# The BPE model options and one added-token record of the checkpoint's tokenizer.json.
_CHECKPOINT_MODEL = {
    "type": "BPE",
    "dropout": None,
    "unk_token": None,
    "continuing_subword_prefix": "",
    "end_of_word_suffix": "",
    "fuse_unk": False,
    "byte_fallback": False,
    "ignore_merges": False,
}
_CHECKPOINT_ADDED_TOKEN = {
    "id": 151644,
    "content": "<|im_start|>",
    "single_word": False,
    "lstrip": False,
    "rstrip": False,
    "normalized": False,
    "special": True,
}


def _tokenizer_definition(**overrides) -> dict:
    return {
        "added_tokens": [_CHECKPOINT_ADDED_TOKEN],
        "normalizer": TOKENIZER_NORMALIZER,
        "pre_tokenizer": TOKENIZER_PRE_TOKENIZER,
        "model": _CHECKPOINT_MODEL,
        **overrides,
    }


def _model(**options) -> dict:
    return {"model": {**_CHECKPOINT_MODEL, **options}}


def _added_token(**fields) -> dict:
    record = {**_CHECKPOINT_ADDED_TOKEN, **fields}
    return {"added_tokens": [{key: value for key, value in record.items() if value is not ...}]}


def _pre_tokenizer(split: dict | None = None, byte_level: dict | None = None) -> dict:
    steps = TOKENIZER_PRE_TOKENIZER["pretokenizers"]
    return {
        "type": "Sequence",
        "pretokenizers": [{**steps[0], **(split or {})}, {**steps[1], **(byte_level or {})}],
    }


def test_the_checkpoint_tokenizer_configuration_is_accepted() -> None:
    require_supported_tokenizer(_tokenizer_definition())
    require_supported_tokenizer(_tokenizer_definition(normalizer=None))
    # Options left out mean the same as the checkpoint's values.
    require_supported_tokenizer(_tokenizer_definition(model={"type": "BPE"}, added_tokens=None))
    require_supported_tokenizer(
        _tokenizer_definition(added_tokens=[{"id": 7, "content": "<|x|>", "special": False}])
    )
    # Offsets are not part of the ids.
    require_supported_tokenizer(
        _tokenizer_definition(pre_tokenizer=_pre_tokenizer(byte_level={"trim_offsets": True}))
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model": {"type": "WordPiece"}}, "requires a BPE"),
        ({"model": None}, "requires a BPE"),
        ({"pre_tokenizer": None}, "pre-tokenizer"),
        ({"pre_tokenizer": TOKENIZER_PRE_TOKENIZER["pretokenizers"][1]}, "pre-tokenizer"),
        (
            {"pre_tokenizer": _pre_tokenizer(split={"pattern": {"Regex": r"\p{N}{1,3}|\s+"}})},
            "pre-tokenizer",
        ),
        ({"pre_tokenizer": _pre_tokenizer(split={"behavior": "Removed"})}, "pre-tokenizer"),
        ({"pre_tokenizer": _pre_tokenizer(split={"invert": True})}, "pre-tokenizer"),
        ({"pre_tokenizer": _pre_tokenizer(byte_level={"use_regex": True})}, "pre-tokenizer"),
        (
            {"pre_tokenizer": _pre_tokenizer(byte_level={"add_prefix_space": True})},
            "pre-tokenizer",
        ),
        ({"normalizer": {"type": "NFKC"}}, "normalizer"),
        ({"normalizer": {"type": "Lowercase"}}, "normalizer"),
        (_model(ignore_merges=True), "ignore_merges"),
        (_model(fuse_unk=True), "fuse_unk"),
        (_model(byte_fallback=True), "byte_fallback"),
        (_model(dropout=0.1), "dropout"),
        (_model(unk_token="<unk>"), "unk_token"),
        (_model(continuing_subword_prefix="##"), "continuing_subword_prefix"),
        (_model(end_of_word_suffix="</w>"), "end_of_word_suffix"),
        (_added_token(single_word=True), "single_word"),
        (_added_token(lstrip=True), "lstrip"),
        (_added_token(rstrip=True), "rstrip"),
        (_added_token(id=...), r"added_tokens\[0\]"),
        (_added_token(content=...), r"added_tokens\[0\]"),
        (_added_token(content=""), r"added_tokens\[0\]"),
        (_added_token(id="151644"), r"added_tokens\[0\]"),
        (_added_token(id=True), r"added_tokens\[0\]"),
        (_added_token(id=-1), r"added_tokens\[0\]"),
        ({"added_tokens": ["<|im_start|>"]}, r"added_tokens\[0\]"),
        ({"added_tokens": {"<|im_start|>": 151644}}, "to be a list"),
    ],
)
def test_other_tokenizer_configurations_are_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        require_supported_tokenizer(_tokenizer_definition(**overrides))


# The text encoder of the checkpoint embeds 151936 ids; its tokenizer uses the first 151669.
_EMBEDDING_ROWS = 151936


def _vocabulary(**ids) -> dict:
    return _tokenizer_definition(model={**_CHECKPOINT_MODEL, "vocab": {"a": 0, "b": 1, **ids}})


def test_token_ids_inside_the_embedding_table_are_accepted() -> None:
    require_embedded_token_ids(_vocabulary(), 151643, _EMBEDDING_ROWS)
    last = _EMBEDDING_ROWS - 1
    require_embedded_token_ids(
        _vocabulary(c=last) | _added_token(id=last), last, _EMBEDDING_ROWS
    )
    require_embedded_token_ids(_vocabulary() | {"added_tokens": None}, 0, _EMBEDDING_ROWS)


@pytest.mark.parametrize(
    ("definition", "pad_token_id", "message"),
    [
        (_vocabulary(c=_EMBEDDING_ROWS), 0, "model.vocab entry 'c'.*151936 is not one of the 151936"),
        (_vocabulary(c=-1), 0, "model.vocab entry 'c'"),
        (_vocabulary(c="2"), 0, "model.vocab entry 'c'"),
        (_vocabulary(c=2.0), 0, "model.vocab entry 'c'"),
        (_vocabulary(c=True), 0, "model.vocab entry 'c'"),
        (
            _vocabulary() | _added_token(id=_EMBEDDING_ROWS),
            0,
            r"added_tokens\[0\] \('<\|im_start\|>'\).*151936 is not one of the 151936",
        ),
        (_vocabulary() | _added_token(id=2**40), 0, r"added_tokens\[0\]"),
        (_vocabulary(), _EMBEDDING_ROWS, "padding token is outside"),
        (_vocabulary(), -1, "padding token is outside"),
        (_tokenizer_definition(), 0, "model.vocab of tokenizer.json to be an object"),
        (_vocabulary() | _added_token(id=...), 0, r"added_tokens\[0\].*a token id"),
    ],
)
def test_token_ids_outside_the_embedding_table_are_rejected(
    definition: dict, pad_token_id: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        require_embedded_token_ids(definition, pad_token_id, _EMBEDDING_ROWS)


def test_flat_images_have_no_spatial_deviation_whatever_their_colour() -> None:
    red = np.zeros((16, 16, 3), dtype=np.float32)
    red[..., 0] = 1.0
    # Over all axes a uniform red image deviates by 0.47; over its pixels it does not.
    assert red.std() > 0.4
    assert spatial_std(red) == 0.0
    assert spatial_std(np.full((16, 16, 3), 0.5, dtype=np.float32)) == 0.0

    ramp = np.zeros((16, 16, 3), dtype=np.float32)
    ramp[..., 1] = np.linspace(0.0, 1.0, 16, dtype=np.float32)[None, :]
    # One channel that varies over the pixels is enough.
    assert spatial_std(ramp) == pytest.approx(ramp[..., 1].std())
    assert spatial_std(ramp) > 0.02


def test_png_levels_are_the_pixels_a_png_reads_back() -> None:
    image = np.array([[[0.0, 0.5, 1.0, 0.2499]], [[-0.1, 1.2, 0.5019, 0.998]]], dtype=np.float32)
    stored = to_png_bytes(image)
    assert stored.dtype == np.uint8
    assert stored.tolist() == [[[0, 128, 255, 64]], [[0, 255, 128, 254]]]
    levels = png_levels(image)
    assert levels.dtype == np.float32
    np.testing.assert_array_equal(levels, stored.astype(np.float32) / np.float32(255.0))
    # An image that already went through a PNG is left as it is.
    np.testing.assert_array_equal(png_levels(levels), levels)
    np.testing.assert_array_equal(to_png_bytes(levels), stored)
