# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the global language model's geometry."""

from __future__ import annotations

import importlib

import pytest

np = pytest.importorskip("numpy")

lm = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.language_model"
)
pf = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.prompt_format"
)


def test_tensor_count_matches_the_published_index() -> None:
    """The published shard index lists 399 tensors."""

    assert lm.total_tensors() == 399
    assert 3 + 36 * 11 == 399


def test_vocabulary_is_extended_past_stock_qwen3() -> None:
    assert lm.VOCAB_SIZE == 200000
    assert lm.VOCAB_SIZE > lm.BASE_QWEN3_VOCAB_SIZE
    # The audio codes fit inside the extension.
    assert pf.AUDIO_CODE_OFFSET + pf.SEMANTIC_VOCAB_SIZE <= lm.VOCAB_SIZE


def test_audio_codes_start_inside_qwen3_and_spill_past_it() -> None:
    """The offset is not an append -- it lands in Qwen3's special-token block.

    151675 is below stock Qwen3's 151936, so the first audio codes overlay the
    reserved region and the range then runs past it into the extension. A
    builder that assumed the codes begin where Qwen3's vocabulary ends would be
    261 rows off.
    """

    assert lm.audio_token(0) == pf.AUDIO_CODE_OFFSET == 151675
    assert lm.audio_token(0) < lm.BASE_QWEN3_VOCAB_SIZE
    assert lm.audio_token(pf.SEMANTIC_VOCAB_SIZE - 1) > lm.BASE_QWEN3_VOCAB_SIZE
    assert lm.audio_token(pf.SEMANTIC_VOCAB_SIZE - 1) < lm.VOCAB_SIZE


def test_audio_token_rejects_a_code_outside_the_semantic_vocabulary() -> None:
    with pytest.raises(ValueError, match="code must be"):
        lm.audio_token(pf.SEMANTIC_VOCAB_SIZE)


def test_grouped_attention_ratio() -> None:
    assert lm.group_size() == 4
    assert lm.query_width() == 4096 == lm.HIDDEN_SIZE
    assert lm.key_value_width() == 1024


def test_head_width_times_query_heads_is_the_model_width() -> None:
    assert lm.NUM_ATTENTION_HEADS * lm.HEAD_DIM == lm.HIDDEN_SIZE


def test_repeat_key_value_expands_in_place() -> None:
    heads = np.arange(lm.NUM_KEY_VALUE_HEADS).reshape(-1, 1, 1)

    expanded = lm.repeat_key_value(heads)

    assert expanded.shape[0] == lm.NUM_ATTENTION_HEADS
    # Head i of the cache serves queries 4i..4i+3, not a strided interleave.
    assert expanded.reshape(-1).tolist() == [
        i for i in range(lm.NUM_KEY_VALUE_HEADS) for _ in range(4)
    ]


def test_repeat_key_value_rejects_a_wrong_head_count() -> None:
    with pytest.raises(ValueError, match="expected 8 heads"):
        lm.repeat_key_value(np.zeros((32, 1, 1)))


def test_rope_is_full_width_and_high_theta() -> None:
    dit = importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.dit"
    )
    cos, sin = lm.rope_tables(4)

    assert cos.shape == (4, lm.HEAD_DIM)
    # The diffusion transformer rotates a third of a narrower head at a
    # hundredth of the frequency base.
    assert lm.HEAD_DIM > dit.ROTARY_DIM
    assert lm.ROPE_THETA == 100 * dit.ROPE_THETA


def test_rope_position_zero_is_the_identity() -> None:
    cos, sin = lm.rope_tables(1)

    assert np.allclose(cos, 1.0, atol=1e-6)
    assert np.allclose(sin, 0.0, atol=1e-6)


def test_rope_offset_continues_the_sequence() -> None:
    whole, _ = lm.rope_tables(8)
    tail, _ = lm.rope_tables(4, offset=4)

    assert np.allclose(whole[4:], tail, atol=1e-6)


def test_rope_rejects_positions_past_the_context() -> None:
    with pytest.raises(ValueError, match="exceed the model"):
        lm.rope_tables(2, offset=lm.MAX_POSITION_EMBEDDINGS - 1)


def test_kv_cache_is_sized_by_grouped_heads() -> None:
    shape = lm.kv_cache_shape(1024)

    assert shape == (1, lm.NUM_KEY_VALUE_HEADS, 1024, lm.HEAD_DIM)
    # Four times smaller than a cache sized by query heads.
    assert shape[1] * lm.group_size() == lm.NUM_ATTENTION_HEADS


def test_context_covers_the_longest_generation() -> None:
    spec = importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.pipeline_spec"
    )

    # 9000 audio frames plus a 5000-token prompt still fits the 10240 context.
    assert spec.MAX_AUDIO_FRAMES < lm.MAX_POSITION_EMBEDDINGS
