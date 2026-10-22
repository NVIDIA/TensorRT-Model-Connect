# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default KV cache length and prefill call size, including the dynamic prefill engine."""

from __future__ import annotations

import pytest

from families.qwen3_8.model import _resolve_lengths

CONTEXT = 262144


def test_default_cache_is_4096_for_a_long_context_model() -> None:
    assert _resolve_lengths(CONTEXT, None, None, False) == (4096, 128)


def test_default_cache_is_capped_by_a_short_context_model() -> None:
    assert _resolve_lengths(2048, None, None, False) == (2048, 128)


def test_explicit_cache_is_kept() -> None:
    assert _resolve_lengths(CONTEXT, 512, None, False) == (512, 128)


def test_explicit_cache_above_the_context_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds checkpoint context"):
        _resolve_lengths(2048, 4096, None, False)


def test_dynamic_prefill_defaults_to_1024_calls_in_the_default_cache() -> None:
    assert _resolve_lengths(CONTEXT, None, None, True) == (4096, 1024)


def test_dynamic_prefill_raises_the_default_cache_to_the_chunk() -> None:
    assert _resolve_lengths(CONTEXT, None, 8192, True) == (8192, 8192)


def test_dynamic_prefill_with_a_too_small_explicit_cache_is_an_error() -> None:
    with pytest.raises(ValueError, match="raise --max-sequence-length"):
        _resolve_lengths(CONTEXT, 512, None, True)
    with pytest.raises(ValueError, match="raise --max-sequence-length"):
        _resolve_lengths(CONTEXT, 2048, 4096, True)


def test_dynamic_prefill_chunk_above_the_context_is_an_error() -> None:
    with pytest.raises(ValueError, match="raise --max-sequence-length"):
        _resolve_lengths(2048, None, 4096, True)


def test_static_prefill_keeps_its_chunk_and_clamps_later() -> None:
    # The static engine's chunk is clamped to the cache in build(); this only passes it on.
    assert _resolve_lengths(CONTEXT, 64, None, False) == (64, 128)
