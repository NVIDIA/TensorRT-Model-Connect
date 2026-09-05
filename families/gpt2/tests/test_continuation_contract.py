# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the GPT-2 continuation contract."""

from __future__ import annotations

import pytest

from families.gpt2.tests.test_e2e import _assert_correctness


_PROMPT = "The capital of France is"


def _verify(actual_text: str, reference_text: str) -> None:
    _assert_correctness(
        {"token_ids": [1], "text": actual_text},
        {"prompt": _PROMPT, "max_new_tokens": 1},
        {},
        [],
        reference_text,
        None,
        actual_text,
    )


def test_rejects_prompt_prefixed_continuation() -> None:
    with pytest.raises(AssertionError):
        _verify("The capital of France is Paris.", "Paris.")


def test_accepts_clean_continuation() -> None:
    _verify("Paris.", "Paris.")
