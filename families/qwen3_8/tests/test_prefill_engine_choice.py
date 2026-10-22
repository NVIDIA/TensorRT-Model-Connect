# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Choosing the dynamic or the static prefill engine: auto falls back, dynamic does not."""

from __future__ import annotations

import pytest

from families.qwen3_8.model import _choose_prefill_engine


def _choose(engine="auto", *, precision="fp16", mtp=False, chunk=None, plugin=None):
    calls = []

    def unavailable():
        calls.append(1)
        return plugin

    result = _choose_prefill_engine(
        engine, precision=precision, mtp=mtp, prefill_chunk_len=chunk, plugin_unavailable=unavailable)
    return result, len(calls)


def test_auto_picks_dynamic_when_everything_is_available() -> None:
    assert _choose()[0] == (True, None)
    assert _choose(precision="bf16")[0] == (True, None)
    assert _choose(chunk=1024)[0] == (True, None)


def test_static_never_asks_for_the_plugin() -> None:
    assert _choose("static") == ((False, None), 0)


def test_no_prefill_engine_is_static_without_a_note() -> None:
    assert _choose(chunk=0) == ((False, None), 0)


@pytest.mark.parametrize(
    "kwargs, text",
    [
        (dict(precision="fp32"), "fp16 or bf16"),
        (dict(mtp=True), "MTP"),
        (dict(plugin="the cuda-tile package is not installed"), "cuda-tile"),
    ],
)
def test_auto_falls_back_to_static_and_says_why(kwargs, text) -> None:
    (dynamic, reason), _ = _choose(**kwargs)
    assert dynamic is False and text in reason


@pytest.mark.parametrize(
    "kwargs, text",
    [
        (dict(precision="fp32"), "fp16 or bf16"),
        (dict(mtp=True), "MTP"),
        (dict(plugin="the cuda-tile package is not installed"), "cuda-tile"),
    ],
)
def test_explicit_dynamic_fails_instead_of_falling_back(kwargs, text) -> None:
    with pytest.raises(ValueError, match=f"not available: .*{text}"):
        _choose("dynamic", **kwargs)


def test_the_plugin_is_only_checked_when_nothing_else_rules_dynamic_out() -> None:
    assert _choose(precision="fp32")[1] == 0
    assert _choose(mtp=True)[1] == 0
    assert _choose()[1] == 1
