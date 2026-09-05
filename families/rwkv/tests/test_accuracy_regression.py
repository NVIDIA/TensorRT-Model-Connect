# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the RWKV continuation contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from families.rwkv.tests.test_e2e import _assert_correctness, _normalized_edit_distance


_REFERENCE = "the capital of the French Republic. The capital of"


def _verify(actual_text: str) -> None:
    _assert_correctness(
        {"token_ids": [1], "text": actual_text},
        {"max_new_tokens": 1},
        {"contract_ned_threshold": 0.2},
        [],
        _REFERENCE,
        None,
        actual_text,
    )


def test_acceptance_manifest_retains_full_precision() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "rwkv-169m.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["precision"] == "fp32"
    assert manifest["testcases"][0]["reference_precision"] == "fp32"


def test_rejects_continuation_above_declared_ned_limit() -> None:
    actual = "the capital of the French Republic. The"
    assert _normalized_edit_distance(actual, _REFERENCE) == pytest.approx(0.22)
    with pytest.raises(AssertionError):
        _verify(actual)


def test_accepts_continuation_within_declared_ned_limit() -> None:
    assert _normalized_edit_distance(_REFERENCE, _REFERENCE) == pytest.approx(0.0)
    _verify(_REFERENCE)
