# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for StableLM continuation accuracy and its CI sentinel."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from families.stablelm.tests.test_e2e import _assert_correctness


_TEST_DIR = Path(__file__).resolve().parent
_MANIFEST_PATH = _TEST_DIR / "manifests" / "stablelm2-1.6b.json"
_THRESHOLD_PATH = _TEST_DIR / "thresholds" / "stablelm2-1.6b.json"
_STABLELM_REVISION = "f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef"
_QA_COMMON_PREFIX = [
    423,
    271,
    10086,
    279,
    2015,
    315,
    279,
    81215,
    8066,
    555,
    320,
    16,
    11,
    220,
    17,
    11,
    220,
    18,
    11,
    220,
    19,
]


def _payload(token_ids: list[int]) -> dict:
    return {"token_ids": token_ids, "text": "same decoded continuation"}


def _case() -> dict:
    return {"max_new_tokens": 22}


def _verify(actual_ids: list[int], reference_ids: list[int]) -> None:
    _assert_correctness(
        _payload(actual_ids),
        _case(),
        {"contract_token_agreement_rate": 1.0},
        reference_ids,
        "same decoded continuation",
        None,
        "same decoded continuation",
    )


def test_stablelm_premerge_case_replays_the_published_accuracy_signal() -> None:
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    testcase = manifest["testcases"][0]
    thresholds = json.loads(_THRESHOLD_PATH.read_text(encoding="utf-8"))["threshold_overrides"]

    assert manifest["hf_revision"] == _STABLELM_REVISION
    prompt = testcase["prompt"]
    assert prompt.startswith("The following are multiple choice questions (with answers)")
    assert prompt.count("\nAnswer:") == 6
    assert prompt.endswith(
        "Let p = (1, 2, 5, 4)(2, 3) in S_5 . Find the index of <p> in S_5.\n"
        "A. 8\nB. 2\nC. 24\nD. 120\nAnswer:"
    )
    assert testcase["max_new_tokens"] >= 22
    assert testcase["reference_precision"] == manifest["precision"] == "fp16"
    assert manifest["fp32_layers"] == [23]
    assert manifest["max_sequence_length"] >= 384
    assert thresholds["contract_token_agreement_rate"] == 1.0


def test_stablelm_contract_rejects_the_published_token_21_divergence() -> None:
    with pytest.raises(AssertionError):
        _verify([*_QA_COMMON_PREFIX, 11], [*_QA_COMMON_PREFIX, 2432])


def test_stablelm_contract_accepts_exact_generated_tokens() -> None:
    exact_tokens = [*_QA_COMMON_PREFIX, 11]
    _verify(exact_tokens, exact_tokens)


def test_stablelm_contract_requires_reference_token_ids_for_strict_gate() -> None:
    with pytest.raises(AssertionError):
        _verify([*_QA_COMMON_PREFIX, 11], [])
