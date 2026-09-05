# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for GPT-NeoX accuracy and its premerge case."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from families.gpt_neox.tests.test_e2e import _assert_correctness


_TEST_DIR = Path(__file__).resolve().parent
_MANIFEST_PATH = _TEST_DIR / "manifests" / "pythia-70m.json"
_THRESHOLD_PATH = _TEST_DIR / "thresholds" / "pythia-70m.json"


def _payload(token_ids: list[int]) -> dict:
    return {"token_ids": token_ids, "text": "same decoded continuation"}


def _verify(actual_ids: list[int], reference_ids: list[int]) -> None:
    _assert_correctness(
        _payload(actual_ids),
        {"max_new_tokens": 32},
        {"contract_token_agreement_rate": 1.0},
        reference_ids,
        "same decoded continuation",
        None,
        "same decoded continuation",
    )


def test_pythia_premerge_case_replays_the_observed_accuracy_failure() -> None:
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    testcase = manifest["testcases"][0]
    thresholds = json.loads(_THRESHOLD_PATH.read_text(encoding="utf-8"))["threshold_overrides"]

    prompt = testcase["prompt"]
    assert prompt.startswith("The following are multiple choice questions (with answers)")
    assert prompt.count("\nAnswer:") == 6
    assert prompt.endswith(
        "Statement 1 | A factor group of a non-Abelian group is non-Abelian. "
        "Statement 2 | If K is a normal subgroup of H and H is a normal subgroup of G, "
        "then K is a normal subgroup of G.\n"
        "A. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer:"
    )
    assert testcase["max_new_tokens"] >= 26
    assert manifest["max_sequence_length"] >= 393
    assert thresholds["contract_token_agreement_rate"] == 1.0


def test_pythia_contract_rejects_a_token_divergence_with_identical_text() -> None:
    with pytest.raises(AssertionError):
        _verify([329, 187, 16708], [329, 187, 11793])


def test_pythia_contract_accepts_exact_generated_tokens() -> None:
    exact_tokens = [329, 187, 16708]
    _verify(exact_tokens, exact_tokens)
