# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Only LFM2.5 manifests opt into reviewed behavior acceptance."""

import pytest

from families.lfm2_moe.tests import test_e2e as e2e


def test_only_new_lfm25_manifests_opt_in():
    for manifest, case in e2e._CASES.values():
        assert ("semantic_case" in case) == manifest["hf_id"].startswith("LiquidAI/LFM2.5-")


def test_e2e_semantics_allow_token_differences_but_require_answer(monkeypatch):
    monkeypatch.setattr(e2e, "record_evidence", lambda *args: None)
    answer = "Caenorhabditis elegans is a nematode and model organism in biological research."
    parts = [{"type": "text", "text": answer}]
    inputs = {
        "native_parts": parts,
        "reference_parts": parts,
        "eos": [9],
        "request": {"config": {"max_new_tokens": 512}},
        "input_count": 17,
    }
    case = {"semantic_case": "model-card-quickstart", "max_new_tokens": 512}
    payload = {"token_ids": [2] * 512, "text": answer}
    e2e._assert_correctness(payload, case, [1] * 512, answer, answer, inputs)
    inputs["native_parts"] = [{"type": "text", "text": "An unrelated nonempty answer."}]
    with pytest.raises(AssertionError, match="identification"):
        e2e._assert_correctness(payload, case, [1] * 512, answer, answer, inputs)


def test_historical_exact_token_gate_is_unchanged():
    with pytest.raises(AssertionError):
        e2e._assert_correctness(
            {"token_ids": [2], "text": "text"}, {"max_new_tokens": 1}, [1], "text", "text"
        )


def test_correct_tokens_cannot_hide_wrong_public_text(monkeypatch):
    monkeypatch.setattr(e2e, "record_evidence", lambda *args: None)
    answer = "Caenorhabditis elegans is a nematode and model organism in biological research."
    parts = [{"type": "text", "text": answer}]
    inputs = {
        "native_parts": parts,
        "reference_parts": parts,
        "eos": [9],
        "request": {"config": {"max_new_tokens": 512}},
        "input_count": 17,
    }
    case = {"semantic_case": "model-card-quickstart", "max_new_tokens": 512}
    payload = {"token_ids": [2] * 512, "text": "WRONG PUBLIC OUTPUT"}
    with pytest.raises(AssertionError, match="public text payload"):
        e2e._assert_correctness(payload, case, [1] * 512, answer, answer, inputs)
