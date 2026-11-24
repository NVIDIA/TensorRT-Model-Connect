# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the E2E harness: case selection and the BF16 departure oracle."""

import pytest

from families.hunyuan_vl.tests import test_e2e as e2e


class _Config:
    def __init__(self, models=(), cases=()):
        self.options = {"--e2e-model": list(models), "--e2e-testcase": list(cases)}

    def getoption(self, name, default=None):
        return self.options.get(name, default)


BF16_CASE = "hunyuanocr-bf16"
BF16_NAMES = {BF16_CASE, e2e.CASES[BF16_CASE][0]["name"]}
FP32_NAMES = {"hunyuanocr-fp32", e2e.CASES["hunyuanocr-fp32"][0]["name"]}
LIMITS = {"bf16_departure_max_logit_gap": 0.5}


def test_explicit_selection_that_matches_no_case_fails():
    typo = _Config(models=["hunyuan_vl"], cases=["hunyuanocr-bf61"])
    try:
        e2e._selected(typo, BF16_NAMES)
    except pytest.fail.Exception as failure:
        assert "selects none of hunyuan_vl" in str(failure)
    except pytest.skip.Exception:
        pytest.fail("an explicit hunyuan_vl selection that matches no case was skipped")
    else:
        pytest.fail("an explicit hunyuan_vl selection that matches no case ran")
    # Another family's selection still skips this file; a matching case runs, the other skips.
    with pytest.raises(pytest.skip.Exception):
        e2e._selected(_Config(models=["qwen"], cases=["qwen-typo"]), BF16_NAMES)
    selected = _Config(models=["hunyuan_vl"], cases=[BF16_CASE])
    e2e._selected(selected, BF16_NAMES)
    with pytest.raises(pytest.skip.Exception):
        e2e._selected(selected, FP32_NAMES)


def _fails(config, names):
    """Whether the selection fails; a skip, which would hide the empty selection, is False."""
    try:
        e2e._selected(config, names)
    except pytest.fail.Exception as failure:
        return "selects none of hunyuan_vl" in str(failure)
    except pytest.skip.Exception:
        return False
    return False


def test_model_and_testcase_that_choose_different_cases_fail():
    crossed = _Config(models=[BF16_CASE], cases=["hunyuanocr-fp32"])
    # Every check fails, the host-side one included, instead of passing without inference.
    for names in (BF16_NAMES, FP32_NAMES, set(e2e.CASES)):
        assert _fails(crossed, names)
    assert not _fails(_Config(models=[BF16_CASE], cases=[BF16_CASE]), set(e2e.CASES))
    assert not _fails(_Config(models=["hunyuan_vl"]), set(e2e.CASES))


def test_departure_reports_the_reference_logit_gap():
    torch = pytest.importorskip("torch")
    logits = [torch.zeros(1, 10) for _ in range(3)]
    logits[1][0, 6], logits[1][0, 9] = 2.0, 1.75
    departure = e2e._departure([5, 6, 7], [5, 9, 7], logits)
    assert departure == {"index": 1, "reference_token": 6, "candidate_token": 9, "logit_gap": 0.25}
    assert e2e._departure([5, 6, 7], [5, 6, 7], logits) is None
    assert e2e._departure([5, 6, 7], [5, 6], logits) == {"index": 2, "logit_gap": None}


def _gap(value):
    return {"index": 3, "reference_token": 1, "candidate_token": 2, "logit_gap": value}


def test_fp32_bundle_must_reproduce_every_token():
    assert e2e._verdict([1, 2], [1, 2], None, None, LIMITS)["passed"]
    assert not e2e._verdict([1, 3], [1, 2], None, _gap(0.0), LIMITS)["passed"]


def test_bf16_stable_request_must_be_exact():
    verdict = e2e._verdict([1, 3], [1, 2], [1, 2], _gap(0.01), LIMITS)
    assert verdict["bf16_stable"] and not verdict["passed"]
    assert e2e._verdict([1, 2], [1, 2], [1, 2], None, LIMITS)["passed"]


def test_bf16_departure_passes_only_at_a_near_tie():
    official = [1, 4]
    assert e2e._verdict([1, 3], [1, 2], official, _gap(0.5), LIMITS)["passed"]
    assert not e2e._verdict([1, 3], [1, 2], official, _gap(0.51), LIMITS)["passed"]
    assert not e2e._verdict([1], [1, 2], official, {"index": 1, "logit_gap": None}, LIMITS)[
        "passed"
    ]
    assert e2e._verdict([1, 2], [1, 2], official, None, LIMITS)["passed"]


def test_verdict_checks_carry_their_own_outcome():
    verdicts = (
        e2e._verdict([1, 2], [1, 2], None, None, LIMITS),
        e2e._verdict([1, 3], [1, 2], None, _gap(0.0), LIMITS),
        e2e._verdict([1, 3], [1, 2], [1, 4], _gap(0.25), LIMITS),
        e2e._verdict([1, 3], [1, 2], [1, 4], _gap(2.0), LIMITS),
        e2e._verdict([1], [1, 2], [1, 4], {"index": 1, "logit_gap": None}, LIMITS),
    )
    evaluate = {"==": lambda a, b: a == b, "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b}
    assert [verdict["passed"] for verdict in verdicts] == [True, False, True, False, False]
    for verdict in verdicts:
        assert verdict["checks"] and verdict["passed"] == all(
            c["passed"] for c in verdict["checks"]
        )
        for check in verdict["checks"]:
            assert check["passed"] == evaluate[check["operator"]](
                check["actual"], check["expected"]
            )
            assert check["scope"] == "contract" or type(check["actual"]) is not bool


def test_native_output_stops_at_its_first_end_token_or_its_budget():
    requests = [{"max_new_tokens": 4}] * 4
    # Ends at its end token; uses the whole budget; continues past an end token; stops early.
    outputs = [{"token_ids": ids} for ids in ([5, 6, 9], [5, 6, 7, 8], [5, 9, 6], [5, 6])]
    assert e2e._stopping_errors(outputs, requests, {9}) == [2, 3]


def test_a_printed_answer_must_be_stated_by_both_outputs():
    requests = [{"answer": "4711"}, {}, {"answer": "4711"}]

    def texts(*values):
        return [{"text": value} for value in values]

    assert (
        e2e._wrong_answers(texts("4711", "x", " 4711\n"), texts("4711", "y", "4711"), requests)
        == []
    )
    assert e2e._wrong_answers(texts("4717", "x", "4711"), texts("4711", "y", "4711"), requests) == [
        0
    ]
    assert e2e._wrong_answers(texts("4711", "x", "4711"), texts("4711", "y", "471"), requests) == [
        2
    ]


def test_native_text_that_does_not_decode_its_tokens_is_reported():
    actual = [{"text": "ab", "token_ids": [1, 2]}, {"text": "a?", "token_ids": [1, 2]}]
    assert e2e._native_text_mismatches(actual, lambda ids: "ab") == [1]
