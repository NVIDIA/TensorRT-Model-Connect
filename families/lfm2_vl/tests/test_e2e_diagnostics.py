# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failed semantic responses remain readable when pytest truncates its assertion."""

from copy import deepcopy
import json

import pytest

from .test_e2e import _finish_reason, _report_semantic_failures


@pytest.mark.parametrize(
    "tokens,eos,limit,reason",
    [
        ([12, 7], 7, 64, "eos_token"),
        ([12, 7], [7, 9], 2, "eos_token"),
        ([12, 9], [7, 9], 64, "eos_token"),
        ([12, 13], 7, 2, "length"),
        ([12, 13], None, 2, "length"),
        ([12], 7, 64, "unknown"),
        ([], 7, 64, "unknown"),
    ],
)
def test_finish_reason_distinguishes_eos_limit_and_unknown(tokens, eos, limit, reason):
    assert _finish_reason(tokens, eos, limit) == reason


def test_failed_rows_retain_exact_responses_without_large_arrays(capsys):
    row = {
        "name": "tool-call",
        "input_tokens": 123,
        "max_new_tokens": 256,
        "reference_text": "",
        "native_text": "Texte exact\n你好\n" + "untruncated " * 256,
        "reference_calls": [
            {"name": "get_weather", "arguments": {"location": "Paris"}, "state": 1}
        ],
        "native_calls": [
            {
                "name": "get_weather",
                "arguments": {"location": "東京", "units": "C"},
                "state": 1,
                "call_id": "call-0",
            }
        ],
        "reference_finish_reason": "eos_token",
        "native_finish_reason": "length",
        "finish_reason_basis": "inferred_from_final_token_and_generation_limit",
        "semantic": {"equivalent": False, "task_correct": False, "native": None},
    }
    failed = {**row, "native_tokens": list(range(32768)), "logits": [0.0] * 32768}
    second = {**failed, "name": "language-ja", "native_text": "赤ではありません。"}
    passed = {**failed, "name": "passing-case", "semantic": {"equivalent": True}}
    evidence = [passed, failed, second]
    before = deepcopy(evidence)

    _report_semantic_failures(evidence)

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    prefix = "LFM2_VL_SEMANTIC_FAILURE "
    assert len(lines) == 2
    assert all(line.startswith(prefix) for line in lines)
    assert [json.loads(line.removeprefix(prefix)) for line in lines] == [
        row,
        {**row, "name": "language-ja", "native_text": "赤ではありません。"},
    ]
    assert captured.err == ""
    assert evidence == before
    assert not all(item["semantic"]["equivalent"] for item in evidence)


def test_passing_semantics_emit_no_failure_diagnostic(capsys):
    _report_semantic_failures([{"semantic": {"equivalent": True}}])
    assert capsys.readouterr().out == ""
