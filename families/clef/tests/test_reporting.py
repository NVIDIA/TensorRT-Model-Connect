# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from html.parser import HTMLParser

import pytest

from families.clef.tests.reporting import record_decision_input, record_decisions
from tools import e2e_evidence
from tools.e2e_evidence import Evidence
from tools.e2e_report import render_case


class _Visible(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.text = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in {"img", "input", "meta", "link", "br", "hr", "source", "path", "polyline"}:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.stack:
            self.stack = self.stack[:len(self.stack) - 1 - self.stack[::-1].index(tag)]

    def handle_data(self, data):
        if not any(tag in self.stack for tag in ("details", "script", "style")):
            self.text.append(data)


def test_default_card_keeps_native_answers_and_independent_reference_probabilities(tmp_path):
    native = {"response": {"answers": {"status": {"choice": "paid"}, "large": {"noul": 0.8}},
                           "usage": {"input_tokens": 150}},
              "scores": [{"question_id": "status", "probabilities": [0.9, 0.1]}]}
    options = {"status": {"option_ids": ["paid", "overdue"], "probabilities": [0.1, 0.9]},
               "large": {"option_ids": ["false", "true"], "probabilities": [0.2, 0.8]}}
    logits = [[0.0, 2.0], [0.0, 1.0]]
    original = copy.deepcopy((native, options, logits))
    recorder = Evidence(tmp_path / "evidence", family="clef", case="display-contract", source_revision="a" * 40)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        record_decision_input({"state": {"invoice": "overdue"}, "questions": {"status": {"type": "choice"}}})
        record_decisions(native, options, logits, 2)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    recorder.data["status"] = "failed"
    document = render_case(recorder.data, recorder.directory)
    visible = _Visible()
    visible.feed(document)
    text = " ".join(visible.text)
    for expected in ("Native output", "Reference output", "Question: status", '"choice": "paid"',
                     "Highest-probability option: overdue", "overdue: 0.9", "true: 0.8",
                     "not a separately formatted SDK answer", "not generated text", "invoice"):
        assert expected in text
    assert "Input tokens 150" not in text
    assert recorder.data["native"]["response"] == native["response"]
    assert recorder.data["native"]["scores"] == native["scores"]
    assert recorder.data["reference"]["logits"] == logits
    assert (native, options, logits) == original


def test_empty_questions_remain_contract_only(tmp_path):
    recorder = Evidence(tmp_path / "evidence", family="clef", case="empty-contract", source_revision="a" * 40)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        record_decisions({"response": {"answers": {}}, "scores": []}, {}, [], 0)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert "No questions; no decisions requested" in recorder.data["native"]["text"]
    assert recorder.data["reference"]["mode"] == "contract_only"
    assert "Highest-probability option" not in recorder.data["reference"]["text"]


@pytest.mark.parametrize("probabilities", [[0.9], [float("nan"), 0.1], []])
def test_incomplete_or_nonfinite_probabilities_do_not_invent_an_option(tmp_path, probabilities):
    recorder = Evidence(tmp_path / "evidence", family="clef", case="partial-contract", source_revision="a" * 40)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        record_decisions(
            {"response": {"answers": {"status": {"choice": "paid"}}}, "scores": []},
            {"status": {"option_ids": ["paid", "overdue"], "probabilities": probabilities}}, [], 1,
        )
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert "unavailable" in recorder.data["reference"]["text"]
    assert "Highest-probability option:" not in recorder.data["reference"]["text"]


def test_disabled_reporting_does_not_inspect_outputs():
    class Unreadable:
        def __getitem__(self, key):
            raise AssertionError("disabled reporting inspected model output")

    record_decisions(Unreadable(), Unreadable(), Unreadable(), 1)
