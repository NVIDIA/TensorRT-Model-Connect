# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from html.parser import HTMLParser

import pytest

from families.laya.tests.reporting import record_decision_input, record_decisions
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


@pytest.mark.parametrize("routing", [None, {"model": "typed-decisions", "language": "english"}])
def test_default_card_preserves_all_typed_answers_and_original_reference(tmp_path, routing):
    answers = {"department": {"choice": "billing", "probabilities": {"billing": 0.9, "other": 0.1}},
               "urgency": {"score": 1.8, "confidence": 0.8},
               "refund": {"noul": 0.75, "action": {"act_probability": 0.25}}}
    reference = {"answers": copy.deepcopy(answers), "usage": {"input_tokens": 150}}
    reference["answers"]["department"]["choice"] = "other"
    native = {"response": {"answers": answers, "usage": {"input_tokens": 150}}, "scores": []}
    if routing is not None:
        native["response"]["routing"] = routing
        reference["routing"] = copy.deepcopy(routing)
    logits = [[1.0, 0.0]]
    original = copy.deepcopy((native, reference, logits))
    recorder = Evidence(tmp_path / "evidence", family="laya", case="display-contract", source_revision="a" * 40)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        record_decision_input({"state": {"subject": "Duplicate charge"}, "questions": {"department": {"type": "choice"}}})
        record_decisions(native, reference, logits)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    recorder.data["status"] = "failed"
    document = render_case(recorder.data, recorder.directory)
    visible = _Visible()
    visible.feed(document)
    text = " ".join(visible.text)
    for expected in ("Native output", "Reference output", '"choice": "billing"', '"choice": "other"',
                     "Question: urgency", '"score": 1.8', '"noul": 0.75', '"act_probability": 0.25',
                     "Original Laya SDK reference", "not generated text", "Duplicate charge"):
        assert expected in text
    assert "Input tokens 150" not in text
    if routing is not None:
        assert "Routing:" in text and "typed-decisions" in text
    assert recorder.data["native"]["response"] == native["response"]
    assert recorder.data["reference"]["response"] == reference
    assert recorder.data["reference"]["logits"] == logits
    assert (native, reference, logits) == original


def test_empty_questions_keep_explicit_contract_scope(tmp_path):
    recorder = Evidence(tmp_path / "evidence", family="laya", case="empty-contract", source_revision="a" * 40)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        record_decisions({"response": {"answers": {}}, "scores": []}, {"answers": {}}, [], empty_questions=True)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert "No questions; no decisions requested" in recorder.data["native"]["text"]
    assert recorder.data["reference"]["mode"] == "contract_only"
    assert recorder.data["reference"]["response"] == {"answers": {}}


def test_disabled_reporting_does_not_inspect_outputs():
    from collections.abc import Mapping

    accesses = []

    class ObservedMapping(Mapping):
        def __getitem__(self, key):
            accesses.append(("getitem", key))
            return {"answers": {}}

        def __iter__(self):
            accesses.append(("iter", None))
            return iter(("response",))

        def __len__(self):
            accesses.append(("len", None))
            return 1

    record_decisions(ObservedMapping(), ObservedMapping(), ObservedMapping())
    assert accesses == []
