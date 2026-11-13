# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep each real reporting operation paired with its current input."""

import copy
import math

from PIL import Image
import pytest

from families.siglip.tests import test_e2e as e2e
from tools import e2e_evidence
from tools.e2e_report import render_case


@pytest.fixture
def recorded_case(tmp_path, monkeypatch):
    image = tmp_path / "input.png"
    Image.new("RGB", (2, 2), "white").save(image)
    case = {"name": "report", "query": "POSITIVE_QUERY", "negative_query": "NEGATIVE_QUERY",
            "prompt": "RECIPE_DEFAULT_PROMPT", "test_image": "input.png"}
    manifest = {"name": "report", "family": "siglip", "task": "text_image_to_relevance",
                "bundle": "model.bundle", "precision": "fp32", "hf_id": "example/model"}
    monkeypatch.setattr(e2e, "CASES", {"report": (tmp_path / "manifest.json", manifest, case)})
    monkeypatch.setattr(e2e, "TEST_ROOT", tmp_path)
    monkeypatch.setattr(e2e, "_model_dir", lambda *args: tmp_path)
    monkeypatch.setattr(e2e, "_runtime", lambda: (tmp_path / "binary", tmp_path))
    monkeypatch.setattr(e2e, "_build", lambda *args: None)
    monkeypatch.setattr(e2e, "_asset", lambda *args: image)
    vectors = {case["query"]: [1.0, 0.0], case["negative_query"]: [-1.0, 0.0]}
    scale, bias = 2.0, -0.5
    reference = ([1.0, 0.0], list(vectors.values()), scale, bias)
    monkeypatch.setattr(e2e, "_reference", lambda *args: reference)
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="siglip", case="report",
                                    source_revision="a" * 40, roots=(tmp_path,))
    events, payloads, failures = [], [], {}

    def observe(operation, text):
        inputs = copy.deepcopy(recorder.data["inputs"])
        events.append((operation, text, inputs))
        assert inputs["text"] == inputs["query"] == text
        assert recorder.data.get("native") is None
        if operation == "relevance":
            assert recorder.data.get("reference") is None
        if text == case["negative_query"] and failures.get("operation") == operation:
            raise RuntimeError("current negative operation failed")

    def embedding(*args):
        if "image_to_embedding" in args:
            return [1.0, 0.0]
        text = args[args.index("--text") + 1]
        observe("text", text)
        return vectors[text]

    def relevance(*args):
        text = args[args.index("--query") + 1]
        observe("relevance", text)
        logit = scale * vectors[text][0] + bias
        score = 1.0 / (1.0 + math.exp(-logit))
        if text == case["negative_query"] and failures.get("bad_score"):
            score += 0.1
        payload = {"score": score, "score_kind": "probability"}
        payloads.append(payload)
        return payload

    monkeypatch.setattr(e2e, "_native_embedding", embedding)
    monkeypatch.setattr(e2e, "_run_json", relevance)

    def run(**failure):
        failures.update(failure)
        token = e2e_evidence._ACTIVE.set(recorder)
        try:
            e2e.test_official_checkpoint_e2e("report", tmp_path)
        finally:
            e2e_evidence._ACTIVE.reset(token)

    return recorder, run, events, payloads, case


def test_positive_and_negative_operations_keep_their_own_inputs(recorded_case):
    recorder, run, events, payloads, case = recorded_case
    original = copy.deepcopy(case)
    run()
    assert [(operation, text) for operation, text, _ in events] == [
        ("text", case["query"]), ("text", case["negative_query"]),
        ("relevance", case["query"]), ("relevance", case["negative_query"]),
    ]
    assert case == original and recorder.data["inputs"]["case"] == original
    assert recorder.data["inputs"]["text"] == case["negative_query"]
    assert all("text" not in payload for payload in payloads)
    top = render_case(recorder.data, recorder.directory).split('<details class="case-details">', 1)[0]
    assert "NEGATIVE_QUERY" in top
    assert "POSITIVE_QUERY" not in top and "RECIPE_DEFAULT_PROMPT" not in top
    assert "Native pair relevance probability:" in top
    assert "Independent reference logit: -2.5" in top
    assert "comparison uses logit space" in top
    assert "<th>Native</th><th>Reference</th>" not in top
    assert recorder.data["native"]["score_kind"] == "probability"
    assert recorder.data["reference"]["score_kind"] == "logit"


@pytest.mark.parametrize("operation", ["text", "relevance"])
def test_failed_negative_call_cannot_reuse_the_previous_output(recorded_case, operation):
    recorder, run, _, _, case = recorded_case
    with pytest.raises(RuntimeError, match="current negative operation failed"):
        run(operation=operation)
    assert recorder.data["inputs"]["text"] == case["negative_query"]
    assert recorder.data["native"] is None
    if operation == "relevance":
        assert recorder.data["reference"] is None
    assert any(item["name"] == "native" and item["value"] for item in recorder.data["observations"])


def test_failed_negative_comparison_keeps_the_matching_reference(recorded_case):
    recorder, run, _, _, case = recorded_case
    with pytest.raises(AssertionError):
        run(bad_score=True)
    assert recorder.data["inputs"]["query"] == case["negative_query"]
    assert recorder.data["reference"]["score"] == -2.5
