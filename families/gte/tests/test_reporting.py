# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A later request must not inherit an earlier embedding or reference."""

import copy
import subprocess

import pytest

from families.gte.tests import test_e2e as e2e
from tools import e2e_evidence
from tools.e2e_report import render_case


@pytest.mark.parametrize("failure", [None, "native", "reference"])
def test_each_embedding_request_has_its_own_current_evidence(tmp_path, monkeypatch, failure):
    case = {"name": "report", "query": "QUERY", "document": "DOCUMENT", "negative_document": "NEGATIVE"}
    manifest = {"name": "report", "family": "gte", "task": "text_to_embedding", "precision": "fp32",
                "bundle": "model.bundle", "hf_id": "example/model"}
    original = copy.deepcopy(case)
    monkeypatch.setattr(e2e, "CASES", {"report": (tmp_path / "manifest.json", manifest, case)})
    monkeypatch.setattr(e2e, "_model_dir", lambda *args: tmp_path)
    monkeypatch.setattr(e2e, "_runtime", lambda: (tmp_path / "binary", tmp_path))
    monkeypatch.setattr(e2e, "_build", lambda *args: None)
    values = {"QUERY": [1.0, 2.0, 3.0], "DOCUMENT": [1.1, 2.1, 3.1], "NEGATIVE": [-1.0, -2.0, -3.0]}
    events = []
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="gte", case="report",
                                    source_revision="a" * 40, roots=(tmp_path,))

    def native(binary, runtime, bundle, settings, text, *roles):
        if roles:
            raise subprocess.CalledProcessError(1, "unsupported role")
        if text == "DOCUMENT" and failure == "native":
            assert recorder.data["inputs"]["text"] == text
            assert recorder.data["native"] is recorder.data["reference"] is None
            raise RuntimeError("second native failed")
        events.append(("native", text))
        return {"values": values[text], "pooling": "cls", "normalization": "none", "embedding_space": ""}

    def reference(model, text):
        text = text.removeprefix("search_query: ").removeprefix("search_document: ")
        assert recorder.data["inputs"]["text"] == text
        assert recorder.data["reference"] is None and recorder.data["views"] == []
        events.append(("reference", text))
        if text == "DOCUMENT" and failure == "reference":
            raise RuntimeError("second reference failed")
        return values[text]

    monkeypatch.setattr(e2e, "_native", native)
    monkeypatch.setattr(e2e, "_reference", reference)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        if failure:
            with pytest.raises(RuntimeError, match="second " + failure + " failed"):
                e2e.test_official_checkpoint_e2e("report", tmp_path)
        else:
            e2e.test_official_checkpoint_e2e("report", tmp_path)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert case == original
    assert recorder.data["inputs"]["case"] == original
    if failure:
        assert recorder.data["inputs"]["text"] == "DOCUMENT"
        assert recorder.data["reference"] is None
        if failure == "native":
            assert recorder.data["native"] is None
        else:
            assert recorder.data["native"]["values"] == values["DOCUMENT"]
        recorder.finish("failed", failure="current operation failed")
        top = render_case(recorder.data, recorder.directory).split('<details class="case-details">', 1)[0]
        assert "DOCUMENT" in top and "QUERY" not in top
    else:
        assert recorder.data["inputs"]["text"] == "NEGATIVE"
        assert recorder.data["native"]["values"] == recorder.data["reference"]["values"] == values["NEGATIVE"]
    assert [(kind, text) for kind, text in events if kind == "reference"][0] == ("reference", "QUERY")
