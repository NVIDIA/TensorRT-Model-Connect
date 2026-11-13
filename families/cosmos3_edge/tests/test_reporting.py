# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each displayed Cosmos request keeps only its own input media."""

import copy

import numpy as np

from families.cosmos3_edge.tests.reporting import record_request_inputs
from tools import e2e_evidence


def test_image_then_text_request_does_not_reuse_image_or_output(tmp_path):
    image = tmp_path / "image.rgb"
    np.arange(12, dtype=np.uint8).tofile(image)
    request = {"prompt": "Describe this image", "image": {"path": str(image), "height": 2, "width": 2}}
    original = copy.deepcopy(request)
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="cosmos3_edge", case="report",
                                    source_revision="a" * 40, roots=(tmp_path,))
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        record_request_inputs(request)
        assert recorder.data["inputs"]["media"]
        e2e_evidence.record_evidence("native", {"text": "image output"})
        e2e_evidence.record_evidence("reference", {"text": "image reference"})
        record_request_inputs({"prompt": "What is 7 times 8?"})
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert request == original
    assert recorder.data["inputs"]["prompt"] == "What is 7 times 8?"
    assert recorder.data["inputs"]["media"] == []
    assert recorder.data["native"] is recorder.data["reference"] is None
    assert recorder.data["views"] == []
    assert any(row["name"] == "native" and row["value"] for row in recorder.data["observations"])


def test_batch_comparisons_have_distinct_recorded_request_indices(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    from families.cosmos3_edge.tests import test_e2e as e2e
    from tools.e2e_report import _assess_reference_comparisons

    requests = [{"prompt": f"Question {index}"} for index in range(3)]
    outputs = [{"text": f"Answer {index}", "token_ids": [index]} for index in range(3)]
    manifest = {"family": "cosmos3_edge", "name": "report", "hf_id": "example/model", "trust_remote_code": False,
                "bundle": "model.bundle", "task": "vision_language_generation", "precision": "fp32",
                "max_sequence_length": 128, "tensor_parallel_size": 1}
    case = {"name": "report", "reference_backend": "hf_transformers", "reference_precision": "fp32"}
    monkeypatch.setattr(e2e, "CASES", {"report": (manifest, case)})
    monkeypatch.setattr(e2e, "ROOT", tmp_path)
    monkeypatch.setattr(e2e, "_selected", lambda *args: None)
    monkeypatch.setattr(e2e, "_runner", lambda *args: tmp_path / "runner")
    monkeypatch.setattr(e2e, "_checkpoint", lambda *args: tmp_path)
    monkeypatch.setattr(e2e, "_inputs", lambda *args: (requests, None, None, None))
    monkeypatch.setattr(e2e, "build", lambda *args: None)
    monkeypatch.setenv("TRTMC_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    (tmp_path / "libtrtmc_model_cosmos3_edge.so").touch()
    (tmp_path / "thresholds").mkdir()
    (tmp_path / "thresholds/report.json").write_text(json.dumps({"threshold_overrides": {"normalized_text_edit_distance": 0.0}}))
    calls = []

    def native(command, **kwargs):
        calls.append("native")
        (tmp_path / "output.json").write_text(json.dumps(outputs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def reference(*args):
        calls.append("reference")
        return outputs

    monkeypatch.setattr(e2e.subprocess, "run", native)
    monkeypatch.setattr(e2e, "_reference", reference)
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="cosmos3_edge", case="report",
                                    source_revision="a" * 40, roots=(tmp_path,))
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        e2e.test_e2e("report", SimpleNamespace(config=None), tmp_path)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert calls == ["native", "reference"]
    comparisons = _assess_reference_comparisons(recorder.data)
    assert len(comparisons) == 3
    assert all(row["label"].startswith(f"Request index {index}:") for index, row in enumerate(comparisons))
