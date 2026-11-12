# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A failed independent validator keeps its written proof and original exception."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from families.qwen3_5_moe.tests import test_e2e as e2e
from tools import e2e_evidence


@pytest.mark.parametrize("proof_problem", [None, "malformed_json", "recording_error"])
def test_failed_reference_driver_keeps_proof_without_replacing_its_exception(tmp_path, monkeypatch, proof_problem):
    case = {"name": "report", "test_image": "input.png", "prompt": "Describe this image."}
    manifest = {"name": "report", "family": "qwen3_5_moe", "task": "vision_language_generation",
                "precision": "fp32", "hf_id": "example/model", "hf_revision": "b" * 40,
                "max_sequence_length": 128, "tensor_parallel_size": 1, "bundle": "model.bundle",
                "testcases": [case]}
    monkeypatch.setattr(e2e, "MANIFEST", manifest)
    monkeypatch.setattr(e2e, "TEST_DIR", tmp_path)
    monkeypatch.setattr(e2e, "_selection", lambda *args: {"report"})
    monkeypatch.setattr(e2e, "build", lambda *args: None)
    monkeypatch.setenv("TRTMC_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setenv("TRTMC_NATIVE_BUILD_DIR", str(tmp_path))
    for name in ("libtrtmc_model_qwen3_5_moe.so", "libtrtmc_backend_trt.so"):
        (tmp_path / name).touch()
    helper = tmp_path / "families/qwen3_5_moe/qwen3_5_moe_edge_inference"
    helper.parent.mkdir(parents=True)
    helper.touch()
    helper.chmod(0o755)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        constants=SimpleNamespace(HF_HUB_OFFLINE=True), snapshot_download=lambda **kwargs: str(tmp_path),
    ))
    calls = []
    native = {"results": [{"id": "raw", "text": "wrong", "token_ids": [1]}]}
    reference = {"threshold": 0.15, "passed": False, "results": [
        {"id": "raw", "prompt": "Actual raw prompt", "actual_text": "wrong", "text": "right",
         "token_ids": [2], "normalized_edit_distance": 1.0, "passed": False},
    ]}
    proof = {"passed": False, "hf_quality": "failed", "hf": reference}
    original = subprocess.CalledProcessError(1, ["independent-reference"], stderr="quality gate failed")

    def gpu(arguments):
        calls.append("gpu")
        arguments.output.mkdir()
        (arguments.output / "public.json").write_text(json.dumps(native))
        (arguments.output / "public.requests.json").write_text(json.dumps([
            {"id": "raw", "prompt": "Actual raw prompt"},
        ]))

    def driver(*args, **kwargs):
        calls.append("reference")
        output = tmp_path / "proof"
        (output / "hf.json").write_text(json.dumps(reference))
        (output / "evidence.json").write_text("{" if proof_problem == "malformed_json" else json.dumps(proof))
        raise original

    monkeypatch.setattr(e2e.edge_validation, "gpu", gpu)
    monkeypatch.setattr(e2e.edge_validation, "run_driver", driver)
    if proof_problem == "recording_error":
        def broken_recording(*args):
            raise OSError("report storage unavailable")
        monkeypatch.setattr(e2e, "record_failed_reference_proof", broken_recording)
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="qwen3_5_moe", case="report",
                                    source_revision="a" * 40, roots=(tmp_path,))
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        with pytest.raises(subprocess.CalledProcessError) as raised:
            e2e.test_official_checkpoint_e2e("report", tmp_path, SimpleNamespace(config=None))
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert raised.value is original
    assert calls == ["gpu", "reference"]
    if proof_problem != "recording_error":
        assert recorder.data["native"]["text"] == "wrong"
        assert recorder.data["reference"]["text"] == "right"
        assert recorder.data["inputs"]["prompt"] == "Actual raw prompt"
        assert recorder.data["inputs"]["media"] == []
        assert recorder.data["diagnostics"]["failed_reference_comparison"]["passed"] is False
        assert recorder.data["thresholds"]["normalized_text_edit_distance"] == 0.15
        assert any(row["name"] == "native" and row["value"] == native for row in recorder.data["observations"])
        assert "reference_comparison" not in recorder.data
    if proof_problem == "malformed_json":
        assert "evidence.json" in recorder.data["proof_recording_error"]
