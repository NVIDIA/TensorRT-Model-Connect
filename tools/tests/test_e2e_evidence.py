# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import e2e_evidence
from tools.e2e_evidence import Evidence, evidence_stage, record_evidence
from tools.e2e_report import render_case

pytest_plugins = ("pytester",)


def _recorder(tmp_path: Path) -> Evidence:
    return Evidence(
        tmp_path / "report",
        family="example",
        case="example-case",
        source_revision="a" * 40,
        roots=(tmp_path,),
    )


def test_capture_keeps_original_values_and_snapshots_repeated_outputs(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    token = e2e_evidence._ACTIVE.set(recorder)
    payload = {"text": "first"}
    try:
        assert record_evidence("native", payload) is payload
        payload["text"] = "second"
        record_evidence("native", payload)
        with evidence_stage("compare"):
            with pytest.raises(AssertionError):
                assert False, "the original comparator still fails"
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert recorder.data["observations"][0]["value"] == {"text": "first"}
    assert recorder.data["native"] == {"text": "second"}
    assert len(recorder.data["observations"]) == 2


def test_context_merges_but_independent_outputs_never_mix(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    recorder.record("inputs", {"manifest": {"family": "example"}})
    recorder.record("inputs", {"prompt": "hello"})
    assert recorder.data["inputs"] == {"manifest": {"family": "example"}, "prompt": "hello"}
    recorder.record("native", {"text": "first", "token_ids": [1]})
    recorder.record("native", {"text": "second"})
    assert recorder.data["native"] == {"text": "second"}
    assert recorder.data["observations"][-2]["value"]["token_ids"] == [1]


def test_files_are_snapshotted_before_reuse_and_arrays_remain_replayable(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    path = tmp_path / "output.txt"
    path.write_text("first")
    recorder.record("native", path)
    first = recorder.data["native"]["artifact"]
    path.write_text("second")
    recorder.record("native", path)
    second = recorder.data["native"]["artifact"]
    assert first != second
    assert (recorder.directory / first).read_text() == "first"
    assert (recorder.directory / second).read_text() == "second"
    values = np.arange(256, dtype=np.float32).reshape(16, 16)
    recorder.record("reference", values)
    restored = np.load(
        recorder.directory / recorder.data["reference"]["artifact"], allow_pickle=False
    )
    assert np.array_equal(restored, values)


def test_capture_rejects_symlinks_and_bounds_large_files(tmp_path: Path, monkeypatch) -> None:
    recorder = _recorder(tmp_path)
    path = tmp_path / "sample.txt"
    path.write_text("retained outside link")
    link = tmp_path / "link.txt"
    link.symlink_to(path)
    recorder.record("native", link)
    assert recorder.data["native"]["available"] is False
    monkeypatch.setattr(e2e_evidence, "_FILE_LIMIT", 1)
    recorder.record("reference", path)
    assert recorder.data["reference"]["omitted"] == "size limit"
    recorder.finish("passed")
    assert (
        json.loads((recorder.directory / "evidence.json").read_text())["evidence_status"]
        == "partial"
    )


def test_html_escapes_observations_and_never_executes_active_artifacts(tmp_path: Path) -> None:
    data = {
        "family": "example",
        "case": "<script>alert(1)</script>",
        "status": "passed",
        "native": {"text": "<img src=x onerror=alert(1)>"},
        "artifacts": [{"path": "../../secret.svg", "media_type": "image/svg+xml"}],
    }
    report = render_case(data, tmp_path)
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in report
    assert "<img src=x onerror=" not in report
    assert "Artifact path must remain within its testcase" in report
    assert "No assertion measurements were recorded" in report
    assert "fetch(" not in report


def test_json_bound_preserves_failed_check_before_passed_details(
    tmp_path: Path, monkeypatch
) -> None:
    recorder = _recorder(tmp_path)
    monkeypatch.setattr(e2e_evidence, "_JSON_LIMIT", 2048)
    recorder.data["checks"] = [
        {"status": "passed", "expression": "setup check", "explanation": "x" * 2000},
        {"status": "failed", "expression": "measured >= threshold", "explanation": "0.8 >= 0.9"},
    ]
    recorder.data["diagnostics"] = "x" * 20000
    recorder.finish("failed", failure="model comparison failed")
    source = recorder.directory / "evidence.json"
    value = json.loads(source.read_text())
    assert source.stat().st_size <= 2048
    assert value["status"] == "failed" and value["evidence_status"] == "partial"
    assert any(
        check["status"] == "failed" and "0.8" in check["explanation"] for check in value["checks"]
    )


def test_real_pytest_failure_retains_passed_checks_outputs_and_failure_stage(
    pytester, monkeypatch
) -> None:
    evidence_root = pytester.path / "captured"
    monkeypatch.setenv("TRTMC_E2E_ARTIFACT_DIR", str(evidence_root))
    monkeypatch.setenv("TRTMC_E2E_SOURCE_REVISION", "b" * 40)
    monkeypatch.setenv("TRTMC_E2E_WORKFLOW_ATTEMPT", "2")
    pytester.makeini("[pytest]\nenable_assertion_pass_hook = true\n")
    pytester.makeconftest('pytest_plugins = ("tools.e2e_evidence",)')
    path = pytester.path / "families/example/tests/test_e2e.py"
    path.parent.mkdir(parents=True)
    path.write_text("""
import pytest
from tools.e2e_evidence import record_evidence, evidence_stage
@pytest.mark.parametrize("case_name", ["example-case"])
def test_official_checkpoint_e2e(case_name, tmp_path):
    import sys
    record_evidence("inputs", {"prompt": "hello"})
    with evidence_stage("native"):
        record_evidence("native", {"text": "actual"})
    with evidence_stage("reference"):
        record_evidence("reference", {"text": "expected"})
    with evidence_stage("compare"):
        measured = 0.8
        assert measured > 0
        print("native diagnostic tail", file=sys.stderr)
        assert measured >= 0.9
""")
    result = pytester.runpytest(str(path), "-q", "-p", "no:cacheprovider")
    result.assert_outcomes(failed=1)
    evidence = json.loads((evidence_root / "evidence/example-case/evidence.json").read_text())
    assert evidence["status"] == "failed"
    assert evidence["source_revision"] == "b" * 40
    assert evidence["workflow_run_attempt"] == 2
    assert evidence["native"] == {"text": "actual"}
    assert evidence["reference"] == {"text": "expected"}
    assert evidence["failure_stage"] == "compare"
    assert "native diagnostic tail" in json.dumps(evidence["captured_output"])
    assert any(
        check["status"] == "passed" and "0.8" in check["explanation"]
        for check in evidence["checks"]
    )
    assert any(
        check["status"] == "failed" and "0.9" in check["explanation"]
        for check in evidence["checks"]
    )
    assert (evidence_root / "evidence/example-case/report.html").is_file()


def test_exact_selection_does_not_write_unselected_skipped_evidence(pytester, monkeypatch) -> None:
    evidence_root = pytester.path / "captured"
    monkeypatch.setenv("TRTMC_E2E_ARTIFACT_DIR", str(evidence_root))
    monkeypatch.setenv("TRTMC_E2E_SOURCE_REVISION", "c" * 40)
    pytester.makeini("[pytest]\nenable_assertion_pass_hook = true\n")
    pytester.makeconftest("""
pytest_plugins = ("tools.e2e_evidence",)
def pytest_addoption(parser):
    parser.addoption("--e2e-testcase", action="append", default=[])
""")
    path = pytester.path / "families/example/tests/test_e2e.py"
    path.parent.mkdir(parents=True)
    path.write_text("""
import pytest
from tools.e2e_evidence import record_evidence
@pytest.mark.parametrize("case_name", ["selected", "not-selected"])
def test_e2e(case_name):
    record_evidence("inputs", {"case": case_name})
    if case_name == "not-selected":
        pytest.skip("not selected")
    record_evidence("native", {"text": "hello"})
    record_evidence("reference", {"text": "hello"})
    assert 1 == 1
""")
    result = pytester.runpytest(
        str(path), "--e2e-testcase", "selected", "-q", "-p", "no:cacheprovider"
    )
    result.assert_outcomes(passed=1, skipped=1)
    assert (evidence_root / "evidence/selected/evidence.json").is_file()
    assert not (evidence_root / "evidence/not-selected").exists()
