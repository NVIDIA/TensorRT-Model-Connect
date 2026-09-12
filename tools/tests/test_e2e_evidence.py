# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import copy
import unittest
import json
import struct
import zlib
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest

from tools import e2e_evidence
from tools.e2e_evidence import Evidence, evidence_stage, record_evidence
from tools.e2e_report import (
    NPY_MAX_BYTES,
    _assessment as assessment,
    classification_index,
    decode_npy,
    _numeric_preview,
    _output_summary,
    render_case,
    render_report,
)

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
    evidence = json.loads(
        (evidence_root / "evidence/example/example-case/evidence.json").read_text()
    )
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
    assert (evidence_root / "evidence/example/example-case/report.html").is_file()


def test_same_case_in_different_families_preserves_each_result(pytester, monkeypatch) -> None:
    from tools.e2e_report import main as render_main

    evidence_root = pytester.path / "captured"
    monkeypatch.setenv("TRTMC_E2E_ARTIFACT_DIR", str(evidence_root))
    monkeypatch.setenv("TRTMC_E2E_SOURCE_REVISION", "b" * 40)
    monkeypatch.setenv("EVIDENCE_TEST_GENERATION", "first")
    monkeypatch.setattr(e2e_evidence, "_environment", lambda: {})
    pytester.makeini("[pytest]\nenable_assertion_pass_hook = true\n")
    pytester.makeconftest('pytest_plugins = ("tools.e2e_evidence",)')
    paths = []
    for family in ("alpha", "beta"):
        path = pytester.path / "families" / family / "tests/test_e2e.py"
        path.parent.mkdir(parents=True)
        path.write_text(f"""
import os
import pytest
from tools.e2e_evidence import record_evidence
@pytest.mark.parametrize("case_name", ["shared-case"])
def test_e2e(case_name):
    record_evidence("inputs", {{"prompt": "{family} input"}})
    record_evidence("native", {{"text": "{family} " + os.environ["EVIDENCE_TEST_GENERATION"]}})
    assert 1 == 1
""")
        paths.append(str(path))

    pytester.runpytest(*paths, "--import-mode=importlib", "-q").assert_outcomes(passed=2)
    alpha = evidence_root / "evidence/alpha/shared-case"
    beta = evidence_root / "evidence/beta/shared-case"
    beta_json = (beta / "evidence.json").read_bytes()
    beta_html = (beta / "report.html").read_bytes()
    assert json.loads((alpha / "evidence.json").read_text())["native"] == {"text": "alpha first"}
    assert json.loads(beta_json)["native"] == {"text": "beta first"}
    stale = alpha / "stale.txt"
    stale.write_text("previous attempt")

    monkeypatch.setenv("EVIDENCE_TEST_GENERATION", "second")
    pytester.runpytest(paths[0], "--import-mode=importlib", "-q").assert_outcomes(passed=1)
    assert json.loads((alpha / "evidence.json").read_text())["native"] == {"text": "alpha second"}
    assert not stale.exists()
    assert (beta / "evidence.json").read_bytes() == beta_json
    assert (beta / "report.html").read_bytes() == beta_html
    report = pytester.path / "combined.html"
    assert render_main([str(evidence_root), "-o", str(report)]) == 0
    assert "2 cases shown" in report.read_text()
    assert "alpha second" in report.read_text() and "beta first" in report.read_text()


@pytest.mark.parametrize("component", ["evidence", "family", "case"])
def test_case_cleanup_rejects_symlinked_namespace(tmp_path: Path, component: str) -> None:
    root = tmp_path / "evidence"
    target = tmp_path / "retained"
    target.mkdir()
    preserved = target / "keep.txt"
    preserved.write_text("original")
    if component == "evidence":
        root.symlink_to(target, target_is_directory=True)
    elif component == "family":
        root.mkdir()
        (root / "alpha").symlink_to(target, target_is_directory=True)
    else:
        (root / "alpha").mkdir(parents=True)
        (root / "alpha/shared-case").symlink_to(target, target_is_directory=True)
    recorder = Evidence(
        root / "alpha/shared-case", family="alpha", case="shared-case", source_revision="a" * 40
    )
    with pytest.raises(ValueError, match="must not be a symlink"):
        recorder.record("native", {"text": "new result"})
    assert preserved.read_text() == "original"
    assert list(target.iterdir()) == [preserved]


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
    assert (evidence_root / "evidence/example/selected/evidence.json").is_file()
    assert not (evidence_root / "evidence/example/not-selected").exists()


@pytest.mark.parametrize("outcome", ["passed", "failed", "skipped"])
@pytest.mark.parametrize("failure", ["finish", "json", "write", "render", "captured_output"])
def test_reporting_failures_preserve_pytest_outcomes(pytester, monkeypatch, outcome, failure):
    from tools import e2e_report

    evidence_root = pytester.path / "captured"
    monkeypatch.setenv("TRTMC_E2E_ARTIFACT_DIR", str(evidence_root))
    monkeypatch.setenv("TRTMC_E2E_SOURCE_REVISION", "a" * 40)
    monkeypatch.setattr(e2e_evidence, "_environment", lambda: {})
    original_finish = Evidence.finish
    original_record = Evidence.record

    def broken_finish(self, *args, **kwargs):
        if failure == "finish":
            raise OSError("synthetic finish failure")
        if failure == "json":
            self.data["unsupported_json_value"] = object()
        if failure == "write":
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / ".evidence.tmp").mkdir()
        return original_finish(self, *args, **kwargs)

    def broken_render(*args, **kwargs):
        raise ValueError("synthetic render failure")

    def broken_record(self, name, value):
        if name == "captured_output":
            raise OSError("synthetic captured output failure")
        return original_record(self, name, value)

    monkeypatch.setattr(Evidence, "finish", broken_finish)
    if failure == "render":
        monkeypatch.setattr(e2e_report, "render_case", broken_render)
    if failure == "captured_output":
        monkeypatch.setattr(Evidence, "record", broken_record)
    pytester.makeini("[pytest]\nenable_assertion_pass_hook = true\n")
    pytester.makeconftest('pytest_plugins = ("tools.e2e_evidence",)')
    path = pytester.path / "families/example/tests/test_e2e.py"
    path.parent.mkdir(parents=True)
    path.write_text(f"""
import pytest
from tools.e2e_evidence import record_evidence
@pytest.mark.parametrize("case_name", ["example-case"])
def test_e2e(case_name):
    record_evidence("inputs", {{"prompt": "A synthetic prompt"}})
    print("A captured diagnostic")
    if {outcome!r} == "skipped":
        pytest.skip("the original skip")
    assert {outcome!r} != "failed", "the original assertion failure"
""")
    junit = pytester.path / "results.xml"
    result = pytester.runpytest(
        str(path), "-q", "-W", "error", "-p", "no:cacheprovider", f"--junitxml={junit}"
    )

    result.assert_outcomes(**{outcome: 1}, errors=0)
    assert result.ret == (1 if outcome == "failed" else 0)
    output = result.stdout.str()
    assert "[evidence]" in output and "Could not" in output
    assert "INTERNALERROR" not in output and "PluggyTeardownRaisedWarning" not in output
    assert 'name="trtmc_evidence_error"' in junit.read_text()
    if outcome == "failed":
        assert "the original assertion failure" in output
    if failure == "captured_output":
        evidence = json.loads(
            (evidence_root / "evidence/example/example-case/evidence.json").read_text()
        )
        assert evidence["status"] == outcome and evidence["evidence_status"] == "partial"


@pytest.mark.parametrize("selection", ["model", "models_file", "not_enabled"])
def test_unselected_skip_with_captured_output_preserves_existing_evidence(
    pytester, monkeypatch, selection
):
    evidence_root = pytester.path / "captured"
    previous = evidence_root / "evidence/example/example-case/evidence.json"
    previous.parent.mkdir(parents=True)
    previous.write_text('{"retained": "previous evidence"}\n')
    original = previous.read_bytes()
    monkeypatch.setenv("TRTMC_E2E_ARTIFACT_DIR", str(evidence_root))
    monkeypatch.setenv("TRTMC_E2E_SOURCE_REVISION", "a" * 40)
    monkeypatch.delenv("TRTMC_E2E", raising=False)
    monkeypatch.setattr(e2e_evidence, "_environment", lambda: {})
    pytester.makeini("[pytest]\nenable_assertion_pass_hook = true\n")
    pytester.makeconftest("""
import pytest
pytest_plugins = ("tools.e2e_evidence",)
def pytest_addoption(parser):
    parser.addoption("--e2e-model", action="append", default=[])
    parser.addoption("--e2e-models-file")
@pytest.fixture(autouse=True)
def captured_setup_and_teardown():
    print("captured setup output")
    yield
    print("captured teardown output")
""")
    path = pytester.path / "families/example/tests/test_e2e.py"
    path.parent.mkdir(parents=True)
    path.write_text("""
import os
from pathlib import Path
import pytest
from tools.e2e_evidence import record_evidence
@pytest.mark.parametrize("case_name", ["example-case"])
def test_e2e(case_name, request):
    print("captured call output")
    selected = set(request.config.getoption("--e2e-model"))
    models_file = request.config.getoption("--e2e-models-file")
    if models_file:
        selected.update(Path(models_file).read_text().splitlines())
    if not selected and os.environ.get("TRTMC_E2E") != "1":
        pytest.skip("E2E is disabled")
    if selected and case_name not in selected:
        pytest.skip("not selected")
    record_evidence("inputs", {"prompt": "Selected input"})
    assert True
""")
    options = []
    if selection == "model":
        options = ["--e2e-model", "another-case"]
    elif selection == "models_file":
        models_file = pytester.path / "selected-models.txt"
        models_file.write_text("another-case\n")
        options = ["--e2e-models-file", str(models_file)]

    result = pytester.runpytest(str(path), *options, "-q", "-W", "error", "-p", "no:cacheprovider")

    result.assert_outcomes(skipped=1, errors=0)
    assert result.ret == 0
    assert previous.read_bytes() == original
    assert sorted(path.name for path in previous.parent.iterdir()) == ["evidence.json"]


class _DefaultText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.closed_details = 0
        self.hidden = 0
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "details":
            self.closed_details += 1
        if tag in {"style", "script"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag == "details":
            self.closed_details -= 1
        if tag in {"style", "script"}:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.closed_details and not self.hidden:
            self.text.append(data)


def _visible(report: str) -> str:
    parser = _DefaultText()
    parser.feed(report)
    return " ".join(parser.text)


def _expanded(report: str) -> str:
    return _visible(report.replace("<details", "<div").replace("</details>", "</div>"))


def _case() -> dict:
    return {
        "family": "example_model",
        "case": "example-greedy-short",
        "status": "passed",
        "source_revision": "a" * 40,
        "evidence_status": "complete",
        "inputs": {
            "manifest": {
                "name": "example-recipe",
                "hf_id": "example/model",
                "task": "text_generation",
                "precision": "fp16",
                "tensor_parallel_size": 2,
                "max_sequence_length": 1024,
            },
            "case": {"name": "example-greedy-short", "prompt": "Old prompt", "max_new_tokens": 8},
            "prompt": "What is the capital of France?",
        },
        "native": {"text": "Paris.", "token_ids": [42, 3]},
        "reference": {
            "reference_text": "Paris.",
            "reference_ids": [42, 3],
            "actual_decoded": "native diagnostic",
        },
        "checks": [
            {
                "status": "passed",
                "expression": "actual == expected",
                "explanation": "original operands",
            }
        ],
        "captured_output": [{"stream": "stderr", "text": "RAW_LOG_SENTINEL"}],
    }


def test_glance_view_uses_recorded_prompt_recipe_and_outputs(tmp_path: Path) -> None:
    report = render_case(_case(), tmp_path)
    visible = _visible(report)
    assert "What is the capital of France?" in visible
    assert "Old prompt" not in visible
    assert visible.count("Paris.") == 2
    assert "Native output" in visible and "Reference output" in visible
    assert "native diagnostic" not in visible
    assert "example-recipe" in visible and "example-greedy-short" not in visible
    assert "example-greedy-short" in report
    assert "example/model" in visible and "FP16 · TP2" in visible
    assert "families/example-model" in report
    assert 'class="output-pair text-comparison"' in report
    assert "RAW_LOG_SENTINEL" not in visible and "RAW_LOG_SENTINEL" in report
    assert "original operands" not in visible and "original operands" in report
    assert "a" * 40 not in visible and "a" * 40 in report
    assert "<details open" not in report


def test_reference_text_does_not_show_native_diagnostics(tmp_path: Path) -> None:
    data = _case()
    data["reference"] = {
        "reference_text": "Expected answer",
        "text": "secondary text",
        "actual_decoded": "wrong answer",
    }
    report = render_case(data, tmp_path)
    visible = _visible(report)
    reference = report.split("<h3>Reference output</h3>")[1].split("</div>")[0]
    assert "Expected answer" in visible and "Expected answer" in reference
    assert "secondary text" not in reference and "wrong answer" not in reference


def test_numeric_output_shows_dimensions_without_vectors(tmp_path: Path) -> None:
    data = _case()
    data["inputs"]["manifest"]["task"] = "embedding"
    values = [0.123456789] * 768
    data["native"] = {"dim": 768, "values": values}
    data["reference"] = {"values": {"shape": [768], "dtype": "float32", "preview": values[:64]}}
    report = render_case(data, tmp_path)
    visible = _visible(report)
    assert "Embedding vector" in visible and "768" in visible
    assert "0.123456789" not in visible and "0.123456789" in report
    assert "float32" not in visible and "float32" in report


@pytest.mark.parametrize("task", ["image_classification", "image_to_class_scores"])
def test_classification_preserves_ids_and_raw_score_meaning(tmp_path: Path, task: str) -> None:
    data = _case()
    data["inputs"]["manifest"]["task"] = task
    data["native"] = {"top_class": 7, "top_score": 12.5, "logits": [0.1] * 1000}
    data["reference"] = 7
    report = render_case(data, tmp_path)
    visible = _visible(report)
    assert "Class ID" in visible and "7" in visible
    assert "12.5" not in visible and "12.5" in report and "top_score" in report
    assert "1000" not in visible and "logits" in report
    assert "confidence" not in visible.lower() and "probability" not in visible.lower()


@pytest.mark.parametrize(
    "native_class,reference_class,status,checked",
    [
        (0, 0, "passed", True),
        (80, 80, "passed", True),
        (7, 80, "failed", False),
        (7, 80, "passed", False),
    ],
)
@pytest.mark.parametrize("task", ["classification", "image_to_class_scores"])
def test_classification_reference_is_visible_beside_native(
    tmp_path: Path, native_class: int, reference_class: int, status: str, checked: bool, task: str
) -> None:
    data = _case()
    data["inputs"]["manifest"]["task"] = task
    data["native"] = {"top_class": native_class}
    data["reference"] = {
        "top_class": reference_class,
        "logits": {"shape": [128], "preview": [0.0] * 64},
    }
    data["status"] = status
    data["checks"] = (
        [{"status": "passed", "expression": 'int(actual["top_class"]) == int(np.argmax(expected))'}]
        if checked
        else []
    )
    report = render_case(data, tmp_path)
    visible = _visible(report)
    before_details = report.split('<details class="case-details">', 1)[0]
    assert 'class="output-pair classification-comparison"' in before_details
    assert "Native output" in visible and "Reference output" in visible
    assert visible.count("Class ID") == 2
    assert f"Class ID <strong>{native_class}</strong>" in before_details
    assert f"Class ID <strong>{reference_class}</strong>" in before_details
    assert "128" not in visible and "logits" in report
    if status == "failed":
        assert 'data-status="failed"' in report
        assert 'class="badge reference"' not in before_details
    elif checked:
        assert "Top class matched the reference; full-logit equality is not asserted." in visible
    else:
        assert 'data-status="unverified"' in report
        assert 'class="badge reference"' not in before_details


def test_classification_scalar_reference_zero_is_visible(tmp_path: Path) -> None:
    data = _case()
    data["inputs"]["manifest"]["task"] = "image_classification"
    data["native"] = {"top_class": 0}
    data["reference"] = 0
    report = render_case(data, tmp_path)
    assert _visible(report).count("Class ID") == 2
    assert (
        report.split('<details class="case-details">', 1)[0].count("Class ID <strong>0</strong>")
        == 2
    )


@pytest.mark.parametrize("reference,expected_class", [({"top_class": 7}, 7), (0, 0)])
def test_classification_reference_only_stays_visible_on_failure(
    tmp_path: Path, reference: dict | int, expected_class: int
) -> None:
    data = _case()
    data["inputs"]["manifest"]["task"] = "classification"
    data["native"] = None
    data["reference"] = reference
    data["status"] = "failed"
    data["failure_stage"] = "native"
    report = render_case(data, tmp_path)
    before_details = report.split('<details class="case-details">', 1)[0]
    assert f"Class ID <strong>{expected_class}</strong>" in before_details
    assert _visible(report).count("Class ID") == 1
    assert "No native output was recorded" in _visible(report)
    assert 'data-status="failed"' in report


def test_classification_partial_reference_logits_do_not_invent_a_class(tmp_path: Path) -> None:
    data = _case()
    data["inputs"]["manifest"]["task"] = "classification"
    data["native"] = {"top_class": 7}
    data["reference"] = {"logits": {"shape": [128], "preview": [0, 10, 2]}}
    data["checks"] = []
    report = render_case(data, tmp_path)
    visible = _visible(report)
    assert visible.count("Class ID") == 1
    assert "Class preview unavailable" in visible
    assert 'data-status="unverified"' in report


@pytest.mark.parametrize("reference", [None, {"mode": "contract_only"}])
def test_classification_without_reference_does_not_invent_a_pair(
    tmp_path: Path, reference: dict | None
) -> None:
    data = _case()
    data["inputs"]["manifest"]["task"] = "classification"
    data["native"] = {"top_class": 0}
    data["reference"] = reference
    report = render_case(data, tmp_path)
    assert "classification-comparison" not in report
    assert _visible(report).count("Class ID") == 1


def test_classification_with_explicit_text_keeps_existing_text_layout(tmp_path: Path) -> None:
    data = _case()
    data["inputs"]["manifest"]["task"] = "classification"
    data["native"]["top_class"] = 7
    data["reference"]["top_class"] = 7
    report = render_case(data, tmp_path)
    assert 'class="output-pair text-comparison"' in report
    assert "classification-comparison" not in report
    assert _visible(report).count("Reference output") == 1


def test_contract_reference_and_failure_are_never_a_paired_pass(tmp_path: Path) -> None:
    data = _case()
    data["status"] = "failed"
    data["failure_stage"] = "reference"
    data["failure"] = {"traceback": "TRACEBACK_SENTINEL"}
    data["reference"] = {"mode": "contract_only", "oracle": "public core invariants"}
    data["evidence_status"] = "partial"
    report = render_case(data, tmp_path)
    visible = _visible(report)
    assert "Reference output" in visible and "No reference text recorded" in visible
    assert "Reference run failed" in visible and "Partial evidence" in visible
    assert "contract_only" in report and "public core invariants" in report
    assert 'data-status="failed"' in report
    assert "TRACEBACK_SENTINEL" not in visible and "TRACEBACK_SENTINEL" in report


def test_missing_settings_and_outputs_remain_explicit(tmp_path: Path) -> None:
    report = render_case({"family": "example", "case": "not-run", "status": "skipped"}, tmp_path)
    visible = _visible(report)
    assert "Not verified" in visible and "Completed execution evidence is unavailable" in visible
    assert "io-panel" not in report.split("<body>", 1)[1]
    assert (
        "Recipe not recorded" not in visible and "Build configuration not recorded" not in visible
    )
    assert "TP1" not in visible and "FP16" not in visible
    assert "No assertion measurements were recorded" in report


def test_run_settings_use_human_labels_and_original_keys(tmp_path: Path) -> None:
    report = render_case(_case(), tmp_path)
    assert "Maximum sequence length" in report and "max_sequence_length" in report
    assert "Maximum new tokens" in report and "max_new_tokens" in report
    assert "Recipe and build" in report and "Request and reference" in report
    assert "1024" not in _visible(report)
    assert "Missing fields have no assumed defaults" in report


def test_media_preserves_identical_bytes_in_distinct_recorded_roles(tmp_path: Path) -> None:
    data = _case()
    gif = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==")
    paths = ["input.gif", "native.gif", "reference.gif", "earlier.gif"]
    for path in paths:
        (tmp_path / path).write_bytes(gif)
    data["inputs"]["asset"] = {"artifact": paths[0]}
    data["native"] = {"artifact": paths[1]}
    data["reference"] = {"artifact": paths[2]}
    data["artifacts"] = [
        {"path": path, "role": "native" if path == "earlier.gif" else "observations", "label": path}
        for path in paths
    ]
    report = render_case(data, tmp_path)
    assert report.count("data:image/gif;base64,") == 3
    assert "Same recorded media as Input" not in report
    assert "More recorded media" not in report
    assert all(path in report for path in paths)
    assert all((tmp_path / path).read_bytes() == gif for path in paths)


def _rgb_png(path: Path, rgb: tuple[int, int, int]) -> str:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload))
        )

    pixels = (b"\0" + bytes(rgb) * 2) * 2
    content = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(content)
    return "data:image/png;base64," + base64.b64encode(content).decode("ascii")


class _MediaMarkup(HTMLParser):
    """Inspect visible media roles and their containing frame groups."""

    def __init__(self, report: str) -> None:
        super().__init__()
        self.groups = []
        self.figures = []
        self.sections = []
        self.details = 0
        self.figure = None
        self.in_caption = False
        self.feed(report)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "details":
            self.details += 1
        if tag == "section":
            group = None
            if "data-media-index" in attrs:
                group = {
                    "kind": attrs["data-media-kind"],
                    "index": int(attrs["data-media-index"]),
                    "paired": attrs["data-media-paired"] == "true",
                    "in_details": bool(self.details),
                    "figures": [],
                }
                self.groups.append(group)
            self.sections.append(group)
        if tag == "figure":
            self.figure = {
                "role": attrs.get("data-media-role"),
                "caption": "",
                "images": [],
                "in_details": bool(self.details),
            }
            self.figures.append(self.figure)
            group = next((item for item in reversed(self.sections) if item is not None), None)
            if group is not None:
                group["figures"].append(self.figure)
        if tag == "figcaption":
            self.in_caption = True
        if tag == "img" and self.figure is not None:
            self.figure["images"].append(attrs["src"])

    def handle_endtag(self, tag):
        if tag == "details":
            self.details -= 1
        if tag == "section":
            self.sections.pop()
        if tag == "figcaption":
            self.in_caption = False
        if tag == "figure":
            self.figure = None

    def handle_data(self, value):
        if self.in_caption and self.figure is not None:
            self.figure["caption"] += value


def _video_evidence(views: list[dict], artifacts: list[dict]) -> dict:
    data = _case()
    data["inputs"]["manifest"]["task"] = "world_model_generation"
    data["native"] = {"frames_count": 2}
    data["reference"] = {"frame_count": 2}
    data["views"] = views
    data["artifacts"] = artifacts
    return data


def test_media_process_inputs_cannot_replace_generated_output(tmp_path: Path) -> None:
    input_image = _rgb_png(tmp_path / "input.png", (220, 30, 20))
    (tmp_path / "process-input.png").write_bytes((tmp_path / "input.png").read_bytes())
    native_image = _rgb_png(tmp_path / "generated.png", (30, 40, 220))
    reference_image = _rgb_png(tmp_path / "reference.png", (20, 210, 40))
    data = _video_evidence(
        [
            {"title": "Frame 0 / native", "image": {"artifact": "generated.png"}},
            {"title": "Frame 0 / reference", "image": {"artifact": "reference.png"}},
        ],
        [
            {"path": "input.png", "role": "inputs"},
            {"path": "process-input.png", "role": "native_process", "label": "argv / image"},
            {"path": "generated.png", "role": "views"},
            {"path": "reference.png", "role": "views"},
        ],
    )
    data["inputs"]["image"] = {"artifact": "input.png"}
    data["native_process"] = {"argv": ["generate-world", {"artifact": "process-input.png"}]}
    original = copy.deepcopy(data)
    report = render_case(data, tmp_path)
    assert "Same recorded media as Input" not in _visible(report)
    markup = _MediaMarkup(report)
    default = [figure for figure in markup.figures if not figure["in_details"]]
    assert not any(
        input_image in figure["images"]
        for figure in markup.figures
        if figure["role"] in {"native", "reference"}
    )
    assert [figure["images"] for figure in default if figure["role"] == "inputs"] == [[input_image]]
    assert [figure["images"] for figure in default if figure["role"] == "native"] == [
        [native_image]
    ]
    assert [figure["images"] for figure in default if figure["role"] == "reference"] == [
        [reference_image]
    ]
    assert data == original


def test_media_canonical_frames_survive_earlier_identical_previews(tmp_path: Path) -> None:
    native_image = _rgb_png(tmp_path / "early-preview.png", (40, 70, 150))
    (tmp_path / "paired-native.png").write_bytes((tmp_path / "early-preview.png").read_bytes())
    reference_image = _rgb_png(tmp_path / "paired-reference.png", (120, 60, 30))
    canonical = [
        {
            "title": "Frame 64 / native",
            "image": {"artifact": "paired-native.png"},
            "caption": "Current native result",
        },
        {
            "title": "Frame 64 / reference",
            "image": {"artifact": "paired-reference.png"},
            "caption": "Current reference result",
        },
    ]
    data = _video_evidence(
        canonical,
        [
            {"path": "early-preview.png", "role": "views", "label": "image"},
            {"path": "paired-native.png", "role": "views", "label": "image"},
            {"path": "paired-reference.png", "role": "views", "label": "image"},
        ],
    )
    data["observations"] = [
        {
            "name": "views",
            "value": [
                {
                    "title": "Native frame 64",
                    "image": {"artifact": "early-preview.png"},
                    "caption": "Earlier preview",
                }
            ],
        },
        {"name": "views", "value": canonical},
    ]
    report = render_case(data, tmp_path)
    assert "Frame 64 / Native" in _visible(report)
    assert "Current native result" in _visible(report)
    groups = _MediaMarkup(report).groups
    assert len(groups) == 1 and groups[0]["paired"] and not groups[0]["in_details"]
    assert [(figure["role"], figure["images"]) for figure in groups[0]["figures"]] == [
        ("native", [native_image]),
        ("reference", [reference_image]),
    ]


@pytest.mark.parametrize("same_path", [False, True], ids=["copied-files", "shared-file"])
def test_media_identical_bytes_keep_both_roles_at_each_frame(
    tmp_path: Path, same_path: bool
) -> None:
    image = _rgb_png(tmp_path / "shared.png", (80, 90, 100))
    views, artifacts = [], []
    for index in (0, 64):
        for role in ("native", "reference"):
            path = "shared.png" if same_path else f"{role}-{index}.png"
            (tmp_path / path).write_bytes((tmp_path / "shared.png").read_bytes())
            views.append({"title": f"Frame {index} / {role}", "image": {"artifact": path}})
            if not same_path:
                artifacts.append({"path": path, "role": "views"})
    if same_path:
        artifacts.append({"path": "shared.png", "role": "views"})
    groups = _MediaMarkup(render_case(_video_evidence(views, artifacts), tmp_path)).groups
    assert [(group["index"], group["paired"], group["in_details"]) for group in groups] == [
        (0, True, False),
        (64, True, True),
    ]
    for group in groups:
        assert [(figure["role"], figure["images"]) for figure in group["figures"]] == [
            ("native", [image]),
            ("reference", [image]),
        ]


def test_media_more_views_compare_the_same_recorded_frame_index(tmp_path: Path) -> None:
    entries = [
        ("reference", 64),
        ("native", 128),
        ("native", 0),
        ("reference", 128),
        ("reference", 0),
        ("native", 64),
    ]
    views, artifacts, expected = [], [], {}
    for position, (role, index) in enumerate(entries):
        path = f"recorded-{position}.png"
        expected[role, index] = _rgb_png(tmp_path / path, (20 + position * 30, 60, 120))
        views.append({"title": f"Frame {index} / {role}", "image": {"artifact": path}})
        artifacts.append({"path": path, "role": "views"})
    report = render_case(_video_evidence(views, artifacts), tmp_path)
    groups = _MediaMarkup(report).groups
    assert [(group["index"], group["in_details"]) for group in groups] == [
        (0, False),
        (64, True),
        (128, True),
    ]
    for group in groups:
        assert group["paired"]
        assert [
            (figure["role"], figure["caption"], figure["images"]) for figure in group["figures"]
        ] == [
            (role, f"Frame {group['index']} / {role.title()}", [expected[role, group["index"]]])
            for role in ("native", "reference")
        ]


@pytest.mark.parametrize(
    "titles",
    [
        ["Frame 0 / native", "Frame 1 / reference"],
        ["Frame 0 / native", "Sample 0 / reference"],
        ["Frame 0 / native", "Frame 0 / native", "Frame 0 / reference"],
        ["Native preview", "Frame 0 / reference"],
    ],
    ids=["mismatched-index", "different-kind", "ambiguous-native", "filename-is-not-index"],
)
def test_media_unmatched_views_never_invent_a_frame_pair(tmp_path: Path, titles: list[str]) -> None:
    views, artifacts = [], []
    for index, title in enumerate(titles):
        path = f"frame-0-{index}.png"
        _rgb_png(tmp_path / path, (30 + index * 70, 50, 90))
        views.append({"title": title, "image": {"artifact": path}})
        artifacts.append({"path": path, "role": "views"})
    report = render_case(_video_evidence(views, artifacts), tmp_path)
    markup = _MediaMarkup(report)
    assert markup.groups and not any(group["paired"] for group in markup.groups)
    assert sum(len(figure["images"]) for figure in markup.figures) == len(titles)
    assert "No unambiguous native/reference pair recorded for this index." in _expanded(report)


@pytest.mark.parametrize("status", ["passed", "failed"])
def test_media_reference_only_frame_remains_visible_without_expanding(
    tmp_path: Path, status: str
) -> None:
    image = _rgb_png(tmp_path / "reference.png", (20, 80, 160))
    data = _video_evidence(
        [{"title": "Frame 64 / reference", "image": {"artifact": "reference.png"}}],
        [{"path": "reference.png", "role": "views"}],
    )
    data["native"] = {}
    data["status"] = status
    report = render_case(data, tmp_path)
    markup = _MediaMarkup(report)
    default = [figure for figure in markup.figures if not figure["in_details"]]
    assert [(figure["role"], figure["images"]) for figure in default] == [("reference", [image])]
    assert not any(group["paired"] for group in markup.groups)
    if status == "failed":
        assert 'data-execution-status="failed"' in report
        assert 'data-status="failed"' in report


def test_media_shared_input_and_indexed_outputs_keep_all_three_roles(tmp_path: Path) -> None:
    image = _rgb_png(tmp_path / "shared.png", (70, 110, 150))
    data = _video_evidence(
        [
            {"title": "Frame 0 / native", "image": {"artifact": "shared.png"}},
            {"title": "Frame 0 / reference", "image": {"artifact": "shared.png"}},
        ],
        [{"path": "shared.png", "role": "views"}],
    )
    data["inputs"]["image"] = {"artifact": "shared.png"}
    markup = _MediaMarkup(render_case(data, tmp_path))
    default = [figure for figure in markup.figures if not figure["in_details"]]
    assert sorted((figure["role"], figure["images"]) for figure in default) == [
        ("inputs", [image]),
        ("native", [image]),
        ("reference", [image]),
    ]
    assert len(markup.groups) == 1 and markup.groups[0]["paired"]
    assert [figure["role"] for figure in markup.groups[0]["figures"]] == ["native", "reference"]


def test_long_text_stays_readable_and_full_evidence_is_expandable(tmp_path: Path) -> None:
    data = _case()
    data["native"]["text"] = "hello " * 100 + "MIDDLE_SENTINEL" + "hello " * 100 + "FULL_TEXT_END"
    report = render_case(data, tmp_path)
    visible = _visible(report)
    assert "hello" in visible and "FULL_TEXT_END" in visible
    assert "middle omitted" not in visible
    assert "MIDDLE_SENTINEL" in visible
    assert "full text" in report and "FULL_TEXT_END" in report
    assert "no-results" in report and "aria-live" in report


def test_forecast_is_identified_as_last_window(tmp_path: Path) -> None:
    data = _case()
    data["inputs"]["manifest"]["task"] = "forecasting"
    data["native"] = {"shape": [1, 24], "values": [0.1] * 24}
    report = render_case(data, tmp_path)
    assert "Last recorded window" in _visible(report)
    assert "1 × 24" in _visible(report)


def test_family_and_settings_cannot_inject_markup(tmp_path: Path) -> None:
    data = _case()
    data["family"] = 'x" onclick="alert(1)'
    data["inputs"]["manifest"]["hf_id"] = "<script>alert(2)</script>"
    data["inputs"]["case"]["seed"] = "<img onerror=alert(3)>"
    report = render_report([(data, tmp_path)])
    assert "<script>alert(2)</script>" not in report
    assert "<img onerror=" not in report
    assert "%22%20onclick%3D%22" in report
    assert "https://nvidia.github.io/TensorRT-Model-Connect/" in report


def test_target_regression_preview_is_not_a_forecast() -> None:
    from tools.e2e_report import _demo_output

    output = _demo_output(
        {"kind": "regression_values", "values": [1.25, -2.5], "target_count": 2,
         "axes": ["target"], "target_names": [], "target_units": []},
        role="native", task="series_to_regression_values")
    assert "Regression target values" in output
    assert "1.25" in output and "-2.5" in output
    assert "All 2 values" in output
    assert "Forecast" not in output and "horizon" not in output


def test_config_values_keep_exact_spelling(tmp_path: Path) -> None:
    data = _case()
    data["inputs"]["case"]["sampling_mode"] = "sample_with_seed"
    report = render_case(data, tmp_path)
    assert "sample_with_seed" in report and "sample with seed" not in report


def test_structured_prompt_is_prose_with_original_collapsed(tmp_path: Path) -> None:
    data = _case()
    data["inputs"]["prompt"] = (
        '{"description":"A robot cleans a plate.","camera":"Close up","duration":"7s"}'
    )
    report = render_case(data, tmp_path)
    visible = _visible(report)
    assert "A robot cleans a plate." in visible and "Close up" in visible
    assert '{"description"' not in visible
    assert "Raw recorded fields" in report and "description" in report
    assert "camera" in report and "7s" in report


def test_forecast_uses_actual_last_input_not_initial_contract(tmp_path: Path) -> None:
    data = _case()
    data["inputs"] = {
        "manifest": {"task": "forecasting"},
        "case": {"inputs": {"past_values": [1.0] * 12}},
    }
    data["native"] = {"input_values": [2.0] * 96, "values": [3.0] * 24}
    visible = _visible(render_case(data, tmp_path))
    assert "Input values" in visible and "96" in visible
    assert "Past values" not in visible


def test_small_score_vectors_show_actual_values_and_documents(tmp_path: Path) -> None:
    data = _case()
    data["inputs"]["manifest"]["task"] = "reranking"
    data["inputs"]["case"]["inputs"] = {
        "documents": ["Candidate A says yes.", "Candidate B says no."]
    }
    data["native"] = {"scores": [-8.96, -10.27]}
    data["reference"] = {"scores": [-8.95, -10.28]}
    visible = _visible(render_case(data, tmp_path))
    assert "Candidate A says yes." in visible and "Candidate B says no." in visible
    assert "Scores" in visible and "All 2 values" in visible
    assert "-8.96" in visible and "-10.28" in visible


def test_masks_and_points_distinguish_counts_shapes_and_values(tmp_path: Path) -> None:
    data = _case()
    data["inputs"]["manifest"]["task"] = "segmentation"
    data["inputs"]["case"].update({"point_x": 0.5, "point_y": 0.25})
    data["native"] = {
        "num_masks": 3,
        "masks": {"shape": [733440], "preview": [0.2] * 64},
        "iou_scores": [0.821, 0.876, 0.986],
        "bbox_xyxy": [10, 20, 30, 40],
        "mask_foreground_pixels": [50, 51, 52, 53, 54],
    }
    report = render_case(data, tmp_path)
    visible = _visible(report)
    assert "Point X: 0.5" in visible and "Point Y: 0.25" in visible
    assert "recorded coordinates" in visible
    assert "All 3 values" in visible and "0.821" in visible
    assert "bbox_xyxy" in report and "40" in report
    assert "mask_foreground_pixels" in report and "54" in report
    assert "733440" not in visible and "733440" in report
    assert "normalized" not in visible.lower()


def test_numeric_preview_uses_same_display_scale_for_both_outputs() -> None:
    native = {"shape": [1, 9, 64], "values": list(range(576))}
    reference = {"values": {"shape": [1, 9, 64], "preview": [value * 2 for value in range(64)]}}
    first = _output_summary(native, role="native", task="forecasting", peer=reference)
    second = _output_summary(reference, role="reference", task="forecasting", peer=native)
    for preview in (first, second):
        assert "First 64 of 576 values" in preview
        assert "1 × 9 × 64" in preview and "flattened order" in preview
        assert "Shared native/reference scale" in preview and ">126</text>" in preview
        assert "<svg" in preview and "median" not in preview
    assert first != second


def test_forecast_input_has_bounded_actual_series_preview(tmp_path: Path) -> None:
    data = _case()
    data["inputs"] = {
        "manifest": {"task": "forecasting"},
        "case": {"inputs": {"past_values": [999] * 12}},
        "window_index": 9,
    }
    data["native"] = {
        "input_values": {"shape": [2048], "preview": [2] * 64},
        "shape": [1, 128],
        "values": [3] * 128,
    }
    data["reference"] = {"values": {"shape": [1, 128], "preview": [3] * 64}}
    report = render_case(data, tmp_path)
    visible = _visible(report)
    assert "First 64 of 2048 values" in visible and "First 64 of 128 values" in visible
    assert "Recorded window 9" in visible and "999" not in visible
    assert report.count("<svg") == 2
    assert report.count('aria-label="Native"') == 1
    assert report.count('aria-label="Reference"') == 1


def test_nested_recorded_summary_is_readable(tmp_path: Path) -> None:
    data = _case()
    data["inputs"]["manifest"]["task"] = "video_generation"
    data["native"] = {"num_frames": 12}
    data["reference"] = {"output": {"summary": {"frame_count": 12, "mean": 0.25}}}
    report = render_case(data, tmp_path)
    assert "Output · Summary · Frame count" not in _visible(report)
    expanded = _expanded(report)
    assert "Output · Summary · Frame count" in expanded
    assert "12" in expanded and "0.25" in expanded


def test_nested_request_options_are_settings_not_just_raw_json(tmp_path: Path) -> None:
    data = _case()
    data["inputs"]["case"]["inputs"] = {
        "cfg_scale": 4.5,
        "num_sampling_steps": 16,
        "generation_mode": "conditional_sample",
        "documents": ["LONG_DOCUMENT_SENTINEL"],
    }
    report = render_case(data, tmp_path)
    settings = report.split("<h3>Request input options</h3>")[1].split("</details>")[0]
    assert "CFG scale" in settings and "4.5" in settings
    assert "num_sampling_steps" in settings and "conditional_sample" in settings
    assert "LONG_DOCUMENT_SENTINEL" not in settings


def test_single_device_requires_both_recorded_parallel_sizes(tmp_path: Path) -> None:
    data = _case()
    data["inputs"]["manifest"]["tensor_parallel_size"] = 1
    assert "Single device" not in _visible(render_case(data, tmp_path))
    data["inputs"]["manifest"]["context_parallel_size"] = 1
    assert "Single device" in _visible(render_case(data, tmp_path))
    data["inputs"]["manifest"]["context_parallel_size"] = 2
    visible = _visible(render_case(data, tmp_path))
    assert "TP1 · CP2" in visible and "Single device" not in visible


def test_numeric_svg_cannot_contain_active_or_nonfinite_values() -> None:
    assert _numeric_preview([float("inf")] * 20, "Output") == ""
    assert _numeric_preview(["<script>bad</script>"] * 20, "Output") == ""
    report = _numeric_preview([1e308] * 20, '<img onerror="bad">')
    assert "<img onerror=" not in report and "&lt;img" in report
    assert "nan" not in report.lower() and "inf" not in report.lower()


def test_saved_complete_logits_supply_class_index_without_mutating_evidence(tmp_path: Path) -> None:
    import copy
    import io

    import numpy as np

    data = _case()
    data["inputs"]["manifest"]["task"] = "image_classification"
    data["native"] = {"top_class": 1, "top_score": 7.0}
    descriptor = {"shape": [1, 1000], "artifact": "logits.npy", "preview": [0.0] * 64}
    data["reference"] = {"logits": descriptor}
    logits = np.zeros((1, 1000), dtype=np.float32)
    logits[0, 777] = 9.0
    buffer = io.BytesIO()
    np.save(buffer, logits, allow_pickle=False)
    (tmp_path / "logits.npy").write_bytes(buffer.getvalue())
    original = copy.deepcopy(data)
    report = render_case(data, tmp_path)
    visible = _visible(report)
    reference = (
        report.split('<details class="case-details">', 1)[0]
        .split("<h4>Reference output</h4>")[1]
        .split("</div>")[0]
    )
    assert "From complete saved logits" in reference and "777" in reference
    assert "777" in visible
    assert visible.count("Class ID") == 2 and data == original
    assert "values" not in descriptor


def test_partial_logits_never_claim_a_derived_class_index(tmp_path: Path) -> None:
    data = _case()
    data["inputs"]["manifest"]["task"] = "image_classification"
    data["reference"] = {
        "logits": {"shape": [1, 1000], "preview": [100.0] * 64, "artifact": "absent.npy"}
    }
    report = render_case(data, tmp_path)
    assert "From complete saved logits" not in report
    assert "complete saved logits are unavailable" in _expanded(report)


def test_numeric_file_path_escape_and_symlinks_are_not_read(tmp_path: Path) -> None:
    import numpy as np

    root = tmp_path / "case"
    root.mkdir()
    np.save(tmp_path / "logits.npy", np.array([0.0, 10.0], dtype=np.float32))
    (root / "link.npy").symlink_to(tmp_path / "logits.npy")
    data = _case()
    data["inputs"]["manifest"]["task"] = "image_classification"
    for artifact in ("../logits.npy", str(tmp_path / "logits.npy"), "link.npy"):
        data["reference"] = {"logits": {"shape": [2], "artifact": artifact}}
        visible = _visible(render_case(data, root))
        assert "Class index (from saved logits)" not in visible


@pytest.mark.parametrize("task", ["text_generation", "text_continuation", "text_translation"])
def test_text_generation_tokens_without_decoded_text_are_explicit(tmp_path: Path, task: str) -> None:
    data = _case()
    data["inputs"]["manifest"]["task"] = task
    data["native"] = {"token_ids": [10, 11]}
    data["reference"].pop("actual_decoded")
    visible = _visible(render_case(data, tmp_path))
    assert "2 recorded tokens; decoded text unavailable" in visible
    assert "token IDs" not in visible and "Token IDs are in Details" in visible


def test_recorded_class_color_key_is_visible_by_default(tmp_path: Path) -> None:
    data = _case()
    path = "legend.gif"
    (tmp_path / path).write_bytes(
        base64.b64decode("R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==")
    )
    data["views"] = [
        {
            "title": "Class color key",
            "caption": "Recorded class IDs use the displayed colors.",
            "image": {"artifact": path},
        }
    ]
    data["artifacts"] = [{"path": path, "role": "views"}]
    report = render_case(data, tmp_path)
    visible = _visible(report)
    assert (
        "Class color key" in visible and "Recorded class IDs use the displayed colors." in visible
    )
    assert report.count("data:image/gif;base64,") == 1
    assert "More recorded media (1)" not in report


def _npy(header: str, payload: bytes = b"", version: int = 1) -> bytes:
    encoded = (header + "\n").encode("ascii")
    return (
        b"\x93NUMPY"
        + bytes([version, 0])
        + len(encoded).to_bytes(2 if version == 1 else 4, "little")
        + encoded
        + payload
    )


@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize(
    "dtype,format_code", [("<f4", "<3f"), (">f8", ">3d"), ("|i1", "3b"), ("<u2", "<3H")]
)
def test_complete_simple_numeric_npy(version: int, dtype: str, format_code: str) -> None:
    data = _npy(
        f"{{'descr': '{dtype}', 'fortran_order': False, 'shape': (1, 3)}}",
        struct.pack(format_code, 1, 2, 3),
        version,
    )
    assert decode_npy(data) == {"shape": [1, 3], "dtype": dtype, "values": [1, 2, 3]}


@pytest.mark.parametrize(
    "header",
    [
        "{'descr': '|O8', 'fortran_order': False, 'shape': (1,)}",
        "{'descr': [('x', '<f4')], 'fortran_order': False, 'shape': (1,)}",
        "{'descr': '<f4', 'fortran_order': True, 'shape': (1,)}",
        "{'descr': '<f4', 'fortran_order': 0, 'shape': (1,)}",
        "{'descr': '<f4', 'fortran_order': False, 'shape': (4097,)}",
        "{'descr': '<f4', 'fortran_order': False, 'shape': (-1,)}",
        "{'descr': '<f4', 'fortran_order': False, 'shape': (True,)}",
        "{'descr': '<f4', 'fortran_order': False, 'shape': [1]}",
        "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), 'extra': 1}",
        "{'descr': '|f4', 'fortran_order': False, 'shape': (1,)}",
        "__import__('os').system('false')",
        "[0] * 4096",
        "{" * 1000,
    ],
)
def test_npy_rejects_unsupported_or_executable_headers(header: str) -> None:
    assert decode_npy(_npy(header, b"\x00" * 4)) is None


def test_npy_rejects_wrong_payload_nonfinite_and_oversize() -> None:
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1,)}"
    valid = _npy(header, struct.pack("<f", 1.0))
    for value in (
        valid[:-1],
        valid + b"extra",
        b"not an npy",
        valid.replace(b"\x01\x00", b"\x04\x00", 1),
        valid + b"0" * NPY_MAX_BYTES,
    ):
        assert decode_npy(value) is None
    assert decode_npy(_npy(header, struct.pack("<f", float("nan")))) is None
    assert decode_npy(_npy(header, struct.pack("<f", float("inf")))) is None
    assert decode_npy(_npy(" " * 4096 + header, struct.pack("<f", 1.0))) is None


def test_class_index_requires_complete_single_batch_logits() -> None:
    assert classification_index({"shape": [1, 3], "values": [-1, 8, 2]}) == (1, 8)
    assert classification_index([[2, 2, 1]]) == (0, 2)
    assert classification_index({"shape": [1000], "preview": [0, 10, 2]}) is None
    assert classification_index({"shape": [1000], "values": [0, 10, 2]}) is None
    assert classification_index({"shape": [2, 3], "values": [1, 2, 3, 4, 5, 6]}) is None
    assert classification_index([[1, 2], [3, 4]]) is None
    assert classification_index([float("nan"), 2]) is None
    assert classification_index([False, True]) is None
    assert classification_index({"shape": [True, 2], "values": [1, 2]}) is None
    assert classification_index({"shape": [1.0, 2], "values": [1, 2]}) is None


def test_nonfinite_preview_warning_is_scoped_and_preserves_result(tmp_path: Path) -> None:
    import copy

    data = _case()
    data["inputs"]["manifest"]["task"] = "monocular_geometry"
    data["native"] = {"depth": {"shape": [382, 640], "preview": ["inf"] * 64}}
    data["reference"] = {"depth": {"shape": [382, 640], "preview": [0.5, "inf"]}}
    original = copy.deepcopy(data)
    report = render_case(data, tmp_path)
    visible = _visible(report)
    assert "Depth (64/64): non-finite saved preview values" in visible
    assert "Depth (1/2): non-finite saved preview values" in _expanded(report)
    assert "preview only, not the full tensor" in visible
    assert 'data-execution-status="passed"' in report and data == original


def test_nonfinite_notice_keeps_other_finite_output_samples(tmp_path: Path) -> None:
    data = _case()
    data["inputs"]["manifest"]["task"] = "monocular_geometry"
    data["native"] = {
        "depth": {"shape": [382, 640], "preview": ["inf"] * 64},
        "actions": {"shape": [100, 14], "preview": [0.125, 0.25, 0.375, 0.5] * 16},
    }
    report = render_case(data, tmp_path)
    visible = _visible(report)
    assert "Depth (64/64): non-finite saved preview values" in visible
    assert "Actions" in visible and "First 64 of 1400 values" in visible
    assert "<svg" in report and "0.125" in report


def test_output_samples_take_priority_over_summary_regardless_of_key_order() -> None:
    from tools.e2e_report import _output_summary

    actions = {"shape": [100, 14], "preview": [0.125, 0.25, 0.375, 0.5] * 16}
    summary = {f"detail_{index}": index for index in range(12)}
    first = _output_summary(
        {"summary": summary, "actions": actions}, role="native", task="robot_control"
    )
    second = _output_summary(
        {"actions": actions, "summary": dict(reversed(list(summary.items())))},
        role="native",
        task="robot_control",
    )
    assert first == second
    assert "Actions shape" in first and "100 × 14" in first
    assert "Actions sample (first values)" in first and "0.125, 0.25, 0.375, 0.5" in first


def test_demo_has_one_outer_details_and_no_execution_bookkeeping(tmp_path: Path) -> None:
    data = _case()
    data["duration_seconds"] = 123.4
    data["stages"] = [{"name": "native", "duration_seconds": 123.4}]
    report = render_case(data, tmp_path)
    card = report.split('<section class="case"', 1)[1]
    default = card.split('<details class="case-details">', 1)[0]
    assert default.count('class="io-panel"') == 3
    assert "<details" not in default
    assert card.count('<details class="case-details">') == 1
    assert "123.4" not in _visible(report) and "123.4" in report
    assert "Recorded tokens" not in _visible(report)
    assert "Recorded assertions:" not in _visible(report)
    assert "Reference output" in default
    assert "<h3>Reference output</h3>" not in card.split('<details class="case-details">', 1)[1]


def test_demo_collapses_same_role_duplicates_without_losing_other_media(tmp_path: Path) -> None:
    data = _case()
    gif = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==")
    paths = ["native.gif", "reference.gif", "extra.gif", "duplicate.gif"]
    blobs = [gif, gif + b"reference", gif + b"extra", gif]
    for path, blob in zip(paths, blobs):
        (tmp_path / path).write_bytes(blob)
    data["native"] = {"artifact": paths[0]}
    data["reference"] = {"artifact": paths[1]}
    data["artifacts"] = [
        {"path": path, "label": path, **({"role": "native"} if path == "duplicate.gif" else {})}
        for path in paths
    ]
    report = render_case(data, tmp_path)
    assert report.count("data:image/gif;base64,") == 3
    default = report.split('<details class="case-details">', 1)[0]
    assert default.count("data:image/gif;base64,") == 1
    assert "More recorded media (1)" in report
    assert all((tmp_path / path).read_bytes() == blob for path, blob in zip(paths, blobs))


def test_demo_snapshot_links_retain_full_unique_values_without_mutation() -> None:
    import copy

    from tools.e2e_report import _demo_raw_data

    data = _case()
    data["observations"] = [
        {"name": "native", "value": {"text": "earlier unique result"}},
        {"name": "native", "value": data["native"]},
        {"name": "reference", "value": data["reference"]},
    ]
    original = copy.deepcopy(data)
    rendered = _demo_raw_data(data)
    assert rendered["native"] == data["native"]
    assert rendered["observations"][0]["value"] == {"text": "earlier unique result"}
    assert rendered["observations"][1]["value"] == {"same_recorded_value_as": "native"}
    assert rendered["observations"][2]["value"] == {"same_recorded_value_as": "reference"}
    assert data == original


def test_demo_audio_output_keeps_spoken_text_without_tensor_bookkeeping() -> None:
    from tools.e2e_report import _demo_output

    output = _demo_output(
        {"text": "Hello there.", "num_samples": 24000, "sample_rate": 24000},
        role="native",
        task="speech_generation",
        media='<audio controls src="recorded.wav"></audio>',
    )
    assert "Hello there." in output and "<audio controls" in output
    assert "24000" not in output


def test_demo_class_index_accepts_complete_direct_tensor_without_probability() -> None:
    from tools.e2e_report import _demo_output

    output = _demo_output(
        {"shape": [1, 3], "values": [0.2, 0.8, 0.5]},
        role="native",
        task="image_classification",
    )
    assert "Class ID" in output and ">1</strong>" in output
    assert "From complete saved logits" in output
    assert "probability" not in output and "0.8" not in output


def test_demo_shared_chart_keeps_scope_and_rejects_invalid_values() -> None:
    from tools.e2e_report import _demo_numeric_comparison

    native = {"shape": [576], "preview": list(range(64))}
    reference = {"shape": [576], "preview": list(range(0, 128, 2))}
    output = _demo_numeric_comparison(native, reference, "encoding")
    assert output.count("<svg") == 1 and output.count("<polyline") == 2
    assert output.count("First 64 of 576 values") == 2  # Caption and accessible chart label.
    assert "shared scale" in output and ">126</text>" in output
    assert _demo_numeric_comparison([float("inf")] * 20) == ""
    assert _demo_numeric_comparison(["<script>bad</script>"] * 20) == ""
    huge = _demo_numeric_comparison([1e308] * 20, [0.0] * 20)
    assert "nan" not in huge.lower() and "inf" not in huge.lower()


@pytest.mark.parametrize(
    "task",
    [
        "text_generation",
        "translation",
        "transcription",
        "transcription_streaming",
        "automatic_speech_recognition",
        "vision_language_generation",
        "ocr",
    ],
)
def test_text_task_references_are_visible_without_opening_details(
    tmp_path: Path, task: str
) -> None:
    data = _case()
    data["inputs"]["manifest"]["task"] = task
    data["native"] = {"generated_text": "Native answer."}
    data["reference"] = {"transcription": "Reference answer."}
    original = copy.deepcopy(data)
    report = render_case(data, tmp_path)
    default, details = report.split('<details class="case-details">', 1)
    assert "Native answer." in _visible(default) and "Reference answer." in _visible(default)
    assert "<h3>Native output</h3>" in default and "<h3>Reference output</h3>" in default
    assert "<h3>Reference output</h3>" not in details
    assert ".pair,.text-comparison{grid-template-columns:1fr}" in report
    assert data == original


def test_long_output_text_stays_in_top_level_comparison(tmp_path: Path) -> None:
    data = _case()
    text = "a" * 450 + "MIDDLE_SENTINEL <script>alert(1)</script>" + "z" * 450
    data["native"] = {"text": text}
    data["reference"] = {"text": text}
    report = render_case(data, tmp_path)
    default = report.split('<details class="case-details">', 1)[0]
    assert _visible(default).count(text) == 2
    assert "middle omitted" not in default
    assert "<script>alert(1)</script>" not in default and "&lt;script&gt;" in default


def test_text_lookup_uses_only_output_containers_and_role_specific_diagnostics() -> None:
    from tools.e2e_report import _demo_text_value

    value = {
        "inputs": {"text": "Wrong input"},
        "stderr": {"text": "Wrong log"},
        "actual_decoded": "Wrong native side",
        "diagnostics": {"text": "Wrong diagnostic", "reference_text": "Expected output"},
    }
    assert _demo_text_value(value, "reference") == "Expected output"
    assert _demo_text_value({"final": {"output": {"text": "Stream result"}}}) == "Stream result"
    assert _demo_text_value({"extras": {"actual_decoded": "Native decoded"}}) == "Native decoded"
    assert _demo_text_value({"reference_text": "Wrong reference side"}) == ""
    assert _demo_text_value({"output": "result.wav", "logs": {"text": "Wrong log"}}) == ""
    assert _demo_text_value({"diagnostics": {"actual_decoded": "Wrong native"}}, "reference") == ""


def test_explicit_decoded_native_diagnostic_stays_on_native_side_without_mutation(
    tmp_path: Path,
) -> None:
    from tools.e2e_report import _demo_native_output

    data = _case()
    data["native"] = {"shape": [1, 3], "preview": [0.1, 0.2, 0.3]}
    data["reference"] = {
        "actual_decoded": "Native token decode",
        "reference_text": "Reference token decode",
        "reference_ids": [1, 2],
    }
    original = copy.deepcopy(data)
    report = render_case(data, tmp_path)
    default = report.split('<details class="case-details">', 1)[0]
    native_panel, reference_panel = default.split("<h3>Reference output</h3>", 1)
    assert "Native token decode" in native_panel and "Reference token decode" not in native_panel
    assert (
        "Reference token decode" in reference_panel and "Native token decode" not in reference_panel
    )
    assert data == original
    assert _demo_native_output(None, data["reference"], "text_generation") is None
    assert (
        _demo_native_output({"text": "Primary native text"}, data["reference"], "text_generation")[
            "text"
        ]
        == "Primary native text"
    )
    assert (
        _demo_native_output(
            data["native"],
            {"reference_text": "Do not substitute", "reference_ids": [1]},
            "text_generation",
        )
        == data["native"]
    )
    assert (
        _demo_native_output(
            data["native"], {"actual_decoded": "Unanchored diagnostic"}, "text_generation"
        )
        == data["native"]
    )

    assert _demo_native_output(data["native"], data["reference"], "encoding") == data["native"]


def test_missing_reference_text_is_explicit_without_empty_nontext_panels(tmp_path: Path) -> None:
    from tools.e2e_report import _demo_text_comparison

    data = _case()
    data["reference"] = {"reference_ids": [1, 2]}
    visible = _visible(render_case(data, tmp_path))
    assert "Reference output" in visible and "decoded text is unavailable" in visible
    data["reference"] = None
    assert "No reference text was recorded" in _visible(render_case(data, tmp_path))
    assert not _demo_text_comparison({"probe_returncode": 0}, None, "text_generation")
    assert not _demo_text_comparison({"values": [1, 2]}, None, "encoding")
    assert not _demo_text_comparison(None, None, "text_generation")


def test_text_reference_media_remains_in_details_without_repeated_text(tmp_path: Path) -> None:
    data = _case()
    gif = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==")
    (tmp_path / "reference.gif").write_bytes(gif)
    data["reference"] = {"text": "Recorded caption", "artifact": "reference.gif"}
    data["artifacts"] = [{"path": "reference.gif", "role": "reference"}]
    report = render_case(data, tmp_path)
    default, details = report.split('<details class="case-details">', 1)
    assert "Recorded caption" in _visible(default) and "data:image/gif" not in default
    assert "<h3>Reference media</h3>" in details and "data:image/gif" in details
    before_raw = details.split("Raw recorded fields, full text and logs", 1)[0]
    assert "Recorded caption" not in before_raw


def test_identical_chart_samples_keep_both_line_styles_and_explicit_scope() -> None:
    from tools.e2e_report import _demo_numeric_comparison

    values = {"shape": [576], "preview": list(range(64))}
    output = _demo_numeric_comparison(values, values, "encoding")
    assert 'aria-label="Native"' in output and 'aria-label="Reference"' in output
    assert 'stroke-width="5"' in output and 'stroke-width="2"' in output
    assert 'stroke-dasharray="6 4"' in output
    assert "Native · solid" in output and "Reference · dashed" in output
    assert "Both curves overlap: 64 displayed paired values are identical." in output
    assert "First 64 of 576 values" in output
    assert "passed" not in output and "full tensor" not in output


def test_chart_distinguishes_visual_overlap_from_identical_values() -> None:
    from tools.e2e_report import _demo_numeric_comparison

    native = list(range(20))
    near = _demo_numeric_comparison(native, [number + 0.00001 for number in native])
    assert 'data-overlap="visual"' in near
    assert "displayed paired values differ" in near and "identical" not in near
    distinct = _demo_numeric_comparison(native, [number + 100 for number in native])
    assert "overlap" not in distinct and distinct.count("<polyline") == 2
    single = _demo_numeric_comparison(native)
    assert single.count("<polyline") == 1 and "Reference" not in single and "overlap" not in single


def test_overlap_note_counts_only_available_paired_samples() -> None:
    from tools.e2e_report import _demo_numeric_comparison

    output = _demo_numeric_comparison(
        {"shape": [128], "preview": list(range(12))},
        {"shape": [128], "preview": list(range(8))},
    )
    assert "8 displayed paired values are identical" in output
    assert "First 12 of 128 values" in output and "reference: first 8 of 128 values" in output
    assert "128 displayed paired values" not in output


def _assessment_fixture(expression=None, explanation="True", native=None, reference=None, **fields):
    data = {
        "status": "passed",
        "native": {"values": [1.0, 2.0]} if native is None else native,
        "reference": {"values": [1.0, 2.0]} if reference is None else reference,
    }
    data["checks"] = (
        []
        if expression is None
        else [{"status": "passed", "expression": expression, "explanation": explanation}]
    )
    data.update(fields)
    return data


class AssessmentTests(unittest.TestCase):
    def test_reference_presence_is_not_parity(self):
        self.assertEqual(assessment(_assessment_fixture())["kind"], "unverified")

    def test_shape_presence_metadata_checks_are_not_parity(self):
        for expression in [
            "native.shape == reference.shape",
            "actual.shape == expected.shape",
            "actual.size == expected.size",
            "actual_num_masks == expected_num_masks",
            "expected_images",
            "reference is not None",
        ]:
            with self.subTest(expression=expression):
                self.assertNotEqual(
                    assessment(_assessment_fixture(expression))["kind"], "reference"
                )

    def test_self_comparisons_are_not_parity(self):
        for expression in [
            "_cosine(native,native) >= 0.99",
            "_cosine(actual, actual) >= 0.99",
            "_relative_l2(actual, actual) <= 0.01",
            "_edit_distance(actual_text, actual_text) <= 0.1",
        ]:
            with self.subTest(expression=expression):
                self.assertNotEqual(
                    assessment(_assessment_fixture(expression, "1.0 >= 0.99"))["kind"], "reference"
                )

    def test_pixel_statistics_are_not_parity(self):
        data = _assessment_fixture(
            'float(pixels.std()) >= float(thresholds["min_pixel_std"])',
            "0.2 >= 0.1",
            thresholds={"min_pixel_std": 0.1},
        )
        self.assertEqual(assessment(data)["kind"], "limited")

    def test_native_reference_cosine_is_parity(self):
        result = assessment(
            _assessment_fixture(
                '_cosine(actual["values"], expected["values"]) >= 0.99', "0.999 >= 0.99"
            )
        )
        self.assertEqual(result["kind"], "reference")
        self.assertIn("0.999 >= 0.99", result["summary"])

    def test_case_expected_ids_are_not_upstream_reference(self):
        result = assessment(
            _assessment_fixture('actual_ids == case["expected_token_ids"]', "[1] == [1]")
        )
        self.assertNotEqual(result["kind"], "reference")

    def test_actual_reference_tokens_are_parity(self):
        self.assertEqual(
            assessment(_assessment_fixture("actual_ids == reference_ids", "[1] == [1]"))["kind"],
            "reference",
        )

    def test_or_answer_bypass_does_not_claim_reference_pass(self):
        data = _assessment_fixture(
            "ned <= threshold or expected_answer_matches",
            "(0.8 <= 0.1 or True)",
            native={"text": "answer one"},
            reference={"reference_text": "answer two"},
        )
        self.assertEqual(assessment(data)["kind"], "limited")

    def test_or_with_observed_reference_branch_pass(self):
        data = _assessment_fixture(
            "ned <= threshold or expected_answer_matches",
            "(0.0 <= 0.1)",
            native={"text": "answer"},
            reference={"reference_text": "answer"},
        )
        self.assertEqual(assessment(data)["kind"], "reference")

    def test_or_unknown_branch_is_not_reference_proof(self):
        data = _assessment_fixture(
            "ned <= threshold or expected_answer_matches",
            "True",
            native={"text": "answer one"},
            reference={"reference_text": "answer two"},
        )
        self.assertEqual(assessment(data)["kind"], "limited")

    def test_prompt_asr_is_limited(self):
        data = _assessment_fixture(
            "_normalized_edit_distance(transcript, prompt) <= limit",
            "0.0 <= 0.15",
            native={"audio": "native.wav"},
            reference={"audio": "reference.wav"},
        )
        self.assertEqual(assessment(data)["kind"], "limited")

    def test_response_fixture_is_limited(self):
        data = _assessment_fixture(
            '_edit_distance(actual_text, expected["text"]) <= limit',
            "0.0 <= 0.1",
            native={"text": "hello"},
            reference={"text": "hello"},
            inputs={"case": {"expected_response_text": "hello"}},
        )
        self.assertEqual(assessment(data)["kind"], "limited")

    def test_contract_declaration_needs_native_and_checks(self):
        data = _assessment_fixture(reference={"mode": "contract_only"})
        self.assertEqual(assessment(data)["kind"], "unverified")
        data["checks"] = [
            {"status": "passed", "expression": "probe.is_file()", "explanation": "True"}
        ]
        self.assertEqual(assessment(data)["kind"], "limited")

    def test_reference_failure_takes_precedence_over_old_checks(self):
        data = _assessment_fixture(
            "_cosine(actual, expected) >= .99",
            "1.0 >= 0.99",
            status="failed",
            failure_stage="reference",
        )
        result = assessment(data)
        self.assertEqual(result["kind"], "failed")
        self.assertEqual(result["label"], "Reference run failed")
        self.assertEqual(assessment(data, "passed")["kind"], "failed")

    def test_failed_assertion_overrides_passed_status(self):
        data = _assessment_fixture("_cosine(actual, expected) >= .99", "0.5 >= 0.99")
        data["checks"][0]["status"] = "failed"
        self.assertEqual(assessment(data)["kind"], "failed")

    def test_missing_execution_is_unverified(self):
        self.assertEqual(assessment({}, "not-run")["kind"], "unverified")
        self.assertEqual(assessment(None)["kind"], "unverified")

    def test_class_margin_exception_is_visible(self):
        data = _assessment_fixture(
            'int(actual["top_class"]) == int(expected["second_class"])',
            "2 == 2",
            native={"top_class": 2},
            reference={"top_class": 1, "second_class": 2, "top1_margin": 0.03},
        )
        data["checks"].append(
            {
                "status": "passed",
                "expression": 'float(expected["top1_margin"]) <= float(margin)',
                "explanation": "0.03 <= 0.1",
            }
        )
        result = assessment(data)
        self.assertEqual(result["kind"], "reference")
        self.assertIn("Top classes differ (2 vs 1)", result["summary"])
        self.assertIn("allowed", result["summary"])

    def test_generic_metric_loop_needs_named_limits_and_pairs(self):
        data = _assessment_fixture(
            "value <= threshold",
            "0.001 <= 0.01",
            native={"depth": [1], "points": [1]},
            reference={"depth": [1], "points": [1]},
            thresholds={"depth_rel_l2": 0.01, "points_rel_l2": 0.02},
        )
        data["checks"].append(
            {"status": "passed", "expression": "value <= threshold", "explanation": "0.001 <= 0.02"}
        )
        self.assertEqual(assessment(data)["kind"], "reference")
        data["thresholds"] = {"min_pixel_mean": 0.01, "min_pixel_std": 0.02}
        self.assertNotEqual(assessment(data)["kind"], "reference")

    def test_does_not_mutate_input(self):
        data = _assessment_fixture("_cosine(actual, expected) >= .99", "1.0 >= 0.99")
        before = copy.deepcopy(data)
        assessment(data)
        self.assertEqual(data, before)

    def test_names_are_irrelevant(self):
        data = _assessment_fixture("_cosine(actual, expected) >= .99", "1.0 >= 0.99")
        result = assessment(data)
        data.update(family="arbitrary-new-family", case="arbitrary-new-case", nodeid="whatever")
        self.assertEqual(assessment(data), result)


def test_text_generation_never_substitutes_logits_for_a_readable_response(tmp_path: Path) -> None:
    import copy

    data = _case()
    logits = {"shape": [1000000], "preview": [0.25] * 64, "dtype": "float16"}
    data["native"] = logits
    data["reference"] = {"logits": logits}
    original = copy.deepcopy(data)
    report = render_case(data, tmp_path)
    visible = _visible(report)
    assert "Decoded text unavailable" in visible and "recorded logits are in Details" in visible
    assert "1000000" not in visible and "float16" not in visible
    assert "<svg" not in report and "1000000" in report
    assert data == original
    data["native"] = {"text": "A real response.", "logits": logits}
    data["reference"] = {"reference_text": "A real response.", "logits": logits}
    report = render_case(data, tmp_path)
    assert "A real response." in _visible(report) and "<svg" not in report


def test_nonfinite_notice_is_one_sentence_without_output_metadata(tmp_path: Path) -> None:
    data = _case()
    data["inputs"]["manifest"]["task"] = "monocular_geometry"
    descriptor = {"shape": [100, 20], "preview": ["inf"] * 64, "dtype": "float32"}
    data["native"] = {"depth": descriptor, "points": descriptor, "dtype": "float32"}
    report = render_case(data, tmp_path)
    visible = _visible(report)
    assert "Depth (64/64), Points (64/64): non-finite saved preview values" in visible
    assert visible.count("preview only, not the full tensor") == 1
    assert "float32" not in visible and "float32" in report
    assert '<dl class="facts">' not in report.split('<details class="case-details">')[0]


def test_numeric_demo_deduplicates_equal_notices_and_keeps_finite_prefix(tmp_path: Path) -> None:
    from tools.e2e_report import _demo_output

    data = _case()
    data["inputs"]["manifest"]["task"] = "monocular_geometry"
    descriptor = {"shape": [100, 20], "preview": ["inf"] * 64}
    data["native"] = {"depth": descriptor, "actions": {"shape": [128], "preview": [0.1] * 64}}
    data["reference"] = {"depth": descriptor, "actions": {"shape": [128], "preview": [0.2] * 64}}
    report = render_case(data, tmp_path)
    visible = _visible(report)
    assert visible.count("preview only, not the full tensor") == 1
    assert "First 64 of 128 values" in visible and report.count("<svg") == 1
    mixed = {"depth": {"shape": [100, 20], "preview": [0.5, "inf", 0.7]}}
    output = _demo_output(mixed, role="native", task="monocular_geometry")
    assert "First 1 of 2000 values" in output and "<circle" in output
    assert "Depth (1/3): non-finite saved preview values" in output
    assert mixed["depth"]["preview"] == [0.5, "inf", 0.7]


def test_long_recorded_prompt_keeps_final_question_visible(tmp_path: Path) -> None:
    data = _case()
    data["inputs"]["prompt"] = (
        "Context begins. " + "Context filler. " * 300 + "What color is the final marker?"
    )
    report = render_case(data, tmp_path)
    visible = _visible(report)
    assert "Context begins." in visible and "What color is the final marker?" in visible
    assert "middle omitted; full text in Details" in visible
    assert 'class="readable text-excerpt"' in report
    assert data["inputs"]["prompt"] in _expanded(report)


def _stress_case():
    return {
        "family": "example",
        "case": "runtime-example",
        "status": "passed",
        "checks": [{"status": "passed", "expression": "count > 0", "explanation": "2 > 0"}],
        "inputs": {
            "manifest": {"task": "text_generation"},
            "prompt": " ".join(["z"] * 12) + "\n",
            "case": {
                "expected_prompt_tokens": 13,
                "expected_prefill_chunks": 4,
                "expected_prefill_chunk_limit": 4,
                "max_new_tokens": 3,
                "prompt_repeat": {"text": "z", "count": 12, "separator": " ", "suffix": "\n"},
            },
        },
        "native": {"text": "Example continuation", "token_ids": [11, 12]},
        "reference": {"mode": "contract_only"},
    }


def test_runtime_stress_demo_does_not_claim_answer_quality(tmp_path):
    data = _stress_case()
    before = copy.deepcopy(data)
    document = render_case(data, tmp_path)
    text = _visible(document)
    assert "Runtime stress test passed" in text
    assert "generated text quality was not evaluated" in text
    assert "'z' × 12" in text and "Expected input: 13 tokens" in text
    assert "Example continuation" in text and "2 generated tokens · configured limit 3" in text
    assert "Not run for this runtime test" in text
    assert assessment(data)["kind"] == "limited"
    assert data == before
    data["status"] = "failed"
    assert "Runtime stress test passed" not in _visible(render_case(data, tmp_path))


def test_repeat_summary_never_overrides_different_actual_input():
    from tools.e2e_report import _demo_input

    data = _stress_case()
    data["inputs"]["prompt"] = data["inputs"]["prompt"].replace("z", "y")
    text = _visible(_demo_input(data))
    assert "y y y" in text and "×" not in text
    assert "differs from the repeat configuration" in text
    data["inputs"]["case"]["prompt_repeat"]["count"] = 10**50
    assert "differs from the repeat configuration" in _demo_input(data)
    data["reference"] = {"text": "Saved reference"}
    assert assessment(data)["label"] != "Runtime stress test passed"


def test_recording_error_keeps_outcome_when_terminal_sink_fails(tmp_path):
    from types import SimpleNamespace
    from tools.e2e_evidence import _report_evidence_error

    def unavailable(*args, **kwargs):
        raise OSError("terminal unavailable")

    terminal = SimpleNamespace(write_line=unavailable)
    item = SimpleNamespace(
        nodeid="sample",
        config=SimpleNamespace(pluginmanager=SimpleNamespace(get_plugin=lambda name: terminal)),
    )
    report = SimpleNamespace(outcome="passed", sections=[], user_properties=[])
    recorder = _recorder(tmp_path)
    _report_evidence_error(item, report, recorder, "write", OSError("record unavailable"))
    assert report.outcome == "passed"
    assert report.user_properties == [
        ("trtmc_evidence_error", "Could not write evidence: OSError: record unavailable")
    ]
    assert recorder.data["evidence_status"] == "partial"


@pytest.mark.parametrize(
    ("suffix", "payload"),
    [
        (".cif", b"data_example\n_atom_site.Cartn_x 1.0\n"),
        (".yaml", b"sequences:\n  - protein:\n      sequence: ACD\n"),
        (".a3m", b">example\nACD\n"),
        (".b2rq", b"\x00B2RQ\x01"),
    ],
)
def test_structure_file_is_copied_as_bounded_inert_evidence(tmp_path, suffix, payload):
    source = tmp_path / f"structure{suffix}"
    source.write_bytes(payload)
    recorder = _recorder(tmp_path)
    recorder.record("native", {"structure": str(source)})
    saved = recorder.data["native"]["structure"]
    assert saved["artifact"].endswith(suffix)
    assert (recorder.directory / saved["artifact"]).read_bytes() == payload
    assert source.read_bytes() == payload


def test_family_observations_cannot_replace_recorder_metadata(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.data.update(
        environment={"python": "recorded-python"},
        repro={"command": "recorded-command"},
        workflow_run_attempt=2,
    )
    trusted = dict(recorder.data)
    names = [*trusted, "failure", "duration_seconds", "evidence_status"]
    for name in names:
        recorder.record(name, {"value": "family observation"})
        if name != "observations":
            assert recorder.data.get(name) == trusted.get(name)
    assert recorder.data["observations"] == [
        {"name": name, "value": {"value": "family observation"}} for name in names
    ]
    recorder.stage = "reference"
    recorder.finish("failed", failure="actual reference failure")
    saved = json.loads((recorder.directory / "evidence.json").read_text())
    assert saved["status"] == "failed"
    assert saved["failure"] == {"message": "actual reference failure"}
    assert saved["failure_stage"] == "reference"
    assert saved["evidence_status"] == "recorded"
    assert isinstance(saved["duration_seconds"], float)
    for name in ("environment", "repro", "workflow_run_attempt", "nodeid"):
        assert saved[name] == trusted[name]


def _library_comparison(label="original"):
    return {
        "label": label,
        "scope": "independent_reference",
        "enforced": True,
        "native": {"artifact": f"artifacts/{label}.cif", "size_bytes": 100},
        "reference": {"artifact": f"artifacts/{label}.npz", "size_bytes": 200},
        "checks": [
            {
                "name": "Finite outputs",
                "scope": "contract",
                "actual": True,
                "operator": "==",
                "expected": True,
                "passed": True,
            },
            {
                "name": "Count",
                "scope": "contract",
                "actual": 8,
                "operator": "==",
                "expected": 8,
                "passed": True,
            },
            {
                "name": "Similarity",
                "scope": "independent_reference",
                "actual": 0.95,
                "operator": ">=",
                "expected": 0.9,
                "passed": True,
            },
            {
                "name": "Aligned error",
                "scope": "independent_reference",
                "actual": 0.02,
                "operator": "<=",
                "expected": 0.1,
                "passed": True,
            },
        ],
    }


def _library_case():
    first, second = _library_comparison(), _library_comparison("variant")
    return {
        "family": "example",
        "case": "example-case",
        "status": "passed",
        "checks": [],
        "inputs": {
            "manifest": {
                "name": "example-recipe",
                "hf_id": "example/checkpoint",
                "task": "structure_prediction",
                "precision": "fp16",
            },
            "text": "Protein A\nACDEFGHI",
        },
        "native": {"confidence_score": 0.9},
        "reference_comparison": second,
        "observations": [
            {"name": "reference_comparison", "value": first},
            {"name": "reference_comparison", "value": second},
        ],
    }


def test_enforced_library_comparisons_need_no_duplicate_pytest_assertions(tmp_path):
    from tools.e2e_report import _assess_reference_comparisons

    data = _library_case()
    original = copy.deepcopy(data)
    result = assessment(data)
    assert result["kind"] == "reference"
    assert (
        "2 recorded comparisons" in result["summary"] and "4 reference checks" in result["summary"]
    )
    assert [record["label"] for record in _assess_reference_comparisons(data)] == [
        "original",
        "variant",
    ]
    document = render_case(data, tmp_path)
    assert "Reference checks passed" in _visible(document)
    assert "Recorded reference comparisons" not in _visible(document)
    expanded = _expanded(document)
    for value in (
        "original",
        "variant",
        "Actual",
        "Limit",
        "Recorded result",
        "0.95",
        "0.9",
        "0.02",
        "0.1",
    ):
        assert value in expanded
    assert document.count("<h4>variant</h4>") == 1
    assert document.index("<h4>original</h4>") < document.index("<h4>variant</h4>")
    assert data == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("scope", "contract_only"),
        ("enforced", False),
        ("enforced", 1),
        ("label", ""),
        ("native", None),
        ("reference", {"path": "missing.npz", "available": False}),
        ("reference", {"artifact": "artifacts/original.cif", "size_bytes": 100}),
        ("reference", {"artifact": "../outside.npz", "size_bytes": 100}),
        (
            "reference",
            {"artifact": "artifacts/reference.npz", "size_bytes": 100, "omitted": "size limit"},
        ),
        ("checks", []),
    ],
)
def test_library_comparison_requires_complete_enforced_distinct_evidence(field, value):
    record = _library_comparison()
    record[field] = value
    data = {"status": "passed", "reference_comparison": record}
    assert assessment(data)["kind"] == "unverified"


@pytest.mark.parametrize(
    "field,value",
    [
        ("actual", 0.5),
        ("actual", float("nan")),
        ("actual", float("inf")),
        ("actual", 10**1000),
        ("actual", "0.95"),
        ("expected", None),
        ("passed", False),
        ("passed", "true"),
        ("operator", ">"),
        ("operator", {}),
        ("scope", {}),
        ("name", ""),
    ],
)
def test_library_comparison_rejects_inconsistent_or_nonfinite_check_rows(field, value):
    record = _library_comparison()
    record["checks"][2][field] = value
    assert assessment({"status": "passed", "reference_comparison": record})["kind"] == "unverified"


def test_library_comparisons_do_not_promote_contracts_or_bare_pass_flags():
    record = _library_comparison()
    record["checks"] = record["checks"][:2]
    assert assessment({"status": "passed", "reference_comparison": record})["kind"] == "unverified"
    record["checks"][0]["scope"] = "independent_reference"
    assert assessment({"status": "passed", "reference_comparison": record})["kind"] == "unverified"
    data = _library_case()
    data["reference"] = {"mode": "contract_only"}
    assert assessment(data)["kind"] != "reference"
    data = _library_case()
    del data["reference_comparison"]
    data["observations"] = []
    data["metrics"] = {"qualification": {"passed": True}}
    data["reference"] = {"artifact": "reference.npz"}
    assert assessment(data)["kind"] == "unverified"


def test_library_comparison_cannot_hide_an_incomplete_or_failed_record():
    data = _library_case()
    del data["observations"][0]["value"]["checks"][2]["expected"]
    assert assessment(data)["kind"] == "unverified"
    data = _library_case()
    data["reference_comparison"]["checks"].append(
        copy.deepcopy(data["reference_comparison"]["checks"][0])
    )
    assert assessment(data)["kind"] == "unverified"
    data = _library_case()
    data.update(status="failed", failure_stage="compare")
    assert assessment(data)["kind"] == "failed"
    assert assessment(data, "passed")["kind"] == "failed"
    data["status"] = "skipped"
    assert assessment(data, "passed")["kind"] == "unverified"


def test_library_comparison_details_keep_failed_verdicts_and_escape_text(tmp_path):
    data = _library_case()
    record = data["observations"][0]["value"]
    record["label"] = "<script>untrusted()</script>"
    record["checks"][2].update(actual=0.5, passed=False)
    data.update(status="failed", failure_stage="compare")
    document = render_case(data, tmp_path)
    assert "Check failed" in _visible(document)
    assert "Failed" in _expanded(document) and "0.5" in _expanded(document)
    assert "<script>untrusted()</script>" not in document
    assert "&lt;script&gt;untrusted()&lt;/script&gt;" in document


def test_library_comparison_uses_human_labels_without_changing_check_identity():
    from tools.e2e_report import _assess_reference_details

    data = _library_case()
    check = data["observations"][0]["value"]["checks"][2]
    check.update(name="raw_similarity_key", label="Readable similarity")
    display = _assess_reference_details(data)
    assert "Readable similarity" in display and "raw_similarity_key" not in display
    del check["label"]
    assert "raw similarity key" in _assess_reference_details(data)
    data["observations"][0]["value"]["checks"].append({**check, "label": "Another label"})
    assert assessment(data)["kind"] == "unverified"
