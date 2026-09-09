# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import json
import struct
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest

from tools import e2e_evidence
from tools.e2e_evidence import Evidence, evidence_stage, record_evidence
from tools.e2e_report import (
    NPY_MAX_BYTES,
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
    assert "native diagnostic" not in visible
    assert "example-recipe" in visible and "example-greedy-short" in visible
    assert "example/model" in visible and "FP16 · TP2" in visible
    assert "families/example-model" in report
    assert 'class="io-grid"' in report
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
    visible = _visible(render_case(data, tmp_path))
    assert "Expected answer" in visible
    assert "secondary text" not in visible and "wrong answer" not in visible


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


def test_classification_preserves_ids_and_raw_score_meaning(tmp_path: Path) -> None:
    data = _case()
    data["inputs"]["manifest"]["task"] = "image_classification"
    data["native"] = {"top_class": 7, "top_score": 12.5, "logits": [0.1] * 1000}
    data["reference"] = 7
    visible = _visible(render_case(data, tmp_path))
    assert "Class ID" in visible and "Raw score" in visible and "12.5" in visible
    assert "1000" in visible
    assert "confidence" not in visible.lower() and "probability" not in visible.lower()


def test_contract_reference_and_failure_are_never_a_paired_pass(tmp_path: Path) -> None:
    data = _case()
    data["status"] = "failed"
    data["failure_stage"] = "reference"
    data["failure"] = {"traceback": "TRACEBACK_SENTINEL"}
    data["reference"] = {"mode": "contract_only", "oracle": "public core invariants"}
    data["evidence_status"] = "partial"
    report = render_case(data, tmp_path)
    visible = _visible(report)
    assert "Contract checks only" in visible and "No reference output was generated" in visible
    assert "Stopped during reference" in visible and "Partial evidence" in visible
    assert 'data-status="failed"' in report
    assert "TRACEBACK_SENTINEL" not in visible and "TRACEBACK_SENTINEL" in report


def test_missing_settings_and_outputs_remain_explicit(tmp_path: Path) -> None:
    report = render_case({"family": "example", "case": "not-run", "status": "skipped"}, tmp_path)
    visible = _visible(report)
    assert visible.count("No output recorded") == 2
    assert "Recipe not recorded" in visible and "Build configuration not recorded" in visible
    assert "TP1" not in visible and "FP16" not in visible
    assert "No assertion measurements were recorded" in visible


def test_run_settings_use_human_labels_and_original_keys(tmp_path: Path) -> None:
    report = render_case(_case(), tmp_path)
    assert "Maximum sequence length" in report and "max_sequence_length" in report
    assert "Maximum new tokens" in report and "max_new_tokens" in report
    assert "Recipe and build" in report and "Request and reference" in report
    assert "1024" not in _visible(report)
    assert "Missing fields have no assumed defaults" in report


def test_media_uses_recorded_references_and_embeds_each_file_once(tmp_path: Path) -> None:
    data = _case()
    gif = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==")
    paths = ["input.gif", "native.gif", "reference.gif", "earlier.gif"]
    for path in paths:
        (tmp_path / path).write_bytes(gif)
    data["inputs"]["asset"] = {"artifact": paths[0]}
    data["native"] = {"artifact": paths[1]}
    data["reference"] = {"artifact": paths[2]}
    data["artifacts"] = [{"path": path, "role": "observations", "label": path} for path in paths]
    report = render_case(data, tmp_path)
    assert report.count("data:image/gif;base64,") == 4
    before_more = report.split("<summary>More recorded media")[0]
    assert before_more.count("data:image/gif;base64,") == 3
    assert "More recorded media (1)" in report
    assert (
        report.index("Input</h3>") < report.index("input.gif") < report.index("Native output</h3>")
    )
    assert (
        report.index("Native output</h3>")
        < report.index("native.gif")
        < report.index("Reference output</h3>")
    )


def test_long_text_stays_readable_and_full_evidence_is_expandable(tmp_path: Path) -> None:
    data = _case()
    data["native"]["text"] = "hello " * 200 + "FULL_TEXT_END"
    report = render_case(data, tmp_path)
    assert "hello" in _visible(report) and "FULL_TEXT_END" not in _visible(report)
    assert "Full text" in report and "FULL_TEXT_END" in report
    assert "no-results" in report and "aria-live" in report


def test_forecast_is_identified_as_last_window(tmp_path: Path) -> None:
    data = _case()
    data["inputs"]["manifest"]["task"] = "forecasting"
    data["native"] = {"shape": [1, 24], "values": [0.1] * 24}
    report = render_case(data, tmp_path)
    assert "Preview of the last recorded window" in _visible(report)
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
    assert "Original structured prompt" in report and "&quot;description&quot;" in report


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
    data["inputs"]["case"]["inputs"] = {
        "documents": ["Candidate A says yes.", "Candidate B says no."]
    }
    data["native"] = {"scores": [-8.96, -10.27]}
    data["reference"] = {"scores": [-8.95, -10.28]}
    visible = _visible(render_case(data, tmp_path))
    assert "Candidate A says yes." in visible and "Candidate B says no." in visible
    assert "Scores shape" in visible and "All 2 values" in visible
    assert "-8.96" in visible and "-10.28" in visible


def test_masks_and_points_distinguish_counts_shapes_and_values(tmp_path: Path) -> None:
    data = _case()
    data["inputs"]["case"].update({"point_x": 0.5, "point_y": 0.25})
    data["native"] = {
        "num_masks": 3,
        "masks": {"shape": [733440], "preview": [0.2] * 64},
        "iou_scores": [0.821, 0.876, 0.986],
        "bbox_xyxy": [10, 20, 30, 40],
        "mask_foreground_pixels": [50, 51, 52, 53, 54],
    }
    visible = _visible(render_case(data, tmp_path))
    assert "Point X (recorded)" in visible and "0.25" in visible
    assert "Generated masks" in visible and "Masks shape" in visible
    assert "All 3 values" in visible and "0.821" in visible
    assert "All 4 values" in visible and "40" in visible
    assert "All 5 values" in visible and "54" in visible
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
    assert "Recorded window index" in visible and "999" not in visible
    assert report.count("<svg") == 3


def test_nested_recorded_summary_is_readable(tmp_path: Path) -> None:
    data = _case()
    data["reference"] = {"output": {"summary": {"frame_count": 12, "mean": 0.25}}}
    visible = _visible(render_case(data, tmp_path))
    assert "Output · Summary · Frame count" in visible
    assert "12" in visible and "0.25" in visible


def test_nested_request_options_are_settings_not_just_raw_json(tmp_path: Path) -> None:
    data = _case()
    data["inputs"]["case"]["inputs"] = {
        "cfg_scale": 4.5,
        "num_sampling_steps": 16,
        "generation_mode": "conditional_sample",
        "documents": ["LONG_DOCUMENT_SENTINEL"],
    }
    report = render_case(data, tmp_path)
    settings = report.split("<h3>Request input options</h3>")[1].split(
        "<details><summary>All recorded input fields"
    )[0]
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
    visible = _visible(render_case(data, tmp_path))
    assert "Class index (from saved logits)" in visible and "777" in visible
    assert "Raw score" in visible and "9.0" in visible
    assert "Class ID" in visible and data == original
    assert "values" not in descriptor


def test_partial_logits_never_claim_a_derived_class_index(tmp_path: Path) -> None:
    data = _case()
    data["inputs"]["manifest"]["task"] = "image_classification"
    data["reference"] = {
        "logits": {"shape": [1, 1000], "preview": [100.0] * 64, "artifact": "absent.npy"}
    }
    visible = _visible(render_case(data, tmp_path))
    assert "Class index (from saved logits)" not in visible
    assert "complete saved logits are unavailable" in visible


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


def test_text_generation_tokens_without_decoded_text_are_explicit(tmp_path: Path) -> None:
    data = _case()
    data["native"] = {"token_ids": [10, 11]}
    assert "Decoded text not recorded" in _visible(render_case(data, tmp_path))


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
    assert "Depth shape" in visible and "382 × 640" in visible
    assert "all 64 values in this recorded preview are non-finite" in visible
    assert "this recorded preview contains 1 non-finite value out of 2" in visible
    assert "preview only, not the full tensor" in visible
    assert 'data-status="passed"' in report and data == original


def test_nonfinite_notice_keeps_other_finite_output_samples(tmp_path: Path) -> None:
    data = _case()
    data["inputs"]["manifest"]["task"] = "monocular_geometry"
    data["native"] = {
        "depth": {"shape": [382, 640], "preview": ["inf"] * 64},
        "actions": {"shape": [100, 14], "preview": [0.125, 0.25, 0.375, 0.5] * 16},
    }
    visible = _visible(render_case(data, tmp_path))
    assert "all 64 values in this recorded preview are non-finite" in visible
    assert "Actions sample (first values)" in visible
    assert "0.125, 0.25, 0.375, 0.5" in visible
