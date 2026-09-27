# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from qualification_tests.benchmark_qualification.render_published import render_archive
from qualification_tests.benchmark_qualification.reporting import (
    render_combined_report,
    write_case_report,
    write_local_summary,
)


def test_local_report_shows_failure_first_with_logs_samples_and_repro(tmp_path: Path) -> None:
    failed = {
        "case": "bad/accuracy/parity",
        "model": "bad",
        "kind": "accuracy",
        "benchmark": "example",
        "status": "failed",
        "metrics": {"samples": 2, "pass_rate": 0.5},
        "gate": {"pass_rate": 1.0},
        "dataset": {"id": "manual-data", "source_mode": "staged", "sha256": "abc"},
        "samples": [
            {"sample_id": "first", "passed": False, "candidate_text": "wrong <text>", "reference_text": "right"},
            {"sample_id": "second", "passed": True},
        ],
    }
    passed = {"case": "good/performance/run", "model": "good", "kind": "performance", "status": "passed"}
    case_dir = tmp_path / "bad/accuracy/parity"
    case_dir.mkdir(parents=True)
    (case_dir / "candidate.stderr.log").write_text("failure at /raid/private/work/model.py\n", encoding="utf-8")
    (case_dir / "candidate-inputs.json").write_text(
        json.dumps([{"sample_id": "first", "request": {"prompt": "hello", "image_path": "/mnt/private/image.jpg"}}]),
        encoding="utf-8",
    )
    write_case_report(case_dir, failed)
    write_local_summary(tmp_path, {"schema_version": "trtmc.qualification-summary/v1", "status": "failed", "cases": [passed, failed]})

    report = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert report.index('class="failed"') < report.index('class="passed"')
    assert 'href="bad/accuracy/parity/candidate.stderr.log"' in report
    assert 'href="bad/accuracy/parity/candidate-inputs.json"' in report
    assert "first" in report and "wrong &lt;text&gt;" in report
    assert "hello" in report and "&lt;staged asset: image.jpg&gt;" in report
    assert "--dataset manual-data=/path/to/manual-data" in report
    assert "/raid/private/" not in report
    assert "Checkpoint</summary>" in report


def test_error_is_not_reported_as_model_gate_failure_and_unsafe_links_are_rejected() -> None:
    cases = [
        {"case": "one/accuracy/case", "kind": "accuracy", "model": "one", "status": "error", "error": "reference failed; see /runs/private"},
        {"case": "two/performance/case", "kind": "performance", "model": "two", "status": "failed", "comparison_status": "contract-mismatch", "comparison_reason": "tokens differ", "observed_metrics": {"candidate_p50_ms": 4.0}},
    ]
    rows = [
        {"model": "two", "status": "failed", "performance": {"case": "two/performance/case", "status": "failed", "report_html": "two/performance/case/report.html"}},
        {"model": "one", "status": "failed", "accuracy": {"case": "one/accuracy/case", "status": "error", "report_html": "../private/report.html"}},
    ]
    report = render_combined_report(rows, cases, available={"../private/result.json"})

    assert '15 models' not in report
    assert '1 failed · 1 error' in report
    assert report.index('class="error"') < report.index('class="failed"')
    assert "reference failed" in report and "/runs/private" not in report
    assert "tokens differ" in report and "observed; not comparable" in report
    assert 'href="../private' not in report


def test_reference_error_shows_attempted_input_without_inventing_an_output(tmp_path: Path) -> None:
    case = {
        "case": "speech/accuracy/wer", "kind": "accuracy", "model": "speech",
        "status": "error", "error": "reference command failed",
    }
    output = tmp_path / "speech/accuracy/wer"
    output.mkdir(parents=True)
    (output / "reference-request.json").write_text(json.dumps({"samples": [
        {"sample_id": "audio-001", "audio_path": "/mnt/data/audio-001.flac"}
    ]}), encoding="utf-8")
    write_case_report(output, case)
    write_local_summary(tmp_path, {"status": "failed", "cases": [case]})

    report = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "First attempted input; no output was recorded" in report
    assert "audio-001" in report and "&lt;staged asset: audio-001.flac&gt;" in report
    assert 'href="speech/accuracy/wer/reference-request.json"' in report


def test_archive_uses_verified_links_and_matrix_reason_without_recalculating_gate(tmp_path: Path) -> None:
    prefix = "report/run"
    report_html = "artifacts/models/model/attempt-1/model/performance/run/report.html"
    matrix_name = report_html.removesuffix("report.html") + "matrix/results.json"
    result_name = report_html.removesuffix("report.html") + "result.json"
    summary = {
        "schema_version": "trtmc.accperf_nas_summary/v1",
        "run_id": "run",
        "rows": [{"model": "model", "status": "failed", "performance": {"case": "model/performance/run", "benchmark": "example", "status": "failed", "report_html": report_html}}],
    }
    results = {"status": "failed", "cases": [{"case": "model/performance/run", "model": "model", "kind": "performance", "status": "failed", "comparison_status": "contract-mismatch", "metrics": {"candidate_p50_ms": None}}]}
    inventory = {"data": {"truncated": False, "matches": [{"path": f"{prefix}/{matrix_name}", "size": 100}, {"path": f"{prefix}/{result_name}", "size": 100}]}}
    cache = tmp_path / "cache"
    matrix = cache / matrix_name
    matrix.parent.mkdir(parents=True)
    matrix.write_text(json.dumps({"rows": [{"comparison": {"reason": "outputs differ"}, "candidate": {"metrics": {"latency_ms": {"p50": 2.0}}, "output_summary": {"text": "candidate"}}, "reference": {"metrics": {"latency_ms": {"p50": 3.0}}, "output_summary": {"text": "reference"}}}]}), encoding="utf-8")
    for name, data in (("summary.json", summary), ("results.json", results), ("inventory.json", inventory)):
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")

    report = render_archive(
        tmp_path / "summary.json", tmp_path / "results.json", tmp_path / "out",
        repository=Path(__file__).resolve().parents[3],
        inventory_paths=(tmp_path / "inventory.json",), inventory_prefix=prefix,
        artifact_cache=cache,
    )

    case = report["cases"][0]
    assert case["status"] == "failed"
    assert case["comparison_reason"] == "outputs differ"
    assert case["observed_metrics"] == {"candidate_p50_ms": 2.0, "reference_p50_ms": 3.0}
    document = (tmp_path / "out/report.html").read_text(encoding="utf-8")
    assert "candidate" in document and "reference" in document
    assert f'href="{matrix_name}"' in document
    assert "not comparable" in document


def test_archive_rejects_truncated_inventory(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"data": {"truncated": True, "matches": []}}), encoding="utf-8")
    from qualification_tests.benchmark_qualification.render_published import _inventory
    import pytest

    with pytest.raises(ValueError, match="truncated"):
        _inventory((inventory,), "report/run")
