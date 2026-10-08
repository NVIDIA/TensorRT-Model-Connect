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
            {
                "sample_id": "first",
                "passed": False,
                "candidate_text": "wrong <text>",
                "reference_text": "right",
            },
            {"sample_id": "second", "passed": True},
        ],
    }
    passed = {
        "case": "good/performance/run",
        "model": "good",
        "kind": "performance",
        "status": "passed",
    }
    case_dir = tmp_path / "bad/accuracy/parity"
    case_dir.mkdir(parents=True)
    (case_dir / "candidate.stderr.log").write_text(
        "failure at /raid/private/work/model.py\n", encoding="utf-8"
    )
    (case_dir / "candidate-inputs.json").write_text(
        json.dumps(
            [
                {
                    "sample_id": "first",
                    "request": {"prompt": "hello", "image_path": "/mnt/private/image.jpg"},
                }
            ]
        ),
        encoding="utf-8",
    )
    write_case_report(case_dir, failed)
    write_local_summary(
        tmp_path,
        {
            "schema_version": "trtmc.qualification-summary/v1",
            "status": "failed",
            "cases": [passed, failed],
        },
    )

    report = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert report.index('class="failed"') < report.index('class="passed"')
    assert 'href="bad/accuracy/parity/candidate.stderr.log"' in report
    assert 'href="bad/accuracy/parity/candidate-inputs.json"' in report
    assert "first" in report and "wrong &lt;text&gt;" in report
    assert "hello" in report and "&lt;staged asset: image.jpg&gt;" in report
    assert "--dataset manual-data=/path/to/manual-data" in report
    assert "/raid/private/" not in report
    assert "Checkpoint</summary>" not in report


def test_error_is_not_reported_as_model_gate_failure_and_unsafe_links_are_rejected() -> None:
    cases = [
        {
            "case": "one/accuracy/case",
            "kind": "accuracy",
            "model": "one",
            "status": "error",
            "error": "reference failed; see /runs/private",
        },
        {
            "case": "two/performance/case",
            "kind": "performance",
            "model": "two",
            "status": "failed",
            "comparison_status": "contract-mismatch",
            "comparison_reason": "tokens differ",
            "observed_metrics": {"candidate_p50_ms": 4.0},
        },
    ]
    rows = [
        {
            "model": "two",
            "status": "failed",
            "performance": {
                "case": "two/performance/case",
                "status": "failed",
                "report_html": "two/performance/case/report.html",
            },
        },
        {
            "model": "one",
            "status": "failed",
            "accuracy": {
                "case": "one/accuracy/case",
                "status": "error",
                "report_html": "../private/report.html",
            },
        },
    ]
    report = render_combined_report(rows, cases, available={"../private/result.json"})

    assert "15 models" not in report
    assert "1 failed · 1 error" in report
    assert report.index('class="error"') < report.index('class="failed"')
    assert "reference failed" in report and "/runs/private" not in report
    assert "tokens differ" in report and "observed; not comparable" in report
    assert 'href="../private' not in report


def test_combined_report_separates_qualification_from_performance_lights() -> None:
    cases = [
        {
            "case": f"{model}/performance/run",
            "kind": "performance",
            "model": model,
            "status": "passed",
            "comparison_status": comparison,
            "metrics": {
                "candidate_p50_ms": candidate,
                "reference_p50_ms": reference,
                "reference_over_candidate_p50": reference / candidate,
            },
        }
        for model, comparison, candidate, reference in (
            ("fast", "green", 6.877179, 29.354052850976586),
            ("similar", "yellow", 10.0, 10.1),
            ("slow", "red", 20.0, 10.0),
        )
    ]
    rows = [
        {
            "model": case["model"],
            "checkpoint": "org/model" if case["model"] == "fast" else "",
            "performance": {
                "case": case["case"],
                "status": "passed",
                "report_html": f"{case['case']}/report.html",
            },
        }
        for case in cases
    ]

    report = render_combined_report(rows, cases, available=set())

    assert "3 models · 3 passed · 0 failed · 0 error" in report
    assert 'class="signal signal-green"' in report
    assert 'class="signal signal-yellow"' in report
    assert 'class="signal signal-red"' in report
    assert "red (slower) is not a qualification failure" in report
    assert "TRTMC p50</span><strong>6.877 ms" in report
    assert "Reference p50</span><strong>29.354 ms" in report
    assert "TRTMC 4.27× faster" not in report
    assert "TRTMC 2× slower" not in report
    assert "p50 TRTMC/HF:" not in report
    assert 'id="model-search"' in report and 'id="status-filter"' in report
    assert "Reasons come from automated checks or execution logs" in report
    assert 'class="checkpoint"><summary>Checkpoint</summary>' in report


def test_single_kind_report_leaves_unrun_side_empty_without_failing_it() -> None:
    accuracy = {
        "case": "accuracy-only/accuracy/run",
        "kind": "accuracy",
        "model": "accuracy-only",
        "status": "passed",
        "metrics": {"pass_rate": 1.0},
    }
    performance = {
        "case": "performance-only/performance/run",
        "kind": "performance",
        "model": "performance-only",
        "status": "passed",
        "comparison_status": "green",
        "metrics": {"candidate_p50_ms": 6.0, "reference_p50_ms": 12.0},
    }
    for kind, case in (("accuracy", accuracy), ("performance", performance)):
        rows = [
            {
                "model": case["model"],
                kind: {
                    "case": case["case"],
                    "status": "passed",
                    "report_html": f"{case['case']}/report.html",
                },
            }
        ]
        report = render_combined_report(rows, [case], available=set())
        assert 'class="not-run">Not run</span>' in report
        assert "1 model · 1 passed · 0 failed · 0 error" in report
        assert "0/1 recorded samples did not pass" not in report
        assert f'href="{case["case"]}/report.html">details</a>' in report
        if kind == "accuracy":
            assert "Accuracy agreement against the model reference." in report
            assert "Performance vs reference ·" not in report
        else:
            assert "Performance against the model reference." in report
            assert "Performance vs reference · 1 case" in report
            assert "TRTMC p50</span><strong>6 ms" in report


def test_reference_error_shows_attempted_input_without_inventing_an_output(tmp_path: Path) -> None:
    case = {
        "case": "speech/accuracy/wer",
        "kind": "accuracy",
        "model": "speech",
        "status": "error",
        "error": "reference command failed",
    }
    output = tmp_path / "speech/accuracy/wer"
    output.mkdir(parents=True)
    (output / "reference-request.json").write_text(
        json.dumps(
            {"samples": [{"sample_id": "audio-001", "audio_path": "/mnt/data/audio-001.flac"}]}
        ),
        encoding="utf-8",
    )
    write_case_report(output, case)
    write_local_summary(tmp_path, {"status": "failed", "cases": [case]})

    report = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "First attempted input; no output was recorded" in report
    assert "audio-001" in report and "&lt;staged asset: audio-001.flac&gt;" in report
    assert 'href="speech/accuracy/wer/reference-request.json"' in report


def test_archive_uses_verified_links_and_matrix_reason_without_recalculating_gate(
    tmp_path: Path,
) -> None:
    prefix = "report/run"
    report_html = "artifacts/models/model/attempt-1/model/performance/run/report.html"
    matrix_name = report_html.removesuffix("report.html") + "matrix/results.json"
    result_name = report_html.removesuffix("report.html") + "result.json"
    summary = {
        "schema_version": "trtmc.accperf_nas_summary/v1",
        "run_id": "run",
        "rows": [
            {
                "model": "model",
                "status": "failed",
                "performance": {
                    "case": "model/performance/run",
                    "benchmark": "example",
                    "status": "failed",
                    "report_html": report_html,
                },
            }
        ],
    }
    results = {
        "status": "failed",
        "cases": [
            {
                "case": "model/performance/run",
                "model": "model",
                "kind": "performance",
                "status": "failed",
                "comparison_status": "contract-mismatch",
                "metrics": {"candidate_p50_ms": None},
            }
        ],
    }
    inventory = {
        "data": {
            "truncated": False,
            "matches": [
                {"path": f"{prefix}/{matrix_name}", "size": 100},
                {"path": f"{prefix}/{result_name}", "size": 100},
            ],
        }
    }
    cache = tmp_path / "cache"
    matrix = cache / matrix_name
    matrix.parent.mkdir(parents=True)
    matrix.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "comparison": {"reason": "outputs differ"},
                        "candidate": {
                            "metrics": {"latency_ms": {"p50": 2.0}},
                            "output_summary": {"text": "candidate"},
                        },
                        "reference": {
                            "metrics": {"latency_ms": {"p50": 3.0}},
                            "output_summary": {"text": "reference"},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    for name, data in (
        ("summary.json", summary),
        ("results.json", results),
        ("inventory.json", inventory),
    ):
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")

    report = render_archive(
        tmp_path / "summary.json",
        tmp_path / "results.json",
        tmp_path / "out",
        repository=Path(__file__).resolve().parents[3],
        inventory_paths=(tmp_path / "inventory.json",),
        inventory_prefix=prefix,
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


def test_archive_rejects_inventory_without_prefix(tmp_path: Path) -> None:
    import pytest

    summary = tmp_path / "summary.json"
    results = tmp_path / "results.json"
    inventory = tmp_path / "inventory.json"
    summary.write_text(
        json.dumps({"schema_version": "trtmc.accperf_nas_summary/v1", "run_id": "run", "rows": []}),
        encoding="utf-8",
    )
    results.write_text(json.dumps({"cases": []}), encoding="utf-8")
    inventory.write_text(
        json.dumps({"data": {"truncated": False, "matches": []}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="--inventory-prefix"):
        render_archive(
            summary,
            results,
            tmp_path / "out",
            repository=Path(__file__).resolve().parents[3],
            inventory_paths=(inventory,),
        )
