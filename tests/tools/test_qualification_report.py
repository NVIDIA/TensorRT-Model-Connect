# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from tools import qualification_report


def test_outcome_accounting_separates_progress_coverage_and_results() -> None:
    accounting = qualification_report.outcome_accounting(
        [
            {"state": "pending", "result": None},
            {"state": "running", "result": None},
            {"state": "terminal", "result": "green"},
            {"state": "terminal", "result": "yellow"},
            {"state": "terminal", "result": "red"},
            {"state": "terminal", "result": "white"},
        ]
    )

    assert accounting["selected"] == 6
    assert accounting["comparable"] == 3
    assert accounting["operational_coverage_percent"] == 50.0
    assert accounting["progress"] == {"pending": 1, "running": 1, "terminal": 4}
    assert accounting["outcomes"] == {
        "green": 1,
        "yellow": 1,
        "red": 1,
        "white": 1,
    }


def test_outcome_accounting_does_not_allow_a_light_for_running_case() -> None:
    with pytest.raises(
        qualification_report.QualificationReportError,
        match="running case cannot have",
    ):
        qualification_report.outcome_accounting(
            [{"state": "running", "result": "white"}]
        )


@pytest.mark.parametrize(
    ("green", "yellow", "red", "white", "selected", "comparable"),
    [
        (87, 0, 8, 0, 95, 95),
        (73, 0, 8, 16, 97, 81),
        (71, 7, 14, 3, 95, 92),
        (60, 6, 16, 15, 97, 82),
    ],
)
def test_audited_campaign_accounting_after_exclusions_are_removed(
    green: int,
    yellow: int,
    red: int,
    white: int,
    selected: int,
    comparable: int,
) -> None:
    rows = [
        {"state": "terminal", "result": result}
        for result, count in (
            ("green", green),
            ("yellow", yellow),
            ("red", red),
            ("white", white),
        )
        for _ in range(count)
    ]

    accounting = qualification_report.outcome_accounting(rows)

    assert accounting["selected"] == selected
    assert accounting["comparable"] == comparable
    assert accounting["outcomes"] == {
        "green": green,
        "yellow": yellow,
        "red": red,
        "white": white,
    }


def test_materialized_html_is_a_renderer_and_not_a_result_source(tmp_path) -> None:
    json_path, html_path, report = qualification_report.materialize_report(
        tmp_path,
        report_kind="accuracy",
        title="Accuracy qualification",
        identity={"run_id": "run-1"},
        run={"source_revision": "abc123"},
        results=[
            {
                "id": "model-a::task-a",
                "model": "model-a",
                "state": "terminal",
                "result": "green",
            }
        ],
    )

    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    assert report["$schema"] == "assets/qualification-report.schema.json"
    document = html_path.read_text(encoding="utf-8")
    assert 'data-report="report.json"' in document
    assert "model-a" not in document
    assert "Comparable results" not in document
    assert (tmp_path / "assets/qualification-report.css").is_file()
    assert (tmp_path / "assets/qualification-report.js").is_file()
    assert (tmp_path / "assets/qualification-report.schema.json").is_file()
    assert report["run"]["environment_href"] == "artifacts/run/environment.json"
    assert (tmp_path / report["run"]["environment_href"]).is_file()
    assert not list(tmp_path.glob(".report.json.*.tmp"))

    static_paths = [
        html_path,
        tmp_path / "assets/qualification-report.css",
        tmp_path / "assets/qualification-report.js",
        tmp_path / "assets/qualification-report.schema.json",
    ]
    mtimes = {path: path.stat().st_mtime_ns for path in static_paths}
    qualification_report.materialize_report(
        tmp_path,
        report_kind="accuracy",
        title="Accuracy qualification",
        identity={"run_id": "run-1"},
        run={"source_revision": "abc123"},
        results=[
            {
                "id": "model-a::task-a",
                "model": "model-a",
                "state": "terminal",
                "result": "green",
            }
        ],
    )
    assert {path: path.stat().st_mtime_ns for path in static_paths} == mtimes


def test_white_case_without_process_output_gets_a_clickable_diagnostic_log(
    tmp_path,
) -> None:
    _, _, report = qualification_report.materialize_report(
        tmp_path,
        report_kind="performance",
        title="Performance qualification",
        identity={"run_id": "run-1"},
        run={},
        results=[
            {
                "id": "model-a.generate",
                "state": "terminal",
                "result": "white",
                "issue": {
                    "priority": "P1",
                    "stage": "reference-preflight",
                    "domain": "harness/unknown",
                    "code": "execution_failure",
                    "message": "profile unavailable",
                },
                "commands": {"resolve": {"rendered": "trtmc-bench run --dry-run"}},
                "debug": {"logs": []},
            }
        ],
    )

    log = report["results"][0]["debug"]["logs"][0]
    assert log["label"] == "Reference Preflight diagnostic"
    assert not log["href"].startswith("/")
    assert "profile unavailable" in (tmp_path / log["href"]).read_text(encoding="utf-8")


def test_materializer_rejects_machine_absolute_artifact_links(tmp_path) -> None:
    absolute_log = tmp_path / "worker.log"
    absolute_log.write_text("failure\n", encoding="utf-8")

    with pytest.raises(
        qualification_report.QualificationReportError,
        match="must be relative",
    ):
        qualification_report.materialize_report(
            tmp_path,
            report_kind="accuracy",
            title="Accuracy qualification",
            identity={"run_id": "run-1"},
            run={},
            results=[
                {
                    "id": "model-a::task-a",
                    "state": "terminal",
                    "result": "white",
                    "issue": {
                        "priority": "P1",
                        "stage": "candidate",
                        "domain": "harness/unknown",
                        "code": "execution_error",
                        "message": "failed",
                    },
                    "debug": {"logs": [{"label": "worker", "href": str(absolute_log)}]},
                }
            ],
        )
