# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from trtmc_benchmark.qualification import (
    CONFIG_SCHEMA,
    ENVIRONMENT_SCHEMA,
    RUN_CONFIGURATION_SCHEMA,
    QualificationCatalog,
    QualificationError,
    QualificationRunner,
    generate_report,
    load_plan,
)
from trtmc_benchmark.qualification_cli import main


def _family(root: Path, family: str, model: str, *, with_config: bool = True) -> Path:
    tests = root / family / "tests"
    manifests = tests / "manifests"
    qualification = tests / "qualification"
    manifests.mkdir(parents=True)
    (manifests / f"{model}.json").write_text(
        json.dumps(
            {
                "name": model,
                "family": family,
                "bundle": f"{model}.bundle",
                "task": "text_generation",
                "precision": "fp32",
                "testcases": [{"name": model}],
            }
        ),
        encoding="utf-8",
    )
    if not with_config:
        return qualification
    qualification.mkdir()
    (qualification / "executor.py").write_text(_FAKE_EXECUTOR, encoding="utf-8")
    (qualification / f"{model}.accuracy.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": CONFIG_SCHEMA,
                "model": model,
                "kind": "accuracy",
                "suites": [
                    {
                        "id": "file-suite",
                        "gate_policy": "blocking",
                        "definition": {"implementation": "fake-suite"},
                        "cases": [{"id": "small"}, {"id": "full"}],
                    },
                    {
                        "id": "inline-suite",
                        "gate_policy": "observation_only",
                        "definition": {"implementation": "inline"},
                        "cases": [{"id": "inspect"}],
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return qualification


_FAKE_EXECUTOR = """\
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--request", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
request = json.loads(args.request.read_text())
item = request["plan_item"]
counter = args.output.parent / "count.txt"
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count))
evidence = args.output.parent / "evidence.json"
evidence.write_text(json.dumps({"count": count}))
result = {
    "schema_version": "trtmc.qualification-result/v1",
    "plan_item_id": item["id"],
    "family": item["family"],
    "model": item["model"],
    "kind": item["kind"],
    "suite_id": item["suite_id"],
    "case_id": item["case_id"],
    "execution": "completed",
    "verdict": "pass" if item["gate_policy"] == "blocking" else None,
    "details": {
        "count": count,
        "metrics": {"count": count},
        "gate_evaluations": [],
    },
    "artifacts": [{"label": "evidence", "path": "evidence.json"}],
}
args.output.write_text(json.dumps(result))
"""


def _environment() -> dict:
    return {
        "schema_version": ENVIRONMENT_SCHEMA,
        "name": "test",
        "execution": {"timeout_seconds": 10},
    }


def test_discovery_reads_only_explicit_family_configs_and_expands_cases(tmp_path: Path) -> None:
    families = tmp_path / "families"
    _family(families, "alpha", "model-a")
    _family(families, "beta", "model-b-l0", with_config=False)

    plan = QualificationCatalog(families).plan("accuracy")

    assert [(item.model, item.suite_id, item.case_id) for item in plan.items] == [
        ("model-a", "file-suite", "small"),
        ("model-a", "file-suite", "full"),
        ("model-a", "inline-suite", "inspect"),
    ]
    assert {item.model for item in plan.items} == {"model-a"}
    assert plan.items[0].definition == {"implementation": "fake-suite"}


def test_an_assessment_with_no_opted_in_models_is_a_valid_empty_run(tmp_path: Path) -> None:
    families = tmp_path / "families"
    _family(families, "alpha", "model-a")
    plan = QualificationCatalog(families).plan("performance")

    report = QualificationRunner(Path(sys.executable)).run(plan, tmp_path / "run", _environment())

    assert plan.items == ()
    assert report["status"] == "empty"
    assert report["summary"]["planned"] == 0
    with pytest.raises(QualificationError, match="unknown models: model-a"):
        QualificationCatalog(families).plan("performance", models=["model-a"])


def test_discovery_filters_exact_names_without_a_second_registry(tmp_path: Path) -> None:
    families = tmp_path / "families"
    _family(families, "alpha", "model-a")
    _family(families, "beta", "model-b")

    plan = QualificationCatalog(families).plan(
        "accuracy", models=["model-b"], suites=["file-suite"], cases=["full"]
    )

    assert len(plan.items) == 1
    assert plan.items[0].model == "model-b"
    assert plan.items[0].case_id == "full"
    with pytest.raises(QualificationError, match="unknown suites: missing"):
        QualificationCatalog(families).plan("accuracy", suites=["missing"])


def test_unselected_family_does_not_break_an_explicit_model_run(tmp_path: Path) -> None:
    families = tmp_path / "families"
    _family(families, "alpha", "model-a")
    broken = _family(families, "beta", "model-b")
    (broken / "model-b.accuracy.yaml").write_text("not: [valid", encoding="utf-8")

    plan = QualificationCatalog(families).plan("accuracy", models=["model-a"])

    assert {item.model for item in plan.items} == {"model-a"}


def test_benchmark_must_be_a_plain_name(tmp_path: Path) -> None:
    families = tmp_path / "families"
    qualification = _family(families, "alpha", "model-a")
    suites = tmp_path / "suites"
    suites.mkdir()
    config = yaml.safe_load((qualification / "model-a.accuracy.yaml").read_text())
    config["suites"][0].pop("id")
    config["suites"][0].pop("definition")
    config["suites"][0]["benchmark"] = "../outside"
    (qualification / "model-a.accuracy.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(QualificationError, match="must be a plain name"):
        QualificationCatalog(families, suites).plan("accuracy")


def test_model_config_cannot_bind_a_resource_class(tmp_path: Path) -> None:
    families = tmp_path / "families"
    qualification = _family(families, "alpha", "model-a")
    config = yaml.safe_load((qualification / "model-a.accuracy.yaml").read_text())
    config["suites"][0]["resource_class"] = "gb300-exclusive"
    (qualification / "model-a.accuracy.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(QualificationError, match="device policy belongs to the campaign"):
        QualificationCatalog(families).plan("accuracy")


def test_suite_source_change_after_planning_fails_closed(tmp_path: Path) -> None:
    families = tmp_path / "families"
    qualification = _family(families, "alpha", "model-a")
    suites = tmp_path / "suites"
    suites.mkdir()
    definition = suites / "file-suite.yaml"
    definition.write_text(yaml.safe_dump({"implementation": "fake-suite"}), encoding="utf-8")
    config_path = qualification / "model-a.accuracy.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["suites"][0].pop("id")
    config["suites"][0].pop("definition")
    config["suites"][0]["benchmark"] = "file-suite"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    plan = QualificationCatalog(families, suites).plan(
        "accuracy", suites=["file-suite"], cases=["small"]
    )
    definition.write_text(yaml.safe_dump({"implementation": "changed"}), encoding="utf-8")

    report = QualificationRunner(Path(sys.executable)).run(plan, tmp_path / "run", _environment())

    assert report["status"] == "error"
    assert report["summary"]["blocking"]["error"] == 1
    assert "source changed after the plan" in report["items"][0]["details"]["error"]


def test_runner_accounts_for_blocking_and_observation_only_results(tmp_path: Path) -> None:
    families = tmp_path / "families"
    _family(families, "alpha", "model-a")
    plan = QualificationCatalog(families).plan("accuracy")
    output = tmp_path / "run"

    report = QualificationRunner(Path(sys.executable)).run(plan, output, _environment())

    assert report["status"] == "pass"
    assert report["summary"]["planned"] == 3
    assert report["summary"]["blocking"] == {
        "pass": 2,
        "fail": 0,
        "error": 0,
        "missing": 0,
    }
    assert report["summary"]["observation_only"] == {
        "completed": 1,
        "error": 0,
        "missing": 0,
    }
    assert (output / "plan.json").is_file()
    assert (output / "report.json").is_file()
    assert (output / "report.html").is_file()
    assert "evidence.json" in (output / "report.html").read_text(encoding="utf-8")


def test_resume_reexecutes_only_a_malformed_result(tmp_path: Path) -> None:
    families = tmp_path / "families"
    _family(families, "alpha", "model-a")
    plan = QualificationCatalog(families).plan("accuracy")
    output = tmp_path / "run"
    runner = QualificationRunner(Path(sys.executable))
    runner.run(plan, output, _environment())
    item_directories = sorted((output / "items").iterdir())
    (item_directories[1] / "result.json").write_text("{broken", encoding="utf-8")

    resumed = runner.run(plan, output, _environment(), resume=True)

    assert resumed["status"] == "pass"
    assert [int((path / "count.txt").read_text()) for path in item_directories] == [1, 2, 1]


def test_resume_archives_an_error_attempt_and_reexecutes_the_case(tmp_path: Path) -> None:
    families = tmp_path / "families"
    _family(families, "alpha", "model-a")
    plan = QualificationCatalog(families).plan("accuracy", suites=["file-suite"], cases=["small"])
    output = tmp_path / "run"
    runner = QualificationRunner(Path(sys.executable))
    runner.run(plan, output, _environment())
    item_dir = next((output / "items").iterdir())
    result_path = item_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(execution="error", verdict=None, details={"error": "temporary failure"})
    result_path.write_text(json.dumps(result), encoding="utf-8")

    resumed = runner.run(plan, output, _environment(), resume=True)

    archived = json.loads((item_dir / "attempts/attempt-1/result.json").read_text(encoding="utf-8"))
    current = json.loads(result_path.read_text(encoding="utf-8"))
    assert resumed["status"] == "pass"
    assert archived["execution"] == "error"
    assert current["execution"] == "completed"


def test_report_fails_closed_when_a_planned_result_is_missing(tmp_path: Path) -> None:
    families = tmp_path / "families"
    _family(families, "alpha", "model-a")
    plan = QualificationCatalog(families).plan("accuracy")
    output = tmp_path / "run"
    QualificationRunner(Path(sys.executable)).run(plan, output, _environment())
    first = sorted((output / "items").iterdir())[0]
    (first / "result.json").unlink()

    report = generate_report(load_plan(output / "plan.json"), output)

    assert report["status"] == "error"
    assert report["summary"]["blocking"]["missing"] == 1


def test_report_exposes_an_observation_execution_error_without_calling_it_a_fail(
    tmp_path: Path,
) -> None:
    families = tmp_path / "families"
    _family(families, "alpha", "model-a")
    plan = QualificationCatalog(families).plan("accuracy")
    output = tmp_path / "run"
    QualificationRunner(Path(sys.executable)).run(plan, output, _environment())
    observation = sorted((output / "items").iterdir())[-1]
    (observation / "result.json").unlink()

    report = generate_report(load_plan(output / "plan.json"), output)

    assert report["status"] == "error"
    assert report["summary"]["observation_only"]["missing"] == 1
    assert report["summary"]["blocking"]["fail"] == 0


def test_observation_execution_failure_is_visible_without_becoming_a_verdict(
    tmp_path: Path,
) -> None:
    families = tmp_path / "families"
    _family(families, "alpha", "model-a")
    plan = QualificationCatalog(families).plan("accuracy")
    output = tmp_path / "run"
    QualificationRunner(Path(sys.executable)).run(plan, output, _environment())
    observation = sorted((output / "items").iterdir())[-1]
    (observation / "result.json").unlink()

    report = generate_report(load_plan(output / "plan.json"), output)

    assert report["status"] == "error"
    assert report["summary"]["observation_only"]["missing"] == 1
    assert report["items"][-1]["verdict"] is None


def test_cli_plan_prints_the_discovered_matrix(tmp_path: Path, capsys) -> None:
    families = tmp_path / "families"
    _family(families, "alpha", "model-a")

    assert (
        main(
            [
                "plan",
                "--kind",
                "accuracy",
                "--families-root",
                str(families),
                "--model",
                "model-a",
                "--suite",
                "file-suite",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert [(item["suite_id"], item["case_id"]) for item in payload["items"]] == [
        ("file-suite", "small"),
        ("file-suite", "full"),
    ]


def test_run_configuration_selects_exact_models_without_requiring_a_model_list(
    tmp_path: Path, capsys
) -> None:
    families = tmp_path / "families"
    _family(families, "alpha", "model-a")
    _family(families, "beta", "model-b")
    environment = tmp_path / "environment.yaml"
    environment.write_text(yaml.safe_dump(_environment()), encoding="utf-8")
    run_configuration = tmp_path / "selected.yaml"
    run_configuration.write_text(
        yaml.safe_dump(
            {
                "schema_version": RUN_CONFIGURATION_SCHEMA,
                "name": "selected-device-run",
                "kind": "accuracy",
                "environment": environment.name,
                "models": ["model-b"],
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "plan",
                "--run-config",
                str(run_configuration),
                "--families-root",
                str(families),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert {item["model"] for item in payload["items"]} == {"model-b"}


def test_resume_rejects_an_environment_change(tmp_path: Path) -> None:
    families = tmp_path / "families"
    _family(families, "alpha", "model-a")
    plan = QualificationCatalog(families).plan("accuracy")
    output = tmp_path / "run"
    runner = QualificationRunner(Path(sys.executable))
    runner.run(plan, output, _environment())
    changed = {**_environment(), "name": "another-device"}

    with pytest.raises(QualificationError, match="resume environment does not match"):
        runner.run(plan, output, changed, resume=True)
