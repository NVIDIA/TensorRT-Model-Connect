# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
import subprocess
import os
import signal
import time
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


def test_checked_in_gpt2_accuracy_suite_is_discoverable() -> None:
    plan = QualificationCatalog(Path(__file__).resolve().parents[4] / "families").plan(
        "accuracy", models=["gpt2-125m"]
    )

    assert [(item.suite_id, item.case_id) for item in plan.items] == [
        ("mmlu_continuation_parity", "smoke")
    ]
    assert plan.items[0].definition["implementation"] == "mmlu_continuation_parity"
    assert "device" not in plan.items[0].case


def test_checked_in_gpt2_performance_suite_is_discoverable() -> None:
    plan = QualificationCatalog(Path(__file__).resolve().parents[4] / "families").plan(
        "performance", models=["gpt2-125m"]
    )

    assert [(item.suite_id, item.case_id) for item in plan.items] == [
        ("text_generation_performance", "generate_64")
    ]
    assert plan.items[0].gate_policy == "observation_only"
    assert plan.items[0].case["candidate"]["measurement"] == {
        "warmup": 5,
        "iterations": 10,
    }
    assert plan.items[0].case["reference"] == {
        "implementation": "hf_transformers",
        "mode": "torch-compile",
        "compile_scope": "model.forward",
        "precision": "fp32",
    }
    assert "device" not in plan.items[0].case


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


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_executor_timeout_stops_descendants_before_returning(tmp_path):
    qualification = _family(tmp_path / "families", "alpha", "model-a")
    pidfile = tmp_path / "child.pid"
    (qualification / "executor.py").write_text(
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"Path({str(pidfile)!r}).write_text(str(child.pid))\n"
        "time.sleep(30)\n"
    )
    plan = QualificationCatalog(tmp_path / "families").plan("accuracy", cases=["small"])
    directory = tmp_path / "item"
    directory.mkdir()
    QualificationRunner()._run_item(
        plan, plan.items[0], directory, {"tools": {"python": sys.executable}}, 1
    )
    assert json.loads((directory / "result.json").read_text())["execution"] == "error"
    pid = int(pidfile.read_text())
    try:
        for _ in range(100):
            status = Path(f"/proc/{pid}/stat")
            if not status.exists() or status.read_text().split()[2] == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail("executor descendant is still running after timeout")
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_inconclusive_family_entrypoint_survives_runner_and_report(tmp_path):
    family = _family(tmp_path / "families", "gpt2", "gpt2-test")
    config = yaml.safe_load((family / "gpt2-test.accuracy.yaml").read_text())
    config["kind"] = "performance"
    config["suites"] = [config["suites"][1]]
    (family / "gpt2-test.performance.yaml").write_text(yaml.safe_dump(config))
    executor_path = (
        Path(__file__).resolve().parents[4] / "families/gpt2/tests/qualification/executor.py"
    )
    # Use the real family entry point with a deterministic unstable measurement.
    (family / "executor.py").write_text(
        "import importlib.util, json\n"
        f"spec = importlib.util.spec_from_file_location('gpt2_executor', {str(executor_path)!r})\n"
        "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
        "def execute(request, item, directory):\n"
        "    result = {'schema_version': module.RESULT_SCHEMA, **module._identity(item),\n"
        "              'execution': 'completed', 'verdict': None, 'details': {}, 'artifacts': []}\n"
        "    if request.get('phase') == 'run':\n"
        "        for name in ('first.json', 'retry.json'):\n"
        "            (directory / name).write_text('{}')\n"
        "        result.update(execution='error',\n"
        "            details={'error': 'measurement_inconclusive', 'comparison_valid': False,\n"
        "                     'measurement_attempts': [{'stable': False}, {'stable': False}]},\n"
        "            artifacts=[{'path': 'first.json'}, {'path': 'retry.json'}])\n"
        "    return result\n"
        "module._execute = execute\n"
        "raise SystemExit(module.main())\n"
    )
    plan = QualificationCatalog(tmp_path / "families").plan("performance")
    report = QualificationRunner().run(plan, tmp_path / "run", _environment())
    assert report["status"] == "error"
    assert report["summary"]["inconclusive"] == 1
    assert report["summary"]["comparable"] == 0
    result = report["items"][0]
    assert len(result["details"]["measurement_attempts"]) == 2
    assert {"first.json", "retry.json"} <= {artifact["path"] for artifact in result["artifacts"]}


def test_family_python_is_used_by_real_executor_without_inherited_packages(tmp_path, monkeypatch):
    families = tmp_path / "families"
    qualification = _family(families, "alpha", "model-a")
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    python = venv / "bin" / "python"
    site = Path(
        subprocess.check_output(
            [
                str(python),
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            text=True,
        ).strip()
    )
    (site / "family_dependency.py").write_text("VALUE = 'family'\n")
    reference_venv = tmp_path / "reference-venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(reference_venv)], check=True)
    reference_python = reference_venv / "bin" / "python"
    reference_site = reference_venv / site.relative_to(venv)
    (reference_site / "family_dependency.py").write_text("VALUE = 'reference'\n")
    injected = tmp_path / "injected"
    injected.mkdir()
    (injected / "family_dependency.py").write_text("VALUE = 'wrong environment'\n")
    monkeypatch.setenv("PYTHONPATH", str(injected))
    (qualification / "executor.py").write_text(
        "import family_dependency, subprocess\nassert family_dependency.VALUE == 'family'\n"
        + _FAKE_EXECUTOR
        + "\nassert subprocess.check_output([request['environment']['tools']['reference_python'], '-c', "
        "\"import family_dependency; print(family_dependency.VALUE)\"], text=True).strip() == 'reference'\n"
    )
    (qualification / "prepare_environment.py").write_text(
        "import argparse, json\nfrom pathlib import Path\n"
        "p = argparse.ArgumentParser(); p.add_argument('--request'); p.add_argument('--output')\n"
        "a = p.parse_args()\n"
        f"Path(a.output).write_text(json.dumps({{'python': {str(python)!r}, 'reference_python': {str(reference_python)!r}}}))\n"
    )
    plan = QualificationCatalog(families).plan("accuracy")
    report = QualificationRunner().run(plan, tmp_path / "run", _environment())
    assert report["status"] == "pass"
    command = json.loads(next((tmp_path / "run" / "items").glob("*/command.json")).read_text())
    assert command["argv"][0] == str(python)
    receipt = json.loads((tmp_path / "run/environments/alpha/environment.json").read_text())
    assert receipt["evidence"]["python"]["prefix"] == str(venv)
    assert receipt["evidence"]["reference_python"]["prefix"] == str(reference_venv)

    metadata = site / "changed_dependency-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: changed-dependency\nVersion: 1.0\n")
    resumed = QualificationRunner().run(plan, tmp_path / "run", _environment(), resume=True)
    assert resumed["status"] == "error"
    assert all("environment changed" in row["details"]["error"] for row in resumed["items"])


def test_environment_failure_is_family_local_and_prepare_does_not_report_accuracy(tmp_path):
    families = tmp_path / "families"
    _family(families, "alpha", "model-a")
    broken = _family(families, "beta", "model-b")
    (broken / "prepare_environment.py").write_text(
        "raise RuntimeError('incompatible dependencies')\n"
    )
    plan = QualificationCatalog(families).plan("accuracy")
    report = QualificationRunner().run(plan, tmp_path / "run", _environment())
    assert report["status"] == "error"
    assert all(
        row["execution"] == "completed" for row in report["items"] if row["family"] == "alpha"
    )
    assert all(row["execution"] == "error" for row in report["items"] if row["family"] == "beta")

    selected = QualificationCatalog(families).plan("accuracy", models=["model-a"])
    root = tmp_path / "prepared"
    receipt = QualificationRunner().run(selected, root, _environment(), prepare_only=True)
    assert receipt["status"] == "prepared"
    assert not (root / "report.json").exists()
    assert not list((root / "items").glob("*/result.json"))
    resumed = QualificationRunner().run(selected, root, _environment(), resume=True)
    assert resumed["status"] == "pass"


def test_resume_rejects_modified_prepared_input_before_reusing_completed_results(tmp_path):
    families = tmp_path / "families"
    qualification = _family(families, "alpha", "model-a")
    artifact = tmp_path / "prepared.bin"
    artifact.write_bytes(b"original")
    source = _FAKE_EXECUTOR.replace(
        "args.output.write_text(json.dumps(result))",
        f"result['details']['prepared'] = {{'input_files': [{str(artifact)!r}]}}\n"
        "args.output.write_text(json.dumps(result))",
    )
    (qualification / "executor.py").write_text(source)
    plan = QualificationCatalog(families).plan("accuracy")
    runner = QualificationRunner()
    assert runner.run(plan, tmp_path / "run", _environment())["status"] == "pass"
    artifact.write_bytes(b"changed")
    report = runner.run(plan, tmp_path / "run", _environment(), resume=True)
    assert report["status"] == "error"
    assert report["summary"]["comparable"] == 0
    assert all("prepared input files changed" in row["details"]["error"] for row in report["items"])


def test_device_run_target_does_not_change_family_observation_verdict(tmp_path):
    families = tmp_path / "families"
    qualification = _family(families, "alpha", "model-a")
    config = yaml.safe_load((qualification / "model-a.accuracy.yaml").read_text())
    config["kind"] = "performance"
    config["suites"] = [config["suites"][1]]
    (qualification / "model-a.performance.yaml").write_text(yaml.safe_dump(config))
    (qualification / "executor.py").write_text(
        _FAKE_EXECUTOR.replace(
            '"metrics": {"count": count}', '"metrics": {"reference_over_candidate_p50": 0.5}'
        )
    )
    plan = QualificationCatalog(families).plan("performance")
    report = QualificationRunner().run(
        plan,
        tmp_path / "run",
        {**_environment(), "performance_target": {"minimum_speedup": 1.0, "blocking": True}},
    )
    assert report["status"] == "fail"
    assert report["items"][0]["verdict"] is None
    assert report["items"][0]["performance_target"]["passed"] is False


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
