# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import yaml

from tools import model_checks


def _platform(*, serial: bool = True, unsupported=()):
    return {
        "id": "test-platform",
        "source": "platform.yaml",
        "execution": {
            "task_order": ["accuracy", "perf"],
            "serial_tasks": serial,
        },
        "unsupported": list(unsupported),
    }


def _accuracy_catalog():
    return {
        "models": {
            "model-a": {
                "default": "suite-a",
                "workloads": ["suite-a", "suite-b"],
            },
            "model-b": {
                "default": "suite-c",
                "workloads": ["suite-c"],
            },
        }
    }


def _perf_cases():
    return [
        {"id": "family-a.default", "family": "family-a", "model": "model-a"},
        {"id": "family-a.long", "family": "family-a", "model": "model-a"},
        {"id": "family-c.default", "family": "family-c", "model": "model-c"},
    ]


def test_plan_defaults_accuracy_and_expands_every_perf_entry():
    plan = model_checks.resolve_plan(
        models=["model-a"],
        tasks=["accuracy", "perf"],
        platform=_platform(),
        accuracy_catalog=_accuracy_catalog(),
        accuracy_workloads=(),
        accuracy_bindings={},
        all_accuracy_workloads=False,
        perf_cases=_perf_cases(),
        perf_exclusions={},
    )

    model = plan["models"][0]
    assert [binding["workload"] for binding in model["tasks"]["accuracy"]["bindings"]] == [
        "suite-a"
    ]
    assert [binding["entry"] for binding in model["tasks"]["perf"]["bindings"]] == [
        "family-a.default",
        "family-a.long",
    ]
    assert plan["summary"] == {
        "model_count": 1,
        "binding_count": 3,
        "blocker_count": 0,
    }


def test_plan_can_expand_all_accuracy_suites():
    plan = model_checks.resolve_plan(
        models=["model-a"],
        tasks=["accuracy"],
        platform=_platform(),
        accuracy_catalog=_accuracy_catalog(),
        accuracy_workloads=(),
        accuracy_bindings={},
        all_accuracy_workloads=True,
        perf_cases=_perf_cases(),
        perf_exclusions={},
    )

    bindings = plan["models"][0]["tasks"]["accuracy"]["bindings"]
    assert [binding["workload"] for binding in bindings] == ["suite-a", "suite-b"]
    assert [binding["id"] for binding in bindings] == [
        "accuracy:model-a:suite-a",
        "accuracy:model-a:suite-b",
    ]


def test_plan_can_select_distinct_accuracy_suites_per_model():
    plan = model_checks.resolve_plan(
        models=["model-a", "model-b"],
        tasks=["accuracy"],
        platform=_platform(),
        accuracy_catalog=_accuracy_catalog(),
        accuracy_workloads=(),
        accuracy_bindings={
            "model-a": ["suite-b"],
            "model-b": ["suite-c"],
        },
        all_accuracy_workloads=False,
        perf_cases=_perf_cases(),
        perf_exclusions={},
    )

    assert [
        (record["model"], binding["workload"])
        for record in plan["models"]
        for binding in record["tasks"]["accuracy"]["bindings"]
    ] == [("model-a", "suite-b"), ("model-b", "suite-c")]


def test_platform_hardware_exclusion_can_target_one_accuracy_suite():
    plan = model_checks.resolve_plan(
        models=["model-a"],
        tasks=["accuracy"],
        platform=_platform(
            unsupported=(
                {
                    "model": "model-a",
                    "task": "accuracy",
                    "binding": "suite-b",
                    "reason": "suite-b exceeds unified memory",
                },
            )
        ),
        accuracy_catalog=_accuracy_catalog(),
        accuracy_workloads=(),
        accuracy_bindings={},
        all_accuracy_workloads=True,
        perf_cases=_perf_cases(),
        perf_exclusions={},
    )

    bindings = plan["models"][0]["tasks"]["accuracy"]["bindings"]
    assert [(binding["workload"], binding["status"]) for binding in bindings] == [
        ("suite-a", "configured"),
        ("suite-b", "unsupported"),
    ]
    assert plan["models"][0]["tasks"]["accuracy"]["status"] == "configured"


def test_missing_task_binding_is_a_blocker_not_hardware_unsupported():
    plan = model_checks.resolve_plan(
        models=["model-b"],
        tasks=["accuracy", "perf"],
        platform=_platform(),
        accuracy_catalog=_accuracy_catalog(),
        accuracy_workloads=(),
        accuracy_bindings={},
        all_accuracy_workloads=False,
        perf_cases=_perf_cases(),
        perf_exclusions={},
    )

    tasks = plan["models"][0]["tasks"]
    assert tasks["accuracy"]["status"] == "configured"
    assert tasks["perf"] == {
        "status": "unconfigured",
        "reason": "model has no Perf release entry",
        "bindings": [],
    }
    assert plan["summary"]["blocker_count"] == 1


def test_explicit_perf_exclusion_is_not_a_blocker():
    plan = model_checks.resolve_plan(
        models=["model-b"],
        tasks=["perf"],
        platform=_platform(),
        accuracy_catalog=_accuracy_catalog(),
        accuracy_workloads=(),
        accuracy_bindings={},
        all_accuracy_workloads=False,
        perf_cases=_perf_cases(),
        perf_exclusions={"model-b": "baseline unavailable"},
    )

    task = plan["models"][0]["tasks"]["perf"]
    assert task["status"] == "excluded"
    assert task["reason"] == "baseline unavailable"
    assert plan["summary"]["blocker_count"] == 0


def test_model_ci_owner_expands_task_profiles_without_a_third_roster():
    profiles = model_checks.model_profiles_for_owners(
        ["family-a"],
        tasks=["accuracy", "perf"],
        accuracy_models={
            "model-a": {"family": "family-a"},
            "model-b": {"family": "family-b"},
        },
        accuracy_catalog=_accuracy_catalog(),
        perf_cases=_perf_cases(),
    )

    assert profiles == ("model-a",)


def test_execution_environment_preserves_command_name_and_resolves_paths(
    tmp_path,
    monkeypatch,
):
    storage = tmp_path / "storage"
    monkeypatch.setenv("TEST_STORAGE", str(storage))
    environment_path = tmp_path / "environment.yaml"
    environment_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": model_checks.ENVIRONMENT_SCHEMA,
                "id": "test-platform",
                "storage": {
                    "root": "${TEST_STORAGE}",
                    "results_root": "${TEST_STORAGE}/results",
                },
                "tasks": {
                    "accuracy": {
                        "runner_python": "python3",
                        "options": {},
                    },
                    "perf": {
                        "runner_python": "python3",
                        "suite": "benchmarks/performance/release.yaml",
                        "environment": (
                            "benchmarks/performance/environments/gb300.yaml"
                        ),
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    environment = model_checks.load_execution_environment(
        str(environment_path),
        platform_id="test-platform",
    )

    assert environment["storage"]["root"] == str(storage)
    assert environment["storage"]["results_root"] == str(storage / "results")
    assert environment["tasks"]["accuracy"]["runner_python"] == "python3"
    assert Path(environment["tasks"]["perf"]["suite"]).is_absolute()


def test_runner_executable_preserves_virtual_environment_symlink(tmp_path):
    runner = tmp_path / "venv/bin/python"
    runner.parent.mkdir(parents=True)
    runner.symlink_to(sys.executable)

    assert model_checks._runner_executable(str(runner), "runner") == str(runner)


def test_l4t_platform_rejects_storage_outside_nvme_partition():
    platform = model_checks.load_platform("l4t-thor")

    with pytest.raises(model_checks.ModelCheckError, match="/dev/nvme0n1p1"):
        model_checks._require_platform_storage_root(Path("/tmp/run"), platform)


def test_run_dry_run_writes_exact_accuracy_bindings(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    monkeypatch.setenv("TRTMC_CHECK_STORAGE_ROOT", str(storage))
    monkeypatch.setenv("TRTMC_CHECK_DATASET_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("TRTMC_CHECK_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("TRTMC_CHECK_PYTHON", sys.executable)

    result = model_checks.main(
        [
            "run",
            "--platform",
            "gb300",
            "--task",
            "accuracy",
            "--model",
            "qwen25vl-3b",
            "--accuracy-binding",
            "qwen25vl-3b=vlm_mmmu_pro_vision_mcq",
            "--run-id",
            "unit-dry-run",
            "--dry-run",
        ]
    )

    assert result == 0
    request = json.loads(
        (storage / "results" / "unit-dry-run" / "request.json").read_text(
            encoding="utf-8"
        )
    )
    command = request["commands"]["accuracy"]
    binding_index = command.index("--binding")
    assert command[binding_index + 1] == (
        "qwen25vl-3b=vlm_mmmu_pro_vision_mcq"
    )
    assert "perf" not in request["commands"]
