# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
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
        "sample_limits": {"suite-a": 5, "suite-b": 5, "suite-c": 5},
        "models": {
            "model-a": {
                "workloads": ["suite-a", "suite-b"],
            },
            "model-b": {
                "workloads": ["suite-c"],
            },
        },
    }


def _perf_cases():
    return [
        {"id": "family-a.default", "family": "family-a", "model": "model-a"},
        {"id": "family-a.long", "family": "family-a", "model": "model-a"},
        {"id": "family-c.default", "family": "family-c", "model": "model-c"},
    ]


def test_plan_expands_every_accuracy_workload_and_perf_entry():
    plan = model_checks.resolve_plan(
        models=["model-a"],
        tasks=["accuracy", "perf"],
        platform=_platform(),
        accuracy_catalog=_accuracy_catalog(),
        accuracy_workloads=(),
        accuracy_bindings={},
        perf_cases=_perf_cases(),
        perf_exclusions={},
    )

    model = plan["models"][0]
    assert [binding["workload"] for binding in model["tasks"]["accuracy"]["bindings"]] == [
        "suite-a",
        "suite-b",
    ]
    assert [binding["entry"] for binding in model["tasks"]["perf"]["bindings"]] == [
        "family-a.default",
        "family-a.long",
    ]
    assert plan["summary"] == {
        "model_count": 1,
        "binding_count": 4,
        "configured_binding_count": 4,
        "unsupported_binding_count": 0,
        "blocker_count": 0,
    }


def test_plan_can_select_one_accuracy_suite_explicitly():
    plan = model_checks.resolve_plan(
        models=["model-a"],
        tasks=["accuracy"],
        platform=_platform(),
        accuracy_catalog=_accuracy_catalog(),
        accuracy_workloads=("suite-b",),
        accuracy_bindings={},
        perf_cases=_perf_cases(),
        perf_exclusions={},
    )

    bindings = plan["models"][0]["tasks"]["accuracy"]["bindings"]
    assert [binding["workload"] for binding in bindings] == ["suite-b"]
    assert [binding["id"] for binding in bindings] == ["accuracy:model-a:suite-b"]


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
        perf_cases=_perf_cases(),
        perf_exclusions={},
    )

    bindings = plan["models"][0]["tasks"]["accuracy"]["bindings"]
    assert [(binding["workload"], binding["status"]) for binding in bindings] == [
        ("suite-a", "configured"),
        ("suite-b", "unsupported"),
    ]
    assert plan["models"][0]["tasks"]["accuracy"]["status"] == "configured"


def test_platform_hardware_exclusion_must_name_a_real_binding():
    with pytest.raises(model_checks.ModelCheckError, match="unknown Accuracy binding"):
        model_checks.audit_platform_unsupported(
            _platform(
                unsupported=(
                    {
                        "model": "model-a",
                        "task": "accuracy",
                        "binding": "missing-suite",
                        "reason": "hardware evidence",
                    },
                )
            ),
            accuracy_catalog=_accuracy_catalog(),
            perf_cases=_perf_cases(),
        )


def test_missing_task_binding_is_a_blocker_not_hardware_unsupported():
    plan = model_checks.resolve_plan(
        models=["model-b"],
        tasks=["accuracy", "perf"],
        platform=_platform(),
        accuracy_catalog=_accuracy_catalog(),
        accuracy_workloads=(),
        accuracy_bindings={},
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


def test_complete_task_matrices_do_not_cross_require_task_bindings():
    plan = model_checks.resolve_plan(
        models=["model-a", "model-c"],
        tasks=["accuracy", "perf"],
        platform=_platform(),
        accuracy_catalog=_accuracy_catalog(),
        accuracy_workloads=(),
        accuracy_bindings={},
        perf_cases=_perf_cases(),
        perf_exclusions={},
        complete_task_matrices=True,
    )

    model_c = next(model for model in plan["models"] if model["model"] == "model-c")
    assert model_c["tasks"]["accuracy"] == {
        "status": "not_applicable",
        "reason": "model belongs only to another selected task's complete matrix",
        "bindings": [],
    }
    assert plan["summary"]["blocker_count"] == 0


def test_all_accuracy_selects_only_accuracy_catalog_models(monkeypatch):
    arguments = model_checks.build_parser().parse_args(
        ["check", "--platform", "gb300", "--task", "accuracy", "--all"]
    )
    captured = {}

    def resolve_plan(**kwargs):
        captured.update(kwargs)
        return {
            "schema_version": "trtmc.model-check-selection/v1",
            "platform": "gb300",
            "platform_source": "platform.yaml",
            "execution": {"task_order": ["accuracy"], "serial_tasks": False},
            "models": [],
            "summary": {
                "model_count": 0,
                "binding_count": 0,
                "configured_binding_count": 0,
                "unsupported_binding_count": 0,
                "blocker_count": 0,
            },
        }

    monkeypatch.setattr(model_checks, "resolve_plan", resolve_plan)
    model_checks._resolve_request(arguments)

    assert set(captured["models"]) == set(captured["accuracy_catalog"]["models"])
    assert captured["complete_task_matrices"] is True


def test_explicit_perf_exclusion_is_not_a_blocker():
    plan = model_checks.resolve_plan(
        models=["model-b"],
        tasks=["perf"],
        platform=_platform(),
        accuracy_catalog=_accuracy_catalog(),
        accuracy_workloads=(),
        accuracy_bindings={},
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
                        "environment": ("benchmarks/performance/environments/gb300.yaml"),
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
    assert environment["storage"]["python_profiles_root"] == str(
        storage / "python-profiles"
    )
    assert environment["tasks"]["accuracy"]["runner_python"] == "python3"
    assert Path(environment["tasks"]["perf"]["suite"]).is_absolute()


def test_runner_executable_preserves_virtual_environment_symlink(tmp_path):
    runner = tmp_path / "venv/bin/python"
    runner.parent.mkdir(parents=True)
    runner.symlink_to(sys.executable)

    assert model_checks._runner_executable(str(runner), "runner") == str(runner)


def test_task_environment_uses_shared_profiles_and_allows_missing_profiles(
    tmp_path,
    monkeypatch,
):
    profiles = tmp_path / "storage/python-profiles"
    monkeypatch.setenv("TRTMC_PYTHON_PROFILE_ROOT", "/opt/trtmc-python-profiles")
    monkeypatch.setenv("TRTMC_PYTHON_PROFILE_PREBUILT_ONLY", "1")

    environment = model_checks._task_environment(
        {"storage": {"python_profiles_root": str(profiles)}}
    )

    assert environment["TRTMC_PYTHON_PROFILE_ROOT"] == str(profiles)
    assert "TRTMC_PYTHON_PROFILE_PREBUILT_ONLY" not in environment
    assert os.environ["TRTMC_PYTHON_PROFILE_ROOT"] == "/opt/trtmc-python-profiles"
    assert os.environ["TRTMC_PYTHON_PROFILE_PREBUILT_ONLY"] == "1"


def test_perf_reference_contracts_come_from_selected_model_owners() -> None:
    plan = {
        "models": [
            {
                "model": model,
                "tasks": {
                    "perf": {
                        "bindings": [
                            {
                                "model": model,
                                "entry": entry,
                                "status": "configured",
                            }
                        ]
                    }
                },
            }
            for model, entry in (
                ("lance-3b-x2t-image", "lance.generate"),
                ("personaplex-7b", "personaplex.speak"),
                ("sana-wm-bidirectional", "sana_wm.generate_image"),
            )
        ]
    }

    contracts = model_checks._selected_perf_reference_contracts(
        plan,
        model_checks.trtmc_validate.DEFAULT_MODELS,
    )

    assert len(contracts) == 3
    assert {contract.environment_variable for contract in contracts} == {
        "TRTMC_LANCE_REFERENCE_REPO",
        "PERSONAPLEX_OFFICIAL_REPO",
        "TRTMC_SANA_WM_REFERENCE_REPO",
    }


def test_prepare_perf_reference_dependencies_warms_once_and_exports_paths(
    tmp_path,
    monkeypatch,
):
    contracts = (
        model_checks.ModelReferenceContract(
            family="family-a",
            repository="https://example.invalid/family-a.git",
            revision="a" * 40,
            relative_path="family-a/reference/source-a",
            entrypoint="entry.py",
            environment_variable="FAMILY_A_REPO",
        ),
        model_checks.ModelReferenceContract(
            family="family-b",
            repository="https://example.invalid/family-b.git",
            revision="b" * 40,
            relative_path="family-b/reference/source-b",
            entrypoint="entry.py",
        ),
    )
    warmed = []

    def warm(_self, contract):
        warmed.append(contract.family)
        return tmp_path / contract.relative_path

    monkeypatch.setattr(model_checks.ModelReferenceCacheWarmer, "warm_contract", warm)

    environment = model_checks._prepare_perf_reference_dependencies(
        contracts,
        tmp_path,
    )

    assert warmed == ["family-a", "family-b"]
    assert environment == {
        "TRTMC_MODEL_REFERENCE_CACHE_ROOT": str(tmp_path),
        "FAMILY_A_REPO": str(tmp_path / "family-a/reference/source-a"),
    }


def test_selected_models_artifact_records_configured_bindings(tmp_path) -> None:
    plan = {
        "models": [
            {
                "tasks": {
                    "accuracy": {
                        "bindings": [
                            {"model": "model-a", "status": "configured"},
                            {"model": "model-b", "status": "unsupported"},
                        ]
                    },
                    "perf": {
                        "bindings": [
                            {"model": "model-a", "status": "configured"},
                            {"model": "model-c", "status": "configured"},
                        ]
                    },
                }
            }
        ]
    }

    selection = model_checks._write_selected_models(plan, tmp_path)

    assert selection.read_text(encoding="utf-8") == "model-a\nmodel-c\n"


@pytest.mark.parametrize(
    ("platform", "hf_cache_mode", "hf_cache_retention"),
    [
        ("gb300", "shared", "retain"),
        ("l4t-thor", "per_model", "delete_always"),
        ("auto-thor", "shared", "retain"),
    ],
)
def test_checked_in_accuracy_environment_deletes_engines_without_fixed_reserve(
    platform,
    hf_cache_mode,
    hf_cache_retention,
):
    path = model_checks.DEFAULT_ENVIRONMENT_ROOT / f"{platform}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    options = raw["tasks"]["accuracy"]["options"]

    assert options["engine-retention"] == "delete_always"
    assert "minimum-free-space-gib" not in options
    assert "local-files-only" not in options
    assert options["hf-cache-mode"] == hf_cache_mode
    assert options["hf-cache-retention"] == hf_cache_retention


def test_l4t_accuracy_environment_bounds_each_model_attempt() -> None:
    raw = model_checks._read_yaml(
        model_checks.DEFAULT_ENVIRONMENT_ROOT / "l4t-thor.yaml",
        "model-check environment",
    )

    assert raw["tasks"]["accuracy"]["options"]["model-timeout-seconds"] == 21600


def test_l4t_marks_minimax_profile_unsupported_by_memory_contract() -> None:
    platform = model_checks.load_platform("l4t-thor")

    assert any(
        item["model"] == "minimax-h3-768p"
        and item["task"] == "accuracy"
        and item["binding"] == "minimax_h3_official_profile_parity"
        and "180 GiB" in item["reason"]
        for item in platform["unsupported"]
    )


def test_l4t_perf_environment_deletes_entry_cache_and_bundle() -> None:
    path = model_checks.REPOSITORY / "benchmarks/performance/environments/l4t-thor.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert raw["storage"]["bundle_retention"] == "delete_always"
    assert raw["execution"]["hf_cache_mode"] == "per_entry"
    assert raw["execution"]["hf_cache_retention"] == "delete_always"


def test_l4t_platform_rejects_unverifiable_nvme_partition():
    platform = model_checks.load_platform("l4t-thor")

    with pytest.raises(model_checks.ModelCheckError, match="/dev/nvme0n1p1"):
        model_checks._require_platform_storage_root(Path("/tmp/run"), platform)


def test_platform_accepts_storage_on_required_device(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    platform = {
        "storage": {"device": str(tmp_path / "device-anchor")},
    }
    (tmp_path / "device-anchor").touch()

    model_checks._require_platform_storage_root(root, platform)


@pytest.mark.parametrize("platform", ["gb300", "l4t-thor", "auto-thor"])
def test_checked_in_platform_resolves_complete_task_matrices(platform):
    assert model_checks.main(["check", "--platform", platform, "--all", "--json"]) == 0


def test_run_dry_run_writes_exact_accuracy_bindings(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    monkeypatch.setenv("TRTMC_CHECK_STORAGE_ROOT", str(storage))
    monkeypatch.setenv("TRTMC_CHECK_DATASET_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("TRTMC_CHECK_BUILD_DIR", str(tmp_path / "runtime"))
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
        (storage / "results" / "unit-dry-run" / "request.json").read_text(encoding="utf-8")
    )
    command = request["commands"]["accuracy"]
    binding_index = command.index("--binding")
    assert command[binding_index + 1] == ("qwen25vl-3b=vlm_mmmu_pro_vision_mcq")
    assert "--local-files-only" not in command
    assert "preparation_commands" not in request
    assert "perf" not in request["commands"]


def test_run_default_output_is_concise_and_ends_with_task_summary(
    tmp_path,
    monkeypatch,
    capsys,
):
    storage = tmp_path / "storage"
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("TRTMC_CHECK_STORAGE_ROOT", str(storage))
    monkeypatch.setenv("TRTMC_CHECK_DATASET_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("TRTMC_CHECK_BUILD_DIR", str(runtime))
    monkeypatch.setenv("TRTMC_CHECK_PYTHON", sys.executable)
    monkeypatch.setenv("TRTMC_PYTHON_PROFILE_PREBUILT_ONLY", "1")
    returncodes = iter((1, 0))
    commands = []
    child_environments = []

    def run(command, **kwargs):
        commands.append(command)
        child_environments.append(kwargs["env"])
        return SimpleNamespace(returncode=next(returncodes))

    monkeypatch.setattr(model_checks.subprocess, "run", run)

    result = model_checks.main(
        [
            "run",
            "--platform",
            "gb300",
            "--model",
            "distilgpt2",
            "--run-id",
            "concise-unit",
        ]
    )

    assert result == 1
    assert len(commands) == 2
    assert all("warm_hf_cache.py" not in command for command in commands)
    assert all(
        environment["TRTMC_PYTHON_PROFILE_ROOT"]
        == str(storage / "python-profiles")
        for environment in child_environments
    )
    assert all(
        "TRTMC_PYTHON_PROFILE_PREBUILT_ONLY" not in environment
        for environment in child_environments
    )
    output = capsys.readouterr().out
    assert "Run: concise-unit" in output
    assert "Order: Accuracy -> Perf" in output
    assert "[1/2] Accuracy" in output
    assert "[2/2] Perf" in output
    assert "Accuracy: FAILED" in output
    assert "Perf: PASSED" in output
    assert "Overall: FAILED" in output
    assert "tools/trtmc_validate.py --binding" not in output
    assert "tools/perf_matrix.py run" not in output


def test_run_verbose_prints_and_forwards_detailed_commands(tmp_path, monkeypatch, capsys):
    storage = tmp_path / "storage"
    monkeypatch.setenv("TRTMC_CHECK_STORAGE_ROOT", str(storage))
    monkeypatch.setenv("TRTMC_CHECK_DATASET_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("TRTMC_CHECK_BUILD_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("TRTMC_CHECK_PYTHON", sys.executable)
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(model_checks.subprocess, "run", run)

    assert (
        model_checks.main(
            [
                "run",
                "--platform",
                "gb300",
                "--task",
                "accuracy",
                "--model",
                "distilgpt2",
                "--run-id",
                "verbose-unit",
                "--verbose",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "tools/trtmc_validate.py --binding" in output
    assert commands[-1][-1] == "--verbose"


def test_run_resume_verifies_request_and_resumes_accuracy(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    monkeypatch.setenv("TRTMC_CHECK_STORAGE_ROOT", str(storage))
    monkeypatch.setenv("TRTMC_CHECK_DATASET_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("TRTMC_CHECK_BUILD_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("TRTMC_CHECK_PYTHON", sys.executable)
    selection = [
        "run",
        "--platform",
        "gb300",
        "--task",
        "accuracy",
        "--model",
        "qwen25vl-3b",
        "--run-id",
        "resume-unit",
    ]
    assert model_checks.main([*selection, "--dry-run"]) == 0

    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(model_checks.subprocess, "run", run)

    assert model_checks.main([*selection, "--resume"]) == 0
    assert commands[-1][-1] == "--resume-existing"
    result = json.loads((storage / "results/resume-unit/result.json").read_text(encoding="utf-8"))
    assert result["resumed"] is True


def test_perf_resume_command_requires_one_existing_run(tmp_path):
    results = tmp_path / "results"
    run = results / "release-family-performance-example"
    run.mkdir(parents=True)
    (run / "results.json").write_text("{}", encoding="utf-8")
    environment = {"tasks": {"perf": {"runner_python": "/venv/bin/python"}}}

    command = model_checks._perf_resume_command(environment, results)

    assert command == [
        "/venv/bin/python",
        str(model_checks.REPOSITORY / "tools/perf_matrix.py"),
        "resume",
        str(run),
    ]


def test_auto_thor_environment_builds_both_task_commands(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("TRTMC_CHECK_STORAGE_ROOT", str(storage))
    monkeypatch.setenv("TRTMC_CHECK_DATASET_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("TRTMC_CHECK_BUILD_DIR", str(runtime))
    monkeypatch.setenv("TRTMC_CHECK_PYTHON", sys.executable)

    assert (
        model_checks.main(
            [
                "run",
                "--platform",
                "auto-thor",
                "--model",
                "distilgpt2",
                "--run-id",
                "auto-thor-unit",
                "--dry-run",
            ]
        )
        == 0
    )

    request = json.loads(
        (storage / "results/auto-thor-unit/request.json").read_text(encoding="utf-8")
    )
    assert set(request["commands"]) == {"accuracy", "perf"}
    assert request["selection"]["execution"]["serial_tasks"] is True
    perf_environment = request["perf_environment_config"]
    assert perf_environment["tools"]["trtmc_worker"] == str(
        runtime / "trtmc_benchmark_worker"
    )
    assert perf_environment["storage"]["bundle_cache"] == str(
        storage / "engines/perf"
    )
    assert perf_environment["storage"]["bundle_roots"] == []
    assert perf_environment["storage"]["runtime_dirs"] == [str(runtime)]
