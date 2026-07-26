# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex

import pytest

from tools import task_eval
from tools import trtmc_compare
from tools import trtmc_disagreements
from tools import trtmc_reference
from tools import trtmc_validate


def test_model_workload_catalog_covers_every_ready_model():
    catalog = trtmc_validate.load_catalog()
    suites = task_eval.load_suites()
    task_models = trtmc_validate._task_eval_models(trtmc_validate.DEFAULT_MODELS)
    ready_models = trtmc_validate.ready_model_names()

    trtmc_validate.audit_catalog(
        catalog,
        ready_models=ready_models,
        suite_names=(suite["id"] for suite in suites),
    )
    trtmc_validate.audit_workload_compatibility(
        catalog,
        suites={suite["id"]: suite for suite in suites},
        task_models=task_models,
    )

    assert len(catalog["models"]) == len(ready_models) == 105


def test_validation_ready_models_exclude_l0_only_profiles():
    records = task_eval.load_manifest_records(trtmc_validate.DEFAULT_MODELS)
    eligible = {
        str(record["name"])
        for record in records
        if not record["requires_multi_device"] and not record.get("skip")
    }
    l0_only = {
        str(record["name"])
        for record in records
        if record.get("ci_tier") == "l0_only"
    }
    selected = set(trtmc_validate.ready_model_names())

    assert l0_only
    assert selected == eligible - l0_only


def test_catalog_defines_sample_limit_for_every_dataset_workload():
    catalog = trtmc_validate.load_catalog()
    configured = set(catalog["sample_limits"])
    declared = {
        workload
        for spec in catalog["models"].values()
        for workload in spec["workloads"]
        if workload != "e2e"
    }

    assert configured == declared
    assert min(catalog["sample_limits"].values()) >= 2
    assert max(catalog["sample_limits"].values()) == 100
    assert catalog["sample_limits"]["mmlu_five_shot_mcq"] == 20
    assert catalog["sample_limits"]["dpg_bench_diffusion_image"] == 5
    assert catalog["sample_limits"]["gedit_bench_image_edit"] == 5


def test_every_dataset_backed_validation_binding_has_native_reference_runner():
    catalog = trtmc_validate.load_catalog()
    suites = {suite["id"]: suite for suite in task_eval.load_suites()}
    bindings = [
        (model_name, workload)
        for model_name, spec in catalog["models"].items()
        for workload in spec["workloads"]
        if workload != "e2e"
    ]
    missing = []
    for model_name, workload in bindings:
        dataset_kind = str(suites[workload]["dataset"]["kind"])
        if trtmc_reference.native_reference_runner_for_dataset_kind(
            dataset_kind
        ) is None:
            missing.append((model_name, workload, dataset_kind))

    assert not missing
    assert len({model for model, _workload in bindings}) == 95


def test_resolve_binding_defaults_and_rejects_undeclared_workload():
    catalog = {
        "models": {
            "model-a": {
                "default": "workload-a",
                "workloads": ["workload-a", "workload-b"],
            }
        }
    }

    assert trtmc_validate.resolve_binding(catalog, "model-a") == (
        trtmc_validate.Binding("model-a", "workload-a")
    )
    assert trtmc_validate.resolve_binding(catalog, "model-a", "workload-b") == (
        trtmc_validate.Binding("model-a", "workload-b")
    )
    with pytest.raises(trtmc_validate.ValidationError, match="does not declare"):
        trtmc_validate.resolve_binding(catalog, "model-a", "workload-c")


def test_resolve_sample_limit_uses_workload_policy_and_cli_override():
    catalog = {
        "sample_limits": {"workload-a": 50},
        "models": {
            "model-a": {
                "default": "workload-a",
                "workloads": ["workload-a"],
            },
            "model-e2e": {
                "default": "e2e",
                "workloads": ["e2e"],
            },
        },
    }

    assert (
        trtmc_validate.resolve_sample_limit(
            catalog,
            trtmc_validate.Binding("model-a", "workload-a"),
            None,
        )
        == 50
    )
    assert (
        trtmc_validate.resolve_sample_limit(
            catalog,
            trtmc_validate.Binding("model-a", "workload-a"),
            7,
        )
        == 7
    )
    assert (
        trtmc_validate.resolve_sample_limit(
            catalog,
            trtmc_validate.Binding("model-a", "workload-a"),
            0,
        )
        == 0
    )
    assert (
        trtmc_validate.resolve_sample_limit(
            catalog,
            trtmc_validate.Binding("model-e2e", "e2e"),
            None,
        )
        == 0
    )


def test_all_defaults_to_continue_and_accepts_stop_policy():
    parser = trtmc_validate.build_parser()

    default = parser.parse_args(["--all"])
    stop = parser.parse_args(["--all", "--on-model-failure", "stop"])

    assert default.on_model_failure == "continue"
    assert stop.on_model_failure == "stop"


@pytest.mark.parametrize(
    ("policy", "expected_models"),
    [
        ("continue", ["model-a", "model-b"]),
        ("stop", ["model-a"]),
    ],
)
def test_all_supervisor_applies_model_failure_policy(
    tmp_path,
    monkeypatch,
    policy,
    expected_models,
):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--on-model-failure",
            policy,
            "--output",
            str(tmp_path / "results"),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
        ]
    )
    bindings = [
        trtmc_validate.Binding("model-a", "workload-a"),
        trtmc_validate.Binding("model-b", "workload-b"),
    ]
    catalog = {
        "sample_limits": {
            "workload-a": 5,
            "workload-b": 10,
        }
    }
    attempted = []

    def run_worker(binding, *, arguments, catalog):
        attempted.append(binding.model)
        status = "failed" if binding.model == "model-a" else "passed"
        return {
            "model": binding.model,
            "workload": binding.workload,
            "validation": {"status": status},
        }

    monkeypatch.setattr(trtmc_validate, "_run_supervised_binding", run_worker)
    monkeypatch.setattr(trtmc_validate, "write_run_metadata", lambda output: output)
    monkeypatch.setattr(
        trtmc_validate,
        "write_report",
        lambda output: (
            output / "report.json",
            output / "report.html",
            {},
        ),
    )
    monkeypatch.setattr(trtmc_validate, "_print_result", lambda *args: None)

    returncode = trtmc_validate._run_all_bindings(
        bindings,
        arguments=arguments,
        catalog=catalog,
    )

    assert returncode == 1
    assert attempted == expected_models


def test_supervised_binding_replaces_stale_result_with_worker_crash(tmp_path, monkeypatch):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--output",
            str(tmp_path / "results"),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
            "--local-files-only",
        ]
    )
    binding = trtmc_validate.Binding("model-a", "workload-a")
    catalog = {"sample_limits": {"workload-a": 5}}
    case_dir = arguments.output / binding.model / binding.workload
    case_dir.mkdir(parents=True)
    comparison = case_dir / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "model": binding.model,
                "workload": binding.workload,
                "status": "passed",
            }
        ),
        encoding="utf-8",
    )

    def crash(command, log_path, env):
        log_path.write_text("worker crashed before comparison\n", encoding="utf-8")
        return 2

    monkeypatch.setattr(trtmc_validate, "_run_subprocess", crash)

    result = trtmc_validate._run_supervised_binding(
        binding,
        arguments=arguments,
        catalog=catalog,
    )

    assert result["execution"] == {"status": "error", "exit_code": 2}
    assert result["comparison"]["status"] == "not_run"
    assert result["validation"]["status"] == "failed"
    assert result["raw_result"]["error_type"] == "WorkerProcessError"
    assert result["reproduce"]["dataset"]["sample_limit"] == 5
    assert "--model-worker" in result["reproduce"]["dataset"]["command"]
    assert "--local-files-only" in result["reproduce"]["dataset"]["command"]
    assert json.loads(comparison.read_text(encoding="utf-8")) == result


def test_supervised_binding_accepts_fresh_worker_result(tmp_path, monkeypatch):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--output",
            str(tmp_path / "results"),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
        ]
    )
    binding = trtmc_validate.Binding("model-a", "workload-a")
    catalog = {"sample_limits": {"workload-a": 5}}
    comparison = (
        arguments.output / binding.model / binding.workload / "comparison.json"
    )

    def pass_worker(command, log_path, env):
        comparison.write_text(
            json.dumps(
                {
                    "model": binding.model,
                    "workload": binding.workload,
                    "status": "passed",
                    "returncode": 0,
                    "raw_result": {"status": "passed"},
                    "reproduce": {
                        "dataset": {
                            "command": "internal worker command",
                            "sample_limit": 5,
                            "prepared_input_count": 5,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(trtmc_validate, "_run_subprocess", pass_worker)

    result = trtmc_validate._run_supervised_binding(
        binding,
        arguments=arguments,
        catalog=catalog,
    )

    assert result["execution"]["status"] == "completed"
    assert result["comparison"]["status"] == "agreement"
    assert result["validation"]["status"] == "passed"
    assert result["worker_log"].endswith("/model-a/workload-a/worker.log")
    assert "--model-worker" not in result["reproduce"]["dataset"]["command"]


def test_all_dry_run_emits_machine_readable_ci_cases(monkeypatch, capsys):
    catalog = {
        "sample_limits": {"workload-a": 5},
        "models": {
            "model-a": {
                "default": "workload-a",
                "workloads": ["workload-a"],
            },
            "model-e2e": {
                "default": "e2e",
                "workloads": ["e2e"],
            },
        },
    }
    monkeypatch.setattr(
        trtmc_validate,
        "_load_validation_inputs",
        lambda arguments: (
            catalog,
            {"workload-a": {}},
            ("model-a", "model-e2e"),
            {},
        ),
    )

    returncode = trtmc_validate.main(["--all", "--dry-run"])

    assert returncode == 0
    assert json.loads(capsys.readouterr().out) == [
        {
            "model": "model-a",
            "workload": "workload-a",
            "sample_limit": 5,
        },
        {
            "model": "model-e2e",
            "workload": "e2e",
            "sample_limit": 0,
        },
    ]


@pytest.mark.parametrize(
    ("validation_status", "comparison_status", "expected_returncode"),
    [
        ("passed", "agreement", 0),
        ("failed", "disagreement", 1),
    ],
)
def test_single_ci_case_writes_stable_result_and_exit_code(
    tmp_path,
    monkeypatch,
    validation_status,
    comparison_status,
    expected_returncode,
):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "model-a",
            "workload-a",
            "--output",
            str(tmp_path / "results"),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
        ]
    )
    binding = trtmc_validate.Binding("model-a", "workload-a")
    catalog = {"sample_limits": {"workload-a": 5}}

    def run_binding(binding, *, arguments, task_models, e2e_models, suites):
        result = {
            "schema_version": "trtmc.validation-result/v2",
            "model": binding.model,
            "workload": binding.workload,
            "execution": {"status": "completed", "exit_code": 0},
            "comparison": {
                "status": comparison_status,
                "mode": "test",
                "primary_metric": None,
                "metrics": {},
                "failures": [],
            },
            "validation": {"status": validation_status},
            "reproduce": {
                "dataset": {
                    "command": "python tools/trtmc_validate.py model-a workload-a",
                    "sample_limit": arguments.limit,
                    "prepared_input_count": arguments.limit,
                },
                "hf": [],
                "trtmc": [],
            },
        }
        case_dir = arguments.output / binding.model / binding.workload
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "comparison.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(trtmc_validate, "_e2e_models", lambda models_dir: {})
    monkeypatch.setattr(trtmc_validate, "run_binding", run_binding)

    returncode = trtmc_validate._run_bindings(
        [binding],
        arguments=arguments,
        catalog=catalog,
        task_models={},
        suites={"workload-a": {}},
    )

    case_dir = arguments.output / "model-a" / "workload-a"
    assert returncode == expected_returncode
    assert (case_dir / "comparison.json").is_file()
    assert (arguments.output / "report.json").is_file()
    assert (arguments.output / "report.html").is_file()


def test_single_ci_case_returns_exit_two_for_setup_error(monkeypatch):
    def fail(arguments):
        raise trtmc_validate.ValidationError("missing CI dataset")

    monkeypatch.setattr(trtmc_validate, "_load_validation_inputs", fail)

    with pytest.raises(SystemExit) as error:
        trtmc_validate.main(["model-a", "workload-a"])

    assert error.value.code == 2


@pytest.mark.parametrize(
    "reference_backend",
    ["hf_transformers", "torch_reference", "diffusers_reference"],
)
def test_default_reference_backends_share_common_environment(reference_backend):
    assert (
        trtmc_validate._declared_profile(
            family="",
            runtime_strategy="",
            reference_backend=reference_backend,
            execution_profiles=None,
        )
        == trtmc_validate.COMMON_REFERENCE_PROFILE
    )


def test_model_specific_reference_environment_keeps_common_validation_base() -> None:
    profiles = trtmc_validate._binding_profiles(
        trtmc_validate.Binding("elf", "dataset"),
        task_models={
            "elf": {
                "family": "elf_flow",
                "runtime_strategy": "elf_flow",
                "reference_backend": "hf_transformers",
            }
        },
        e2e_models={},
    )

    assert profiles == (
        trtmc_validate.COMMON_REFERENCE_PROFILE,
        "elf_flow_reference",
    )


def test_ensure_environments_reports_create_only_when_resolver_creates(monkeypatch, capsys):
    calls = 0

    def resolve(name, base_python, *, on_create):
        nonlocal calls
        calls += 1
        if calls == 1:
            on_create(name)
        return f"/profiles/{name}/bin/python"

    monkeypatch.setattr(trtmc_validate, "resolve_profile_python", resolve)

    cold = trtmc_validate.ensure_environments(
        [trtmc_validate.COMMON_REFERENCE_PROFILE],
        "/base/python",
    )
    cold_output = capsys.readouterr().out
    warm = trtmc_validate.ensure_environments(
        [trtmc_validate.COMMON_REFERENCE_PROFILE],
        "/base/python",
    )
    warm_output = capsys.readouterr().out

    assert "Creating reference environment: reference_common" in cold_output
    assert "Using reference environment: /profiles/reference_common/bin/python" in (cold_output)
    assert "Creating reference environment" not in warm_output
    assert "Using reference environment: /profiles/reference_common/bin/python" in (warm_output)
    assert cold.base_python == "/profiles/reference_common/bin/python"
    assert warm.base_python == cold.base_python


def test_reference_sources_create_once_then_reuse(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    source = trtmc_validate.ReferenceSource(
        name="ELF",
        repository="https://example.invalid/ELF.git",
        revision="0123456789abcdef",
        relative_checkout=Path("elf/reference/ELF-0123456789ab"),
        entrypoint=Path("src/entrypoint.py"),
    )
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        if command[1] == "-C":
            checkout = Path(command[2])
            entrypoint = checkout / source.entrypoint
            entrypoint.parent.mkdir(parents=True)
            entrypoint.write_text("# reference\n", encoding="utf-8")
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(trtmc_validate.subprocess, "run", fake_run)

    cold = trtmc_validate._ensure_reference_source(source, tmp_path)
    cold_output = capsys.readouterr().out
    command_count = len(commands)
    warm = trtmc_validate._ensure_reference_source(source, tmp_path)
    warm_output = capsys.readouterr().out

    assert cold == warm == tmp_path / source.relative_checkout
    assert command_count == 2
    assert len(commands) == command_count
    assert "Creating reference source: ELF" in cold_output
    assert f"Using reference source: {cold}" in cold_output
    assert "Creating reference source" not in warm_output
    assert f"Using reference source: {warm}" in warm_output


def test_elf_reference_source_is_pinned_to_upstream_pytorch_implementation() -> None:
    assert trtmc_validate.ELF_SOURCE.revision == (
        "b29d8833609e9ab7f67cd9da39435ac5cea04837"
    )
    assert trtmc_validate.ELF_SOURCE.relative_checkout == Path(
        "elf/reference/ELF-b29d8833609e"
    )


def test_reference_sources_select_model_specific_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared: list[str] = []

    def prepare(source, cache_root):
        prepared.append(source.name)
        checkout = cache_root / source.relative_checkout
        entrypoint = checkout / source.entrypoint
        entrypoint.parent.mkdir(parents=True)
        entrypoint.write_text("# reference\n", encoding="utf-8")
        return checkout

    monkeypatch.setattr(trtmc_validate, "_ensure_reference_source", prepare)

    elf = trtmc_validate.ensure_reference_sources("elf_flow", tmp_path)
    sana = trtmc_validate.ensure_reference_sources("sana_wm", tmp_path)
    common = trtmc_validate.ensure_reference_sources("bert", tmp_path)

    assert prepared == ["ELF", "SANA-WM"]
    assert (
        elf.elf_reference_repo
        == tmp_path / trtmc_validate.ELF_SOURCE.relative_checkout
    )
    assert elf.environment["TRTMC_STORAGE_ROOT"] == str(tmp_path)
    assert sana.environment["SANA_WM_SCRIPT"] == str(
        tmp_path
        / trtmc_validate.SANA_WM_SOURCE.relative_checkout
        / trtmc_validate.SANA_WM_SOURCE.entrypoint
    )
    assert common.elf_reference_repo is None
    assert common.environment == {"TRTMC_STORAGE_ROOT": str(tmp_path)}


def test_print_result_only_exposes_raw_commands_and_result_locations(tmp_path, capsys):
    comparison = tmp_path / "comparison.json"
    report = tmp_path / "report.html"
    trtmc_validate._print_result(
        {
            "reproduce": {
                "dataset": {
                    "command": "python tools/trtmc_validate.py model-a --limit 1000",
                    "sample_limit": 1000,
                    "prepared_input_count": 1000,
                },
                "hf": ["python hf_reference.py --model model-a"],
                "trtmc": ["trtmc run --model model-a"],
            }
        },
        comparison,
        report,
    )

    output = capsys.readouterr().out
    assert output == (
        "\n"
        "Reproduce dataset run:\n"
        "  python tools/trtmc_validate.py model-a --limit 1000\n"
        "\n"
        "Reproduce representative HF:\n"
        "  python hf_reference.py --model model-a\n"
        "\n"
        "Reproduce representative TRTMC:\n"
        "  trtmc run --model model-a\n"
        "\n"
        f"Compare result: {comparison}\n"
        f"Report:         {report}\n"
    )
    assert "package" not in output.lower()
    assert "token-agreement" not in output
    assert "env action" not in output.lower()


def test_print_result_does_not_mislabel_validation_wrapper_as_raw_command(tmp_path, capsys):
    comparison = tmp_path / "comparison.json"
    report = tmp_path / "report.html"
    trtmc_validate._print_result(
        {
            "reproduce": {
                "hf": [],
                "trtmc": [],
                "validation": "python tools/trtmc_validate.py model-a",
            }
        },
        comparison,
        report,
    )

    output = capsys.readouterr().out
    assert output.count("unavailable; see comparison result") == 3
    assert "python tools/trtmc_validate.py model-a" not in output


def test_write_report_links_each_comparison(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    comparison = case_dir / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "passed",
                "reference_environment": [
                    {"name": "reference_common", "python": "/profiles/python"}
                ],
                "reproduce": {
                    "hf": ["python hf.py"],
                    "trtmc": ["trtmc run"],
                    "dataset": {
                        "command": "python tools/trtmc_validate.py model-a",
                        "sample_limit": 500,
                        "prepared_input_count": 1000,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    json_path, html_path, report = trtmc_validate.write_report(tmp_path)

    assert report["summary"] == {
        "cases": 1,
        "execution_completed": 1,
        "execution_errors": 0,
        "agreements": 1,
        "disagreements": 0,
        "not_compared": 0,
        "validation_passed": 1,
        "validation_failed": 0,
        "validation_skipped": 0,
        "selected_samples": 500,
    }
    assert report["validation_status"] == "passed"
    assert report["results"][0]["execution"]["status"] == "completed"
    assert report["results"][0]["comparison"]["status"] == "agreement"
    assert report["results"][0]["validation"]["status"] == "passed"
    assert "status" not in report["results"][0]
    assert json_path == tmp_path / "report.json"
    assert html_path == tmp_path / "report.html"
    document = html_path.read_text(encoding="utf-8")
    assert "model-a/workload-a/comparison.json" in document
    assert "Agreement" in document
    assert "Completed" in document
    assert "TRTMC Reference Consistency Report" in document
    assert "Vanilla reproduction" in document
    assert "Dataset · Reference 1/1 · TRTMC 1/1" in document
    assert "Dataset slice (500 samples)" in document
    assert "<th>Samples</th>" in document
    assert "<td>500</td>" in document
    assert report["summary"]["selected_samples"] == 500
    assert "prepared inputs" not in document
    assert "$ python tools/trtmc_validate.py model-a" in document
    assert "$ python hf.py" in document
    assert "$ trtmc run" in document


def test_write_report_records_total_duration(tmp_path, monkeypatch):
    started_at = "2026-07-25T01:02:03+00:00"
    finished_at = datetime(2026, 7, 25, 4, 4, 6, 500000, tzinfo=timezone.utc)
    (tmp_path / "run.json").write_text(
        json.dumps({"started_at": started_at}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        trtmc_validate,
        "_utc_now",
        lambda: finished_at,
    )

    _, html_path, report = trtmc_validate.write_report(tmp_path)

    assert report["summary"]["duration_seconds"] == 10_923.5
    assert "3h 02m 04s total duration" in html_path.read_text(encoding="utf-8")


def test_write_report_does_not_render_validation_wrapper(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "failed",
                "reproduce": {
                    "hf": ["python hf.py --prompt '<hello>'"],
                    "trtmc": [],
                    "validation": "python tools/task_eval.py eval --model model-a",
                },
            }
        ),
        encoding="utf-8",
    )

    _, html_path, _ = trtmc_validate.write_report(tmp_path)

    document = html_path.read_text(encoding="utf-8")
    assert "python hf.py --prompt &#x27;&lt;hello&gt;&#x27;" in document
    assert "Not reached; see comparison.json." in document
    assert "task_eval.py" not in document
    migrated = json.loads((case_dir / "comparison.json").read_text(encoding="utf-8"))
    assert "validation" not in migrated["reproduce"]
    assert set(migrated["reproduce"]) == {
        "command_count",
        "command_logs",
        "commands_shown",
        "dataset",
        "hf",
        "representative",
        "trtmc",
    }


def test_write_report_recovers_json_logged_runner_command(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    work_dir = case_dir / "task-eval"
    work_dir.mkdir(parents=True)
    (work_dir / "trtfb_run.log").write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "command": ["trtmc", "solve", "model.trtfb", "--field-input", "1,2"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "passed",
                "raw_result": {"work_dir": str(work_dir)},
                "reproduce": {"hf": ["python hf.py"], "trtmc": ["trtmc build"]},
            }
        ),
        encoding="utf-8",
    )

    _, html_path, report = trtmc_validate.write_report(tmp_path)

    assert report["results"][0]["reproduce"]["trtmc"] == [
        "trtmc build",
        "trtmc solve model.trtfb --field-input 1,2",
    ]
    assert "$ trtmc solve model.trtfb --field-input 1,2" in html_path.read_text(
        encoding="utf-8"
    )


def test_report_bounds_large_sample_commands_and_selects_disagreement(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    work_dir = case_dir / "validation" / "workload-a" / "model-a"
    work_dir.mkdir(parents=True)
    sample_count = 10_000
    (work_dir / "prompts.jsonl").write_text(
        "".join(
            json.dumps({"sample_id": f"sample-{index}", "prompt": f"prompt-{index}"})
            + "\n"
            for index in range(sample_count)
        ),
        encoding="utf-8",
    )
    (work_dir / "trtfb_run.log").write_text(
        "".join(
            f"$ trtmc run model.trtfb --prompt prompt-{index}\n"
            for index in range(sample_count)
        ),
        encoding="utf-8",
    )
    (work_dir / "summary.json").write_text(
        json.dumps({"disagreements": [{"sample_id": "sample-9999"}]}),
        encoding="utf-8",
    )
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "failed",
                "raw_result": {
                    "status": "failed",
                    "work_dir": str(work_dir),
                },
                "reproduce": {
                    "dataset": {
                        "command": (
                            "python tools/trtmc_validate.py model-a --limit 10000"
                        ),
                        "prepared_input_count": sample_count,
                    },
                    "hf": [],
                    "trtmc": [],
                },
            }
        ),
        encoding="utf-8",
    )

    _, html_path, report = trtmc_validate.write_report(tmp_path)

    reproduction = report["results"][0]["reproduce"]
    assert reproduction["command_count"]["trtmc"] == sample_count
    assert reproduction["commands_shown"]["trtmc"] == 1
    assert reproduction["trtmc"] == [
        "trtmc run model.trtfb --prompt prompt-9999"
    ]
    assert reproduction["representative"] == {
        "sample_id": "sample-9999",
        "reason": "first_disagreement",
    }
    assert reproduction["command_logs"]["trtmc"] == ["trtfb_run.log"]
    assert "prompt-5000" not in json.dumps(report)
    document = html_path.read_text(encoding="utf-8")
    assert "Showing 1 of 10000 commands" in document
    assert "prompt-9999" in document
    assert "prompt-5000" not in document
    assert (case_dir / "comparison.json").stat().st_size < 20_000
    assert (tmp_path / "report.json").stat().st_size < 20_000


def test_report_adds_failed_sample_results_and_native_commands(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    work_dir = case_dir / "validation" / "workload-a" / "model-a"
    work_dir.mkdir(parents=True)
    prompt = {
        "sample_id": "sample-7",
        "eval_index": 7,
        "prompt": "Complete this sentence",
    }
    (work_dir / "prompts.jsonl").write_text(
        json.dumps(prompt) + "\n",
        encoding="utf-8",
    )
    (work_dir / "summary.json").write_text(
        json.dumps(
            {
                "disagreements": [
                    {
                        "sample_id": "sample-7",
                        "hf_prediction": "reference answer",
                        "trtfb_prediction": "TRTMC answer",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "hf_predictions.json").write_text(
        json.dumps(
            {
                "responses": [
                    {
                        "sample_id": "sample-7",
                        "output_text": "reference answer",
                        "generated_token_ids": [1, 2],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "trtfb_predictions.json").write_text(
        json.dumps(
            {
                "responses": [
                    {
                        "sample_id": "sample-7",
                        "output_text": "TRTMC answer",
                        "generated_token_ids": [1, 3],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "hf_native_repro.json").write_text(
        json.dumps(
            {
                "command": [
                    "/profiles/reference/bin/python",
                    "/workspace/trtmc/tools/reference/transformers_text.py",
                    "--prompts",
                    "{work_dir}/prompts.jsonl",
                    "--sample-id",
                    "{sample_id}",
                    "--predictions",
                    "{reference_predictions_json}",
                ]
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "trtfb_repro.json").write_text(
        json.dumps(
            {
                "command": [
                    "/workspace/build/trtmc_dataset_benchmark",
                    "model.trtfb",
                    "{input_jsonl}",
                    "{trtmc_raw_jsonl}",
                    "--max-new-tokens",
                    "8",
                    "--seed",
                    "{sample_seed}",
                ],
                "base_seed": 42,
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "failed",
                "raw_result": {"status": "failed", "work_dir": str(work_dir)},
            }
        ),
        encoding="utf-8",
    )

    _, html_path, report = trtmc_validate.write_report(tmp_path)

    metadata = report["results"][0]["disagreements"]
    assert metadata["count"] == 1
    artifact = case_dir / metadata["path"]
    records = [
        json.loads(line)
        for line in artifact.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["input"] == prompt
    assert records[0]["reference_result"]["output_text"] == "reference answer"
    assert records[0]["trtmc_result"]["output_text"] == "TRTMC answer"
    assert records[0]["reproduce"]["reference"].startswith(
        "/profiles/reference/bin/python "
        "/workspace/trtmc/tools/reference/transformers_text.py"
    )
    assert records[0]["reproduce"]["trtmc"].startswith(
        "/workspace/build/trtmc_dataset_benchmark model.trtfb"
    )
    assert records[0]["reproduce"]["trtmc"].endswith("--seed 49")
    assert (case_dir / records[0]["artifacts"]["trtmc_input"]).read_text(
        encoding="utf-8"
    ) == json.dumps(prompt, ensure_ascii=False) + "\n"
    rendered = html_path.read_text(encoding="utf-8")
    assert "1 failed samples · results and vanilla commands" in rendered
    assert "Reference result" in rendered
    assert "TRTMC result" in rendered
    assert "reference answer" in rendered
    assert "TRTMC answer" in rendered
    assert "Reference vanilla command" in rendered
    assert "TRTMC vanilla command" in rendered
    for wrapper in (
        "task_eval.py",
        "trtmc_compare.py",
        "trtmc_reference.py",
        "trtmc_validate.py",
    ):
        assert wrapper not in json.dumps(records)


def test_cached_reference_command_is_relocated_to_current_work_dir(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "current run"
    work_dir.mkdir()
    (work_dir / "prompts.jsonl").write_text(
        json.dumps({"sample_id": "sample-1"}) + "\n",
        encoding="utf-8",
    )
    old_work_dir = Path("/runs/results/old-run/model/workload")
    (work_dir / "hf_native_run.log").write_text(
        "$ python reference.py "
        f"--prompts {shlex.quote(str(old_work_dir / 'prompts.jsonl'))} "
        f"--answers {shlex.quote(str(old_work_dir / 'answers.json'))} "
        f"--manifest {shlex.quote(str(old_work_dir / 'manifest.json'))} "
        f"--output {shlex.quote(str(old_work_dir / 'hf_predictions.json'))}\n",
        encoding="utf-8",
    )

    command = trtmc_validate._commands_from_logs(work_dir)["hf"][0]

    assert str(old_work_dir) not in command
    assert shlex.quote(str(work_dir / "prompts.jsonl")) in command
    assert shlex.quote(str(work_dir / "hf_predictions.json")) in command


def test_failed_sample_uses_recorded_trtmc_command_and_copies_media(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    work_dir = case_dir / "validation" / "workload-a" / "model-a"
    frames = work_dir / "reference_frames"
    frames.mkdir(parents=True)
    input_image = work_dir / "input.png"
    reference_image = frames / "000.png"
    reference_visualization = work_dir / "reference_visualization.png"
    trtmc_visualization = work_dir / "trtmc_visualization.png"
    trtmc_audio = work_dir / "output.wav"
    input_image.write_bytes(b"input-image")
    reference_image.write_bytes(b"reference-image")
    reference_visualization.write_bytes(b"reference-visualization")
    trtmc_visualization.write_bytes(b"trtmc-visualization")
    trtmc_audio.write_bytes(b"RIFFfake-wave")
    prompt = {
        "sample_id": "sample-9",
        "prompt": "Describe",
        "images": [str(input_image)],
    }
    (work_dir / "prompts.jsonl").write_text(
        json.dumps(prompt) + "\n",
        encoding="utf-8",
    )
    (work_dir / "answers.json").write_text(
        json.dumps({"requests": [{"sample_id": "sample-9", "answer": "A"}]}),
        encoding="utf-8",
    )
    (work_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "backend_mean_iou": 0.90,
                "gates": {"min_backend_mean_iou": 0.95},
                "cases": [
                    {
                        "sample_id": "sample-9",
                        "backend_mean_iou": 0.90,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "hf_predictions.json").write_text(
        json.dumps(
            {
                "responses": [
                    {
                        "sample_id": "sample-9",
                        "frames_dir": str(frames),
                        "visualization_path": str(reference_visualization),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "trtfb_predictions.json").write_text(
        json.dumps(
            {
                "responses": [
                    {
                        "sample_id": "sample-9",
                        "wav_path": str(trtmc_audio),
                        "visualization_path": str(trtmc_visualization),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    native_command = [
        "/workspace/build/trtmc",
        "run",
        "/runs/engines/model.trtfb",
        "--prompt",
        "Describe",
    ]
    (work_dir / "trtfb_native_commands.jsonl").write_text(
        json.dumps({"sample_id": "sample-9", "command": native_command}) + "\n",
        encoding="utf-8",
    )
    reference_command = [
        "/profiles/reference/bin/python",
        "/workspace/model/reference.py",
        "--input",
        str(input_image),
    ]
    (work_dir / "hf_native_commands.jsonl").write_text(
        json.dumps(
            {"sample_id": "sample-9", "command": reference_command}
        )
        + "\n",
        encoding="utf-8",
    )

    metadata = trtmc_disagreements.build_disagreement_artifact(
        work_dir=work_dir,
        case_dir=case_dir,
    )

    record = json.loads(
        (case_dir / metadata["path"]).read_text(encoding="utf-8")
    )
    assert record["reproduce"]["trtmc"] == (
        "/workspace/build/trtmc run /runs/engines/model.trtfb "
        "--prompt Describe"
    )
    assert record["reproduce"]["reference"].startswith(
        "/profiles/reference/bin/python /workspace/model/reference.py"
    )
    media = record["artifacts"]["media"]
    assert {item["kind"] for item in media} == {"image", "audio"}
    assert len(media) == 5
    assert {item["label"] for item in media} >= {
        "Reference visualization_path",
        "TRTMC visualization_path",
    }
    assert all((case_dir / item["path"]).is_file() for item in media)
    rendered = trtmc_validate._render_disagreement_record(
        record,
        asset_base=Path("model-a/workload-a"),
    )
    assert "<img " in rendered
    assert "<audio " in rendered
    assert "task_eval.py" not in rendered


def test_failed_encoder_pair_expands_to_both_reproducible_samples():
    rows = [
        {
            "pair_id": "sts-4",
            "passed": False,
            "cosine_abs_delta": 0.2,
        }
    ]
    prompts = {
        "sts-4-a": {
            "sample_id": "sts-4-a",
            "pair_id": "sts-4",
            "pair_side": "sentence1",
        },
        "sts-4-b": {
            "sample_id": "sts-4-b",
            "pair_id": "sts-4",
            "pair_side": "sentence2",
        },
    }

    expanded = trtmc_disagreements._expand_pair_disagreements(rows, prompts)

    assert [row["sample_id"] for row in expanded] == [
        "sts-4-a",
        "sts-4-b",
    ]


def test_summary_gate_failure_selects_worst_sample_for_reproduction():
    rows = trtmc_disagreements._summary_disagreements(
        {
            "status": "failed",
            "backend_mean_iou": 0.92,
            "gates": {"min_backend_mean_iou": 0.95},
            "cases": [
                {"sample_id": "sample-good", "backend_mean_iou": 0.97},
                {"sample_id": "sample-worst", "backend_mean_iou": 0.90},
            ],
        }
    )

    assert rows == [
        {
            "sample_id": "sample-worst",
            "backend_mean_iou": 0.90,
            "status": "failed",
            "reason": "summary_gate_failure",
            "failed_gates": [
                {
                    "gate": "min_backend_mean_iou",
                    "metric": "backend_mean_iou",
                    "actual": 0.92,
                    "threshold": 0.95,
                }
            ],
        }
    ]


def test_report_bounds_inline_failed_samples_but_keeps_full_artifact(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    work_dir = case_dir / "validation" / "workload-a" / "model-a"
    work_dir.mkdir(parents=True)
    sample_count = 25
    prompts = [
        {"sample_id": f"sample-{index}", "prompt": f"prompt-{index}"}
        for index in range(sample_count)
    ]
    (work_dir / "prompts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in prompts),
        encoding="utf-8",
    )
    (work_dir / "summary.json").write_text(
        json.dumps(
            {
                "disagreements": [
                    {"sample_id": row["sample_id"], "reason": "token_mismatch"}
                    for row in prompts
                ]
            }
        ),
        encoding="utf-8",
    )
    for name, prefix in (
        ("hf_predictions.json", "reference"),
        ("trtfb_predictions.json", "trtmc"),
    ):
        (work_dir / name).write_text(
            json.dumps(
                {
                    "responses": [
                        {
                            "sample_id": row["sample_id"],
                            "output_text": f"{prefix}-{index}",
                        }
                        for index, row in enumerate(prompts)
                    ]
                }
            ),
            encoding="utf-8",
        )
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "failed",
                "raw_result": {"status": "failed", "work_dir": str(work_dir)},
            }
        ),
        encoding="utf-8",
    )

    _, html_path, report = trtmc_validate.write_report(tmp_path)

    metadata = report["results"][0]["disagreements"]
    artifact = case_dir / metadata["path"]
    assert metadata["count"] == sample_count
    assert len(artifact.read_text(encoding="utf-8").splitlines()) == sample_count
    rendered = html_path.read_text(encoding="utf-8")
    assert "Showing 20 of 25" in rendered
    assert "sample-19" in rendered
    assert "sample-20" not in rendered
    assert "View all in disagreements.jsonl" in rendered
    assert (case_dir / "comparison.json").stat().st_size < 20_000
    assert (tmp_path / "report.json").stat().st_size < 20_000


def test_report_does_not_treat_shared_task_failure_as_disagreement(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    work_dir = case_dir / "validation" / "workload-a" / "model-a"
    work_dir.mkdir(parents=True)
    (work_dir / "prompts.jsonl").write_text(
        json.dumps({"sample_id": "sample-0", "prompt": "hello"}) + "\n",
        encoding="utf-8",
    )
    (work_dir / "trtfb_run.log").write_text(
        "$ trtmc run model.trtfb --prompt hello\n",
        encoding="utf-8",
    )
    (work_dir / "summary.json").write_text(
        json.dumps(
            {
                "disagreements": [],
                "hf": {"samples": [{"sample_id": "sample-0", "passed": False}]},
                "trtfb": {
                    "samples": [{"sample_id": "sample-0", "passed": False}]
                },
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "passed",
                "raw_result": {"work_dir": str(work_dir)},
            }
        ),
        encoding="utf-8",
    )

    _, _, report = trtmc_validate.write_report(tmp_path)

    assert report["results"][0]["reproduce"]["representative"] == {
        "sample_id": "sample-0",
        "reason": "first_input",
    }


def test_run_metadata_records_source_and_exact_command(monkeypatch, tmp_path):
    monkeypatch.setenv("TRTMC_VALIDATION_SOURCE_REVISION", "abc123")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(
        trtmc_validate.sys,
        "argv",
        ["tools/trtmc_validate.py", "model-a"],
    )

    path = trtmc_validate.write_run_metadata(tmp_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))

    assert metadata["source_revision"] == "abc123"
    assert metadata["cuda_visible_devices"] == "1"
    assert metadata["command"] == "tools/trtmc_validate.py model-a"


def test_comparison_command_uses_validation_entrypoint(tmp_path):
    arguments = argparse.Namespace(
        engine_dir=tmp_path / "engines",
        reference_cache_dir=tmp_path / "references",
        trtmc_binary=tmp_path / "trtmc",
        benchmark_binary=tmp_path / "trtmc_dataset_benchmark",
        limit=2,
        force_hf=True,
        force_build=False,
        no_build=True,
        local_files_only=True,
        backend_dir=None,
        model_plugin_dir=None,
        cuda_visible_devices="1",
    )

    command = trtmc_validate._comparison_command(
        trtmc_validate.Binding("model-a", "workload-a"),
        case_dir=tmp_path / "case",
        dataset=tmp_path / "dataset.json",
        arguments=arguments,
        reference_python="/profiles/python",
    )

    assert command[:2] == [
        "/profiles/python",
        str(trtmc_validate.REPO_ROOT / "tools" / "trtmc_compare.py"),
    ]
    assert "task_eval.py" not in " ".join(command)
    assert command[command.index("--work-root") + 1] == str(
        tmp_path / "case" / "validation"
    )
    assert command[command.index("--model") + 1] == "model-a"
    assert command[command.index("--suite") + 1] == "workload-a"
    assert command[command.index("--hf-python") + 1] == "/profiles/python"
    assert command[command.index("--reference-cache-dir") + 1] == str(
        tmp_path / "references"
    )
    assert "--replace-bundle-on-build" in command
    assert "--force-hf" in command
    assert "--require-prebuilt-bundles" in command
    assert "--local-files-only" in command


def test_comparison_command_passes_elf_reference_checkout(tmp_path):
    arguments = trtmc_validate.build_parser().parse_args([])
    arguments.engine_dir = tmp_path / "engines"
    arguments.reference_cache_dir = tmp_path / "references"
    arguments.trtmc_binary = tmp_path / "trtmc"
    arguments.benchmark_binary = tmp_path / "trtmc_dataset_benchmark"
    arguments.limit = 1
    reference_sources = trtmc_validate.ReferenceSourceSelection(
        environment={},
        elf_reference_repo=tmp_path / "sources" / "elf",
    )

    command = trtmc_validate._comparison_command(
        trtmc_validate.Binding("elf-b", "elf-workload"),
        case_dir=tmp_path / "case",
        dataset=tmp_path / "dataset.json",
        arguments=arguments,
        reference_python="/profiles/python",
        reference_sources=reference_sources,
    )

    assert command[command.index("--elf-reference-repo") + 1] == str(
        reference_sources.elf_reference_repo
    )


def test_run_binding_wires_reference_source_command_and_environment(
    tmp_path,
    monkeypatch,
):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "elf-b",
            "elf-workload",
            "--output",
            str(tmp_path / "results"),
            "--dataset",
            str(tmp_path / "dataset.json"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
        ]
    )
    selection = trtmc_validate.ReferenceSourceSelection(
        environment={
            "TRTMC_STORAGE_ROOT": str(tmp_path / "references"),
            "EXTERNAL_REFERENCE_SENTINEL": "present",
        },
        elf_reference_repo=tmp_path / "references" / "elf",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        trtmc_validate,
        "ensure_environments",
        lambda _profiles, _base: trtmc_validate.EnvironmentSelection(
            base_python="/profiles/python",
            names_and_paths=(),
            overrides={},
        ),
    )
    monkeypatch.setattr(
        trtmc_validate,
        "ensure_reference_sources",
        lambda _family, _cache: selection,
    )

    def run(command, _log_path, environment):
        captured["command"] = command
        captured["environment"] = environment
        return 0

    monkeypatch.setattr(trtmc_validate, "_run_subprocess", run)
    monkeypatch.setattr(
        trtmc_validate,
        "_comparison_result",
        lambda binding, **_kwargs: {
            "model": binding.model,
            "workload": binding.workload,
        },
    )

    trtmc_validate.run_binding(
        trtmc_validate.Binding("elf-b", "elf-workload"),
        arguments=arguments,
        task_models={
            "elf-b": {
                "family": "elf_flow",
                "runtime_strategy": "elf_flow",
                "reference_backend": "torch_reference",
                "execution_profiles": {},
            }
        },
        e2e_models={},
        suites={"elf-workload": {}},
    )

    command = captured["command"]
    assert command[command.index("--elf-reference-repo") + 1] == str(
        selection.elf_reference_repo
    )
    assert captured["environment"]["EXTERNAL_REFERENCE_SENTINEL"] == "present"


def test_compare_entrypoint_forwards_to_validation_backend(monkeypatch):
    captured = []

    def run(arguments):
        captured.extend(arguments)
        return 7

    monkeypatch.setattr(trtmc_compare.task_eval, "main", run)

    assert trtmc_compare.main(["--suite", "suite-a"]) == 7
    assert captured == ["eval", "--suite", "suite-a"]


@pytest.mark.parametrize(
    ("raw_result", "execution", "comparison", "validation"),
    [
        (
            {"status": "passed", "prediction_agreement_rate": 1.0},
            "completed",
            "agreement",
            "passed",
        ),
        (
            {"status": "failed", "prediction_agreement_rate": 0.5},
            "completed",
            "disagreement",
            "failed",
        ),
        (
            {
                "status": "failed",
                "prediction_agreement_rate": 0.5,
                "gate_failures": [
                    {
                        "gate": "min_prediction_agreement_rate",
                        "actual": 0.5,
                        "required": 0.98,
                    }
                ],
                "error_type": "BenchmarkGateError",
                "error": (
                    "min_prediction_agreement_rate "
                    "actual=0.5 required=0.98"
                ),
            },
            "completed",
            "disagreement",
            "failed",
        ),
        (
            {"status": "failed", "error": "runner crashed"},
            "error",
            "not_run",
            "failed",
        ),
    ],
)
def test_result_statuses_separate_execution_from_agreement(
    raw_result,
    execution,
    comparison,
    validation,
):
    result = trtmc_validate._normalize_result(
        {
            "model": "model-a",
            "workload": "workload-a",
            "status": raw_result["status"],
            "raw_result": raw_result,
        }
    )

    assert result["execution"]["status"] == execution
    assert result["comparison"]["status"] == comparison
    assert result["validation"]["status"] == validation


@pytest.mark.parametrize("returncode", [0, 1])
def test_comparison_result_marks_missing_summary_as_execution_error(
    tmp_path,
    returncode,
):
    result = trtmc_validate._comparison_result(
        trtmc_validate.Binding("model-a", "workload-a"),
        case_dir=tmp_path,
        returncode=returncode,
        reference_environment=trtmc_validate.EnvironmentSelection(
            base_python="/profiles/python",
            names_and_paths=(("reference_common", "/profiles/python"),),
            overrides={},
        ),
        dataset_command="python tools/trtmc_validate.py model-a",
    )

    assert result["execution"] == {
        "status": "error",
        "exit_code": returncode,
    }
    assert result["comparison"]["status"] == "not_run"
    assert result["validation"]["status"] == "failed"
    assert result["raw_result"]["error_type"] == "ComparisonProcessError"
    assert "without writing" in result["raw_result"]["error"]
