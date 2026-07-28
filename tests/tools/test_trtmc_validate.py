# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys

import pytest

from tools import task_eval
from tools import trtmc_compare
from tools import trtmc_disagreements
from tools import trtmc_reference
from tools import trtmc_validate


def _raw_evidence(
    model: str,
    workload: str,
    status: str,
    **extra,
) -> dict:
    evidence = {
        "model": model,
        "suite": workload,
        "status": status,
    }
    if status in {"pass", "passed"}:
        evidence.update(
            {
                "mode": "mcq",
                "prediction_agreement_rate": 1.0,
                "accuracy_drop_from_hf": 0.0,
                "valid_count": 1,
                "skipped_count": 0,
                "total_count": 1,
                "gates": {
                    "max_accuracy_drop_from_hf": 0.0,
                    "min_prediction_agreement": 1.0,
                },
            }
        )
    evidence.update(extra)
    mode = evidence.get("mode")
    if (
        status in {"pass", "passed"}
        and isinstance(mode, str)
        and trtmc_validate._PASSED_COMPLETE_COUNT_FIELD_BY_MODE.get(
            mode
        )
        == "sample_count"
    ):
        evidence.setdefault("sample_count", evidence.get("valid_count"))
        if "total_count" not in extra:
            evidence.pop("total_count", None)
        if "skipped_count" not in extra:
            evidence.pop("skipped_count", None)
    return evidence


def _test_gate_context(result_paths) -> dict:
    context = {}
    for path in result_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        workload = result.get("workload")
        if workload is None:
            continue
        raw_result = result.get("raw_result", {})
        gates = (
            raw_result.get("gates", {})
            if isinstance(raw_result, dict)
            else {}
        )
        assert isinstance(gates, dict)
        context[(result["model"], workload)] = dict(gates)
    return context


def _assert_output_lock_held(output: Path) -> None:
    descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(BlockingIOError):
            trtmc_validate.fcntl.flock(
                descriptor,
                trtmc_validate.fcntl.LOCK_EX
                | trtmc_validate.fcntl.LOCK_NB,
            )
    finally:
        os.close(descriptor)


def _assert_output_lock_available(output: Path) -> None:
    descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        trtmc_validate.fcntl.flock(
            descriptor,
            trtmc_validate.fcntl.LOCK_EX
            | trtmc_validate.fcntl.LOCK_NB,
        )
        trtmc_validate.fcntl.flock(
            descriptor,
            trtmc_validate.fcntl.LOCK_UN,
        )
    finally:
        os.close(descriptor)


def _canonical_disagreement_record(
    sample_id: str,
    *,
    artifacts: dict | None = None,
) -> dict:
    return {
        "schema_version": trtmc_disagreements.SCHEMA_VERSION,
        "sample_id": sample_id,
        "reason": "comparison_threshold",
        "input": {},
        "reference_result": {},
        "trtmc_result": {},
        "comparison": {},
        "reproduce": {},
        "artifacts": artifacts or {},
    }


def test_model_workload_catalog_covers_every_validation_eligible_model():
    catalog = trtmc_validate.load_catalog()
    suites = task_eval.load_suites()
    task_models = trtmc_validate._task_eval_models(trtmc_validate.DEFAULT_MODELS)
    eligible_models = trtmc_validate.ready_model_names()

    trtmc_validate.audit_catalog(
        catalog,
        ready_models=eligible_models,
        suite_names=(suite["id"] for suite in suites),
    )
    trtmc_validate.audit_workload_compatibility(
        catalog,
        suites={suite["id"]: suite for suite in suites},
        task_models=task_models,
    )

    assert len(catalog["models"]) == len(eligible_models) == 105
    assert sum(
        "not_compared_reason" in spec for spec in catalog["models"].values()
    ) == 14
    assert all(
        "e2e" not in spec.get("workloads", [])
        for spec in catalog["models"].values()
    )
    assert (
        catalog["models"]["flux-2-dev"]["reference_cache_identity"]
        == catalog["models"]["flux-2-dev-fp8"]["reference_cache_identity"]
    )
    qwen_identities = {
        catalog["models"][name]["reference_cache_identity"]
        for name in (
            "qwen3-0.6b-fp16",
            "qwen3-0.6b-fp8",
            "qwen3-0.6b-topp",
        )
    }
    assert len(qwen_identities) == 1


def test_authoritative_gates_use_resolved_cli_suite_and_model_context():
    catalog = {
        "models": {
            "model-a": {
                "workloads": ["suite-a"],
            }
        }
    }
    suites = {
        "suite-a": {
            "id": "suite-a",
            "gates": {
                "min_agreement": 0.5,
                "max_error": 0.4,
            },
            "family_profiles": {
                "family-a": {
                    "gates": {
                        "min_agreement": 0.75,
                    }
                }
            },
            "model_profiles": {
                "model-a": {
                    "gates": {
                        "max_error": 0.1,
                    }
                }
            },
        }
    }
    task_models = {
        "model-a": {
            "name": "model-a",
            "family": "family-a",
        }
    }

    assert trtmc_validate._authoritative_gates_by_binding(
        catalog,
        suites=suites,
        task_models=task_models,
    ) == {
        ("model-a", "suite-a"): {
            "min_agreement": 0.75,
            "max_error": 0.1,
        }
    }


def test_complete_count_policy_covers_every_supported_noncontinuation_mode():
    assert set(
        trtmc_validate._PASSED_COMPLETE_COUNT_FIELD_BY_MODE
    ) == (
        set(trtmc_validate._PRIMARY_METRIC_BY_MODE)
        - {"continuation"}
    )


def test_gate_configuration_rejects_unrepresentably_large_integer():
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="must be a finite number",
    ):
        trtmc_validate._validated_gate_configuration(
            {"min_score": 10**10_000},
            field="expected_gates",
        )


def test_validation_eligible_models_apply_all_selection_boundaries():
    records = task_eval.load_manifest_records(trtmc_validate.DEFAULT_MODELS)
    eligible = {
        str(record["name"])
        for record in records
        if not record["requires_multi_device"] and not record.get("skip")
    }
    l0_only = {str(record["name"]) for record in records if record.get("ci_tier") == "l0_only"}
    selected = set(trtmc_validate.ready_model_names())

    assert l0_only
    assert selected == eligible - l0_only


def test_all_help_and_docs_name_validation_eligibility_boundaries():
    help_text = " ".join(trtmc_validate.build_parser().format_help().split())
    assert "run every validation-eligible ready single-device non-l0-only model" in help_text

    repo_root = Path(__file__).resolve().parents[2]
    for relative_path in (
        "website/docs/reference/testing.md",
        "tests/validation/README.md",
    ):
        text = " ".join((repo_root / relative_path).read_text(encoding="utf-8").split())
        assert "validation-eligible ready single-device model" in text
        assert "`ci_tier: l0_only`" in text
        assert "readiness alone does not select a model" in text
    website_text = " ".join(
        (repo_root / "website/docs/reference/testing.md")
        .read_text(encoding="utf-8")
        .split()
    )
    assert (
        "91 use dataset-backed, threshold-gated reference workloads"
        in website_text
    )
    assert "14 are explicitly marked `not_compared_reason`" in website_text
    assert "explicit E2E fallback" not in website_text
    assert "E2E bindings have no dataset slice" not in website_text


def test_catalog_defines_sample_limit_for_every_dataset_workload():
    catalog = trtmc_validate.load_catalog()
    configured = set(catalog["sample_limits"])
    declared = {
        workload
        for spec in catalog["models"].values()
        for workload in spec.get("workloads", [])
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
        for workload in spec.get("workloads", [])
    ]
    missing = []
    for model_name, workload in bindings:
        dataset_kind = str(suites[workload]["dataset"]["kind"])
        if trtmc_reference.native_reference_runner_for_dataset_kind(dataset_kind) is None:
            missing.append((model_name, workload, dataset_kind))

    assert not missing
    assert len({model for model, _workload in bindings}) == 91


def test_resolve_binding_defaults_and_rejects_undeclared_workload():
    catalog = {
        "models": {
            "model-a": {
                "default": "workload-a",
                "workloads": ["workload-a", "workload-b"],
                "reference_cache_identity": "org/model/reference-contract-v1",
            }
        }
    }

    assert trtmc_validate.resolve_binding(catalog, "model-a") == (
        trtmc_validate.Binding(
            "model-a",
            "workload-a",
            reference_cache_identity="org/model/reference-contract-v1",
        )
    )
    assert trtmc_validate.resolve_binding(catalog, "model-a", "workload-b") == (
        trtmc_validate.Binding(
            "model-a",
            "workload-b",
            reference_cache_identity="org/model/reference-contract-v1",
        )
    )
    with pytest.raises(trtmc_validate.ValidationError, match="does not declare"):
        trtmc_validate.resolve_binding(catalog, "model-a", "workload-c")


def test_resolve_binding_keeps_unimplemented_model_visible_but_not_runnable():
    catalog = {
        "models": {
            "model-a": {
                "not_compared_reason": "Reference comparator is missing.",
            }
        }
    }

    binding = trtmc_validate.resolve_binding(catalog, "model-a")

    assert binding == trtmc_validate.Binding(
        "model-a",
        None,
        "Reference comparator is missing.",
    )
    assert not binding.runnable
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="has no reference-consistency workloads",
    ):
        trtmc_validate.resolve_binding(catalog, "model-a", "workload-a")


def test_catalog_rejects_e2e_as_reference_consistency_workload(tmp_path):
    catalog_path = tmp_path / "model_workloads.yaml"
    catalog_path.write_text(
        """
version: 1
sample_limits:
  workload-a: 1
models:
  model-a:
    default: e2e
    workloads: [e2e]
""",
        encoding="utf-8",
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="cannot use e2e",
    ):
        trtmc_validate.load_catalog(catalog_path)


def test_catalog_rejects_cache_identity_across_different_reference_contracts(
    monkeypatch,
) -> None:
    catalog = {
        "models": {
            "model-a": {
                "default": "workload-a",
                "workloads": ["workload-a"],
                "reference_cache_identity": "shared-reference",
            },
            "model-b": {
                "default": "workload-a",
                "workloads": ["workload-a"],
                "reference_cache_identity": "shared-reference",
            },
        }
    }
    task_models = {
        "model-a": {
            "hf_id": "org/model-a",
            "family": "family",
            "reference_backend": "hf_transformers",
            "reference_family": "causal",
        },
        "model-b": {
            "hf_id": "org/model-b",
            "family": "family",
            "reference_backend": "hf_transformers",
            "reference_family": "causal",
        },
    }
    monkeypatch.setattr(
        trtmc_validate.task_eval,
        "suite_match_reason",
        lambda _suite, _model: (True, ""),
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="spans different reference contracts",
    ):
        trtmc_validate.audit_workload_compatibility(
            catalog,
            suites={"workload-a": {}},
            task_models=task_models,
        )


def test_resolve_sample_limit_uses_workload_policy_and_cli_override():
    catalog = {
        "sample_limits": {"workload-a": 50},
        "models": {
            "model-a": {
                "default": "workload-a",
                "workloads": ["workload-a"],
            },
            "model-not-compared": {
                "not_compared_reason": "Reference comparator is missing.",
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
            trtmc_validate.Binding(
                "model-not-compared",
                None,
                "Reference comparator is missing.",
            ),
            7,
        )
        == 0
    )
    assert (
        trtmc_validate.resolve_sample_limit(
            catalog,
            trtmc_validate.Binding(
                "model-not-compared",
                None,
                "Reference comparator is missing.",
            ),
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
    assert default.model_attempts == 2
    assert default.model_retry_delay_seconds == 5.0


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_model_retry_delay_rejects_nonfinite_values(value):
    parser = trtmc_validate.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--all", "--model-retry-delay-seconds", value]
        )


def test_model_retry_delay_accepts_finite_zero():
    arguments = trtmc_validate.build_parser().parse_args(
        ["--all", "--model-retry-delay-seconds", "0"]
    )

    assert arguments.model_retry_delay_seconds == 0.0


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
            "execution": {"status": "completed"},
            "validation": {"status": status},
        }

    monkeypatch.setattr(
        trtmc_validate,
        "_run_supervised_binding_with_retries",
        run_worker,
    )
    monkeypatch.setattr(trtmc_validate, "write_run_metadata", lambda output: output)
    monkeypatch.setattr(
        trtmc_validate,
        "finalize_run_metadata",
        lambda output: output,
    )
    monkeypatch.setattr(
        trtmc_validate,
        "write_report",
        lambda output, **_kwargs: (
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


def test_all_supervisor_propagates_authoritative_gates_to_worker_and_report(
    tmp_path,
    monkeypatch,
):
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
    expected_gates = {"min_score": 0.75}
    gates_by_binding = {
        ("model-a", "workload-a"): expected_gates,
    }
    worker_contexts = []
    report_contexts = []

    def run_worker(
        selected,
        *,
        arguments,
        catalog,
        expected_gates,
    ):
        del arguments, catalog
        worker_contexts.append((selected, expected_gates))
        return {
            "model": selected.model,
            "workload": selected.workload,
            "validation": {"status": "passed"},
        }

    def write_report(_output, **kwargs):
        report_contexts.append(
            kwargs.get("expected_gates_by_binding")
        )
        return (
            tmp_path / "report.json",
            tmp_path / "report.html",
            {},
        )

    monkeypatch.setattr(
        trtmc_validate,
        "_run_supervised_binding_with_retries",
        run_worker,
    )
    monkeypatch.setattr(
        trtmc_validate,
        "write_report",
        write_report,
    )
    monkeypatch.setattr(
        trtmc_validate,
        "write_run_metadata",
        lambda output: output,
    )
    monkeypatch.setattr(
        trtmc_validate,
        "finalize_run_metadata",
        lambda output: output,
    )
    monkeypatch.setattr(
        trtmc_validate,
        "_print_result",
        lambda *_args: None,
    )

    returncode = trtmc_validate._run_all_bindings(
        [binding],
        arguments=arguments,
        catalog={"sample_limits": {"workload-a": 1}},
        expected_gates_by_binding=gates_by_binding,
    )

    assert returncode == 0
    assert worker_contexts == [(binding, expected_gates)]
    assert report_contexts
    assert all(
        context is gates_by_binding
        for context in report_contexts
    )


def test_all_supervisor_records_unexpected_run_failure(
    tmp_path,
    monkeypatch,
):
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

    def fail_worker(*_args, **_kwargs):
        raise RuntimeError("UNIQUE-EARLY-ERROR")

    monkeypatch.setattr(
        trtmc_validate,
        "_run_supervised_binding_with_retries",
        fail_worker,
    )
    monkeypatch.setattr(
        trtmc_validate,
        "_source_revision",
        lambda: "test-revision",
    )

    with pytest.raises(RuntimeError, match="UNIQUE-EARLY-ERROR"):
        trtmc_validate._run_all_bindings(
            [binding],
            arguments=arguments,
            catalog={"sample_limits": {"workload-a": 1}},
        )

    run = json.loads(
        (arguments.output / "run.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (arguments.output / "report.json").read_text(encoding="utf-8")
    )
    assert run["status"] == "failed"
    assert run["finished_at"]
    assert "UNIQUE-EARLY-ERROR" in run["error"]
    assert report["run"]["error"] == run["error"]
    assert report["summary"]["cases"] == 0
    assert "UNIQUE-EARLY-ERROR" in (
        arguments.output / "report.html"
    ).read_text(encoding="utf-8")


def test_supervisor_retries_execution_error_but_not_disagreement(
    tmp_path,
    monkeypatch,
):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--model-retry-delay-seconds",
            "0",
            "--output",
            str(tmp_path / "results"),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
        ]
    )
    binding = trtmc_validate.Binding("model-a", "workload-a")
    attempts = []

    def run_worker(binding, *, arguments, catalog, attempt):
        attempts.append(attempt)
        execution_status = "error" if attempt == 1 else "completed"
        validation_status = "failed" if attempt == 1 else "passed"
        result = {
            "model": binding.model,
            "workload": binding.workload,
            "execution": {"status": execution_status, "exit_code": 1 if attempt == 1 else 0},
            "validation": {"status": validation_status},
            "raw_result": {
                "status": validation_status,
                "error_type": "WorkerProcessError" if attempt == 1 else "",
            },
            "worker_log": str(tmp_path / f"worker-{attempt}.log"),
        }
        case_dir = trtmc_validate._case_directory(arguments.output, binding)
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "comparison.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(trtmc_validate, "_run_supervised_binding", run_worker)

    result = trtmc_validate._run_supervised_binding_with_retries(
        binding,
        arguments=arguments,
        catalog={"sample_limits": {"workload-a": 5}},
    )

    assert attempts == [1, 2]
    assert result["execution"]["status"] == "completed"
    assert result["execution"]["attempt_count"] == 2
    assert result["execution"]["retry_count"] == 1

    attempts.clear()

    def disagree(binding, *, arguments, catalog, attempt):
        attempts.append(attempt)
        return {
            "model": binding.model,
            "workload": binding.workload,
            "execution": {"status": "completed", "exit_code": 1},
            "validation": {"status": "failed"},
            "raw_result": {"status": "failed"},
            "worker_log": str(tmp_path / "worker-disagreement.log"),
        }

    monkeypatch.setattr(trtmc_validate, "_run_supervised_binding", disagree)

    disagreement = trtmc_validate._run_supervised_binding_with_retries(
        binding,
        arguments=arguments,
        catalog={"sample_limits": {"workload-a": 5}},
    )

    assert attempts == [1]
    assert disagreement["execution"]["status"] == "completed"
    assert disagreement["execution"]["attempt_count"] == 1


def test_all_supervisor_records_not_compared_without_launching_worker(
    tmp_path,
    monkeypatch,
):
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
    binding = trtmc_validate.Binding(
        "model-a",
        None,
        "Reference comparator is missing.",
    )

    def unexpected_worker(*_args, **_kwargs):
        raise AssertionError("not-compared models must not launch a worker")

    monkeypatch.setattr(
        trtmc_validate,
        "_run_supervised_binding_with_retries",
        unexpected_worker,
    )
    monkeypatch.setattr(trtmc_validate, "write_run_metadata", lambda output: output)
    monkeypatch.setattr(
        trtmc_validate,
        "finalize_run_metadata",
        lambda output: output,
    )
    monkeypatch.setattr(
        trtmc_validate,
        "write_report",
        lambda output, **_kwargs: (
            output / "report.json",
            output / "report.html",
            {},
        ),
    )
    monkeypatch.setattr(trtmc_validate, "_print_result", lambda *args: None)

    returncode = trtmc_validate._run_all_bindings(
        [binding],
        arguments=arguments,
        catalog={"sample_limits": {}},
    )

    comparison = (
        arguments.output
        / "model-a"
        / trtmc_validate.NOT_COMPARED_DIRECTORY
        / "comparison.json"
    )
    result = json.loads(comparison.read_text(encoding="utf-8"))
    assert returncode == 0
    assert result["execution"]["status"] == "not_run"
    assert result["comparison"]["status"] == "not_run"
    assert result["validation"]["status"] == "not_compared"
    assert result["not_compared_reason"] == "Reference comparator is missing."


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


def test_supervised_binding_rejects_stale_result_swapped_after_read(
    tmp_path,
    monkeypatch,
):
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
    case_dir = arguments.output / binding.model / binding.workload
    case_dir.mkdir(parents=True)
    comparison = case_dir / "comparison.json"
    stale = {
        "model": binding.model,
        "workload": binding.workload,
        "returncode": 0,
        "raw_result": {
            "model": binding.model,
            "suite": binding.workload,
            "status": "passed",
        },
    }
    comparison.write_text(json.dumps(stale), encoding="utf-8")
    replacement = case_dir / "replacement.json"
    replacement.write_text(json.dumps(stale), encoding="utf-8")
    real_read = trtmc_validate._read_report_result
    swapped = False

    def swap_after_read(output, path, **kwargs):
        nonlocal swapped
        loaded = real_read(output, path, **kwargs)
        if not swapped:
            os.replace(replacement, comparison)
            swapped = True
        return loaded

    monkeypatch.setattr(
        trtmc_validate,
        "_run_subprocess",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr(
        trtmc_validate,
        "_read_report_result",
        swap_after_read,
    )

    result = trtmc_validate._run_supervised_binding(
        binding,
        arguments=arguments,
        catalog={"sample_limits": {"workload-a": 5}},
    )

    assert swapped
    assert result["execution"] == {"status": "error", "exit_code": 0}
    assert result["validation"]["status"] == "failed"
    assert result["raw_result"]["error_type"] == "WorkerProcessError"
    assert "changed after it was read" in result["raw_result"]["error"]


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
    comparison = arguments.output / binding.model / binding.workload / "comparison.json"

    def pass_worker(command, log_path, env):
        comparison.write_text(
            json.dumps(
                {
                    "model": binding.model,
                    "workload": binding.workload,
                    "status": "passed",
                    "returncode": 0,
                    "raw_result": _raw_evidence(
                        binding.model,
                        binding.workload,
                        "passed",
                    ),
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


def test_supervised_binding_rejects_passing_result_from_crashed_worker(
    tmp_path,
    monkeypatch,
):
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
    comparison = (
        arguments.output
        / binding.model
        / binding.workload
        / "comparison.json"
    )

    def crash_after_pass(_command, _log_path, _env):
        comparison.write_text(
            json.dumps(
                {
                    "model": binding.model,
                    "workload": binding.workload,
                    "returncode": 0,
                    "raw_result": _raw_evidence(
                        binding.model,
                        binding.workload,
                        "passed",
                    ),
                }
            ),
            encoding="utf-8",
        )
        return 9

    monkeypatch.setattr(
        trtmc_validate,
        "_run_subprocess",
        crash_after_pass,
    )

    result = trtmc_validate._run_supervised_binding(
        binding,
        arguments=arguments,
        catalog={"sample_limits": {"workload-a": 5}},
    )

    assert result["execution"] == {"status": "error", "exit_code": 9}
    assert result["comparison"]["status"] == "not_run"
    assert result["validation"]["status"] == "failed"
    assert result["raw_result"]["error_type"] == "WorkerProcessError"
    assert "requires exit code 0" in result["raw_result"]["error"]


def test_supervised_binding_accepts_worker_disagreement_exit_code(
    tmp_path,
    monkeypatch,
):
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
    comparison = (
        arguments.output
        / binding.model
        / binding.workload
        / "comparison.json"
    )

    def disagree(_command, _log_path, _env):
        comparison.write_text(
            json.dumps(
                {
                    "model": binding.model,
                    "workload": binding.workload,
                    "returncode": 1,
                    "raw_result": {
                        "model": binding.model,
                        "suite": binding.workload,
                        "status": "failed",
                    },
                }
            ),
            encoding="utf-8",
        )
        return 1

    monkeypatch.setattr(
        trtmc_validate,
        "_run_subprocess",
        disagree,
    )

    result = trtmc_validate._run_supervised_binding(
        binding,
        arguments=arguments,
        catalog={"sample_limits": {"workload-a": 5}},
    )

    assert result["execution"] == {"status": "completed", "exit_code": 1}
    assert result["comparison"]["status"] == "disagreement"
    assert result["validation"]["status"] == "failed"
    assert result["raw_result"].get("error_type") is None


def test_supervised_binding_rejects_not_compared_runnable_result(
    tmp_path,
    monkeypatch,
):
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
    comparison = (
        arguments.output
        / binding.model
        / binding.workload
        / "comparison.json"
    )

    def not_compared(_command, _log_path, _env):
        comparison.write_text(
            json.dumps(
                {
                    "model": binding.model,
                    "workload": binding.workload,
                    "not_compared_reason": "unexpected",
                    "execution": {
                        "status": "not_run",
                        "exit_code": None,
                    },
                    "comparison": {"status": "not_run"},
                    "validation": {"status": "not_compared"},
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(
        trtmc_validate,
        "_run_subprocess",
        not_compared,
    )

    result = trtmc_validate._run_supervised_binding(
        binding,
        arguments=arguments,
        catalog={"sample_limits": {"workload-a": 5}},
    )

    assert result["execution"] == {"status": "error", "exit_code": 0}
    assert result["validation"]["status"] == "failed"
    assert "runnable binding" in result["raw_result"]["error"]


@pytest.mark.parametrize(
    "payload",
    [
        {
            "model": "model-a",
            "workload": "workload-a",
            "execution": {"status": "completed", "exit_code": 0},
            "comparison": {"status": "agreement"},
            "validation": {"status": "passed"},
        },
        {
            "model": "model-a",
            "workload": "workload-a",
            "raw_result": {
                "model": "model-a",
                "status": "skipped",
            },
        },
        {
            "model": "model-a",
            "workload": "workload-a",
            "raw_result": {
                "model": "model-a",
                "status": "passed",
            },
            "execution": {"status": "completed", "exit_code": None},
        },
        {
            "model": "model-a",
            "workload": "workload-a",
            "returncode": 0,
            "raw_result": {
                "model": "model-a",
                "status": "passed",
                "prediction_agreement_rate": 0.25,
            },
            "comparison": {
                "status": "agreement",
                "metrics": {"prediction_agreement_rate": 1.0},
                "primary_metric": {
                    "name": "prediction_agreement_rate",
                    "value": 1.0,
                },
            },
        },
        {
            "model": "model-a",
            "workload": "workload-a",
            "returncode": 0,
            "raw_result": {
                "model": "model-b",
                "status": "passed",
            },
        },
        {
            "model": "model-a",
            "workload": "workload-a",
            "status": "passed",
            "returncode": 0,
            "raw_result": {"mode": "fabricated"},
        },
        {
            "model": "model-a",
            "workload": "workload-a",
            "returncode": 0,
            "raw_result": {"status": "passed"},
        },
        {
            "model": "model-a",
            "workload": "workload-a",
            "returncode": 0,
            "raw_result": {
                "model": [],
                "suite": "workload-a",
                "status": "passed",
            },
        },
    ],
    ids=[
        "canonical-only",
        "skipped",
        "missing-exit-code",
        "conflicting-metrics",
        "raw-binding-mismatch",
        "missing-raw-status",
        "missing-raw-binding",
        "malformed-raw-binding",
    ],
)
def test_supervised_binding_rejects_unverifiable_worker_evidence(
    tmp_path,
    monkeypatch,
    payload,
):
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
    comparison = (
        arguments.output
        / binding.model
        / binding.workload
        / "comparison.json"
    )

    def write_unverifiable_result(_command, _log_path, _env):
        comparison.write_text(json.dumps(payload), encoding="utf-8")
        return 0

    monkeypatch.setattr(
        trtmc_validate,
        "_run_subprocess",
        write_unverifiable_result,
    )

    result = trtmc_validate._run_supervised_binding(
        binding,
        arguments=arguments,
        catalog={"sample_limits": {"workload-a": 5}},
    )

    assert result["execution"] == {"status": "error", "exit_code": 0}
    assert result["comparison"]["status"] == "not_run"
    assert result["validation"]["status"] == "failed"
    assert result["raw_result"]["error_type"] == "WorkerProcessError"


def test_all_dry_run_emits_machine_readable_ci_cases(monkeypatch, capsys):
    catalog = {
        "sample_limits": {"workload-a": 5},
        "models": {
            "model-a": {
                "default": "workload-a",
                "workloads": ["workload-a"],
            },
            "model-not-compared": {
                "not_compared_reason": "Reference comparator is missing.",
            },
        },
    }
    monkeypatch.setattr(
        trtmc_validate,
        "_load_validation_inputs",
        lambda arguments: (
            catalog,
            {"workload-a": {}},
            ("model-a", "model-not-compared"),
            {},
        ),
    )

    returncode = trtmc_validate.main(["--all", "--dry-run", "--limit", "7"])

    assert returncode == 0
    assert json.loads(capsys.readouterr().out) == [
        {
            "model": "model-a",
            "workload": "workload-a",
            "sample_limit": 7,
        },
        {
            "model": "model-not-compared",
            "workload": None,
            "sample_limit": 0,
            "status": "not_compared",
            "reason": "Reference comparator is missing.",
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

    def run_binding(binding, *, arguments, task_models, suites):
        raw_result = (
            _raw_evidence(
                binding.model,
                binding.workload,
                "passed",
            )
            if validation_status == "passed"
            else {
                "model": binding.model,
                "suite": binding.workload,
                "mode": "mcq",
                "status": "failed",
                "prediction_agreement_rate": 0.5,
                "gate_failures": [
                    {
                        "gate": "min_prediction_agreement_rate",
                        "metric": "prediction_agreement_rate",
                        "actual": 0.5,
                        "required": 0.98,
                    }
                ],
                "error_type": "BenchmarkGateError",
                "error": "comparison gate failed",
            }
        )
        result = trtmc_validate._normalize_result(
            {
                "schema_version": "trtmc.validation-result/v2",
                "model": binding.model,
                "workload": binding.workload,
                "returncode": 0,
                "raw_result": raw_result,
                "reproduce": {
                    "dataset": {
                        "command": (
                            "python tools/trtmc_validate.py "
                            "model-a workload-a"
                        ),
                        "sample_limit": arguments.limit,
                        "prepared_input_count": arguments.limit,
                    },
                    "hf": [],
                    "trtmc": [],
                },
            }
        )
        case_dir = arguments.output / binding.model / binding.workload
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "comparison.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(trtmc_validate, "run_binding", run_binding)

    returncode = trtmc_validate._run_bindings(
        [binding],
        arguments=arguments,
        catalog=catalog,
        task_models={},
        suites={"workload-a": {}},
        expected_gates_by_binding={
            ("model-a", "workload-a"): _raw_evidence(
                "model-a",
                "workload-a",
                "passed",
            )["gates"],
        },
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
    assert trtmc_validate.ELF_SOURCE.revision == ("b29d8833609e9ab7f67cd9da39435ac5cea04837")
    assert trtmc_validate.ELF_SOURCE.relative_checkout == Path("elf/reference/ELF-b29d8833609e")


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
    assert elf.elf_reference_repo == tmp_path / trtmc_validate.ELF_SOURCE.relative_checkout
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
                    "returncode": 0,
                    "raw_result": _raw_evidence(
                        "model-a",
                        "workload-a",
                        "passed",
                    ),
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

    json_path, html_path, report = trtmc_validate.write_report(
        tmp_path,
        expected_gates_by_binding=_test_gate_context(
            [comparison]
        ),
    )

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
    assert "🟢 1 &nbsp; 🟡 0 &nbsp;" in document
    assert "🔴 0 &nbsp; ⚪ 0" in document
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


def test_write_report_holds_output_lock_for_entire_publication(
    tmp_path,
    monkeypatch,
):
    locked = False
    operations = []
    expected = (
        tmp_path / "report.json",
        tmp_path / "report.html",
        {"validation_status": "passed"},
    )

    def record_lock(_descriptor, operation):
        nonlocal locked
        operations.append(operation)
        if operation == trtmc_validate.fcntl.LOCK_EX:
            assert not locked
            locked = True
        else:
            assert operation == trtmc_validate.fcntl.LOCK_UN
            assert locked
            locked = False

    def publish_while_locked(output, *, result_paths):
        assert locked
        assert output == tmp_path
        assert result_paths == []
        return expected

    monkeypatch.setattr(trtmc_validate.fcntl, "flock", record_lock)
    monkeypatch.setattr(
        trtmc_validate,
        "_write_report_locked",
        publish_while_locked,
    )

    assert trtmc_validate.write_report(
        tmp_path,
        result_paths=[],
    ) == expected
    assert not locked
    assert operations == [
        trtmc_validate.fcntl.LOCK_EX,
        trtmc_validate.fcntl.LOCK_UN,
    ]


def test_write_report_releases_output_lock_after_failure(
    tmp_path,
    monkeypatch,
):
    locked = False

    def record_lock(_descriptor, operation):
        nonlocal locked
        if operation == trtmc_validate.fcntl.LOCK_EX:
            locked = True
        else:
            assert operation == trtmc_validate.fcntl.LOCK_UN
            assert locked
            locked = False

    def fail_while_locked(_output, *, result_paths):
        assert locked
        assert result_paths == []
        raise OSError("injected publication failure")

    monkeypatch.setattr(trtmc_validate.fcntl, "flock", record_lock)
    monkeypatch.setattr(
        trtmc_validate,
        "_write_report_locked",
        fail_while_locked,
    )

    with pytest.raises(OSError, match="injected publication failure"):
        trtmc_validate.write_report(
            tmp_path,
            result_paths=[],
        )
    assert not locked


def test_write_report_creates_missing_output_root(
    tmp_path,
):
    output = tmp_path / "new-output"

    json_path, html_path, report = trtmc_validate.write_report(
        output,
        result_paths=[],
    )

    assert output.is_dir()
    assert json_path == output / "report.json"
    assert html_path == output / "report.html"
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    assert html_path.is_file()
    assert report["validation_status"] == "failed"


def test_output_publication_lock_preserves_body_error_when_unlock_fails(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "output"
    output.mkdir()
    real_flock = trtmc_validate.fcntl.flock

    def fail_unlock(descriptor, operation):
        if operation == trtmc_validate.fcntl.LOCK_UN:
            raise OSError("injected unlock failure")
        return real_flock(descriptor, operation)

    monkeypatch.setattr(
        trtmc_validate.fcntl,
        "flock",
        fail_unlock,
    )

    with pytest.raises(RuntimeError, match="business failure") as raised:
        with trtmc_validate._validation_output_publication_lock(output):
            raise RuntimeError("business failure")

    assert any(
        "injected unlock failure" in note
        for note in getattr(raised.value, "__notes__", [])
    )
    descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        real_flock(
            descriptor,
            trtmc_validate.fcntl.LOCK_EX
            | trtmc_validate.fcntl.LOCK_NB,
        )
        real_flock(descriptor, trtmc_validate.fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def test_output_publication_lock_serializes_independent_processes(
    tmp_path,
):
    marker = tmp_path / "child-entered"
    script = (
        "import fcntl, os, pathlib, sys\n"
        "descriptor = os.open(sys.argv[1], os.O_RDONLY | os.O_DIRECTORY)\n"
        "print('ready', flush=True)\n"
        "fcntl.flock(descriptor, fcntl.LOCK_EX)\n"
        "pathlib.Path(sys.argv[2]).write_text('entered', encoding='utf-8')\n"
        "fcntl.flock(descriptor, fcntl.LOCK_UN)\n"
        "os.close(descriptor)\n"
    )
    process = None
    with trtmc_validate._validation_output_publication_lock(
        tmp_path
    ):
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(tmp_path),
                str(marker),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        assert not marker.exists()

    assert process is not None
    _stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, stderr
    assert marker.read_text(encoding="utf-8") == "entered"


def test_output_publication_lock_rejects_replaced_visible_root(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "output"
    output.mkdir()
    moved = tmp_path / "output-before-lock"
    real_flock = trtmc_validate.fcntl.flock
    replaced = False

    def replace_after_lock(descriptor, operation):
        nonlocal replaced
        real_flock(descriptor, operation)
        if (
            operation == trtmc_validate.fcntl.LOCK_EX
            and not replaced
        ):
            output.rename(moved)
            output.mkdir()
            replaced = True

    monkeypatch.setattr(
        trtmc_validate.fcntl,
        "flock",
        replace_after_lock,
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="output changed while acquiring",
    ):
        trtmc_validate.write_report(
            output,
            result_paths=[],
        )

    assert replaced
    assert not (output / "report.json").exists()
    assert not (moved / "report.json").exists()


def test_report_lock_covers_all_successful_publication_phases(
    tmp_path,
    monkeypatch,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    comparison = case_dir / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "returncode": 0,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "passed",
                ),
            }
        ),
        encoding="utf-8",
    )
    phases = set()
    real_preflight = trtmc_validate._preflight_report_disagreements
    real_commit = trtmc_validate._commit_file_update
    real_verify = trtmc_validate._verify_report_transaction_visibility
    real_finalize = trtmc_validate._finalize_file_update

    def check_preflight(*args, **kwargs):
        _assert_output_lock_held(tmp_path)
        phases.add("preflight")
        return real_preflight(*args, **kwargs)

    def check_commit(update):
        _assert_output_lock_held(tmp_path)
        phases.add("commit")
        return real_commit(update)

    def check_visibility(entries):
        _assert_output_lock_held(tmp_path)
        phases.add("visibility")
        return real_verify(entries)

    def check_finalize(update):
        _assert_output_lock_held(tmp_path)
        phases.add("finalize")
        return real_finalize(update)

    monkeypatch.setattr(
        trtmc_validate,
        "_preflight_report_disagreements",
        check_preflight,
    )
    monkeypatch.setattr(
        trtmc_validate,
        "_commit_file_update",
        check_commit,
    )
    monkeypatch.setattr(
        trtmc_validate,
        "_verify_report_transaction_visibility",
        check_visibility,
    )
    monkeypatch.setattr(
        trtmc_validate,
        "_finalize_file_update",
        check_finalize,
    )

    trtmc_validate.write_report(
        tmp_path,
        result_paths=[comparison],
        expected_gates_by_binding=_test_gate_context(
            [comparison]
        ),
    )

    assert phases == {
        "preflight",
        "commit",
        "visibility",
        "finalize",
    }
    _assert_output_lock_available(tmp_path)


def test_report_lock_covers_rollback_and_releases_after_failure(
    tmp_path,
    monkeypatch,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    comparison = case_dir / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "returncode": 0,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "passed",
                ),
            }
        ),
        encoding="utf-8",
    )
    real_commit = trtmc_validate._commit_file_update
    real_rollback = trtmc_validate._rollback_file_update
    injected = False
    rollback_checked = False

    def fail_after_commit(update):
        nonlocal injected
        real_commit(update)
        if not injected:
            injected = True
            raise OSError("injected commit failure")

    def check_rollback(update):
        nonlocal rollback_checked
        _assert_output_lock_held(tmp_path)
        rollback_checked = True
        return real_rollback(update)

    monkeypatch.setattr(
        trtmc_validate,
        "_commit_file_update",
        fail_after_commit,
    )
    monkeypatch.setattr(
        trtmc_validate,
        "_rollback_file_update",
        check_rollback,
    )

    with pytest.raises(OSError, match="injected commit failure"):
        trtmc_validate.write_report(
            tmp_path,
            result_paths=[comparison],
            expected_gates_by_binding=_test_gate_context(
                [comparison]
            ),
        )

    assert injected
    assert rollback_checked
    _assert_output_lock_available(tmp_path)


def test_all_report_only_contains_results_created_by_current_run(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "results"
    stale_dir = output / "removed-model" / "old-workload"
    stale_dir.mkdir(parents=True)
    stale_comparison = stale_dir / "comparison.json"
    stale_comparison.write_text(
        json.dumps(
            {
                "model": "removed-model",
                "workload": "old-workload",
                "status": "failed",
            }
        ),
        encoding="utf-8",
    )
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--output",
            str(output),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
        ]
    )
    binding = trtmc_validate.Binding("current-model", "current-workload")

    def run_worker(
        selected,
        *,
        arguments,
        catalog,
        expected_gates,
    ):
        del catalog
        assert expected_gates == _raw_evidence(
            selected.model,
            selected.workload,
            "passed",
        )["gates"]
        result = {
            "model": selected.model,
            "workload": selected.workload,
            "status": "passed",
            "returncode": 0,
            "raw_result": _raw_evidence(
                selected.model,
                selected.workload,
                "passed",
            ),
            "reproduce": {},
        }
        case_dir = trtmc_validate._prepare_case_directory(
            arguments.output,
            selected,
        )
        trtmc_validate._atomic_write_json(
            case_dir / "comparison.json",
            result,
        )
        return trtmc_validate._normalize_result(result)

    monkeypatch.setattr(
        trtmc_validate,
        "_run_supervised_binding_with_retries",
        run_worker,
    )
    monkeypatch.setattr(trtmc_validate, "_print_result", lambda *args: None)

    returncode = trtmc_validate._run_all_bindings(
        [binding],
        arguments=arguments,
        catalog={"sample_limits": {"current-workload": 2}},
        expected_gates_by_binding={
            ("current-model", "current-workload"): _raw_evidence(
                "current-model",
                "current-workload",
                "passed",
            )["gates"],
        },
    )

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert returncode == 0
    assert report["validation_status"] == "passed"
    assert report["summary"]["cases"] == 1
    assert [result["model"] for result in report["results"]] == [
        "current-model"
    ]
    assert stale_comparison.is_file()


def test_write_report_does_not_follow_workload_directory_symlink(tmp_path):
    output = tmp_path / "results"
    model_dir = output / "model-a"
    model_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    external_comparison = outside / "comparison.json"
    original = json.dumps(
        {
            "model": "outside",
            "workload": "outside",
            "status": "passed",
        }
    )
    external_comparison.write_text(original, encoding="utf-8")
    (model_dir / "workload-a").symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="must not be a symlink",
    ):
        trtmc_validate.write_report(output)

    assert external_comparison.read_text(encoding="utf-8") == original


def test_report_read_rejects_swapped_workload_directory(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "results"
    case_dir = output / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    internal = case_dir / "comparison.json"
    internal.write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "passed",
                "returncode": 0,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "passed",
                ),
            }
        ),
        encoding="utf-8",
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "comparison.json").write_text(
        json.dumps(
            {
                "model": "outside",
                "workload": "outside",
                "status": "failed",
            }
        ),
        encoding="utf-8",
    )
    original_case_dir = output / "model-a" / "workload-original"
    real_open = os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "comparison.json" and dir_fd is not None and not swapped:
            swapped = True
            case_dir.rename(original_case_dir)
            case_dir.symlink_to(outside, target_is_directory=True)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(trtmc_validate.os, "open", swapping_open)
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="parent changed while writing",
    ):
        trtmc_validate._read_report_result(output, internal)

    assert swapped
    assert json.loads(
        (original_case_dir / "comparison.json").read_text()
    )["model"] == "model-a"
    assert json.loads((outside / "comparison.json").read_text())["model"] == "outside"


def test_report_read_rejects_comparison_replaced_after_open(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "results"
    case_dir = output / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    comparison = case_dir / "comparison.json"
    comparison.write_text('{"model":"old"}', encoding="utf-8")
    opened_copy = case_dir / "comparison.opened.json"
    real_fdopen = os.fdopen
    swapped = False

    def swapping_fdopen(descriptor, *args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            comparison.rename(opened_copy)
            comparison.write_text('{"model":"new"}', encoding="utf-8")
        return real_fdopen(descriptor, *args, **kwargs)

    monkeypatch.setattr(trtmc_validate.os, "fdopen", swapping_fdopen)
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="changed while reading",
    ):
        trtmc_validate._read_report_result(output, comparison)

    assert swapped
    assert json.loads(comparison.read_text(encoding="utf-8"))["model"] == "new"


def test_case_artifact_read_rejects_log_replaced_after_open(
    tmp_path,
    monkeypatch,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    work_dir = case_dir / "validation"
    work_dir.mkdir(parents=True)
    log_path = work_dir / "trtfb_run.log"
    log_path.write_text("OLD-CONTENT", encoding="utf-8")
    opened_copy = work_dir / "trtfb_run.opened.log"
    real_fdopen = os.fdopen
    swapped = False

    def swapping_fdopen(descriptor, *args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            log_path.rename(opened_copy)
            log_path.write_text("NEW-CONTENT", encoding="utf-8")
        return real_fdopen(descriptor, *args, **kwargs)

    monkeypatch.setattr(trtmc_validate.os, "fdopen", swapping_fdopen)
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="changed while reading",
    ):
        trtmc_validate._read_case_text_artifact(
            log_path,
            case_dir=case_dir,
        )

    assert swapped
    assert log_path.read_text(encoding="utf-8") == "NEW-CONTENT"


def test_json_artifact_read_rejects_file_replaced_after_open(
    tmp_path,
    monkeypatch,
):
    metadata_path = tmp_path / "run.json"
    metadata_path.write_text('{"source":"old"}', encoding="utf-8")
    opened_copy = tmp_path / "run.opened.json"
    real_fdopen = os.fdopen
    swapped = False

    def swapping_fdopen(descriptor, *args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            metadata_path.rename(opened_copy)
            metadata_path.write_text('{"source":"new"}', encoding="utf-8")
        return real_fdopen(descriptor, *args, **kwargs)

    monkeypatch.setattr(trtmc_validate.os, "fdopen", swapping_fdopen)
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="changed while reading",
    ):
        trtmc_validate._read_json_artifact(metadata_path)

    assert swapped
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["source"] == "new"


def test_atomic_report_write_does_not_follow_swapped_parent(
    tmp_path,
    monkeypatch,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    destination = case_dir / "comparison.json"
    destination.write_text("OLD-INTERNAL", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "comparison.json"
    external.write_text("DO-NOT-OVERWRITE", encoding="utf-8")
    original_case_dir = tmp_path / "model-a" / "workload-original"
    real_open = os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if (
            isinstance(path, str)
            and path.startswith(".comparison.json.")
            and path.endswith(".tmp")
            and dir_fd is not None
            and not swapped
        ):
            swapped = True
            case_dir.rename(original_case_dir)
            case_dir.symlink_to(outside, target_is_directory=True)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(trtmc_validate.os, "open", swapping_open)
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="parent changed while writing",
    ):
        trtmc_validate._atomic_write_text(destination, "NEW-CONTENT")

    assert swapped
    assert external.read_text(encoding="utf-8") == "DO-NOT-OVERWRITE"
    assert (original_case_dir / "comparison.json").read_text() == "NEW-CONTENT"


def test_worker_log_write_is_anchored_when_parent_is_swapped(
    tmp_path,
    monkeypatch,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    log_path = case_dir / "worker.log"
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "worker.log"
    external.write_text("DO-NOT-APPEND", encoding="utf-8")
    original_case_dir = tmp_path / "model-a" / "workload-original"
    real_open = os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if (
            isinstance(path, str)
            and path.startswith(".worker.log.")
            and path.endswith(".tmp")
            and dir_fd is not None
            and not swapped
        ):
            swapped = True
            case_dir.rename(original_case_dir)
            case_dir.symlink_to(outside, target_is_directory=True)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(trtmc_validate.os, "open", swapping_open)
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="parent changed while writing",
    ):
        trtmc_validate._run_subprocess(
            [sys.executable, "-c", "print('worker-output')"],
            log_path,
            {},
        )

    assert swapped
    assert external.read_text(encoding="utf-8") == "DO-NOT-APPEND"
    internal_log = (original_case_dir / "worker.log").read_text(encoding="utf-8")
    assert "worker-output" in internal_log


def test_worker_log_replaces_hard_link_without_overwriting_external_file(
    tmp_path,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    external = tmp_path / "external.log"
    external.write_text("DO-NOT-TRUNCATE", encoding="utf-8")
    log_path = case_dir / "worker.log"
    os.link(external, log_path)

    returncode = trtmc_validate._run_subprocess(
        [sys.executable, "-c", "print('worker-output')"],
        log_path,
        {},
    )

    assert returncode == 0
    assert external.read_text(encoding="utf-8") == "DO-NOT-TRUNCATE"
    assert external.stat().st_nlink == 1
    assert "worker-output" in log_path.read_text(encoding="utf-8")


def test_supervisor_does_not_unlink_through_swapped_model_directory(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "output"
    case_dir = output / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    (case_dir / "comparison.json").write_text("STALE", encoding="utf-8")
    outside_model = tmp_path / "outside-model"
    outside_case = outside_model / "workload-a"
    outside_case.mkdir(parents=True)
    external = outside_case / "comparison.json"
    external.write_text("DO-NOT-DELETE", encoding="utf-8")
    original_model = output / "model-original"
    real_prepare = trtmc_validate._prepare_case_directory

    def prepare_and_swap(output_path, binding):
        prepared = real_prepare(output_path, binding)
        (output / "model-a").rename(original_model)
        (output / "model-a").symlink_to(
            outside_model,
            target_is_directory=True,
        )
        return prepared

    monkeypatch.setattr(
        trtmc_validate,
        "_prepare_case_directory",
        prepare_and_swap,
    )
    arguments = argparse.Namespace(output=output)

    with pytest.raises(trtmc_validate.ValidationError):
        trtmc_validate._run_supervised_binding(
            trtmc_validate.Binding("model-a", "workload-a"),
            arguments=arguments,
            catalog={},
        )

    assert external.read_text(encoding="utf-8") == "DO-NOT-DELETE"
    assert (original_model / "workload-a" / "comparison.json").is_file()


def test_report_does_not_refresh_disagreement_artifact_from_raw_work_dir(
    tmp_path,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    comparison = case_dir / "comparison.json"
    comparison.write_text(
        json.dumps(
                {
                    "model": "model-a",
                    "workload": "workload-a",
                    "status": "passed",
                    "returncode": 0,
                    "raw_result": _raw_evidence(
                        "model-a",
                        "workload-a",
                        "passed",
                        work_dir=str(work_dir),
                    ),
                }
        ),
        encoding="utf-8",
    )
    external = tmp_path / "external-disagreements.jsonl"
    external.write_text("DO-NOT-TRUNCATE\n", encoding="utf-8")
    (case_dir / "disagreements.jsonl").symlink_to(external)

    _, _, report = trtmc_validate.write_report(
        tmp_path,
        expected_gates_by_binding=_test_gate_context(
            [comparison]
        ),
    )

    assert report["validation_status"] == "passed"
    assert external.read_text(encoding="utf-8") == "DO-NOT-TRUNCATE\n"


def test_report_rejects_disagreement_artifact_path_outside_case(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    external = tmp_path / "outside.jsonl"
    external.write_text(
        json.dumps({"sample_id": "secret", "input": "LOCAL-SECRET-MARKER"})
        + "\n",
        encoding="utf-8",
    )
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "failed",
                "returncode": 1,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "failed",
                ),
                "disagreements": {
                    "count": 1,
                    "path": str(external),
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="artifact path must be disagreements.jsonl",
    ):
        trtmc_validate.write_report(tmp_path)

    assert "LOCAL-SECRET-MARKER" in external.read_text(encoding="utf-8")
    assert not (tmp_path / "report.html").exists()


@pytest.mark.parametrize("artifact_kind", ["symlink", "fifo", "hardlink"])
def test_report_rejects_unsafe_disagreement_artifact(
    tmp_path,
    artifact_kind,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "failed",
                "returncode": 1,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "failed",
                ),
                "disagreements": {
                    "count": 1,
                    "path": "disagreements.jsonl",
                },
            }
        ),
        encoding="utf-8",
    )
    artifact = case_dir / "disagreements.jsonl"
    external = tmp_path / "external.jsonl"
    external.write_text(
        json.dumps({"sample_id": "secret", "input": "DO-NOT-READ"}) + "\n",
        encoding="utf-8",
    )
    if artifact_kind == "symlink":
        artifact.symlink_to(external)
    elif artifact_kind == "fifo":
        os.mkfifo(artifact)
    else:
        os.link(external, artifact)

    with pytest.raises(trtmc_validate.ValidationError):
        trtmc_validate.write_report(tmp_path)

    assert "DO-NOT-READ" in external.read_text(encoding="utf-8")


@pytest.mark.parametrize("media_kind", ["symlink", "hardlink"])
def test_report_does_not_render_unsafe_case_media(
    tmp_path,
    media_kind,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    sample_id = "sample-1"
    sample_directory = (
        trtmc_disagreements._sample_directory_name(sample_id)
    )
    relative_media = (
        Path("repro") / sample_directory / "media" / "01-secret.png"
    )
    media_path = case_dir / relative_media
    media_path.parent.mkdir(parents=True)
    external = tmp_path / "secret.png"
    external.write_bytes(b"DO-NOT-SERVE")
    if media_kind == "symlink":
        media_path.symlink_to(external)
    else:
        os.link(external, media_path)
    (case_dir / "disagreements.jsonl").write_text(
        json.dumps(
            _canonical_disagreement_record(
                sample_id,
                artifacts={
                    "media": [
                        {
                            "label": "secret",
                            "kind": "image",
                            "path": str(relative_media),
                        }
                    ]
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (case_dir / "comparison.json").write_text(
        json.dumps(
                {
                    "model": "model-a",
                    "workload": "workload-a",
                    "status": "failed",
                    "returncode": 1,
                    "raw_result": _raw_evidence(
                        "model-a",
                        "workload-a",
                        "failed",
                    ),
                    "disagreements": {
                    "count": 1,
                    "path": "disagreements.jsonl",
                },
            }
        ),
        encoding="utf-8",
    )

    _, html_path, _ = trtmc_validate.write_report(tmp_path)

    rendered = html_path.read_text(encoding="utf-8")
    assert str(relative_media) not in rendered
    assert external.read_bytes() == b"DO-NOT-SERVE"


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "01-%2e%2e-secret.png",
        r"01-..\secret.png",
    ],
)
def test_report_does_not_render_noncanonical_media_url(
    tmp_path,
    unsafe_name,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    sample_id = "sample"
    relative_media = (
        Path("repro")
        / trtmc_disagreements._sample_directory_name(sample_id)
        / "media"
        / unsafe_name
    )
    media_path = case_dir / relative_media
    media_path.parent.mkdir(parents=True)
    media_path.write_bytes(b"UNSAFE-MEDIA")
    (case_dir / "disagreements.jsonl").write_text(
        json.dumps(
            _canonical_disagreement_record(
                sample_id,
                artifacts={
                    "media": [
                        {
                            "label": "secret",
                            "kind": "image",
                            "path": relative_media.as_posix(),
                        }
                    ]
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (case_dir / "comparison.json").write_text(
        json.dumps(
                {
                    "model": "model-a",
                    "workload": "workload-a",
                    "status": "failed",
                    "returncode": 1,
                    "raw_result": _raw_evidence(
                        "model-a",
                        "workload-a",
                        "failed",
                    ),
                    "disagreements": {
                    "count": 1,
                    "path": "disagreements.jsonl",
                },
            }
        ),
        encoding="utf-8",
    )

    _, html_path, _ = trtmc_validate.write_report(tmp_path)

    assert relative_media.as_posix() not in html_path.read_text(
        encoding="utf-8"
    )


def test_report_rejects_symlinked_work_dir_inside_case(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    outside = tmp_path / "outside-work"
    outside.mkdir()
    external_log = outside / "trtfb_run.log"
    external_log.write_text("$ trtmc run SECRET\n", encoding="utf-8")
    work_dir = case_dir / "task-eval"
    work_dir.symlink_to(outside, target_is_directory=True)
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "passed",
                "returncode": 0,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "passed",
                    work_dir=str(work_dir),
                ),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(trtmc_validate.ValidationError):
        trtmc_validate.write_report(tmp_path)

    assert external_log.read_text(encoding="utf-8") == "$ trtmc run SECRET\n"


def test_report_allows_external_input_media_but_rejects_external_outputs(
    tmp_path,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    work_dir = case_dir / "validation" / "workload-a" / "model-a"
    work_dir.mkdir(parents=True)
    input_backing = tmp_path / "dataset-input-backing.png"
    input_backing.write_bytes(b"EXPECTED-DATASET-INPUT")
    external_input = tmp_path / "dataset-input.png"
    os.link(input_backing, external_input)
    external_output = tmp_path / "private-reference-output.png"
    external_output.write_bytes(b"DO-NOT-COPY-PRIVATE-OUTPUT")
    sample_id = "sample-1"
    (work_dir / "prompts.jsonl").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "images": [str(external_input)],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (work_dir / "summary.json").write_text(
        json.dumps({"disagreements": [{"sample_id": sample_id}]}),
        encoding="utf-8",
    )
    (work_dir / "hf_predictions.json").write_text(
        json.dumps(
            {
                "responses": [
                    {
                        "sample_id": sample_id,
                        "hf_image": str(external_output),
                    }
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
                "returncode": 1,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "failed",
                    work_dir=str(work_dir),
                ),
            }
        ),
        encoding="utf-8",
    )

    _, _, report = trtmc_validate.write_report(tmp_path)

    artifact = case_dir / report["results"][0]["disagreements"]["path"]
    record = json.loads(artifact.read_text(encoding="utf-8").splitlines()[0])
    media = record["artifacts"]["media"]
    assert [item["label"] for item in media] == ["Input input image 1"]
    copied_input = case_dir / media[0]["path"]
    assert copied_input.read_bytes() == b"EXPECTED-DATASET-INPUT"
    assert all(
        path.read_bytes() != b"DO-NOT-COPY-PRIVATE-OUTPUT"
        for path in (case_dir / "repro").rglob("*")
        if path.is_file()
    )


@pytest.mark.parametrize(
    "payload",
    [
        '{"responses":NaN}',
        "{",
        '{"nested":' * 300 + "0" + "}" * 300,
    ],
    ids=["nan", "invalid", "deep"],
)
def test_report_wraps_invalid_disagreement_json_as_validation_error(
    tmp_path,
    payload,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    work_dir = case_dir / "validation" / "workload-a" / "model-a"
    work_dir.mkdir(parents=True)
    (work_dir / "hf_predictions.json").write_text(payload, encoding="utf-8")
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "failed",
                "returncode": 1,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "failed",
                    work_dir=str(work_dir),
                ),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="invalid disagreement evidence",
    ):
        trtmc_validate.write_report(tmp_path)


def test_report_wraps_invalid_reproduction_seed_as_validation_error(
    tmp_path,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    work_dir = case_dir / "validation" / "workload-a" / "model-a"
    work_dir.mkdir(parents=True)
    (work_dir / "prompts.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "prompt": "hello",
                "seed_index": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (work_dir / "summary.json").write_text(
        json.dumps({"disagreements": [{"sample_id": "sample-1"}]}),
        encoding="utf-8",
    )
    (work_dir / "trtfb_repro.json").write_text(
        json.dumps(
            {
                "base_seed": 1,
                "command": ["trtmc", "run", "--seed", "{sample_seed}"],
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
                "returncode": 1,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "failed",
                    work_dir=str(work_dir),
                ),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="invalid disagreement evidence",
    ):
        trtmc_validate.write_report(tmp_path)


def test_report_wraps_nul_work_dir_as_validation_error(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "passed",
                "raw_result": {"work_dir": "bad\u0000directory"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(trtmc_validate.ValidationError):
        trtmc_validate.write_report(tmp_path)


def test_atomic_copy_enforces_size_limit_when_source_grows_after_open(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.png"
    source.write_bytes(b"1234")
    destination = tmp_path / "copied.png"
    real_fdopen = os.fdopen
    grew = False

    def growing_fdopen(descriptor, *args, **kwargs):
        nonlocal grew
        if args and args[0] == "rb" and not grew:
            grew = True
            with source.open("ab") as source_file:
                source_file.write(b"5678")
        return real_fdopen(descriptor, *args, **kwargs)

    monkeypatch.setattr(trtmc_validate.os, "fdopen", growing_fdopen)
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="exceeds 4 bytes while copying",
    ):
        trtmc_validate._atomic_copy_regular_file(
            source,
            destination,
            maximum_bytes=4,
        )

    assert grew
    assert source.read_bytes() == b"12345678"
    assert not destination.exists()


def test_atomic_copy_closes_source_directory_if_destination_open_fails(
    tmp_path,
    monkeypatch,
):
    source_dir = tmp_path / "source"
    destination_dir = tmp_path / "destination"
    source_dir.mkdir()
    destination_dir.mkdir()
    source = source_dir / "source.txt"
    source.write_text("data", encoding="utf-8")
    real_open = trtmc_validate._open_real_directory
    opened_descriptors = []

    def fail_destination(path):
        if path == source_dir:
            descriptor = real_open(path)
            opened_descriptors.append(descriptor)
            return descriptor
        raise trtmc_validate.ValidationError("destination open failed")

    monkeypatch.setattr(
        trtmc_validate,
        "_open_real_directory",
        fail_destination,
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="destination open failed",
    ):
        trtmc_validate._atomic_copy_regular_file(
            source,
            destination_dir / "copy.txt",
        )

    assert opened_descriptors
    for descriptor in opened_descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_retry_archive_rejects_destination_symlink_without_overwriting_target(
    tmp_path,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    (case_dir / "comparison.json").write_text("CURRENT", encoding="utf-8")
    external = tmp_path / "external-attempt.json"
    external.write_text("DO-NOT-OVERWRITE", encoding="utf-8")
    (case_dir / "comparison.attempt-1.json").symlink_to(external)

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="must be a regular file",
    ):
        trtmc_validate._archive_failed_attempt(case_dir, 1)

    assert external.read_text(encoding="utf-8") == "DO-NOT-OVERWRITE"


def test_write_report_rejects_fifo_comparison_without_blocking(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    os.mkfifo(case_dir / "comparison.json")

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="must be a regular file",
    ):
        trtmc_validate.write_report(tmp_path)


def test_write_report_rejects_fifo_run_metadata_without_blocking(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "passed",
            }
        ),
        encoding="utf-8",
    )
    os.mkfifo(tmp_path / "run.json")

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="must be a regular file",
    ):
        trtmc_validate.write_report(tmp_path)


def test_primary_json_readers_enforce_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        trtmc_validate,
        "MAX_REPORT_ARTIFACT_BYTES",
        8,
    )
    metadata = tmp_path / "run.json"
    metadata.write_text('{"value":1}', encoding="utf-8")
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="exceeds 8 bytes",
    ):
        trtmc_validate._read_json_artifact(metadata)

    output = tmp_path / "results"
    case_dir = output / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    comparison = case_dir / "comparison.json"
    comparison.write_text('{"value":1}', encoding="utf-8")
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="exceeds 8 bytes",
    ):
        trtmc_validate._read_report_result(output, comparison)


def test_write_report_rejects_nonstandard_json_constants(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    (case_dir / "comparison.json").write_text(
        (
            '{"model":"model-a","workload":"workload-a",'
            '"status":"passed","metric":NaN}'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="non-standard JSON constant",
    ):
        trtmc_validate.write_report(tmp_path)


@pytest.mark.parametrize(
    "extra_field",
    [
        '"metric":1e309',
        '"note":"\\ud800"',
    ],
    ids=["overflow-to-infinity", "isolated-surrogate"],
)
def test_write_report_rejects_invalid_json_scalars(
    tmp_path,
    extra_field,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    (case_dir / "comparison.json").write_text(
        (
            '{"model":"model-a","workload":"workload-a",'
            f'"status":"passed",{extra_field}}}'
        ),
        encoding="utf-8",
    )

    with pytest.raises(trtmc_validate.ValidationError):
        trtmc_validate.write_report(tmp_path)


def test_atomic_json_write_rejects_nonfinite_values_before_replacing_file(
    tmp_path,
):
    destination = tmp_path / "comparison.json"
    destination.write_text("KEEP", encoding="utf-8")

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="non-finite number",
    ):
        trtmc_validate._atomic_write_json(destination, {"metric": float("nan")})

    assert destination.read_text(encoding="utf-8") == "KEEP"


@pytest.mark.parametrize(
    "line",
    [
        '{"command":["trtmc",NaN]}',
        '{"command":["trtmc",1e309]}',
        '{"command":["trtmc","\\ud800"]}',
        "[" * 2000 + "0" + "]" * 2000,
    ],
    ids=["nan", "overflow-to-infinity", "isolated-surrogate", "deep"],
)
def test_command_log_records_reject_nonstandard_or_deep_json(line):
    with pytest.raises(trtmc_validate.ValidationError):
        trtmc_validate._command_record_from_log_line(line)


@pytest.mark.parametrize(
    "line",
    [
        '{"sample_id":"sample","value":1e309}',
        '{"sample_id":"sample","value":"\\ud800"}',
    ],
    ids=["overflow-to-infinity", "isolated-surrogate"],
)
def test_disagreement_preview_rejects_invalid_json_scalars(
    tmp_path,
    line,
):
    artifact = tmp_path / "disagreements.jsonl"
    artifact.write_text(line + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        trtmc_disagreements.load_disagreement_preview(artifact)


def test_disagreement_builder_validates_final_record_depth(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    work_dir = case_dir / "validation" / "workload-a" / "model-a"
    work_dir.mkdir(parents=True)
    nested = 0
    for _ in range(255):
        nested = [nested]
    prompts = work_dir / "prompts.jsonl"
    prompts.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "nested": nested,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (work_dir / "summary.json").write_text(
        '{"disagreements":[{"sample_id":"sample"}]}',
        encoding="utf-8",
    )
    assert trtmc_disagreements._load_jsonl(prompts)

    with pytest.raises(ValueError, match="nesting exceeds"):
        trtmc_disagreements.build_disagreement_artifact(
            work_dir=work_dir,
            case_dir=case_dir,
        )

    assert not (case_dir / "disagreements.jsonl").exists()


def test_disagreement_builder_enforces_preview_reader_size_limit(
    tmp_path,
    monkeypatch,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    work_dir = case_dir / "validation" / "workload-a" / "model-a"
    work_dir.mkdir(parents=True)
    (work_dir / "prompts.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "prompt": "x" * 200,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (work_dir / "summary.json").write_text(
        '{"disagreements":[{"sample_id":"sample"}]}',
        encoding="utf-8",
    )
    artifact = case_dir / "disagreements.jsonl"
    artifact.write_text("KEEP\n", encoding="utf-8")
    monkeypatch.setattr(
        trtmc_disagreements,
        "MAX_DISAGREEMENT_ARTIFACT_BYTES",
        128,
    )

    with pytest.raises(ValueError, match="exceeds 128 bytes"):
        trtmc_disagreements.build_disagreement_artifact(
            work_dir=work_dir,
            case_dir=case_dir,
        )

    assert artifact.read_text(encoding="utf-8") == "KEEP\n"


def test_atomic_json_write_enforces_reader_size_limit_before_replace(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "comparison.json"
    original = '{"keep":true}'
    destination.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        trtmc_validate,
        "MAX_REPORT_ARTIFACT_BYTES",
        64,
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="exceeds 64 bytes",
    ):
        trtmc_validate._atomic_write_json(
            destination,
            {"payload": "x" * 80},
        )

    assert destination.read_text(encoding="utf-8") == original


def test_run_metadata_depth_is_preflighted_before_result_rewrite(
    tmp_path,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    comparison = case_dir / "comparison.json"
    original = (
        '{"model":"model-a","workload":"workload-a",'
        '"status":"passed"}'
    )
    comparison.write_text(original, encoding="utf-8")
    nested = 0
    for _ in range(255):
        nested = [nested]
    (tmp_path / "run.json").write_text(
        json.dumps({"nested": nested}),
        encoding="utf-8",
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="nesting exceeds 255",
    ):
        trtmc_validate.write_report(tmp_path)

    assert comparison.read_text(encoding="utf-8") == original


def test_report_rejects_result_too_deep_for_report_envelope(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    nested = 0
    for _ in range(
        trtmc_validate.MAX_VALIDATION_RESULT_JSON_DEPTH + 1
    ):
        nested = [nested]
    comparison = case_dir / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "passed",
                "nested": nested,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="nesting exceeds 254",
    ):
        trtmc_validate.write_report(tmp_path)

    assert not (tmp_path / "report.json").exists()


def test_result_rejects_unhashable_primary_metric_name(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "raw_result": {"status": "passed"},
                "execution": {"status": "completed"},
                "comparison": {
                    "status": "agreement",
                    "metrics": {"accuracy": 1.0},
                    "primary_metric": {
                        "name": [],
                        "value": 1.0,
                    },
                },
                "validation": {"status": "passed"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="primary_metric.name",
    ):
        trtmc_validate.write_report(tmp_path)


def test_result_nested_status_defaults_are_normalized(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "raw_result": {
                    "status": "failed",
                    "error": "runner crashed",
                },
                "execution": {},
                "comparison": {},
                "validation": {},
                "reference_environment": None,
            }
        ),
        encoding="utf-8",
    )

    _, _, report = trtmc_validate.write_report(tmp_path)

    result = report["results"][0]
    assert result["execution"]["status"] == "error"
    assert result["comparison"]["status"] == "not_run"
    assert result["validation"]["status"] == "failed"
    assert result["reference_environment"] == []


def test_result_rejects_invalid_attempt_count(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "raw_result": {"status": "passed"},
                "execution": {
                    "status": "completed",
                    "attempt_count": {},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="attempt_count must be a positive integer",
    ):
        trtmc_validate.write_report(tmp_path)


@pytest.mark.parametrize(
    ("execution", "message"),
    [
        (
            {
                "status": "completed",
                "exit_code": 0,
                "attempt_count": 1,
                "max_attempts": 0,
                "retry_count": 0,
                "attempts": [
                    {
                        "attempt": 1,
                        "execution_status": "completed",
                        "validation_status": "passed",
                    }
                ],
            },
            "max_attempts",
        ),
        (
            {
                "status": "completed",
                "exit_code": 0,
                "attempt_count": 1,
                "max_attempts": 1,
                "retry_count": 99,
                "attempts": [
                    {
                        "attempt": 1,
                        "execution_status": "completed",
                        "validation_status": "passed",
                    }
                ],
            },
            "retry_count",
        ),
        (
            {
                "status": "completed",
                "exit_code": 0,
                "attempt_count": 1,
                "max_attempts": 1,
                "retry_count": 0,
                "attempts": [
                    {
                        "attempt": True,
                        "execution_status": "completed",
                        "validation_status": "passed",
                    }
                ],
            },
            "integer, contiguous",
        ),
        (
            {
                "status": "completed",
                "exit_code": 0,
                "attempt_count": 1,
                "max_attempts": 1,
                "retry_count": 0,
                "attempts": [
                    {
                        "attempt": 1,
                        "execution_status": [],
                        "validation_status": "passed",
                    }
                ],
            },
            "execution_status is invalid",
        ),
        (
            {
                "status": "completed",
                "exit_code": 0,
                "attempt_count": 1,
                "max_attempts": 1,
                "retry_count": 0,
                "attempts": [
                    {
                        "attempt": 1,
                        "execution_status": "completed",
                        "validation_status": {},
                    }
                ],
            },
            "validation_status is invalid",
        ),
        (
            {
                "status": "completed",
                "exit_code": 0,
                "attempt_count": 2,
                "max_attempts": 2,
                "retry_count": 1,
                "attempts": [
                    {
                        "attempt": 1,
                        "execution_status": "completed",
                        "validation_status": "passed",
                    },
                    {
                        "attempt": 2,
                        "execution_status": "completed",
                        "validation_status": "passed",
                    },
                ],
            },
            "non-final retry attempts",
        ),
        (
            {
                "status": "completed",
                "exit_code": 0,
                "attempt_count": 2,
                "max_attempts": 2,
                "retry_count": 1,
                "attempts": [
                    {
                        "attempt": 1,
                        "execution_status": "error",
                        "validation_status": "passed",
                        "error_type": "WorkerProcessError",
                    },
                    {
                        "attempt": 2,
                        "execution_status": "completed",
                        "validation_status": "passed",
                    },
                ],
            },
            "non-final retry attempts",
        ),
        (
            {
                "status": "completed",
                "exit_code": 0,
                "attempt_count": 1,
                "max_attempts": 1,
                "retry_count": 0,
                "attempts": [
                    {
                        "attempt": 1,
                        "execution_status": "completed",
                        "validation_status": "passed",
                        "error_type": "WorkerProcessError",
                        "error": "worker crashed",
                    }
                ],
            },
            "incompatible error evidence",
        ),
        (
            {
                "status": "completed",
                "exit_code": 0,
                "attempt_count": 1,
                "max_attempts": 1,
                "retry_count": 0,
                "attempts": [
                    {
                        "attempt": 1,
                        "execution_status": "error",
                        "validation_status": "failed",
                        "error_type": "WorkerProcessError",
                        "error": "worker crashed",
                    }
                ],
            },
            "final retry attempt must match",
        ),
    ],
)
def test_result_rejects_inconsistent_retry_evidence(
    execution,
    message,
):
    with pytest.raises(trtmc_validate.ValidationError, match=message):
        trtmc_validate._normalize_result(
            {
                "schema_version": "trtmc.validation-result/v2",
                "model": "model-a",
                "workload": "workload-a",
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "passed",
                ),
                "execution": execution,
                "comparison": {"status": "agreement"},
                "validation": {"status": "passed"},
            }
        )


@pytest.mark.parametrize("attempt", [True, 1.0])
def test_result_rejects_non_integer_retry_attempt_number(attempt):
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="integer, contiguous",
    ):
        trtmc_validate._normalize_result(
            {
                "schema_version": "trtmc.validation-result/v2",
                "model": "model-a",
                "workload": "workload-a",
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "passed",
                ),
                "execution": {
                    "status": "completed",
                    "exit_code": 0,
                    "attempt_count": 1,
                    "max_attempts": 1,
                    "retry_count": 0,
                    "attempts": [
                        {
                            "attempt": attempt,
                            "execution_status": "completed",
                            "validation_status": "passed",
                        }
                    ],
                },
            }
        )


@pytest.mark.parametrize(
    ("raw_extra", "exit_code", "execution_status", "final_error"),
    [
        (
            {
                "error_type": "WorkerProcessError",
                "error": "worker crashed",
            },
            1,
            "error",
            {
                "error_type": "DifferentError",
                "error": "different failure",
            },
        ),
        (
            {
                "error_type": "BenchmarkGateError",
                "error": "gate failed",
                "gate_failures": [{"gate": "minimum"}],
            },
            1,
            "completed",
            {},
        ),
    ],
)
def test_result_rejects_final_retry_error_evidence_mismatch(
    raw_extra,
    exit_code,
    execution_status,
    final_error,
):
    validation_status = "failed"
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="must match raw_result",
    ):
        trtmc_validate._normalize_result(
            {
                "schema_version": "trtmc.validation-result/v2",
                "model": "model-a",
                "workload": "workload-a",
                "returncode": exit_code,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "failed",
                    **raw_extra,
                ),
                "execution": {
                    "status": execution_status,
                    "exit_code": exit_code,
                    "attempt_count": 1,
                    "max_attempts": 1,
                    "retry_count": 0,
                    "attempts": [
                        {
                            "attempt": 1,
                            "execution_status": execution_status,
                            "validation_status": validation_status,
                            **final_error,
                        }
                    ],
                },
                "validation": {"status": validation_status},
            }
        )


def test_report_rejects_result_identity_mismatch(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    work_dir = case_dir / "validation" / "workload-a" / "model-a"
    work_dir.mkdir(parents=True)
    (work_dir / "prompts.jsonl").write_text(
        '{"sample_id":"sample-1","prompt":"SECRET"}\n',
        encoding="utf-8",
    )
    (work_dir / "summary.json").write_text(
        '{"disagreements":[{"sample_id":"sample-1"}]}',
        encoding="utf-8",
    )
    (work_dir / "trtfb_repro.json").write_text(
        '{"command":["trtmc","run","--input","{input_jsonl}"]}',
        encoding="utf-8",
    )
    disagreements = case_dir / "disagreements.jsonl"
    disagreements.write_text("KEEP\n", encoding="utf-8")
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-b",
                "workload": "workload-b",
                "status": "passed",
                "raw_result": {
                    "status": "passed",
                    "work_dir": str(work_dir),
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="model does not match its path",
    ):
        trtmc_validate.write_report(tmp_path)

    assert disagreements.read_text(encoding="utf-8") == "KEEP\n"
    assert not (case_dir / "repro").exists()


def test_report_preflights_all_result_identities_before_rewrite(tmp_path):
    first_dir = tmp_path / "model-a" / "workload-a"
    second_dir = tmp_path / "model-b" / "workload-b"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    first = first_dir / "comparison.json"
    first_text = (
        '{"model":"model-a","workload":"workload-a",'
        '"status":"passed"}'
    )
    first.write_text(first_text, encoding="utf-8")
    second = second_dir / "comparison.json"
    second.write_text(
        (
            '{"model":"WRONG","workload":"workload-b",'
            '"status":"passed"}'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="model does not match its path",
    ):
        trtmc_validate.write_report(
            tmp_path,
            result_paths=[first, second],
        )

    assert first.read_text(encoding="utf-8") == first_text


def test_report_requires_count_in_disagreement_metadata(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "failed",
                "disagreements": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="must include count",
    ):
        trtmc_validate.write_report(tmp_path)


def test_write_report_converts_deep_json_recursion_to_validation_error(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    (case_dir / "comparison.json").write_text(
        '{"nested":' * 2000 + "0" + "}" * 2000,
        encoding="utf-8",
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="invalid validation result JSON",
    ):
        trtmc_validate.write_report(tmp_path)


def test_write_report_surfaces_quantized_reference_precision_contract(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "quantized-model" / "mmlu_five_shot_mcq"
    case_dir.mkdir(parents=True)
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "quantized-model",
                "workload": "mmlu_five_shot_mcq",
                "status": "passed",
                "raw_result": {
                    "status": "passed",
                    "mode": "mcq",
                    "prediction_agreement_rate": 1.0,
                    "accuracy_drop_from_hf": 0.0,
                    "valid_count": 1,
                    "skipped_count": 0,
                    "total_count": 1,
                    "gates": {
                        "min_prediction_agreement_rate": 0.98,
                        "max_accuracy_drop_from_hf": 0.01,
                    },
                    "precision_contract": {
                        "trtmc_base_precision": "bf16",
                        "trtmc_quantization": "fp8",
                        "reference_precision": "bf16",
                        "reference_dtype": "bfloat16",
                        "comparison": "quantized_vs_unquantized_reference",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    _, html_path, report = trtmc_validate.write_report(tmp_path)

    assert report["results"][0]["precision_contract"] == {
        "trtmc_base_precision": "bf16",
        "trtmc_quantization": "fp8",
        "reference_precision": "bf16",
        "reference_dtype": "bfloat16",
        "comparison": "quantized_vs_unquantized_reference",
    }
    document = html_path.read_text(encoding="utf-8")
    assert "TRTMC FP8 (BF16 base) vs HF BF16" in document
    assert "Quantized candidate vs unquantized reference" in document


def test_traffic_light_counts_are_mutually_exclusive():
    def result(validation, comparison):
        return {
            "validation": {"status": validation},
            "comparison": {"status": comparison},
        }

    assert trtmc_validate._traffic_light_counts(
        [
            result("passed", "agreement"),
            result("skipped", "not_run"),
            result("failed", "disagreement"),
            result("not_compared", "not_run"),
        ]
    ) == {
        "green": 1,
        "yellow": 1,
        "red": 1,
        "white": 1,
    }


def test_diffusion_report_flattens_nested_reference_metrics():
    comparison = trtmc_validate._comparison_details(
        {
            "status": "passed",
            "mode": "diffusion_image_clip_parity",
            "overall_pass_rate": 1.0,
            "passed_count": 5,
            "valid_count": 5,
            "skipped_count": 0,
            "metrics": {
                "trt_hf_image_clip_cosine": {
                    "mean": 0.91,
                    "min": 0.87,
                    "max": 0.95,
                    "count": 5,
                },
                "psnr": {
                    "mean": 12.5,
                    "min": 11.0,
                    "max": 14.0,
                    "count": 5,
                },
            },
        },
        {"status": "completed"},
    )

    assert comparison["primary_metric"] == {
        "name": "overall_pass_rate",
        "value": 1.0,
    }
    assert comparison["metrics"]["trt_hf_image_clip_cosine"] == 0.91
    assert comparison["metrics"]["psnr"] == 12.5
    assert "No metrics" not in trtmc_validate._render_metrics(
        {"comparison": comparison}
    )


def test_legacy_e2e_result_is_not_reported_as_reference_agreement(tmp_path):
    case_dir = tmp_path / "model-a" / "e2e"
    case_dir.mkdir(parents=True)
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "e2e",
                "executor": "e2e",
                "status": "passed",
                "returncode": 0,
                "raw_results": [{"status": "pass"}],
            }
        ),
        encoding="utf-8",
    )

    _, html_path, report = trtmc_validate.write_report(tmp_path)

    result = report["results"][0]
    assert report["validation_status"] == "incomplete"
    assert report["summary"]["agreements"] == 0
    assert report["summary"]["not_compared"] == 1
    assert result["execution"]["status"] == "not_run"
    assert result["comparison"]["status"] == "not_run"
    assert result["validation"]["status"] == "not_compared"
    assert result["not_compared_reason"] == trtmc_validate.LEGACY_E2E_REASON
    document = html_path.read_text(encoding="utf-8")
    assert "🟢 0 &nbsp; 🟡 0 &nbsp;" in document
    assert "🔴 0 &nbsp; ⚪ 1" in document
    assert "E2E execution does not compare aligned reference" in document


def test_not_compared_result_replaces_legacy_e2e_row_without_deleting_evidence(
    tmp_path,
):
    legacy_dir = tmp_path / "model-a" / "e2e"
    legacy_dir.mkdir(parents=True)
    legacy_comparison = legacy_dir / "comparison.json"
    legacy_comparison.write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "e2e",
                "executor": "e2e",
                "status": "passed",
                "raw_results": [{"status": "pass"}],
            }
        ),
        encoding="utf-8",
    )
    trtmc_validate._write_not_compared_case(
        trtmc_validate.Binding(
            "model-a",
            None,
            "Reference comparator is missing.",
        ),
        tmp_path,
    )

    _, _, report = trtmc_validate.write_report(tmp_path)

    assert legacy_comparison.is_file()
    assert report["summary"]["cases"] == 1
    assert report["summary"]["not_compared"] == 1
    assert (
        report["results"][0]["not_compared_reason"]
        == "Reference comparator is missing."
    )


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


def test_write_report_preserves_finalized_duration(tmp_path, monkeypatch):
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "started_at": "2026-07-25T01:02:03+00:00",
                "finished_at": "2026-07-25T01:02:13+00:00",
                "duration_seconds": 10.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        trtmc_validate,
        "_utc_now",
        lambda: datetime(2026, 7, 25, 4, 4, 6, tzinfo=timezone.utc),
    )

    _, html_path, report = trtmc_validate.write_report(tmp_path)

    assert report["summary"]["duration_seconds"] == 10.0
    assert "0h 00m 10s total duration" in html_path.read_text(encoding="utf-8")


def test_write_report_does_not_render_validation_wrapper(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "failed",
                "returncode": 1,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "failed",
                ),
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
    comparison = case_dir / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "passed",
                "returncode": 0,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "passed",
                    work_dir=str(work_dir),
                ),
                "reproduce": {"hf": ["python hf.py"], "trtmc": ["trtmc build"]},
            }
        ),
        encoding="utf-8",
    )

    _, html_path, report = trtmc_validate.write_report(
        tmp_path,
        expected_gates_by_binding=_test_gate_context(
            [comparison]
        ),
    )

    assert report["results"][0]["reproduce"]["trtmc"] == [
        "trtmc build",
        "trtmc solve model.trtfb --field-input 1,2",
    ]
    assert "$ trtmc solve model.trtfb --field-input 1,2" in html_path.read_text(encoding="utf-8")


def test_report_bounds_large_sample_commands_and_selects_disagreement(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    work_dir = case_dir / "validation" / "workload-a" / "model-a"
    work_dir.mkdir(parents=True)
    sample_count = 10_000
    (work_dir / "prompts.jsonl").write_text(
        "".join(
            json.dumps({"sample_id": f"sample-{index}", "prompt": f"prompt-{index}"}) + "\n"
            for index in range(sample_count)
        ),
        encoding="utf-8",
    )
    (work_dir / "trtfb_run.log").write_text(
        "".join(
            f"$ trtmc run model.trtfb --prompt prompt-{index}\n" for index in range(sample_count)
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
                "returncode": 1,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "failed",
                    work_dir=str(work_dir),
                ),
                "reproduce": {
                    "dataset": {
                        "command": ("python tools/trtmc_validate.py model-a --limit 10000"),
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
    assert reproduction["trtmc"] == ["trtmc run model.trtfb --prompt prompt-9999"]
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
                "returncode": 1,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "failed",
                    work_dir=str(work_dir),
                ),
            }
        ),
        encoding="utf-8",
    )

    _, html_path, report = trtmc_validate.write_report(tmp_path)

    metadata = report["results"][0]["disagreements"]
    assert metadata["count"] == 1
    artifact = case_dir / metadata["path"]
    records = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines()]
    assert records[0]["input"] == prompt
    assert records[0]["reference_result"]["output_text"] == "reference answer"
    assert records[0]["trtmc_result"]["output_text"] == "TRTMC answer"
    assert records[0]["reproduce"]["reference"].startswith(
        "/profiles/reference/bin/python /workspace/trtmc/tools/reference/transformers_text.py"
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


def test_commands_from_logs_use_native_trtmc_jsonl(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "prompts.jsonl").write_text(
        "\n".join(
            json.dumps({"sample_id": sample_id})
            for sample_id in ("sample-1", "sample-2")
        )
        + "\n",
        encoding="utf-8",
    )
    (work_dir / "trtfb_run.log").write_text(
        "$ python task_eval.py run-trtfb\n",
        encoding="utf-8",
    )
    commands = (
        {
            "sample_id": "sample-1",
            "command": ["trtmc", "segment-prompted", "model.trtfb", "--prompt", "cat"],
        },
        {
            "sample_id": "sample-2",
            "command": ["trtmc", "segment-prompted", "model.trtfb", "--prompt", "dog"],
        },
    )
    (work_dir / "trtfb_native_commands.jsonl").write_text(
        "".join(json.dumps(command) + "\n" for command in commands),
        encoding="utf-8",
    )

    reproduction = trtmc_validate._commands_from_logs(work_dir)

    assert reproduction["trtmc"] == [
        "trtmc segment-prompted model.trtfb --prompt cat"
    ]
    assert reproduction["command_count"]["trtmc"] == 2
    assert reproduction["command_logs"]["trtmc"] == [
        "trtfb_native_commands.jsonl"
    ]


def test_commands_from_logs_prefer_native_reference_jsonl(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "prompts.jsonl").write_text(
        json.dumps({"sample_id": "sample-1"}) + "\n",
        encoding="utf-8",
    )
    (work_dir / "hf_run.log").write_text(
        "$ python trtmc_reference.py run\n",
        encoding="utf-8",
    )
    (work_dir / "hf_native_run.log").write_text(
        "$ python plugin_reference.py\n",
        encoding="utf-8",
    )
    (work_dir / "hf_native_commands.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "command": ["python", "model_reference.py", "--prompt", "cat"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    reproduction = trtmc_validate._commands_from_logs(work_dir)

    assert reproduction["hf"] == ["python model_reference.py --prompt cat"]
    assert reproduction["command_count"]["hf"] == 1
    assert reproduction["command_logs"]["hf"] == [
        "hf_native_commands.jsonl"
    ]


@pytest.mark.parametrize(
    "line",
    [
        '{"sample_id":"sample","command":',
        "$ trtmc run NOT-JSON",
        '{"sample_id":"sample","command":[{}]}',
        '{"sample_id":"sample","command":[null]}',
        '{"sample_id":"sample","command":[1]}',
        json.dumps(
            {
                "sample_id": "sample",
                "command": ["trtmc", "bad\x00token"],
            }
        ),
        json.dumps(
            {
                "sample_id": "sample",
                "command": "trtmc run\x00hidden",
            }
        ),
        '{"command":["trtmc","run"]}',
        '{"sample_id":7,"command":["trtmc","run"]}',
        '{"sample_id":"","command":["trtmc","run"]}',
        json.dumps(
            {
                "sample_id": "sample\x00hidden",
                "command": ["trtmc", "run"],
            }
        ),
        '{"sample_id":"sample","command":["","argument"]}',
        '{"sample_id":"sample","command":["   ","argument"]}',
    ],
    ids=[
        "truncated",
        "shell-shortcut",
        "object-token",
        "null-token",
        "number-token",
        "nul-token",
        "nul-command",
        "missing-sample-id",
        "non-string-sample-id",
        "empty-sample-id",
        "nul-sample-id",
        "empty-executable",
        "whitespace-executable",
    ],
)
def test_native_command_jsonl_is_strict_json(tmp_path, line):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "trtfb_native_commands.jsonl").write_text(
        line + "\n",
        encoding="utf-8",
    )

    with pytest.raises(trtmc_validate.ValidationError):
        trtmc_validate._commands_from_logs(work_dir)


def test_native_command_jsonl_requires_utf8(tmp_path):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "trtfb_native_commands.jsonl").write_bytes(
        b'{"sample_id":"sample","command":["trtmc","'
        + b"\xff"
        + b'"]}\n'
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="not UTF-8",
    ):
        trtmc_validate._commands_from_logs(work_dir)


def test_command_log_discovery_enforces_entry_budget(
    tmp_path,
    monkeypatch,
):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    for index in range(4):
        (work_dir / f"artifact-{index}.txt").write_text(
            "data",
            encoding="utf-8",
        )
    monkeypatch.setattr(
        trtmc_validate,
        "MAX_COMMAND_LOG_DISCOVERY_ENTRIES",
        3,
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="scan exceeds 3 entries",
    ):
        trtmc_validate._secure_command_log_paths(work_dir)


def test_command_log_discovery_enforces_file_and_byte_budgets(
    tmp_path,
    monkeypatch,
):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    for index in range(3):
        (work_dir / f"run-{index}.log").write_text(
            "1234",
            encoding="utf-8",
        )
    monkeypatch.setattr(
        trtmc_validate,
        "MAX_COMMAND_LOG_FILES",
        2,
    )
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="log count exceeds 2",
    ):
        trtmc_validate._secure_command_log_paths(work_dir)

    monkeypatch.setattr(
        trtmc_validate,
        "MAX_COMMAND_LOG_FILES",
        128,
    )
    monkeypatch.setattr(
        trtmc_validate,
        "MAX_COMMAND_LOG_TOTAL_BYTES",
        8,
    )
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="logs exceed 8 bytes",
    ):
        trtmc_validate._secure_command_log_paths(work_dir)


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
        json.dumps({"sample_id": "sample-9", "command": reference_command}) + "\n",
        encoding="utf-8",
    )

    metadata = trtmc_disagreements.build_disagreement_artifact(
        work_dir=work_dir,
        case_dir=case_dir,
    )

    record = json.loads((case_dir / metadata["path"]).read_text(encoding="utf-8"))
    assert record["reproduce"]["trtmc"] == (
        "/workspace/build/trtmc run /runs/engines/model.trtfb --prompt Describe"
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
        case_dir=case_dir,
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
                    {"sample_id": row["sample_id"], "reason": "token_mismatch"} for row in prompts
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
                "returncode": 1,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "failed",
                    work_dir=str(work_dir),
                ),
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


def test_report_validates_disagreement_rows_beyond_inline_preview(
    tmp_path,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    artifact = case_dir / "disagreements.jsonl"
    artifact.write_text(
        "".join(
            json.dumps(
                _canonical_disagreement_record(
                    f"sample-{index}"
                )
            )
            + "\n"
            for index in range(
                trtmc_disagreements.INLINE_DISAGREEMENT_LIMIT
            )
        )
        + "NOT-JSON\n",
        encoding="utf-8",
    )
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "failed",
                "returncode": 1,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "failed",
                ),
                "disagreements": {
                    "count": (
                        trtmc_disagreements.INLINE_DISAGREEMENT_LIMIT + 1
                    ),
                    "path": "disagreements.jsonl",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="invalid validation disagreement artifact",
    ):
        trtmc_validate.write_report(tmp_path)


def test_report_rejects_disagreement_metadata_count_mismatch(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    (case_dir / "disagreements.jsonl").write_text(
        json.dumps(_canonical_disagreement_record("sample-1")) + "\n",
        encoding="utf-8",
    )
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "failed",
                "returncode": 1,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "failed",
                ),
                "disagreements": {
                    "count": 2,
                    "path": "disagreements.jsonl",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="artifact count is 1, expected 2",
    ):
        trtmc_validate.write_report(tmp_path)


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "count": -1,
            "path": "disagreements.jsonl",
        },
        {
            "count": True,
            "path": "disagreements.jsonl",
        },
        {
            "count": 1,
            "inline_limit": "not-an-integer",
            "path": "disagreements.jsonl",
        },
    ],
    ids=["negative-count", "boolean-count", "invalid-inline-limit"],
)
def test_report_rejects_invalid_disagreement_metadata(
    tmp_path,
    metadata,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    (case_dir / "disagreements.jsonl").write_text(
        "NOT-JSON\n",
        encoding="utf-8",
    )
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "failed",
                "disagreements": metadata,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="disagreement metadata",
    ):
        trtmc_validate.write_report(tmp_path)


def test_report_validates_zero_count_disagreement_artifact(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    (case_dir / "disagreements.jsonl").write_text(
        "NOT-JSON\n",
        encoding="utf-8",
    )
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "passed",
                "disagreements": {
                    "count": 0,
                    "path": "disagreements.jsonl",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="invalid validation disagreement artifact",
    ):
        trtmc_validate.write_report(tmp_path)


def test_report_render_failure_does_not_partially_publish_outputs(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    (tmp_path / "report.json").write_text(
        "OLD-JSON",
        encoding="utf-8",
    )
    (tmp_path / "report.html").write_text(
        "OLD-HTML",
        encoding="utf-8",
    )
    (case_dir / "disagreements.jsonl").write_text(
        "NOT-JSON\n",
        encoding="utf-8",
    )
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "failed",
                "disagreements": {
                    "count": 1,
                    "path": "disagreements.jsonl",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(trtmc_validate.ValidationError):
        trtmc_validate.write_report(tmp_path)

    assert (tmp_path / "report.json").read_text(
        encoding="utf-8"
    ) == "OLD-JSON"
    assert (tmp_path / "report.html").read_text(
        encoding="utf-8"
    ) == "OLD-HTML"


def test_disagreement_reproduction_paths_are_unique_for_colliding_sample_ids(
    tmp_path,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    work_dir = case_dir / "validation" / "workload-a" / "model-a"
    work_dir.mkdir(parents=True)
    sample_ids = ["a/b", "a?b", "x" * 300]
    image_paths = []
    for index, sample_id in enumerate(sample_ids):
        image_path = work_dir / f"input-{index}.png"
        image_path.write_bytes(f"IMAGE-{sample_id}".encode())
        image_paths.append(image_path)
    prompts = [
        {
            "sample_id": sample_id,
            "prompt": f"PROMPT-{index}",
            "images": [str(image_paths[index])],
        }
        for index, sample_id in enumerate(sample_ids)
    ]
    (work_dir / "prompts.jsonl").write_text(
        "".join(json.dumps(prompt) + "\n" for prompt in prompts),
        encoding="utf-8",
    )
    (work_dir / "summary.json").write_text(
        json.dumps(
            {
                "disagreements": [
                    {"sample_id": prompt["sample_id"]}
                    for prompt in prompts
                ]
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "trtfb_repro.json").write_text(
        json.dumps(
            {
                "command": [
                    "trtmc",
                    "run",
                    "model.trtfb",
                    "--input",
                    "{input_jsonl}",
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
                "returncode": 1,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "failed",
                    work_dir=str(work_dir),
                ),
            }
        ),
        encoding="utf-8",
    )

    _, _, report = trtmc_validate.write_report(tmp_path)

    artifact = case_dir / report["results"][0]["disagreements"]["path"]
    records = [
        json.loads(line)
        for line in artifact.read_text(encoding="utf-8").splitlines()
    ]
    paths = [
        case_dir / record["artifacts"]["trtmc_input"]
        for record in records
    ]
    media_paths = [
        case_dir / record["artifacts"]["media"][0]["path"]
        for record in records
    ]
    assert len(set(paths)) == len(sample_ids)
    assert len(set(media_paths)) == len(sample_ids)
    for index, (input_path, media_path) in enumerate(
        zip(paths, media_paths, strict=True)
    ):
        assert (
            json.loads(input_path.read_text(encoding="utf-8"))["prompt"]
            == f"PROMPT-{index}"
        )
        assert media_path.read_bytes() == image_paths[index].read_bytes()
        assert len(input_path.parent.name.encode()) < 255


def test_disagreement_media_is_capped_per_sample(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    work_dir = case_dir / "validation" / "workload-a" / "model-a"
    work_dir.mkdir(parents=True)
    images = []
    for index in range(
        trtmc_disagreements.MAX_MEDIA_FILES_PER_SAMPLE + 4
    ):
        image = work_dir / f"input-{index:02d}.png"
        image.write_bytes(f"IMAGE-{index}".encode())
        images.append(str(image))
    (work_dir / "prompts.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "images": images,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (work_dir / "summary.json").write_text(
        '{"disagreements":[{"sample_id":"sample"}]}',
        encoding="utf-8",
    )

    trtmc_disagreements.build_disagreement_artifact(
        work_dir=work_dir,
        case_dir=case_dir,
    )

    record = json.loads(
        (case_dir / "disagreements.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert (
        len(record["artifacts"]["media"])
        == trtmc_disagreements.MAX_MEDIA_FILES_PER_SAMPLE
    )


def test_frame_scan_consumes_only_the_configured_entry_budget(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "frames"
    root.mkdir()
    consumed = 0

    class FakeEntry:
        def __init__(self, index):
            self.name = f"{index:06d}.txt"

        def is_dir(self, *, follow_symlinks):
            assert follow_symlinks is False
            return False

        def is_file(self, *, follow_symlinks):
            assert follow_symlinks is False
            return False

    class FakeScandir:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return self

        def __next__(self):
            nonlocal consumed
            consumed += 1
            if (
                consumed
                > trtmc_disagreements.MAX_MEDIA_SCAN_ENTRIES * 2
            ):
                raise StopIteration
            return FakeEntry(consumed)

    monkeypatch.setattr(
        trtmc_disagreements.os,
        "scandir",
        lambda _descriptor: FakeScandir(),
    )

    assert (
        trtmc_disagreements._frame_candidates(
            "TRTMC",
            str(root),
            trusted_output_root=tmp_path,
        )
        == []
    )
    assert consumed == trtmc_disagreements.MAX_MEDIA_SCAN_ENTRIES


def test_anchored_frame_scan_rejects_directory_swap(tmp_path):
    work_dir = tmp_path / "work"
    frames = work_dir / "frames"
    frames.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    (external / "secret.png").write_bytes(b"SECRET")
    moved = work_dir / "original-frames"

    def swap_then_scan(root, maximum_entries):
        root.rename(moved)
        root.symlink_to(external, target_is_directory=True)
        return trtmc_validate._scan_disagreement_media(
            root,
            maximum_entries,
        )

    with pytest.raises(trtmc_validate.ValidationError):
        trtmc_disagreements._frame_candidates(
            "TRTMC",
            str(frames),
            trusted_output_root=work_dir,
            scan_artifacts=swap_then_scan,
        )


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
                "trtfb": {"samples": [{"sample_id": "sample-0", "passed": False}]},
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
    assert metadata["finished_at"] is None
    assert metadata["duration_seconds"] is None


def test_finalize_run_metadata_records_completion(monkeypatch, tmp_path):
    started_at = datetime(2026, 7, 25, 1, 2, 3, tzinfo=timezone.utc)
    finished_at = datetime(2026, 7, 25, 4, 4, 6, 500000, tzinfo=timezone.utc)
    (tmp_path / "run.json").write_text(
        json.dumps({"started_at": started_at.isoformat()}),
        encoding="utf-8",
    )
    monkeypatch.setattr(trtmc_validate, "_utc_now", lambda: finished_at)

    path = trtmc_validate.finalize_run_metadata(tmp_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))

    assert metadata["finished_at"] == finished_at.isoformat()
    assert metadata["duration_seconds"] == 10_923.5


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
        trtmc_validate.Binding(
            "model-a",
            "workload-a",
            reference_cache_identity="org/model/reference-contract-v1",
        ),
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
    assert command[command.index("--work-root") + 1] == str(tmp_path / "case" / "validation")
    assert command[command.index("--model") + 1] == "model-a"
    assert command[command.index("--suite") + 1] == "workload-a"
    assert command[command.index("--hf-python") + 1] == "/profiles/python"
    assert command[command.index("--reference-cache-dir") + 1] == str(
        tmp_path / "references"
    )
    assert command[
        command.index("--reference-cache-identity") + 1
    ] == "org/model/reference-contract-v1"
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
    summary = (
        arguments.output
        / "elf-b"
        / "elf-workload"
        / "validation"
        / "elf-workload"
        / "eval_summary.json"
    )
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "results": [
                    {"model": "elf-b", "status": "passed"}
                ]
            }
        ),
        encoding="utf-8",
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
        captured["summary_before_run"] = json.loads(
            summary.read_text(encoding="utf-8")
        )
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
        suites={"elf-workload": {}},
    )

    command = captured["command"]
    assert command[command.index("--elf-reference-repo") + 1] == str(selection.elf_reference_repo)
    assert captured["environment"]["EXTERNAL_REFERENCE_SENTINEL"] == "present"
    assert captured["summary_before_run"] == {"results": []}


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
                "error": ("min_prediction_agreement_rate actual=0.5 required=0.98"),
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
    if raw_result["status"] == "passed":
        raw_result = {
            **_raw_evidence(
                "model-a",
                "workload-a",
                "passed",
            ),
            **raw_result,
        }
    else:
        raw_result = {
            **raw_result,
            "model": "model-a",
            "suite": "workload-a",
        }
    result = trtmc_validate._normalize_result(
        {
            "model": "model-a",
            "workload": "workload-a",
            "status": raw_result["status"],
            "returncode": (
                0 if raw_result["status"] == "passed" else 1
            ),
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


@pytest.mark.parametrize(
    "results",
    [
        [{"model": "model-other", "status": "passed"}],
        [
            {"model": "model-a", "status": "passed"},
            {"model": "model-a", "status": "passed"},
        ],
        [
            {"model": "model-a", "status": "passed"},
            {"model": "model-other", "status": "failed"},
        ],
    ],
    ids=["wrong-model", "duplicate-model", "extra-model"],
)
def test_comparison_result_requires_one_exact_model_result(
    tmp_path,
    results,
):
    summary = (
        tmp_path
        / "validation"
        / "workload-a"
        / "eval_summary.json"
    )
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps({"results": results}),
        encoding="utf-8",
    )

    result = trtmc_validate._comparison_result(
        trtmc_validate.Binding("model-a", "workload-a"),
        case_dir=tmp_path,
        returncode=0,
        reference_environment=trtmc_validate.EnvironmentSelection(
            base_python="/profiles/python",
            names_and_paths=(),
            overrides={},
        ),
        dataset_command="python tools/trtmc_validate.py model-a",
    )

    assert result["execution"]["status"] == "error"
    assert result["comparison"]["status"] == "not_run"
    assert result["validation"]["status"] == "failed"
    assert result["raw_result"]["error_type"] == "ComparisonProcessError"


@pytest.mark.parametrize("reported_status", ["passed", "skipped"])
def test_comparison_result_rejects_success_from_crashed_process(
    tmp_path,
    reported_status,
):
    summary = (
        tmp_path
        / "validation"
        / "workload-a"
        / "eval_summary.json"
    )
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "model": "model-a",
                        "status": reported_status,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = trtmc_validate._comparison_result(
        trtmc_validate.Binding("model-a", "workload-a"),
        case_dir=tmp_path,
        returncode=9,
        reference_environment=trtmc_validate.EnvironmentSelection(
            base_python="/profiles/python",
            names_and_paths=(),
            overrides={},
        ),
        dataset_command="python tools/trtmc_validate.py model-a",
    )

    assert result["execution"] == {"status": "error", "exit_code": 9}
    assert result["comparison"]["status"] == "not_run"
    assert result["validation"]["status"] == "failed"
    assert result["raw_result"]["error_type"] == "ComparisonProcessError"
    assert "exited with code 9" in result["raw_result"]["error"]


def test_comparison_result_rejects_skipped_requested_comparison(tmp_path):
    summary = (
        tmp_path
        / "validation"
        / "workload-a"
        / "eval_summary.json"
    )
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "model": "model-a",
                        "status": "skipped",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = trtmc_validate._comparison_result(
        trtmc_validate.Binding("model-a", "workload-a"),
        case_dir=tmp_path,
        returncode=0,
        reference_environment=trtmc_validate.EnvironmentSelection(
            base_python="/profiles/python",
            names_and_paths=(),
            overrides={},
        ),
        dataset_command="python tools/trtmc_validate.py model-a",
    )

    assert result["execution"] == {"status": "error", "exit_code": 0}
    assert result["comparison"]["status"] == "not_run"
    assert result["validation"]["status"] == "failed"
    assert result["raw_result"]["error_type"] == "ComparisonProcessError"


def test_empty_task_eval_boundaries_publish_standard_failed_reports(
    tmp_path,
):
    summaries = {
        "time-series": (
            "time_series_parity",
            task_eval.compare_time_series_prediction_sets(
                {"responses": []},
                {"responses": []},
                gates={
                    "max_relative_l2": 0.01,
                    "max_absolute_error": 0.1,
                    "min_sample_agreement_rate": 1.0,
                },
            ),
        ),
        "encoder": (
            "encoder_embedding_parity",
            task_eval.compare_encoder_embedding_prediction_sets(
                {"responses": []},
                {"responses": []},
                gates={
                    "min_vector_cosine": 0.99,
                    "min_vector_pass_rate": 1.0,
                    "max_pair_cosine_abs_delta": 0.02,
                },
            ),
        ),
    }
    result_paths = []
    for model, (mode, summary) in summaries.items():
        assert summary["status"] == "failed"
        json.dumps(summary, allow_nan=False)
        case_dir = tmp_path / model / "workload-a"
        case_dir.mkdir(parents=True)
        result_path = case_dir / "comparison.json"
        result_path.write_text(
            json.dumps(
                {
                    "schema_version": "trtmc.validation-result/v2",
                    "model": model,
                    "workload": "workload-a",
                    "returncode": 1,
                    "raw_result": {
                        **summary,
                        "model": model,
                        "suite": "workload-a",
                        "mode": mode,
                    },
                },
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        result_paths.append(result_path)

    _, _, report = trtmc_validate.write_report(
        tmp_path,
        result_paths=result_paths,
    )

    assert report["validation_status"] == "failed"
    assert [result["validation"]["status"] for result in report["results"]] == [
        "failed",
        "failed",
    ]
    assert all(
        result["execution"]["status"] == "completed"
        for result in report["results"]
    )


@pytest.mark.parametrize(
    "error_type",
    ["WorkerProcessError", "ComparisonProcessError", "ValueError"],
)
def test_result_treats_error_type_only_as_execution_error(error_type):
    result = trtmc_validate._normalize_result(
        {
            "model": "model-a",
            "workload": "workload-a",
            "returncode": 1,
            "raw_result": {
                "status": "failed",
                "error_type": error_type,
            },
        }
    )

    assert result["execution"] == {"status": "error", "exit_code": 1}
    assert result["comparison"]["status"] == "not_run"
    assert result["validation"]["status"] == "failed"


def test_result_rejects_canonical_pass_that_conflicts_with_raw_error():
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="conflict with raw evidence",
    ):
        trtmc_validate._normalize_result(
            {
                "schema_version": "trtmc.validation-result/v2",
                "model": "model-a",
                "workload": "workload-a",
                "raw_result": {
                    "status": "failed",
                    "error": "backend crashed",
                },
                "execution": {
                    "status": "completed",
                    "exit_code": 0,
                },
                "comparison": {"status": "agreement"},
                "validation": {"status": "passed"},
            }
        )


def test_result_rejects_gate_failures_reported_as_pass():
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="conflict with raw evidence",
    ):
        trtmc_validate._normalize_result(
            {
                "model": "model-a",
                "workload": "workload-a",
                "raw_result": {
                    "status": "passed",
                    "error_type": "BenchmarkGateError",
                    "error": "gate failed",
                    "gate_failures": [{"gate": "minimum"}],
                },
                "execution": {
                    "status": "completed",
                    "exit_code": 0,
                },
                "comparison": {"status": "agreement"},
                "validation": {"status": "passed"},
            }
        )


@pytest.mark.parametrize(
    "gate_failures",
    [{"gate": "minimum"}, "minimum", 1, None],
)
def test_result_rejects_non_list_raw_gate_failures(gate_failures):
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="raw_result.gate_failures must be a list",
    ):
        trtmc_validate._normalize_result(
            {
                "model": "model-a",
                "workload": "workload-a",
                "returncode": 0,
                "raw_result": {
                    "status": "passed",
                    "gate_failures": gate_failures,
                },
            }
        )


@pytest.mark.parametrize("metrics", [[], "accuracy", 1, None])
def test_result_rejects_non_object_raw_metrics(metrics):
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="raw_result.metrics must be an object",
    ):
        trtmc_validate._normalize_result(
            {
                "model": "model-a",
                "workload": "workload-a",
                "returncode": 0,
                "raw_result": {
                    "status": "passed",
                    "metrics": metrics,
                },
            }
        )


@pytest.mark.parametrize("mode", [[], {}, 0, False])
def test_result_rejects_non_string_raw_mode(mode):
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="raw_result.mode must be a string",
    ):
        trtmc_validate._normalize_result(
            {
                "model": "model-a",
                "workload": "workload-a",
                "returncode": 0,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "passed",
                    mode=mode,
                ),
            }
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("passed_count", -1, "non-negative integer"),
        ("valid_count", 1.5, "non-negative integer"),
        ("prediction_agreement_rate", 2.5, r"must be in \[0, 1\]"),
        ("divergence_rate", -1.0, r"must be in \[0, 1\]"),
        ("mean_vector_cosine", 1.5, r"must be in \[-1, 1\]"),
        ("mean_relative_l2", -0.1, "must be non-negative"),
    ],
)
def test_result_rejects_out_of_range_raw_metrics(
    field,
    value,
    message,
):
    with pytest.raises(trtmc_validate.ValidationError, match=message):
        trtmc_validate._normalize_result(
            {
                "model": "model-a",
                "workload": "workload-a",
                "returncode": 0,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "passed",
                    **{field: value},
                ),
            }
        )


@pytest.mark.parametrize(
    ("metrics", "message"),
    [
        ({"score": 0.5}, "score must be an object"),
        (
            {"score": {"mean": "bad"}},
            "score.mean must be a finite number",
        ),
        (
            {"score": {"mean": 0.5, "min": 0.8}},
            "min <= mean <= max",
        ),
        (
            {"score": {"mean": 0.5, "count": -1}},
            "count must be a non-negative integer",
        ),
        (
            {
                "score": {
                    "mean": 0.5,
                    "count": 1,
                    "gated_count": -1,
                }
            },
            "gated_count must be a non-negative integer",
        ),
        (
            {
                "score": {
                    "mean": 0.5,
                    "count": 1,
                    "gated_count": 2,
                    "passed_count": 1,
                }
            },
            "gated_count cannot exceed count",
        ),
        (
            {
                "score": {
                    "mean": 0.5,
                    "count": 2,
                    "gated_count": 1,
                    "passed_count": 2,
                }
            },
            "passed_count cannot exceed gated_count",
        ),
    ],
)
def test_result_rejects_malformed_nested_raw_metrics(
    metrics,
    message,
):
    with pytest.raises(trtmc_validate.ValidationError, match=message):
        trtmc_validate._normalize_result(
            {
                "model": "model-a",
                "workload": "workload-a",
                "returncode": 0,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "passed",
                    metrics=metrics,
                ),
            }
        )


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (
            {
                "passed_count": 2,
                "valid_count": 1,
            },
            "passed_count cannot exceed valid_count",
        ),
        (
            {
                "passed_count": 1,
                "valid_count": 2,
                "overall_pass_rate": 1.0,
            },
            "overall_pass_rate conflicts",
        ),
        (
            {
                "overall_pass_rate": 1.0,
                "metrics": {
                    "overall_pass_rate": {
                        "mean": 0.0,
                    }
                },
            },
            "raw metric conflicts with nested mean",
        ),
        (
            {
                "hf_accuracy": 0.8,
                "trtfb_accuracy": 0.7,
                "accuracy_delta_trtfb_minus_hf": 0.2,
            },
            "accuracy_delta_trtfb_minus_hf conflicts",
        ),
        (
            {
                "hf_mean_ground_truth_iou": 0.8,
                "trtfb_mean_ground_truth_iou": 0.7,
                "ground_truth_iou_drop_from_hf": -0.1,
            },
            "ground_truth_iou_drop_from_hf conflicts",
        ),
    ],
)
def test_result_rejects_internally_inconsistent_raw_metrics(
    extra,
    message,
):
    with pytest.raises(trtmc_validate.ValidationError, match=message):
        trtmc_validate._normalize_result(
            {
                "model": "model-a",
                "workload": "workload-a",
                "returncode": 0,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "passed",
                    **extra,
                ),
            }
        )


@pytest.mark.parametrize(
    ("mode", "primary_metric"),
    sorted(trtmc_validate._PRIMARY_METRIC_BY_MODE.items()),
)
def test_passed_known_mode_requires_its_primary_metric(
    mode,
    primary_metric,
):
    with pytest.raises(
        trtmc_validate.ValidationError,
        match=rf"raw_result\.{primary_metric}",
    ):
        trtmc_validate._normalize_result(
            {
                "schema_version": "trtmc.validation-result/v2",
                "model": "model-a",
                "workload": "workload-a",
                "returncode": 0,
                "raw_result": {
                    "model": "model-a",
                    "suite": "workload-a",
                    "status": "passed",
                    "mode": mode,
                },
            }
        )


def test_passed_prompted_segmentation_requires_ground_truth_gate_metrics():
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="missing raw metric evidence",
    ):
        trtmc_validate._normalize_result(
            {
                "schema_version": "trtmc.validation-result/v2",
                "model": "model-a",
                "workload": "workload-a",
                "returncode": 0,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "passed",
                    mode="prompted_segmentation_parity",
                    mean_backend_mask_iou=1.0,
                ),
            }
        )


@pytest.mark.parametrize(
    "counts",
    [
        {
            "overall_pass_rate": 0.0,
            "passed_count": 0,
            "valid_count": 0,
            "skipped_count": 0,
        },
        {
            "overall_pass_rate": 1.0,
            "passed_count": 1,
            "valid_count": 1,
            "skipped_count": 9,
        },
        {
            "overall_pass_rate": 0.5,
            "passed_count": 1,
            "valid_count": 2,
            "skipped_count": 0,
        },
    ],
)
def test_passed_diffusion_result_requires_consistent_nonempty_counts(
    counts,
):
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="passed diffusion comparison",
    ):
        trtmc_validate._normalize_result(
            {
                "schema_version": "trtmc.validation-result/v2",
                "model": "model-a",
                "workload": "workload-a",
                "returncode": 0,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "passed",
                    mode="diffusion_image_clip_parity",
                    **counts,
                ),
            }
        )


def test_passed_continuation_result_rejects_contradictory_divergence():
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="exact_count plus divergent_count",
    ):
        trtmc_validate._normalize_result(
            {
                "schema_version": "trtmc.validation-result/v2",
                "model": "model-a",
                "workload": "workload-a",
                "returncode": 0,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "passed",
                    mode="continuation",
                    count=1,
                    exact_count=1,
                    divergent_count=1,
                    exact_match_rate=1.0,
                    divergence_rate=0.0,
                    token_prefix_agreement=1.0,
                ),
            }
        )


def test_diagnostic_only_continuation_cannot_publish_passed_agreement():
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="diagnostic-only continuation evidence",
    ):
        trtmc_validate._normalize_result(
            {
                "schema_version": "trtmc.validation-result/v2",
                "model": "model-a",
                "workload": "workload-a",
                "returncode": 0,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "passed",
                    mode="continuation",
                    evaluation_policy="diagnostic_only",
                    gates={},
                    count=1,
                    exact_count=0,
                    divergent_count=1,
                    exact_match_rate=0.0,
                    divergence_rate=1.0,
                    token_prefix_agreement=0.0,
                ),
            }
        )


@pytest.mark.parametrize(
    "extra",
    [
        {
            "mode": "encoder_embedding_parity",
            "vector_pass_rate": 0.0,
            "min_vector_cosine": 0.0,
            "max_pair_cosine_abs_delta": 1.0,
            "gates": {
                "min_vector_cosine": 0.999,
                "min_vector_pass_rate": 1.0,
                "max_pair_cosine_abs_delta": 0.02,
            },
        },
        {
            "mode": "reranking_parity",
            "mean_pairwise_ordering_agreement": 0.0,
            "sample_pass_rate": 0.0,
            "metrics": {
                "pairwise_ordering_agreement": {
                    "mean": 0.0,
                    "min": 0.0,
                }
            },
            "gates": {
                "pairwise_ordering_agreement": 1.0,
                "min_sample_pass_rate": 1.0,
            },
        },
        {
            "mode": "time_series_parity",
            "sample_agreement_rate": 0.0,
            "mean_relative_l2": 1.0,
            "max_relative_l2": 1.0,
            "max_absolute_error": 1.0,
            "gates": {
                "min_sample_agreement_rate": 1.0,
                "max_relative_l2": 0.01,
                "max_absolute_error": 0.1,
            },
        },
        {
            "mode": "image_classification_parity",
            "top1_agreement": 0.0,
            "top1_accuracy_drop_from_hf": 1.0,
            "gates": {
                "min_top1_agreement": 0.98,
                "max_top1_accuracy_drop_from_hf": 0.01,
            },
        },
        {
            "mode": "semantic_segmentation_parity",
            "backend_pixel_agreement": 0.0,
            "backend_mean_iou": 0.0,
            "mean_iou_drop_from_hf": 1.0,
            "gates": {
                "min_backend_pixel_agreement": 0.98,
                "min_backend_mean_iou": 0.95,
                "max_mean_iou_drop_from_hf": 0.01,
            },
        },
        {
            "mode": "prompted_segmentation_parity",
            "mean_backend_mask_iou": 0.0,
            "hf_mean_ground_truth_iou": 1.0,
            "trtfb_mean_ground_truth_iou": 0.0,
            "ground_truth_iou_drop_from_hf": 1.0,
            "gates": {
                "min_backend_mask_iou": 0.7,
                "max_ground_truth_iou_drop_from_hf": 0.05,
            },
        },
        {
            "mode": "diffusion_text_parity",
            "token_agreement_rate": 0.0,
            "shared_sampling_inputs_match_rate": 0.0,
            "corpus_bleu_abs_delta": 1.0,
            "gates": {
                "max_corpus_bleu_abs_delta": 0.5,
                "min_shared_sampling_inputs_match_rate": 1.0,
            },
        },
        {
            "mode": "diffusion_image_clip_parity",
            "overall_pass_rate": 1.0,
            "passed_count": 1,
            "valid_count": 1,
            "skipped_count": 0,
            "total_count": 1,
            "metrics": {
                "pixel_mean": {
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "count": 1,
                    "gated_count": 1,
                    "passed_count": 1,
                }
            },
            "gates": {"min_pixel_mean": 0.15},
        },
        {
            "mode": "continuation",
            "evaluation_policy": "threshold_gated",
            "count": 1,
            "exact_count": 0,
            "divergent_count": 1,
            "exact_match_rate": 0.0,
            "divergence_rate": 1.0,
            "token_prefix_agreement": 0.0,
            "gates": {"min_token_prefix_agreement": 0.98},
        },
    ],
    ids=[
        "encoder",
        "reranking",
        "time-series",
        "image-classification",
        "semantic-segmentation",
        "prompted-segmentation",
        "diffusion-text",
        "diffusion-image",
        "continuation",
    ],
)
def test_passed_result_recomputes_supported_mode_gates(extra):
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="violates raw_result.gates",
    ):
        trtmc_validate._normalize_result(
            {
                "schema_version": "trtmc.validation-result/v2",
                "model": "model-a",
                "workload": "workload-a",
                "returncode": 0,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "passed",
                    **extra,
                ),
            }
        )


@pytest.mark.parametrize(
    "extra",
    [
        {
            "mode": "encoder_embedding_parity",
            "vector_pass_rate": 1.0,
            "min_vector_cosine": 0.999,
            "max_pair_cosine_abs_delta": 0.01,
            "gates": {
                "min_vector_cosine": 0.999,
                "min_vector_pass_rate": 1.0,
                "max_pair_cosine_abs_delta": 0.02,
            },
        },
        {
            "mode": "reranking_parity",
            "mean_pairwise_ordering_agreement": 1.0,
            "sample_pass_rate": 1.0,
            "metrics": {
                "pairwise_ordering_agreement": {
                    "mean": 1.0,
                    "min": 1.0,
                }
            },
            "gates": {
                "pairwise_ordering_agreement": 1.0,
                "min_sample_pass_rate": 1.0,
            },
        },
        {
            "mode": "time_series_parity",
            "sample_agreement_rate": 1.0,
            "mean_relative_l2": 0.001,
            "max_relative_l2": 0.005,
            "max_absolute_error": 0.05,
            "gates": {
                "min_sample_agreement_rate": 1.0,
                "max_relative_l2": 0.01,
                "max_absolute_error": 0.1,
            },
        },
    ],
    ids=["encoder", "reranking", "time-series"],
)
def test_passed_result_accepts_satisfied_specialized_gates(extra):
    result = trtmc_validate._normalize_result(
        {
            "schema_version": "trtmc.validation-result/v2",
            "model": "model-a",
            "workload": "workload-a",
            "returncode": 0,
            "raw_result": _raw_evidence(
                "model-a",
                "workload-a",
                "passed",
                **extra,
            ),
        }
    )

    assert result["validation"]["status"] == "passed"


@pytest.mark.parametrize(
    ("counterexample", "message"),
    [
        (
            "missing-complete-count",
            "valid_count and sample_count evidence",
        ),
        ("explicit-skipped-sample", "zero skipped samples"),
        ("weakened-threshold", "changed thresholds"),
        ("unknown-gate", "unknown gates"),
    ],
)
def test_passed_raw_evidence_rejects_incomplete_counts_or_untrusted_gates(
    counterexample,
    message,
):
    authoritative_gates = {
        "min_vector_cosine": 0.99,
        "min_vector_pass_rate": 1.0,
        "max_pair_cosine_abs_delta": 0.02,
    }
    raw_result = _raw_evidence(
        "model-a",
        "workload-a",
        "passed",
        mode="encoder_embedding_parity",
        vector_pass_rate=1.0,
        min_vector_cosine=1.0,
        max_pair_cosine_abs_delta=0.0,
        gates=dict(authoritative_gates),
    )
    if counterexample == "missing-complete-count":
        raw_result.pop("sample_count")
    elif counterexample == "explicit-skipped-sample":
        raw_result["skipped_count"] = 1
    elif counterexample == "weakened-threshold":
        raw_result["gates"]["min_vector_pass_rate"] = 0.5
    else:
        raw_result["gates"]["min_untrusted_score"] = 0.0

    with pytest.raises(
        trtmc_validate.ValidationError,
        match=message,
    ):
        trtmc_validate._normalize_result(
            {
                "schema_version": "trtmc.validation-result/v2",
                "model": "model-a",
                "workload": "workload-a",
                "returncode": 0,
                "raw_result": raw_result,
            },
            expected_gates=authoritative_gates,
        )


def test_report_rejects_passed_supported_mode_without_gate_context(
    tmp_path,
    monkeypatch,
):
    model = "model-a"
    workload = "workload-a"
    case_dir = tmp_path / model / workload
    case_dir.mkdir(parents=True)
    comparison = case_dir / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "schema_version": "trtmc.validation-result/v2",
                "model": model,
                "workload": workload,
                "returncode": 0,
                "raw_result": _raw_evidence(
                    model,
                    workload,
                    "passed",
                ),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        trtmc_validate,
        "_default_authoritative_gates_by_binding",
        lambda: {},
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="no authoritative gate configuration",
    ):
        trtmc_validate.write_report(
            tmp_path,
            result_paths=[comparison],
        )


def test_report_rejects_threshold_gated_continuation_without_gate_context(
    tmp_path,
    monkeypatch,
):
    model = "model-a"
    workload = "workload-a"
    case_dir = tmp_path / model / workload
    case_dir.mkdir(parents=True)
    comparison = case_dir / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "schema_version": "trtmc.validation-result/v2",
                "model": model,
                "workload": workload,
                "returncode": 0,
                "raw_result": _raw_evidence(
                    model,
                    workload,
                    "passed",
                    mode="continuation",
                    evaluation_policy="threshold_gated",
                    count=1,
                    exact_count=1,
                    divergent_count=0,
                    exact_match_rate=1.0,
                    divergence_rate=0.0,
                    token_prefix_agreement=1.0,
                    gates={
                        "min_token_prefix_agreement": 0.98,
                    },
                ),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        trtmc_validate,
        "_default_authoritative_gates_by_binding",
        lambda: {},
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="no authoritative gate configuration",
    ):
        trtmc_validate.write_report(
            tmp_path,
            result_paths=[comparison],
        )


def test_report_rechecks_raw_gates_against_authoritative_binding(
    tmp_path,
    monkeypatch,
):
    model = "model-a"
    workload = "workload-a"
    case_dir = tmp_path / model / workload
    case_dir.mkdir(parents=True)
    expected_gates = {
        "max_accuracy_drop_from_hf": 0.0,
        "min_prediction_agreement": 1.0,
    }
    raw_result = _raw_evidence(
        model,
        workload,
        "passed",
        gates={
            **expected_gates,
            "min_untrusted_score": 0.0,
        },
    )
    comparison = case_dir / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "schema_version": "trtmc.validation-result/v2",
                "model": model,
                "workload": workload,
                "returncode": 0,
                "raw_result": raw_result,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        trtmc_validate,
        "_default_authoritative_gates_by_binding",
        lambda: {(model, workload): expected_gates},
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="unknown gates",
    ):
        trtmc_validate.write_report(
            tmp_path,
            result_paths=[comparison],
        )


def test_real_time_series_producer_count_reaches_validation(
    tmp_path,
):
    gates = {
        "max_relative_l2": 0.01,
        "max_absolute_error": 0.1,
        "min_sample_agreement_rate": 1.0,
    }
    predictions = {
        "responses": [
            {
                "sample_id": "one",
                "output_values": [1.0, 2.0],
                "output_shape": [2],
            }
        ]
    }
    producer_summary = task_eval.compare_time_series_prediction_sets(
        predictions,
        predictions,
        gates=gates,
    )
    assert producer_summary["status"] == "passed"
    assert producer_summary["sample_count"] == 1

    validation_dir = tmp_path / "validation" / "workload-a"
    validation_dir.mkdir(parents=True)
    raw_result = {
        "model": "model-a",
        "suite": "workload-a",
        "mode": "time_series_parity",
        "status": producer_summary["status"],
        "sample_count": producer_summary["sample_count"],
        "valid_count": producer_summary["valid_count"],
        "passed_count": producer_summary["passed_count"],
        "sample_agreement_rate": producer_summary[
            "sample_agreement_rate"
        ],
        "mean_relative_l2": producer_summary["mean_relative_l2"],
        "max_relative_l2": producer_summary["max_relative_l2"],
        "max_absolute_error": producer_summary["max_absolute_error"],
        "gates": producer_summary["gates"],
    }
    (validation_dir / "eval_summary.json").write_text(
        json.dumps({"results": [raw_result]}),
        encoding="utf-8",
    )

    result = trtmc_validate._comparison_result(
        trtmc_validate.Binding("model-a", "workload-a"),
        case_dir=tmp_path,
        returncode=0,
        reference_environment=trtmc_validate.EnvironmentSelection(
            base_python="/profiles/python",
            names_and_paths=(),
            overrides={},
        ),
        dataset_command="python tools/trtmc_validate.py model-a",
        expected_gates=gates,
    )

    assert result["validation"]["status"] == "passed"
    assert result["raw_result"]["sample_count"] == 1
    assert result["raw_result"]["valid_count"] == 1


def test_real_mcq_producer_status_reaches_canonical_report(tmp_path):
    answers = {
        "requests": [
            {
                "answer": "A",
                "subject": "subject",
            }
        ]
    }
    hf = {"responses": [{"sample_id": "one", "output_text": "A"}]}
    gates = {
        "max_accuracy_drop_from_hf": 0.01,
        "min_prediction_agreement": 0.98,
    }
    expected = {
        "matching": (hf, "passed"),
        "divergent": (
            {"responses": [{"sample_id": "one", "output_text": "B"}]},
            "failed",
        ),
    }

    for label, (trtfb, expected_status) in expected.items():
        output = tmp_path / label
        model = f"model-{label}"
        workload = "mmlu_five_shot_mcq"
        case_dir = output / model / workload
        case_dir.mkdir(parents=True)
        raw_result = task_eval.build_reference_comparison_result(
            base_result={"model": model, "suite": workload},
            scorer="mcq",
            summary=task_eval.compare_prediction_sets(
                hf,
                trtfb,
                answers,
                scorer="mcq",
            ),
            gates=gates,
        )
        (case_dir / "comparison.json").write_text(
            json.dumps(
                {
                    "schema_version": "trtmc.validation-result/v2",
                    "model": model,
                    "workload": workload,
                    "returncode": (
                        0 if raw_result["status"] == "passed" else 1
                    ),
                    "raw_result": raw_result,
                }
            ),
            encoding="utf-8",
        )

        _, _, report = trtmc_validate.write_report(
            output,
            expected_gates_by_binding={
                (model, workload): gates,
            },
        )

        assert report["results"][0]["validation"]["status"] == expected_status


def test_result_rejects_conflicting_outer_and_raw_status():
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="legacy status conflicts with raw_result.status",
    ):
        trtmc_validate._normalize_result(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "failed",
                "returncode": 0,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "passed",
                ),
            }
        )


def test_result_does_not_pass_without_integer_exit_evidence():
    result = trtmc_validate._normalize_result(
        {
            "schema_version": "trtmc.validation-result/v2",
            "model": "model-a",
            "workload": "workload-a",
            "raw_result": {"status": "passed"},
        }
    )

    assert result["execution"] == {
        "status": "error",
        "exit_code": None,
    }
    assert result["validation"] == {"status": "failed"}


def test_result_completed_requires_exact_raw_binding():
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="must include exact raw model and suite",
    ):
        trtmc_validate._normalize_result(
            {
                "schema_version": "trtmc.validation-result/v2",
                "model": "model-a",
                "workload": "workload-a",
                "raw_result": {"status": "passed"},
                "execution": {
                    "status": "completed",
                    "exit_code": 0,
                },
                "comparison": {"status": "agreement"},
                "validation": {"status": "passed"},
            }
        )


@pytest.mark.parametrize("returncode", [[], {}])
def test_result_rejects_malformed_legacy_returncode(returncode):
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="returncode must be an integer or null",
    ):
        trtmc_validate._normalize_result(
            {
                "model": "model-a",
                "workload": "workload-a",
                "returncode": returncode,
                "raw_result": {"status": "passed"},
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("error_type", 0),
        ("error", {}),
        ("exception", []),
        ("traceback", {}),
        ("failure_class", False),
    ],
)
def test_result_rejects_non_string_raw_error_fields(field, value):
    with pytest.raises(
        trtmc_validate.ValidationError,
        match=rf"raw_result.{field} must be a string",
    ):
        trtmc_validate._normalize_result(
            {
                "model": "model-a",
                "workload": "workload-a",
                "returncode": 0,
                "raw_result": {
                    "status": "passed",
                    field: value,
                },
            }
        )


@pytest.mark.parametrize("status", ["banana", "success", "error"])
def test_result_rejects_unknown_raw_status(status):
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="raw_result.status must be one of",
    ):
        trtmc_validate._normalize_result(
            {
                "schema_version": "trtmc.validation-result/v2",
                "model": "model-a",
                "workload": "workload-a",
                "returncode": 0,
                "raw_result": {
                    "model": "model-a",
                    "suite": "workload-a",
                    "status": status,
                },
            }
        )


def test_result_rejects_non_string_canonical_comparison_mode():
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="comparison.mode must be a string",
    ):
        trtmc_validate._normalize_result(
            {
                "schema_version": "trtmc.validation-result/v2",
                "model": "model-a",
                "workload": "workload-a",
                "returncode": 0,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "passed",
                ),
                "comparison": {"mode": []},
            }
        )


@pytest.mark.parametrize(
    "raw_result",
    [{"mode": "fabricated"}, {}, None],
)
def test_result_v2_rejects_missing_raw_comparison_evidence(raw_result):
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="v2 runnable result must include a non-empty raw_result",
    ):
        trtmc_validate._normalize_result(
            {
                "schema_version": "trtmc.validation-result/v2",
                "model": "model-a",
                "workload": "workload-a",
                "status": "passed",
                "returncode": 0,
                "raw_result": raw_result,
            }
        )


def test_result_v2_rejects_canonical_only_runnable_pass():
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="v2 runnable result must include a non-empty raw_result",
    ):
        trtmc_validate._normalize_result(
            {
                "schema_version": "trtmc.validation-result/v2",
                "model": "model-a",
                "workload": "workload-a",
                "execution": {
                    "status": "completed",
                    "exit_code": 0,
                },
                "comparison": {"status": "agreement"},
                "validation": {"status": "passed"},
            }
        )


def test_result_rejects_legacy_canonical_only_runnable_pass():
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="must include raw comparison evidence",
    ):
        trtmc_validate._normalize_result(
            {
                "model": "model-a",
                "workload": "workload-a",
                "execution": {
                    "status": "completed",
                    "exit_code": 0,
                },
                "comparison": {"status": "agreement"},
                "validation": {"status": "passed"},
            }
        )


def test_result_legacy_migration_is_idempotent():
    migrated = trtmc_validate._normalize_result(
        {
            "model": "model-a",
            "workload": "workload-a",
            "status": "passed",
            "returncode": 0,
            "raw_result": _raw_evidence(
                "model-a",
                "workload-a",
                "passed",
            ),
        }
    )

    assert migrated["schema_version"] == "trtmc.validation-result/v2"
    assert migrated["raw_result"] == _raw_evidence(
        "model-a",
        "workload-a",
        "passed",
    )
    assert trtmc_validate._normalize_result(migrated) == migrated


@pytest.mark.parametrize("raw_result", [[], "passed", 1])
def test_result_rejects_non_object_raw_evidence(raw_result):
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="raw_result must be an object or null",
    ):
        trtmc_validate._normalize_result(
            {
                "schema_version": "trtmc.validation-result/v2",
                "model": "model-a",
                "workload": "workload-a",
                "raw_result": raw_result,
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "model-b"),
        ("suite", "workload-b"),
        ("workload", "workload-b"),
    ],
)
def test_write_report_rejects_raw_binding_mismatch(
    tmp_path,
    field,
    value,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    comparison = case_dir / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "schema_version": "trtmc.validation-result/v2",
                "model": "model-a",
                "workload": "workload-a",
                "returncode": 0,
                "raw_result": {
                    "status": "passed",
                    field: value,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match=rf"raw_result.{field} conflicts with the canonical binding",
    ):
        trtmc_validate.write_report(
            tmp_path,
            result_paths=[comparison],
        )

    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "report.html").exists()


def test_comparison_result_rejects_invalid_summary_status(tmp_path):
    summary = (
        tmp_path
        / "validation"
        / "workload-a"
        / "eval_summary.json"
    )
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "results": [
                    {"model": "model-a", "status": "unknown"}
                ]
            }
        ),
        encoding="utf-8",
    )

    result = trtmc_validate._comparison_result(
        trtmc_validate.Binding("model-a", "workload-a"),
        case_dir=tmp_path,
        returncode=0,
        reference_environment=trtmc_validate.EnvironmentSelection(
            base_python="/profiles/python",
            names_and_paths=(),
            overrides={},
        ),
        dataset_command="python tools/trtmc_validate.py model-a",
    )

    assert result["execution"]["status"] == "error"
    assert result["comparison"]["status"] == "not_run"
    assert result["validation"]["status"] == "failed"
    assert "invalid status" in result["raw_result"]["error"]


@pytest.mark.parametrize(
    "payload",
    [
        "NOT-JSON",
        json.dumps([]),
        json.dumps({"results": {}}),
        json.dumps(
            {
                "results": [
                    {
                        "model": "model-a",
                        "suite": "workload-a",
                        "status": "passed",
                        "gate_failures": {},
                    }
                ]
            }
        ),
    ],
    ids=[
        "invalid-json",
        "non-object-summary",
        "non-list-results",
        "malformed-model-evidence",
    ],
)
def test_comparison_result_classifies_malformed_output_as_process_error(
    tmp_path,
    payload,
):
    summary = (
        tmp_path
        / "validation"
        / "workload-a"
        / "eval_summary.json"
    )
    summary.parent.mkdir(parents=True)
    summary.write_text(payload, encoding="utf-8")

    result = trtmc_validate._comparison_result(
        trtmc_validate.Binding("model-a", "workload-a"),
        case_dir=tmp_path,
        returncode=0,
        reference_environment=trtmc_validate.EnvironmentSelection(
            base_python="/profiles/python",
            names_and_paths=(),
            overrides={},
        ),
        dataset_command="python tools/trtmc_validate.py model-a",
    )

    assert result["execution"] == {
        "status": "error",
        "exit_code": 0,
    }
    assert result["comparison"]["status"] == "not_run"
    assert result["validation"]["status"] == "failed"
    assert result["raw_result"]["error_type"] == (
        "ComparisonProcessError"
    )


def test_write_report_rolls_back_all_outputs_after_commit_failure(
    tmp_path,
    monkeypatch,
):
    original_case_artifacts = {}
    result_paths = []
    for suffix in ("a", "b"):
        model = f"model-{suffix}"
        workload = f"workload-{suffix}"
        case_dir = tmp_path / model / workload
        work_dir = case_dir / "validation" / workload / model
        work_dir.mkdir(parents=True)
        repro = case_dir / "repro"
        repro.mkdir()
        (repro / "old.txt").write_text(
            f"OLD-REPRO-{suffix}",
            encoding="utf-8",
        )
        disagreements = case_dir / "disagreements.jsonl"
        disagreements.write_text(
            f"OLD-DISAGREEMENTS-{suffix}\n",
            encoding="utf-8",
        )
        comparison = case_dir / "comparison.json"
        comparison.write_text(
            json.dumps(
                {
                    "model": model,
                    "workload": workload,
                    "status": "passed",
                    "raw_result": {
                        "status": "passed",
                        "work_dir": str(work_dir),
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        original_case_artifacts[case_dir] = {
            "comparison": comparison.read_bytes(),
            "disagreements": disagreements.read_bytes(),
            "repro": (repro / "old.txt").read_bytes(),
        }
        result_paths.append(comparison)

    report_json = tmp_path / "report.json"
    report_html = tmp_path / "report.html"
    report_json.write_bytes(b"OLD-REPORT-JSON")
    report_html.write_bytes(b"OLD-REPORT-HTML")
    original_commit = trtmc_validate._commit_file_update
    commit_count = 0

    def fail_after_every_file_was_committed(update):
        nonlocal commit_count
        original_commit(update)
        commit_count += 1
        if commit_count == 6:
            raise OSError("injected file commit failure")

    monkeypatch.setattr(
        trtmc_validate,
        "_commit_file_update",
        fail_after_every_file_was_committed,
    )

    with pytest.raises(OSError, match="injected file commit failure"):
        trtmc_validate.write_report(
            tmp_path,
            result_paths=result_paths,
        )

    assert commit_count == 6
    assert report_json.read_bytes() == b"OLD-REPORT-JSON"
    assert report_html.read_bytes() == b"OLD-REPORT-HTML"
    for case_dir, expected in original_case_artifacts.items():
        assert (
            case_dir / "comparison.json"
        ).read_bytes() == expected["comparison"]
        assert (
            case_dir / "disagreements.jsonl"
        ).read_bytes() == expected["disagreements"]
        assert (case_dir / "repro" / "old.txt").read_bytes() == expected["repro"]
        assert not any(
            child.name.startswith(
                (
                    ".report-stage-",
                    ".repro-next.",
                    ".repro-previous.",
                )
            )
            for child in case_dir.iterdir()
        )


def test_write_report_isolates_cleanup_errors_and_closes_all_anchors(
    tmp_path,
    monkeypatch,
):
    result_paths = []
    for suffix in ("a", "b"):
        model = f"model-{suffix}"
        workload = f"workload-{suffix}"
        case_dir = tmp_path / model / workload
        case_dir.mkdir(parents=True)
        comparison = case_dir / "comparison.json"
        comparison.write_text(
            json.dumps(
                {
                    "model": model,
                    "workload": workload,
                    "status": "passed",
                }
            ),
            encoding="utf-8",
        )
        result_paths.append(comparison)
    (tmp_path / "report.json").write_bytes(b"OLD-REPORT")
    (tmp_path / "report.html").write_bytes(b"OLD-HTML")

    cleanup_phase = False
    injected = False
    finalize_calls = 0
    real_verify = trtmc_validate._verify_report_transaction_visibility
    real_digest = trtmc_validate._regular_file_digest_at
    real_finalize = trtmc_validate._finalize_file_update

    def verify_then_mark_cleanup(entries):
        nonlocal cleanup_phase
        real_verify(entries)
        cleanup_phase = True

    def fail_first_cleanup_digest(directory_fd, name, *, maximum_bytes):
        nonlocal injected
        if (
            cleanup_phase
            and not injected
            and ".previous." in name
        ):
            injected = True
            raise OSError("injected cleanup digest failure")
        return real_digest(
            directory_fd,
            name,
            maximum_bytes=maximum_bytes,
        )

    def count_finalize(update):
        nonlocal finalize_calls
        finalize_calls += 1
        return real_finalize(update)

    monkeypatch.setattr(
        trtmc_validate,
        "_verify_report_transaction_visibility",
        verify_then_mark_cleanup,
    )
    monkeypatch.setattr(
        trtmc_validate,
        "_regular_file_digest_at",
        fail_first_cleanup_digest,
    )
    monkeypatch.setattr(
        trtmc_validate,
        "_finalize_file_update",
        count_finalize,
    )
    descriptors_before = len(os.listdir("/proc/self/fd"))

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="validation report cleanup incomplete",
    ):
        trtmc_validate.write_report(
            tmp_path,
            result_paths=result_paths,
        )

    descriptors_after = len(os.listdir("/proc/self/fd"))
    assert injected
    assert finalize_calls == 4
    assert descriptors_after == descriptors_before
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.html").is_file()
    preserved = [
        path
        for path in tmp_path.rglob("*")
        if path.name.startswith(".comparison.json.previous.")
        or path.name.startswith(".report.json.previous.")
        or path.name.startswith(".report.html.previous.")
    ]
    assert len(preserved) == 1
    _assert_output_lock_available(tmp_path)


def test_file_transaction_rejects_original_swap_before_backup(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "report.json"
    target.write_bytes(b"ORIGINAL")
    original_away = tmp_path / "original-away.json"
    update = trtmc_validate._prepare_file_update(
        target,
        b"TRANSACTION",
    )
    original_verify = trtmc_validate._verify_file_update_target
    swapped = False

    def verify_then_swap(candidate):
        nonlocal swapped
        original_verify(candidate)
        if not swapped:
            target.rename(original_away)
            target.write_bytes(b"CONCURRENT")
            swapped = True

    monkeypatch.setattr(
        trtmc_validate,
        "_verify_file_update_target",
        verify_then_swap,
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="changed while moving to backup",
    ):
        trtmc_validate._commit_file_update(update)
    trtmc_validate._rollback_file_update(update)
    trtmc_validate._finalize_file_update(update)

    assert swapped
    assert target.read_bytes() == b"CONCURRENT"
    assert original_away.read_bytes() == b"ORIGINAL"
    assert not any(
        child.name.startswith(
            (".report.json.previous.", ".report.json.rollback.")
        )
        for child in tmp_path.iterdir()
    )


def test_directory_transaction_rejects_original_swap_before_backup(
    tmp_path,
    monkeypatch,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    repro = case_dir / "repro"
    repro.mkdir(parents=True)
    (repro / "original.txt").write_bytes(b"ORIGINAL")
    stage = trtmc_validate._create_case_artifact_stage(case_dir)
    staged_repro = stage.path / "repro"
    staged_repro.mkdir()
    (staged_repro / "new.txt").write_bytes(b"TRANSACTION")
    update = trtmc_validate._prepare_case_directory_update(stage)
    original_away = case_dir / "original-away"
    original_verify = trtmc_validate._verify_case_directory_target
    swapped = False

    def verify_then_swap(candidate):
        nonlocal swapped
        original_verify(candidate)
        if not swapped:
            repro.rename(original_away)
            repro.mkdir()
            (repro / "concurrent.txt").write_bytes(b"CONCURRENT")
            swapped = True

    monkeypatch.setattr(
        trtmc_validate,
        "_verify_case_directory_target",
        verify_then_swap,
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="changed while moving it to transaction backup",
    ):
        trtmc_validate._commit_case_directory_update(update)
    trtmc_validate._rollback_case_directory_update(update)
    trtmc_validate._finalize_case_directory_update(update)
    trtmc_validate._cleanup_case_artifact_stage(stage)

    assert swapped
    assert (repro / "concurrent.txt").read_bytes() == b"CONCURRENT"
    assert (original_away / "original.txt").read_bytes() == b"ORIGINAL"
    assert not any(
        child.name.startswith(
            (".repro-previous.", ".repro-next.", ".repro-rollback.")
        )
        for child in case_dir.iterdir()
    )


def test_file_transaction_reconciles_backup_rename_success_then_error(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "report.json"
    target.write_bytes(b"ORIGINAL")
    update = trtmc_validate._prepare_file_update(
        target,
        b"TRANSACTION",
    )
    real_rename = trtmc_validate._rename_noreplace_at
    injected = False

    def rename_then_raise(
        source_fd,
        source_name,
        destination_fd,
        destination_name,
    ):
        nonlocal injected
        real_rename(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )
        if (
            not injected
            and source_name == target.name
            and destination_name.startswith(
                ".report.json.previous."
            )
        ):
            injected = True
            raise OSError("injected after successful backup rename")

    monkeypatch.setattr(
        trtmc_validate,
        "_rename_noreplace_at",
        rename_then_raise,
    )

    with pytest.raises(
        OSError,
        match="after successful backup rename",
    ):
        trtmc_validate._commit_file_update(update)
    trtmc_validate._rollback_file_update(update)
    assert trtmc_validate._finalize_file_update(update) == []

    assert injected
    assert target.read_bytes() == b"ORIGINAL"
    assert [path.name for path in tmp_path.iterdir()] == [
        "report.json"
    ]


def test_directory_transaction_reconciles_backup_rename_success_then_error(
    tmp_path,
    monkeypatch,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    repro = case_dir / "repro"
    repro.mkdir(parents=True)
    (repro / "original.txt").write_bytes(b"ORIGINAL")
    stage = trtmc_validate._create_case_artifact_stage(case_dir)
    staged_repro = stage.path / "repro"
    staged_repro.mkdir()
    (staged_repro / "new.txt").write_bytes(b"TRANSACTION")
    update = trtmc_validate._prepare_case_directory_update(stage)
    real_rename = trtmc_validate._rename_noreplace_at
    injected = False

    def rename_then_raise(
        source_fd,
        source_name,
        destination_fd,
        destination_name,
    ):
        nonlocal injected
        real_rename(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )
        if (
            not injected
            and source_name == "repro"
            and destination_name.startswith(".repro-previous.")
        ):
            injected = True
            raise OSError("injected after successful backup rename")

    monkeypatch.setattr(
        trtmc_validate,
        "_rename_noreplace_at",
        rename_then_raise,
    )

    with pytest.raises(
        OSError,
        match="after successful backup rename",
    ):
        trtmc_validate._commit_case_directory_update(update)
    trtmc_validate._rollback_case_directory_update(update)
    assert trtmc_validate._finalize_case_directory_update(update) == []
    trtmc_validate._cleanup_case_artifact_stage(stage)

    assert injected
    assert (repro / "original.txt").read_bytes() == b"ORIGINAL"
    assert [path.name for path in case_dir.iterdir()] == ["repro"]


def test_directory_prepare_recovers_stage_after_rename_success_then_error(
    tmp_path,
    monkeypatch,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    repro = case_dir / "repro"
    repro.mkdir(parents=True)
    (repro / "original.txt").write_bytes(b"ORIGINAL")
    stage = trtmc_validate._create_case_artifact_stage(case_dir)
    staged_repro = stage.path / "repro"
    staged_repro.mkdir()
    (staged_repro / "new.txt").write_bytes(b"TRANSACTION")
    real_rename = trtmc_validate._rename_noreplace_at
    injected = False

    def rename_then_raise(
        source_fd,
        source_name,
        destination_fd,
        destination_name,
    ):
        nonlocal injected
        real_rename(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )
        if (
            not injected
            and source_name == "repro"
            and destination_name.startswith(".repro-next.")
        ):
            injected = True
            raise OSError("injected after successful prepare rename")

    monkeypatch.setattr(
        trtmc_validate,
        "_rename_noreplace_at",
        rename_then_raise,
    )

    with pytest.raises(
        OSError,
        match="after successful prepare rename",
    ):
        trtmc_validate._prepare_case_directory_update(stage)

    assert injected
    assert (stage.path / "repro" / "new.txt").read_bytes() == b"TRANSACTION"
    assert (repro / "original.txt").read_bytes() == b"ORIGINAL"
    assert not any(
        child.name.startswith(".repro-next.")
        for child in case_dir.iterdir()
    )
    trtmc_validate._cleanup_case_artifact_stage(stage)
    assert [path.name for path in case_dir.iterdir()] == ["repro"]


def test_directory_prepare_recovers_stage_after_fingerprint_failure(
    tmp_path,
    monkeypatch,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    repro = case_dir / "repro"
    repro.mkdir(parents=True)
    (repro / "original.txt").write_bytes(b"ORIGINAL")
    stage = trtmc_validate._create_case_artifact_stage(case_dir)
    staged_repro = stage.path / "repro"
    staged_repro.mkdir()
    (staged_repro / "new.txt").write_bytes(b"TRANSACTION")
    real_fingerprint = trtmc_validate._directory_tree_fingerprint_at
    injected = False

    def fail_next_fingerprint(directory_fd, name):
        nonlocal injected
        if (
            not injected
            and directory_fd == stage.case_fd
            and name.startswith(".repro-next.")
        ):
            injected = True
            raise OSError("injected post-move fingerprint failure")
        return real_fingerprint(directory_fd, name)

    monkeypatch.setattr(
        trtmc_validate,
        "_directory_tree_fingerprint_at",
        fail_next_fingerprint,
    )

    with pytest.raises(
        OSError,
        match="post-move fingerprint failure",
    ):
        trtmc_validate._prepare_case_directory_update(stage)

    assert injected
    assert (stage.path / "repro" / "new.txt").read_bytes() == b"TRANSACTION"
    assert (repro / "original.txt").read_bytes() == b"ORIGINAL"
    assert not any(
        child.name.startswith(".repro-next.")
        for child in case_dir.iterdir()
    )
    trtmc_validate._cleanup_case_artifact_stage(stage)
    assert [path.name for path in case_dir.iterdir()] == ["repro"]


def test_file_transaction_rejects_staged_content_mutation_during_install(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "report.json"
    target.write_bytes(b"ORIGINAL")
    update = trtmc_validate._prepare_file_update(
        target,
        b"TRANSACTION",
    )
    real_rename = trtmc_validate._rename_noreplace_at
    injected = False

    def mutate_then_rename(
        source_fd,
        source_name,
        destination_fd,
        destination_name,
    ):
        nonlocal injected
        if (
            not injected
            and source_name == update.next_name
            and destination_name == target.name
        ):
            descriptor = os.open(
                source_name,
                os.O_WRONLY,
                dir_fd=source_fd,
            )
            try:
                os.pwrite(descriptor, b"MALICIOUS!!", 0)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.utime(
                source_name,
                ns=(
                    update.next_metadata.st_atime_ns,
                    update.next_metadata.st_mtime_ns,
                ),
                dir_fd=source_fd,
                follow_symlinks=False,
            )
            injected = True
        real_rename(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )

    monkeypatch.setattr(
        trtmc_validate,
        "_rename_noreplace_at",
        mutate_then_rename,
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="staged file changed during publication",
    ):
        trtmc_validate._commit_file_update(update)
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="rollback incomplete",
    ):
        trtmc_validate._rollback_file_update(update)
    trtmc_validate._finalize_file_update(update)

    assert injected
    assert target.read_bytes() == b"ORIGINAL"
    recovery = [
        path
        for path in tmp_path.iterdir()
        if ".rollback." in path.name
    ]
    assert len(recovery) == 1
    assert recovery[0].read_bytes() == b"MALICIOUS!!"


def test_directory_transaction_rejects_staged_mutation_during_install(
    tmp_path,
    monkeypatch,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    repro = case_dir / "repro"
    repro.mkdir(parents=True)
    (repro / "original.txt").write_bytes(b"ORIGINAL")
    stage = trtmc_validate._create_case_artifact_stage(case_dir)
    staged_repro = stage.path / "repro"
    staged_repro.mkdir()
    (staged_repro / "new.txt").write_bytes(b"TRANSACTION")
    update = trtmc_validate._prepare_case_directory_update(stage)
    real_rename = trtmc_validate._rename_noreplace_at
    injected = False

    def mutate_then_rename(
        source_fd,
        source_name,
        destination_fd,
        destination_name,
    ):
        nonlocal injected
        if (
            not injected
            and source_name == update.next_name
            and destination_name == "repro"
        ):
            directory_fd = os.open(
                source_name,
                os.O_RDONLY | os.O_DIRECTORY,
                dir_fd=source_fd,
            )
            try:
                descriptor = os.open(
                    "attacker.txt",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=directory_fd,
                )
                try:
                    os.write(descriptor, b"MALICIOUS")
                finally:
                    os.close(descriptor)
            finally:
                os.close(directory_fd)
            injected = True
        real_rename(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )

    monkeypatch.setattr(
        trtmc_validate,
        "_rename_noreplace_at",
        mutate_then_rename,
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="staged reproduction changed during report publication",
    ):
        trtmc_validate._commit_case_directory_update(update)
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="rollback incomplete",
    ):
        trtmc_validate._rollback_case_directory_update(update)
    trtmc_validate._finalize_case_directory_update(update)
    trtmc_validate._cleanup_case_artifact_stage(stage)

    assert injected
    assert (repro / "original.txt").read_bytes() == b"ORIGINAL"
    recovery = [
        path
        for path in case_dir.iterdir()
        if path.name.startswith(".repro-rollback.")
    ]
    assert len(recovery) == 1
    assert (recovery[0] / "attacker.txt").read_bytes() == b"MALICIOUS"


def test_file_rollback_preserves_in_place_concurrent_modification(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "report.json"
    target.write_bytes(b"ORIGINAL")
    update = trtmc_validate._prepare_file_update(
        target,
        b"TRANSACTION",
    )
    trtmc_validate._commit_file_update(update)
    real_rename = trtmc_validate._rename_noreplace_at
    injected = False

    def modify_then_rename(
        source_fd,
        source_name,
        destination_fd,
        destination_name,
    ):
        nonlocal injected
        if (
            not injected
            and source_name == target.name
            and ".rollback." in destination_name
        ):
            descriptor = os.open(
                source_name,
                os.O_WRONLY | os.O_TRUNC,
                dir_fd=source_fd,
            )
            try:
                os.write(descriptor, b"CONCURRENT-MODIFICATION")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            injected = True
        real_rename(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )

    monkeypatch.setattr(
        trtmc_validate,
        "_rename_noreplace_at",
        modify_then_rename,
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="rollback incomplete",
    ):
        trtmc_validate._rollback_file_update(update)
    trtmc_validate._finalize_file_update(update)

    assert injected
    assert target.read_bytes() == b"ORIGINAL"
    recovery = [
        path
        for path in tmp_path.iterdir()
        if ".rollback." in path.name
    ]
    assert len(recovery) == 1
    assert recovery[0].read_bytes() == b"CONCURRENT-MODIFICATION"


def test_directory_rollback_preserves_in_place_concurrent_modification(
    tmp_path,
    monkeypatch,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    repro = case_dir / "repro"
    repro.mkdir(parents=True)
    (repro / "original.txt").write_bytes(b"ORIGINAL")
    stage = trtmc_validate._create_case_artifact_stage(case_dir)
    staged_repro = stage.path / "repro"
    staged_repro.mkdir()
    (staged_repro / "new.txt").write_bytes(b"TRANSACTION")
    update = trtmc_validate._prepare_case_directory_update(stage)
    trtmc_validate._commit_case_directory_update(update)
    real_rename = trtmc_validate._rename_noreplace_at
    injected = False

    def modify_then_rename(
        source_fd,
        source_name,
        destination_fd,
        destination_name,
    ):
        nonlocal injected
        if (
            not injected
            and source_name == "repro"
            and destination_name.startswith(".repro-rollback.")
        ):
            directory_fd = os.open(
                source_name,
                os.O_RDONLY | os.O_DIRECTORY,
                dir_fd=source_fd,
            )
            try:
                descriptor = os.open(
                    "concurrent.txt",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=directory_fd,
                )
                try:
                    os.write(descriptor, b"CONCURRENT")
                finally:
                    os.close(descriptor)
            finally:
                os.close(directory_fd)
            injected = True
        real_rename(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )

    monkeypatch.setattr(
        trtmc_validate,
        "_rename_noreplace_at",
        modify_then_rename,
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="rollback incomplete",
    ):
        trtmc_validate._rollback_case_directory_update(update)
    trtmc_validate._finalize_case_directory_update(update)
    trtmc_validate._cleanup_case_artifact_stage(stage)

    assert injected
    assert (repro / "original.txt").read_bytes() == b"ORIGINAL"
    recovery = [
        path
        for path in case_dir.iterdir()
        if path.name.startswith(".repro-rollback.")
    ]
    assert len(recovery) == 1
    assert (recovery[0] / "concurrent.txt").read_bytes() == b"CONCURRENT"


def test_file_finalize_quarantines_and_preserves_racing_replacement(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "report.json"
    target.write_bytes(b"ORIGINAL")
    update = trtmc_validate._prepare_file_update(
        target,
        b"TRANSACTION",
    )
    trtmc_validate._commit_file_update(update)
    backup_name = update.backup_name
    preserved = tmp_path / "preserved-original"
    real_rename = trtmc_validate._rename_noreplace_at
    injected = False

    def replace_then_rename(
        source_fd,
        source_name,
        destination_fd,
        destination_name,
    ):
        nonlocal injected
        if not injected and source_name == backup_name:
            os.rename(
                source_name,
                preserved.name,
                src_dir_fd=source_fd,
                dst_dir_fd=source_fd,
            )
            descriptor = os.open(
                source_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=source_fd,
            )
            try:
                os.write(descriptor, b"CONCURRENT")
            finally:
                os.close(descriptor)
            injected = True
        real_rename(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )

    monkeypatch.setattr(
        trtmc_validate,
        "_rename_noreplace_at",
        replace_then_rename,
    )

    cleanup_errors = trtmc_validate._finalize_file_update(update)

    assert injected
    assert cleanup_errors
    assert target.read_bytes() == b"TRANSACTION"
    assert preserved.read_bytes() == b"ORIGINAL"
    quarantined = [
        path
        for path in tmp_path.iterdir()
        if ".cleanup." in path.name
    ]
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"CONCURRENT"


def test_directory_finalize_quarantines_and_preserves_racing_replacement(
    tmp_path,
    monkeypatch,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    repro = case_dir / "repro"
    repro.mkdir(parents=True)
    (repro / "original.txt").write_bytes(b"ORIGINAL")
    stage = trtmc_validate._create_case_artifact_stage(case_dir)
    staged_repro = stage.path / "repro"
    staged_repro.mkdir()
    (staged_repro / "new.txt").write_bytes(b"TRANSACTION")
    update = trtmc_validate._prepare_case_directory_update(stage)
    trtmc_validate._commit_case_directory_update(update)
    backup_name = update.backup_name
    preserved = case_dir / "preserved-original"
    real_rename = trtmc_validate._rename_noreplace_at
    injected = False

    def replace_then_rename(
        source_fd,
        source_name,
        destination_fd,
        destination_name,
    ):
        nonlocal injected
        if not injected and source_name == backup_name:
            os.rename(
                source_name,
                preserved.name,
                src_dir_fd=source_fd,
                dst_dir_fd=source_fd,
            )
            os.mkdir(source_name, 0o700, dir_fd=source_fd)
            descriptor = os.open(
                f"{source_name}/concurrent.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=source_fd,
            )
            try:
                os.write(descriptor, b"CONCURRENT")
            finally:
                os.close(descriptor)
            injected = True
        real_rename(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )

    monkeypatch.setattr(
        trtmc_validate,
        "_rename_noreplace_at",
        replace_then_rename,
    )

    cleanup_errors = (
        trtmc_validate._finalize_case_directory_update(update)
    )
    trtmc_validate._cleanup_case_artifact_stage(stage)

    assert injected
    assert cleanup_errors
    assert (repro / "new.txt").read_bytes() == b"TRANSACTION"
    assert (preserved / "original.txt").read_bytes() == b"ORIGINAL"
    quarantined = [
        path
        for path in case_dir.iterdir()
        if path.name.startswith(".repro-cleanup.")
    ]
    assert len(quarantined) == 1
    assert (
        quarantined[0] / "concurrent.txt"
    ).read_bytes() == b"CONCURRENT"


def test_file_transaction_never_overwrites_concurrent_install_target(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "report.json"
    target.write_bytes(b"ORIGINAL")
    update = trtmc_validate._prepare_file_update(
        target,
        b"TRANSACTION",
    )
    original_rename = trtmc_validate._rename_noreplace_at
    injected = False

    def inject_target(
        source_fd,
        source_name,
        destination_fd,
        destination_name,
    ):
        nonlocal injected
        if (
            source_name == update.next_name
            and destination_name == target.name
            and not injected
        ):
            target.write_bytes(b"CONCURRENT")
            injected = True
        original_rename(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )

    monkeypatch.setattr(
        trtmc_validate,
        "_rename_noreplace_at",
        inject_target,
    )

    with pytest.raises(FileExistsError):
        trtmc_validate._commit_file_update(update)
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="concurrent file prevented rollback",
    ) as raised:
        trtmc_validate._rollback_file_update(update)
    trtmc_validate._finalize_file_update(update)

    assert injected
    assert target.read_bytes() == b"CONCURRENT"
    backups = tuple(tmp_path.glob(".report.json.previous.*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"ORIGINAL"
    assert str(backups[0]) in str(raised.value)


def test_directory_transaction_never_overwrites_concurrent_install_target(
    tmp_path,
    monkeypatch,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    repro = case_dir / "repro"
    repro.mkdir(parents=True)
    (repro / "original.txt").write_bytes(b"ORIGINAL")
    stage = trtmc_validate._create_case_artifact_stage(case_dir)
    staged_repro = stage.path / "repro"
    staged_repro.mkdir()
    (staged_repro / "new.txt").write_bytes(b"TRANSACTION")
    update = trtmc_validate._prepare_case_directory_update(stage)
    original_rename = trtmc_validate._rename_noreplace_at
    injected = False

    def inject_target(
        source_fd,
        source_name,
        destination_fd,
        destination_name,
    ):
        nonlocal injected
        if (
            source_name == update.next_name
            and destination_name == "repro"
            and not injected
        ):
            repro.mkdir()
            (repro / "concurrent.txt").write_bytes(b"CONCURRENT")
            injected = True
        original_rename(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )

    monkeypatch.setattr(
        trtmc_validate,
        "_rename_noreplace_at",
        inject_target,
    )

    with pytest.raises(FileExistsError):
        trtmc_validate._commit_case_directory_update(update)
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="concurrent reproduction prevented rollback",
    ) as raised:
        trtmc_validate._rollback_case_directory_update(update)
    trtmc_validate._finalize_case_directory_update(update)
    trtmc_validate._cleanup_case_artifact_stage(stage)

    assert injected
    assert (repro / "concurrent.txt").read_bytes() == b"CONCURRENT"
    backups = tuple(case_dir.glob(".repro-previous.*"))
    assert len(backups) == 1
    assert (backups[0] / "original.txt").read_bytes() == b"ORIGINAL"
    assert str(backups[0]) in str(raised.value)


def test_file_transaction_rejects_target_swap_after_install(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "report.json"
    target.write_bytes(b"ORIGINAL")
    transaction_away = tmp_path / "transaction-away.json"
    update = trtmc_validate._prepare_file_update(
        target,
        b"TRANSACTION",
    )
    original_verify = trtmc_validate._verify_committed_file_update
    swapped = False

    def swap_then_verify(candidate):
        nonlocal swapped
        if not swapped:
            target.rename(transaction_away)
            target.write_bytes(b"CONCURRENT")
            swapped = True
        original_verify(candidate)

    monkeypatch.setattr(
        trtmc_validate,
        "_verify_committed_file_update",
        swap_then_verify,
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="not visible after publication",
    ):
        trtmc_validate._commit_file_update(update)
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="concurrent file prevented rollback",
    ):
        trtmc_validate._rollback_file_update(update)
    trtmc_validate._finalize_file_update(update)

    assert swapped
    assert target.read_bytes() == b"CONCURRENT"
    assert transaction_away.read_bytes() == b"TRANSACTION"
    backups = tuple(tmp_path.glob(".report.json.previous.*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"ORIGINAL"


def test_directory_transaction_rejects_target_swap_after_install(
    tmp_path,
    monkeypatch,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    repro = case_dir / "repro"
    repro.mkdir(parents=True)
    (repro / "original.txt").write_bytes(b"ORIGINAL")
    stage = trtmc_validate._create_case_artifact_stage(case_dir)
    staged_repro = stage.path / "repro"
    staged_repro.mkdir()
    (staged_repro / "new.txt").write_bytes(b"TRANSACTION")
    update = trtmc_validate._prepare_case_directory_update(stage)
    transaction_away = case_dir / "transaction-away"
    original_verify = (
        trtmc_validate._verify_committed_case_directory_update
    )
    swapped = False

    def swap_then_verify(candidate):
        nonlocal swapped
        if not swapped:
            repro.rename(transaction_away)
            repro.mkdir()
            (repro / "concurrent.txt").write_bytes(b"CONCURRENT")
            swapped = True
        original_verify(candidate)

    monkeypatch.setattr(
        trtmc_validate,
        "_verify_committed_case_directory_update",
        swap_then_verify,
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="not visible after report publication",
    ):
        trtmc_validate._commit_case_directory_update(update)
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="concurrent reproduction prevented rollback",
    ):
        trtmc_validate._rollback_case_directory_update(update)
    trtmc_validate._finalize_case_directory_update(update)
    trtmc_validate._cleanup_case_artifact_stage(stage)

    assert swapped
    assert (repro / "concurrent.txt").read_bytes() == b"CONCURRENT"
    assert (transaction_away / "new.txt").read_bytes() == b"TRANSACTION"
    backups = tuple(case_dir.glob(".repro-previous.*"))
    assert len(backups) == 1
    assert (backups[0] / "original.txt").read_bytes() == b"ORIGINAL"


def test_report_transaction_rejects_parent_swap_after_final_precheck(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "results"
    output.mkdir()
    (output / "report.json").write_bytes(b"OLD-JSON")
    (output / "report.html").write_bytes(b"OLD-HTML")
    moved_output = tmp_path / "results-original"
    original_verify = trtmc_validate._verify_file_update_target
    swapped = False

    def verify_then_swap(update):
        nonlocal swapped
        original_verify(update)
        if update.path.name == "report.html" and not swapped:
            output.rename(moved_output)
            output.mkdir()
            (output / "sentinel").write_bytes(b"REPLACEMENT")
            swapped = True

    monkeypatch.setattr(
        trtmc_validate,
        "_verify_file_update_target",
        verify_then_swap,
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="parent changed after publication",
    ):
        trtmc_validate.write_report(output, result_paths=[])

    assert swapped
    assert (output / "sentinel").read_bytes() == b"REPLACEMENT"
    assert sorted(child.name for child in output.iterdir()) == [
        "sentinel"
    ]
    assert (moved_output / "report.json").read_bytes() == b"OLD-JSON"
    assert (moved_output / "report.html").read_bytes() == b"OLD-HTML"
    assert not any(
        child.name.startswith(".report")
        for child in moved_output.iterdir()
    )


def test_report_final_visibility_sweep_preserves_concurrent_target(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "results"
    output.mkdir()
    report_json = output / "report.json"
    report_html = output / "report.html"
    report_json.write_bytes(b"OLD-JSON")
    report_html.write_bytes(b"OLD-HTML")
    transaction_json = output / "transaction-report.json"
    original_verify = (
        trtmc_validate._verify_report_transaction_visibility
    )
    swapped = False

    def swap_then_verify(entries):
        nonlocal swapped
        if not swapped:
            report_json.rename(transaction_json)
            report_json.write_bytes(b"CONCURRENT")
            swapped = True
        original_verify(entries)

    monkeypatch.setattr(
        trtmc_validate,
        "_verify_report_transaction_visibility",
        swap_then_verify,
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="rollback was incomplete",
    ) as raised:
        trtmc_validate.write_report(output, result_paths=[])

    assert swapped
    assert report_json.read_bytes() == b"CONCURRENT"
    assert report_html.read_bytes() == b"OLD-HTML"
    assert transaction_json.exists()
    backups = tuple(output.glob(".report.json.previous.*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"OLD-JSON"
    assert str(backups[0]) in str(raised.value)


def test_transaction_cleanup_handles_1100_levels_without_recursion(
    monkeypatch,
):
    depth = 1100
    removed = []
    directory_metadata = os.stat_result(
        (stat.S_IFDIR | 0o700, 1, 1, 1, 0, 0, 0, 0, 0, 0)
    )

    class Entry:
        name = "d"

        @staticmethod
        def stat(*, follow_symlinks):
            assert follow_symlinks is False
            return directory_metadata

    class Entries:
        def __init__(self, directory_depth):
            self.directory_depth = directory_depth

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            if self.directory_depth < depth:
                return iter((Entry(),))
            return iter(())

    monkeypatch.setattr(trtmc_validate.os, "open", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(trtmc_validate.os, "close", lambda _descriptor: None)
    monkeypatch.setattr(
        trtmc_validate.os,
        "fstat",
        lambda _descriptor: directory_metadata,
    )
    monkeypatch.setattr(
        trtmc_validate,
        "_stat_at",
        lambda _descriptor, _name: directory_metadata,
    )
    monkeypatch.setattr(
        trtmc_validate.os,
        "scandir",
        lambda descriptor: Entries(descriptor),
    )
    monkeypatch.setattr(
        trtmc_validate.os,
        "rmdir",
        lambda name, *, dir_fd: removed.append((name, dir_fd)),
    )
    monkeypatch.setattr(
        trtmc_validate,
        "_open_relative_directory_at",
        lambda _root_fd, components: len(components),
    )

    trtmc_validate._remove_directory_tree_at(10, "root")

    assert len(removed) == depth + 1
    assert removed[-1] == ("root", 10)


def test_execution_error_with_zero_disagreements_clears_stale_repro(
    tmp_path,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    repro = case_dir / "repro"
    repro.mkdir(parents=True)
    stale = repro / "stale-input.jsonl"
    stale.write_text("SECRET\n", encoding="utf-8")
    disagreements = case_dir / "disagreements.jsonl"
    disagreements.write_text("", encoding="utf-8")
    comparison = case_dir / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "raw_result": {
                    "status": "failed",
                    "error_type": "WorkerProcessError",
                    "error": "worker crashed",
                },
                "execution": {"status": "error", "exit_code": 1},
                "comparison": {
                    "status": "not_run",
                    "mode": "",
                    "primary_metric": None,
                    "metrics": {},
                    "failures": [],
                },
                "validation": {"status": "failed"},
                "disagreements": {
                    "count": 0,
                    "path": "disagreements.jsonl",
                },
            }
        ),
        encoding="utf-8",
    )

    trtmc_validate.write_report(
        tmp_path,
        result_paths=[comparison],
    )

    assert repro.is_dir()
    assert list(repro.iterdir()) == []
    assert disagreements.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    "reproduce",
    [
        {"hf": ["python reference.py", 7]},
        {"trtmc": ["trtmc run\x00hidden"]},
        {"dataset": {"command": 7}},
        {"dataset": {"command": "prepare\x00hidden"}},
        {"representative": {"sample_id": 7}},
        {"representative": {"sample_id": "sample\x00hidden"}},
        {"representative": {"reason": "failure\x00hidden"}},
    ],
    ids=[
        "non-string-command",
        "nul-command",
        "non-string-dataset-command",
        "nul-dataset-command",
        "non-string-representative",
        "nul-representative-sample-id",
        "nul-representative-reason",
    ],
)
def test_result_reproduction_rejects_non_string_or_nul_fields(reproduce):
    with pytest.raises(trtmc_validate.ValidationError):
        trtmc_validate._normalize_result(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "passed",
                "reproduce": reproduce,
            }
        )


def test_not_compared_result_has_canonical_shape():
    result = trtmc_validate._not_compared_result(
        trtmc_validate.Binding(
            "model-a",
            None,
            "No aligned reference comparator.",
        )
    )

    assert result["workload"] is None
    assert result["not_compared_reason"] == (
        "No aligned reference comparator."
    )
    assert result["execution"] == {
        "status": "not_run",
        "exit_code": None,
    }
    assert result["comparison"] == {
        "status": "not_run",
        "mode": "",
        "primary_metric": None,
        "metrics": {},
        "failures": [],
    }
    assert result["validation"] == {"status": "not_compared"}


def test_result_rejects_primary_metric_value_mismatch():
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="primary_metric.value must exactly match",
    ):
        trtmc_validate._normalize_result(
            {
                "model": "model-a",
                "workload": "workload-a",
                "raw_result": {"status": "passed"},
                "execution": {"status": "completed"},
                "comparison": {
                    "status": "agreement",
                    "metrics": {"accuracy": 1.0},
                    "primary_metric": {
                        "name": "accuracy",
                        "value": 0.5,
                    },
                },
                "validation": {"status": "passed"},
            }
        )


@pytest.mark.parametrize(
    ("run_status", "expected_report_status"),
    [
        ("running", "incomplete"),
        ("failed", "failed"),
    ],
)
def test_report_aggregates_running_and_failed_run_status(
    tmp_path,
    run_status,
    expected_report_status,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    comparison = case_dir / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "passed",
                "returncode": 0,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "passed",
                ),
            }
        ),
        encoding="utf-8",
    )
    run = {
        "schema_version": "trtmc.validation-run/v1",
        "started_at": "2026-07-27T01:00:00+00:00",
        "finished_at": None,
        "duration_seconds": None,
        "status": run_status,
    }
    if run_status == "failed":
        run.update(
            {
                "finished_at": "2026-07-27T01:00:01+00:00",
                "duration_seconds": 1.0,
                "error": "worker failed",
            }
        )
    (tmp_path / "run.json").write_text(
        json.dumps(run),
        encoding="utf-8",
    )

    _, _, report = trtmc_validate.write_report(
        tmp_path,
        result_paths=[comparison],
        expected_gates_by_binding=_test_gate_context(
            [comparison]
        ),
    )

    assert report["validation_status"] == expected_report_status
    assert report["summary"]["cases"] == 1
    assert report["summary"]["validation_passed"] == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "started_at",
            "not-a-timestamp",
            "started_at is not ISO-8601",
        ),
        (
            "started_at",
            "2026-07-27T01:00:00",
            "started_at must include a timezone",
        ),
        (
            "finished_at",
            "not-a-timestamp",
            "finished_at is not ISO-8601",
        ),
        (
            "finished_at",
            "2026-07-27T01:00:01",
            "finished_at must include a timezone",
        ),
    ],
    ids=[
        "started-not-iso",
        "started-no-timezone",
        "finished-not-iso",
        "finished-no-timezone",
    ],
)
def test_canonical_run_rejects_invalid_timestamps(
    tmp_path,
    field,
    value,
    message,
):
    run = {
        "schema_version": "trtmc.validation-run/v1",
        "started_at": "2026-07-27T01:00:00+00:00",
        "finished_at": "2026-07-27T01:00:01+00:00",
        "duration_seconds": 1.0,
        "status": "completed",
    }
    run[field] = value
    (tmp_path / "run.json").write_text(
        json.dumps(run),
        encoding="utf-8",
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match=message,
    ):
        trtmc_validate.write_report(
            tmp_path,
            result_paths=[],
        )

    assert not (tmp_path / "report.json").exists()


def test_canonical_run_rejects_finished_before_started(tmp_path):
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "schema_version": "trtmc.validation-run/v1",
                "started_at": "2026-07-27T01:00:00+00:00",
                "finished_at": "2026-07-27T00:59:59+00:00",
                "duration_seconds": 1.0,
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="finished_at precedes started_at",
    ):
        trtmc_validate.write_report(
            tmp_path,
            result_paths=[],
        )

    assert not (tmp_path / "report.json").exists()


def test_report_refresh_preserves_command_count_and_representative(
    tmp_path,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    work_dir = case_dir / "validation" / "workload-a" / "model-a"
    work_dir.mkdir(parents=True)
    comparison = case_dir / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "passed",
                "returncode": 0,
                "raw_result": _raw_evidence(
                    "model-a",
                    "workload-a",
                    "passed",
                    work_dir=str(work_dir),
                ),
                "reproduce": {
                    "hf": ["python reference.py"],
                    "trtmc": ["trtmc run model.trtfb"],
                    "command_count": {"hf": 7, "trtmc": 5},
                    "representative": {
                        "sample_id": "sample-77",
                        "reason": "first_disagreement",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    _, _, report = trtmc_validate.write_report(
        tmp_path,
        result_paths=[comparison],
        expected_gates_by_binding=_test_gate_context(
            [comparison]
        ),
    )

    reproduce = report["results"][0]["reproduce"]
    assert reproduce["command_count"] == {"hf": 7, "trtmc": 5}
    assert reproduce["representative"] == {
        "sample_id": "sample-77",
        "reason": "first_disagreement",
    }
    assert reproduce["hf"] == ["python reference.py"]
    assert reproduce["trtmc"] == ["trtmc run model.trtfb"]


def test_report_result_budget_uses_payload_length_from_the_same_read(
    tmp_path,
    monkeypatch,
):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    comparison = case_dir / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "passed",
                "padding": "x" * 512,
            }
        ),
        encoding="utf-8",
    )
    actual_payload_bytes = comparison.stat().st_size
    assert actual_payload_bytes > 128
    monkeypatch.setattr(
        trtmc_validate,
        "MAX_REPORT_RESULT_BYTES",
        128,
    )
    real_fstat = trtmc_validate.os.fstat
    real_stat = trtmc_validate.os.stat

    class SmallStat:
        def __init__(self, metadata):
            self._metadata = metadata
            self.st_size = 1

        def __getattr__(self, name):
            return getattr(self._metadata, name)

    def small_regular_fstat(descriptor):
        metadata = real_fstat(descriptor)
        if trtmc_validate.stat.S_ISREG(metadata.st_mode):
            return SmallStat(metadata)
        return metadata

    def small_regular_stat(*args, **kwargs):
        metadata = real_stat(*args, **kwargs)
        if trtmc_validate.stat.S_ISREG(metadata.st_mode):
            return SmallStat(metadata)
        return metadata

    monkeypatch.setattr(
        trtmc_validate.os,
        "fstat",
        small_regular_fstat,
    )
    monkeypatch.setattr(
        trtmc_validate.os,
        "stat",
        small_regular_stat,
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="comparison inputs exceed 128 bytes",
    ):
        trtmc_validate.write_report(
            tmp_path,
            result_paths=[comparison],
        )

    assert actual_payload_bytes > 128
    assert not (tmp_path / "report.json").exists()


def test_write_report_handles_40_staged_cases_with_nofile_limit_64(
    tmp_path,
):
    resource = pytest.importorskip("resource")
    proc_fds = Path("/proc/self/fd")
    if not proc_fds.is_dir():
        pytest.skip("file descriptor accounting requires /proc/self/fd")

    result_paths = []
    for index in range(40):
        model = f"model-{index:02d}"
        workload = f"workload-{index:02d}"
        case_dir = tmp_path / model / workload
        work_dir = case_dir / "validation" / workload / model
        work_dir.mkdir(parents=True)
        comparison = case_dir / "comparison.json"
        comparison.write_text(
            json.dumps(
                {
                    "model": model,
                    "workload": workload,
                    "status": "passed",
                    "returncode": 0,
                    "raw_result": _raw_evidence(
                        model,
                        workload,
                        "passed",
                        work_dir=str(work_dir),
                    ),
                }
            ),
            encoding="utf-8",
        )
        result_paths.append(comparison)

    original_limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    original_soft, hard = original_limits
    target = 64
    if (
        original_soft != resource.RLIM_INFINITY
        and original_soft < target
    ) or (
        hard != resource.RLIM_INFINITY
        and hard < target
    ):
        pytest.skip("RLIMIT_NOFILE cannot be set to 64")
    descriptors_before = len(tuple(proc_fds.iterdir()))
    if descriptors_before >= target - 8:
        pytest.skip("test process already uses too many descriptors")

    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (OSError, ValueError):
        pytest.skip("RLIMIT_NOFILE cannot be lowered on this platform")
    try:
        _, _, report = trtmc_validate.write_report(
            tmp_path,
            result_paths=result_paths,
            expected_gates_by_binding=_test_gate_context(
                result_paths
            ),
        )
        descriptors_after = len(tuple(proc_fds.iterdir()))
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, original_limits)

    assert report["summary"]["cases"] == 40
    assert report["validation_status"] == "passed"
    assert descriptors_after == descriptors_before


def test_report_enforces_aggregate_disagreement_source_budget(
    tmp_path,
    monkeypatch,
):
    result_paths = []
    source_payload = json.dumps({"disagreements": []})
    monkeypatch.setattr(
        trtmc_validate,
        "MAX_REPORT_DISAGREEMENT_SOURCE_BYTES",
        len(source_payload.encode("utf-8")) * 2 - 1,
    )
    for suffix in ("a", "b"):
        model = f"model-{suffix}"
        workload = f"workload-{suffix}"
        case_dir = tmp_path / model / workload
        work_dir = case_dir / "validation" / workload / model
        work_dir.mkdir(parents=True)
        (work_dir / "summary.json").write_text(
            source_payload,
            encoding="utf-8",
        )
        comparison = case_dir / "comparison.json"
        comparison.write_text(
            json.dumps(
                {
                    "model": model,
                    "workload": workload,
                    "status": "passed",
                    "raw_result": {
                        "status": "passed",
                        "work_dir": str(work_dir),
                    },
                }
            ),
            encoding="utf-8",
        )
        result_paths.append(comparison)

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="artifact exceeds",
    ):
        trtmc_validate.write_report(
            tmp_path,
            result_paths=result_paths,
        )

    assert not (tmp_path / "report.json").exists()
    assert not any(tmp_path.rglob(".report-stage-*"))


def test_report_enforces_aggregate_disagreement_media_budget(
    tmp_path,
    monkeypatch,
):
    result_paths = []
    monkeypatch.setattr(
        trtmc_validate,
        "MAX_REPORT_MEDIA_FILES",
        1,
    )
    for suffix in ("a", "b"):
        model = f"model-{suffix}"
        workload = f"workload-{suffix}"
        case_dir = tmp_path / model / workload
        work_dir = case_dir / "validation" / workload / model
        work_dir.mkdir(parents=True)
        image = work_dir / "input.png"
        image.write_bytes(f"image-{suffix}".encode())
        (work_dir / "prompts.jsonl").write_text(
            json.dumps(
                {
                    "sample_id": "sample-1",
                    "images": [str(image)],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (work_dir / "summary.json").write_text(
            json.dumps(
                {
                    "disagreements": [
                        {"sample_id": "sample-1"}
                    ]
                }
            ),
            encoding="utf-8",
        )
        comparison = case_dir / "comparison.json"
        comparison.write_text(
            json.dumps(
                {
                    "model": model,
                    "workload": workload,
                    "status": "failed",
                    "returncode": 1,
                    "raw_result": _raw_evidence(
                        model,
                        workload,
                        "failed",
                        work_dir=str(work_dir),
                    ),
                }
            ),
            encoding="utf-8",
        )
        result_paths.append(comparison)

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="media exceeds",
    ):
        trtmc_validate.write_report(
            tmp_path,
            result_paths=result_paths,
        )

    assert not (tmp_path / "report.json").exists()
    assert not any(tmp_path.rglob(".report-stage-*"))
