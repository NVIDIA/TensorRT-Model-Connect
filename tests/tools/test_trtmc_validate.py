# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json

import pytest

from tools import task_eval
from tools import trtmc_validate


def test_model_workload_catalog_covers_every_ready_model():
    catalog = trtmc_validate.load_catalog()
    suites = task_eval.load_suites()
    task_models = trtmc_validate._task_eval_models(trtmc_validate.DEFAULT_MODELS)

    trtmc_validate.audit_catalog(
        catalog,
        ready_models=trtmc_validate.ready_model_names(),
        suite_names=(suite["id"] for suite in suites),
    )
    trtmc_validate.audit_workload_compatibility(
        catalog,
        suites={suite["id"]: suite for suite in suites},
        task_models=task_models,
    )

    assert len(catalog["models"]) == len(trtmc_validate.ready_model_names())


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


def test_print_result_only_exposes_raw_commands_and_result_locations(tmp_path, capsys):
    comparison = tmp_path / "comparison.json"
    report = tmp_path / "report.html"
    trtmc_validate._print_result(
        {
            "reproduce": {
                "hf": ["python hf_reference.py --model model-a"],
                "trtmc": ["trtmc run --model model-a"],
                "validation": "python tools/trtmc_validate.py model-a",
            }
        },
        comparison,
        report,
    )

    output = capsys.readouterr().out
    assert output == (
        "\n"
        "Reproduce HF:\n"
        "  python hf_reference.py --model model-a\n"
        "\n"
        "Reproduce TRTMC:\n"
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
    assert output.count("unavailable; see comparison result") == 2
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
                },
            }
        ),
        encoding="utf-8",
    )

    json_path, html_path, report = trtmc_validate.write_report(tmp_path)

    assert report["summary"] == {
        "cases": 1,
        "passed": 1,
        "failed": 0,
        "skipped": 0,
    }
    assert json_path == tmp_path / "report.json"
    assert html_path == tmp_path / "report.html"
    assert "model-a/workload-a/comparison.json" in html_path.read_text(encoding="utf-8")


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


def test_task_eval_command_is_directly_reproducible(tmp_path):
    arguments = argparse.Namespace(
        engine_dir=tmp_path / "engines",
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

    command = trtmc_validate._task_eval_command(
        trtmc_validate.Binding("model-a", "workload-a"),
        case_dir=tmp_path / "case",
        dataset=tmp_path / "dataset.json",
        arguments=arguments,
        reference_python="/profiles/python",
    )

    assert command[:3] == [
        trtmc_validate.sys.executable,
        str(trtmc_validate.REPO_ROOT / "tools" / "task_eval.py"),
        "eval",
    ]
    assert command[command.index("--model") + 1] == "model-a"
    assert command[command.index("--suite") + 1] == "workload-a"
    assert command[command.index("--hf-python") + 1] == "/profiles/python"
    assert "--force-hf" in command
    assert "--require-prebuilt-bundles" in command
    assert "--local-files-only" in command
