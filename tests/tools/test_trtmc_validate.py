# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json

import pytest

from tools import task_eval
from tools import trtmc_compare
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
                "dataset": {
                    "command": "python tools/trtmc_validate.py model-a --limit 1000",
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
        "Reproduce full dataset:\n"
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
    assert "Full dataset (1000 prepared inputs)" in document
    assert "$ python tools/trtmc_validate.py model-a" in document
    assert "$ python hf.py" in document
    assert "$ trtmc run" in document


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
        trtmc_validate.sys.executable,
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
