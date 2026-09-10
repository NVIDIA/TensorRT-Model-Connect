# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import random
import subprocess
import sys

from families.chronos_bolt.tests.qualification import executor, prepare_environment, reference


def _environment(data_root: Path) -> dict:
    return {
        "schema_version": "trtmc.qualification-environment/v1",
        "storage": {"data_root": str(data_root)},
    }


def _definition(**dataset_overrides) -> dict:
    return {
        "dataset": {"relative_path": "ETTh1/ETTh1.csv", **dataset_overrides},
        "selection": {"method": "seeded_windows", "seed": 20260715},
    }


def _dataset_evidence(root: Path) -> dict:
    return {"path": str((root / "ETTh1/ETTh1.csv").resolve())}


def _case() -> dict:
    return {
        "sample_limit": 2,
        "window": {
            "input_columns": ["OT"],
            "context_length": 3,
            "prediction_length": 2,
            "stride": 2,
            "test_target_start": 20,
            "test_end": 40,
            "frequency": 0,
        },
    }


def _write_etth1(root: Path) -> Path:
    path = root / "ETTh1/ETTh1.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "date,OT\n"
        + "".join(f"2026-01-{index + 1:02d},{100 + index / 10}\n" for index in range(40)),
        encoding="utf-8",
    )
    return path


def test_etth1_selection_reproduces_historical_seeded_windows(tmp_path: Path) -> None:
    _write_etth1(tmp_path)

    samples, path = executor._load_samples(
        _definition(), _case(), _environment(tmp_path), _dataset_evidence(tmp_path)
    )

    starts = list(range(17, 36, 2))
    random.Random(20260715).shuffle(starts)
    assert path == (tmp_path / "ETTh1/ETTh1.csv").resolve()
    assert [sample["dataset_index"] for sample in samples] == [start + 3 for start in starts[:2]]
    assert all(len(sample["past_values"]) == 3 for sample in samples)


def test_time_series_parity_requires_every_sample_to_pass() -> None:
    case = {"gate": {"max_relative_l2": 1.0e-3, "max_absolute_error": 1.0e-2}}
    reference = [
        {"sample_id": "a", "values": [1.0, 2.0], "shape": [1, 2]},
        {"sample_id": "b", "values": [3.0, 4.0], "shape": [1, 2]},
    ]
    candidate = [
        {"sample_id": "a", "values": [1.0, 2.0001], "shape": [1, 2]},
        {"sample_id": "b", "values": [3.0, 4.1], "shape": [1, 2]},
    ]

    comparison = executor._compare_samples(reference, candidate, case)

    assert comparison["verdict"] == "fail"
    assert comparison["failed_sample_count"] == 1
    assert comparison["metrics"]["sample_agreement_rate"] == 0.5


def test_time_series_parity_rejects_shape_only_matches() -> None:
    case = {"gate": {"max_relative_l2": 1.0e-3, "max_absolute_error": 1.0e-2}}
    reference = [{"sample_id": "a", "values": [1.0, 2.0], "shape": [1, 2]}]
    candidate = [{"sample_id": "a", "values": [1.0, 2.0], "shape": [2, 1]}]

    comparison = executor._compare_samples(reference, candidate, case)

    assert comparison["verdict"] == "fail"
    assert comparison["metrics"]["shape_match_rate"] == 0.0
    assert comparison["gate_evaluations"][0]["passed"] is False


def test_forecast_output_parser_ignores_logs_and_accepts_identical_rank_output() -> None:
    output = 'runtime log\n[0]<stdout>: {"values":[1.0],"shape":[1]}\n'

    assert executor._command_json(output, "case") == {"values": [1.0], "shape": [1]}


def test_chronos_environment_reuses_compatible_common_python(tmp_path, monkeypatch) -> None:
    request = tmp_path / "request.json"
    output = tmp_path / "output.json"
    request.write_text(
        json.dumps({"common_python": sys.executable, "allow_create": False}),
        encoding="utf-8",
    )
    monkeypatch.setattr(prepare_environment, "_compatible", lambda _python: True)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_environment.py", "--request", str(request), "--output", str(output)],
    )

    prepare_environment.main()

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "python": sys.executable,
        "reference_python": sys.executable,
    }


def test_chronos_environment_is_used_for_build_and_reference(tmp_path, monkeypatch) -> None:
    root = tmp_path / "environments/chronos_bolt"
    target = root / "chronos-test"
    request = tmp_path / "request.json"
    output = tmp_path / "output.json"
    request.write_text(
        json.dumps(
            {
                "common_python": sys.executable,
                "environment_directory": str(root),
                "allow_create": True,
            }
        ),
        encoding="utf-8",
    )
    calls = []
    inherited = []
    monkeypatch.setattr(
        prepare_environment.tempfile,
        "mkdtemp",
        lambda **_kwargs: (target.mkdir(parents=True), str(target))[1],
    )
    monkeypatch.setattr(
        prepare_environment,
        "_compatible",
        lambda python: str(python) == str(target / "bin/python"),
    )
    monkeypatch.setattr(
        prepare_environment,
        "_inherit_common_environment",
        lambda common, selected: inherited.append((common, selected)),
    )

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(prepare_environment.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_environment.py", "--request", str(request), "--output", str(output)],
    )

    prepare_environment.main()

    selected = str(target / "bin/python")
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "python": selected,
        "reference_python": selected,
    }
    assert calls[0] == [
        sys.executable,
        "-m",
        "venv",
        "--system-site-packages",
        str(target),
    ]
    assert inherited == [(sys.executable, selected)]
    assert "--no-deps" in calls[1]
    assert calls[1][-2:] == [
        "--requirement",
        str(prepare_environment.REQUIREMENTS),
    ]


def test_family_venv_inherits_packages_from_common_venv(tmp_path: Path) -> None:
    common = tmp_path / "common"
    target = tmp_path / "target"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(common)], check=True)
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(target)], check=True)
    common_python = str(common / "bin/python")
    target_python = str(target / "bin/python")
    common_site = prepare_environment._purelib(common_python)
    target_site = prepare_environment._purelib(target_python)
    (common_site / "common_cuda_base.py").write_text("VALUE = 'shared-base'\n", encoding="utf-8")
    (common_site / "family_override.py").write_text("VALUE = 'common'\n", encoding="utf-8")
    (target_site / "family_override.py").write_text("VALUE = 'family'\n", encoding="utf-8")

    prepare_environment._inherit_common_environment(common_python, target_python)

    completed = subprocess.run(
        [
            target_python,
            "-c",
            "import common_cuda_base, family_override; "
            "print(common_cuda_base.VALUE, family_override.VALUE)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "shared-base family"


def test_historical_stability_contract_is_preserved() -> None:
    assert executor.measurement_stability([100.0] * 8 + [90.0, 110.0])["stable"]
    assert not executor.measurement_stability([100.0] * 5 + [106.0] * 5)["stable"]


def test_reference_entrypoint_has_no_repository_import_requirement() -> None:
    runner = Path(__file__).with_name("reference.py")

    completed = subprocess.run(
        [sys.executable, "-I", str(runner), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "usage:" in completed.stdout


def test_chronos_compile_specializes_the_fixed_benchmark_shape(monkeypatch) -> None:
    import torch
    from torch._dynamo.backends import registry

    forward = object()
    pipeline = type("Pipeline", (), {})()
    pipeline.model = type("Model", (), {"forward": forward})()
    call = {}

    def fake_compile(target, **options):
        call.update({"target": target, **options})
        return "compiled-forward"

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(registry, "lookup_backend", lambda name: f"{name}-backend")

    evidence = reference._compile(pipeline)

    assert pipeline.model.forward == "compiled-forward"
    assert call["target"] is forward
    assert callable(call["backend"])
    assert call["fullgraph"] is False
    assert call["dynamic"] is False
    assert evidence["dynamic"] is False


def test_eager_reference_measurement_requires_no_compiled_graph(monkeypatch) -> None:
    import torch

    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    samples, output = reference._measure(
        lambda: calls.append("invoke") or "forecast",
        warmup=2,
        iterations=3,
        compile_evidence={"applied": False, "compiled_graph_count": 0},
    )

    assert calls == ["invoke"] * 5
    assert len(samples) == 3
    assert output == "forecast"


def _performance_definition() -> dict:
    return {
        "stability": {
            "samples": 10,
            "median_drift_limit": 0.05,
            "within_median_fraction": 0.05,
            "minimum_close_samples": 8,
            "retries": 1,
        }
    }


def _performance_item() -> dict:
    return {
        "id": "item",
        "family": "chronos_bolt",
        "model": "chronos-bolt-tiny-official",
        "kind": "performance",
        "suite_id": "time_series_performance",
        "case_id": "forecast_64",
    }


def _performance_case() -> dict:
    return {"reference": {"mode": "torch-compile", "fallback": "eager"}}


def _performance_result(item: dict, mode: str, samples: list[float]) -> dict:
    return {
        **executor._identity(item),
        "schema_version": executor.RESULT_SCHEMA,
        "execution": "completed",
        "verdict": None,
        "details": {
            "candidate": {"samples_ms": samples},
            "reference": {"mode": mode, "samples_ms": samples},
            "comparison": {"reference_over_candidate_p50": 1.0},
            "metrics": {"reference_over_candidate_p50": 1.0},
        },
        "artifacts": [{"label": "attempt", "path": "result.json"}],
    }


def _execute_performance(tmp_path: Path, monkeypatch, behavior) -> tuple[dict, list[str]]:
    calls = []

    def fake_attempt(*args, reference_mode, **kwargs):
        calls.append(reference_mode)
        return behavior(args[1], reference_mode)

    monkeypatch.setattr(executor, "_performance_attempt", fake_attempt)
    result = executor._execute_performance(
        {},
        _performance_item(),
        {},
        _performance_definition(),
        _performance_case(),
        {},
        {},
        tmp_path,
    )
    return result, calls


def test_performance_prefers_a_valid_compiled_reference(tmp_path, monkeypatch) -> None:
    result, calls = _execute_performance(
        tmp_path,
        monkeypatch,
        lambda item, mode: _performance_result(item, mode, [100.0] * 10),
    )

    assert calls == ["torch-compile"]
    assert result["execution"] == "completed"
    assert result["details"]["reference_selection"] == {
        "policy": "prefer_torch_compile_then_eager",
        "preferred_mode": "torch-compile",
        "fallback_mode": "eager",
        "selected_mode": "torch-compile",
        "fallback_used": False,
    }


def test_performance_falls_back_to_eager_when_compiled_reference_fails(
    tmp_path, monkeypatch
) -> None:
    def behavior(item, mode):
        if mode == "torch-compile":
            raise executor.PerformanceReferenceError("compiled output parity failed")
        return _performance_result(item, mode, [100.0] * 10)

    result, calls = _execute_performance(tmp_path, monkeypatch, behavior)

    assert calls == ["torch-compile", "eager"]
    assert result["execution"] == "completed"
    assert result["details"]["reference"]["mode"] == "eager"
    assert result["details"]["reference_selection"]["fallback_used"] is True
    assert result["details"]["measurement_attempts"][0]["error"] == (
        "compiled output parity failed"
    )


def test_performance_falls_back_after_compiled_measurements_stay_unstable(
    tmp_path, monkeypatch
) -> None:
    unstable = [100.0] * 5 + [106.0] * 5

    def behavior(item, mode):
        samples = unstable if mode == "torch-compile" else [100.0] * 10
        return _performance_result(item, mode, samples)

    result, calls = _execute_performance(tmp_path, monkeypatch, behavior)

    assert calls == ["torch-compile", "torch-compile", "eager"]
    assert result["execution"] == "completed"
    assert result["details"]["reference_selection"]["selected_mode"] == "eager"


def test_performance_errors_only_after_compiled_and_eager_references_fail(
    tmp_path, monkeypatch
) -> None:
    def behavior(_item, mode):
        raise executor.PerformanceReferenceError(f"{mode} failed")

    result, calls = _execute_performance(tmp_path, monkeypatch, behavior)

    assert calls == ["torch-compile", "eager"]
    assert result["execution"] == "error"
    assert result["details"]["error"] == "all_performance_references_failed"
    assert result["details"]["reference_selection"]["selected_mode"] is None
