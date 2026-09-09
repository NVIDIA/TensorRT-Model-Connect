# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import random
import subprocess
import sys

from families.chronos_bolt.tests.qualification import executor, prepare_environment


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
    assert "--no-deps" in calls[1]
    assert calls[1][-2:] == [
        "--requirement",
        str(prepare_environment.REQUIREMENTS),
    ]


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
