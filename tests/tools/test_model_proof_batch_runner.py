# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the bounded multi-model proof runner."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / ".github" / "scripts" / "run-model-proof-batch.sh"


def _write_fake_proof_runner(tmp_path: Path) -> tuple[Path, Path]:
    calls = tmp_path / "calls.tsv"
    runner = tmp_path / "fake-proof.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "model=''\n"
        "revision=''\n"
        "suite=''\n"
        "output_dir=''\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  case "$1" in\n'
        '    --model) model="$2"; shift 2 ;;\n'
        '    --revision) revision="$2"; shift 2 ;;\n'
        '    --suite) suite="$2"; shift 2 ;;\n'
        '    --output-dir) output_dir="$2"; shift 2 ;;\n'
        "    *) exit 98 ;;\n"
        "  esac\n"
        "done\n"
        'printf \'%s\\t%s\\t%s\\t%s\\n\' "$model" "${TRTMC_GPU_ID:-missing}" "$revision" "$suite" >> "$FAKE_CALLS"\n'
        'mkdir -p "$output_dir/artifacts"\n'
        'printf \'<!doctype html><title>%s</title>\\n\' "$model" > "$output_dir/artifacts/model-proof-report.html"\n'
        'if [ "${FAKE_BLOCK:-0}" = 1 ]; then\n'
        '  trap \'printf "%s\\n" "$model" >> "$FAKE_TERMINATED"; exit 143\' TERM INT HUP\n'
        "  while :; do sleep 0.1; done\n"
        "fi\n"
        'sleep "${FAKE_DELAY_SECONDS:-0}"\n'
        'case ",${FAKE_FAIL_MODELS:-}," in\n'
        '  *,"$model",*) exit 7 ;;\n'
        "esac\n"
        'if [ "${FAKE_OMIT_EVIDENCE:-}" != proof.json ]; then\n'
        '  printf \'{"model":"%s"}\\n\' "$model" > "$output_dir/artifacts/proof.json"\n'
        "fi\n"
        'if [ "${FAKE_OMIT_EVIDENCE:-}" != model-proof-status.json ]; then\n'
        '  printf \'{"model":"%s","outcome":"passed"}\\n\' "$model" > "$output_dir/artifacts/model-proof-status.json"\n'
        "fi\n"
        'if [ "${FAKE_OMIT_EVIDENCE:-}" = model-proof-report.html ]; then\n'
        '  rm -f "$output_dir/artifacts/model-proof-report.html"\n'
        "fi\n",
        encoding="utf-8",
    )
    return runner, calls


def _environment(
    runner: Path,
    calls: Path,
    *,
    gpu_ids: str = "2,5",
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "TRTMC_MODEL_PROOF_RUNNER": str(runner),
            "TRTMC_MODEL_PROOF_GPU_IDS": gpu_ids,
            "TRTMC_GPU_ID": "99",
            "FAKE_CALLS": str(calls),
        }
    )
    return env


def _command(models: list[str], output_dir: Path) -> list[str]:
    return [
        "bash",
        str(RUNNER),
        "--models-json",
        json.dumps(models),
        "--expected-count",
        str(len(models)),
        "--revision",
        "a" * 40,
        "--suite",
        "premerge",
        "--output-dir",
        str(output_dir),
    ]


def _calls(path: Path) -> dict[str, tuple[str, str, str]]:
    rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()]
    return {model: (gpu, revision, suite) for model, gpu, revision, suite in rows}


def test_batch_runner_assigns_deterministic_gpu_workers_and_writes_index(
    tmp_path: Path,
) -> None:
    fake_runner, calls = _write_fake_proof_runner(tmp_path)
    models = ["alpha", "beta", "gamma", "delta", "epsilon"]
    output_dir = tmp_path / "batch"
    env = _environment(fake_runner, calls)
    env["FAKE_DELAY_SECONDS"] = "0.05"

    result = subprocess.run(
        _command(models, output_dir),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    recorded = _calls(calls)
    assert set(recorded) == set(models)
    assert {model: recorded[model][0] for model in models} == {
        "alpha": "2",
        "beta": "5",
        "gamma": "2",
        "delta": "5",
        "epsilon": "2",
    }
    assert all(recorded[model][1:] == ("a" * 40, "premerge") for model in models)
    assert all((output_dir / model / "batch.log").is_file() for model in models)
    assert all(
        (output_dir / model / "artifacts" / "model-proof-report.html").is_file() for model in models
    )

    status = json.loads((output_dir / "batch-status.json").read_text(encoding="utf-8"))
    assert status["schema_version"] == 1
    assert status["report_kind"] == "model_proof_batch"
    assert status["outcome"] == "passed"
    assert status["expected_count"] == len(models)
    assert status["model_count"] == len(models)
    assert status["passed_count"] == len(models)
    assert status["failed_count"] == 0
    assert status["gpu_ids"] == ["2", "5"]
    assert [entry["model"] for entry in status["models"]] == models
    assert all(entry["status"] == "passed" for entry in status["models"])
    assert [entry["gpu_id"] for entry in status["models"]] == [
        "2",
        "5",
        "2",
        "5",
        "2",
    ]

    index = (output_dir / "model-proof-index.html").read_text(encoding="utf-8")
    for model in models:
        assert f"{model}/artifacts/model-proof-report.html" in index
        assert f"{model}/batch.log" in index


def test_batch_runner_attempts_every_model_before_returning_failure(tmp_path: Path) -> None:
    fake_runner, calls = _write_fake_proof_runner(tmp_path)
    models = ["passes-first", "fails", "passes-after-failure", "also-passes"]
    output_dir = tmp_path / "batch"
    env = _environment(fake_runner, calls)
    env["FAKE_FAIL_MODELS"] = "fails"

    result = subprocess.run(
        _command(models, output_dir),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert set(_calls(calls)) == set(models)
    status = json.loads((output_dir / "batch-status.json").read_text(encoding="utf-8"))
    assert status["outcome"] == "failed"
    assert status["passed_count"] == 3
    assert status["failed_count"] == 1
    by_model = {entry["model"]: entry for entry in status["models"]}
    assert by_model["fails"]["status"] == "failed"
    assert by_model["fails"]["exit_code"] == 7
    assert by_model["passes-after-failure"]["status"] == "passed"


@pytest.mark.parametrize(
    ("models_json", "expected_error"),
    [
        ("not-json", "invalid --models-json"),
        ('{"model":"alpha"}', "must be a JSON array"),
        ('["alpha","alpha"]', "duplicate model id"),
        ('["alpha","../escape"]', "unsafe model id"),
        ('["alpha",".."]', "unsafe model id"),
        ('["alpha","Uppercase"]', "unsafe model id"),
        ('["alpha","_leading"]', "unsafe model id"),
        ('["alpha",true]', "unsafe model id"),
    ],
)
def test_batch_runner_rejects_invalid_model_json_before_running_proofs(
    tmp_path: Path,
    models_json: str,
    expected_error: str,
) -> None:
    fake_runner, calls = _write_fake_proof_runner(tmp_path)
    command = _command(["alpha", "beta"], tmp_path / "batch")
    command[command.index("--models-json") + 1] = models_json

    result = subprocess.run(
        command,
        env=_environment(fake_runner, calls),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert expected_error in result.stderr
    assert not calls.exists()


def test_batch_runner_rejects_expected_count_mismatch(tmp_path: Path) -> None:
    fake_runner, calls = _write_fake_proof_runner(tmp_path)
    command = _command(["alpha", "beta"], tmp_path / "batch")
    command[command.index("--expected-count") + 1] = "3"

    result = subprocess.run(
        command,
        env=_environment(fake_runner, calls),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "does not match --models-json length" in result.stderr
    assert not calls.exists()


@pytest.mark.parametrize(
    "missing_evidence",
    ["proof.json", "model-proof-status.json", "model-proof-report.html"],
)
def test_batch_runner_rejects_zero_exit_without_required_evidence(
    tmp_path: Path,
    missing_evidence: str,
) -> None:
    fake_runner, calls = _write_fake_proof_runner(tmp_path)
    output_dir = tmp_path / "batch"
    env = _environment(fake_runner, calls)
    env["FAKE_OMIT_EVIDENCE"] = missing_evidence

    result = subprocess.run(
        _command(["alpha"], output_dir),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert f"required evidence: {missing_evidence}" in result.stderr
    batch_log = (output_dir / "alpha" / "batch.log").read_text(encoding="utf-8")
    assert f"required evidence: {missing_evidence}" in batch_log
    status = json.loads((output_dir / "batch-status.json").read_text(encoding="utf-8"))
    assert status["outcome"] == "failed"
    assert status["failed_count"] == 1
    assert status["models"][0]["exit_code"] == 1


@pytest.mark.parametrize("gpu_ids", ["", "0,,1", "01", "2,2", "-1,2"])
def test_batch_runner_rejects_invalid_or_duplicate_gpu_lists(
    tmp_path: Path,
    gpu_ids: str,
) -> None:
    fake_runner, calls = _write_fake_proof_runner(tmp_path)
    env = _environment(fake_runner, calls, gpu_ids=gpu_ids)

    result = subprocess.run(
        _command(["alpha"], tmp_path / "batch"),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "TRTMC_MODEL_PROOF_GPU_IDS" in result.stderr
    assert not calls.exists()


def test_batch_runner_terminates_children_and_finalizes_reports_on_term(
    tmp_path: Path,
) -> None:
    fake_runner, calls = _write_fake_proof_runner(tmp_path)
    output_dir = tmp_path / "batch"
    terminated = tmp_path / "terminated.txt"
    env = _environment(fake_runner, calls)
    env.update({"FAKE_BLOCK": "1", "FAKE_TERMINATED": str(terminated)})
    process = subprocess.Popen(
        _command(["alpha", "beta"], output_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if calls.is_file() and len(calls.read_text(encoding="utf-8").splitlines()) == 2:
            break
        time.sleep(0.05)
    else:
        process.kill()
        process.wait(timeout=5)
        pytest.fail("fake model proofs did not start")

    process.terminate()
    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == 143, stdout + stderr
    assert "Received TERM" in stderr
    status = json.loads((output_dir / "batch-status.json").read_text(encoding="utf-8"))
    assert status["outcome"] == "interrupted"
    assert status["interrupted_count"] == 2
    assert all(entry["status"] == "interrupted" for entry in status["models"])
    assert (output_dir / "model-proof-index.html").is_file()
    assert set(terminated.read_text(encoding="utf-8").splitlines()) == {"alpha", "beta"}
