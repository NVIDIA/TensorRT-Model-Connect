# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LeRobot ACT bundle, native Task API, and exact-source reference proof."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from families.lerobot_act.model import ACTION_MAX, ACTION_MIN


FAMILY = "lerobot_act"
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"
_ACTION_MIN = np.asarray(ACTION_MIN, dtype=np.float32)
_ACTION_MAX = np.asarray(ACTION_MAX, dtype=np.float32)


def _cases() -> dict[str, tuple[dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] == "robot_control"
        assert manifest["tensor_parallel_size"] == 1
        for case in manifest["testcases"]:
            assert case["name"] not in result
            result[case["name"]] = (manifest, case)
    assert result
    return result


CASES = _cases()


def _selected(config) -> set[str]:
    values = set()
    for option in ("--e2e-model", "--e2e-testcase"):
        for raw in config.getoption(option) or []:
            values.update(item.strip() for item in str(raw).split(",") if item.strip())
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        values.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return values


def pytest_generate_tests(metafunc) -> None:
    if "case_name" not in metafunc.fixturenames:
        return
    selected = _selected(metafunc.config)
    names = sorted(
        name
        for name, (manifest, _) in CASES.items()
        if not selected or {FAMILY, manifest["name"], name}.intersection(selected)
    )
    parameters = (
        names
        if selected
        else [
            pytest.param(name, marks=pytest.mark.skip(reason="real family E2E was not selected"))
            for name in names
        ]
    )
    metafunc.parametrize("case_name", parameters, ids=names)


def _required_path(value: str | None, name: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {name}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {name} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get("TRTMC_LEROBOT_ACT_MODEL_DIR")
    if explicit:
        return _required_path(explicit, "TRTMC_LEROBOT_ACT_MODEL_DIR")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=manifest["hf_id"],
            revision=manifest["hf_revision"],
            allow_patterns=("config.json", "model.safetensors"),
            local_files_only=True,
        )
    )


def _asset(case: dict, name: str) -> Path:
    path = TEST_ROOT / case["inputs"][name]
    assert path.is_file(), path
    return path


@cache
def _qualification_binary() -> tuple[Path, Path]:
    build_dir = _required_path(
        os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR"
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--parallel",
            "8",
            "--target",
            "lerobot_act_qualification",
            "trtmc_backend_trt",
            "trtmc_model_lerobot_act",
        ],
        check=True,
        timeout=600,
    )
    binary = build_dir / "families" / FAMILY / "lerobot_act_qualification"
    assert binary.is_file(), f"selected {FAMILY} qualification binary is missing: {binary}"
    return binary, build_dir


def _run_native(binary: Path, runtime_root: Path, bundle: Path, case: dict, tmp_path: Path):
    output = tmp_path / "actions.f32"
    completed = subprocess.run(
        [
            str(binary),
            "control",
            str(bundle),
            "--runtime-root",
            str(runtime_root),
            "--image",
            str(_asset(case, "image")),
            "--state",
            str(_asset(case, "state")),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    summary = json.loads(completed.stdout)
    actions = np.fromfile(output, dtype="<f4").reshape(100, 14)

    qualification_binary, qualification_root = _qualification_binary()
    qualification_output = tmp_path / "qualification-actions.f32"
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        value
        for value in (
            str(qualification_root),
            str(runtime_root),
            environment.get("LD_LIBRARY_PATH", ""),
        )
        if value
    )
    completed = subprocess.run(
        [
            str(qualification_binary),
            str(bundle),
            str(qualification_root),
            str(_asset(case, "image")),
            str(_asset(case, "state")),
            str(qualification_output),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=1800,
    )
    summary.update(json.loads(completed.stdout))
    qualification_actions = np.fromfile(qualification_output, dtype="<f4").reshape(100, 14)
    assert np.array_equal(actions, qualification_actions)
    assert summary["num_actions"] == 100
    assert summary["action_dim"] == 14
    return summary, actions


def _run_reference(model_dir: Path, case: dict, tmp_path: Path) -> np.ndarray:
    source = _required_path(
        os.environ.get("TRTMC_REFERENCE_SOURCE_DIR"), "TRTMC_REFERENCE_SOURCE_DIR"
    )
    output = tmp_path / "reference.npz"
    subprocess.run(
        [
            os.environ.get("TRTMC_REFERENCE_PYTHON", sys.executable),
            str(TEST_ROOT / "official_reference.py"),
            "--source-root",
            str(source),
            "--checkpoint-dir",
            str(model_dir),
            "--image",
            str(_asset(case, "image")),
            "--state",
            str(_asset(case, "state")),
            "--output",
            str(output),
        ],
        check=True,
        timeout=1800,
    )
    with np.load(output, allow_pickle=False) as payload:
        return np.array(payload["actions"], copy=True)


def _assert_operational_summary(summary: dict) -> None:
    assert float(summary["control_frequency_hz"]) == 50.0
    for field in (
        "action_step_capacity_hz",
        "chunk_inference_p50_ms",
        "chunk_inference_p95_ms",
        "chunk_throughput_per_second",
        "gpu_memory_delta_mib",
        "gpu_memory_total_mib",
        "peak_resident_memory_mib",
        "startup_ms",
    ):
        value = float(summary[field])
        assert np.isfinite(value) and value > 0.0
    effective_hz = float(summary["control_effective_hz"])
    assert np.isfinite(effective_hz) and effective_hz >= 49.0
    jitter = float(summary["control_p99_abs_jitter_ms"])
    assert np.isfinite(jitter) and 0.0 <= jitter <= 5.0
    assert int(summary["control_missed_deadlines"]) == 0
    assert float(summary["gpu_memory_delta_mib"]) <= 1024.0
    assert float(summary["peak_resident_memory_mib"]) <= 2048.0
    assert float(summary["startup_ms"]) <= 5000.0


def test_operational_summary_keeps_the_old_limits() -> None:
    _assert_operational_summary(
        {
            "control_frequency_hz": 50.0,
            "action_step_capacity_hz": 100.0,
            "chunk_inference_p50_ms": 1.0,
            "chunk_inference_p95_ms": 2.0,
            "chunk_throughput_per_second": 1000.0,
            "control_effective_hz": 50.0,
            "control_p99_abs_jitter_ms": 1.0,
            "control_missed_deadlines": 0,
            "gpu_memory_delta_mib": 1.0,
            "gpu_memory_total_mib": 1024.0,
            "peak_resident_memory_mib": 1.0,
            "startup_ms": 1.0,
        }
    )


def test_operational_summary_rejects_invalid_active_invariants() -> None:
    summary = {
        "control_frequency_hz": 50.0,
        "action_step_capacity_hz": 100.0,
        "chunk_inference_p50_ms": 1.0,
        "chunk_inference_p95_ms": 2.0,
        "chunk_throughput_per_second": 1000.0,
        "control_effective_hz": 50.0,
        "control_p99_abs_jitter_ms": 1.0,
        "control_missed_deadlines": 0,
        "gpu_memory_delta_mib": 1.0,
        "gpu_memory_total_mib": 1024.0,
        "peak_resident_memory_mib": 1.0,
        "startup_ms": 1.0,
    }
    for field in ("chunk_inference_p50_ms", "chunk_throughput_per_second", "gpu_memory_total_mib"):
        with pytest.raises(AssertionError):
            _assert_operational_summary({**summary, field: 0.0})
    with pytest.raises(AssertionError):
        _assert_operational_summary({**summary, "control_frequency_hz": 49.0})
    with pytest.raises(AssertionError):
        _assert_operational_summary({**summary, "control_p99_abs_jitter_ms": -0.1})
    with pytest.raises(AssertionError):
        _assert_operational_summary({**summary, "control_effective_hz": np.inf})


def _assert_actions_in_training_bounds(actions: np.ndarray) -> None:
    values = np.asarray(actions, dtype=np.float32)
    assert values.shape == (100, 14)
    in_bounds = np.logical_and(values >= _ACTION_MIN, values <= _ACTION_MAX)
    assert float(np.mean(in_bounds)) >= 1.0


def test_actual_actions_must_be_inside_training_bounds() -> None:
    actions = np.broadcast_to((_ACTION_MIN + _ACTION_MAX) / 2.0, (100, 14)).copy()
    _assert_actions_in_training_bounds(actions)
    actions[0, 0] = _ACTION_MIN[0] - 0.1
    with pytest.raises(AssertionError):
        _assert_actions_in_training_bounds(actions)


def test_e2e(case_name: str, tmp_path: Path) -> None:
    manifest, case = CASES[case_name]
    import torch

    assert torch.cuda.is_available()
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    model_dir = _model_dir(manifest)
    bundle = tmp_path / manifest["bundle"]
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task="robot_control",
            precision=manifest["precision"],
        )
    )

    summary, actual = _run_native(binary, runtime_root, bundle, case, tmp_path)
    expected = _run_reference(model_dir, case, tmp_path)
    assert actual.shape == expected.shape == (100, 14)
    _assert_actions_in_training_bounds(actual)
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    thresholds = json.loads((THRESHOLD_ROOT / f"{case_name}.json").read_text(encoding="utf-8"))[
        "threshold_overrides"
    ]
    assert np.max(np.abs(delta)) <= float(thresholds["action_max_abs_error"])
    assert np.mean(np.abs(delta)) <= float(thresholds["action_mean_abs_error"])
    assert np.sqrt(np.mean(np.square(delta))) <= float(thresholds["action_rmse"])
    inference_ms = float(summary["chunk_inference_p95_ms"])
    assert inference_ms <= float(thresholds["chunk_inference_p95_ms"])
    assert float(summary["action_step_capacity_hz"]) >= float(
        thresholds["action_step_capacity_hz"]
    )
    _assert_operational_summary(summary)
