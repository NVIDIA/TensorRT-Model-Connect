# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused governance tests for the OpenPI E2E plugins."""

from __future__ import annotations

import hashlib
import json
import struct
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from tests.e2e.models.openpi import e2e_plugins
from tests.e2e.models.openpi.e2e_plugins import performance
from tests.e2e.models.openpi.e2e_plugins.comparators.robot_action import (
    RobotActionGenerationComparator,
)
from tests.e2e.models.openpi.e2e_plugins.references import upstream_replay
from tests.e2e.models.openpi.e2e_plugins.runners import robot_action
from tests.e2e_harness.contracts import (
    E2ECase,
    RunContext,
    StageOutput,
    StageSpec,
    ThresholdProfile,
)
from tests.e2e_harness.registry import (
    activate_model_plugins,
    get_comparator,
    get_reference,
    get_runner,
    reset,
)


def _case(tmp_path: Path) -> E2ECase:
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "prompt": "pick up the block",
                "cameras": {
                    "base_0_rgb": {"path": "base.ppm", "valid": True},
                    "left_wrist_0_rgb": {"path": "wrist.ppm", "valid": True},
                    "right_wrist_0_rgb": {"path": "missing.ppm", "valid": False},
                },
                "state": [0.0, 0.1],
                "initial_noise": [0.0, 0.0, 0.0, 0.0],
            }
        ),
        encoding="utf-8",
    )
    return E2ECase(
        name="openpi-unit",
        hf_id=e2e_plugins.OPENPI_SNAPSHOT_REPO_ID,
        family="openpi",
        runtime_strategy="openpi_vla",
        task_strategy="robot_action_generation",
        reference_backend="upstream_replay",
        bundle="pi05-droid.trtfb",
        inputs={
            "request_json": str(request),
            "profile": "pi05_droid",
            "reference_case_id": "droid_iris_20231204154425_f000005",
            "action_horizon": 15,
            "internal_action_dim": 32,
            "flow_steps": 10,
            "fixed_external_noise": True,
            "action_spans": [2.0, 4.0],
        },
        metadata={
            "performance_qualification": {
                "native_backend": "tensorrt_cpp",
                "baseline_backend": "pytorch_eager",
                "baseline_artifact": "performance/pytorch-eager.json",
                "baseline_sha256": performance.TORCH_EAGER_SHA256,
                "iterations": 1000,
                "warmups": 100,
            }
        },
    )


def _context(case: E2ECase, tmp_path: Path) -> RunContext:
    return RunContext(
        case=case,
        binary_path="/opt/trtmc/bin/trtmc",
        engine_dir=str(tmp_path / "engines"),
        artifacts_dir=str(tmp_path / "artifacts"),
    )


def _threshold() -> ThresholdProfile:
    return ThresholdProfile(
        task_strategy="robot_action_generation",
        metrics={
            "normalized_action_cosine_min": 0.9995,
            "normalized_action_mae_max": 0.003,
            "normalized_action_p99_abs_max": 0.01,
            "normalized_action_max_abs_max": 0.02,
            "physical_action_p99_span_fraction_max": 0.01,
            "stage_velocity_cosine_min": 0.9995,
            "stage_velocity_rmse_max": 0.005,
            "stage_velocity_max_abs_max": 0.05,
            "image_uint8_max_lsb": 1.0,
            "normalization_max_abs": 1e-6,
            "native_latency_p50_ms_max": 50.0,
            "native_latency_p95_ms_max": 60.0,
            "torch_eager_speedup_p50_min": 5.0,
            "torch_eager_speedup_p95_min": 5.0,
        },
    )


def _baseline_summary() -> dict:
    return {
        "artifact_sha256": performance.TORCH_EAGER_SHA256,
        "backend": "pytorch_eager",
        "profile_name": "pi05_droid",
        "iterations": 1000,
        "warmups": 100,
        "latency_ms": {
            "mean_ms": 385.0395223489992,
            "p50_ms": 384.67197,
            "p95_ms": 430.593005,
        },
        "effective_compile_mode": None,
        "torch_compile_guard_invocation_count": 0,
        "upstream_commit": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
        "hardware": {
            "gpu_name": "NVIDIA GB300",
            "tensorrt_version": "11.2.0.113",
        },
        "workload": dict(performance._EXPECTED_WORKLOAD),
    }


def _performance_receipt(
    *,
    mean_ms: float = 40.0,
    p50_ms: float = 40.0,
    p95_ms: float = 50.0,
) -> dict:
    stderr = (
        "[trtmc.openpi.benchmark] "
        f"action_ms={mean_ms:.6f} p50_ms={p50_ms:.6f} p95_ms={p95_ms:.6f} "
        "iterations=1000 warmup=100\n"
    )
    return {
        "schema_version": 1,
        "artifact_type": "openpi_performance_receipt",
        "native": performance.parse_native_benchmark(stderr),
        "torch_eager": _baseline_summary(),
        "native_stderr": stderr,
    }


def test_runner_invokes_native_act_and_parses_action_tensor(monkeypatch, tmp_path: Path) -> None:
    case = _case(tmp_path)
    context = _context(case, tmp_path)
    invocation = 0
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        nonlocal invocation
        invocation += 1
        commands.append(command)
        assert kwargs["env"].get("PYTHONPATH") is None or isinstance(
            kwargs["env"]["PYTHONPATH"], str
        )
        output_path = Path(command[command.index("--output-json") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "actions": [[0.1, -0.2], [0.3, 0.4]],
                    "horizon": 2,
                    "action_dim": 2,
                    "timings": {
                        "preprocess_ms": 1.0,
                        "prefill_ms": 2.0,
                        "denoise_ms": 3.0 + invocation,
                        "postprocess_ms": 0.5,
                    },
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=0,
            stdout=f"runtime log {invocation}",
            stderr=(
                f"runtime warning {invocation}\n"
                "[trtmc.openpi.benchmark] action_ms=40.000000 p50_ms=40.000000 "
                "p95_ms=50.000000 iterations=1000 warmup=100\n"
                if "--benchmark" in command
                else f"runtime warning {invocation}"
            ),
        )

    monkeypatch.setattr(robot_action.subprocess, "run", fake_run)
    monkeypatch.setattr(performance, "load_torch_eager_baseline", _baseline_summary)
    runner = robot_action.RobotActionGenerationRunner()
    output = runner.run_stage(case, StageSpec(name="actions"), context)
    rerun = runner.run_stage(case, StageSpec(name="actions"), context)

    command = output.metadata["command"]
    assert command[:2] == [
        "/opt/trtmc/bin/trtmc-openpi",
        str(tmp_path / "engines" / "pi05-droid.trtfb"),
    ]
    assert command[2:4] == ["--request-json", str(tmp_path / "request.json")]
    assert "--hf-python" not in command
    assert output.data["actions"] == [[0.1, -0.2], [0.3, 0.4]]
    assert output.data["horizon"] == 2
    assert output.data["action_dim"] == 2
    assert output.data["returncode"] == 0
    assert "timings" not in output.data
    assert "benchmark" not in output.data
    assert "stdout" not in output.data
    assert "stderr" not in output.data
    assert output.data == rerun.data
    assert invocation == 2
    assert sum("--benchmark" in command for command in commands) == 1
    assert output.metadata["command"][-4:] == ["--benchmark", "1000", "--warmup", "100"]
    assert "--benchmark" not in rerun.metadata["command"]
    assert output.metadata["runtime_timings"] != rerun.metadata["runtime_timings"]
    assert output.metadata["performance"] == rerun.metadata["performance"]
    assert output.metadata["runtime_stdout"] != rerun.metadata["runtime_stdout"]
    assert output.metadata["runtime_stderr"] != rerun.metadata["runtime_stderr"]
    assert output.metadata["runtime_contract"] == "native_cpp_tensorrt"

    same_path_context = _context(case, tmp_path)
    runner.run_stage(case, StageSpec(name="actions"), same_path_context)
    other_context = _context(case, tmp_path)
    other_context.artifacts_dir = str(tmp_path / "other-artifacts")
    runner.run_stage(case, StageSpec(name="actions"), other_context)
    renamed_case = deepcopy(case)
    renamed_case.name = "openpi-other-run"
    runner.run_stage(renamed_case, StageSpec(name="actions"), context)
    assert invocation == 5
    assert sum("--benchmark" in command for command in commands) == 4


def test_runner_preserves_failed_process_diagnostics(monkeypatch, tmp_path: Path) -> None:
    case = _case(tmp_path)
    context = _context(case, tmp_path)
    monkeypatch.setattr(
        robot_action.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            returncode=7,
            stdout="partial output",
            stderr="runtime failure",
        ),
    )

    output = robot_action.RobotActionGenerationRunner().run_stage(
        case, StageSpec(name="actions"), context
    )

    assert output.data == {
        "returncode": 7,
        "stdout": "partial output",
        "stderr": "runtime failure",
    }
    assert output.metadata["returncode"] == 7


def test_pinned_snapshot_resolver_is_offline_and_rejects_unsafe_paths(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(e2e_plugins, "snapshot_download", fake_snapshot_download)
    e2e_plugins.openpi_snapshot_root.cache_clear()

    assert e2e_plugins.openpi_snapshot_path("request", "request.json") == (
        snapshot / "request" / "request.json"
    )
    assert e2e_plugins.openpi_proof_path("request", "request.json") == (
        snapshot / "trtmc_openpi" / "request" / "request.json"
    )
    assert e2e_plugins.openpi_snapshot_root() == snapshot
    assert calls == [
        {
            "repo_id": e2e_plugins.OPENPI_SNAPSHOT_REPO_ID,
            "revision": e2e_plugins.OPENPI_SNAPSHOT_REVISION,
            "allow_patterns": list(e2e_plugins.OPENPI_SNAPSHOT_ALLOW_PATTERNS),
            "local_files_only": True,
        }
    ]
    for unsafe in ("../request.json", "/request.json"):
        try:
            e2e_plugins.openpi_snapshot_path(unsafe)
        except ValueError as error:
            assert "snapshot-relative" in str(error)
        else:
            raise AssertionError(f"unsafe snapshot path {unsafe!r} was accepted")
    e2e_plugins.openpi_snapshot_root.cache_clear()


def test_runner_request_falls_back_to_pinned_snapshot(monkeypatch, tmp_path: Path) -> None:
    case = _case(tmp_path)
    case.inputs.pop("request_json")
    snapshot = tmp_path / "snapshot"
    request = snapshot / "request" / "request.json"
    request.parent.mkdir(parents=True)
    request.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        robot_action,
        "openpi_proof_path",
        lambda *parts: snapshot.joinpath(*parts),
    )

    assert robot_action._resolve_request_path(case) == request

    explicit = tmp_path / "explicit-request.json"
    explicit.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("TRTMC_OPENPI_REQUEST_JSON", str(explicit))
    assert robot_action._resolve_request_path(case) == explicit


def test_runner_uses_one_native_capture_for_stagewise_outputs(monkeypatch, tmp_path: Path) -> None:
    case = _case(tmp_path)
    context = _context(case, tmp_path)
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        del kwargs
        calls.append(command)
        output_path = Path(command[command.index("--output-json") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "actions": [[0.1, -0.2], [0.3, 0.4]],
                    "horizon": 2,
                    "action_dim": 2,
                    "timings": {
                        "preprocess_ms": 1.0,
                        "prefill_ms": 2.0,
                        "denoise_ms": 3.0,
                        "postprocess_ms": 0.5,
                    },
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    descriptors = {
        name: {
            "path": str(tmp_path / f"{name}.bin"),
            "dtype": "float32",
            "shape": [1],
            "byte_length": 4,
            "sha256": "a" * 64,
        }
        for names in robot_action._STAGE_TENSORS.values()
        for name in names
    }
    descriptors["normalized_actions"] = {
        "path": str(tmp_path / "normalized_actions.bin"),
        "dtype": "float32",
        "shape": [1, 2, 2],
        "byte_length": 16,
        "sha256": "a" * 64,
    }
    monkeypatch.setattr(robot_action.subprocess, "run", fake_run)
    monkeypatch.setattr(
        robot_action, "_load_native_diagnostic_manifest", lambda path, profile: descriptors
    )
    runner = robot_action.RobotActionGenerationRunner()
    preprocess = runner.run_stage(case, StageSpec(name="preprocess"), context)
    vision = runner.run_stage(case, StageSpec(name="vision"), context)
    action_output = StageOutput(stage_name="actions", data={})
    runner._attach_normalized_actions(action_output, case, context)

    assert len(calls) == 1
    assert "--qualification-diagnostics" in calls[0]
    assert set(preprocess.data["tensor_files"]) == set(robot_action._STAGE_TENSORS["preprocess"])
    assert set(vision.data["tensor_files"]) == {"vision_tokens"}
    assert set(action_output.data["tensor_files"]) == {"normalized_actions"}
    assert preprocess.metadata["runtime_contract"] == "native_cpp_tensorrt"


def test_native_capture_manifest_is_hash_and_contract_checked(monkeypatch, tmp_path: Path) -> None:
    tensor_dir = tmp_path / "tensors"
    tensor_dir.mkdir()
    payload = tensor_dir / "token_ids.bin"
    payload.write_bytes(struct.pack("<i", 7))
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    contract = {
        "token_ids": {
            "stage": "preprocess",
            "role": "intermediate",
            "dtype": "int32",
            "shape": [1],
        }
    }
    manifest = {
        "schema_version": 1,
        "artifact_type": "trtmc_action_qualification_diagnostics",
        "runtime_contract": "native_cpp_tensorrt",
        "model_id": "openpi-test",
        "tensors": {
            "token_ids": {
                "path": "tensors/token_ids.bin",
                **contract["token_ids"],
                "byte_length": 4,
                "sha256": digest,
            }
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(robot_action.qualification, "load_contract", lambda profile: {})
    monkeypatch.setattr(
        robot_action.qualification, "_expected_tensor_contract", lambda loaded: contract
    )
    descriptors = robot_action._load_native_diagnostic_manifest(manifest_path, "pi05_droid")
    assert descriptors["token_ids"]["path"] == str(payload.resolve())

    payload.write_bytes(struct.pack("<i", 8))
    try:
        robot_action._load_native_diagnostic_manifest(manifest_path, "pi05_droid")
    except ValueError as error:
        assert "SHA-256 mismatch" in str(error)
    else:
        raise AssertionError("modified native capture was accepted")


def test_action_comparator_enforces_span_normalized_and_decision_gates(
    monkeypatch,
) -> None:
    monkeypatch.setattr(performance, "load_torch_eager_baseline", _baseline_summary)
    comparator = RobotActionGenerationComparator()
    reference = StageOutput(
        stage_name="actions",
        data={
            "physical_actions": [[0.10, -0.20], [0.30, 0.40]],
            "normalized_actions": [[0.10, -0.20], [0.30, 0.40]],
            "horizon": 2,
            "action_dim": 2,
            "action_spans": [2.0, 4.0],
            "decision_indices": [1],
        },
    )
    close = StageOutput(
        stage_name="actions",
        data={
            "actions": [[0.101, -0.199], [0.299, 0.401]],
            "normalized_actions": [[0.101, -0.199], [0.299, 0.401]],
            "horizon": 2,
            "action_dim": 2,
            "returncode": 0,
        },
        metadata={"performance": _performance_receipt()},
    )
    passed = comparator.compare(close, reference, _threshold(), StageSpec(name="actions"))
    assert passed.status == "passed"
    assert passed.metrics["physical_action_p99_span_fraction"].passed
    assert passed.metrics["normalized_action_cosine"].passed
    performance_metrics = {
        "native_latency_p50_ms",
        "native_latency_p95_ms",
        "torch_eager_latency_p50_ms",
        "torch_eager_latency_p95_ms",
        "torch_eager_speedup_p50",
        "torch_eager_speedup_p95",
        "torch_compile_invocation_count",
    }
    assert performance_metrics <= set(passed.metrics)
    assert passed.metrics["torch_eager_latency_p50_ms"].threshold is None
    assert passed.metrics["torch_eager_latency_p95_ms"].threshold is None
    for metric_name in (
        "physical_action_cosine",
        "physical_action_mae",
        "physical_action_max_abs",
    ):
        assert passed.metrics[metric_name].threshold is None
        assert passed.metrics[metric_name].operator == "info"

    drifted = StageOutput(
        stage_name="actions",
        data={
            "actions": [[0.20, 0.20], [0.30, 0.40]],
            "normalized_actions": [[0.20, 0.20], [0.30, 0.40]],
            "horizon": 2,
            "action_dim": 2,
            "returncode": 0,
        },
        metadata={"performance": _performance_receipt()},
    )
    failed = comparator.compare(drifted, reference, _threshold(), StageSpec(name="actions"))
    assert failed.status == "failed"
    assert not failed.metrics["physical_action_p99_span_fraction"].passed
    assert not failed.metrics["sign_or_gripper_decision_changes"].passed

    missing_normalized = deepcopy(close)
    missing_normalized.data.pop("normalized_actions")
    missing_result = comparator.compare(
        missing_normalized, reference, _threshold(), StageSpec(name="actions")
    )
    assert missing_result.status == "error"
    assert "normalized action parity evidence is required" in missing_result.message


def test_action_comparator_enforces_all_performance_gates(monkeypatch) -> None:
    baseline = _baseline_summary()
    monkeypatch.setattr(performance, "load_torch_eager_baseline", lambda: baseline)
    reference = StageOutput(
        stage_name="actions",
        data={
            "physical_actions": [[0.1, -0.2], [0.3, 0.4]],
            "normalized_actions": [[0.1, -0.2], [0.3, 0.4]],
            "horizon": 2,
            "action_dim": 2,
            "action_spans": [2.0, 4.0],
        },
    )
    actual = StageOutput(
        stage_name="actions",
        data={
            "actions": [[0.1, -0.2], [0.3, 0.4]],
            "normalized_actions": [[0.1, -0.2], [0.3, 0.4]],
            "horizon": 2,
            "action_dim": 2,
            "returncode": 0,
        },
        metadata={"performance": _performance_receipt(p50_ms=80.0, p95_ms=90.0)},
    )

    result = RobotActionGenerationComparator().compare(
        actual, reference, _threshold(), StageSpec(name="actions")
    )
    assert result.status == "failed"
    for name in (
        "native_latency_p50_ms",
        "native_latency_p95_ms",
        "torch_eager_speedup_p50",
        "torch_eager_speedup_p95",
    ):
        assert not result.metrics[name].passed

    compile_baseline = _baseline_summary()
    compile_baseline["torch_compile_guard_invocation_count"] = 1
    monkeypatch.setattr(performance, "load_torch_eager_baseline", lambda: compile_baseline)
    actual.metadata["performance"]["torch_eager"] = compile_baseline
    compile_result = RobotActionGenerationComparator().compare(
        actual, reference, _threshold(), StageSpec(name="actions")
    )
    assert compile_result.status == "failed"
    assert not compile_result.metrics["torch_compile_invocation_count"].passed

    missing_threshold = _threshold()
    missing_threshold.metrics.pop("torch_eager_speedup_p95_min")
    missing_result = RobotActionGenerationComparator().compare(
        actual, reference, missing_threshold, StageSpec(name="actions")
    )
    assert missing_result.status == "error"


def test_preprocess_comparator_requires_exact_tokens_masks_and_noise() -> None:
    comparator = RobotActionGenerationComparator()
    reference = StageOutput(
        stage_name="preprocess",
        data={
            "token_ids": [2, 10, 1, 0],
            "token_mask": [1, 1, 1, 0],
            "image_mask": [1, 1, 0],
            "initial_noise": [0.25, -0.5],
            "preprocessed_images": [-1.0, 0.0, 1.0],
            "normalized_state": [0.1, -0.2],
        },
    )
    actual = StageOutput(stage_name="preprocess", data=dict(reference.data))
    passed = comparator.compare(actual, reference, _threshold(), StageSpec(name="preprocess"))
    assert passed.status == "passed"

    actual.data["token_ids"] = [2, 11, 1, 0]
    failed = comparator.compare(actual, reference, _threshold(), StageSpec(name="preprocess"))
    assert failed.status == "failed"
    assert failed.metrics["token_mismatch_count"].value == 1.0


def test_upstream_reference_requires_a_validated_512_case_set(monkeypatch, tmp_path: Path) -> None:
    index = tmp_path / "reference-set.json"
    index.write_text(json.dumps({"artifact_type": "openpi_pinned_reference_set"}), encoding="utf-8")
    observed: dict[str, int] = {}

    def fake_validate(path, *, verify_payloads=True, minimum_cases=1):
        del path, verify_payloads
        observed["minimum_cases"] = minimum_cases
        return {"artifact_type": "openpi_pinned_reference_set", "cases": []}

    upstream_replay._load_validated_document.cache_clear()
    monkeypatch.setattr(upstream_replay.qualification, "validate_reference_set", fake_validate)
    upstream_replay._load_validated_document(str(index))
    assert observed["minimum_cases"] == 512


def test_upstream_reference_and_norm_stats_fall_back_to_pinned_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    case = _case(tmp_path)
    case.inputs.pop("action_spans")
    snapshot = tmp_path / "snapshot"
    reference = snapshot / "reference" / "reference-set.json"
    reference.parent.mkdir(parents=True)
    reference.write_text("{}", encoding="utf-8")
    normalization = snapshot / "preprocessor_config.json"
    normalization.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "actions": {
                        "q01": [-1.0, -2.0],
                        "q99": [1.0, 2.0],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "openpi_config.json").write_text(
        json.dumps(
            {
                "profile": "pi05_droid",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        upstream_replay,
        "openpi_snapshot_path",
        lambda *parts: snapshot.joinpath(*parts),
    )
    monkeypatch.setattr(
        upstream_replay,
        "openpi_proof_path",
        lambda *parts: snapshot.joinpath(*parts),
    )

    assert upstream_replay._replay_path(case) == reference
    assert upstream_replay._load_action_spans(case, 2) == [2.0, 4.0]

    explicit_reference = tmp_path / "explicit-reference.json"
    explicit_reference.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("TRTMC_OPENPI_REFERENCE_ARTIFACT", str(explicit_reference))
    assert upstream_replay._replay_path(case) == explicit_reference


def test_upstream_reference_decodes_pinned_action_payload(monkeypatch, tmp_path: Path) -> None:
    normalized_path = tmp_path / "normalized.bin"
    physical_path = tmp_path / "physical.bin"
    normalized_path.write_bytes(struct.pack("<4f", 0.1, -0.2, 0.3, 0.4))
    physical_path.write_bytes(struct.pack("<4f", 1.0, -2.0, 3.0, 4.0))
    artifact = {
        "profile_name": "pi05_droid",
        "case": {"id": "case-0001"},
        "upstream": {"checkpoint": {"sha256": "a" * 64}},
        "tensors": {
            "normalized_actions": {
                "path": normalized_path.name,
                "dtype": "float32",
                "shape": [1, 2, 2],
                "sha256": "b" * 64,
            },
            "physical_actions": {
                "path": physical_path.name,
                "dtype": "float32",
                "shape": [1, 2, 2],
                "sha256": "c" * 64,
            },
        },
    }
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        upstream_replay,
        "_resolve_case_artifact",
        lambda case: (artifact_path, artifact),
    )
    case = _case(tmp_path)
    output = upstream_replay.UpstreamReplayReference().run_stage(
        case, StageSpec(name="actions"), _context(case, tmp_path)
    )
    assert output.data["physical_actions"] == [[1.0, -2.0], [3.0, 4.0]]
    assert output.data["action_spans"] == [2.0, 4.0]
    assert output.metadata["upstream_commit"] == ("15a9616a00943ada6c20a0f158e3adb39df2ccac")


def test_model_plugin_activation_resolves_all_openpi_protocols() -> None:
    model_dir = Path(__file__).resolve().parent
    try:
        activate_model_plugins(model_dir)
        assert type(get_runner("robot_action_generation")).__module__.startswith(
            "tests.e2e.models.openpi.e2e_plugins"
        )
        assert type(get_reference("upstream_replay")).__module__.startswith(
            "tests.e2e.models.openpi.e2e_plugins"
        )
        assert type(get_comparator("robot_action_generation")).__module__.startswith(
            "tests.e2e.models.openpi.e2e_plugins"
        )
    finally:
        reset()


def test_runtime_strategy_matrix_declares_native_action_governance() -> None:
    matrix_path = Path(__file__).resolve().parents[3] / "runtime_strategy_matrix.yaml"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    entry = matrix["runtime_strategies"]["openpi_vla"]
    assert entry["task_strategy"] == "robot_action_generation"
    assert entry["cli_commands"] == ["trtmc-openpi"]
    assert entry["performance_mode"] == "multi_stage"
    assert entry["diff_framework_check_classes"] == []
    assert "No diff_framework check currently registers" in entry["diff_framework_exemption"]
    assert "openpi_vla" in matrix["new_runtime_guard_strategies"]
