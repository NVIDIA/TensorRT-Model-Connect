# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for E2EOrchestrator lifecycle phase outcomes."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from tests.e2e_harness import orchestrator
from tests.e2e_harness.contracts import (
    CompareResult,
    E2ECase,
    E2EStatus,
    FailureType,
    MetricResult,
    PreflightRequirement,
    RunContext,
    StageOutput,
    StageSpec,
    StageStatus,
)
from tests.e2e_harness.orchestrator import E2EOrchestrator


def test_manifest_build_args_include_decoder_engine_layout() -> None:
    command = ["trtmc", "build"]

    orchestrator._append_manifest_build_args(
        command,
        {"decoder_engine_layout": "dual_profile"},
    )

    assert command[-2:] == ["--decoder-engine-layout", "dual_profile"]


class _FakeRunner:
    strategy_name = "unit_task"

    def __init__(
        self,
        outputs: list[StageOutput] | None = None,
        error: str | None = None,
    ) -> None:
        self._outputs = outputs or []
        self._error = error
        self.calls = 0

    def run_stage(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
    ) -> StageOutput:
        self.calls += 1
        if self._error is not None:
            raise RuntimeError(self._error)
        if self._outputs:
            index = min(self.calls - 1, len(self._outputs) - 1)
            return self._outputs[index]
        return StageOutput(stage_name=stage.name, text="trt", data={"value": 1})


class _FakeReference:
    backend_name = "unit_ref"

    def __init__(self, error: str | None = None) -> None:
        self._error = error
        self.calls = 0

    def run_stage(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
    ) -> StageOutput:
        self.calls += 1
        if self._error is not None:
            raise RuntimeError(self._error)
        return StageOutput(stage_name=stage.name, text="ref", data={"value": 1})


class _FakeComparator:
    task_strategy = "unit_task"

    def __init__(
        self,
        result: CompareResult | None = None,
        error: str | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls = 0

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: Any,
        stage: StageSpec,
    ) -> CompareResult:
        self.calls += 1
        if self._error is not None:
            raise RuntimeError(self._error)
        if self._result is not None:
            return self._result
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value,
            metrics={
                "value_match": MetricResult(
                    value=1.0,
                    threshold=1.0,
                    operator="==",
                    passed=True,
                ),
            },
            message="matched",
        )


@pytest.fixture(autouse=True)
def _stable_env_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tests.e2e_harness.artifact_sink._collect_env_fingerprint",
        lambda ctx=None: {"unit": "env"},
    )


def test_manifest_build_env_resolves_model_relative_paths(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    case = E2ECase(
        name="unit",
        hf_id="unit/model",
        family="unit",
        runtime_strategy="unit_runtime",
        metadata={
            "model_test_dir": str(model_dir),
            "build_env": {
                "UNIT_MODEL_ASSET": {
                    "path": "data/input.png",
                    "relative_to": "model",
                },
                "UNIT_LITERAL": "enabled",
            },
        },
    )

    env: dict[str, str] = {}
    orchestrator._apply_manifest_build_env(env, case)

    assert env["UNIT_MODEL_ASSET"] == str(model_dir / "data/input.png")
    assert env["UNIT_LITERAL"] == "enabled"


def test_manifest_build_env_requires_injected_values(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "private.cache"
    asset.write_bytes(b"cache")
    case = E2ECase(
        name="unit",
        hf_id="unit/model",
        family="unit",
        runtime_strategy="unit_runtime",
        metadata={
            "build_env": {
                "TRTMC_BARK_TIMING_CACHE_PATH": {
                    "required_from_env": True,
                    "path_like": True,
                },
                "TRTMC_BARK_TIMING_CACHE_SHA256": {
                    "required_from_env": True,
                },
            },
        },
    )

    env = {
        "TRTMC_BARK_TIMING_CACHE_PATH": str(asset),
        "TRTMC_BARK_TIMING_CACHE_SHA256": "opaque-digest",
        "TRTMC_ELF_TIMING_CACHE_PATH": str(asset),
        "TRTMC_ELF_TIMING_CACHE_METADATA_PATH": str(asset),
    }
    orchestrator._apply_manifest_build_env(env, case)

    assert env["TRTMC_BARK_TIMING_CACHE_PATH"] == str(asset)
    assert env["TRTMC_BARK_TIMING_CACHE_SHA256"] == "opaque-digest"
    assert "TRTMC_ELF_TIMING_CACHE_PATH" not in env
    assert "TRTMC_ELF_TIMING_CACHE_METADATA_PATH" not in env

    with pytest.raises(
        RuntimeError,
        match=(
            "required build environment variable "
            "TRTMC_BARK_TIMING_CACHE_PATH is missing"
        ),
    ):
        orchestrator._apply_manifest_build_env({}, case)


def _make_case(
    name: str,
    *,
    preflight: list[PreflightRequirement] | None = None,
    determinism: dict[str, Any] | None = None,
) -> E2ECase:
    return E2ECase(
        name=name,
        hf_id="hf/unit-model",
        family="unit",
        runtime_strategy="unit_runtime",
        task_strategy="unit_task",
        reference_backend="unit_ref",
        bundle=f"{name}.bundle",
        preflight=preflight or [],
        stages=[StageSpec(name="generate")],
        determinism=determinism or {},
    )


def _make_ctx(tmp_path: Path, case: E2ECase) -> RunContext:
    engine_dir = tmp_path / "engines"
    engine_dir.mkdir()
    return RunContext(
        case=case,
        artifacts_dir=str(tmp_path / "artifacts"),
        binary_path="/tmp/trtmc",
        hf_python=sys.executable,
        engine_dir=str(engine_dir),
    )


def test_ci_engine_build_guard_passes_manifest_identity_to_builder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case("unit-build")
    case.metadata.update(
        {
            "model_name": "unit-model-config",
            "manifest_path": "/src/tests/e2e/models/unit/manifests/unit.json",
        }
    )
    ctx = _make_ctx(tmp_path, case)
    guard_dir = tmp_path / "engine-builds"
    monkeypatch.setenv("TRTMC_ENGINE_BUILD_GUARD_DIR", str(guard_dir))
    monkeypatch.setenv("TRTMC_ENGINE_BUILD_REVISION", "abc123")
    build_environments: list[dict[str, str]] = []

    def fake_run(cmd, **kwargs):
        build_environments.append(kwargs["env"])
        bundle_path = Path(cmd[cmd.index("-o") + 1])
        bundle_path.write_bytes(b"bundle")
        timing_path = Path(cmd[cmd.index("--build-timing-json") + 1])
        timing_path.write_text('{"total_s": 1.0}\n', encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="built", stderr="")

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)

    result = orchestrator._resolve_bundle(case, ctx)

    assert result[0] == str(Path(ctx.engine_dir) / case.bundle)
    assert result[2] == ""
    assert len(build_environments) == 1
    build_env = build_environments[0]
    assert build_env["TRTMC_ENGINE_BUILD_IDENTITY"] == "unit-model-config"
    assert build_env["TRTMC_ENGINE_BUILD_REVISION"] == "abc123"
    command = json.loads(build_env["TRTMC_ENGINE_BUILD_COMMAND_JSON"])
    assert command[command.index("-o") + 1] == str(Path(ctx.engine_dir) / case.bundle)


def test_bundle_build_resolves_native_plugins_beside_runtime_binary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case("unit-build")
    ctx = _make_ctx(tmp_path, case)
    native_dir = tmp_path / "native"
    native_dir.mkdir()
    ctx.binary_path = str(native_dir / "trtmc")
    build_environments: list[dict[str, str]] = []

    def fake_run(cmd, **kwargs):
        build_environments.append(kwargs["env"])
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"bundle")
        return subprocess.CompletedProcess(cmd, 0, stdout="built", stderr="")

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)

    bundle, _elapsed, error, _build_info = orchestrator._resolve_bundle(case, ctx)

    assert bundle == str(Path(ctx.engine_dir) / case.bundle)
    assert error == ""
    assert len(build_environments) == 1
    assert build_environments[0]["_TRTMC_INTERNAL_NATIVE_BIN_DIR"] == str(native_dir)


@pytest.mark.parametrize(
    ("max_cache_length", "expected_flag"),
    ((None, False), (256, True)),
)
def test_bundle_build_only_emits_declared_max_cache_length(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    max_cache_length: int | None,
    expected_flag: bool,
) -> None:
    case = _make_case("unit-build")
    if max_cache_length is not None:
        case.inputs["max_cache_length"] = max_cache_length
    ctx = _make_ctx(tmp_path, case)
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"bundle")
        Path(cmd[cmd.index("--build-timing-json") + 1]).write_text(
            '{"total_s": 1.0}\n',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="built", stderr="")

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)

    bundle, _elapsed, error, _build_info = orchestrator._resolve_bundle(
        case,
        ctx,
    )
    repro = orchestrator._build_repro_commands(case, ctx, bundle, {})

    assert error == ""
    assert len(commands) == 1
    assert ("--max-cache-length" in commands[0]) is expected_flag
    assert ("--max-cache-length" in repro["build_bundle"]) is expected_flag
    if expected_flag:
        index = commands[0].index("--max-cache-length")
        assert commands[0][index + 1] == str(max_cache_length)


def test_ci_bundle_build_recovers_one_sigsegv_in_a_fresh_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case("unit-build")
    case.metadata["model_name"] = "unit-model-config"
    ctx = _make_ctx(tmp_path, case)
    monkeypatch.setenv(
        "TRTMC_ENGINE_BUILD_GUARD_DIR",
        str(tmp_path / "engine-builds"),
    )
    case.metadata["build_timeout_s"] = 100
    build_environments: list[dict[str, str]] = []
    build_timeouts: list[float] = []
    clock = {"now": 0.0}
    monkeypatch.setattr(orchestrator.time, "monotonic", lambda: clock["now"])

    def fake_run(cmd, **kwargs):
        build_environments.append(dict(kwargs["env"]))
        build_timeouts.append(kwargs["timeout"])
        bundle_path = Path(cmd[cmd.index("-o") + 1])
        timing_path = Path(cmd[cmd.index("--build-timing-json") + 1])
        if len(build_environments) == 1:
            bundle_path.write_bytes(b"partial")
            timing_path.write_text('{"partial": true}\n', encoding="utf-8")
            clock["now"] = 90.0
            return subprocess.CompletedProcess(
                cmd,
                -signal.SIGSEGV,
                stdout="",
                stderr="native builder crashed",
            )
        assert not bundle_path.exists()
        assert not timing_path.exists()
        bundle_path.write_bytes(b"bundle")
        timing_path.write_text('{"total_s": 1.0}\n', encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="built", stderr="")

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)

    bundle, _elapsed, error, build_info = orchestrator._resolve_bundle(case, ctx)

    assert bundle == str(Path(ctx.engine_dir) / case.bundle)
    assert error == ""
    assert len(build_environments) == 2
    assert "TRTMC_ENGINE_BUILD_RECOVERY_ATTEMPT" not in build_environments[0]
    assert build_environments[1]["TRTMC_ENGINE_BUILD_RECOVERY_ATTEMPT"] == "2"
    assert build_environments[1]["TRTMC_ENGINE_BUILD_RECOVERY_SIGNAL"] == str(signal.SIGSEGV)
    assert build_timeouts == [100, 10.0]
    assert build_info["attempt_count"] == 2
    assert [
        (attempt["attempt"], attempt["returncode"], attempt["signal"])
        for attempt in build_info["recovery_attempts"]
    ] == [(1, -signal.SIGSEGV, signal.SIGSEGV)]
    recovery_timing = (
        Path(ctx.artifacts_dir) / case.name / "build_timing.attempt-1.json"
    )
    assert json.loads(recovery_timing.read_text(encoding="utf-8")) == {
        "partial": True
    }
    assert build_info["recovery_attempts"][0]["timing_path"] == str(
        recovery_timing
    )


def test_ci_bundle_build_does_not_retry_an_ordinary_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case("unit-build")
    case.metadata["model_name"] = "unit-model-config"
    ctx = _make_ctx(tmp_path, case)
    monkeypatch.setenv(
        "TRTMC_ENGINE_BUILD_GUARD_DIR",
        str(tmp_path / "engine-builds"),
    )
    calls = 0

    def fake_run(cmd, **kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            cmd,
            1,
            stdout="",
            stderr="deterministic build failure",
        )

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)

    bundle, _elapsed, error, build_info = orchestrator._resolve_bundle(case, ctx)

    assert bundle is None
    assert "rc=1" in error
    assert calls == 1
    assert build_info["attempt_count"] == 1
    assert build_info["recovery_attempts"] == []


def test_ci_bundle_build_fails_after_a_second_sigsegv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case("unit-build")
    case.metadata["model_name"] = "unit-model-config"
    ctx = _make_ctx(tmp_path, case)
    monkeypatch.setenv(
        "TRTMC_ENGINE_BUILD_GUARD_DIR",
        str(tmp_path / "engine-builds"),
    )
    calls = 0

    def fake_run(cmd, **kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            cmd,
            -signal.SIGSEGV,
            stdout="",
            stderr=f"native builder crash {calls}",
        )

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)

    bundle, _elapsed, error, build_info = orchestrator._resolve_bundle(case, ctx)

    assert bundle is None
    assert f"rc={-signal.SIGSEGV}" in error
    assert calls == 2
    assert build_info["attempt_count"] == 2
    assert len(build_info["recovery_attempts"]) == 1


def test_ci_bundle_build_does_not_claim_a_retry_without_time_remaining(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case("unit-build")
    case.metadata.update(
        {
            "model_name": "unit-model-config",
            "build_timeout_s": 100,
        }
    )
    ctx = _make_ctx(tmp_path, case)
    monkeypatch.setenv(
        "TRTMC_ENGINE_BUILD_GUARD_DIR",
        str(tmp_path / "engine-builds"),
    )
    clock = {"now": 0.0}
    monkeypatch.setattr(orchestrator.time, "monotonic", lambda: clock["now"])
    calls = 0

    def fake_run(cmd, **kwargs):
        nonlocal calls
        calls += 1
        clock["now"] = 100.0
        return subprocess.CompletedProcess(
            cmd,
            -signal.SIGSEGV,
            stdout="",
            stderr="native builder crash at deadline",
        )

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)

    bundle, _elapsed, error, build_info = orchestrator._resolve_bundle(case, ctx)

    assert bundle is None
    assert f"rc={-signal.SIGSEGV}" in error
    assert calls == 1
    assert build_info["attempt_count"] == 1
    assert build_info["recovery_attempts"] == []


def test_hf_auth_preflight_accepts_a_warmed_offline_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case("gated-offline")
    ctx = _make_ctx(tmp_path, case)
    snapshot = tmp_path / "hf-cache" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    def resolve_offline(
        hf_id: str,
        *,
        allow_patterns: list[str],
        local_files_only: bool,
    ) -> str:
        assert hf_id == case.hf_id
        assert "config.json" in allow_patterns
        assert "model.safetensors" in allow_patterns
        assert "LICENSE" not in allow_patterns
        assert "sam3.pt" not in allow_patterns
        assert local_files_only is True
        return str(snapshot)

    huggingface_hub = types.ModuleType("huggingface_hub")
    huggingface_hub.snapshot_download = resolve_offline
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)

    passed, message = orchestrator._check_hf_auth(
        ctx,
        PreflightRequirement(
            kind="hf_auth_token_present",
            args={"hf_id": case.hf_id},
        ),
    )

    assert passed is True
    assert message == f"HF snapshot available offline: {case.hf_id}"


def test_hf_auth_preflight_rejects_a_missing_offline_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case("gated-missing")
    ctx = _make_ctx(tmp_path, case)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    def missing(*_args, **_kwargs):
        raise RuntimeError("not cached")

    huggingface_hub = types.ModuleType("huggingface_hub")
    huggingface_hub.snapshot_download = missing
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)

    passed, message = orchestrator._check_hf_auth(
        ctx,
        PreflightRequirement(
            kind="hf_auth_token_present",
            args={"hf_id": case.hf_id},
        ),
    )

    assert passed is False
    assert "snapshot is unavailable offline" in message


def test_bundle_build_honors_manifest_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case("custom-build-timeout")
    case.metadata["build_timeout_s"] = 5400
    ctx = _make_ctx(tmp_path, case)
    captured: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)

    bundle, _elapsed, error, _build_info = orchestrator._resolve_bundle(case, ctx)

    assert bundle == str(Path(ctx.engine_dir) / case.bundle)
    assert error == ""
    assert captured["kwargs"]["timeout"] == 5400


def _patch_bundle_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / "engines" / "unit.bundle"
    bundle_path.parent.mkdir(exist_ok=True)
    bundle_path.write_text("bundle", encoding="utf-8")
    monkeypatch.setattr(
        orchestrator,
        "_resolve_bundle",
        lambda case, ctx: (str(bundle_path), None, "", {}),
    )


def _patch_plugins(
    monkeypatch: pytest.MonkeyPatch,
    *,
    runner: _FakeRunner | None = None,
    reference: _FakeReference | None = None,
    comparator: _FakeComparator | None = None,
) -> None:
    monkeypatch.setattr(orchestrator, "get_runner", lambda name: runner)
    monkeypatch.setattr(orchestrator, "get_reference", lambda name: reference)
    monkeypatch.setattr(orchestrator, "get_comparator", lambda name: comparator)
    monkeypatch.setattr(orchestrator, "get_contract_plugin", lambda name: None)


def _read_result_json(ctx: RunContext, case: E2ECase) -> dict[str, Any]:
    path = Path(ctx.artifacts_dir) / case.name / "result.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_auto_register_artifacts_includes_reference_visuals_and_metadata_audio(
    tmp_path: Path,
) -> None:
    class Sink:
        base_dir = tmp_path

        def __init__(self) -> None:
            self.artifacts: dict[str, str] = {}

        def register_artifact(self, key: str, rel_path: str) -> None:
            self.artifacts[key] = rel_path

    sink = Sink()
    input_image = tmp_path / "input.png"
    visual = tmp_path / "hf_seg_viz.png"
    audio = tmp_path / "talker_decode.wav"
    input_image.write_bytes(b"png")
    visual.write_bytes(b"png")
    audio.write_bytes(b"wav")
    output = StageOutput(
        stage_name="full_inference",
        data={"input_image_path": str(input_image), "viz_path": str(visual)},
        metadata={"audio_output_path": str(audio)},
    )

    orchestrator._auto_register_artifacts(sink, output, "ref")

    assert sink.artifacts == {
        "ref_input_image": "input.png",
        "ref_segmentation_map": "hf_seg_viz.png",
        "ref_wav": "talker_decode.wav",
    }

    sibling_prefix = Path(f"{tmp_path}-sibling") / "outside.wav"
    outside = StageOutput(
        stage_name="full_inference",
        data={"wav_path": str(sibling_prefix)},
    )
    orchestrator._auto_register_artifacts(sink, outside, "trt")
    assert sink.artifacts["trt_wav"] == str(sibling_prefix)


def test_run_returns_preflight_skip_without_resolving_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case(
        "preflight-skip",
        preflight=[PreflightRequirement(kind="unknown_preflight")],
    )
    ctx = _make_ctx(tmp_path, case)
    monkeypatch.setattr(
        orchestrator,
        "_resolve_bundle",
        lambda case, ctx: pytest.fail("bundle resolution should not run"),
    )

    result = E2EOrchestrator().run(case, ctx)

    assert result.status == E2EStatus.SKIP.value
    assert result.failure_type == FailureType.PRECHECK_FAIL.value
    assert result.stages == {}
    assert result.determinism["preflight"][0]["kind"] == "unknown_preflight"
    data = _read_result_json(ctx, case)
    assert data["status"] == E2EStatus.SKIP.value
    assert data["failure_type"] == FailureType.PRECHECK_FAIL.value
    assert data["stages"] == {}


def test_python_module_preflight_rejects_module_that_fails_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_name = "broken_preflight_module"
    (tmp_path / f"{module_name}.py").write_text(
        'raise OSError("native extension failed to load")\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    case = _make_case("broken-module")
    ctx = _make_ctx(tmp_path, case)

    passed, message = orchestrator._check_python_module(
        ctx,
        PreflightRequirement(
            kind="python_module_available",
            args={"module": module_name, "phase": "build"},
        ),
    )

    assert not passed
    assert "not importable in build profile" in message
    assert "native extension failed to load" in message


def test_run_returns_build_failure_and_logs_build_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case("build-failure")
    ctx = _make_ctx(tmp_path, case)
    build_info = {
        "command": [sys.executable, "-m", "builder"],
        "returncode": 2,
        "stdout": "",
        "stderr": "build stderr",
    }
    monkeypatch.setattr(
        orchestrator,
        "_resolve_bundle",
        lambda case, ctx: (None, 0.25, "build exploded", build_info),
    )

    result = E2EOrchestrator().run(case, ctx)

    assert result.status == E2EStatus.FAIL.value
    assert result.failure_type == FailureType.BUILD_FAIL.value
    assert result.determinism == {"build_error": "build exploded"}
    data = _read_result_json(ctx, case)
    assert data["failure_type"] == FailureType.BUILD_FAIL.value
    assert data["commands"][0]["command"] == build_info["command"]
    assert data["timing"]["bundle_build_s"] == 0.25
    assert "repro_commands" in data


def test_run_preserves_recovered_build_attempt_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case("recovered-build-failure")
    ctx = _make_ctx(tmp_path, case)
    model_artifacts = Path(ctx.artifacts_dir) / case.name
    model_artifacts.mkdir(parents=True)
    recovery_timing = model_artifacts / "build_timing.attempt-1.json"
    recovery_timing.write_text('{"partial": true}\n', encoding="utf-8")
    command = [sys.executable, "-m", "builder"]
    build_info = {
        "command": command,
        "returncode": 2,
        "stdout": "",
        "stderr": "final build failure",
        "recovery_attempts": [
            {
                "attempt": 1,
                "returncode": -signal.SIGSEGV,
                "stdout": "",
                "stderr": "native builder crashed",
                "timing_path": str(recovery_timing),
            }
        ],
    }
    monkeypatch.setattr(
        orchestrator,
        "_resolve_bundle",
        lambda case, ctx: (None, 0.25, "build exploded", build_info),
    )

    result = E2EOrchestrator().run(case, ctx)

    assert result.status == E2EStatus.FAIL.value
    data = _read_result_json(ctx, case)
    assert [entry["returncode"] for entry in data["commands"]] == [
        -signal.SIGSEGV,
        2,
    ]
    assert [entry["label"] for entry in data["commands"]] == [
        "build_recovery_attempt_1",
        "build",
    ]
    assert data["artifacts"]["build_recovery_attempt_1_timing_json"] == (
        "build_timing.attempt-1.json"
    )
    log_text = (model_artifacts / "e2e_run.log").read_text(encoding="utf-8")
    assert "[build_recovery_attempt_1] rc=-11" in log_text
    assert "native builder crashed" in log_text


def test_run_preserves_recovered_success_for_ledger_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case("recovered-build-success")
    ctx = _make_ctx(tmp_path, case)
    bundle_path = Path(ctx.engine_dir) / case.bundle
    bundle_path.write_bytes(b"bundle")
    model_artifacts = Path(ctx.artifacts_dir) / case.name
    model_artifacts.mkdir(parents=True)
    recovery_timing = model_artifacts / "build_timing.attempt-1.json"
    recovery_timing.write_text('{"partial": true}\n', encoding="utf-8")
    command = [sys.executable, "-m", "builder"]
    build_info = {
        "command": command,
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "recovery_attempts": [
            {
                "attempt": 1,
                "returncode": -signal.SIGSEGV,
                "stdout": "",
                "stderr": "native builder crashed",
                "timing_path": str(recovery_timing),
            }
        ],
    }
    monkeypatch.setattr(
        orchestrator,
        "_resolve_bundle",
        lambda case, ctx: (str(bundle_path), 0.25, "", build_info),
    )
    _patch_plugins(
        monkeypatch,
        runner=_FakeRunner(),
        reference=_FakeReference(),
        comparator=_FakeComparator(),
    )

    result = E2EOrchestrator().run(case, ctx)

    assert result.status == E2EStatus.PASS.value
    data = _read_result_json(ctx, case)
    assert [
        (entry["label"], entry["returncode"])
        for entry in data["commands"]
    ] == [
        ("build_recovery_attempt_1", -signal.SIGSEGV),
        ("build", 0),
    ]
    assert data["artifacts"]["build_recovery_attempt_1_timing_json"] == (
        "build_timing.attempt-1.json"
    )
    log_text = (model_artifacts / "e2e_run.log").read_text(encoding="utf-8")
    assert "[build_recovery_attempt_1] rc=-11" in log_text


def test_run_classifies_trt_stage_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case("trt-error")
    ctx = _make_ctx(tmp_path, case)
    _patch_bundle_success(monkeypatch, tmp_path)
    _patch_plugins(
        monkeypatch,
        runner=_FakeRunner(error="trt boom"),
        reference=_FakeReference(),
        comparator=_FakeComparator(),
    )

    result = E2EOrchestrator().run(case, ctx)

    assert result.status == E2EStatus.FAIL.value
    assert result.failure_type == FailureType.TRT_RUN_FAIL.value
    assert result.stages["generate"].status == StageStatus.ERROR.value
    assert "TRT run failed: trt boom" in result.stages["generate"].message
    data = _read_result_json(ctx, case)
    assert data["stages"]["generate"]["status"] == StageStatus.ERROR.value


def test_run_stops_after_trt_process_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case("trt-nonzero-returncode")
    ctx = _make_ctx(tmp_path, case)
    _patch_bundle_success(monkeypatch, tmp_path)
    reference = _FakeReference()
    _patch_plugins(
        monkeypatch,
        runner=_FakeRunner(
            outputs=[
                StageOutput(
                    stage_name="generate",
                    data={"returncode": 1, "stderr": "runtime boom"},
                    metadata={"command": ["trtmc", "run"]},
                )
            ]
        ),
        reference=reference,
        comparator=_FakeComparator(),
    )

    result = E2EOrchestrator().run(case, ctx)

    assert result.status == E2EStatus.FAIL.value
    assert result.failure_type == FailureType.TRT_RUN_FAIL.value
    assert result.stages["generate"].status == StageStatus.ERROR.value
    assert "TRT run failed" in result.stages["generate"].message
    assert "returncode=1" in result.stages["generate"].message
    assert "runtime boom" in result.stages["generate"].message
    assert reference.calls == 0


def test_run_classifies_reference_stage_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case("reference-error")
    ctx = _make_ctx(tmp_path, case)
    _patch_bundle_success(monkeypatch, tmp_path)
    _patch_plugins(
        monkeypatch,
        runner=_FakeRunner(),
        reference=_FakeReference(error="reference boom"),
        comparator=_FakeComparator(),
    )

    result = E2EOrchestrator().run(case, ctx)

    assert result.status == E2EStatus.FAIL.value
    assert result.failure_type == FailureType.REFERENCE_RUN_FAIL.value
    assert result.stages["generate"].status == StageStatus.ERROR.value
    assert "Reference run failed: reference boom" in result.stages["generate"].message
    data = _read_result_json(ctx, case)
    assert data["failure_type"] == FailureType.REFERENCE_RUN_FAIL.value


def test_run_classifies_comparison_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case("comparison-failure")
    ctx = _make_ctx(tmp_path, case)
    _patch_bundle_success(monkeypatch, tmp_path)
    failed_compare = CompareResult(
        stage_name="generate",
        status=StageStatus.FAILED.value,
        metrics={"value_match": MetricResult(value=0.0, threshold=1.0, passed=False)},
        message="mismatch",
    )
    _patch_plugins(
        monkeypatch,
        runner=_FakeRunner(),
        reference=_FakeReference(),
        comparator=_FakeComparator(result=failed_compare),
    )

    result = E2EOrchestrator().run(case, ctx)

    assert result.status == E2EStatus.FAIL.value
    assert result.failure_type == FailureType.COMPARE_FAIL.value
    assert result.stages["generate"].status == StageStatus.FAILED.value
    data = _read_result_json(ctx, case)
    assert data["stages"]["generate"]["metrics"]["value_match"]["passed"] is False


def test_run_classifies_determinism_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case("determinism-failure", determinism={"reruns": 1})
    ctx = _make_ctx(tmp_path, case)
    _patch_bundle_success(monkeypatch, tmp_path)
    runner = _FakeRunner(
        outputs=[
            StageOutput(stage_name="generate", text="baseline", data={"value": 1}),
            StageOutput(stage_name="generate", text="rerun drift", data={"value": 1}),
        ]
    )
    _patch_plugins(
        monkeypatch,
        runner=runner,
        reference=_FakeReference(),
        comparator=_FakeComparator(),
    )

    result = E2EOrchestrator().run(case, ctx)

    assert result.status == E2EStatus.FAIL.value
    assert result.failure_type == FailureType.DETERMINISM_FAIL.value
    assert result.stages["generate"].status == StageStatus.PASSED.value
    assert result.determinism["status"] == "non_deterministic"
    assert result.determinism["per_stage"]["generate"][0]["text_match"] is False
    data = _read_result_json(ctx, case)
    assert data["failure_type"] == FailureType.DETERMINISM_FAIL.value
    assert data["determinism"]["status"] == "non_deterministic"
