"""Tests for E2E orchestrator TRT runtime path validation guard.

Trace: ARCH-E2E-001, UD-E2E-RUNTIME-GUARD
Intent: Validate that the runtime path guard accepts new-runtime markers and rejects legacy runtime markers
Preconditions: StageOutput metadata contains stderr with runtime backend markers
Postconditions: New runtime markers pass validation; legacy runtime markers trigger rejection
"""

from __future__ import annotations

from pathlib import Path

from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput
from tests.e2e_harness.orchestrator import _validate_trt_runtime_path


def _make_case(strategy: str) -> E2ECase:
    return E2ECase(
        name="guard-case",
        hf_id="hf/test-model",
        family="unit",
        runtime_strategy=strategy,
        bundle="guard-case.trtfb",
        stages=[],
    )


def _make_ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        case=_make_case("decoder_kv_cache"),
        artifacts_dir=str(tmp_path),
        binary_path="/tmp/build/trtmc",
        hf_python="/usr/bin/python3",
        engine_dir=str(tmp_path),
    )


def test_runtime_guard_accepts_new_runtime_marker_in_metadata(tmp_path: Path) -> None:
    case = _make_case("decoder_kv_cache")
    ctx = _make_ctx(tmp_path)
    output = StageOutput(
        stage_name="full_generation",
        metadata={
            "command": [ctx.binary_path, "run", "/tmp/model.trtfb"],
            "stderr": "[trtmc] Runtime ready (backend=trt_new_runtime_default, strategy=decoder_kv_cache)",
        },
    )

    assert _validate_trt_runtime_path(case, ctx, output) is None


def test_runtime_guard_rejects_legacy_runtime_marker(tmp_path: Path) -> None:
    case = _make_case("decoder_kv_cache")
    ctx = _make_ctx(tmp_path)
    output = StageOutput(
        stage_name="full_generation",
        metadata={
            "command": [ctx.binary_path, "run", "/tmp/model.trtfb"],
            "stderr": "[trtmc] Runtime path: compatibility factory mode",
        },
    )

    message = _validate_trt_runtime_path(case, ctx, output)
    assert message is not None
    assert "legacy compatibility factory mode" in message


def test_runtime_guard_rejects_missing_new_runtime_confirmation(tmp_path: Path) -> None:
    case = _make_case("decoder_kv_cache")
    ctx = _make_ctx(tmp_path)
    output = StageOutput(
        stage_name="full_generation",
        metadata={
            "command": [ctx.binary_path, "run", "/tmp/model.trtfb"],
            "stderr": "tokens: 1 2 3 4",
        },
    )

    message = _validate_trt_runtime_path(case, ctx, output)
    assert message is not None
    assert "did not confirm the new runtime path" in message


def test_runtime_guard_covers_voxcpm2_text_to_audio_strategy(tmp_path: Path) -> None:
    case = _make_case("text_to_audio_voxcpm2")
    ctx = _make_ctx(tmp_path)
    output = StageOutput(
        stage_name="full_generation",
        metadata={
            "command": [ctx.binary_path, "generate-audio", "/tmp/voxcpm2.trtfb"],
            "stderr": "[trtmc] Runtime path: compatibility factory mode",
        },
    )

    message = _validate_trt_runtime_path(case, ctx, output)
    assert message is not None
    assert "legacy compatibility factory mode" in message


def test_runtime_guard_reads_stderr_log_from_stage_data(tmp_path: Path) -> None:
    case = _make_case("speech_to_text")
    ctx = _make_ctx(tmp_path)
    stderr_log = tmp_path / "speech_to_text_stderr.log"
    stderr_log.write_text(
        "[trtmc] Runtime ready (backend=trt_new_runtime_default, strategy=speech_to_text)\n",
        encoding="utf-8",
    )
    output = StageOutput(
        stage_name="full_inference",
        data={
            "stderr_log": str(stderr_log),
            "stderr": "truncated tail",
        },
        metadata={
            "command": [ctx.binary_path, "transcribe", "/tmp/model.trtfb"],
        },
    )

    assert _validate_trt_runtime_path(case, ctx, output) is None


def test_runtime_guard_ignores_legacy_marker_from_unrelated_subprocess(tmp_path: Path) -> None:
    case = _make_case("decoder_kv_cache")
    ctx = _make_ctx(tmp_path)
    output = StageOutput(
        stage_name="full_generation",
        metadata={
            "cpp": {
                "command": [ctx.binary_path, "run", "/tmp/model.trtfb"],
                "stderr": "[trtmc] Runtime ready (backend=trt_new_runtime_default, strategy=decoder_kv_cache)",
            },
            "debug_runner": {
                "command": [ctx.hf_python, "-m", "tensorrt_model_connect.debug_runner"],
                "stderr": "[trtmc] Runtime path: compatibility factory mode",
            },
        },
    )

    assert _validate_trt_runtime_path(case, ctx, output) is None


def test_runtime_guard_skips_unknown_strategies(tmp_path: Path) -> None:
    case = _make_case("future_unknown_strategy")
    ctx = _make_ctx(tmp_path)
    output = StageOutput(
        stage_name="full_generation",
        metadata={
            "command": [ctx.binary_path, "run", "/tmp/model.trtfb"],
            "stderr": "[trtmc] Runtime path: compatibility factory mode",
        },
    )

    assert _validate_trt_runtime_path(case, ctx, output) is None


def test_runtime_guard_ignores_nonzero_cli_parse_errors_without_runtime_markers(tmp_path: Path) -> None:
    case = _make_case("prompted_segmentation")
    ctx = _make_ctx(tmp_path)
    output = StageOutput(
        stage_name="full_inference",
        metadata={
            "command": [ctx.binary_path, "segment", "/tmp/model.trtfb"],
            "returncode": 1,
            "stderr": "Error: Unknown flag: --point",
        },
    )

    assert _validate_trt_runtime_path(case, ctx, output) is None
