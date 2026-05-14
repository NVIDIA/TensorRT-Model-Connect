"""Tests for text-generation runner process-result handling."""

from __future__ import annotations

from tests.e2e_harness.runners.text_generation import _detect_trt_runtime_error


def test_detects_trt_runtime_error_from_zero_rc_stderr() -> None:
    stderr = "\n".join(
        [
            "[trtmc] Pipeline loaded (strategy=seq2seq_encoder_decoder)",
            "[trt] ERROR: IExecutionContext::enqueueV3: Error Code 1: Cuda Runtime",
            "Internal Error: CHECK(false) failed.",
        ]
    )

    error = _detect_trt_runtime_error(stderr)

    assert "enqueueV3" in error


def test_ignores_normal_trtmc_stderr() -> None:
    stderr = "\n".join(
        [
            "[trtmc] Backend loaded: trt",
            "[trtmc.load_timing] label=\"engine_plan\" load_deserialize_ms=10.5 plan_bytes=4",
        ]
    )

    assert _detect_trt_runtime_error(stderr) == ""
