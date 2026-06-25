"""Qwen-owned tests for text-generation runner helper behavior."""

from __future__ import annotations

from tests.e2e.models.qwen.e2e_plugins.runners.text_generation import (
    _detect_trt_runtime_error,
    _extract_trtmc_load_timing,
    _extract_trtmc_timing,
)


def test_detects_trt_runtime_error_from_zero_rc_stderr() -> None:
    stderr = "\n".join(
        [
            "[trtmc] Pipeline loaded (strategy=qwen_decoder_kv_cache)",
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


def test_extracts_engine_timing_from_cli_stderr() -> None:
    timing = _extract_trtmc_timing(
        "noise\n[trtmc.timing] prefill_ms=12.500000 decode_ms=7.250000 total_ms=19.750000\n"
    )

    assert timing["trt_engine_prefill_s"] == 0.0125
    assert timing["trt_engine_decode_s"] == 0.00725
    assert timing["trt_engine_s"] == 0.01975


def test_extracts_load_deserialize_timing_from_cli_stderr() -> None:
    timing = _extract_trtmc_load_timing(
        "\n".join(
            [
                "[trtmc.load_timing] label=\"engine_plan\" load_deserialize_ms=10.500000 plan_bytes=4",
                "[trtmc.load_timing] label=\"extra\" load_deserialize_ms=2.250000 plan_bytes=8",
            ]
        )
    )

    assert timing["trt_load_deserialize_s"] == 0.01275
