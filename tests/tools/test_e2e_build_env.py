"""E2E bundle-build environment tests."""

from __future__ import annotations

from tests.e2e_harness.contracts import E2ECase
from tests.e2e_harness.orchestrator import _apply_build_env_overrides


def test_manifest_builder_optimization_level_overrides_build_env() -> None:
    case = E2ECase(
        name="fnet-base",
        hf_id="google/fnet-base",
        family="fnet",
        runtime_strategy="encoder_only",
        metadata={"builder_optimization_level": 0},
    )
    env = {
        "TRTMC_BUILDER_OPTIMIZATION_LEVEL": "1",
        "TRTMC_TRT_TIMING_CACHE_PATH": "/tmp/trt-timing-cache/tensorrt-opt1.cache",
    }

    _apply_build_env_overrides(case, env)

    assert env["TRTMC_BUILDER_OPTIMIZATION_LEVEL"] == "0"
    assert env["TRTMC_TRT_TIMING_CACHE_PATH"].endswith("tensorrt-opt0.cache")


def test_manifest_builder_optimization_level_appends_cache_suffix_when_missing() -> None:
    case = E2ECase(
        name="fnet-base",
        hf_id="google/fnet-base",
        family="fnet",
        runtime_strategy="encoder_only",
        metadata={"builder_optimization_level": 0},
    )
    env = {"TRTMC_TRT_TIMING_CACHE_PATH": "/tmp/trt-timing-cache/tensorrt.cache"}

    _apply_build_env_overrides(case, env)

    assert env["TRTMC_TRT_TIMING_CACHE_PATH"].endswith("tensorrt-opt0.cache")
