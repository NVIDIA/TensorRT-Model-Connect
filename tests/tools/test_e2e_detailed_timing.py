# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for E2E detailed timing normalization."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRTMC_BUILD_ROOT = REPO_ROOT / "python"
if str(TRTMC_BUILD_ROOT) not in sys.path:
    sys.path.insert(0, str(TRTMC_BUILD_ROOT))

from tests.e2e_harness.contracts import StageOutput  # noqa: E402
from tests.e2e_harness.orchestrator import (  # noqa: E402
    _build_detailed_timing,
    _collect_trt_stage_timing,
)
from tensorrt_model_connect.engine_builder import (  # noqa: E402
    _compile_time_excluding_component_weight_load,
    _untracked_compile_time,
)


def test_detailed_timing_uses_actual_phase_measurements_without_overlap():
    details = _build_detailed_timing(
        {
            "bundle_build_s": 30.0,
            "trt_generate_s": 3.0,
            "trt_engine_generate_s": 1.2,
            "trt_load_deserialize_generate_s": 0.8,
            "trt_compile_s": 99.0,
            "ref_generate_s": 2.0,
            "contract_generate_s": 0.3,
            "compare_generate_s": 0.2,
            "preflight_s": 0.1,
        },
        {
            "timing": {
                "total_s": 25.0,
                "phases": {
                    "weights_loading_s": 1.5,
                    "trt_compile_s": 20.0,
                    "trt_compile_main_engine_s": 20.0,
                    "bundle_write_s": 0.4,
                },
            },
        },
    )

    assert details["weights_loading_s"] == 1.5
    assert details["trt_compile_s"] == 20.0
    assert details["trt_compile_main_engine_s"] == 20.0
    assert details["bundle_write_s"] == 0.4
    assert details["inference_s"] == 1.2
    assert details["trt_load_deserialization_s"] == 0.8
    assert details["trt_validation_s"] == 3.0
    assert details["reference_s"] == 2.0
    assert details["comparison_s"] == 0.5
    assert details["preflight_s"] == 0.1
    assert "build_total_s" not in details
    assert "bundle_total_s" not in details
    assert "build_overhead_s" not in details


def test_detailed_timing_does_not_treat_trt_wall_time_as_inference():
    details = _build_detailed_timing(
        {
            "trt_generate_s": 3.0,
            "trt_load_deserialize_generate_s": 0.8,
            "ref_generate_s": 2.0,
        },
        {},
    )

    assert "inference_s" not in details
    assert details["trt_load_deserialization_s"] == 0.8
    assert details["trt_validation_s"] == 3.0
    assert details["reference_s"] == 2.0


def test_orchestrator_does_not_double_count_saved_stderr_log(tmp_path):
    log_text = "\n".join([
        '[trtmc.load_timing] label="engine_plan" load_deserialize_ms=10.500000 plan_bytes=4',
        '[trtmc.engine_timing] label="engine_plan" execute_ms=3.250000 launches=1',
    ])
    log_path = tmp_path / "generate_stderr.log"
    log_path.write_text(log_text, encoding="utf-8")
    timing = _collect_trt_stage_timing(
        StageOutput(
            stage_name="generate",
            data={
                "stderr": log_text,
                "stderr_log": str(log_path),
            },
        ),
        "generate",
    )

    assert timing["trt_load_deserialize_generate_s"] == 0.0105
    assert timing["trt_component_load_deserialize_generate_engine_plan_s"] == 0.0105
    assert timing["trt_engine_generate_s"] == 0.00325
    assert timing["trt_component_engine_generate_engine_plan_s"] == 0.00325


def test_diffusion_compile_time_excludes_component_weight_loading():
    timing = {
        "phases": {
            "weights_loading_s": 13.0,
            "weights_loading_decoder_block_s": 8.0,
            "weights_loading_denoiser_s": 5.0,
        },
    }

    compile_s = _compile_time_excluding_component_weight_load(
        components_elapsed=100.0,
        weights_before_components=1.0,
        build_timing=timing,
    )

    assert compile_s == 88.0


def test_diffusion_compile_time_adds_only_untracked_compile_residual():
    timing = {
        "phases": {
            "trt_compile_s": 80.0,
            "trt_compile_decoder_block_s": 30.0,
            "trt_compile_denoiser_s": 50.0,
        },
    }

    residual = _untracked_compile_time(
        measured_compile_elapsed=88.0,
        compile_before_components=0.0,
        build_timing=timing,
    )

    assert residual == 8.0


def test_component_weight_timings_are_preserved_in_detailed_timing():
    details = _build_detailed_timing(
        {},
        {
            "timing": {
                "phases": {
                    "weights_loading_s": 13.0,
                    "weights_loading_decoder_block_s": 8.0,
                    "weights_loading_denoiser_s": 5.0,
                    "trt_compile_s": 88.0,
                    "trt_compile_decoder_block_s": 30.0,
                    "trt_compile_denoiser_s": 50.0,
                },
            },
        },
    )

    assert details["weights_loading_s"] == 13.0
    assert details["weights_loading_decoder_block_s"] == 8.0
    assert details["weights_loading_denoiser_s"] == 5.0
    assert details["trt_compile_s"] == 88.0
    assert details["trt_compile_decoder_block_s"] == 30.0
    assert details["trt_compile_denoiser_s"] == 50.0


def test_component_runtime_timings_are_preserved_in_detailed_timing():
    details = _build_detailed_timing(
        {
            "trt_engine_end_to_end_s": 0.75,
            "trt_component_engine_end_to_end_denoiser_plan_s": 0.6,
            "trt_component_engine_end_to_end_vae_decoder_plan_s": 0.15,
            "trt_load_deserialize_end_to_end_s": 3.0,
            "trt_component_load_deserialize_end_to_end_denoiser_plan_s": 2.5,
            "trt_component_load_deserialize_end_to_end_vae_decoder_plan_s": 0.5,
        },
        {},
    )

    assert details["inference_s"] == 0.75
    assert details["trt_load_deserialization_s"] == 3.0
    assert details["trt_component_engine_end_to_end_denoiser_plan_s"] == 0.6
    assert details["trt_component_engine_end_to_end_vae_decoder_plan_s"] == 0.15
    assert details["trt_component_load_deserialize_end_to_end_denoiser_plan_s"] == 2.5
    assert details["trt_component_load_deserialize_end_to_end_vae_decoder_plan_s"] == 0.5
