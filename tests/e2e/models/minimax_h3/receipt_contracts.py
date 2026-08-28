# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed cross-backend receipt contracts for MiniMax-H3."""

from __future__ import annotations

import numpy as np

from tensorrt_model_connect.families.minimax_h3.provenance import (
    DIFFUSERS_REFERENCE_REVISION,
    ref2va_input_specification_record,
)


_AUDIO_ONLY_SHIM_NAME = "pinned-diffusers-ref2va-audio-only-input-gate"
_AUDIO_ONLY_SHIM_METHOD = (
    "diffusers.modular_pipelines.minimax_h3.before_encoder.MiniMaxH3Ref2VASetupStep._check_inputs"
)
_AUDIO_ONLY_SHIM_METHOD_SHA256 = "fd232215f0ceb47b463e2d29a52307faabe25d06db56f16f35d278d528b58df3"
_AUDIO_ONLY_SHIM_ERROR = (
    "An audio reference has to be paired with at least one image or video reference and "
    "cannot be used on its own."
)


def validate_ref2va_receipt_contract(trt_receipt: dict, ref_receipt: dict) -> None:
    """Require matching Ref2VA inputs and fail-closed audio-only runtime evidence."""

    trt_workload = trt_receipt.get("workload")
    ref_workload = ref_receipt.get("request")
    trt_ref2va = isinstance(trt_workload, dict) and trt_workload.get("workflow") == "ref2va"
    ref_ref2va = isinstance(ref_workload, dict) and ref_workload.get("workflow") == "ref2va"
    if not trt_ref2va and not ref_ref2va:
        return
    if not trt_ref2va or not ref_ref2va:
        raise ValueError("MiniMax-H3 TRT and HF receipts disagree on the Ref2VA workflow")
    trt_kinds = trt_workload.get("reference_kinds")
    ref_kinds = ref_workload.get("reference_kinds")
    if (
        not isinstance(trt_kinds, list)
        or not trt_kinds
        or trt_kinds != ref_kinds
        or any(kind not in {"image", "video", "audio"} for kind in trt_kinds)
    ):
        raise ValueError("MiniMax-H3 TRT and HF receipts identify different Ref2VA inputs")
    if set(trt_kinds) != {"audio"}:
        return
    if len(trt_kinds) > 3:
        raise ValueError("MiniMax-H3 audio-only receipts exceed the three-audio reference cap")

    specification = ref2va_input_specification_record()
    if (
        trt_receipt.get("official_input_specification") != specification
        or ref_receipt.get("official_input_specification") != specification
    ):
        raise ValueError(
            "MiniMax-H3 audio-only receipts do not bind the official input specification"
        )

    compat = ref_receipt.get("ref2va_audio_only_compatibility")
    request = ref_receipt.get("request")
    if not isinstance(compat, dict) or not isinstance(request, dict):
        raise ValueError("MiniMax-H3 HF audio-only receipt has no compatibility evidence")
    warmup = request.get("warmup")
    measure = request.get("measure")
    suppressed_calls = compat.get("suppressed_calls")
    if (
        compat.get("name") != _AUDIO_ONLY_SHIM_NAME
        or compat.get("diffusers_revision") != DIFFUSERS_REFERENCE_REVISION
        or compat.get("upstream_method") != _AUDIO_ONLY_SHIM_METHOD
        or compat.get("upstream_method_source_sha256") != _AUDIO_ONLY_SHIM_METHOD_SHA256
        or compat.get("suppressed_error") != _AUDIO_ONLY_SHIM_ERROR
        or compat.get("scope") != "audio-only Ref2VA requests"
        or compat.get("official_input_specification") != specification
        or isinstance(warmup, bool)
        or not isinstance(warmup, int)
        or warmup < 0
        or isinstance(measure, bool)
        or not isinstance(measure, int)
        or measure < 1
        or isinstance(suppressed_calls, bool)
        or not isinstance(suppressed_calls, int)
        or suppressed_calls != warmup + measure
    ):
        raise ValueError("MiniMax-H3 HF audio-only compatibility evidence is inconsistent")

    runtime = trt_receipt.get("runtime")
    engine_execute = trt_receipt.get("engine_execute")
    if not isinstance(runtime, dict) or not isinstance(engine_execute, dict):
        raise ValueError("MiniMax-H3 native audio-only receipt has no runtime engine evidence")
    condition_audio_rows = runtime.get("condition_audio_rows")
    audio_encoder_ms = engine_execute.get("audio_vae_encoder_plan_ms")
    if (
        runtime.get("references") != len(trt_kinds)
        or runtime.get("condition_video_rows") != 0
        or isinstance(condition_audio_rows, bool)
        or not isinstance(condition_audio_rows, int)
        or condition_audio_rows <= 0
        or "vision_conditioner_plan_ms" in engine_execute
        or "vae_encoder_tile_t1_plan_ms" in engine_execute
        or "vae_encoder_tile_t17_plan_ms" in engine_execute
        or isinstance(audio_encoder_ms, bool)
        or not isinstance(audio_encoder_ms, (int, float))
        or not np.isfinite(audio_encoder_ms)
        or audio_encoder_ms <= 0.0
    ):
        raise ValueError("MiniMax-H3 native audio-only engine routing is inconsistent")
