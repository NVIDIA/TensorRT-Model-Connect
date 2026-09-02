# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import pytest

from tensorrt_model_connect.families.minimax_h3.ref2va_bundle_contract import (
    REF2VA_PLAN_SECTIONS,
    REF2VA_SHARED_SECTIONS,
    ref2va_bundle_metadata,
    ref2va_storage_estimate,
)
from tensorrt_model_connect.families.minimax_h3.ref2va_checkpoint import (
    CHECKPOINT_REVISION,
    COMPONENT_NAME,
    MODEL_ID,
    TOTAL_TENSOR_BYTES,
    TransformerRefIdentity,
)


def _identity() -> TransformerRefIdentity:
    return TransformerRefIdentity(
        model_id=MODEL_ID,
        revision=CHECKPOINT_REVISION,
        component=COMPONENT_NAME,
        tensor_bytes=TOTAL_TENSOR_BYTES,
        tensor_count=638,
        inventory_sha256="0" * 64,
        files={},
    )


def test_ref2va_sections_share_qwen_and_are_all_lazy_plan_units() -> None:
    assert tuple(component for component, _filename, _section in REF2VA_PLAN_SECTIONS) == (
        "ref2va_denoiser",
        "ref2va_adaln_precompute",
        "ref2va_video_vae_encoder",
        "ref2va_audio_vae_encoder",
    )
    assert REF2VA_SHARED_SECTIONS["text_encoder"] == "text_encoder_plan"
    assert REF2VA_SHARED_SECTIONS["vision_encoder"] == "vision_encoder_plan"
    assert all("qwen" not in filename for _component, filename, _section in REF2VA_PLAN_SECTIONS)


def test_incremental_storage_budget_exposes_packaging_copy_peak() -> None:
    estimate = ref2va_storage_estimate()
    assert estimate.transformer_ref_download == 66_280_504_216
    assert estimate.all_ref2va_plan_budget == 68_027_502_954
    assert estimate.checkpoint_plus_plans_peak == 134_308_007_170
    assert estimate.checkpoint_plans_and_bundle_copy_peak == 202_335_510_124
    assert estimate.denoiser_workspace_limit == 96 << 30
    assert estimate.adaln_workspace_limit == 64 << 30
    assert estimate.encoder_workspace_limit == 32 << 30
    assert estimate.shared_text_encoder_workspace_limit == 96 << 30
    assert estimate.shared_vision_encoder_workspace_limit == 32 << 30
    assert estimate.max_profile_hidden_tensor_lower_bound == 6_777_093_120


def test_bundle_metadata_requires_strict_transformer_ref_identity() -> None:
    metadata = ref2va_bundle_metadata(_identity())
    assert metadata["ref2va_supported"] is True
    assert metadata["ref2va_transformer_fallback_allowed"] is False
    assert metadata["ref2va_runtime_dependencies"] == ["TensorRT-RTX"]
    assert metadata["ref2va_limits"]["audio_can_be_sole_input"] is True
    assert "audio_cannot_be_sole_input" not in metadata["ref2va_limits"]
    assert metadata["ref2va_scheduler"] == {
        "sigma_grid_points": 50,
        "transformer_forwards": 49,
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "guidance_scale": 1.0,
        "guidance_distilled": True,
    }
    assert metadata["ref2va_shared_sections"]["text_encoder"] == "text_encoder_plan"
    assert metadata["ref2va_shared_qwen_profiles"]["vision_encoder_plan"][
        "patch_rows_per_call"
    ] == [2_304, 4_032, 65_536]
    assert metadata["ref2va_shared_qwen_profiles"]["text_encoder_plan"]["sequence_rows"] == [
        1,
        1_144,
        262_144,
    ]
    assert (
        metadata["ref2va_shared_qwen_profiles"]["vision_encoder_plan"]["spatial_chunking_allowed"]
        is False
    )
    assert metadata["ref2va_transformer_ref"]["runtime_framework"] is None
    abis = metadata["ref2va_plan_abis"]
    assert abis["ref2va_denoiser_plan"]["inputs"][0] == {
        "name": "video_hidden_states",
        "dtype": "float32",
        "min_shape": [25_408, 96],
        "opt_shape": [44_592, 96],
        "max_shape": [364_608, 96],
    }
    assert abis["ref2va_adaln_precompute_plan"]["inputs"][0]["max_shape"] == [4, 256]
    assert abis["ref2va_video_vae_encoder_plan"]["outputs"][0]["max_shape"] == [
        1,
        48,
        5,
        16,
        16,
    ]
    assert abis["ref2va_audio_vae_encoder_plan"]["outputs"][0]["max_shape"] == [
        2,
        32,
        600,
    ]
    with pytest.raises(ValueError, match="provenance is incompatible"):
        ref2va_bundle_metadata(replace(_identity(), revision="main"))
    with pytest.raises(TypeError, match="validated transformer_ref"):
        ref2va_bundle_metadata(object())  # type: ignore[arg-type]
