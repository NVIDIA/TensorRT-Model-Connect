# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import pytest

from families.minimax_h3.ref2va_bundle_contract import (
    REF2VA_FIRST_BLOCK_CACHE_SECTIONS,
    REF2VA_PLAN_SECTIONS,
    REF2VA_SHARED_SECTIONS,
    ref2va_bundle_metadata,
)
from families.minimax_h3.ref2va_checkpoint import (
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


def test_bundle_metadata_requires_strict_transformer_ref_identity() -> None:
    metadata = ref2va_bundle_metadata(_identity())
    assert metadata["ref2va_supported"] is True
    assert metadata["ref2va_schema_version"] == 4
    assert metadata["ref2va_limits"]["audio_can_be_sole_input"] is True
    assert metadata["ref2va_limits"]["max_total_video_soundtrack_seconds"] == 15.0
    assert "requires_image_or_video" not in metadata["ref2va_limits"]
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
    ] == [1_620, 4_032, 65_536]
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
    assert metadata["ref2va_denoiser_profile_count"] == 2
    assert metadata["ref2va_denoiser_profile_layout"] == "five_second_common_then_public_dynamic"
    assert metadata["ref2va_denoiser_profiles"] == [
        {
            "name": "five_second_common",
            "video_rows": [14_985, 28_224, 77_256],
            "audio_rows": [414, 754, 1_214],
            "text_rows": [1, 2_571, 8_192],
            "packed_rows": [15_400, 31_549, 86_662],
        },
        {
            "name": "public_dynamic",
            "video_rows": [14_985, 44_592, 364_608],
            "audio_rows": [414, 414, 3_558],
            "text_rows": [1, 7_433, 262_144],
            "packed_rows": [15_400, 52_439, 630_310],
        },
    ]
    abis = metadata["ref2va_plan_abis"]
    assert "optimization_profiles" not in abis["ref2va_denoiser_plan"]
    assert abis["ref2va_denoiser_plan"]["inputs"][0] == {
        "name": "video_hidden_states",
        "dtype": "float32",
        "min_shape": [14_985, 96],
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


def test_cache_metadata_uses_split_engines_and_independent_threshold() -> None:
    metadata = ref2va_bundle_metadata(_identity(), first_block_cache=True)
    assert metadata["ref2va_schema_version"] == 5
    assert metadata["ref2va_first_block_cache"] == {"enabled": True, "threshold": 0.08}
    assert metadata["ref2va_plan_sections"] == {
        name: section for name, _filename, section in REF2VA_FIRST_BLOCK_CACHE_SECTIONS
    }
    abis = metadata["ref2va_plan_abis"]
    assert "ref2va_denoiser_plan" not in abis
    assert len(abis) == 6
    head = abis["ref2va_dit_head_plan"]
    assert {item["name"] for item in head["outputs"]} == {
        "head_hidden",
        "head_residual",
        "cache_metric",
    }
    assert {"cache_video_indices", "cache_audio_indices"} <= {
        item["name"] for item in head["inputs"]
    }
    assert (
        ref2va_bundle_metadata(_identity(), first_block_cache=True, first_block_cache_threshold=0)[
            "ref2va_first_block_cache"
        ]["threshold"]
        == 0.0
    )


@pytest.mark.parametrize("first_block_cache", (False, True))
def test_ref2va_metadata_accepts_only_distinct_full_quantized_source(
    first_block_cache: bool,
) -> None:
    from families.minimax_h3.quantized_checkpoint import (
        QUANTIZED_CHECKPOINT_IDENTITY,
        QUANTIZED_REF2VA_CHECKPOINT_IDENTITY,
    )

    identity = QUANTIZED_REF2VA_CHECKPOINT_IDENTITY
    metadata = ref2va_bundle_metadata(identity, first_block_cache=first_block_cache)
    assert metadata["ref2va_supported"] is True
    assert metadata["ref2va_transformer_ref"] == identity.bundle_metadata()
    assert metadata["ref2va_transformer_ref"]["model_id"] == "Comfy-Org/MiniMax-H3"
    assert metadata["ref2va_transformer_ref"]["quantization"] == "int8_tensorwise_convrot"
    assert metadata["ref2va_first_block_cache"]["enabled"] is first_block_cache
    assert metadata["ref2va_capacity"]["video_rows"][0] == 14_985
    assert metadata["ref2va_capacity"]["packed_rows"][0] == 15_400
    for incompatible in (
        QUANTIZED_CHECKPOINT_IDENTITY,
        replace(identity, revision="main"),
        replace(
            identity, filename="diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
        ),
        replace(identity, size_bytes=identity.size_bytes - 1),
        replace(identity, quantized_weight_count=200),
    ):
        with pytest.raises(ValueError, match="distinct full Comfy INT8 checkpoint"):
            ref2va_bundle_metadata(incompatible, first_block_cache=first_block_cache)


@pytest.mark.parametrize("threshold", [-0.1, float("nan"), float("inf"), True, "0.08"])
def test_cache_metadata_rejects_invalid_threshold(threshold) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ref2va_bundle_metadata(_identity(), first_block_cache_threshold=threshold)
