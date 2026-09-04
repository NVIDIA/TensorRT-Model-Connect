# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed bundle/provenance contract for native MiniMax-H3 Ref2VA.

This module does not package files itself.  It gives the staged bundle writer
one authoritative list of additional sections and path-free metadata, while
making the large checkpoint/build storage requirement visible before any
download or engine build starts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .config import (
    ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES,
    DENOISER_DEFAULT_WORKSPACE_BYTES,
    TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES,
    VISION_ENCODER_DEFAULT_WORKSPACE_BYTES,
)
from .fl2va_contract import PlanAbi, TensorAbi
from .ref2va_checkpoint import (
    CHECKPOINT_REVISION,
    COMPONENT_NAME,
    MODEL_ID,
    SHARDS,
    TransformerRefIdentity,
)
from .ref2va_contract import (
    MAX_AUDIOS,
    MAX_IMAGES,
    MAX_REFERENCES,
    MAX_REFERENCE_DURATION_SECONDS,
    MAX_TOTAL_AUDIO_DURATION_SECONDS,
    MAX_TOTAL_VIDEO_DURATION_SECONDS,
    MAX_VIDEOS,
    MIN_REFERENCE_DURATION_SECONDS,
    Ref2VADenoiserProfile,
    ref2va_denoiser_abi,
)
from .ref2va_qwen_contract import ref2va_shared_qwen_profile_metadata


# The Qwen language/vision plans are intentionally not listed here: Ref2VA
# binds the same ``text_encoder_plan`` and ``vision_encoder_plan`` sections
# already shared by T2VA/FL2VA.  No second Qwen weight copy is permitted.
REF2VA_PLAN_SECTIONS = (
    ("ref2va_denoiser", "ref2va_denoiser.plan", "ref2va_denoiser_plan"),
    (
        "ref2va_adaln_precompute",
        "ref2va_adaln_precompute.plan",
        "ref2va_adaln_precompute_plan",
    ),
    (
        "ref2va_video_vae_encoder",
        "ref2va_video_vae_encoder.plan",
        "ref2va_video_vae_encoder_plan",
    ),
    (
        "ref2va_audio_vae_encoder",
        "ref2va_audio_vae_encoder.plan",
        "ref2va_audio_vae_encoder_plan",
    ),
)

REF2VA_SHARED_SECTIONS = {
    "text_encoder": "text_encoder_plan",
    "vision_encoder": "vision_encoder_plan",
    "image_vae_encoder": "fl2va_keyframe_vae_encoder_plan",
    "video_vae_decoder": "vae_tile_decoder_plan",
    "audio_vae_decoder": "audio_vae_decoder_plan",
}

# Exact checkpoint tensor bytes by exhaustive key partition.  The estimates
# reserve one percent above parameters for TensorRT serialization metadata and
# are deliberately budgets, not claimed engine sizes.  The two encoder bases
# are measured real TRT-RTX 1.6.1.120 builds on this machine (348,786,444 and
# 724,746,796 bytes respectively), again padded by one percent here.
REF2VA_DENOISER_PARAMETER_BYTES = 40_138_350_592
REF2VA_ADALN_PARAMETER_BYTES = 26_142_079_488
REF2VA_AUDIO_ENCODER_MEASURED_PLAN_BYTES = 348_786_444
REF2VA_VIDEO_ENCODER_MEASURED_PLAN_BYTES = 724_746_796
REF2VA_ENCODER_DEFAULT_WORKSPACE_BYTES = 32 << 30

# The public transformer_ref partition is the guidance-distilled 50-point
# rectified-flow recipe.  ``num_inference_steps`` counts the terminal zero, so
# the transformer executes once for each of the first 49 grid points.  Keep
# this mode-local instead of inheriting the optional FastH3 schedule used by
# the transformer partition for T2VA/FL2VA.
REF2VA_SCHEDULER_GRID_POINTS = 50
REF2VA_TRANSFORMER_FORWARDS = 49
REF2VA_VIDEO_FLOW_SHIFT = 12.0
REF2VA_AUDIO_FLOW_SHIFT = 3.0
REF2VA_GUIDANCE_SCALE = 1.0


def _ref2va_adaln_abi() -> PlanAbi:
    outputs = tuple(
        TensorAbi(
            f"block_modulation_{index}",
            "bfloat16",
            (12, 6, 5_376),
            (12, 6, 5_376),
            (12, 6, 5_376),
        )
        for index in range(50)
    ) + (
        TensorAbi(
            "final_modulation",
            "bfloat16",
            (4, 2, 5_376),
            (4, 2, 5_376),
            (4, 2, 5_376),
        ),
    )
    return PlanAbi(
        filename="ref2va_adaln_precompute.plan",
        inputs=(
            TensorAbi(
                "timestep_features",
                "float32",
                (4, 256),
                (4, 256),
                (4, 256),
            ),
        ),
        outputs=outputs,
    )


def _ref2va_video_encoder_abi() -> PlanAbi:
    return PlanAbi(
        filename="ref2va_video_vae_encoder.plan",
        inputs=(
            TensorAbi(
                "pixel_tile_clip",
                "float32",
                (1, 3, 17, 256, 256),
                (1, 3, 17, 256, 256),
                (1, 3, 17, 256, 256),
            ),
        ),
        outputs=(
            TensorAbi(
                "posterior_parameter_tile_clip",
                "float32",
                (1, 48, 5, 16, 16),
                (1, 48, 5, 16, 16),
                (1, 48, 5, 16, 16),
            ),
        ),
    )


def _ref2va_audio_encoder_abi() -> PlanAbi:
    return PlanAbi(
        filename="ref2va_audio_vae_encoder.plan",
        inputs=(
            TensorAbi(
                "audio_samples",
                "float32",
                (2, 1, 64_000),
                (2, 1, 165_600),
                (2, 1, 480_000),
            ),
        ),
        outputs=(
            TensorAbi(
                "posterior_mean",
                "float32",
                (2, 32, 80),
                (2, 32, 207),
                (2, 32, 600),
            ),
        ),
    )


def ref2va_plan_abi_metadata(
    profile: Ref2VADenoiserProfile = Ref2VADenoiserProfile(),
) -> dict[str, object]:
    """Serialize every dedicated plan binding without importing TensorRT."""

    profile.validate()

    def tensor(binding: TensorAbi) -> dict[str, object]:
        return {
            "name": binding.name,
            "dtype": binding.dtype,
            "min_shape": list(binding.min_shape),
            "opt_shape": list(binding.opt_shape),
            "max_shape": list(binding.max_shape),
        }

    plans = (
        ("ref2va_denoiser_plan", ref2va_denoiser_abi(profile)),
        ("ref2va_adaln_precompute_plan", _ref2va_adaln_abi()),
        ("ref2va_video_vae_encoder_plan", _ref2va_video_encoder_abi()),
        ("ref2va_audio_vae_encoder_plan", _ref2va_audio_encoder_abi()),
    )
    return {
        section: {
            "filename": abi.filename,
            "inputs": [tensor(binding) for binding in abi.inputs],
            "outputs": [tensor(binding) for binding in abi.outputs],
        }
        for section, abi in plans
    }


@dataclass(frozen=True)
class Ref2VAStorageEstimate:
    """Conservative incremental storage/workspace budget in bytes."""

    transformer_ref_download: int
    denoiser_plan_budget: int
    adaln_plan_budget: int
    audio_encoder_plan_budget: int
    video_encoder_plan_budget: int
    all_ref2va_plan_budget: int
    checkpoint_plus_plans_peak: int
    checkpoint_plans_and_bundle_copy_peak: int
    denoiser_workspace_limit: int
    adaln_workspace_limit: int
    encoder_workspace_limit: int
    shared_text_encoder_workspace_limit: int
    shared_vision_encoder_workspace_limit: int
    max_profile_hidden_tensor_lower_bound: int


def ref2va_storage_estimate(
    profile: Ref2VADenoiserProfile = Ref2VADenoiserProfile(),
) -> Ref2VAStorageEstimate:
    """Return an auditable disk budget before downloading ``transformer_ref``.

    ``checkpoint_plus_plans_peak`` retains checkpoint shards while all four
    plans exist. ``checkpoint_plans_and_bundle_copy_peak`` additionally keeps
    a copied bundle during packaging.  Shared Qwen/decoder sections and their
    bundle copy are outside these *incremental Ref2VA* numbers.
    """

    profile.validate()

    def plan_budget(value: int) -> int:
        return math.ceil(value * 1.01)

    checkpoint = sum(size for _name, size, _sha256 in SHARDS)
    denoiser = plan_budget(REF2VA_DENOISER_PARAMETER_BYTES)
    adaln = plan_budget(REF2VA_ADALN_PARAMETER_BYTES)
    audio = plan_budget(REF2VA_AUDIO_ENCODER_MEASURED_PLAN_BYTES)
    video = plan_budget(REF2VA_VIDEO_ENCODER_MEASURED_PLAN_BYTES)
    plans = denoiser + adaln + audio + video
    hidden_lower_bound = profile.max_packed_rows * 5_376 * 2
    return Ref2VAStorageEstimate(
        transformer_ref_download=checkpoint,
        denoiser_plan_budget=denoiser,
        adaln_plan_budget=adaln,
        audio_encoder_plan_budget=audio,
        video_encoder_plan_budget=video,
        all_ref2va_plan_budget=plans,
        checkpoint_plus_plans_peak=checkpoint + plans,
        checkpoint_plans_and_bundle_copy_peak=checkpoint + 2 * plans,
        denoiser_workspace_limit=DENOISER_DEFAULT_WORKSPACE_BYTES,
        adaln_workspace_limit=ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES,
        encoder_workspace_limit=REF2VA_ENCODER_DEFAULT_WORKSPACE_BYTES,
        shared_text_encoder_workspace_limit=TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES,
        shared_vision_encoder_workspace_limit=VISION_ENCODER_DEFAULT_WORKSPACE_BYTES,
        max_profile_hidden_tensor_lower_bound=hidden_lower_bound,
    )


def ref2va_bundle_metadata(
    transformer_ref: TransformerRefIdentity,
    profile: Ref2VADenoiserProfile = Ref2VADenoiserProfile(),
) -> dict[str, object]:
    """Produce path-free metadata only after strict checkpoint validation."""

    profile.validate()
    if not isinstance(transformer_ref, TransformerRefIdentity):
        raise TypeError("MiniMax-H3 Ref2VA metadata requires validated transformer_ref identity")
    if (
        transformer_ref.model_id != MODEL_ID
        or transformer_ref.revision != CHECKPOINT_REVISION
        or transformer_ref.component != COMPONENT_NAME
        or transformer_ref.tensor_count != 638
    ):
        raise ValueError("MiniMax-H3 Ref2VA transformer_ref provenance is incompatible")
    estimate = ref2va_storage_estimate(profile)
    return {
        "ref2va_schema_version": 2,
        "ref2va_supported": True,
        "ref2va_context_ir_supported": False,
        "ref2va_regenerate_2k_supported": False,
        "ref2va_runtime_language": "c++/cuda",
        "ref2va_runtime_dependencies": ["TensorRT-RTX"],
        "ref2va_transformer_fallback_allowed": False,
        "ref2va_scheduler": {
            "sigma_grid_points": REF2VA_SCHEDULER_GRID_POINTS,
            "transformer_forwards": REF2VA_TRANSFORMER_FORWARDS,
            "video_shift": REF2VA_VIDEO_FLOW_SHIFT,
            "audio_shift": REF2VA_AUDIO_FLOW_SHIFT,
            "guidance_scale": REF2VA_GUIDANCE_SCALE,
            "guidance_distilled": True,
        },
        "ref2va_transformer_ref": transformer_ref.bundle_metadata(),
        "ref2va_plan_sections": {
            component: section for component, _filename, section in REF2VA_PLAN_SECTIONS
        },
        "ref2va_plan_abis": ref2va_plan_abi_metadata(profile),
        "ref2va_shared_sections": dict(REF2VA_SHARED_SECTIONS),
        "ref2va_shared_qwen_profiles": ref2va_shared_qwen_profile_metadata(),
        "ref2va_limits": {
            "max_images": MAX_IMAGES,
            "max_videos": MAX_VIDEOS,
            "max_explicit_audios": MAX_AUDIOS,
            "max_reference_files": MAX_REFERENCES,
            "min_seconds_each_video_or_audio": MIN_REFERENCE_DURATION_SECONDS,
            "max_seconds_each_video_or_audio": MAX_REFERENCE_DURATION_SECONDS,
            "max_total_video_seconds": MAX_TOTAL_VIDEO_DURATION_SECONDS,
            "max_total_explicit_audio_seconds": MAX_TOTAL_AUDIO_DURATION_SECONDS,
            "requires_image_or_video": True,
            "video_soundtrack_stays_attached": True,
        },
        "ref2va_capacity": {
            "video_rows": [
                profile.min_video_rows,
                profile.opt_video_rows,
                profile.max_video_rows,
            ],
            "audio_rows": [
                profile.min_audio_rows,
                profile.opt_audio_rows,
                profile.max_audio_rows,
            ],
            "text_rows": [
                profile.min_text_rows,
                profile.opt_text_rows,
                profile.max_text_rows,
            ],
            "packed_rows": [
                profile.min_packed_rows,
                profile.opt_packed_rows,
                profile.max_packed_rows,
            ],
        },
        "ref2va_build_storage_estimate": asdict(estimate),
    }
