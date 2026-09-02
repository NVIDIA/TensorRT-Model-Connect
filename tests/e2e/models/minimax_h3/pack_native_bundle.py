# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream already-qualified MiniMax-H3 plans into a runnable TRTMC bundle."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from tensorrt_model_connect import engine_builder
from tensorrt_model_connect.bundle_writer import (
    BundleInfo,
    BundleSection,
    _bundle_section_from_file,
    write_bundle,
)
from tensorrt_model_connect.families.minimax_h3.config import (
    AUDIO_LATENT_FRAMES_MAX,
    AUDIO_LATENT_FRAMES_MIN,
    AUDIO_LATENT_FRAMES_OPT,
    CANVAS_MAX_ASPECT_RATIO,
    CANVAS_MAX_PIXELS,
    CANVAS_MIN_ASPECT_RATIO,
    CANVAS_MULTIPLE,
    CANVAS_SHORT_EDGE,
    NATIVE_EXPLICIT_CANVAS_SIZES,
    FASTH3_GUIDANCE_SCALE,
    FASTH3_SCHEDULER_GRID_POINTS,
    FASTH3_TRANSFORMER_FORWARDS,
    FASTH3_VSA_MAX_VIDEO_TILES,
    FASTH3_VSA_TILE_SIZE,
    SOL_ENGINE_1344X768_124_TO_345F,
    VIDEO_NUM_FRAMES_MAX,
    VIDEO_NUM_FRAMES_MIN,
    VIDEO_NUM_FRAMES_OPT,
)
from tensorrt_model_connect.families.minimax_h3.checkpoint import (
    FASTH3_VSA_ADAPTER_BASE_REVISION,
    FASTH3_VSA_ADAPTER_BYTES,
    FASTH3_VSA_ADAPTER_FINETUNED_REVISION,
    FASTH3_VSA_ADAPTER_MODEL_ID,
    FASTH3_VSA_ADAPTER_SHA256,
    FASTH3_VSA_ADAPTER_SOURCE_REVISION,
)
from tensorrt_model_connect.families.minimax_h3.provenance import (
    CHECKPOINT_REVISION,
    validate_build_receipt,
    validate_source_revision,
)
from tensorrt_model_connect.families.minimax_h3.ref2va_bundle_contract import (
    REF2VA_PLAN_SECTIONS as REF2VA_COMPONENTS,
    ref2va_bundle_metadata,
)
from tensorrt_model_connect.families.minimax_h3.ref2va_checkpoint import (
    validate_transformer_ref_checkpoint,
)
from tensorrt_model_connect.families.minimax_h3.ref2va_qwen_contract import (
    ref2va_shared_qwen_profile_metadata,
)

PLAN_SECTIONS = {
    "text_encoder_plan": "text_encoder.plan",
    "vision_encoder_plan": "vision_encoder.plan",
    "adaln_precompute_plan": "adaln_precompute.plan",
    "denoiser_plan": "denoiser.plan",
    "fl2va_keyframe_vae_encoder_plan": "fl2va_keyframe_vae_encoder.plan",
    "vae_tile_decoder_plan": "vae_tile_decoder.plan",
    "audio_vae_decoder_plan": "audio_vae_decoder.plan",
}
FIRST_BLOCK_CACHE_PLAN_SECTIONS = {
    "text_encoder_plan": "text_encoder.plan",
    "vision_encoder_plan": "vision_encoder.plan",
    "adaln_precompute_plan": "adaln_precompute.plan",
    "denoiser_head_plan": "denoiser_head.plan",
    "denoiser_tail_plan": "denoiser_tail.plan",
    "denoiser_finish_plan": "denoiser_finish.plan",
    "fl2va_keyframe_vae_encoder_plan": "fl2va_keyframe_vae_encoder.plan",
    "vae_tile_decoder_plan": "vae_tile_decoder.plan",
    "audio_vae_decoder_plan": "audio_vae_decoder.plan",
}
FASTH3_DENOISER_PLAN_SECTIONS = {
    "denoiser_entry_plan": "denoiser_entry.plan",
    **{
        f"denoiser_transition_{index:02d}_plan": f"denoiser_transition_{index:02d}.plan"
        for index in range(SOL_ENGINE_1344X768_124_TO_345F.num_layers - 1)
    },
    "denoiser_finish_plan": "denoiser_finish.plan",
}
FASTH3_PLAN_SECTIONS = {
    "text_encoder_plan": "text_encoder.plan",
    "vision_encoder_plan": "vision_encoder.plan",
    "adaln_precompute_plan": "adaln_precompute.plan",
    **FASTH3_DENOISER_PLAN_SECTIONS,
    "fl2va_keyframe_vae_encoder_plan": "fl2va_keyframe_vae_encoder.plan",
    "vae_tile_decoder_plan": "vae_tile_decoder.plan",
    "audio_vae_decoder_plan": "audio_vae_decoder.plan",
}
REF2VA_PLAN_SECTIONS = {section: filename for _component, filename, section in REF2VA_COMPONENTS}
EAGER_BUNDLE_SECTIONS = ("tokenizer.json", "config.json")
LAZY_BUNDLE_SECTIONS = tuple(PLAN_SECTIONS)


def _target_metadata() -> tuple[str, str, str]:
    """Bind a bundle to the TensorRT ABI and GPU that built its plans."""

    trt_version = engine_builder._get_trt_version()
    trt_abi = engine_builder._trt_abi_from_version(trt_version)
    gpu_name = engine_builder._get_gpu_name()
    if trt_version == "unknown" or not trt_abi or not gpu_name:
        raise RuntimeError(
            "MiniMax-H3 bundle packaging requires a detected TensorRT version and GPU"
        )
    return trt_version, trt_abi, gpu_name


def _bundle_loading_policy(plan_sections=PLAN_SECTIONS) -> dict[str, object]:
    """Keep only metadata resident; H3 loads one large plan at a time."""

    return {
        "mode": "staged",
        "eager_sections": list(EAGER_BUNDLE_SECTIONS),
        "lazy_sections": list(plan_sections),
    }


def _audio_vae_metadata(model: Path, profile) -> dict[str, object]:
    path = model / "audio_vae" / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing MiniMax-H3 AudioVAE config: {path}")
    config = json.loads(path.read_text())
    rates = config.get("decoder_rates")
    latent_mean = config.get("latents_mean")
    latent_std = config.get("latents_std")
    if (
        not isinstance(rates, list)
        or not rates
        or not isinstance(latent_mean, list)
        or not isinstance(latent_std, list)
        or len(latent_mean) != profile.audio_in_channels
        or len(latent_std) != profile.audio_in_channels
    ):
        raise ValueError("MiniMax-H3 AudioVAE config has invalid decoder metadata")
    try:
        hop_length = math.prod(int(value) for value in rates)
        sampling_rate = int(config["sampling_rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("MiniMax-H3 AudioVAE config has invalid decoder metadata") from error
    if hop_length <= 0 or sampling_rate <= 0 or profile.audio_rows % 2:
        raise ValueError("MiniMax-H3 AudioVAE config has invalid decoder metadata")
    return {
        "audio_latent_frames": AUDIO_LATENT_FRAMES_OPT,
        "audio_latent_frames_min": AUDIO_LATENT_FRAMES_MIN,
        "audio_latent_frames_opt": AUDIO_LATENT_FRAMES_OPT,
        "audio_latent_frames_max": AUDIO_LATENT_FRAMES_MAX,
        "audio_sample_rate": sampling_rate,
        "audio_hop_length": hop_length,
        "audio_channels": 2,
        "audio_vae_precision": "fp32",
        "audio_vae_input_normalized": False,
        "audio_latents_mean": [float(value) for value in latent_mean],
        "audio_latents_std": [float(value) for value in latent_std],
    }


def _validated_fast_h3_metadata(value: object, profile) -> dict[str, object]:
    """Fail closed on the path-free adapter identity stored by the build helper."""

    if not isinstance(value, dict):
        raise ValueError("FastH3 packaging requires validated adapter metadata")
    expected = {
        "schema_version": 1,
        "adapter_model_id": FASTH3_VSA_ADAPTER_MODEL_ID,
        "adapter_source_revision": FASTH3_VSA_ADAPTER_SOURCE_REVISION,
        "adapter_sha256": FASTH3_VSA_ADAPTER_SHA256,
        "adapter_bytes": FASTH3_VSA_ADAPTER_BYTES,
        "adapter_tensor_count": 856,
        "adapter_low_rank_tensor_count": 724,
        "adapter_diff_tensor_count": 82,
        "adapter_set_weight_tensor_count": 50,
        "adapter_gate_tensor_count": 50,
        "adapter_base_revision": FASTH3_VSA_ADAPTER_BASE_REVISION,
        "adapter_finetuned_revision": FASTH3_VSA_ADAPTER_FINETUNED_REVISION,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"FastH3 build receipt has invalid {key}")
    counts = value.get("adapter_partition_tensor_counts")
    expected_partitions = {
        "adaln_precompute",
        "denoiser_entry",
        *(f"denoiser_transition_{index:02d}" for index in range(profile.num_layers - 1)),
        "denoiser_finish",
    }
    if (
        not isinstance(counts, dict)
        or set(counts) != expected_partitions
        or any(
            not isinstance(count, int) or isinstance(count, bool) or count <= 0
            for count in counts.values()
        )
        or sum(counts.values()) != expected["adapter_tensor_count"]
    ):
        raise ValueError("FastH3 build receipt has invalid adapter partition accounting")
    if set(value) != {*expected, "adapter_partition_tensor_counts"}:
        raise ValueError("FastH3 build receipt has unknown adapter metadata")
    return dict(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--transformer-ref",
        help="Strict transformer_ref checkpoint used to build all four Ref2VA plans.",
    )
    denoiser_group = parser.add_mutually_exclusive_group()
    denoiser_group.add_argument(
        "--first-block-cache",
        action="store_true",
        help="Package split head/tail/finish plans instead of denoiser.plan.",
    )
    denoiser_group.add_argument(
        "--fast-h3",
        action="store_true",
        help="Package the adapter-merged 51-plan native CUDA VSA denoiser.",
    )
    args = parser.parse_args()
    plans = Path(args.plans_dir)
    model = Path(args.model_path)
    output = Path(args.output)
    source_revision = validate_source_revision(args.source_revision)
    profile = replace(
        SOL_ENGINE_1344X768_124_TO_345F,
        first_block_cache=args.first_block_cache,
    )
    if args.fast_h3:
        plan_sections = FASTH3_PLAN_SECTIONS
    elif profile.first_block_cache:
        plan_sections = FIRST_BLOCK_CACHE_PLAN_SECTIONS
    else:
        plan_sections = PLAN_SECTIONS
    transformer_ref_identity = None
    if args.transformer_ref:
        transformer_ref_identity = validate_transformer_ref_checkpoint(args.transformer_ref)
        plan_sections = {**plan_sections, **REF2VA_PLAN_SECTIONS}
    trt_version, trt_abi, gpu_name = _target_metadata()
    receipt_path = plans / "build_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Missing native build receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    tokenizer = (model / "tokenizer" / "tokenizer.json").resolve(strict=True)
    audio_vae_metadata = _audio_vae_metadata(model, profile)
    expected_source_sha, recorded, tokenizer_record, snapshot_record = validate_build_receipt(
        receipt,
        plans_dir=plans,
        snapshot=model,
        tokenizer=tokenizer,
        build_helper=Path(__file__).with_name("build_native_components.py"),
        source_revision=source_revision,
        profile=profile,
        hash_files=False,
        segmented_vsa=args.fast_h3,
    )
    fast_h3_metadata = (
        _validated_fast_h3_metadata(receipt.get("fast_h3"), profile) if args.fast_h3 else None
    )
    expected_mode = (
        "segmented_vsa"
        if args.fast_h3
        else "first_block"
        if profile.first_block_cache
        else "monolithic"
    )
    if receipt.get("denoiser_mode") != expected_mode:
        raise ValueError("MiniMax-H3 build receipt denoiser mode does not match packaging mode")
    if not args.fast_h3 and receipt.get("fast_h3") is not None:
        raise ValueError("Dense MiniMax-H3 packaging rejects FastH3 adapter metadata")
    if transformer_ref_identity is not None:
        if receipt.get("transformer_ref") != transformer_ref_identity.bundle_metadata():
            raise ValueError(
                "Ref2VA build receipt transformer_ref provenance does not match the "
                "strictly validated checkpoint"
            )
        missing_ref_plans = sorted(set(REF2VA_PLAN_SECTIONS.values()) - set(recorded))
        if missing_ref_plans:
            raise ValueError(
                f"Ref2VA packaging receipt is missing strictly built plans: {missing_ref_plans}"
            )
    elif receipt.get("transformer_ref") is not None:
        raise ValueError("Non-Ref2VA packaging rejects transformer_ref receipt metadata")

    sections: list[BundleSection] = []
    for section_name, filename in plan_sections.items():
        path = plans / filename
        sections.append(
            _bundle_section_from_file(
                section_name,
                path,
                expected_sha256=recorded[filename]["sha256"],
            )
        )
    sections.append(
        _bundle_section_from_file(
            "tokenizer.json", tokenizer, expected_sha256=tokenizer_record["sha256"]
        )
    )
    if transformer_ref_identity is None:
        text_sequence_profile = [1, 1144, 2641]
        vision_patch_profile = [2040, 4032, 4176]
        vision_row_profile = [1, 1008, 2088]
    else:
        shared_qwen = ref2va_shared_qwen_profile_metadata()
        text_sequence_profile = shared_qwen["text_encoder_plan"]["sequence_rows"]
        vision_patch_profile = shared_qwen["vision_encoder_plan"]["patch_rows_per_call"]
        vision_row_profile = shared_qwen["text_encoder_plan"]["compact_vision_rows"]
    config = {
        "model_type": "minimax_h3",
        "runtime_strategy": "diffusion_minimax_h3",
        "precision": "bf16",
        "engine_backend": "trt_rtx",
        "trt_version": trt_version,
        "trt_abi": trt_abi,
        "bundle_loading": _bundle_loading_policy(plan_sections),
        "tokenizer_add_special_tokens": 0,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "source_revision": source_revision,
        "builder_source_sha256": expected_source_sha,
        "build_helper_sha256": receipt["build_helper_sha256"],
        "checkpoint_inventory_sha256": snapshot_record["inventory_sha256"],
        "workspace_limit_bytes": dict(receipt["workspace_limit_bytes"]),
        "plan_sha256": {
            filename: recorded[filename]["sha256"] for filename in plan_sections.values()
        },
        "first_block_cache": profile.first_block_cache,
        "denoiser_cache_mode": expected_mode,
        "first_block_cache_threshold": 0.025,
        "height": 768,
        "width": 1344,
        "canvas_multiple": CANVAS_MULTIPLE,
        "canvas_short_edge": CANVAS_SHORT_EDGE,
        "canvas_max_pixels": CANVAS_MAX_PIXELS,
        "explicit_canvas_sizes": [list(size) for size in NATIVE_EXPLICIT_CANVAS_SIZES],
        "min_aspect_ratio": CANVAS_MIN_ASPECT_RATIO,
        "max_aspect_ratio": CANVAS_MAX_ASPECT_RATIO,
        "public_workflows": [
            "t2va",
            "fl2va",
            *(["ref2va"] if transformer_ref_identity is not None else []),
        ],
        "conditioning": {
            "implementation": "shared_native_qwen3_vl",
            "text_encoder_section": "text_encoder_plan",
            "vision_encoder_section": "vision_encoder_plan",
            "keyframe_vae_encoder_section": "fl2va_keyframe_vae_encoder_plan",
            "text_sequence_profile": text_sequence_profile,
            "vision_patch_profile": vision_patch_profile,
            "vision_row_profile": vision_row_profile,
            "t2va_dummy_vision_rows": 1,
            "t2va_vision_count": 0,
            "t2va_vision_mask_nonzero": 0,
            "keyframe_vae_tile_batch_profile": [1, 28, 33],
            "reachable_canvas_count": 95,
            "max_rounded_canvas": [576, 1856],
            "max_condition_video_rows": 2088,
            "mode_coupled_profile_required": True,
        },
        "num_frames": VIDEO_NUM_FRAMES_OPT,
        "num_frames_min": VIDEO_NUM_FRAMES_MIN,
        "num_frames_opt": VIDEO_NUM_FRAMES_OPT,
        "num_frames_max": VIDEO_NUM_FRAMES_MAX,
        "fps": 24,
        "num_inference_steps": FASTH3_TRANSFORMER_FORWARDS if args.fast_h3 else 50,
        "guidance_scale": FASTH3_GUIDANCE_SCALE if args.fast_h3 else 1.0,
        "scheduler_grid_points": FASTH3_SCHEDULER_GRID_POINTS if args.fast_h3 else 50,
        "transformer_forwards": FASTH3_TRANSFORMER_FORWARDS if args.fast_h3 else 49,
        "attention_mode": "native_vsa" if args.fast_h3 else "dense",
        "text_rows": profile.text_rows,
        "text_rows_min": profile.min_text_rows,
        "text_rows_opt": profile.opt_text_rows,
        "text_rows_max": profile.text_rows,
        "audio_rows": profile.opt_audio_rows,
        "audio_rows_min": profile.min_audio_rows,
        "audio_rows_opt": profile.opt_audio_rows,
        "audio_rows_max": profile.audio_rows,
        **audio_vae_metadata,
        "video_rows": profile.opt_video_rows,
        "video_rows_min": profile.min_video_rows,
        "video_rows_opt": profile.opt_video_rows,
        "video_rows_max": profile.video_rows,
        "packed_sequence_length_min": profile.min_sequence_length,
        "packed_sequence_length_opt": profile.opt_sequence_length,
        "packed_sequence_length_max": profile.sequence_length,
        "padded_sequence_length": profile.padded_sequence_length,
        "max_timestep_count": 4,
        "context_parallel_size": 1,
        "vae_tile_batch": 28,
        "vae_tile_batch_min": 15,
        "vae_tile_batch_opt": 28,
        "vae_tile_batch_max": 33,
        "vae_tile_size": 256,
        "vae_tile_overlap": 64,
    }
    if fast_h3_metadata is not None:
        config["fast_h3"] = fast_h3_metadata
        config["vsa"] = {
            "implementation": "native_cuda_segmented",
            "tile_size": FASTH3_VSA_TILE_SIZE,
            "video_tile_shape": [4, 4, 4],
            "video_keep_numerator": 1,
            "video_keep_denominator": 10,
            "max_video_tiles": FASTH3_VSA_MAX_VIDEO_TILES,
            "max_total_tiles": (profile.vsa_prefix_tile_profile[2] + FASTH3_VSA_MAX_VIDEO_TILES),
            "packed_row_to_tile_slot_profile": list(profile.packed_row_profile),
            "prefix_valid_sizes_profile": list(profile.vsa_prefix_tile_profile),
            "video_valid_sizes_profile": list(profile.vsa_video_tile_abi_profile),
            "runtime_metadata_abi": {
                "producer": "modelconnect_cpp",
                "packed_row_to_tile_slot": {
                    "dtype": "int32",
                    "shape": ["S"],
                    "profile": list(profile.packed_row_profile),
                },
                "prefix_valid_sizes": {
                    "dtype": "int32",
                    "shape": ["P"],
                    "profile": list(profile.vsa_prefix_tile_profile),
                },
                "video_valid_sizes": {
                    "dtype": "int32",
                    "shape": ["Vtiles"],
                    "profile": list(profile.vsa_video_tile_abi_profile),
                },
            },
            "segment_count": len(FASTH3_DENOISER_PLAN_SECTIONS),
            "transition_count": profile.num_layers - 1,
            "attention_calls_per_forward": profile.num_layers,
            "fbc_composable": False,
            "tensor_abi": {
                "dtype": "bf16",
                "residual_input": "residual_hidden",
                "residual_output": "next_residual_hidden",
                "attention_input": "vsa_attention_output",
                "query_output": "vsa_query",
                "key_output": "vsa_key",
                "value_output": "vsa_value",
                "gate_output": "vsa_gate",
                "residual_shape": ["S", profile.hidden_size],
                "attention_shape": [profile.num_heads, "S", profile.head_dim],
                "packed_rows_profile": list(profile.packed_row_profile),
            },
            "segments": [
                {
                    "component": section.removesuffix("_plan"),
                    "filename": filename,
                    "section": section,
                }
                for section, filename in FASTH3_DENOISER_PLAN_SECTIONS.items()
            ],
        }
    if transformer_ref_identity is not None:
        config.update(ref2va_bundle_metadata(transformer_ref_identity))
    sections.append(BundleSection("config.json", json.dumps(config, indent=2).encode()))
    info = BundleInfo(
        model_id="MiniMaxAI/MiniMax-H3",
        model_type="minimax_h3",
        family="minimax_h3",
        trt_version=trt_version,
        trt_abi=trt_abi,
        gpu_name=gpu_name,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        runtime_strategy="diffusion_minimax_h3",
        precision="bf16",
        tokenizer_add_special_tokens=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_bundle(output, info, sections)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
