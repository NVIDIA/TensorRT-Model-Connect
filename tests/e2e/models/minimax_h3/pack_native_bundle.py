# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream already-qualified MiniMax-H3 plans into a runnable TRTMC bundle."""

from __future__ import annotations

import argparse
import json
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
    FL2VA_PLAN_FILENAMES,
    FL2VA_PROCESSOR_ASSET_SECTIONS,
    MINIMAX_H3_NATIVE_PLUGIN_ABI,
    MINIMAX_H3_NATIVE_PLUGIN_FILENAME,
    MINIMAX_H3_NATIVE_PLUGIN_IDENTITY,
    MINIMAX_H3_NATIVE_PLUGIN_SECTION,
    REF2VA_AUDIO_ENCODER_HOP_LENGTH,
    REF2VA_AUDIO_ENCODER_IMPLEMENTATION,
    REF2VA_AUDIO_ENCODER_INPUT_PROFILE,
    REF2VA_AUDIO_ENCODER_MODULE_FORMAT,
    REF2VA_AUDIO_ENCODER_OUTPUT_CHANNELS,
    REF2VA_AUDIO_ENCODER_PLUGIN_COUNT,
    REF2VA_AUDIO_ENCODER_WEIGHT_NORM,
    REF2VA_MAX_CONDITION_AUDIO_ROWS,
    REF2VA_MAX_CONDITION_VIDEO_ROWS,
    REF2VA_IMAGE_VISION_ATTENTION_IMPLEMENTATION,
    REF2VA_IMAGE_VISION_ATTENTION_PRECISION,
    REF2VA_IMAGE_VISION_ATTENTION_SCALE,
    REF2VA_IMAGE_VISION_LINEAR_COUNT,
    REF2VA_IMAGE_VISION_LINEAR_IMPLEMENTATION,
    REF2VA_IMAGE_VISION_LAYER_NORM_COUNT,
    REF2VA_IMAGE_VISION_LAYER_NORM_IMPLEMENTATION,
    REF2VA_IMAGE_VISION_PATCH_BIAS_SHAPE,
    REF2VA_IMAGE_VISION_PATCH_IMPLEMENTATION,
    REF2VA_IMAGE_VISION_PATCH_INPUT_SHAPE,
    REF2VA_IMAGE_VISION_PATCH_KERNEL,
    REF2VA_IMAGE_VISION_PATCH_OUTPUT_SHAPE,
    REF2VA_IMAGE_VISION_PATCH_PRECISION,
    REF2VA_IMAGE_VISION_PATCH_PROFILE,
    REF2VA_IMAGE_VISION_PATCH_STRIDE,
    REF2VA_IMAGE_VISION_PATCH_WEIGHT_SHAPE,
    REF2VA_LANGUAGE_ATTENTION_IMPLEMENTATION,
    REF2VA_LANGUAGE_ATTENTION_PRECISION,
    REF2VA_LANGUAGE_Q_PRE_SCALE_PRECISION,
    REF2VA_MAX_TEXT_ROWS,
    REF2VA_MIN_CONDITION_VIDEO_ROWS,
    REF2VA_OPT_CONDITION_VIDEO_ROWS,
    REF2VA_PLAN_FILENAMES,
    REF2VA_VIDEO_VISION_ATTENTION_IMPLEMENTATION,
    REF2VA_VIDEO_VISION_ATTENTION_PRECISION,
    REF2VA_VIDEO_VISION_PATCH_PROFILE,
    REF2VA_VIDEO_VISION_Q_PRE_SCALE_PRECISION,
    REF2VA_VISION_PLAN_LAYOUT,
    SOL_ENGINE_1344X768_124F,
)
from tensorrt_model_connect.families.minimax_h3.provenance import (
    CHECKPOINT_REVISION,
    validate_build_receipt,
    validate_source_revision,
)

PLAN_SECTIONS = {
    "text_encoder_plan": "text_encoder.plan",
    "adaln_precompute_plan": "adaln_precompute.plan",
    "denoiser_plan": "denoiser.plan",
    "vae_tile_decoder_plan": "vae_tile_decoder.plan",
    "audio_vae_decoder_plan": "audio_vae_decoder.plan",
}
FIRST_BLOCK_CACHE_PLAN_SECTIONS = {
    "text_encoder_plan": "text_encoder.plan",
    "adaln_precompute_plan": "adaln_precompute.plan",
    "denoiser_head_plan": "denoiser_head.plan",
    "denoiser_tail_plan": "denoiser_tail.plan",
    "denoiser_finish_plan": "denoiser_finish.plan",
    "vae_tile_decoder_plan": "vae_tile_decoder.plan",
    "audio_vae_decoder_plan": "audio_vae_decoder.plan",
}
FL2VA_PLAN_SECTIONS = {
    f"{filename.removesuffix('.plan')}_plan": filename for filename in FL2VA_PLAN_FILENAMES
}
REF2VA_PLAN_SECTIONS = {
    f"{filename.removesuffix('.plan')}_plan": filename for filename in REF2VA_PLAN_FILENAMES
}
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


def _bundle_loading_policy(
    plan_sections=PLAN_SECTIONS,
    *,
    processor_sections: tuple[str, ...] = (),
    native_plugin_section: str | None = None,
) -> dict[str, object]:
    """Keep only metadata resident; H3 loads one large plan at a time."""

    return {
        "mode": "staged",
        "eager_sections": [
            "tokenizer.json",
            *processor_sections,
            *([native_plugin_section] if native_plugin_section is not None else []),
            "config.json",
        ],
        "lazy_sections": list(plan_sections),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--workflow", choices=("t2va", "fl2va", "ref2va"), default="t2va")
    parser.add_argument(
        "--first-block-cache",
        action="store_true",
        help="Package split head/tail/finish plans instead of denoiser.plan.",
    )
    args = parser.parse_args()
    if args.workflow != "t2va" and args.first_block_cache:
        parser.error(f"MiniMax-H3 {args.workflow.upper()} does not support --first-block-cache")
    plans = Path(args.plans_dir)
    model = Path(args.model_path)
    output = Path(args.output)
    source_revision = validate_source_revision(args.source_revision)
    workflow = args.workflow
    profile = replace(SOL_ENGINE_1344X768_124F, first_block_cache=args.first_block_cache)
    if workflow == "ref2va":
        plan_sections = REF2VA_PLAN_SECTIONS
        processor_sections = FL2VA_PROCESSOR_ASSET_SECTIONS
        native_plugin_section = MINIMAX_H3_NATIVE_PLUGIN_SECTION
    elif workflow == "fl2va":
        plan_sections = FL2VA_PLAN_SECTIONS
        processor_sections = FL2VA_PROCESSOR_ASSET_SECTIONS
        native_plugin_section = None
    else:
        plan_sections = (
            FIRST_BLOCK_CACHE_PLAN_SECTIONS if profile.first_block_cache else PLAN_SECTIONS
        )
        processor_sections = ()
        native_plugin_section = None
    trt_version, trt_abi, gpu_name = _target_metadata()
    receipt_path = plans / "build_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Missing native build receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    tokenizer = (model / "tokenizer" / "tokenizer.json").resolve(strict=True)
    expected_source_sha, recorded, tokenizer_record, snapshot_record = validate_build_receipt(
        receipt,
        plans_dir=plans,
        snapshot=model,
        tokenizer=tokenizer,
        build_helper=Path(__file__).with_name("build_native_components.py"),
        source_revision=source_revision,
        profile=profile,
        hash_files=False,
        workflow=workflow,
    )

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
    for section_name in processor_sections:
        sections.append(
            _bundle_section_from_file(
                section_name,
                (model / section_name).resolve(strict=True),
                expected_sha256=receipt["assets"][section_name]["sha256"],
            )
        )
    if native_plugin_section is not None:
        sections.append(
            _bundle_section_from_file(
                native_plugin_section,
                (plans / MINIMAX_H3_NATIVE_PLUGIN_FILENAME).resolve(strict=True),
                expected_sha256=receipt["assets"][native_plugin_section]["sha256"],
            )
        )
    config = {
        "model_type": "minimax_h3",
        "runtime_strategy": "diffusion_minimax_h3",
        "workflow": workflow,
        "checkpoint_partition": "transformer_ref" if workflow == "ref2va" else "transformer",
        "precision": "bf16",
        "engine_backend": "trt",
        "trt_version": trt_version,
        "trt_abi": trt_abi,
        "bundle_loading": _bundle_loading_policy(
            plan_sections,
            processor_sections=processor_sections,
            native_plugin_section=native_plugin_section,
        ),
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
        "denoiser_cache_mode": "first_block" if profile.first_block_cache else "monolithic",
        "first_block_cache_threshold": 0.025,
        "height": 768,
        "width": 1344,
        "num_frames": 124,
        "fps": 24,
        "num_inference_steps": 50,
        "text_rows": 537,
        "audio_rows": 414,
        "video_rows": 37296,
        "padded_sequence_length": 38247,
        "max_timestep_count": 4,
        "context_parallel_size": 1,
        "vae_tile_batch": 28,
        "vae_tile_size": 256,
        "vae_tile_overlap": 64,
        "audio_sample_rate": 32000,
        "audio_latent_frames": 207,
        "audio_output_samples": 165600,
    }
    if workflow == "fl2va":
        config.update(
            {
                "min_text_rows": profile.min_text_rows,
                "max_text_rows": profile.max_text_rows,
                "fl2va_keyframe_counts": [0, 1, 2],
                "fl2va_keyframe_rows": 1008,
                "fl2va_vae_tile_size": 256,
                "fl2va_vae_tile_min_overlap": 64,
                "fl2va_vae_temporal_frames": [1],
                "processor_asset_sections": list(processor_sections),
                "asset_sha256": {
                    name: receipt["assets"][name]["sha256"]
                    for name in ("tokenizer.json", *processor_sections)
                },
            }
        )
    elif workflow == "ref2va":
        audio_encoder_metadata = recorded["audio_vae_encoder.plan"]["build_metadata"]
        config.update(
            {
                "min_text_rows": profile.ref2va_min_text_rows,
                "opt_text_rows": profile.ref2va_opt_text_rows,
                "max_text_rows": REF2VA_MAX_TEXT_ROWS,
                "ref2va_min_condition_video_rows": REF2VA_MIN_CONDITION_VIDEO_ROWS,
                "ref2va_opt_condition_video_rows": REF2VA_OPT_CONDITION_VIDEO_ROWS,
                "ref2va_min_condition_audio_rows": profile.ref2va_min_condition_audio_rows,
                "ref2va_opt_condition_audio_rows": profile.ref2va_opt_condition_audio_rows,
                "ref2va_max_condition_video_rows": REF2VA_MAX_CONDITION_VIDEO_ROWS,
                "ref2va_max_condition_audio_rows": REF2VA_MAX_CONDITION_AUDIO_ROWS,
                "ref2va_max_images": 9,
                "ref2va_max_videos": 3,
                "ref2va_max_audios": 3,
                "ref2va_max_references": 12,
                "ref2va_vision_plan_layout": REF2VA_VISION_PLAN_LAYOUT,
                "minimax_h3_native_plugin_section": MINIMAX_H3_NATIVE_PLUGIN_SECTION,
                "minimax_h3_native_plugin_artifact": MINIMAX_H3_NATIVE_PLUGIN_FILENAME,
                "minimax_h3_native_plugin_abi": MINIMAX_H3_NATIVE_PLUGIN_ABI,
                "minimax_h3_native_plugin_identity": MINIMAX_H3_NATIVE_PLUGIN_IDENTITY,
                "ref2va_audio_encoder_implementation": REF2VA_AUDIO_ENCODER_IMPLEMENTATION,
                "ref2va_audio_encoder_plugin_count": REF2VA_AUDIO_ENCODER_PLUGIN_COUNT,
                "ref2va_audio_encoder_module_format": REF2VA_AUDIO_ENCODER_MODULE_FORMAT,
                "ref2va_audio_encoder_weight_norm": REF2VA_AUDIO_ENCODER_WEIGHT_NORM,
                "ref2va_audio_encoder_input_profile": list(REF2VA_AUDIO_ENCODER_INPUT_PROFILE),
                "ref2va_audio_encoder_hop_length": REF2VA_AUDIO_ENCODER_HOP_LENGTH,
                "ref2va_audio_encoder_output_channels": REF2VA_AUDIO_ENCODER_OUTPUT_CHANNELS,
                "ref2va_audio_encoder_cuda_graphs": audio_encoder_metadata["cuda_graphs"],
                "ref2va_audio_encoder_cudnn_tf32": audio_encoder_metadata["cudnn_tf32"],
                "ref2va_audio_encoder_matmul_tf32": audio_encoder_metadata["matmul_tf32"],
                "ref2va_audio_encoder_graph_optimizer": audio_encoder_metadata["graph_optimizer"],
                "ref2va_audio_encoder_cudnn_enabled": audio_encoder_metadata["cudnn_enabled"],
                "ref2va_audio_encoder_cudnn_benchmark": audio_encoder_metadata["cudnn_benchmark"],
                "ref2va_audio_encoder_cudnn_deterministic": audio_encoder_metadata[
                    "cudnn_deterministic"
                ],
                "ref2va_audio_encoder_module_bytes": audio_encoder_metadata["module_bytes"],
                "ref2va_audio_encoder_module_sha256": audio_encoder_metadata["module_sha256"],
                "ref2va_language_attention_implementation": (
                    REF2VA_LANGUAGE_ATTENTION_IMPLEMENTATION
                ),
                "ref2va_language_attention_precision": REF2VA_LANGUAGE_ATTENTION_PRECISION,
                "ref2va_language_q_pre_scale_precision": (REF2VA_LANGUAGE_Q_PRE_SCALE_PRECISION),
                "ref2va_image_vision_attention_implementation": (
                    REF2VA_IMAGE_VISION_ATTENTION_IMPLEMENTATION
                ),
                "ref2va_image_vision_attention_precision": (
                    REF2VA_IMAGE_VISION_ATTENTION_PRECISION
                ),
                "ref2va_image_vision_attention_scale": (REF2VA_IMAGE_VISION_ATTENTION_SCALE),
                "ref2va_image_vision_linear_implementation": (
                    REF2VA_IMAGE_VISION_LINEAR_IMPLEMENTATION
                ),
                "ref2va_image_vision_linear_count": REF2VA_IMAGE_VISION_LINEAR_COUNT,
                "ref2va_image_vision_layer_norm_implementation": (
                    REF2VA_IMAGE_VISION_LAYER_NORM_IMPLEMENTATION
                ),
                "ref2va_image_vision_layer_norm_count": REF2VA_IMAGE_VISION_LAYER_NORM_COUNT,
                "ref2va_image_vision_patch_implementation": (
                    REF2VA_IMAGE_VISION_PATCH_IMPLEMENTATION
                ),
                "ref2va_image_vision_patch_precision": REF2VA_IMAGE_VISION_PATCH_PRECISION,
                "ref2va_image_vision_patch_input_shape": list(
                    REF2VA_IMAGE_VISION_PATCH_INPUT_SHAPE
                ),
                "ref2va_image_vision_patch_weight_shape": list(
                    REF2VA_IMAGE_VISION_PATCH_WEIGHT_SHAPE
                ),
                "ref2va_image_vision_patch_bias_shape": list(REF2VA_IMAGE_VISION_PATCH_BIAS_SHAPE),
                "ref2va_image_vision_patch_kernel": list(REF2VA_IMAGE_VISION_PATCH_KERNEL),
                "ref2va_image_vision_patch_stride": list(REF2VA_IMAGE_VISION_PATCH_STRIDE),
                "ref2va_image_vision_patch_output_shape": list(
                    REF2VA_IMAGE_VISION_PATCH_OUTPUT_SHAPE
                ),
                "ref2va_video_vision_attention_implementation": (
                    REF2VA_VIDEO_VISION_ATTENTION_IMPLEMENTATION
                ),
                "ref2va_video_vision_attention_precision": (
                    REF2VA_VIDEO_VISION_ATTENTION_PRECISION
                ),
                "ref2va_video_vision_q_pre_scale_precision": (
                    REF2VA_VIDEO_VISION_Q_PRE_SCALE_PRECISION
                ),
                "ref2va_image_vision_patch_profile": list(REF2VA_IMAGE_VISION_PATCH_PROFILE),
                "ref2va_video_vision_patch_profile": list(REF2VA_VIDEO_VISION_PATCH_PROFILE),
                "ref2va_reference_min_seconds": 2,
                "ref2va_reference_max_seconds": 15,
                "ref2va_vae_tile_size": 256,
                "ref2va_vae_tile_min_overlap": 64,
                "ref2va_vae_temporal_frames": [1, 17],
                "processor_asset_sections": list(processor_sections),
                "asset_sha256": {
                    name: receipt["assets"][name]["sha256"]
                    for name in (
                        "tokenizer.json",
                        *processor_sections,
                        MINIMAX_H3_NATIVE_PLUGIN_SECTION,
                    )
                },
            }
        )
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
