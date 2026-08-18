# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream already-qualified MiniMax-H3 plans into a runnable TRTMC bundle."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from tensorrt_model_connect.bundle_writer import (
    BundleInfo,
    BundleSection,
    _bundle_section_from_file,
    gpu_name,
    tensorrt_abi,
    tensorrt_version,
    write_bundle,
)
from tensorrt_model_connect.families.minimax_h3.config import SOL_ENGINE_1344X768_124F
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
}
FIRST_BLOCK_CACHE_PLAN_SECTIONS = {
    "text_encoder_plan": "text_encoder.plan",
    "adaln_precompute_plan": "adaln_precompute.plan",
    "denoiser_head_plan": "denoiser_head.plan",
    "denoiser_tail_plan": "denoiser_tail.plan",
    "denoiser_finish_plan": "denoiser_finish.plan",
    "vae_tile_decoder_plan": "vae_tile_decoder.plan",
}
EAGER_BUNDLE_SECTIONS = ("tokenizer.json", "config.json")
LAZY_BUNDLE_SECTIONS = tuple(PLAN_SECTIONS)


def _target_metadata() -> tuple[str, str, str]:
    """Bind a bundle to the TensorRT ABI and GPU that built its plans."""

    version = tensorrt_version()
    abi = tensorrt_abi(version)
    gpu = gpu_name()
    if version == "unknown" or not abi or not gpu:
        raise RuntimeError(
            "MiniMax-H3 bundle packaging requires a detected TensorRT version and GPU"
        )
    return version, abi, gpu


def _bundle_loading_policy(plan_sections=PLAN_SECTIONS) -> dict[str, object]:
    """Keep only metadata resident; H3 loads one large plan at a time."""

    return {
        "mode": "staged",
        "eager_sections": list(EAGER_BUNDLE_SECTIONS),
        "lazy_sections": list(plan_sections),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--first-block-cache",
        action="store_true",
        help="Package split head/tail/finish plans instead of denoiser.plan.",
    )
    args = parser.parse_args()
    plans = Path(args.plans_dir)
    model = Path(args.model_path)
    output = Path(args.output)
    source_revision = validate_source_revision(args.source_revision)
    profile = replace(SOL_ENGINE_1344X768_124F, first_block_cache=args.first_block_cache)
    plan_sections = FIRST_BLOCK_CACHE_PLAN_SECTIONS if profile.first_block_cache else PLAN_SECTIONS
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
    config = {
        "model_type": "minimax_h3",
        "runtime_strategy": "diffusion_minimax_h3",
        "precision": "bf16",
        "engine_backend": "trt",
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
    }
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
