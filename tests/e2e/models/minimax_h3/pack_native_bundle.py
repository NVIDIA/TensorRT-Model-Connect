# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream already-qualified MiniMax-H3 plans into a runnable TRTMC bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from tensorrt_model_connect.bundle_writer import (
    BundleInfo,
    BundleSection,
    _bundle_section_from_file,
    write_bundle,
)


PLAN_SECTIONS = {
    "text_encoder_plan": "text_encoder.plan",
    "adaln_precompute_plan": "adaln_precompute.plan",
    "denoiser_plan_cp": "denoiser_cp.plan",
    "vae_tile_decoder_plan": "vae_tile_decoder.plan",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plans = Path(args.plans_dir)
    model = Path(args.model_path)
    output = Path(args.output)
    receipt_path = plans / "build_receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    recorded = receipt.get("components", {})

    sections: list[BundleSection] = []
    for section_name, filename in PLAN_SECTIONS.items():
        path = plans / filename
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Missing native plan: {path}")
        sections.append(
            _bundle_section_from_file(
                section_name,
                path,
                expected_sha256=recorded.get(filename, {}).get("sha256"),
            )
        )
    tokenizer = (model / "tokenizer" / "tokenizer.json").resolve(strict=True)
    sections.append(_bundle_section_from_file("tokenizer.json", tokenizer))
    config = {
        "model_type": "minimax_h3",
        "runtime_strategy": "diffusion_minimax_h3",
        "precision": "bf16",
        "engine_backend": "trt",
        "trt_version": "11.2.0.113",
        "trt_abi": "11.2",
        "tokenizer_add_special_tokens": 0,
        "checkpoint_revision": "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc",
        "height": 768,
        "width": 1344,
        "num_frames": 124,
        "fps": 24,
        "num_inference_steps": 50,
        "text_rows": 537,
        "audio_rows": 414,
        "video_rows": 37296,
        "padded_sequence_length": 38272,
        "max_timestep_count": 4,
        "context_parallel_size": 4,
        "vae_tile_batch": 7,
        "vae_tile_size": 256,
        "vae_tile_overlap": 64,
    }
    sections.append(BundleSection("config.json", json.dumps(config, indent=2).encode()))
    info = BundleInfo(
        model_id="MiniMaxAI/MiniMax-H3",
        model_type="minimax_h3",
        family="minimax_h3",
        trt_version="11.2.0.113",
        trt_abi="11.2",
        gpu_name="NVIDIA GB300",
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
