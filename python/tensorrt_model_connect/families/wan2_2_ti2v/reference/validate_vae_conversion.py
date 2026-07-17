#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the family-owned Wan2.2 VAE conversion against upstream."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan

from tensorrt_model_connect.families.wan2_2_ti2v.checkpoint_mapper import (
    VAE22_CONFIG,
    convert_vae_state_dict,
    load_native_vae_state_dict,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--latent-frames", type=int, default=2)
    parser.add_argument("--latent-height", type=int, default=2)
    parser.add_argument("--latent-width", type=int, default=2)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.official_source.resolve()))
    from wan.modules.vae2_2 import Wan2_2_VAE  # pylint: disable=import-outside-toplevel

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    started = time.perf_counter()
    native = Wan2_2_VAE(vae_pth=str(args.checkpoint / "Wan2.2_VAE.pth"), device=device)
    native_loaded = time.perf_counter()
    candidate = AutoencoderKLWan(**VAE22_CONFIG)
    candidate.load_state_dict(
        convert_vae_state_dict(load_native_vae_state_dict(args.checkpoint)),
        strict=True,
    )
    candidate.eval().requires_grad_(False).to(device)
    candidate_loaded = time.perf_counter()

    generator = torch.Generator(device=device).manual_seed(args.seed)
    latent = torch.randn(
        1,
        48,
        args.latent_frames,
        args.latent_height,
        args.latent_width,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    mean = torch.tensor(VAE22_CONFIG["latents_mean"], device=device, dtype=latent.dtype).view(
        1, 48, 1, 1, 1
    )
    std = torch.tensor(VAE22_CONFIG["latents_std"], device=device, dtype=latent.dtype).view(
        1, 48, 1, 1, 1
    )

    with torch.inference_mode():
        reference = native.decode([latent[0]])[0].unsqueeze(0)
        torch.cuda.synchronize(device)
        native_done = time.perf_counter()
        converted = candidate.decode(latent * std + mean, return_dict=False)[0]
        torch.cuda.synchronize(device)
        candidate_done = time.perf_counter()

    reference = reference.float().cpu()
    converted = converted.float().cpu()
    delta = converted - reference
    report = {
        "kind": "wan2_2_ti2v_vae_conversion_parity",
        "device": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "latent_shape": list(latent.shape),
        "output_shape": list(reference.shape),
        "timing_seconds": {
            "native_load": native_loaded - started,
            "canonical_load": candidate_loaded - native_loaded,
            "native_decode": native_done - candidate_loaded,
            "canonical_decode": candidate_done - native_done,
        },
        "metrics": {
            "max_abs_error": float(delta.abs().max()),
            "mean_abs_error": float(delta.abs().mean()),
            "rmse": float(delta.square().mean().sqrt()),
            "cosine_similarity": float(
                torch.nn.functional.cosine_similarity(
                    reference.flatten().double(), converted.flatten().double(), dim=0
                )
            ),
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
