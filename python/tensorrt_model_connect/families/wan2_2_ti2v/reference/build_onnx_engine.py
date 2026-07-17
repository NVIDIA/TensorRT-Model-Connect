#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse a fixed-shape ONNX component and serialize a TensorRT plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorrt_model_connect.families.wan2_2_ti2v.vae_builder import (
    OFFICIAL_VAE_DECODER_PROFILE,
    OFFICIAL_VAE_WORKSPACE_GIB,
    Wan22VaeDecoderProfile,
    build_onnx_engine_file,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace-gib", type=int, default=OFFICIAL_VAE_WORKSPACE_GIB)
    parser.add_argument("--official-vae-profile", action="store_true")
    parser.add_argument("--latent-frames", type=int)
    parser.add_argument("--latent-height", type=int)
    parser.add_argument("--latent-width", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    custom_geometry = (args.latent_frames, args.latent_height, args.latent_width)
    if args.official_vae_profile and any(value is not None for value in custom_geometry):
        parser.error("--official-vae-profile cannot be combined with custom latent geometry")
    if any(value is not None for value in custom_geometry) and not all(
        value is not None for value in custom_geometry
    ):
        parser.error("custom VAE geometry requires all three latent dimensions")
    if args.official_vae_profile:
        profile = OFFICIAL_VAE_DECODER_PROFILE
    elif all(value is not None for value in custom_geometry):
        profile = Wan22VaeDecoderProfile(
            latent_frames=args.latent_frames,
            latent_height=args.latent_height,
            latent_width=args.latent_width,
        )
    else:
        profile = None

    build_onnx_engine_file(
        args.onnx,
        args.output,
        profile=profile,
        workspace_gib=args.workspace_gib,
        verbose=args.verbose,
    )
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
