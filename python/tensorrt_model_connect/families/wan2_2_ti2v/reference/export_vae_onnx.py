#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the fixed-shape Wan2.2 VAE decoder for TensorRT parser probing."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorrt_model_connect.families.wan2_2_ti2v.vae_builder import (
    Wan22VaeDecoderProfile,
    export_vae_decoder_onnx,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latent-frames", type=int, default=2)
    parser.add_argument("--latent-height", type=int, default=2)
    parser.add_argument("--latent-width", type=int, default=2)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    profile = Wan22VaeDecoderProfile(
        latent_frames=args.latent_frames,
        latent_height=args.latent_height,
        latent_width=args.latent_width,
    )
    export_vae_decoder_onnx(args.checkpoint, args.output, profile=profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
