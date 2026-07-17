#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build native initializer/recurrent Wan2.2 VAE decoder engines."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from tensorrt_model_connect.families.wan2_2_ti2v.vae_step_builder import (
    OFFICIAL_VAE_STEP_PROFILE,
    SMALL_VAE_STEP_PROFILE,
    build_vae_step_engine,
    load_vae_step_weights,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("small", "official"), default="official")
    parser.add_argument("--kind", choices=("initializer", "recurrent", "both"), default="both")
    parser.add_argument("--workspace-gib", type=int, default=64)
    parser.add_argument(
        "--max-aux-streams",
        type=int,
        help="Override TensorRT's maximum auxiliary stream count (0 forces one stream)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile = OFFICIAL_VAE_STEP_PROFILE if args.profile == "official" else SMALL_VAE_STEP_PROFILE
    weights = load_vae_step_weights(args.checkpoint)
    kinds = ("initializer", "recurrent") if args.kind == "both" else (args.kind,)
    for kind in kinds:
        plan = build_vae_step_engine(
            weights,
            profile=profile,
            first_frame_only=kind == "initializer",
            workspace_gib=args.workspace_gib,
            max_aux_streams=args.max_aux_streams,
            verbose=args.verbose,
        )
        output = args.output_dir / f"vae_{args.profile}_{kind}.plan"
        output.write_bytes(plan)
        print(f"wrote {output} ({len(plan)} bytes)")
        del plan
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
