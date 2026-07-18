#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a fixed-shape Wan2.2 denoiser component probe."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorrt_model_connect.families.wan2_2_ti2v.dit_builder import (
    build_dit_engine,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latent-frames", type=int, default=1)
    parser.add_argument("--latent-height", type=int, default=4)
    parser.add_argument("--latent-width", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--workspace-gib", type=int, default=96)
    parser.add_argument("--builder-optimization-level", type=int)
    parser.add_argument("--round-residual-bf16", action="store_true")
    parser.add_argument("--emulate-bf16-gemm", action="store_true")
    parser.add_argument("--debug-layers", default="")
    parser.add_argument("--debug-sub-layers", default="")
    parser.add_argument("--debug-full-attention-layers", default="")
    parser.add_argument("--debug-full-norm-layers", default="")
    parser.add_argument("--debug-full-substage-layers", default="")
    parser.add_argument("--debug-cross-k-norm-layers", default="")
    parser.add_argument("--debug-embeddings", action="store_true")
    parser.add_argument("--debug-final-stages", action="store_true")
    parser.add_argument("--cross-attention-fp32", action="store_true")
    parser.add_argument("--source-attention-plugin", type=Path)
    parser.add_argument("--cuda-bf16-plugin", type=Path)
    parser.add_argument("--dit-cuda-plugin", type=Path)
    parser.add_argument("--disable-dit-bf16-linear", action="store_true")
    parser.add_argument("--enable-dit-time-silu", action="store_true")
    parser.add_argument("--enable-dit-time-linear2", action="store_true")
    parser.add_argument("--enable-dit-time-projection", action="store_true")
    parser.add_argument("--enable-dit-block-layer-norm", action="store_true")
    parser.add_argument("--enable-dit-adaptive-norm", action="store_true")
    parser.add_argument("--enable-dit-rms-norm", action="store_true")
    parser.add_argument("--enable-dit-self-gated-residual", action="store_true")
    parser.add_argument("--enable-dit-ffn-gated-residual", action="store_true")
    parser.add_argument("--enable-dit-cross-affine-layer-norm", action="store_true")
    parser.add_argument("--enable-dit-final-projection", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    debug_layers = tuple(int(value) for value in args.debug_layers.split(",") if value.strip())
    debug_sub_layers = tuple(
        int(value) for value in args.debug_sub_layers.split(",") if value.strip()
    )
    debug_full_attention_layers = tuple(
        int(value) for value in args.debug_full_attention_layers.split(",") if value.strip()
    )
    debug_full_norm_layers = tuple(
        int(value) for value in args.debug_full_norm_layers.split(",") if value.strip()
    )
    debug_full_substage_layers = tuple(
        int(value) for value in args.debug_full_substage_layers.split(",") if value.strip()
    )
    debug_cross_k_norm_layers = tuple(
        int(value) for value in args.debug_cross_k_norm_layers.split(",") if value.strip()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plan = build_dit_engine(
        str(args.checkpoint),
        latent_frames=args.latent_frames,
        latent_height=args.latent_height,
        latent_width=args.latent_width,
        num_layers=args.num_layers,
        round_residual_bf16=args.round_residual_bf16,
        emulate_bf16_gemm=args.emulate_bf16_gemm,
        debug_layers=debug_layers,
        debug_sub_layers=debug_sub_layers,
        debug_full_attention_layers=debug_full_attention_layers,
        debug_full_norm_layers=debug_full_norm_layers,
        debug_full_substage_layers=debug_full_substage_layers,
        debug_cross_k_norm_layers=debug_cross_k_norm_layers,
        debug_embeddings=args.debug_embeddings,
        debug_final_stages=args.debug_final_stages,
        cross_attention_fp32=args.cross_attention_fp32,
        source_attention_plugin=(
            str(args.source_attention_plugin.resolve())
            if args.source_attention_plugin is not None
            else None
        ),
        cuda_bf16_plugin=(
            str(args.cuda_bf16_plugin.resolve()) if args.cuda_bf16_plugin is not None else None
        ),
        dit_cuda_plugin=(
            str(args.dit_cuda_plugin.resolve()) if args.dit_cuda_plugin is not None else None
        ),
        dit_bf16_linear=not args.disable_dit_bf16_linear,
        dit_time_silu=args.enable_dit_time_silu,
        dit_time_linear2=args.enable_dit_time_linear2,
        dit_time_projection=args.enable_dit_time_projection,
        dit_block_layer_norm=args.enable_dit_block_layer_norm,
        dit_adaptive_norm=args.enable_dit_adaptive_norm,
        dit_rms_norm=args.enable_dit_rms_norm,
        dit_self_gated_residual=args.enable_dit_self_gated_residual,
        dit_ffn_gated_residual=args.enable_dit_ffn_gated_residual,
        dit_cross_affine_layer_norm=args.enable_dit_cross_affine_layer_norm,
        dit_final_projection=args.enable_dit_final_projection,
        workspace_size=args.workspace_gib << 30,
        builder_optimization_level=args.builder_optimization_level,
        verbose=args.verbose,
    )
    args.output.write_bytes(plan)
    print(f"wrote {args.output} ({len(plan)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
