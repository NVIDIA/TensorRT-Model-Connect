# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Causal 3D VAE decoder engine builder.

Builds a TensorRT engine for a causal 3D VAE decoder that processes one
latent frame at a time, using temporal caches for causal convolutions.

Reusable by: Wan2.1, Hunyuan Video (same causal 3D VAE architecture).

For Wan2.1:
  base_dim=96, dim_mult=[1,2,4,4], z_dim=16
  Decoder channels (reversed): [384, 384, 192, 96]
  up_blocks: 4 levels, each with 3 resnets (num_res_blocks+1)
  Temporal upsample at blocks 0, 1 (reversed from encoder's [False, True, True])
  Spatial upsample at blocks 0, 1, 2 (not at last block)

Weight naming (diffusers WanDecoder3d):
  - Norms use `.gamma` (shape [C,1,1,1]) — WanRMS_norm, not GroupNorm
  - Resnets: `decoder.up_blocks.{i}.resnets.{j}.{conv1|conv2|norm1|norm2}`
  - Channel change shortcut: `resnets.{j}.conv_shortcut`
  - Spatial upsample: `upsamplers.0.resample.1` (2D Conv after nearest-upsample)
  - Temporal upsample: `upsamplers.0.time_conv` (CausalConv3D for pixel-shuffle-in-time)
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict


def load_vae_weights(
    model_dir: str,
    *,
    z_dim: int = 16,
    base_dim: int = 96,
    dim_mult: tuple[int, ...] = (1, 2, 4, 4),
    num_res_blocks: int = 2,
    norm_type: str = "l2_channel_norm",
) -> "WeightDict":
    """Load VAE decoder weights from a diffusers-format vae directory.

    Only loads decoder weights (not encoder). Returns raw weight arrays
    (no transposition — conv weights are used as-is).

    Args:
        norm_type: "l2_channel_norm" loads .gamma keys (Wan-style),
                   "group_norm" loads .weight/.bias keys.
    """
    from pathlib import Path
    from .checkpoint_mapper import WeightDict, _open_safetensors, _load_tensor, _has_tensor

    model_path = Path(model_dir)
    readers = _open_safetensors(model_path)
    weights = WeightDict()

    def _w(name: str) -> np.ndarray:
        return _load_tensor(readers, name).astype(np.float32)

    def _maybe(name: str) -> np.ndarray | None:
        if _has_tensor(readers, name):
            return _w(name)
        return None

    def _load_norm(prefix: str, channels: int) -> None:
        """Load norm weights based on norm_type."""
        if norm_type == "l2_channel_norm":
            weights[f"{prefix}.gamma"] = _w(f"{prefix}.gamma")
        else:
            weights[f"{prefix}.weight"] = _w(f"{prefix}.weight")
            weights[f"{prefix}.bias"] = _w(f"{prefix}.bias")

    # post_quant_conv [z_dim, z_dim, 1, 1, 1]
    weights["post_quant_conv.weight"] = _w("post_quant_conv.weight")
    weights["post_quant_conv.bias"] = _w("post_quant_conv.bias")

    # conv_in [384, 16, 3, 3, 3]
    weights["decoder.conv_in.weight"] = _w("decoder.conv_in.weight")
    weights["decoder.conv_in.bias"] = _w("decoder.conv_in.bias")

    # mid_block: 2 resnets + 1 attention
    channels_list = [base_dim * m for m in dim_mult]
    mid_ch = channels_list[-1]
    for i in range(2):
        p = f"decoder.mid_block.resnets.{i}"
        _load_norm(f"{p}.norm1", mid_ch)
        _load_norm(f"{p}.norm2", mid_ch)
        weights[f"{p}.conv1.weight"] = _w(f"{p}.conv1.weight")
        weights[f"{p}.conv1.bias"] = _w(f"{p}.conv1.bias")
        weights[f"{p}.conv2.weight"] = _w(f"{p}.conv2.weight")
        weights[f"{p}.conv2.bias"] = _w(f"{p}.conv2.bias")

    # mid_block attention norm
    attn_prefix = "decoder.mid_block.attentions.0"
    if norm_type == "l2_channel_norm":
        weights[f"{attn_prefix}.norm.gamma"] = _w(f"{attn_prefix}.norm.gamma")
    else:
        weights[f"{attn_prefix}.norm.weight"] = _w(f"{attn_prefix}.norm.weight")
        weights[f"{attn_prefix}.norm.bias"] = _w(f"{attn_prefix}.norm.bias")
    weights[f"{attn_prefix}.to_qkv.weight"] = _w(f"{attn_prefix}.to_qkv.weight")
    weights[f"{attn_prefix}.to_qkv.bias"] = _w(f"{attn_prefix}.to_qkv.bias")
    weights[f"{attn_prefix}.proj.weight"] = _w(f"{attn_prefix}.proj.weight")
    weights[f"{attn_prefix}.proj.bias"] = _w(f"{attn_prefix}.proj.bias")

    # up_blocks
    num_levels = len(dim_mult)
    for level in range(num_levels):
        for blk in range(num_res_blocks + 1):
            p = f"decoder.up_blocks.{level}.resnets.{blk}"
            ch = channels_list[-(level + 1)]
            _load_norm(f"{p}.norm1", ch)
            _load_norm(f"{p}.norm2", ch)
            weights[f"{p}.conv1.weight"] = _w(f"{p}.conv1.weight")
            weights[f"{p}.conv1.bias"] = _w(f"{p}.conv1.bias")
            weights[f"{p}.conv2.weight"] = _w(f"{p}.conv2.weight")
            weights[f"{p}.conv2.bias"] = _w(f"{p}.conv2.bias")
            # Channel shortcut
            sc_prefix = f"{p}.conv_shortcut" if norm_type == "l2_channel_norm" else f"{p}.shortcut"
            sc_w = _maybe(f"{sc_prefix}.weight")
            if sc_w is not None:
                weights[f"{sc_prefix}.weight"] = sc_w
                weights[f"{sc_prefix}.bias"] = _w(f"{sc_prefix}.bias")

        # Upsampler (spatial 2D conv + optional temporal)
        sp_w = _maybe(f"decoder.up_blocks.{level}.upsamplers.0.resample.1.weight")
        if sp_w is not None:
            weights[f"decoder.up_blocks.{level}.upsamplers.0.resample.1.weight"] = sp_w
            weights[f"decoder.up_blocks.{level}.upsamplers.0.resample.1.bias"] = _w(
                f"decoder.up_blocks.{level}.upsamplers.0.resample.1.bias")
        tc_w = _maybe(f"decoder.up_blocks.{level}.upsamplers.0.time_conv.weight")
        if tc_w is not None:
            weights[f"decoder.up_blocks.{level}.upsamplers.0.time_conv.weight"] = tc_w
            weights[f"decoder.up_blocks.{level}.upsamplers.0.time_conv.bias"] = _w(
                f"decoder.up_blocks.{level}.upsamplers.0.time_conv.bias")

    # Output norm + conv
    if norm_type == "l2_channel_norm":
        weights["decoder.norm_out.gamma"] = _w("decoder.norm_out.gamma")
    else:
        weights["decoder.norm_out.weight"] = _w("decoder.norm_out.weight")
        weights["decoder.norm_out.bias"] = _w("decoder.norm_out.bias")
    weights["decoder.conv_out.weight"] = _w("decoder.conv_out.weight")
    weights["decoder.conv_out.bias"] = _w("decoder.conv_out.bias")

    return weights


def count_vae_caches(
    dim_mult: tuple[int, ...] = (1, 2, 4, 4),
    num_res_blocks: int = 2,
    temporal_upsample: tuple[bool, ...] = (False, True, True),
) -> int:
    """Count causal conv cache slots needed for the VAE decoder.

    Each CausalConv3D with temporal kernel > 1 needs one cache.
    """
    count = 0
    num_levels = len(dim_mult)
    temp_up = list(reversed(temporal_upsample))

    # conv_in: CausalConv3d(kt=3) -> 1 cache
    count += 1

    # mid_block: 2 resnets × 2 causal convs each = 4 caches
    count += 4

    # up_blocks: each level has (num_res_blocks+1) resnets × 2 caches
    for level in range(num_levels):
        count += (num_res_blocks + 1) * 2

        # Spatial-only levels still have spatial upsample convs but those
        # use 2D conv (no temporal cache needed)
        # Temporal upsample: time_conv is CausalConv3d -> 1 cache
        if level < num_levels - 1:
            if level < len(temp_up) and temp_up[level]:
                count += 1

    # conv_out: CausalConv3d(kt=3) -> 1 cache
    count += 1

    return count


def build_causal_vae_3d_engine(
    weights: "WeightDict",
    *,
    z_dim: int = 16,
    base_dim: int = 96,
    dim_mult: tuple[int, ...] = (1, 2, 4, 4),
    num_res_blocks: int = 2,
    temporal_upsample: tuple[bool, ...] = (False, True, True),
    h_lat: int = 60,
    w_lat: int = 104,
    out_channels: int = 3,
    norm_type: str = "l2_channel_norm",
    num_groups: int = 32,
    eps: float = 1e-6,
    verbose: bool = False,
) -> bytes:
    """Build causal 3D VAE decoder TRT engine plan.

    Full causal decoder with 32 cache I/O tensors. Processes one latent frame
    at a time; temporal expansion happens via pixel-shuffle inside the graph.

    Output temporal dim = scale_factor_temporal (4 for Wan2.1). The caller
    trims extra frames from frame-0 to get the correct total count.
    """
    from tensorrt_model_connect import trt_compat
    trt = trt_compat.get_trt()
    from . import graph_ops, graph_blocks

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    num_levels = len(dim_mult)
    channels_list = [base_dim * m for m in dim_mult]  # [96, 192, 384, 384]
    # Decoder processes in reversed order
    dec_channels = list(reversed(channels_list))       # [384, 384, 192, 96]
    mid_ch = dec_channels[0]                           # 384
    temp_up = list(reversed(temporal_upsample))        # [True, True, False]

    # --- Track spatial/temporal dims through the graph ---
    cur_h, cur_w = h_lat, w_lat
    cur_t = 1  # Starts at T=1

    # --- Input ---
    latent = network.add_input(
        "latent_frame", trt.float32, (1, z_dim, 1, h_lat, w_lat))

    # --- Cache inputs ---
    cache_idx = 0
    cache_inputs = {}   # idx -> trt.ITensor
    cache_outputs = {}  # idx -> trt.ITensor

    def _add_cache_input(channels: int, t_cache: int, h_c: int, w_c: int) -> trt.ITensor:
        nonlocal cache_idx
        name = f"cache_{cache_idx}"
        shape = (1, channels, t_cache, h_c, w_c)
        t = network.add_input(name, trt.float32, shape)
        cache_inputs[cache_idx] = t
        cache_idx += 1
        return t

    def _set_cache_output(idx: int, tensor: trt.ITensor) -> None:
        cache_outputs[idx] = tensor

    # --- post_quant_conv: 1x1x1 Conv3D [z_dim -> z_dim] ---
    x = graph_ops.add_conv3d_as_conv2d(
        network, latent,
        weight=weights["post_quant_conv.weight"],
        bias=weights["post_quant_conv.bias"],
        out_channels=z_dim,
        kernel_size=(1, 1, 1),
    )

    # --- conv_in: CausalConv3D [z_dim -> mid_ch, (3,3,3)] ---
    ci_cache = _add_cache_input(z_dim, 2, cur_h, cur_w)
    x, ci_cache_out = graph_ops.add_causal_conv3d(
        network, x, ci_cache,
        weight=weights["decoder.conv_in.weight"],
        bias=weights["decoder.conv_in.bias"],
        out_channels=mid_ch,
        kernel_size=(3, 3, 3),
        padding_hw=(1, 1),
    )
    _set_cache_output(cache_idx - 1, ci_cache_out)
    print(f"[vae-3d] conv_in: [{z_dim}]->[{mid_ch}], "
          f"T={cur_t}, {cur_h}x{cur_w}", file=sys.stderr)

    # --- mid_block: resnet.0 -> attention -> resnet.1 ---
    for mi in range(2):
        prefix = f"decoder.mid_block.resnets.{mi}"
        c1 = _add_cache_input(mid_ch, 2, cur_h, cur_w)
        c2 = _add_cache_input(mid_ch, 2, cur_h, cur_w)
        x, co1, co2 = graph_blocks.add_vae_resblock_3d(
            network, x, c1, c2,
            weights=weights, prefix=prefix,
            in_channels=mid_ch, out_channels=mid_ch,
            norm_type=norm_type, num_groups=num_groups, eps=eps)
        _set_cache_output(cache_idx - 2, co1)
        _set_cache_output(cache_idx - 1, co2)

        # Attention after first mid resnet
        if mi == 0:
            x = graph_blocks.add_vae_spatial_attention(
                network, x,
                weights=weights,
                prefix="decoder.mid_block.attentions.0",
                channels=mid_ch,
                norm_type=norm_type, num_groups=num_groups, eps=eps)

    print(f"[vae-3d] mid_block done, T={cur_t}, {cur_h}x{cur_w}",
          file=sys.stderr)

    # --- up_blocks ---
    # Channel assignment: each level's resnets output dec_channels[level].
    # The spatial upsample conv transitions channels to the next level's input.
    # Level 0: resnets 384, spatial 384->192
    # Level 1: resnets 192->384 (shortcut), spatial 384->192
    # Level 2: resnets 192, spatial 192->96
    # Level 3: resnets 96, no upsample
    prev_ch = mid_ch
    for level in range(num_levels):
        out_ch = dec_channels[level]
        has_spatial = level < num_levels - 1
        has_temporal = (level < len(temp_up) and temp_up[level]
                        and level < num_levels - 1)

        # 3 resnets per block (num_res_blocks + 1)
        for blk in range(num_res_blocks + 1):
            prefix = f"decoder.up_blocks.{level}.resnets.{blk}"
            in_ch = prev_ch if blk == 0 else out_ch

            c1 = _add_cache_input(in_ch, 2, cur_h, cur_w)
            c2 = _add_cache_input(out_ch, 2, cur_h, cur_w)
            x, co1, co2 = graph_blocks.add_vae_resblock_3d(
                network, x, c1, c2,
                weights=weights, prefix=prefix,
                in_channels=in_ch, out_channels=out_ch,
                norm_type=norm_type, num_groups=num_groups, eps=eps)
            _set_cache_output(cache_idx - 2, co1)
            _set_cache_output(cache_idx - 1, co2)

        prev_ch = out_ch
        print(f"[vae-3d] up_block {level}: ch={out_ch}, T={cur_t}, "
              f"{cur_h}x{cur_w}", file=sys.stderr)

        # HF order: temporal upsample FIRST, then spatial upsample
        if has_temporal:
            # Temporal pixel-shuffle: time_conv (C->2C, kt=3,1,1) + reshape
            tc_prefix = f"decoder.up_blocks.{level}.upsamplers.0.time_conv"
            tc_w = weights[f"{tc_prefix}.weight"]
            tc_in_ch = tc_w.shape[1]   # Input channels to time_conv
            tc_out_ch = tc_w.shape[0]  # Output channels (= 2 * tc_in_ch)
            tc_cache = _add_cache_input(tc_in_ch, 2, cur_h, cur_w)
            x, tc_cache_out = graph_ops.add_causal_conv3d(
                network, x, tc_cache,
                weight=tc_w,
                bias=weights[f"{tc_prefix}.bias"],
                out_channels=tc_out_ch,
                kernel_size=(3, 1, 1),
                padding_hw=(0, 0),
            )
            _set_cache_output(cache_idx - 1, tc_cache_out)

            # Pixel shuffle: [1, 2C, T, H, W] -> [1, C, 2T, H, W]
            x = graph_ops.add_temporal_pixel_shuffle(network, x, factor=2)
            prev_ch = tc_in_ch  # After pixel shuffle, channels = tc_in_ch
            cur_t *= 2
            print(f"[vae-3d]   temporal 2x -> {tc_in_ch}ch, T={cur_t}",
                  file=sys.stderr)

        if has_spatial:
            # Spatial 2x upsample: nearest + Conv3D(1,3,3)
            # The conv may change channels (e.g., 384->192 between levels)
            sp_prefix = f"decoder.up_blocks.{level}.upsamplers.0.resample.1"
            sp_w = weights[f"{sp_prefix}.weight"]
            sp_out_ch = sp_w.shape[0]  # Detect output channels from weight
            x = graph_ops.add_spatial_upsample_with_conv(
                network, x,
                weight=sp_w,
                bias=weights[f"{sp_prefix}.bias"],
                scale=2)
            cur_h *= 2
            cur_w *= 2
            prev_ch = sp_out_ch  # Track channel change from spatial conv
            print(f"[vae-3d]   spatial 2x -> {sp_out_ch}ch, {cur_h}x{cur_w}",
                  file=sys.stderr)

    # --- norm_out + SiLU ---
    if norm_type == "l2_channel_norm":
        x = graph_ops.add_l2_channel_norm(
            network, x, prev_ch,
            weights["decoder.norm_out.gamma"], eps)
    else:
        x = graph_ops.add_group_norm(
            network, x, prev_ch, num_groups,
            weights["decoder.norm_out.weight"],
            weights["decoder.norm_out.bias"], eps)
    x = graph_ops.add_silu(network, x)

    # --- conv_out: CausalConv3D [96 -> 3, (3,3,3)] ---
    co_cache = _add_cache_input(prev_ch, 2, cur_h, cur_w)
    x, co_cache_out = graph_ops.add_causal_conv3d(
        network, x, co_cache,
        weight=weights["decoder.conv_out.weight"],
        bias=weights["decoder.conv_out.bias"],
        out_channels=out_channels,
        kernel_size=(3, 3, 3),
        padding_hw=(1, 1),
    )
    _set_cache_output(cache_idx - 1, co_cache_out)

    # --- Mark outputs ---
    cast_x = network.add_cast(x, trt.float32)
    x_out = cast_x.get_output(0)
    x_out.name = "video_frame"
    network.mark_output(x_out)

    # Mark cache outputs
    for idx in sorted(cache_outputs.keys()):
        t = cache_outputs[idx]
        cast_t = network.add_cast(t, trt.float32)
        t_out = cast_t.get_output(0)
        t_out.name = f"cache_out_{idx}"
        network.mark_output(t_out)

    total_caches = cache_idx
    print(f"[vae-3d] Building TRT engine: {total_caches} caches, "
          f"output [1, {out_channels}, {cur_t}, {cur_h}, {cur_w}]",
          file=sys.stderr)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for 3D VAE")
    return bytes(plan)
