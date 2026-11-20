# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT engine for the ``AutoencoderKLQwenImage21`` decoder on one image.

The checkpoint's "causal 3D" convolutions are 2D convolutions applied to a single frame, and on a
first (and only) frame the temporal upsampling convolutions are skipped, so the image decode path
is a plain 2D residual decoder:

  post_quant_conv -> conv_in -> mid block (residual, single-head attention, residual)
  -> five up blocks (three residual blocks each; the first four upsample 2x and add a
     parameter-free shortcut that spreads the block's input channels over the new pixels)
  -> channel norm -> SiLU -> conv_out -> clamp to [-1, 1]

Engine contract (FP32 I/O):
  ``latents``  [1, z_dim, h, w]            normalised latents, as the denoiser produces them
  ``image``    [1, out_channels, 16h, 16w] RGBA in [-1, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import tensorrt as trt

from . import trt_ops
from .resample import shortcut_channels
from .trt_ops import BF16, FP32, Graph
from .weights import read_json, read_safetensors, take

_OPTIMIZATION_LEVEL = 3
_WORKSPACE_BYTES = 16 << 30


@dataclass(frozen=True)
class VaeConfig:
    z_dim: int
    decoder_dim: int
    dim_mult: tuple[int, ...]
    num_res_blocks: int
    temporal_upsample: tuple[bool, ...]
    out_channels: int
    spatial_scale: int
    latents_mean: tuple[float, ...]
    latents_std: tuple[float, ...]


def load_vae(vae_dir: Path) -> tuple[VaeConfig, dict[str, np.ndarray]]:
    raw = read_json(vae_dir / "config.json")
    if raw.get("_class_name") != "AutoencoderKLQwenImage21":
        raise ValueError(f"unsupported VAE class {raw.get('_class_name')!r}")
    if not bool(raw["is_residual"]) or raw["attn_scales"] or raw["patch_size"] is not None:
        raise ValueError("unsupported AutoencoderKLQwenImage21 configuration")
    dim_mult = tuple(int(value) for value in raw["dim_mult"])
    # The decoder upsamples in the reverse order of the encoder's downsampling.
    temporal_upsample = tuple(bool(value) for value in reversed(raw["temperal_downsample"]))
    config = VaeConfig(
        z_dim=int(raw["z_dim"]),
        decoder_dim=int(raw["decoder_base_dim"] or raw["base_dim"]),
        dim_mult=dim_mult,
        num_res_blocks=int(raw["num_res_blocks"]),
        temporal_upsample=temporal_upsample,
        out_channels=int(raw["out_channels"]),
        spatial_scale=int(raw["scale_factor_spatial"]),
        latents_mean=tuple(float(value) for value in raw["latents_mean"]),
        latents_std=tuple(float(value) for value in raw["latents_std"]),
    )
    if len(temporal_upsample) != len(dim_mult) - 1:
        raise ValueError("temperal_downsample must have one entry per resampling stage")
    if config.spatial_scale != 2 ** (len(dim_mult) - 1):
        raise ValueError("scale_factor_spatial does not match the number of upsampling stages")
    if len(config.latents_mean) != config.z_dim or len(config.latents_std) != config.z_dim:
        raise ValueError("latents_mean and latents_std must have z_dim entries")
    return config, read_safetensors(vae_dir, prefixes=("decoder.", "post_quant_conv."))


def _shortcut(
    graph: Graph, tensor: trt.ITensor, in_dim: int, out_dim: int, factor_t: int
) -> trt.ITensor:
    _, _, height, width = trt_ops.shape_of(tensor)
    indices = trt_ops.constant(graph, shortcut_channels(in_dim, out_dim, factor_t))
    spread = graph.network.add_gather(tensor, indices, 1).get_output(0)
    spread = trt_ops.reshape(graph, spread, (1, out_dim, 2, 2, height, width))
    spread = trt_ops.transpose(graph, spread, (0, 1, 4, 2, 5, 3))
    return trt_ops.reshape(graph, spread, (1, out_dim, height * 2, width * 2))


def _residual_block(
    graph: Graph, weights: dict, prefix: str, tensor: trt.ITensor, in_dim: int, out_dim: int
) -> trt.ITensor:
    def conv(name: str, source: trt.ITensor, channels: int, kernel: int) -> trt.ITensor:
        return trt_ops.conv2d(
            graph,
            source,
            take(weights, f"{prefix}.{name}.weight", (out_dim, channels, kernel, kernel)),
            take(weights, f"{prefix}.{name}.bias", (out_dim,)),
            padding=kernel // 2,
        )

    shortcut = tensor if in_dim == out_dim else conv("conv_shortcut", tensor, in_dim, 1)
    hidden = trt_ops.channel_l2_norm(
        graph, tensor, take(weights, f"{prefix}.norm1.gamma", (in_dim, 1, 1, 1))
    )
    hidden = conv("conv1", trt_ops.silu(graph, hidden), in_dim, 3)
    hidden = trt_ops.channel_l2_norm(
        graph, hidden, take(weights, f"{prefix}.norm2.gamma", (out_dim, 1, 1, 1))
    )
    hidden = conv("conv2", trt_ops.silu(graph, hidden), out_dim, 3)
    return trt_ops.add(graph, hidden, shortcut)


def _attention_block(
    graph: Graph, weights: dict, prefix: str, tensor: trt.ITensor, channels: int
) -> trt.ITensor:
    _, _, height, width = trt_ops.shape_of(tensor)
    tokens = height * width
    normed = trt_ops.channel_l2_norm(
        graph, tensor, take(weights, f"{prefix}.norm.gamma", (channels, 1, 1))
    )
    qkv = trt_ops.conv2d(
        graph,
        normed,
        take(weights, f"{prefix}.to_qkv.weight", (3 * channels, channels, 1, 1)),
        take(weights, f"{prefix}.to_qkv.bias", (3 * channels,)),
    )
    qkv = trt_ops.transpose(graph, trt_ops.reshape(graph, qkv, (1, 3 * channels, tokens)), (0, 2, 1))
    query, key, value = (
        trt_ops.reshape(
            graph, trt_ops.slice_axis(graph, qkv, 2, index * channels, channels), (1, 1, tokens, channels)
        )
        for index in range(3)
    )
    context = trt_ops.attention(graph, query, key, value)
    context = trt_ops.transpose(graph, trt_ops.reshape(graph, context, (1, tokens, channels)), (0, 2, 1))
    context = trt_ops.reshape(graph, context, (1, channels, height, width))
    projected = trt_ops.conv2d(
        graph,
        context,
        take(weights, f"{prefix}.proj.weight", (channels, channels, 1, 1)),
        take(weights, f"{prefix}.proj.bias", (channels,)),
    )
    return trt_ops.add(graph, projected, tensor)


def build_vae_decoder_plan(
    config: VaeConfig,
    weights: dict[str, np.ndarray],
    output: BinaryIO,
    *,
    latent_height: int,
    latent_width: int,
    verbose: bool,
    compute: trt.DataType = BF16,
) -> None:
    """Build the image decoder for one static latent size and write its plan to ``output``.

    ``compute`` is BF16 for bundles; the parity tests pass FP32 to build the same graph without
    BF16 rounding.
    """
    if latent_height < 1 or latent_width < 1:
        raise ValueError("VAE latent size must be positive")
    graph = trt_ops.create_graph(
        verbose=verbose,
        workspace_bytes=_WORKSPACE_BYTES,
        optimization_level=_OPTIMIZATION_LEVEL,
        compute=compute,
    )
    network = graph.network
    z_dim = config.z_dim
    latents = network.add_input("latents", FP32, (1, z_dim, latent_height, latent_width))

    # The pipeline de-normalises latents before decoding: z * std + mean, per channel.
    std = np.asarray(config.latents_std, dtype=np.float32).reshape(1, z_dim, 1, 1)
    mean = np.asarray(config.latents_mean, dtype=np.float32).reshape(1, z_dim, 1, 1)
    hidden = trt_ops.add(
        graph, trt_ops.mul(graph, latents, trt_ops.constant(graph, std)), trt_ops.constant(graph, mean)
    )
    hidden = trt_ops.cast(graph, hidden, graph.compute)
    hidden = trt_ops.conv2d(
        graph,
        hidden,
        take(weights, "post_quant_conv.weight", (z_dim, z_dim, 1, 1)),
        take(weights, "post_quant_conv.bias", (z_dim,)),
    )

    dims = [config.decoder_dim * multiplier for multiplier in (config.dim_mult[-1], *config.dim_mult[::-1])]
    hidden = trt_ops.conv2d(
        graph,
        hidden,
        take(weights, "decoder.conv_in.weight", (dims[0], z_dim, 3, 3)),
        take(weights, "decoder.conv_in.bias", (dims[0],)),
        padding=1,
    )
    hidden = _residual_block(graph, weights, "decoder.mid_block.resnets.0", hidden, dims[0], dims[0])
    hidden = _attention_block(graph, weights, "decoder.mid_block.attentions.0", hidden, dims[0])
    hidden = _residual_block(graph, weights, "decoder.mid_block.resnets.1", hidden, dims[0], dims[0])

    stages = len(config.dim_mult)
    for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        prefix = f"decoder.up_blocks.{index}"
        block_input = hidden
        channels = in_dim
        for resnet in range(config.num_res_blocks + 1):
            hidden = _residual_block(
                graph, weights, f"{prefix}.resnets.{resnet}", hidden, channels, out_dim
            )
            channels = out_dim
        if index == stages - 1:
            continue
        hidden = trt_ops.conv2d(
            graph,
            trt_ops.upsample_nearest_2x(graph, hidden),
            take(weights, f"{prefix}.upsampler.resample.1.weight", (out_dim, out_dim, 3, 3)),
            take(weights, f"{prefix}.upsampler.resample.1.bias", (out_dim,)),
            padding=1,
        )
        factor_t = 2 if config.temporal_upsample[index] else 1
        hidden = trt_ops.add(graph, hidden, _shortcut(graph, block_input, in_dim, out_dim, factor_t))

    hidden = trt_ops.channel_l2_norm(
        graph, hidden, take(weights, "decoder.norm_out.gamma", (dims[-1], 1, 1, 1))
    )
    hidden = trt_ops.conv2d(
        graph,
        trt_ops.silu(graph, hidden),
        take(weights, "decoder.conv_out.weight", (config.out_channels, dims[-1], 3, 3)),
        take(weights, "decoder.conv_out.bias", (config.out_channels,)),
        padding=1,
    )
    image = trt_ops.clamp(graph, trt_ops.cast(graph, hidden, FP32), -1.0, 1.0)
    image.name = "image"
    network.mark_output(image)

    output.write(memoryview(trt_ops.serialize(graph, "VAE decoder")))
