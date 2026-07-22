# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native recurrent Wan2.2 TI2V-5B VAE decoder engines.

The Diffusers decoder consumes one latent frame per call.  Thirty-two causal
convolutions carry the two preceding feature frames between calls.  Encoding
that source contract as two fixed TensorRT engines avoids the 281.6 GB
activation arena of the fully unrolled 31-latent graph:

* the initializer consumes one latent and zero caches and emits one video frame;
* the recurrent step consumes one latent and the 32 caches and emits four frames.

Both engines expose identical fixed-size ``cache_0`` ... ``cache_31`` inputs
and ``cache_out_0`` ... ``cache_out_31`` outputs.  Initializer cache outputs are
left-zero-padded to two frames, replacing Diffusers' Python-only ``"Rep"``
sentinel with an exactly equivalent native tensor contract.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .checkpoint_mapper import VAE22_CONFIG, convert_vae_state_dict, load_native_vae_state_dict


OFFICIAL_VAE_WORKSPACE_GIB = 64


@dataclass(frozen=True)
class Wan22VaeStepProfile:
    """Static spatial contract shared by initializer and recurrent engines."""

    latent_height: int
    latent_width: int

    def __post_init__(self) -> None:
        if self.latent_height <= 0 or self.latent_width <= 0:
            raise ValueError(
                "Wan2.2 VAE step latent dimensions must be positive, got "
                f"{self.latent_height}x{self.latent_width}"
            )

    @property
    def latent_shape(self) -> tuple[int, int, int, int, int]:
        return (1, 48, 1, self.latent_height, self.latent_width)

    def video_shape(self, *, first_frame_only: bool) -> tuple[int, int, int, int, int]:
        return (
            1,
            3,
            1 if first_frame_only else 4,
            self.latent_height * 16,
            self.latent_width * 16,
        )


@dataclass(frozen=True)
class Wan22VaeCacheSpec:
    """One source-ordered causal feature-cache tensor."""

    index: int
    logical_name: str
    channels: int
    spatial_scale: int

    def shape(self, profile: Wan22VaeStepProfile) -> tuple[int, int, int, int, int]:
        return (
            1,
            self.channels,
            2,
            profile.latent_height * self.spatial_scale,
            profile.latent_width * self.spatial_scale,
        )


# Execution order is the mutable feat_idx order in Diffusers WanDecoder3d.
# The decoder owns 34 WanCausalConv3d modules, but the two channel-changing
# conv_shortcut modules are ordinary residual projections and never use caches.
VAE_STEP_CACHE_SPECS = (
    Wan22VaeCacheSpec(0, "decoder.conv_in", 48, 1),
    Wan22VaeCacheSpec(1, "decoder.mid_block.resnets.0.conv1", 1024, 1),
    Wan22VaeCacheSpec(2, "decoder.mid_block.resnets.0.conv2", 1024, 1),
    Wan22VaeCacheSpec(3, "decoder.mid_block.resnets.1.conv1", 1024, 1),
    Wan22VaeCacheSpec(4, "decoder.mid_block.resnets.1.conv2", 1024, 1),
    *(
        Wan22VaeCacheSpec(
            5 + resnet * 2 + conv - 1, f"decoder.up_blocks.0.resnets.{resnet}.conv{conv}", 1024, 1
        )
        for resnet in range(3)
        for conv in (1, 2)
    ),
    Wan22VaeCacheSpec(11, "decoder.up_blocks.0.upsampler.time_conv", 1024, 1),
    *(
        Wan22VaeCacheSpec(
            12 + resnet * 2 + conv - 1, f"decoder.up_blocks.1.resnets.{resnet}.conv{conv}", 1024, 2
        )
        for resnet in range(3)
        for conv in (1, 2)
    ),
    Wan22VaeCacheSpec(18, "decoder.up_blocks.1.upsampler.time_conv", 1024, 2),
    Wan22VaeCacheSpec(19, "decoder.up_blocks.2.resnets.0.conv1", 1024, 4),
    Wan22VaeCacheSpec(20, "decoder.up_blocks.2.resnets.0.conv2", 512, 4),
    Wan22VaeCacheSpec(21, "decoder.up_blocks.2.resnets.1.conv1", 512, 4),
    Wan22VaeCacheSpec(22, "decoder.up_blocks.2.resnets.1.conv2", 512, 4),
    Wan22VaeCacheSpec(23, "decoder.up_blocks.2.resnets.2.conv1", 512, 4),
    Wan22VaeCacheSpec(24, "decoder.up_blocks.2.resnets.2.conv2", 512, 4),
    Wan22VaeCacheSpec(25, "decoder.up_blocks.3.resnets.0.conv1", 512, 8),
    Wan22VaeCacheSpec(26, "decoder.up_blocks.3.resnets.0.conv2", 256, 8),
    Wan22VaeCacheSpec(27, "decoder.up_blocks.3.resnets.1.conv1", 256, 8),
    Wan22VaeCacheSpec(28, "decoder.up_blocks.3.resnets.1.conv2", 256, 8),
    Wan22VaeCacheSpec(29, "decoder.up_blocks.3.resnets.2.conv1", 256, 8),
    Wan22VaeCacheSpec(30, "decoder.up_blocks.3.resnets.2.conv2", 256, 8),
    Wan22VaeCacheSpec(31, "decoder.conv_out", 256, 8),
)

if tuple(spec.index for spec in VAE_STEP_CACHE_SPECS) != tuple(range(32)):
    raise RuntimeError("Wan2.2 VAE cache specification must contain consecutive indices 0..31")


def load_vae_step_weights(checkpoint: str | Path) -> dict[str, np.ndarray]:
    """Load only converted decoder and post-quant weights as contiguous FP32."""

    converted = convert_vae_state_dict(load_native_vae_state_dict(checkpoint))
    return {
        name: np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
        for name, value in converted.items()
        if name.startswith("decoder.") or name.startswith("post_quant_conv.")
    }


def _add_dup_up3d(
    trt,
    network,
    tensor,
    *,
    out_channels: int,
    factor_t: int,
    factor_s: int,
    first_frame_only: bool,
):
    """TensorRT spelling of Diffusers DupUp3D, including first-chunk trim."""

    batch, in_channels, frames, height, width = tuple(tensor.shape)
    factor = factor_t * factor_s * factor_s
    if out_channels * factor % in_channels != 0:
        raise RuntimeError(
            f"Invalid Wan2.2 DupUp3D contract {in_channels}->{out_channels}, factor={factor}"
        )
    repeats = out_channels * factor // in_channels
    expanded = network.add_shuffle(tensor)
    expanded.reshape_dims = (batch, in_channels, 1, frames, height, width)
    repeated = network.add_concatenation([expanded.get_output(0)] * repeats)
    repeated.axis = 2
    factored = network.add_shuffle(repeated.get_output(0))
    factored.reshape_dims = (
        batch,
        out_channels,
        factor_t,
        factor_s,
        factor_s,
        frames,
        height,
        width,
    )
    arranged = network.add_shuffle(factored.get_output(0))
    arranged.first_transpose = trt.Permutation([0, 1, 5, 2, 6, 3, 7, 4])
    output = network.add_shuffle(arranged.get_output(0))
    output.reshape_dims = (
        batch,
        out_channels,
        frames * factor_t,
        height * factor_s,
        width * factor_s,
    )
    result = output.get_output(0)
    if first_frame_only and factor_t > 1:
        trim = network.add_slice(
            result,
            start=(0, 0, factor_t - 1, 0, 0),
            shape=(batch, out_channels, frames, height * factor_s, width * factor_s),
            stride=(1, 1, 1, 1, 1),
        )
        result = trim.get_output(0)
    return result


def _add_unpatchify(trt, network, tensor, *, patch_size: int = 2):
    """TensorRT spelling of Diffusers unpatchify for the Wan2.2 12-channel head."""

    batch, patched_channels, frames, height, width = tuple(tensor.shape)
    channels = patched_channels // (patch_size * patch_size)
    factored = network.add_shuffle(tensor)
    factored.reshape_dims = (
        batch,
        channels,
        patch_size,
        patch_size,
        frames,
        height,
        width,
    )
    arranged = network.add_shuffle(factored.get_output(0))
    arranged.first_transpose = trt.Permutation([0, 1, 4, 5, 3, 6, 2])
    output = network.add_shuffle(arranged.get_output(0))
    output.reshape_dims = (
        batch,
        channels,
        frames,
        height * patch_size,
        width * patch_size,
    )
    clip = network.add_activation(output.get_output(0), trt.ActivationType.CLIP)
    clip.alpha = -1.0
    clip.beta = 1.0
    return clip.get_output(0)


def build_vae_step_engine(
    weights: dict[str, np.ndarray],
    *,
    profile: Wan22VaeStepProfile,
    first_frame_only: bool,
    verbose: bool = False,
) -> bytes:
    """Build one fixed initializer or recurrent Wan2.2 VAE engine."""

    from tensorrt_model_connect import trt_compat
    from . import graph_blocks, graph_ops

    if (profile.latent_height, profile.latent_width) not in {
        (24, 42),
        (44, 80),
    }:
        raise ValueError(
            "Wan2.2 VAE step engines are qualified only for 24x42 and 44x80 latent spatial "
            f"profiles, got {profile.latent_height}x{profile.latent_width}"
        )

    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, OFFICIAL_VAE_WORKSPACE_GIB << 30)

    latent = network.add_input("latent_frame", trt.float32, profile.latent_shape)
    cache_inputs = [
        network.add_input(f"cache_{spec.index}", trt.float32, spec.shape(profile))
        for spec in VAE_STEP_CACHE_SPECS
    ]
    cache_outputs = [None] * len(VAE_STEP_CACHE_SPECS)
    cache_index = 0

    def take_cache(logical_name: str, channels: int, scale: int):
        nonlocal cache_index
        spec = VAE_STEP_CACHE_SPECS[cache_index]
        if (spec.logical_name, spec.channels, spec.spatial_scale) != (
            logical_name,
            channels,
            scale,
        ):
            raise RuntimeError(
                f"Wan2.2 VAE cache {cache_index} is {spec}, expected "
                f"{logical_name}/{channels}/scale{scale}"
            )
        index = cache_index
        cache_index += 1
        return index, cache_inputs[index]

    mean = graph_ops.add_constant(
        network,
        (1, 48, 1, 1, 1),
        np.asarray(VAE22_CONFIG["latents_mean"], dtype=np.float32).reshape(1, 48, 1, 1, 1),
    )
    std = graph_ops.add_constant(
        network,
        (1, 48, 1, 1, 1),
        np.asarray(VAE22_CONFIG["latents_std"], dtype=np.float32).reshape(1, 48, 1, 1, 1),
    )
    scaled = network.add_elementwise(latent, std, trt.ElementWiseOperation.PROD)
    denormalized = network.add_elementwise(scaled.get_output(0), mean, trt.ElementWiseOperation.SUM)
    x = graph_ops.add_conv3d_as_conv2d(
        network,
        denormalized.get_output(0),
        weight=weights["post_quant_conv.weight"],
        bias=weights["post_quant_conv.bias"],
        out_channels=48,
        kernel_size=(1, 1, 1),
    )

    current_scale = 1
    index, cache = take_cache("decoder.conv_in", 48, current_scale)
    x, cache_outputs[index] = graph_ops.add_causal_conv3d(
        network,
        x,
        cache,
        weight=weights["decoder.conv_in.weight"],
        bias=weights["decoder.conv_in.bias"],
        out_channels=1024,
        kernel_size=(3, 3, 3),
        padding_hw=(1, 1),
    )

    def add_resnet(tensor, *, prefix: str, input_channels: int, output_channels: int):
        norm1 = graph_ops.add_l2_channel_norm(
            network,
            tensor,
            input_channels,
            weights[f"{prefix}.norm1.gamma"],
            1.0e-12,
        )
        act1 = graph_ops.add_silu(network, norm1)
        cache1_index, cache1 = take_cache(f"{prefix}.conv1", input_channels, current_scale)
        conv1, cache_outputs[cache1_index] = graph_ops.add_causal_conv3d(
            network,
            act1,
            cache1,
            weight=weights[f"{prefix}.conv1.weight"],
            bias=weights[f"{prefix}.conv1.bias"],
            out_channels=output_channels,
            kernel_size=(3, 3, 3),
            padding_hw=(1, 1),
        )
        norm2 = graph_ops.add_l2_channel_norm(
            network,
            conv1,
            output_channels,
            weights[f"{prefix}.norm2.gamma"],
            1.0e-12,
        )
        act2 = graph_ops.add_silu(network, norm2)
        cache2_index, cache2 = take_cache(f"{prefix}.conv2", output_channels, current_scale)
        conv2, cache_outputs[cache2_index] = graph_ops.add_causal_conv3d(
            network,
            act2,
            cache2,
            weight=weights[f"{prefix}.conv2.weight"],
            bias=weights[f"{prefix}.conv2.bias"],
            out_channels=output_channels,
            kernel_size=(3, 3, 3),
            padding_hw=(1, 1),
        )
        if input_channels == output_channels:
            shortcut = tensor
        else:
            shortcut = graph_ops.add_conv3d_as_conv2d(
                network,
                tensor,
                weight=weights[f"{prefix}.conv_shortcut.weight"],
                bias=weights[f"{prefix}.conv_shortcut.bias"],
                out_channels=output_channels,
                kernel_size=(1, 1, 1),
            )
        return network.add_elementwise(conv2, shortcut, trt.ElementWiseOperation.SUM).get_output(0)

    for middle in range(2):
        prefix = f"decoder.mid_block.resnets.{middle}"
        x = add_resnet(x, prefix=prefix, input_channels=1024, output_channels=1024)
        if middle == 0:
            x = graph_blocks.add_vae_spatial_attention(
                network,
                x,
                weights=weights,
                prefix="decoder.mid_block.attentions.0",
                channels=1024,
                eps=1.0e-12,
            )

    decoder_channels = (1024, 1024, 512, 256)
    previous_channels = 1024
    for level, output_channels in enumerate(decoder_channels):
        block_input = x
        for resnet in range(3):
            input_channels = previous_channels if resnet == 0 else output_channels
            prefix = f"decoder.up_blocks.{level}.resnets.{resnet}"
            x = add_resnet(
                x,
                prefix=prefix,
                input_channels=input_channels,
                output_channels=output_channels,
            )
        previous_channels = output_channels

        has_upsampler = level < 3
        has_temporal = level in (0, 1)
        if has_temporal:
            prefix = f"decoder.up_blocks.{level}.upsampler.time_conv"
            index, cache = take_cache(prefix, output_channels, current_scale)
            if first_frame_only:
                # TensorRT does not allow one ITensor to be both a network
                # input and output.  Diffusers' first call leaves these two
                # zero/sentinel caches unchanged, so materialize an explicit
                # identity output with the same source semantics.
                cache_outputs[index] = network.add_identity(cache).get_output(0)
            else:
                x, cache_outputs[index] = graph_ops.add_causal_conv3d(
                    network,
                    x,
                    cache,
                    weight=weights[f"{prefix}.weight"],
                    bias=weights[f"{prefix}.bias"],
                    out_channels=output_channels * 2,
                    kernel_size=(3, 1, 1),
                )
                x = graph_ops.add_temporal_pixel_shuffle(network, x, factor=2)

        if has_upsampler:
            prefix = f"decoder.up_blocks.{level}.upsampler.resample.1"
            x = graph_ops.add_spatial_upsample_with_conv(
                network,
                x,
                weight=weights[f"{prefix}.weight"],
                bias=weights[f"{prefix}.bias"],
                scale=2,
            )
            current_scale *= 2

            shortcut = _add_dup_up3d(
                trt,
                network,
                block_input,
                out_channels=output_channels,
                factor_t=2 if has_temporal else 1,
                factor_s=2,
                first_frame_only=first_frame_only,
            )
            x = network.add_elementwise(x, shortcut, trt.ElementWiseOperation.SUM).get_output(0)

    x = graph_ops.add_l2_channel_norm(
        network,
        x,
        256,
        weights["decoder.norm_out.gamma"],
        1.0e-12,
    )
    x = graph_ops.add_silu(network, x)
    index, cache = take_cache("decoder.conv_out", 256, current_scale)
    x, cache_outputs[index] = graph_ops.add_causal_conv3d(
        network,
        x,
        cache,
        weight=weights["decoder.conv_out.weight"],
        bias=weights["decoder.conv_out.bias"],
        out_channels=12,
        kernel_size=(3, 3, 3),
        padding_hw=(1, 1),
    )
    x = _add_unpatchify(trt, network, x)

    if cache_index != len(VAE_STEP_CACHE_SPECS) or any(value is None for value in cache_outputs):
        raise RuntimeError(
            f"Wan2.2 VAE step graph populated {cache_index} caches with "
            f"{sum(value is not None for value in cache_outputs)} outputs, expected 32"
        )
    expected_video_shape = profile.video_shape(first_frame_only=first_frame_only)
    if tuple(x.shape) != expected_video_shape:
        raise RuntimeError(
            f"Wan2.2 VAE step output is {tuple(x.shape)}, expected {expected_video_shape}"
        )
    x.name = "video_frame"
    network.mark_output(x)
    for spec, cache_output in zip(VAE_STEP_CACHE_SPECS, cache_outputs):
        if tuple(cache_output.shape) != spec.shape(profile):
            raise RuntimeError(
                f"Wan2.2 VAE cache_out_{spec.index} is {tuple(cache_output.shape)}, "
                f"expected {spec.shape(profile)}"
            )
        cache_output.name = f"cache_out_{spec.index}"
        network.mark_output(cache_output)

    kind = "initializer" if first_frame_only else "recurrent"
    print(
        f"[wan22-vae-step] building {kind}: video={expected_video_shape}, "
        f"caches={len(VAE_STEP_CACHE_SPECS)}",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TensorRT returned no serialized Wan2.2 VAE {kind} engine")
    return bytes(plan)
