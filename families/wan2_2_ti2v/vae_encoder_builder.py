# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Wan2.2 TI2V-5B VAE encoder engine for the image-to-video first frame.

Image-to-video encodes only the conditioning image. Upstream ``WanVAE_.encode``
runs the encoder on that single frame with empty causal caches, so every causal
3x3x3 convolution sees two zero frames before the image and only its last
temporal tap contributes, and each temporal downsampler only caches the frame.
The engine is therefore a 2D network over one frame. It follows the official
``vae2_2.Encoder3d`` and its native weight names, and ends in the normalized
latent ``(mu - mean) * (1 / std)`` that the DiT consumes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

from .checkpoint_mapper import VAE22_CONFIG, load_native_vae_state_dict
from .vae_step_builder import (
    OFFICIAL_VAE_WORKSPACE_GIB,
    QUALIFIED_LATENT_SIZES,
    Wan22VaeStepProfile,
)


# Encoder3d(dim=160, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
# temperal_downsample=[False, True, True]); the last block neither downsamples.
BASE_CHANNELS = 160
BLOCK_CHANNELS = (160, 320, 640, 640)
TEMPORAL_DOWNSAMPLE = (False, True, True, False)
LATENT_CHANNELS = 48
PATCH_SIZE = 2
NORM_EPS = 1.0e-12


def load_vae_encoder_weights(checkpoint: str | Path) -> dict[str, np.ndarray]:
    """Load the native encoder and its quant convolution (``conv1``) as contiguous FP32."""

    state = load_native_vae_state_dict(checkpoint)
    return {
        name: np.ascontiguousarray(value.detach().float().cpu().numpy(), dtype=np.float32)
        for name, value in state.items()
        if name.startswith("encoder.") or name.startswith("conv1.")
    }


class _EncoderGraph:
    def __init__(self, network, weights: dict[str, np.ndarray]):
        from . import graph_ops

        self.network = network
        self.weights = weights
        self.ops = graph_ops
        # TensorRT reads weight pointers at build time: keep derived arrays alive until then.
        self.kept: list[np.ndarray] = []

    def add(self, left, right):
        return self.network.add_elementwise(left, right, trt.ElementWiseOperation.SUM).get_output(0)

    def conv(self, tensor, name: str, out_channels: int):
        """A causal Conv3d on the first frame: only its last temporal tap meets the image."""

        weight = self.weights[f"{name}.weight"]
        taps, kernel_h, kernel_w = weight.shape[2:]
        last_tap = np.ascontiguousarray(weight[:, :, taps - 1 : taps])
        self.kept.append(last_tap)
        return self.ops.add_conv3d_as_conv2d(
            self.network,
            tensor,
            weight=last_tap,
            bias=self.weights[f"{name}.bias"],
            out_channels=out_channels,
            kernel_size=(1, kernel_h, kernel_w),
            padding=(0, kernel_h // 2, kernel_w // 2),
        )

    def norm_silu(self, tensor, channels: int, gamma_name: str):
        normalized = self.ops.add_l2_channel_norm(
            self.network, tensor, channels, self.weights[gamma_name], NORM_EPS
        )
        return self.ops.add_silu(self.network, normalized)

    def residual(self, tensor, prefix: str, in_channels: int, out_channels: int):
        """ResidualBlock: RMS norm, SiLU, conv, RMS norm, SiLU, conv, plus the shortcut."""

        if in_channels == out_channels:
            shortcut = tensor
        else:
            shortcut = self.conv(tensor, f"{prefix}.shortcut", out_channels)
        hidden = self.norm_silu(tensor, in_channels, f"{prefix}.residual.0.gamma")
        hidden = self.conv(hidden, f"{prefix}.residual.2", out_channels)
        hidden = self.norm_silu(hidden, out_channels, f"{prefix}.residual.3.gamma")
        hidden = self.conv(hidden, f"{prefix}.residual.6", out_channels)
        return self.add(hidden, shortcut)

    def downsample(self, tensor, name: str, channels: int):
        """Resample's ZeroPad2d((0, 1, 0, 1)) and stride-2 3x3 Conv2d on every frame."""

        batch, _, frames, height, width = tuple(tensor.shape)
        rows = self.network.add_shuffle(tensor)
        rows.first_transpose = trt.Permutation([0, 2, 1, 3, 4])
        rows.reshape_dims = (batch * frames, channels, height, width)
        convolution = self.network.add_convolution_nd(
            rows.get_output(0),
            num_output_maps=channels,
            kernel_shape=(3, 3),
            kernel=trt.Weights(self.weights[f"{name}.weight"]),
            bias=trt.Weights(self.weights[f"{name}.bias"]),
        )
        convolution.stride_nd = (2, 2)
        convolution.pre_padding = (0, 0)
        convolution.post_padding = (1, 1)
        frames_back = self.network.add_shuffle(convolution.get_output(0))
        frames_back.reshape_dims = (batch, frames, channels, height // 2, width // 2)
        frames_back.second_transpose = trt.Permutation([0, 2, 1, 3, 4])
        return frames_back.get_output(0)

    def average_down(self, tensor, out_channels: int, factor_t: int, factor_s: int):
        """AvgDown3D: zero-pad time at the front, fold the factors into channels, then average."""

        batch, channels, frames, height, width = tuple(tensor.shape)
        pad_t = (factor_t - frames % factor_t) % factor_t
        if pad_t:
            zero = self.ops.add_constant(
                self.network, (1, 1, 1, 1, 1), np.zeros(1, dtype=np.float32)
            )
            zeros = self.network.add_elementwise(tensor, zero, trt.ElementWiseOperation.PROD)
            padding = self.network.add_concatenation([zeros.get_output(0)] * pad_t + [tensor])
            padding.axis = 2
            tensor, frames = padding.get_output(0), frames + pad_t
        group = channels * factor_t * factor_s * factor_s // out_channels
        if group == 1 and factor_t == factor_s == 1:
            return tensor
        folded = self.network.add_shuffle(tensor)
        folded.reshape_dims = (
            batch,
            channels,
            frames // factor_t,
            factor_t,
            height // factor_s,
            factor_s,
            width // factor_s,
            factor_s,
        )
        folded.second_transpose = trt.Permutation([0, 1, 3, 5, 7, 2, 4, 6])
        grouped = self.network.add_shuffle(folded.get_output(0))
        grouped.reshape_dims = (
            batch,
            out_channels,
            group,
            frames // factor_t,
            height // factor_s,
            width // factor_s,
        )
        return self.network.add_reduce(
            grouped.get_output(0), trt.ReduceOperation.AVG, 1 << 2, False
        ).get_output(0)


def _patchify(network, image, height: int, width: int):
    """``b c f (h q) (w r) -> b (c r q) f h w`` with q = r = 2."""

    factored = network.add_shuffle(image)
    factored.reshape_dims = (
        1,
        3,
        1,
        height // PATCH_SIZE,
        PATCH_SIZE,
        width // PATCH_SIZE,
        PATCH_SIZE,
    )
    factored.second_transpose = trt.Permutation([0, 1, 6, 4, 2, 3, 5])
    patched = network.add_shuffle(factored.get_output(0))
    patched.reshape_dims = (
        1,
        3 * PATCH_SIZE * PATCH_SIZE,
        1,
        height // PATCH_SIZE,
        width // PATCH_SIZE,
    )
    return patched.get_output(0)


def build_vae_encoder_engine(
    weights: dict[str, np.ndarray],
    *,
    profile: Wan22VaeStepProfile,
    verbose: bool = False,
) -> bytes:
    """Build the fixed first-frame encoder: ``image`` in [-1, 1] to ``image_latent``."""

    from . import graph_blocks

    if (profile.latent_height, profile.latent_width) not in QUALIFIED_LATENT_SIZES:
        raise ValueError(
            "Wan2.2 VAE encoder is qualified only for the generation profiles' latent sizes, "
            f"got {profile.latent_height}x{profile.latent_width}"
        )
    height, width = profile.latent_height * 16, profile.latent_width * 16
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, OFFICIAL_VAE_WORKSPACE_GIB << 30)

    graph = _EncoderGraph(network, weights)
    image = network.add_input("image", trt.float32, (1, 3, 1, height, width))
    x = graph.conv(_patchify(network, image, height, width), "encoder.conv1", BASE_CHANNELS)
    in_channels = BASE_CHANNELS
    for index, out_channels in enumerate(BLOCK_CHANNELS):
        prefix = f"encoder.downsamples.{index}"
        block_input = x
        x = graph.residual(x, f"{prefix}.downsamples.0", in_channels, out_channels)
        x = graph.residual(x, f"{prefix}.downsamples.1", out_channels, out_channels)
        spatial = index != len(BLOCK_CHANNELS) - 1
        if spatial:
            # A downsample3d time_conv only caches the first frame, so it adds nothing here.
            x = graph.downsample(x, f"{prefix}.downsamples.2.resample.1", out_channels)
        shortcut = graph.average_down(
            block_input,
            out_channels,
            factor_t=2 if TEMPORAL_DOWNSAMPLE[index] else 1,
            factor_s=2 if spatial else 1,
        )
        x = graph.add(x, shortcut)
        in_channels = out_channels

    x = graph.residual(x, "encoder.middle.0", in_channels, in_channels)
    x = graph_blocks.add_vae_spatial_attention(
        network, x, weights=weights, prefix="encoder.middle.1", channels=in_channels, eps=NORM_EPS
    )
    x = graph.residual(x, "encoder.middle.2", in_channels, in_channels)
    x = graph.norm_silu(x, in_channels, "encoder.head.0.gamma")
    x = graph.conv(x, "encoder.head.2", 2 * LATENT_CHANNELS)
    x = graph.conv(x, "conv1", 2 * LATENT_CHANNELS)  # the quant convolution

    mu = network.add_slice(
        x,
        (0, 0, 0, 0, 0),
        (1, LATENT_CHANNELS, 1, profile.latent_height, profile.latent_width),
        (1, 1, 1, 1, 1),
    ).get_output(0)
    mean = np.asarray(VAE22_CONFIG["latents_mean"], dtype=np.float32).reshape(1, -1, 1, 1, 1)
    inverse_std = (
        np.float32(1.0) / np.asarray(VAE22_CONFIG["latents_std"], dtype=np.float32)
    ).reshape(1, -1, 1, 1, 1)
    centered = network.add_elementwise(
        mu,
        graph.ops.add_constant(network, mean.shape, mean),
        trt.ElementWiseOperation.SUB,
    ).get_output(0)
    latent = network.add_elementwise(
        centered,
        graph.ops.add_constant(network, inverse_std.shape, inverse_std),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    latent.name = "image_latent"
    network.mark_output(latent)

    print(
        f"[wan2.2-ti2v] building VAE encoder: image={width}x{height}, "
        f"latent={profile.latent_height}x{profile.latent_width}",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the Wan2.2 VAE encoder")
    return bytes(plan)
