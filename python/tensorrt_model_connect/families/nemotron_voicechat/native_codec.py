# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-native decoder for the VoiceChat RVQ-VAE audio codec.

The public VoiceChat checkpoint stores the frozen codec below
``tts_model.audio_codec``.  One generated codec frame contains 31 residual-VQ
indices.  The indices are looked up in 31 independent 1024 by 512 codebooks,
summed, and decoded by three causal ConvNeXt/transpose-convolution stages.

This module intentionally stops at the real-valued 18-channel spectral
representation.  The small, stateful ISTFT is implemented by the C++ runtime
in ``codec_reconstruction.cpp``; no Python audio stack is needed at runtime.

The engine contract is deliberately frame-oriented for duplex use:

* ``codec_codes``: ``[1, 31]`` INT32
* ``codec_cache_in_{0..8}``: the six causal samples preceding each ConvNeXt
  block, in ``[1, channels, 6]`` layout
* ``spectral_params``: ``[18, 441]`` FP32, with magnitude
  logits in channels 0..8 and phase in channels 9..17
* ``codec_cache_out_{0..8}``: replacement causal caches for the next call

The engine consumes exactly one 80 ms model frame and produces 441 spectral
frames, which the C++ reconstruction turns
into exactly 1764 mono samples at 22050 Hz.  Keeping the causal state explicit
also makes reset and conversation/session ownership unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


CHECKPOINT_PREFIX = "tts_model.audio_codec."
CODEC_ENGINE_SECTION = "codec_engine_plan"
CODE_INPUT_NAME = "codec_codes"
SPECTRAL_OUTPUT_NAME = "spectral_params"
CAUSAL_CACHE_WIDTH = 6


@dataclass(frozen=True)
class CodecArchitecture:
    """Frozen architecture carried by NVIDIA-NemotronLabs-VoiceChat-11B."""

    num_quantizers: int = 31
    codebook_size: int = 1024
    latent_size: int = 512
    base_hidden_size: int = 384
    channel_mult: tuple[int, ...] = (4, 2, 1)
    rates: tuple[int, ...] = (9, 7, 7)
    num_blocks: int = 3
    convnext_kernel_size: int = 7
    n_fft: int = 16
    hop_length: int = 4
    layer_norm_epsilon: float = 1e-6
    samples_per_codec_frame: int = 1764

    @property
    def spectral_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def spectral_channels(self) -> int:
        return self.n_fft + 2

    @property
    def spectral_frames_per_codec_frame(self) -> int:
        result = 1
        for rate in self.rates:
            result *= rate
        return result

    @property
    def cache_channels(self) -> tuple[int, ...]:
        return tuple(
            self.base_hidden_size * multiplier
            for multiplier in self.channel_mult
            for _ in range(self.num_blocks)
        )

    def validate(self) -> None:
        if len(self.channel_mult) != len(self.rates):
            raise ValueError("codec channel_mult and rates must have the same length")
        if self.convnext_kernel_size - 1 != CAUSAL_CACHE_WIDTH:
            raise ValueError("VoiceChat codec requires a six-sample ConvNeXt cache")
        produced = self.spectral_frames_per_codec_frame * self.hop_length
        if produced != self.samples_per_codec_frame:
            raise ValueError(
                "codec upsampling/hop does not match samples_per_codec_frame: "
                f"{produced} != {self.samples_per_codec_frame}"
            )


VOICECHAT_CODEC = CodecArchitecture()
VOICECHAT_CODEC.validate()


def cache_input_name(block: int) -> str:
    return f"codec_cache_in_{block}"


def cache_output_name(block: int) -> str:
    return f"codec_cache_out_{block}"


def expected_weight_shapes(
    architecture: CodecArchitecture = VOICECHAT_CODEC,
) -> dict[str, tuple[int, ...]]:
    """Return the exact public-checkpoint tensor names and shapes we consume."""

    shapes: dict[str, tuple[int, ...]] = {}
    for quantizer in range(architecture.num_quantizers):
        shapes[f"prvq.mus_list.{quantizer}"] = (
            architecture.codebook_size,
            architecture.latent_size,
        )

    layer = 0
    in_channels = architecture.latent_size
    for multiplier, rate in zip(architecture.channel_mult, architecture.rates):
        channels = architecture.base_hidden_size * multiplier
        # PyTorch ConvTranspose1d stores [in_channels, out_channels/groups, K].
        shapes[f"decoder.layers.{layer}.weight"] = (in_channels, channels, rate)
        layer += 1
        for _ in range(architecture.num_blocks):
            intermediate = channels * 4
            stem = f"decoder.layers.{layer}"
            shapes[f"{stem}.dwconv.weight"] = (channels, 1, architecture.convnext_kernel_size)
            shapes[f"{stem}.dwconv.bias"] = (channels,)
            shapes[f"{stem}.norm.weight"] = (channels,)
            shapes[f"{stem}.norm.bias"] = (channels,)
            shapes[f"{stem}.pwconv1.weight"] = (intermediate, channels, 1)
            shapes[f"{stem}.pwconv1.bias"] = (intermediate,)
            shapes[f"{stem}.pwconv2.weight"] = (channels, intermediate, 1)
            shapes[f"{stem}.pwconv2.bias"] = (channels,)
            layer += 1
        in_channels = channels

    shapes[f"decoder.layers.{layer}.weight"] = (
        architecture.spectral_channels,
        in_channels,
        1,
    )
    return shapes


def _as_float32(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "float") and hasattr(value, "numpy"):
        value = value.float().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.ascontiguousarray(value, dtype=np.float32)


def validate_codec_weights(
    weights: Mapping[str, object],
    architecture: CodecArchitecture = VOICECHAT_CODEC,
) -> dict[str, np.ndarray]:
    """Validate and normalize an already-loaded codec-only state dictionary."""

    normalized: dict[str, np.ndarray] = {}
    for name, expected_shape in expected_weight_shapes(architecture).items():
        if name not in weights:
            raise KeyError(f"VoiceChat codec tensor not found: {name}")
        value = _as_float32(weights[name])
        if value.shape != expected_shape:
            raise ValueError(
                f"VoiceChat codec tensor {name} has shape {value.shape}; expected {expected_shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"VoiceChat codec tensor contains non-finite values: {name}")
        normalized[name] = value
    return normalized


def load_codec_weights(
    model_dir: str | Path,
    architecture: CodecArchitecture = VOICECHAT_CODEC,
) -> dict[str, np.ndarray]:
    """Load only ``tts_model.audio_codec`` tensors from HF safetensors.

    The checkpoint is about 42 GiB, so this deliberately reads the codec's
    tensors one at a time instead of materializing the full state dictionary.
    """

    # Import lazily so shape/contract tests do not require safetensors or torch.
    from .checkpoint_mapper import _load_tensor, _open_safetensors

    readers = _open_safetensors(Path(model_dir))
    loaded = {
        name: _load_tensor(readers, CHECKPOINT_PREFIX + name)
        for name in expected_weight_shapes(architecture)
    }
    return validate_codec_weights(loaded, architecture)


def dequantize_codes_numpy(
    codes: np.ndarray,
    weights: Mapping[str, np.ndarray],
    architecture: CodecArchitecture = VOICECHAT_CODEC,
) -> np.ndarray:
    """CPU reference for the exact additive RVQ lookup used by the TRT graph."""

    codes = np.asarray(codes)
    if codes.ndim != 2 or codes.shape[1] != architecture.num_quantizers:
        raise ValueError(
            "VoiceChat codec codes must have shape "
            f"[frames, {architecture.num_quantizers}], got {codes.shape}"
        )
    if not np.issubdtype(codes.dtype, np.integer):
        raise TypeError(f"VoiceChat codec codes must be integers, got {codes.dtype}")
    if codes.size and (int(codes.min()) < 0 or int(codes.max()) >= architecture.codebook_size):
        raise ValueError(f"VoiceChat codec codes must be in [0, {architecture.codebook_size})")

    latent = np.zeros((codes.shape[0], architecture.latent_size), dtype=np.float32)
    for quantizer in range(architecture.num_quantizers):
        table = np.asarray(weights[f"prvq.mus_list.{quantizer}"], dtype=np.float32)
        latent += table[codes[:, quantizer]]
    return latent


def _build_trt_graph(
    network,
    trt,
    weights: Mapping[str, np.ndarray],
    architecture: CodecArchitecture,
) -> None:
    """Populate ``network`` with the strongly-typed streaming codec graph."""

    def constant(values: np.ndarray, shape: tuple[int, ...]):
        values = np.ascontiguousarray(values, dtype=np.float32).reshape(shape)
        return network.add_constant(shape, trt.Weights(values)).get_output(0)

    def shuffle(tensor, shape: tuple[int, ...], permutation=None):
        layer = network.add_shuffle(tensor)
        if permutation is not None:
            layer.first_transpose = trt.Permutation(permutation)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def convolution_1d(
        tensor,
        weight: np.ndarray,
        bias: np.ndarray | None,
        out_channels: int,
        kernel_size: int,
        *,
        groups: int = 1,
    ):
        in_channels = int(tensor.shape[1])
        length = int(tensor.shape[2])
        tensor_4d = shuffle(tensor, (1, in_channels, 1, length))
        weight_4d = np.ascontiguousarray(
            weight.reshape(out_channels, in_channels // groups, 1, kernel_size),
            dtype=np.float32,
        )
        bias_weights = (
            trt.Weights(np.ascontiguousarray(bias, dtype=np.float32))
            if bias is not None
            else trt.Weights()
        )
        layer = network.add_convolution_nd(
            tensor_4d,
            num_output_maps=out_channels,
            kernel_shape=(1, kernel_size),
            kernel=trt.Weights(weight_4d),
            bias=bias_weights,
        )
        layer.num_groups = groups
        output_length = length - kernel_size + 1
        return shuffle(layer.get_output(0), (1, out_channels, output_length))

    def transpose_convolution_1d(
        tensor,
        weight: np.ndarray,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ):
        in_channels = int(tensor.shape[1])
        length = int(tensor.shape[2])
        tensor_4d = shuffle(tensor, (1, in_channels, 1, length))
        weight_4d = np.ascontiguousarray(
            weight.reshape(in_channels, out_channels, 1, kernel_size),
            dtype=np.float32,
        )
        layer = network.add_deconvolution_nd(
            tensor_4d,
            num_output_maps=out_channels,
            kernel_shape=(1, kernel_size),
            kernel=trt.Weights(weight_4d),
            bias=trt.Weights(),
        )
        layer.stride_nd = (1, stride)
        return shuffle(layer.get_output(0), (1, out_channels, length * stride))

    def layer_norm_channels(tensor, weight: np.ndarray, bias: np.ndarray):
        value = tensor
        mean = network.add_reduce(
            value, trt.ReduceOperation.AVG, 1 << 1, keep_dims=True
        ).get_output(0)
        centered = network.add_elementwise(value, mean, trt.ElementWiseOperation.SUB).get_output(0)
        square = network.add_elementwise(
            centered, centered, trt.ElementWiseOperation.PROD
        ).get_output(0)
        variance = network.add_reduce(
            square, trt.ReduceOperation.AVG, 1 << 1, keep_dims=True
        ).get_output(0)
        epsilon = constant(
            np.asarray([architecture.layer_norm_epsilon], dtype=np.float32),
            (1, 1, 1),
        )
        denominator = network.add_elementwise(
            variance, epsilon, trt.ElementWiseOperation.SUM
        ).get_output(0)
        reciprocal_std = network.add_unary(denominator, trt.UnaryOperation.SQRT).get_output(0)
        reciprocal_std = network.add_unary(reciprocal_std, trt.UnaryOperation.RECIP).get_output(0)
        value = network.add_elementwise(
            centered, reciprocal_std, trt.ElementWiseOperation.PROD
        ).get_output(0)
        scale = network.add_constant(
            (1, weight.size, 1),
            trt.Weights(np.ascontiguousarray(weight.reshape(1, -1, 1), dtype=np.float32)),
        ).get_output(0)
        shift = network.add_constant(
            (1, bias.size, 1),
            trt.Weights(np.ascontiguousarray(bias.reshape(1, -1, 1), dtype=np.float32)),
        ).get_output(0)
        value = network.add_elementwise(value, scale, trt.ElementWiseOperation.PROD).get_output(0)
        value = network.add_elementwise(value, shift, trt.ElementWiseOperation.SUM).get_output(0)
        return value

    def exact_gelu(tensor):
        # nn.GELU() in the public NeMo codec uses approximate="none".
        value = tensor
        inv_sqrt_two = network.add_constant(
            (1, 1, 1),
            trt.Weights(np.asarray([1.0 / np.sqrt(2.0)], dtype=np.float32)),
        ).get_output(0)
        half = network.add_constant(
            (1, 1, 1), trt.Weights(np.asarray([0.5], dtype=np.float32))
        ).get_output(0)
        one = network.add_constant(
            (1, 1, 1), trt.Weights(np.asarray([1.0], dtype=np.float32))
        ).get_output(0)
        scaled = network.add_elementwise(
            value, inv_sqrt_two, trt.ElementWiseOperation.PROD
        ).get_output(0)
        erf = network.add_unary(scaled, trt.UnaryOperation.ERF).get_output(0)
        shifted = network.add_elementwise(erf, one, trt.ElementWiseOperation.SUM).get_output(0)
        result = network.add_elementwise(value, shifted, trt.ElementWiseOperation.PROD).get_output(
            0
        )
        result = network.add_elementwise(result, half, trt.ElementWiseOperation.PROD).get_output(0)
        return result

    def convnext_block(tensor, stem: str, block: int):
        channels = int(tensor.shape[1])
        length = int(tensor.shape[2])
        cache = network.add_input(
            cache_input_name(block),
            trt.float32,
            (1, channels, CAUSAL_CACHE_WIDTH),
        )
        padded = network.add_concatenation([cache, tensor])
        padded.axis = 2
        padded_tensor = padded.get_output(0)

        new_cache = network.add_slice(
            padded_tensor,
            start=(0, 0, length),
            shape=(1, channels, CAUSAL_CACHE_WIDTH),
            stride=(1, 1, 1),
        ).get_output(0)
        new_cache.name = cache_output_name(block)
        network.mark_output(new_cache)

        residual = tensor
        value = convolution_1d(
            padded_tensor,
            weights[f"{stem}.dwconv.weight"],
            weights[f"{stem}.dwconv.bias"],
            channels,
            architecture.convnext_kernel_size,
            groups=channels,
        )
        value = layer_norm_channels(
            value,
            weights[f"{stem}.norm.weight"],
            weights[f"{stem}.norm.bias"],
        )
        intermediate = channels * 4
        value = convolution_1d(
            value,
            weights[f"{stem}.pwconv1.weight"],
            weights[f"{stem}.pwconv1.bias"],
            intermediate,
            1,
        )
        value = exact_gelu(value)
        value = convolution_1d(
            value,
            weights[f"{stem}.pwconv2.weight"],
            weights[f"{stem}.pwconv2.bias"],
            channels,
            1,
        )
        return network.add_elementwise(residual, value, trt.ElementWiseOperation.SUM).get_output(0)

    codes = network.add_input(
        CODE_INPUT_NAME,
        trt.int32,
        (1, architecture.num_quantizers),
    )

    # Residual VQ dequantization: sum 31 independent embedding lookups.
    latent = None
    for quantizer in range(architecture.num_quantizers):
        table = constant(
            weights[f"prvq.mus_list.{quantizer}"],
            (architecture.codebook_size, architecture.latent_size),
        )
        index_slice = network.add_slice(
            codes,
            start=(0, quantizer),
            shape=(1, 1),
            stride=(1, 1),
        ).get_output(0)
        indices = shuffle(index_slice, (1,))
        selected = network.add_gather(table, indices, axis=0).get_output(0)
        latent = (
            selected
            if latent is None
            else network.add_elementwise(latent, selected, trt.ElementWiseOperation.SUM).get_output(
                0
            )
        )

    # [frames, latent] -> [1, latent, frames]
    value = shuffle(latent, (1, 1, architecture.latent_size))
    value = shuffle(
        value,
        (1, architecture.latent_size, 1),
        permutation=(0, 2, 1),
    )

    layer = 0
    block = 0
    for multiplier, rate in zip(architecture.channel_mult, architecture.rates):
        channels = architecture.base_hidden_size * multiplier
        value = transpose_convolution_1d(
            value,
            weights[f"decoder.layers.{layer}.weight"],
            channels,
            rate,
            rate,
        )
        layer += 1
        for _ in range(architecture.num_blocks):
            value = convnext_block(value, f"decoder.layers.{layer}", block)
            layer += 1
            block += 1

    value = convolution_1d(
        value,
        weights[f"decoder.layers.{layer}.weight"],
        None,
        architecture.spectral_channels,
        1,
    )
    value = shuffle(
        value, (architecture.spectral_channels, architecture.spectral_frames_per_codec_frame)
    )
    value.name = SPECTRAL_OUTPUT_NAME
    network.mark_output(value)


def build_codec_engine(
    weights: Mapping[str, object],
    *,
    verbose: bool = False,
    architecture: CodecArchitecture = VOICECHAT_CODEC,
) -> bytes:
    """Build the single-frame TensorRT Native API RVQ/decoder plan."""
    architecture.validate()
    normalized = validate_codec_weights(weights, architecture)

    # Keep TensorRT optional at import time; only native engine construction
    # requires it.
    from tensorrt_model_connect import trt_compat

    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    # NeMo's RVQVAEModel.decode enters disable_tf32(); preserve that
    # numerics boundary for model-card parity.
    config.clear_flag(trt.BuilderFlag.TF32)
    _build_trt_graph(
        network,
        trt,
        normalized,
        architecture,
    )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the VoiceChat codec engine")
    return bytes(plan)


def build_codec_engine_from_checkpoint(
    model_dir: str | Path,
    *,
    verbose: bool = False,
    architecture: CodecArchitecture = VOICECHAT_CODEC,
) -> bytes:
    """Load exact checkpoint codec weights and build the native decoder plan."""

    weights = load_codec_weights(model_dir, architecture)
    return build_codec_engine(weights, verbose=verbose, architecture=architecture)


__all__ = [
    "CAUSAL_CACHE_WIDTH",
    "CHECKPOINT_PREFIX",
    "CODEC_ENGINE_SECTION",
    "CODE_INPUT_NAME",
    "CodecArchitecture",
    "SPECTRAL_OUTPUT_NAME",
    "VOICECHAT_CODEC",
    "build_codec_engine",
    "build_codec_engine_from_checkpoint",
    "cache_input_name",
    "cache_output_name",
    "dequantize_codes_numpy",
    "expected_weight_shapes",
    "load_codec_weights",
    "validate_codec_weights",
]
