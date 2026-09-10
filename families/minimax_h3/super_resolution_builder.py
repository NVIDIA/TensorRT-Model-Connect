# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT video super-resolution plan for MiniMax-H3.

The build-time source is the public Real-ESRGAN ``realesr-general-x4v3``
SRVGGNetCompact checkpoint and its weak-denoise companion.  Checkpoint
loading and denoise-strength interpolation happen only while building the
bundle.  The serialized runtime graph contains TensorRT-native convolution,
PReLU, shuffle, resize, elementwise, and cast layers; it has no framework,
plugin, or custom-CUDA dependency.

The released network first reconstructs at 4x around its nearest-neighbor
base, as SRVGGNetCompact specifies.  The base remains unscaled while the
learned detail residual defaults to a conservative 0.25 strength to protect
video identity and temporal continuity.  A native cubic resize then produces
the aspect-preserving 1296x720 delivery canvas.
"""

from __future__ import annotations

from collections.abc import Mapping
import gc
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

from . import trt_compat

from . import graph_ops as op


trt = trt_compat.get_trt()

SUPER_RESOLUTION_PLAN_FILENAME = "video_super_resolution.plan"
INPUT_NAME = "frames"
OUTPUT_NAME = "upscaled_frames"

SOURCE_HEIGHT = 480
SOURCE_WIDTH = 864
TARGET_HEIGHT = 720
TARGET_WIDTH = 1296
CHANNELS = 3

BATCH_MIN = 1
BATCH_OPT = 4
BATCH_MAX = 8

MODEL_UPSCALE = 4
FEATURE_CHANNELS = 64
NUM_BODY_CONVOLUTIONS = 32
DEFAULT_DENOISE_STRENGTH = 0.5
DEFAULT_LEARNED_RESIDUAL_STRENGTH = 0.25

GENERAL_CHECKPOINT_FILENAME = "realesr-general-x4v3.pth"
WEAK_DENOISE_CHECKPOINT_FILENAME = "realesr-general-wdn-x4v3.pth"
GENERAL_CHECKPOINT_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
)
WEAK_DENOISE_CHECKPOINT_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth"
)


def _validate_denoise_strength(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("MiniMax-H3 super-resolution denoise_strength must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(
            "MiniMax-H3 super-resolution denoise_strength must be finite and in [0, 1]"
        )
    return result


def _validate_learned_residual_strength(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("MiniMax-H3 super-resolution learned_residual_strength must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(
            "MiniMax-H3 super-resolution learned_residual_strength must be finite and in [0, 1]"
        )
    return result


def checkpoint_shapes() -> dict[str, tuple[int, ...]]:
    """Return the exact public SRVGGNetCompact state-dict contract."""

    shapes: dict[str, tuple[int, ...]] = {
        "body.0.weight": (FEATURE_CHANNELS, CHANNELS, 3, 3),
        "body.0.bias": (FEATURE_CHANNELS,),
    }
    # The public architecture has an initial Conv/PReLU pair followed by
    # 32 more Conv/PReLU pairs.  PyTorch ModuleList indices therefore
    # alternate even convolutions and odd PReLUs through body.65.
    for body_index in range(2, 2 * (NUM_BODY_CONVOLUTIONS + 1), 2):
        shapes[f"body.{body_index}.weight"] = (
            FEATURE_CHANNELS,
            FEATURE_CHANNELS,
            3,
            3,
        )
        shapes[f"body.{body_index}.bias"] = (FEATURE_CHANNELS,)
    for body_index in range(1, 2 * (NUM_BODY_CONVOLUTIONS + 1), 2):
        shapes[f"body.{body_index}.weight"] = (FEATURE_CHANNELS,)
    output_index = 2 * (NUM_BODY_CONVOLUTIONS + 1)
    shapes[f"body.{output_index}.weight"] = (
        CHANNELS * MODEL_UPSCALE * MODEL_UPSCALE,
        FEATURE_CHANNELS,
        3,
        3,
    )
    shapes[f"body.{output_index}.bias"] = (CHANNELS * MODEL_UPSCALE * MODEL_UPSCALE,)
    return shapes


def checkpoint_keys() -> tuple[str, ...]:
    """Return the exhaustive SRVGGNetCompact tensor names."""

    return tuple(checkpoint_shapes())


def super_resolution_metadata(
    *,
    denoise_strength: float = DEFAULT_DENOISE_STRENGTH,
    learned_residual_strength: float = DEFAULT_LEARNED_RESIDUAL_STRENGTH,
) -> dict[str, object]:
    """Return public, path-free metadata for the serialized plan."""

    strength = _validate_denoise_strength(denoise_strength)
    detail_strength = _validate_learned_residual_strength(learned_residual_strength)
    return {
        "schema_version": 1,
        "architecture": "SRVGGNetCompact",
        "source_models": [
            GENERAL_CHECKPOINT_FILENAME,
            *([WEAK_DENOISE_CHECKPOINT_FILENAME] if strength != 1.0 else []),
        ],
        "denoise_strength": strength,
        "learned_residual_strength": detail_strength,
        "model_upscale": MODEL_UPSCALE,
        "source_shape": [SOURCE_HEIGHT, SOURCE_WIDTH],
        "target_shape": [TARGET_HEIGHT, TARGET_WIDTH],
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "batch_profile": [BATCH_MIN, BATCH_OPT, BATCH_MAX],
        "compute_precision": "fp16",
        "io_dtype": "float32",
        "layout": "NHWC",
        "implementation": "tensorrt_native",
        "runtime_framework": None,
    }


def _checkpoint_state(payload: Any, path: Path) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Real-ESRGAN checkpoint {path} is not a state-dict mapping")
    for key in ("params_ema", "params"):
        state = payload.get(key)
        if isinstance(state, Mapping):
            return state
    # Accept a raw state dict while rejecting unrelated checkpoint metadata.
    if payload and all(isinstance(name, str) and name.startswith("body.") for name in payload):
        return payload
    raise ValueError(f"Real-ESRGAN checkpoint {path} has no params or params_ema state dict")


def _numpy_tensor(value: Any, *, name: str, path: Path) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Real-ESRGAN tensor {name!r} in {path} is not numeric") from error
    return np.ascontiguousarray(array)


def _validated_state(payload: Any, path: Path) -> dict[str, np.ndarray]:
    state = _checkpoint_state(payload, path)
    expected = checkpoint_shapes()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(
            "Real-ESRGAN SRVGGNetCompact checkpoint mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    result: dict[str, np.ndarray] = {}
    for name, shape in expected.items():
        value = _numpy_tensor(state[name], name=name, path=path)
        if value.shape != shape:
            raise ValueError(
                f"Real-ESRGAN tensor {name!r} has shape {value.shape}, expected {shape}"
            )
        result[name] = value
    return result


def load_super_resolution_weights(
    primary_checkpoint: str | Path,
    weak_denoise_checkpoint: str | Path | None = None,
    *,
    denoise_strength: float = DEFAULT_DENOISE_STRENGTH,
) -> dict[str, np.ndarray]:
    """Load and interpolate the two official Real-ESRGAN checkpoints.

    ``denoise_strength=1`` selects the primary checkpoint exactly.  Other
    values follow Real-ESRGAN's official dynamic-network-interpolation order:
    ``primary * strength + weak_denoise * (1 - strength)``.
    """

    strength = _validate_denoise_strength(denoise_strength)
    primary_path = Path(primary_checkpoint)
    if not primary_path.is_file():
        raise ValueError(f"Real-ESRGAN primary checkpoint does not exist: {primary_path}")
    if weak_denoise_checkpoint is None and strength != 1.0:
        raise ValueError(
            "Real-ESRGAN weak-denoise checkpoint is required when denoise_strength != 1"
        )

    # PyTorch is a build-only checkpoint decoder for the public .pth files.
    # It is absent from the serialized engine and native C++ runtime path.
    import torch

    primary = _validated_state(
        torch.load(primary_path, map_location="cpu", weights_only=True), primary_path
    )
    if strength == 1.0:
        return primary

    weak_path = Path(weak_denoise_checkpoint)
    if not weak_path.is_file():
        raise ValueError(f"Real-ESRGAN weak-denoise checkpoint does not exist: {weak_path}")
    weak = _validated_state(torch.load(weak_path, map_location="cpu", weights_only=True), weak_path)
    inverse = 1.0 - strength
    return {
        name: np.ascontiguousarray(primary[name] * strength + weak[name] * inverse)
        for name in checkpoint_keys()
    }


def _validate_weight_dict(weights: Mapping[str, np.ndarray]) -> None:
    expected = checkpoint_shapes()
    missing = sorted(set(expected) - set(weights))
    unexpected = sorted(set(weights) - set(expected))
    if missing or unexpected:
        raise ValueError(
            "MiniMax-H3 super-resolution checkpoint partition mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    for name, shape in expected.items():
        if np.asarray(weights[name]).shape != shape:
            raise ValueError(
                f"MiniMax-H3 super-resolution tensor {name!r} has shape "
                f"{np.asarray(weights[name]).shape}, expected {shape}"
            )


def _conv3x3(network, hidden, weights: Mapping[str, np.ndarray], body_index: int):
    prefix = f"body.{body_index}"
    kernel = weights[f"{prefix}.weight"]
    bias = weights[f"{prefix}.bias"]
    layer = network.add_convolution_nd(
        hidden,
        int(kernel.shape[0]),
        (3, 3),
        kernel,
        bias,
    )
    if layer is None:
        raise RuntimeError(f"TensorRT rejected MiniMax-H3 super-resolution {prefix}")
    layer.name = f"video_super_resolution.{prefix}"
    layer.padding_nd = (1, 1)
    return layer.get_output(0)


def _prelu(network, hidden, weights: Mapping[str, np.ndarray], body_index: int):
    prefix = f"body.{body_index}"
    slopes = op.weight_constant(
        network,
        np.asarray(weights[f"{prefix}.weight"]).reshape(1, FEATURE_CHANNELS, 1, 1),
    )
    slopes = op.cast(network, slopes, hidden.dtype)
    layer = network.add_parametric_relu(hidden, slopes)
    if layer is None:
        raise RuntimeError(f"TensorRT rejected MiniMax-H3 super-resolution {prefix}")
    layer.name = f"video_super_resolution.{prefix}"
    return layer.get_output(0)


def _pixel_shuffle_x4(network, hidden):
    channels = CHANNELS * MODEL_UPSCALE * MODEL_UPSCALE
    if tuple(hidden.shape)[1:] != (channels, SOURCE_HEIGHT, SOURCE_WIDTH):
        raise RuntimeError(
            "MiniMax-H3 super-resolution PixelShuffle received invalid static geometry"
        )
    split = network.add_shuffle(hidden)
    split.name = "video_super_resolution.pixel_shuffle.split"
    split.reshape_dims = (
        -1,
        CHANNELS,
        MODEL_UPSCALE,
        MODEL_UPSCALE,
        SOURCE_HEIGHT,
        SOURCE_WIDTH,
    )
    interleave = network.add_shuffle(split.get_output(0))
    interleave.name = "video_super_resolution.pixel_shuffle.interleave"
    interleave.first_transpose = (0, 1, 4, 2, 5, 3)
    interleave.reshape_dims = (
        -1,
        CHANNELS,
        SOURCE_HEIGHT * MODEL_UPSCALE,
        SOURCE_WIDTH * MODEL_UPSCALE,
    )
    return interleave.get_output(0)


@op.cleanup_failed_build
def build_super_resolution_engine_from_weights(
    weights: dict[str, np.ndarray],
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    learned_residual_strength: float = DEFAULT_LEARNED_RESIDUAL_STRENGTH,
    output_path: str | Path | None = None,
) -> bytes | dict[str, int | str]:
    """Build the native FP16-compute, FP32-NHWC-I/O SR plan."""

    _validate_weight_dict(weights)
    detail_strength = _validate_learned_residual_strength(learned_residual_strength)
    # Retain one compact FP16 copy for the complete TensorRT build.  Keeping
    # these arrays alive is required because convolution weights are supplied
    # to the network by pointer.
    fp16_weights = {
        name: np.ascontiguousarray(np.asarray(value), dtype=np.float16)
        for name, value in weights.items()
    }

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    op.configure_builder(config, weight_streaming=weight_streaming)
    if workspace_bytes is not None:
        op.configure_workspace(config, workspace_bytes, default_bytes=32 << 30)

    frames = network.add_input(
        INPUT_NAME,
        trt.float32,
        (-1, SOURCE_HEIGHT, SOURCE_WIDTH, CHANNELS),
    )
    profile = builder.create_optimization_profile()
    profile.set_shape(
        INPUT_NAME,
        (BATCH_MIN, SOURCE_HEIGHT, SOURCE_WIDTH, CHANNELS),
        (BATCH_OPT, SOURCE_HEIGHT, SOURCE_WIDTH, CHANNELS),
        (BATCH_MAX, SOURCE_HEIGHT, SOURCE_WIDTH, CHANNELS),
    )
    config.add_optimization_profile(profile)

    to_nchw = network.add_shuffle(frames)
    to_nchw.name = "video_super_resolution.nhwc_to_nchw"
    to_nchw.first_transpose = (0, 3, 1, 2)
    source = op.cast(network, to_nchw.get_output(0), trt.float16)

    hidden = source
    for convolution_index in range(NUM_BODY_CONVOLUTIONS + 1):
        body_index = 2 * convolution_index
        hidden = _conv3x3(network, hidden, fp16_weights, body_index)
        hidden = _prelu(network, hidden, fp16_weights, body_index + 1)
    output_index = 2 * (NUM_BODY_CONVOLUTIONS + 1)
    hidden = _conv3x3(network, hidden, fp16_weights, output_index)
    learned_x4 = _pixel_shuffle_x4(network, hidden)

    learned_scale = op.constant(
        network,
        np.full((1, 1, 1, 1), detail_strength, dtype=np.float16),
        dtype=np.float16,
    )
    learned_x4 = network.add_elementwise(
        learned_x4,
        learned_scale,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)

    residual_resize = network.add_resize(source)
    residual_resize.name = "video_super_resolution.nearest_residual"
    residual_resize.resize_mode = trt.InterpolationMode.NEAREST
    residual_resize.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC
    residual_resize.scales = (1.0, 1.0, float(MODEL_UPSCALE), float(MODEL_UPSCALE))
    reconstructed = network.add_elementwise(
        learned_x4,
        residual_resize.get_output(0),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)

    delivery_resize = network.add_resize(reconstructed)
    delivery_resize.name = "video_super_resolution.cubic_delivery_resize"
    delivery_resize.resize_mode = trt.InterpolationMode.CUBIC
    delivery_resize.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
    delivery_resize.exclude_outside = 1
    delivery_resize.cubic_coeff = -0.75
    delivery_resize.scales = (
        1.0,
        1.0,
        TARGET_HEIGHT / (SOURCE_HEIGHT * MODEL_UPSCALE),
        TARGET_WIDTH / (SOURCE_WIDTH * MODEL_UPSCALE),
    )
    delivered = delivery_resize.get_output(0)

    zero = op.constant(network, np.zeros((1, 1, 1, 1), dtype=np.float16), dtype=np.float16)
    one = op.constant(network, np.ones((1, 1, 1, 1), dtype=np.float16), dtype=np.float16)
    delivered = network.add_elementwise(delivered, zero, trt.ElementWiseOperation.MAX).get_output(0)
    delivered = network.add_elementwise(delivered, one, trt.ElementWiseOperation.MIN).get_output(0)
    delivered = op.cast(network, delivered, trt.float32)

    to_nhwc = network.add_shuffle(delivered)
    to_nhwc.name = "video_super_resolution.nchw_to_nhwc"
    to_nhwc.first_transpose = (0, 2, 3, 1)
    output = to_nhwc.get_output(0)
    output.name = OUTPUT_NAME
    network.mark_output(output)

    op.validate_native_network(network, expected_attentions=0, label="video super-resolution")
    print(
        "[minimax-h3] building native video super-resolution: "
        f"batch={BATCH_MIN}/{BATCH_OPT}/{BATCH_MAX}, "
        f"{SOURCE_WIDTH}x{SOURCE_HEIGHT} -> {TARGET_WIDTH}x{TARGET_HEIGHT}, "
        f"learned_residual_strength={detail_strength:g}, fp16",
        file=sys.stderr,
    )

    plan = None
    record = None
    try:
        if output_path is None:
            plan = builder.build_serialized_network(network, config)
        else:
            record = trt_compat.build_serialized_network_to_file(
                builder, network, config, output_path
            )
    finally:
        op.release_weight_buffers(network)
        if consume_weights:
            weights.clear()
    if output_path is None and plan is None:
        raise RuntimeError("TensorRT failed to build MiniMax-H3 video super-resolution engine")
    del fp16_weights, network, config, builder
    gc.collect()
    return record if record is not None else bytes(plan)


def build_super_resolution_engine(
    primary_checkpoint: str | Path,
    weak_denoise_checkpoint: str | Path | None = None,
    *,
    denoise_strength: float = DEFAULT_DENOISE_STRENGTH,
    verbose: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    learned_residual_strength: float = DEFAULT_LEARNED_RESIDUAL_STRENGTH,
    output_path: str | Path | None = None,
) -> bytes | dict[str, int | str]:
    """Load the public checkpoints and build the native SR plan."""

    weights = load_super_resolution_weights(
        primary_checkpoint,
        weak_denoise_checkpoint,
        denoise_strength=denoise_strength,
    )
    return build_super_resolution_engine_from_weights(
        weights,
        verbose=verbose,
        consume_weights=True,
        workspace_bytes=workspace_bytes,
        weight_streaming=weight_streaming,
        learned_residual_strength=learned_residual_strength,
        output_path=output_path,
    )
