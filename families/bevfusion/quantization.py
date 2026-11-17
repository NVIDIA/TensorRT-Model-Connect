# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BEV-owned FP8 convolution weights and calibrated, static activation Q/DQ."""

import json
from pathlib import Path

import numpy as np


SCALES_FILE = "fp8_activation_scales.json"
FP8_MAX = 448.0
MIN_SCALE = 2.0**-14
CONVOLUTIONS = (
    "fuser.0",
    *(f"decoder.backbone.blocks.{stage}.{3 * index}" for stage in range(2) for index in range(6)),
    "decoder.neck.deblocks.0.0",
    "heads.map.classifier.0",
    "heads.map.classifier.3",
)
# The fuser/backbone are sensitive on tuning seeds 16..19. The neck and the
# two large head convolutions retain an error margin before held-out validation.
DEFAULT_EXCLUSIONS = tuple(
    name for name in CONVOLUTIONS if name == "fuser.0" or name.startswith("decoder.backbone.")
)


def checkpoint_key(values):
    """Select calibrated weights using numeric weight summaries, as for DenseNet."""
    names = ("fuser.0.weight", "heads.map.classifier.6.weight")
    sums = [np.abs(values[name].detach().float().numpy()).sum(dtype=np.float64) for name in names]
    return "bevfusion-bev128-" + "-".join(f"{value:.8e}" for value in sums)


def load_scales(checkpoint, values):
    """Require a complete scale table for this checkpoint; null explicitly retains FP16."""
    path = Path(checkpoint).parent / SCALES_FILE
    if not path.is_file():
        path = Path(__file__).with_name(SCALES_FILE)
    try:
        document = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError("BEVFusion FP8 requires calibrate_fp8_scales.py output") from error
    if (
        not isinstance(document, dict)
        or document.get("version") != 1
        or document.get("checkpoint_key") != checkpoint_key(values)
    ):
        raise ValueError("BEVFusion FP8 scales do not match the checkpoint; recalibrate it")
    scales = document.get("layers", {})
    if not isinstance(scales, dict) or set(scales) != set(CONVOLUTIONS):
        raise ValueError("BEVFusion FP8 scales must cover every selected convolution exactly")
    if not isinstance(document.get("calibration"), dict):
        raise ValueError("BEVFusion FP8 scales require calibration provenance")
    for name, scale in scales.items():
        if scale is not None and (
            type(scale) not in (int, float)
            or not np.isfinite(scale)
            or scale < MIN_SCALE
            or scale > float(np.finfo(np.float16).max)
        ):
            raise ValueError(
                f"BEVFusion FP8 scale of {name} must be finite and representable in FP16"
            )
    if not any(value is not None for value in scales.values()):
        raise ValueError("BEVFusion FP8 recipe must quantize at least one convolution")
    return scales, document["calibration"]


def quantize_weight(weight):
    """Pack E4M3 weights with one FP16-representable scale per output channel."""
    import ml_dtypes

    weight = np.asarray(weight, dtype=np.float32)
    if weight.ndim != 4 or not all(weight.shape) or not np.isfinite(weight).all():
        raise ValueError("BEVFusion FP8 requires a finite OIHW convolution weight")
    amax = np.abs(weight).max(axis=(1, 2, 3))
    scale = np.maximum(amax / FP8_MAX, MIN_SCALE).astype(np.float16)
    if not np.isfinite(scale).all():
        raise ValueError("BEVFusion FP8 weight scales overflow FP16")
    normalized = np.clip(weight / scale.astype(np.float32)[:, None, None, None], -FP8_MAX, FP8_MAX)
    packed = normalized.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return np.ascontiguousarray(packed), scale


def add_convolution(graph, value, weight, input_scale):
    """Add activation Q/DQ and packed-weight DQ; TensorRT owns FP8 lowering."""
    trt, network = graph.trt, graph.network
    if value.dtype != trt.float16:
        raise ValueError("BEVFusion FP8 convolutions require FP16 Q/DQ boundaries")

    def constant(values):
        array = graph.keep(np.asarray(values, dtype=np.float16).reshape(-1))
        return network.add_constant(array.shape, array).get_output(0)

    activation_scale = constant([input_scale])
    quantized = network.add_quantize(value, activation_scale, trt.DataType.FP8).get_output(0)
    activation = network.add_dequantize(quantized, activation_scale, trt.float16).get_output(0)
    packed, scale = quantize_weight(weight)
    packed = graph.keep(packed)
    stored = network.add_constant(
        packed.shape, trt.Weights(trt.DataType.FP8, packed.ctypes.data, packed.size)
    ).get_output(0)
    dequantized = network.add_dequantize(stored, constant(scale), trt.float16)
    dequantized.axis = 0
    layer = network.add_convolution_nd(
        activation, int(weight.shape[0]), tuple(weight.shape[2:]), trt.Weights(), trt.Weights()
    )
    layer.set_input(1, dequantized.get_output(0))
    return layer
