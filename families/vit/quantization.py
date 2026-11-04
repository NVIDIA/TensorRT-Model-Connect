# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ViT-owned FP8 TensorRT Q/DQ graph context.

FP8 (``quantization="fp8"``) is self-quantization of the original checkpoint, because
no ModelOpt FP8 checkpoint exists for the transformers ViT. Weights are quantized at build
time, one projection at a time: ``scale = max(abs(weight)) / 448`` (448 is the largest
finite e4m3 value), and the FP8 bytes are fed to the engine as a constant. The activation
scale of each projection input needs real data, so it was calibrated once offline with
``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``) and is read from
``fp8_activation_scales.json``, one table per model config. Users do not run calibration.
FP8 weights are never stored.

A model config with no table, or a table that does not cover every projection, is an error
that names the calibration tool: there is no silent fallback. The one exception is an
outlier projection, whose scale is stored as ``null`` on purpose and which stays in the
work precision.

Each quantized projection becomes ``activation -> Quantize -> Dequantize -> MatMul`` with
``FP8 weight -> Dequantize`` on the other input. The Dequantize layers and the matmul use
the type of the activation, so the matmul output has the same type and no cast is added.
The family builds float32 engines, so they run in float32.

Quantized projections of every encoder layer: ``query``, ``key``, ``value``, the attention
output ``attention.output.dense``, ``intermediate.dense`` and ``output.dense``. The patch
convolution, the embeddings, the layer norms, the attention matmuls and the classifier stay
in the work precision. Weights keep the checkpoint's ``[out_features, in_features]`` layout,
so the matmul transposes the weight operand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
import tensorrt as trt

from . import graph

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"
CALIBRATION_TOOL = "python -m families.vit.calibrate_fp8_scales"

_SUPPORTED = ("fp8",)
LAYER_PROJECTIONS = (
    "attention.attention.query",
    "attention.attention.key",
    "attention.attention.value",
    "attention.output.dense",
    "intermediate.dense",
    "output.dense",
)


@dataclass(frozen=True)
class _FP8Weight:
    """One projection weight stored as raw fp8_e4m3 bytes with one per-tensor scale."""

    packed: np.ndarray  # [out_features, in_features] raw fp8_e4m3 bytes
    weight_scale: float
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _LayerScales:
    """Activation scale and the packed weight of one quantized projection."""

    input_scale: float
    weight: _FP8Weight


def _scale_const(network, value: float, dtype: trt.DataType):
    """One-element constant holding a Q/DQ scale in the Dequantize output type."""
    np_dtype = np.float16 if dtype == trt.float16 else np.float32
    return graph.add_constant(network, (1,), np.asarray([value], dtype=np_dtype), dtype=np_dtype)


def _wrap_fp8_matmul(network, activation, scales: _LayerScales, *, keep_alive: list):
    """Add ``activation @ weight.T`` with an FP8 weight and a static FP8 activation scale.

    With an fp16 activation TensorRT's FP8 GEMM keeps the raw accumulator in fp16 and
    applies the scales after, which overflows once an output exceeds about
    ``65504 * input_scale * weight_scale``. The float32 engines this family builds have no
    such limit.
    """
    dtype = activation.dtype
    if dtype not in (trt.float16, trt.float32):
        raise ValueError(f"ViT FP8 supports fp16 or fp32 activations, got {dtype}")
    w = scales.weight
    n, k = w.out_features, w.in_features

    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(w.packed)
    w_fp8 = trt.Weights(trt.DataType.FP8, w.packed.ctypes.data, n * k)
    w_const = network.add_constant((n, k), w_fp8).get_output(0)
    w_dequant = network.add_dequantize(w_const, _scale_const(network, w.weight_scale, dtype), dtype)

    act_scale = _scale_const(network, scales.input_scale, dtype)
    a_quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
    a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, dtype)

    return network.add_matrix_multiply(
        a_dequant.get_output(0),
        trt.MatrixOperation.NONE,
        w_dequant.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)


@dataclass(frozen=True)
class VitQuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    keep_alive: list = None  # type: ignore[assignment]

    @property
    def has_fp8_weights(self) -> bool:
        """True when any projection is FP8 (selects the FP8 build options)."""
        return bool(self.scales)

    def __post_init__(self):
        """Create the list that keeps weight buffers alive during the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, name: str) -> bool:
        """True when ``name`` (for example ``vit.encoder.layer.3.output.dense``) is quantized."""
        return name in self.scales

    def matmul(self, name: str) -> Any:
        """The FP8 matmul callable for ``name`` (``(network, input) -> output``), or None."""
        entry = self.scales.get(name)
        if entry is None:
            return None

        def run(network, activation):
            return _wrap_fp8_matmul(network, activation, entry, keep_alive=self.keep_alive)

        return run


def scale_key(cfg: dict) -> str:
    """Key of this model's entry in the activation-scale file, from the resolved config.

    Hidden size, layers, MLP size, patch size and input size are the values that change
    the activation ranges.
    """
    return (
        f"vit-h{cfg['hidden_size']}-l{cfg['num_layers']}-i{cfg['intermediate_size']}"
        f"-p{cfg['patch_size']}-s{cfg['image_size']}"
    )


def quantize_fp8_weight(weight_fp32: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize one weight to FP8 e4m3 with a single scalar scale.

    Returns the raw fp8 bytes and ``scale = max(abs(weight)) / 448``; the same scale is
    the dequantize scale in the engine.
    """
    amax = float(np.abs(weight_fp32).max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scale = amax / FP8_E4M3_MAX
    scaled = np.clip(weight_fp32 / scale, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return np.ascontiguousarray(packed), scale


def load_activation_scales(cfg: dict) -> dict[str, float | None]:
    """Return the committed activation scales for this model config, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(
            f"ViT FP8 build requires {path}, which was not found. "
            f"Run `{CALIBRATION_TOOL} calibrate` to create it."
        ) from error
    key = scale_key(cfg)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for model config {key!r}; available: "
            f"{sorted(table)}. Run `{CALIBRATION_TOOL} calibrate` for this model."
        )
    return table[key]


def projections(cfg: dict) -> list[str]:
    """Checkpoint names (without the ``.weight`` suffix) of every quantizable projection."""
    prefix = cfg["prefix"]
    return [
        f"{prefix}encoder.layer.{layer}.{stem}"
        for layer in range(cfg["num_layers"])
        for stem in LAYER_PROJECTIONS
    ]


def calibrate_vit(weights: dict, cfg: dict, quantization: str) -> VitQuantContext:
    """Quantize the checkpoint one projection at a time and pair it with the stored scales."""
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"ViT supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    activation_scales = load_activation_scales(cfg)
    names = projections(cfg)
    missing = [name for name in names if name not in activation_scales]
    unused = sorted(set(activation_scales) - set(names))
    if missing or unused:
        raise ValueError(
            f"FP8 scale table {scale_key(cfg)!r} does not match the model: "
            f"{len(missing)} projections without an entry (first: {missing[:3]}), "
            f"{len(unused)} entries without a projection (first: {unused[:3]}). "
            f"Re-run `{CALIBRATION_TOOL} calibrate` for this model."
        )
    scales: dict[str, _LayerScales] = {}
    for name in names:
        if f"{name}.weight_scale" in weights:
            raise ValueError(
                f"{name} is already quantized; fp8 self-quantization needs the original checkpoint"
            )
        input_scale = activation_scales[name]
        if input_scale is None:
            continue  # outlier projection, kept in the work precision on purpose
        input_scale = float(input_scale)
        if not np.isfinite(input_scale) or input_scale <= 0:
            raise ValueError(f"FP8 activation scale of {name!r} must be finite and positive")
        key = f"{name}.weight"
        if key not in weights:
            raise ValueError(f"checkpoint has no {key}")
        weight = np.asarray(weights[key], dtype=np.float32)
        packed, weight_scale = quantize_fp8_weight(weight)
        out_features, in_features = weight.shape
        scales[name] = _LayerScales(
            input_scale, _FP8Weight(packed, weight_scale, out_features, in_features)
        )
    if not scales:
        raise RuntimeError("ViT FP8 self-quantization matched no projection in the checkpoint")
    return VitQuantContext(scales=scales)
