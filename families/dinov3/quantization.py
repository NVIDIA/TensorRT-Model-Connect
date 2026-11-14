# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3-owned FP8 TensorRT Q/DQ graph context.

FP8 (``quantization="fp8"``) is self-quantization of the original checkpoint, because
no ModelOpt FP8 checkpoint exists for DINOv3. Weights are quantized at build time, one
projection at a time: ``scale = max(abs(weight)) / 448`` (448 is the largest finite
e4m3 value), and the FP8 bytes are fed to the engine as a constant. The activation scale
of each projection input needs real data, so it was calibrated once offline with
``calibrate_fp8_scales.py`` and is read from ``fp8_activation_scales.json``, one table per
model shape and resolution. Users do not run calibration. A shape with no table, or a
projection with no entry in the table (or an entry with no projection), is an error: there
is no silent fallback to an unquantized projection. Each quantized projection becomes
``activation -> Quantize -> Dequantize -> MatMul`` with ``FP8 weight -> Dequantize`` on the
other input.

The Dequantize layers and the matmul run in the work precision of the engine. In an fp16
engine the matmul output is fp16, so TensorRT writes it from the FP8 GEMM epilogue and
fuses the bias add and the GELU into it. In an fp32 engine they are fp32.

ViT-L and ViT-H+ cannot use an fp16 engine: their residual stream reaches about 1.6e5,
above the fp16 maximum 65504, so even the unquantized fp16 engine gives NaN. Their FP8 build
requires ``precision="fp32"`` (the family has no bf16) and is rejected with fp16.

Quantized projections (names are the builder's logical weight names):

* ViT: the fused q/k/v projection (``layer.N.attention.qkv``, one weight scale over the
  three weights, as the builder runs them as one GEMM), ``attention.o_proj``,
  ``mlp.up_proj``, ``mlp.down_proj`` (and ``mlp.gate_proj`` when the MLP is gated).
* ConvNeXt: both pointwise projections of every block (``stage.S.block.B.pointwise1`` and
  ``pointwise2``).

The patch embedding, the convolutions, the layer norms, layer scales, RoPE and attention
matmuls stay in the work precision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
import tensorrt as trt

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"

# ViT with at least this hidden size (ViT-L, ViT-H+, ViT-7B) overflows fp16 activations.
FP16_OVERFLOW_HIDDEN_SIZE = 1024
# Smallest normal fp16 value; a smaller Q/DQ scale would lose precision as an fp16 constant.
_FP16_MIN_NORMAL = 2.0**-14

_SUPPORTED = ("fp8",)


@dataclass(frozen=True)
class _FP8Weight:
    """One projection weight stored as raw fp8_e4m3 bytes with one per-tensor scale."""

    packed: np.ndarray  # [in_features, out_features] raw fp8_e4m3 bytes
    weight_scale: float
    in_features: int
    out_features: int


@dataclass(frozen=True)
class _LayerScales:
    """Activation scale and the packed weight of one quantized projection."""

    input_scale: float
    weight: _FP8Weight


def _scale_const(network, value: float, dtype: trt.DataType):
    """One-element constant holding a Q/DQ scale in the Dequantize output type."""
    np_dtype = np.float16 if dtype == trt.float16 else np.float32
    data = np.asarray([value], dtype=np_dtype)
    return network.add_constant((1,), trt.Weights(data)).get_output(0)


def _wrap_fp8_matmul(network, activation, scales: _LayerScales, *, keep_alive: list):
    """Add ``activation @ weight`` with an FP8 weight and a static FP8 activation scale.

    The Dequantize layers and the matmul use the type of ``activation`` (fp16 or fp32), so
    the matmul output has the same type and no cast is added. With an fp16 activation
    TensorRT's FP8 GEMM keeps the raw accumulator in fp16 and applies the scales after,
    which overflows once an output exceeds about ``65504 * input_scale * weight_scale``;
    the scan in the tests and the lab checks real inputs for that.
    """
    dtype = activation.dtype
    if dtype not in (trt.float16, trt.float32):
        raise ValueError(f"DINOv3 FP8 supports fp16 or fp32 activations, got {dtype}")
    w = scales.weight
    rank = len(tuple(activation.shape))
    shape = (1,) * max(0, rank - 2) + (w.in_features, w.out_features)

    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(w.packed)
    w_fp8 = trt.Weights(trt.DataType.FP8, w.packed.ctypes.data, w.in_features * w.out_features)
    w_const = network.add_constant(shape, w_fp8).get_output(0)
    w_dequant = network.add_dequantize(w_const, _scale_const(network, w.weight_scale, dtype), dtype)

    act_scale = _scale_const(network, scales.input_scale, dtype)
    a_quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
    a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, dtype)

    return network.add_matrix_multiply(
        a_dequant.get_output(0),
        trt.MatrixOperation.NONE,
        w_dequant.get_output(0),
        trt.MatrixOperation.NONE,
    ).get_output(0)


@dataclass(frozen=True)
class Dinov3QuantContext:
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
        """True when ``name`` (for example ``layer.3.mlp.up_proj``) is quantized."""
        return name in self.scales

    def matmul(self, network, activation, name: str):
        """``activation @ weight`` through FP8 Q/DQ for the projection ``name``."""
        return _wrap_fp8_matmul(network, activation, self.scales[name], keep_alive=self.keep_alive)


def scale_key(model_type: str, cfg: dict) -> str:
    """Key of this model's table in the activation-scale file, from the resolved geometry.

    The key holds every shape value that changes the activation ranges: hidden size,
    layers, intermediate size, patch size, gated MLP and the input resolution (ViT), or
    channels, depths and resolution (ConvNeXt).
    """
    if model_type == "dinov3_vit":
        gated = "-gated" if cfg["use_gated_mlp"] else ""
        return (
            f"dinov3_vit-h{cfg['hidden_size']}-l{cfg['num_hidden_layers']}"
            f"-i{cfg['intermediate_size']}-p{cfg['patch_size']}{gated}"
            f"-s{cfg['image_h']}x{cfg['image_w']}"
        )
    if model_type == "dinov3_convnext":
        channels = "x".join(str(v) for v in cfg["hidden_sizes"])
        depths = "x".join(str(v) for v in cfg["depths"])
        return f"dinov3_convnext-c{channels}-d{depths}-s{cfg['image_h']}x{cfg['image_w']}"
    raise ValueError(f"unsupported DINOv3 model_type for FP8: {model_type!r}")


def requires_fp32_work(model_type: str, cfg: dict) -> bool:
    """True when the fp16 engine of this shape overflows, so FP8 needs ``precision="fp32"``."""
    return model_type == "dinov3_vit" and cfg["hidden_size"] >= FP16_OVERFLOW_HIDDEN_SIZE


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


def load_activation_scales(model_type: str, cfg: dict) -> dict[str, float]:
    """Return the committed activation scales for this model shape, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"DINOv3 FP8 build requires {path}, which was not found") from error
    key = scale_key(model_type, cfg)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model shape."
        )
    return table[key]


def projection_weights(model_type: str, cfg: dict, weights: Any) -> dict[str, np.ndarray]:
    """Logical name to ``[in_features, out_features]`` weight of every quantizable projection."""
    found: dict[str, np.ndarray] = {}
    if model_type == "dinov3_vit":
        for layer in range(cfg["num_hidden_layers"]):
            prefix = f"layer.{layer}"
            found[f"{prefix}.attention.qkv"] = np.concatenate(
                [weights[f"{prefix}.attention.{name}_proj.weight"] for name in ("q", "k", "v")],
                axis=1,
            )
            found[f"{prefix}.attention.o_proj"] = weights[f"{prefix}.attention.o_proj.weight"]
            mlp = ["up_proj", "down_proj"] + (["gate_proj"] if cfg["use_gated_mlp"] else [])
            for name in mlp:
                found[f"{prefix}.mlp.{name}"] = weights[f"{prefix}.mlp.{name}.weight"]
    elif model_type == "dinov3_convnext":
        for stage, depth in enumerate(cfg["depths"]):
            for block in range(depth):
                prefix = f"stage.{stage}.block.{block}"
                for name in ("pointwise1", "pointwise2"):
                    found[f"{prefix}.{name}"] = weights[f"{prefix}.{name}.weight"]
    else:
        raise ValueError(f"unsupported DINOv3 model_type for FP8: {model_type!r}")
    return found


def calibrate_dinov3(
    model_type: str,
    cfg: dict,
    weights: Any,
    quantization: str,
    *,
    precision: str | None = None,
) -> Dinov3QuantContext:
    """Quantize the checkpoint one projection at a time and pair it with the stored scales.

    ``precision`` is the work precision of the engine being built; when given, an fp16
    engine of a shape whose fp16 activations overflow is rejected.
    """
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"DINOv3 supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    if precision == "fp16" and requires_fp32_work(model_type, cfg):
        raise ValueError(
            f"DINOv3 {model_type} hidden_size={cfg['hidden_size']} overflows fp16 "
            "(residual stream above 65504, NaN even without FP8); build it with precision fp32"
        )
    activation_scales = load_activation_scales(model_type, cfg)
    projections = projection_weights(model_type, cfg, weights)
    missing = sorted(set(projections) - set(activation_scales))
    unused = sorted(set(activation_scales) - set(projections))
    if missing or unused:
        raise ValueError(
            f"FP8 scale table {scale_key(model_type, cfg)!r} does not match the model: "
            f"{len(missing)} projections without a scale (first: {missing[:3]}), "
            f"{len(unused)} scales without a projection (first: {unused[:3]}). "
            "Re-run calibrate_fp8_scales.py for this model shape."
        )
    scales: dict[str, _LayerScales] = {}
    for name, weight in projections.items():
        input_scale = float(activation_scales[name])
        if not np.isfinite(input_scale) or input_scale <= 0:
            raise ValueError(f"FP8 activation scale of {name!r} must be finite and positive")
        weight = np.asarray(weight, dtype=np.float32)
        packed, weight_scale = quantize_fp8_weight(weight)
        if precision == "fp16" and min(input_scale, weight_scale) < _FP16_MIN_NORMAL:
            raise ValueError(
                f"FP8 scale of {name!r} is below the smallest normal fp16 value; "
                "build this model with precision fp32"
            )
        in_features, out_features = weight.shape
        scales[name] = _LayerScales(
            input_scale, _FP8Weight(packed, weight_scale, in_features, out_features)
        )
    if not scales:
        raise RuntimeError("DINOv3 FP8 self-quantization matched no projection in the checkpoint")
    return Dinov3QuantContext(scales=scales)
