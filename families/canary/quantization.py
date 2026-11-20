# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canary-owned FP8 TensorRT Q/DQ graph context.

FP8 (``quantization="fp8"``) is self-quantization of the original checkpoint, because
no ModelOpt FP8 checkpoint exists for Canary. Weights are quantized at build time, one
projection at a time, when the graph asks for the matmul: ``scale = max(abs(weight)) /
448`` (448 is the largest finite e4m3 value), and the FP8 bytes are fed to the engine as
a constant. The activation scale of each projection input needs real audio, so it was
calibrated once offline with ``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``)
and is read from ``fp8_activation_scales.json``. Users do not run calibration. A
projection with no entry in that file stays unquantized. Each quantized projection
becomes ``activation -> Quantize -> Dequantize -> MatMul`` with ``FP8 weight ->
Dequantize`` on the other input.

Quantized projections (weight-dict names, ``N`` is the layer index):

  encoder  ``el.N.w_{q,k,v,o}`` and the four feed-forward matrices
           ``el.N.ff{1,2}.w{1,2}``
  decoder  ``layer.N.w_{q,k,v}`` (self-attention), ``layer.N.xw_{q,k,v}``
           (cross-attention) and ``layer.N.w_fc{1,2}``; the attention output
           projections ``layer.N.w_o`` and ``layer.N.xw_o`` stay unquantized (see
           ``_UNQUANTIZED``)

The subsampling convolutions, the conformer convolution modules, the positional
projection, the embeddings, the layer norms, the biases and the output projection stay
unquantized. Weights are stored ``[in_features, out_features]``, so the matmul uses the
FP8 constant as is.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
import tensorrt as trt

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"

_SUPPORTED = ("fp8",)
# Weight-dict stems of the quantizable projections.
_ENCODER_STEMS = (
    "w_q",
    "w_k",
    "w_v",
    "w_o",
    "ff1.w1",
    "ff1.w2",
    "ff2.w1",
    "ff2.w2",
)
_DECODER_STEMS = (
    "w_q",
    "w_k",
    "w_v",
    "w_o",
    "xw_q",
    "xw_k",
    "xw_v",
    "xw_o",
    "w_fc1",
    "w_fc2",
)
# Projections that are never quantized, even when the scale file has an entry: the decoder
# attention output projections. TensorRT 11.4 cannot compile an FP8 Quantize right after the
# fused attention (``element type (fp8_e4m3) differs from input 1's element type (half)`` in
# the bias add). They hold about 1/10 of a decoder layer's weights.
_UNQUANTIZED = re.compile(r"^layer\.\d+\.x?w_o$")
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"


@dataclass(frozen=True)
class CanaryQuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, float]  # weight name -> activation scale
    graph_ops: Any
    keep_alive: list = None  # type: ignore[assignment]

    @property
    def has_fp8_weights(self) -> bool:
        """True when any projection is FP8 (selects the FP8 build options)."""
        return bool(self.scales)

    def __post_init__(self):
        """Create the list that keeps weight buffers alive during the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, weight_name: str) -> bool:
        """True when ``weight_name`` (for example ``layer.3.xw_k``) is a quantized projection."""
        return weight_name in self.scales

    def maybe_quantized_matmul(
        self,
        network,
        lhs,
        lhs_width: int,
        rhs_width: int,
        rhs_weights: np.ndarray,
        weight_name: str,
        dtype: np.dtype = np.float32,
    ):
        """Quantized matmul for quantized projections; the regular constant matmul otherwise."""
        input_scale = self.scales.get(weight_name)
        if input_scale is None:
            return self.graph_ops.add_matmul_rhs_constant(
                network, lhs, lhs_width, rhs_width, rhs_weights, dtype=dtype
            )
        packed, weight_scale = quantize_fp8_weight(
            np.asarray(rhs_weights, dtype=np.float32).reshape(lhs_width, rhs_width)
        )
        return _wrap_fp8_matmul(
            network,
            lhs,
            packed,
            weight_scale,
            input_scale,
            graph_ops=self.graph_ops,
            keep_alive=self.keep_alive,
        )


def _scale_const(network, graph_ops, value: float):
    """One-element fp32 constant holding a Q/DQ scale."""
    return graph_ops.add_constant(
        network, (1,), np.asarray([value], dtype=np.float32), dtype=np.float32
    )


def _wrap_fp8_matmul(
    network,
    activation,
    packed: np.ndarray,
    weight_scale: float,
    input_scale: float,
    *,
    graph_ops,
    keep_alive: list,
):
    """Add ``activation @ weight`` with an FP8 weight and a static FP8 activation scale.

    The Dequantize layers and the matmul run in float32 and the result is cast back.
    With fp16 Dequantize outputs, TensorRT's FP8 GEMM keeps the raw accumulator in
    fp16 and applies the scales afterwards, so it overflows to NaN as soon as an
    output exceeds about ``65504 * input_scale * weight_scale``.
    """
    k, n = packed.shape
    rank = len(tuple(activation.shape))

    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(packed)
    w_fp8 = trt.Weights(trt.DataType.FP8, packed.ctypes.data, k * n)
    w_const = network.add_constant((1,) * (rank - 2) + (k, n), w_fp8).get_output(0)
    w_dequant = network.add_dequantize(
        w_const, _scale_const(network, graph_ops, weight_scale), trt.float32
    )

    act_scale = _scale_const(network, graph_ops, input_scale)
    a_quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
    a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, trt.float32)

    output = network.add_matrix_multiply(
        a_dequant.get_output(0),
        trt.MatrixOperation.NONE,
        w_dequant.get_output(0),
        trt.MatrixOperation.NONE,
    ).get_output(0)
    if output.dtype == activation.dtype:
        return output
    return network.add_cast(output, activation.dtype).get_output(0)


def configure_quantized_build(trt_config, quant_ctx) -> None:
    """Apply the FP8 build options and keep layer names for kernel inspection."""
    if quant_ctx is None:
        return
    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if quant_ctx.has_fp8_weights and _DUAL_GEMM_KNOB in getattr(trt_config, "all_build_routes", ""):
        # TensorRT's fused dual-GEMM kernel fails to compile with scalar-scale FP8.
        # The route is set only when this TensorRT build lists the knob.
        trt_config.build_route = f"-{_DUAL_GEMM_KNOB}=off"


def scale_key(weights) -> str:
    """Key of this model's entry in the activation-scale file."""
    return (
        f"canary-h{int(weights['_hidden'])}"
        f"-e{int(weights['_enc_layers'])}"
        f"-d{int(weights['_dec_layers'])}"
        f"-m{int(weights['_mel_bins'])}"
    )


def projection_names(weights) -> list[str]:
    """Weight-dict names of every quantizable projection of this model."""
    names = [
        f"el.{layer}.{stem}"
        for layer in range(int(weights["_enc_layers"]))
        for stem in _ENCODER_STEMS
    ]
    names += [
        f"layer.{layer}.{stem}"
        for layer in range(int(weights["_dec_layers"]))
        for stem in _DECODER_STEMS
    ]
    return names


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


def load_activation_scales(key: str) -> dict[str, float]:
    """Return the committed activation scales for this model shape, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Canary FP8 build requires {path}, which was not found") from error
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model shape."
        )
    return table[key]


def calibrate_canary(weights, graph_ops, quantization: str) -> CanaryQuantContext:
    """Pair the stored activation scales with the loaded projections of this checkpoint."""
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"Canary supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    activation_scales = load_activation_scales(scale_key(weights))
    scales: dict[str, float] = {}
    for name in projection_names(weights):
        input_scale = activation_scales.get(name)
        if input_scale is None or _UNQUANTIZED.match(name):
            continue
        if name not in weights:
            raise ValueError(f"checkpoint has no {name}")
        scales[name] = float(input_scale)
    if not scales:
        raise RuntimeError("Canary FP8 self-quantization matched no projection in the checkpoint")
    return CanaryQuantContext(scales=scales, graph_ops=graph_ops)
