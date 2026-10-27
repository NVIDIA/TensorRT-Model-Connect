# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PersonaPlex-owned FP8 TensorRT Q/DQ graph context.

FP8 (``quantization="fp8"``) is self-quantization of the original checkpoint, because no
ModelOpt FP8 checkpoint exists for PersonaPlex. Weights are quantized at build time, one
projection at a time: ``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3
value), and the FP8 bytes are fed to the engine as a constant. The activation scale of
each projection input needs real speech, so it was calibrated once offline with
``calibrate_fp8_scales.py`` and is read from ``fp8_activation_scales.json``. Users do not
run calibration. A projection with no entry in that file stays unquantized. Each quantized
projection becomes ``activation -> Quantize -> Dequantize -> MatMul`` with
``FP8 weight -> Dequantize`` on the other input.

Quantized projections (engine weight names, ``N`` is the temporal layer index):

  temporal  ``layer.N.w_{q,k,v,gate,up}``

Two projections per layer stay fp16 on purpose. The attention output projection ``w_o``:
TensorRT 11.4 cannot compile an FP8 Quantize right after the fused attention. The MLP
down projection ``w_down``: its FP8 GEMM (K = 11264) measured slower than the fp16 one at
batch 1, and it is the most sensitive projection (its input has the heaviest tails), so
quantizing it costs both speed and accuracy. The depth transformer, both Mimi codecs, the
embeddings, the norms and the text output projection stay unquantized.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
import tensorrt as trt

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"

_SUPPORTED = ("fp8",)
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"
# Temporal projections that are quantized, in engine weight-name form.
PROJECTION_STEMS = ("w_q", "w_k", "w_v", "w_gate", "w_up")
_PROJECTION_NAME = re.compile(r"^layer\.(\d+)\.(" + "|".join(PROJECTION_STEMS) + r")$")


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


def _scale_const(network, graph_ops, value: float):
    """One-element fp32 constant holding a Q/DQ scale."""
    return graph_ops.add_constant(
        network, (1,), np.asarray([value], dtype=np.float32), dtype=np.float32
    )


def _wrap_fp8_matmul(network, activation, scales: _LayerScales, *, graph_ops, keep_alive: list):
    """Add ``activation @ weight.T`` with FP8 weights and a static FP8 activation scale.

    The Dequantize layers and the matmul run in float32 and the result is cast back.
    With fp16 Dequantize outputs, TensorRT's FP8 GEMM keeps the raw accumulator in
    fp16 and applies the scales afterwards, so it overflows to NaN as soon as an
    output exceeds about ``65504 * input_scale * weight_scale``.
    """
    w = scales.weight
    n, k = w.out_features, w.in_features

    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(w.packed)
    w_fp8 = trt.Weights(trt.DataType.FP8, w.packed.ctypes.data, n * k)
    w_const = network.add_constant((n, k), w_fp8).get_output(0)
    w_dequant = network.add_dequantize(
        w_const, _scale_const(network, graph_ops, w.weight_scale), trt.float32
    )

    act_scale = _scale_const(network, graph_ops, scales.input_scale)
    a_quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
    a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, trt.float32)

    output = network.add_matrix_multiply(
        a_dequant.get_output(0),
        trt.MatrixOperation.NONE,
        w_dequant.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    if output.dtype == activation.dtype:
        return output
    return network.add_cast(output, activation.dtype).get_output(0)


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


@dataclass
class PersonaPlexQuantContext:
    """Routes the quantized temporal projections through Q/DQ; others stay unquantized.

    The checkpoint loader asks ``should_quantize`` per projection and hands the fp32
    ``[out, in]`` weight to ``add_weight``, which stores only the FP8 bytes.
    """

    activation_scales: dict[str, float]
    graph_ops: Any
    scales: dict[str, _LayerScales] = field(default_factory=dict)
    keep_alive: list = field(default_factory=list)

    def should_quantize(self, weight_name: str) -> bool:
        """True when the loader must hand ``weight_name`` (for example ``layer.3.w_q``) over."""
        return weight_name in self.activation_scales

    def add_weight(self, weight_name: str, weight_out_in: np.ndarray) -> tuple[int, int]:
        """Quantize one ``[out, in]`` fp32 weight; returns ``(in_features, out_features)``."""
        if weight_out_in.ndim != 2:
            raise ValueError(f"{weight_name} is not a matrix")
        packed, weight_scale = quantize_fp8_weight(weight_out_in)
        out_features, in_features = weight_out_in.shape
        self.scales[weight_name] = _LayerScales(
            float(self.activation_scales[weight_name]),
            _FP8Weight(packed, weight_scale, out_features, in_features),
        )
        return in_features, out_features

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
        entry = self.scales.get(weight_name)
        if entry is None:
            if self.should_quantize(weight_name):
                raise RuntimeError(f"{weight_name} has a calibrated scale but was never quantized")
            return self.graph_ops.add_matmul_rhs_constant(
                network, lhs, lhs_width, rhs_width, rhs_weights, dtype=dtype
            )
        return _wrap_fp8_matmul(
            network, lhs, entry, graph_ops=self.graph_ops, keep_alive=self.keep_alive
        )


def configure_fp8_build(trt_config, quant_ctx) -> None:
    """Apply the FP8 build options and keep layer names for kernel inspection."""
    if quant_ctx is None:
        return
    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if _DUAL_GEMM_KNOB in getattr(trt_config, "all_build_routes", ""):
        # TensorRT's fused dual-GEMM kernel fails to compile with scalar-scale FP8.
        # The route is set only when this TensorRT build lists the knob.
        trt_config.build_route = f"-{_DUAL_GEMM_KNOB}=off"


def scale_key(inferred: dict) -> str:
    """Key of this model's entry in the activation-scale file."""
    return (
        f"personaplex-h{int(inferred['hidden_size'])}"
        f"-l{int(inferred['num_hidden_layers'])}"
        f"-i{int(inferred['intermediate_size'])}"
    )


def load_activation_scales(inferred: dict) -> dict[str, float]:
    """Return the committed activation scales for this model shape, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"PersonaPlex FP8 build requires {path}, which was not found") from error
    key = scale_key(inferred)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model shape."
        )
    scales = {name: value for name, value in table[key].items() if _PROJECTION_NAME.match(name)}
    if not scales:
        raise RuntimeError(f"FP8 activation scales for {key!r} hold no temporal projection")
    return scales


def create_quant_context(readers, inferred: dict, graph_ops, quantization: str):
    """Build the FP8 context for the original checkpoint, or raise for anything else."""
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"PersonaPlex supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    if any(name.endswith(".weight_scale") for name in readers.tensor_map):
        raise ValueError(
            "checkpoint is already quantized; fp8 self-quantization needs the original checkpoint"
        )
    return PersonaPlexQuantContext(load_activation_scales(inferred), graph_ops)
