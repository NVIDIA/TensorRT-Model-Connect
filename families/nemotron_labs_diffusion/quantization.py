# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-Labs-Diffusion-owned FP8 self-quantization of the decoder projections.

``quantization="fp8"`` quantizes the original BF16 checkpoint at build time; no FP8
checkpoint is needed or written. Weights are quantized one projection at a time:
``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3 value), and the FP8
bytes are fed to the engine as a constant. The activation scale of each projection input
needs real activations, so it was calibrated once offline with
``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``) and is read from
``fp8_activation_scales.json``. Users do not run calibration. A projection with no entry
in that file stays in fp16.

Each quantized projection becomes::

    activation -> Quantize -> Dequantize -> MatMul -> cast to fp16
    FP8 weight -> Dequantize --------------^

Quantized projections of every decoder layer ``layer.N``: ``w_q``, ``w_k``, ``w_v``,
``w_gate``, ``w_up`` and ``w_down``. The attention output projection ``w_o`` stays in fp16:
TensorRT 11.4 cannot compile an FP8 Quantize on the output of the fused attention. The
embedding, the diffusion head and the norms stay in fp16.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import ml_dtypes
import numpy as np
import tensorrt as trt

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"

_SUPPORTED = ("fp8",)
# Build-route knob that turns off TensorRT's fused dual-GEMM kernel.
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"

# Projection stems of one decoder layer (engine weight names are ``layer.N.<stem>``).
PROJECTIONS = ("w_q", "w_k", "w_v", "w_o", "w_gate", "w_up", "w_down")
# Quantized subset; ``w_o`` follows the fused attention (see the module docstring).
QUANTIZED = ("w_q", "w_k", "w_v", "w_gate", "w_up", "w_down")


def scale_key(config) -> str:
    """Key of this model's entry in the activation-scale file."""
    return (
        f"{str(config.model_type).lower()}-h{int(config.hidden_size)}"
        f"-l{int(config.num_hidden_layers)}-i{int(config.intermediate_size)}"
    )


def quantized_names(num_layers: int, stems: tuple[str, ...] = QUANTIZED) -> list[str]:
    """Names (``layer.N.<stem>``) of the projections that can be quantized."""
    return [f"layer.{i}.{stem}" for i in range(num_layers) for stem in stems]


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


def load_activation_scales(config) -> dict[str, float]:
    """Return the committed activation scales for this model shape, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Nemotron Labs Diffusion FP8 build requires {path}, which was not found"
        ) from error
    key = scale_key(config)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model shape."
        )
    return table[key]


def _scale_const(network, value: float):
    """One-element fp32 constant holding a Q/DQ scale."""
    arr = np.asarray([value], dtype=np.float32)
    return network.add_constant((1,), trt.Weights(arr)).get_output(0)


@dataclass
class NemotronQuantContext:
    """Routes the quantized projections through FP8 Q/DQ; others stay in fp16."""

    scales: dict[str, float]
    graph_ops: object
    keep_alive: list = field(default_factory=list)

    @property
    def has_fp8_weights(self) -> bool:
        """True when any projection is FP8 (selects the FP8 build options)."""
        return bool(self.scales)

    def should_quantize(self, weight_name: str) -> bool:
        """True when ``weight_name`` (for example ``layer.3.w_k``) is a quantized projection."""
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
        """FP8 matmul for quantized projections; the regular constant matmul otherwise.

        ``rhs_weights`` is the engine's ``[in, out]`` weight. The Dequantize layers and the
        matmul run in float32 and the result is cast back to the activation type. With
        16-bit Dequantize outputs, TensorRT's FP8 GEMM keeps the raw accumulator in 16 bits
        and applies the scales afterwards, so it overflows once an output exceeds about
        ``65504 * input_scale * weight_scale``.
        """
        if weight_name not in self.scales:
            return self.graph_ops.add_matmul_rhs_constant(
                network, lhs, lhs_width, rhs_width, rhs_weights, dtype=dtype
            )
        packed, weight_scale = quantize_fp8_weight(np.asarray(rhs_weights, dtype=np.float32))
        # trt.Weights holds a raw pointer, so the buffer must outlive the build.
        self.keep_alive.append(packed)
        rank = len(tuple(lhs.shape))
        shape = (lhs_width, rhs_width) if rank <= 2 else (1,) * (rank - 2) + (lhs_width, rhs_width)
        w_const = network.add_constant(
            shape, trt.Weights(trt.DataType.FP8, packed.ctypes.data, lhs_width * rhs_width)
        ).get_output(0)
        w_dequant = network.add_dequantize(
            w_const, _scale_const(network, weight_scale), trt.float32
        )

        act_scale = _scale_const(network, self.scales[weight_name])
        a_quant = network.add_quantize(lhs, act_scale, trt.DataType.FP8)
        a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, trt.float32)

        out = network.add_matrix_multiply(
            a_dequant.get_output(0),
            trt.MatrixOperation.NONE,
            w_dequant.get_output(0),
            trt.MatrixOperation.NONE,
        ).get_output(0)
        return network.add_cast(out, lhs.dtype).get_output(0)


def configure_build(trt_config) -> None:
    """Apply the FP8 build options and keep layer names for kernel inspection."""
    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    # TensorRT's fused dual-GEMM kernel fails to compile with scalar-scale FP8.
    # The route is set only when this TensorRT build lists the knob.
    if _DUAL_GEMM_KNOB in getattr(trt_config, "all_build_routes", ""):
        trt_config.build_route = f"-{_DUAL_GEMM_KNOB}=off"


def calibrate_nemotron(
    weights, config, graph_ops, quantization: str, *, stems: tuple[str, ...] = QUANTIZED
) -> NemotronQuantContext:
    """Pair the committed activation scales with the checkpoint, after validating it."""
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"Nemotron Labs Diffusion supports quantization {list(_SUPPORTED)} or none; "
            f"got {quantization!r}"
        )
    if any(str(key).endswith((".weight_scale", ".input_scale")) for key in weights):
        raise ValueError(
            "the checkpoint is already quantized; fp8 self-quantization needs the original checkpoint"
        )
    activation_scales = load_activation_scales(config)
    scales: dict[str, float] = {}
    for name in quantized_names(int(config.num_hidden_layers), stems):
        scale = activation_scales.get(name)
        if scale is None:
            continue
        if name not in weights:
            raise ValueError(f"checkpoint has no weight for {name}")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"activation scale of {name} must be finite and positive")
        scales[name] = float(scale)
    if not scales:
        raise RuntimeError(
            "Nemotron Labs Diffusion FP8 self-quantization matched no projection in the checkpoint"
        )
    return NemotronQuantContext(scales=scales, graph_ops=graph_ops)


def configure_quant_build(quant_ctx, trt_config) -> None:
    """Apply the FP8 build options when a quantization context is present."""
    if quant_ctx is not None:
        configure_build(trt_config)
