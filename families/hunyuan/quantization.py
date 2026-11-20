# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hunyuan-owned FP8 self-quantization and TensorRT Q/DQ graph context.

The original BF16 Hunyuan checkpoint is used as-is; no FP8 checkpoint is needed.

Weights are quantized on the fly at build time, one projection at a time:
``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3 value), and
the FP8 bytes are fed to the engine as a constant. The weights are taken after
``_HunyuanModel.load_weights`` has absorbed the Hunyuan multipliers, so the
quantized weight is the one the unquantized engine would use. The activation
scale of each projection input needs real data, so it was calibrated once
offline with ``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``) and is
read from ``fp8_activation_scales.json``. Users do not run calibration.

Each quantized projection becomes::

    activation -> Quantize -> Dequantize(fp16) -> MatMul(fp16)
    FP8 weight -> Dequantize(fp16) -------------------^

Both Dequantize layers output the activation type (fp16), so the FP8 GEMM writes fp16 and the
SwiGLU and residual math after it stay fused. A float32 Dequantize makes the GEMM write float32
and adds a cast pass. The risk of fp16 is overflow above 65504; the validation scans real inputs
for non-finite values (see the pull request).

Quantized (internal weight names, ``N`` is the layer index): ``layer.N.w_q``, ``w_k``, ``w_v``,
``w_o`` (attention output projection), ``w_gate``, ``w_up``, ``w_down``. Not quantized: ``lm_head``
(tied to the token embedding), the token embedding and the RMSNorms. The attention output
projection builds as FP8 on TensorRT 11.4 and 11.1 for this family (the Hunyuan decoder uses the
native attention layer; other families fail to compile an FP8 Quantize after a fused attention on
11.4, see the pull request), so it is quantized here. A projection whose input has a massive
activation (a scale more than 20 times the median of the same projection, see
``calibrate_fp8_scales.split_outlier_scales``) stays in fp16; its entry in the scale file is
``null``. A projection with no entry in the file is an error, never a silent fp16 fallback.
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

# HF projection stem -> internal weight-dict stem used by graph_blocks.
PROJECTIONS = (
    ("self_attn.q_proj", "w_q"),
    ("self_attn.k_proj", "w_k"),
    ("self_attn.v_proj", "w_v"),
    ("self_attn.o_proj", "w_o"),
    ("mlp.gate_proj", "w_gate"),
    ("mlp.up_proj", "w_up"),
    ("mlp.down_proj", "w_down"),
)
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"


def scale_key(config, final_norm) -> str:
    """Key of this model's entry in the activation-scale file.

    The model config alone does not identify a checkpoint: Hy-MT2-7B and HY-MT1.5-7B share one
    config and have different weights. The key therefore ends with the numeric sum of the final
    RMSNorm weight (``n<sum>``), which differs between checkpoints of the same config.
    """
    fingerprint = float(np.sum(np.asarray(final_norm, dtype=np.float64)))
    return (
        f"{str(config.model_type).lower()}"
        f"-h{int(config.hidden_size)}"
        f"-l{int(config.num_hidden_layers)}"
        f"-i{int(config.intermediate_size)}"
        f"-n{fingerprint:.4f}"
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
    """Static activation scale and FP8 weight of one quantized projection."""

    input_scale: float
    weight: _FP8Weight


def _scale_const(network, graph_ops, value: float):
    """Add a one-element fp32 scale constant to the network."""
    return graph_ops.add_constant(
        network, (1,), np.asarray([value], dtype=np.float32), dtype=np.float32
    )


def _wrap_fp8_matmul(network, activation, scales: _LayerScales, *, graph_ops, keep_alive: list):
    """Add the FP8 Q/DQ matmul for one projection; Dequantize and matmul use the activation type."""
    w = scales.weight
    n, k = w.out_features, w.in_features

    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(w.packed)
    w_fp8 = trt.Weights(trt.DataType.FP8, w.packed.ctypes.data, n * k)
    w_const = network.add_constant((n, k), w_fp8).get_output(0)
    w_dequant = network.add_dequantize(
        w_const, _scale_const(network, graph_ops, w.weight_scale), activation.dtype
    )

    act_scale = _scale_const(network, graph_ops, scales.input_scale)
    a_quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
    a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, activation.dtype)

    return network.add_matrix_multiply(
        a_dequant.get_output(0),
        trt.MatrixOperation.NONE,
        w_dequant.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)


@dataclass(frozen=True)
class HunyuanQuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    graph_ops: Any
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        """Give each context its own list for buffers that must outlive the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, weight_name: str) -> bool:
        """Return True when this weight has a calibrated activation scale."""
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
        """Add the projection as an FP8 matmul, or as the unquantized matmul when it has no scale."""
        entry = self.scales.get(weight_name)
        if entry is None:
            return self.graph_ops.add_matmul_rhs_constant(
                network, lhs, lhs_width, rhs_width, rhs_weights, dtype=dtype
            )
        return _wrap_fp8_matmul(
            network, lhs, entry, graph_ops=self.graph_ops, keep_alive=self.keep_alive
        )


def configure_fp8_build(trt_config, quant_ctx) -> None:
    """Apply the FP8 build options; a no-op for an unquantized build."""
    if quant_ctx is None:
        return
    # Keeps per-layer tactic names in the plan so FP8 lowering can be inspected.
    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    # TensorRT's fused dual-GEMM (gate/up) kernel is switched off: it does not compile for an
    # FP8 consumer of the SwiGLU output on TensorRT 11.1 in other families, and on TensorRT 11.4,
    # where it builds, it made decode 5% to 7% slower here. The route is set only when this
    # TensorRT build lists the knob.
    if _DUAL_GEMM_KNOB in getattr(trt_config, "all_build_routes", ""):
        trt_config.build_route = f"-{_DUAL_GEMM_KNOB}=off"


def quantize_fp8_weight(weight_fp32: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize one weight to FP8 e4m3 with a single scalar scale.

    Returns the raw fp8 bytes and ``scale = max(abs(weight)) / 448``; the same
    scale is the dequantize scale in the engine.
    """
    amax = float(np.abs(weight_fp32).max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scale = amax / FP8_E4M3_MAX
    scaled = np.clip(weight_fp32 / scale, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return np.ascontiguousarray(packed), scale


def load_activation_scales(config, final_norm) -> dict[str, float | None]:
    """Return the committed activation scales for this model config, or raise when none exist.

    An entry of ``None`` marks a projection the calibration left unquantized on purpose.
    """
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Hunyuan FP8 build requires {path}, which was not found") from error
    key = scale_key(config, final_norm)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model config."
        )
    return table[key]


def projection_names(config) -> list[str]:
    """Internal weight names of every quantized projection of this model config."""
    return [
        f"layer.{layer}.{weight_stem}"
        for layer in range(int(config.num_hidden_layers))
        for _, weight_stem in PROJECTIONS
    ]


def calibrate_hunyuan_fp8(config, weights, graph_ops) -> HunyuanQuantContext:
    """Build the Q/DQ context from the loaded Hunyuan weights (``[in, out]`` layout).

    Quantizes one projection at a time, so peak extra memory is one weight tensor. A projection
    with no stored activation scale, or a stored scale for a projection this model config does
    not quantize, is an error.
    """
    if config.raw.get("quantization_config") or any(
        str(key).endswith((".weight_scale", ".input_scale")) for key in weights
    ):
        raise ValueError(
            "the checkpoint is already quantized; FP8 self-quantization needs the original checkpoint"
        )
    activation_scales = load_activation_scales(config, weights["final_norm"])

    names = projection_names(config)
    unexpected = sorted(set(activation_scales) - set(names))
    if unexpected:
        raise ValueError(
            f"{ACTIVATION_SCALES_FILENAME} has scales for projections that are not quantized, "
            f"for example {unexpected[0]}; run calibrate_fp8_scales.py for this model config"
        )
    scales: dict[str, _LayerScales] = {}
    for name in names:
        if name not in activation_scales:
            raise ValueError(
                f"{name} has no activation scale in {ACTIVATION_SCALES_FILENAME}; "
                "run calibrate_fp8_scales.py"
            )
        input_scale = activation_scales[name]
        if input_scale is None:
            continue  # massive-activation projection, left in fp16 by the calibration
        if not np.isfinite(input_scale) or input_scale <= 0:
            raise ValueError(f"activation scale of {name} must be finite and positive")
        if name not in weights:
            raise ValueError(f"checkpoint has no {name}")
        transposed = np.asarray(weights[name])
        in_features, out_features = transposed.shape
        packed, weight_scale = quantize_fp8_weight(
            np.ascontiguousarray(transposed.T, dtype=np.float32)
        )
        scales[name] = _LayerScales(
            input_scale=float(input_scale),
            weight=_FP8Weight(packed, weight_scale, out_features, in_features),
        )
    if not scales:
        raise RuntimeError("Hunyuan FP8 self-quantization matched no projection in the checkpoint")
    return HunyuanQuantContext(scales=scales, graph_ops=graph_ops)
