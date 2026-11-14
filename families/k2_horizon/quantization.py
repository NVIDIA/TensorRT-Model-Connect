# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""K2-Horizon-owned FP8 self-quantization and TensorRT Q/DQ graph context.

The original BF16 K2-Horizon checkpoint is used as-is; no FP8 checkpoint is needed.

Weights are quantized on the fly at build time, one projection at a time:
``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3 value), and
the FP8 bytes are fed to the engine as a constant. The activation scale of each
projection input needs real data, so it was calibrated once offline with
``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``) and is read from
``fp8_activation_scales.json``. Users do not run calibration.

Each quantized projection becomes::

    activation -> Quantize -> Dequantize(bf16) -> MatMul(bf16)
    FP8 weight -> Dequantize(bf16) -------------------^

Both Dequantize layers output the activation type (bf16), so the FP8 GEMM writes bf16 and the
SwiGLU and residual math after it stay fused. bf16 has the float32 exponent range, so the
Dequantize cannot overflow.

Quantized (internal weight names, ``N`` is the layer index): ``layer.N.w_q``, ``w_k``, ``w_v``,
``w_o`` (attention output projection), ``w_gate``, ``w_up``, ``w_down``. This is the layer list of
the publisher's own FP8 export (every ``Linear`` except ``lm_head``). Not quantized: ``lm_head``,
the token embedding and the grouped RMSNorms. The attention of this family is an explicit
matmul and softmax (no fused attention layer), so an FP8 Quantize after it builds on TensorRT
11.4 and 11.1.

Projections whose entry in the scale file is ``null`` stay in bf16 on purpose: every projection
of the first two decoder layers (most of the FP8 loss of K2-Horizon-7B is there) and a projection
whose input has a massive activation (a scale more than 20 times the median of the same
projection); see ``calibrate_fp8_scales.recipe_scales``. A projection with no entry in the file is
an error, never a silent bf16 fallback.
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

# HF projection stem -> internal weight-dict stem used by model.py.
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

    The model config alone does not identify a checkpoint (fine-tunes share it), so the key ends
    with the numeric sum of the final RMSNorm weight (``n<sum>``).
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


def _scale_const(network, value: float, keep_alive: list):
    """Add a one-element fp32 scale constant to the network."""
    array = np.asarray([value], dtype=np.float32)
    keep_alive.append(array)
    return network.add_constant((1,), trt.Weights(array)).get_output(0)


def _wrap_fp8_matmul(network, activation, scales: _LayerScales, *, keep_alive: list, name: str):
    """Add the FP8 Q/DQ matmul for one projection; Dequantize and matmul use the activation type."""
    w = scales.weight
    n, k = w.out_features, w.in_features

    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(w.packed)
    w_fp8 = trt.Weights(trt.DataType.FP8, w.packed.ctypes.data, n * k)
    w_const = network.add_constant((n, k), w_fp8).get_output(0)
    w_dequant = network.add_dequantize(
        w_const, _scale_const(network, w.weight_scale, keep_alive), activation.dtype
    )

    act_scale = _scale_const(network, scales.input_scale, keep_alive)
    a_quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
    a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, activation.dtype)

    layer = network.add_matrix_multiply(
        a_dequant.get_output(0),
        trt.MatrixOperation.NONE,
        w_dequant.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    )
    if layer is None:
        raise RuntimeError(f"TensorRT failed to add K2-Horizon FP8 matmul {name}")
    layer.name = name
    return layer.get_output(0)


@dataclass(frozen=True)
class K2HorizonQuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    keep_alive: list = field(default_factory=list)

    def should_quantize(self, weight_name: str) -> bool:
        """Return True when this weight has a calibrated activation scale."""
        return weight_name in self.scales

    def quantized_matmul(self, network, lhs, weight_name: str, *, name: str):
        """Add the FP8 matmul of one projection that has a calibrated scale."""
        return _wrap_fp8_matmul(
            network, lhs, self.scales[weight_name], keep_alive=self.keep_alive, name=name
        )


def configure_fp8_build(trt_config, quant_ctx) -> None:
    """Apply the FP8 build options; a no-op for an unquantized build."""
    if quant_ctx is None:
        return
    # Keeps per-layer tactic names in the plan so FP8 lowering can be inspected.
    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    # TensorRT's fused dual-GEMM (gate/up) kernel is switched off: it does not compile for an
    # FP8 consumer of the SwiGLU output on TensorRT 11.1 in other families. The route is set only
    # when this TensorRT build lists the knob.
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
        raise RuntimeError(f"K2-Horizon FP8 build requires {path}, which was not found") from error
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


def _bf16_bits_to_fp32(array: np.ndarray) -> np.ndarray:
    """Return the float32 values of a weight stored as BF16 bit patterns (``uint16``)."""
    array = np.asarray(array)
    if array.dtype == np.uint16:
        return array.view(ml_dtypes.bfloat16).astype(np.float32)
    return array.astype(np.float32)


def calibrate_k2_horizon_fp8(config, weights) -> K2HorizonQuantContext:
    """Build the Q/DQ context from the loaded K2-Horizon weights (``[in, out]`` BF16 bits).

    Quantizes one projection at a time, so peak extra memory is one weight tensor. A projection
    with no stored activation scale, or a stored scale for a projection this model config does
    not quantize, is an error.
    """
    raw = getattr(config, "raw", {})
    if (isinstance(raw, dict) and raw.get("quantization_config")) or any(
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
            continue  # massive-activation projection, left in bf16 by the calibration
        if not np.isfinite(input_scale) or input_scale <= 0:
            raise ValueError(f"activation scale of {name} must be finite and positive")
        if name not in weights:
            raise ValueError(f"checkpoint has no {name}")
        transposed = np.asarray(weights[name])
        in_features, out_features = transposed.shape
        packed, weight_scale = quantize_fp8_weight(
            np.ascontiguousarray(_bf16_bits_to_fp32(transposed).T)
        )
        scales[name] = _LayerScales(
            input_scale=float(input_scale),
            weight=_FP8Weight(packed, weight_scale, out_features, in_features),
        )
    if not scales:
        raise RuntimeError("K2-Horizon FP8 self-quantization matched no projection")
    return K2HorizonQuantContext(scales=scales)
