# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-owned FP8 self-quantization and TensorRT Q/DQ graph context.

The original BF16 Nemotron checkpoint is used as-is; no FP8 checkpoint is needed.

Weights are quantized on the fly at build time, one projection at a time:
``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3 value), and
the FP8 bytes are fed to the engine as a constant. The weights are taken after
``_NemotronModel.load_weights``, so the quantized weight is the one the unquantized
engine would use. The activation
scale of each projection input needs real data, so it was calibrated once
offline with ``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``) and is
read from ``fp8_activation_scales.json``. Users do not run calibration.

Each quantized projection becomes::

    activation -> Quantize -> Dequantize -> MatMul
    FP8 weight -> Dequantize ----------------^

Both Dequantize layers output the activation type (fp16 or bf16), so the FP8 GEMM
writes fp16 and TensorRT fuses the bias and the squared ReLU into its epilogue.

Quantized projections of every decoder layer: ``q_proj``, ``k_proj``, ``v_proj``
and the two MLP projections (``up_proj``, ``down_proj``; Nemotron has no gate
projection). The attention output projection (``w_o``) stays unquantized:
TensorRT 11.4 cannot compile an FP8 Quantize right after a fused attention.
``lm_head``, the embedding and the layer norms stay unquantized as well.

The JSON file has one entry per projection and checkpoint (the key is the model config plus
the final norm sum, see ``scale_key``). Only Nemotron-Mini-4B-Instruct is calibrated:
Nemotron-4-Mini-Hindi-4B-Base has the same model config but different activation ranges, so it
raises until it is calibrated. A projection with
no entry raises an error that names the calibration tool. A projection whose
entry is ``null`` was left unquantized on purpose by the calibration tool
(calibrated scale above 20 times the median of its kind).
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
    ("mlp.up_proj", "w_fc1"),
    ("mlp.down_proj", "w_fc2"),
)
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"


def scale_key(config, final_norm: np.ndarray) -> str:
    """Key of this checkpoint's entry in the activation-scale file.

    ``config`` is the model config. Nemotron-Mini-4B-Instruct and Nemotron-4-Mini-Hindi-4B-Base
    have the same config but different activation ranges (their calibrated scales differ
    by 0.62x to 2.15x), so the key adds the sum of the final norm weight (after the +1
    offset of the layer norm), which tells the checkpoints apart.
    """
    norm_sum = float(np.sum(np.asarray(final_norm, dtype=np.float32), dtype=np.float64))
    return (
        f"{str(config.model_type).lower()}"
        f"-h{int(config.hidden_size)}"
        f"-l{int(config.num_hidden_layers)}"
        f"-i{int(config.intermediate_size)}"
        f"-n{norm_sum:.4f}"
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
    """Add the FP8 Q/DQ matmul for one projection; Dequantize and matmul use the activation type.

    A fp32 Dequantize makes the GEMM write fp32 and adds a cast pass (decode 0.99x of the fp16
    engine, against 1.07x to 1.09x with the activation type). The risk of fp16 is overflow
    above 65504 in a projection output. A scan of the outputs of all 160 projections over 528
    real prompts (mean 1010 tokens) found no non-finite value: the largest output was 4480,
    6.8% of 65504 (``down_proj`` of layer 29). bf16 has the fp32 exponent range.
    """
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
class NemotronQuantContext:
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
    # TensorRT's fused dual-GEMM (gate/up) kernel fails to compile with scalar-scale FP8
    # inputs. The route is set only when this TensorRT build lists the knob.
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


def load_activation_scales(config, final_norm: np.ndarray) -> dict[str, float | None]:
    """Return the committed activation scales for this checkpoint, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Nemotron FP8 build requires {path}, which was not found") from error
    key = scale_key(config, final_norm)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this checkpoint."
        )
    return table[key]


def calibrate_nemotron_fp8(config, weights, graph_ops) -> NemotronQuantContext:
    """Build the Q/DQ context from the loaded Nemotron weights (``[in, out]`` layout).

    Quantizes one projection at a time, so peak extra memory is one weight tensor.
    """
    if any(str(key).endswith((".weight_scale", ".input_scale")) for key in weights):
        raise ValueError("the checkpoint is already quantized; FP8 self-quantization needs BF16")
    activation_scales = load_activation_scales(config, weights["final_norm"])

    scales: dict[str, _LayerScales] = {}
    for layer in range(int(config.num_hidden_layers)):
        for _, weight_stem in PROJECTIONS:
            name = f"layer.{layer}.{weight_stem}"
            if name not in activation_scales:
                raise ValueError(
                    f"{name} has no activation scale in {ACTIVATION_SCALES_FILENAME}; "
                    "run calibrate_fp8_scales.py"
                )
            input_scale = activation_scales[name]
            if input_scale is None:
                continue  # left unquantized on purpose (outlier), see the module docstring
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
        raise RuntimeError("Nemotron FP8 self-quantization matched no projection in the checkpoint")
    return NemotronQuantContext(scales=scales, graph_ops=graph_ops)
