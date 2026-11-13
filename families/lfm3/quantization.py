# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LFM3-owned FP8 self-quantization and TensorRT Q/DQ graph context.

The original BF16 LFM3 checkpoint is used as-is; no FP8 checkpoint is needed.
LiquidAI publishes no FP8 or NVFP4 LFM3 checkpoint.

Weights are quantized on the fly at build time, one projection at a time:
``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3 value), and
the FP8 bytes are fed to the engine as a constant. The activation scale of each
projection input needs real data, so it was calibrated once offline with
``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``) and is read from
``fp8_activation_scales.json``, keyed by the model config and a weight
fingerprint. Users do not run calibration.

Each quantized projection becomes::

    activation -> Quantize -> Dequantize(activation type) -> MatMul
    FP8 weight -> Dequantize(activation type) ----------------^

Quantized: the attention q/k/v projections, the SwiGLU w1/w2/w3 projections and
the conv-block in/out projections that surround the depthwise short convolution.
Not quantized: the depthwise short convolution and its state, the per-head
attention gate projection (one output per head), the attention output projection
(it builds as FP8 here, but adds most of the held-out perplexity loss for about
2% of decode time), ``lm_head``, the embedding and the norms. A projection whose calibrated scale was an outlier is
stored as ``null`` in the JSON file and stays unquantized.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np
import tensorrt as trt

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"
CALIBRATION_TOOL = "families/lfm3/calibrate_fp8_scales.py"

# HF projection stem -> weight-dict stem used by model.py, per layer kind.
CONV_PROJECTIONS = (
    ("conv.in_proj", "conv_in"),
    ("conv.out_proj", "conv_out"),
)
ATTENTION_PROJECTIONS = (
    ("self_attn.q_proj", "w_q"),
    ("self_attn.k_proj", "w_k"),
    ("self_attn.v_proj", "w_v"),
)
MLP_PROJECTIONS = (
    ("feed_forward.w1", "w1"),
    ("feed_forward.w3", "w3"),
    ("feed_forward.w2", "w2"),
)
PROJECTIONS = CONV_PROJECTIONS + ATTENTION_PROJECTIONS + MLP_PROJECTIONS
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"


def layer_projections(layer_type: str) -> tuple[str, ...]:
    """Weight stems of the quantized projections that a layer of this type carries."""
    first = CONV_PROJECTIONS if layer_type == "conv" else ATTENTION_PROJECTIONS
    return tuple(stem for _, stem in first + MLP_PROJECTIONS)


def scale_key(config, final_norm: np.ndarray) -> str:
    """Key of this checkpoint's entry in the activation-scale file.

    ``config`` is the validated ``DenseLfm3Config``. The model config fields name the
    architecture; the sum of the final norm weight tells checkpoints (base, instruct,
    fine-tunes) of the same model config apart, since their activation ranges differ.
    """
    norm_sum = float(np.sum(np.asarray(final_norm, dtype=np.float32), dtype=np.float64))
    return (
        f"lfm3-h{int(config.hidden_size)}"
        f"-l{int(config.num_hidden_layers)}"
        f"-i{int(config.intermediate_size)}"
        f"-kv{int(config.num_key_value_heads)}"
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


def _scale_const(network, value: float):
    """Add a one-element fp32 scale constant to the network."""
    array = np.asarray([value], dtype=np.float32)
    return network.add_constant((1,), trt.Weights(array)).get_output(0)


def _wrap_fp8_matmul(network, activation, scales: _LayerScales, *, keep_alive: list, name: str):
    """Add the FP8 Q/DQ matmul for one projection; Dequantize writes the activation type.

    The FP8 GEMM then writes fp16 or bf16 directly, so no fp32 output and cast pass is
    added. A NaN scan on real prompts found no overflow with this layout.
    """
    w = scales.weight
    n, k = w.out_features, w.in_features
    dtype = activation.dtype

    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(w.packed)
    w_fp8 = trt.Weights(trt.DataType.FP8, w.packed.ctypes.data, n * k)
    w_const = network.add_constant((n, k), w_fp8).get_output(0)
    w_dequant = network.add_dequantize(w_const, _scale_const(network, w.weight_scale), dtype)

    act_scale = _scale_const(network, scales.input_scale)
    a_quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
    a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, dtype)

    layer = network.add_matrix_multiply(
        a_dequant.get_output(0),
        trt.MatrixOperation.NONE,
        w_dequant.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    )
    if layer is None:
        raise RuntimeError(f"TensorRT failed to add LFM3 FP8 matmul {name}")
    layer.name = name
    return layer.get_output(0)


@dataclass(frozen=True)
class Lfm3QuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        """Give each context its own list for buffers that must outlive the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, weight_name: str) -> bool:
        """Return True when this weight is quantized in this build."""
        return weight_name in self.scales

    def quantized_matmul(self, network, lhs, weight_name: str, *, name: str):
        """Add the projection as an FP8 matmul; the caller checks ``should_quantize`` first."""
        return _wrap_fp8_matmul(
            network, lhs, self.scales[weight_name], keep_alive=self.keep_alive, name=name
        )


def configure_fp8_build(trt_config, quant_ctx) -> None:
    """Apply the FP8 build options; a no-op for an unquantized build."""
    if quant_ctx is None:
        return
    # Keeps per-layer tactic names in the plan so FP8 lowering can be inspected.
    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    # TensorRT's fused dual-GEMM (w1/w3) kernel fails to compile with scalar-scale FP8
    # inputs. The route is set only when this TensorRT build lists the knob.
    if _DUAL_GEMM_KNOB in getattr(trt_config, "all_build_routes", ""):
        route = str(getattr(trt_config, "build_route", "") or "")
        knob = f"-{_DUAL_GEMM_KNOB}=off"
        trt_config.build_route = f"{route} {knob}".strip() if route else knob


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
        raise RuntimeError(f"LFM3 FP8 build requires {path}, which was not found") from error
    key = scale_key(config, final_norm)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for the LFM3 model config {key!r}; "
            f"available: {sorted(table)}. Run {CALIBRATION_TOOL} for this checkpoint."
        )
    return table[key]


def calibrate_lfm3_fp8(config, weights) -> Lfm3QuantContext:
    """Build the Q/DQ context from the loaded LFM3 weights (``[in, out]`` layout).

    ``config`` is the validated ``DenseLfm3Config``. Every quantizable projection of
    every layer must have an entry in the scale file: a positive scale (quantized) or
    ``null`` (outlier, kept in 16 bit). Quantizes one projection at a time, so peak
    extra memory is one weight tensor.
    """
    if any(str(key).endswith((".weight_scale", ".input_scale")) for key in weights):
        raise ValueError("the checkpoint is already quantized; FP8 self-quantization needs BF16")
    activation_scales = load_activation_scales(config, weights["final_norm"])

    expected = {
        f"layer.{layer}.{stem}"
        for layer, layer_type in enumerate(config.layer_types)
        for stem in layer_projections(layer_type)
    }
    unexpected = sorted(set(activation_scales) - expected)
    if unexpected:
        raise ValueError(f"LFM3 scale file has entries for unquantized projections: {unexpected}")

    scales: dict[str, _LayerScales] = {}
    for name in sorted(expected, key=lambda n: (int(n.split(".")[1]), n)):
        if name not in activation_scales:
            raise ValueError(f"LFM3 scale file has no entry for {name}; rerun {CALIBRATION_TOOL}")
        input_scale = activation_scales[name]
        if input_scale is None:
            continue  # outlier input: the projection stays in 16 bit
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
        raise RuntimeError("LFM3 FP8 self-quantization matched no projection in the checkpoint")
    return Lfm3QuantContext(scales=scales)
