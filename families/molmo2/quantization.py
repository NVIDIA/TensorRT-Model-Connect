# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Molmo2-owned FP8 self-quantization of the Qwen3-derived text decoder.

No ModelOpt FP8 checkpoint of Molmo2 exists, so the original checkpoint is
quantized at build time, one projection at a time. Only the decoder projections
are quantized: the fused ``att_proj`` (split into q, k and v, one shared input
scale), ``attn_out``, the fused ``ff_proj`` (split into up and gate, one shared
input scale) and ``ff_out``. The SigLIP tower, the pooling attention, the image
projector, the embeddings, the language-model head and all norms keep BF16.

FP8 (W8A8): ``weight_scale = amax(W) / 448`` per projection, computed from the
checkpoint on every build; the activation scale is static, calibrated once
offline with ModelOpt ``FP8_DEFAULT_CFG`` (max) by ``calibrate_scales.py`` and
read from ``fp8_activation_scales.json``::

    activation -> Quantize -> Dequantize(bf16) -> MatMul
    FP8 weight -> Dequantize(bf16) -----------------^

A projection whose calibrated scale is ``null`` (an outlier input) keeps BF16.
Users do not run calibration; an unknown model config or another checkpoint of
the same config raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import ml_dtypes
import numpy as np

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"
# Checkpoint projection stems of one decoder block, in graph order.
PROJECTIONS = ("self_attn.att_proj", "self_attn.attn_out", "mlp.ff_proj", "mlp.ff_out")
QUANTIZATIONS = ("fp8",)
# Scales are tied to one checkpoint through the first values of its final norm weight.
IDENTITY_TENSOR = "model.transformer.ln_f.weight"
IDENTITY_VALUES = 8
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"


def scale_key(text: dict) -> str:
    """Key of one decoder config in the activation-scale file."""
    return (
        f"molmo2-h{int(text['hidden_size'])}-l{int(text['num_hidden_layers'])}"
        f"-i{int(text['intermediate_size'])}"
    )


def checkpoint_identity(weights) -> list[float]:
    """The first values of the final norm weight (BF16-rounded), which tell checkpoints apart."""
    values = np.asarray(weights(IDENTITY_TENSOR), np.float32).reshape(-1)[:IDENTITY_VALUES]
    return [float(v) for v in values.astype(ml_dtypes.bfloat16).astype(np.float32)]


def load_activation_scales(text: dict, identity: list[float]) -> dict[str, float | None]:
    """The committed FP8 activation scales of this decoder config and checkpoint, or raise."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    key = scale_key(text)
    if key not in table:
        raise NotImplementedError(
            f"Molmo2 has no calibrated activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_scales.py for this model config."
        )
    if table[key]["identity"] != list(identity):
        raise NotImplementedError(
            f"Molmo2 activation scales for {key!r} were calibrated on another checkpoint "
            f"({table[key]['checkpoint']}); run calibrate_scales.py for this checkpoint."
        )
    return table[key]["scales"]


def quantize_fp8_weight(weight: np.ndarray) -> tuple[np.ndarray, float]:
    """Raw e4m3 bytes of ``weight`` and its per-tensor scale ``amax / 448``."""
    weight = np.asarray(weight, dtype=np.float32)
    amax = float(np.abs(weight).max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scale = amax / FP8_E4M3_MAX
    scaled = np.clip(weight / np.float32(scale), -FP8_E4M3_MAX, FP8_E4M3_MAX)
    return np.ascontiguousarray(scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)), scale


@dataclass
class DecoderQuantization:
    """FP8 activation scale of each quantized projection.

    Keys are checkpoint prefixes, ``model.transformer.blocks.N.<stem>``.
    """

    input_scales: dict[str, float]
    keep_alive: list = field(default_factory=list)

    def configure(self, trt_config) -> None:
        """Keep layer names in the plan and turn the fused gate/up kernel off.

        TensorRT 11.1 cannot compile the fused gate/up kernel when its BF16
        output feeds an FP8 Quantize; the knob exists only where TensorRT lists it.
        """
        import tensorrt as trt

        trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        if _DUAL_GEMM_KNOB in getattr(trt_config, "all_build_routes", ""):
            trt_config.build_route = f"-{_DUAL_GEMM_KNOB}=off"

    def linear(self, graph, value, weight, prefix):
        """``value @ W.T`` for a quantized projection, or None when ``prefix`` keeps BF16."""
        scale = self.input_scales.get(prefix)
        if scale is None:
            return None
        return _fp8_matmul(graph.net, value, weight, scale, self.keep_alive)


def _scalar(network, value: float):
    array = np.asarray([value], dtype=np.float32)
    return network.add_constant((1,), array).get_output(0), array


def _fp8_matmul(network, value, weight, input_scale: float, keep_alive: list):
    import tensorrt as trt

    dtype = value.dtype
    packed, weight_scale = quantize_fp8_weight(weight)
    rows, columns = packed.shape
    keep_alive.append(packed)
    constant = network.add_constant(
        (rows, columns), trt.Weights(trt.DataType.FP8, packed.ctypes.data, packed.size)
    ).get_output(0)
    w_scale, a = _scalar(network, weight_scale)
    a_scale, b = _scalar(network, input_scale)
    keep_alive += [a, b]
    w = network.add_dequantize(constant, w_scale, dtype).get_output(0)
    q = network.add_quantize(value, a_scale, trt.DataType.FP8).get_output(0)
    x = network.add_dequantize(q, a_scale, dtype).get_output(0)
    return network.add_matrix_multiply(
        x, trt.MatrixOperation.NONE, w, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)


def checkpoint_is_quantized(config: dict, weight_names) -> bool:
    """True for a checkpoint that already carries quantized weights or quantization metadata."""
    return bool(
        config.get("quantization_config")
        or config.get("text_config", {}).get("quantization_config")
        or any(
            name.endswith((".weight_scale", ".input_scale", ".weight_scale_2", ".weight_packed"))
            for name in weight_names
        )
    )


def decoder_quantization(config: dict, weights, quantization: str) -> DecoderQuantization:
    """The decoder projections to quantize for ``quantization`` (only ``fp8``).

    ``weights`` is the family checkpoint reader (``model.Weights``).
    """
    if quantization not in QUANTIZATIONS:
        raise ValueError(f"Molmo2 quantization must be none or fp8, not {quantization!r}")
    if checkpoint_is_quantized(config, weights.index):
        raise ValueError(
            "Molmo2 checkpoint is already quantized; self-quantization needs the original "
            "checkpoint"
        )
    text = config["text_config"]
    scales = load_activation_scales(text, checkpoint_identity(weights))
    input_scales = {}
    for layer in range(int(text["num_hidden_layers"])):
        for stem in PROJECTIONS:
            name = f"layer.{layer}.{stem}"
            if name not in scales:
                raise ValueError(f"Molmo2 activation scale file has no entry for {name}")
            scale = scales[name]
            if scale is None:
                continue
            if not (np.isfinite(scale) and scale > 0):
                raise ValueError(f"Molmo2 activation scale of {name} must be finite and positive")
            input_scales[f"model.transformer.blocks.{layer}.{stem}"] = float(scale)
    return DecoderQuantization(input_scales)
