# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma 3n-owned FP8 self-quantization of the text decoder.

No ModelOpt FP8 checkpoint of Gemma 3n exists, so ``quantization="fp8"``
quantizes the original checkpoint at build time, one projection at a time.
Weights are never stored in FP8.

Recipe (W8A8, per-tensor scales): the gate, up and down projections of every
decoder layer. The attention projections (q, k, v, o) stay BF16: in FP8 they
raise the KL divergence of the decoder against the original model by 40% and
shorten the decode step by about 1%. The MobileNetV5 tower, the multimodal
embedders, the token and per-layer embeddings, the tied language-model head,
the AltUp, LAuReL and per-layer input projections and the norms keep the build
precision as well. A projection whose calibrated scale is ``null`` (an outlier
input, more than 20 times the median of its kind) stays BF16.

Weight scale ``amax(W) / 448``; the activation scale is static, calibrated
once offline with ModelOpt ``FP8_DEFAULT_CFG`` (max) by ``calibrate_scales.py``
and read from ``fp8_activation_scales.json``. Users do not run calibration; a
checkpoint without scales raises. Each quantized projection becomes::

    activation -> Quantize -> Dequantize(bf16) -> MatMul
    FP8 weight -> Dequantize(bf16) ----------------^
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
QUANTIZATIONS = ("fp8",)
IDENTITY_VALUES = 8
_TEXT = "model.language_model"
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"
# HF stems (under ``model.language_model.layers.N.``) of the projections the FP8 recipe quantizes.
PROJECTIONS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def scale_key(text) -> str:
    """Key of one Gemma 3n text decoder config in the activation-scale file."""
    sizes = "_".join(str(int(size)) for size in sorted(set(text.intermediate_sizes)))
    return f"gemma3n-h{int(text.hidden_size)}-l{int(text.num_layers)}-i{sizes}"


def final_norm_head(checkpoint) -> list[float]:
    """The first 8 values of the final norm weight (float32), which tell checkpoints apart."""
    norm = np.asarray(checkpoint.get(f"{_TEXT}.norm.weight"), dtype=np.float32).reshape(-1)
    return [float(value) for value in norm[:IDENTITY_VALUES]]


def load_activation_scales(text, checkpoint) -> dict[str, float | None]:
    """The committed FP8 activation scales of this checkpoint, or raise.

    Checkpoints of the same decoder config (for example the base and the
    instruction-tuned E4B) have other weights and activation ranges, so an
    entry also stores the first values of the final norm weight of the
    calibrated checkpoint, and another checkpoint is refused.
    """
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    key = scale_key(text)
    entry = table.get(key)
    if entry is None or entry["final_norm_head"] != final_norm_head(checkpoint):
        raise NotImplementedError(
            f"Gemma 3n has no calibrated FP8 activation scales for this checkpoint ({key}); "
            f"calibrated configs: {sorted(table)}. Run calibrate_scales.py for it."
        )
    return entry["scales"]


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
    """Static FP8 activation scale of each quantized projection.

    Keys are checkpoint prefixes, ``model.language_model.layers.N.<stem>``; a
    projection without a key keeps the build precision.
    """

    input_scales: dict[str, float]
    keep_alive: list = field(default_factory=list)

    def configure(self, trt_config) -> None:
        """Keep per-layer names in the plan and turn the fused gate/up kernel off.

        With the fused gate/up (dual GEMM) kernel the FP8 decode step is 7%
        slower on B200 and equal on B300 (TensorRT 11.4). The route is set only
        when this TensorRT build lists the knob.
        """
        trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        if _DUAL_GEMM_KNOB in getattr(trt_config, "all_build_routes", ""):
            trt_config.build_route = f"-{_DUAL_GEMM_KNOB}=off"

    def matmul(self, builder, value, prefix: str, shape):
        """``value @ W.T`` as an FP8 Q/DQ matmul, or None when ``prefix`` is not quantized."""
        input_scale = self.input_scales.get(prefix)
        if input_scale is None:
            return None
        network, dtype = builder.net, value.dtype
        packed, weight_scale = quantize_fp8_weight(
            builder.checkpoint.get_f32(prefix + ".weight", shape)
        )
        rows, columns = packed.shape
        scales = (np.asarray([weight_scale], np.float32), np.asarray([input_scale], np.float32))
        # trt.Weights and constants keep raw pointers: the buffers must outlive the build.
        self.keep_alive += [packed, *scales]
        weight = network.add_constant(
            (rows, columns), trt.Weights(trt.DataType.FP8, packed.ctypes.data, packed.size)
        )
        w_scale, a_scale = (
            network.add_constant((1,), trt.Weights(scale)).get_output(0) for scale in scales
        )
        w = network.add_dequantize(weight.get_output(0), w_scale, dtype).get_output(0)
        q = network.add_quantize(value, a_scale, trt.DataType.FP8).get_output(0)
        x = network.add_dequantize(q, a_scale, dtype).get_output(0)
        return network.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, w, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)


def decoder_quantization(text, checkpoint, quantization: str) -> DecoderQuantization:
    """The FP8 recipe for this checkpoint: one activation scale per quantized projection."""
    if quantization not in QUANTIZATIONS:
        raise ValueError(f"Gemma 3n quantization must be none or fp8, not {quantization!r}")
    if any(
        name.endswith((".weight_scale", ".input_scale", ".weight_scale_2"))
        for name in checkpoint.mapping
    ):
        raise ValueError(
            "Gemma 3n checkpoint is already quantized; FP8 self-quantization needs "
            "the original checkpoint"
        )
    scales = load_activation_scales(text, checkpoint)
    input_scales = {}
    for layer in range(int(text.num_layers)):
        for stem in PROJECTIONS:
            name = f"layer.{layer}.{stem}"
            if name not in scales:
                raise ValueError(f"Gemma 3n activation scale file has no entry for {name}")
            scale = scales[name]
            if scale is None:
                continue
            if isinstance(scale, bool) or not (np.isfinite(scale) and scale > 0):
                raise ValueError(f"Gemma 3n activation scale of {name} must be finite and positive")
            input_scales[f"{_TEXT}.layers.{layer}.{stem}"] = float(scale)
    return DecoderQuantization(input_scales)
