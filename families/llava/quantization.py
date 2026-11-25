# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLaVA-owned FP8 self-quantization of the Llama decoder.

No ModelOpt FP8 checkpoint of LLaVA 1.5 exists, so ``quantization="fp8"``
quantizes the original checkpoint while the prefill and decode engines are
built, one projection at a time. Weights are never stored in FP8.

Recipe (W8A8, per-tensor scales): the q, k, v, o, gate, up and down projections
of every decoder layer. The CLIP tower, the projector, the token embedding, the
language-model head and the norms keep the build precision (FP16). A projection
whose calibrated scale is ``null`` (an outlier input, more than 20 times the
median of its kind: the down projection of layer 1) stays FP16.

Weight scale ``amax(W) / 448``; the activation scale is static, calibrated once
offline with ModelOpt ``FP8_DEFAULT_CFG`` (max) by ``calibrate_scales.py`` and
read from ``fp8_activation_scales.json``. Users do not run calibration; a
checkpoint without scales raises. Each quantized projection becomes::

    activation -> Quantize -> Dequantize(fp16) -> MatMul
    FP8 weight -> Dequantize(fp16) ----------------^

The attention is explicit (FP32 scores, softmax, matmul), so the FP8 Quantize
in front of the attention output projection builds on TensorRT 11.4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tensorrt as trt

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"
QUANTIZATIONS = ("fp8",)
IDENTITY_VALUES = 8
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"
# HF stem under ``language_model.layers.N.`` -> builder weight stem.
PROJECTIONS = (
    ("self_attn.q_proj", "w_q"),
    ("self_attn.k_proj", "w_k"),
    ("self_attn.v_proj", "w_v"),
    ("self_attn.o_proj", "w_o"),
    ("mlp.gate_proj", "w_gate"),
    ("mlp.up_proj", "w_up"),
    ("mlp.down_proj", "w_down"),
)


def scale_key(config) -> str:
    """Key of one Llama decoder config in the activation-scale file."""
    return (
        f"llava-h{int(config.hidden_size)}-l{int(config.num_hidden_layers)}"
        f"-i{int(config.intermediate_size)}"
    )


def final_norm_head(weights) -> list[float]:
    """The first 8 values of the final norm weight (float32), which tell checkpoints apart."""
    norm = np.asarray(weights["final_norm"], dtype=np.float32).reshape(-1)
    return [float(value) for value in norm[:IDENTITY_VALUES]]


def load_activation_scales(config, weights) -> dict[str, float | None]:
    """The committed FP8 activation scales of this checkpoint, or raise.

    Fine-tunes of LLaVA 1.5 7B share the decoder config but not the weights or
    the activation ranges, so an entry also stores the first values of the
    final norm weight of the calibrated checkpoint, and another checkpoint is
    refused.
    """
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    key = scale_key(config)
    entry = table.get(key)
    if entry is None or entry["final_norm_head"] != final_norm_head(weights):
        raise NotImplementedError(
            f"LLaVA has no calibrated FP8 activation scales for this checkpoint ({key}); "
            f"calibrated configs: {sorted(table)}. Run calibrate_scales.py for it."
        )
    return entry["scales"]


def quantize_fp8_weight(weight) -> tuple[np.ndarray, float]:
    """Raw e4m3 bytes of ``weight`` ``[out, in]`` and its per-tensor scale ``amax / 448``."""
    import torch

    tensor = torch.as_tensor(np.ascontiguousarray(weight)).float()
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    amax = float(tensor.abs().max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scale = amax / FP8_E4M3_MAX
    scaled = (tensor / torch.tensor(scale, dtype=torch.float32)).clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy()
    return np.ascontiguousarray(packed), scale


def _scalar(network, value: float, keep_alive: list):
    array = np.asarray([value], dtype=np.float32)
    keep_alive.append(array)
    return network.add_constant((1,), array).get_output(0)


@dataclass
class LlavaQuantContext:
    """Static FP8 activation scale of each quantized projection.

    Keys are builder weight names, ``layer.N.w_q``; a projection without a key
    (and the language-model head ``w_out``) keeps the build precision. The
    prefill and decode engines share one context, so each weight is quantized
    once per build.
    """

    input_scales: dict[str, float]
    keep_alive: list = field(default_factory=list)
    _packed: dict = field(default_factory=dict)

    def configure(self, trt_config) -> None:
        """Keep per-layer names in the plan and turn the fused gate/up kernel off.

        TensorRT 11.1 fails to compile the fused gate/up (dual GEMM) kernel
        whose output feeds an FP8 Quantize, and on TensorRT 11.4 the build
        without it decodes 19% faster on B200 (equal on B300). The route is set
        only when this TensorRT build lists the knob.
        """
        trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        if _DUAL_GEMM_KNOB in getattr(trt_config, "all_build_routes", ""):
            trt_config.build_route = f"-{_DUAL_GEMM_KNOB}=off"

    def maybe_quantized_matmul(
        self, network, lhs, lhs_width, rhs_width, rhs_weights, weight_name, dtype=np.float32
    ):
        """``lhs @ rhs_weights`` (``[in, out]``) as an FP8 Q/DQ matmul when ``weight_name`` is listed."""
        input_scale = self.input_scales.get(weight_name)
        if input_scale is None:
            from .graph_ops import add_matmul_rhs_constant

            return add_matmul_rhs_constant(
                network, lhs, lhs_width, rhs_width, rhs_weights, dtype=dtype
            )
        packed = self._packed.get(weight_name)
        if packed is None:
            packed = quantize_fp8_weight(np.asarray(rhs_weights).T)
            self._packed[weight_name] = packed
        weight, weight_scale = packed
        rows, columns = weight.shape
        # trt.Weights and constants keep raw pointers: the buffers must outlive the build.
        self.keep_alive.append(weight)
        constant = network.add_constant(
            (rows, columns), trt.Weights(trt.DataType.FP8, weight.ctypes.data, weight.size)
        ).get_output(0)
        out_dtype = lhs.dtype
        w = network.add_dequantize(
            constant, _scalar(network, weight_scale, self.keep_alive), out_dtype
        ).get_output(0)
        a_scale = _scalar(network, input_scale, self.keep_alive)
        q = network.add_quantize(lhs, a_scale, trt.DataType.FP8).get_output(0)
        x = network.add_dequantize(q, a_scale, out_dtype).get_output(0)
        return network.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, w, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)


def decoder_quantization(config, weights, quantization: str) -> LlavaQuantContext:
    """The FP8 recipe for this checkpoint: one activation scale per quantized projection."""
    if quantization not in QUANTIZATIONS:
        raise ValueError(f"LLaVA quantization must be none or fp8, not {quantization!r}")
    scales = load_activation_scales(config, weights)
    input_scales = {}
    for layer in range(int(config.num_hidden_layers)):
        for _, stem in PROJECTIONS:
            name = f"layer.{layer}.{stem}"
            if name not in scales:
                raise ValueError(f"LLaVA activation scale file has no entry for {name}")
            scale = scales[name]
            if scale is None:
                continue
            if isinstance(scale, bool) or not (np.isfinite(scale) and scale > 0):
                raise ValueError(f"LLaVA activation scale of {name} must be finite and positive")
            input_scales[name] = float(scale)
    return LlavaQuantContext(input_scales)
