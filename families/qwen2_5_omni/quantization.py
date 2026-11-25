# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen2.5 Omni-owned FP8 self-quantization of the Thinker text decoder.

No ModelOpt FP8 or NVFP4 checkpoint of Qwen2.5-Omni-7B exists, so
``quantization="fp8"`` quantizes the original checkpoint while the engine is
built, one projection at a time. Weights are never stored in FP8.

Recipe (W8A8, per-tensor scales): the q, k, v, o, gate, up and down projections
of every Thinker decoder layer. The token embedding, the language-model head,
the norms and the attention core keep the build precision (FP32). A projection
whose calibrated scale is ``null`` (an outlier input, more than 20 times the
median of its kind: the down projection of layer 26) stays FP32.

Weight scale ``amax(W) / 448``; q, k and v share one weight scale (the largest
amax of the three), as TensorRT runs the three projections of one input as one
GEMM. The activation scale is static, calibrated once offline with ModelOpt
``FP8_DEFAULT_CFG`` (max) by ``calibrate_scales.py`` and read from
``fp8_activation_scales.json``. Users do not run calibration; a checkpoint
without scales raises. Each quantized projection becomes::

    activation -> Quantize -> Dequantize(fp32) -> MatMul (+ bias)
    FP8 weight -> Dequantize(fp32) ----------------^

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
# Projection stems under ``model.layers.N.`` of the Thinker text decoder.
PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
_SHARED_INPUT = PROJECTIONS[:3]


def scale_key(cfg) -> str:
    """Key of one Thinker text decoder config in the activation-scale file."""
    return (
        f"qwen2_5_omni-h{int(cfg['hidden_size'])}-l{int(cfg['num_hidden_layers'])}"
        f"-i{int(cfg['intermediate_size'])}"
    )


def final_norm_head(weights) -> list[float]:
    """The first 8 values of the final norm weight (float32), which tell checkpoints apart."""
    norm = np.asarray(weights["model.norm.weight"], dtype=np.float32).reshape(-1)
    return [float(value) for value in norm[:IDENTITY_VALUES]]


def load_activation_scales(cfg, weights) -> dict[str, float | None]:
    """The committed FP8 activation scales of this checkpoint, or raise.

    Fine-tunes share the decoder config but not the weights or the activation
    ranges, so an entry also stores the first values of the final norm weight
    of the calibrated checkpoint, and another checkpoint is refused.
    """
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    key = scale_key(cfg)
    entry = table.get(key)
    if entry is None or entry["final_norm_head"] != final_norm_head(weights):
        raise NotImplementedError(
            f"Qwen2.5 Omni has no calibrated FP8 activation scales for this checkpoint ({key}); "
            f"calibrated configs: {sorted(table)}. Run calibrate_scales.py for it."
        )
    return entry["scales"]


def quantize_fp8_weight(weight, amax: float | None = None) -> tuple[np.ndarray, float]:
    """Raw e4m3 bytes of ``weight`` ``[out, in]`` and its per-tensor scale ``amax / 448``.

    ``amax`` defaults to the weight's own; projections that share a scale pass the shared one.
    """
    import torch

    tensor = torch.as_tensor(np.ascontiguousarray(weight)).float()
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    if amax is None:
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
class ThinkerQuantContext:
    """Static FP8 activation scale (and shared weight amax) of each quantized projection.

    Keys are checkpoint weight stems, ``model.layers.N.mlp.up_proj``; a
    projection without a key (and the language-model head) keeps FP32.
    """

    input_scales: dict[str, float]
    weight_amax: dict[str, float] = field(default_factory=dict)
    keep_alive: list = field(default_factory=list)

    def configure(self, trt_config) -> None:
        """Keep per-layer names in the plan and turn the fused gate/up kernel off.

        TensorRT 11.1 fails to compile the fused gate/up (dual GEMM) kernel
        whose output feeds an FP8 Quantize. The route is set only when this
        TensorRT build lists the knob.
        """
        trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        if _DUAL_GEMM_KNOB in getattr(trt_config, "all_build_routes", ""):
            trt_config.build_route = f"-{_DUAL_GEMM_KNOB}=off"

    def covers(self, key: str) -> bool:
        return key in self.input_scales

    def matmul(self, network, value, key: str, weight):
        """``value @ weight.T`` for the ``[out, in]`` FP32 ``weight`` of projection ``key``."""
        packed, weight_scale = quantize_fp8_weight(weight, self.weight_amax.get(key))
        rows, columns = packed.shape
        # trt.Weights and constants keep raw pointers: the buffers must outlive the build.
        self.keep_alive.append(packed)
        constant = network.add_constant(
            (rows, columns), trt.Weights(trt.DataType.FP8, packed.ctypes.data, packed.size)
        ).get_output(0)
        w = network.add_dequantize(
            constant, _scalar(network, weight_scale, self.keep_alive), value.dtype
        ).get_output(0)
        a_scale = _scalar(network, self.input_scales[key], self.keep_alive)
        q = network.add_quantize(value, a_scale, trt.DataType.FP8).get_output(0)
        x = network.add_dequantize(q, a_scale, value.dtype).get_output(0)
        return network.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, w, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)


def thinker_quantization(cfg, weights, quantization: str) -> ThinkerQuantContext:
    """The FP8 recipe for this checkpoint: one activation scale per quantized projection.

    q, k and v read the same input and TensorRT runs them as one GEMM; with a
    weight scale per projection the Qwen2.5-Omni-7B engine is wrong on TensorRT
    11.1 and 11.4 (each one alone is right), so the three share the largest
    weight amax of the layer.
    """
    if quantization not in QUANTIZATIONS:
        raise ValueError(f"Qwen2.5 Omni quantization must be none or fp8, not {quantization!r}")
    scales = load_activation_scales(cfg, weights)
    input_scales = {}
    for layer in range(int(cfg["num_hidden_layers"])):
        for stem in PROJECTIONS:
            name = f"model.layers.{layer}.{stem}"
            if name not in scales:
                raise ValueError(f"Qwen2.5 Omni activation scale file has no entry for {name}")
            scale = scales[name]
            if scale is None:
                continue
            if isinstance(scale, bool) or not (np.isfinite(scale) and scale > 0):
                raise ValueError(
                    f"Qwen2.5 Omni activation scale of {name} must be finite and positive"
                )
            input_scales[name] = float(scale)
    weight_amax = {}
    for layer in range(int(cfg["num_hidden_layers"])):
        names = [f"model.layers.{layer}.{stem}" for stem in _SHARED_INPUT]
        if all(name in input_scales for name in names):
            amax = max(float(np.abs(weights[name + ".weight"]).max()) for name in names)
            weight_amax.update(dict.fromkeys(names, amax))
    return ThinkerQuantContext(input_scales, weight_amax)
