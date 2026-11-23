# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-Music3 FP8 self-quantization and TensorRT Q/DQ helpers.

The original checkpoint is used as is; no FP8 checkpoint exists for this model. Weights are
quantized on every build, one projection at a time: ``scale = max(abs(weight)) / 448`` (448 is
the largest finite e4m3 value), and the FP8 bytes go into the engine as a constant. The static
activation scale of each projection input was calibrated once with
``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``, max calibration, whole generations)
and is read from ``fp8_activation_scales.json``. Users do not run calibration.

Three engines carry FP8 projections:

* ``language_model`` (the Qwen3 decoder that runs once per frame and branch): ``w_q``, ``w_k``,
  ``w_v``, ``w_gate``, ``w_up``, ``w_down`` of every layer. The decoder runs in bf16, so the
  Dequantize writes bf16 (the float32 exponent range, no fp16 overflow).
* ``depth_decoder`` (the RVQ depth decoder that runs fifteen times per frame): the attention
  and MLP projections of its four layers. The blocks run in bf16 as in the unquantized build.
* ``dit`` (the flow-matching transformer that runs sixty times per window): ``ff_in`` and
  ``ff_out`` of its 36 blocks. The blocks run in bf16 as in the unquantized build. The attention
  projections stay bf16: quantizing them as well doubled the latent error of a denoised window.

Each quantized projection becomes::

    activation -> Quantize -> Dequantize(activation type) -> MatMul
    FP8 weight [out, in] -> Dequantize(activation type) ----^ (transposed)

Not quantized: the language model's attention output projection ``w_o`` (an FP8 Quantize
after its fused attention does not build on TensorRT 11.4), embeddings, the language model
head, the depth decoder's input projection and output heads, the DiT attention projections, stem
convolutions, ``proj_in``, ``proj_out`` and timestep embedding, the condition encoder and the
vocoder, every norm. A projection whose stored scale is ``null`` (more
than 20 times the median of its kind at calibration) stays unquantized on purpose; a projection
with no entry is an error, never a silent fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"
SUPPORTED_QUANTIZATIONS = ("fp8",)

#: Language-model projections (the decoder's weight-dict stems).
LM_PROJECTIONS = ("w_q", "w_k", "w_v", "w_gate", "w_up", "w_down")
#: Depth-decoder projections (checkpoint names without ``.weight``).
DEPTH_PROJECTIONS = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out",
    "gate_proj",
    "up_proj",
    "down_proj",
)
#: DiT block projections (checkpoint names without ``.weight``).
DIT_PROJECTIONS = ("ff_in", "ff_out")

_DUAL_GEMM_KNOB = "peep:match_dual_gemm"


def _lm_names() -> list[str]:
    from .language_model import NUM_HIDDEN_LAYERS

    return [
        f"layer.{layer}.{stem}" for layer in range(NUM_HIDDEN_LAYERS) for stem in LM_PROJECTIONS
    ]


def _depth_names() -> list[str]:
    from .depth_decoder import NUM_LAYERS

    return [f"layers.{layer}.{stem}" for layer in range(NUM_LAYERS) for stem in DEPTH_PROJECTIONS]


def _dit_names() -> list[str]:
    from .dit import NUM_LAYERS

    return [
        f"transformer_blocks.{layer}.{stem}"
        for layer in range(NUM_LAYERS)
        for stem in DIT_PROJECTIONS
    ]


#: Calibration component -> every projection name it quantizes.
ENGINE_PROJECTIONS = {
    "language_model": _lm_names,
    "rvq_depth_decoder": _depth_names,
    "transformer": _dit_names,
}

# Component -> (model-config part of the key, tensor whose sum fingerprints the weights).
_KEY_PARTS = {
    "language_model": ("lm-h4096-l36-i12288", "model.norm.weight"),
    "rvq_depth_decoder": ("depth-h4096-l4-i6144", "norm.weight"),
    "transformer": ("dit-h2048-l36-f8192", "transformer_blocks.35.norm2.weight"),
}


def scale_key(component: str, tensors: dict) -> str:
    """Key of one component's entry in the scale file: model config plus a weight fingerprint.

    The fingerprint is the numeric sum of one norm weight, so a fine-tune with the same model
    config gets its own key (and an error until it is calibrated).
    """
    config, tensor = _KEY_PARTS[component]
    if tensor not in tensors:
        raise ValueError(f"{component} checkpoint has no {tensor}")
    total = float(np.sum(np.asarray(tensors[tensor], dtype=np.float64)))
    return f"minimax_music3-{config}-n{total:.4f}"


def quantize_fp8_weight(weight_fp32: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize one ``[out, in]`` weight to FP8 e4m3 bytes with ``scale = amax / 448``."""
    import ml_dtypes

    amax = float(np.abs(weight_fp32).max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scale = amax / FP8_E4M3_MAX
    scaled = np.clip(np.asarray(weight_fp32, dtype=np.float32) / scale, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    return np.ascontiguousarray(scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)), scale


def load_activation_scales(component: str, tensors: dict) -> dict[str, float | None]:
    """Return the committed scales of one component, or raise when this checkpoint has none."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(
            f"MiniMax-Music3 FP8 build requires {path}, which was not found"
        ) from error
    key = scale_key(component, tensors)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: {sorted(table)}. "
            "Run calibrate_fp8_scales.py for this checkpoint."
        )
    scales = table[key]
    expected = ENGINE_PROJECTIONS[component]()
    unexpected = sorted(set(scales) - set(expected))
    if unexpected:
        raise ValueError(
            f"{ACTIVATION_SCALES_FILENAME} has a scale for {unexpected[0]}, which is not quantized"
        )
    missing = sorted(set(expected) - set(scales))
    if missing:
        raise ValueError(
            f"{missing[0]} has no activation scale in {ACTIVATION_SCALES_FILENAME}; "
            "run calibrate_fp8_scales.py"
        )
    for name, value in scales.items():
        if value is not None and not (np.isfinite(value) and value > 0):
            raise ValueError(f"activation scale of {name} must be finite and positive")
    return scales


def reject_quantized_checkpoint(tensors: dict) -> None:
    """Refuse a checkpoint that already carries quantization scales."""
    if any(
        str(key).endswith((".weight_scale", ".input_scale", ".weight_scale_inv")) for key in tensors
    ):
        raise ValueError(
            "the checkpoint is already quantized; FP8 self-quantization needs the original checkpoint"
        )


def add_fp8_linear(network, hidden, weight_out_in, input_scale: float, keep_alive: list):
    """Add ``hidden @ W.T`` as an FP8 Q/DQ matmul; Dequantize outputs ``hidden``'s type.

    ``weight_out_in`` is the checkpoint's ``[out, in]`` weight. The FP8 constant gets leading
    ones up to ``hidden``'s rank so the two operands of the matmul have the same rank.
    """
    import tensorrt as trt

    packed, weight_scale = quantize_fp8_weight(np.asarray(weight_out_in, dtype=np.float32))
    keep_alive.append(packed)  # trt.Weights holds a raw pointer until the build ends
    out_features, in_features = packed.shape
    shape = (1,) * (len(hidden.shape) - 2) + (out_features, in_features)
    weight = network.add_constant(
        shape, trt.Weights(trt.DataType.FP8, packed.ctypes.data, packed.size)
    ).get_output(0)

    def scale_const(value: float):
        array = np.asarray([value], dtype=np.float32)
        keep_alive.append(array)
        return network.add_constant((1,), trt.Weights(array)).get_output(0)

    weight_dq = network.add_dequantize(weight, scale_const(weight_scale), hidden.dtype).get_output(
        0
    )
    act_scale = scale_const(input_scale)
    quantized = network.add_quantize(hidden, act_scale, trt.DataType.FP8).get_output(0)
    hidden_dq = network.add_dequantize(quantized, act_scale, hidden.dtype).get_output(0)
    return network.add_matrix_multiply(
        hidden_dq, trt.MatrixOperation.NONE, weight_dq, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)


@dataclass
class BlockQuant:
    """FP8 projections of a hand-built graph (DiT or depth decoder), keyed by checkpoint name."""

    scales: dict[str, float | None]
    keep_alive: list = field(default_factory=list)
    quantized: list = field(default_factory=list)

    def linear(self, network, hidden, weight, name: str):
        """Return the FP8 matmul for ``name``, or None when the projection stays unquantized."""
        if name not in self.scales:
            return None
        scale = self.scales[name]
        if scale is None:
            return None
        self.quantized.append(name)
        return add_fp8_linear(network, hidden, weight, float(scale), self.keep_alive)


@dataclass
class LanguageModelQuantContext:
    """``quant_ctx`` of the family decoder: FP8 for the projections that have a scale."""

    scales: dict[str, float | None]
    graph_ops: Any
    keep_alive: list = field(default_factory=list)
    quantized: list = field(default_factory=list)

    def maybe_quantized_matmul(
        self,
        network,
        lhs,
        lhs_width: int,
        rhs_width: int,
        rhs_weights: np.ndarray,
        weight_name: str,
        dtype=np.float32,
    ):
        """Add the projection as an FP8 matmul, or unquantized when it has no scale."""
        scale = self.scales.get(weight_name)
        if scale is None:
            return self.graph_ops.add_matmul_rhs_constant(
                network, lhs, lhs_width, rhs_width, rhs_weights, dtype=dtype
            )
        self.quantized.append(weight_name)
        # The decoder weight dict holds [in, out]; the FP8 constant is [out, in].
        weight = np.ascontiguousarray(np.asarray(rhs_weights, dtype=np.float32).T)
        return add_fp8_linear(network, lhs, weight, float(scale), self.keep_alive)


def configure_fp8_build(trt_config) -> None:
    """Build options of an FP8 DiT or depth-decoder plan."""
    import tensorrt as trt

    # Keeps per-layer tactic names in the plan so the FP8 lowering can be inspected.
    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED


def configure_lm_fp8_build(trt_config) -> None:
    """Build options of an FP8 language-model plan."""
    configure_fp8_build(trt_config)
    # TensorRT's fused gate/up (dual-GEMM) kernel is switched off: it does not compile for an
    # FP8 consumer of the SwiGLU output on TensorRT 11.1 in other families. The route is set
    # only when this TensorRT build lists the knob.
    if LM_DUAL_GEMM_OFF and _DUAL_GEMM_KNOB in getattr(trt_config, "all_build_routes", ""):
        trt_config.build_route = f"-{_DUAL_GEMM_KNOB}=off"


LM_DUAL_GEMM_OFF = True
