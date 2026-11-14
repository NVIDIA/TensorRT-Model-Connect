# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModernBERT-owned FP8 self-quantization and TensorRT Q/DQ graph context.

FP8 (``quantization="fp8"``) quantizes the original checkpoint at build time, because no
FP8 checkpoint exists for the ModernBERT encoders. Weights are quantized one checkpoint
module at a time: ``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3
value), and the FP8 bytes are fed to the engine as constants. FP8 weights are never stored.
The activation scale of each module input needs real data, so it was calibrated once offline
with ``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``, max calibration, WikiText
training text) and is read from ``fp8_activation_scales.json``. Users do not run
calibration.

Each quantized projection becomes::

    activation -> Quantize -> Dequantize(fp16) -> MatMul
    FP8 weight -> Dequantize(fp16) -----------------^

Both Dequantize layers output fp16, so the FP8 GEMM writes fp16 and no cast pass is added.
The risk of fp16 is an overflow above 65504 in a projection output; the NaN scan on
held-out text found none.

Quantized (recipe: the encoder linears, as in the public FP8 recipes of small text
encoders): per layer the fused query/key/value module ``attn.Wqkv``, the fused GeGLU input
module ``mlp.Wi`` and the MLP output module ``mlp.Wo``. The family splits ``Wqkv`` into
three and ``Wi`` into two matmuls; the parts of one module share its input Q/DQ and its
weight scale, as one checkpoint module has one input and one weight quantizer. The attention
output projection ``attn.Wo`` stays 16-bit: attention is one fused attention layer, and
TensorRT 11.4 cannot compile an FP8 Quantize right after a fused attention; retest it on
every newer TensorRT release. Embeddings, layer norms, RoPE, attention and GELU stay 16-bit.

The GeGLU input module of the first layer (``layers.0.mlp.Wi``) also stays 16-bit: a
sensitivity sweep over single layers and module kinds showed that this one module causes
most of the FP8 error of the masked-LM head (ModernBERT-base: KL against the PyTorch model
0.66 with it, 0.022 without; ModernBERT-large: 0.18 and 0.013), while it costs no measurable
speed. Calibration stores it as ``null``.

A module whose calibrated input scale is more than 20 times the median of the same module
kind has a massive activation: its entry in the scale file is ``null`` and it stays 16-bit
on purpose. A module missing from the file is an error.
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
# Smallest normal fp16 value; a smaller Q/DQ scale would lose precision as an fp16 constant.
_FP16_MIN_NORMAL = 2.0**-14

_SUPPORTED = ("fp8",)
# Quantized checkpoint modules of one layer and the family weight of each part, in order.
PROJECTIONS = {
    "attn.Wqkv": (("q", "w_q"), ("k", "w_k"), ("v", "w_v")),
    "mlp.Wi": (("input", "w_mlp_input"), ("gate", "w_mlp_gate")),
    "mlp.Wo": (("down", "w_down"),),
}
# Modules of the recipe that stay 16-bit in every model config (see the module docstring).
UNQUANTIZED_MODULES = ("layers.0.mlp.Wi",)


@dataclass(frozen=True)
class _FP8Weight:
    """One projection weight stored as raw fp8_e4m3 bytes with one per-tensor scale."""

    packed: np.ndarray  # [in_features, out_features] raw fp8_e4m3 bytes
    weight_scale: float
    in_features: int
    out_features: int


@dataclass(frozen=True)
class _ModuleScales:
    """Static input scale of one checkpoint module and the FP8 weight of each part."""

    input_scale: float
    parts: dict[str, _FP8Weight]


def _scale_const(network, value: float):
    """One-element fp16 constant holding a Q/DQ scale (the Dequantize output type)."""
    data = np.asarray([value], dtype=np.float16)
    return network.add_constant((1,), trt.Weights(data)).get_output(0)


def _quantize_dequantize(network, activation, input_scale: float):
    """``activation -> Quantize -> Dequantize`` with the static per-tensor scale."""
    if activation.dtype != trt.float16:
        raise ValueError(f"ModernBERT FP8 needs an fp16 activation, got {activation.dtype}")
    act_scale = _scale_const(network, input_scale)
    quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
    return network.add_dequantize(quant.get_output(0), act_scale, trt.float16).get_output(0)


def _fp8_weight_matmul(network, dequantized_activation, weight: _FP8Weight, *, keep_alive: list):
    """Add ``activation @ weight`` for an FP8 weight constant with fp16 Dequantize."""
    rank = len(tuple(dequantized_activation.shape))
    shape = (1,) * max(0, rank - 2) + (weight.in_features, weight.out_features)
    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(weight.packed)
    w_fp8 = trt.Weights(
        trt.DataType.FP8, weight.packed.ctypes.data, weight.in_features * weight.out_features
    )
    w_const = network.add_constant(shape, w_fp8).get_output(0)
    w_dequant = network.add_dequantize(
        w_const, _scale_const(network, weight.weight_scale), trt.float16
    )
    return network.add_matrix_multiply(
        dequantized_activation,
        trt.MatrixOperation.NONE,
        w_dequant.get_output(0),
        trt.MatrixOperation.NONE,
    ).get_output(0)


@dataclass(frozen=True)
class ModernBertQuantContext:
    """Routes the quantized modules through Q/DQ; other projections stay unquantized."""

    scales: dict[str, _ModuleScales]
    keep_alive: list = field(default_factory=list)

    def should_quantize(self, module: str) -> bool:
        """True when ``module`` (for example ``layers.3.mlp.Wi``) runs as FP8."""
        return module in self.scales

    def quantize_input(self, network, activation, module: str):
        """The Quantize/Dequantize pair on the input of ``module``, shared by its parts."""
        return _quantize_dequantize(network, activation, self.scales[module].input_scale)

    def matmul(self, network, dequantized_activation, module: str, part: str):
        """``activation @ weight`` for the FP8 weight of one part of ``module``."""
        return _fp8_weight_matmul(
            network,
            dequantized_activation,
            self.scales[module].parts[part],
            keep_alive=self.keep_alive,
        )

    def configure_build(self, builder_config) -> None:
        """Keep per-layer tactic names in a quantized plan so FP8 lowering can be inspected."""
        builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED


def _abs_sum(array) -> float:
    return float(np.abs(np.asarray(array, dtype=np.float32)).sum(dtype=np.float64))


def scale_key(config, weights) -> str:
    """Key of this model's entry in the activation-scale file.

    The key holds the model config values that change the activation ranges (hidden size,
    layers, intermediate size, heads). Checkpoints of the same architecture differ in
    training, so the key also holds two numeric weight sums (sum of absolute values,
    5 significant digits): the embedding norm and the query weight of layer 0.
    """
    return (
        f"modernbert-h{config.hidden_size}-l{config.num_hidden_layers}"
        f"-i{config.intermediate_size}-n{config.num_attention_heads}"
        f"-w{_abs_sum(weights['embed_norm']):.4e}-{_abs_sum(weights['layer.0.w_q']):.4e}"
    )


def quantize_fp8_weight(weight_fp32: np.ndarray, scale: float | None = None):
    """Quantize one weight to FP8 e4m3 with a single scalar scale.

    Returns the raw fp8 bytes and ``scale`` (default ``max(abs(weight)) / 448``); the same
    scale is the dequantize scale in the engine.
    """
    if scale is None:
        amax = float(np.abs(weight_fp32).max())
        if not np.isfinite(amax) or amax <= 0:
            raise ValueError("weight tensor has no finite positive amax to scale from")
        scale = amax / FP8_E4M3_MAX
    scaled = np.clip(weight_fp32 / scale, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return np.ascontiguousarray(packed), float(scale)


def load_activation_scales(config, weights) -> dict[str, float | None]:
    """Return the committed activation scales for this model config, or raise when none exist.

    An entry of ``None`` marks a module the calibration left unquantized on purpose.
    """
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"ModernBERT FP8 build requires {path}, which was not found") from error
    key = scale_key(config, weights)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model config."
        )
    return table[key]


def module_names(num_layers: int) -> list[str]:
    """Checkpoint module names of every quantizable module of this model config."""
    return [f"layers.{layer}.{stem}" for layer in range(num_layers) for stem in PROJECTIONS]


def refuse_quantized_checkpoint(config, tensor_names) -> None:
    """Raise when the checkpoint is already quantized (scale tensors or a quantization config)."""
    if config.raw.get("quantization_config") or any(
        str(name).endswith((".weight_scale", ".input_scale")) for name in tensor_names
    ):
        raise ValueError(
            "the checkpoint is already quantized; FP8 self-quantization needs the original"
        )


def calibrate_modernbert(config, weights, quantization: str, *, precision: str):
    """Quantize the loaded weights (``[in, out]`` layout) one module at a time.

    Pairs each module with its stored activation scale. A module with no stored scale, or a
    stored scale for a module this model config does not have, is an error.
    """
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"ModernBERT supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    if precision != "fp16":
        raise ValueError(f"ModernBERT FP8 needs precision fp16, got {precision!r}")
    activation_scales = load_activation_scales(config, weights)

    names = module_names(config.num_hidden_layers)
    unexpected = sorted(set(activation_scales) - set(names))
    if unexpected:
        raise ValueError(
            f"{ACTIVATION_SCALES_FILENAME} has scales for modules that are not quantized, "
            f"for example {unexpected[0]}; run calibrate_fp8_scales.py for this model config"
        )
    scales: dict[str, _ModuleScales] = {}
    for name in names:
        if name not in activation_scales:
            raise ValueError(
                f"{name} has no activation scale in {ACTIVATION_SCALES_FILENAME}; "
                "run calibrate_fp8_scales.py"
            )
        input_scale = activation_scales[name]
        if input_scale is None:
            continue  # massive-activation module, left 16-bit by the calibration
        if not np.isfinite(input_scale) or input_scale <= 0:
            raise ValueError(f"activation scale of {name} must be finite and positive")
        layer, stem = name.split(".", 2)[1], name.split(".", 2)[2]
        matrices = {}
        for part, logical in PROJECTIONS[stem]:
            key = f"layer.{layer}.{logical}"
            if key not in weights:
                raise ValueError(f"checkpoint has no weight for {name} ({key})")
            matrices[part] = np.asarray(weights[key], dtype=np.float32)  # [in, out]
        # One weight scale per checkpoint module, shared by its parts.
        amax = max(float(np.abs(matrix).max()) for matrix in matrices.values())
        if not np.isfinite(amax) or amax <= 0:
            raise ValueError(f"weight of {name} has no finite positive amax to scale from")
        weight_scale = amax / FP8_E4M3_MAX
        if min(input_scale, weight_scale) < _FP16_MIN_NORMAL:
            raise ValueError(f"FP8 scale of {name} is below the smallest normal fp16 value")
        parts = {}
        for part, matrix in matrices.items():
            packed, _ = quantize_fp8_weight(matrix, weight_scale)
            parts[part] = _FP8Weight(packed, weight_scale, matrix.shape[0], matrix.shape[1])
        scales[name] = _ModuleScales(float(input_scale), parts)
    if not scales:
        raise RuntimeError("ModernBERT FP8 self-quantization matched no module in the checkpoint")
    return ModernBertQuantContext(scales=scales)
