# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RWKV-owned FP8 self-quantization and TensorRT Q/DQ graph context.

FP8 (``quantization="fp8"``) quantizes the original checkpoint at build time, because no
FP8 checkpoint exists for the RWKV-4 models. Weights are quantized one projection at a
time: ``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3 value), and the
FP8 bytes are fed to the engine as constants. FP8 weights are never stored. The activation
scale of each projection input needs real data, so it was calibrated once offline with
``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``, max calibration, WikiText
training text) and is read from ``fp8_activation_scales.json``. Users do not run
calibration.

Each quantized projection becomes::

    activation -> Quantize -> Dequantize(fp16) -> MatMul
    FP8 weight -> Dequantize(fp16) -----------------^

Both Dequantize layers output fp16, so the FP8 GEMM writes fp16 and no cast pass is added.
The risk of fp16 is an overflow above 65504 in a projection output; the NaN scan on
held-out text found none.

Quantized (recipe: every linear projection of the time-mixing and channel-mixing blocks,
as in the public FP8 recipes of recurrent and hybrid models): per layer
``attention.key``, ``attention.value``, ``attention.receptance``, ``attention.output``,
``feed_forward.key``, ``feed_forward.receptance`` and ``feed_forward.value``. Each reads its
own time-shift blend (or gated WKV output), so each has its own Quantize. RWKV has no
softmax attention, so the time-mixing output projection quantizes like the others.

Kept 16-bit: every projection of the LAST layer, whose time-mixing output and feed-forward
inputs reach 2 to 16 times the median of their kind and which carries most of the FP8 loss
(ModelOpt fake quantization, WikiText-2 test: rwkv-4-3b KL 0.026 with it and 0.008
without, rwkv-4-169m 0.019 and 0.013); the embedding, the LM head, layer norms, the
time-shift blends and the WKV recurrence (which stays fp32 in every build).

A projection whose calibrated input scale is more than 20 times the median of the same
projection kind has a massive activation: its entry in the scale file is ``null`` and it
stays 16-bit on purpose. A projection missing from the file is an error.
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
# Quantized checkpoint projection of one layer (name below ``rwkv.blocks.N``) and the
# family weight that holds it.
PROJECTIONS = {
    "attention.key": "w_attn_k",
    "attention.value": "w_attn_v",
    "attention.receptance": "w_attn_r",
    "attention.output": "w_attn_o",
    "feed_forward.key": "w_ffn_k",
    "feed_forward.receptance": "w_ffn_r",
    "feed_forward.value": "w_ffn_v",
}


@dataclass(frozen=True)
class _FP8Weight:
    """One projection weight stored as raw fp8_e4m3 bytes with one per-tensor scale."""

    packed: np.ndarray  # [in_features, out_features] raw fp8_e4m3 bytes
    weight_scale: float
    in_features: int
    out_features: int


@dataclass(frozen=True)
class _ProjectionScales:
    """Static input scale and FP8 weight of one projection."""

    input_scale: float
    weight: _FP8Weight


def _scale_const(network, value: float):
    """One-element fp16 constant holding a Q/DQ scale (the Dequantize output type)."""
    data = np.asarray([value], dtype=np.float16)
    return network.add_constant((1,), trt.Weights(data)).get_output(0)


@dataclass(frozen=True)
class RwkvQuantContext:
    """Routes the quantized projections through Q/DQ; other projections stay unquantized."""

    scales: dict[str, _ProjectionScales]
    keep_alive: list = field(default_factory=list)

    def should_quantize(self, module: str) -> bool:
        """True when ``module`` (for example ``layer.3.feed_forward.key``) runs as FP8."""
        return module in self.scales

    def matmul(self, network, activation, module: str):
        """``Quantize -> Dequantize(fp16)`` on ``activation``, then ``@`` the FP8 weight."""
        if activation.dtype != trt.float16:
            raise ValueError(f"RWKV FP8 needs an fp16 activation, got {activation.dtype}")
        entry = self.scales[module]
        weight = entry.weight
        act_scale = _scale_const(network, entry.input_scale)
        quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
        dequant = network.add_dequantize(quant.get_output(0), act_scale, trt.float16)

        rank = len(tuple(activation.shape))
        shape = (1,) * max(0, rank - 2) + (weight.in_features, weight.out_features)
        # trt.Weights holds a raw pointer, so the buffer must outlive the build.
        self.keep_alive.append(weight.packed)
        w_fp8 = trt.Weights(
            trt.DataType.FP8, weight.packed.ctypes.data, weight.in_features * weight.out_features
        )
        w_const = network.add_constant(shape, w_fp8).get_output(0)
        w_dequant = network.add_dequantize(
            w_const, _scale_const(network, weight.weight_scale), trt.float16
        )
        return network.add_matrix_multiply(
            dequant.get_output(0),
            trt.MatrixOperation.NONE,
            w_dequant.get_output(0),
            trt.MatrixOperation.NONE,
        ).get_output(0)

    def configure_build(self, builder_config) -> None:
        """Keep per-layer tactic names in a quantized plan so FP8 lowering can be inspected."""
        builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED


def _abs_sum(array) -> float:
    return float(np.abs(np.asarray(array, dtype=np.float32)).sum(dtype=np.float64))


def scale_key(config, weights) -> str:
    """Key of this model's entry in the activation-scale file.

    The key holds the model config values that change the activation ranges (hidden size,
    layers, intermediate size, vocabulary). Checkpoints of the same architecture differ in
    training (the Pile models and their Raven chat fine-tunes share one config), so the key
    also holds two numeric weight sums (sum of absolute values, 5 significant digits): the
    first time-mixing layer norm and the time-mixing key weight of layer 0.
    """
    return (
        f"rwkv-h{config.hidden_size}-l{config.num_hidden_layers}"
        f"-i{int(weights['_intermediate_size'])}-v{config.vocab_size}"
        f"-w{_abs_sum(weights['layer.0.attn_norm']):.4e}-{_abs_sum(weights['layer.0.w_attn_k']):.4e}"
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

    An entry of ``None`` marks a projection the calibration left unquantized on purpose.
    """
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"RWKV FP8 build requires {path}, which was not found") from error
    key = scale_key(config, weights)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model config."
        )
    return table[key]


def quantized_layers(num_layers: int) -> range:
    """Layers whose projections run as FP8: all but the last one (see the module docstring)."""
    if num_layers < 2:
        raise ValueError("RWKV FP8 needs at least 2 layers (the last layer stays 16-bit)")
    return range(num_layers - 1)


def module_names(num_layers: int) -> list[str]:
    """Names of every quantizable projection of this model config."""
    return [
        f"layer.{layer}.{stem}" for layer in quantized_layers(num_layers) for stem in PROJECTIONS
    ]


def refuse_quantized_checkpoint(config, tensor_names, model_dir=None) -> None:
    """Raise when the checkpoint is already quantized.

    Signs: ``quantization_config`` in ``config.json``, an ``hf_quant_config.json`` next to it,
    or scale tensors in the checkpoint.
    """
    if (
        config.raw.get("quantization_config")
        or (model_dir is not None and (Path(model_dir) / "hf_quant_config.json").is_file())
        or any(str(name).endswith((".weight_scale", ".input_scale")) for name in tensor_names)
    ):
        raise ValueError(
            "the checkpoint is already quantized; FP8 self-quantization needs the original"
        )


def calibrate_rwkv(config, weights, quantization: str, *, precision: str):
    """Quantize the loaded weights (``[in, out]`` layout) one projection at a time.

    Pairs each projection with its stored activation scale. A projection with no stored
    scale, or a stored scale for a projection this model config does not quantize, is an
    error.
    """
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"RWKV supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    if precision != "fp16":
        raise ValueError(f"RWKV FP8 needs precision fp16, got {precision!r}")
    names = module_names(config.num_hidden_layers)
    activation_scales = load_activation_scales(config, weights)

    unexpected = sorted(set(activation_scales) - set(names))
    if unexpected:
        raise ValueError(
            f"{ACTIVATION_SCALES_FILENAME} has scales for projections that are not quantized, "
            f"for example {unexpected[0]}; run calibrate_fp8_scales.py for this model config"
        )
    scales: dict[str, _ProjectionScales] = {}
    for name in names:
        if name not in activation_scales:
            raise ValueError(
                f"{name} has no activation scale in {ACTIVATION_SCALES_FILENAME}; "
                "run calibrate_fp8_scales.py"
            )
        input_scale = activation_scales[name]
        if input_scale is None:
            continue  # massive-activation projection, left 16-bit by the calibration
        if not np.isfinite(input_scale) or input_scale <= 0:
            raise ValueError(f"activation scale of {name} must be finite and positive")
        _, layer, stem = name.split(".", 2)
        key = f"layer.{layer}.{PROJECTIONS[stem]}"
        if key not in weights:
            raise ValueError(f"checkpoint has no weight for {name} ({key})")
        matrix = np.asarray(weights[key], dtype=np.float32)  # [in, out]
        packed, weight_scale = quantize_fp8_weight(matrix)
        if min(input_scale, weight_scale) < _FP16_MIN_NORMAL:
            raise ValueError(f"FP8 scale of {name} is below the smallest normal fp16 value")
        scales[name] = _ProjectionScales(
            float(input_scale),
            _FP8Weight(packed, weight_scale, matrix.shape[0], matrix.shape[1]),
        )
    if not scales:
        raise RuntimeError("RWKV FP8 self-quantization matched no projection in the checkpoint")
    return RwkvQuantContext(scales=scales)
