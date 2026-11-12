# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pi 0.5-owned FP8 TensorRT Q/DQ graph context.

FP8 (``quantization="fp8"``) is self-quantization of the original checkpoint: no ModelOpt
FP8 checkpoint of a LeRobot Pi 0.5 policy exists. Weights are quantized at build time, one
projection at a time, with one per-tensor scale ``max(abs(weight)) / 448`` (448 is the
largest finite e4m3 value), and the FP8 bytes go to the engine as a constant. The input
scale of each projection needs real observations, so it was calibrated once offline with
``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``, max calibration over LIBERO
frames and all denoising steps) and is read from ``fp8_activation_scales.json``. Users do
not run calibration. FP8 weights are never stored. A quantized projection becomes
``activation -> Quantize -> Dequantize -> MatMul`` with ``FP8 weight -> Dequantize`` on the
other input.

Recipe: gate/up/down of every Gemma-2B prefix layer whose MLP the engine computes (the
last prefix layer only emits its keys and values) and of every Gemma-300M action-expert
layer. The public VLM recipes quantize the language model only, attention included;
here q/k/v/o FP8 was measured to add about 30% to the action error for no end-to-end
speed (the expert runs 50 tokens, the prefix attention is a small share), so attention
stays in the engine precision. The SigLIP vision tower and projector (bf16 sublayers on a
float32 residual stream), the embedding table, the norms and their adaptive modulation,
the time MLP, the action in/out projections and the attention matmuls stay in the engine
precision too. A projection whose calibrated input scale is above 20x the median of its
kind (tower and stem) is an outlier: the table stores ``null`` for it and it stays
unquantized. A projection with no
entry, or a table that does not match the model, is an error: there is no silent
unquantized fallback.

FP8 needs the bf16 build precision; fp32 + FP8 and an already quantized checkpoint are
refused before the weights load. Both Dequantize layers and the matmul run in bf16, so the
GEMM writes bf16 directly. A bf16 Dequantize has the float32 exponent range and
cannot overflow the way an fp16 one can. The Q/DQ scales are constants of that type, so
the scales are rounded up to it before the weight is quantized: the engine then
dequantizes with exactly the scale the bytes were made with.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np


FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"
CALIBRATION_TOOL = "python -m families.lerobot_pi05.calibrate_fp8_scales"
# A projection input scale above this multiple of the median of its kind is an outlier.
OUTLIER_RATIO = 20.0
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"
_SUPPORTED = ("fp8",)

PREFIX_ROOT = "paligemma.model.language_model.layers"
EXPERT_ROOT = "gemma_expert.model.layers"
TOWERS = (("prefix", PREFIX_ROOT), ("expert", EXPERT_ROOT))
# Quantized projections of each layer (the recipe above).
STEMS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
_NAME = re.compile(
    r"^(?:paligemma\.model\.language_model|gemma_expert\.model)\.layers\.\d+\."
    r"mlp\.(?:gate|up|down)_proj$"
)


def round_up_to(value: float, dtype: Any) -> float:
    """The smallest value of ``dtype`` (``ml_dtypes.bfloat16`` or ``np.float32``) not below ``value``."""
    rounded = np.asarray(value, dtype=np.float32).astype(dtype)
    if float(rounded) < value:
        rounded = np.nextafter(rounded, np.asarray(np.inf, dtype=dtype))
    return float(rounded)


def _np_dtype(precision: str):
    return ml_dtypes.bfloat16 if precision == "bf16" else np.float32


def quantize_fp8_weight(
    weight_fp32: np.ndarray, precision: str = "bf16"
) -> tuple[np.ndarray, float]:
    """Quantize one weight to FP8 e4m3 with one per-tensor scale.

    Returns the raw fp8 bytes and ``scale``: ``max(abs(weight)) / 448`` rounded up to the
    engine precision, which is also the dequantize scale in the engine.
    """
    amax = float(np.abs(weight_fp32).max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scale = round_up_to(amax / FP8_E4M3_MAX, _np_dtype(precision))
    scaled = np.clip(weight_fp32 / np.float32(scale), -FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return np.ascontiguousarray(packed), scale


@dataclass(frozen=True)
class _FP8Weight:
    """One projection weight stored as raw fp8_e4m3 bytes with one per-tensor scale."""

    packed: np.ndarray  # [out_features, in_features] raw fp8_e4m3 bytes
    weight_scale: float
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _LayerScales:
    input_scale: float
    weight: _FP8Weight


def _scale_const(network, value: float, dtype, keep_alive: list):
    """A one-element Q/DQ scale constant stored directly in ``dtype`` (bf16 or fp32)."""
    import tensorrt as trt

    if dtype == trt.bfloat16:
        bits = np.asarray([value], dtype=ml_dtypes.bfloat16).view(np.uint16)
        keep_alive.append(bits)
        weights = trt.Weights(trt.DataType.BF16, bits.ctypes.data, 1)
    else:
        bits = np.asarray([value], dtype=np.float32)
        keep_alive.append(bits)
        weights = trt.Weights(bits)
    return network.add_constant((1,), weights).get_output(0)


def _wrap_fp8_matmul(network, activation, scales: _LayerScales, *, keep_alive: list):
    """``activation @ weight.T`` with an FP8 weight and a static FP8 activation scale.

    Both Dequantize layers and the matmul use the type of ``activation`` (bf16 or fp32),
    so the output has that type and no cast is added.
    """
    import tensorrt as trt

    dtype = activation.dtype
    if dtype not in (trt.bfloat16, trt.float32):
        raise ValueError(f"lerobot_pi05 FP8 supports bf16 or fp32 activations, got {dtype}")
    w = scales.weight
    n, k = w.out_features, w.in_features
    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(w.packed)
    w_fp8 = trt.Weights(trt.DataType.FP8, w.packed.ctypes.data, n * k)
    w_const = network.add_constant((n, k), w_fp8).get_output(0)
    w_dequant = network.add_dequantize(
        w_const, _scale_const(network, w.weight_scale, dtype, keep_alive), dtype
    )

    act_scale = _scale_const(network, scales.input_scale, dtype, keep_alive)
    a_quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
    a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, dtype)
    return network.add_matrix_multiply(
        a_dequant.get_output(0),
        trt.MatrixOperation.NONE,
        w_dequant.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)


@dataclass(frozen=True)
class Pi05QuantContext:
    """Routes the quantized projections through Q/DQ; every other matmul is unchanged."""

    scales: dict[str, _LayerScales]
    keep_alive: list = field(default_factory=list)

    @property
    def has_fp8_weights(self) -> bool:
        return bool(self.scales)

    def should_quantize(self, name: str) -> bool:
        """True when ``name`` (for example ``gemma_expert.model.layers.3.mlp.up_proj``) is FP8."""
        return name in self.scales

    def linear(self, network, x, name: str, weights: dict):
        """``x @ W^T`` for the bias-free projection ``name``: FP8 when quantized, else plain."""
        entry = self.scales.get(name)
        if entry is None:
            from . import graph

            return graph.linear(network, x, weights[name + ".weight"])
        return _wrap_fp8_matmul(network, x, entry, keep_alive=self.keep_alive)


def configure_quantized_build(trt_config, quant_ctx: Pi05QuantContext | None) -> None:
    """Keep layer names for kernel inspection and turn off the fused dual GEMM for FP8."""
    if quant_ctx is None:
        return
    import tensorrt as trt

    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if quant_ctx.has_fp8_weights and _DUAL_GEMM_KNOB in getattr(trt_config, "all_build_routes", ""):
        # TensorRT's fused gate/up kernel fails to compile with scalar-scale FP8 inputs.
        # The route is set only when this TensorRT build lists the knob.
        trt_config.build_route = f"-{_DUAL_GEMM_KNOB}=off"


def projection_names(depth: int = 18) -> list[str]:
    """Checkpoint names (without ``.weight``) of every quantized projection the engines compute."""
    names = []
    for _, root in TOWERS:
        # The last prefix layer feeds the expert only through its keys and values: no MLP.
        layers = depth - 1 if root == PREFIX_ROOT else depth
        for layer in range(layers):
            names.extend(f"{root}.{layer}.{stem}" for stem in STEMS)
    return names


def is_projection(name: str) -> bool:
    """True for a checkpoint-style module name of a quantized projection kind."""
    return bool(_NAME.match(name))


def scale_key(cfg, weights: dict) -> str:
    """Key of this checkpoint's entry in the activation-scale file.

    Every Pi 0.5 policy shares the Gemma-2B / Gemma-300M geometry, but fine-tunes differ in
    weights and so in activation ranges: the first values of the action output bias are
    part of the key. The camera slots, prompt length and chunk size set the token counts.
    """
    head = np.asarray(weights["action_out_proj.bias"], dtype=np.float32).reshape(-1)[:3]
    fingerprint = "_".join(f"{value:+.5f}" for value in head)
    return (
        f"pi05-gemma_2b-gemma_300m-s{cfg.camera_slots}-p{cfg.tokenizer_max_length}"
        f"-c{cfg.chunk_size}-{fingerprint}"
    )


def refuse_quantized_checkpoint(model_dir: Path) -> None:
    """Raise for a checkpoint that is already quantized: FP8 self-quantizes the original."""
    model_dir = Path(model_dir)
    try:
        raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        raw = {}
    markers = [
        name
        for name in ("hf_quant_config.json", "quantize_config.json")
        if (model_dir / name).is_file()
    ]
    if (isinstance(raw, dict) and raw.get("quantization_config")) or markers:
        found = markers[0] if markers else "quantization_config in config.json"
        raise ValueError(
            f"the checkpoint is already quantized ({found}); fp8 self-quantization needs the original"
        )


def load_activation_scales(key: str) -> dict[str, float | None]:
    """The committed activation scales for this checkpoint, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(
            f"lerobot_pi05 FP8 build requires {path}, which was not found"
        ) from error
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: {sorted(table)}. "
            f"Run `{CALIBRATION_TOOL}` for this checkpoint."
        )
    return table[key]


def build_quant_context(
    weights: dict, cfg, quantization: str, *, precision: str
) -> Pi05QuantContext:
    """Quantize the projections one at a time and pair them with the stored input scales."""
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"lerobot_pi05 supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    if precision != "bf16":
        raise ValueError(f"lerobot_pi05 fp8 needs precision bf16, got {precision!r}")
    key = scale_key(cfg, weights)
    activation_scales = load_activation_scales(key)
    names = projection_names()
    missing = sorted(set(names) - set(activation_scales))
    unused = sorted(set(activation_scales) - set(names))
    if missing or unused:
        raise ValueError(
            f"FP8 scale table {key!r} does not match the model: {len(missing)} projections "
            f"without an entry (first: {missing[:3]}), {len(unused)} entries without a "
            f"projection (first: {unused[:3]}). Re-run `{CALIBRATION_TOOL}` for this checkpoint."
        )
    scales: dict[str, _LayerScales] = {}
    for name in names:
        input_scale = activation_scales[name]
        if input_scale is None:
            continue  # outlier projection: stays unquantized
        if not np.isfinite(input_scale) or input_scale <= 0:
            raise ValueError(f"{name}: activation scale {input_scale!r} is not finite and positive")
        if f"{name}.weight_scale" in weights:
            raise ValueError(
                f"{name} is already quantized; fp8 self-quantization needs the original checkpoint"
            )
        weight = np.asarray(weights[f"{name}.weight"], dtype=np.float32)
        packed, weight_scale = quantize_fp8_weight(weight, precision)
        out_features, in_features = weight.shape
        scales[name] = _LayerScales(
            round_up_to(float(input_scale), _np_dtype(precision)),
            _FP8Weight(packed, weight_scale, out_features, in_features),
        )
    if not scales:
        raise RuntimeError("lerobot_pi05 FP8 self-quantization matched no projection")
    return Pi05QuantContext(scales=scales)
