# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Depth Anything-owned FP8 and NVFP4 TensorRT Q/DQ graph context.

FP8 (``quantization="fp8"``) is self-quantization of the original checkpoint, because
no ModelOpt FP8 checkpoint exists for Depth Anything. Weights are quantized at build
time, one projection at a time: ``scale = max(abs(weight)) / 448`` (448 is the largest
finite e4m3 value), and the FP8 bytes are fed to the engine as a constant. FP8 weights are
never stored. The activation scale of each projection input needs real data, so it was
calibrated once offline with ``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``, max
calibration) and is read from ``fp8_activation_scales.json``, one table per model config.
Users do not run calibration. Each quantized projection becomes
``activation -> Quantize -> Dequantize -> MatMul`` with ``FP8 weight -> Dequantize`` on
the other input.

Recipe (the public ViT recipes of ModelOpt and TensorRT, adapted to the per-tensor scales
of this path): only the DINOv2 ViT backbone is in scope. Its query, key and value
projections, attention output projection and both MLP projections are quantized in every
encoder layer. The patch convolution, the embeddings, layer norms, LayerScale, the GELU,
the attention matmuls and the whole DPT neck and head (convolutions, resizes) stay in the
work precision. A projection whose calibrated input scale is more than 20 times the median
of its kind is stored as ``null`` in the table and stays unquantized.

A model config with no table, or a projection with no entry in the table (a ``null`` entry
is an explicit entry), raises and names the calibration tool: there is no silent fallback.

The Dequantize layers and the matmul run in the type of the activation (the family work
precision), so no Cast layers are added.

NVFP4 (``quantization="nvfp4"``, Blackwell GPUs) is also self-quantization of the original
checkpoint, for the Base and Large model configs only (``NVFP4_MODEL_CONFIGS``; Small raises,
its NVFP4 quality loss is about 2x the FP8 loss or more). Only the first MLP projection (``mlp.fc1``) of every encoder layer is NVFP4 W4A4;
the other five projections are exactly the FP8 recipe above. NVFP4 on the other projections
was measured slower than this layout and returned NaN or wrong depth for some batch sizes and
graphs, so they stay FP8. The fc1 weight is quantized at build time:
``weight_scale_2 = amax * fp32(1 / 2688)`` (2688 = 6 * 448), one e4m3 block scale per 16 inputs
(``block_amax / 6 / weight_scale_2``, clamped to 448), values rounded to the E2M1 grid and packed
two per byte (first value in the low nibble), the same bytes as a ModelOpt NVFP4 export. The
fc1 input is quantized at run time with ``add_dynamic_quantize`` (per-16 e4m3 block scales) and
a static global scale ``input_scale = amax / 2688`` read from ``nvfp4_activation_scales.json``.
That amax is the FP8 calibration amax of the same input, so the NVFP4 table is the FP8 scale
divided by 6 (``calibrate_fp8_scales.py`` writes both tables). The fc1 activation is reshaped
from ``[1, tokens, K]`` to ``[tokens, K]`` for the quantize and the matmul and back afterwards.
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
E2M1_MAX = 6.0
NVFP4_BLOCK_SIZE = 16
NVFP4_MAX = E2M1_MAX * FP8_E4M3_MAX
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"
NVFP4_ACTIVATION_SCALES_FILENAME = "nvfp4_activation_scales.json"

_SUPPORTED = ("fp8", "nvfp4")
LAYER_PROJECTIONS = (
    "attention.attention.query",
    "attention.attention.key",
    "attention.attention.value",
    "attention.output.dense",
    "mlp.fc1",
    "mlp.fc2",
)
# The projections that are NVFP4 under quantization="nvfp4"; the others keep the FP8 recipe.
NVFP4_PROJECTIONS = ("mlp.fc1",)
# Model configs, by backbone hidden size, that support quantization="nvfp4". Small (hidden size
# 384) does not: its NVFP4 depth error is 1.97x (fp16) and 2.08x (fp32) the FP8 error, at or
# above the 2x limit, and its worst held-out image drops from 0.914 (FP8) to 0.723 correlation.
NVFP4_MODEL_CONFIGS = {768: "Base", 1024: "Large"}
_MODEL_CONFIG_NAMES = {384: "Small", 768: "Base", 1024: "Large"}
# A projection whose input scale exceeds this multiple of the median of its kind stays unquantized.
OUTLIER_RATIO = 20.0


@dataclass(frozen=True)
class _FP8Weight:
    """One projection weight stored as raw fp8_e4m3 bytes with one per-tensor scale."""

    packed: np.ndarray  # [out_features, in_features] raw fp8_e4m3 bytes
    weight_scale: float
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _FP4Weight:
    """One projection weight in NVFP4: packed E2M1 bytes, e4m3 block scales, one global scale."""

    packed: np.ndarray  # [out_features, in_features / 2] uint8, first value in the low nibble
    block_scale: np.ndarray  # [out_features, in_features / 16] raw e4m3 bytes
    weight_scale_2: float
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _LayerScales:
    """Activation scale and the packed weight (FP8 or NVFP4) of one quantized projection."""

    input_scale: float
    weight: _FP8Weight | _FP4Weight


def _scale_const(network, value: float, dtype: trt.DataType):
    """One-element constant holding a Q/DQ scale in the Dequantize output type."""
    np_dtype = np.float16 if dtype == trt.float16 else np.float32
    data = np.asarray([value], dtype=np_dtype)
    return network.add_constant((1,), trt.Weights(data)).get_output(0)


def _wrap_fp8_matmul(network, activation, scales: _LayerScales, *, keep_alive: list):
    """Add ``activation @ weight.T`` with an FP8 weight and a static FP8 activation scale.

    The Dequantize layers and the matmul use the type of ``activation``, so the matmul
    output has the same type and no cast is added. Weights keep the checkpoint's
    ``[out_features, in_features]`` layout, so the matmul transposes the weight operand.
    """
    dtype = activation.dtype
    if dtype not in (trt.float16, trt.float32):
        raise ValueError(f"Depth Anything FP8 supports fp16 or fp32 activations, got {dtype}")
    w = scales.weight
    rank = len(tuple(activation.shape))
    shape = (1,) * max(0, rank - 2) + (w.out_features, w.in_features)

    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(w.packed)
    w_fp8 = trt.Weights(trt.DataType.FP8, w.packed.ctypes.data, w.out_features * w.in_features)
    w_const = network.add_constant(shape, w_fp8).get_output(0)
    w_dequant = network.add_dequantize(w_const, _scale_const(network, w.weight_scale, dtype), dtype)

    act_scale = _scale_const(network, scales.input_scale, dtype)
    a_quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
    a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, dtype)

    return network.add_matrix_multiply(
        a_dequant.get_output(0),
        trt.MatrixOperation.NONE,
        w_dequant.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)


def _fp32_const(network, value: float):
    """One-element fp32 constant (the NVFP4 global scales)."""
    data = np.asarray([value], dtype=np.float32)
    return network.add_constant((1,), trt.Weights(data)).get_output(0)


def _wrap_nvfp4_matmul(network, activation, scales: _LayerScales, *, keep_alive: list):
    """Add ``activation @ weight.T`` with an NVFP4 weight and a block-scaled FP4 activation.

    The ``[1, tokens, K]`` activation is reshaped to ``[tokens, K]`` and quantized at run time
    with ``add_dynamic_quantize`` (per-16 e4m3 block scales, global scale ``input_scale``). The
    weight is an FP4 constant that reaches the matmul through a double Dequantize (block scale
    times ``weight_scale_2``, then the weight). TensorRT fuses the pattern into a block-scaled
    FP4 GEMM. The Dequantize layers and the matmul use the type of ``activation``; the output
    is reshaped back to ``[1, tokens, N]``.
    """
    dtype = activation.dtype
    if dtype not in (trt.float16, trt.float32):
        raise ValueError(f"Depth Anything NVFP4 supports fp16 or fp32 activations, got {dtype}")
    w = scales.weight
    n, k = w.out_features, w.in_features
    shape = tuple(int(v) for v in activation.shape)
    if len(shape) != 3 or shape[-1] != k:
        raise ValueError(f"Depth Anything NVFP4 expects a [batch, tokens, {k}] input, got {shape}")
    rows = shape[0] * shape[1]

    # trt.Weights holds a raw pointer, so the buffers must outlive the build.
    keep_alive.append(w.packed)
    keep_alive.append(w.block_scale)
    w_fp4 = trt.Weights(trt.DataType.FP4, w.packed.ctypes.data, n * k)
    w_const = network.add_constant((n, k), w_fp4).get_output(0)
    w_fp8_bs = trt.Weights(trt.DataType.FP8, w.block_scale.ctypes.data, w.block_scale.size)
    w_bs_const = network.add_constant((n, k // NVFP4_BLOCK_SIZE), w_fp8_bs).get_output(0)
    w_scale = network.add_dequantize(w_bs_const, _fp32_const(network, w.weight_scale_2), dtype)
    w_dequant = network.add_dequantize(w_const, w_scale.get_output(0), dtype)
    w_dequant.axis = 1

    flat = network.add_shuffle(activation)
    flat.reshape_dims = (rows, k)
    act_scale = _fp32_const(network, scales.input_scale)
    a_dynq = network.add_dynamic_quantize(
        flat.get_output(0), 1, NVFP4_BLOCK_SIZE, trt.DataType.FP4, trt.DataType.FP8
    )
    a_dynq.set_input(1, act_scale)
    a_scale = network.add_dequantize(a_dynq.get_output(1), act_scale, dtype)
    a_dequant = network.add_dequantize(a_dynq.get_output(0), a_scale.get_output(0), dtype)
    a_dequant.axis = 1

    product = network.add_matrix_multiply(
        a_dequant.get_output(0),
        trt.MatrixOperation.NONE,
        w_dequant.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    unflat = network.add_shuffle(product)
    unflat.reshape_dims = (shape[0], shape[1], n)
    return unflat.get_output(0)


@dataclass(frozen=True)
class DepthAnythingQuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    keep_alive: list = None  # type: ignore[assignment]

    @property
    def has_fp8_weights(self) -> bool:
        """True when any projection is FP8 (selects the FP8 build options)."""
        return any(isinstance(entry.weight, _FP8Weight) for entry in self.scales.values())

    @property
    def has_fp4_weights(self) -> bool:
        """True when any projection is NVFP4."""
        return any(isinstance(entry.weight, _FP4Weight) for entry in self.scales.values())

    def __post_init__(self):
        """Create the list that keeps weight buffers alive during the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, name: str) -> bool:
        """True when ``name`` (for example ``backbone.encoder.layer.3.mlp.fc1``) is quantized."""
        return name in self.scales

    def matmul(self, name: str) -> Any:
        """The FP8 or NVFP4 matmul for ``name`` (``(network, input) -> output``), or None."""
        entry = self.scales.get(name)
        if entry is None:
            return None
        wrap = _wrap_nvfp4_matmul if isinstance(entry.weight, _FP4Weight) else _wrap_fp8_matmul

        def run(network, activation):
            return wrap(network, activation, entry, keep_alive=self.keep_alive)

        return run


# Weight whose sum tells checkpoints of the same model config apart (for example Depth Anything
# V1 and V2 share their model configs but not their weights, so they must not share scales).
_FINGERPRINT_WEIGHT = "backbone.encoder.layer.0.attention.attention.query.weight"


def scale_key(cfg: dict, weights: dict) -> str:
    """Key of this model's table in the activation-scale file.

    The model config (width, depth, heads, patch, image size) plus a numeric fingerprint of the
    weights: the sum of one fixed projection weight, in five significant digits.
    """
    if _FINGERPRINT_WEIGHT not in weights:
        raise ValueError(f"checkpoint has no {_FINGERPRINT_WEIGHT}")
    total = float(np.asarray(weights[_FINGERPRINT_WEIGHT], dtype=np.float64).sum())
    return (
        f"depth_anything-h{cfg['hidden_size']}-l{cfg['num_hidden_layers']}"
        f"-n{cfg['num_attention_heads']}-p{cfg['patch_size']}-s{cfg['image_size']}"
        f"-w{total:.4e}"
    )


def quantize_fp8_weight(weight_fp32: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize one weight to FP8 e4m3 with a single scalar scale.

    Returns the raw fp8 bytes and ``scale = max(abs(weight)) / 448``; the same scale is
    the dequantize scale in the engine.
    """
    amax = float(np.abs(weight_fp32).max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scale = amax / FP8_E4M3_MAX
    scaled = np.clip(weight_fp32 / scale, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return np.ascontiguousarray(packed), scale


_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Midpoints between neighbouring E2M1 values. A value exactly on a midpoint goes to the neighbour
# with an even code (0, 2, 4, 6): the first, third, fifth and seventh midpoint round down and the
# others round up.
_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_MIDPOINT_ROUNDS_UP = (False, True, False, True, False, True, False)


def _e2m1_codes(scaled: np.ndarray) -> np.ndarray:
    """Sign-magnitude E2M1 codes (0..15) of already block-scaled values."""
    magnitude = np.abs(scaled)
    code = np.zeros(scaled.shape, dtype=np.uint8)
    for midpoint, up in zip(_MIDPOINTS, _MIDPOINT_ROUNDS_UP, strict=True):
        code += (magnitude >= midpoint if up else magnitude > midpoint).astype(np.uint8)
    return code | ((scaled < 0).astype(np.uint8) << 3)


def quantize_nvfp4_weight(weight_fp32: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Quantize one ``[out, in]`` weight to NVFP4 (the ModelOpt NVFP4 weight format).

    Returns the packed bytes ``[out, in / 2]`` (first value in the low nibble), the raw e4m3
    block-scale bytes ``[out, in / 16]`` and ``weight_scale_2 = amax * fp32(1 / 2688)``. The
    block scale is ``block_amax / 6 / weight_scale_2`` clamped to 448 and cast to e4m3; an
    all-zero block (or one whose scale underflows e4m3) gets scale 1. All arithmetic is fp32.
    """
    weight = np.asarray(weight_fp32, dtype=np.float32)
    if weight.ndim != 2:
        raise ValueError(f"expected a [out, in] weight, got shape {weight.shape}")
    out_features, in_features = weight.shape
    if in_features <= 0 or in_features % NVFP4_BLOCK_SIZE:
        raise ValueError(f"in_features={in_features} is not a multiple of {NVFP4_BLOCK_SIZE}")
    amax = np.float32(np.abs(weight).max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    # ModelOpt computes amax * fp32(1 / 2688) (a reciprocal multiply), which can differ in the
    # last bit from the correctly rounded division; the same operation gives the same bytes.
    weight_scale_2 = np.float32(amax * np.float32(1.0 / NVFP4_MAX))
    blocks = weight.reshape(out_features, in_features // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    block_amax = np.abs(blocks).max(axis=-1)
    block_scale = np.minimum(
        block_amax / np.float32(E2M1_MAX) / weight_scale_2, np.float32(FP8_E4M3_MAX)
    ).astype(ml_dtypes.float8_e4m3fn)
    scale_float = block_scale.astype(np.float32)
    scale_float = np.where(scale_float == 0, np.float32(1.0), scale_float).astype(np.float32)
    block_scale = scale_float.astype(ml_dtypes.float8_e4m3fn)
    scaled = blocks / (scale_float[..., None] * weight_scale_2)
    code = _e2m1_codes(scaled).reshape(out_features, in_features)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
    return (
        np.ascontiguousarray(packed),
        np.ascontiguousarray(block_scale.view(np.uint8)),
        float(weight_scale_2),
    )


def dequantize_nvfp4_weight(
    packed: np.ndarray, block_scale: np.ndarray, weight_scale_2: float
) -> np.ndarray:
    """The float32 values the NVFP4 bytes stand for, ``[out, in]`` (what the engine uses)."""
    packed = np.asarray(packed, dtype=np.uint8)
    codes = np.stack([packed & 0xF, packed >> 4], axis=-1).reshape(packed.shape[0], -1)
    values = np.asarray(_E2M1_GRID, np.float32)[codes & 7] * np.where(codes >= 8, -1.0, 1.0)
    scale = np.asarray(block_scale, np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    expanded = np.repeat(scale, NVFP4_BLOCK_SIZE, axis=-1)
    return (values * expanded * np.float32(weight_scale_2)).astype(np.float32)


def load_activation_scales(cfg: dict, weights: dict) -> dict[str, float | None]:
    """Return the committed activation scales for this model config, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Depth Anything FP8 build requires {path}, which was not found"
        ) from error
    key = scale_key(cfg, weights)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run families/depth_anything/calibrate_fp8_scales.py for this "
            "model config and checkpoint."
        )
    return table[key]


def load_nvfp4_activation_scales(cfg: dict, weights: dict) -> dict[str, float | None]:
    """Return the committed NVFP4 input scales for this model config, or raise when none exist."""
    path = Path(__file__).parent / NVFP4_ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Depth Anything NVFP4 build requires {path}, which was not found"
        ) from error
    key = scale_key(cfg, weights)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated NVFP4 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run families/depth_anything/calibrate_fp8_scales.py for this "
            "model config and checkpoint."
        )
    return table[key]


def projections(cfg: dict) -> list[str]:
    """Checkpoint names (without the ``.weight`` suffix) of every quantizable projection."""
    return [
        f"backbone.encoder.layer.{layer}.{stem}"
        for layer in range(cfg["num_hidden_layers"])
        for stem in LAYER_PROJECTIONS
    ]


def check_nvfp4_model_config(cfg: dict) -> None:
    """Raise unless ``quantization="nvfp4"`` is supported for this model config (Base, Large)."""
    hidden = int(cfg["hidden_size"])
    if hidden in NVFP4_MODEL_CONFIGS:
        return
    allowed = " and ".join(
        f"{name} (hidden size {size})" for size, name in sorted(NVFP4_MODEL_CONFIGS.items())
    )
    known = _MODEL_CONFIG_NAMES.get(hidden)
    got = f"hidden size {hidden}" + (f" ({known})" if known else "")
    raise NotImplementedError(
        f"depth_anything quantization=nvfp4 supports only the {allowed} model configs; this "
        f"checkpoint has {got}. Use quantization=fp8 or none for this model config."
    )


def nvfp4_projections(cfg: dict) -> list[str]:
    """Checkpoint names of the projections that are NVFP4 under ``quantization="nvfp4"``."""
    return [
        f"backbone.encoder.layer.{layer}.{stem}"
        for layer in range(cfg["num_hidden_layers"])
        for stem in NVFP4_PROJECTIONS
    ]


def _nvfp4_entries(weights: dict, cfg: dict) -> dict[str, _LayerScales | None]:
    """NVFP4 weight and input scale of every ``nvfp4_projections`` entry (None: unquantized)."""
    table = load_nvfp4_activation_scales(cfg, weights)
    names = nvfp4_projections(cfg)
    missing = sorted(set(names) - set(table))
    unused = sorted(set(table) - set(names))
    if missing or unused:
        raise ValueError(
            f"NVFP4 scale table {scale_key(cfg, weights)!r} does not match the model: "
            f"{len(missing)} projections without an entry (first: {missing[:3]}), "
            f"{len(unused)} entries without an NVFP4 projection (first: {unused[:3]}). "
            "Re-run families/depth_anything/calibrate_fp8_scales.py for this model config."
        )
    entries: dict[str, _LayerScales | None] = {}
    for name in names:
        input_scale = table[name]
        if input_scale is None:
            entries[name] = None  # outlier projection, stays unquantized
            continue
        if not np.isfinite(input_scale) or input_scale <= 0:
            raise ValueError(f"NVFP4 activation scale of {name!r} must be finite and positive")
        key = f"{name}.weight"
        if key not in weights:
            raise ValueError(f"checkpoint has no {key}")
        weight = np.asarray(weights[key], dtype=np.float32)
        packed, block_scale, weight_scale_2 = quantize_nvfp4_weight(weight)
        out_features, in_features = weight.shape
        entries[name] = _LayerScales(
            float(input_scale),
            _FP4Weight(packed, block_scale, weight_scale_2, out_features, in_features),
        )
    return entries


def calibrate_depth_anything(weights: dict, cfg: dict, quantization: str):
    """Quantize the checkpoint one projection at a time and pair it with the stored scales.

    ``fp8``: every projection FP8. ``nvfp4``: the ``NVFP4_PROJECTIONS`` NVFP4, the rest FP8.
    """
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"Depth Anything supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    if quantization == "nvfp4":
        check_nvfp4_model_config(cfg)
    activation_scales = load_activation_scales(cfg, weights)
    names = projections(cfg)
    missing = sorted(set(names) - set(activation_scales))
    unused = sorted(set(activation_scales) - set(names))
    if missing or unused:
        raise ValueError(
            f"FP8 scale table {scale_key(cfg, weights)!r} does not match the model: "
            f"{len(missing)} projections without an entry (first: {missing[:3]}), "
            f"{len(unused)} entries without a projection (first: {unused[:3]}). "
            "Re-run families/depth_anything/calibrate_fp8_scales.py for this model config."
        )
    for name in names:
        if f"{name}.weight_scale" in weights:
            raise ValueError(
                f"{name} is already quantized; {quantization} self-quantization needs the "
                "original checkpoint"
            )
    fp4_entries = _nvfp4_entries(weights, cfg) if quantization == "nvfp4" else {}
    scales: dict[str, _LayerScales] = {}
    for name in names:
        if name in fp4_entries:
            if fp4_entries[name] is not None:
                scales[name] = fp4_entries[name]
            continue
        input_scale = activation_scales[name]
        if input_scale is None:
            continue  # outlier projection, stays unquantized
        if not np.isfinite(input_scale) or input_scale <= 0:
            raise ValueError(f"FP8 activation scale of {name!r} must be finite and positive")
        key = f"{name}.weight"
        if key not in weights:
            raise ValueError(f"checkpoint has no {key}")
        weight = np.asarray(weights[key], dtype=np.float32)
        packed, weight_scale = quantize_fp8_weight(weight)
        out_features, in_features = weight.shape
        scales[name] = _LayerScales(
            float(input_scale), _FP8Weight(packed, weight_scale, out_features, in_features)
        )
    if not scales:
        raise RuntimeError(
            "Depth Anything FP8 self-quantization matched no projection in the checkpoint"
        )
    return DepthAnythingQuantContext(scales=scales)
