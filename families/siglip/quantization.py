# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SigLIP-owned FP8 TensorRT Q/DQ graph context.

FP8 (``quantization="fp8"``) is self-quantization of the original checkpoint, because
no ModelOpt FP8 checkpoint exists for SigLIP. Weights are quantized at build time, one
projection at a time: ``scale = max(abs(weight)) / 448`` (448 is the largest finite
e4m3 value), and the FP8 bytes are fed to the engine as a constant. The activation scale
of each projection input needs real data, so it was calibrated once offline with
``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``) and is read from
``fp8_activation_scales.json``. Users do not run calibration. A projection with no entry
in that file stays unquantized. Each quantized projection becomes
``activation -> Quantize -> Dequantize -> MatMul`` with ``FP8 weight -> Dequantize`` on
the other input.

Quantized projections are the q/k/v/out projections and the two MLP projections of
every encoder layer in both towers (weight names such as
``vision_model.encoder.layers.3.mlp.fc1``). The patch convolution, the embeddings, the
layer norms, the pooling head and the final text projection stay in float32.
Weights keep the checkpoint's ``[out_features, in_features]`` layout, so the matmul
transposes the weight operand.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
import tensorrt as trt

from . import graph

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"

_SUPPORTED = ("fp8",)
TOWERS = ("vision_model", "text_model")
_LAYER_PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
    "mlp.fc1",
    "mlp.fc2",
)
# Projections that stay in float32 even when a scale exists (none at present).
_UNQUANTIZED = re.compile(r"$^")


@dataclass(frozen=True)
class _FP8Weight:
    """One projection weight stored as raw fp8_e4m3 bytes with one per-tensor scale."""

    packed: np.ndarray  # [out_features, in_features] raw fp8_e4m3 bytes
    weight_scale: float
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _LayerScales:
    """Activation scale and the packed weight of one quantized projection."""

    input_scale: float
    weight: _FP8Weight


def _scale_const(network, value: float):
    """One-element fp32 constant holding a Q/DQ scale."""
    return graph.add_constant(network, (1,), np.asarray([value], dtype=np.float32))


def _wrap_fp8_matmul(network, activation, scales: _LayerScales, *, keep_alive: list):
    """Add ``activation @ weight.T`` with an FP8 weight and a static FP8 activation scale.

    The Dequantize layers and the matmul run in float32 and the result is cast back to the
    activation type. With fp16 Dequantize outputs, TensorRT's FP8 GEMM keeps the raw
    accumulator in fp16 and applies the scales afterwards, which overflows once an
    output exceeds about ``65504 * input_scale * weight_scale``.
    """
    w = scales.weight
    n, k = w.out_features, w.in_features

    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(w.packed)
    w_fp8 = trt.Weights(trt.DataType.FP8, w.packed.ctypes.data, n * k)
    w_const = network.add_constant((n, k), w_fp8).get_output(0)
    w_dequant = network.add_dequantize(w_const, _scale_const(network, w.weight_scale), trt.float32)

    act_scale = _scale_const(network, scales.input_scale)
    a_quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
    a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, trt.float32)

    output = network.add_matrix_multiply(
        a_dequant.get_output(0),
        trt.MatrixOperation.NONE,
        w_dequant.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    if output.dtype == activation.dtype:
        return output
    return network.add_cast(output, activation.dtype).get_output(0)


@dataclass(frozen=True)
class SiglipQuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    keep_alive: list = None  # type: ignore[assignment]

    @property
    def has_fp8_weights(self) -> bool:
        """True when any projection is FP8 (selects the FP8 build options)."""
        return bool(self.scales)

    def __post_init__(self):
        """Create the list that keeps weight buffers alive during the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, name: str) -> bool:
        """True when ``name`` (for example ``text_model.encoder.layers.3.mlp.fc1``) is quantized."""
        return name in self.scales

    def matmul(self, name: str) -> Any:
        """The FP8 matmul callable for ``name`` (``(network, input) -> output``), or None."""
        entry = self.scales.get(name)
        if entry is None:
            return None

        def run(network, activation):
            return _wrap_fp8_matmul(network, activation, entry, keep_alive=self.keep_alive)

        return run


def scale_key(cfg: dict) -> str:
    """Key of this model's entry in the activation-scale file, from the resolved geometry."""
    vision, text = cfg["vision"], cfg["text"]
    return (
        f"siglip-v{vision['hidden_size']}x{vision['num_layers']}"
        f"p{vision['patch_size']}i{vision['image_size']}"
        f"-t{text['hidden_size']}x{text['num_layers']}w{text['max_text_len']}"
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


def load_activation_scales(cfg: dict) -> dict[str, float]:
    """Return the committed activation scales for this model shape, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"SigLIP FP8 build requires {path}, which was not found") from error
    key = scale_key(cfg)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model shape."
        )
    return table[key]


def projections(cfg: dict) -> list[str]:
    """Checkpoint names (without the ``.weight`` suffix) of every quantizable projection."""
    names = []
    for tower in TOWERS:
        layers = cfg["vision" if tower == "vision_model" else "text"]["num_layers"]
        for layer in range(layers):
            names.extend(f"{tower}.encoder.layers.{layer}.{stem}" for stem in _LAYER_PROJECTIONS)
    return names


def calibrate_siglip(weights: dict, cfg: dict, quantization: str) -> SiglipQuantContext:
    """Quantize the checkpoint one projection at a time and pair it with the stored scales."""
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"SigLIP supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    activation_scales = load_activation_scales(cfg)
    scales: dict[str, _LayerScales] = {}
    for name in projections(cfg):
        if f"{name}.weight_scale" in weights:
            raise ValueError(
                f"{name} is already quantized; fp8 self-quantization needs the original checkpoint"
            )
        input_scale = activation_scales.get(name)
        if input_scale is None or _UNQUANTIZED.match(name):
            continue
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
        raise RuntimeError("SigLIP FP8 self-quantization matched no projection in the checkpoint")
    return SiglipQuantContext(scales=scales)
