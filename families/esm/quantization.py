# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ESM-owned FP8 TensorRT Q/DQ graph context.

FP8 (``quantization="fp8"``) is self-quantization of the original checkpoint:
no ModelOpt FP8 or NVFP4 checkpoint of ESM-2 is published. Weights are
quantized at build time, one projection at a time: ``scale = max(abs(weight))
/ 448`` (448 is the largest finite e4m3 value), and the FP8 bytes are fed to the
engine as a constant. FP8 weights are never stored. The activation scale of each
projection input needs real protein sequences, so it was calibrated once offline
with ``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``, max calibration)
and is read from ``fp8_activation_scales.json``. Users do not run calibration.

Each quantized projection is ``activation -> Quantize -> Dequantize -> MatMul``
with ``FP8 weight -> Dequantize`` on the other input. The query, key and value
projections read the same normalised rows with the same scale, so they share
one Quantize/Dequantize pair. Dequantize and the matmul run in the work dtype of
the graph (fp16: FP8 builds require it), so the FP8 GEMM writes fp16 and the
bias add fuses into its epilogue. The FP8 GEMM accumulates in fp32; no
non-finite output was seen on 2013 real and stress inputs per engine (NaN scan).

Quantized projections (short names, ``N`` is the layer index):
``layer.N.query``, ``key``, ``value``, ``attn_out`` (``attention.output.dense``),
``fc1`` (``intermediate.dense``) and ``fc2`` (``output.dense``). The attention
is explicit matmul and softmax in this family, so its output projection is
quantized as well. The word embedding, the layer norms, the attention matmuls,
the ``lm_head`` (dense, norm and the tied decoder) stay unquantized. A stored
scale of ``null`` marks a projection the calibration left unquantized (outlier
rule, see the calibration tool); a projection missing from the table is an
error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
import tensorrt as trt

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"

_SUPPORTED = ("fp8",)
STEMS = ("query", "key", "value", "attn_out", "fc1", "fc2")
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"
# Checkpoint module (inside ``encoder.layer.N``) of each short name.
HF_SUFFIX = {
    "query": "attention.self.query",
    "key": "attention.self.key",
    "value": "attention.self.value",
    "attn_out": "attention.output.dense",
    "fc1": "intermediate.dense",
    "fc2": "output.dense",
}


@dataclass
class EsmQuantContext:
    """Routes the quantized projections through Q/DQ; the others stay unquantized."""

    scales: dict[str, float]  # short name -> activation scale (quantized projections only)
    graph: Any
    keep_alive: list = field(default_factory=list)
    _shared: dict = field(default_factory=dict)

    @property
    def has_fp8_weights(self) -> bool:
        """True when any projection is FP8 (selects the FP8 build options)."""
        return bool(self.scales)

    def should_quantize(self, name: str) -> bool:
        """True when ``name`` (for example ``layer.3.fc1``) is a quantized projection."""
        return name in self.scales

    def linear(self, network, inp, name: str, weight, bias, dtype):
        """``inp @ weight.T + bias``; FP8 for quantized projections, the plain linear otherwise."""
        input_scale = self.scales.get(name)
        if input_scale is None:
            return self.graph.add_linear(network, inp, weight, bias, dtype=dtype)
        packed, weight_scale = quantize_fp8_weight(np.asarray(weight, dtype=np.float32))
        activation = self._quantized_activation(network, inp, input_scale)
        out = _fp8_weight_matmul(
            network, activation, packed, weight_scale, graph=self.graph, keep_alive=self.keep_alive
        )
        if bias is None:
            return out
        rank = len(inp.shape)
        shape = (1,) * (rank - 1) + (-1,)
        bias = np.asarray(bias, dtype=dtype).reshape(shape)
        constant = self.graph.add_constant(network, bias.shape, bias, dtype=dtype)
        return network.add_elementwise(out, constant, trt.ElementWiseOperation.SUM).get_output(0)

    def _quantized_activation(self, network, inp, input_scale: float):
        """One Quantize/Dequantize pair per (input, scale), shared by q, k and v."""
        key = (id(inp), float(input_scale))
        cached = self._shared.get(key)
        if cached is not None:
            return cached[1]
        scale = _scale_const(network, self.graph, input_scale)
        quant = network.add_quantize(inp, scale, trt.DataType.FP8)
        dequant = network.add_dequantize(quant.get_output(0), scale, inp.dtype).get_output(0)
        # Keep ``inp`` referenced so its id cannot be reused by another tensor.
        self._shared[key] = (inp, dequant)
        return dequant


def _scale_const(network, graph, value: float):
    """One-element fp32 constant holding a Q/DQ scale."""
    return graph.add_constant(
        network, (1,), np.asarray([value], dtype=np.float32), dtype=np.float32
    )


def _fp8_weight_matmul(
    network, activation, packed: np.ndarray, weight_scale: float, *, graph, keep_alive
):
    """``activation @ dequantize(weight).T`` with an FP8 e4m3 weight constant."""
    n, k = packed.shape
    rank = len(tuple(activation.shape))
    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(packed)
    w_fp8 = trt.Weights(trt.DataType.FP8, packed.ctypes.data, n * k)
    w_const = network.add_constant((1,) * (rank - 2) + (n, k), w_fp8).get_output(0)
    w_dequant = network.add_dequantize(
        w_const, _scale_const(network, graph, weight_scale), activation.dtype
    ).get_output(0)
    return network.add_matrix_multiply(
        activation, trt.MatrixOperation.NONE, w_dequant, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)


def configure_quantized_build(trt_config, quant_ctx) -> None:
    """Apply the FP8 build options and keep layer names for kernel inspection."""
    if quant_ctx is None:
        return
    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if quant_ctx.has_fp8_weights and _DUAL_GEMM_KNOB in getattr(trt_config, "all_build_routes", ""):
        # TensorRT's fused dual-GEMM kernel fails to compile with scalar-scale FP8.
        # The route is set only when this TensorRT build lists the knob.
        trt_config.build_route = f"-{_DUAL_GEMM_KNOB}=off"


def scale_key(cfg: dict, weights) -> str:
    """Key of this checkpoint's entry in the activation-scale file.

    The model config (width, depth, FFN width, heads) plus the first three
    query biases of layer 0, so a fine-tuned checkpoint of the same model
    config does not silently reuse the scales of the base model.
    """
    bias = np.asarray(
        weights[f"{cfg['prefix']}encoder.layer.0.attention.self.query.bias"], dtype=np.float32
    ).reshape(-1)[:3]
    fingerprint = "_".join(f"{value:+.4f}" for value in bias)
    return (
        f"esm-h{int(cfg['hidden_size'])}-l{int(cfg['num_layers'])}"
        f"-i{int(cfg['intermediate_size'])}-a{int(cfg['num_heads'])}-{fingerprint}"
    )


def projection_names(cfg: dict) -> list[str]:
    """Short names of every quantizable projection of this model."""
    return [f"layer.{layer}.{stem}" for layer in range(int(cfg["num_layers"])) for stem in STEMS]


def quantize_fp8_weight(weight_fp32: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize one weight to FP8 e4m3 with a single scalar scale.

    Returns the raw fp8 bytes and ``scale = max(abs(weight)) / 448``; the same
    scale is the dequantize scale in the engine.
    """
    amax = float(np.abs(weight_fp32).max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scale = amax / FP8_E4M3_MAX
    scaled = np.clip(weight_fp32 / scale, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return np.ascontiguousarray(packed), scale


def load_activation_scales(key: str) -> dict:
    """Return the committed activation scales for this checkpoint, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"esm FP8 build requires {path}, which was not found") from error
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for the esm model config {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this checkpoint."
        )
    return table[key]


def calibrate_esm(weights, cfg: dict, graph, quantization: str) -> EsmQuantContext:
    """Pair the stored activation scales with the projections of this checkpoint."""
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"esm supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    activation_scales = load_activation_scales(scale_key(cfg, weights))
    prefix = cfg["prefix"]
    scales: dict[str, float] = {}
    for name in projection_names(cfg):
        if name not in activation_scales:
            raise ValueError(
                f"{name} has no entry in {ACTIVATION_SCALES_FILENAME}; run calibrate_fp8_scales.py"
            )
        layer, stem = name.split(".")[1], name.split(".")[2]
        module = f"{prefix}encoder.layer.{layer}.{HF_SUFFIX[stem]}"
        if f"{module}.weight" not in weights:
            raise ValueError(f"checkpoint has no {module}.weight")
        if f"{module}.weight_scale" in weights or f"{module}.input_scale" in weights:
            raise ValueError(
                f"{module} is already quantized; fp8 self-quantization needs the original "
                "checkpoint"
            )
        input_scale = activation_scales[name]
        if input_scale is None:
            continue  # outlier: calibrated, but kept unquantized on purpose
        if not float(input_scale) > 0:
            raise ValueError(f"{name} has a non-positive activation scale {input_scale!r}")
        scales[name] = float(input_scale)
    if not scales:
        raise RuntimeError("esm FP8 self-quantization matched no projection in the checkpoint")
    return EsmQuantContext(scales=scales, graph=graph)
