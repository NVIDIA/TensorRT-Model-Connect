# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3-owned FP8 self-quantization of the DiT projections.

``quantization="fp8"`` quantizes the original bf16 transformer at build time; no FP8
checkpoint is needed or written. Weights are quantized one projection at a time:
``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3 value), and the FP8 bytes
are fed to the engine as a constant. The activation scale of each projection input needs
real denoising activations, so it was calibrated once offline with
``calibrate_fp8_scales.py`` (forward hooks, ``max(abs(input)) / 448``) and is read from
``fp8_activation_scales.json``. Users do not run calibration. A projection with no entry in
that file is an error.

Each quantized projection becomes::

    activation (bf16) -> Quantize -> Dequantize (bf16) -> MatMul (bf16)
    FP8 weight [out, in] -> Dequantize (bf16) -> transpose ---^

The Dequantize layers use the activation type (bf16), so TensorRT keeps the bias and the
elementwise ops that follow in the GEMM epilogue. A bf16 accumulator has the float32
exponent range, so the fp16 overflow of a Dequantize cannot happen here.

Quantized projections of the ``transformer_blocks.N`` stack (``QUANTIZED``). The fused
``to_q``/``to_k``/``to_v`` GEMM and the fused gate/up GEMM (``ff.net.0.proj``) take one
weight scale over the whole fused weight (the maximum of its parts) and one input scale.

Everything else stays bf16 (or float32 where the checkpoint is float32), for these reasons:

  ``attn.to_out.0``     TensorRT 11.4 cannot compile an FP8 Quantize on the output of the fused
                        attention; TensorRT 11.1 can.
  token refiner         two blocks over at most 537 text rows: no gain, kept exact.
  adaLN projections, time embedder, patch and context projections, ``proj_out``, norms, RoPE,
  text encoder, VAE      not part of the DiT GEMM stack that dominates a call.
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

_SUPPORTED = ("fp8",)

# Quantized projections inside one block (engine names are ``transformer_blocks.N.<stem>``).
# ``attn.qkv`` is the fused query/key/value GEMM (inputs shared by to_q, to_k, to_v).
QUANTIZED = ("attn.qkv", "ff.net.0.proj", "ff.net.2")


def scale_key(hidden_size: int, num_layers: int, ffn_dim: int) -> str:
    """Key of this model's entry in the activation-scale file."""
    return f"minimax_h3-h{int(hidden_size)}-l{int(num_layers)}-i{int(ffn_dim)}"


def quantized_names(num_layers: int, stems: tuple[str, ...] = QUANTIZED) -> list[str]:
    """Names (``transformer_blocks.N.<stem>``) of the projections that are quantized."""
    return [f"transformer_blocks.{i}.{stem}" for i in range(num_layers) for stem in stems]


def quantize_fp8_weight(weight: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize one weight to FP8 e4m3 with a single scalar scale.

    Returns the raw fp8 bytes and ``scale = max(abs(weight)) / 448``; the same scale is the
    dequantize scale in the engine.
    """
    weight_fp32 = np.asarray(weight, dtype=np.float32)
    amax = float(np.abs(weight_fp32).max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scale = amax / FP8_E4M3_MAX
    scaled = np.clip(weight_fp32 / scale, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return np.ascontiguousarray(packed), scale


def load_activation_scales(hidden_size: int, num_layers: int, ffn_dim: int) -> dict[str, float]:
    """Return the committed activation scales for this model config, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"MiniMax-H3 FP8 build requires {path}, which was not found") from error
    key = scale_key(hidden_size, num_layers, ffn_dim)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model config."
        )
    return table[key]


def _scale_const(network, value: float):
    """One-element fp32 constant holding a Q/DQ scale."""
    arr = np.asarray([value], dtype=np.float32)
    return network.add_constant((1,), trt.Weights(arr)).get_output(0)


@dataclass
class MiniMaxQuantContext:
    """Routes the quantized projections through FP8 Q/DQ; others stay bf16."""

    scales: dict[str, float]
    keep_alive: list = field(default_factory=list)

    @property
    def has_fp8_weights(self) -> bool:
        """True when any projection is FP8."""
        return bool(self.scales)

    def should_quantize(self, name: str) -> bool:
        """True when ``name`` (for example ``transformer_blocks.3.ff.net.2``) is quantized."""
        return name in self.scales

    def linear(self, network, x, weight_oi: np.ndarray, name: str):
        """``x @ weight_oi.T`` for ``x`` of shape ``[rows, in]`` with an FP8 weight.

        ``weight_oi`` is the PyTorch ``[out, in]`` weight (a fused weight for qkv and
        gate/up). Both Dequantize layers and the matmul use the activation type (bf16).
        """
        out_dim, in_dim = weight_oi.shape
        dtype = x.dtype
        packed, weight_scale = quantize_fp8_weight(weight_oi)
        # trt.Weights holds a raw pointer, so the buffer must outlive the build.
        self.keep_alive.append(packed)
        w_const = network.add_constant(
            (out_dim, in_dim),
            trt.Weights(trt.DataType.FP8, packed.ctypes.data, out_dim * in_dim),
        ).get_output(0)
        w_dequant = network.add_dequantize(w_const, _scale_const(network, weight_scale), dtype)

        act_scale = _scale_const(network, self.scales[name])
        a_quant = network.add_quantize(x, act_scale, trt.DataType.FP8)
        a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, dtype)

        return network.add_matrix_multiply(
            a_dequant.get_output(0),
            trt.MatrixOperation.NONE,
            w_dequant.get_output(0),
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)


def calibrate_minimax(
    weights,
    *,
    hidden_size: int,
    num_layers: int,
    ffn_dim: int,
    quantization: str,
    stems: tuple[str, ...] = QUANTIZED,
) -> MiniMaxQuantContext:
    """Pair the committed activation scales with the checkpoint, after validating it."""
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"MiniMax-H3 supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    if any(str(name).endswith((".weight_scale", ".input_scale")) for name in weights):
        raise ValueError(
            "the transformer is already quantized; fp8 self-quantization needs the original "
            "bf16 checkpoint"
        )
    activation_scales = load_activation_scales(hidden_size, num_layers, ffn_dim)
    scales: dict[str, float] = {}
    for name in quantized_names(num_layers, stems):
        scale = activation_scales.get(name)
        if scale is None:
            raise ValueError(
                f"{name} has no activation scale in {ACTIVATION_SCALES_FILENAME}; "
                "run calibrate_fp8_scales.py"
            )
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"activation scale of {name} must be finite and positive")
        probe = name.replace("attn.qkv", "attn.to_q")
        if f"{probe}.weight" not in weights:
            raise ValueError(f"checkpoint has no weight for {name}")
        scales[name] = float(scale)
    if not scales:
        raise RuntimeError("MiniMax-H3 FP8 self-quantization matched no projection")
    return MiniMaxQuantContext(scales=scales)
