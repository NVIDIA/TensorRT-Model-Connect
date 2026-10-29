# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan-T2V-owned FP8 self-quantization of the DiT feed-forward projections.

``quantization="fp8"`` quantizes the original transformer at build time; no FP8 checkpoint
is needed or written. Weights are quantized one projection at a time:
``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3 value), and the FP8
bytes are fed to the engine as a constant. The activation scale of each projection input
needs real denoising activations, so it was calibrated once offline with
``calibrate_fp8_scales.py`` (max of the input over real prompts) and is read from
``fp8_activation_scales.json``. Users do not run calibration. A projection with no entry
in that file stays in fp16.

Each quantized projection becomes::

    activation -> Quantize -> Dequantize (fp16) -> MatMul (fp16)
    FP8 weight -> Dequantize (fp16) ------------------^

The Dequantize layers output fp16, so TensorRT keeps the bias add and the GELU in the GEMM
epilogue. A float32 Dequantize makes the GEMM write float32 and adds a separate cast, bias and
GELU kernel that cancels most of the gain (measured 1.03x to 1.08x against 1.10x to 1.20x
per DiT call), and it crashes the builder at 81 frames. A projection output above
65504 would give inf or NaN in fp16; none appeared on real denoising inputs (every output of
every step of 24 videos was scanned, see the PR).

Quantized projections, for every block ``blocks.N`` (60 in total):

  feed-forward  ``ffn.net.0.proj`` (fc1), ``ffn.net.2`` (fc2)

Everything else stays in fp16: the self- and cross-attention q/k/v and output projections,
the adaLN modulation, the patch, time and text embeddings (they run in the C++ runtime), the
final projection, the text encoder and the VAE. The attention output projections cannot be
quantized on TensorRT 11.4 (an FP8 Quantize on the output of the fused attention fails to
build, also for a tiny model; 11.1 builds it). Quantizing q/k/v too gave 4% to 8% more speed and no
consistent quality gain (see the PR).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import ml_dtypes
import numpy as np
import tensorrt as trt

from . import graph_ops
from .checkpoint_mapper import _open_safetensors

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"

_SUPPORTED = ("fp8",)
# Projection stems inside one block (the weight key is ``blocks.N.<stem>.weight``).
BLOCK_PROJECTIONS = ("ffn.net.0.proj", "ffn.net.2")


def scale_key(dim: int, num_layers: int, ffn_dim: int) -> str:
    """Key of this model shape's entry in the activation-scale file."""
    return f"wan_t2v-h{int(dim)}-l{int(num_layers)}-i{int(ffn_dim)}"


def quantized_names(num_layers: int) -> list[str]:
    """Names (``blocks.N.<stem>``) of every projection that can be quantized."""
    return [f"blocks.{block}.{stem}" for block in range(num_layers) for stem in BLOCK_PROJECTIONS]


def quantize_fp8_weight(weight_fp32: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize one ``[out, in]`` weight to FP8 e4m3 with a single scalar scale.

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


def load_activation_scales(dim: int, num_layers: int, ffn_dim: int) -> dict[str, float]:
    """Return the committed activation scales for this model shape, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Wan T2V FP8 build requires {path}, which was not found") from error
    key = scale_key(dim, num_layers, ffn_dim)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model shape."
        )
    return table[key]


def reject_quantized_checkpoint(transformer_dir: str | Path) -> None:
    """Refuse a transformer that already stores quantization scales."""
    readers = _open_safetensors(Path(transformer_dir))
    if any(name.endswith((".weight_scale", ".input_scale")) for name in readers.tensor_map):
        raise ValueError(
            "the transformer is already quantized; fp8 self-quantization needs the original checkpoint"
        )


@dataclass
class WanQuantContext:
    """Routes the quantized projections through FP8 Q/DQ; others stay in fp16."""

    scales: dict[str, float]
    keep_alive: list = field(default_factory=list)

    @property
    def has_fp8_weights(self) -> bool:
        """True when any projection is FP8 (selects the FP8 build options)."""
        return bool(self.scales)

    def should_quantize(self, name: str) -> bool:
        """True when ``name`` (for example ``blocks.3.attn1.to_k``) has a scale."""
        return name in self.scales

    def maybe_quantized_matmul(
        self,
        network,
        lhs,
        lhs_width: int,
        rhs_width: int,
        rhs_weights: np.ndarray,
        weight_name: str,
        dtype: np.dtype = np.float32,
    ):
        """``lhs @ rhs_weights`` as an FP8 matmul, or the plain constant matmul without a scale.

        ``rhs_weights`` is the engine layout ``[in, out]``; the FP8 constant keeps the
        checkpoint layout ``[out, in]`` and the matmul transposes it.
        """
        if weight_name not in self.scales:
            return graph_ops.add_matmul_rhs_constant(
                network, lhs, lhs_width, rhs_width, rhs_weights, dtype=dtype
            )
        return self._fp8_matmul(network, lhs, rhs_weights, weight_name)

    def _fp8_matmul(self, network, x, weight_in_out: np.ndarray, name: str):
        """``x @ weight`` for a 2-D fp16 ``x`` with an FP8 weight; every layer outputs fp16."""
        if x.dtype != trt.float16:
            raise ValueError("Wan DiT FP8 projections need fp16 activations")
        w_hf = np.ascontiguousarray(np.asarray(weight_in_out, dtype=np.float32).T)
        out_dim, in_dim = w_hf.shape
        packed, weight_scale = quantize_fp8_weight(w_hf)
        # trt.Weights holds a raw pointer, so the buffer must outlive the build.
        self.keep_alive.append(packed)
        w_const = network.add_constant(
            (out_dim, in_dim),
            trt.Weights(trt.DataType.FP8, packed.ctypes.data, out_dim * in_dim),
        ).get_output(0)
        w_dequant = network.add_dequantize(
            w_const, _scale_const(network, weight_scale), trt.float16
        )

        act_scale = _scale_const(network, self.scales[name])
        a_quant = network.add_quantize(x, act_scale, trt.DataType.FP8)
        a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, trt.float16)

        return network.add_matrix_multiply(
            a_dequant.get_output(0),
            trt.MatrixOperation.NONE,
            w_dequant.get_output(0),
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)


def _scale_const(network, value: float):
    """One-element fp32 constant holding a Q/DQ scale."""
    arr = np.asarray([value], dtype=np.float32)
    return network.add_constant((1,), trt.Weights(arr)).get_output(0)


def configure_build(trt_config) -> None:
    """Keep layer names and kernel names in the plan for inspection."""
    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED


def calibrate_wan_t2v(
    weights: dict, quantization: str, *, dim: int, num_layers: int, ffn_dim: int
) -> WanQuantContext:
    """Pair the committed activation scales with the loaded DiT weights, after validating them."""
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"Wan T2V supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    activation_scales = load_activation_scales(dim, num_layers, ffn_dim)
    scales: dict[str, float] = {}
    for name in quantized_names(num_layers):
        scale = activation_scales.get(name)
        if scale is None:
            continue
        if f"{name}.weight" not in weights:
            raise ValueError(f"checkpoint has no {name}.weight")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"activation scale of {name} must be finite and positive")
        scales[name] = float(scale)
    if not scales:
        raise RuntimeError("Wan T2V FP8 self-quantization matched no projection in the checkpoint")
    return WanQuantContext(scales=scales)
