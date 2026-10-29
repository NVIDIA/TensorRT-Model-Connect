# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AST-owned FP8 TensorRT Q/DQ graph context.

FP8 (``quantization="fp8"``) is self-quantization of the original checkpoint, because
no ModelOpt FP8 checkpoint exists for AST. Weights are quantized at build time, one
projection at a time: ``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3
value), and the FP8 bytes are fed to the engine as a constant. The activation scale of
each projection input needs real audio, so it was calibrated once offline with
``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG`` on AudioSet clips) and is read
from ``fp8_activation_scales.json``. Users do not run calibration. A projection with no
entry in that file is an error. Each quantized projection becomes
``activation -> Quantize -> Dequantize -> MatMul`` with ``FP8 weight -> Dequantize`` on
the other input.

Dequantize and the matmul run in the work dtype (fp16; FP8 builds require it). TensorRT
then fuses the bias and the output cast into the FP8 GEMM epilogue, which is faster than a
float32 Dequantize with a separate cast pass. The FP8 GEMM accumulates in float32 and
applies the scales there, so the output stayed finite on 292 real and stress inputs
(NaN scan). A projection output above the fp16 range would give inf; that scan is the guard.

Quantized projections (short names, ``N`` is the layer index): ``layer.N.query``, ``key``,
``value``, ``attn_out`` (the attention output), ``fc1`` (``intermediate.dense``) and ``fc2``
(``output.dense``) of all twelve layers. The attention is decomposed, so its output
projection is quantized as well. The patch projection, the layer norms, the positions and
the classifier head stay unquantized. Weights keep the checkpoint's ``[out_features,
in_features]`` layout, so the matmul transposes the weight operand.
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
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"

_SUPPORTED = ("fp8",)
_STEMS = ("query", "key", "value", "attn_out", "fc1", "fc2")
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"
# Checkpoint tensor (inside ``layer.N``) of each short name.
HF_SUFFIX = {
    "query": "attention.attention.query",
    "key": "attention.attention.key",
    "value": "attention.attention.value",
    "attn_out": "attention.output.dense",
    "fc1": "intermediate.dense",
    "fc2": "output.dense",
}


@dataclass(frozen=True)
class AstQuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, float]  # short name -> activation scale
    graph: Any
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
        """True when ``name`` (for example ``layer.3.fc1``) is a quantized projection."""
        return name in self.scales

    def linear(self, network, inp, name: str, weight, bias, dtype):
        """``inp @ weight.T + bias``; FP8 for quantized projections, the plain linear otherwise."""
        input_scale = self.scales.get(name)
        if input_scale is None:
            return self.graph.add_linear(network, inp, weight, bias, dtype=dtype)
        packed, weight_scale = quantize_fp8_weight(np.asarray(weight, dtype=np.float32))
        out = _wrap_fp8_matmul(
            network,
            inp,
            packed,
            weight_scale,
            input_scale,
            graph=self.graph,
            keep_alive=self.keep_alive,
        )
        if bias is None:
            return out
        rank = len(inp.shape)
        shape = (1,) * (rank - 1) + (-1,)
        bias = np.asarray(bias, dtype=dtype).reshape(shape)
        constant = self.graph.add_constant(network, bias.shape, bias, dtype=dtype)
        return network.add_elementwise(out, constant, trt.ElementWiseOperation.SUM).get_output(0)


def _scale_const(network, graph, value: float):
    """One-element fp32 constant holding a Q/DQ scale."""
    return graph.add_constant(
        network, (1,), np.asarray([value], dtype=np.float32), dtype=np.float32
    )


def _wrap_fp8_matmul(
    network,
    activation,
    packed: np.ndarray,
    weight_scale: float,
    input_scale: float,
    *,
    graph,
    keep_alive,
):
    """Add ``activation @ weight.T`` with an FP8 weight and a static FP8 activation scale.

    The Dequantize layers and the matmul run in the activation dtype (fp16). The FP8 GEMM
    accumulates in float32 and applies the scales in its epilogue, so large outputs stayed
    finite on 292 real and stress inputs (NaN scan), and the bias and the output cast fuse
    into the GEMM. An output beyond the fp16 range would give inf; the scan guards it.
    """
    n, k = packed.shape
    rank = len(tuple(activation.shape))
    work_type = activation.dtype

    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(packed)
    w_fp8 = trt.Weights(trt.DataType.FP8, packed.ctypes.data, n * k)
    w_const = network.add_constant((1,) * (rank - 2) + (n, k), w_fp8).get_output(0)
    w_dequant = network.add_dequantize(
        w_const, _scale_const(network, graph, weight_scale), work_type
    )

    act_scale = _scale_const(network, graph, input_scale)
    a_quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
    a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, work_type)

    output = network.add_matrix_multiply(
        a_dequant.get_output(0),
        trt.MatrixOperation.NONE,
        w_dequant.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    return output


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

    AudioSet checkpoints of one geometry (the 0.4593, 0.450 and 0.448 releases) share every
    shape but not the weights, so the first classifier biases is part of the key.
    """
    head = np.asarray(weights["classifier.dense.bias"], dtype=np.float32).reshape(-1)[:3]
    fingerprint = "_".join(f"{value:+.4f}" for value in head)
    return f"ast-h{int(cfg['hidden_size'])}-l{int(cfg['num_layers'])}-s{int(cfg['seq_len'])}-{fingerprint}"


def projection_names(cfg: dict) -> list[str]:
    """Short names of every quantizable projection of this model."""
    return [f"layer.{layer}.{stem}" for layer in range(int(cfg["num_layers"])) for stem in _STEMS]


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


def load_activation_scales(key: str) -> dict[str, float]:
    """Return the committed activation scales for this checkpoint, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"AST FP8 build requires {path}, which was not found") from error
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this checkpoint."
        )
    return table[key]


def calibrate_ast(weights, cfg: dict, graph, quantization: str) -> AstQuantContext:
    """Pair the stored activation scales with the loaded projections of this checkpoint."""
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"AST supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    activation_scales = load_activation_scales(scale_key(cfg, weights))
    scales: dict[str, float] = {}
    for name in projection_names(cfg):
        input_scale = activation_scales.get(name)
        if input_scale is None:
            raise ValueError(
                f"{name} has no activation scale in {ACTIVATION_SCALES_FILENAME}; "
                "run calibrate_fp8_scales.py"
            )
        layer, stem = name.split(".")[1], name.split(".")[2]
        key = f"audio_spectrogram_transformer.encoder.layer.{layer}.{HF_SUFFIX[stem]}.weight"
        if key not in weights:
            raise ValueError(f"checkpoint has no {key}")
        if f"{key[: -len('.weight')]}.weight_scale" in weights:
            raise ValueError(
                f"{key} is already quantized; fp8 self-quantization needs the original checkpoint"
            )
        scales[name] = float(input_scale)
    if not scales:
        raise RuntimeError("AST FP8 self-quantization matched no projection in the checkpoint")
    return AstQuantContext(scales=scales, graph=graph)
