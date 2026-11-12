# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lance-owned FP8 self-quantization and TensorRT Q/DQ graph context.

The original BF16 Lance checkpoint is used as-is; no FP8 checkpoint is needed
(the only public quantized Lance checkpoints are community MLX / custom-format
files, not ModelOpt exports TensorRT can consume).

Only the language-model projections of the understanding decoder are
quantized (q, k, v, gate, up, down of every layer). The ViT, the vision
projector, lm_head, the embedding, the norms and the attention output
projection stay unquantized.

Weights are quantized on the fly at build time, one projection at a time:
``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3 value), and
the FP8 bytes are fed to the engine as a constant. The activation scale of each
projection input needs real data, so it was calibrated once offline with
``calibrate_fp8_scales.py`` (image-text and text prompts through the BF16
decoder) and is read from ``fp8_activation_scales.json``. Users do not run
calibration.

Each quantized projection becomes::

    activation -> Quantize -> Dequantize -> MatMul -> Cast
    FP8 weight -> Dequantize -----------------^

The Dequantize and the matmul run in fp32 and the result is cast back to bf16
(``DEQUANTIZE_DTYPE = "float32"``). A Dequantize in the engine dtype (bf16) gives
the same speed, but on TensorRT 11.4 the quantized engine then returns wrong
logits (first-token cosine 0.40 against the unquantized engine; TensorRT 11.1 is
correct), so it is not used. The build refuses fp16 because the down-projection
inputs of Lance reach 1e5, above the fp16 range. The down projections of layers 2
to 10 and 35 have input maxima 20 to 250 times the median of their kind; they are
listed under ``excluded`` in the scales file and stay unquantized.

The attention output projection (``w_o``) stays unquantized: TensorRT 11.4
cannot compile an FP8 Quantize right after a fused attention.
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

# HF projection stem -> internal weight-dict stem used by graph_blocks.
PROJECTIONS = (
    ("self_attn.q_proj", "w_q"),
    ("self_attn.k_proj", "w_k"),
    ("self_attn.v_proj", "w_v"),
    ("mlp.gate_proj", "w_gate"),
    ("mlp.up_proj", "w_up"),
    ("mlp.down_proj", "w_down"),
)
_PROJECTION_STEMS = frozenset(stem for _, stem in PROJECTIONS)
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"
# "float32" runs Dequantize and the matmul in fp32 (then casts back);
# "activation" runs them in the activation dtype of the engine (wrong results on TensorRT 11.4).
DEQUANTIZE_DTYPE = "float32"


def scale_key(config) -> str:
    """Key of this model's entry in the activation-scale file."""
    return (
        f"{str(config.model_type).lower()}"
        f"-h{int(config.hidden_size)}"
        f"-l{int(config.num_hidden_layers)}"
        f"-i{int(config.intermediate_size)}"
    )


@dataclass(frozen=True)
class _FP8Weight:
    """One projection weight stored as raw fp8_e4m3 bytes with one per-tensor scale."""

    packed: np.ndarray  # [out_features, in_features] raw fp8_e4m3 bytes
    weight_scale: float
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _LayerScales:
    """Static activation scale and FP8 weight of one quantized projection."""

    input_scale: float
    weight: _FP8Weight


def _scale_const(network, graph_ops, value: float):
    """Add a one-element fp32 scale constant to the network."""
    return graph_ops.add_constant(
        network, (1,), np.asarray([value], dtype=np.float32), dtype=np.float32
    )


def _wrap_fp8_matmul(network, activation, scales: _LayerScales, *, graph_ops, keep_alive: list):
    """Add the FP8 Q/DQ matmul for one projection."""
    w = scales.weight
    n, k = w.out_features, w.in_features
    dq_dtype = trt.float32 if DEQUANTIZE_DTYPE == "float32" else activation.dtype

    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(w.packed)
    w_fp8 = trt.Weights(trt.DataType.FP8, w.packed.ctypes.data, n * k)
    w_const = network.add_constant((n, k), w_fp8).get_output(0)
    w_dequant = network.add_dequantize(
        w_const, _scale_const(network, graph_ops, w.weight_scale), dq_dtype
    )

    act_scale = _scale_const(network, graph_ops, scales.input_scale)
    a_quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
    a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, dq_dtype)

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
class LanceQuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    graph_ops: Any
    excluded: frozenset = frozenset()
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        """Give each context its own list for buffers that must outlive the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, weight_name: str) -> bool:
        """Return True when this weight has a calibrated activation scale."""
        return weight_name in self.scales

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
        """Add the projection as an FP8 matmul, or as the unquantized matmul."""
        entry = self.scales.get(weight_name)
        if entry is None:
            stem = weight_name.rsplit(".", 1)[-1]
            if stem in _PROJECTION_STEMS and weight_name not in self.excluded:
                raise ValueError(
                    f"projection {weight_name} has no FP8 activation scale and is not "
                    f"excluded; run calibrate_fp8_scales.py for this model config"
                )
            return self.graph_ops.add_matmul_rhs_constant(
                network, lhs, lhs_width, rhs_width, rhs_weights, dtype=dtype
            )
        return _wrap_fp8_matmul(
            network, lhs, entry, graph_ops=self.graph_ops, keep_alive=self.keep_alive
        )


def configure_fp8_build(trt_config, quant_ctx) -> None:
    """Apply the FP8 build options; a no-op for an unquantized build."""
    if quant_ctx is None:
        return
    # Keeps per-layer tactic names in the plan so FP8 lowering can be inspected.
    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    # TensorRT's fused dual-GEMM (gate/up) kernel fails to compile with scalar-scale FP8
    # inputs. The route is set only when this TensorRT build lists the knob.
    if _DUAL_GEMM_KNOB in getattr(trt_config, "all_build_routes", ""):
        trt_config.build_route = f"-{_DUAL_GEMM_KNOB}=off"


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


def load_activation_scales(config) -> tuple[dict[str, float], frozenset]:
    """Return the committed activation scales and exclusions for this model config."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Lance FP8 build requires {path}, which was not found") from error
    key = scale_key(config)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model config."
        )
    entry = table[key]
    return entry["input_scales"], frozenset(entry.get("excluded", ()))


def calibrate_lance_fp8(config, weights, graph_ops) -> LanceQuantContext:
    """Build the Q/DQ context from the loaded Lance weights (``[in, out]`` layout).

    Quantizes one projection at a time, so peak extra memory is one weight tensor.
    """
    if any(str(key).endswith((".weight_scale", ".input_scale")) for key in weights):
        raise ValueError("the checkpoint is already quantized; FP8 self-quantization needs BF16")
    activation_scales, excluded = load_activation_scales(config)

    scales: dict[str, _LayerScales] = {}
    for layer in range(int(config.num_hidden_layers)):
        for _, weight_stem in PROJECTIONS:
            name = f"layer.{layer}.{weight_stem}"
            if name in excluded:
                continue
            input_scale = activation_scales.get(name)
            if input_scale is None:
                raise ValueError(
                    f"projection {name} has no FP8 activation scale in "
                    f"{ACTIVATION_SCALES_FILENAME} and is not excluded; run "
                    f"calibrate_fp8_scales.py for this model config"
                )
            if not np.isfinite(input_scale) or input_scale <= 0:
                raise ValueError(f"activation scale of {name} must be finite and positive")
            if name not in weights:
                raise ValueError(f"checkpoint has no {name}")
            transposed = np.asarray(weights[name])
            in_features, out_features = transposed.shape
            packed, weight_scale = quantize_fp8_weight(
                np.ascontiguousarray(transposed.T, dtype=np.float32)
            )
            scales[name] = _LayerScales(
                input_scale=float(input_scale),
                weight=_FP8Weight(packed, weight_scale, out_features, in_features),
            )
    if not scales:
        raise RuntimeError("Lance FP8 self-quantization matched no projection in the checkpoint")
    return LanceQuantContext(scales=scales, graph_ops=graph_ops, excluded=excluded)
