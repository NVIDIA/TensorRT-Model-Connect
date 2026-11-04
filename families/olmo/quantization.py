# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OLMo-owned FP8 self-quantization and TensorRT Q/DQ graph context.

The original OLMo checkpoint is used as-is; no FP8 checkpoint is needed (none exists on
the Hub for OLMo-1B-hf).

Weights are quantized on the fly at build time, one projection at a time:
``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3 value), and
the FP8 bytes are fed to the engine as a constant. The weights are taken as
``_OlmoModel.load_weights`` returns them, so the quantized weight is the one the
unquantized engine would use. FP8 weights are never stored. The activation
scale of each projection input needs real data, so it was calibrated once
offline with ``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``) and is
read from ``fp8_activation_scales.json``, keyed by the model config. Users do not
run calibration. A projection without an entry in that file is an error.

Each quantized projection becomes::

    activation -> Quantize -> Dequantize -> MatMul
    FP8 weight -> Dequantize ------------------^

Both Dequantize layers output the activation type, so the FP8 GEMM writes fp16 (or
bf16) and TensorRT fuses the following ops into its epilogue.

Quantized projections: q, k, v, gate, up and down of every layer, except the ``down_proj``
of the layers listed in ``EXCLUDED_PROJECTIONS``. In OLMo-1B the input of those six
``down_proj`` layers has a massive activation (calibrated range 45 to 253 against a
median of 13), so one static 8-bit scale for the whole tensor flattens every other
token. The calibration tool leaves out any projection whose scale is more than 3 times
the median of the same projection (``calibrate_fp8_scales.drop_outlier_scales``). The
usual 20 times limit keeps all of them (the largest is 19 times) and gives a mean KL of
0.069 against the HF model; 3 times gives 0.030 (see the PR text for the measurements).

The attention output projection (``w_o``) stays unquantized: TensorRT 11.4
cannot compile an FP8 Quantize right after a fused attention. ``lm_head`` and
the embedding stay unquantized as well. Tensor parallel builds are not supported.
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
# Projections left unquantized on purpose, by model config (see the module docstring). They
# have no entry in ``fp8_activation_scales.json``; every other projection must have one.
EXCLUDED_PROJECTIONS = {
    "olmo-h2048-l16-i8192": frozenset(f"layer.{layer}.w_down" for layer in (1, 2, 8, 9, 10, 15)),
}
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"
# ``build_route`` is one string: several knobs are joined by this separator.
_ROUTE_SEPARATOR = " "


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
    """Add the FP8 Q/DQ matmul for one projection; Dequantize and matmul use the activation type."""
    w = scales.weight
    n, k = w.out_features, w.in_features

    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(w.packed)
    w_fp8 = trt.Weights(trt.DataType.FP8, w.packed.ctypes.data, n * k)
    w_const = network.add_constant((n, k), w_fp8).get_output(0)
    w_dequant = network.add_dequantize(
        w_const, _scale_const(network, graph_ops, w.weight_scale), activation.dtype
    )

    act_scale = _scale_const(network, graph_ops, scales.input_scale)
    a_quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
    a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, activation.dtype)

    return network.add_matrix_multiply(
        a_dequant.get_output(0),
        trt.MatrixOperation.NONE,
        w_dequant.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)


@dataclass(frozen=True)
class OlmoQuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    graph_ops: Any
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
        """Add the projection as an FP8 matmul, or as the unquantized matmul when it is not quantized."""
        entry = self.scales.get(weight_name)
        if entry is None:
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
    append_build_route(trt_config, _DUAL_GEMM_KNOB)


def append_build_route(trt_config, knob: str) -> None:
    """Turn ``knob`` off in the TensorRT build route, keeping the knobs already set.

    ``build_route`` is a single string, so assigning it would drop an earlier knob. The knob
    is added only when this TensorRT build lists it.
    """
    if knob not in getattr(trt_config, "all_build_routes", ""):
        return
    route = f"-{knob}=off"
    existing = trt_config.build_route
    if route not in existing.split(_ROUTE_SEPARATOR):
        trt_config.build_route = f"{existing}{_ROUTE_SEPARATOR}{route}" if existing else route


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


def load_activation_scales(config) -> dict[str, float]:
    """Return the committed activation scales for this model config, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"OLMo FP8 build requires {path}, which was not found") from error
    key = scale_key(config)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model config."
        )
    return table[key]


def calibrate_olmo_fp8(config, weights, graph_ops) -> OlmoQuantContext:
    """Build the Q/DQ context from the loaded OLMo weights (``[in, out]`` layout).

    Quantizes one projection at a time, so peak extra memory is one weight tensor.
    """
    if any(str(key).endswith((".weight_scale", ".input_scale")) for key in weights):
        raise ValueError(
            "the checkpoint is already quantized; FP8 self-quantization needs the original checkpoint"
        )
    activation_scales = load_activation_scales(config)
    excluded = EXCLUDED_PROJECTIONS.get(scale_key(config), frozenset())

    scales: dict[str, _LayerScales] = {}
    for layer in range(int(config.num_hidden_layers)):
        for _, weight_stem in PROJECTIONS:
            name = f"layer.{layer}.{weight_stem}"
            if name in excluded:
                continue
            input_scale = activation_scales.get(name)
            if input_scale is None:
                raise ValueError(
                    f"{name} has no activation scale in {ACTIVATION_SCALES_FILENAME}; "
                    "run calibrate_fp8_scales.py"
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
        raise RuntimeError("OLMo FP8 self-quantization matched no projection in the checkpoint")
    return OlmoQuantContext(scales=scales, graph_ops=graph_ops)
