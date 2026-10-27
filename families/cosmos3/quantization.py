# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3-owned FP8 lowering of the official ModelOpt FP8 checkpoint.

``quantization="fp8"`` builds from the official ModelOpt FP8 transformer
(``nvidia/Cosmos3-Nano``, ``revision="fp8"``). Its packed weights and scales are used
as stored: no calibration and no re-quantization. Every linear projection of the dual-stream
layers carries an e4m3 ``weight``, a scalar ``weight_scale`` and a scalar static
``input_scale``. Each one becomes::

    activation -> Quantize -> Dequantize -> MatMul -> cast to BF16
    FP8 weight -> Dequantize --------------^

The Dequantize layers and the matmul run in float32. With 16-bit Dequantize outputs
TensorRT's FP8 GEMM keeps the raw accumulator in 16 bits and applies the scales afterwards,
so it overflows once an output exceeds about ``65504 * input_scale * weight_scale``.

Quantized projections, for every layer ``layers.N``:

  understanding stream  ``self_attn.to_q``, ``to_k``, ``to_v`` and
                        ``mlp.gate_proj``, ``up_proj``, ``down_proj``
  generation stream     ``self_attn.add_q_proj``, ``add_k_proj``, ``add_v_proj`` and
                        ``mlp_moe_gen.gate_proj``, ``up_proj``, ``down_proj``

The attention output projections ``self_attn.to_out`` and ``self_attn.to_add_out`` run in BF16
with the checkpoint's FP8 values dequantized at build time (``fp8 * weight_scale``). TensorRT
11.4 cannot compile an FP8 Quantize on the output of the fused attention: the compiler fails
with ``Could not infer output types for operation: softmax`` on an fp8 operand, also for a tiny
model. TensorRT 11.1 builds them either way. They hold about 1/9 of the projection weights.

The embedding, the norms, the time embedder and the input and output projections stay in
BF16, as in the checkpoint. The checkpoint's step policy (BF16-activation steps at the start
and the end of the schedule) is not applied: a TensorRT plan has one precision for every step.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import tensorrt as trt

FP8_E4M3_MAX = 448.0

# Build-route knob that turns off TensorRT's fused dual-GEMM kernel.
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"

# Projection stems inside one layer (the weight key is ``layers.N.<stem>.weight``).
LAYER_PROJECTIONS = (
    "self_attn.to_q",
    "self_attn.to_k",
    "self_attn.to_v",
    "self_attn.to_out",
    "self_attn.add_q_proj",
    "self_attn.add_k_proj",
    "self_attn.add_v_proj",
    "self_attn.to_add_out",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "mlp_moe_gen.gate_proj",
    "mlp_moe_gen.up_proj",
    "mlp_moe_gen.down_proj",
)

# Projections kept in BF16 although the checkpoint stores them in FP8: the attention output
# projections (see the module docstring). Their weights are dequantized once at build time.
_BF16_PROJECTIONS = re.compile(r"\.self_attn\.(to_out|to_add_out)$")


def projection_names(num_layers: int) -> list[str]:
    """Names (``layers.N.<stem>``) of every projection the checkpoint quantizes."""
    return [f"layers.{layer}.{stem}" for layer in range(num_layers) for stem in LAYER_PROJECTIONS]


def is_fp8_state(state: Mapping[str, Any]) -> bool:
    """True when the state dict holds ModelOpt FP8 projections (it has ``weight_scale`` tensors)."""
    return any(key.endswith(".weight_scale") for key in state)


def is_fp8_checkpoint(transformer_dir: str | Path) -> bool:
    """True when the transformer index lists ``weight_scale`` tensors (no weights are read)."""
    index = Path(transformer_dir) / "diffusion_pytorch_model.safetensors.index.json"
    if not index.is_file():
        return False
    weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
    return any(str(key).endswith(".weight_scale") for key in weight_map)


@dataclass(frozen=True)
class _FP8Weight:
    """One projection weight as the checkpoint's raw e4m3 bytes with its per-tensor scale."""

    packed: np.ndarray  # [out_features, in_features] raw fp8_e4m3 bytes
    weight_scale: float
    out_features: int
    in_features: int


def _positive_scalar(value: Any, name: str) -> float:
    """One finite positive scale from a scalar tensor."""
    array = np.asarray(value.detach().float().cpu().numpy() if hasattr(value, "detach") else value)
    array = array.astype(np.float32).reshape(-1)
    if array.size != 1 or not np.isfinite(array[0]) or array[0] <= 0:
        raise ValueError(f"{name} must hold one finite positive scale")
    return float(array[0])


def _raw_fp8_bytes(tensor: Any, name: str) -> np.ndarray:
    """The checkpoint's e4m3 weight as a contiguous uint8 array (no dtype conversion)."""
    import torch

    if getattr(tensor, "dtype", None) != torch.float8_e4m3fn:
        raise ValueError(f"{name} is not a float8_e4m3 tensor")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 weight")
    return np.ascontiguousarray(tensor.view(torch.uint8).cpu().numpy())


@dataclass
class Cosmos3QuantContext:
    """Routes the quantized projections through FP8 Q/DQ; others stay in BF16."""

    weights: dict[str, _FP8Weight]
    input_scales: dict[str, float]
    keep_alive: list = field(default_factory=list)

    @property
    def has_fp8_weights(self) -> bool:
        """True when any projection is FP8 (selects the FP8 build options)."""
        return bool(self.weights)

    def should_quantize(self, name: str) -> bool:
        """True when ``name`` (for example ``layers.3.self_attn.to_k``) runs as an FP8 GEMM."""
        return name in self.weights and not _BF16_PROJECTIONS.search(name)

    def dequantized_weight(self, name: str) -> np.ndarray:
        """The projection weight as float32 ``[out, in]``, for projections that stay in BF16."""
        import ml_dtypes

        weight = self.weights[name]
        values = weight.packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
        return values * np.float32(weight.weight_scale)

    def matmul(self, network, x, name: str):
        """``x @ weight.T`` for ``x`` of shape ``[rows, in]`` with the checkpoint's FP8 weight."""
        weight = self.weights[name]
        n, k = weight.out_features, weight.in_features
        # trt.Weights holds a raw pointer, so the buffer must outlive the build.
        self.keep_alive.append(weight.packed)
        w_const = network.add_constant(
            (n, k), trt.Weights(trt.DataType.FP8, weight.packed.ctypes.data, n * k)
        ).get_output(0)
        w_dequant = network.add_dequantize(
            w_const, _scale_const(network, weight.weight_scale), trt.float32
        )
        act_scale = _scale_const(network, self.input_scales[name])
        a_quant = network.add_quantize(x, act_scale, trt.DataType.FP8)
        a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, trt.float32)
        out = network.add_matrix_multiply(
            a_dequant.get_output(0),
            trt.MatrixOperation.NONE,
            w_dequant.get_output(0),
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        return network.add_cast(out, x.dtype).get_output(0)


def _scale_const(network, value: float):
    """One-element fp32 constant holding a Q/DQ scale."""
    arr = np.asarray([value], dtype=np.float32)
    return network.add_constant((1,), trt.Weights(arr)).get_output(0)


def configure_build(trt_config) -> None:
    """Apply the FP8 build options and keep layer names for kernel inspection."""
    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    # TensorRT's fused dual-GEMM kernel fails to compile with scalar-scale FP8.
    # The route is set only when this TensorRT build lists the knob.
    if _DUAL_GEMM_KNOB in getattr(trt_config, "all_build_routes", ""):
        trt_config.build_route = f"-{_DUAL_GEMM_KNOB}=off"


def load_fp8_projections(state: Mapping[str, Any], num_layers: int) -> Cosmos3QuantContext:
    """Collect the checkpoint's FP8 weights and scales, after checking every projection.

    A projection is quantized when the checkpoint stores it in FP8 with both scales. One that
    is stored in BF16 stays BF16. A half-quantized projection (FP8 bytes without scales, or
    scales on a BF16 weight) is an error, so a malformed file never builds silently.
    """
    import torch

    weights: dict[str, _FP8Weight] = {}
    input_scales: dict[str, float] = {}
    for name in projection_names(num_layers):
        weight = state.get(f"{name}.weight")
        if weight is None:
            raise ValueError(f"checkpoint has no {name}.weight")
        has_scales = f"{name}.weight_scale" in state and f"{name}.input_scale" in state
        is_fp8 = getattr(weight, "dtype", None) == torch.float8_e4m3fn
        if is_fp8 != has_scales:
            raise ValueError(
                f"{name} is {'FP8' if is_fp8 else 'not FP8'} but "
                f"{'has' if has_scales else 'lacks'} weight_scale and input_scale"
            )
        if not is_fp8:
            continue
        packed = _raw_fp8_bytes(weight, f"{name}.weight")
        weights[name] = _FP8Weight(
            packed,
            _positive_scalar(state[f"{name}.weight_scale"], f"{name}.weight_scale"),
            packed.shape[0],
            packed.shape[1],
        )
        input_scales[name] = _positive_scalar(state[f"{name}.input_scale"], f"{name}.input_scale")
    if not weights:
        raise ValueError(
            "no FP8 projections found; fp8 builds need the ModelOpt FP8 checkpoint "
            "(nvidia/Cosmos3-Nano, revision fp8)"
        )
    return Cosmos3QuantContext(weights=weights, input_scales=input_scales)
