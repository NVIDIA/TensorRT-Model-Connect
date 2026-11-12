# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eagle-VLM-owned FP8 TensorRT Q/DQ graph context for ModelOpt checkpoints.

Supported checkpoints are the ModelOpt exports ``nvidia/llama-nemotron-embed-vl-1b-v2-fp8``
and ``nvidia/llama-nemotron-rerank-vl-1b-v2-fp8``. Every language-model attention
(q/k/v/o) and MLP (gate/up/down) projection carries:

  ``<name>.weight`` (float8_e4m3), ``<name>.weight_scale`` (per-tensor fp32) and
  ``<name>.input_scale`` (per-tensor fp32).

The attention output projection ``o_proj`` is not used from the checkpoint: it stays fp16
because TensorRT 11.4 cannot compile an FP8 Quantize right after the fused attention.
The vision tower (``vision_model.*``), the projector (``mlp1.*``), the embedding, the norms
and the score head stay unquantized. The checkpoint's ``hf_quant_config.json`` lists them in
``exclude_modules``. A language-model projection without scales that is not listed there is
an error.

The checkpoint's own FP8 bytes are fed to the engine as-is (no dequantize/re-quantize round
trip). TensorRT's ``add_dequantize(value, scale)`` computes ``value * scale``, which
reproduces the exported weight exactly, and the Quantize -> Dequantize -> MatMul pattern
is fused into a native FP8 tensor-core GEMM. Weights keep the checkpoint's
``[out_features, in_features]`` layout, so the matmul transposes the weight operand.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

_DUAL_GEMM_KNOB = "peep:match_dual_gemm"
# HF projection stem -> internal weight-dict stem used by the engine builder.
PROJECTIONS = (
    ("self_attn.q_proj", "w_q"),
    ("self_attn.k_proj", "w_k"),
    ("self_attn.v_proj", "w_v"),
    ("self_attn.o_proj", "w_o"),
    ("mlp.gate_proj", "w_gate"),
    ("mlp.up_proj", "w_up"),
    ("mlp.down_proj", "w_down"),
)
_SUPPORTED = ("fp8",)
# Attention output projections stay in fp16 even though ModelOpt calibrates them:
# TensorRT 11.4 cannot compile an FP8 Quantize right after the fused attention.
UNQUANTIZED_STEMS = ("w_o",)
LAYER_PREFIXES = ("language_model.layers", "model.language_model.layers")


@dataclass(frozen=True)
class _FP8Weight:
    """The checkpoint's own FP8 weight, reused bit-exact."""

    packed: np.ndarray  # [out_features, in_features] raw fp8_e4m3 bytes
    weight_scale: float
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _LayerScales:
    """Activation scale and the packed weight of one quantized projection."""

    input_scale: float
    weight: _FP8Weight


def _scale_const(network, graph_ops, value: float):
    """One-element fp32 constant holding a Q/DQ scale."""
    return graph_ops.add_constant(
        network, (1,), np.asarray([value], dtype=np.float32), dtype=np.float32
    )


def _wrap_fp8_matmul(network, activation, scales: _LayerScales, *, graph_ops, keep_alive: list):
    """Add ``activation @ weight.T`` with FP8 weights and a static FP8 activation scale.

    The Dequantize layers and the matmul run in float32 and the result is cast back.
    With fp16 Dequantize outputs, TensorRT's FP8 GEMM keeps the raw accumulator in
    fp16 and applies the scales afterwards, so it overflows to NaN as soon as an
    output exceeds about ``65504 * input_scale * weight_scale``. Large outputs do
    occur in real models.
    """
    import tensorrt as trt

    w = scales.weight
    n, k = w.out_features, w.in_features
    # The reranker runs on [batch, tokens, hidden] activations: the weight gets leading
    # unit dimensions so both matmul operands have the same rank.
    rank = len(tuple(activation.shape))
    w_shape = (n, k) if rank <= 2 else (1,) * (rank - 2) + (n, k)

    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(w.packed)
    w_fp8 = trt.Weights(trt.DataType.FP8, w.packed.ctypes.data, n * k)
    w_const = network.add_constant(w_shape, w_fp8).get_output(0)
    w_dequant = network.add_dequantize(
        w_const, _scale_const(network, graph_ops, w.weight_scale), trt.float32
    )

    act_scale = _scale_const(network, graph_ops, scales.input_scale)
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
class EagleQuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    graph_ops: Any
    # Projections kept fp16 although the checkpoint stores them as FP8: the dequantized
    # weight (``fp8 * weight_scale``, [out_features, in_features], float32) by internal name.
    dense_weights: dict[str, np.ndarray] = None  # type: ignore[assignment]
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        """Create the list that keeps weight buffers alive during the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])
        if self.dense_weights is None:
            object.__setattr__(self, "dense_weights", {})

    def should_quantize(self, weight_name: str) -> bool:
        """True when ``weight_name`` (for example ``layer.3.w_q``) is a quantized projection."""
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
        """Quantized matmul for quantized projections; the regular constant matmul otherwise."""
        entry = self.scales.get(weight_name)
        if entry is None:
            return self.graph_ops.add_matmul_rhs_constant(
                network, lhs, lhs_width, rhs_width, rhs_weights, dtype=dtype
            )
        return _wrap_fp8_matmul(
            network, lhs, entry, graph_ops=self.graph_ops, keep_alive=self.keep_alive
        )


def configure_fp8_build(trt_config, quant_ctx) -> None:
    """Apply the FP8 build options and keep layer names for kernel inspection."""
    if quant_ctx is None:
        return
    import tensorrt as trt

    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if _DUAL_GEMM_KNOB in getattr(trt_config, "all_build_routes", ""):
        # TensorRT's fused dual-GEMM kernel fails to compile with scalar-scale FP8.
        # The route is set only when this TensorRT build lists the knob.
        trt_config.build_route = f"-{_DUAL_GEMM_KNOB}=off"


def _open_raw_readers(model_dir: Path, *, framework: str = "pt") -> dict[str, Any]:
    """Map tensor name -> reader (raw dtypes, no conversion).

    ``framework="numpy"`` is enough to list tensor names and does not need torch.
    """
    single = model_dir / "model.safetensors"
    if single.is_file():
        reader = safe_open(str(single), framework=framework)
        return {name: reader for name in reader.keys()}
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"No model.safetensors checkpoint in {model_dir}")
    weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model.safetensors.index.json has no weight_map")
    by_file = {
        shard: safe_open(str(model_dir / shard), framework=framework)
        for shard in sorted(set(weight_map.values()))
    }
    return {name: by_file[shard] for name, shard in weight_map.items()}


def _scalar(tensors: dict[str, Any], key: str) -> float | None:
    """The tensor as a positive finite float, or None when it is absent or not valid."""
    reader = tensors.get(key)
    if reader is None:
        return None
    value = reader.get_tensor(key).float().numpy().reshape(-1)
    if value.size != 1 or not np.isfinite(value[0]) or value[0] <= 0:
        return None
    return float(value[0])


def _raw_bytes(tensors: dict[str, Any], key: str) -> np.ndarray:
    """The tensor's raw bytes as a contiguous uint8 array (no dtype conversion)."""
    import torch

    tensor = tensors[key].get_tensor(key)
    return np.ascontiguousarray(tensor.view(torch.uint8).numpy())


def is_quantized_checkpoint(model_dir: Path) -> bool:
    """True when the checkpoint stores quantized projections (it has ``weight_scale`` tensors)."""
    try:
        tensors = _open_raw_readers(Path(model_dir), framework="numpy")
    except FileNotFoundError:
        return False
    return any(name.endswith(".weight_scale") for name in tensors)


def _excluded_modules(model_dir: Path) -> frozenset[str]:
    """The module names or glob patterns ModelOpt left unquantized, from ``hf_quant_config.json``."""
    path = Path(model_dir) / "hf_quant_config.json"
    if not path.is_file():
        return frozenset()
    quant = json.loads(path.read_text(encoding="utf-8")).get("quantization", {})
    return frozenset(quant.get("exclude_modules", ()))


def _read_fp8(tensors, prefix: str) -> _FP8Weight | None:
    """The FP8 weight under ``prefix``, or None when it is not a float8_e4m3 tensor."""
    weight_key, scale_key = f"{prefix}.weight", f"{prefix}.weight_scale"
    if weight_key not in tensors or scale_key not in tensors:
        return None
    if tensors[weight_key].get_slice(weight_key).get_dtype() != "F8_E4M3":
        return None
    packed = _raw_bytes(tensors, weight_key)
    weight_scale = _scalar(tensors, scale_key)
    if weight_scale is None:
        raise ValueError(f"{scale_key} must be one positive finite value")
    out_features, in_features = packed.shape
    return _FP8Weight(packed, weight_scale, out_features, in_features)


def _dequantized_weight(tensors, prefix: str) -> np.ndarray:
    """The checkpoint's FP8 weight as float32 (``fp8 * weight_scale``) for a projection kept fp16."""
    import torch

    weight = _read_fp8(tensors, prefix)
    if weight is None:
        raise ValueError(f"{prefix} is not an fp8 ModelOpt tensor")
    values = torch.from_numpy(weight.packed).view(torch.float8_e4m3fn).float().numpy()
    return values * np.float32(weight.weight_scale)


def layer_prefix(tensors) -> str:
    """Checkpoint path prefix of the language-model layers (without the layer index)."""
    for prefix in LAYER_PREFIXES:
        if f"{prefix}.0.input_layernorm.weight" in tensors:
            return prefix
    raise RuntimeError("Cannot find text decoder layer weights")


def calibrate_eagle(model_dir: Path, graph_ops, quantization: str, config) -> EagleQuantContext:
    """Build the Q/DQ context from a ModelOpt FP8 Eagle checkpoint.

    The checkpoint already carries calibrated weights and activation scales, so no
    forward pass is needed. Raises when the checkpoint is not an FP8 ModelOpt export,
    instead of silently falling back to unquantized weights.
    """
    if quantization not in _SUPPORTED:
        raise NotImplementedError(f"eagle_vlm does not support quantization={quantization!r}")
    model_dir = Path(model_dir)
    tensors = _open_raw_readers(model_dir)
    excluded = _excluded_modules(model_dir)
    hint = "nvidia/llama-nemotron-embed-vl-1b-v2-fp8 or nvidia/llama-nemotron-rerank-vl-1b-v2-fp8"
    prefix_root = layer_prefix(tensors)

    scales: dict[str, _LayerScales] = {}
    dense: dict[str, np.ndarray] = {}
    for layer in range(int(config.num_hidden_layers)):
        for hf_stem, weight_stem in PROJECTIONS:
            prefix = f"{prefix_root}.{layer}.{hf_stem}"
            if weight_stem in UNQUANTIZED_STEMS:
                if f"{prefix}.weight_scale" in tensors:
                    dense[f"layer.{layer}.{weight_stem}"] = _dequantized_weight(tensors, prefix)
                continue
            if f"{prefix}.weight_scale" not in tensors:
                # ModelOpt left this projection unquantized. That is only valid when
                # the checkpoint says so, and never for a plain full-precision file.
                if any(fnmatch.fnmatchcase(prefix, pattern) for pattern in excluded):
                    continue
                raise ValueError(
                    f"{prefix} has no fp8 scales and is not listed in exclude_modules; "
                    f"eagle_vlm fp8 builds need a ModelOpt checkpoint such as {hint}"
                )
            scale_key = f"{prefix}.input_scale"
            weight = _read_fp8(tensors, prefix)
            if weight is None or scale_key not in tensors:
                raise ValueError(
                    f"{prefix} is not an fp8 ModelOpt tensor; "
                    f"eagle_vlm fp8 builds need a checkpoint such as {hint}"
                )
            input_scale = _scalar(tensors, scale_key)
            if input_scale is None:
                raise ValueError(f"{scale_key} must be one positive finite value")
            scales[f"layer.{layer}.{weight_stem}"] = _LayerScales(input_scale, weight)
    if not scales:
        raise ValueError(f"no fp8 projections found in {model_dir}")
    return EagleQuantContext(scales=scales, graph_ops=graph_ops, dense_weights=dense)
