# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5-owned NVFP4 TensorRT Q/DQ graph context for ModelOpt checkpoints.

Supported checkpoints are ModelOpt NVFP4 exports of Qwen3.5 text models, such as
``AxionML/Qwen3.5-9B-NVFP4``. Every projection of the language model carries:

  ``<name>.weight`` (E2M1, two values per uint8), ``<name>.weight_scale``
  (per-16-element float8_e4m3), ``<name>.weight_scale_2`` (global fp32) and
  ``<name>.input_scale`` (fp32).

That covers the MLP of every layer, the q/k/v/o projections of the full-attention
layers, and ``in_proj_qkv``, ``in_proj_z``, ``in_proj_a``, ``in_proj_b`` and
``out_proj`` of the DeltaNet layers. ``conv1d``, the norms, the embedding and
``lm_head`` stay unquantized.

The checkpoint's own quantized bytes are fed to the engine as-is (no
dequantize/re-quantize round trip). TensorRT's ``add_dequantize(value, scale)``
computes ``value * scale``, which reproduces the exported weight exactly, and the
DynamicQuantize -> Dequantize -> MatMul pattern is fused into a native FP4
tensor-core GEMM. Weights keep the checkpoint's ``[out_features, in_features]``
layout, so the matmul transposes the weight operand instead of repacking
sub-byte data.

The full-attention ``q_proj`` holds the query and the output gate of every head
interleaved along its rows. The engine uses them as two projections, so the rows
are split here. Rows are independent in the packed weight and the block scales,
so each half is still the checkpoint's own bytes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorrt as trt
import torch
from safetensors import safe_open

NVFP4_BLOCK_SIZE = 16

_SUPPORTED = ("nvfp4",)
# (HF projection stem, internal weight-dict stem) for each layer kind.
_MLP_PROJECTIONS = (
    ("mlp.gate_proj", "w_gate"),
    ("mlp.up_proj", "w_up"),
    ("mlp.down_proj", "w_down"),
)
_ATTENTION_PROJECTIONS = (
    ("self_attn.k_proj", "w_k"),
    ("self_attn.v_proj", "w_v"),
    ("self_attn.o_proj", "w_o"),
)
_DELTANET_PROJECTIONS = (
    ("linear_attn.in_proj_qkv", "deltanet_in_proj_qkv"),
    ("linear_attn.in_proj_z", "deltanet_z_proj"),
    ("linear_attn.in_proj_a", "deltanet_a_proj"),
    ("linear_attn.in_proj_b", "deltanet_b_proj"),
    ("linear_attn.out_proj", "deltanet_out_proj"),
)
_Q_PROJECTION = "self_attn.q_proj"


@dataclass(frozen=True)
class _NVFP4Weight:
    """The checkpoint's own packed NVFP4 weight, reused bit-exact."""

    packed: np.ndarray  # [out_features, in_features // 2] uint8
    block_scale_f8: np.ndarray  # [out_features, in_features // 16] raw fp8_e4m3 bytes
    global_scale: float
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _LayerScales:
    """Activation scale and the packed weight of one quantized projection."""

    input_scale: float
    weight: _NVFP4Weight


def _scale_const(network, graph_ops, value: float):
    """One-element fp32 constant holding a Q/DQ scale."""
    return graph_ops.add_constant(
        network, (1,), np.asarray([value], dtype=np.float32), dtype=np.float32
    )


def _wrap_nvfp4_matmul(network, activation, scales: _LayerScales, *, graph_ops, keep_alive: list):
    """Add ``activation @ weight.T`` with packed FP4 weights and block-scaled FP4 activations."""
    output_dtype = activation.dtype
    w = scales.weight
    n, k = w.out_features, w.in_features

    # trt.Weights holds a raw pointer, so the buffers must outlive the build.
    keep_alive.append(w.packed)
    keep_alive.append(w.block_scale_f8)

    w_fp4 = trt.Weights(trt.DataType.FP4, w.packed.ctypes.data, n * k)
    w_const = network.add_constant((n, k), w_fp4).get_output(0)
    w_fp8_bs = trt.Weights(trt.DataType.FP8, w.block_scale_f8.ctypes.data, w.block_scale_f8.size)
    w_bs_const = network.add_constant((n, k // NVFP4_BLOCK_SIZE), w_fp8_bs).get_output(0)
    w_gs = _scale_const(network, graph_ops, w.global_scale)
    w_effective_scale = network.add_dequantize(w_bs_const, w_gs, output_dtype)
    w_dequant = network.add_dequantize(w_const, w_effective_scale.get_output(0), output_dtype)
    w_dequant.axis = 1

    act_gs = _scale_const(network, graph_ops, scales.input_scale)
    a_dynq = network.add_dynamic_quantize(
        activation, 1, NVFP4_BLOCK_SIZE, trt.DataType.FP4, trt.DataType.FP8
    )
    a_dynq.set_input(1, act_gs)
    a_effective_scale = network.add_dequantize(a_dynq.get_output(1), act_gs, output_dtype)
    a_dequant = network.add_dequantize(
        a_dynq.get_output(0), a_effective_scale.get_output(0), output_dtype
    )
    a_dequant.axis = 1

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
class Qwen35QuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    graph_ops: Any
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        """Create the list that keeps weight buffers alive during the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

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
        return _wrap_nvfp4_matmul(
            network, lhs, entry, graph_ops=self.graph_ops, keep_alive=self.keep_alive
        )


def _open_raw_readers(model_dir: Path) -> dict[str, Any]:
    """Map tensor name -> torch-framework reader (raw dtypes, no conversion)."""
    single = model_dir / "model.safetensors"
    if single.is_file():
        reader = safe_open(str(single), framework="pt")
        return {name: reader for name in reader.keys()}
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"No model.safetensors checkpoint in {model_dir}")
    weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model.safetensors.index.json has no weight_map")
    by_file = {
        shard: safe_open(str(model_dir / shard), framework="pt")
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
    tensor = tensors[key].get_tensor(key)
    return np.ascontiguousarray(tensor.view(torch.uint8).numpy())


def is_nvfp4_checkpoint(model_dir: Path) -> bool:
    """True when the checkpoint stores NVFP4 projections (it has ``weight_scale_2`` tensors)."""
    try:
        tensors = _open_raw_readers(Path(model_dir))
    except FileNotFoundError:
        return False
    return any(name.endswith(".weight_scale_2") for name in tensors)


def _read_nvfp4(tensors, prefix: str) -> _NVFP4Weight | None:
    """The NVFP4 weight under ``prefix``, or None when its tensors are not in the checkpoint."""
    keys = [f"{prefix}.{s}" for s in ("weight", "weight_scale", "weight_scale_2")]
    if not all(key in tensors for key in keys):
        return None
    packed = _raw_bytes(tensors, keys[0])
    block_scale = _raw_bytes(tensors, keys[1])
    out_features, packed_in = packed.shape
    in_features = packed_in * 2
    if block_scale.shape != (out_features, in_features // NVFP4_BLOCK_SIZE):
        raise ValueError(
            f"{keys[1]} has shape {block_scale.shape}, expected "
            f"({out_features}, {in_features // NVFP4_BLOCK_SIZE})"
        )
    global_scale = _scalar(tensors, keys[2])
    if global_scale is None:
        raise ValueError(f"{keys[2]} must be one positive finite value")
    return _NVFP4Weight(packed, block_scale, global_scale, out_features, in_features)


def _split_query_gate(
    weight: _NVFP4Weight, num_heads: int, head_dim: int
) -> tuple[_NVFP4Weight, _NVFP4Weight]:
    """Split the rows of a ``q_proj`` into the query half and the output-gate half.

    Each head owns ``2 * head_dim`` consecutive rows: ``head_dim`` query rows
    followed by ``head_dim`` gate rows.
    """
    if weight.out_features != num_heads * 2 * head_dim:
        raise ValueError(
            f"q_proj has {weight.out_features} rows, expected {num_heads * 2 * head_dim}"
        )

    def rows(array: np.ndarray, part: slice) -> np.ndarray:
        per_head = array.reshape(num_heads, 2 * head_dim, array.shape[1])
        return np.ascontiguousarray(per_head[:, part, :].reshape(num_heads * head_dim, -1))

    halves = []
    for part in (slice(0, head_dim), slice(head_dim, 2 * head_dim)):
        halves.append(
            _NVFP4Weight(
                rows(weight.packed, part),
                rows(weight.block_scale_f8, part),
                weight.global_scale,
                num_heads * head_dim,
                weight.in_features,
            )
        )
    return halves[0], halves[1]


def calibrate_qwen35(
    model_dir: Path,
    graph_ops,
    quantization: str,
    *,
    layer_types: list[str],
    num_heads: int,
    head_dim: int,
) -> Qwen35QuantContext:
    """Build the Q/DQ context from a ModelOpt NVFP4 Qwen3.5 checkpoint.

    ``layer_types`` lists ``"deltanet"`` or ``"attention"`` per layer. The checkpoint
    already carries calibrated weights and activation scales, so no forward pass is
    needed. Raises when the checkpoint does not match the requested format, instead
    of silently falling back to unquantized weights.
    """
    if quantization not in _SUPPORTED:
        raise NotImplementedError(f"qwen3_5 does not support quantization={quantization!r}")
    tensors = _open_raw_readers(Path(model_dir))
    layers = (
        "model.language_model.layers"
        if "model.language_model.layers.0.mlp.gate_proj.weight" in tensors
        else "model.layers"
    )

    def read(prefix: str) -> tuple[_NVFP4Weight, float]:
        scale_key = f"{prefix}.input_scale"
        weight = _read_nvfp4(tensors, prefix)
        if weight is None or scale_key not in tensors:
            raise ValueError(
                f"{prefix} is not an nvfp4 ModelOpt tensor; qwen3_5 nvfp4 builds need a "
                "ModelOpt NVFP4 checkpoint such as AxionML/Qwen3.5-9B-NVFP4"
            )
        input_scale = _scalar(tensors, scale_key)
        if input_scale is None:
            raise ValueError(f"{scale_key} must be one positive finite value")
        return weight, input_scale

    scales: dict[str, _LayerScales] = {}
    for layer, kind in enumerate(layer_types):
        mixer = _DELTANET_PROJECTIONS if kind == "deltanet" else _ATTENTION_PROJECTIONS
        for hf_stem, weight_stem in (*mixer, *_MLP_PROJECTIONS):
            weight, input_scale = read(f"{layers}.{layer}.{hf_stem}")
            scales[f"layer.{layer}.{weight_stem}"] = _LayerScales(input_scale, weight)
        if kind == "attention":
            weight, input_scale = read(f"{layers}.{layer}.{_Q_PROJECTION}")
            query, gate = _split_query_gate(weight, num_heads, head_dim)
            scales[f"layer.{layer}.w_q"] = _LayerScales(input_scale, query)
            scales[f"layer.{layer}.w_gate_attn"] = _LayerScales(input_scale, gate)
    return Qwen35QuantContext(scales=scales, graph_ops=graph_ops)
