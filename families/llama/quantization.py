# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Llama-owned FP8 / NVFP4 TensorRT Q/DQ graph context for ModelOpt checkpoints.

Supported checkpoints are ModelOpt exports such as
``nvidia/Llama-3.1-8B-Instruct-FP8`` and ``nvidia/Llama-3.1-8B-Instruct-NVFP4``.
Every attention (q/k/v/o) and MLP (gate/up/down) projection carries:

  FP8    ``<name>.weight`` (float8_e4m3), ``<name>.weight_scale`` (per-tensor
         fp32) and ``<name>.input_scale`` (per-tensor fp32).
  NVFP4  ``<name>.weight`` (E2M1, two values per uint8), ``<name>.weight_scale``
         (per-16-element float8_e4m3), ``<name>.weight_scale_2`` (global fp32)
         and ``<name>.input_scale`` (fp32).

``lm_head`` and the embedding stay unquantized.

The checkpoint's own quantized bytes are fed to the engine as-is (no
dequantize/re-quantize round trip). TensorRT's ``add_dequantize(value, scale)``
computes ``value * scale``, which reproduces the exported weight exactly, and
the Quantize/DynamicQuantize -> Dequantize -> MatMul pattern is fused into a
native FP8 or FP4 tensor-core GEMM. Weights keep the checkpoint's
``[out_features, in_features]`` layout, so the matmul transposes the weight
operand instead of repacking sub-byte data.

Activations use ``add_quantize`` for FP8 and ``add_dynamic_quantize`` for NVFP4
(blockwise ``add_quantize`` to an FP4 output is rejected by the shape checker).
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

# HF projection stem -> internal weight-dict stem used by graph_blocks.
_PROJECTIONS = (
    ("self_attn.q_proj", "w_q"),
    ("self_attn.k_proj", "w_k"),
    ("self_attn.v_proj", "w_v"),
    ("self_attn.o_proj", "w_o"),
    ("mlp.gate_proj", "w_gate"),
    ("mlp.up_proj", "w_up"),
    ("mlp.down_proj", "w_down"),
)
_SUPPORTED = ("fp8", "nvfp4")


@dataclass(frozen=True)
class _NVFP4Weight:
    """The checkpoint's own packed NVFP4 weight, reused bit-exact."""

    packed: np.ndarray          # [out_features, in_features // 2] uint8
    block_scale_f8: np.ndarray  # [out_features, in_features // 16] raw fp8_e4m3 bytes
    global_scale: float
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _FP8Weight:
    """The checkpoint's own FP8 weight, reused bit-exact."""

    packed: np.ndarray  # [out_features, in_features] raw fp8_e4m3 bytes
    weight_scale: float
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _LayerScales:
    input_scale: float
    weight: _NVFP4Weight | _FP8Weight


def _scale_const(network, graph_ops, value: float):
    return graph_ops.add_constant(
        network, (1,), np.asarray([value], dtype=np.float32), dtype=np.float32
    )


def _finish_matmul(network, activation, a_dequant, w_dequant):
    output = network.add_matrix_multiply(
        a_dequant,
        trt.MatrixOperation.NONE,
        w_dequant,
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    if output.dtype == activation.dtype:
        return output
    return network.add_cast(output, activation.dtype).get_output(0)


class _NVFP4Format:
    @staticmethod
    def wrap_matmul(network, activation, scales: _LayerScales, *, graph_ops, keep_alive: list):
        output_dtype = activation.dtype
        w = scales.weight
        n, k = w.out_features, w.in_features

        # trt.Weights holds a raw pointer, so the buffers must outlive the build.
        keep_alive.append(w.packed)
        keep_alive.append(w.block_scale_f8)

        w_fp4 = trt.Weights(trt.DataType.FP4, w.packed.ctypes.data, n * k)
        w_const = network.add_constant((n, k), w_fp4).get_output(0)
        w_fp8_bs = trt.Weights(
            trt.DataType.FP8, w.block_scale_f8.ctypes.data, w.block_scale_f8.size
        )
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

        return _finish_matmul(
            network, activation, a_dequant.get_output(0), w_dequant.get_output(0)
        )


class _FP8Format:
    @staticmethod
    def wrap_matmul(network, activation, scales: _LayerScales, *, graph_ops, keep_alive: list):
        output_dtype = activation.dtype
        w = scales.weight
        n, k = w.out_features, w.in_features

        keep_alive.append(w.packed)

        w_fp8 = trt.Weights(trt.DataType.FP8, w.packed.ctypes.data, n * k)
        w_const = network.add_constant((n, k), w_fp8).get_output(0)
        w_dequant = network.add_dequantize(
            w_const, _scale_const(network, graph_ops, w.weight_scale), output_dtype
        )

        act_scale = _scale_const(network, graph_ops, scales.input_scale)
        a_quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
        a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, output_dtype)

        return _finish_matmul(
            network, activation, a_dequant.get_output(0), w_dequant.get_output(0)
        )


@dataclass(frozen=True)
class LlamaQuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    graph_ops: Any
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, weight_name: str) -> bool:
        return weight_name in self.scales

    @property
    def disable_dual_gemm_fusion(self) -> bool:
        """True for FP8: TensorRT's fused gate/up (dual-GEMM) kernel fails to compile
        with scalar-scale FP8 Q/DQ inputs ('arith.divf' type mismatch), so the builders
        turn that fusion off. NVFP4 fuses correctly and keeps it."""
        return any(isinstance(s.weight, _FP8Weight) for s in self.scales.values())

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
        entry = self.scales.get(weight_name)
        if entry is None:
            return self.graph_ops.add_matmul_rhs_constant(
                network, lhs, lhs_width, rhs_width, rhs_weights, dtype=dtype
            )
        fmt = _NVFP4Format if isinstance(entry.weight, _NVFP4Weight) else _FP8Format
        return fmt.wrap_matmul(
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
    reader = tensors.get(key)
    if reader is None:
        return None
    value = reader.get_tensor(key).float().numpy().reshape(-1)
    if value.size != 1 or not np.isfinite(value[0]) or value[0] <= 0:
        return None
    return float(value[0])


def _raw_bytes(tensors: dict[str, Any], key: str) -> np.ndarray:
    tensor = tensors[key].get_tensor(key)
    return np.ascontiguousarray(tensor.view(torch.uint8).numpy())


def _read_nvfp4(tensors, prefix: str) -> _NVFP4Weight | None:
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


def _read_fp8(tensors, prefix: str) -> _FP8Weight | None:
    weight_key, scale_key = f"{prefix}.weight", f"{prefix}.weight_scale"
    if weight_key not in tensors or scale_key not in tensors:
        return None
    if tensors[weight_key].get_tensor(weight_key).dtype != torch.float8_e4m3fn:
        return None
    packed = _raw_bytes(tensors, weight_key)
    weight_scale = _scalar(tensors, scale_key)
    if weight_scale is None:
        raise ValueError(f"{scale_key} must be one positive finite value")
    out_features, in_features = packed.shape
    return _FP8Weight(packed, weight_scale, out_features, in_features)


def calibrate_llama(
    model_dir: Path, config, graph_ops, quantization: str,
) -> LlamaQuantContext:
    """Build the Q/DQ context from a ModelOpt FP8 or NVFP4 Llama checkpoint.

    The checkpoint already carries calibrated weights and activation scales, so
    no forward pass is needed. Raises when the checkpoint does not match the
    requested format, instead of silently falling back to unquantized weights.
    """
    if quantization not in _SUPPORTED:
        raise NotImplementedError(f"llama does not support quantization={quantization!r}")
    tensors = _open_raw_readers(Path(model_dir))
    read_weight = _read_nvfp4 if quantization == "nvfp4" else _read_fp8

    scales: dict[str, _LayerScales] = {}
    for layer in range(int(config.num_hidden_layers)):
        for hf_stem, weight_stem in _PROJECTIONS:
            prefix = f"model.layers.{layer}.{hf_stem}"
            input_scale = _scalar(tensors, f"{prefix}.input_scale")
            weight = read_weight(tensors, prefix)
            if input_scale is None or weight is None:
                raise ValueError(
                    f"{prefix} is not a {quantization} ModelOpt tensor; "
                    f"llama {quantization} builds need a checkpoint such as "
                    f"nvidia/Llama-3.1-8B-Instruct-{quantization.upper()}"
                )
            scales[f"layer.{layer}.{weight_stem}"] = _LayerScales(input_scale, weight)
    return LlamaQuantContext(scales=scales, graph_ops=graph_ops)
