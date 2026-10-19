# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-MoE-owned quantization context for ModelOpt NVFP4 checkpoints.

Supported checkpoints are ModelOpt NVFP4 exports such as
``nvidia/Qwen3-30B-A3B-NVFP4``. The attention projections (q/k/v/o) and every
routed expert projection (gate/up/down) carry:

  ``<name>.weight``          E2M1, two values per uint8
  ``<name>.weight_scale``    per-16-element float8_e4m3
  ``<name>.weight_scale_2``  global fp32
  ``<name>.input_scale``     fp32

The router (``mlp.gate``), the embedding and ``lm_head`` stay unquantized.

Attention (W4A4)
    The checkpoint's own packed bytes are fed to the engine as-is. Weights keep
    the checkpoint's ``[out_features, in_features]`` layout and the matmul
    transposes the weight operand; activations use dynamic NVFP4 quantization.

Experts (W4A16)
    The native TensorRT MoE layer in TensorRT 11.1 rejects block-scaled 4-bit
    weights and is slow for a single token, so the experts are built from plain
    layers instead: the router's top-k indices gather the selected experts'
    packed FP4 weights (and their block scales), which are dequantized and fed
    to batched fp16 matmuls. Weights keep the checkpoint's ``[out, in]`` layout
    and bytes. The block scale and the global scale are folded into one fp16
    scale per 16 weights. Activations stay fp16, so the checkpoint's expert
    ``input_scale`` is not used.

    TensorRT folds a dequantize of constants into a full fp16 constant, which
    would store the experts at 16 bits. To keep the 4-bit weights, the scale is
    multiplied by a value that is 1 at run time but is computed from the router
    indices, so it cannot be folded.
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
_E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])

# HF attention projection stem -> internal weight-dict stem.
_ATTENTION = (
    ("self_attn.q_proj", "w_q"),
    ("self_attn.k_proj", "w_k"),
    ("self_attn.v_proj", "w_v"),
    ("self_attn.o_proj", "w_o"),
)
# HF expert projection stem -> internal expert stem.
_EXPERTS = (("gate_proj", "w_gate"), ("up_proj", "w_up"), ("down_proj", "w_down"))


@dataclass(frozen=True)
class _NVFP4Weight:
    """Packed NVFP4 weight with its scales, reused bit-exact."""

    packed: np.ndarray  # [out, in // 2] uint8
    block_scale_f8: np.ndarray  # [out, in // 16] raw fp8_e4m3 bytes
    global_scale: float
    input_scale: float

    @property
    def out_features(self) -> int:
        return self.packed.shape[0]

    @property
    def in_features(self) -> int:
        return self.packed.shape[1] * 2


@dataclass(frozen=True)
class _ExpertLayer:
    """The routed experts of one layer as packed FP4, in the checkpoint layout."""

    gate: np.ndarray  # [experts, inter, hidden // 2] uint8
    up: np.ndarray  # [experts, inter, hidden // 2]
    down: np.ndarray  # [experts, hidden, inter // 2]
    gate_scale: np.ndarray  # [experts, inter, hidden // 16] fp16 (block * global scale)
    up_scale: np.ndarray
    down_scale: np.ndarray  # [experts, hidden, inter // 16]


def _scale_const(network, graph_ops, values) -> trt.ITensor:
    values = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    return graph_ops.add_constant(network, values.shape, values, dtype=np.float32)


def _nvfp4_matmul(network, activation, weight: _NVFP4Weight, *, graph_ops, keep_alive: list):
    output_dtype = activation.dtype
    n, k = weight.out_features, weight.in_features

    # trt.Weights holds a raw pointer, so the buffers must outlive the build.
    keep_alive.append(weight.packed)
    keep_alive.append(weight.block_scale_f8)

    w_fp4 = trt.Weights(trt.DataType.FP4, weight.packed.ctypes.data, n * k)
    w_const = network.add_constant((n, k), w_fp4).get_output(0)
    w_fp8_bs = trt.Weights(
        trt.DataType.FP8, weight.block_scale_f8.ctypes.data, weight.block_scale_f8.size
    )
    w_bs_const = network.add_constant((n, k // NVFP4_BLOCK_SIZE), w_fp8_bs).get_output(0)
    w_gs = _scale_const(network, graph_ops, [weight.global_scale])
    w_effective_scale = network.add_dequantize(w_bs_const, w_gs, output_dtype)
    w_dequant = network.add_dequantize(w_const, w_effective_scale.get_output(0), output_dtype)
    w_dequant.axis = 1

    act_gs = _scale_const(network, graph_ops, [weight.input_scale])
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
class QwenMoeQuantContext:
    """Routes quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _NVFP4Weight]  # attention: "layer.N.w_q", ...
    experts: dict[str, _ExpertLayer]  # "layer.N"
    graph_ops: Any
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, weight_name: str) -> bool:
        if ".experts." in weight_name:
            return weight_name.split(".experts.")[0] in self.experts
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
        entry = self.scales.get(weight_name)
        if entry is None:
            return self.graph_ops.add_matmul_rhs_constant(
                network, lhs, lhs_width, rhs_width, rhs_weights, dtype=dtype
            )
        return _nvfp4_matmul(
            network, lhs, entry, graph_ops=self.graph_ops, keep_alive=self.keep_alive
        )

    def gathered_moe(
        self, network, hidden, top_indices, top_scores, layer_name: str, hidden_size: int
    ) -> trt.ITensor:
        """Routed-expert output ``[1, hidden]`` from the gathered FP4 experts.

        ``hidden`` is ``[1, hidden]``; ``top_indices`` and ``top_scores`` are
        ``[1, top_k]`` (scores already renormalized).
        """
        layer = self.experts[layer_name]
        dtype = hidden.dtype
        top_k = top_indices.shape[-1]
        elementwise = trt.ElementWiseOperation

        def reshape(tensor, dims):
            shuffle = network.add_shuffle(tensor)
            shuffle.reshape_dims = dims
            return shuffle.get_output(0)

        def constant(array: np.ndarray, weights=None) -> trt.ITensor:
            self.keep_alive.append(array)
            weights = weights or trt.Weights(array)
            return network.add_constant(array.shape, weights).get_output(0)

        indices = reshape(top_indices, (top_k,))
        # 1.0 that depends on the router output, so the dequantize below is not folded.
        zero = network.add_elementwise(
            network.add_cast(top_indices, trt.DataType.FLOAT).get_output(0),
            constant(np.zeros((1, 1), dtype=np.float32)),
            elementwise.PROD,
        ).get_output(0)
        zero = network.add_reduce(zero, trt.ReduceOperation.SUM, 0b11, True).get_output(0)
        one = network.add_elementwise(
            zero, constant(np.ones((1, 1), dtype=np.float32)), elementwise.SUM
        ).get_output(0)
        one = network.add_cast(reshape(one, (1, 1, 1)), dtype).get_output(0)

        # [top_k, 1, hidden]: the same row once per selected expert.
        rows = network.add_elementwise(
            reshape(hidden, (1, 1, hidden_size)),
            constant(np.zeros((top_k, 1, hidden_size), dtype=np.float16)),
            elementwise.SUM,
        ).get_output(0)

        def project(activation, packed: np.ndarray, scale: np.ndarray) -> trt.ITensor:
            experts, out_features, packed_in = packed.shape
            in_features = packed_in * 2
            weights = trt.Weights(
                trt.DataType.FP4, packed.ctypes.data, experts * out_features * in_features
            )
            self.keep_alive.append(packed)
            w_const = network.add_constant(
                (experts, out_features, in_features), weights
            ).get_output(0)
            block_scale = network.add_elementwise(constant(scale), one, elementwise.PROD)
            dequant = network.add_dequantize(w_const, block_scale.get_output(0), dtype)
            dequant.axis = 2
            selected = network.add_gather(dequant.get_output(0), indices, 0).get_output(0)
            return network.add_matrix_multiply(
                activation, trt.MatrixOperation.NONE, selected, trt.MatrixOperation.TRANSPOSE
            ).get_output(0)

        gate = project(rows, layer.gate, layer.gate_scale)
        up = project(rows, layer.up, layer.up_scale)
        silu = network.add_elementwise(
            gate,
            network.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0),
            elementwise.PROD,
        ).get_output(0)
        inner = network.add_elementwise(silu, up, elementwise.PROD).get_output(0)
        down = project(inner, layer.down, layer.down_scale)  # [top_k, 1, hidden]

        scores = reshape(network.add_cast(top_scores, dtype).get_output(0), (top_k, 1, 1))
        weighted = network.add_elementwise(down, scores, elementwise.PROD).get_output(0)
        output = network.add_reduce(weighted, trt.ReduceOperation.SUM, 0b001, False).get_output(0)
        return reshape(output, (1, hidden_size))


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
    keys = [f"{prefix}.{s}" for s in ("weight", "weight_scale", "weight_scale_2", "input_scale")]
    if not all(key in tensors for key in keys):
        return None
    packed = _raw_bytes(tensors, keys[0])
    block_scale = _raw_bytes(tensors, keys[1])
    out_features, packed_in = packed.shape
    if block_scale.shape != (out_features, packed_in * 2 // NVFP4_BLOCK_SIZE):
        raise ValueError(
            f"{keys[1]} has shape {block_scale.shape}, expected "
            f"({out_features}, {packed_in * 2 // NVFP4_BLOCK_SIZE})"
        )
    global_scale, input_scale = _scalar(tensors, keys[2]), _scalar(tensors, keys[3])
    if global_scale is None or input_scale is None:
        raise ValueError(f"{prefix} scales must each be one positive finite value")
    return _NVFP4Weight(packed, block_scale, global_scale, input_scale)


def dequantize_nvfp4(weight: _NVFP4Weight) -> torch.Tensor:
    """The exact float32 values the NVFP4 weight stands for, ``[out, in]``."""
    packed = torch.from_numpy(weight.packed)
    low, high = (packed & 0xF).long(), (packed >> 4).long()
    values = torch.stack([_E2M1[low], _E2M1[high]], dim=-1).reshape(packed.shape[0], -1)
    block_scale = torch.from_numpy(weight.block_scale_f8).view(torch.float8_e4m3fn).float()
    block_scale = block_scale.repeat_interleave(NVFP4_BLOCK_SIZE, dim=1)
    return values * block_scale * weight.global_scale


def _read_experts(tensors, layer_prefix: str, num_experts: int) -> _ExpertLayer:
    """Stack the FP4 experts of one layer: packed bytes and one fp16 scale per 16 weights."""
    packed: dict[str, list[np.ndarray]] = {key: [] for _, key in _EXPERTS}
    scales: dict[str, list[np.ndarray]] = {key: [] for _, key in _EXPERTS}
    for expert in range(num_experts):
        for hf_stem, stem in _EXPERTS:
            name = f"{layer_prefix}.mlp.experts.{expert}.{hf_stem}"
            entry = _read_nvfp4(tensors, name)
            if entry is None:
                raise ValueError(f"{name} is not an nvfp4 ModelOpt tensor")
            block = torch.from_numpy(entry.block_scale_f8).view(torch.float8_e4m3fn).float()
            packed[stem].append(entry.packed)
            scales[stem].append((block * entry.global_scale).half().numpy())
    stack = lambda group, key: np.ascontiguousarray(np.stack(group[key]))  # noqa: E731
    return _ExpertLayer(
        gate=stack(packed, "w_gate"),
        up=stack(packed, "w_up"),
        down=stack(packed, "w_down"),
        gate_scale=stack(scales, "w_gate"),
        up_scale=stack(scales, "w_up"),
        down_scale=stack(scales, "w_down"),
    )


def calibrate_qwen_moe(
    model_dir: Path, config, graph_ops, quantization: str
) -> QwenMoeQuantContext:
    """Build the quantization context from a ModelOpt NVFP4 Qwen-MoE checkpoint.

    The checkpoint already carries calibrated weights and activation scales, so
    no forward pass is needed. Raises when the checkpoint does not match the
    requested format, instead of silently falling back to unquantized weights.
    """
    if quantization != "nvfp4":
        raise NotImplementedError(f"qwen_moe does not support quantization={quantization!r}")
    raw = config.raw
    if raw.get("mlp_only_layers"):
        raise NotImplementedError("quantized Qwen-MoE builds do not support dense MLP layers")
    num_experts = int(raw.get("num_experts", 128))
    tensors = _open_raw_readers(Path(model_dir))
    if "model.layers.0.mlp.shared_expert.gate_proj.weight" in tensors:
        raise NotImplementedError("quantized Qwen-MoE builds do not support a shared expert")

    scales: dict[str, _NVFP4Weight] = {}
    experts: dict[str, _ExpertLayer] = {}
    for layer in range(int(config.num_hidden_layers)):
        layer_prefix = f"model.layers.{layer}"
        for hf_stem, weight_stem in _ATTENTION:
            entry = _read_nvfp4(tensors, f"{layer_prefix}.{hf_stem}")
            if entry is None:
                raise ValueError(
                    f"{layer_prefix}.{hf_stem} is not an nvfp4 ModelOpt tensor; "
                    "qwen_moe nvfp4 builds need a checkpoint such as "
                    "nvidia/Qwen3-30B-A3B-NVFP4"
                )
            scales[f"layer.{layer}.{weight_stem}"] = entry
        experts[f"layer.{layer}"] = _read_experts(tensors, layer_prefix, num_experts)
    return QwenMoeQuantContext(scales=scales, experts=experts, graph_ops=graph_ops)
