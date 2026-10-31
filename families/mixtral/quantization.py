# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixtral-owned quantization context for ModelOpt FP8 checkpoints.

Supported checkpoints are ModelOpt FP8 exports such as
``nvidia/Mixtral-8x7B-Instruct-v0.1-FP8``. The attention projections (q/k/v/o)
and every routed expert projection (``w1`` gate, ``w3`` up, ``w2`` down) carry:

  ``<name>.weight``        float8_e4m3, ``[out_features, in_features]``
  ``<name>.weight_scale``  per-tensor fp32
  ``<name>.input_scale``   per-tensor fp32

The router (``block_sparse_moe.gate``), the embedding and ``lm_head`` stay
unquantized.

Attention (W8A8)
    The q/k/v projections feed the checkpoint's own FP8 bytes to the engine
    as-is. Weights keep the checkpoint's ``[out_features, in_features]``
    layout and the matmul transposes the weight operand; the activation goes
    through Quantize and Dequantize with the checkpoint's ``input_scale``.
    The output projection (``o_proj``) stays a fp16 matmul with the
    dequantized weight: TensorRT 11.4 cannot compile an FP8 Quantize right
    after fused attention.

Experts (W8A8)
    Every expert is a separate set of three projections with the same FP8 Q/DQ
    pattern as attention and the expert's own scales. Like the unquantized
    path, all experts run and the ones the router did not select get a routing
    weight of 0. Gate and up of an expert read one activation and share its
    ``input_scale``.
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

# HF attention projection stem -> internal weight-dict stem.
_ATTENTION = (
    ("self_attn.q_proj", "w_q"),
    ("self_attn.k_proj", "w_k"),
    ("self_attn.v_proj", "w_v"),
    ("self_attn.o_proj", "w_o"),
)
# HF expert projection stem -> internal expert stem.
_EXPERTS = (("w1", "w_gate"), ("w3", "w_up"), ("w2", "w_down"))


@dataclass(frozen=True)
class _FP8Weight:
    """The checkpoint's own FP8 weight, reused bit-exact."""

    packed: np.ndarray  # [out, in] raw fp8_e4m3 bytes (uint8)
    weight_scale: float
    input_scale: float

    @property
    def out_features(self) -> int:
        return self.packed.shape[0]

    @property
    def in_features(self) -> int:
        return self.packed.shape[1]


@dataclass(frozen=True)
class _ExpertLayer:
    """The routed experts of one layer, one FP8 projection per expert."""

    gate: tuple[_FP8Weight, ...]
    up: tuple[_FP8Weight, ...]
    down: tuple[_FP8Weight, ...]


def _scale_const(network, graph_ops, value: float) -> trt.ITensor:
    return graph_ops.add_constant(
        network, (1,), np.asarray([value], dtype=np.float32), dtype=np.float32
    )


def _fp8_activation(
    network, activation, input_scale: float, *, graph_ops, dtype=None
) -> trt.ITensor:
    """Quantize then dequantize an activation with the checkpoint's ``input_scale``."""
    scale = _scale_const(network, graph_ops, input_scale)
    quantized = network.add_quantize(activation, scale, trt.DataType.FP8)
    return network.add_dequantize(
        quantized.get_output(0), scale, dtype or activation.dtype
    ).get_output(0)


def _fp8_weight_matmul(
    network, dequantized_activation, weight: _FP8Weight, *, graph_ops, keep_alive
):
    """``activation @ weight.T`` with the checkpoint's FP8 bytes behind a Dequantize."""
    output_dtype = dequantized_activation.dtype
    n, k = weight.out_features, weight.in_features

    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(weight.packed)
    w_fp8 = trt.Weights(trt.DataType.FP8, weight.packed.ctypes.data, n * k)
    w_const = network.add_constant((n, k), w_fp8).get_output(0)
    w_dequant = network.add_dequantize(
        w_const, _scale_const(network, graph_ops, weight.weight_scale), output_dtype
    )
    output = network.add_matrix_multiply(
        dequantized_activation,
        trt.MatrixOperation.NONE,
        w_dequant.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    if output.dtype == output_dtype:
        return output
    return network.add_cast(output, output_dtype).get_output(0)


def _fp8_matmul(network, activation, weight: _FP8Weight, *, graph_ops, keep_alive: list):
    activation = _fp8_activation(network, activation, weight.input_scale, graph_ops=graph_ops)
    return _fp8_weight_matmul(
        network, activation, weight, graph_ops=graph_ops, keep_alive=keep_alive
    )


@dataclass(frozen=True)
class MixtralQuantContext:
    """Routes quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _FP8Weight]  # attention: "layer.N.w_q", ...
    # FP8 projections that run as fp16 matmuls with the dequantized weight ("layer.N.w_o").
    dense: dict[str, _FP8Weight]
    experts: dict[str, _ExpertLayer]  # "layer.N"
    graph_ops: Any
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def dense_weight(self, weight_name: str) -> np.ndarray:
        """The dequantized weight of a ``dense`` projection as fp16 ``[in, out]``."""
        return np.ascontiguousarray(
            dequantize_fp8(self.dense[weight_name]).T.numpy().astype(np.float16)
        )

    @property
    def has_fp8_weights(self) -> bool:
        return True

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
        return _fp8_matmul(
            network, lhs, entry, graph_ops=self.graph_ops, keep_alive=self.keep_alive
        )

    def routed_experts(
        self, network, hidden, top_indices, top_scores, layer_name: str, hidden_size: int
    ) -> trt.ITensor:
        """Routed-expert output ``[1, hidden]`` from the FP8 experts.

        ``hidden`` is ``[1, hidden]``; ``top_indices`` (int32) and ``top_scores``
        (fp32, already renormalized) are ``[1, top_k]``. Every expert runs and
        an expert that was not selected gets a routing weight of 0, like the
        unquantized path.
        """
        layer = self.experts[layer_name]
        dtype = hidden.dtype
        elementwise = trt.ElementWiseOperation
        num_experts = len(layer.gate)

        def constant(array: np.ndarray) -> trt.ITensor:
            self.keep_alive.append(array)
            return network.add_constant(array.shape, trt.Weights(array)).get_output(0)

        # [experts, top_k]: 1 where expert e is the k-th selected expert.
        matches = network.add_cast(
            network.add_elementwise(
                constant(np.arange(num_experts, dtype=np.int32).reshape(num_experts, 1)),
                top_indices,
                elementwise.EQUAL,
            ).get_output(0),
            trt.DataType.FLOAT,
        ).get_output(0)
        # [experts, 1]: the routing weight of every expert (0 when not selected).
        weights = network.add_reduce(
            network.add_elementwise(matches, top_scores, elementwise.PROD).get_output(0),
            trt.ReduceOperation.SUM,
            0b10,
            True,
        ).get_output(0)

        routed = None
        for expert in range(num_experts):
            gate_w, up_w, down_w = layer.gate[expert], layer.up[expert], layer.down[expert]
            # Gate and up read the same activation with one input scale.
            shared = _fp8_activation(network, hidden, gate_w.input_scale, graph_ops=self.graph_ops)
            gate = _fp8_weight_matmul(
                network, shared, gate_w, graph_ops=self.graph_ops, keep_alive=self.keep_alive
            )
            up = _fp8_weight_matmul(
                network, shared, up_w, graph_ops=self.graph_ops, keep_alive=self.keep_alive
            )
            silu = network.add_elementwise(
                gate,
                network.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0),
                elementwise.PROD,
            ).get_output(0)
            inner = network.add_elementwise(silu, up, elementwise.PROD).get_output(0)
            down = _fp8_weight_matmul(
                network,
                _fp8_activation(network, inner, down_w.input_scale, graph_ops=self.graph_ops),
                down_w,
                graph_ops=self.graph_ops,
                keep_alive=self.keep_alive,
            )
            weight = network.add_slice(weights, (expert, 0), (1, 1), (1, 1)).get_output(0)
            scaled = network.add_elementwise(
                network.add_cast(down, trt.DataType.FLOAT).get_output(0),
                weight,
                elementwise.PROD,
            ).get_output(0)
            contribution = network.add_cast(scaled, dtype).get_output(0)
            routed = (
                contribution
                if routed is None
                else network.add_elementwise(routed, contribution, elementwise.SUM).get_output(0)
            )
        return routed


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


def _raw_fp8(tensors: dict[str, Any], key: str) -> np.ndarray:
    tensor = tensors[key].get_tensor(key)
    if tensor.dtype != torch.float8_e4m3fn:
        raise ValueError(f"{key} is {tensor.dtype}, expected float8_e4m3fn")
    return np.ascontiguousarray(tensor.view(torch.uint8).numpy())


def _read_fp8(tensors, prefix: str) -> _FP8Weight | None:
    keys = [f"{prefix}.{s}" for s in ("weight", "weight_scale", "input_scale")]
    if not all(key in tensors for key in keys):
        return None
    weight_scale, input_scale = _scalar(tensors, keys[1]), _scalar(tensors, keys[2])
    if weight_scale is None or input_scale is None:
        raise ValueError(f"{prefix} scales must each be one positive finite value")
    packed = _raw_fp8(tensors, keys[0])
    if packed.ndim != 2:
        raise ValueError(f"{keys[0]} must be a matrix, got shape {packed.shape}")
    return _FP8Weight(packed, weight_scale, input_scale)


def dequantize_fp8(weight: _FP8Weight) -> torch.Tensor:
    """The exact float32 values the FP8 weight stands for, ``[out, in]``."""
    values = torch.from_numpy(weight.packed).view(torch.float8_e4m3fn).float()
    return values * weight.weight_scale


def _read_experts(tensors, layer_prefix: str, num_experts: int) -> _ExpertLayer:
    """Read the FP8 experts of one layer, keeping the checkpoint bytes."""
    projections = {}
    for hf_stem, stem in _EXPERTS:
        entries = []
        for expert in range(num_experts):
            name = f"{layer_prefix}.block_sparse_moe.experts.{expert}.{hf_stem}"
            entry = _read_fp8(tensors, name)
            if entry is None:
                raise ValueError(f"{name} is not an fp8 ModelOpt tensor")
            entries.append(entry)
        projections[stem] = tuple(entries)
    for gate, up in zip(projections["w_gate"], projections["w_up"]):
        if gate.input_scale != up.input_scale:
            raise ValueError(f"{layer_prefix}: gate and up of one expert need one input scale")
    return _ExpertLayer(
        gate=projections["w_gate"], up=projections["w_up"], down=projections["w_down"]
    )


def calibrate_mixtral(model_dir: Path, config, graph_ops, quantization: str) -> MixtralQuantContext:
    """Build the quantization context from a ModelOpt FP8 Mixtral checkpoint.

    The checkpoint already carries calibrated weights and activation scales, so
    no forward pass is needed. Raises when the checkpoint does not match the
    requested format, instead of silently falling back to unquantized weights.
    """
    if quantization != "fp8":
        raise NotImplementedError(f"mixtral does not support quantization={quantization!r}")
    num_experts = int(config.raw.get("num_local_experts", 8))
    tensors = _open_raw_readers(Path(model_dir))

    scales: dict[str, _FP8Weight] = {}
    dense: dict[str, _FP8Weight] = {}
    experts: dict[str, _ExpertLayer] = {}
    for layer in range(int(config.num_hidden_layers)):
        layer_prefix = f"model.layers.{layer}"
        for hf_stem, weight_stem in _ATTENTION:
            entry = _read_fp8(tensors, f"{layer_prefix}.{hf_stem}")
            if entry is None:
                raise ValueError(
                    f"{layer_prefix}.{hf_stem} is not an fp8 ModelOpt tensor; "
                    "mixtral fp8 builds need a checkpoint such as "
                    "nvidia/Mixtral-8x7B-Instruct-v0.1-FP8"
                )
            if weight_stem == "w_o":
                dense[f"layer.{layer}.{weight_stem}"] = entry
            else:
                scales[f"layer.{layer}.{weight_stem}"] = entry
        experts[f"layer.{layer}"] = _read_experts(tensors, layer_prefix, num_experts)
    return MixtralQuantContext(scales=scales, experts=experts, dense=dense, graph_ops=graph_ops)
