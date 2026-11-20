# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS-owned NVFP4 context for the routed experts.

``openai/gpt-oss-20b`` and ``openai/gpt-oss-120b`` ship the routed experts as
MXFP4 and everything else (attention, router, embedding, ``lm_head``, norms,
attention sinks) in bf16. Per layer the checkpoint holds::

    mlp.experts.gate_up_proj_blocks   uint8 [E, 2*I, K/32, 16]  E2M1, first value in the low nibble
    mlp.experts.gate_up_proj_scales   uint8 [E, 2*I, K/32]      e8m0, scale = 2^(x - 127)
    mlp.experts.down_proj_blocks      uint8 [E, K, I/32, 16]
    mlp.experts.down_proj_scales      uint8 [E, K, I/32]
    mlp.experts.*_bias                bf16  [E, 2*I] and [E, K]

Gate and up rows are interleaved in ``gate_up_proj`` (even rows gate, odd rows up).

Repack, bit exact
    TensorRT block-scaled FP4 GEMMs take NVFP4 (E2M1 values, one e4m3 scale per
    16 values, one fp32 global scale per tensor). The checkpoint's E2M1 bytes are
    used as they are. Every power-of-two e8m0 scale of a tensor fits an e4m3
    scale next to one global scale ``2^g`` (the exponents of one gpt-oss tensor
    span at most 17 values; e4m3 holds 18 powers of two). The scale of each
    32-value block is repeated for its two 16-value blocks. The dequantized
    weights are identical to the MXFP4 checkpoint values, no re-quantization.
    A block more than 18 powers of two below the tensor's largest scale would
    not fit and is set to zero; ``read_experts`` counts them.

Experts (W4A4)
    The native TensorRT MoE layer rejects block-scaled 4-bit weights, and a
    router-driven gather cannot select FP4 weights (a constant of type FP4 may
    only feed a Dequantize), so selecting 4 of 32 experts costs a full fp16
    dequantize of every expert. Instead the FP4 GEMMs run for all experts as one
    batched matmul (TensorRT reads each expert's 4-bit weights once, at memory
    speed) and the router's top-k then picks the rows. The expert inputs
    (``norm2`` output and the gated activation) are dynamically quantized to FP4
    with calibrated per-layer global scales from ``nvfp4_activation_scales.json``.

Attention, router, embedding, ``lm_head``, norms and sinks stay fp16.
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
MXFP4_BLOCK_SIZE = 32
NVFP4_MAX = 6.0 * 448.0
ACTIVATION_SCALES_FILENAME = "nvfp4_activation_scales.json"
# e4m3 holds 2^-9 (subnormal) .. 2^8 exactly; the largest tensor scale is placed at 2^8.
_E4M3_MAX_EXPONENT = 8
_E4M3_MIN_EXPONENT = -9
_E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def scale_key(config) -> str:
    """Key of this model's entry in the activation-scale file."""
    return (
        f"{str(config.model_type).lower()}"
        f"-h{int(config.hidden_size)}"
        f"-l{int(config.num_hidden_layers)}"
        f"-i{int(config.intermediate_size)}"
    )


@dataclass(frozen=True)
class _NVFP4Experts:
    """One expert projection of one layer, every expert, as NVFP4 bytes."""

    packed: np.ndarray  # [E, out, in // 2] uint8 (E2M1, first value in the low nibble)
    block_scale: np.ndarray  # [E, out, in // 16] raw float8_e4m3 bytes
    global_scale: float  # 2^g, one per tensor
    input_scale: float  # global scale of the dynamically quantized input

    @property
    def num_experts(self) -> int:
        return self.packed.shape[0]

    @property
    def out_features(self) -> int:
        return self.packed.shape[1]

    @property
    def in_features(self) -> int:
        return self.packed.shape[2] * 2


@dataclass(frozen=True)
class _ExpertLayer:
    """The routed experts of one layer."""

    gate_up: _NVFP4Experts  # out = 2 * intermediate, gate and up interleaved
    down: _NVFP4Experts
    gate_bias: np.ndarray  # [E, intermediate] fp16
    up_bias: np.ndarray
    down_bias: np.ndarray  # [E, hidden]
    flushed_blocks: int  # scales that did not fit the e4m3 window and were set to zero


def repack_mxfp4_to_nvfp4(
    blocks: torch.Tensor, scales: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Repack one MXFP4 expert tensor as NVFP4 without changing a single weight value.

    ``blocks`` is uint8 ``[E, out, in // 32, 16]``, ``scales`` is uint8 ``[E, out, in // 32]``.
    Returns the packed bytes ``[E, out, in // 2]``, the raw e4m3 block scales
    ``[E, out, in // 16]``, the global scale and the number of blocks flushed to zero.
    """
    if blocks.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("MXFP4 blocks and scales must be uint8")
    experts, out_features, groups, half_block = blocks.shape
    if half_block != MXFP4_BLOCK_SIZE // 2 or scales.shape != (experts, out_features, groups):
        raise ValueError(f"unexpected MXFP4 shapes {tuple(blocks.shape)} and {tuple(scales.shape)}")
    packed = blocks.reshape(experts, out_features, groups * half_block).contiguous()

    exponent = scales.int() - 127
    top = int(exponent.max())
    global_exponent = top - _E4M3_MAX_EXPONENT
    relative = exponent - global_exponent  # <= 8
    fits = relative >= _E4M3_MIN_EXPONENT
    # Powers of two inside the e4m3 range convert exactly; the rest become zero.
    block = torch.where(fits, torch.exp2(relative.float()), torch.zeros(()))
    block_scale = block.to(torch.float8_e4m3fn).view(torch.uint8)
    # Each 32-value block becomes two 16-value blocks with the same scale.
    block_scale = block_scale.repeat_interleave(MXFP4_BLOCK_SIZE // NVFP4_BLOCK_SIZE, dim=-1)
    return (
        np.ascontiguousarray(packed.numpy()),
        np.ascontiguousarray(block_scale.numpy()),
        float(2.0**global_exponent),
        int((~fits).sum()),
    )


def dequantize_nvfp4(experts: _NVFP4Experts) -> torch.Tensor:
    """The float32 values the NVFP4 tensor stands for, ``[E, out, in]``."""
    packed = torch.from_numpy(experts.packed)
    low, high = (packed & 0xF).long(), (packed >> 4).long()
    values = torch.stack([_E2M1[low], _E2M1[high]], dim=-1).reshape(*packed.shape[:2], -1)
    block = torch.from_numpy(experts.block_scale).view(torch.float8_e4m3fn).float()
    block = block.repeat_interleave(NVFP4_BLOCK_SIZE, dim=-1)
    return values * block * experts.global_scale


def dequantize_mxfp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Reference MXFP4 dequantization (the checkpoint's own definition), ``[E, out, in]``."""
    low, high = (blocks & 0xF).long(), (blocks >> 4).long()
    values = torch.stack([_E2M1[low], _E2M1[high]], dim=-1)
    values = values.reshape(*blocks.shape[:3], MXFP4_BLOCK_SIZE)
    values = values * torch.exp2(scales.float() - 127).unsqueeze(-1)
    return values.reshape(blocks.shape[0], blocks.shape[1], -1)


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


def _bias_fp16(tensors, key: str) -> np.ndarray:
    return np.ascontiguousarray(tensors[key].get_tensor(key).to(torch.float16).numpy())


def _read_projection(tensors, layer_prefix: str, name: str, input_scale: float) -> tuple:
    blocks_key = f"{layer_prefix}.mlp.experts.{name}_blocks"
    scales_key = f"{layer_prefix}.mlp.experts.{name}_scales"
    if blocks_key not in tensors or scales_key not in tensors:
        raise ValueError(
            f"{blocks_key} is not an MXFP4 tensor; gpt_oss nvfp4 builds need an official "
            "MXFP4 checkpoint such as openai/gpt-oss-20b"
        )
    packed, block_scale, global_scale, flushed = repack_mxfp4_to_nvfp4(
        tensors[blocks_key].get_tensor(blocks_key), tensors[scales_key].get_tensor(scales_key)
    )
    return _NVFP4Experts(packed, block_scale, global_scale, input_scale), flushed


def read_experts(tensors, layer: int, activation_scales: dict[str, float]) -> _ExpertLayer:
    """Read and repack the experts of one layer."""
    layer_prefix = f"model.layers.{layer}"
    gate_up, flushed_gate_up = _read_projection(
        tensors, layer_prefix, "gate_up_proj", activation_scales[f"layer.{layer}.experts.gate_up"]
    )
    down, flushed_down = _read_projection(
        tensors, layer_prefix, "down_proj", activation_scales[f"layer.{layer}.experts.down"]
    )
    gate_up_bias = _bias_fp16(tensors, f"{layer_prefix}.mlp.experts.gate_up_proj_bias")
    return _ExpertLayer(
        gate_up=gate_up,
        down=down,
        gate_bias=np.ascontiguousarray(gate_up_bias[:, ::2]),
        up_bias=np.ascontiguousarray(gate_up_bias[:, 1::2]),
        down_bias=_bias_fp16(tensors, f"{layer_prefix}.mlp.experts.down_proj_bias"),
        flushed_blocks=flushed_gate_up + flushed_down,
    )


def load_activation_scales(config) -> dict[str, float]:
    """Return the committed activation scales for this model shape, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"GPT-OSS NVFP4 build requires {path}, which was not found") from error
    key = scale_key(config)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated NVFP4 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_nvfp4_scales.py for this model shape."
        )
    return table[key]


@dataclass(frozen=True)
class GptOssQuantContext:
    """Routes the routed experts through batched NVFP4 GEMMs; everything else stays fp16."""

    experts: dict[str, _ExpertLayer]  # "layer.N"
    graph_ops: Any
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        """Give each context its own list for buffers that must outlive the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, weight_name: str) -> bool:
        """Return True for the expert weights of a quantized layer."""
        if ".experts" in weight_name:
            return weight_name.split(".experts")[0] in self.experts
        return False

    def _const(self, network, array: np.ndarray, data_type=None) -> trt.ITensor:
        """Add a constant that keeps its host buffer alive until the build ends."""
        array = np.ascontiguousarray(array)
        self.keep_alive.append(array)
        weights = (
            trt.Weights(data_type, array.ctypes.data, array.size)
            if data_type is not None
            else trt.Weights(array)
        )
        return network.add_constant(array.shape, weights).get_output(0)

    def _scalar(self, network, value: float) -> trt.ITensor:
        return self._const(network, np.asarray([value], dtype=np.float32))

    def _quantized_activation(self, network, activation, input_scale: float, axis: int):
        """Dynamic NVFP4 quantize and dequantize of an activation (the block scales stay on-chip)."""
        dtype = activation.dtype
        act_scale = self._scalar(network, input_scale)
        quantize = network.add_dynamic_quantize(
            activation, axis, NVFP4_BLOCK_SIZE, trt.DataType.FP4, trt.DataType.FP8
        )
        quantize.set_input(1, act_scale)
        effective = network.add_dequantize(quantize.get_output(1), act_scale, dtype)
        dequantize = network.add_dequantize(quantize.get_output(0), effective.get_output(0), dtype)
        dequantize.axis = axis
        return dequantize.get_output(0)

    def _batched_matmul(self, network, activation, weights: _NVFP4Experts, dtype):
        """``activation [1, 1, in]`` or ``[E, 1, in]`` times every expert's ``[out, in]`` weight."""
        experts, out_features, in_features = (
            weights.num_experts,
            weights.out_features,
            weights.in_features,
        )
        self.keep_alive.append(weights.packed)
        w_const = network.add_constant(
            (experts, out_features, in_features),
            trt.Weights(
                trt.DataType.FP4, weights.packed.ctypes.data, experts * out_features * in_features
            ),
        ).get_output(0)
        self.keep_alive.append(weights.block_scale)
        block_scale = network.add_constant(
            (experts, out_features, in_features // NVFP4_BLOCK_SIZE),
            trt.Weights(
                trt.DataType.FP8, weights.block_scale.ctypes.data, weights.block_scale.size
            ),
        ).get_output(0)
        effective = network.add_dequantize(
            block_scale, self._scalar(network, weights.global_scale), dtype
        )
        w_dequant = network.add_dequantize(w_const, effective.get_output(0), dtype)
        w_dequant.axis = 2
        return network.add_matrix_multiply(
            activation,
            trt.MatrixOperation.NONE,
            w_dequant.get_output(0),
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)

    def moe(
        self, network, hidden, top_indices, top_scores, layer_name: str, hidden_size: int
    ) -> trt.ITensor:
        """Routed-expert output ``[1, hidden]``.

        ``hidden`` is ``[1, hidden]``; ``top_indices`` and ``top_scores`` are
        ``[1, top_k]`` (scores already soft-maxed over the selected experts).
        """
        layer = self.experts[layer_name]
        dtype = hidden.dtype
        experts = layer.gate_up.num_experts
        inter = layer.gate_up.out_features // 2
        top_k = top_indices.shape[-1]
        elementwise = trt.ElementWiseOperation

        def reshape(tensor, dims):
            shuffle = network.add_shuffle(tensor)
            shuffle.reshape_dims = dims
            return shuffle.get_output(0)

        def scalar(value: float):
            return self._const(network, np.full((1, 1, 1), value, dtype=np.float16))

        def interleaved(tensor, offset: int):
            return network.add_slice(
                tensor, (0, 0, offset), (experts, 1, inter), (1, 1, 2)
            ).get_output(0)

        x = self._quantized_activation(network, hidden, layer.gate_up.input_scale, 1)
        gate_up = self._batched_matmul(
            network, reshape(x, (1, 1, hidden_size)), layer.gate_up, dtype
        )
        bias = lambda array: self._const(network, array.reshape(experts, 1, -1))  # noqa: E731
        gate = network.add_elementwise(
            interleaved(gate_up, 0), bias(layer.gate_bias), elementwise.SUM
        ).get_output(0)
        up = network.add_elementwise(
            interleaved(gate_up, 1), bias(layer.up_bias), elementwise.SUM
        ).get_output(0)

        # GPT-OSS gate: clamp(gate, max=7), clamp(up, -7, 7), gate * sigmoid(1.702 * gate) * (up + 1)
        gate = network.add_elementwise(gate, scalar(7.0), elementwise.MIN).get_output(0)
        up = network.add_elementwise(up, scalar(7.0), elementwise.MIN).get_output(0)
        up = network.add_elementwise(up, scalar(-7.0), elementwise.MAX).get_output(0)
        scaled = network.add_elementwise(gate, scalar(1.702), elementwise.PROD).get_output(0)
        sigmoid = network.add_activation(scaled, trt.ActivationType.SIGMOID).get_output(0)
        glu = network.add_elementwise(gate, sigmoid, elementwise.PROD).get_output(0)
        up_plus_one = network.add_elementwise(up, scalar(1.0), elementwise.SUM).get_output(0)
        gated = network.add_elementwise(up_plus_one, glu, elementwise.PROD).get_output(0)

        gated = self._quantized_activation(network, gated, layer.down.input_scale, 2)
        down = self._batched_matmul(network, gated, layer.down, dtype)  # [E, 1, hidden]
        down = network.add_elementwise(down, bias(layer.down_bias), elementwise.SUM).get_output(0)

        selected = network.add_gather(down, reshape(top_indices, (top_k,)), 0).get_output(0)
        weights = reshape(network.add_cast(top_scores, dtype).get_output(0), (top_k, 1, 1))
        weighted = network.add_elementwise(selected, weights, elementwise.PROD).get_output(0)
        total = network.add_reduce(weighted, trt.ReduceOperation.SUM, 0b001, False).get_output(0)
        return reshape(total, (1, hidden_size))


def calibrate_gpt_oss(model_dir: Path, config, graph_ops, quantization: str) -> GptOssQuantContext:
    """Build the quantization context from an official MXFP4 GPT-OSS checkpoint.

    Reads and repacks every layer's experts (no forward pass). Raises when the
    checkpoint is not an MXFP4 checkpoint instead of falling back to unquantized weights.
    """
    if quantization != "nvfp4":
        raise NotImplementedError(f"gpt_oss does not support quantization={quantization!r}")
    activation_scales = load_activation_scales(config)
    tensors = _open_raw_readers(Path(model_dir))
    experts: dict[str, _ExpertLayer] = {}
    for layer in range(int(config.num_hidden_layers)):
        experts[f"layer.{layer}"] = read_experts(tensors, layer, activation_scales)
    return GptOssQuantContext(experts=experts, graph_ops=graph_ops)
