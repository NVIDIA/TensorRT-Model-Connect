# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LFM2 MoE FP8 self-quantization of the routed experts.

``quantization="fp8"`` quantizes the routed experts of the original BF16 checkpoint at build
time. LiquidAI's ``LFM2.5-24B-A2B-FP8`` checkpoint uses 128x128 block weight scales and dynamic
per-group activations (compressed-tensors), which TensorRT cannot run as an FP8 GEMM, so it is
not used. FP8 weights are never stored in the repository or on the Hub.

Recipe (W8A8, static per-tensor scales)
    Quantized: the routed experts of every MoE layer, the fused gate/up projection
    (``w1``/``w3``) and the down projection (``w2``). They hold about 95 % of the weights.
    One weight scale per layer and projection (all experts share it):
    ``scale = max(abs(weight)) / 448``. The activation scale of each expert input is
    ``amax / 448`` with ``amax`` the largest input any expert of the layer saw over real
    prompts; it was calibrated once offline with ``calibrate_fp8_scales.py`` and is read from
    ``fp8_activation_scales.json``, keyed by model config and final-norm checksum. A down
    projection whose input amax is more than 20 times the median of all layers is stored as
    ``null`` and stays fp32 (four layers of LFM2.5-24B-A2B).
    Not quantized: attention, the conv blocks, the two dense MLP layers, the router and its
    expert bias, the embedding, ``lm_head`` and the norms. Quantizing attention, the dense MLP
    or the conv projections as well gives about 1 % more decode speed and a larger error.

Graph
    Each selected expert is picked with a dynamic slice (the router's top-k indices give the
    start) from an fp16 constant that holds the FP8 grid values (exact e4m3 values), then
    ``Quantize(1.0)`` and ``Dequantize(weight scale)``. TensorRT moves the Quantize onto the
    constant, so the plan stores the experts as FP8 and the slices copy FP8 bytes. The
    activation goes through ``Quantize``/``Dequantize`` with its static scale, so the batched
    expert matmuls run as FP8 GEMMs. Dequantize and matmul run in fp32, the family precision.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tensorrt as trt

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"
# A down-projection input whose amax exceeds this multiple of the median stays fp32.
OUTLIER_RATIO = 20.0
PROJECTIONS = ("gate_up", "down")
_QUANT_SIDE_TENSORS = ("weight_scale", "weight_scale_inv", "weight_scale_2", "input_scale")


def scale_key(cfg: dict, final_norm: np.ndarray) -> str:
    """Key of a checkpoint in the activation-scale file: model config plus final-norm checksum."""
    norm_sum = float(np.sum(np.asarray(final_norm, dtype=np.float32), dtype=np.float64))
    return (
        f"lfm2_moe-h{int(cfg['hidden_size'])}"
        f"-l{int(cfg['num_hidden_layers'])}"
        f"-e{int(cfg['num_experts'])}"
        f"-i{int(cfg['moe_intermediate_size'])}"
        f"-n{norm_sum:.4f}"
    )


def moe_layers(cfg: dict) -> range:
    """Indices of the layers with routed experts."""
    return range(int(cfg["num_dense_layers"]), int(cfg["num_hidden_layers"]))


def scale_name(layer: int, projection: str) -> str:
    return f"layer.{layer}.experts.{projection}"


def outlier_scales(scales: dict[str, float]) -> dict[str, float | None]:
    """Replace the down-projection scales above ``OUTLIER_RATIO`` x their median with None."""
    downs = [value for name, value in scales.items() if name.endswith(".down")]
    limit = OUTLIER_RATIO * statistics.median(downs)
    return {
        name: None if name.endswith(".down") and value > limit else value
        for name, value in scales.items()
    }


def load_activation_scales(cfg: dict, final_norm: np.ndarray, path: Path | None = None) -> dict:
    """The committed activation scales of this checkpoint; raise when none exist."""
    path = path or Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    key = scale_key(cfg, final_norm)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated LFM2 MoE FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run families/lfm2_moe/calibrate_fp8_scales.py for this checkpoint."
        )
    scales = table[key]
    for layer in moe_layers(cfg):
        for projection in PROJECTIONS:
            name = scale_name(layer, projection)
            if name not in scales:
                raise ValueError(f"LFM2 MoE FP8 scale file has no {name} for {key!r}")
            value = scales[name]
            if value is None and projection == "gate_up":
                raise ValueError(f"LFM2 MoE FP8 scale file has no scale for {name}")
            if value is not None and not (np.isfinite(value) and value > 0):
                raise ValueError(
                    f"LFM2 MoE FP8 activation scale of {name} must be finite and positive"
                )
    return scales


def quantize_fp8_stack(stack: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize an expert stack to FP8 e4m3 with one scale; return the grid values as fp16.

    The fp16 values are exactly the e4m3 values (every e4m3 value is an fp16 value), and
    ``scale = max(abs(weight)) / 448`` is the dequantize scale.
    """
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(stack, dtype=np.float32))
    amax = float(tensor.abs().max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("expert weights have no finite positive amax to scale from")
    scale = amax / FP8_E4M3_MAX
    grid = (tensor / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return np.ascontiguousarray(grid.to(torch.float16).numpy()), scale


@dataclass(frozen=True)
class FP8Experts:
    """One expert projection of one layer, every expert, as FP8 grid values."""

    grid: np.ndarray  # [experts, out, in] float16 holding e4m3 values
    weight_scale: float
    input_scale: float


@dataclass
class Lfm2MoeFP8Context:
    """FP8 experts per MoE layer: ``{layer: {"gate_up": FP8Experts, "down": FP8Experts | None}}``."""

    experts: dict[int, dict[str, FP8Experts | None]]
    keep_alive: list = field(default_factory=list)

    def _scalar(self, network, value: float):
        array = np.asarray([value], dtype=np.float32)
        self.keep_alive.append(array)
        return network.add_constant((1,), trt.Weights(array)).get_output(0)

    def activation(self, network, tensor, input_scale: float):
        """``Dequantize(Quantize(tensor))`` with the static per-tensor scale."""
        scale = self._scalar(network, input_scale)
        quantize = network.add_quantize(tensor, scale, trt.DataType.FP8)
        return network.add_dequantize(quantize.get_output(0), scale, tensor.dtype).get_output(0)

    def selected(self, network, stack, expert_ids, top: int):
        """The selected experts ``[top, out, in]`` of a constant stack, one dynamic slice each."""
        zeros = np.zeros(2, dtype=np.int32)
        self.keep_alive.append(zeros)
        zeros = network.add_constant((2,), trt.Weights(zeros)).get_output(0)
        rows = []
        for slot in range(top):
            expert = network.add_slice(expert_ids, (slot,), (1,), (1,)).get_output(0)
            start = network.add_concatenation([expert, zeros]).get_output(0)
            pick = network.add_slice(stack, (0, 0, 0), (1, *stack.shape[1:]), (1, 1, 1))
            pick.set_input(1, start)
            rows.append(pick.get_output(0))
        joined = network.add_concatenation(rows)
        joined.axis = 0
        return joined.get_output(0)

    def selected_weight(self, network, experts: FP8Experts, expert_ids, top: int):
        """Dequantized FP8 weights of the selected experts, fp32 ``[top, out, in]``."""
        self.keep_alive.append(experts.grid)
        grid = network.add_constant(experts.grid.shape, trt.Weights(experts.grid)).get_output(0)
        picked = self.selected(network, grid, expert_ids, top)
        quantize = network.add_quantize(picked, self._scalar(network, 1.0), trt.DataType.FP8)
        return network.add_dequantize(
            quantize.get_output(0), self._scalar(network, experts.weight_scale), trt.float32
        ).get_output(0)


def calibrate_lfm2_moe_fp8(cfg: dict, weights: dict, scales_path: Path | None = None):
    """Quantize the expert stacks of the loaded fp32 weights; consumes the quantized stacks.

    ``weights`` comes from ``model.load_weights``. Each quantized stack is removed from it
    after quantization, so the fp32 copy is not kept next to the FP8 one.
    """
    if any(str(key).rsplit(".", 1)[-1] in _QUANT_SIDE_TENSORS for key in weights):
        raise ValueError(
            "LFM2 MoE FP8 self-quantization needs the original BF16 checkpoint; this checkpoint "
            "is already quantized"
        )
    scales = load_activation_scales(cfg, weights["model.embedding_norm.weight"], scales_path)
    experts: dict[int, dict[str, FP8Experts | None]] = {}
    for layer in moe_layers(cfg):
        entry: dict[str, FP8Experts | None] = {}
        for projection in PROJECTIONS:
            input_scale = scales[scale_name(layer, projection)]
            key = f"model.layers.{layer}.feed_forward.experts.{projection}"
            if input_scale is None:
                entry[projection] = None
                continue
            grid, weight_scale = quantize_fp8_stack(weights.pop(key))
            entry[projection] = FP8Experts(grid, weight_scale, float(input_scale))
        experts[layer] = entry
    return Lfm2MoeFP8Context(experts)
