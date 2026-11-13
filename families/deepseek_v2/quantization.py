# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V2-owned FP8 weight-only quantization of the routed experts.

``quantization="fp8"`` quantizes the routed experts of the original BF16 checkpoint at build
time, one projection at a time, and feeds the FP8 bytes to the engine as constants. No ModelOpt
FP8 checkpoint of DeepSeek-V2-Lite exists on the Hub (the FP8 repositories are community
exports: DeepSeek-V3 style 128x128 block scales, or a per-expert static format of the Coder
fine-tune only), so the BF16 checkpoint is the single source. FP8 weights are never stored in
the repository or on the Hub.

Only the routed experts ``layer.N.experts.w_{gate,up,down}`` are quantized (internal weight
names, ``N`` is the layer index). They hold about 93 % of the weights of DeepSeek-V2-Lite. The
format is W8A16: FP8 weights, activations stay in the layer's compute type, so no activation
scale and no calibration data are needed. ``scale = max(abs(weight)) / 448`` (448 is the
largest finite e4m3 value), one scalar per layer and projection (all experts of a layer share
it).

The engine gathers the selected experts' weights per token from the router's top-k indices,
dequantizes them and runs batched matmuls. TensorRT folds a Dequantize of constants into a full
fp16 constant, which would store the experts at 16 bits, and it only accepts an FP8 constant in
front of a Dequantize (a Gather directly on FP8 bytes is rejected on TensorRT 11.1 and 11.4).
So the Dequantize runs on the whole expert stack and its scale is multiplied by a value that is
1 at run time but is computed from the router indices, so it cannot be folded; TensorRT then
fuses the dequantize into the gather.

Not quantized: the MLA attention projections, the dense MLP of the first layer, the shared
experts, the router, the embedding, ``lm_head`` and the norms. Quantizing them as W8A8 gives no
decode speed and a 2.7 times larger KL to the unquantized engine (see the pull request).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np

from .checkpoint_mapper import _has_tensor, _open_safetensors

FP8_E4M3_MAX = 448.0

_SUPPORTED = ("fp8",)
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"
_QUANTIZE_PPG_KNOB = "quantize_ppg"
# B300 (sm103): TensorRT 11.4 picks a slow fused Dequantize+Gather kernel for the routed experts.
_SLOW_EXPERT_GATHER_SM = (10, 3)
# (HF stem under ``mlp.experts.E``, internal weight stem)
EXPERT_PROJECTIONS = (
    ("gate_proj", "w_gate"),
    ("up_proj", "w_up"),
    ("down_proj", "w_down"),
)
# Tensors a quantized checkpoint carries next to ``<name>.weight``.
_QUANT_SIDE_TENSORS = ("weight_scale", "weight_scale_inv", "weight_scale_2", "input_scale")
_FLOAT_DTYPES = (np.dtype(np.float16), np.dtype(np.float32), np.dtype(ml_dtypes.bfloat16))


@dataclass(frozen=True)
class _ExpertStack:
    """One routed-expert projection of a layer: all experts as FP8 bytes with one scale."""

    packed: np.ndarray  # [experts, in_features, out_features] raw fp8_e4m3 bytes
    weight_scale: float

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the stack as ``(experts, in_features, out_features)``."""
        return tuple(int(v) for v in self.packed.shape)  # type: ignore[return-value]


@dataclass(frozen=True)
class _ExpertLayer:
    """The routed experts of one layer."""

    gate: _ExpertStack
    up: _ExpertStack
    down: _ExpertStack


@dataclass(frozen=True)
class DeepseekV2QuantContext:
    """Owns the FP8 routed experts; everything else stays unquantized."""

    experts: dict[str, _ExpertLayer]  # keyed by layer prefix: "layer.N"
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        """Create the list that keeps weight buffers alive during the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def quantizes_experts(self, layer_prefix: str) -> bool:
        """True when the routed experts of ``layer.N`` are FP8 constants of this context."""
        return layer_prefix in self.experts

    def routed_experts(
        self,
        network,
        inp,
        top_indices,
        routing_weights,
        layer_name: str,
        hidden_size: int,
        top_k: int,
    ):
        """Routed-expert output ``[tokens, hidden]`` from the gathered FP8 experts.

        ``inp`` is ``[tokens, hidden]`` in the layer's compute type; ``top_indices`` and
        ``routing_weights`` are ``[tokens, top_k]`` (weights already normalized or scaled).
        """
        import tensorrt as trt

        layer = self.experts[layer_name]
        dtype = inp.dtype
        elementwise = trt.ElementWiseOperation

        def reshape(tensor, dims):
            shuffle = network.add_shuffle(tensor)
            shuffle.reshape_dims = dims
            return shuffle.get_output(0)

        def constant(array: np.ndarray):
            self.keep_alive.append(array)
            return network.add_constant(array.shape, trt.Weights(array)).get_output(0)

        def run_time_one():
            """A ``[1]`` fp32 tensor equal to 1.0 at run time that depends on the router output.

            It multiplies the dequantize scale, so TensorRT cannot fold the dequantize of the
            FP8 constants to fp16.
            """
            first = network.add_slice(top_indices, (0, 0), (1, 1), (1, 1)).get_output(0)
            first = network.add_cast(first, trt.DataType.FLOAT).get_output(0)
            zero = network.add_elementwise(
                first, constant(np.zeros((1, 1), dtype=np.float32)), elementwise.PROD
            ).get_output(0)
            one = network.add_elementwise(
                zero, constant(np.ones((1, 1), dtype=np.float32)), elementwise.SUM
            ).get_output(0)
            return reshape(one, (1,))

        unit = run_time_one()

        def gathered(stack: _ExpertStack):
            experts, in_features, out_features = stack.shape
            self.keep_alive.append(stack.packed)
            weights = trt.Weights(trt.DataType.FP8, stack.packed.ctypes.data, stack.packed.size)
            const = network.add_constant((experts, in_features, out_features), weights).get_output(
                0
            )
            scale = network.add_elementwise(
                constant(np.asarray([stack.weight_scale], dtype=np.float32)),
                unit,
                elementwise.PROD,
            ).get_output(0)
            dequant = network.add_dequantize(const, scale, dtype).get_output(0)
            return network.add_gather(dequant, top_indices, 0).get_output(0)

        rows = reshape(inp, (-1, 1, 1, hidden_size))

        def batched_matmul(lhs, rhs):
            return network.add_matrix_multiply(
                lhs, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.NONE
            ).get_output(0)

        gate = batched_matmul(rows, gathered(layer.gate))
        up = batched_matmul(rows, gathered(layer.up))
        silu = network.add_elementwise(
            gate,
            network.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0),
            elementwise.PROD,
        ).get_output(0)
        inner = network.add_elementwise(silu, up, elementwise.PROD).get_output(0)
        down = batched_matmul(inner, gathered(layer.down))  # [tokens, top_k, 1, hidden]

        per_expert = reshape(down, (-1, top_k, hidden_size))
        weights_3d = reshape(routing_weights, (-1, top_k, 1))
        if weights_3d.dtype != per_expert.dtype:
            weights_3d = network.add_cast(weights_3d, per_expert.dtype).get_output(0)
        weighted = network.add_elementwise(per_expert, weights_3d, elementwise.PROD).get_output(0)
        return network.add_reduce(
            weighted, trt.ReduceOperation.SUM, 1 << 1, keep_dims=False
        ).get_output(0)


def _knob_listed(trt_config, knob: str) -> bool:
    """True when this TensorRT build lists ``knob`` (exact option name) in its build routes."""
    return f'"-{knob}"' in getattr(trt_config, "all_build_routes", "")


def _compute_capability() -> tuple[int, int]:
    """Compute capability of the current CUDA device, the one TensorRT builds for."""
    from cuda.bindings import runtime

    status, device = runtime.cudaGetDevice()
    if status != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGetDevice failed: {status}")
    status, props = runtime.cudaGetDeviceProperties(device)
    if status != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGetDeviceProperties failed: {status}")
    return int(props.major), int(props.minor)


def configure_fp8_build(trt_config, quant_ctx) -> None:
    """Apply the build options of FP8 builds; unquantized builds (``quant_ctx`` None) are untouched.

    - TensorRT's fused dual-GEMM kernel (gate and up of the unquantized shared experts and dense
      MLP) is turned off. With it on, the FP8 engine decodes 5 % slower on B200.
    - On B300 only, quantize propagation is turned off. TensorRT 11.4 picks a slow fused
      Dequantize+Gather kernel for the routed experts on B300 (98 us per layer against 46 us
      on B200); with the knob off it uses a faster one (71 us). The same knob slows B200
      (46 -> 74 us), so it is set on B300 only.

    Each knob is set only when this TensorRT build lists it.
    """
    if quant_ctx is None:
        return
    routes = []
    if _knob_listed(trt_config, _DUAL_GEMM_KNOB):
        routes.append(f"-{_DUAL_GEMM_KNOB}=off")
    if (
        _knob_listed(trt_config, _QUANTIZE_PPG_KNOB)
        and _compute_capability() == _SLOW_EXPERT_GATHER_SM
    ):
        routes.append(f"-{_QUANTIZE_PPG_KNOB}=off")
    if routes:
        trt_config.build_route = " ".join(routes)  # several knobs: space separated


def quantize_fp8_weight(weight: np.ndarray, scale: float | None = None) -> tuple[np.ndarray, float]:
    """Quantize one weight to FP8 e4m3 with a single scalar scale.

    ``scale`` defaults to ``max(abs(weight)) / 448``. Returns the raw fp8 bytes (same shape as
    the weight) and the scale; the same scale is the dequantize scale in the engine.
    """
    values = np.asarray(weight, dtype=np.float32)
    if scale is None:
        amax = float(np.abs(values).max())
        if not np.isfinite(amax) or amax <= 0:
            raise ValueError("weight tensor has no finite positive amax to scale from")
        scale = amax / FP8_E4M3_MAX
    scaled = np.clip(values / np.float32(scale), -FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return np.ascontiguousarray(packed), float(scale)


def dequantize_fp8_weight(packed: np.ndarray, scale: float) -> np.ndarray:
    """The float32 values the FP8 bytes stand for (what the engine dequantizes to)."""
    return packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * np.float32(scale)


def _read_expert_weight(readers, name: str) -> np.ndarray:
    """Read ``<name>.weight`` in its checkpoint dtype, refusing an already quantized one."""
    if any(_has_tensor(readers, f"{name}.{side}") for side in _QUANT_SIDE_TENSORS):
        raise ValueError(
            f"{name} is already quantized; fp8 self-quantization needs the original checkpoint"
        )
    key = f"{name}.weight"
    if not _has_tensor(readers, key):
        raise ValueError(f"checkpoint has no {key}")
    tensor = readers.tensor_map[key].get_tensor(key)
    if tensor.dtype not in _FLOAT_DTYPES:
        raise ValueError(
            f"{key} has dtype {tensor.dtype}; fp8 self-quantization needs the original "
            "bf16 / fp16 / fp32 checkpoint"
        )
    return tensor


def _quantize_expert_stack(
    readers, layer_prefix: str, num_experts: int, hf_stem: str
) -> _ExpertStack:
    """All experts of one projection: one scale from the layer-wide amax, bytes as [E, in, out]."""
    weights = [
        _read_expert_weight(readers, f"{layer_prefix}.mlp.experts.{e}.{hf_stem}")
        for e in range(num_experts)
    ]
    shapes = {tuple(w.shape) for w in weights}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError(f"{layer_prefix} experts {hf_stem} do not share one rank-2 shape")
    amax = max(float(np.abs(w.astype(np.float32)).max()) for w in weights)
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError(f"{layer_prefix} experts {hf_stem} have no finite positive amax")
    scale = amax / FP8_E4M3_MAX
    out_features, in_features = weights[0].shape
    packed = np.empty((num_experts, in_features, out_features), dtype=np.uint8)
    for expert, weight in enumerate(weights):
        # checkpoint [out, in] -> engine [in, out]
        packed[expert] = quantize_fp8_weight(weight.T, scale)[0]
    return _ExpertStack(packed=packed, weight_scale=scale)


def quantize_deepseek_v2_experts(
    model_dir: Path, quantization: str, config
) -> DeepseekV2QuantContext:
    """Quantize the routed experts of the checkpoint, one projection at a time."""
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"DeepSeek-V2 supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    if str(config.model_type).lower() != "deepseek_v2":
        raise NotImplementedError(
            f"DeepSeek-V2 fp8 supports model_type deepseek_v2; got {config.model_type!r}"
        )
    raw = config.raw
    if raw.get("quantization_config"):
        raise ValueError(
            "the checkpoint config has a quantization_config; fp8 self-quantization needs the "
            "original bf16 checkpoint"
        )
    readers = _open_safetensors(Path(model_dir))
    num_experts = int(raw.get("n_routed_experts", 64))
    first_moe = int(raw.get("first_k_dense_replace", 1))
    moe_freq = int(raw.get("moe_layer_freq", 1))
    experts: dict[str, _ExpertLayer] = {}
    for layer in range(int(config.num_hidden_layers)):
        if layer < first_moe or (layer - first_moe) % moe_freq != 0:
            continue
        stacks = {
            stem: _quantize_expert_stack(readers, f"model.layers.{layer}", num_experts, hf_stem)
            for hf_stem, stem in EXPERT_PROJECTIONS
        }
        experts[f"layer.{layer}"] = _ExpertLayer(
            gate=stacks["w_gate"], up=stacks["w_up"], down=stacks["w_down"]
        )
    if not experts:
        raise RuntimeError("DeepSeek-V2 FP8 quantization found no routed-expert layer")
    return DeepseekV2QuantContext(experts=experts)
