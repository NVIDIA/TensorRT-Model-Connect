# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phi-MoE-owned FP8 self-quantization and TensorRT Q/DQ graph context.

No ModelOpt FP8 or NVFP4 checkpoint of Phi-3.5-MoE exists on the Hub (the quantized
repositories are community exports in GGUF, AWQ, GPTQ or compressed-tensors format with
per-token dynamic activation scales, which do not fit the static ``input_scale`` path). So
``quantization="fp8"`` quantizes the original BF16 checkpoint at build time, one tensor at a
time, and feeds the FP8 bytes to the engine as constants. FP8 weights are never stored in the
repository or on the Hub.

Two kinds of projections are quantized (internal weight names, ``N`` is the layer index):

Attention q/k/v, W8A8 (FP8 weights and FP8 activations)
    ``layer.N.w_{q,k,v}``. ``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3
    value). The activation scale of each projection input needs real data, so it was calibrated
    once offline with ``calibrate_fp8_scales.py`` and is read from
    ``fp8_activation_scales.json``. Users do not run calibration. A projection with no entry in
    that file is an error (a ValueError names the projection and the calibration tool), except
    ``w_o``, which is never quantized. Each quantized projection becomes
    ``activation -> Quantize -> Dequantize -> MatMul`` with ``FP8 weight -> Dequantize``. The
    Dequantize layers and the matmul run in float32 and the result is cast back, because a
    float16 Dequantize makes the fused FP8 GEMM overflow to NaN when an output is large.

Routed experts, W8A16 (FP8 weights, activations stay in the compute type)
    ``layer.N.experts.w_{gate,up,down}`` hold most of the decoder weights. Both the
    unquantized and the FP8 engine gather the two experts the SparseMixer top-2 indices select
    from one stacked constant per projection and run batched matmuls (``_add_moe_block``); the
    FP8 engine only changes how the stacked constant is stored and dequantized. One scalar
    weight scale per layer and projection (all experts share it); no activation scale is needed.
    TensorRT rejects a Gather directly on an FP8 constant and folds a Dequantize of constants
    into a full fp16 constant (which would store the experts at 16 bits). The "grid" route
    avoids both: the stack is an fp16 constant that holds the FP8 grid values (the exact e4m3
    values), the gathered experts go through ``Quantize`` with scale 1.0 and then ``Dequantize``
    with the weight scale. TensorRT moves the Quantize onto the constant, so the plan stores
    the experts as FP8 and the gather reads FP8 bytes, and only the two selected experts are
    dequantized. This is a size option: at decode the dequantize-and-gather kernel is slower
    than the plain fp16 gather (see the pull request for the measurements).

    The batched prefill engine of a split bundle runs every expert on every token over the
    whole stack, so it has no gather to hide the dequantize behind: it uses the FP8 stack
    constant and a Dequantize whose scale is multiplied by a value that is 1 at run time but is
    computed from the router indices, so TensorRT cannot fold the dequantize into an fp16
    constant and the plan keeps the experts as FP8.

Not quantized: the attention output projection ``w_o`` (TensorRT 11.4 cannot compile an FP8
Quantize right after the fused attention), the router, the embedding, ``lm_head``, the norms and
all biases. Weights keep the value layout of the unquantized engine, ``[in, out]``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorrt as trt
import torch
from safetensors import safe_open

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"

_SUPPORTED = ("fp8",)
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"
# (HF stem under ``model.layers.N``, internal weight stem)
ATTENTION_PROJECTIONS = (
    ("self_attn.q_proj", "w_q"),
    ("self_attn.k_proj", "w_k"),
    ("self_attn.v_proj", "w_v"),
)
# (HF stem under ``model.layers.N.block_sparse_moe.experts.E``, internal stem)
EXPERT_PROJECTIONS = (
    ("w1", "w_gate"),
    ("w3", "w_up"),
    ("w2", "w_down"),
)
# The attention output projection stays in fp16 even if a scale file lists it: TensorRT 11.4
# cannot compile an FP8 Quantize right after the fused attention.
UNQUANTIZED = re.compile(r"^layer\.\d+\.w_o$")


@dataclass(frozen=True)
class _FP8Weight:
    """One projection weight stored as raw fp8_e4m3 bytes with one per-tensor scale."""

    packed: np.ndarray  # [out_features, in_features] raw fp8_e4m3 bytes (checkpoint layout)
    weight_scale: float
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _LayerScales:
    """Activation scale and the packed weight of one quantized attention projection."""

    input_scale: float
    weight: _FP8Weight


@dataclass(frozen=True)
class _ExpertStack:
    """One routed-expert projection of a layer: all experts as FP8 bytes with one scale."""

    packed: np.ndarray  # [experts, in_features, out_features] raw fp8_e4m3 bytes
    weight_scale: float

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return ``(experts, in_features, out_features)``."""
        return tuple(int(v) for v in self.packed.shape)  # type: ignore[return-value]


@dataclass(frozen=True)
class _ExpertLayer:
    """The routed experts of one layer."""

    gate: _ExpertStack
    up: _ExpertStack
    down: _ExpertStack


def _scale_const(network, graph_ops, value: float):
    """One-element fp32 constant holding a Q/DQ scale."""
    return graph_ops.add_constant(
        network, (1,), np.asarray([value], dtype=np.float32), dtype=np.float32
    )


def _wrap_fp8_matmul(network, activation, scales: _LayerScales, *, graph_ops, keep_alive: list):
    """Add ``activation @ weight.T`` with FP8 weights and a static FP8 activation scale.

    The Dequantize layers and the matmul run in float32 and the result is cast back. With
    fp16 Dequantize outputs, TensorRT's FP8 GEMM keeps the raw accumulator in fp16 and applies
    the scales afterwards, so it overflows to NaN as soon as an output exceeds about
    ``65504 * input_scale * weight_scale``.
    """
    w = scales.weight
    n, k = w.out_features, w.in_features

    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(w.packed)
    w_fp8 = trt.Weights(trt.DataType.FP8, w.packed.ctypes.data, n * k)
    w_const = network.add_constant((n, k), w_fp8).get_output(0)
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
class PhiMoeQuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]  # attention projections: "layer.N.w_q", ...
    experts: dict[str, _ExpertLayer]  # routed experts: "layer.N"
    graph_ops: Any
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        """Create the list that keeps weight buffers alive during the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, weight_name: str) -> bool:
        """True for a quantized attention projection (``layer.3.w_q``) or an expert stack."""
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
        """Quantized matmul for quantized projections; the regular constant matmul otherwise."""
        entry = self.scales.get(weight_name)
        if entry is None:
            return self.graph_ops.add_matmul_rhs_constant(
                network, lhs, lhs_width, rhs_width, rhs_weights, dtype=dtype
            )
        return _wrap_fp8_matmul(
            network, lhs, entry, graph_ops=self.graph_ops, keep_alive=self.keep_alive
        )

    def expert_gatherer(self, network, layer_name: str, top_indices, dtype):
        """Return ``gathered(stem)`` that gathers one FP8 expert projection of a layer.

        ``stem`` is ``w_gate``, ``w_up`` or ``w_down``; the result is ``[1, top_k, in, out]`` in
        ``dtype`` (a TensorRT type). It replaces the fp16 constant of the unquantized gather
        (grid route, see the module docstring): ``top_indices`` (int32 ``[1, top_k]``) picks the
        experts from an fp16 constant that holds the FP8 grid values, then ``Quantize(1.0)`` and
        ``Dequantize(weight scale)``. The rest of the MoE block is shared with the unquantized
        path.
        """
        layer = self.experts[layer_name]

        def constant(array: np.ndarray):
            self.keep_alive.append(array)
            return network.add_constant(array.shape, trt.Weights(array)).get_output(0)

        unit_scale = constant(np.ones((1,), dtype=np.float32))

        def gathered(stem: str):
            stack: _ExpertStack = {"w_gate": layer.gate, "w_up": layer.up, "w_down": layer.down}[
                stem
            ]
            grid = torch.from_numpy(stack.packed).view(torch.float8_e4m3fn).half().numpy()
            picked = network.add_gather(constant(grid), top_indices, 0).get_output(0)
            quantized = network.add_quantize(picked, unit_scale, trt.DataType.FP8)
            return network.add_dequantize(
                quantized.get_output(0),
                constant(np.asarray([stack.weight_scale], dtype=np.float32)),
                dtype,
            ).get_output(0)

        return gathered

    def expert_stacker(self, network, layer_name: str, top_indices, dtype):
        """Return ``stacked(stem)`` for the batched prefill engine: all experts, ``[E, in, out]``.

        The FP8 stack constant goes through ``Dequantize`` with the weight scale times a value
        that is 1 at run time (derived from ``top_indices``), so TensorRT keeps the constant in
        FP8 instead of folding the dequantize into a 16-bit constant.
        """
        layer = self.experts[layer_name]

        def constant(array: np.ndarray):
            self.keep_alive.append(array)
            return network.add_constant(array.shape, trt.Weights(array)).get_output(0)

        elementwise = trt.ElementWiseOperation
        zero = network.add_elementwise(
            network.add_cast(top_indices, trt.DataType.FLOAT).get_output(0),
            constant(np.zeros((1, 1), dtype=np.float32)),
            elementwise.PROD,
        ).get_output(0)
        zero = network.add_reduce(zero, trt.ReduceOperation.SUM, 0b11, True).get_output(0)
        unit = network.add_elementwise(
            zero, constant(np.ones((1, 1), dtype=np.float32)), elementwise.SUM
        ).get_output(0)
        shuffle = network.add_shuffle(unit)
        shuffle.reshape_dims = (1,)
        unit = shuffle.get_output(0)

        def stacked(stem: str):
            stack: _ExpertStack = {"w_gate": layer.gate, "w_up": layer.up, "w_down": layer.down}[
                stem
            ]
            self.keep_alive.append(stack.packed)
            weights = trt.Weights(trt.DataType.FP8, stack.packed.ctypes.data, stack.packed.size)
            const = network.add_constant(stack.shape, weights).get_output(0)
            scale = network.add_elementwise(
                constant(np.asarray([stack.weight_scale], dtype=np.float32)),
                unit,
                elementwise.PROD,
            ).get_output(0)
            return network.add_dequantize(const, scale, dtype).get_output(0)

        return stacked


def configure_fp8_build(trt_config, quant_ctx) -> None:
    """Apply the FP8 build options and keep layer names for kernel inspection."""
    if quant_ctx is None:
        return
    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if _DUAL_GEMM_KNOB in getattr(trt_config, "all_build_routes", ""):
        # TensorRT's fused dual-GEMM kernel fails to compile with scalar-scale FP8.
        # The route is set only when this TensorRT build lists the knob.
        trt_config.build_route = f"-{_DUAL_GEMM_KNOB}=off"


def scale_key(config) -> str:
    """Key of this model's entry in the activation-scale file."""
    return (
        f"{str(config.model_type).lower()}"
        f"-h{int(config.hidden_size)}"
        f"-l{int(config.num_hidden_layers)}"
        f"-i{int(config.intermediate_size)}"
    )


def quantize_fp8_weight(weight: torch.Tensor | np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize one weight to FP8 e4m3 with a single scalar scale.

    Returns the raw fp8 bytes (same shape as the weight) and
    ``scale = max(abs(weight)) / 448``; the same scale is the dequantize scale in the engine.
    """
    tensor = torch.as_tensor(weight).float()
    amax = float(tensor.abs().max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    return _to_fp8_bytes(tensor, amax / FP8_E4M3_MAX), amax / FP8_E4M3_MAX


def _to_fp8_bytes(weight_fp32: torch.Tensor, scale: float) -> np.ndarray:
    """Round ``weight / scale`` to e4m3 (nearest even, saturating) and return the raw bytes."""
    scaled = (weight_fp32 / scale).clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return np.ascontiguousarray(scaled.to(torch.float8_e4m3fn).view(torch.uint8).numpy())


def dequantize_fp8_weight(packed: np.ndarray, scale: float) -> torch.Tensor:
    """The float32 values the FP8 bytes stand for (what the engine dequantizes to)."""
    return torch.from_numpy(packed).view(torch.float8_e4m3fn).float() * scale


def load_activation_scales(config) -> dict[str, float]:
    """Return the committed activation scales for this model config, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Phi-MoE FP8 build requires {path}, which was not found") from error
    key = scale_key(config)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model config."
        )
    return table[key]


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


def _read_weight(tensors: dict[str, Any], hf_prefix: str) -> torch.Tensor:
    """Read ``<hf_prefix>.weight`` as float32, refusing an already quantized checkpoint."""
    key = f"{hf_prefix}.weight"
    if f"{hf_prefix}.weight_scale" in tensors or f"{hf_prefix}.weight_scale_2" in tensors:
        raise ValueError(
            f"{hf_prefix} is already quantized; fp8 self-quantization needs the original checkpoint"
        )
    if key not in tensors:
        raise ValueError(f"checkpoint has no {key}")
    tensor = tensors[key].get_tensor(key)
    if not tensor.dtype.is_floating_point or tensor.element_size() == 1:
        raise ValueError(
            f"{key} has dtype {tensor.dtype}; fp8 self-quantization needs the original "
            "bf16 / fp16 checkpoint"
        )
    return tensor.float()


def _quantize_expert_stack(
    tensors: dict[str, Any], layer_prefix: str, num_experts: int, hf_stem: str
) -> _ExpertStack:
    """All experts of one projection: one scale from the layer-wide amax, bytes as [E, in, out]."""
    names = [f"{layer_prefix}.block_sparse_moe.experts.{e}.{hf_stem}" for e in range(num_experts)]
    amax = 0.0
    for name in names:
        amax = max(amax, float(_read_weight(tensors, name).abs().max()))
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError(f"{layer_prefix} experts {hf_stem} have no finite positive amax")
    scale = amax / FP8_E4M3_MAX
    packed = None
    for expert, name in enumerate(names):
        # checkpoint [out, in] -> engine [in, out]
        weight = _read_weight(tensors, name).t()
        if packed is None:
            packed = np.empty((num_experts, *weight.shape), dtype=np.uint8)
        packed[expert] = _to_fp8_bytes(weight, scale)
    return _ExpertStack(packed=packed, weight_scale=scale)


def calibrate_phi_moe(model_dir: Path, config, graph_ops, quantization: str) -> PhiMoeQuantContext:
    """Quantize the checkpoint one tensor at a time and pair it with the stored scales."""
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"Phi-MoE supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    if config.raw.get("quantization_config"):
        raise ValueError(
            "the checkpoint config has a quantization_config; fp8 self-quantization needs the "
            "original bf16 checkpoint"
        )
    activation_scales = load_activation_scales(config)
    tensors = _open_raw_readers(Path(model_dir))
    num_experts = int(config.raw.get("num_local_experts", 16))
    scales: dict[str, _LayerScales] = {}
    experts: dict[str, _ExpertLayer] = {}
    for layer in range(int(config.num_hidden_layers)):
        layer_prefix = f"model.layers.{layer}"
        for hf_stem, stem in ATTENTION_PROJECTIONS:
            name = f"layer.{layer}.{stem}"
            if UNQUANTIZED.match(name):
                continue
            input_scale = activation_scales.get(name)
            if input_scale is None:
                raise ValueError(
                    f"no FP8 activation scale for {name} in {ACTIVATION_SCALES_FILENAME}; "
                    "run calibrate_fp8_scales.py for this model config"
                )
            if not np.isfinite(input_scale) or input_scale <= 0:
                raise ValueError(f"activation scale of {name} must be finite and positive")
            weight = _read_weight(tensors, f"{layer_prefix}.{hf_stem}")
            packed, weight_scale = quantize_fp8_weight(weight)
            out_features, in_features = weight.shape
            scales[name] = _LayerScales(
                float(input_scale), _FP8Weight(packed, weight_scale, out_features, in_features)
            )
        stacks = {
            stem: _quantize_expert_stack(tensors, layer_prefix, num_experts, hf_stem)
            for hf_stem, stem in EXPERT_PROJECTIONS
        }
        experts[f"layer.{layer}"] = _ExpertLayer(
            gate=stacks["w_gate"], up=stacks["w_up"], down=stacks["w_down"]
        )
    if not scales or not experts:
        raise RuntimeError("Phi-MoE FP8 self-quantization matched no projection")
    return PhiMoeQuantContext(scales=scales, experts=experts, graph_ops=graph_ops)
