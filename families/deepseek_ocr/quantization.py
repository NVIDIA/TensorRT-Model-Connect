# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-OCR-owned FP8 TensorRT Q/DQ graph context for the language decoder.

FP8 (``quantization="fp8"``) is self-quantization of the original checkpoint: no
ModelOpt FP8 checkpoint exists for DeepSeek-OCR-2 (the FP8 / 4-bit repositories on the Hub
are community exports in compressed-tensors or GPTQ format with per-token dynamic activation
scales, which TensorRT does not run as an FP8 GEMM). Weights are quantized at build time, one
tensor at a time, and the FP8 bytes are fed to the engine as constants. FP8 weights are never
stored in the repository or on the Hub.

Two kinds of projections are quantized (internal weight names, ``N`` is the layer index):

Dense projections, W8A8 (FP8 weights and FP8 activations)
    ``layer.N.w_{q,k,v}`` (attention), ``layer.0.w_{gate,up,down}`` (the dense MLP of layer 0)
    and ``layer.N.shared.w_{gate,up,down}`` (the two shared experts of every MoE layer).
    ``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3 value); the activation
    scale of each projection input needs real images and prompts, so it was calibrated once
    offline with ``calibrate_fp8_scales.py`` and is read from ``fp8_activation_scales.json``.
    Users do not run calibration. A projection with no entry in that file stays unquantized.
    Each quantized projection becomes
    ``activation -> Quantize -> Dequantize -> MatMul`` with ``FP8 weight -> Dequantize``.

Routed experts, W8A16 (FP8 weights, activations stay in the layer's compute type)
    ``layer.N.experts.w_{gate,up,down}`` hold about 93 % of the decoder-layer weights. The
    engine gathers the selected experts' weights per token from the router's top-k indices,
    dequantizes them and runs batched matmuls. The weights are constants of the engine in FP8,
    one scalar scale per projection tensor (all experts of a layer share it). No activation
    scale is needed. TensorRT folds a Dequantize of constants into a full fp16 constant,
    which would store the experts at 16 bits, and it only accepts an FP8 constant in front of a
    Dequantize (a Gather directly on FP8 bytes is rejected). So the Dequantize runs on the
    whole expert stack and the scale is multiplied by a value that is 1 at run time but is
    computed from the router indices, so it cannot be folded; TensorRT then fuses the
    dequantize into the gather. The 1.0 of the up and down projections is computed from the
    output of the previous projection: this makes TensorRT gather the three weight stacks one
    after the other instead of all at once, which would triple the activation memory of the
    prefill engine (the gathered weights of 256 tokens are several GiB per projection).

Not quantized: the attention output projection ``w_o`` (TensorRT 11.4 cannot compile an FP8
Quantize right after the fused attention), the router, the embedding, ``lm_head``, the norms and
the whole vision engine. The vision part stays fp16. Weights keep the checkpoint's value layout
transposed to ``[in, out]`` like the unquantized engine, so the matmuls need no transpose.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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
DENSE_MLP_PROJECTIONS = (
    ("mlp.gate_proj", "w_gate"),
    ("mlp.up_proj", "w_up"),
    ("mlp.down_proj", "w_down"),
)
SHARED_EXPERT_PROJECTIONS = (
    ("mlp.shared_experts.gate_proj", "shared.w_gate"),
    ("mlp.shared_experts.up_proj", "shared.w_up"),
    ("mlp.shared_experts.down_proj", "shared.w_down"),
)
EXPERT_PROJECTIONS = (
    ("gate_proj", "w_gate"),
    ("up_proj", "w_up"),
    ("down_proj", "w_down"),
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
    """Activation scale and the packed weight of one quantized dense projection."""

    input_scale: float
    weight: _FP8Weight


@dataclass(frozen=True)
class _ExpertStack:
    """One routed-expert projection of a layer: all experts as FP8 bytes with one scale."""

    packed: np.ndarray  # [experts, in_features, out_features] raw fp8_e4m3 bytes
    weight_scale: float

    @property
    def shape(self) -> tuple[int, int, int]:
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

    The Dequantize layers and the matmul run in float32 and the result is cast back.
    With fp16 Dequantize outputs, TensorRT's FP8 GEMM keeps the raw accumulator in
    fp16 and applies the scales afterwards, so it overflows to NaN as soon as an
    output exceeds about ``65504 * input_scale * weight_scale``. Large outputs do
    occur in real models.
    """
    import tensorrt as trt

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
class DeepseekOcrQuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]  # dense projections: "layer.N.w_q", "layer.N.shared.w_up", ...
    experts: dict[str, _ExpertLayer]  # routed experts: "layer.N"
    graph_ops: Any
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        """Create the list that keeps weight buffers alive during the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, weight_name: str) -> bool:
        """True for a quantized dense projection (``layer.3.w_q``) or a quantized expert stack."""
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

        def unit_after(tensor):
            """A ``[1]`` fp32 tensor equal to 1.0 at run time that depends on one element of ``tensor``.

            It multiplies the dequantize scale, so TensorRT cannot fold the dequantize of the
            FP8 constants to fp16, and it orders the three gathers of one layer: the weights
            of the next projection are only gathered after the previous matmul has run.
            Without that ordering TensorRT gathers all three at once and the prefill
            engine's activation memory triples.
            """
            dims = len(tuple(tensor.shape))
            first = network.add_slice(tensor, (0,) * dims, (1,) * dims, (1,) * dims).get_output(0)
            if first.dtype not in (trt.DataType.HALF, trt.DataType.FLOAT):
                first = network.add_cast(first, trt.DataType.FLOAT).get_output(0)
            zero_dtype = np.float16 if first.dtype == trt.DataType.HALF else np.float32
            zero = network.add_elementwise(
                first, constant(np.zeros((1,) * dims, dtype=zero_dtype)), elementwise.PROD
            ).get_output(0)
            zero = network.add_cast(zero, trt.DataType.FLOAT).get_output(0)
            one = network.add_elementwise(
                reshape(zero, (1, 1)), constant(np.ones((1, 1), dtype=np.float32)), elementwise.SUM
            ).get_output(0)
            return reshape(one, (1,))

        def gathered(stack: _ExpertStack, unit):
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

        gate = batched_matmul(rows, gathered(layer.gate, unit_after(top_indices)))
        up = batched_matmul(rows, gathered(layer.up, unit_after(gate)))
        silu = network.add_elementwise(
            gate,
            network.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0),
            elementwise.PROD,
        ).get_output(0)
        inner = network.add_elementwise(silu, up, elementwise.PROD).get_output(0)
        down = batched_matmul(
            inner, gathered(layer.down, unit_after(up))
        )  # [tokens, top_k, 1, hidden]

        per_expert = reshape(down, (-1, top_k, hidden_size))
        weights_3d = reshape(routing_weights, (-1, top_k, 1))
        weighted = network.add_elementwise(per_expert, weights_3d, elementwise.PROD).get_output(0)
        return network.add_reduce(
            weighted, trt.ReduceOperation.SUM, 1 << 1, keep_dims=False
        ).get_output(0)

    def routed_experts_dense(
        self,
        network,
        inp,
        top_indices,
        routing_weights,
        layer_name: str,
        hidden_size: int,
        top_k: int,
    ):
        """All-experts variant of :meth:`routed_experts` for many tokens (prefill).

        Three plain matmuls over the whole FP8 expert stacks (``[hidden, E * inter]`` for
        gate and up, ``[E * inter, hidden]`` for down), see
        ``graph_blocks.add_dense_routed_swiglu_experts``. The stacks are dequantized with the
        run-time 1.0 trick so TensorRT keeps the FP8 constants.
        """
        import tensorrt as trt

        from . import graph_blocks

        layer = self.experts[layer_name]
        dtype = inp.dtype
        elementwise = trt.ElementWiseOperation
        experts, hidden, inter = layer.gate.shape

        # ``[E, in, out]`` bytes re-laid for the all-experts matmuls
        gate_packed = np.ascontiguousarray(np.transpose(layer.gate.packed, (1, 0, 2)))
        up_packed = np.ascontiguousarray(np.transpose(layer.up.packed, (1, 0, 2)))
        down_packed = np.ascontiguousarray(layer.down.packed)

        def constant(array: np.ndarray):
            self.keep_alive.append(array)
            return network.add_constant(array.shape, trt.Weights(array)).get_output(0)

        first = network.add_slice(top_indices, (0, 0), (1, 1), (1, 1)).get_output(0)
        zero = network.add_elementwise(
            network.add_cast(first, trt.DataType.FLOAT).get_output(0),
            constant(np.zeros((1, 1), dtype=np.float32)),
            elementwise.PROD,
        ).get_output(0)
        unit = network.add_elementwise(
            zero, constant(np.ones((1, 1), dtype=np.float32)), elementwise.SUM
        ).get_output(0)
        unit = network.add_shuffle(unit)
        unit.reshape_dims = (1,)
        unit = unit.get_output(0)

        def dequantized(packed: np.ndarray, weight_scale: float, shape):
            self.keep_alive.append(packed)
            weights = trt.Weights(trt.DataType.FP8, packed.ctypes.data, packed.size)
            const = network.add_constant(shape, weights).get_output(0)
            scale = network.add_elementwise(
                constant(np.asarray([weight_scale], dtype=np.float32)), unit, elementwise.PROD
            ).get_output(0)
            return network.add_dequantize(const, scale, dtype).get_output(0)

        return graph_blocks.add_dense_routed_swiglu_experts(
            network,
            inp,
            top_indices,
            routing_weights,
            dequantized(gate_packed, layer.gate.weight_scale, (hidden, experts * inter)),
            dequantized(up_packed, layer.up.weight_scale, (hidden, experts * inter)),
            dequantized(down_packed, layer.down.weight_scale, (experts * inter, hidden)),
            hidden_size=hidden_size,
            top_k=top_k,
            num_experts=experts,
            inter_size=inter,
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
    """Return the committed activation scales for this model shape, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(
            f"DeepSeek-OCR FP8 build requires {path}, which was not found"
        ) from error
    key = scale_key(config)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model shape."
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
    names = [f"{layer_prefix}.mlp.experts.{e}.{hf_stem}" for e in range(num_experts)]
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


def calibrate_deepseek_ocr(
    model_dir: Path, graph_ops, quantization: str, config
) -> DeepseekOcrQuantContext:
    """Quantize the checkpoint one tensor at a time and pair it with the stored scales."""
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"DeepSeek-OCR supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    raw = config.raw
    if raw.get("quantization_config") or (raw.get("language_config") or {}).get(
        "quantization_config"
    ):
        raise ValueError(
            "the checkpoint config has a quantization_config; fp8 self-quantization needs the "
            "original bf16 checkpoint"
        )
    activation_scales = load_activation_scales(config)
    tensors = _open_raw_readers(Path(model_dir))
    # DeepSeek-VL2 style checkpoints keep the decoder under "language.model.*".
    model_prefix = "language.model" if "language.model.embed_tokens.weight" in tensors else "model"
    num_experts = int(raw.get("n_routed_experts", 64))
    first_moe = int(raw.get("first_k_dense_replace", 1))
    scales: dict[str, _LayerScales] = {}
    experts: dict[str, _ExpertLayer] = {}
    for layer in range(int(config.num_hidden_layers)):
        layer_prefix = f"{model_prefix}.layers.{layer}"
        mlp = DENSE_MLP_PROJECTIONS if layer < first_moe else SHARED_EXPERT_PROJECTIONS
        for hf_stem, stem in (*ATTENTION_PROJECTIONS, *mlp):
            name = f"layer.{layer}.{stem}"
            hf_prefix = f"{layer_prefix}.{hf_stem}"
            input_scale = activation_scales.get(name)
            if input_scale is None or UNQUANTIZED.match(name):
                continue
            weight = _read_weight(tensors, hf_prefix)
            packed, weight_scale = quantize_fp8_weight(weight)
            out_features, in_features = weight.shape
            scales[name] = _LayerScales(
                float(input_scale), _FP8Weight(packed, weight_scale, out_features, in_features)
            )
        if layer >= first_moe:
            stacks = {
                stem: _quantize_expert_stack(tensors, layer_prefix, num_experts, hf_stem)
                for hf_stem, stem in EXPERT_PROJECTIONS
            }
            experts[f"layer.{layer}"] = _ExpertLayer(
                gate=stacks["w_gate"], up=stacks["w_up"], down=stacks["w_down"]
            )
    if not scales or not experts:
        raise RuntimeError("DeepSeek-OCR FP8 self-quantization matched no projection")
    return DeepseekOcrQuantContext(scales=scales, experts=experts, graph_ops=graph_ops)
