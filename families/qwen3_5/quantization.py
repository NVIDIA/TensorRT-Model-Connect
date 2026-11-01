# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5-owned NVFP4 and FP8 TensorRT Q/DQ graph context.

FP8 (``quantization="fp8"``) is self-quantization of the original BF16 checkpoint,
because no ModelOpt FP8 checkpoint exists for these sizes. Weights are quantized
at build time, one projection at a time: ``scale = max(abs(weight)) / 448`` (448 is
the largest finite e4m3 value), and the FP8 bytes are fed to the engine as a
constant. The activation scale of each projection input needs real data, so it was
calibrated once offline with ``calibrate_fp8_scales.py`` (ModelOpt
``FP8_DEFAULT_CFG``) and is read from ``fp8_activation_scales.json``. Users do not
run calibration. A projection with no entry in that file stays unquantized. Each
quantized projection becomes ``activation -> Quantize -> Dequantize -> MatMul`` with
``FP8 weight -> Dequantize`` on the other input.

NVFP4 (``quantization="nvfp4"``) reads ModelOpt NVFP4 exports of Qwen3.5 text models,
such as ``AxionML/Qwen3.5-9B-NVFP4``. Every projection of the language model carries:

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

import ml_dtypes
import numpy as np
import tensorrt as trt
import torch
from safetensors import safe_open

NVFP4_BLOCK_SIZE = 16
FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"

_SUPPORTED = ("nvfp4", "fp8")
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
class _FP8Weight:
    """One projection weight stored as raw fp8_e4m3 bytes with one per-tensor scale."""

    packed: np.ndarray  # [out_features, in_features] raw fp8_e4m3 bytes
    weight_scale: float
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _LayerScales:
    """Activation scale and the packed weight of one quantized projection."""

    input_scale: float
    weight: _NVFP4Weight | _FP8Weight


def _scale_const(network, graph_ops, value: float):
    """One-element fp32 constant holding a Q/DQ scale."""
    return graph_ops.add_constant(
        network, (1,), np.asarray([value], dtype=np.float32), dtype=np.float32
    )


def _finish_matmul(network, activation, a_dequant, w_dequant):
    """Matmul of the dequantized activation with the transposed weight, in the input dtype."""
    output = network.add_matrix_multiply(
        a_dequant,
        trt.MatrixOperation.NONE,
        w_dequant,
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    if output.dtype == activation.dtype:
        return output
    return network.add_cast(output, activation.dtype).get_output(0)


def _wrap_fp8_matmul(network, activation, scales: _LayerScales, *, graph_ops, keep_alive: list):
    """Add ``activation @ weight.T`` with FP8 weights and a static FP8 activation scale.

    The Dequantize layers and the matmul run in float32 and the result is cast back.
    With fp16 Dequantize outputs, TensorRT's FP8 GEMM keeps the raw accumulator in
    fp16 and applies the scales afterwards, so it overflows to NaN as soon as an
    output exceeds about ``65504 * input_scale * weight_scale``. Large outputs do
    occur in real models.
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

    return _finish_matmul(network, activation, a_dequant.get_output(0), w_dequant.get_output(0))


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

    return _finish_matmul(network, activation, a_dequant.get_output(0), w_dequant.get_output(0))


@dataclass(frozen=True)
class Qwen35QuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    graph_ops: Any
    keep_alive: list = None  # type: ignore[assignment]

    @property
    def has_fp8_weights(self) -> bool:
        """True when any projection is FP8 (TensorRT's fused gate/up kernel then fails to compile)."""
        return any(isinstance(entry.weight, _FP8Weight) for entry in self.scales.values())

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
        wrap = _wrap_fp8_matmul if isinstance(entry.weight, _FP8Weight) else _wrap_nvfp4_matmul
        return wrap(network, lhs, entry, graph_ops=self.graph_ops, keep_alive=self.keep_alive)


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
    config=None,
) -> Qwen35QuantContext:
    """Build the Q/DQ context for ``quantization`` (``"nvfp4"`` or ``"fp8"``).

    NVFP4 reads a ModelOpt NVFP4 Qwen3.5 checkpoint. FP8 quantizes the BF16 checkpoint
    and needs ``config`` for the activation-scale key.

    ``layer_types`` lists ``"deltanet"`` or ``"attention"`` per layer. The checkpoint
    already carries calibrated weights and activation scales, so no forward pass is
    needed. Raises when the checkpoint does not match the requested format, instead
    of silently falling back to unquantized weights.
    """
    if quantization not in _SUPPORTED:
        raise NotImplementedError(f"qwen3_5 does not support quantization={quantization!r}")
    if quantization == "fp8":
        return _calibrate_fp8(
            Path(model_dir),
            graph_ops,
            config,
            layer_types=layer_types,
            num_heads=num_heads,
            head_dim=head_dim,
        )
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


def scale_key(config) -> str:
    """Key of this model's entry in the activation-scale file."""
    # The builder accepts both "qwen3_5" and "qwen3.5"; the table keys use the first.
    return (
        f"{str(config.model_type).lower().replace('.', '_')}"
        f"-h{int(config.hidden_size)}"
        f"-l{int(config.num_hidden_layers)}"
        f"-i{int(config.intermediate_size)}"
    )


def quantize_fp8_weight(weight_fp32: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize one weight to FP8 e4m3 with a single scalar scale.

    Returns the raw fp8 bytes and ``scale = max(abs(weight)) / 448``; the same scale is
    the dequantize scale in the engine.
    """
    amax = float(np.abs(weight_fp32).max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scale = amax / FP8_E4M3_MAX
    scaled = np.clip(weight_fp32 / scale, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return np.ascontiguousarray(packed), scale


def load_activation_scales(config) -> dict[str, float]:
    """Return the committed activation scales for this model shape, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Qwen3.5 FP8 build requires {path}, which was not found") from error
    key = scale_key(config)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model shape."
        )
    return table[key]


def _calibrate_fp8(
    model_dir: Path, graph_ops, config, *, layer_types, num_heads: int, head_dim: int
) -> Qwen35QuantContext:
    """Quantize the BF16 checkpoint one projection at a time and pair it with the stored scales."""
    if config is None:
        raise ValueError("fp8 self-quantization needs the model config")
    activation_scales = load_activation_scales(config)
    tensors = _open_raw_readers(model_dir)
    layers = (
        "model.language_model.layers"
        if "model.language_model.layers.0.mlp.gate_proj.weight" in tensors
        else "model.layers"
    )

    def read(prefix: str) -> np.ndarray:
        key = f"{prefix}.weight"
        if key not in tensors:
            raise ValueError(f"checkpoint has no {key}")
        if f"{prefix}.weight_scale" in tensors or f"{prefix}.weight_scale_2" in tensors:
            raise ValueError(
                f"{prefix} is already quantized; fp8 self-quantization needs the BF16 checkpoint"
            )
        return tensors[key].get_tensor(key).float().numpy()

    def entry(name: str, weight: np.ndarray, input_scale: float) -> _LayerScales:
        packed, weight_scale = quantize_fp8_weight(weight)
        out_features, in_features = weight.shape
        return _LayerScales(
            float(input_scale), _FP8Weight(packed, weight_scale, out_features, in_features)
        )

    scales: dict[str, _LayerScales] = {}
    for layer, kind in enumerate(layer_types):
        mixer = _DELTANET_PROJECTIONS if kind == "deltanet" else _ATTENTION_PROJECTIONS
        for hf_stem, weight_stem in (*mixer, *_MLP_PROJECTIONS):
            name = f"layer.{layer}.{weight_stem}"
            input_scale = activation_scales.get(name)
            if input_scale is not None:
                scales[name] = entry(name, read(f"{layers}.{layer}.{hf_stem}"), input_scale)
        name = f"layer.{layer}.w_q"
        input_scale = activation_scales.get(name) if kind == "attention" else None
        if input_scale is not None:
            raw = read(f"{layers}.{layer}.{_Q_PROJECTION}")
            if raw.shape[0] != num_heads * 2 * head_dim:
                raise ValueError(
                    f"q_proj has {raw.shape[0]} rows, expected {num_heads * 2 * head_dim}"
                )
            per_head = raw.reshape(num_heads, 2 * head_dim, raw.shape[1])
            for stem, part in (
                (name, slice(0, head_dim)),
                (f"layer.{layer}.w_gate_attn", slice(head_dim, 2 * head_dim)),
            ):
                half = np.ascontiguousarray(per_head[:, part, :].reshape(num_heads * head_dim, -1))
                scales[stem] = entry(stem, half, input_scale)
    if not scales:
        raise RuntimeError("Qwen3.5 FP8 self-quantization matched no projection in the checkpoint")
    return Qwen35QuantContext(scales=scales, graph_ops=graph_ops)
