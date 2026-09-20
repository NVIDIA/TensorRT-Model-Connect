# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8-owned NVFP4 + FP8 TensorRT Q/DQ graph context.

RadixArk/Qwen3.8-27B-NVFP4 is a ModelOpt MIXED_PRECISION export that carries
two quantization schemes side by side:

  NVFP4  MLP projections (gate/up/down) and lm_head: E2M1 values packed two
         per uint8, a per-16-element "<name>.weight_scale" (fp8_e4m3), and a
         global "<name>.weight_scale_2" (fp32).
  FP8    Attention and DeltaNet projections (q, k, v, o, in_proj_qkv,
         in_proj_z, out_proj): float8_e4m3 weights with a single per-tensor
         "<name>.weight_scale" (fp32, no block/global split).

Both schemes also carry a per-tensor "<name>.input_scale" for their
activations.

This module reuses the checkpoint's own quantized bytes directly, bit-exact --
it does NOT dequantize then re-quantize the weight (checkpoint_mapper.py's
dequantized float array is only used by the *unquantized* fallback path, e.g.
the DeltaNet decay/beta projections this checkpoint leaves unquantized).
TensorRT's ``add_dequantize(value, scale)`` computes ``value * scale``, which
is exactly the formula checkpoint_mapper.py's own dequantization helpers use,
so feeding the checkpoint's packed bytes + scales into the graph reconstructs
the identical value the original quantization produced.

Weights are fed in their native checkpoint layout ([out_features, in_features])
rather than transposed to TensorRT's usual [in, out] convention -- transposing
packed sub-byte (NVFP4: 2-per-uint8) data would require a full
unpack/transpose/repack pass. Instead the matmul itself does the transpose
(``MatrixOperation.TRANSPOSE`` on the weight operand).

The Q/DQ pairs below are TensorRT's standard explicit-quantization idiom, not
a literal "dequantize then multiply" at runtime: Myelin's CASK6/CUTLASS SM100
backend recognizes the Quantize/DynamicQuantize -> Dequantize -> MatMul
pattern and fuses it into a native FP4 or FP8 tensor-core GEMM kernel
operating on the packed data directly (verified via IEngineInspector tactic
names, e.g. ``tensorop256x128x64...`` for FP4 and a ``CastMulCast`` ->
``Fc`` kernel pair for FP8) -- nothing gets materialized at full precision in
the built engine or at inference time.

The activation side uses two different mechanisms depending on format:
  - NVFP4: ``add_dynamic_quantize`` (IDynamicQuantizeLayer), because
    blockwise ``add_quantize`` to an FP4 output type is unconditionally
    rejected by Myelin's shape checker; IDynamicQuantizeLayer is the one
    layer type with a fused FP4 tensor-core kernel.
  - FP8: plain ``add_quantize``/``add_dequantize`` (FP8 has no blockwise
    requirement), matching families/qwen's proven FP8 pattern.
Both use the checkpoint's calibrated "<name>.input_scale" as a build-time
constant, since activations are only known at runtime.

``calibrate_qwen3_8_fp8`` below is a separate scheme for the *original*
(unquantized) ``Qwen/Qwen3.8-27B`` BF16 checkpoint: it self-quantizes MLP
(gate/up/down), attention (q/k/v/o), and DeltaNet (in_proj_qkv/in_proj_z/
out_proj) to FP8 on the fly at build time (bit-exact-style scalar scale,
same mechanism as ``_FP8Format`` above, just computed here instead of read
from a checkpoint), using a small one-time-calibrated activation-scale file
bundled with this family instead of a checkpoint-shipped ``input_scale``. See
that function's docstring for why (two other published FP8 checkpoint
formats were tried first and found architecturally unable to reach a real
fused FP8 GEMM kernel).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tensorrt as trt

from .checkpoint_mapper import _open_safetensors, _has_tensor, _get_raw_tensor, _to_numpy_fp32


NVFP4_BLOCK_SIZE = 16

# Real HF tensor names (relative to a layer) that carry calibrated NVFP4
# scales in the checkpoint, paired with the internal weight-dict name used by
# graph_blocks.add_swiglu_mlp / engine_builder.
_NVFP4_MLP_PROJECTIONS = (
    ("mlp.gate_proj", "w_gate"),
    ("mlp.up_proj", "w_up"),
    ("mlp.down_proj", "w_down"),
)
_NVFP4_LM_HEAD_HF_NAME = "lm_head"
_NVFP4_LM_HEAD_WEIGHT_NAME = "w_lm_head"

# FP8-quantized DeltaNet and attention projections. Decay/beta (in_proj_a,
# in_proj_b) are deliberately excluded: the checkpoint carries no scales for
# them (too small/sensitive to quantize), so they stay in the unquantized
# fallback path automatically (should_quantize() returns False for names not
# in the scales dict).
_FP8_DELTANET_PROJECTIONS = (
    ("linear_attn.in_proj_qkv", "deltanet_in_proj_qkv"),
    ("linear_attn.in_proj_z", "deltanet_z_proj"),
    ("linear_attn.out_proj", "deltanet_out_proj"),
)
_FP8_ATTENTION_SIMPLE_PROJECTIONS = (
    ("self_attn.k_proj", "w_k"),
    ("self_attn.v_proj", "w_v"),
    ("self_attn.o_proj", "w_o"),
)

# Self-quantized FP8 (families/qwen3_8/quantization.py::calibrate_qwen3_8_fp8):
# weight scale is computed on the fly from the original checkpoint's own
# BF16 weights (single scalar per tensor, matching the scheme proven to fuse
# into a real FP8 tensor-core GEMM), and the activation scale is read from a
# one-time-calibrated scales file bundled with this family (see
# calibrate_qwen3_8_fp8's docstring).
_FP8_SELF_QUANTIZED_MLP_PROJECTIONS = (
    ("mlp.gate_proj", "w_gate"),
    ("mlp.up_proj", "w_up"),
    ("mlp.down_proj", "w_down"),
)
_FP8_ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"

# Passed as IBuilderConfig.build_route by engine_builder.py when
# Qwen38QuantContext.disable_dual_gemm_fusion is set, to work around the
# TRT dual-GEMM fusion bug described above. "-peep:match_dual_gemm" is a
# whitelisted TRT compiler knob (confirmed via
# IBuilderConfig.all_build_routes on the public TensorRT 11.1 wheel).
_DISABLE_DUAL_GEMM_BUILD_ROUTE = "-peep:match_dual_gemm=off"


@dataclass(frozen=True)
class _NVFP4Weight:
    """The checkpoint's own packed NVFP4 weight, reused bit-exact."""

    packed: np.ndarray          # [out_features, in_features // 2] uint8
    block_scale_f8: np.ndarray  # [out_features, in_features // 16] raw fp8_e4m3 bytes (as uint8)
    global_scale: float
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _FP8Weight:
    """The checkpoint's own packed FP8 weight, reused bit-exact."""

    packed: np.ndarray  # [out_features, in_features] raw fp8_e4m3 bytes (as uint8)
    weight_scale: float
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _LayerScales:
    input_scale: float
    weight: _NVFP4Weight | _FP8Weight


class _NVFP4Format:
    name = "nvfp4"

    @staticmethod
    def wrap_matmul(
        network,
        activation,
        scales: _LayerScales,
        *,
        dtype: np.dtype,
        graph_ops,
        keep_alive: list,
    ):
        output_dtype = activation.dtype
        w = scales.weight
        n, k = w.out_features, w.in_features

        # Keep the backing numpy buffers alive until the engine is built --
        # trt.Weights holds a raw pointer into them, not a copy.
        keep_alive.append(w.packed)
        keep_alive.append(w.block_scale_f8)

        def scale_const(value: float):
            return graph_ops.add_constant(
                network, (1,), np.asarray([value], dtype=np.float32), dtype=np.float32
            )

        # --- weight side: reuse the checkpoint's own packed FP4 bytes as-is ---
        w_fp4 = trt.Weights(trt.DataType.FP4, w.packed.ctypes.data, n * k)
        w_const = network.add_constant((n, k), w_fp4).get_output(0)

        w_fp8_bs = trt.Weights(trt.DataType.FP8, w.block_scale_f8.ctypes.data, w.block_scale_f8.size)
        w_bs_const = network.add_constant((n, k // NVFP4_BLOCK_SIZE), w_fp8_bs).get_output(0)

        w_gs = scale_const(w.global_scale)
        w_effective_scale = network.add_dequantize(w_bs_const, w_gs, output_dtype)
        w_dequant = network.add_dequantize(w_const, w_effective_scale.get_output(0), output_dtype)
        w_dequant.axis = 1

        # --- activation side: dynamic-quantize with the checkpoint's real scale ---
        act_gs = scale_const(scales.input_scale)
        a_dynq = network.add_dynamic_quantize(
            activation, 1, NVFP4_BLOCK_SIZE, trt.DataType.FP4, trt.DataType.FP8
        )
        a_dynq.set_input(1, act_gs)
        a_f4 = a_dynq.get_output(0)
        a_block_scales_f8 = a_dynq.get_output(1)

        a_effective_scale = network.add_dequantize(a_block_scales_f8, act_gs, output_dtype)
        a_dequant = network.add_dequantize(a_f4, a_effective_scale.get_output(0), output_dtype)
        a_dequant.axis = 1

        output = network.add_matrix_multiply(
            a_dequant.get_output(0),
            trt.MatrixOperation.NONE,
            w_dequant.get_output(0),
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        return (
            output
            if output.dtype == output_dtype
            else network.add_cast(output, output_dtype).get_output(0)
        )


class _FP8Format:
    name = "fp8"

    @staticmethod
    def wrap_matmul(
        network,
        activation,
        scales: _LayerScales,
        *,
        dtype: np.dtype,
        graph_ops,
        keep_alive: list,
    ):
        output_dtype = activation.dtype
        w = scales.weight
        n, k = w.out_features, w.in_features

        keep_alive.append(w.packed)

        def scale_const(value: float):
            return graph_ops.add_constant(
                network, (1,), np.asarray([value], dtype=np.float32), dtype=np.float32
            )

        # --- weight side: reuse the checkpoint's own packed FP8 bytes as-is ---
        w_fp8 = trt.Weights(trt.DataType.FP8, w.packed.ctypes.data, n * k)
        w_const = network.add_constant((n, k), w_fp8).get_output(0)
        w_scale = scale_const(w.weight_scale)
        w_dequant = network.add_dequantize(w_const, w_scale, output_dtype)

        # --- activation side: quantize with the checkpoint's real scale ---
        act_scale = scale_const(scales.input_scale)
        a_quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
        a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, output_dtype)

        output = network.add_matrix_multiply(
            a_dequant.get_output(0),
            trt.MatrixOperation.NONE,
            w_dequant.get_output(0),
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        return (
            output
            if output.dtype == output_dtype
            else network.add_cast(output, output_dtype).get_output(0)
        )


def _wrap_matmul(network, activation, scales: _LayerScales, *, dtype, graph_ops, keep_alive):
    if isinstance(scales.weight, _NVFP4Weight):
        fmt = _NVFP4Format
    elif isinstance(scales.weight, _FP8Weight):
        fmt = _FP8Format
    else:
        raise TypeError(f"Unsupported quantized weight type: {type(scales.weight)!r}")
    return fmt.wrap_matmul(
        network, activation, scales, dtype=dtype, graph_ops=graph_ops, keep_alive=keep_alive)


@dataclass(frozen=True)
class _Profile:
    scales: dict[str, _LayerScales]

    def should_quantize(self, name: str) -> bool:
        return name in self.scales


@dataclass(frozen=True)
class Qwen38QuantContext:
    profile: _Profile
    graph_ops: Any
    keep_alive: list = None  # type: ignore[assignment]
    disable_dual_gemm_fusion: bool = False

    def __post_init__(self):
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

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
        if not self.profile.should_quantize(weight_name):
            return self.graph_ops.add_matmul_rhs_constant(
                network, lhs, lhs_width, rhs_width, rhs_weights, dtype=dtype
            )
        return _wrap_matmul(
            network,
            lhs,
            self.profile.scales[weight_name],
            dtype=dtype,
            graph_ops=self.graph_ops,
            keep_alive=self.keep_alive,
        )


def _read_input_scale(readers, hf_prefix: str) -> float | None:
    key = f"{hf_prefix}.input_scale"
    if not _has_tensor(readers, key):
        return None
    value = _to_numpy_fp32(_get_raw_tensor(readers, key)).reshape(-1)
    if value.size != 1 or not np.isfinite(value[0]) or value[0] <= 0:
        return None
    return float(value[0])


def _read_nvfp4_weight(readers, hf_prefix: str, out_features: int, in_features: int) -> _NVFP4Weight | None:
    weight_key = f"{hf_prefix}.weight"
    block_scale_key = f"{hf_prefix}.weight_scale"
    global_scale_key = f"{hf_prefix}.weight_scale_2"
    if not (
        _has_tensor(readers, weight_key)
        and _has_tensor(readers, block_scale_key)
        and _has_tensor(readers, global_scale_key)
    ):
        return None

    packed_raw = _get_raw_tensor(readers, weight_key)
    packed = packed_raw.numpy() if hasattr(packed_raw, "numpy") else np.asarray(packed_raw)
    packed = np.ascontiguousarray(packed.astype(np.uint8, copy=False))
    if packed.shape != (out_features, in_features // 2):
        raise ValueError(
            f"{weight_key} has shape {packed.shape}, expected "
            f"({out_features}, {in_features // 2})")

    block_scale_raw = _get_raw_tensor(readers, block_scale_key)
    if not hasattr(block_scale_raw, "view"):
        raise TypeError(f"{block_scale_key} is not a torch tensor (fp8 needs a raw byte view)")
    block_scale_f8 = np.ascontiguousarray(
        block_scale_raw.view(torch.uint8).numpy())
    if block_scale_f8.shape != (out_features, in_features // NVFP4_BLOCK_SIZE):
        raise ValueError(
            f"{block_scale_key} has shape {block_scale_f8.shape}, expected "
            f"({out_features}, {in_features // NVFP4_BLOCK_SIZE})")

    global_scale = float(
        _to_numpy_fp32(_get_raw_tensor(readers, global_scale_key)).reshape(-1)[0])

    return _NVFP4Weight(
        packed=packed,
        block_scale_f8=block_scale_f8,
        global_scale=global_scale,
        out_features=out_features,
        in_features=in_features,
    )


def _read_fp8_weight(readers, hf_prefix: str) -> _FP8Weight | None:
    """Read a whole-tensor FP8 weight, inferring shape from the tensor itself."""
    weight_key = f"{hf_prefix}.weight"
    scale_key = f"{hf_prefix}.weight_scale"
    if not (_has_tensor(readers, weight_key) and _has_tensor(readers, scale_key)):
        return None
    raw = _get_raw_tensor(readers, weight_key)
    if not hasattr(raw, "view"):
        raise TypeError(f"{weight_key} is not a torch tensor (fp8 needs a raw byte view)")
    packed = np.ascontiguousarray(raw.view(torch.uint8).numpy())
    out_features, in_features = packed.shape
    weight_scale = float(_to_numpy_fp32(_get_raw_tensor(readers, scale_key)).reshape(-1)[0])
    return _FP8Weight(
        packed=packed, weight_scale=weight_scale,
        out_features=out_features, in_features=in_features)


def _read_fp8_weight_split_q(readers, hf_prefix: str, num_heads: int, head_dim: int):
    """Read q_proj and split it into (w_q, w_gate_attn), mirroring
    engine_builder._load_attention_weights' split of the dequantized version.
    Both halves share the original tensor's single weight_scale/input_scale.
    """
    weight_key = f"{hf_prefix}.weight"
    scale_key = f"{hf_prefix}.weight_scale"
    if not (_has_tensor(readers, weight_key) and _has_tensor(readers, scale_key)):
        return None, None
    raw = _get_raw_tensor(readers, weight_key)
    if not hasattr(raw, "view"):
        raise TypeError(f"{weight_key} is not a torch tensor (fp8 needs a raw byte view)")
    packed = np.ascontiguousarray(raw.view(torch.uint8).numpy())  # [2*attn_size, hidden]
    hidden = packed.shape[1]
    attn_size = num_heads * head_dim
    reshaped = packed.reshape(num_heads, 2 * head_dim, hidden)
    q_part = np.ascontiguousarray(reshaped[:, :head_dim, :].reshape(attn_size, hidden))
    gate_part = np.ascontiguousarray(reshaped[:, head_dim:, :].reshape(attn_size, hidden))
    weight_scale = float(_to_numpy_fp32(_get_raw_tensor(readers, scale_key)).reshape(-1)[0])
    q_w = _FP8Weight(packed=q_part, weight_scale=weight_scale,
                      out_features=attn_size, in_features=hidden)
    gate_w = _FP8Weight(packed=gate_part, weight_scale=weight_scale,
                         out_features=attn_size, in_features=hidden)
    return q_w, gate_w


def _quantize_fp8_weight(weight_fp32: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize one (already-upcast-to-fp32) weight tensor to FP8 e4m3 with a
    single scalar scale.

    scale = max(abs(weight)) / 448 (448 is e4m3's largest finite magnitude),
    matching the exact formula families/qwen/quantization.py::_scalar() uses.
    Operates on one already-loaded (small) tensor at a time -- the caller is
    responsible for reading/discarding one projection's weight at a time so
    peak memory stays bounded (see calibrate_qwen3_8_fp8's docstring).
    """
    amax = float(np.abs(weight_fp32).max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scale = amax / 448.0
    fp8_tensor = torch.from_numpy(weight_fp32 / scale).to(torch.float8_e4m3fn)
    packed = np.ascontiguousarray(fp8_tensor.view(torch.uint8).numpy())
    return packed, scale


def _quantize_fp8_weight_split_q(
    weight_fp32: np.ndarray, num_heads: int, head_dim: int,
) -> tuple[_FP8Weight, _FP8Weight]:
    """Quantize a raw BF16 q_proj weight ([2*attn_size, hidden]) to FP8 with a
    single scalar scale shared by both halves, then split it into (w_q,
    w_gate_attn) -- mirroring `_read_fp8_weight_split_q`'s per-head interleave
    split of an already-quantized checkpoint tensor, just computing the FP8
    bytes here instead of reading them.
    """
    attn_size = num_heads * head_dim
    hidden = weight_fp32.shape[1]
    packed, weight_scale = _quantize_fp8_weight(weight_fp32)
    reshaped = packed.reshape(num_heads, 2 * head_dim, hidden)
    q_part = np.ascontiguousarray(reshaped[:, :head_dim, :].reshape(attn_size, hidden))
    gate_part = np.ascontiguousarray(reshaped[:, head_dim:, :].reshape(attn_size, hidden))
    q_w = _FP8Weight(packed=q_part, weight_scale=weight_scale,
                      out_features=attn_size, in_features=hidden)
    gate_w = _FP8Weight(packed=gate_part, weight_scale=weight_scale,
                         out_features=attn_size, in_features=hidden)
    return q_w, gate_w


def calibrate_qwen3_8_fp8(
    model_dir: Path, config, graph_ops, *, readers=None,
) -> Qwen38QuantContext:
    """Build the Q/DQ context for the real, original `Qwen/Qwen3.8-27B` BF16
    checkpoint, self-quantizing MLP (gate/up/down), attention (q/k/v/o), and
    DeltaNet (in_proj_qkv/in_proj_z/out_proj) projections to the same
    scalar-scale FP8 scheme already proven to fuse into a real Blackwell
    tensor-core GEMM (confirmed via RadixArk/Qwen3.8-27B-NVFP4's FP8
    attention/DeltaNet layers -- see this module's history/PR discussion).
    This mirrors the official `Qwen/Qwen3.8-27B-FP8` checkpoint's own choice
    of which layers to quantize (same `modules_to_not_convert` scope: norms,
    lm_head, embeddings, and DeltaNet's decay/beta/gate parameters all stay
    unquantized) -- only the numeric scheme differs, since that checkpoint's
    2D block-scale format is the one that cannot reach a real fused GEMM in
    this stack.

    Two published Qwen3.8-27B-FP8-style checkpoints were evaluated and found
    architecturally unable to reach a real fused FP8 tensor-core GEMM: the
    official checkpoint's 2D 128x128 block-scale format has no matching
    kernel path in TensorRT's public API and dense-GEMM fusion, or
    tensorrt-edge-llm's CUTLASS plugin; a per-channel-weight +
    dynamic-per-token-activation checkpoint builds and runs correctly but
    TRT's dense-FC fusion pass does not recognize that scale scheme and
    silently falls back to dequantize-to-FP16 + plain FP16 GEMM.

    Weight quantization needs no calibration -- `scale = max(abs(weight)) /
    448` is a pure function of the weight tensor itself -- so it happens here
    on the fly, one projection at a time (bounded peak memory: `readers`
    already does lazy per-tensor reads via safetensors, and each BF16
    projection tensor is a few hundred MB, not the whole ~54GB model).

    Activation `input_scale`, by contrast, genuinely requires observing real
    activation values from a forward pass -- that was calibrated once,
    offline, against this same checkpoint (real bfloat16 execution via plain
    `transformers.AutoModelForCausalLM`, forward hooks on gate/up/down,
    amax over a representative prompt set) and is read here from the small
    JSON file bundled with this family (`_FP8_ACTIVATION_SCALES_FILENAME`),
    not recomputed -- that is the one part of this whole scheme that is not
    cheap to redo on every build.

    Sets `disable_dual_gemm_fusion=True` on the returned context so
    `engine_builder.py` disables TRT's dual-GEMM auto-fusion at build
    time (see `_FP8_SELF_QUANTIZED_MLP_PROJECTIONS`'s comment above for why).
    """
    if readers is None:
        readers = _open_safetensors(Path(model_dir))
    num_layers = int(config.num_hidden_layers)
    num_heads = int(config.num_attention_heads)
    head_dim = int(config.head_dim)

    scales_path = Path(__file__).parent / _FP8_ACTIVATION_SCALES_FILENAME
    try:
        activation_scales = json.loads(scales_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Qwen3.8 FP8 self-quantization requires {scales_path} (one-time "
            "calibrated activation scales); it was not found next to "
            "quantization.py"
        ) from error

    scales: dict[str, _LayerScales] = {}

    def self_quantize_simple(layer: int, hf_stem: str, weight_stem: str) -> None:
        name = f"layer.{layer}.{weight_stem}"
        input_scale = activation_scales.get(name)
        if input_scale is None:
            return
        hf_prefix = f"model.language_model.layers.{layer}.{hf_stem}"
        weight_key = f"{hf_prefix}.weight"
        if not _has_tensor(readers, weight_key):
            return
        raw = _to_numpy_fp32(_get_raw_tensor(readers, weight_key))
        out_features, in_features = raw.shape
        packed, weight_scale = _quantize_fp8_weight(raw)
        weight = _FP8Weight(
            packed=packed, weight_scale=weight_scale,
            out_features=out_features, in_features=in_features)
        scales[name] = _LayerScales(input_scale=float(input_scale), weight=weight)

    for layer in range(num_layers):
        for hf_stem, weight_stem in _FP8_SELF_QUANTIZED_MLP_PROJECTIONS:
            self_quantize_simple(layer, hf_stem, weight_stem)

        for hf_stem, weight_stem in _FP8_ATTENTION_SIMPLE_PROJECTIONS:
            self_quantize_simple(layer, hf_stem, weight_stem)

        for hf_stem, weight_stem in _FP8_DELTANET_PROJECTIONS:
            self_quantize_simple(layer, hf_stem, weight_stem)

        # --- attention q_proj: split into (w_q, w_gate_attn), sharing one
        # calibrated input_scale and one self-quantized weight_scale ---
        q_input_scale = activation_scales.get(f"layer.{layer}.w_q")
        q_hf_prefix = f"model.language_model.layers.{layer}.self_attn.q_proj"
        q_weight_key = f"{q_hf_prefix}.weight"
        if q_input_scale is not None and _has_tensor(readers, q_weight_key):
            raw = _to_numpy_fp32(_get_raw_tensor(readers, q_weight_key))
            q_weight, gate_weight = _quantize_fp8_weight_split_q(raw, num_heads, head_dim)
            scales[f"layer.{layer}.w_q"] = _LayerScales(
                input_scale=float(q_input_scale), weight=q_weight)
            scales[f"layer.{layer}.w_gate_attn"] = _LayerScales(
                input_scale=float(q_input_scale), weight=gate_weight)

    if not scales:
        raise RuntimeError(
            "Qwen3.8 FP8 self-quantization found no matching MLP tensors in "
            "the checkpoint; is this the original Qwen/Qwen3.8-27B checkpoint?"
        )

    return Qwen38QuantContext(
        profile=_Profile(scales),
        graph_ops=graph_ops,
        disable_dual_gemm_fusion=True,
    )


def calibrate_qwen3_8_nvfp4(
    model_dir: Path, config, graph_ops, *, readers=None,
) -> Qwen38QuantContext:
    """Build the NVFP4 + FP8 Q/DQ context from the checkpoint's own quantized
    weights and calibrated activation scales.

    This checkpoint already ships a fully calibrated ModelOpt MIXED_PRECISION
    export (real packed weights + real activation scales for both schemes),
    so no forward-pass recalibration is needed -- just read the tensors
    straight from the safetensors files.

    `readers` may be a pre-opened `_ReaderCollection` (shared with
    `Qwen38Model.load_weights()` to avoid indexing the checkpoint's shards
    twice); if omitted, one is opened here.
    """
    if readers is None:
        readers = _open_safetensors(Path(model_dir))
    num_layers = int(config.num_hidden_layers)
    hidden = int(config.hidden_size)
    mlp_size = int(config.intermediate_size)
    vocab = int(config.vocab_size)
    num_heads = int(config.num_attention_heads)
    head_dim = int(config.head_dim)

    nvfp4_dims = {
        "w_gate": (mlp_size, hidden),
        "w_up": (mlp_size, hidden),
        "w_down": (hidden, mlp_size),
    }

    scales: dict[str, _LayerScales] = {}

    for layer in range(num_layers):
        # --- NVFP4: MLP (gate/up/down) ---
        for hf_stem, weight_stem in _NVFP4_MLP_PROJECTIONS:
            hf_prefix = f"model.language_model.layers.{layer}.{hf_stem}"
            input_scale = _read_input_scale(readers, hf_prefix)
            if input_scale is None:
                continue
            out_features, in_features = nvfp4_dims[weight_stem]
            weight = _read_nvfp4_weight(readers, hf_prefix, out_features, in_features)
            if weight is None:
                continue
            scales[f"layer.{layer}.{weight_stem}"] = _LayerScales(
                input_scale=input_scale, weight=weight)

        # --- FP8: DeltaNet projections ---
        for hf_stem, weight_stem in _FP8_DELTANET_PROJECTIONS:
            hf_prefix = f"model.language_model.layers.{layer}.{hf_stem}"
            input_scale = _read_input_scale(readers, hf_prefix)
            if input_scale is None:
                continue
            weight = _read_fp8_weight(readers, hf_prefix)
            if weight is None:
                continue
            scales[f"layer.{layer}.{weight_stem}"] = _LayerScales(
                input_scale=input_scale, weight=weight)

        # --- FP8: attention q_proj (split into w_q + w_gate_attn) ---
        q_hf_prefix = f"model.language_model.layers.{layer}.self_attn.q_proj"
        q_input_scale = _read_input_scale(readers, q_hf_prefix)
        if q_input_scale is not None:
            q_weight, gate_weight = _read_fp8_weight_split_q(
                readers, q_hf_prefix, num_heads, head_dim)
            if q_weight is not None:
                scales[f"layer.{layer}.w_q"] = _LayerScales(
                    input_scale=q_input_scale, weight=q_weight)
                scales[f"layer.{layer}.w_gate_attn"] = _LayerScales(
                    input_scale=q_input_scale, weight=gate_weight)

        # --- FP8: attention k_proj/v_proj/o_proj ---
        for hf_stem, weight_stem in _FP8_ATTENTION_SIMPLE_PROJECTIONS:
            hf_prefix = f"model.language_model.layers.{layer}.{hf_stem}"
            input_scale = _read_input_scale(readers, hf_prefix)
            if input_scale is None:
                continue
            weight = _read_fp8_weight(readers, hf_prefix)
            if weight is None:
                continue
            scales[f"layer.{layer}.{weight_stem}"] = _LayerScales(
                input_scale=input_scale, weight=weight)

    # --- NVFP4: lm_head ---
    lm_head_scale = _read_input_scale(readers, _NVFP4_LM_HEAD_HF_NAME)
    if lm_head_scale is not None:
        lm_head_weight = _read_nvfp4_weight(readers, _NVFP4_LM_HEAD_HF_NAME, vocab, hidden)
        if lm_head_weight is not None:
            scales[_NVFP4_LM_HEAD_WEIGHT_NAME] = _LayerScales(
                input_scale=lm_head_scale, weight=lm_head_weight)

    if not scales:
        raise RuntimeError(
            "Qwen3.8 quantization calibration found no quantized tensors in "
            "the checkpoint; is this a RadixArk/Qwen3.8-27B-NVFP4-style "
            "checkpoint?"
        )

    return Qwen38QuantContext(
        profile=_Profile(scales),
        graph_ops=graph_ops,
    )
