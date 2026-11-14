# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Llama-owned FP8 / NVFP4 / INT4-AWQ TensorRT Q/DQ graph context.

FP8 and NVFP4 read ModelOpt exports such as
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

INT4-AWQ (W4A16) quantizes the original checkpoint at build time. No public
W4A16 Llama checkpoint exists, so ``calibrate_awq_scales.py`` runs ModelOpt
``INT4_AWQ_CFG`` (``awq_lite``, no weight-clipping search) once per checkpoint
and stores only the per-projection activation smoothing ``pre_quant_scale``
under ``awq_scales/``. The build divides each weight by its smoothing scale,
takes group-128 block scales as ``amax / 7`` and rounds to signed INT4, which is
the weight ModelOpt's ``awq_lite`` fake quantization evaluates. The engine
multiplies the activation by the smoothing scale and runs
INT4 constant -> Dequantize -> MatMul with 16-bit activations.
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
INT4_AWQ_BLOCK_SIZE = 128
INT4_MAX = 7
AWQ_SCALES_DIR = Path(__file__).parent / "awq_scales"
# Small per-checkpoint tensors whose exact bytes identify the checkpoint an AWQ scale file
# belongs to; the scale file stores them as ``identity.<name>``.
AWQ_IDENTITY_TENSORS = ("model.norm.weight", "model.layers.0.input_layernorm.weight")
# Leading values of every decoder projection's first row, also stored as identity bytes, so
# a checkpoint that changes any projection (e.g. a merged fine-tune) never matches.
AWQ_PROJECTION_IDENTITY_VALUES = 64

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
_SUPPORTED = ("fp8", "nvfp4", "int4_awq")


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
    """The checkpoint's own FP8 weight, reused bit-exact."""

    packed: np.ndarray  # [out_features, in_features] raw fp8_e4m3 bytes
    weight_scale: float
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _AWQWeight:
    """INT4 weight quantized at build time from the original checkpoint."""

    packed: np.ndarray  # [out_features, in_features // 2] uint8, two codes per byte along in
    block_scale: np.ndarray  # [out_features, in_features // 128] float16
    pre_quant_scale: np.ndarray  # [in_features] float16, multiplies the activation
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _LayerScales:
    """Activation scale (None for weight-only INT4-AWQ) and the packed weight of one projection."""

    input_scale: float | None
    weight: _NVFP4Weight | _FP8Weight | _AWQWeight


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


class _NVFP4Format:
    """Builds the NVFP4 weight and dynamic-activation Q/DQ pattern."""

    @staticmethod
    def wrap_matmul(network, activation, scales: _LayerScales, *, graph_ops, keep_alive: list):
        """Add ``activation @ weight.T`` with packed FP4 weights and block-scaled FP4 activations."""
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

        return _finish_matmul(network, activation, a_dequant.get_output(0), w_dequant.get_output(0))


class _FP8Format:
    """Builds the FP8 weight and static-activation Q/DQ pattern."""

    @staticmethod
    def wrap_matmul(network, activation, scales: _LayerScales, *, graph_ops, keep_alive: list):
        """Add ``activation @ weight.T`` with FP8 weights and a static FP8 activation scale."""
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

        return _finish_matmul(network, activation, a_dequant.get_output(0), w_dequant.get_output(0))


class _AWQFormat:
    """Builds the INT4 weight-only pattern with AWQ activation smoothing."""

    @staticmethod
    def wrap_matmul(network, activation, scales: _LayerScales, *, graph_ops, keep_alive: list):
        """Add ``(activation * pre_quant_scale) @ dequantize(int4_weight).T``."""
        output_dtype = activation.dtype
        w = scales.weight
        n, k = w.out_features, w.in_features

        def half_const(values: np.ndarray, shape: tuple[int, ...]):
            const = graph_ops.add_constant(network, shape, values, dtype=np.float16)
            if const.dtype == output_dtype:
                return const
            return network.add_cast(const, output_dtype).get_output(0)

        keep_alive.append(w.packed)
        w_int4 = trt.Weights(trt.DataType.INT4, w.packed.ctypes.data, n * k)
        w_const = network.add_constant((n, k), w_int4).get_output(0)
        w_dequant = network.add_dequantize(
            w_const, half_const(w.block_scale, w.block_scale.shape), output_dtype
        )
        w_dequant.axis = 1

        rank = len(tuple(activation.shape))
        smooth = half_const(w.pre_quant_scale, (1,) * (rank - 1) + (k,))
        smoothed = network.add_elementwise(
            activation, smooth, trt.ElementWiseOperation.PROD
        ).get_output(0)
        return _finish_matmul(network, activation, smoothed, w_dequant.get_output(0))


@dataclass(frozen=True)
class LlamaQuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    graph_ops: Any
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        """Create the list that keeps weight buffers alive during the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, weight_name: str) -> bool:
        """True when ``weight_name`` (for example ``layer.3.w_q``) is a quantized projection."""
        return weight_name in self.scales

    @property
    def has_nvfp4_weights(self) -> bool:
        """Whether any projection uses packed NVFP4 weights."""
        return any(isinstance(entry.weight, _NVFP4Weight) for entry in self.scales.values())

    @property
    def disable_dual_gemm_fusion(self) -> bool:
        """True for FP8: TensorRT's fused gate/up (dual-GEMM) kernel fails to compile
        with scalar-scale FP8 Q/DQ inputs ('arith.divf' type mismatch), so the builders
        turn that fusion off. NVFP4 fuses correctly and keeps it."""
        return any(isinstance(s.weight, _FP8Weight) for s in self.scales.values())

    @property
    def builder_optimization_level(self) -> int:
        """3 for INT4-AWQ, 1 otherwise. At level 1 TensorRT 11.4 dequantizes every INT4
        weight to 16 bits in a separate kernel on each step instead of fusing the
        dequantization into the GEMM, which makes decode slower than fp16."""
        return 3 if any(isinstance(s.weight, _AWQWeight) for s in self.scales.values()) else 1

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
        if isinstance(entry.weight, _AWQWeight):
            fmt = _AWQFormat
        elif isinstance(entry.weight, _NVFP4Weight):
            fmt = _NVFP4Format
        else:
            fmt = _FP8Format
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


def _read_fp8(tensors, prefix: str) -> _FP8Weight | None:
    """The FP8 weight under ``prefix``, or None when it is not a float8_e4m3 tensor."""
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


def quantize_awq_weight(weight: np.ndarray, pre_quant_scale: np.ndarray) -> _AWQWeight:
    """Quantize one ``[out, in]`` weight to group-128 signed INT4 after AWQ smoothing.

    The engine multiplies the activation by ``pre_quant_scale``, so the weight is
    divided by it first. Block scales are ``amax / 7`` rounded to float16, the
    precision the engine dequantizes with, and the codes are rounded with that
    same scale.
    """
    weight = np.asarray(weight, dtype=np.float32)
    out_features, in_features = weight.shape
    if in_features % INT4_AWQ_BLOCK_SIZE:
        raise ValueError(f"INT4-AWQ needs in_features divisible by {INT4_AWQ_BLOCK_SIZE}")
    smooth = np.asarray(pre_quant_scale, dtype=np.float16).reshape(-1)
    if smooth.shape != (in_features,) or not np.isfinite(smooth).all() or np.any(smooth <= 0):
        raise ValueError(f"pre_quant_scale must be {in_features} positive finite values")
    blocks = (weight / smooth.astype(np.float32)).reshape(
        out_features, in_features // INT4_AWQ_BLOCK_SIZE, INT4_AWQ_BLOCK_SIZE
    )
    block_scale = (np.abs(blocks).max(axis=2) / INT4_MAX).astype(np.float16)
    # An all-zero block quantizes to zero codes under any scale.
    block_scale[block_scale == 0] = 1.0
    codes = np.clip(np.rint(blocks / block_scale[..., None].astype(np.float32)), -8, 7)
    nibbles = codes.astype(np.int8).reshape(out_features, in_features).view(np.uint8) & 15
    packed = np.ascontiguousarray(nibbles[:, 0::2] | (nibbles[:, 1::2] << 4))
    return _AWQWeight(
        packed, np.ascontiguousarray(block_scale), np.ascontiguousarray(smooth),
        out_features, in_features,
    )


def awq_identity(tensors: dict[str, Any]) -> dict[str, np.ndarray]:
    """Raw bytes that identify an AWQ checkpoint, keyed as in the scale file: the norm tensors
    in full and the leading values of every decoder projection."""
    identity = {}
    for key in AWQ_IDENTITY_TENSORS:
        if key not in tensors:
            raise ValueError(f"INT4-AWQ checkpoint identity needs {key}")
        identity[f"identity.{key}"] = _raw_bytes(tensors, key).reshape(-1)
    suffixes = tuple(f".{stem}.weight" for stem, _ in _PROJECTIONS)
    for key in sorted(k for k in tensors if k.startswith("model.layers.") and k.endswith(suffixes)):
        head = tensors[key].get_slice(key)[0:1, :AWQ_PROJECTION_IDENTITY_VALUES]
        identity[f"identity.{key}"] = np.ascontiguousarray(
            head.contiguous().view(torch.uint8).numpy().reshape(-1)
        )
    return identity


def _load_awq_scales(scales_dir: Path, identity: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The pre_quant_scale table calibrated for the checkpoint with these identity tensors."""
    for path in sorted(Path(scales_dir).glob("*.npz")):
        with np.load(path, allow_pickle=False) as table:
            if all(
                key in table.files and np.array_equal(table[key], value)
                for key, value in identity.items()
            ):
                return {name: table[name] for name in table.files if name.startswith("layer.")}
    raise ValueError(
        f"no INT4-AWQ scales in {scales_dir} match this checkpoint; "
        "run python -m families.llama.calibrate_awq_scales"
    )


def _quantize_llama_awq(model_dir: Path, config, graph_ops, scales_dir: Path) -> LlamaQuantContext:
    """Quantize every decoder projection of the original checkpoint to INT4-AWQ."""
    if (
        config.raw.get("quantization_config")
        or (Path(model_dir) / "hf_quant_config.json").is_file()
    ):
        raise ValueError(
            f"llama int4_awq quantizes the original 16-bit checkpoint; {model_dir} is already "
            "quantized (quantization_config or hf_quant_config.json)"
        )
    tensors = _open_raw_readers(Path(model_dir))
    table = _load_awq_scales(scales_dir, awq_identity(tensors))
    scales: dict[str, _LayerScales] = {}
    for layer in range(int(config.num_hidden_layers)):
        for hf_stem, weight_stem in _PROJECTIONS:
            name = f"layer.{layer}.{weight_stem}"
            key = f"model.layers.{layer}.{hf_stem}.weight"
            if name not in table:
                raise ValueError(
                    f"INT4-AWQ scales have no {name}; "
                    "run python -m families.llama.calibrate_awq_scales"
                )
            if key not in tensors:
                raise ValueError(f"{key} is missing from the checkpoint")
            source = tensors[key].get_tensor(key)
            if not source.is_floating_point() or source.element_size() < 2:
                raise ValueError(f"INT4-AWQ needs the original 16/32-bit {key}, not {source.dtype}")
            weight = quantize_awq_weight(source.float().numpy(), table[name])
            scales[name] = _LayerScales(None, weight)
    return LlamaQuantContext(scales=scales, graph_ops=graph_ops)


def calibrate_llama(
    model_dir: Path,
    config,
    graph_ops,
    quantization: str,
    *,
    awq_scales_dir: Path = AWQ_SCALES_DIR,
) -> LlamaQuantContext:
    """Build the Q/DQ context for a Llama checkpoint.

    FP8 and NVFP4 checkpoints already carry calibrated weights and activation
    scales, so no forward pass is needed. INT4-AWQ quantizes the original
    weights with the committed smoothing scales of the matching checkpoint.
    Raises when the checkpoint does not match the requested format, instead of
    silently falling back to unquantized weights.
    """
    if quantization not in _SUPPORTED:
        raise NotImplementedError(f"llama does not support quantization={quantization!r}")
    if quantization == "int4_awq":
        return _quantize_llama_awq(Path(model_dir), config, graph_ops, Path(awq_scales_dir))
    tensors = _open_raw_readers(Path(model_dir))
    read_weight = _read_nvfp4 if quantization == "nvfp4" else _read_fp8

    scales: dict[str, _LayerScales] = {}
    for layer in range(int(config.num_hidden_layers)):
        for hf_stem, weight_stem in _PROJECTIONS:
            prefix = f"model.layers.{layer}.{hf_stem}"
            scale_key = f"{prefix}.input_scale"
            weight = read_weight(tensors, prefix)
            if weight is None or scale_key not in tensors:
                raise ValueError(
                    f"{prefix} is not a {quantization} ModelOpt tensor; "
                    f"llama {quantization} builds need a checkpoint such as "
                    f"nvidia/Llama-3.1-8B-Instruct-{quantization.upper()}"
                )
            input_scale = _scalar(tensors, scale_key)
            if input_scale is None:
                raise ValueError(f"{scale_key} must be one positive finite value")
            scales[f"layer.{layer}.{weight_stem}"] = _LayerScales(input_scale, weight)
    return LlamaQuantContext(scales=scales, graph_ops=graph_ops)
