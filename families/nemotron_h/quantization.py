# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-H-owned FP8 / NVFP4 TensorRT Q/DQ graph context for ModelOpt checkpoints.

Supported checkpoints are ModelOpt exports such as
``nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8`` and ``nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4``.
The Mamba-2 ``in_proj`` / ``out_proj`` and the MLP ``up_proj`` / ``down_proj`` projections
carry:

  FP8    ``<name>.weight`` (float8_e4m3), ``<name>.weight_scale`` (per-tensor
         fp32) and ``<name>.input_scale`` (per-tensor fp32).
  NVFP4  ``<name>.weight`` (E2M1, two values per uint8), ``<name>.weight_scale``
         (per-16-element float8_e4m3), ``<name>.weight_scale_2`` (global fp32)
         and ``<name>.input_scale`` (fp32).

The attention projections, ``conv1d``, the norms, the embedding and ``lm_head`` stay
unquantized. The NVFP4 checkpoint also keeps the Mamba and MLP projections of the first
two and last two layers unquantized. The checkpoint's ``hf_quant_config.json`` lists these
in ``exclude_modules``. A projection without scales that is not listed there is an error.

The checkpoint's own quantized bytes are fed to the engine as-is (no
dequantize/re-quantize round trip). TensorRT's ``add_dequantize(value, scale)``
computes ``value * scale``, which reproduces the exported weight exactly, and
the Quantize/DynamicQuantize -> Dequantize -> MatMul pattern is fused into a
native FP8 or FP4 tensor-core GEMM. Weights keep the checkpoint's
``[out_features, in_features]`` layout, so the matmul transposes the weight
operand instead of repacking sub-byte data.

Activations use ``add_quantize`` for FP8 and ``add_dynamic_quantize`` for NVFP4
(blockwise ``add_quantize`` to an FP4 output is rejected by the shape checker).
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

# HF projection stem -> internal weight-dict stem, per layer kind.
_MAMBA_PROJECTIONS = (
    ("mixer.in_proj", "mamba_in_proj"),
    ("mixer.out_proj", "mamba_out_proj"),
)
_MLP_PROJECTIONS = (
    ("mixer.up_proj", "w_up"),
    ("mixer.down_proj", "w_down"),
)
_ATTENTION_PROJECTIONS = (
    ("mixer.q_proj", "w_q"),
    ("mixer.k_proj", "w_k"),
    ("mixer.v_proj", "w_v"),
    ("mixer.o_proj", "w_o"),
)
_PROJECTIONS_BY_KIND = {
    "mamba2": _MAMBA_PROJECTIONS,
    "mlp": _MLP_PROJECTIONS,
    "attention": _ATTENTION_PROJECTIONS,
}
_SUPPORTED = ("fp8", "nvfp4")


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
        """Add ``activation @ weight.T`` with FP8 weights and a static FP8 activation scale.

        The Dequantize layers and the matmul run in float32 and the result is cast back.
        With fp16 Dequantize outputs, TensorRT's FP8 GEMM keeps the raw accumulator in
        fp16 and applies the scales afterwards, so it overflows to NaN as soon as an
        output exceeds about ``65504 * input_scale * weight_scale`` (around 150 for a
        typical projection). Large outputs are normal here, for example the BOS token
        in the Mamba ``out_proj`` of layer 6.
        """
        w = scales.weight
        n, k = w.out_features, w.in_features

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


@dataclass(frozen=True)
class NemotronHQuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    graph_ops: Any
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        """Create the list that keeps weight buffers alive during the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, weight_name: str) -> bool:
        """True when ``weight_name`` (for example ``layer.3.w_up``) is a quantized projection."""
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
        fmt = _NVFP4Format if isinstance(entry.weight, _NVFP4Weight) else _FP8Format
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


def is_quantized_checkpoint(model_dir: Path) -> bool:
    """True when the checkpoint stores quantized projections (it has ``weight_scale`` tensors)."""
    try:
        tensors = _open_raw_readers(Path(model_dir))
    except FileNotFoundError:
        return False
    return any(name.endswith(".weight_scale") for name in tensors)


def _excluded_modules(model_dir: Path) -> frozenset[str]:
    """The modules ModelOpt left unquantized, from ``hf_quant_config.json``."""
    path = Path(model_dir) / "hf_quant_config.json"
    if not path.is_file():
        return frozenset()
    quant = json.loads(path.read_text(encoding="utf-8")).get("quantization", {})
    return frozenset(quant.get("exclude_modules", ()))


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


def calibrate_nemotron_h(
    model_dir: Path,
    graph_ops,
    quantization: str,
    *,
    layer_types: list[str],
) -> NemotronHQuantContext:
    """Build the Q/DQ context from a ModelOpt FP8 or NVFP4 Nemotron-H checkpoint.

    ``layer_types`` lists ``"mamba2"``, ``"mlp"`` or ``"attention"`` per layer. The
    checkpoint already carries calibrated weights and activation scales, so no forward
    pass is needed. Raises when the checkpoint does not match the requested format,
    instead of silently falling back to unquantized weights.
    """
    if quantization not in _SUPPORTED:
        raise NotImplementedError(f"nemotron_h does not support quantization={quantization!r}")
    model_dir = Path(model_dir)
    tensors = _open_raw_readers(model_dir)
    excluded = _excluded_modules(model_dir)
    read_weight = _read_nvfp4 if quantization == "nvfp4" else _read_fp8

    scales: dict[str, _LayerScales] = {}
    for layer, kind in enumerate(layer_types):
        for hf_stem, weight_stem in _PROJECTIONS_BY_KIND[kind]:
            prefix = f"backbone.layers.{layer}.{hf_stem}"
            if f"{prefix}.weight_scale" not in tensors:
                # ModelOpt left this projection unquantized. That is only valid when
                # the checkpoint says so, and never for a plain full-precision file.
                if prefix in excluded:
                    continue
                raise ValueError(
                    f"{prefix} has no {quantization} scales and is not listed in "
                    f"exclude_modules; nemotron_h {quantization} builds need a ModelOpt "
                    f"checkpoint such as nvidia/NVIDIA-Nemotron-Nano-9B-v2-{quantization.upper()}"
                )
            scale_key = f"{prefix}.input_scale"
            weight = read_weight(tensors, prefix)
            if weight is None or scale_key not in tensors:
                raise ValueError(
                    f"{prefix} is not a {quantization} ModelOpt tensor; "
                    f"nemotron_h {quantization} builds need a checkpoint such as "
                    f"nvidia/NVIDIA-Nemotron-Nano-9B-v2-{quantization.upper()}"
                )
            input_scale = _scalar(tensors, scale_key)
            if input_scale is None:
                raise ValueError(f"{scale_key} must be one positive finite value")
            scales[f"layer.{layer}.{weight_stem}"] = _LayerScales(input_scale, weight)
    if not scales:
        raise ValueError(f"no {quantization} projections found in {model_dir}")
    return NemotronHQuantContext(scales=scales, graph_ops=graph_ops)
