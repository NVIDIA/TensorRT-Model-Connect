# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned prepared linear weights; no calibration or precision substitution."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path

import numpy as np
import tensorrt as trt


@dataclass(frozen=True)
class PreparedLinear:
    """Original packed output-major weights and their calibrated scale tensors."""

    format: str
    packed: np.ndarray
    input_scale: float
    weight_scale: float | np.ndarray
    block_scale: np.ndarray | None = None
    smooth_scale: np.ndarray | None = None
    keep_alive: list = field(default_factory=list, repr=False, compare=False)

    @property
    def shape(self) -> tuple[int, int]:
        multiplier = 2 if self.format in {"nvfp4", "int4_awq"} else 1
        return self.packed.shape[1] * multiplier, self.packed.shape[0]

    def select_outputs(self, indices: np.ndarray) -> PreparedLinear:
        """Split interleaved query/gate rows without decoding or requantizing."""
        return replace(self, packed=np.ascontiguousarray(self.packed[indices]),
                       weight_scale=(np.ascontiguousarray(self.weight_scale[indices])
                                     if self.format == "int8_sq" else self.weight_scale),
                       block_scale=(None if self.block_scale is None else
                                    np.ascontiguousarray(self.block_scale[indices])))

    def matmul(self, network, activation, lhs_width: int, rhs_width: int,
               *, fp32_accumulation: bool = False):
        if self.shape != (lhs_width, rhs_width):
            raise ValueError(f"Prepared projection shape {self.shape} != {(lhs_width, rhs_width)}")
        if len(activation.shape) != 2:
            raise ValueError("Qwen3.5 prepared projections require rank-two activations")
        dtype = activation.dtype
        if dtype != trt.float16:
            raise ValueError("Prepared Qwen3.5 projections currently require FP16 compute")

        def multiply(lhs, rhs):
            # Block drafts require the source's FP32 dot-product accumulation
            # and explicit FP16 output rounding. Keep the calibrated Q/DQ
            # tensors intact; this is not a dense-weight or activation bypass.
            if fp32_accumulation:
                lhs = network.add_cast(lhs, trt.float32).get_output(0)
                rhs = network.add_cast(rhs, trt.float32).get_output(0)
            output = network.add_matrix_multiply(
                lhs, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.TRANSPOSE
            ).get_output(0)
            if fp32_accumulation:
                output = network.add_cast(output, dtype).get_output(0)
            return output

        def scalar(value):
            values = np.asarray([value], dtype=np.float32)
            self.keep_alive.append(values)
            return network.add_constant((1,), values).get_output(0)

        n, k = rhs_width, lhs_width
        if self.format == "mxfp8":
            raw = trt.Weights(trt.fp8, self.packed.ctypes.data, n * k)
            packed = network.add_constant((n, k), raw).get_output(0)
            scale_weights = trt.Weights(trt.DataType.E8M0, self.block_scale.ctypes.data,
                                       self.block_scale.size)
            scales = network.add_constant(self.block_scale.shape, scale_weights).get_output(0)
            weight = network.add_dequantize(packed, scales, dtype)
            quant = network.add_dynamic_quantize(activation, 1, 32, trt.fp8, trt.DataType.E8M0)
            inputs = network.add_dequantize(quant.get_output(0), quant.get_output(1), dtype)
            weight.axis = inputs.axis = 1
            return multiply(inputs.get_output(0), weight.get_output(0))
        if self.format == "int8_sq":
            smooth = network.add_constant((1, k), self.smooth_scale).get_output(0)
            activation = network.add_elementwise(activation, smooth, trt.ElementWiseOperation.PROD).get_output(0)
            # Source SQ saturates symmetrically at -127, not INT8's -128.
            # Clip in FP32 so the calibrated endpoint is not rounded to FP16.
            full = network.add_cast(activation, trt.float32).get_output(0)
            clipped = network.add_activation(full, trt.ActivationType.CLIP)
            clipped.alpha = -127 * self.input_scale
            clipped.beta = 127 * self.input_scale
            a_scale = scalar(self.input_scale)
            quant = network.add_quantize(clipped.get_output(0), a_scale, trt.int8)
            inputs = network.add_dequantize(quant.get_output(0), a_scale, dtype).get_output(0)
            packed = network.add_constant((n, k), self.packed).get_output(0)
            scales = network.add_constant((n,), self.weight_scale).get_output(0)
            weight = network.add_dequantize(packed, scales, dtype)
            weight.axis = 0
            return multiply(inputs, weight.get_output(0))
        if self.format == "int4_awq":
            # ModelOpt smooths activations in FP16; there is no activation
            # quantizer in this W4A16 path. TensorRT owns INT4 lowering.
            smooth = network.add_constant((1, k), self.smooth_scale).get_output(0)
            activation = network.add_elementwise(activation, smooth, trt.ElementWiseOperation.PROD).get_output(0)
            raw = trt.Weights(trt.DataType.INT4, self.packed.ctypes.data, n * k)
            packed = network.add_constant((n, k), raw).get_output(0)
            scales = network.add_constant(self.block_scale.shape, self.block_scale).get_output(0)
            weight = network.add_dequantize(packed, scales, dtype)
            weight.axis = 1
            return multiply(activation, weight.get_output(0))
        weight_type = trt.DataType.FP4 if self.format == "nvfp4" else trt.DataType.FP8
        # The weights object owns all backing arrays for the complete build.
        raw = trt.Weights(weight_type, self.packed.ctypes.data, n * k)
        packed = network.add_constant((n, k), raw).get_output(0)
        w_scale = scalar(self.weight_scale)
        a_scale = scalar(self.input_scale)
        if self.format == "nvfp4":
            raw_scale = trt.Weights(trt.DataType.FP8, self.block_scale.ctypes.data,
                                   self.block_scale.size)
            block = network.add_constant((n, k // 16), raw_scale).get_output(0)
            w_scale = network.add_dequantize(block, w_scale, dtype).get_output(0)
            quant = network.add_dynamic_quantize(activation, 1, 16,
                                                 trt.DataType.FP4, trt.DataType.FP8)
            quant.set_input(1, a_scale)
            # Source dynamic quantization uses FP32 scale, but activation
            # reconstruction first rounds the global scale to FP16.
            reconstruction_scale = network.add_cast(a_scale, dtype).get_output(0)
            a_scale = network.add_dequantize(quant.get_output(1), reconstruction_scale, dtype).get_output(0)
        else:
            quant = network.add_quantize(activation, a_scale, trt.DataType.FP8)
        weight = network.add_dequantize(packed, w_scale, dtype)
        inputs = network.add_dequantize(quant.get_output(0), a_scale, dtype)
        if self.format == "nvfp4":
            weight.axis = inputs.axis = 1
        return multiply(inputs.get_output(0), weight.get_output(0))


def prepared_format(model_dir: Path, raw: dict, requested: str | None, *,
                    allow_sidecar_only: bool = False) -> str:
    """Validate prepared metadata independently of optional Edge installation."""
    text = raw.get("text_config", raw)
    embedded = raw.get("quantization_config", text.get("quantization_config"))
    sidecar_path = model_dir / "hf_quant_config.json"
    document = json.loads(sidecar_path.read_text()) if sidecar_path.is_file() else {}
    sidecar = document.get("quantization", {})
    # ModelOpt DFlash exports keep quantization metadata in the sidecar only.
    # Targets retain the stricter embedded-metadata contract by default.
    if allow_sidecar_only and embedded is None and sidecar:
        producer = document.get("producer", {})
        if not isinstance(producer, dict) or producer.get("name") != "modelopt":
            raise ValueError("Native Qwen3.5 draft sidecar requires explicit ModelOpt producer")
        embedded = {"quant_method": "modelopt", "quant_algo": sidecar.get("quant_algo")}
    if not embedded and not sidecar:
        if requested not in (None, "none"):
            raise ValueError("Native Qwen3.5 quantization requires an already-prepared checkpoint")
        return "none"
    if not isinstance(embedded, dict) or embedded.get("quant_method") != "modelopt":
        raise ValueError("Native Qwen3.5 requires explicit ModelOpt prepared metadata")
    algorithm = embedded.get("quant_algo")
    if sidecar and sidecar.get("quant_algo") != algorithm:
        raise ValueError("Embedded and sidecar quantization algorithms disagree")
    formats = {"NVFP4": "nvfp4", "FP8": "fp8", "W4A16_AWQ": "int4_awq",
               "W8A8_SQ_PER_CHANNEL": "int8_sq", "INT8_SQ": "int8_sq",
               "MXFP8": "mxfp8", "FP8_PB_WO": "mxfp8"}
    if algorithm not in formats:
        raise NotImplementedError(f"Native Qwen3.5 prepared format {algorithm!r} is not implemented yet")
    result = formats[algorithm]
    if requested not in (None, result):
        raise ValueError("Requested quantization differs from the prepared checkpoint")
    scheme = embedded.get("kv_cache_scheme")
    if scheme and (not isinstance(scheme, dict) or scheme.get("num_bits") != 8
                   or scheme.get("type") != "float" or scheme.get("dynamic") is not False):
        raise ValueError("Native Qwen3.5 requires static E4M3 KV-cache quantization")
    if sidecar.get("kv_cache_quant_algo") not in (None, "FP8"):
        raise ValueError("Native Qwen3.5 requires FP8 KV-cache quantization")
    # Exported LLM K-scale tensors decide cache storage, not the sidecar:
    # ModelOpt also sets this field when only visual attention is quantized.
    return result


def load_projection(readers, prefix: str):
    """Read source exclusions densely and quantized projections byte-exactly."""
    from .checkpoint_mapper import _transpose_2d

    def raw(name):
        reader = readers.tensor_map.get(name)
        if reader is None:
            raise KeyError(f"Missing prepared projection tensor {name}")
        return np.asarray(reader.get_tensor(name))

    def scale(name):
        value = raw(name).astype(np.float32).reshape(-1)
        if value.size != 1 or not np.isfinite(value[0]) or value[0] <= 0:
            raise ValueError(f"{name} must contain one finite positive scale")
        return float(value[0])

    key = prefix + ".weight"
    if prefix + ".weight_scale" not in readers.tensor_map:
        value = raw(key)
        if value.dtype.kind not in "f" and str(value.dtype) != "bfloat16":
            raise ValueError(f"{key}: packed weight is missing its calibrated scales")
        return _transpose_2d(value, prefix)
    value = raw(key)
    if value.ndim != 2:
        raise ValueError(f"{key}: prepared weight must be rank two")
    scales = raw(prefix + ".weight_scale")
    if str(value.dtype) == "float8_e4m3fn" and scales.dtype == np.uint8:
        n, k = value.shape
        if k % 32 or scales.shape != (n, k // 32) or np.any(scales == 255):
            raise ValueError(f"{prefix}: invalid MXFP8 E8M0 block scales")
        if prefix + ".pre_quant_scale" in readers.tensor_map:
            raise ValueError(f"{prefix}: MXFP8 does not define activation smoothing")
        return PreparedLinear("mxfp8", np.ascontiguousarray(value.view(np.uint8)), 1.0, 1.0,
                              np.ascontiguousarray(scales))
    if value.dtype == np.int8:
        n, k = value.shape
        scales = raw(prefix + ".weight_scale").astype(np.float32).reshape(-1)
        if scales.size != n or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError(f"{prefix}: invalid per-output-channel SmoothQuant scales")
        smooth_key = prefix + ".pre_quant_scale"
        smooth = raw(smooth_key).astype(np.float16).reshape(-1) if smooth_key in readers.tensor_map else np.ones(k, np.float16)
        if smooth.size != k or not np.isfinite(smooth).all() or np.any(smooth <= 0):
            raise ValueError(f"{prefix}: invalid SmoothQuant activation smoothing scales")
        return PreparedLinear("int8_sq", np.ascontiguousarray(value), scale(prefix + ".input_scale"),
                              np.ascontiguousarray(scales), smooth_scale=np.ascontiguousarray(smooth.reshape(1, k)))
    if value.dtype == np.uint8 and prefix + ".weight_scale_2" not in readers.tensor_map:
        # ModelOpt packs output rows into low/even and high/odd nibbles.
        # TensorRT's constant packs adjacent input columns. Reorder bits only,
        # retaining two's-complement INT4 codes without dense dequantization.
        n_half, k = value.shape
        n = 2 * n_half
        scales = raw(prefix + ".weight_scale").astype(np.float16)
        if (k % 2 or scales.ndim != 2 or scales.shape[0] != n
                or scales.shape[1] == 0 or k % scales.shape[1]
                or k // scales.shape[1] not in (64, 128)):
            raise ValueError(f"{prefix}: malformed ModelOpt AWQ block geometry")
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError(f"{prefix}: AWQ scales must be finite and positive")
        codes = np.empty((n, k), dtype=np.uint8)
        codes[0::2] = value & 15
        codes[1::2] = value >> 4
        packed = np.ascontiguousarray(codes[:, 0::2] | (codes[:, 1::2] << 4))
        smooth_key = prefix + ".pre_quant_scale"
        smooth = raw(smooth_key).astype(np.float16).reshape(-1) if smooth_key in readers.tensor_map else np.ones(k, np.float16)
        if smooth.size != k or not np.isfinite(smooth).all() or np.any(smooth <= 0):
            raise ValueError(f"{prefix}: invalid AWQ activation smoothing scales")
        return PreparedLinear("int4_awq", packed, 1.0, 1.0,
                              np.ascontiguousarray(scales), np.ascontiguousarray(smooth.reshape(1, k)))
    if prefix + ".pre_quant_scale" in readers.tensor_map:
        raise NotImplementedError("Prepared smoothing must be explicitly implemented, not ignored")
    input_scale = scale(prefix + ".input_scale")
    if prefix + ".weight_scale_2" in readers.tensor_map:
        blocks = raw(prefix + ".weight_scale")
        if value.dtype != np.uint8 or str(blocks.dtype) != "float8_e4m3fn":
            raise ValueError(f"{prefix}: NVFP4 requires packed U8 weights and E4M3 block scales")
        n, packed_k = value.shape
        if packed_k % 8 or blocks.shape != (n, packed_k // 8):
            raise ValueError(f"{prefix}: malformed NVFP4 block geometry")
        block_values = blocks.astype(np.float32)
        if not np.isfinite(block_values).all() or np.any(block_values < 0):
            raise ValueError(f"{prefix}: NVFP4 block scales must be finite and nonnegative")
        return PreparedLinear("nvfp4", np.ascontiguousarray(value), input_scale,
                              scale(prefix + ".weight_scale_2"),
                              np.ascontiguousarray(blocks.view(np.uint8)))
    if str(value.dtype) != "float8_e4m3fn":
        raise ValueError(f"{prefix}: FP8 requires original E4M3 packed weights")
    return PreparedLinear("fp8", np.ascontiguousarray(value.view(np.uint8)), input_scale,
                          scale(prefix + ".weight_scale"))


def load_fp8_kv_scales(readers, layer_types: list[str]) -> dict:
    """Read exported LLM scales, matching Edge's checkpoint-based detection."""
    prefixes = {
        f"layer.{i}": f"model.language_model.layers.{i}.self_attn"
        for i, kind in enumerate(layer_types) if kind == "attention"
    }
    if not any(f"{prefix}.k_proj.k_scale" in readers.tensor_map
               for prefix in prefixes.values()):
        return {"_fp8_kv_cache": False}
    result = {"_fp8_kv_cache": True}
    for name, prefix in prefixes.items():
        scales = []
        for projection in ("k", "v"):
            key = f"{prefix}.{projection}_proj.{projection}_scale"
            reader = readers.tensor_map.get(key)
            # Edge initializes individual absent scales to one once enabled.
            value = np.asarray(reader.get_tensor(key)) if reader is not None else np.array(1.0)
            if value.size != 1 or not np.isfinite(value).all() or (value <= 0).any():
                raise ValueError(f"Invalid Qwen3.5 FP8 cache scale: {key}")
            scales.append(float(value.item()))
        result[f"{name}.kv_scales"] = tuple(scales)
    return result


def prepare_fp8_cache(network, current, cache, scale_value: float):
    """Store post-RoPE E4M3 bytes and attend to calibrated current/past values."""
    if cache.dtype != trt.fp8:
        raise ValueError("Qwen3.5 calibrated KV cache requires FP8 storage inputs")
    scale = network.add_constant((1,), np.asarray([scale_value], dtype=np.float32)).get_output(0)
    packed = network.add_quantize(current, scale, trt.fp8).get_output(0)
    decoded_current = network.add_dequantize(packed, scale, current.dtype).get_output(0)
    decoded_cache = network.add_dequantize(cache, scale, current.dtype).get_output(0)
    return packed, decoded_current, decoded_cache
