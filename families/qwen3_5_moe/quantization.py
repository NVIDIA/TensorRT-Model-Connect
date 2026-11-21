# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5-MoE packed projections expressed with TensorRT standard operators."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


def _positive_scalar(value, name: str) -> float:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != 1 or not np.isfinite(array[0]) or array[0] <= 0:
        raise ValueError(f"{name} requires one finite positive scale")
    return float(array[0])


@dataclass(frozen=True)
class PackedLinear:
    """Output-major original codes and scales, retained for the full TRT build.

    GPTQ packing changes bits/layout only. NVFP4 experts are W4A16: their
    input_scale metadata must not accidentally enable W4A4 activation rounding.
    FP8 attention projections do retain their calibrated activation quantizer.
    """

    format: str
    packed: np.ndarray
    scales: np.ndarray
    global_scale: float = 1.0
    input_scale: float | None = None
    keep_alive: list = field(default_factory=list, compare=False, repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        """Return logical input/output dimensions, without expanding weights."""
        factor = 1 if self.format == "fp8" else 2
        return self.packed.shape[1] * factor, self.packed.shape[0]

    def select_outputs(self, indices: np.ndarray) -> PackedLinear:
        """Split interleaved query/gate outputs without changing source codes."""
        return replace(self, packed=np.ascontiguousarray(self.packed[indices]),
                       scales=(self.scales if self.format == "fp8" else
                               np.ascontiguousarray(self.scales[indices])), keep_alive=[])

    def weight_tensor(self, network, dtype):
        """Reconstruct source weights in graph; TensorRT owns lowering/storage."""
        import tensorrt as trt

        k, n = self.shape
        kind = {"gptq": trt.DataType.INT4, "nvfp4_a16": trt.DataType.FP4,
                "fp8": trt.DataType.FP8}[self.format]
        raw = trt.Weights(kind, self.packed.ctypes.data, n * k)
        weight = network.add_constant((n, k), raw).get_output(0)
        self.keep_alive.extend((self.packed, self.scales, raw))
        if self.format == "nvfp4_a16":
            raw_scales = trt.Weights(trt.DataType.FP8, self.scales.ctypes.data, self.scales.size)
            blocks = network.add_constant(self.scales.shape, raw_scales).get_output(0)
            global_array = np.asarray([self.global_scale], dtype=np.float32)
            global_scale = network.add_constant((1,), global_array).get_output(0)
            self.keep_alive.extend((raw_scales, global_array))
            scales = network.add_dequantize(blocks, global_scale, trt.float32).get_output(0)
            decoded = network.add_dequantize(weight, scales, trt.float32)
            decoded.axis = 1
            # Do not round the intermediate FP8-block/global product to FP16.
            return network.add_cast(decoded.get_output(0), dtype).get_output(0)
        scales = network.add_constant(self.scales.shape, self.scales).get_output(0)
        decoded = network.add_dequantize(weight, scales, dtype)
        if self.format == "gptq":
            decoded.axis = 1
        return decoded.get_output(0)

    def matmul(self, network, activation, input_width: int, output_width: int,
               *, fp32_accumulation: bool = False):
        """Apply source projection; never quantize W4A16 activations."""
        import tensorrt as trt

        if self.shape != (input_width, output_width):
            raise ValueError("Packed projection dimensions disagree with the model graph")
        if self.format == "fp8":
            scale_array = np.asarray([self.input_scale], dtype=np.float32)
            self.keep_alive.append(scale_array)
            scale = network.add_constant((1,), scale_array).get_output(0)
            quantized = network.add_quantize(activation, scale, trt.DataType.FP8).get_output(0)
            activation = network.add_dequantize(quantized, scale, activation.dtype).get_output(0)
        weight = self.weight_tensor(network, activation.dtype)
        dtype = activation.dtype
        if fp32_accumulation:
            activation = network.add_cast(activation, trt.float32).get_output(0)
            weight = network.add_cast(weight, trt.float32).get_output(0)
        result = network.add_matrix_multiply(
            activation, trt.MatrixOperation.NONE, weight, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        return network.add_cast(result, dtype).get_output(0) if fp32_accumulation else result


def gptq_projection(qweight, qzeros, scales, group_index) -> PackedLinear:
    """Preserve symmetric group128 GPTQ v1/v2 without clipping or calibration."""
    qweight, qzeros = np.asarray(qweight), np.asarray(qzeros)
    scales, group_index = np.asarray(scales), np.asarray(group_index)
    if qweight.dtype != np.int32 or qweight.ndim != 2:
        raise ValueError("GPTQ qweight requires rank-two packed INT32")
    k, n = qweight.shape[0] * 8, qweight.shape[1]
    if k % 128 or n % 8 or scales.shape != (k // 128, n) or scales.dtype != np.float16:
        raise ValueError("GPTQ requires group128 and original FP16 scales")
    if not np.isfinite(scales).all() or np.any(scales < 0):
        raise ValueError("GPTQ scales must be finite and nonnegative")
    if (group_index.dtype != np.int32 or group_index.shape != (k,)
            or not np.array_equal(group_index, np.arange(k) // 128)):
        raise ValueError("GPTQ act-order is not part of this checkpoint contract")
    if qzeros.dtype != np.int32 or qzeros.shape != (k // 128, n // 8):
        raise ValueError("GPTQ qzeros geometry differs from group128")
    # GPTQ v1 stores zero-1; v2 stores zero. Both symmetric formats decode at8.
    zero_words = qzeros.view(np.uint32)
    if not (np.all(zero_words == np.uint32(0x77777777))
            or np.all(zero_words == np.uint32(0x88888888))):
        raise ValueError("Symmetric GPTQ requires an unambiguous zero point of8")
    # A little-endian word already contains four adjacent packed input pairs.
    # Transpose words, then flip each nibble sign bit instead of expanding INT4.
    words = np.array(qweight.T, dtype="<i4", copy=True, order="C").view("<u4")
    words ^= np.uint32(0x88888888)
    packed = words.view(np.uint8).reshape(n, k // 2)
    return PackedLinear("gptq", packed, np.ascontiguousarray(scales.T))


def nvfp4_projection(packed, block_scales, global_scale) -> PackedLinear:
    """Retain E2M1 bytes and original E4M3 block/global scales for W4A16."""
    packed, block_scales = np.asarray(packed), np.asarray(block_scales)
    if packed.dtype != np.uint8 or packed.ndim != 2:
        raise ValueError("NVFP4 requires rank-two packed UINT8")
    n, half = packed.shape
    if half % 8 or block_scales.shape != (n, half // 8):
        raise ValueError("NVFP4 block scales require group16 geometry")
    if str(block_scales.dtype) != "float8_e4m3fn":
        raise ValueError("NVFP4 requires original E4M3 block-scale bytes")
    values = block_scales.astype(np.float32)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("NVFP4 block scales must be finite and nonnegative")
    return PackedLinear("nvfp4_a16", np.ascontiguousarray(packed),
                        np.ascontiguousarray(block_scales.view(np.uint8)),
                        _positive_scalar(global_scale, "NVFP4 global"))


def fp8_projection(packed, weight_scale, input_scale) -> PackedLinear:
    """Retain E4M3 weight bytes and explicit calibrated W8A8 scales."""
    packed = np.asarray(packed)
    if str(packed.dtype) != "float8_e4m3fn" or packed.ndim != 2:
        raise ValueError("FP8 requires rank-two original E4M3 weight bytes")
    return PackedLinear("fp8", np.ascontiguousarray(packed.view(np.uint8)),
                        np.asarray([_positive_scalar(weight_scale, "FP8 weight")], np.float32),
                        input_scale=_positive_scalar(input_scale, "FP8 activation"))


def prepared_format(model_dir, raw: dict, requested: str | None, *,
                    allow_sidecar_only: bool = False) -> str:
    """Use the same exact source policy for offload and native construction."""
    from pathlib import Path
    from types import SimpleNamespace
    from .edge_llm.precision import source_policy

    root = Path(model_dir)
    text = raw.get("text_config", raw)
    embedded = raw.get("quantization_config", text.get("quantization_config"))
    sidecars = any((root / name).exists() for name in
                   ("hf_quant_config.json", "quantize_config.json", "quant_config.json"))
    if not embedded and not sidecars:
        if requested not in (None, "none"):
            raise ValueError("Native quantization requires an already-prepared checkpoint")
        return "none"
    if allow_sidecar_only:
        raise ValueError("Qwen3.5-MoE requires its original unquantized DFlash draft")
    return source_policy(SimpleNamespace(model_dir=root, quantization=requested), raw)["weights"]


def load_projection(readers, prefix: str):
    """Read source codes/scales, or transpose an original excluded projection."""
    def raw(suffix):
        key = prefix + suffix
        reader = readers.tensor_map.get(key)
        if reader is None:
            raise KeyError(f"Missing projection tensor {key}")
        return np.asarray(reader.get_tensor(key))

    if prefix + ".qweight" in readers.tensor_map:
        return gptq_projection(raw(".qweight"), raw(".qzeros"), raw(".scales"), raw(".g_idx"))
    value = raw(".weight")
    if value.ndim != 2:
        raise ValueError(f"{prefix}: projection weight must be rank two")
    if value.dtype == np.uint8:
        return nvfp4_projection(value, raw(".weight_scale"), raw(".weight_scale_2"))
    if str(value.dtype) == "float8_e4m3fn":
        return fp8_projection(value, raw(".weight_scale"), raw(".input_scale"))
    if value.dtype not in (np.dtype("float16"), np.dtype("float32")) and str(value.dtype) != "bfloat16":
        raise ValueError(f"{prefix}: unsupported source projection dtype {value.dtype}")
    if prefix + ".weight_scale" in readers.tensor_map:
        raise ValueError(f"{prefix}: original floating projection has unexpected quantization scales")
    # Original BF16/FP16 exclusions are consumed as FP16 by the admitted target.
    # Retain FP32 inputs for the independent mathematical test fixtures.
    dtype = np.float32 if value.dtype == np.float32 else np.float16
    return np.ascontiguousarray(value.T, dtype=dtype)


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
        result[f"{name}.kv_scales"] = load_attention_kv_scales(readers, prefix)
    return result


def load_attention_kv_scales(readers, prefix: str) -> tuple[float, float]:
    """Read one attention layer's calibrated scales, including excluded MTP heads."""
    scales = []
    for projection in ("k", "v"):
        key = f"{prefix}.{projection}_proj.{projection}_scale"
        reader = readers.tensor_map.get(key)
        # Edge initializes individual absent scales to one once FP8 KV is enabled.
        value = np.asarray(reader.get_tensor(key)) if reader is not None else np.array(1.0)
        if value.size != 1 or not np.isfinite(value).all() or (value <= 0).any():
            raise ValueError(f"Invalid Qwen3.5-MoE FP8 cache scale: {key}")
        scales.append(float(value.item()))
    return scales[0], scales[1]


def prepare_fp8_cache(network, current, cache, scale_value: float):
    """Store post-RoPE E4M3 bytes and attend to calibrated current/past values."""
    import tensorrt as trt

    if cache.dtype != trt.fp8:
        raise ValueError("Qwen3.5-MoE calibrated KV cache requires FP8 storage inputs")
    scale = network.add_constant((1,), np.asarray([scale_value], dtype=np.float32)).get_output(0)
    packed = network.add_quantize(current, scale, trt.fp8).get_output(0)
    decoded_current = network.add_dequantize(packed, scale, current.dtype).get_output(0)
    decoded_cache = network.add_dequantize(cache, scale, current.dtype).get_output(0)
    return packed, decoded_current, decoded_cache
