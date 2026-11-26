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

    GPTQ packing changes bits/layout only. The mixed 35B recipe uses W4A16;
    only the ordinary NVFP4 recipe rounds activations to E2M1.
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

    def _lookup_packed_weights(self, network, bits: int, table: np.ndarray):
        """Decode original checkpoint codes, low bits first, using standard TRT ops."""
        import tensorrt as trt

        k, n = self.shape
        lanes = 32 // bits
        # INT8/FP8 constants would be interpreted as quantized weights; carry
        # the unchanged source bytes in INT32 words instead. Padding is per row.
        extra = (-self.packed.shape[1]) % 4
        data = np.pad(self.packed, ((0, 0), (0, extra))) if extra else self.packed
        words = np.ascontiguousarray(data).view("<i4").reshape(n, -1, 1)
        divisors = (1 << (bits * np.arange(lanes, dtype=np.int32))).reshape(1, 1, lanes)
        radix = np.full((1, 1, 1), 1 << bits, np.int32)
        self.keep_alive.extend((words, divisors, radix, table))

        def constant(array):
            return network.add_constant(array.shape, array).get_output(0)

        def binary(left, right, op):
            return network.add_elementwise(left, right, op).get_output(0)

        op = trt.ElementWiseOperation
        shifted = binary(constant(words), constant(divisors), op.FLOOR_DIV)
        base = constant(radix)
        # Euclidean modulo also extracts the high code of negative signed words.
        indices = binary(shifted, binary(binary(shifted, base, op.FLOOR_DIV), base, op.PROD), op.SUB)
        values = network.add_gather(constant(table), indices, 0).get_output(0)
        view = network.add_shuffle(values)
        view.reshape_dims = (n, words.shape[1] * lanes)
        weight = view.get_output(0)
        return network.add_slice(weight, (0, 0), (n, k), (1, 1)).get_output(0) if extra else weight

    def weight_tensor(self, network, dtype):
        """Reconstruct source weights in graph; TensorRT owns lowering/storage."""
        import tensorrt as trt

        k, n = self.shape
        if self.format == "fp8":
            import ml_dtypes

            values = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn)
            table = values.astype(np.float32) * self.scales[0]
            return network.add_cast(self._lookup_packed_weights(network, 8, table), dtype).get_output(0)
        if self.format in {"nvfp4_a16", "nvfp4_a4"}:
            import ml_dtypes

            positive = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)
            values = self._lookup_packed_weights(network, 4, np.concatenate((positive, -positive)))
            blocks = network.add_shuffle(values)
            blocks.reshape_dims = (n, k // 16, 16)
            # The serialized FP8 scale is shared by sixteen E2M1 weights.
            # Combine it with the per-projection scale before any FP16 cast.
            scales = self.scales.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
            scales = (scales * np.float32(self.global_scale)).reshape(n, k // 16, 1)
            self.keep_alive.append(scales)
            scale = network.add_constant(scales.shape, scales).get_output(0)
            weight = network.add_elementwise(blocks.get_output(0), scale, trt.ElementWiseOperation.PROD)
            view = network.add_shuffle(weight.get_output(0))
            view.reshape_dims = (n, k)
            return network.add_cast(view.get_output(0), dtype).get_output(0)
        if self.format != "gptq":
            raise ValueError(f"Unknown packed projection format: {self.format}")
        raw = trt.Weights(trt.DataType.INT4, self.packed.ctypes.data, n * k)
        weight = network.add_constant((n, k), raw).get_output(0)
        self.keep_alive.extend((self.packed, self.scales, raw))
        scales = network.add_constant(self.scales.shape, self.scales).get_output(0)
        decoded = network.add_dequantize(weight, scales, dtype)
        decoded.axis = 1
        return decoded.get_output(0)

    def _round_e4m3_magnitude(self, network, magnitude, scale):
        """Nearest-even finite E4M3 code; compare calibrated midpoints before division."""
        import tensorrt as trt

        rank = len(magnitude.shape)
        e, u = trt.ElementWiseOperation, trt.UnaryOperation

        def constant(value):
            array = np.full((1,) * rank, value, np.float32)
            self.keep_alive.append(array)
            return network.add_constant(array.shape, array).get_output(0)

        def binary(left, right, op):
            return network.add_elementwise(left, right, op).get_output(0)

        def unary(value, op):
            return network.add_unary(value, op).get_output(0)

        raw = binary(magnitude, scale, e.DIV)
        raw = binary(raw, constant(448), e.MIN)
        normal = binary(raw, constant(2 ** -6), e.MAX)
        exponent = unary(binary(unary(normal, u.LOG), constant(np.log(2)), e.DIV), u.FLOOR)
        step = binary(binary(constant(2), exponent, e.POW), constant(0.125), e.PROD)
        lower = unary(binary(raw, step, e.DIV), u.FLOOR)
        midpoint = binary(binary(binary(lower, constant(0.5), e.SUM), step, e.PROD), scale, e.PROD)
        half = binary(lower, constant(0.5), e.PROD)
        odd = binary(half, unary(half, u.FLOOR), e.GREATER)
        up = binary(binary(magnitude, midpoint, e.GREATER),
                    binary(odd, binary(magnitude, midpoint, e.EQUAL), e.AND), e.OR)
        code = binary(lower, network.add_cast(up, trt.float32).get_output(0), e.SUM)
        return binary(binary(code, step, e.PROD), constant(448), e.MIN)

    def round_nvfp4_input(self, network, activation):
        """Apply this projection's original per-16 E2M1 / E4M3 activation scales."""
        import tensorrt as trt

        if self.format != "nvfp4_a4" or self.input_scale is None:
            raise ValueError("NVFP4 activation quantization requires the source W4A4 scale")
        shape = network.add_shape(activation).get_output(0)
        width = self.shape[0]
        blocks = network.add_shuffle(activation)
        rank = len(activation.shape)
        # Preserve the independent token and routed-expert axes; only split K.
        blocks.reshape_dims = (0,) * (rank - 1) + (width // 16, 16)
        work = network.add_cast(blocks.get_output(0), trt.float32).get_output(0)
        e, u = trt.ElementWiseOperation, trt.UnaryOperation

        def constant(value):
            array = np.full((1,) * (rank + 1), value, np.float32)
            self.keep_alive.append(array)
            return network.add_constant(array.shape, array).get_output(0)

        def binary(left, right, op):
            return network.add_elementwise(left, right, op).get_output(0)

        def unary(value, op):
            return network.add_unary(value, op).get_output(0)

        # Reconstruct the ModelOpt block scale in FP32. E4M3FN has three
        # mantissa bits, minimum normal exponent -6 and maximum finite 448.
        # Floating point graph math keeps this route valid without FP8 hardware.
        magnitude = unary(work, u.ABS)
        maximum = network.add_reduce(magnitude, trt.ReduceOperation.MAX, 1 << rank, True).get_output(0)
        global_scale = constant(self.input_scale)
        six_global = constant(np.float32(6) * np.float32(self.input_scale))
        block_scale = self._round_e4m3_magnitude(network, maximum, six_global)
        effective = binary(block_scale, global_scale, e.PROD)
        # Compare in the original scale domain. Division can be lowered to an
        # approximate reciprocal, moving exactly representable FP4 ties by one ulp.
        levels = constant(0)
        for midpoint, upper, upper_even in ((0.25, 0.5, False), (0.75, 1, True),
                                             (1.25, 1.5, False), (1.75, 2, True),
                                             (2.5, 3, False), (3.5, 4, True), (5, 6, False)):
            threshold = binary(effective, constant(midpoint), e.PROD)
            select_upper = (unary(binary(magnitude, threshold, e.LESS), u.NOT) if upper_even
                            else binary(magnitude, threshold, e.GREATER))
            levels = network.add_select(select_upper, constant(upper), levels).get_output(0)
        levels = binary(levels, unary(work, u.SIGN), e.PROD)
        # Match the effective-scale DQ of the requested compute type.
        scale = network.add_cast(effective, activation.dtype).get_output(0)
        result = binary(network.add_cast(levels, activation.dtype).get_output(0), scale, e.PROD)
        restored = network.add_shuffle(result)
        restored.set_input(1, shape)
        return restored.get_output(0)

    def round_fp8_input(self, network, activation):
        """Quantize and dequantize using the checkpoint's per-tensor E4M3 input scale."""
        import tensorrt as trt

        if self.format != "fp8" or self.input_scale is None:
            raise ValueError("FP8 activation quantization requires its calibrated input scale")
        scale_array = np.full((1,) * len(activation.shape), self.input_scale, np.float32)
        self.keep_alive.append(scale_array)
        scale = network.add_constant(scale_array.shape, scale_array).get_output(0)
        value = network.add_cast(activation, trt.float32).get_output(0)
        mag = network.add_unary(value, trt.UnaryOperation.ABS).get_output(0)
        code = self._round_e4m3_magnitude(network, mag, scale)
        sign = network.add_unary(value, trt.UnaryOperation.SIGN).get_output(0)
        v = network.add_elementwise(code, sign, trt.ElementWiseOperation.PROD).get_output(0)
        v = network.add_elementwise(v, scale, trt.ElementWiseOperation.PROD).get_output(0)
        return network.add_cast(v, activation.dtype).get_output(0)

    def matmul(self, network, activation, input_width: int, output_width: int,
               *, fp32_accumulation: bool = False):
        """Apply source projection; never quantize W4A16 activations."""
        import tensorrt as trt

        if self.shape != (input_width, output_width):
            raise ValueError("Packed projection dimensions disagree with the model graph")
        if self.format == "nvfp4_a4":
            activation = self.round_nvfp4_input(network, activation)
        if self.format == "fp8":
            activation = self.round_fp8_input(network, activation)
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


def nvfp4_projection(packed, block_scales, global_scale, *, input_scale=None) -> PackedLinear:
    """Retain packed E2M1 weights; only an explicit W4A4 recipe supplies input_scale."""
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
    return PackedLinear("nvfp4_a16" if input_scale is None else "nvfp4_a4", np.ascontiguousarray(packed),
                        np.ascontiguousarray(block_scales.view(np.uint8)),
                        _positive_scalar(global_scale, "NVFP4 global"),
                        None if input_scale is None else _positive_scalar(input_scale, "NVFP4 activation"))


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


def load_projection(readers, prefix: str, *, quantize_nvfp4_input: bool = False):
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
        return nvfp4_projection(value, raw(".weight_scale"), raw(".weight_scale_2"),
                                input_scale=raw(".input_scale") if quantize_nvfp4_input else None)
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
