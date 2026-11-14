# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 prepared NVFP4 weights and source-calibrated dense activation Q/DQ."""

from __future__ import annotations

import json
from pathlib import Path

import ml_dtypes
import numpy as np
import tensorrt as trt


def configure_aux_streams(build_config, raw):
    """Disable auxiliary streams for prepared dense NVFP4 text graphs."""
    if ((raw.get("quantization_config") or {}).get("quant_algo") == "NVFP4"
            and not raw.get("enable_moe_block")):
        build_config.max_aux_streams = 0


def validate_source(source: Path, raw, readers):
    """Admit prepared experts or dense MLPs with original-precision attention/cache."""
    quant = raw.get("quantization_config") or {}
    sidecar = source / "hf_quant_config.json"
    if not quant and not sidecar.exists():
        if any(name.endswith((".weight_scale", ".weight_scale_2", ".input_scale",
                              ".k_scale", ".v_scale", ".q_scale")) for name in readers.tensor_map):
            raise ValueError("Gemma4 prepared tensors require explicit quantization metadata")
        return
    exported = json.loads(sidecar.read_text()).get("quantization", {}) if sidecar.is_file() else {}
    if (quant.get("quant_method") != "modelopt"
            or quant.get("quant_algo") != "NVFP4" or exported.get("quant_algo") != "NVFP4"
            or exported.get("group_size") != 16):
        raise NotImplementedError("Gemma4 native prepared weights require ModelOpt NVFP4 block16")
    dense = not raw.get("enable_moe_block")
    expected = set()
    if dense:
        layout = {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16}
        if quant.get("config_groups") != {"group_0": {
                "weights": layout, "input_activations": layout, "targets": ["Linear"]}}:
            raise NotImplementedError("Gemma4 dense NVFP4 requires the prepared W4A4 contract")
        prefix = next((p for p in ("model.language_model", "language_model.model", "model")
                       if p + ".embed_tokens.weight" in readers.tensor_map), None)
        if prefix is None:
            raise ValueError("Gemma4 dense NVFP4 target embedding is missing")
        expected = {f"{prefix}.layers.{i}.mlp.{projection}.{suffix}"
                    for i in range(raw["num_hidden_layers"])
                    for projection in ("gate_proj", "up_proj", "down_proj")
                    for suffix in ("weight_scale", "weight_scale_2", "input_scale")}
        actual = {name for name in readers.tensor_map
                  if name.endswith((".weight_scale", ".weight_scale_2", ".input_scale"))}
        if actual != expected:
            raise NotImplementedError("Gemma4 dense NVFP4 requires all and only prepared MLP projections")
    # The sidecar's FP8 label alone is not evidence of calibrated KV. The
    # admitted published source carries no cache scales; never discard any.
    for name in readers.tensor_map:
        if name.endswith((".k_scale", ".v_scale", ".q_scale")):
            raise NotImplementedError("Gemma4 native calibrated attention quantization is not mapped")
        if (not dense and name.endswith((".weight_scale", ".weight_scale_2", ".input_scale"))
                and ".experts." not in name):
            raise NotImplementedError("Gemma4 native packed weights currently map routed experts only")


def expert_tensor(graph, prefix, projection, loop):
    """Decode one loop-selected expert, keeping the source's packed storage.

    Ordinary integer operators interpret the source FP4 bits. Iterating before
    decoding avoids retaining expanded weights for all experts in a layer.
    No hardware-specific quantized kernel or activation rewrite is required.
    """
    experts = graph.raw["num_experts"]
    hidden, width = graph.raw["hidden_size"], graph.raw["moe_intermediate_size"]
    names = ("gate_proj", "up_proj") if projection == "gate_up_proj" else ("down_proj",)
    rows, columns = (width, hidden) if len(names) == 2 else (hidden, width)
    if columns % 16:
        raise ValueError("Gemma4 NVFP4 expert input width must be divisible by16")
    packed = np.empty((experts, rows * len(names), columns // 2), np.uint8)
    scales = np.empty((experts, rows * len(names), columns // 16), np.float32)

    def read(name):
        reader = graph.readers.tensor_map.get(name)
        if reader is None:
            raise ValueError(f"Gemma4 packed expert tensor is missing: {name}")
        return reader.get_tensor(name)

    for expert in range(experts):
        for section, name in enumerate(names):
            key = f"{graph.prefix}.{prefix}.experts.{expert}.{name}"
            weight = read(key + ".weight")
            block = read(key + ".weight_scale")
            global_scale = np.asarray(read(key + ".weight_scale_2"), dtype=np.float32).reshape(-1)
            if (weight.dtype != np.uint8 or weight.shape != (rows, columns // 2)
                    or block.dtype != np.dtype(ml_dtypes.float8_e4m3fn)
                    or block.shape != (rows, columns // 16)
                    or global_scale.size != 1 or not np.isfinite(global_scale[0])
                    or global_scale[0] <= 0):
                raise ValueError(f"Gemma4 packed expert geometry or scales are invalid: {key}")
            block = np.asarray(block, dtype=np.float32)
            if not np.isfinite(block).all() or (block < 0).any():
                raise ValueError(f"Gemma4 packed expert block scales are invalid: {key}")
            # Edge's Gemma4 expert path is W4A16: checkpoint input_scale does
            # not quantize activations. Preserve every source weight scale.
            packed[expert, section * rows:(section + 1) * rows] = weight
            combined = block * global_scale[0]
            if not np.isfinite(combined).all():
                raise ValueError(f"Gemma4 packed expert combined scales are not finite: {key}")
            scales[expert, section * rows:(section + 1) * rows] = combined
    # Eight FP4 values per INT32 constant: preserve both byte order and the
    # compact source size. Signed floor division below also handles bit31.
    stored = graph.const(packed.view("<i4").astype(np.int32, copy=False))
    values = loop.add_iterator(stored, 0, False).get_output(0)
    block = loop.add_iterator(graph.const(scales), 0, False).get_output(0)
    count = rows * len(names)
    divisors = graph.const(np.array([16**i for i in range(8)], np.int32).reshape(1, 1, 8))
    quotient = graph.op(graph.reshape(values, (count, columns // 8, 1)),
                         divisors, trt.ElementWiseOperation.FLOOR_DIV)
    sixteen = graph.const(np.array([[[16]]], np.int32))
    high = graph.op(quotient, sixteen, trt.ElementWiseOperation.FLOOR_DIV)
    codes = graph.op(quotient, graph.op(high, sixteen, trt.ElementWiseOperation.PROD),
                      trt.ElementWiseOperation.SUB)
    lut = graph.const(np.array([0, .5, 1, 1.5, 2, 3, 4, 6,
                                -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32))
    decoded = graph.net.add_gather(lut, graph.reshape(codes, (-1,)), 0).get_output(0)
    decoded = graph.op(graph.reshape(decoded, (count, columns // 16, 16)),
                        graph.reshape(block, (count, columns // 16, 1)),
                        trt.ElementWiseOperation.PROD)
    return graph.cast(graph.reshape(decoded, (count, columns)), graph.dtype)


def dense_linear(graph, value, name):
    """Source-packed dense W4A4, expressed only with TensorRT native Q/DQ.

    This is not the routed-expert W4A16 path. In particular, input_scale is
    the calibrated global activation scale, not a reason to bypass quantization.
    TensorRT owns lowering or reporting unsupported hardware; no SM selection
    or hardware-specific kernel is part of the mathematical graph.
    """
    stem = f"{graph.prefix}.{name}"

    def read(suffix):
        key = stem + suffix
        reader = graph.readers.tensor_map.get(key)
        if reader is None:
            raise ValueError(f"Missing Gemma4 dense NVFP4 tensor: {key}")
        return reader.get_tensor(key)

    packed = read(".weight")
    blocks = read(".weight_scale")
    if packed.ndim != 2 or packed.dtype != np.uint8:
        raise ValueError("Gemma4 dense NVFP4 requires packed rank-two uint8 weights")
    rows, columns = packed.shape[0], packed.shape[1] * 2
    if (columns % 16 or value.shape[-1] != columns
            or blocks.dtype != np.dtype(ml_dtypes.float8_e4m3fn)
            or blocks.shape != (rows, columns // 16)):
        raise ValueError("Gemma4 dense NVFP4 weight or block-scale geometry differs")
    expanded_scales = blocks.astype(np.float32)
    if not np.isfinite(expanded_scales).all() or (expanded_scales < 0).any():
        raise ValueError("Gemma4 dense NVFP4 block scales must be finite and nonnegative")

    def scalar(suffix):
        scale = np.asarray(read(suffix), dtype=np.float32).reshape(-1)
        if scale.size != 1 or not np.isfinite(scale[0]) or scale[0] <= 0:
            raise ValueError("Gemma4 dense NVFP4 global scales must be finite and positive")
        return scale

    global_scale, input_scale = scalar(".weight_scale_2"), scalar(".input_scale")
    packed = np.ascontiguousarray(packed)
    blocks = np.ascontiguousarray(blocks.view(np.uint8))
    graph.buffers.extend((packed, blocks))  # Keep native weight pointers alive.
    net, dtype = graph.net, graph.dtype
    weight = net.add_constant((rows, columns), trt.Weights(
        trt.DataType.FP4, packed.ctypes.data, rows * columns)).get_output(0)
    block = net.add_constant((rows, columns // 16), trt.Weights(
        trt.DataType.FP8, blocks.ctypes.data, blocks.size)).get_output(0)
    scale = net.add_dequantize(block, graph.const(global_scale), dtype).get_output(0)
    dequant = net.add_dequantize(weight, scale, dtype)
    dequant.axis = 1
    weight = dequant.get_output(0)
    activation_scale = graph.const(input_scale)
    quant = net.add_dynamic_quantize(graph.cast(value, dtype), 1, 16,
                                     trt.DataType.FP4, trt.DataType.FP8)
    quant.set_input(1, activation_scale)
    scale = net.add_dequantize(quant.get_output(1), activation_scale, dtype).get_output(0)
    dequant = net.add_dequantize(quant.get_output(0), scale, dtype)
    dequant.axis = 1
    activation = dequant.get_output(0)
    # Preserve the owning MLP's FP32 accumulation boundary only AFTER source
    # quantization and dequantization rounding, never instead of that Q/DQ.
    activation = graph.cast(activation, value.dtype)
    weight = graph.cast(weight, value.dtype)
    return net.add_matrix_multiply(activation, trt.MatrixOperation.NONE,
                                    weight, trt.MatrixOperation.TRANSPOSE).get_output(0)
