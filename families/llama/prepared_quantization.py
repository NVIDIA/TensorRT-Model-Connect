# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared Llama linear weights lowered through standard TensorRT Q/DQ."""

from dataclasses import dataclass, field, replace
import json
from pathlib import Path

import numpy as np
import ml_dtypes
from safetensors import safe_open



class PreparedSafetensorsReader:
    """Validated, read-only NumPy access without a training-framework dependency.

    safe_open validates the file layout first. Its NumPy backend does not expose
    FP8 on all supported NumPy versions, so map those bytes with the declared
    ml_dtypes dependency rather than modifying NumPy or unpacking whole shards.
    """

    def __init__(self, path: Path):
        self.path = path
        self.reader = safe_open(str(path), framework="numpy")
        with path.open("rb") as source:
            length = int.from_bytes(source.read(8), "little")
            self.header = json.loads(source.read(length))
        self.data_offset = 8 + length

    def keys(self):
        return self.reader.keys()

    def get_tensor(self, name: str):
        tensor = self.header[name]
        dtype = {"F8_E4M3": ml_dtypes.float8_e4m3fn, "BF16": ml_dtypes.bfloat16}.get(tensor["dtype"])
        if dtype is None:
            return self.reader.get_tensor(name)
        # Layout, shape, dtype and data bounds were already checked by safe_open.
        return np.memmap(self.path, mode="r", dtype=dtype, shape=tuple(tensor["shape"]),
                         offset=self.data_offset + tensor["data_offsets"][0])


def compressed_static_fp8(metadata: dict) -> bool:
    """Recognize only the calibrated W8A8 linear scheme used by Llama 3.2."""
    if metadata.get("quant_method") != "compressed-tensors":
        return False
    groups = metadata.get("config_groups")
    if (metadata.get("format") != "float-quantized"
            or metadata.get("quantization_status") != "compressed"
            or metadata.get("kv_cache_scheme") is not None
            or metadata.get("ignore") != ["lm_head"]
            or not isinstance(groups, dict) or len(groups) != 1):
        raise ValueError("Unsupported compressed-tensors Llama quantization contract")
    group = next(iter(groups.values()))
    if (not isinstance(group, dict) or group.get("targets") != ["Linear"]
            or group.get("output_activations") is not None):
        raise ValueError("Llama requires one FP8 group targeting Linear projections")
    for key in ("weights", "input_activations"):
        scheme = group.get(key)
        if (not isinstance(scheme, dict) or scheme.get("num_bits") != 8
                or scheme.get("type") != "float" or scheme.get("strategy") != "tensor"
                or scheme.get("symmetric") is not True or scheme.get("dynamic") is not False
                or scheme.get("group_size") is not None or scheme.get("block_structure") is not None):
            raise ValueError("Llama requires static per-tensor FP8 weights and input activations")
    return True


def check_compressed_projections(model_dir: Path, layers: int) -> None:
    """Check FP8 storage and scalar calibrations without reading the weight payloads."""
    tensors = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        reader = PreparedSafetensorsReader(path)
        for key in reader.keys():
            if key in tensors:
                raise ValueError(f"Duplicate prepared Llama tensor: {key}")
            tensors[key] = reader
    for layer in range(layers):
        for stem in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            prefix = f"model.layers.{layer}.{stem}"
            weight = prefix + ".weight"
            if weight not in tensors or tensors[weight].header[weight]["dtype"] != "F8_E4M3":
                raise ValueError(f"{weight}: expected a serialized FP8 E4M3 projection")
            for suffix in ("input_scale", "weight_scale"):
                key = f"{prefix}.{suffix}"
                if key not in tensors:
                    raise ValueError(f"Missing calibrated Llama FP8 scale: {key}")
                positive_scale(tensors[key].get_tensor(key), key)


def checkpoint_format(model_dir: Path) -> str | None:
    config = model_dir / "config.json"
    raw = json.loads(config.read_text()) if config.is_file() else {}
    embedded = raw.get("quantization_config") or {}
    sidecar = model_dir / "hf_quant_config.json"
    prepared = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    if compressed_static_fp8(embedded):
        if sidecar.exists():
            raise ValueError("Conflicting prepared Llama quantization metadata")
        check_compressed_projections(model_dir, raw["num_hidden_layers"])
        return "fp8"
    if not embedded and not prepared:
        return None
    quant = prepared.get("quantization", embedded)
    method = embedded.get("quant_method", prepared.get("producer", {}).get("name"))
    algorithm = quant.get("quant_algo")
    if method != "modelopt" or algorithm not in {"FP8", "NVFP4"}:
        raise NotImplementedError("Native Llama prepared weights require ModelOpt FP8 or NVFP4")
    if embedded.get("quant_algo", algorithm) != algorithm:
        raise ValueError("Conflicting prepared Llama quantization metadata")
    if algorithm == "NVFP4" and quant.get("group_size", 16) != 16:
        raise ValueError("Native Llama NVFP4 requires source group size16")
    return algorithm.lower()


def positive_scale(value, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().numpy()
    result = np.ascontiguousarray(value, dtype=np.float32).reshape(-1)
    if result.size != 1 or not np.isfinite(result[0]) or result[0] <= 0:
        raise ValueError(f"{name}: expected one finite positive scale")
    return result


@dataclass
class PreparedLinear:
    """Source [out,in] packed storage with an explicit logical transpose view."""

    packed: np.ndarray
    input_scale: np.ndarray
    weight_scale: np.ndarray
    out_features: int
    in_features: int
    format: str
    global_scale: np.ndarray | None = None
    transposed: bool = False
    keep_alive: list = field(default_factory=list, repr=False)

    @property
    def ndim(self):
        return 2

    @property
    def shape(self):
        return ((self.in_features, self.out_features) if self.transposed
                else (self.out_features, self.in_features))

    def transpose(self):
        if self.transposed:
            raise ValueError("Prepared linear weight was already transposed")
        return replace(self, transposed=True)

    @classmethod
    def from_tensor(cls, tensor, name, get_tensor):
        if tensor.ndim != 2 or not name.endswith(".weight"):
            raise ValueError(f"{name}: packed linear weight must be rank2")
        prefix = name.removesuffix(".weight")
        input_scale = positive_scale(get_tensor(prefix + ".input_scale"), name)
        scale = get_tensor(prefix + ".weight_scale")
        if tensor.dtype == np.dtype(ml_dtypes.float8_e4m3fn):
            return cls(np.ascontiguousarray(tensor.view(np.uint8)),
                       input_scale, positive_scale(scale, name), *tensor.shape, "fp8")
        if tensor.dtype != np.uint8:
            raise ValueError(f"{name}: unsupported prepared tensor dtype {tensor.dtype}")
        rows, columns = tensor.shape[0], tensor.shape[1] * 2
        if columns % 16 or scale.dtype != np.dtype(ml_dtypes.float8_e4m3fn) or tuple(scale.shape) != (rows, columns // 16):
            raise ValueError(f"{name}: incompatible source NVFP4 block scales")
        if not np.isfinite(scale.astype(np.float32)).all() or (scale.astype(np.float32) < 0).any():
            raise ValueError(f"{name}: invalid NVFP4 block scales")
        return cls(np.ascontiguousarray(tensor), input_scale,
                   np.ascontiguousarray(scale.view(np.uint8)), rows, columns, "nvfp4",
                   positive_scale(get_tensor(prefix + ".weight_scale_2"), name))

    def matmul(self, network, activation, lhs_width, rhs_width):
        import tensorrt as trt

        if not self.transposed or self.shape != (lhs_width, rhs_width):
            raise ValueError("Prepared Llama projection dimensions differ from the graph")
        if len(activation.shape) != 2:
            raise ValueError("Prepared Llama projection requires a rank2 activation")
        dtype = activation.dtype
        n, k = self.out_features, self.in_features

        def constant(value):
            self.keep_alive.append(value)
            return network.add_constant(tuple(value.shape), trt.Weights(value)).get_output(0)

        act_scale = constant(self.input_scale)
        if self.format == "fp8":
            weights = trt.Weights(trt.DataType.FP8, self.packed.ctypes.data, n * k)
            raw = network.add_constant((n, k), weights).get_output(0)
            weight = network.add_dequantize(raw, constant(self.weight_scale), dtype).get_output(0)
            packed_activation = network.add_quantize(activation, act_scale, trt.DataType.FP8).get_output(0)
            lhs = network.add_dequantize(packed_activation, act_scale, dtype).get_output(0)
        else:
            weights = trt.Weights(trt.DataType.FP4, self.packed.ctypes.data, n * k)
            raw = network.add_constant((n, k), weights).get_output(0)
            scale_weights = trt.Weights(trt.DataType.FP8, self.weight_scale.ctypes.data, self.weight_scale.size)
            block_scale = network.add_constant((n, k // 16), scale_weights).get_output(0)
            scale = network.add_dequantize(block_scale, constant(self.global_scale), dtype).get_output(0)
            dequant = network.add_dequantize(raw, scale, dtype)
            dequant.axis = 1
            weight = dequant.get_output(0)
            dynamic = network.add_dynamic_quantize(activation, 1, 16, trt.DataType.FP4, trt.DataType.FP8)
            dynamic.set_input(1, act_scale)
            scale = network.add_dequantize(dynamic.get_output(1), act_scale, dtype).get_output(0)
            dequant = network.add_dequantize(dynamic.get_output(0), scale, dtype)
            dequant.axis = 1
            lhs = dequant.get_output(0)
        result = network.add_matrix_multiply(
            lhs, trt.MatrixOperation.NONE, weight, trt.MatrixOperation.TRANSPOSE).get_output(0)
        return result if result.dtype == dtype else network.add_cast(result, dtype).get_output(0)


def quantize_cache_value(network, value, scale):
    """Keep source FP8 cache values in the existing native compute-type buffers."""
    import tensorrt as trt

    constant = network.add_constant((1,), trt.Weights(scale)).get_output(0)
    quantized = network.add_quantize(value, constant, trt.DataType.FP8).get_output(0)
    return network.add_dequantize(quantized, constant, value.dtype).get_output(0)


def packed_cache_rows(network, current, past, scale):
    """Export E4M3 bytes. Attention reads both old and new rows at compute precision."""
    import tensorrt as trt

    constant = network.add_constant((1,), trt.Weights(scale)).get_output(0)
    packed = network.add_quantize(current, constant, trt.DataType.FP8).get_output(0)
    return (
        packed,
        network.add_dequantize(past, constant, current.dtype).get_output(0),
        network.add_dequantize(packed, constant, current.dtype).get_output(0),
    )
