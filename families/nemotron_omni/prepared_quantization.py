# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron Omni prepared weights: source packing and scales, native TensorRT Q/DQ."""

from dataclasses import dataclass, field, replace
import fnmatch
import json
from pathlib import Path

import ml_dtypes
import numpy as np


@dataclass(frozen=True)
class PreparedPolicy:
    algorithm: str
    layers: dict
    excluded: tuple[str, ...]

    @property
    def request_format(self):
        return "fp8" if self.algorithm == "FP8" else "nvfp4"

    def linear_format(self, name):
        module = name.removesuffix(".weight")
        if any(fnmatch.fnmatchcase(module, pattern) for pattern in self.excluded):
            return None
        algorithm = self.algorithm
        if algorithm == "MIXED_PRECISION":
            entry = self.layers.get(module)
            if entry is None:
                return None
            algorithm = entry["quant_algo"]
        return algorithm.lower()


def checkpoint_policy(
    model_dir: Path, *, allow_legacy_producer: bool = False
) -> PreparedPolicy | None:
    path = model_dir / "config.json"
    raw = json.loads(path.read_text()) if path.is_file() else {}
    embedded = raw.get("quantization_config") or {}
    sidecar = model_dir / "hf_quant_config.json"
    prepared = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    if not embedded and not prepared:
        if any(
            (model_dir / name).exists() for name in ("quant_config.json", "quantize_config.json")
        ):
            raise ValueError("Nemotron Omni requires ModelOpt prepared quantization metadata")
        return None
    if embedded.get("quant_method", "modelopt") != "modelopt" or (
        prepared
        and prepared.get("producer", {}).get("name") != "modelopt"
        and not (allow_legacy_producer and "producer" not in prepared)
    ):
        raise ValueError("Nemotron Omni prepared weights require ModelOpt metadata")
    quant = prepared.get("quantization", embedded)
    if not isinstance(quant, dict):
        raise ValueError("Invalid Nemotron Omni prepared quantization metadata")
    algorithm = quant.get("quant_algo")
    supported = {"FP8", "NVFP4", "W4A16_NVFP4", "MIXED_PRECISION"}
    if algorithm not in supported or embedded.get("quant_algo", algorithm) != algorithm:
        raise ValueError("Unsupported or conflicting Nemotron Omni quantization algorithm")
    layers = quant.get("quantized_layers", {})
    if algorithm == "MIXED_PRECISION":
        if not isinstance(layers, dict) or not layers:
            raise ValueError("Mixed Nemotron Omni quantization requires per-layer policies")
        policies = list(layers.values())
    else:
        policies = [quant]
    for policy in policies:
        algo = policy.get("quant_algo") if isinstance(policy, dict) else None
        if algo not in supported - {"MIXED_PRECISION"}:
            raise ValueError("Unsupported prepared Nemotron Omni linear format")
        allowed = {1, 16} if algo == "FP8" else {16}
        if policy.get("group_size", 1 if algo == "FP8" else 16) not in allowed:
            raise ValueError("Unsupported prepared Nemotron Omni group size")
    excluded = quant.get("exclude_modules", quant.get("ignore", []))
    if not isinstance(excluded, list) or any(not isinstance(x, str) for x in excluded):
        raise ValueError("Invalid Nemotron Omni quantization exclusions")
    if embedded and prepared:
        for key in ("group_size", "kv_cache_quant_algo", "quantized_layers"):
            if key in embedded and key in quant and embedded[key] != quant[key]:
                raise ValueError(f"Conflicting Nemotron Omni quantization {key}")
    return PreparedPolicy(algorithm, layers, tuple(excluded))


def positive_scale(value, name):
    value = np.ascontiguousarray(value, dtype=np.float32).reshape(-1)
    if value.size != 1 or not np.isfinite(value[0]) or value[0] <= 0:
        raise ValueError(f"{name}: expected one finite positive scale")
    return value


@dataclass
class PreparedLinear:
    """Packed source [out,in], never materialized as an unquantized matrix."""

    packed: np.ndarray
    weight_scale: np.ndarray
    out_features: int
    in_features: int
    format: str
    input_scale: np.ndarray | None = None
    global_scale: np.ndarray | None = None
    transposed: bool = False
    keep_alive: list = field(default_factory=list, repr=False)

    @property
    def ndim(self):
        return 2

    @property
    def shape(self):
        shape = (self.out_features, self.in_features)
        return shape[::-1] if self.transposed else shape

    def transpose(self):
        if self.transposed:
            raise ValueError("Prepared Nemotron Omni weight was already transposed")
        return replace(self, transposed=True, keep_alive=[])

    @classmethod
    def from_tensor(cls, tensor, name, get_tensor, format):
        if tensor.ndim != 2 or not name.endswith(".weight"):
            raise ValueError(f"{name}: prepared linear must have rank2")
        prefix = name.removesuffix(".weight")
        scale = get_tensor(prefix + ".weight_scale")
        input_scale = (
            None
            if format == "w4a16_nvfp4"
            else positive_scale(get_tensor(prefix + ".input_scale"), name)
        )
        if format == "fp8":
            if tensor.dtype != np.dtype(ml_dtypes.float8_e4m3fn):
                raise ValueError(f"{name}: FP8 requires E4M3 source weights")
            return cls(
                np.ascontiguousarray(tensor.view(np.uint8)),
                positive_scale(scale, name),
                *tensor.shape,
                format,
                input_scale,
            )
        if format not in {"nvfp4", "w4a16_nvfp4"} or tensor.dtype != np.uint8:
            raise ValueError(f"{name}: incompatible prepared source format {format}")
        rows, columns = tensor.shape[0], tensor.shape[1] * 2
        if (
            columns % 16
            or scale.dtype != np.dtype(ml_dtypes.float8_e4m3fn)
            or scale.shape != (rows, columns // 16)
        ):
            raise ValueError(f"{name}: expected E4M3 block16 scales for packed FP4 weights")
        if not np.isfinite(scale.astype(np.float32)).all() or (scale.astype(np.float32) < 0).any():
            raise ValueError(f"{name}: invalid NVFP4 block scales")
        return cls(
            np.ascontiguousarray(tensor),
            np.ascontiguousarray(scale.view(np.uint8)),
            rows,
            columns,
            format,
            input_scale,
            positive_scale(get_tensor(prefix + ".weight_scale_2"), name),
        )

    def matmul(self, network, activation, lhs_width, rhs_width):
        import tensorrt as trt

        if not self.transposed or self.shape != (lhs_width, rhs_width):
            raise ValueError("Prepared Nemotron Omni weight dimensions differ from graph")
        # Every decoder/expert invocation consumes one supplied embedding.
        dtype = activation.dtype
        n, k = self.out_features, self.in_features

        def constant(value):
            self.keep_alive.append(value)
            return network.add_constant(tuple(value.shape), trt.Weights(value)).get_output(0)

        if self.format == "fp8":
            raw = network.add_constant(
                (n, k), trt.Weights(trt.DataType.FP8, self.packed.ctypes.data, n * k)
            ).get_output(0)
            weight = network.add_dequantize(raw, constant(self.weight_scale), dtype).get_output(0)
            scale = constant(self.input_scale)
            # Conditional expert outputs acquire their rank during graph build.
            # Construct rank-two Q/DQ first, then wire the real activation; the
            # temporary constant is disconnected from the completed network.
            seed = constant(
                np.zeros((1, lhs_width), dtype=np.float16 if dtype == trt.float16 else np.float32)
            )
            quantize = network.add_quantize(seed, scale, trt.DataType.FP8)
            lhs = network.add_dequantize(quantize.get_output(0), scale, dtype).get_output(0)
            quantize.set_input(0, activation)
        else:
            raw = network.add_constant(
                (n, k), trt.Weights(trt.DataType.FP4, self.packed.ctypes.data, n * k)
            ).get_output(0)
            blocks = network.add_constant(
                (n, k // 16),
                trt.Weights(
                    trt.DataType.FP8, self.weight_scale.ctypes.data, self.weight_scale.size
                ),
            ).get_output(0)
            scale = network.add_dequantize(blocks, constant(self.global_scale), dtype).get_output(0)
            dequant = network.add_dequantize(raw, scale, dtype)
            dequant.axis = 1
            weight = dequant.get_output(0)
            lhs = activation
            if self.format == "nvfp4":
                global_scale = constant(self.input_scale)
                # As with FP8 above, defer conditional-output binding until
                # rank-sensitive Q/DQ construction is complete. The seed is
                # disconnected; the source activation and scales are unchanged.
                seed = constant(
                    np.zeros(
                        (1, lhs_width), dtype=np.float16 if dtype == trt.float16 else np.float32
                    )
                )
                packed = network.add_dynamic_quantize(
                    seed, 1, 16, trt.DataType.FP4, trt.DataType.FP8
                )
                packed.set_input(1, global_scale)
                scale = network.add_dequantize(
                    packed.get_output(1), global_scale, dtype
                ).get_output(0)
                dequant = network.add_dequantize(packed.get_output(0), scale, dtype)
                dequant.axis = 1
                lhs = dequant.get_output(0)
                packed.set_input(0, activation)
        # Match the owning unquantized graph accumulation boundary. Keep the
        # source Q/DQ rounding and packed constants; only accumulate in FP32.
        from .graph_ops import _add_matrix_multiply_with_fp32_accumulation

        return _add_matrix_multiply_with_fp32_accumulation(
            network, lhs, trt.MatrixOperation.NONE, weight, trt.MatrixOperation.TRANSPOSE
        )


class PreparedQuantContext:
    def maybe_quantized_matmul(
        self, network, lhs, lhs_width, rhs_width, weight, weight_name, *, dtype
    ):
        from . import graph_ops

        if isinstance(weight, PreparedLinear):
            return weight.matmul(network, lhs, lhs_width, rhs_width)
        return graph_ops.add_matmul_rhs_constant(
            network, lhs, lhs_width, rhs_width, weight, dtype=dtype, rank=2
        )
