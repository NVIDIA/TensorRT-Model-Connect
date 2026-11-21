# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared ModelOpt linear weights lowered through standard TensorRT Q/DQ."""

from dataclasses import dataclass, field, replace
import json
from pathlib import Path

import numpy as np
import ml_dtypes


def checkpoint_format(model_dir: Path) -> str | None:
    config = model_dir / "config.json"
    raw = json.loads(config.read_text()) if config.is_file() else {}
    embedded = raw.get("quantization_config") or {}
    sidecar = model_dir / "hf_quant_config.json"
    prepared = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    if not embedded and not prepared:
        return None
    if prepared and not isinstance(prepared.get("quantization"), dict):
        raise ValueError("Invalid Phi4 prepared quantization metadata")
    quant = prepared.get("quantization", embedded)
    method = embedded.get("quant_method", prepared.get("producer", {}).get("name"))
    algorithm = quant.get("quant_algo")
    if method != "modelopt" or algorithm not in {"FP8", "NVFP4", "W4A16_AWQ"}:
        raise NotImplementedError(
            "Native Phi4 prepared weights require ModelOpt FP8, NVFP4 or W4A16_AWQ"
        )
    if embedded.get("quant_algo", algorithm) != algorithm:
        raise ValueError("Conflicting prepared Phi4 quantization metadata")
    if algorithm == "NVFP4" and quant.get("group_size", 16) != 16:
        raise ValueError("Native Phi4 NVFP4 requires source group size16")
    expected_group = {"FP8": None, "NVFP4": 16, "W4A16_AWQ": 128}[algorithm]
    for metadata in (quant, embedded):
        group = metadata.get("group_size", expected_group)
        if group != expected_group and not (algorithm == "FP8" and group == 0):
            raise ValueError("Conflicting Phi4 prepared group size")
        if metadata.get("kv_cache_quant_algo") not in (None, "NONE"):
            raise NotImplementedError("Phi4 native prepared KV quantization is not implemented")
    return "int4_awq" if algorithm == "W4A16_AWQ" else algorithm.lower()


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
    input_scale: np.ndarray | None
    weight_scale: np.ndarray
    out_features: int
    in_features: int
    format: str
    global_scale: np.ndarray | None = None
    transposed: bool = False
    pre_quant_scale: np.ndarray | None = None
    keep_alive: list = field(default_factory=list, repr=False)

    @property
    def ndim(self):
        return 2

    @property
    def shape(self):
        return (
            (self.in_features, self.out_features)
            if self.transposed
            else (self.out_features, self.in_features)
        )

    def transpose(self):
        if self.transposed:
            raise ValueError("Prepared linear weight was already transposed")
        return replace(self, transposed=True)

    @classmethod
    def from_tensor(cls, tensor, name, get_tensor, format, pre_quant_scale=None):
        if tensor.ndim != 2 or not name.endswith(".weight"):
            raise ValueError(f"{name}: packed linear weight must be rank2")
        prefix = name.removesuffix(".weight")
        scale = get_tensor(prefix + ".weight_scale")
        if format == "int4_awq":
            if tensor.dtype != np.uint8 or tensor.ndim != 2:
                raise ValueError(f"{name}: AWQ requires packed uint8 [out/2,in] weights")
            rows, columns = tensor.shape[0] * 2, tensor.shape[1]
            scale = np.asarray(scale, dtype=np.float32)
            if columns % 128 or scale.shape != (rows, columns // 128):
                raise ValueError(f"{name}: AWQ requires per-row input-group128 scales")
            if not np.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError(f"{name}: invalid AWQ scales")
            # ModelOpt HF exports pairs of OUTPUT channels in a byte. TensorRT
            # INT4 constants pack the contiguous INPUT channels, low nibble first.
            nibbles = np.empty((rows, columns), dtype=np.uint8)
            nibbles[0::2] = tensor & 15
            nibbles[1::2] = tensor >> 4
            packed = np.ascontiguousarray(nibbles[:, 0::2] | (nibbles[:, 1::2] << 4))
            if pre_quant_scale is not None:
                pre_quant_scale = np.asarray(pre_quant_scale, dtype=np.float32).reshape(-1)
                if (
                    pre_quant_scale.shape != (columns,)
                    or not np.isfinite(pre_quant_scale).all()
                    or (pre_quant_scale <= 0).any()
                ):
                    raise ValueError(f"{name}: invalid AWQ input smoothing scales")
                pre_quant_scale = np.ascontiguousarray(pre_quant_scale)
            return cls(
                packed,
                None,
                np.ascontiguousarray(scale),
                rows,
                columns,
                format,
                pre_quant_scale=pre_quant_scale,
            )
        input_scale = positive_scale(get_tensor(prefix + ".input_scale"), name)
        if format == "fp8" and tensor.dtype == np.dtype(ml_dtypes.float8_e4m3fn):
            return cls(
                np.ascontiguousarray(tensor.view(np.uint8)),
                input_scale,
                positive_scale(scale, name),
                *tensor.shape,
                "fp8",
            )
        if format != "nvfp4" or tensor.dtype != np.uint8:
            raise ValueError(f"{name}: unsupported prepared tensor dtype {tensor.dtype}")
        rows, columns = tensor.shape[0], tensor.shape[1] * 2
        if (
            columns % 16
            or scale.dtype != np.dtype(ml_dtypes.float8_e4m3fn)
            or tuple(scale.shape) != (rows, columns // 16)
        ):
            raise ValueError(f"{name}: incompatible source NVFP4 block scales")
        if not np.isfinite(scale.astype(np.float32)).all() or (scale.astype(np.float32) < 0).any():
            raise ValueError(f"{name}: invalid NVFP4 block scales")
        return cls(
            np.ascontiguousarray(tensor),
            input_scale,
            np.ascontiguousarray(scale.view(np.uint8)),
            rows,
            columns,
            "nvfp4",
            positive_scale(get_tensor(prefix + ".weight_scale_2"), name),
        )

    def __getitem__(self, selection):
        """Split fused QKV/gate-up along output rows without discarding scales."""
        rows, columns = selection
        if self.transposed or not isinstance(rows, slice) or columns != slice(None):
            raise ValueError("Prepared Phi4 only supports unswizzled output-row slices")
        start, stop, step = rows.indices(self.out_features)
        if step != 1 or stop <= start:
            raise ValueError("Prepared Phi4 row slice must be nonempty and contiguous")
        scale = self.weight_scale if self.format == "fp8" else self.weight_scale[start:stop]
        return replace(
            self,
            packed=np.ascontiguousarray(self.packed[start:stop]),
            weight_scale=np.ascontiguousarray(scale),
            out_features=stop - start,
            keep_alive=[],
        )

    def matmul(self, network, activation, lhs_width, rhs_width):
        import tensorrt as trt

        if not self.transposed or self.shape != (lhs_width, rhs_width):
            raise ValueError("Prepared Phi4 projection dimensions differ from the graph")
        if len(activation.shape) != 2:
            raise ValueError("Prepared Phi4 projection requires a rank2 activation")
        dtype = activation.dtype
        n, k = self.out_features, self.in_features

        def constant(value):
            self.keep_alive.append(value)
            return network.add_constant(tuple(value.shape), trt.Weights(value)).get_output(0)

        if self.format == "int4_awq":
            weights = trt.Weights(trt.DataType.INT4, self.packed.ctypes.data, n * k)
            raw = network.add_constant((n, k), weights).get_output(0)
            scale = constant(self.weight_scale.astype(np.float16))
            if scale.dtype != dtype:
                scale = network.add_cast(scale, dtype).get_output(0)
            dequant = network.add_dequantize(raw, scale, dtype)
            dequant.axis = 1
            weight = dequant.get_output(0)
            lhs = activation
            if self.pre_quant_scale is not None:
                smooth = constant(self.pre_quant_scale.reshape(1, k))
                if smooth.dtype != dtype:
                    smooth = network.add_cast(smooth, dtype).get_output(0)
                lhs = network.add_elementwise(
                    lhs, smooth, trt.ElementWiseOperation.PROD
                ).get_output(0)
        elif self.format == "fp8":
            act_scale = constant(self.input_scale)
            weights = trt.Weights(trt.DataType.FP8, self.packed.ctypes.data, n * k)
            raw = network.add_constant((n, k), weights).get_output(0)
            weight = network.add_dequantize(raw, constant(self.weight_scale), dtype).get_output(0)
            packed_activation = network.add_quantize(
                activation, act_scale, trt.DataType.FP8
            ).get_output(0)
            lhs = network.add_dequantize(packed_activation, act_scale, dtype).get_output(0)
        else:
            act_scale = constant(self.input_scale)
            weights = trt.Weights(trt.DataType.FP4, self.packed.ctypes.data, n * k)
            raw = network.add_constant((n, k), weights).get_output(0)
            scale_weights = trt.Weights(
                trt.DataType.FP8, self.weight_scale.ctypes.data, self.weight_scale.size
            )
            block_scale = network.add_constant((n, k // 16), scale_weights).get_output(0)
            scale = network.add_dequantize(
                block_scale, constant(self.global_scale), dtype
            ).get_output(0)
            dequant = network.add_dequantize(raw, scale, dtype)
            dequant.axis = 1
            weight = dequant.get_output(0)
            dynamic = network.add_dynamic_quantize(
                activation, 1, 16, trt.DataType.FP4, trt.DataType.FP8
            )
            dynamic.set_input(1, act_scale)
            scale = network.add_dequantize(dynamic.get_output(1), act_scale, dtype).get_output(0)
            dequant = network.add_dequantize(dynamic.get_output(0), scale, dtype)
            dequant.axis = 1
            lhs = dequant.get_output(0)
        if self.format == "int4_awq":
            # Keep the dequantized FP16 source values, but accumulate their
            # products in FP32 to avoid intermediate half-precision rounding.
            from .graph_ops import _add_matrix_multiply_with_fp32_accumulation

            return _add_matrix_multiply_with_fp32_accumulation(
                network, lhs, trt.MatrixOperation.NONE, weight, trt.MatrixOperation.TRANSPOSE
            )
        result = network.add_matrix_multiply(
            lhs, trt.MatrixOperation.NONE, weight, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        return result if result.dtype == dtype else network.add_cast(result, dtype).get_output(0)


class PreparedQuantContext:
    """Route only prepared projections through native TensorRT Q/DQ."""

    def maybe_quantized_matmul(
        self, network, lhs, lhs_width, rhs_width, weight, weight_name, *, dtype
    ):
        from . import graph_ops

        if isinstance(weight, PreparedLinear):
            return weight.matmul(network, lhs, lhs_width, rhs_width)
        return graph_ops.add_matmul_rhs_constant(
            network, lhs, lhs_width, rhs_width, weight, dtype=dtype
        )
