# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quantization format strategies.

Each QuantFormat implementation encapsulates how to insert Q/DQ nodes
around a matmul for one specific quantization scheme. The format owns
all TRT API calls for that scheme — no switch/case logic elsewhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from tensorrt_model_connect import trt_compat
    trt = trt_compat.get_trt()

from .scales import LayerScales

# Packed NVFP4 weight buffers must outlive the engine build (TRT holds the raw
# pointers until serialization); keep references alive here.
_NVFP4_WEIGHT_KEEPALIVE: list = []


@runtime_checkable
class QuantFormat(Protocol):
    """Strategy for inserting Q/DQ nodes around a matmul."""

    @property
    def name(self) -> str:
        """Format identifier (e.g., 'fp8', 'int8_sq', 'int4_awq')."""
        ...

    def wrap_matmul(
        self,
        network: trt.INetworkDefinition,
        activation: trt.ITensor,
        weight_array: np.ndarray,
        scales: LayerScales,
        *,
        lhs_width: int,
        rhs_width: int,
        dtype: np.dtype,
        graph_ops: Any,
    ) -> trt.ITensor:
        """Insert format-specific Q/DQ around a matmul.

        Args:
            network: TRT network definition.
            activation: LHS input tensor (activations).
            weight_array: RHS weight numpy array [lhs_width, rhs_width].
            scales: Per-layer scales for this operation.
            lhs_width: Input feature dimension.
            rhs_width: Output feature dimension.
            dtype: Working numpy dtype (np.float16 for FP16 mode).
            graph_ops: Family-owned graph helper module.

        Returns:
            Dequantized output tensor after quantized matmul.
        """
        ...

    def wrap_conv2d(
        self,
        network: trt.INetworkDefinition,
        activation: trt.ITensor,
        weight_array: np.ndarray,
        bias_array: np.ndarray | None,
        scales: LayerScales,
        *,
        out_channels: int,
        kernel_shape: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        groups: int = 1,
        dtype: np.dtype,
        graph_ops: Any,
    ) -> trt.ITensor:
        """Insert format-specific Q/DQ around a Conv2D."""
        ...


# ---------------------------------------------------------------------------
# Concrete format implementations
# ---------------------------------------------------------------------------

def _np_to_trt_dtype(dtype: np.dtype):
    """Convert numpy dtype to TRT DataType."""
    from tensorrt_model_connect import trt_compat
    trt = trt_compat.get_trt()
    if dtype == np.float16:
        return trt.float16
    return trt.float32


def _cast_output_dtype(
    network: "trt.INetworkDefinition",
    tensor: "trt.ITensor",
    target_dtype: "trt.DataType",
) -> "trt.ITensor":
    if tensor.dtype == target_dtype:
        return tensor
    return network.add_cast(tensor, target_dtype).get_output(0)


class FP8Format:
    """FP8 E4M3 per-tensor quantization.

    Both weights and activations are quantized to FP8 with per-tensor
    scales. TRT fuses the Q/DQ pairs with the adjacent matmul into a
    single FP8 Tensor Core kernel.
    """

    @property
    def name(self) -> str:
        return "fp8"

    def wrap_matmul(
        self,
        network: trt.INetworkDefinition,
        activation: trt.ITensor,
        weight_array: np.ndarray,
        scales: LayerScales,
        *,
        lhs_width: int,
        rhs_width: int,
        dtype: np.dtype,
        graph_ops: Any,
    ) -> trt.ITensor:
        from tensorrt_model_connect import trt_compat
        trt = trt_compat.get_trt()

        out_trt_dtype = activation.dtype
        # With fp16 base, materializing the fp8 GEMM result in fp16 overflows
        # fp16's narrow exponent range on bf16-native models (garbage output):
        # the running partial sum of many dequantized terms exceeds ~65504.
        # Dequantize/accumulate in fp32 ONLY for fp16 base, then cast back to the
        # base dtype. bf16/fp32 base keep their native dtype (range headroom) so
        # the fast bf16 path sees no perf regression.
        acc_dtype = trt.float32 if out_trt_dtype == trt.float16 else out_trt_dtype

        # Weight Q/DQ: constant -> quantize(FP8) -> dequantize(acc_dtype).
        # STRONGLY_TYPED networks require a quantize's input+scale to share a
        # dtype and a dequantize's scale to match its output dtype, so cast the
        # weight const to the work dtype and give each role its own scale cast.
        # All casts are no-ops for an fp32 base (regression-safe); required for
        # a bf16/fp16 base, where an fp32 scale would fail the build (arith.divf).
        weight_const = graph_ops.add_constant(
            network, (lhs_width, rhs_width), weight_array, dtype=dtype)
        weight_const = _cast_output_dtype(network, weight_const, out_trt_dtype)
        w_scale = np.array(
            [scales.weight_scale] if np.isscalar(scales.weight_scale)
            else scales.weight_scale, dtype=np.float32)
        w_scale_t = graph_ops.add_constant(
            network, w_scale.shape, w_scale, dtype=np.float32)
        w_scale_q = _cast_output_dtype(network, w_scale_t, out_trt_dtype)
        w_scale_dq = _cast_output_dtype(network, w_scale_t, acc_dtype)
        q_w = network.add_quantize(weight_const, w_scale_q, trt.fp8)
        dq_w = network.add_dequantize(
            q_w.get_output(0), w_scale_dq, acc_dtype)

        # Activation Q/DQ: input -> quantize(FP8) -> dequantize(acc_dtype)
        a_scale = np.array(
            [scales.input_scale] if np.isscalar(scales.input_scale)
            else scales.input_scale, dtype=np.float32)
        a_scale_t = graph_ops.add_constant(
            network, a_scale.shape, a_scale, dtype=np.float32)
        a_scale_q = _cast_output_dtype(network, a_scale_t, out_trt_dtype)
        a_scale_dq = _cast_output_dtype(network, a_scale_t, acc_dtype)
        q_a = network.add_quantize(activation, a_scale_q, trt.fp8)
        dq_a = network.add_dequantize(
            q_a.get_output(0), a_scale_dq, acc_dtype)

        # Matmul on dequantized tensors (TRT fuses Q/DQ + matmul)
        mm = network.add_matrix_multiply(
            dq_a.get_output(0), trt.MatrixOperation.NONE,
            dq_w.get_output(0), trt.MatrixOperation.NONE)
        return _cast_output_dtype(network, mm.get_output(0), out_trt_dtype)

    def wrap_conv2d(
        self,
        network: trt.INetworkDefinition,
        activation: trt.ITensor,
        weight_array: np.ndarray,
        bias_array: np.ndarray | None,
        scales: LayerScales,
        *,
        out_channels: int,
        kernel_shape: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        groups: int = 1,
        dtype: np.dtype,
        graph_ops: Any,
    ) -> trt.ITensor:
        from tensorrt_model_connect import trt_compat
        trt = trt_compat.get_trt()

        out_trt_dtype = activation.dtype

        # Weight Q/DQ: constant -> quantize(FP8) -> dequantize(work_dtype)
        w_scale = np.array(
            [scales.weight_scale] if np.isscalar(scales.weight_scale)
            else scales.weight_scale, dtype=np.float32)
        w_scale_t = graph_ops.add_constant(
            network, w_scale.shape, w_scale, dtype=np.float32)
        w_const = graph_ops.add_constant(
            network, weight_array.shape, weight_array, dtype=dtype)
        q_w = network.add_quantize(w_const, w_scale_t, trt.fp8)
        dq_w = network.add_dequantize(
            q_w.get_output(0), w_scale_t, out_trt_dtype)

        # Activation Q/DQ: input -> quantize(FP8) -> dequantize(work_dtype)
        a_scale = np.array(
            [scales.input_scale] if np.isscalar(scales.input_scale)
            else scales.input_scale, dtype=np.float32)
        a_scale_t = graph_ops.add_constant(
            network, a_scale.shape, a_scale, dtype=np.float32)
        q_a = network.add_quantize(activation, a_scale_t, trt.fp8)
        dq_a = network.add_dequantize(
            q_a.get_output(0), a_scale_t, out_trt_dtype)

        # Conv2D on dequantized tensors
        b_trt = trt.Weights(np.ascontiguousarray(
            bias_array, dtype=dtype)) if bias_array is not None else trt.Weights()
        conv = network.add_convolution_nd(
            dq_a.get_output(0), out_channels,
            kernel_shape, dq_w.get_output(0), b_trt)
        conv.stride_nd = stride
        conv.padding_nd = padding
        if groups > 1:
            conv.num_groups = groups
        return _cast_output_dtype(network, conv.get_output(0), out_trt_dtype)


class INT8SmoothQuantFormat:
    """INT8 SmoothQuant: per-channel weights, per-tensor activations."""

    @property
    def name(self) -> str:
        return "int8_sq"

    def wrap_matmul(
        self,
        network: trt.INetworkDefinition,
        activation: trt.ITensor,
        weight_array: np.ndarray,
        scales: LayerScales,
        *,
        lhs_width: int,
        rhs_width: int,
        dtype: np.dtype,
        graph_ops: Any,
    ) -> trt.ITensor:
        from tensorrt_model_connect import trt_compat
        trt = trt_compat.get_trt()

        out_trt_dtype = activation.dtype

        # SmoothQuant: migrate per-input-channel range from activations into
        # weights. ModelOpt stores pre_quant_scale (=1/s) on the input quantizer
        # and calibrates the weight amax on the SMOOTHED weight. MC loads the
        # ORIGINAL weights, so reproduce the smoothing here: activations are
        # scaled by pre_quant_scale and weights by 1/pre_quant_scale (per input
        # channel) — x@W == (x*pqs)@(W/pqs). When pre_quant_scale is absent the
        # format degrades to plain per-channel-weight int8.
        pqs = getattr(scales, "pre_quant_scale", None)
        weight_for_const = weight_array
        if pqs is not None:
            pqs = np.asarray(pqs, dtype=np.float32).reshape(-1)  # [lhs_width=in]
            inv = (1.0 / pqs).astype(np.float32)
            weight_for_const = np.ascontiguousarray(
                weight_array.astype(np.float32) * inv[:, None]).astype(
                    weight_array.dtype)

        # Weight Q/DQ: per-channel (axis=1 for [in, out] layout)
        weight_const = graph_ops.add_constant(
            network, (lhs_width, rhs_width), weight_for_const, dtype=dtype)
        w_scale = np.asarray(scales.weight_scale, dtype=np.float32)
        if w_scale.ndim == 0:
            w_scale = np.full(rhs_width, float(w_scale), dtype=np.float32)
        w_scale_t = graph_ops.add_constant(
            network, w_scale.shape, w_scale, dtype=np.float32)
        q_w = network.add_quantize(weight_const, w_scale_t, trt.int8)
        q_w.axis = 1  # per-channel along output dimension
        dq_w = network.add_dequantize(
            q_w.get_output(0), w_scale_t, out_trt_dtype)
        dq_w.axis = 1

        # SmoothQuant activation scaling: x_smooth = x * pre_quant_scale
        # (per input channel), broadcast over the leading (token) dim.
        act = activation
        if pqs is not None:
            pqs_t = graph_ops.add_constant(
                network, (1, lhs_width), pqs.reshape(1, lhs_width),
                dtype=np.float32)
            pqs_in = pqs_t
            if activation.dtype != trt.float32:
                pqs_in = network.add_cast(pqs_t, activation.dtype).get_output(0)
            act = network.add_elementwise(
                activation, pqs_in, trt.ElementWiseOperation.PROD).get_output(0)

        # Activation Q/DQ: per-tensor
        a_scale = np.array(
            [scales.input_scale] if np.isscalar(scales.input_scale)
            else scales.input_scale, dtype=np.float32)
        a_scale_t = graph_ops.add_constant(
            network, a_scale.shape, a_scale, dtype=np.float32)
        q_a = network.add_quantize(act, a_scale_t, trt.int8)
        dq_a = network.add_dequantize(
            q_a.get_output(0), a_scale_t, out_trt_dtype)

        mm = network.add_matrix_multiply(
            dq_a.get_output(0), trt.MatrixOperation.NONE,
            dq_w.get_output(0), trt.MatrixOperation.NONE)
        return _cast_output_dtype(network, mm.get_output(0), out_trt_dtype)

    def wrap_conv2d(
        self,
        network: trt.INetworkDefinition,
        activation: trt.ITensor,
        weight_array: np.ndarray,
        bias_array: np.ndarray | None,
        scales: LayerScales,
        *,
        out_channels: int,
        kernel_shape: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        groups: int = 1,
        dtype: np.dtype,
        graph_ops: Any,
    ) -> trt.ITensor:
        from tensorrt_model_connect import trt_compat
        trt = trt_compat.get_trt()

        out_trt_dtype = activation.dtype

        # Weight Q/DQ: per-channel (axis=0 for conv [OC, IC, kH, kW])
        w_const = graph_ops.add_constant(
            network, weight_array.shape, weight_array, dtype=dtype)
        w_scale = np.asarray(scales.weight_scale, dtype=np.float32)
        if w_scale.ndim == 0:
            w_scale = np.full(out_channels, float(w_scale), dtype=np.float32)
        w_scale_t = graph_ops.add_constant(
            network, w_scale.shape, w_scale, dtype=np.float32)
        q_w = network.add_quantize(w_const, w_scale_t, trt.int8)
        q_w.axis = 0  # per-channel along output channels for conv
        dq_w = network.add_dequantize(
            q_w.get_output(0), w_scale_t, out_trt_dtype)
        dq_w.axis = 0

        # Activation Q/DQ: per-tensor
        a_scale = np.array(
            [scales.input_scale] if np.isscalar(scales.input_scale)
            else scales.input_scale, dtype=np.float32)
        a_scale_t = graph_ops.add_constant(
            network, a_scale.shape, a_scale, dtype=np.float32)
        q_a = network.add_quantize(activation, a_scale_t, trt.int8)
        dq_a = network.add_dequantize(
            q_a.get_output(0), a_scale_t, out_trt_dtype)

        # Conv2D on dequantized tensors
        b_trt = trt.Weights(np.ascontiguousarray(
            bias_array, dtype=dtype)) if bias_array is not None else trt.Weights()
        conv = network.add_convolution_nd(
            dq_a.get_output(0), out_channels,
            kernel_shape, dq_w.get_output(0), b_trt)
        conv.stride_nd = stride
        conv.padding_nd = padding
        if groups > 1:
            conv.num_groups = groups
        return _cast_output_dtype(network, conv.get_output(0), out_trt_dtype)


class INT4AWQFormat:
    """INT4 AWQ: weight-only block quantization with FP16 activations."""

    @property
    def name(self) -> str:
        return "int4_awq"

    def wrap_matmul(
        self,
        network: trt.INetworkDefinition,
        activation: trt.ITensor,
        weight_array: np.ndarray,
        scales: LayerScales,
        *,
        lhs_width: int,
        rhs_width: int,
        dtype: np.dtype,
        graph_ops: Any,
    ) -> trt.ITensor:
        from tensorrt_model_connect import trt_compat
        trt = trt_compat.get_trt()

        out_trt_dtype = activation.dtype
        block_size = scales.block_size or 128

        # Weight Q/DQ: INT4 with block quantization
        weight_const = graph_ops.add_constant(
            network, (lhs_width, rhs_width), weight_array, dtype=dtype)
        w_scale = np.asarray(scales.weight_scale, dtype=np.float32)
        w_scale_t = graph_ops.add_constant(
            network, w_scale.shape, w_scale, dtype=np.float32)
        q_w = network.add_quantize(weight_const, w_scale_t, trt.int4)
        q_w.block_shape = trt.Dims([block_size])
        dq_w = network.add_dequantize(
            q_w.get_output(0), w_scale_t, out_trt_dtype)
        dq_w.block_shape = trt.Dims([block_size])

        # Activation: stays in working dtype (no activation quantization)
        mm = network.add_matrix_multiply(
            activation, trt.MatrixOperation.NONE,
            dq_w.get_output(0), trt.MatrixOperation.NONE)
        return _cast_output_dtype(network, mm.get_output(0), out_trt_dtype)

    def wrap_conv2d(
        self,
        network: trt.INetworkDefinition,
        activation: trt.ITensor,
        weight_array: np.ndarray,
        bias_array: np.ndarray | None,
        scales: LayerScales,
        *,
        out_channels: int,
        kernel_shape: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        groups: int = 1,
        dtype: np.dtype,
        graph_ops: Any,
    ) -> trt.ITensor:
        from tensorrt_model_connect import trt_compat
        trt = trt_compat.get_trt()

        out_trt_dtype = activation.dtype
        block_size = scales.block_size or 128

        # Weight Q/DQ: INT4 with block quantization
        w_const = graph_ops.add_constant(
            network, weight_array.shape, weight_array, dtype=dtype)
        w_scale = np.asarray(scales.weight_scale, dtype=np.float32)
        w_scale_t = graph_ops.add_constant(
            network, w_scale.shape, w_scale, dtype=np.float32)
        q_w = network.add_quantize(w_const, w_scale_t, trt.int4)
        q_w.block_shape = trt.Dims([block_size])
        dq_w = network.add_dequantize(
            q_w.get_output(0), w_scale_t, out_trt_dtype)
        dq_w.block_shape = trt.Dims([block_size])

        # Activation: stays in working dtype (weight-only quantization)
        b_trt = trt.Weights(np.ascontiguousarray(
            bias_array, dtype=dtype)) if bias_array is not None else trt.Weights()
        conv = network.add_convolution_nd(
            activation, out_channels,
            kernel_shape, dq_w.get_output(0), b_trt)
        conv.stride_nd = stride
        conv.padding_nd = padding
        if groups > 1:
            conv.num_groups = groups
        return _cast_output_dtype(network, conv.get_output(0), out_trt_dtype)


class NVFP4Format:
    """NVFP4: Blackwell-native 4-bit float with dynamic activation quantization."""

    @property
    def name(self) -> str:
        return "nvfp4"

    def wrap_matmul(
        self,
        network: trt.INetworkDefinition,
        activation: trt.ITensor,
        weight_array: np.ndarray,
        scales: LayerScales,
        *,
        lhs_width: int,
        rhs_width: int,
        dtype: np.dtype,
        graph_ops: Any,
    ) -> trt.ITensor:
        from tensorrt_model_connect import trt_compat
        trt = trt_compat.get_trt()

        out_trt_dtype = activation.dtype
        block_size = scales.block_size or 16

        # NVFP4 = FP4 (E2M1) values + FP8 (E4M3) per-block scales + FP32 per-tensor
        # "double" scale. The block structure comes from the dynamic-quantize layer;
        # we deliberately do NOT set block_shape on the dequantizes — that kNDBLOCKED
        # path triggers a Myelin quantize-op fusion failure inside the full decoder.
        def _nvfp4_dynamic(t, axis, amax):
            # Per-block scales computed at runtime; per-tensor global = amax.
            ds = graph_ops.add_constant(
                network, (1,),
                np.array([max(amax / (6.0 * 448.0), 1e-8)], dtype=np.float32),
                dtype=np.float32)
            dq = network.add_dynamic_quantize(
                t, axis, block_size, trt.DataType.FP4, trt.fp8)
            dq.set_input(1, ds)
            sc = network.add_dequantize(
                dq.get_output(1), ds, out_trt_dtype).get_output(0)
            return network.add_dequantize(
                dq.get_output(0), sc, out_trt_dtype).get_output(0)

        wf = np.ascontiguousarray(np.asarray(weight_array, dtype=np.float32))
        w_amax = float(np.abs(wf).max())

        # Weight: static packed FP4 + FP8 per-block scales + FP32 global, so the
        # engine stores 4-bit weights (~2-3x smaller). Falls back to the dynamic
        # (correct but uncompressed) weight path on any failure.
        try:
            import ml_dtypes
            if lhs_width % block_size:
                raise ValueError(
                    "in-dim %d not divisible by block %d" % (lhs_width, block_size))
            nb = lhs_width // block_size
            w_global = np.float32(max(w_amax / (6.0 * 448.0), 1e-8))
            woi = np.ascontiguousarray(wf.T)                       # [out, in]
            wob = woi.reshape(rhs_width, nb, block_size)           # [out, nb, blk]
            real_blk = np.maximum(np.abs(wob).max(axis=2) / 6.0, 1e-8)
            fp8_blk = (real_blk / w_global).astype(ml_dtypes.float8_e4m3fn)
            wq = (wob / real_blk[:, :, None]).reshape(
                rhs_width, lhs_width).astype(ml_dtypes.float4_e2m1fn)
            raw = wq.view(np.uint8).reshape(-1)
            packed = np.ascontiguousarray(
                (raw[0::2] & 0x0F) | ((raw[1::2] & 0x0F) << 4))    # 2 fp4/byte
            fp8_bytes = np.ascontiguousarray(fp8_blk.view(np.uint8))
            g_arr = np.ascontiguousarray([w_global], dtype=np.float32)
            _NVFP4_WEIGHT_KEEPALIVE.extend([packed, fp8_bytes, g_arr])
            w_const = network.add_constant(
                (rhs_width, lhs_width),
                trt.Weights(trt.DataType.FP4, packed.ctypes.data,
                            rhs_width * lhs_width)).get_output(0)
            s8_const = network.add_constant(
                (rhs_width, nb),
                trt.Weights(trt.DataType.FP8, fp8_bytes.ctypes.data,
                            rhs_width * nb)).get_output(0)
            g_const = graph_ops.add_constant(network, (1,), g_arr, dtype=np.float32)
            real_scale = network.add_dequantize(
                s8_const, g_const, out_trt_dtype).get_output(0)
            dq_w = network.add_dequantize(
                w_const, real_scale, out_trt_dtype).get_output(0)
            weight_transposed = True
        except Exception:
            weight_const = graph_ops.add_constant(
                network, (lhs_width, rhs_width), weight_array, dtype=dtype)
            dq_w = _nvfp4_dynamic(weight_const, 0, w_amax)
            weight_transposed = False

        # Activation: dynamic per-block; per-tensor global = calibrated amax.
        NVFP4_MAXBOUND = 6.0
        a_amax = (float(scales.input_scale) * NVFP4_MAXBOUND
                  if np.isscalar(scales.input_scale)
                  and float(scales.input_scale) not in (0.0, 1.0)
                  else w_amax)
        dq_a = _nvfp4_dynamic(activation, len(activation.shape) - 1, a_amax)

        w_op = (trt.MatrixOperation.TRANSPOSE if weight_transposed
                else trt.MatrixOperation.NONE)
        mm = network.add_matrix_multiply(
            dq_a, trt.MatrixOperation.NONE, dq_w, w_op)
        return _cast_output_dtype(network, mm.get_output(0), out_trt_dtype)

    def wrap_conv2d(
        self,
        network: trt.INetworkDefinition,
        activation: trt.ITensor,
        weight_array: np.ndarray,
        bias_array: np.ndarray | None,
        scales: LayerScales,
        *,
        out_channels: int,
        kernel_shape: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        groups: int = 1,
        dtype: np.dtype,
        graph_ops: Any,
    ) -> trt.ITensor:
        from tensorrt_model_connect import trt_compat
        trt = trt_compat.get_trt()

        out_trt_dtype = activation.dtype
        block_size = scales.block_size or 16

        # Weight: static FP4 quantization with block shape
        w_const = graph_ops.add_constant(
            network, weight_array.shape, weight_array, dtype=dtype)
        w_scale = np.asarray(scales.weight_scale, dtype=np.float32)
        w_scale_t = graph_ops.add_constant(
            network, w_scale.shape, w_scale, dtype=np.float32)
        q_w = network.add_quantize(w_const, w_scale_t, trt.DataType.FP4)
        q_w.block_shape = trt.Dims([block_size])
        dq_w = network.add_dequantize(
            q_w.get_output(0), w_scale_t, out_trt_dtype)
        dq_w.block_shape = trt.Dims([block_size])

        # Activation: dynamic quantization (scales computed at runtime)
        dq_a = network.add_dynamic_quantize_v2(
            activation,
            trt.Dims([block_size]),
            trt.DataType.FP4,
            trt.float32,  # scale type
        )

        # Conv2D on quantized tensors
        b_trt = trt.Weights(np.ascontiguousarray(
            bias_array, dtype=dtype)) if bias_array is not None else trt.Weights()
        conv = network.add_convolution_nd(
            dq_a.get_output(0), out_channels,
            kernel_shape, dq_w.get_output(0), b_trt)
        conv.stride_nd = stride
        conv.padding_nd = padding
        if groups > 1:
            conv.num_groups = groups
        return _cast_output_dtype(network, conv.get_output(0), out_trt_dtype)


class W4A8Format:
    """W4A8: INT4 weights with INT8 activations (mixed precision)."""

    @property
    def name(self) -> str:
        return "w4a8"

    def wrap_matmul(
        self,
        network: trt.INetworkDefinition,
        activation: trt.ITensor,
        weight_array: np.ndarray,
        scales: LayerScales,
        *,
        lhs_width: int,
        rhs_width: int,
        dtype: np.dtype,
        graph_ops: Any,
    ) -> trt.ITensor:
        from tensorrt_model_connect import trt_compat
        trt = trt_compat.get_trt()

        out_trt_dtype = activation.dtype
        block_size = scales.block_size or 128

        # Weight: INT4 block quantization
        weight_const = graph_ops.add_constant(
            network, (lhs_width, rhs_width), weight_array, dtype=dtype)
        w_scale = np.asarray(scales.weight_scale, dtype=np.float32)
        w_scale_t = graph_ops.add_constant(
            network, w_scale.shape, w_scale, dtype=np.float32)
        q_w = network.add_quantize(weight_const, w_scale_t, trt.int4)
        q_w.block_shape = trt.Dims([block_size])
        dq_w = network.add_dequantize(
            q_w.get_output(0), w_scale_t, out_trt_dtype)
        dq_w.block_shape = trt.Dims([block_size])

        # Activation: INT8 per-tensor quantization
        a_scale = np.array(
            [scales.input_scale] if np.isscalar(scales.input_scale)
            else scales.input_scale, dtype=np.float32)
        a_scale_t = graph_ops.add_constant(
            network, a_scale.shape, a_scale, dtype=np.float32)
        q_a = network.add_quantize(activation, a_scale_t, trt.int8)
        dq_a = network.add_dequantize(
            q_a.get_output(0), a_scale_t, out_trt_dtype)

        mm = network.add_matrix_multiply(
            dq_a.get_output(0), trt.MatrixOperation.NONE,
            dq_w.get_output(0), trt.MatrixOperation.NONE)
        return _cast_output_dtype(network, mm.get_output(0), out_trt_dtype)

    def wrap_conv2d(
        self,
        network: trt.INetworkDefinition,
        activation: trt.ITensor,
        weight_array: np.ndarray,
        bias_array: np.ndarray | None,
        scales: LayerScales,
        *,
        out_channels: int,
        kernel_shape: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        groups: int = 1,
        dtype: np.dtype,
        graph_ops: Any,
    ) -> trt.ITensor:
        from tensorrt_model_connect import trt_compat
        trt = trt_compat.get_trt()

        out_trt_dtype = activation.dtype
        block_size = scales.block_size or 128

        # Weight: INT4 block quantization
        w_const = graph_ops.add_constant(
            network, weight_array.shape, weight_array, dtype=dtype)
        w_scale = np.asarray(scales.weight_scale, dtype=np.float32)
        w_scale_t = graph_ops.add_constant(
            network, w_scale.shape, w_scale, dtype=np.float32)
        q_w = network.add_quantize(w_const, w_scale_t, trt.int4)
        q_w.block_shape = trt.Dims([block_size])
        dq_w = network.add_dequantize(
            q_w.get_output(0), w_scale_t, out_trt_dtype)
        dq_w.block_shape = trt.Dims([block_size])

        # Activation: INT8 per-tensor quantization
        a_scale = np.array(
            [scales.input_scale] if np.isscalar(scales.input_scale)
            else scales.input_scale, dtype=np.float32)
        a_scale_t = graph_ops.add_constant(
            network, a_scale.shape, a_scale, dtype=np.float32)
        q_a = network.add_quantize(activation, a_scale_t, trt.int8)
        dq_a = network.add_dequantize(
            q_a.get_output(0), a_scale_t, out_trt_dtype)

        # Conv2D on quantized tensors
        b_trt = trt.Weights(np.ascontiguousarray(
            bias_array, dtype=dtype)) if bias_array is not None else trt.Weights()
        conv = network.add_convolution_nd(
            dq_a.get_output(0), out_channels,
            kernel_shape, dq_w.get_output(0), b_trt)
        conv.stride_nd = stride
        conv.padding_nd = padding
        if groups > 1:
            conv.num_groups = groups
        return _cast_output_dtype(network, conv.get_output(0), out_trt_dtype)
