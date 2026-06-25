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

        # Weight Q/DQ: constant -> quantize(FP8) -> dequantize(work_dtype)
        weight_const = graph_ops.add_constant(
            network, (lhs_width, rhs_width), weight_array, dtype=dtype)
        w_scale = np.array(
            [scales.weight_scale] if np.isscalar(scales.weight_scale)
            else scales.weight_scale, dtype=np.float32)
        w_scale_t = graph_ops.add_constant(
            network, w_scale.shape, w_scale, dtype=np.float32)
        q_w = network.add_quantize(weight_const, w_scale_t, trt.fp8)
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

        # Weight Q/DQ: per-channel (axis=1 for [in, out] layout)
        weight_const = graph_ops.add_constant(
            network, (lhs_width, rhs_width), weight_array, dtype=dtype)
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

        # Activation Q/DQ: per-tensor
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
        q_w.set_block_shape(trt.Dims([block_size]))
        dq_w = network.add_dequantize(
            q_w.get_output(0), w_scale_t, out_trt_dtype)
        dq_w.set_block_shape(trt.Dims([block_size]))

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
        q_w.set_block_shape(trt.Dims([block_size]))
        dq_w = network.add_dequantize(
            q_w.get_output(0), w_scale_t, out_trt_dtype)
        dq_w.set_block_shape(trt.Dims([block_size]))

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

        # Weight: static FP4 quantization with block shape
        weight_const = graph_ops.add_constant(
            network, (lhs_width, rhs_width), weight_array, dtype=dtype)
        w_scale = np.asarray(scales.weight_scale, dtype=np.float32)
        w_scale_t = graph_ops.add_constant(
            network, w_scale.shape, w_scale, dtype=np.float32)
        q_w = network.add_quantize(weight_const, w_scale_t, trt.DataType.FP4)
        q_w.set_block_shape(trt.Dims([block_size]))
        dq_w = network.add_dequantize(
            q_w.get_output(0), w_scale_t, out_trt_dtype)
        dq_w.set_block_shape(trt.Dims([block_size]))

        # Activation: dynamic quantization (scales computed at runtime)
        dq_a = network.add_dynamic_quantize_v2(
            activation,
            trt.Dims([block_size]),
            trt.DataType.FP4,
            trt.float32,  # scale type
        )

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
        block_size = scales.block_size or 16

        # Weight: static FP4 quantization with block shape
        w_const = graph_ops.add_constant(
            network, weight_array.shape, weight_array, dtype=dtype)
        w_scale = np.asarray(scales.weight_scale, dtype=np.float32)
        w_scale_t = graph_ops.add_constant(
            network, w_scale.shape, w_scale, dtype=np.float32)
        q_w = network.add_quantize(w_const, w_scale_t, trt.DataType.FP4)
        q_w.set_block_shape(trt.Dims([block_size]))
        dq_w = network.add_dequantize(
            q_w.get_output(0), w_scale_t, out_trt_dtype)
        dq_w.set_block_shape(trt.Dims([block_size]))

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
        q_w.set_block_shape(trt.Dims([block_size]))
        dq_w = network.add_dequantize(
            q_w.get_output(0), w_scale_t, out_trt_dtype)
        dq_w.set_block_shape(trt.Dims([block_size]))

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
        q_w.set_block_shape(trt.Dims([block_size]))
        dq_w = network.add_dequantize(
            q_w.get_output(0), w_scale_t, out_trt_dtype)
        dq_w.set_block_shape(trt.Dims([block_size]))

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
