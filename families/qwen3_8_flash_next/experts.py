# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse routed experts for Flash-Next; all projection and quantization math is TRT."""

import ml_dtypes
import numpy as np
import tensorrt as trt


_LEVELS = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)


class _ExpertGraph:
    def __init__(self, network):
        self.n = network

    def const(self, value):
        array = np.asarray(value)
        return self.n.add_constant(array.shape, np.ascontiguousarray(array)).get_output(0)

    def op(self, a, b, kind):
        return self.n.add_elementwise(a, b, kind).get_output(0)

    def reshape(self, x, dims):
        layer = self.n.add_shuffle(x)
        layer.reshape_dims = dims
        return layer.get_output(0)

    def cast(self, x, dtype):
        return self.n.add_cast(x, dtype).get_output(0)

    def gather(self, lut, codes):
        return self.n.add_gather(lut, codes, 0).get_output(0)

    def unpack(self, words, bits, table):
        """INT32 is a transport for the original little-endian bytes, not a quantizer."""
        ew = trt.ElementWiseOperation
        shift = (1 << (bits * np.arange(32 // bits, dtype=np.int32))).reshape(1, 1, -1)
        shifted = self.op(self.reshape(words, (0, 0, 1)), self.const(shift), ew.FLOOR_DIV)
        base = self.const(np.full((1, 1, 1), 1 << bits, np.int32))
        whole = self.op(shifted, base, ew.FLOOR_DIV)
        codes = self.op(shifted, self.op(whole, base, ew.PROD), ew.SUB)
        return self.gather(self.const(table), codes)

    def round_input(self, x, scale, width, limits):
        """Per-token blocks of 16: E4M3 scale and E2M1 levels, both ties-to-even."""
        n, ew, un = self.n, trt.ElementWiseOperation, trt.UnaryOperation
        blocks = self.reshape(self.cast(x, trt.float32), (0, width // 16, 16))
        absval = n.add_unary(blocks, un.ABS).get_output(0)
        maximum = n.add_reduce(absval, trt.ReduceOperation.MAX, 1 << 2, True).get_output(0)

        # limits is a compact representation of the checkpoint-calibrated
        # FP32 division and nearest-even cut points over the FP16 input domain.
        def limit_column(i):
            return n.add_gather(limits, self.const(np.array(i, np.int32)), 1).get_output(0)

        bounds = limit_column(0)
        zero = self.const(np.zeros((1, 1, 1), np.int32))
        code = zero
        for bit in (64, 32, 16, 8, 4, 2, 1):
            upper = self.op(code, self.const(np.full((1, 1, 1), bit, np.int32)), ew.SUM)
            too_low = self.op(maximum, self.gather(bounds, upper), ew.LESS)
            code = n.add_select(too_low, code, upper).get_output(0)
        values = np.arange(127, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
        effective = self.op(
            self.gather(self.const(values), code), self.reshape(scale, (1, 1, 1)), ew.PROD
        )

        result = self.const(np.zeros((1, 1, 1), np.float32))
        for i in range(1, len(_LEVELS)):
            too_low = self.op(absval, self.gather(limit_column(i), code), ew.LESS)
            result = n.add_select(
                too_low, result, self.const(np.full((1, 1, 1), _LEVELS[i], np.float32))
            ).get_output(0)
        sign = n.add_unary(blocks, un.SIGN).get_output(0)
        signed = self.op(result, sign, ew.PROD)
        return self.reshape(self.cast(self.op(signed, effective, ew.PROD), trt.float16), (0, width))

    def projection(self, x, name, rows, width):
        n, ew = self.n, trt.ElementWiseOperation
        q = n.add_input(name + "_codes", trt.int32, (rows, width // 8))
        s = n.add_input(name + "_blocks", trt.int32, (rows, width // 64))
        alpha = n.add_input(name + "_weight_scale", trt.float32, (1,))
        beta = n.add_input(name + "_input_scale", trt.float32, (1,))
        levels = np.concatenate((_LEVELS, -_LEVELS))
        decoded = self.reshape(self.unpack(q, 4, levels), (rows, width // 16, 16))
        fp8 = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
        scales = self.reshape(self.unpack(s, 8, fp8), (rows, width // 16, 1))
        scales = self.op(scales, self.reshape(alpha, (1, 1, 1)), ew.PROD)
        w = self.cast(self.reshape(self.op(decoded, scales, ew.PROD), (rows, width)), trt.float16)
        limits = n.add_input(name + "_limits", trt.float32, (128, 8))
        inputs = self.cast(self.round_input(x, beta, width, limits), trt.float32)
        # Quantized operands round to the requested compute type before
        # multiplication; GEMM accumulates in FP32 and rounds once at output.
        accum = n.add_matrix_multiply(
            inputs,
            trt.MatrixOperation.NONE,
            self.cast(w, trt.float32),
            trt.MatrixOperation.TRANSPOSE,
        )
        return self.cast(accum.get_output(0), trt.float16)


def routed_expert(network, tokens):
    """Compute one selected expert for its gathered token rows.

    All 48 layers and 512 experts have the same geometry. Keeping the packed
    parameters as inputs permits the native runtime to load only selected experts.
    The tokenwise router coefficients are applied after the down projection.
    """
    g = _ExpertGraph(network)
    gate = g.projection(tokens, "gate", 640, 2560)
    up = g.projection(tokens, "up", 640, 2560)
    calc = g.cast(gate, trt.float32)
    s = network.add_activation(calc, trt.ActivationType.SIGMOID).get_output(0)
    gate = g.cast(g.op(calc, s, trt.ElementWiseOperation.PROD), trt.float16)
    return g.projection(g.op(gate, up, trt.ElementWiseOperation.PROD), "down", 2560, 640)
