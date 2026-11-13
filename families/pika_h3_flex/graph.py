# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family-owned typed graph operations for the mixed-precision Pika H3 Flex model."""

import json

import numpy as np
import tensorrt as trt


class Weights:
    def __init__(self, directory, stack):
        from safetensors import safe_open

        self.files, self.index, self.adapter = {}, {}, None
        for path in sorted(directory.glob("*.safetensors")):
            reader = stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
            self.files[path.name] = reader
            for key in reader.keys():
                if key in self.index:
                    raise ValueError(f"Duplicate checkpoint tensor: {key}")
                self.index[key] = path.name
        indexes = list(directory.glob("*.safetensors.index.json"))
        if indexes:
            expected = json.loads(indexes[0].read_text())["weight_map"]
            if expected != self.index:
                raise ValueError(f"Incomplete checkpoint component: {directory}")
        if not self.index:
            raise ValueError(f"Missing safe weights: {directory}")

    def read(self, key):
        import torch

        return self.files[self.index[key]].get_tensor(key).to(torch.float32).numpy()


class PlanWriter(trt.IStreamWriter):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def write(self, data):
        return self.stream.write(data)


class Graph:
    def __init__(self, weights, logger, minimum=1, optimum=64, maximum=256):
        self.weights, self.logger = weights, logger
        self.builder = trt.Builder(logger)
        self.net = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        self.bounds = minimum, optimum, maximum
        self.profiles = {}

    def cast(self, x, dtype):
        return x if x.dtype == dtype else self.net.add_cast(x, dtype).get_output(0)

    def const(self, value, dtype=trt.bfloat16):
        storage = np.int32 if dtype == trt.int32 else np.float32
        value = np.ascontiguousarray(value, dtype=storage)
        return self.cast(self.net.add_constant(value.shape, value).get_output(0), dtype)

    def scalar(self, value, x):
        return self.const(np.full((1,) * len(x.shape), value), x.dtype)

    def op(self, x, y, kind):
        dtype = x.dtype
        if dtype == trt.bfloat16:
            # Each explicit PyTorch binary operation rounds its BF16 result.
            # Preserve that boundary when TensorRT fuses neighboring operations.
            x = self.cast(x, trt.float32)
            if isinstance(y, trt.ITensor):
                y = self.cast(y, trt.float32)
        if not isinstance(y, trt.ITensor):
            y = self.scalar(y, x)
        result = self.net.add_elementwise(x, y, kind).get_output(0)
        return (
            self.cast(result, dtype)
            if dtype == trt.bfloat16 and result.dtype != trt.bool
            else result
        )

    def affine_bias(self, value, bias):
        # Bias belongs to the affine operator, before its output is rounded.
        return self.net.add_elementwise(value, bias, trt.ElementWiseOperation.SUM).get_output(0)

    def add(self, x, y):
        return self.op(x, y, trt.ElementWiseOperation.SUM)

    def mul(self, x, y):
        return self.op(x, y, trt.ElementWiseOperation.PROD)

    def div(self, x, y):
        return self.op(x, y, trt.ElementWiseOperation.DIV)

    def reshape(self, x, shape):
        layer = self.net.add_shuffle(x)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def transpose(self, x, order):
        layer = self.net.add_shuffle(x)
        layer.second_transpose = order
        return layer.get_output(0)

    def concat(self, values, axis):
        layer = self.net.add_concatenation(values)
        layer.axis = axis
        return layer.get_output(0)

    def gather(self, x, indices, axis=0):
        return self.net.add_gather(x, indices, axis).get_output(0)

    def channels(self, x, first, count):
        # Gather supports a dynamic row count without baking a slice size into
        # the graph. All channel selections are structural constants.
        return self.gather(
            x, self.const(np.arange(first, first + count), trt.int32), len(x.shape) - 1
        )

    def mm(self, x, y, transpose=False):
        return self.net.add_matrix_multiply(
            x,
            trt.MatrixOperation.NONE,
            y,
            trt.MatrixOperation.TRANSPOSE if transpose else trt.MatrixOperation.NONE,
        ).get_output(0)

    def linear(self, x, name, dtype=None, bias=True):
        dtype = x.dtype if dtype is None else dtype
        x = self.cast(x, dtype)
        value = self.mm(x, self.const(self.weights.read(name + ".weight"), dtype), True)
        if bias and name + ".bias" in self.weights.index:
            weight = self.weights.read(name + ".bias").reshape((1,) * (len(x.shape) - 1) + (-1,))
            value = self.affine_bias(value, self.const(weight, dtype))
        adapter = self.weights.adapter
        if adapter is not None and name + ".lora_A.default.weight" in adapter.index:
            # PEFT casts the released FP32 storage to the base layer's BF16
            # compute dtype. Keep the two low-rank projections and their
            # rounding boundaries instead of folding the update into W.
            a = self.const(adapter.read(name + ".lora_A.default.weight"), dtype)
            b = self.const(adapter.read(name + ".lora_B.default.weight"), dtype)
            value = self.add(value, self.mm(self.mm(x, a, True), b, True))
        return value

    def silu(self, x):
        value = self.cast(x, trt.float32)
        sigmoid = self.net.add_activation(value, trt.ActivationType.SIGMOID).get_output(0)
        return self.cast(self.mul(value, sigmoid), x.dtype)

    def norm(self, x, name, eps, *, affine=True, round_before_affine=False, layer=False):
        value = self.cast(x, trt.float32)
        axis = 1 << (len(x.shape) - 1)
        if layer:
            mean = self.net.add_reduce(value, trt.ReduceOperation.AVG, axis, True).get_output(0)
            value = self.op(value, mean, trt.ElementWiseOperation.SUB)
        square = self.mul(value, value)
        mean = self.net.add_reduce(square, trt.ReduceOperation.AVG, axis, True).get_output(0)
        root = self.net.add_unary(self.add(mean, eps), trt.UnaryOperation.SQRT).get_output(0)
        result = self.div(value, root)
        if round_before_affine:
            result = self.cast(result, x.dtype)
        if affine:
            shape = (1,) * (len(x.shape) - 1) + (-1,)
            result = self.mul(
                result, self.const(self.weights.read(name + ".weight").reshape(shape), result.dtype)
            )
            if layer:
                result = self.add(
                    result,
                    self.const(self.weights.read(name + ".bias").reshape(shape), result.dtype),
                )
        return self.cast(result, x.dtype)

    def rope(self, x, cos, sin, rotary_dim):
        rotated = self.channels(x, 0, rotary_dim)
        half = rotary_dim // 2
        swap = self.concat(
            [self.mul(self.channels(rotated, half, half), -1), self.channels(rotated, 0, half)],
            len(x.shape) - 1,
        )
        # Qwen and H3 round each low-precision product before adding them.
        # Explicit FP32 opmath and casts preserve those boundaries when TensorRT
        # fuses the rotary expression.
        products = [
            self.cast(
                self.mul(self.cast(value, trt.float32), self.cast(angle, trt.float32)), x.dtype
            )
            for value, angle in (
                (rotated, self.cast(cos, x.dtype)),
                (swap, self.cast(sin, x.dtype)),
            )
        ]
        result = self.cast(
            self.add(*(self.cast(value, trt.float32) for value in products)), x.dtype
        )
        if x.shape[-1] > rotary_dim:
            result = self.concat(
                [result, self.channels(x, rotary_dim, x.shape[-1] - rotary_dim)], len(x.shape) - 1
            )
        return result

    def attention(
        self,
        hidden,
        prefix,
        heads,
        dim,
        eps,
        *,
        cos=None,
        sin=None,
        rotary_dim=0,
        bias=False,
        affine=True,
    ):
        projections = []
        for kind in ("q", "k", "v"):
            value = self.reshape(
                self.linear(hidden, prefix + ".to_" + kind, bias=bias), (1, -1, heads, dim)
            )
            if kind != "v":
                value = self.norm(value, prefix + ".norm_" + kind, eps, affine=affine)
                if cos is not None:
                    value = self.rope(value, cos, sin, rotary_dim)
            projections.append(self.transpose(value, (0, 2, 1, 3)))
        # PyTorch SDPA accumulates scores, softmax and the value reduction in
        # FP32 for BF16 inputs. Keep those mathematical intermediates explicit.
        query, key, value = [self.cast(x, trt.float32) for x in projections]
        layer = self.net.add_attention(
            self.mul(query, dim**-0.5), key, value, trt.AttentionNormalizationOp.SOFTMAX, False
        )
        layer.decomposable = True
        attended = self.cast(layer.get_output(0), hidden.dtype)
        attended = self.reshape(self.transpose(attended, (0, 2, 1, 3)), (-1, heads * dim))
        return self.linear(attended, prefix + ".to_out.0", bias=bias)

    def ff(self, hidden, prefix, middle, bias=False):
        combined = self.linear(hidden, prefix + ".net.0.proj", bias=bias)
        # Diffusers SwiGLU stores [value, gate], unlike Qwen's separate gate/up.
        value, gate = self.channels(combined, 0, middle), self.channels(combined, middle, middle)
        return self.linear(self.mul(value, self.silu(gate)), prefix + ".net.2", bias=bias)

    def output(self, x, name):
        x.name = name
        self.net.mark_output(x)

    def save(self, stream):
        options = self.builder.create_builder_config()
        options.clear_flag(trt.BuilderFlag.TF32)
        options.builder_optimization_level = 3
        dynamic = [
            self.net.get_input(i)
            for i in range(self.net.num_inputs)
            if -1 in self.net.get_input(i).shape
        ]
        if dynamic:
            profile = self.builder.create_optimization_profile()
            for value in dynamic:
                shapes = self.profiles.get(value.name)
                if shapes is None:
                    shapes = [
                        tuple(bound if d == -1 else d for d in value.shape) for bound in self.bounds
                    ]
                profile.set_shape(value.name, *shapes)
            options.add_optimization_profile(profile)
        if not self.builder.build_serialized_network_to_stream(
            self.net, options, PlanWriter(stream)
        ):
            raise RuntimeError("Pika-H3-Flex engine build failed")
