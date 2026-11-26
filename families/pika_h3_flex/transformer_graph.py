# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The mixed-precision transformer graph and its mathematical boundaries."""

import numpy as np

from .graph import Graph
from .native_math import add_math


def _axis(input_index, axis):
    return {"input": input_index, "axis": axis}


class TransformerGraph(Graph):
    def linear(self, x, name, dtype=None, bias=True):
        dtype = x.dtype if dtype is None else dtype
        x = self.cast(x, dtype)
        weight = self.weights.read(name + ".weight")
        shape = [_axis(0, axis) for axis in range(len(x.shape) - 1)] + [weight.shape[0]]
        inputs = [x, self.const(weight, dtype)]
        operation = "linear"
        if bias and name + ".bias" in self.weights.index:
            inputs.append(self.const(self.weights.read(name + ".bias"), dtype))
            operation = "linear_bias"
        value = add_math(self.net, operation, inputs, shape=shape)
        adapter = self.weights.adapter
        if adapter is not None and name + ".lora_A.default.weight" in adapter.index:
            a = adapter.read(name + ".lora_A.default.weight")
            b = adapter.read(name + ".lora_B.default.weight")
            update = add_math(
                self.net, "linear", [x, self.const(a, dtype)], shape=shape[:-1] + [a.shape[0]]
            )
            update = add_math(self.net, "linear", [update, self.const(b, dtype)], shape=shape)
            value = self.add(value, update)
        return value

    def norm(self, x, name, eps, *, affine=True, round_before_affine=False, layer=False):
        width = x.shape[-1]
        scale = self.weights.read(name + ".weight") if affine else np.ones(width, np.float32)
        bias = self.weights.read(name + ".bias") if layer else np.zeros(width, np.float32)
        return add_math(
            self.net,
            "norm",
            [x, self.const(scale, x.dtype), self.const(bias, x.dtype)],
            extra_args=[
                {"type": "float", "value": float(eps)},
                {"type": "int", "value": 2 if layer else 1 if round_before_affine else 0},
            ],
        )

    def silu(self, x):
        return add_math(self.net, "silu", [x])

    def mm(self, x, y, transpose=False):
        rank = max(len(x.shape), len(y.shape))
        shape = []
        for axis in range(rank - 2):
            left, right = axis - rank + len(x.shape), axis - rank + len(y.shape)
            if left < 0 or x.shape[left] == 1:
                shape.append(1 if right < 0 else _axis(1, right))
            else:
                shape.append(_axis(0, left))
        shape.extend([_axis(0, len(x.shape) - 2), _axis(1, len(y.shape) - (2 if transpose else 1))])
        return add_math(
            self.net,
            "mm",
            [x, y],
            shape=shape,
            extra_args=[{"type": "int", "value": int(transpose)}],
        )

    def time_features(self, timestep, channels):
        return add_math(
            self.net,
            "time_features",
            [timestep],
            shape=[_axis(0, 0), int(channels)],
            extra_args=[{"type": "int", "value": int(channels)}],
        )

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
            projections.append(value)
        # Preserve BF16 Q/K/V and the BSHD projection layout used by SDPA.
        # Promoting the attention to FP32 changes the distilled trajectory.
        attended = add_math(self.net, "sdpa", projections)
        return self.linear(
            self.reshape(attended, (-1, heads * dim)), prefix + ".to_out.0", bias=bias
        )
