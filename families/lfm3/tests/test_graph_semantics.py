# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest


class Tensor:
    def __init__(self, values, dtype=None):
        self.values = np.asarray(values, dtype=dtype)
        self.dtype = self.values.dtype
        self.shape = self.values.shape


def layer(value):
    return SimpleNamespace(get_output=lambda i: value)


class Network:
    def add_constant(self, shape, weights):
        return layer(Tensor(weights.reshape(shape)))

    def add_cast(self, value, dtype):
        return layer(Tensor(value.values, dtype))

    def add_shape(self, value):
        return layer(Tensor(value.shape, np.int64))

    def add_slice(self, value, start, shape, stride):
        return layer(
            Tensor(
                value.values[tuple(slice(a, a + b * c, c) for a, b, c in zip(start, shape, stride))]
            )
        )

    def add_elementwise(self, lhs, rhs, op):
        f = dict(
            sum=np.add,
            sub=np.subtract,
            prod=np.multiply,
            less=np.less,
            greater=np.greater,
            and_=np.logical_and,
        )[op]
        return layer(Tensor(f(lhs.values, rhs.values)))

    def add_shuffle(self, value):
        inputs = {}
        result = SimpleNamespace(
            reshape_dims=value.shape, set_input=lambda i, v: inputs.update({i: v})
        )
        result.get_output = lambda i: Tensor(
            value.values.reshape(tuple(inputs[1].values) if 1 in inputs else result.reshape_dims)
        )
        return result

    def add_fill(self, shape, operation, dtype):
        inputs = {}
        return SimpleNamespace(
            set_input=lambda i, v: inputs.update({i: v}),
            get_output=lambda i: Tensor(
                np.arange(int(inputs[0].values[0])) * inputs[2].values + inputs[1].values, dtype
            ),
        )

    def add_concatenation(self, values):
        return layer(Tensor(np.concatenate([v.values for v in values])))

    def add_unary(self, tensor, op):
        function = dict(cos=np.cos, sin=np.sin)[op]
        return layer(Tensor(function(tensor.values), tensor.dtype))

    def add_activation(self, tensor, op):
        assert op == "sigmoid"
        return layer(Tensor(1 / (1 + np.exp(-tensor.values)), tensor.dtype))

    def add_rotary_embedding(self, tensor, cos, sin, interleaved, dimension):
        assert not interleaved
        assert tensor.dtype == cos.dtype == sin.dtype == np.float32
        assert cos.shape[-1] == sin.shape[-1] == dimension // 2
        left, right = np.split(tensor.values[..., :dimension], 2, axis=-1)
        c, s = cos.values[:, None], sin.values[:, None]
        rotated = np.concatenate(
            [left * c - right * s, right * c + left * s, tensor.values[..., dimension:]], axis=-1
        )
        return layer(Tensor(rotated, tensor.dtype))


def source(name):
    tree = ast.parse((Path(__file__).resolve().parents[1] / name).read_text())
    tree.body = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *[n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))],
    ]
    ast.fix_missing_locations(tree)
    trt = SimpleNamespace(
        float32=np.dtype(np.float32),
        float16=np.dtype(np.float16),
        int32=np.dtype(np.int32),
        Weights=lambda v: v,
        UnaryOperation=SimpleNamespace(COS="cos", SIN="sin"),
        ActivationType=SimpleNamespace(SIGMOID="sigmoid"),
        FillOperation=SimpleNamespace(LINSPACE="linspace"),
        ElementWiseOperation=SimpleNamespace(
            SUM="sum", SUB="sub", PROD="prod", LESS="less", GREATER="greater", AND="and_"
        ),
    )
    ns = {"np": np, "trt": trt, "NamedTuple": NamedTuple}
    exec(compile(tree, name, "exec"), ns)
    return ns


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "2"])
def test_sequence_capacity_rejects_non_integer_or_non_positive_requests(value):
    helper = source("model.py")
    with pytest.raises(ValueError, match="positive integer"):
        helper["_positive_int"](value, "max_sequence_length")


@pytest.mark.parametrize("value", [True, [], [False], -1, 16, [1, 16]])
def test_generation_eos_ids_fail_before_bundle_publication(tmp_path, value):
    helper = source("model.py")
    helper["json"] = json
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": value}))
    config = SimpleNamespace(
        bundle_overrides=lambda: {}, vocab_size=16, bos_token_id=0, eos_token_id=1, pad_token_id=2
    )
    with pytest.raises(ValueError, match="eos_token_id"):
        helper["_runtime_config"](tmp_path, config)


@pytest.mark.parametrize("position", [0, 3, 255, 256, 300])
def test_sliding_mask_uses_absolute_positions_without_evicting_cache(position):
    helper = source("native_kv_attention_builder.py")
    mask = helper["add_active_prefix_causal_masks"](
        Network(),
        Tensor([1], np.int32),
        Tensor([position], np.int32),
        Tensor([position + 1], np.int32),
        512,
        sliding_window=256,
    )
    want = np.zeros(512, bool)
    want[max(0, position - 255) : position + 1] = True
    np.testing.assert_array_equal(mask.attention.values.reshape(-1), want)
    np.testing.assert_array_equal(mask.active_prefix.values.reshape(-1), np.arange(512) <= position)


def test_zero_tail_rope_keeps_full_head_pairing_and_fp32_math():
    helper = source("model.py")
    tensor = Tensor([[1, 2, 3, 4]], np.float16)
    result = helper["_apply_rope"](
        Network(),
        tensor,
        Tensor([[[0, 1]]], np.float16),
        Tensor([[[1, 0]]], np.float16),
        num_heads=1,
        head_dim=4,
    )
    np.testing.assert_array_equal(result.values, [[-3, 2, 1, 4]])
    assert result.dtype == np.float16


def test_rope_frequency_uses_full_head_denominator_and_zero_tail():
    helper = source("model.py")
    cosine, sine = helper["_active_rope_cache"](
        Network(),
        Tensor([1], np.int32),
        head_dim=8,
        rotary_dim=4,
        rope_theta=1e6,
        work_dtype=np.dtype(np.float16),
    )
    angles = np.array([1, 10**-1.5, 0, 0], np.float32)
    np.testing.assert_array_equal(cosine.values.reshape(-1), np.cos(angles).astype(np.float16))
    np.testing.assert_array_equal(sine.values.reshape(-1), np.sin(angles).astype(np.float16))
    assert cosine.dtype == np.float16


def test_attention_output_gate_scales_heads_independently():
    helper = source("model.py")
    out = helper["_gate_attention"](
        Network(),
        Tensor([[1.0, 2.0, 3.0, 4.0]], np.float32),
        Tensor([[0.0, np.log(3)]], np.float32),
        num_heads=2,
        head_dim=2,
    )
    np.testing.assert_allclose(out.values, [[0.5, 1.0, 2.25, 3.0]], rtol=1e-6)
