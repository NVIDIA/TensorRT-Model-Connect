# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the family graph's RMSNorm arithmetic against the upstream layer."""

import numpy as np
import pytest
import tensorrt as trt
import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from families.llava import graph_ops


class Tensor:
    def __init__(self, value):
        self.value = value
        self.dtype = trt.float16 if value.dtype == torch.float16 else trt.float32

    def get_output(self, _index):
        return self


class ArithmeticNetwork:
    def add_cast(self, tensor, dtype):
        return Tensor(tensor.value.to(torch.float16 if dtype == trt.float16 else torch.float32))

    def add_elementwise(self, left, right, operation):
        assert left.dtype == right.dtype, "Strongly typed arithmetic requires equal input dtypes"
        if operation == trt.ElementWiseOperation.PROD:
            return Tensor(left.value * right.value)
        assert operation == trt.ElementWiseOperation.SUM
        return Tensor(left.value + right.value)

    def add_reduce(self, tensor, operation, axes, keep_dims):
        assert operation == trt.ReduceOperation.AVG
        dims = tuple(axis for axis in range(tensor.value.ndim) if axes & (1 << axis))
        return Tensor(tensor.value.mean(dim=dims, keepdim=keep_dims))

    def add_unary(self, tensor, operation):
        if operation == trt.UnaryOperation.SQRT:
            return Tensor(tensor.value.sqrt())
        assert operation == trt.UnaryOperation.RECIP
        return Tensor(tensor.value.reciprocal())


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_rms_norm_matches_upstream_rounding(monkeypatch, dtype):
    torch_dtype = torch.float16 if dtype == np.float16 else torch.float32
    values = torch.tensor([
        [-0.89013671875, 0.468505859375, 0.8310546875, 0.98291015625],
        [-1.0947265625, 0.3798828125, 1.1513671875, 0.260986328125],
    ], dtype=torch_dtype)
    gamma = np.array([0.91064453125, 0.351806640625, 0.8173828125, 0.650390625], dtype=dtype)
    reference = LlamaRMSNorm(4, eps=1e-6).to(dtype=torch_dtype)
    with torch.no_grad():
        reference.weight.copy_(torch.from_numpy(gamma))
        expected = reference(values)

    def constant(_network, shape, array, dtype=np.float32):
        return Tensor(torch.from_numpy(np.asarray(array, dtype=dtype).reshape(shape)))

    monkeypatch.setattr(graph_ops, "add_constant", constant)
    observed = graph_ops.add_rms_norm(
        ArithmeticNetwork(), Tensor(values), 4, gamma,
        Tensor(torch.tensor([[1e-6]], dtype=torch.float32)), dtype=dtype,
    ).value
    if dtype == np.float16:
        normalized = values.float() * torch.rsqrt(values.float().square().mean(-1, keepdim=True) + 1e-6)
        late_rounding = (normalized * reference.weight.float()).half()
        assert not torch.equal(expected, late_rounding), "Fixture must expose the FP16 rounding boundary"
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    else:
        torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-7)
