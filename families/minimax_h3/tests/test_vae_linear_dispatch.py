# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only instrumentation contracts, not model/kernel qualification."""

import pytest
import torch

from families.minimax_h3.tests.vae_linear_dispatch import capture_linear_calls


@pytest.mark.parametrize("contiguous", [False, True])
def test_linear_dispatch_capture_preserves_values_and_records_actual_layout(contiguous):
    module = torch.nn.Linear(4, 5)
    source = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3).transpose(1, 2)
    if contiguous:
        source = source.contiguous()
    with torch.no_grad():
        expected = module(source)
    records = []
    with torch.no_grad(), capture_linear_calls({"projection": module}, records):
        actual = module(source)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(records) == 1
    row = records[0]
    assert row["module"] == "projection"
    assert row["input"]["shape"] == [2, 3, 4]
    assert row["input"]["stride"] == list(source.stride())
    assert row["input"]["contiguous"] is contiguous
    assert row["input"]["dtype"] == row["output"]["dtype"] == "torch.float32"
    assert row["output"]["shape"] == [2, 3, 5]
    products = [
        op
        for op in row["operators"]
        if op["operator"] in {"aten.mm.default", "aten.bmm.default", "aten.addmm.default"}
    ]
    assert products and all(op["operands"] and op["output"] for op in products)
    assert not module._forward_pre_hooks and not module._forward_hooks


def test_linear_capture_removes_hooks_on_failure():
    module = torch.nn.Linear(4, 5)
    records = []
    with pytest.raises(RuntimeError), capture_linear_calls({"projection": module}, records):
        module(torch.zeros(2, 3))
    assert not module._forward_pre_hooks and not module._forward_hooks
    assert len(records) == 1 and records[0]["output"] is None
