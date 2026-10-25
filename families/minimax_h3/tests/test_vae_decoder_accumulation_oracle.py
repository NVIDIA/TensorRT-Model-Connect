# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host guards for the diagnostic, not trained-convolution qualification."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from families.minimax_h3.tests import vae_decoder_accumulation_oracle as oracle


def test_accumulation_cases_and_retained_evidence_guards(tmp_path):
    assert len(oracle.CASES) == len(set(oracle.CASES)) == 30
    value = np.linspace(-1, 1, 24, dtype=np.float16).reshape(1, 1, 24)
    record = oracle.save_array(tmp_path, "complete", value, {})
    np.testing.assert_array_equal(oracle.checked_array(tmp_path, record), value)
    for replacement in (
        dict(path="../complete.npy"),
        dict(shape=[24]),
        dict(dtype="float32"),
        dict(content_sha256="0" * 64),
    ):
        with pytest.raises(AssertionError):
            oracle.checked_array(tmp_path, record | replacement)
    changed = value.copy()
    changed[0, 0, 0] = 0
    with (tmp_path / "complete.npy").open("wb") as file:
        np.save(file, changed, allow_pickle=False)
    with pytest.raises(AssertionError):
        oracle.checked_array(tmp_path, record)


def test_counterfactuals_keep_half_rounding_boundaries_and_real_bias():
    module = torch.nn.Conv3d(2, 2, 1)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([0.12345, -0.23456, 0.34567, 0.45678]).view(2, 2, 1, 1, 1))
        module.bias.copy_(torch.tensor([0.00321, -0.04567]))
    value = torch.tensor([0.12345, 3.45678]).view(1, 2, 1, 1, 1)
    functional = torch.nn.functional
    with torch.no_grad():
        expected = {
            "explicit-half-convolution": functional.conv3d(
                value.half(), module.weight.half(), module.bias.half()
            ),
            "half-operands-float-accumulation": functional.conv3d(
                value.half().float(), module.weight.half().float(), module.bias.half().float()
            ).half(),
            "half-convolution-separate-bias": functional.conv3d(value.half(), module.weight.half())
            + module.bias.half().view(1, -1, 1, 1, 1),
            "original-float-convolution": functional.conv3d(
                value, module.weight, module.bias
            ).half(),
        }
        for variant, reference in expected.items():
            actual = oracle.convolution_counterfactual(variant, module, value)
            assert actual.dtype == torch.float16
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        with pytest.raises(ValueError, match="Unknown convolution"):
            oracle.convolution_counterfactual("invalid", module, value)


def test_dispatch_captures_actual_aten_operands_without_executing_a_gpu():
    module = torch.nn.Conv3d(2, 2, 1)
    value = torch.ones(1, 2, 1, 1, 1)
    with patch.object(oracle, "convolution_counterfactual", side_effect=lambda _v, m, x: m(x)):
        events, arrays, output = oracle.capture_dispatch(module, value)
    convolution = next(row for row in events if "convolution" in row["operator"])
    assert [operand["dtype"] for operand in convolution["operands"]] == ["torch.float32"] * 3
    assert [operand["shape"] for operand in convolution["operands"]] == [
        list(value.shape),
        list(module.weight.shape),
        list(module.bias.shape),
    ]
    assert len(arrays) == 3 and output.shape == value.shape
