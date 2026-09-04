# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from families.bert.weights import _ReaderCollection, _load_layer_norm


class _Reader:
    def __init__(self, tensors: dict[str, np.ndarray]):
        self.tensors = tensors

    def keys(self):
        return self.tensors.keys()

    def get_tensor(self, name: str) -> np.ndarray:
        return self.tensors[name]


@pytest.mark.parametrize(
    ("weight_name", "bias_name"),
    (("norm.weight", "norm.bias"), ("norm.gamma", "norm.beta")),
)
def test_layer_norm_accepts_one_exact_hugging_face_schema(
    weight_name: str,
    bias_name: str,
) -> None:
    readers = _ReaderCollection(
        [_Reader({weight_name: np.array([1.0]), bias_name: np.array([2.0])})]
    )

    weight, bias = _load_layer_norm(readers, "norm")

    np.testing.assert_array_equal(weight, np.array([1.0], dtype=np.float32))
    np.testing.assert_array_equal(bias, np.array([2.0], dtype=np.float32))


@pytest.mark.parametrize(
    "tensors",
    (
        {},
        {"norm.weight": np.array([1.0])},
        {
            "norm.weight": np.array([1.0]),
            "norm.bias": np.array([2.0]),
            "norm.gamma": np.array([1.0]),
            "norm.beta": np.array([2.0]),
        },
    ),
)
def test_layer_norm_rejects_missing_or_ambiguous_schemas(
    tensors: dict[str, np.ndarray],
) -> None:
    readers = _ReaderCollection([_Reader(tensors)])

    with pytest.raises(KeyError, match="exactly one naming schema"):
        _load_layer_norm(readers, "norm")
