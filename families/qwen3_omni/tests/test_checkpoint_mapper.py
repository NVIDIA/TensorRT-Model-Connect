# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint loading keeps BF16 storage until each graph boundary needs a cast."""

from types import SimpleNamespace

import ml_dtypes
import numpy as np

from .. import checkpoint_mapper


def test_load_tensor_preserves_bf16_values_without_a_full_precision_copy() -> None:
    values = np.array([[1.001, -0.3333], [17.0625, 0.0]], dtype=ml_dtypes.bfloat16)
    reader = SimpleNamespace(get_tensor=lambda _name: values)
    readers = checkpoint_mapper._ReaderCollection({"weight": reader})

    loaded = checkpoint_mapper._load_tensor(readers, "weight")

    assert loaded.dtype == values.dtype
    assert loaded.nbytes == values.nbytes
    assert np.shares_memory(loaded, values)
    np.testing.assert_array_equal(loaded, values)
