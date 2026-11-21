# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint dtype preservation, shard selection and missing tensor failures."""
import json
import numpy as np
import ml_dtypes
import pytest
from safetensors.numpy import save_file
from families.d3.weights import _open_safetensors, _load_tensor, load_projection


def test_original_bf16_projection(tmp_path):
    # Values above the FP16 range must survive the original BF16 path.
    matrix = np.array([[1, 100000], [0.5, -100000]], dtype=ml_dtypes.bfloat16)
    save_file({"projection.weight": matrix}, tmp_path / "model.safetensors")
    stack, readers = _open_safetensors(tmp_path)
    with stack:
        actual = load_projection(readers, "projection")
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, matrix.astype(np.float32).T)


def test_indexed_checkpoint_and_separate_readout(tmp_path):
    save_file({"a": np.array([2], np.float32)}, tmp_path / "shard.safetensors")
    save_file({"weight": np.zeros((255, 1024), np.float32)}, tmp_path / "readout.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "shard.safetensors"}}))
    stack, readers = _open_safetensors(tmp_path)
    with stack:
        np.testing.assert_array_equal(_load_tensor(readers, "a"), [2])
        with pytest.raises(ValueError, match="missing D3 tensor"):
            _load_tensor(readers, "weight")


@pytest.mark.parametrize("values", [np.array([1], np.float16), np.array([np.nan], np.float32)])
def test_reject_changed_precision_or_corrupt_weights(tmp_path, values):
    save_file({"x": values}, tmp_path / "model.safetensors")
    stack, readers = _open_safetensors(tmp_path)
    with stack, pytest.raises(ValueError):
        _load_tensor(readers, "x")
