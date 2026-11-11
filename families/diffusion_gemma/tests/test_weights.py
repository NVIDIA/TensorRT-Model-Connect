# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import struct

import ml_dtypes
import numpy as np
import pytest

from families.diffusion_gemma.config import Config
from families.diffusion_gemma.weights import Checkpoint, _is_finite


def config():
    return Config.from_dict(json.loads((Path(__file__).parent / "data/diffusiongemma-26b-a4b-config.json").read_text()))


def write_shard(path, values, dtype="BF16"):
    values = np.asarray(values, dtype=ml_dtypes.bfloat16 if dtype == "BF16" else np.float32)
    header = json.dumps({"test.weight": {
        "dtype": dtype, "shape": list(values.shape), "data_offsets": [0, values.nbytes],
    }}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + values.tobytes())


def test_bf16_is_mapped_without_loading_torch_or_converting_storage(tmp_path):
    write_shard(tmp_path / "model.safetensors", [[1, -2], [0.25, 3.5]])
    reader = Checkpoint(tmp_path, config())
    result = reader.tensor("test.weight", (2, 2))
    assert isinstance(result, np.memmap)
    assert result.dtype == np.dtype(ml_dtypes.bfloat16)
    np.testing.assert_array_equal(result.astype(np.float32), [[1, -2], [0.25, 3.5]])


def test_wrong_weight_shape_is_an_error(tmp_path):
    write_shard(tmp_path / "model.safetensors", [1, 2])
    with pytest.raises(ValueError, match="expected"):
        Checkpoint(tmp_path, config()).tensor("test.weight", (2, 1))


def test_truncated_tensor_data_is_an_error(tmp_path):
    path = tmp_path / "model.safetensors"
    write_shard(path, [1, 2])
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="byte range"):
        Checkpoint(tmp_path, config()).tensor("test.weight", (2,))


def test_index_cannot_escape_checkpoint_directory(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"test.weight": "../outside.safetensors"},
    }))
    with pytest.raises(ValueError, match="inside the checkpoint"):
        Checkpoint(tmp_path, config()).tensor("test.weight", (2,))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_bf16_special_exponents_are_not_accepted_as_finite(value):
    assert not _is_finite(np.array([0, value, 1], dtype=ml_dtypes.bfloat16))


def test_bf16_finite_check_keeps_subnormals_and_negative_values():
    assert _is_finite(np.array([0, -1, 1e-40, 3.3e38], dtype=ml_dtypes.bfloat16))
