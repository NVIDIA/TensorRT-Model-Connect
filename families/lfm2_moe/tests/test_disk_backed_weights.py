# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disk-backed constants retain their exact FP32 representation and ownership."""

import tempfile

import numpy as np
import pytest

from families.lfm2_moe.checkpoint_mapper import WeightDict


@pytest.fixture
def storage(tmp_path):
    with tempfile.TemporaryFile(dir=tmp_path) as stream:
        yield stream


def test_disk_weights_preserve_bits_layout_and_metadata(storage):
    # Include signed zero, BF16 subnormals, infinities and a NaN payload.
    words = np.array(
        [0, 0x80000000, 0x00010000, 0x3F810000, 0xBF810000, 0x7F800000, 0xFF800000, 0x7FC10000],
        dtype=np.uint32,
    ).reshape(2, 4)
    source = words.view(np.float32).T
    weights = WeightDict(storage_file=storage)
    weights["projection"] = source
    weights["_layer_types"] = ["conv"]
    stored = weights["projection"]
    assert isinstance(stored, np.memmap)
    assert stored.dtype == np.float32
    assert stored.flags.c_contiguous
    assert not stored.flags.writeable
    np.testing.assert_array_equal(stored.view(np.uint32), words.T)
    assert weights["_layer_types"] == ["conv"]
    with pytest.raises(ValueError, match="read-only"):
        stored[0, 0] = 1


def test_disk_weights_do_not_alias_input_or_reused_key(storage):
    source = np.arange(8, dtype=np.float32).reshape(2, 4)
    weights = WeightDict(storage_file=storage)
    weights["projection"] = source
    first = weights["projection"]
    source.fill(17)
    weights["projection"] = source
    np.testing.assert_array_equal(first, np.arange(8, dtype=np.float32).reshape(2, 4))
    np.testing.assert_array_equal(weights["projection"], source)
    assert first.offset != weights["projection"].offset


def test_default_weight_storage_keeps_existing_array_object():
    source = np.arange(8, dtype=np.float32).reshape(2, 4).T
    weights = WeightDict()
    weights["projection"] = source
    assert weights["projection"] is source


@pytest.mark.parametrize("fail", [False, True])
def test_mapped_values_survive_storage_owner_cleanup(tmp_path, fail):
    retained = []
    try:
        with tempfile.TemporaryFile(dir=tmp_path) as stream:
            weights = WeightDict(storage_file=stream)
            weights["prefix"] = np.array([3], dtype=np.float32)
            weights["projection"] = np.arange(12, dtype=np.float32).reshape(3, 4)
            retained.append(weights["projection"])
            assert retained[0].offset == 4
            if fail:
                raise RuntimeError("engine construction failed")
    except RuntimeError as error:
        assert fail and str(error) == "engine construction failed"
        assert error.__traceback__ is not None
    assert stream.closed
    assert not list(tmp_path.iterdir())
    np.testing.assert_array_equal(retained[0], np.arange(12, dtype=np.float32).reshape(3, 4))
