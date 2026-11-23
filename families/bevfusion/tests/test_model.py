# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from families.bevfusion.model import Weights, build_engine, grid_coordinates


def test_grid_preserves_xy_order_and_source_pixel_centers():
    grid = grid_coordinates()
    assert grid.shape == (1, 200, 200, 2)
    assert grid.dtype == np.float32
    expected = (np.array([-49.75, 49.75]) + 51.2) / 102.4 * 2 - 1
    np.testing.assert_allclose(grid[0, 0, [0, -1], 0], expected, atol=1e-7)
    assert grid[0, 0, 1, 0] > grid[0, 0, 0, 0]
    assert grid[0, 0, 1, 1] == grid[0, 0, 0, 1]
    np.testing.assert_array_equal(grid[0, :, 0, 1], grid[0, 0, :, 0])


def test_weights_reject_wrong_shape_nonfinite_and_unused(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "checkpoint.pth"
    torch.save(
        {"state_dict": {"fuser.0.weight": torch.ones(2), "encoders.other": torch.ones(1)}}, path
    )
    weights = Weights(path)
    with pytest.raises(ValueError, match="shape"):
        weights.get("fuser.0.weight", (3,))
    with pytest.raises(ValueError, match="unexpected BEV weights"):
        weights.finish()
    np.testing.assert_array_equal(weights.get("fuser.0.weight", (2,)), np.ones(2))
    weights.finish()
    torch.save({"state_dict": {"fuser.0.weight": torch.tensor([float("nan")])}}, path)
    with pytest.raises(ValueError, match="non-finite"):
        Weights(path).get("fuser.0.weight", (1,))


def test_unsupported_precision_fails_before_reading_checkpoint(tmp_path):
    pytest.importorskip("tensorrt")
    with pytest.raises(ValueError, match="only fp32 and fp16"):
        build_engine(tmp_path / "missing", "bf16")
