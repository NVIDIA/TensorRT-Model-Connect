# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prove the graph's grid bounds preserve the family's supported inputs."""

from types import SimpleNamespace

import numpy as np
import pytest

from families.moge import model


class _ArrayGraph(model._NativeMogeGraph):
    def __init__(self, *, fast_path=False):
        operations = SimpleNamespace(
            MIN=np.minimum, MAX=np.maximum, DIV=np.divide, PROD=np.multiply
        )
        trt = SimpleNamespace(
            ElementWiseOperation=operations,
            UnaryOperation=SimpleNamespace(SQRT=np.sqrt, ROUND=np.rint),
            InterpolationMode=SimpleNamespace(LINEAR="linear"),
            float32=np.float32,
            int64=np.int64,
        )
        super().__init__(trt, None, {}, fast_path=fast_path)
        self.bounds = []

    def shape(self, image_shape, name):
        return image_shape

    def shape_index(self, shape, index, name):
        return shape[index]

    def cast(self, value, dtype, name):
        return np.asarray(value, dtype=dtype)

    def constant(self, value, *, dtype=np.float32, name="constant"):
        return np.asarray(value, dtype=dtype)

    def shape_value(self, value, name):
        return np.int64(value)

    def binary(self, left, right, operation, name):
        if operation in (np.minimum, np.maximum):
            self.bounds.append(operation)
        return operation(left, right)

    def unary(self, value, operation, name):
        return operation(value)

    def shape_concat(self, values, name):
        return values

    def resize(self, tensor, output_shape, interpolation, name):
        raise _AtImageResize(output_shape)


class _AtImageResize(Exception):
    def __init__(self, dimensions):
        super().__init__("encoder image-grid boundary")
        self.dimensions = dimensions


@pytest.mark.parametrize(
    "fast_path,height_range,width_range",
    [
        (False, *((model._MIN_IMAGE_SIZE, model._MAX_IMAGE_SIZE),) * 2),
        (
            True,
            (model._FAST_MIN_IMAGE_HEIGHT, model._FAST_MAX_IMAGE_HEIGHT),
            (model._FAST_MIN_IMAGE_WIDTH, model._FAST_MAX_IMAGE_WIDTH),
        ),
    ],
    ids=["fp32-profile", "fp16-profile"],
)
def test_grid_bounds_preserve_all_supported_image_sizes(fast_path, height_range, width_range):
    graph = _ArrayGraph(fast_path=fast_path)
    widths = np.arange(width_range[0], width_range[1] + 1, dtype=np.float32)
    checked = 0
    # Work a row at a time to cover both profiles without a large allocation.
    for height in range(height_range[0], height_range[1] + 1):
        aspect = widths / np.float32(height)
        supported = (aspect >= np.float32(0.5)) & (aspect <= np.float32(2.0))
        image_widths = widths[supported]
        aspect = aspect[supported]
        if not len(aspect):
            continue
        base_h = np.rint(np.sqrt(np.float32(model._NUM_TOKENS) / aspect)).astype(np.int64)
        base_w = np.rint(np.sqrt(np.float32(model._NUM_TOKENS) * aspect)).astype(np.int64)
        graph.bounds.clear()
        with pytest.raises(_AtImageResize) as boundary:
            graph.encoder((1, 3, height, image_widths))
        dimensions = boundary.value.dimensions
        assert dimensions[:2] == [1, 3]
        np.testing.assert_array_equal(dimensions[2], base_h * model._PATCH)
        np.testing.assert_array_equal(dimensions[3], base_w * model._PATCH)
        assert graph.bounds.count(np.minimum) == (0 if fast_path else 2)
        assert graph.bounds.count(np.maximum) == (0 if fast_path else 2)
        checked += len(aspect)
    expected = sum(
        max(0, min(width_range[1], 2 * height) - max(width_range[0], (height + 1) // 2) + 1)
        for height in range(height_range[0], height_range[1] + 1)
    )
    assert checked == expected > 0
