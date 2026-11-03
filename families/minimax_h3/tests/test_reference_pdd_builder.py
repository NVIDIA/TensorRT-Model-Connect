# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invalid projection requests fail before CUDA or TensorRT builder creation."""

from copy import deepcopy

import numpy as np
import pytest

from families.minimax_h3 import reference_pdd_builder as p


def banks():
    return {
        name: {
            "weight": np.ones((8, width, 17), dtype=np.float32),
            "bias": np.ones((8, width), dtype=np.float32),
        }
        for name, width in [("video", 96), ("audio", 32)]
    }


@pytest.mark.parametrize(
    "problem",
    ["missing", "steps", "width", "hidden", "bias", "nonfinite", "precision", "unrounded"],
)
def test_invalid_parameters_never_reach_a_GPU_builder(monkeypatch, problem):
    def forbidden(*_args):
        raise AssertionError("An invalid bank reached CUDA")

    monkeypatch.setattr(p.trt, "Builder", forbidden)
    value = banks()
    if problem == "missing":
        value.pop("audio")
    elif problem == "steps":
        value["video"]["weight"] = value["video"]["weight"][:-1]
    elif problem == "width":
        value["video"]["weight"] = value["video"]["weight"][:, :-1]
    elif problem == "hidden":
        value["video"]["weight"] = value["video"]["weight"][:, :, :-1]
    elif problem == "bias":
        value["audio"]["bias"] = value["audio"]["bias"][:, :-1]
    elif problem == "nonfinite":
        value["video"]["weight"][0, 0, 0] = np.nan
    elif problem == "precision":
        value["audio"]["weight"] = value["audio"]["weight"].astype(np.float16)
    else:
        value["video"]["weight"][0, 0, 0] = np.float32(1.0001)
    with pytest.raises(ValueError):
        p.build_projection_engine(value)


@pytest.mark.parametrize("rows", [(0, 1, 2), (1, 5, 2), (True, 1, 2), [1, 2, 3], (1, 2)])
def test_invalid_profile_never_reaches_a_GPU_builder(monkeypatch, rows):
    def forbidden(*_args):
        raise AssertionError("An invalid profile reached CUDA")

    monkeypatch.setattr(p.trt, "Builder", forbidden)
    with pytest.raises(ValueError):
        p.build_projection_engine(banks(), video_rows=rows)


def test_parameter_validation_retains_input_arrays():
    value = banks()
    original = deepcopy(value)
    assert p.validate_head_banks(value) == 17
    for name in value:
        for key in value[name]:
            np.testing.assert_array_equal(value[name][key], original[name][key])
