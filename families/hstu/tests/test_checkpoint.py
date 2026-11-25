# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete checkpoint, profile and input-boundary contracts."""

import numpy as np
import pytest
from ..checkpoint import expected_shapes, validate_weights
from ..config import Config, ServingProfile
from ..reference import example_inputs


def test_official_architecture_and_public_inputs():
    shapes = expected_shapes()
    assert len(shapes) == 86
    assert sum(np.prod(v) for v in shapes.values()) == 16693594
    c = Config()
    schema = c.runtime(4, "bf16")
    assert schema["stages"][0]["outputs"][0]["shape"] == [-1, 5, 4, 513, 128]
    inputs = example_inputs(c, 4)
    assert set(inputs) == {x["name"] for x in schema["inputs"]}
    for spec in schema["inputs"]:
        assert inputs[spec["name"]].shape == (4, *spec["shape"][1:])
        assert str(inputs[spec["name"]].dtype) == spec["dtype"]
    assert schema["outputs"] == [dict(name="scores", dtype="float32", shape=[-1, 128])]


@pytest.mark.parametrize(
    "profile",
    [
        dict(history_length=0),
        dict(candidates=2049),
        dict(batching="other"),
        dict(min_batch_size=True),
    ],
)
def test_invalid_serving_profile_is_rejected(profile):
    with pytest.raises(ValueError):
        ServingProfile(**profile)


def test_larger_profile_preserves_official_architecture():
    c = Config(serving=ServingProfile(history_length=2048, candidates=256))
    assert c.runtime(4, "fp32")["stages"][0]["outputs"][0]["shape"] == [-1, 5, 4, 2049, 128]
    with pytest.raises(ValueError):
        c.runtime(0, "bf16")
    with pytest.raises(ValueError):
        c.runtime(4, "fp16")


def test_converter_rejects_partial_models():
    with pytest.raises(ValueError, match="86"):
        validate_weights({})


def test_unknown_bf16_policy_is_rejected():
    with pytest.raises(ValueError, match="BF16 policy"):
        Config(bf16_policy="unknown")


def test_bf16_policy_keeps_reduced_operations_explicit():
    config = Config(fp32_block_indices=tuple(range(5)), bf16_policy="conservative")
    with pytest.raises(ValueError, match="reduced-precision"):
        config.runtime(4, "bf16")
    config = Config(fp32_block_indices=tuple(range(5)), bf16_policy="expanded")
    assert config.runtime(4, "bf16")["bf16_extra_ops"] == [
        "candidate_av",
        "item_mlp",
        "time_projection",
        "prediction_hidden",
    ]
    assert config.runtime(4, "fp32")["bf16_extra_ops"] == []
