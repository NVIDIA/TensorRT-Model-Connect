# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from tensorrt_model_connect.families.boltz2.feature_bundle import (
    FEATURE_NAMES,
    INT32_FEATURE_NAMES,
    deserialize_features,
    serialize_features,
)


def test_feature_section_round_trip_is_stable():
    features = {
        name: np.asarray(
            [index, index + 1],
            dtype=np.int64 if name in INT32_FEATURE_NAMES else np.float32,
        )
        for index, name in enumerate(FEATURE_NAMES)
    }
    first = serialize_features(features)
    second = serialize_features(features)
    assert first == second
    decoded = deserialize_features(first)
    assert tuple(decoded) == FEATURE_NAMES
    for name in FEATURE_NAMES:
        assert np.array_equal(decoded[name], features[name])
        expected = np.dtype("int32" if name in INT32_FEATURE_NAMES else "float32")
        assert decoded[name].dtype == expected


def test_feature_section_rejects_int64_overflow():
    features = {
        name: np.zeros(1, dtype=np.int32 if name in INT32_FEATURE_NAMES else np.float32)
        for name in FEATURE_NAMES
    }
    features["res_type"] = np.asarray([np.iinfo(np.int32).max + 1], dtype=np.int64)
    with pytest.raises(ValueError, match="outside the INT32 range"):
        serialize_features(features)


def test_feature_section_rejects_missing_tensor():
    with pytest.raises(ValueError, match="feature set is missing"):
        serialize_features({})


def test_feature_section_rejects_truncation():
    features = {
        name: np.zeros(1, dtype=np.int32 if name in INT32_FEATURE_NAMES else np.float32)
        for name in FEATURE_NAMES
    }
    with pytest.raises(ValueError, match="truncated"):
        deserialize_features(serialize_features(features)[:-1])
