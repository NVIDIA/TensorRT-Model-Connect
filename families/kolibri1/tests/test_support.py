# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family
import numpy as np
import tensorrt as trt
from families.kolibri1.model import Graph


def test_kolibri1_checkpoint_has_one_owner():
    family, support = resolve_family(ModelMetadata({"model_type": "kolibri1"}, {}))
    assert family == "kolibri1"
    assert support.tasks == ("text_continuation",)
    assert support.default_task == "text_continuation"


def test_production_attention_scale_uses_fp32_opmath():
    class NumpyGraph(Graph):
        def __init__(self):
            pass

        def cast(self, value, dtype):
            return value.astype(np.float32 if dtype == trt.float32 else np.float16)

        def const(self, value, dtype=np.float16):
            return np.asarray(value, dtype=dtype)

        def mul(self, left, right):
            return left * right

    values = np.linspace(-16, 16, 65536, dtype=np.float32).astype(np.float16)
    expected = (values.astype(np.float32) * np.float32(128**-0.5)).astype(np.float16)
    rounded_scale = values * np.float16(128**-0.5)
    assert np.any(expected != rounded_scale), "fixture must detect a half-rounded scale"
    np.testing.assert_array_equal(NumpyGraph().scale_attention(values, 128), expected)
