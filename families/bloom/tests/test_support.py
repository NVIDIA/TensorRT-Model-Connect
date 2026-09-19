# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover the family's semantic Task without importing TensorRT."""

from families.bloom.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_semantic_primary_task():
    support = describe(ModelMetadata(config={"model_type": "bloom"}, model_index={}))
    assert support is not None
    assert support.tasks == ("text_continuation",)
    assert support.default_task == "text_continuation"
