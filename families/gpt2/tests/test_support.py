# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for gpt2."""

from families.gpt2.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_primary_task_is_text_continuation():
    support = describe(ModelMetadata(config={"model_type": "gpt2"}, model_index={}))
    assert support is not None
    assert support.default_task == "text_continuation"
    assert support.tasks == ("text_continuation",)
    assert support.default_precision == "fp32"


def test_unrelated_model_type_is_not_claimed():
    assert describe(ModelMetadata(config={"model_type": "bert"}, model_index={})) is None
    assert describe(ModelMetadata(config={}, model_index={})) is None
