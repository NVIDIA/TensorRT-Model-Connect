# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for bark."""

from families.bark.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_primary_task_is_text_to_audio():
    support = describe(ModelMetadata(config={"model_type": "bark"}, model_index={}))
    assert support is not None
    assert support.default_task == "text_to_audio"


def test_declared_tasks():
    support = describe(ModelMetadata(config={"model_type": "bark"}, model_index={}))
    assert support is not None
    assert support.tasks == ("text_to_audio",)


def test_unrelated_model_type_is_not_claimed():
    assert describe(ModelMetadata(config={"model_type": "gpt2"}, model_index={})) is None


def test_missing_model_type_is_not_claimed():
    assert describe(ModelMetadata(config={}, model_index={})) is None
