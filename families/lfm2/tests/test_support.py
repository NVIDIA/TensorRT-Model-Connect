# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint identity and build-task discovery without Torch or TensorRT."""

from tensorrt_model_connect.model_support import ModelMetadata

from families.lfm2.support import describe


def test_checkpoint_identity_and_default_task():
    metadata = ModelMetadata({"model_type": "lfm2", "architectures": ["Lfm2ForCausalLM"]}, {})
    support = describe(metadata)
    assert support is not None
    assert support.default_task == "text_generation"
    assert support.tasks == ("text_generation",)


def test_unrelated_checkpoint_is_not_claimed():
    assert describe(ModelMetadata({"model_type": "unrelated_model"}, {})) is None
