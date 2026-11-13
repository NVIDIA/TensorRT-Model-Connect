# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free identity and default-task tests for HSTU support."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata


def _metadata(config):
    return ModelMetadata(config=config, model_index={}, files=("config.json", "model.safetensors"))


def test_matches_the_synthetic_hstu_ranking_checkpoint():
    from families.hstu.support import describe

    support = describe(
        _metadata({"model_type": "hstu_ranking", "architectures": ["HSTURankingGR"]})
    )
    assert support == FamilySupport(
        tasks=("structured_decision",),
        default_task="structured_decision",
        default_precision="bf16",
    )
    assert support.default_task == "structured_decision"


def test_rejects_unrelated_models():
    from families.hstu.support import describe

    assert describe(_metadata({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]})) is None
    assert describe(_metadata({})) is None
