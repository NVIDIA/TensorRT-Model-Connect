# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for lerobot_act."""

from families.lerobot_act.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_primary_task_matches_the_semantic_runtime():
    support = describe(ModelMetadata(config={"type": "act"}, model_index={}))
    assert support is not None
    assert support.tasks == ("image_state_to_action_chunk",)
    assert support.default_task == "image_state_to_action_chunk"


def test_lerobot_policy_names_resolve_to_the_family():
    for model_type in ("act", "ACTPolicy", "lerobot-act"):
        support = describe(ModelMetadata(config={"model_type": model_type}, model_index={}))
        assert support is not None
        assert support.default_task == "image_state_to_action_chunk"


def test_unrelated_identity_is_not_claimed():
    assert describe(ModelMetadata(config={"model_type": "unrelated-model"}, model_index={})) is None
