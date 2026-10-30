# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for LeRobot GR00T N1.7."""

from families.lerobot_groot.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_groot_policy_is_claimed_with_its_one_task():
    support = describe(ModelMetadata(config={"type": "groot"}, model_index={}))
    assert support is not None
    assert support.tasks == ("images_state_text_to_action_chunk",)
    assert support.default_task == "images_state_text_to_action_chunk"
    assert support.default_precision == "bf16"


def test_resolution_picks_exactly_this_family():
    family, resolved = resolve_family(ModelMetadata(config={"type": "groot"}, model_index={}))
    assert family == "lerobot_groot"
    assert resolved.default_task == "images_state_text_to_action_chunk"


def test_other_lerobot_policies_are_not_claimed():
    for config in ({"type": "pi05"}, {"type": "act"}, {"type": "smolvla"}, {}):
        assert describe(ModelMetadata(config=config, model_index={})) is None
