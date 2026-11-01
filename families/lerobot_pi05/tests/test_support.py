# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for LeRobot Pi 0.5."""

from families.lerobot_pi05.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_pi05_policy_is_claimed_with_its_one_task():
    support = describe(ModelMetadata(config={"type": "pi05"}, model_index={}))
    assert support is not None
    assert support.tasks == ("images_state_text_to_action_chunk",)
    assert support.default_task == "images_state_text_to_action_chunk"
    assert support.default_precision == "bf16"


def test_resolution_picks_exactly_this_family():
    family, resolved = resolve_family(ModelMetadata(config={"type": "pi05"}, model_index={}))
    assert family == "lerobot_pi05"
    assert resolved.default_task == "images_state_text_to_action_chunk"


def test_other_lerobot_policies_are_not_claimed():
    # Pi 0, Pi 0-FAST and the other policy types are separate families.
    for config in ({"type": "pi0"}, {"type": "pi0_fast"}, {"type": "act"}, {}):
        assert describe(ModelMetadata(config=config, model_index={})) is None
