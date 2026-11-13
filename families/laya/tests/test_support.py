# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from families.laya.cli import BuildRequest
from families.laya.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_released_checkpoint_has_one_decision_owner():
    metadata = ModelMetadata(
        {"model_type": "laya"}, {}, ("rl_agent_config.json", "model.safetensors")
    )
    family, support = resolve_family(metadata)
    assert family == "laya"
    assert support.tasks == ("structured_decision",)
    assert support.default_precision == "bf16"


def test_modernbert_is_not_a_laya_checkpoint():
    assert describe(ModelMetadata({"model_type": "modernbert"}, {})) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"variant": "other"},
        {"precision": "fp16"},
        {"backend": "torch"},
        {"task": "text_generation"},
        {"max_sequence_length": 0},
        {"max_sequence_length": 8193},
        {"max_sequence_length": True},
        {"max_batch_size": 0},
        {"max_options": True},
    ],
)
def test_invalid_build_policy_fails_closed(tmp_path, kwargs):
    with pytest.raises(ValueError):
        BuildRequest(tmp_path, **kwargs)
