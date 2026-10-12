# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint stop tokens must survive the GLM-ASR bundle contract."""

import json

import pytest

from families.glmasr.config import ModelConfig


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"text_config": {"eos_token_id": [59246, 59253, 59255]}}, [59246, 59253, 59255]),
        ({"eos_token_id": 17}, [17]),
        ({"text_config": {"eos_token_id": 29}, "eos_token_id": 17}, [29]),
        ({"text_config": {}, "eos_token_id": [17, 29]}, [17, 29]),
        ({"text_config": {"eos_token_id": []}, "eos_token_id": 17}, [17]),
    ],
)
def test_checkpoint_preserves_all_stop_tokens(raw, expected):
    assert ModelConfig.from_json(json.dumps(raw)).eos_token_ids == expected


@pytest.mark.parametrize("raw", [{}, {"text_config": {"eos_token_id": []}}])
def test_checkpoint_requires_stop_tokens(raw):
    with pytest.raises(ValueError, match="no eos_token_id"):
        _ = ModelConfig.from_json(json.dumps(raw)).eos_token_ids
