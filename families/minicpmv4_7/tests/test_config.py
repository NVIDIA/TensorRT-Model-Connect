# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from families.minicpmv4_7.config import ModelConfig

DATA = Path(__file__).parent / "data"


def config(name):
    return json.loads((DATA / f"minicpm-v-4.7-{name}-config.json").read_text(encoding="utf-8"))


def test_official_dense_and_moe_geometry():
    dense = ModelConfig.from_dict(config("1b"))
    assert not dense.text.is_moe
    assert dense.text.intermediate_size == 3584
    assert (dense.text.hidden_size, dense.text.vocab_size) == (1024, 248094)
    assert (dense.text.num_linear_layers, dense.text.num_full_layers) == (18, 6)
    moe = ModelConfig.from_dict(config("35b"))
    assert moe.text.is_moe
    assert (moe.text.num_experts, moe.text.num_experts_per_tok) == (256, 8)
    assert (moe.text.num_linear_layers, moe.text.num_full_layers) == (30, 10)
    assert dense.vision == moe.vision


@pytest.mark.parametrize("width", [None, 0, -1, True, 3.5])
def test_dense_requires_positive_integer_width(width):
    raw = config("1b")
    raw["text_config"]["intermediate_size"] = width
    with pytest.raises(ValueError, match="intermediate_size"):
        ModelConfig.from_dict(raw)


@pytest.mark.parametrize("model_type", ["qwen3", "qwen3_5", None])
def test_other_decoder_architectures_are_rejected(model_type):
    raw = config("1b")
    raw["text_config"]["model_type"] = model_type
    with pytest.raises(ValueError, match="language model"):
        ModelConfig.from_dict(raw)


def test_moe_routing_validation_remains_strict():
    raw = config("35b")
    raw["text_config"]["num_experts_per_tok"] = 257
    with pytest.raises(ValueError, match="experts"):
        ModelConfig.from_dict(raw)
