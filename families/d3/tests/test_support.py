# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the global resolver, including both Qwen owners."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family, FamilyResolutionError

CONFIG = json.loads((Path(__file__).parent / "fixtures/config.json").read_text())


def metadata(config=None, files=("decision_config.json", "readout.safetensors")):
    return ModelMetadata(deepcopy(CONFIG if config is None else config), {}, files)


def test_edge_has_exactly_one_owner():
    family, support = resolve_family(metadata())
    assert family == "d3"
    assert support.tasks == ("structured_decision",)
    assert support.default_task == "structured_decision"
    assert support.default_precision == "bf16"


def test_plain_qwen_retains_owner():
    assert resolve_family(metadata(files=()))[0] == "qwen3_5"


@pytest.mark.parametrize("hidden,layers,gate", [(1024, 24, None), (5120, 64, "swish")])
def test_unknown_d3_topologies_do_not_fall_back_to_generation(hidden, layers, gate):
    c = deepcopy(CONFIG)
    c["text_config"].update(hidden_size=hidden, num_hidden_layers=layers)
    if gate:
        c["text_config"]["output_gate_type"] = gate
        c["text_config"].pop("mlp_only_layers")
    with pytest.raises(FamilyResolutionError, match="no family"):
        resolve_family(metadata(c))


def test_missing_readout_cannot_fall_back_to_qwen():
    with pytest.raises(FamilyResolutionError, match="no family"):
        resolve_family(metadata(files=("decision_config.json",)))


@pytest.mark.parametrize("name", ["edge", "lite", "nano", "mini", "flash"])
def test_released_dense_profiles_have_one_owner(name):
    path = Path(__file__).parent / "fixtures/profiles" / name / "config.json"
    config = json.loads(path.read_text())
    assert resolve_family(metadata(config))[0] == "d3"
    assert resolve_family(metadata(config, files=()))[0] == "qwen3_5"
