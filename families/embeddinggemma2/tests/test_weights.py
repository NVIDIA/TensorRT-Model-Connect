# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import json
import pytest
import torch
from safetensors.torch import save_file

from families.embeddinggemma2.config import Config
from families.embeddinggemma2.weights import load_weights
from families.embeddinggemma2.tests.test_config import tiny_config


def checkpoint(config):
    return {
        "language_model." + key: torch.full(shape, 0.75, dtype=torch.bfloat16)
        for key, shape in config.weight_shapes().items()
    }


def test_loads_only_text_parameters_and_preserves_bfloat16_values(tmp_path):
    config = Config.from_dict(tiny_config())
    tensors = checkpoint(config)
    tensors["vision_tower.unused"] = torch.ones(1)
    save_file(tensors, tmp_path / "model.safetensors")
    actual = load_weights(tmp_path, config)
    assert set(actual) == set(config.weight_shapes())
    for key, shape in config.weight_shapes().items():
        assert actual[key].shape == shape
        assert actual[key].dtype == np.float32
        np.testing.assert_array_equal(actual[key], np.full(shape, 0.75))


@pytest.mark.parametrize("failure", ["missing", "shape"])
def test_rejects_missing_or_wrong_shape_including_saved_layer_scalar(tmp_path, failure):
    config = Config.from_dict(tiny_config())
    tensors = checkpoint(config)
    key = "language_model.layers.1.layer_scalar"
    if failure == "missing":
        del tensors[key]
    else:
        tensors[key] = torch.ones(2)
    save_file(tensors, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="layer_scalar"):
        load_weights(tmp_path, config)


def test_sharded_checkpoint_does_not_open_unneeded_multimodal_shards(tmp_path):
    config = Config.from_dict(tiny_config())
    tensors = checkpoint(config)
    save_file(tensors, tmp_path / "text.safetensors")
    mapping = {key: "text.safetensors" for key in tensors}
    mapping["vision_tower.unused"] = "unavailable-vision.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    assert set(load_weights(tmp_path, config)) == set(config.weight_shapes())
