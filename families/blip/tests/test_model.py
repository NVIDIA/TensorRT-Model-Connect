# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

from families.blip.model import build, resolve


@pytest.fixture
def checkpoint(tmp_path):
    common = {
        "hidden_size": 8,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "layer_norm_eps": 1e-5,
        "hidden_act": "gelu",
    }
    config = {
        "model_type": "blip",
        "architectures": ["BlipForConditionalGeneration"],
        "vision_config": {**common, "image_size": 32, "patch_size": 16},
        "text_config": {**common, "is_decoder": True},
    }
    processor = {"size": 32, "image_mean": [0.5] * 3, "image_std": [0.25] * 3}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(processor))
    return tmp_path


def test_resolves_checkpoint_owned_parameters(checkpoint):
    cfg, processor = resolve(checkpoint)
    assert cfg["vision_config"]["patch_size"] == 16
    assert processor["image_std"] == [0.25] * 3


@pytest.mark.parametrize(
    "file,section,key,value",
    [
        ("config.json", "vision_config", "hidden_act", "quick_gelu"),
        ("config.json", "text_config", "is_decoder", False),
        ("config.json", "text_config", "position_embedding_type", "relative_key"),
        ("config.json", "vision_config", "num_attention_heads", 3),
        ("config.json", "vision_config", "patch_size", 7),
        ("config.json", "text_config", "layer_norm_eps", 0),
        ("preprocessor_config.json", None, "resample", 2),
        ("preprocessor_config.json", None, "size", 48),
        ("preprocessor_config.json", None, "do_resize", False),
        ("preprocessor_config.json", None, "do_normalize", False),
        ("preprocessor_config.json", None, "image_std", [0, 1, 1]),
    ],
)
def test_unsupported_semantics_fail_before_build(checkpoint, file, section, key, value):
    path = checkpoint / file
    cfg = json.loads(path.read_text())
    (cfg[section] if section else cfg)[key] = value
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        resolve(checkpoint)


def test_precision_is_not_silently_ignored():
    with pytest.raises(ValueError, match="precision=fp32"):
        build(SimpleNamespace(task="images_text_to_text", precision="fp16", backend="trt"), None)
