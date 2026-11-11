# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject model geometries whose declared operators cannot implement the checkpoint."""

import json
from pathlib import Path

import pytest

from families.lighton_ocr.config import Config


def checkpoint_config():
    return json.loads((Path(__file__).parent / "data/checkpoint-config.json").read_text())


@pytest.mark.parametrize("architecture", [
    "LightOnOCRForConditionalGeneration", "LightOnOcrForConditionalGeneration",
])
def test_original_legacy_metadata_preserves_both_model_geometries(tmp_path, architecture):
    raw = checkpoint_config()
    raw["architectures"] = [architecture]
    (tmp_path / "config.json").write_text(json.dumps(raw))
    config = Config.from_dir(tmp_path)
    assert config.raw["model_type"] == "mistral3"
    assert config.text_config["num_hidden_layers"] == 28
    assert config.text_config["num_attention_heads"] == 16
    assert config.text_config["num_key_value_heads"] == 8
    assert config.text_config["head_dim"] == 128
    assert config.vision_config["num_hidden_layers"] == 24
    assert config.vision_config["head_dim"] == 64
    assert config.spatial_merge_size == 2 and config.maximum_image_tokens == 3025
    assert (config.image_token_id, config.eos_token_id, config.pad_token_id) == (151655, 151645, 151643)


@pytest.mark.parametrize(("path", "value"), [
    (("architectures",), ["Mistral3ForConditionalGeneration"]),
    (("text_config", "model_type"), "qwen2"),
    (("vision_config", "model_type"), "siglip"),
    (("text_config", "num_key_value_heads"), 0),
    (("text_config", "num_attention_heads"), 15),
    (("text_config", "head_dim"), 127),
    (("text_config", "head_dim"), True),
    (("text_config", "attention_bias"), True),
    (("text_config", "tie_word_embeddings"), False),
    (("text_config", "use_qk_norm"), False),
    (("text_config", "layer_types"), ["full_attention"]),
    (("text_config", "rms_norm_eps"), float("nan")),
    (("vision_config", "num_channels"), 1),
    (("vision_config", "hidden_size"), 1025),
    (("vision_config", "image_size"), 1539),
    (("spatial_merge_size",), 3),
    (("vision_feature_layer",), 0),
    (("projector_hidden_act",), "relu"),
    (("multimodal_projector_bias",), True),
    (("image_token_id",), 151936),
    (("eos_token_id",), -1),
    (("pad_token_id",), True),
])
def test_incompatible_checkpoint_declarations_fail_before_graph_build(path, value):
    raw = checkpoint_config()
    target = raw
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        Config.from_dict(raw)


def test_axial_rope_requires_quarter_head_dimensions():
    raw = checkpoint_config()
    raw["vision_config"].update(hidden_size=96, head_dim=6)
    with pytest.raises(ValueError, match="axial rotary"):
        Config.from_dict(raw)


def test_nondefault_rotary_schedule_is_not_silently_treated_as_default():
    raw = checkpoint_config()
    raw["text_config"]["rope_parameters"]["rope_type"] = "dynamic"
    with pytest.raises(ValueError, match="default rotary"):
        Config.from_dict(raw)
