# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pytest
from families.llava.config import ModelConfig
from families.llava.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_exact_family_ownership():
    metadata = ModelMetadata(config={"model_type": "llava"}, model_index={})
    support = describe(metadata)
    assert support.default_task == "vision_language_generation"
    assert support.default_precision == "fp16"
    assert resolve_family(metadata)[0] == "llava"
    assert describe(ModelMetadata(config={"model_type": "llava_next"}, model_index={})) is None
    assert describe(ModelMetadata(config={"model_type": "clip"}, model_index={})) is None


def test_original_checkpoint_omitted_llama_dimensions(tmp_path):
    config = {"model_type": "llava", "text_config": {"model_type": "llama", "vocab_size": 32064},
              "vision_config": {"image_size": 336, "patch_size": 14}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    parsed = ModelConfig.from_dir(tmp_path)
    assert (parsed.hidden_size, parsed.num_hidden_layers, parsed.num_attention_heads) == (4096, 32, 32)
    config["vision_feature_layer"] = -1
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="penultimate"):
        ModelConfig.from_dir(tmp_path)


def test_other_decoder_is_rejected_before_llama_defaults_are_applied(tmp_path):
    config = {"model_type": "llava", "text_config": {"model_type": "qwen2"},
              "vision_config": {"model_type": "clip_vision_model"}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Llama decoder"):
        ModelConfig.from_dir(tmp_path)


@pytest.mark.parametrize("rope", [
    {"rope_scaling": {"type": "linear", "factor": 2.0}},
    {"rope_parameters": {"rope_type": "linear", "factor": 2.0}},
])
def test_scaled_rope_is_rejected_before_transformers_normalizes_it(tmp_path, rope):
    config = {"model_type": "llava", "text_config": {"model_type": "llama", **rope},
              "vision_config": {"image_size": 336, "patch_size": 14}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unscaled RoPE"):
        ModelConfig.from_dir(tmp_path)
