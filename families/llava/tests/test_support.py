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
    assert parsed.rope_theta == 10000.0
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
    {"rope_parameters": {"type": "linear", "factor": 2.0}},
])
def test_scaled_rope_is_rejected_after_transformers_normalization(tmp_path, rope):
    config = {"model_type": "llava", "text_config": {"model_type": "llama", **rope},
              "vision_config": {"image_size": 336, "patch_size": 14}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unscaled RoPE"):
        ModelConfig.from_dir(tmp_path)


@pytest.mark.parametrize("rope", [
    {"rope_theta": 20000.0},
    {"rope_parameters": {"rope_type": "default", "rope_theta": 20000.0}},
    {"rope_parameters": {"type": "default", "rope_theta": 20000.0}},
])
def test_unscaled_rope_retains_the_normalized_base(tmp_path, rope):
    config = {"model_type": "llava", "text_config": {"model_type": "llama", **rope},
              "vision_config": {"image_size": 336, "patch_size": 14}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert ModelConfig.from_dir(tmp_path).rope_theta == 20000.0


@pytest.mark.parametrize("tokenizer_kind", ["missing", "directory"])
def test_native_tokenizer_is_required_before_config_or_engines(tmp_path, monkeypatch, tokenizer_kind):
    from tensorrt_model_connect import BuildRequest
    from families.llava import model

    if tokenizer_kind == "directory":
        (tmp_path / "tokenizer.json").mkdir()

    def config_must_not_be_loaded(*_args):
        raise AssertionError("A missing native tokenizer must fail before checkpoint loading")

    monkeypatch.setattr(model.ModelConfig, "from_dir", config_must_not_be_loaded)
    request = BuildRequest(model_dir=tmp_path, output_path=tmp_path / "unused.bundle",
                           family="llava", task="vision_language_generation", precision="fp16")
    with pytest.raises(ValueError, match="require tokenizer.json"):
        model.build(request, object())
