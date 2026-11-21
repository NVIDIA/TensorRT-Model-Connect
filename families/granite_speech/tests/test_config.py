# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry resolution, refusals, tensor inventory and sample limits (CPU, no checkpoint)."""

from __future__ import annotations

import copy
import json

import pytest

from families.granite_speech import checkpoint, config as config_module

# The geometry of ibm-granite/granite-4.0-1b-speech (config.json and preprocessor_config.json).
_CONFIG = {
    "architectures": ["GraniteSpeechForConditionalGeneration"],
    "model_type": "granite_speech",
    "audio_token_index": 100352,
    "downsample_rate": 5,
    "window_size": 15,
    "has_lora_adapter": False,
    "tie_word_embeddings": False,
    "encoder_config": {
        "model_type": "granite_speech_encoder", "context_size": 200, "conv_expansion_factor": 2,
        "conv_kernel_size": 15, "dim_head": 128, "feedforward_mult": 4, "hidden_dim": 1024,
        "input_dim": 160, "max_pos_emb": 512, "num_heads": 8, "num_layers": 16, "output_dim": 348,
    },
    "projector_config": {
        "model_type": "blip_2_qformer", "cross_attention_frequency": 1, "encoder_hidden_size": 1024,
        "hidden_act": "gelu", "hidden_size": 1024, "intermediate_size": 4096, "layer_norm_eps": 1e-12,
        "num_attention_heads": 16, "num_hidden_layers": 2, "use_qformer_text_input": False,
    },
    "text_config": {
        "model_type": "granite", "attention_bias": False, "attention_multiplier": 0.0078125,
        "embedding_multiplier": 12.0, "eos_token_id": 100257, "hidden_act": "silu", "hidden_size": 2048,
        "intermediate_size": 4096, "logits_scaling": 8.0, "max_position_embeddings": 4096,
        "mlp_bias": False, "num_attention_heads": 16, "num_hidden_layers": 40, "num_key_value_heads": 4,
        "residual_multiplier": 0.22, "rms_norm_eps": 1e-05,
        "rope_parameters": {"rope_theta": 10000, "rope_type": "default"}, "rope_scaling": None,
        "rope_theta": 10000.0, "vocab_size": 100353,
    },
}
_PREPROCESSOR = {
    "feature_extractor_type": "GraniteSpeechFeatureExtractor",
    "melspec_kwargs": {"hop_length": 160, "n_fft": 512, "n_mels": 80, "sample_rate": 16000, "win_length": 400},
    "projector_downsample_rate": 5, "projector_window_size": 15, "sampling_rate": 16000,
}


def _write(tmp_path, cfg=None, pre=None):
    (tmp_path / "config.json").write_text(json.dumps(cfg or _CONFIG))
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(pre or _PREPROCESSOR))
    return tmp_path


def _variant(path: str, value):
    cfg = copy.deepcopy(_CONFIG)
    node = cfg
    keys = path.split(".")
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    return cfg


def test_resolves_the_checkpoint_geometry(tmp_path):
    cfg = config_module.resolve(_write(tmp_path))
    assert cfg.encoder.num_layers == 16 and cfg.encoder.context_size == 200
    assert cfg.projector.num_queries == 3
    assert cfg.text.kv_width == 512 and cfg.text.head_dim == 128
    assert cfg.text.rope_theta == 10000.0 and cfg.text.eos_token_id == 100257
    assert cfg.front_end.n_fft == 512 and cfg.front_end.win_length == 400 and cfg.front_end.n_mels == 80
    assert cfg.audio_token_id == 100352


@pytest.mark.parametrize("path, value, fragment", [
    ("has_lora_adapter", True, "LoRA adapter"),
    ("architectures", ["GraniteSpeechForCTC"], "GraniteSpeechForConditionalGeneration"),
    ("tie_word_embeddings", True, "untied"),
    ("window_size", 20, "projector_window_size"),
    ("encoder_config.model_type", "conformer", "granite_speech_encoder"),
    ("encoder_config.context_size", 600, "context_size"),
    ("encoder_config.input_dim", 128, "two stacked mel frames"),
    ("projector_config.use_qformer_text_input", True, "query-only"),
    ("projector_config.hidden_act", "gelu_new", "exact GELU"),
    ("projector_config.encoder_hidden_size", 512, "encoder_hidden_size"),
    ("text_config.rope_scaling", {"rope_type": "yarn", "factor": 4.0}, "default rotary"),
    ("text_config.rope_parameters", {"rope_type": "yarn", "factor": 4.0}, "default rotary"),
    ("text_config.hidden_act", "gelu", "SiLU"),
    ("text_config.attention_bias", True, "without bias"),
    ("text_config.model_type", "llama", "granite language model"),
])
def test_refuses_variants_it_does_not_build(tmp_path, path, value, fragment):
    # Each of these loads in Transformers and changes the numbers; refusing is the only
    # thing between them and a plausible wrong transcript.
    with pytest.raises(NotImplementedError, match=fragment):
        config_module.resolve(_write(tmp_path, cfg=_variant(path, value)))


def test_refuses_a_different_feature_front_end(tmp_path):
    pre = dict(_PREPROCESSOR, feature_extractor_type="WhisperFeatureExtractor")
    with pytest.raises(NotImplementedError, match="GraniteSpeechFeatureExtractor"):
        config_module.resolve(_write(tmp_path, pre=pre))


def test_extractor_reads_only_its_constructor_arguments(tmp_path):
    # The reference extractor overwrites melspec_kwargs with its constructor values; a different
    # entry there does not change the features, a top-level argument does.
    pre = dict(_PREPROCESSOR, melspec_kwargs={"n_fft": 1024})
    assert config_module.resolve(_write(tmp_path, pre=pre)).front_end.n_fft == 512
    pre = dict(_PREPROCESSOR, n_fft=1024)
    assert config_module.resolve(_write(tmp_path, pre=pre)).front_end.n_fft == 1024


def test_placeholder_count_follows_the_processor(tmp_path):
    cfg = config_module.resolve(_write(tmp_path))
    # The checkpoint's multilingual_sample.wav: 399040 samples, 1247 frames, 252 placeholders.
    assert cfg.stacked_frames(399040) == 1247
    assert cfg.audio_tokens(399040) == 252


def test_inventory_matches_the_reference_module_tree(tmp_path):
    cfg = config_module.resolve(_write(tmp_path))
    names = checkpoint.inventory(cfg)
    # 534 encoder + 57 projector + 363 language-model tensors: the checkpoint index lists 954.
    assert len(names) == 954
    assert names["encoder.layers.0.attn.rel_pos_emb.weight"] == (1025, 128)
    assert names["encoder.layers.15.conv.depth_conv.conv.weight"] == (2048, 1, 15)
    assert names["projector.query"] == (1, 3, 1024)
    assert names["projector.qformer.encoder.layer.1.crossattention.attention.key.weight"] == (1024, 1024)
    assert names["language_model.model.layers.39.self_attn.k_proj.weight"] == (512, 2048)
    assert names["language_model.lm_head.weight"] == (100353, 2048)


def test_checkpoint_refuses_missing_and_unexpected_tensors(tmp_path):
    cfg = config_module.resolve(_write(tmp_path))
    names = sorted(checkpoint.inventory(cfg))
    mapping = {name: "model-00001-of-00001.safetensors" for name in names[1:]}
    mapping["encoder.extra.weight"] = "model-00001-of-00001.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(NotImplementedError, match=names[0].replace(".", r"\.")):
        checkpoint.Checkpoint(tmp_path, cfg)


def test_sample_limits_are_tight_for_the_capacity(tmp_path):
    cfg = config_module.resolve(_write(tmp_path))
    prompt = 18  # 3 ids before the placeholders and 15 after them
    limits = config_module.sample_limits(cfg, prompt, 4096)
    assert limits["min_samples"] == 257  # reflect padding by 256 needs 257 samples
    assert limits["max_samples"] == 6523359  # 407.7 s at 16 kHz
    assert prompt + cfg.audio_tokens(limits["max_samples"]) + 1 <= 4096
    assert prompt + cfg.audio_tokens(limits["max_samples"] + 1) + 1 > 4096
    assert limits["opt_samples"] == 30 * 16000
    with pytest.raises(ValueError, match="no room"):
        config_module.sample_limits(cfg, prompt, 20)
