# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free identity, default Task and build-request policy of openai_privacy_filter."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from families.openai_privacy_filter.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_token_classifier_offers_head_scores():
    support = describe(
        ModelMetadata(
            config={
                "model_type": "openai_privacy_filter",
                "architectures": ["OpenAIPrivacyFilterForTokenClassification"],
            },
            model_index={},
        )
    )
    assert support is not None
    assert support.tasks == ("text_to_head_scores",)
    assert support.default_task == "text_to_head_scores"
    assert support.default_precision == "fp32"


def test_the_parent_and_sibling_architectures_are_not_claimed():
    for config in (
        {"model_type": "gpt_oss", "architectures": ["GptOssForCausalLM"]},
        {"model_type": "bert", "architectures": ["BertForTokenClassification"]},
        {"model_type": "privacy_filter"},
    ):
        assert describe(ModelMetadata(config=config, model_index={})) is None


_CONFIG = {
    "model_type": "openai_privacy_filter",
    "architectures": ["OpenAIPrivacyFilterForTokenClassification"],
    "hidden_size": 640,
    "num_hidden_layers": 8,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "intermediate_size": 640,
    "num_local_experts": 128,
    "num_experts_per_tok": 4,
    "vocab_size": 200064,
    "max_position_embeddings": 131072,
    "sliding_window": 128,
    "rms_norm_eps": 1e-5,
    "attention_bias": True,
    "rope_parameters": {
        "rope_type": "yarn",
        "rope_theta": 150000.0,
        "factor": 32.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "original_max_position_embeddings": 4096,
        "truncate": False,
    },
    "id2label": {"0": "O", "1": "B-email", "2": "I-email", "3": "E-email", "4": "S-email"},
}


def _request(model_dir, **overrides):
    request = {
        "model_dir": model_dir,
        "task": "text_to_head_scores",
        "precision": "fp32",
        "backend": "trt",
        "max_sequence_length": None,
        "image_height": None,
        "image_width": None,
        "video_num_frames": None,
        "max_batch_size": 1,
        "tensor_parallel_size": 1,
        "context_parallel_size": 1,
        "quantization": None,
        "fp32_layers": (),
        "dynamic_kv_cache": False,
        "verbose": False,
        "mtp_seq_len": None,
    }
    request.update(overrides)
    return SimpleNamespace(**request)


@pytest.mark.parametrize(
    "overrides",
    (
        {"task": "text_to_token_features"},
        {"precision": "fp16"},
        {"precision": "bf16"},
        {"backend": "edgellm"},
        {"quantization": "fp8"},
        {"fp32_layers": (1,)},
        {"tensor_parallel_size": 2},
        {"context_parallel_size": 2},
        {"max_batch_size": 4},
        {"dynamic_kv_cache": True},
        {"mtp_seq_len": 2},
        {"image_height": 224},
        {"max_sequence_length": 131073},
        {"max_sequence_length": 0},
    ),
)
def test_build_refuses_options_it_cannot_honour(tmp_path, overrides):
    from families.openai_privacy_filter import model

    (tmp_path / "config.json").write_text(json.dumps(_CONFIG), encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    writer = SimpleNamespace()
    with pytest.raises((ValueError, NotImplementedError)):
        model.build(_request(tmp_path, **overrides), writer)


@pytest.mark.parametrize(
    "edit",
    (
        {"model_type": "gpt_oss"},
        {"architectures": ["OpenAIPrivacyFilterModel"]},
        {"num_key_value_heads": 3},
        {"num_experts_per_tok": 129},
        {"id2label": {"0": "O", "2": "B-email"}},
        {"rope_parameters": {"rope_type": "longrope", "rope_theta": 1.0}},
    ),
)
def test_configuration_outside_the_architecture_is_refused(edit):
    from families.openai_privacy_filter.config import ModelConfig

    with pytest.raises((ValueError, NotImplementedError)):
        ModelConfig.from_raw({**_CONFIG, **edit})


def test_configuration_defaults_follow_the_reference():
    from families.openai_privacy_filter.config import ModelConfig

    config = ModelConfig.from_raw(_CONFIG)
    assert config.labels == ("O", "B-email", "I-email", "E-email", "S-email")
    assert (config.swiglu_alpha, config.swiglu_limit) == (1.702, 7.0)
    assert config.rope.rope_type == "yarn"
    assert config.rope.attention_factor == pytest.approx(0.1 * math.log(32.0) + 1.0)
