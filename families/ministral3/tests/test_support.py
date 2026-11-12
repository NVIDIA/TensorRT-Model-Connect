# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free checkpoint identity and default-task policy of ministral3."""

from families.ministral3.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family

EMBEDDING_CONFIG = {
    "architectures": ["Ministral3Model"],
    "model_type": "ministral3",
    "is_causal": False,
}


def test_embedding_checkpoint_identity_and_default_task():
    support = describe(ModelMetadata(config=EMBEDDING_CONFIG, model_index={}))
    assert support is not None
    assert support.tasks == (
        "text_to_embedding",
        "text_to_pooled_features",
        "text_to_token_features",
    )
    assert support.default_task == "text_to_embedding"
    assert support.default_precision == "bf16"


def test_the_checkpoint_resolves_to_exactly_this_family():
    family, support = resolve_family(ModelMetadata(config=EMBEDDING_CONFIG, model_index={}))
    assert family == "ministral3"
    assert support.default_task == "text_to_embedding"


def test_other_ministral_identities_are_not_claimed():
    for config in (
        {"model_type": "ministral3", "architectures": ["Ministral3ForCausalLM"]},
        {"model_type": "mistral3", "architectures": ["Mistral3ForConditionalGeneration"]},
        {"model_type": "mistral", "architectures": ["MistralModel"]},
    ):
        assert describe(ModelMetadata(config=config, model_index={})) is None


def test_causal_ministral3_backbones_are_not_claimed():
    """``is_causal`` true, absent (causal by default) or not a boolean is not this family."""
    for value in (True, None, "false", 0):
        config = dict(EMBEDDING_CONFIG, is_causal=value)
        if value is None:
            del config["is_causal"]
        assert describe(ModelMetadata(config=config, model_index={})) is None
