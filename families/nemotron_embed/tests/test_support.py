# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family
from families.nemotron_embed.support import describe


def test_bidirectional_embed_checkpoint_has_one_owner():
    metadata = ModelMetadata(
        {
            "model_type": "mistral3",
            "pooling": "avg",
            "text_config": {"model_type": "ministral3", "is_causal": False},
        },
        {},
    )
    family, support = resolve_family(metadata)
    assert family == "nemotron_embed"
    assert support.tasks == ("text_to_embedding",)
    assert support.default_task == "text_to_embedding"


def test_generative_ministral_is_not_claimed():
    assert (
        describe(
            ModelMetadata(
                {
                    "model_type": "mistral3",
                    "pooling": "avg",
                    "text_config": {"model_type": "ministral3", "is_causal": True},
                },
                {},
            )
        )
        is None
    )
