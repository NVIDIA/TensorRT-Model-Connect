# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_eagle_vlm_default_task_is_checkpoint_owned() -> None:
    _, embedding = resolve_family(
        ModelMetadata(
            {
                "model_type": "llama_nemotron_vl",
                "architectures": ["LlamaNemotronVLModel"],
            },
            {},
        )
    )
    _, reranking = resolve_family(
        ModelMetadata(
            {
                "model_type": "llama_nemotron_vl_rerank",
                "architectures": ["LlamaNemotronVLForSequenceClassification"],
            },
            {},
        )
    )

    assert embedding.default_task == "embedding"
    assert reranking.default_task == "reranking"
