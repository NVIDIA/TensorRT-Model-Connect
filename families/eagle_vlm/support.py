# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for eagle_vlm."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata, family_support


_TASKS = (
    "text_to_embedding",
    "text_to_pooled_features",
    "text_query_documents_to_relevance",
    "text_pair_to_relevance",
)
_EMBED = family_support(
    model_types=("llama_nemotron_vl",),
    architectures=("LlamaNemotronVLModel",),
    tasks=_TASKS,
    default_task="text_to_embedding",
)
_RERANK = family_support(
    model_types=("llama_nemotron_vl_rerank",),
    architectures=("LlamaNemotronVLForSequenceClassification",),
    tasks=_TASKS,
    default_task="text_query_documents_to_relevance",
)
_GENERIC = family_support(
    model_types=("eagle_vlm", "eaglevlm"),
    tasks=_TASKS,
    default_task="text_to_pooled_features",
)


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    return _EMBED(metadata) or _RERANK(metadata) or _GENERIC(metadata)
