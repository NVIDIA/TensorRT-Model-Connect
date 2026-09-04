# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for eagle_vlm."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata, family_support


_TASKS = ("embedding", "encoding", "reranking")
_EMBED = family_support(
    model_types=("llama_nemotron_vl",),
    architectures=("LlamaNemotronVLModel",),
    tasks=_TASKS,
    default_task="embedding",
)
_RERANK = family_support(
    model_types=("llama_nemotron_vl_rerank",),
    architectures=("LlamaNemotronVLForSequenceClassification",),
    tasks=_TASKS,
    default_task="reranking",
)
_GENERIC = family_support(
    model_types=("eagle_vlm", "eaglevlm"),
    tasks=_TASKS,
    default_task="encoding",
)


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    return _EMBED(metadata) or _RERANK(metadata) or _GENERIC(metadata)
