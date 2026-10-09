# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep every Qwen checkpoint variant on the semantic Task contract."""

import pytest

from families.qwen.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


@pytest.mark.parametrize("model_type", ("qwen", "Qwen2", "qwen2", "qwen3", "qwq"))
def test_semantic_primary_task(model_type: str) -> None:
    support = describe(ModelMetadata(config={"model_type": model_type}, model_index={}))
    assert support is not None
    assert support.tasks == ("text_continuation",)
    assert support.default_task == "text_continuation"


def test_embedding_has_one_qwen_owner():
    metadata = ModelMetadata(
        config={"model_type": "qwen3"},
        model_index={},
        files=("modules.json", "1_Pooling/config.json"),
    )
    family, support = resolve_family(metadata)
    assert family == "qwen"
    assert support.default_task == "embedding"
    assert support.default_precision == "bf16"
    assert support.tasks == ("embedding",)


def test_generation_discovery_is_unchanged():
    family, support = resolve_family(ModelMetadata(config={"model_type": "qwen3"}, model_index={}))
    assert family == "qwen"
    assert support.tasks == ("text_continuation",)
