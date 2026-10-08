# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


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
    assert support.tasks == ("text_generation",)
