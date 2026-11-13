# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_text_embedding_checkpoint_has_one_owner_and_default_task():
    family, support = resolve_family(
        ModelMetadata(config={"model_type": "embedding_gemma2"}, model_index={})
    )
    assert family == "embeddinggemma2"
    assert support.default_task == "embedding"
    assert support.tasks == ("embedding",)
    assert support.default_precision == "bf16"
