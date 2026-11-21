# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_longformer_has_one_owner():
    family, support = resolve_family(ModelMetadata({"model_type": "longformer"}, {}))
    assert family == "longformer"
    assert support.tasks == ("text_to_token_features",)
    assert support.default_task == "text_to_token_features"
