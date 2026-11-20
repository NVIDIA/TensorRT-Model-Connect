# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_mt5_has_a_distinct_owner():
    family, support = resolve_family(
        ModelMetadata({"model_type": "mt5", "architectures": ["MT5ForConditionalGeneration"]}, {})
    )
    assert family == "mt5"
    assert support.tasks == ("text_generation",)
    assert support.default_task == "text_generation"
