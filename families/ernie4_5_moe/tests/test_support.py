# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_ernie4_5_moe_checkpoint_has_one_owner():
    family, support = resolve_family(ModelMetadata({"model_type": "ernie4_5_moe"}, {}))
    assert family == "ernie4_5_moe"
    assert support.tasks == ("text_continuation",)
    assert support.default_task == "text_continuation"
