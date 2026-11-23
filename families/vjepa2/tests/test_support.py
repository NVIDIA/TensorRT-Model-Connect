# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_vjepa2_checkpoint_has_one_owner():
    family, support = resolve_family(ModelMetadata({"model_type": "vjepa2"}, {}))
    assert family == "vjepa2"
    assert support.tasks == ("video_to_token_features",)
    assert support.default_task == "video_to_token_features"
