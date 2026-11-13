# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_idefics3_checkpoint_has_one_owner():
    family, support = resolve_family(ModelMetadata({"model_type": "idefics3"}, {}))
    assert family == "idefics3"
    assert support.tasks == ("images_text_to_text",)
