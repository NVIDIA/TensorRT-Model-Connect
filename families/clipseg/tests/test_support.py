# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_clipseg_owns_text_prompted_segmentation():
    family, support = resolve_family(
        ModelMetadata(
            {"model_type": "clipseg", "architectures": ["CLIPSegForImageSegmentation"]}, {}
        )
    )
    assert family == "clipseg"
    assert support.tasks == ("text_prompted_segmentation",)
    assert support.default_task == "text_prompted_segmentation"
