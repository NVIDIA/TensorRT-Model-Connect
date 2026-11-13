# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_p_h3_image_checkpoint_has_one_owner():
    family, support = resolve_family(ModelMetadata({}, {"_class_name": "PH3ImageModularPipeline"}))
    assert family == "p_h3_image"
    assert support.tasks == ("text_to_image",)
    assert support.default_precision == "bf16"
