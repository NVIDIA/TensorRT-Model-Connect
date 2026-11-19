# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family
from families.dinov2_with_registers.support import describe


def test_exact_register_backbone_has_one_owner():
    for identity in (
        {"model_type": "dinov2_with_registers"},
        {"architectures": ["Dinov2WithRegistersModel"]},
    ):
        family, support = resolve_family(ModelMetadata(identity, {}))
        assert family == "dinov2_with_registers"
        assert support.tasks == ("image_to_token_and_pooled_features",)
        assert support.default_task == "image_to_token_and_pooled_features"
        assert support.default_precision == "fp32"


def test_other_backbones_and_classification_heads_are_not_claimed():
    for identity in (
        {"model_type": "dinov2"},
        {"model_type": "dinov3_vit"},
        {"architectures": ["Dinov2WithRegistersForImageClassification"]},
    ):
        assert describe(ModelMetadata(identity, {})) is None
