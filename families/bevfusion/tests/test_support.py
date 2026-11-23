# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family
from families.bevfusion.support import describe


def test_bev_checkpoint_has_one_owner():
    metadata = ModelMetadata(config={}, model_index={}, files=("bevfusion-seg.pth",))
    support = describe(metadata)
    assert support is not None
    assert support.tasks == ("spatial_features_to_class_scores",)
    assert support.default_task == "spatial_features_to_class_scores"
    assert support.default_precision == "fp32"
    assert resolve_family(metadata)[0] == "bevfusion"
    assert describe(ModelMetadata(config={"model_type": "bevfusion"}, model_index={})) is None
