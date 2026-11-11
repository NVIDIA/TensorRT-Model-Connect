# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free exact checkpoint ownership and default Task."""

import pytest
from families.table_transformer.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "table-transformer"},
        {"architectures": ["TableTransformerForObjectDetection"]},
    ],
)
def test_structure_checkpoint_identity(config):
    support = describe(ModelMetadata(config=config, model_index={}))
    assert support is not None
    assert support.tasks == ("image_to_boxes",)
    assert support.default_task == "image_to_boxes"


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"model_type": "detr"},
        {"model_type": "table-transformer-other"},
        {"architectures": ["DetrForObjectDetection"]},
    ],
)
def test_other_checkpoints_not_claimed(config):
    assert describe(ModelMetadata(config=config, model_index={})) is None
