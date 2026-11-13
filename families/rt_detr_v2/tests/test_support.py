# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover the published RT-DETR v2 checkpoint through its config identity."""

import json

import pytest

from families.rt_detr_v2.support import describe
from tensorrt_model_connect.model_support import (
    ModelMetadata,
    load_model_metadata,
    resolve_family,
)


@pytest.mark.parametrize(
    "config",
    [
        # PekingU/rtdetr_v2_r18vd@5650961749fa93567c0d46fc7f43ea4f9e914107.
        {"model_type": "rt_detr_v2", "architectures": ["RtDetrV2ForObjectDetection"]},
        {"model_type": "rt_detr_v2"},
        {"architectures": ["RtDetrV2ForObjectDetection"]},
    ],
)
def test_checkpoint_config_resolves_one_owner_without_a_pipeline_index(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    metadata = load_model_metadata(tmp_path)
    assert metadata.model_index == {}
    family, support = resolve_family(metadata)
    assert family == "rt_detr_v2"
    assert support.tasks == ("object_detection",)
    assert support.default_task == "object_detection"


def test_v1_checkpoint_does_not_match_the_v2_family():
    metadata = ModelMetadata(
        {"model_type": "rt_detr", "architectures": ["RTDetrForObjectDetection"]}, {}
    )
    assert describe(metadata) is None
