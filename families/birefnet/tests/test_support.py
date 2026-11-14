# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover BiRefNet's published config, which declares no model_type."""

import json

from families.birefnet.support import describe
from tensorrt_model_connect.model_support import (
    ModelMetadata,
    load_model_metadata,
    resolve_family,
)


def test_lite_checkpoint_resolves_one_owner_without_a_model_type_or_pipeline_index(tmp_path):
    # ZhengPeng7/BiRefNet_lite@aa62cd87eafb9cc43056d08ef3615a14628b831d.
    config = {
        "architectures": ["BiRefNet"],
        "auto_map": {
            "AutoConfig": "BiRefNet_config.BiRefNetConfig",
            "AutoModelForImageSegmentation": "birefnet.BiRefNet",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    metadata = load_model_metadata(tmp_path)
    assert metadata.model_type == "" and metadata.model_index == {}
    family, support = resolve_family(metadata)
    assert family == "birefnet"
    assert support.tasks == ("segmentation",)
    assert support.default_task == "segmentation"


def test_an_unrelated_segmenter_is_not_claimed():
    metadata = ModelMetadata(
        {"model_type": "segformer", "architectures": ["SegformerForSemanticSegmentation"]}, {}
    )
    assert describe(metadata) is None
