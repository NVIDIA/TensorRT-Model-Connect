# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover Deformable DETR from its Transformers config (it has no Diffusers pipeline index)."""

import json

from families.deformable_detr.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, load_model_metadata, resolve_family


def test_published_checkpoint_resolves_to_this_family(tmp_path):
    # SenseTime/deformable-detr@83ecd26945199939cb82806f988debdb71e6f43e (the deformable-detr-r50 profile).
    config = {"model_type": "deformable_detr", "architectures": ["DeformableDetrForObjectDetection"]}
    (tmp_path / "config.json").write_text(json.dumps(config))
    family, support = resolve_family(load_model_metadata(tmp_path))
    assert family == "deformable_detr"
    assert support.default_task == "object_detection"


def test_detr_is_not_claimed():
    metadata = ModelMetadata({"model_type": "detr", "architectures": ["DetrForObjectDetection"]}, {})
    assert describe(metadata) is None
