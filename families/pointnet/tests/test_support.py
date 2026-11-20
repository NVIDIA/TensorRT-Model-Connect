# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free identity and default-task tests for PointNet support."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata


def _metadata(model_type="", files=()):
    return ModelMetadata(
        config={"model_type": model_type} if model_type else {},
        model_index={},
        files=files,
    )


def test_matches_pointnet_model_type():
    from families.pointnet.support import describe

    support = describe(_metadata(model_type="pointnet"))
    assert support == FamilySupport(
        tasks=("points_to_semantic_segmentation",),
        default_task="points_to_semantic_segmentation",
        default_precision="fp32",
    )


def test_matches_pointnet_onnx_file():
    from families.pointnet.support import describe

    support = describe(_metadata(files=("pointnet.onnx", "config.json")))
    assert support is not None
    assert support.default_task == "points_to_semantic_segmentation"
    assert support.default_precision == "fp32"


def test_rejects_unrelated_model():
    from families.pointnet.support import describe

    assert describe(_metadata(model_type="gpt2")) is None
