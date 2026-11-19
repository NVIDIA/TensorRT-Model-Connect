# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free identity and default-task tests for Uni3D support."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata


def _metadata(model_type="", files=()):
    return ModelMetadata(
        config={"model_type": model_type} if model_type else {},
        model_index={},
        files=files,
    )


def test_matches_uni3d_model_type():
    from families.uni3d.support import describe

    support = describe(_metadata(model_type="uni3d"))
    assert support == FamilySupport(
        tasks=("points_to_embedding",),
        default_task="points_to_embedding",
        default_precision="fp16",
    )


def test_matches_uni3d_onnx_file():
    from families.uni3d.support import describe

    support = describe(_metadata(files=("uni3d.onnx", "config.json")))
    assert support is not None
    assert support.default_task == "points_to_embedding"


def test_rejects_unrelated_model():
    from families.uni3d.support import describe

    assert describe(_metadata(model_type="pointnet")) is None
