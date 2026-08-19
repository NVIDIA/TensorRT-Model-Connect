# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owner-local image path contracts for the timm ViT E2E runner."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tensorrt_model_connect.models.timm_vit.tests.e2e_plugins.runners import (
    image_classification,
)


def test_repo_relative_owner_image_is_normalized_inside_the_owner(tmp_path) -> None:
    model_test_dir = image_classification._MODEL_TEST_DIR
    repo_root = model_test_dir.parents[4]
    relative_test_dir = model_test_dir.relative_to(repo_root)
    case = SimpleNamespace(
        inputs={"image": str(relative_test_dir / "data/test_img.jpeg")},
        metadata={"model_test_dir": str(relative_test_dir)},
    )
    shadow = tmp_path / "data/test_img.jpeg"
    shadow.parent.mkdir()
    shadow.write_bytes(b"not-the-owner-asset")
    context = SimpleNamespace(engine_dir=str(tmp_path))

    assert image_classification.ImageClassificationRunner()._resolve_image_path(
        case, context
    ) == str(model_test_dir / "data/test_img.jpeg")


def test_repo_relative_image_cannot_escape_the_owner() -> None:
    model_test_dir = image_classification._MODEL_TEST_DIR
    repo_root = model_test_dir.parents[4]
    relative_test_dir = model_test_dir.relative_to(repo_root)
    case = SimpleNamespace(
        inputs={"image": f"{relative_test_dir}/../other/data/test_img.jpeg"},
        metadata={"model_test_dir": str(relative_test_dir)},
    )
    context = SimpleNamespace(engine_dir="/tmp/missing-engines")

    with pytest.raises(FileNotFoundError, match="Model-owned image asset"):
        image_classification.ImageClassificationRunner()._resolve_image_path(
            case, context
        )
