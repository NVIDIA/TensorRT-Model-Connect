# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for YOLOv8 builds."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


try:
    import tensorrt  # noqa: F401
except ModuleNotFoundError:
    sys.modules["tensorrt"] = SimpleNamespace()

from families.yolov8 import model  # noqa: E402
from families.yolov8.checkpoint import Checkpoint  # noqa: E402
from families.yolov8.support import describe  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


def _checkpoint(*, classes: int = 80, image_size: int = 640) -> Checkpoint:
    names = tuple(f"class{index}" for index in range(classes))
    return Checkpoint(
        {"model.0.conv.weight": np.zeros((16, 3, 3, 3), np.float32)}, names, image_size
    )


def test_support_claims_a_directory_by_its_archive() -> None:
    """An Ultralytics release carries no config.json, so the file is the identity."""
    assert describe(ModelMetadata(config={}, model_index={}, files=("yolov8n.pt",))) is not None
    assert describe(ModelMetadata(config={}, model_index={}, files=("yolov10n.pt",))) is None
    assert describe(ModelMetadata(config={"model_type": "detr"}, model_index={})) is None


def test_checkpoint_opens_the_archive_it_is_named_for(tmp_path: Path) -> None:
    """A release ships every width, and a build request cannot say which."""
    with pytest.raises(FileNotFoundError, match="yolov8n.pt"):
        Checkpoint.open(tmp_path)


def test_config_comes_from_the_archive() -> None:
    config = model.describe_checkpoint(_checkpoint(classes=3))
    assert config["names"] == {0: "class0", 1: "class1", 2: "class2"}
    assert config["imgsz"] == 640


def test_config_rejects_a_size_the_backbone_cannot_halve() -> None:
    """The stride 32 backbone needs the input to divide by 32."""
    with pytest.raises(ValueError, match="divisible by 32"):
        model.describe_checkpoint(_checkpoint(image_size=700))


def test_config_rejects_an_archive_with_no_classes() -> None:
    with pytest.raises(ValueError, match="names no classes"):
        model.describe_checkpoint(Checkpoint({}, (), 640))


def test_only_the_stage_table_convolutions_change_scale() -> None:
    """Seven convolutions halve the resolution; the total stride is 32."""
    assert model._STRIDED_CONVS == frozenset({0, 1, 3, 5, 7, 16, 19})
    backbone = [index for index in model._STRIDED_CONVS if index < 10]
    assert len(backbone) == 5


def test_backbone_blocks_add_their_input_back_and_neck_blocks_do_not() -> None:
    """The channel shapes match either way, so this cannot be read from weights.

    An earlier YOLOv10 draft inferred it from shapes and silently added a
    residual the reference does not have.
    """
    assert model._RESIDUAL == {"c2f": True, "c2f_plain": False}
    kinds = {index: kind for index, kind, _ in model._STAGES}
    assert [kinds[index] for index in (2, 4, 6, 8)] == ["c2f"] * 4
    assert [kinds[index] for index in (12, 15, 18, 21)] == ["c2f_plain"] * 4


def test_the_head_reads_the_three_neck_outputs() -> None:
    assert model._HEAD_INDEX == 22
    assert model._HEAD_SOURCES == (15, 18, 21)
    assert model._STRIDES == (8, 16, 32)


def test_anchors_cover_every_cell_at_each_stride() -> None:
    points, strides = model._anchors(640, 640)
    expected = sum((640 // stride) ** 2 for stride in model._STRIDES)
    assert points.shape == (expected, 2)
    # The stride column keeps a trailing axis so it broadcasts over x and y.
    assert strides.shape == (expected, 1)
    # Each level contributes its own cell count at its own stride.
    for stride in model._STRIDES:
        assert int((strides == stride).sum()) == (640 // stride) ** 2
    # Cell centres sit half a cell in from the corner.
    assert float(points[0][0]) == pytest.approx(0.5)


def test_batch_norm_uses_the_ultralytics_epsilon() -> None:
    """Ultralytics builds with 1e-3, not the PyTorch default of 1e-5."""
    assert model._BATCH_NORM_EPSILON == 1e-3
