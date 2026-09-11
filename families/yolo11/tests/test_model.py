# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for YOLO11 builds."""

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

from families.yolo11 import model  # noqa: E402
from families.yolo11.checkpoint import Checkpoint  # noqa: E402
from families.yolo11.support import describe  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


def _checkpoint(*, classes: int = 80, image_size: int = 640) -> Checkpoint:
    names = tuple(f"class{index}" for index in range(classes))
    return Checkpoint(
        {"model.0.conv.weight": np.zeros((16, 3, 3, 3), np.float32)}, names, image_size
    )


def test_support_claims_a_directory_by_its_archive() -> None:
    """An Ultralytics release carries no config.json, so the file is the identity."""
    assert describe(ModelMetadata(config={}, model_index={}, files=("yolo11n.pt",))) is not None
    assert describe(ModelMetadata(config={}, model_index={}, files=("yolov10n.pt",))) is None
    assert describe(ModelMetadata(config={"model_type": "detr"}, model_index={})) is None


def test_checkpoint_opens_the_archive_it_is_named_for(tmp_path: Path) -> None:
    """A release ships every width, and a build request cannot say which."""
    with pytest.raises(FileNotFoundError, match="yolo11n.pt"):
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
    assert model._STRIDED_CONVS == frozenset({0, 1, 3, 5, 7, 17, 20})
    backbone = [index for index in model._STRIDED_CONVS if index < 10]
    assert len(backbone) == 5


def test_every_block_adds_its_input_back() -> None:
    """YOLO11 keeps the residual in the neck, where YOLOv8 drops it.

    The channel shapes match either way, so an engine built with the YOLOv8
    rule still runs and still returns plausible detections. A first draft of
    this family carried that rule over and the neck stages were wrong: the
    median box moved by 13 pixels and only 3453 of 8400 anchors agreed on a
    class. This is read off the reference, not assumed.
    """
    assert model._RESIDUAL == {"c3k2": True}
    kinds = {index: kind for index, kind, _ in model._STAGES}
    for index in (2, 4, 6, 8, 13, 16, 19, 22):
        assert kinds[index] == "c3k2"


def test_the_inner_block_kind_is_read_from_the_checkpoint() -> None:
    """A C3k carries a third convolution; a bottleneck never does.

    That is the only difference visible without running the model, and it is
    what decides which inner block a C3k2 builds.
    """
    source = Path(model.__file__).read_text(encoding="utf-8")
    assert 'weights.exists(f"{leaf}.cv3.conv.weight")' in source


def test_psa_head_size_is_the_documented_constant() -> None:
    """PSA fixes the size of a head, not how many there are.

    Dividing by the wrong number still builds. An earlier YOLOv10 draft used
    32 here and the attention stage fell to 0.557 correlation.
    """
    assert model._PSA_HEAD_DIM == 64


def test_the_head_reads_the_three_neck_outputs() -> None:
    assert model._HEAD_INDEX == 23
    assert model._HEAD_SOURCES == (16, 19, 22)
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
