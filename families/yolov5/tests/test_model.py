# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for YOLOv5 builds."""

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

from families.yolov5 import checkpoint as checkpoint_module  # noqa: E402
from families.yolov5 import model  # noqa: E402
from families.yolov5.checkpoint import Checkpoint  # noqa: E402
from families.yolov5.support import describe  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


def _checkpoint(
    *,
    classes: int = 80,
    image_size: int = 640,
    strides: tuple[int, ...] = (8, 16, 32),
) -> Checkpoint:
    names = tuple(f"class{index}" for index in range(classes))
    return Checkpoint(
        {"model.0.conv.weight": np.zeros((16, 3, 6, 6), np.float32)},
        names,
        image_size,
        strides,
    )


def test_support_claims_a_directory_by_its_archive() -> None:
    """A YOLOv5 release carries no config.json, so the file is the identity."""
    assert describe(ModelMetadata(config={}, model_index={}, files=("yolov5n.pt",))) is not None
    assert describe(ModelMetadata(config={}, model_index={}, files=("yolov8n.pt",))) is None
    assert describe(ModelMetadata(config={"model_type": "detr"}, model_index={})) is None


def test_checkpoint_opens_the_archive_it_is_named_for(tmp_path: Path) -> None:
    """A release ships every width, and a build request cannot say which."""
    with pytest.raises(FileNotFoundError, match="yolov5n.pt"):
        Checkpoint.open(tmp_path)


def test_a_placeholder_still_fails_a_dunder_lookup() -> None:
    """The stand-in answers class names, not module attributes.

    `inspect` reads `__file__` off every module handed to it, and `torch.load`
    puts it on that path. Answering a dunder with a class makes unpickling die
    inside the standard library with an unrelated AttributeError.
    """
    assert issubclass(checkpoint_module._placeholder_for("Detect"), checkpoint_module._Placeholder)
    with pytest.raises(AttributeError):
        checkpoint_module._placeholder_for("__file__")


def test_a_placeholder_restores_attributes_without_running_anything() -> None:
    """Unpickling a module only assigns state, so nothing needs to execute."""
    node = checkpoint_module._Placeholder()
    node.__setstate__({"_parameters": {}, "nc": 80})
    assert node.nc == 80


def test_collect_walks_a_module_tree_the_way_state_dict_does() -> None:
    """Parameters and buffers are both kept, and names join with a dot."""
    leaf = checkpoint_module._Placeholder()
    leaf.__setstate__({"_parameters": {"weight": "w"}, "_buffers": {"running_mean": "m"}})
    root = checkpoint_module._Placeholder()
    root.__setstate__({"_modules": {"0": leaf}, "_buffers": {"anchors": "a"}})
    collected: dict[str, object] = {}
    checkpoint_module._collect(root, "", collected)
    assert collected == {"anchors": "a", "0.weight": "w", "0.running_mean": "m"}


def test_config_comes_from_the_archive() -> None:
    config = model.describe_checkpoint(_checkpoint(classes=3))
    assert config["names"] == {0: "class0", 1: "class1", 2: "class2"}
    assert config["imgsz"] == 640
    assert config["strides"] == [8, 16, 32]


def test_config_rejects_a_size_the_backbone_cannot_halve() -> None:
    """The stride 32 backbone needs the input to divide by 32."""
    with pytest.raises(ValueError, match="divisible by 32"):
        model.describe_checkpoint(_checkpoint(image_size=700))


def test_config_rejects_an_archive_with_no_classes() -> None:
    with pytest.raises(ValueError, match="names no classes"):
        model.describe_checkpoint(Checkpoint({}, (), 640, (8, 16, 32)))


def test_preprocess_rejects_a_stride_count_the_head_cannot_use() -> None:
    """The head has one branch per level, so the strides have to match it."""
    with pytest.raises(ValueError, match="3 detection levels"):
        model._preprocess_config(model.describe_checkpoint(_checkpoint(strides=(8, 16))))


def test_preprocess_uses_the_yolov5_letterbox_values() -> None:
    config = model._preprocess_config(model.describe_checkpoint(_checkpoint()))
    assert config["mean"] == [0.0, 0.0, 0.0]
    assert config["std"] == [1.0, 1.0, 1.0]
    assert config["pad_value"] == pytest.approx(114.0 / 255.0)


def test_only_the_stage_table_convolutions_change_scale() -> None:
    """Seven convolutions halve the resolution; the backbone stride totals 32."""
    assert model._STRIDED_CONVS == frozenset({0, 1, 3, 5, 7, 18, 21})
    backbone = [index for index in model._STRIDED_CONVS if index < 10]
    assert len(backbone) == 5
    assert 2 ** len(backbone) == 32


def test_the_stem_pads_by_two_not_by_half_its_kernel() -> None:
    """The stem is a 6x6 kernel padded by 2, the one convolution that differs.

    Every other convolution pads by half its kernel. Padding the stem by 3
    would add one row and one column, and every box would land offset.
    """
    assert model._STEM_PADDING == {0: 2}
    assert 0 in model._STRIDED_CONVS


def test_the_neck_drops_the_residual_and_the_backbone_keeps_it() -> None:
    """The published configuration builds every neck C3 with shortcut=False.

    The channel shapes match either way, so an engine built with the wrong
    rule still runs and still returns plausible detections. This is read off
    the reference, not assumed: with it right, all 25200 anchors agree with
    the reference on their class.
    """
    assert model._RESIDUAL == {"c3": True, "c3_plain": False}
    kinds = {index: kind for index, kind, _ in model._STAGES}
    for index in (2, 4, 6, 8):
        assert kinds[index] == "c3"
    for index in (13, 17, 20, 23):
        assert kinds[index] == "c3_plain"


def test_the_head_reads_the_three_neck_outputs() -> None:
    assert model._HEAD_INDEX == 24
    assert model._HEAD_SOURCES == (17, 20, 23)
    # The head is the one stage the table does not list, because it is built
    # by the detect path rather than by the backbone loop.
    assert model._HEAD_INDEX not in {index for index, _, _ in model._STAGES}


def test_the_stage_table_wires_each_concat_to_a_kept_output() -> None:
    """Nothing may refer forward, and every source has to exist."""
    seen: set[int] = set()
    for index, kind, sources in model._STAGES:
        if kind == "concat":
            for source in sources:
                assert source == -1 or source in seen, (index, source)
        seen.add(index)
    for source in model._HEAD_SOURCES:
        assert source in seen


def test_the_layout_check_reports_a_topology_mismatch() -> None:
    """A checkpoint that is not this topology has to say which stage is off."""
    bare = Checkpoint({"anchors": np.zeros(1, np.float32)}, ("a",), 640, (8, 16, 32))
    with pytest.raises(ValueError, match="has no model"):
        model._layout(bare)
    tensors = {f"model.{index}.conv.weight": np.zeros(1, np.float32) for index in range(3)}
    with pytest.raises(ValueError, match="missing="):
        model._layout(Checkpoint(tensors, ("a",), 640, (8, 16, 32)))


def test_each_level_predicts_against_three_anchors() -> None:
    """Objectness and the four box values sit ahead of the class scores."""
    assert model._ANCHORS_PER_LEVEL == 3
    assert model._BOX_VALUES == 4
    assert model._OBJECTNESS_VALUES == 1
    # This is the published head width for the 80-class COCO releases.
    assert model._ANCHORS_PER_LEVEL * (model._BOX_VALUES + model._OBJECTNESS_VALUES + 80) == 255


def test_cell_coordinates_run_row_by_row_as_x_then_y() -> None:
    """The head flattens height before width, and reports x before y."""
    grid = model._cell_grid(2, 3)
    assert grid.shape == (2, 6)
    assert list(grid[0]) == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    assert list(grid[1]) == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_batch_norm_uses_the_yolov5_epsilon() -> None:
    """YOLOv5 resets every norm to 1e-3 after building the model.

    The value is pickled with the module but is not in the state dict. With
    the PyTorch default of 1e-5 the engine still detects the same object, and
    a weak false detection appears that the reference does not report.
    """
    assert model._BATCH_NORM_EPSILON == 1e-3


def test_folding_rejects_a_norm_that_does_not_match_its_convolution() -> None:
    tensors = {
        "block.conv.weight": np.zeros((8, 3, 3, 3), np.float32),
        "block.bn.weight": np.ones(4, np.float32),
        "block.bn.bias": np.zeros(4, np.float32),
        "block.bn.running_mean": np.zeros(4, np.float32),
        "block.bn.running_var": np.ones(4, np.float32),
    }
    with pytest.raises(ValueError, match="does not match its convolution"):
        model._fold(Checkpoint(tensors, ("a",), 640, (8, 16, 32)), "block", np.float32)


def test_folding_keeps_the_statistics_in_float32() -> None:
    """A small running variance loses its precision if divided in fp16."""
    tensors = {
        "block.conv.weight": np.ones((1, 1, 1, 1), np.float32),
        "block.bn.weight": np.full(1, 2.0, np.float32),
        "block.bn.bias": np.full(1, 1.0, np.float32),
        "block.bn.running_mean": np.full(1, 3.0, np.float32),
        "block.bn.running_var": np.full(1, 4.0, np.float32),
    }
    weight, bias = model._fold(Checkpoint(tensors, ("a",), 640, (8, 16, 32)), "block", np.float16)
    scale = 2.0 / np.sqrt(4.0 + model._BATCH_NORM_EPSILON)
    assert float(weight.reshape(-1)[0]) == pytest.approx(scale, rel=1e-3)
    assert float(bias.reshape(-1)[0]) == pytest.approx(1.0 - 3.0 * scale, rel=1e-3)
