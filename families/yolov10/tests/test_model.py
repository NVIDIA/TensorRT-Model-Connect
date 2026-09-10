# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for YOLOv10 builds."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file


try:
    import tensorrt  # noqa: F401
except ModuleNotFoundError:
    sys.modules["tensorrt"] = SimpleNamespace()

from families.yolov10 import model  # noqa: E402
from families.yolov10.checkpoint import Checkpoint  # noqa: E402
from families.yolov10.support import describe  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


def _random(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _conv_norm(tensors: dict[str, np.ndarray], prefix: str, out_ch: int = 4, kernel: int = 3):
    tensors[f"{prefix}.conv.weight"] = _random(out_ch, 4, kernel, kernel)
    tensors[f"{prefix}.bn.weight"] = _random(out_ch)
    tensors[f"{prefix}.bn.bias"] = _random(out_ch)
    tensors[f"{prefix}.bn.running_mean"] = _random(out_ch)
    tensors[f"{prefix}.bn.running_var"] = np.abs(_random(out_ch)) + 1.0


def _stage_set(tmp_path: Path, indices) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    for index in indices:
        _conv_norm(tensors, f"model.model.{index}.cv1")
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def _config(**overrides):
    raw = {"model": "yolov10n.yaml", "task": "detect", "names": {str(i): str(i) for i in range(80)}}
    raw.update(overrides)
    return raw


def _metadata(config) -> ModelMetadata:
    return ModelMetadata(config=config, model_index={})


def test_support_matches_the_exact_root_shape() -> None:
    """These checkpoints carry no model_type, so identity is the JSON shape."""
    assert describe(_metadata(_config())) is not None
    # A detector from another family must not match on `task` alone.
    assert describe(_metadata(_config(model="rtdetr.yaml"))) is None
    # Nor a YOLOv10 config that is not a detector.
    assert describe(_metadata(_config(task="segment"))) is None
    # Nor one without classes.
    assert describe(_metadata(_config(names={}))) is None


def test_support_ignores_metadata_without_a_config() -> None:
    assert describe(SimpleNamespace(config=None)) is None


def test_read_config_rejects_another_family(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(
        json.dumps({"model": "yolov8n.yaml", "task": "detect", "names": {"0": "person"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported YOLOv10 model identity"):
        model._read_config(tmp_path)


def test_read_config_rejects_a_non_detection_task(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(
        json.dumps({"model": "yolov10n.yaml", "task": "segment", "names": {"0": "person"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported YOLOv10 task"):
        model._read_config(tmp_path)


def test_preprocess_config_requires_a_size_the_strides_divide() -> None:
    with pytest.raises(ValueError, match="must divide"):
        model._preprocess_config(_config(imgsz=641))


def test_preprocess_config_reads_the_class_count_from_names() -> None:
    config = model._preprocess_config(_config())
    assert config["num_classes"] == 80
    # Ultralytics scales to [0, 1] and does not mean-subtract; the runtime seam
    # depends on those exact values.
    assert config["mean"] == [0.0, 0.0, 0.0]
    assert config["std"] == [1.0, 1.0, 1.0]


def test_layout_accepts_the_expected_stage_set(tmp_path: Path) -> None:
    weighted = [index for index, kind, _ in model._STAGES if kind not in {"upsample", "concat"}]
    model._layout(Checkpoint.open(_stage_set(tmp_path, weighted + [model._HEAD_INDEX])))


def test_layout_rejects_a_missing_stage(tmp_path: Path) -> None:
    weighted = [index for index, kind, _ in model._STAGES if kind not in {"upsample", "concat"}]
    with pytest.raises(ValueError, match="missing="):
        model._layout(Checkpoint.open(_stage_set(tmp_path, weighted)))


def test_layout_rejects_an_unexpected_stage(tmp_path: Path) -> None:
    weighted = [index for index, kind, _ in model._STAGES if kind not in {"upsample", "concat"}]
    with pytest.raises(ValueError, match="unexpected="):
        model._layout(
            Checkpoint.open(_stage_set(tmp_path, weighted + [model._HEAD_INDEX, 99]))
        )


def test_neck_bottlenecks_drop_the_residual_backbone_ones_keep() -> None:
    """The shortcut flag is architecture, not shape: the channel counts match
    either way, so inferring it from the tensors adds a residual the model does
    not have.

    This applies only to plain bottleneck inner blocks. A CIB always carries its
    residual, which `_c2f` decides from the keys, because the wider widths put
    CIB blocks in neck stages where the narrow ones put bottlenecks.
    """
    kinds = {index: kind for index, kind, _ in model._STAGES}
    assert [kinds[i] for i in (2, 4, 6, 8)] == ["c2f"] * 4
    assert [kinds[i] for i in (13, 16, 19)] == ["c2f_plain"] * 3
    assert model._RESIDUAL["c2f"] is True
    assert model._RESIDUAL["c2f_plain"] is False


def test_batch_norm_uses_the_ultralytics_epsilon() -> None:
    """1e-5 still builds and still detects, with every box slightly wrong."""
    assert model._BATCH_NORM_EPSILON == 1e-3


def test_psa_head_size_is_the_documented_constant() -> None:
    """PSA fixes head size, not head count. Halving it silently changes the
    attention shape and drops correlation to about 0.56."""
    assert model._PSA_HEAD_DIM == 64


def test_anchors_cover_every_cell_at_each_stride() -> None:
    points, strides = model._anchors(640, 640)
    expected = sum((640 // stride) ** 2 for stride in model._STRIDES)
    assert points.shape == (expected, 2)
    assert strides.shape == (expected, 1)
    # Centres sit at the middle of their cell, and each level carries its stride.
    assert points[0].tolist() == [0.5, 0.5]
    assert set(np.unique(strides).tolist()) == {float(s) for s in model._STRIDES}
