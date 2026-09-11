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


@pytest.mark.parametrize("layout", ["single", "indexed"])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32", "int64"])
def test_reference_checkpoint_preserves_torch_values_and_dtypes(
    tmp_path: Path, layout: str, dtype_name: str
) -> None:
    import torch
    from safetensors.torch import save_file as save_torch_file

    from families.yolov10.tests.test_e2e import _reference_tensors

    dtype = getattr(torch, dtype_name)
    values = {
        "model.weight": torch.tensor([[1, 2], [3, 4]], dtype=dtype),
        "model.bias": torch.tensor([5, 6], dtype=dtype),
    }
    if dtype_name == "int64":
        values["model.weight"][0, 0] = 2**40 + 3
    if layout == "single":
        save_torch_file(values, str(tmp_path / "model.safetensors"))
    else:
        save_torch_file(
            {"model.weight": values["model.weight"]}, str(tmp_path / "part-1.safetensors")
        )
        save_torch_file({"model.bias": values["model.bias"]}, str(tmp_path / "part-2.safetensors"))
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "model.weight": "part-1.safetensors",
                        "model.bias": "part-2.safetensors",
                    }
                }
            ),
            encoding="utf-8",
        )
    actual = _reference_tensors(tmp_path)
    assert set(actual) == set(values)
    for name, expected in values.items():
        assert actual[name].dtype == dtype
        assert torch.equal(actual[name], expected)
    assert actual["model.weight"].device.type == "cpu"


def test_checkpoint_default_keeps_numpy_build_behavior(tmp_path: Path) -> None:
    value = np.array([1.25, 2.5], dtype=np.float16)
    save_file({"model.weight": value}, str(tmp_path / "model.safetensors"))
    checkpoint = Checkpoint.open(tmp_path)
    raw = checkpoint.tensor_map["model.weight"].get_tensor("model.weight")
    assert isinstance(raw, np.ndarray) and raw.dtype == np.float16
    assert checkpoint.tensor("model.weight").dtype == np.float32
    assert np.array_equal(checkpoint.tensor("model.weight"), value)


def test_reference_checkpoint_prefers_single_file_over_an_index(tmp_path: Path) -> None:
    import torch
    from safetensors.torch import save_file as save_torch_file

    from families.yolov10.tests.test_e2e import _reference_tensors

    value = torch.tensor([3.0], dtype=torch.float16)
    save_torch_file({"model.weight": value}, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text("malformed index", encoding="utf-8")
    assert torch.equal(_reference_tensors(tmp_path)["model.weight"], value)
    assert Checkpoint.open(tmp_path).names == {"model.weight"}


def test_reference_checkpoint_uses_declared_mapping_not_all_shard_keys(tmp_path: Path) -> None:
    import torch
    from safetensors.torch import save_file as save_torch_file

    from families.yolov10.tests.test_e2e import _reference_tensors

    save_torch_file(
        {"model.weight": torch.tensor([1.0]), "off_index": torch.tensor([99.0])},
        str(tmp_path / "part-1.safetensors"),
    )
    save_torch_file(
        {"model.weight": torch.tensor([7.0]), "model.bias": torch.tensor([2.0])},
        str(tmp_path / "part-2.safetensors"),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.weight": "part-1.safetensors",
                    "model.bias": "part-2.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    checkpoint = Checkpoint.open(tmp_path)
    actual = _reference_tensors(tmp_path)
    assert set(actual) == checkpoint.names == {"model.weight", "model.bias"}
    assert actual["model.weight"].item() == 1.0
    assert actual["model.bias"].item() == 2.0
    assert np.array_equal(actual["model.weight"].numpy(), checkpoint.tensor("model.weight"))


def test_reference_checkpoint_rejects_a_missing_declared_tensor(tmp_path: Path) -> None:
    import torch
    from safetensors import SafetensorError
    from safetensors.torch import save_file as save_torch_file

    from families.yolov10.tests.test_e2e import _reference_tensors

    save_torch_file({"other": torch.tensor([1.0])}, str(tmp_path / "part.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.missing": "part.safetensors"}}), encoding="utf-8"
    )
    with pytest.raises(SafetensorError):
        _reference_tensors(tmp_path)
    with pytest.raises(SafetensorError):
        Checkpoint.open(tmp_path).tensor("model.missing")


@pytest.mark.parametrize("framework", ["numpy", "pt"])
def test_checkpoint_rejects_a_missing_indexed_shard(tmp_path: Path, framework: str) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": "missing.safetensors"}}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError):
        Checkpoint.open(tmp_path, framework=framework)


@pytest.mark.parametrize("framework", ["numpy", "pt"])
@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"weight_map": {}},
        {"weight_map": []},
        {"weight_map": {"": "part.safetensors"}},
        {"weight_map": {"model.weight": 7}},
    ],
)
def test_checkpoint_rejects_malformed_index_mapping(
    tmp_path: Path, framework: str, payload
) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        Checkpoint.open(tmp_path, framework=framework)


@pytest.mark.parametrize("framework", ["numpy", "pt"])
@pytest.mark.parametrize(
    "shard", ["../outside.safetensors", "/outside.safetensors", "nested/part.safetensors"]
)
def test_checkpoint_rejects_unsafe_indexed_shard_paths(
    tmp_path: Path, framework: str, shard: str
) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": shard}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="direct relative files"):
        Checkpoint.open(tmp_path, framework=framework)


@pytest.mark.parametrize("framework", ["numpy", "pt"])
def test_checkpoint_rejects_invalid_index_json(tmp_path: Path, framework: str) -> None:
    (tmp_path / "model.safetensors.index.json").write_text("{invalid json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        Checkpoint.open(tmp_path, framework=framework)
