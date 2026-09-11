# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for timm Res2Net builds."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file


try:
    import tensorrt  # noqa: F401
except ModuleNotFoundError:
    sys.modules["tensorrt"] = SimpleNamespace()

from families.timm_res2net import model  # noqa: E402
from families.timm_res2net.checkpoint import Checkpoint  # noqa: E402
from families.timm_res2net.support import describe  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


def _random(*shape: int) -> np.ndarray:
    return np.random.RandomState(3).randn(*shape).astype(np.float32)


def _norm(tensors: dict[str, np.ndarray], prefix: str, channels: int) -> None:
    tensors[f"{prefix}.weight"] = _random(channels)
    tensors[f"{prefix}.bias"] = _random(channels)
    tensors[f"{prefix}.running_mean"] = _random(channels)
    tensors[f"{prefix}.running_var"] = np.abs(_random(channels)) + 1.0


def _checkpoint(
    tmp_path: Path,
    *,
    scale: int = 4,
    deep_stem: bool = False,
    pooled_shortcut: bool = False,
    depths: tuple[int, ...] = (1, 1, 1, 1),
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architecture": "res2net50_26w_4s",
                "num_classes": 5,
                "pretrained_cfg": {
                    "input_size": [3, 224, 224],
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "crop_pct": 0.875,
                    "interpolation": "bilinear",
                },
            }
        ),
        encoding="utf-8",
    )
    width = 4
    tensors: dict[str, np.ndarray] = {}
    if deep_stem:
        for conv, norm, out_ch in (
            ("conv1.0", "conv1.1", 4),
            ("conv1.3", "conv1.4", 4),
            ("conv1.6", "bn1", 8),
        ):
            tensors[f"{conv}.weight"] = _random(out_ch, 4, 3, 3)
            _norm(tensors, norm, out_ch)
    else:
        tensors["conv1.weight"] = _random(8, 3, 7, 7)
        _norm(tensors, "bn1", 8)

    for stage in ("layer1", "layer2", "layer3", "layer4"):
        for index in range(depths[0]):
            prefix = f"{stage}.{index}"
            tensors[f"{prefix}.conv1.weight"] = _random(width * scale, 8, 1, 1)
            _norm(tensors, f"{prefix}.bn1", width * scale)
            for rung in range(scale - 1):
                tensors[f"{prefix}.convs.{rung}.weight"] = _random(width, width, 3, 3)
                _norm(tensors, f"{prefix}.bns.{rung}", width)
            tensors[f"{prefix}.conv3.weight"] = _random(8, width * scale, 1, 1)
            _norm(tensors, f"{prefix}.bn3", 8)
            if index == 0:
                if pooled_shortcut:
                    # Index 0 is the pool and carries no tensor of its own.
                    tensors[f"{prefix}.downsample.1.weight"] = _random(8, 8, 1, 1)
                    _norm(tensors, f"{prefix}.downsample.2", 8)
                else:
                    tensors[f"{prefix}.downsample.0.weight"] = _random(8, 8, 1, 1)
                    _norm(tensors, f"{prefix}.downsample.1", 8)
    tensors["fc.weight"] = _random(5, 8)
    tensors["fc.bias"] = _random(5)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def _metadata(model_type: str) -> ModelMetadata:
    return ModelMetadata(config={"model_type": model_type}, model_index={})


def test_support_claims_only_its_own_identity() -> None:
    assert describe(_metadata("res2net50_26w_4s")) is not None
    assert describe(_metadata("res2next50")) is not None
    assert describe(_metadata("seresnet50")) is None


@pytest.mark.parametrize("missing_library", ["backend", "model", None])
def test_e2e_runtime_library_checks_record_setup_before_build(
    tmp_path: Path, monkeypatch, missing_library: str | None
) -> None:
    from contextlib import contextmanager

    from families.timm_res2net.tests import test_e2e
    from tools import e2e_evidence

    binary = tmp_path / "trtmc"
    binary.touch()
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    libraries = {
        "backend": runtime_root / "libtrtmc_backend_trt.so",
        "model": runtime_root / "libtrtmc_model_timm_res2net.so",
    }
    for role, path in libraries.items():
        if role != missing_library:
            path.touch()
    monkeypatch.setenv("TRTMC_BINARY", str(binary))
    monkeypatch.setenv("TRTMC_RUNTIME_ROOT", str(runtime_root))
    recorder = e2e_evidence.Evidence(
        tmp_path / "evidence", family="timm_res2net", case="runtime-library-control",
        source_revision="a" * 40, roots=(tmp_path,),
    )
    events = []
    original_is_file = Path.is_file

    def is_file(path):
        if path in libraries.values():
            events.append(path.name)
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", is_file)

    def model_dir(manifest):
        events.append("model_dir")
        return tmp_path

    build_stop = RuntimeError("CPU control reached the original build boundary")

    def build(request):
        events.append("build")
        assert request.model_dir == tmp_path
        raise build_stop

    monkeypatch.setattr(test_e2e, "_model_dir", model_dir)
    monkeypatch.setattr(test_e2e, "build", build)
    setup_errors = []

    @contextmanager
    def observed_stage(name):
        with e2e_evidence.evidence_stage(name):
            try:
                yield
            except AssertionError as error:
                if name == "setup":
                    setup_errors.append(error)
                raise

    monkeypatch.setattr(test_e2e, "evidence_stage", observed_stage)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        if missing_library:
            with pytest.raises(AssertionError) as caught:
                test_e2e.test_official_checkpoint_e2e("res2net50-26w-4s-in1k", tmp_path)
            assert len(setup_errors) == 1 and setup_errors[0] is caught.value
            assert recorder.data["failure_stage"] == "setup"
            assert [(row["stage"], row["status"]) for row in recorder.data["timing"]] == [
                ("setup", "failed")
            ]
            assert events == [libraries["backend"].name] + (
                [libraries["model"].name] if missing_library == "model" else []
            )
        else:
            with pytest.raises(RuntimeError) as caught:
                test_e2e.test_official_checkpoint_e2e("res2net50-26w-4s-in1k", tmp_path)
            assert caught.value is build_stop
            assert events == [libraries["backend"].name, libraries["model"].name, "model_dir", "build"]
            assert [(row["stage"], row["status"]) for row in recorder.data["timing"]] == [
                ("setup", "passed"), ("build", "failed")
            ]
    finally:
        e2e_evidence._ACTIVE.reset(token)


def test_layout_reads_the_scale_from_the_convolution_chain(tmp_path: Path) -> None:
    """The last chunk skips the chain, so scale is one more than the rungs."""
    for scale in (2, 4, 8):
        layout = model._layout(Checkpoint.open(_checkpoint(tmp_path / f"s{scale}", scale=scale)))
        assert layout["scale"] == scale


def test_layout_rejects_a_bottleneck_with_no_chain(tmp_path: Path) -> None:
    """A plain ResNet has every stage but no convs chain, so it is rejected."""
    tensors: dict[str, np.ndarray] = {}
    for stage in ("layer1", "layer2", "layer3", "layer4"):
        tensors[f"{stage}.0.conv1.weight"] = _random(4, 4, 1, 1)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="no convs chain"):
        model._layout(Checkpoint.open(tmp_path))


def test_layout_separates_the_two_stem_shapes(tmp_path: Path) -> None:
    plain = model._layout(Checkpoint.open(_checkpoint(tmp_path / "a", deep_stem=False)))
    deep = model._layout(Checkpoint.open(_checkpoint(tmp_path / "b", deep_stem=True)))
    assert not plain["deep_stem"]
    assert deep["deep_stem"]


def test_pooled_shortcut_is_decided_by_index_zero_not_index_one(tmp_path: Path) -> None:
    """Index 1 exists in both layouts, so it cannot tell them apart.

    In the plain shortcut index 1 is the norm; in the pooled one it is the
    convolution. Only index 0 distinguishes them.
    """
    plain = model._layout(Checkpoint.open(_checkpoint(tmp_path / "a", pooled_shortcut=False)))
    pooled = model._layout(Checkpoint.open(_checkpoint(tmp_path / "b", pooled_shortcut=True)))
    assert not plain["pooled_shortcut"]
    assert pooled["pooled_shortcut"]


def test_pooled_shortcut_rejects_a_block_with_no_projection() -> None:
    with pytest.raises(ValueError, match="no projection shortcut"):
        model._pooled_shortcut(frozenset({"layer1.0.conv1.weight"}))


def test_weights_read_every_rung_of_the_chain(tmp_path: Path) -> None:
    checkpoint = Checkpoint.open(_checkpoint(tmp_path, scale=8))
    layout = model._layout(checkpoint)
    weights = model._weights(checkpoint, layout, np.float32)
    for rung in range(7):
        assert f"layer1.0.convs.{rung}.weight" in weights
    assert "layer1.0.convs.7.weight" not in weights


def test_fold_reproduces_the_batch_norm_it_replaces(tmp_path: Path) -> None:
    checkpoint = Checkpoint.open(_checkpoint(tmp_path))
    weight, bias = model._fold_norm(checkpoint, "conv1.weight", "bn1", np.float32)
    raw = checkpoint.tensor("conv1.weight")
    gamma = checkpoint.tensor("bn1.weight")
    beta = checkpoint.tensor("bn1.bias")
    mean = checkpoint.tensor("bn1.running_mean")
    variance = checkpoint.tensor("bn1.running_var")
    scale = gamma / np.sqrt(variance + model._BATCH_NORM_EPSILON)
    np.testing.assert_allclose(weight, raw * scale.reshape(-1, 1, 1, 1), rtol=1e-6)
    np.testing.assert_allclose(bias, beta - mean * scale, rtol=1e-6)


def test_read_config_rejects_another_family(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"architecture": "seresnet50"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unsupported timm Res2Net model identity"):
        model._read_config(tmp_path)


def test_preprocess_config_rejects_a_degenerate_std() -> None:
    raw = {
        "num_classes": 5,
        "pretrained_cfg": {
            "input_size": [3, 224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.0, 0.225],
            "crop_pct": 0.875,
            "interpolation": "bilinear",
        },
    }
    with pytest.raises(ValueError, match="preprocessing or classifier config is invalid"):
        model._preprocess_config(raw)


class _RecordingConfig:
    """Records the builder flags a build would clear."""

    def __init__(self) -> None:
        self.cleared: list[object] = []

    def clear_flag(self, flag: object) -> None:
        self.cleared.append(flag)


def test_fp32_builds_switch_off_the_reduced_precision_path(monkeypatch) -> None:
    """An fp32 build must not silently run convolutions in TF32.

    TF32 keeps ten mantissa bits. On res2net50_26w_8s that is enough to change
    the predicted class against timm, so fp32 has to mean fp32.
    """
    trt = SimpleNamespace(BuilderFlag=SimpleNamespace(TF32=object()))
    monkeypatch.setattr(model, "trt", trt)

    config = _RecordingConfig()
    model._configure_precision(config, "fp32")
    assert config.cleared == [trt.BuilderFlag.TF32]


def test_fp16_builds_leave_the_precision_flags_alone() -> None:
    config = _RecordingConfig()
    model._configure_precision(config, "fp16")
    assert config.cleared == []
