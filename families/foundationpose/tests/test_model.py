# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from families.foundationpose import model
from families.foundationpose.tests import test_e2e
from tensorrt_model_connect import BuildRequest


class RecordingWriter:
    def __init__(self) -> None:
        self.header = None
        self.sections: dict[str, bytes] = {}

    def set_header(self, **header) -> None:
        self.header = header

    def add_bytes(self, name: str, value: bytes) -> None:
        assert name not in self.sections
        self.sections[name] = value


def _checkpoint(root: Path) -> Path:
    root.mkdir()
    (root / "refine_model.onnx").write_bytes(b"refiner weights")
    (root / "score_model.onnx").write_bytes(b"scorer weights")
    return root


def _request(model_dir: Path, **changes) -> BuildRequest:
    request = BuildRequest(
        model_dir=model_dir,
        output_path=model_dir.parent / "foundationpose.bundle",
        family="foundationpose",
        task="pose_hypothesis_refinement",
        precision="fp16",
        max_sequence_length=1,
    )
    return replace(request, **changes)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_build_emits_two_exact_family_plans(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, precision: str
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    calls = []

    def build_engine(path: str, **options) -> bytes:
        calls.append((Path(path).name, options))
        return options["kind"].encode()

    monkeypatch.setattr(model, "build_foundationpose_engine", build_engine)
    writer = RecordingWriter()

    model.build(_request(checkpoint, precision=precision), writer)

    assert writer.header == {
        "family": "foundationpose",
        "task": "pose_hypothesis_refinement",
        "backend": "trt",
    }
    assert writer.sections == {"engine.plan": b"refiner", "score.plan": b"scorer"}
    assert calls == [
        (
            "refine_model.onnx",
            {"kind": "refiner", "max_batch": 42, "precision": precision, "verbose": False},
        ),
        (
            "score_model.onnx",
            {"kind": "scorer", "max_batch": 252, "precision": precision, "verbose": False},
        ),
    ]


@pytest.mark.parametrize("missing", ["refine_model.onnx", "score_model.onnx"])
def test_build_requires_both_exact_weight_files(tmp_path: Path, missing: str) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    (checkpoint / missing).unlink()

    with pytest.raises(FileNotFoundError, match=missing):
        model.build(_request(checkpoint), RecordingWriter())


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"task": "classification"}, "task=pose_hypothesis_refinement"),
        ({"precision": "bf16"}, "fp16 or fp32"),
        ({"max_sequence_length": 2}, "max_sequence_length=1"),
        ({"image_height": 160}, "image_height"),
        ({"image_width": 160}, "image_width"),
        ({"video_num_frames": 1}, "video_num_frames"),
        ({"max_batch_size": 2}, "max_batch_size"),
        ({"tensor_parallel_size": 2}, "tensor parallelism"),
        ({"context_parallel_size": 2}, "context parallelism"),
        ({"quantization": "fp8"}, "quantization"),
        ({"fp32_layers": (0,)}, "mixed-precision"),
    ],
)
def test_build_rejects_every_unsupported_profile(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    with pytest.raises((ValueError, NotImplementedError), match=message):
        model.build(_request(tmp_path / "missing", **changes), RecordingWriter())


def test_builder_has_no_environment_or_compatibility_side_channel() -> None:
    source = (Path(model.__file__).with_name("builder.py")).read_text(encoding="utf-8")
    assert "trt_compat" not in source
    assert "TRTMC_" not in source
    assert "hasattr(" not in source


def test_manifest_declares_the_exact_ngc_inputs() -> None:
    path = Path(__file__).with_name("manifests") / "foundationpose-ngc-1.0.1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["external_files"] == [
        {
            "path": "refine_model.onnx",
            "url": "https://api.ngc.nvidia.com/v2/models/nvidia/isaac/foundationpose/versions/1.0.1_onnx/files/refine_model.onnx",
        },
        {
            "path": "score_model.onnx",
            "url": "https://api.ngc.nvidia.com/v2/models/nvidia/isaac/foundationpose/versions/1.0.1_onnx/files/score_model.onnx",
        },
    ]


def test_e2e_requires_trusted_external_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTMC_FOUNDATIONPOSE_MODEL_DIR", raising=False)

    with pytest.raises(AssertionError, match="TRTMC_FOUNDATIONPOSE_MODEL_DIR"):
        test_e2e._model_dir({"external_files": []})
