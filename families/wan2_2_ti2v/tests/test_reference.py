# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contract tests for the model-owned Wan2.2 references."""

from __future__ import annotations

import json
import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np
import pytest

from . import frame_accuracy, official_reference


TEST_ROOT = Path(__file__).resolve().parent
MODEL_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
SOURCE_REVISION = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"


def _write_png(path: Path, width: int, height: int, value: int) -> None:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes([value, value, value]) * width
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(row * height))
        + chunk(b"IEND", b"")
    )


def _frame_paths(root: Path, count: int) -> list[str]:
    root.mkdir()
    paths = []
    for index in range(count):
        path = root / f"frame_{index:04d}.png"
        path.touch()
        paths.append(str(path))
    return paths


def test_manifests_keep_the_original_reference_routes() -> None:
    l0 = json.loads((TEST_ROOT / "manifests/wan22-ti2v-5b-l0.json").read_text())
    full = json.loads((TEST_ROOT / "manifests/wan22-ti2v-5b.json").read_text())
    source = json.loads((TEST_ROOT / "reference-source.json").read_text())

    assert l0["hf_revision"] == full["hf_revision"] == MODEL_REVISION
    assert l0["testcases"][0]["reference_backend"] == "invariant_only"
    assert full["testcases"][0]["reference_backend"] == "wan_official"
    assert source == {
        "repository": "Wan-Video/Wan2.2",
        "revision": SOURCE_REVISION,
    }


def test_official_reference_consumes_the_materialized_raw_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    entrypoint = source / official_reference.SOURCE_ENTRYPOINT
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# pinned official Wan entrypoint\n", encoding="utf-8")
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
    assert not (model_dir / "model_index.json").exists()
    monkeypatch.setenv(official_reference.SOURCE_ENVIRONMENT, str(source))
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        captured["command"] = command
        captured["kwargs"] = kwargs
        frames = tmp_path / "artifacts/reference-frames"
        for index in range(5):
            _write_png(frames / f"frame_{index:04d}.png", 32, 16, index * 20)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(official_reference.subprocess, "run", fake_run)
    output = official_reference.generate(
        model_dir,
        tmp_path / "artifacts",
        prompt="A cat wearing boxing gloves",
        height=16,
        width=32,
        num_frames=5,
        num_steps=7,
        guidance_scale=5.0,
        flow_shift=5.0,
        seed=42,
        timeout_s=60,
    )

    command = captured["command"]
    assert isinstance(command, list)
    script = command[2]
    assert f"model_ref = {str(model_dir)!r}" in script
    assert f"official_source = {str(source)!r}" in script
    assert "sys.path.insert(0, official_source)" in script
    assert "from wan.configs.wan_ti2v_5B import ti2v_5B" in script
    assert "from wan.textimage2video import WanTI2V" in script
    assert "checkpoint_dir=model_ref" in script
    assert "DiffusionPipeline" not in script
    assert "from_pretrained" not in script
    assert "types.ModuleType" not in script
    assert 'sys.modules["wan"]' not in script
    assert 'sys.modules["easydict"]' not in script
    assert 'sys.modules["imageio"]' not in script
    assert "wan_model_module.flash_attention" not in script
    assert "setattr(tokenizer" not in script
    assert "Official Wan tokenizer {role} binding mismatch" in script
    assert output["num_frames"] == 5
    assert [Path(path).name for path in output["frame_paths"]] == [
        f"frame_{index:04d}.png" for index in range(5)
    ]
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["timeout"] == 60
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"


def test_official_reference_declares_its_real_import_dependencies() -> None:
    requirements = {
        line
        for line in (TEST_ROOT.parent / "requirements.txt").read_text().splitlines()
        if line and not line.startswith("#")
    }
    assert {
        "accelerate>=1.1.1",
        "dashscope",
        "decord",
        "diffusers>=0.31.0",
        "easydict",
        "flash-attn",
        "imageio[ffmpeg]",
        "librosa",
        "opencv-python>=4.9.0.80",
        "peft",
        "tokenizers>=0.20.3",
        "tqdm",
        "transformers>=4.49.0,<=4.51.3",
    } <= requirements


def test_official_reference_requires_the_declared_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(official_reference.SOURCE_ENVIRONMENT, raising=False)
    with pytest.raises(RuntimeError, match=official_reference.SOURCE_ENVIRONMENT):
        official_reference._source()


def test_frame_accuracy_reads_every_frame(tmp_path: Path) -> None:
    references = tmp_path / "references"
    actuals = tmp_path / "actuals"
    references.mkdir()
    actuals.mkdir()
    reference_paths = []
    actual_paths = []
    for index, value in enumerate((20, 40, 80, 60, 100)):
        name = f"frame_{index:04d}.png"
        reference_path = references / name
        actual_path = actuals / name
        _write_png(reference_path, 2, 1, value)
        _write_png(actual_path, 2, 1, value)
        reference_paths.append(str(reference_path))
        actual_paths.append(str(actual_path))

    exact = frame_accuracy.compare_png_sequences(reference_paths, actual_paths)
    assert exact["frame_count"] == 5.0
    assert exact["maximum_frame_rmse_uint8"] == 0.0
    assert exact["temporal_profile_correlation"] == pytest.approx(1.0)

    _write_png(Path(actual_paths[-1]), 2, 1, 200)
    changed = frame_accuracy.compare_png_sequences(reference_paths, actual_paths)
    assert changed["maximum_frame_rmse_uint8"] > 0.0


def test_frame_accuracy_rejects_noncontiguous_missing_and_unequal_frames(tmp_path: Path) -> None:
    references = _frame_paths(tmp_path / "references", 2)
    actuals = _frame_paths(tmp_path / "actuals", 2)
    renamed = Path(actuals[1]).with_name("frame_0002.png")
    Path(actuals[1]).rename(renamed)
    actuals[1] = str(renamed)
    with pytest.raises(ValueError, match="TensorRT frame list is not contiguous"):
        frame_accuracy.compare_png_sequences(references, actuals)

    Path(actuals[0]).unlink()
    with pytest.raises(ValueError, match="TensorRT frame files are missing"):
        frame_accuracy.compare_png_sequences(references, [actuals[0]])

    Path(actuals[0]).touch()
    with pytest.raises(ValueError, match="reference=2, TensorRT=1"):
        frame_accuracy.compare_png_sequences(references, [actuals[0]])


def test_frame_accuracy_traverses_all_121_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    references = _frame_paths(tmp_path / "references", 121)
    actuals = _frame_paths(tmp_path / "actuals", 121)
    pixels = {}
    for index, (reference, actual) in enumerate(zip(references, actuals)):
        frame = np.full((1, 1, 3), 50 + (index * index) % 150, dtype=np.uint8)
        pixels[reference] = frame
        pixels[actual] = frame
    loaded: list[str] = []

    def load_rgb(path: Path) -> np.ndarray:
        loaded.append(str(path))
        return pixels[str(path)]

    monkeypatch.setattr(frame_accuracy, "_load_rgb", load_rgb)
    metrics = frame_accuracy.compare_png_sequences(references, actuals)

    assert metrics["frame_count"] == 121.0
    assert loaded == [path for pair in zip(references, actuals) for path in pair]


def test_frame_accuracy_rejects_late_shape_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    references = _frame_paths(tmp_path / "references", 121)
    actuals = _frame_paths(tmp_path / "actuals", 121)
    pixels = {path: np.zeros((1, 1, 3), dtype=np.uint8) for path in [*references, *actuals]}
    pixels[actuals[-1]] = np.zeros((2, 1, 3), dtype=np.uint8)
    monkeypatch.setattr(frame_accuracy, "_load_rgb", lambda path: pixels[str(path)])

    with pytest.raises(ValueError, match="frame 120 shape mismatch"):
        frame_accuracy.compare_png_sequences(references, actuals)


def test_frame_accuracy_rejects_frozen_reference_and_detects_frozen_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    references = _frame_paths(tmp_path / "references", 5)
    actuals = _frame_paths(tmp_path / "actuals", 5)
    pixels = {path: np.full((1, 1, 3), 100, dtype=np.uint8) for path in references + actuals}
    monkeypatch.setattr(frame_accuracy, "_load_rgb", lambda path: pixels[str(path)])
    with pytest.raises(ValueError, match="reference video has no temporal activity"):
        frame_accuracy.compare_png_sequences(references, actuals)

    for index, reference in enumerate(references):
        pixels[reference] = np.full((1, 1, 3), 40 + index * index, dtype=np.uint8)
    metrics = frame_accuracy.compare_png_sequences(references, actuals)
    assert metrics["trt_temporal_mae_uint8"] == 0.0
    assert metrics["trt_active_transition_fraction"] == 0.0
    assert metrics["temporal_motion_ratio"] == 0.0


def test_frame_accuracy_detects_wrong_temporal_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    references = _frame_paths(tmp_path / "references", 121)
    actuals = _frame_paths(tmp_path / "actuals", 121)
    values = [(index * 73) % 256 for index in range(121)]
    swapped = [index ^ 1 if index < 120 else index for index in range(121)]
    pixels = {
        **{
            path: np.full((1, 1, 3), value, dtype=np.uint8)
            for path, value in zip(references, values)
        },
        **{
            path: np.full((1, 1, 3), values[index], dtype=np.uint8)
            for path, index in zip(actuals, swapped)
        },
    }
    monkeypatch.setattr(frame_accuracy, "_load_rgb", lambda path: pixels[str(path)])

    metrics = frame_accuracy.compare_png_sequences(references, actuals)
    assert 0.5 <= metrics["temporal_motion_ratio"] <= 2.0
    assert metrics["reference_active_transition_fraction"] > 0.9
    assert metrics["trt_active_transition_fraction"] > 0.9
    assert metrics["temporal_profile_correlation"] < 0.75


def test_official_reference_rejects_incomplete_noncontiguous_and_wrong_sized_frames(
    tmp_path: Path,
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    _write_png(frames / "frame_0000.png", 32, 16, 0)
    with pytest.raises(RuntimeError, match="incomplete frame sequence"):
        official_reference._validate_frames(
            frames, expected_count=2, expected_width=32, expected_height=16
        )

    _write_png(frames / "frame_0002.png", 32, 16, 0)
    with pytest.raises(RuntimeError, match="incomplete frame sequence"):
        official_reference._validate_frames(
            frames, expected_count=2, expected_width=32, expected_height=16
        )

    (frames / "frame_0002.png").unlink()
    with pytest.raises(RuntimeError, match="wrong dimensions"):
        official_reference._validate_frames(
            frames, expected_count=1, expected_width=16, expected_height=32
        )
