# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from families.lance.tests.official_reference import (
    _image_only_source,
    _result_text,
    _source_dir,
    run_official_generation,
)


def test_reference_source_is_required_and_fail_closed(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TRTMC_REFERENCE_SOURCE_DIR", raising=False)
    with pytest.raises(AssertionError, match="TRTMC_REFERENCE_SOURCE_DIR"):
        _source_dir()
    monkeypatch.setenv("TRTMC_REFERENCE_SOURCE_DIR", str(tmp_path))
    with pytest.raises(AssertionError, match="inference_lance.py"):
        _source_dir()


def _fake_source(root) -> None:
    dataset = root / "data/datasets_custom/validation_dataset.py"
    dataset.parent.mkdir(parents=True)
    (root / "data/__init__.py").write_text("", encoding="utf-8")
    (dataset.parent / "__init__.py").write_text(
        "from .validation_dataset import read\n", encoding="utf-8"
    )
    dataset.write_text(
        "import decord\n"
        "from decord import VideoReader\n"
        "def read(video: VideoReader):\n"
        "    return VideoReader, decord, video\n",
        encoding="utf-8",
    )
    (root / "inference_lance.py").write_text("# pinned upstream entrypoint\n", encoding="utf-8")
    (root / "modeling").mkdir()


def test_image_source_removes_only_the_eager_video_import(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _fake_source(source)
    output = tmp_path / "output"
    output.mkdir()

    image_source = _image_only_source(source, output)
    patched = (image_source / "data/datasets_custom/validation_dataset.py").read_text()
    assert "import decord" not in patched
    assert "from decord" not in patched
    assert "video: VideoReader" not in patched
    assert "return VideoReader, decord, video" in patched
    assert (image_source / "modeling").is_symlink()


def test_result_requires_one_nonempty_answer(tmp_path) -> None:
    result = tmp_path / "result.json"
    result.write_text(json.dumps([{"answer": " White "}]), encoding="utf-8")
    assert _result_text(result) == "White"
    result.write_text("[]", encoding="utf-8")
    with pytest.raises(AssertionError):
        _result_text(result)


def test_generation_invokes_only_the_pinned_upstream_entrypoint(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    entrypoint = source / "inference_lance.py"
    _fake_source(source)
    model_root = tmp_path / "model"
    model = model_root / "Lance_3B"
    vision = model_root / "Qwen2.5-VL-ViT"
    model.mkdir(parents=True)
    vision.mkdir()
    (model / "llm_config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"model")
    (vision / "config.json").write_text("{}", encoding="utf-8")
    (vision / "vit.safetensors").write_bytes(b"vision")
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setenv("TRTMC_REFERENCE_SOURCE_DIR", str(source))

    def fake_run(command, **kwargs):
        assert Path(command[1]).name == entrypoint.name
        assert Path(command[1]).parent.name == "official_image_source"
        assert command[command.index("--task") + 1] == "x2t_image"
        assert command[command.index("--model_path") + 1] == str(model)
        assert kwargs["cwd"] == output / "official_workspace"
        assert "reference_compat" not in kwargs["env"].get("PYTHONPATH", "")
        result_dir = output / "official_output"
        result_dir.mkdir()
        (result_dir / "result.json").write_text(json.dumps([{"answer": "White"}]), encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("families.lance.tests.official_reference.subprocess.run", fake_run)
    assert run_official_generation(model_root, image, "Color?", output, 10) == "White"
