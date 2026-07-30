# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lance official-reference platform compatibility contracts."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from tests.e2e.models.lance.e2e_plugins.references import lance_official


def test_lance_image_reference_provides_decord_import_without_video_support() -> None:
    environment = lance_official._image_reference_environment(
        {**os.environ, "PYTHONPATH": "/existing/python/path"}
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import decord; "
                "assert callable(decord.cpu); "
                "decord.VideoReader('unused.mp4')"
            ),
        ],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "image-only Lance reference" in result.stderr
    assert environment["PYTHONPATH"].endswith(":/existing/python/path")


def test_lance_image_reference_keeps_upstream_visual_generation_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "inference_lance.py").write_text("", encoding="utf-8")
    model_root = tmp_path / "model"
    (model_root / "Lance_3B").mkdir(parents=True)
    (model_root / "Qwen2.5-VL-ViT").mkdir()
    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    captured: dict[str, list[str]] = {}

    def run(command, **_kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(lance_official, "_official_source", lambda: source)
    monkeypatch.setattr(
        lance_official,
        "_cached_model_root",
        lambda _model_id: model_root,
    )
    monkeypatch.setattr(
        lance_official,
        "_case_artifact_dir",
        lambda *_args: str(artifact_dir),
    )
    monkeypatch.setattr(lance_official.subprocess, "run", run)
    monkeypatch.setattr(
        lance_official,
        "save_full_stderr",
        lambda *_args: ("", ""),
    )
    monkeypatch.setattr(
        lance_official,
        "_result_text",
        lambda _path: "reference answer",
    )

    lance_official.plugin.run_stage(
        SimpleNamespace(
            task_strategy="vision_language_generation",
            inputs={"image": str(image), "prompt": "Describe the image."},
            hf_id="bytedance-research/Lance",
            determinism={"seed": 42},
            name="lance-test",
        ),
        SimpleNamespace(name="full_generation"),
        SimpleNamespace(
            artifacts_dir=str(artifact_dir),
            reference_python_path=lambda: sys.executable,
        ),
    )

    command = captured["command"]
    visual_gen = command.index("--visual_gen")
    assert command[visual_gen + 1] == "true"
