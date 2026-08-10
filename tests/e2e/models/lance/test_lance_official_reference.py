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


def test_lance_image_reference_checks_only_its_image_checkpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def snapshot_download(model_id: str, **kwargs) -> str:
        captured["model_id"] = model_id
        captured.update(kwargs)
        return str(tmp_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )

    assert (
        lance_official._cached_model_root(
            "bytedance-research/Lance",
            local_files_only=True,
        )
        == tmp_path
    )
    assert captured == {
        "model_id": "bytedance-research/Lance",
        "allow_patterns": [
            "Lance_3B/**",
            "Qwen2.5-VL-ViT/**",
        ],
        "local_files_only": True,
    }


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
    captured_local_files_only: list[bool] = []

    def resolve_model(_model_id, *, local_files_only):
        captured_local_files_only.append(local_files_only)
        return model_root

    def run(command, **_kwargs):
        captured["command"] = command
        output_dir = Path(command[command.index("--save_path_gen") + 1])
        output_dir.mkdir(parents=True)
        (output_dir / "result.json").write_text(
            '[{"answer": "reference answer"}]\n',
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(lance_official, "_official_source", lambda: source)
    monkeypatch.setattr(
        lance_official,
        "_cached_model_root",
        resolve_model,
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
    output = lance_official.plugin.run_stage(
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
            local_files_only=False,
        ),
    )

    command = captured["command"]
    visual_gen = command.index("--visual_gen")
    assert command[visual_gen + 1] == "true"
    text_template = command.index("--text_template")
    assert command[text_template + 1] == "true"
    resolution = command.index("--resolution")
    assert command[resolution + 1] == "image_512res"
    chdir_argument = next(
        value
        for value in command
        if value.startswith("--chdir=")
    )
    workspace = Path(chdir_argument.removeprefix("--chdir="))
    assert (workspace / "downloads").resolve() == model_root.resolve()
    preserved_result = artifact_dir / "official_output/official_reference_result.json"
    assert output.text == "reference answer"
    assert captured_local_files_only == [False]
    assert output.data["result_path"] == str(preserved_result)
    assert preserved_result.is_file()
    assert not (artifact_dir / "official_output/result.json").exists()
