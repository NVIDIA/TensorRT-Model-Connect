# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lance official-reference platform compatibility contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

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
            revision="7395315758865e6f56ab87ad06a88c7ac172f056",
            local_files_only=True,
        )
        == tmp_path
    )
    assert captured == {
        "model_id": "bytedance-research/Lance",
        "allow_patterns": [
            "Lance_3B/**",
            "Qwen2.5-VL-ViT/**",
            "Wan2.2_VAE.pth",
        ],
        "revision": "7395315758865e6f56ab87ad06a88c7ac172f056",
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


def test_lance_image_reference_adds_explicit_sdpa_fallback() -> None:
    environment = lance_official._image_reference_environment(
        {"TRTMC_LANCE_REFERENCE_ATTENTION_BACKEND": "torch_sdpa"}
    )

    assert environment["PYTHONPATH"].split(os.pathsep)[:2] == [
        str(lance_official._IMAGE_REFERENCE_ATTENTION_COMPAT),
        str(lance_official._IMAGE_REFERENCE_COMPAT),
    ]


def test_lance_sdpa_fallback_is_importable_without_distribution_metadata() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.metadata as metadata; "
                "from flash_attn import flash_attn_varlen_func; "
                "assert callable(flash_attn_varlen_func); "
                "\ntry:\n metadata.version('flash_attn')\n"
                "except metadata.PackageNotFoundError:\n pass\n"
                "else:\n raise AssertionError('fallback must not publish metadata')"
            ),
        ],
        env=lance_official._image_reference_environment(
            {
                **os.environ,
                "TRTMC_LANCE_REFERENCE_ATTENTION_BACKEND": "torch_sdpa",
            }
        ),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_lance_sdpa_vit_overlay_preserves_checkpoint_and_changes_only_config(
    tmp_path: Path,
) -> None:
    vit_path = tmp_path / "Qwen2.5-VL-ViT"
    vit_path.mkdir()
    config_path = vit_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "_attn_implementation": "flash_attention_2",
                "hidden_size": 1280,
            }
        ),
        encoding="utf-8",
    )
    weights = vit_path / "vit.safetensors"
    weights.write_bytes(b"weights")

    overlay = lance_official._image_reference_vit_path(
        tmp_path / "artifacts",
        vit_path,
        attention_backend="torch_sdpa",
    )

    assert overlay != vit_path
    assert json.loads((overlay / "config.json").read_text(encoding="utf-8")) == {
        "_attn_implementation": "sdpa",
        "hidden_size": 1280,
    }
    assert (overlay / "vit.safetensors").is_symlink()
    assert (overlay / "vit.safetensors").resolve() == weights.resolve()
    assert json.loads(config_path.read_text(encoding="utf-8"))[
        "_attn_implementation"
    ] == "flash_attention_2"


def test_lance_native_flash_attention_uses_original_vit_checkpoint(
    tmp_path: Path,
) -> None:
    vit_path = tmp_path / "Qwen2.5-VL-ViT"
    vit_path.mkdir()

    assert (
        lance_official._image_reference_vit_path(
            tmp_path / "artifacts",
            vit_path,
            attention_backend="flash_attn",
        )
        == vit_path
    )


def test_lance_image_reference_keeps_explicit_native_flash_attention() -> None:
    environment = lance_official._image_reference_environment(
        {"TRTMC_LANCE_REFERENCE_ATTENTION_BACKEND": "flash_attn"}
    )

    assert str(lance_official._IMAGE_REFERENCE_ATTENTION_COMPAT) not in environment[
        "PYTHONPATH"
    ].split(os.pathsep)


def test_lance_image_reference_rejects_unknown_attention_backend() -> None:
    with pytest.raises(RuntimeError, match="unsupported Lance reference attention backend"):
        lance_official._image_reference_environment(
            {"TRTMC_LANCE_REFERENCE_ATTENTION_BACKEND": "automatic"}
        )


def test_lance_image_reference_keeps_upstream_visual_generation_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "inference_lance.py").write_text("", encoding="utf-8")
    model_root = tmp_path / "model"
    (model_root / "Lance_3B").mkdir(parents=True)
    vit_path = model_root / "Qwen2.5-VL-ViT"
    vit_path.mkdir()
    (vit_path / "config.json").write_text(
        '{"_attn_implementation": "flash_attention_2"}\n',
        encoding="utf-8",
    )
    (vit_path / "vit.safetensors").write_bytes(b"weights")
    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    captured: dict[str, list[str]] = {}
    captured_local_files_only: list[bool] = []

    def resolve_model(_model_id, *, revision, local_files_only):
        assert revision == "7395315758865e6f56ab87ad06a88c7ac172f056"
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
    monkeypatch.setenv(
        "TRTMC_LANCE_REFERENCE_ATTENTION_BACKEND",
        "torch_sdpa",
    )
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
            hf_revision="7395315758865e6f56ab87ad06a88c7ac172f056",
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
    command_vit_path = Path(command[command.index("--vit_path") + 1])
    assert command_vit_path != vit_path
    assert json.loads(
        (command_vit_path / "config.json").read_text(encoding="utf-8")
    )["_attn_implementation"] == "sdpa"
    assert (command_vit_path / "vit.safetensors").resolve() == (
        vit_path / "vit.safetensors"
    ).resolve()
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
