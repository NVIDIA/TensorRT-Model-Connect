# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect import build_cli


def _stub_family_resolution(monkeypatch) -> None:
    support = SimpleNamespace(tasks=("example_task",), default_task="example_task")
    monkeypatch.setattr(build_cli, "load_model_metadata", lambda _model_dir: object())
    monkeypatch.setattr(
        build_cli,
        "resolve_family",
        lambda _metadata: ("example", support),
    )


def test_build_command_forwards_only_direct_inputs(monkeypatch, tmp_path: Path) -> None:
    captured = []
    monkeypatch.setattr(build_cli, "build", captured.append)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"patchtsmixer"}', encoding="utf-8")
    output = tmp_path / "model.bundle"

    assert (
        build_cli.main(
            [
                "build",
                str(model),
                "--output",
                str(output),
                "--task",
                "time_series_forecast",
                "--precision",
                "fp16",
                "--backend",
                "trt_rtx",
                "--max-sequence-length",
                "1024",
                "--image-height",
                "512",
                "--image-width",
                "768",
                "--video-num-frames",
                "17",
                "--max-batch-size",
                "3",
                "--tensor-parallel-size",
                "2",
                "--context-parallel-size",
                "4",
                "--quantization",
                "fp8",
                "--fp32-layer",
                "2",
                "--fp32-layer",
                "5",
                "--dynamic-kv-cache",
                "--verbose",
            ]
        )
        == 0
    )
    request = captured[0]
    assert request.model_dir == model
    assert request.output_path == output
    assert request.family == "patchtsmixer"
    assert request.task == "time_series_forecast"
    assert request.precision == "fp16"
    assert request.backend == "trt_rtx"
    assert request.max_sequence_length == 1024
    assert request.image_height == 512
    assert request.image_width == 768
    assert request.video_num_frames == 17
    assert request.max_batch_size == 3
    assert request.tensor_parallel_size == 2
    assert request.context_parallel_size == 4
    assert request.quantization == "fp8"
    assert request.fp32_layers == (2, 5)
    assert request.dynamic_kv_cache is True
    assert request.verbose is True


def test_build_command_uses_the_family_owned_default_task(monkeypatch, tmp_path: Path) -> None:
    captured = []
    monkeypatch.setattr(build_cli, "build", captured.append)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")

    assert build_cli.main(["build", str(model), "--output", str(tmp_path / "out.bundle")]) == 0

    assert captured[0].family == "gpt2"
    assert captured[0].task == "text_generation"
    assert captured[0].precision == "fp32"


def test_hugging_face_model_id_resolves_to_a_local_snapshot(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )

    assert build_cli._resolve_model("openai-community/gpt2", "revision-1") == tmp_path
    assert calls == [{"repo_id": "openai-community/gpt2", "revision": "revision-1"}]


def test_build_command_preserves_resolved_checkpoint_and_source_revisions(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint_revision = "b" * 40
    source_revision = "a" * 40
    snapshot = tmp_path / "models--example-org--example-model" / "snapshots" / checkpoint_revision
    snapshot.mkdir(parents=True)
    (snapshot / "metadata.json").write_text("{}", encoding="utf-8")
    captured = []

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **_kwargs: str(snapshot)),
    )
    monkeypatch.setenv("TRTMC_ENGINE_BUILD_REVISION", source_revision)
    monkeypatch.setattr(build_cli, "build", captured.append)
    _stub_family_resolution(monkeypatch)

    assert (
        build_cli.main(
            [
                "build",
                "example-org/example-model",
                "--revision",
                "main",
                "--output",
                str(tmp_path / "out.bundle"),
            ]
        )
        == 0
    )
    request = captured[0]
    assert request.model_dir == snapshot
    assert request.checkpoint_id == "example-org/example-model"
    assert request.checkpoint_revision == checkpoint_revision
    assert request.source_revision == source_revision


def test_local_snapshot_can_preserve_a_canonical_checkpoint_id(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint_revision = "b" * 40
    source_revision = "a" * 40
    snapshot = tmp_path / "downloaded-checkpoint"
    snapshot.mkdir()
    (snapshot / "metadata.json").write_text("{}", encoding="utf-8")
    captured = []

    monkeypatch.setenv("TRTMC_ENGINE_BUILD_REVISION", source_revision)
    monkeypatch.setattr(build_cli, "build", captured.append)
    _stub_family_resolution(monkeypatch)

    assert (
        build_cli.main(
            [
                "build",
                str(snapshot),
                "--checkpoint-id",
                "example-org/example-model",
                "--revision",
                checkpoint_revision,
                "--output",
                str(tmp_path / "out.bundle"),
            ]
        )
        == 0
    )
    request = captured[0]
    assert request.model_dir == snapshot
    assert request.checkpoint_id == "example-org/example-model"
    assert request.checkpoint_revision == checkpoint_revision


def test_build_command_derives_source_revision_from_the_checkout(
    monkeypatch, tmp_path: Path
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "metadata.json").write_text("{}", encoding="utf-8")
    captured = []
    monkeypatch.delenv("TRTMC_ENGINE_BUILD_REVISION", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setattr(build_cli, "build", captured.append)
    _stub_family_resolution(monkeypatch)

    assert build_cli.main(["build", str(model), "-o", str(tmp_path / "out.bundle")]) == 0

    repository = Path(__file__).resolve().parents[3]
    expected = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    assert captured[0].source_revision == expected


def test_local_checkpoint_rejects_a_non_exact_requested_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact 40-character Git SHA"):
        build_cli._checkpoint_revision(
            tmp_path / "local-checkpoint",
            requested="main",
            require_exact=False,
        )


def test_build_command_rejects_a_task_the_family_does_not_own(monkeypatch, tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")
    monkeypatch.setattr(build_cli, "build", lambda request: None)

    with pytest.raises(ValueError, match="does not support task 'embedding'"):
        build_cli.main(
            [
                "build",
                str(model),
                "--output",
                str(tmp_path / "out.bundle"),
                "--task",
                "embedding",
            ]
        )


def test_prepare_structure_dispatches_to_the_resolved_family(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"boltz2"}', encoding="utf-8")
    request = tmp_path / "request.yaml"
    request.write_text("version: 1\n", encoding="utf-8")
    output = tmp_path / "request.b2rq"
    cache = tmp_path / "cache"
    calls = []

    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return {"family": "boltz2", "cache_hit": False}

    monkeypatch.setattr(
        build_cli,
        "_load_family",
        lambda family: SimpleNamespace(prepare_structure_request=prepare),
    )

    assert (
        build_cli.main(
            [
                "prepare-structure",
                str(model),
                "--input",
                str(request),
                "--output",
                str(output),
                "--cache-dir",
                str(cache),
            ]
        )
        == 0
    )
    assert calls == [((model, request, output), {"cache_dir": cache})]
    assert json.loads(capsys.readouterr().out) == {
        "cache_hit": False,
        "family": "boltz2",
    }
