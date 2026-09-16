# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark build arguments follow the selected owner's declaration."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tensorrt_model_connect import family_cli
from trtmc_benchmark.builder import _build_command
from trtmc_benchmark.catalog import ManifestCatalog
from trtmc_benchmark.types import BenchmarkError


@pytest.fixture
def owner_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "families"
    owner = root / "example_owner"
    owner.mkdir(parents=True)
    descriptor = {
        "version": 1,
        "commands": [{
            "name": "build", "help": "Build example", "executor": "python", "handler": "model:build",
            "arguments": [
                {"name": "model", "type": "path"},
                {"name": "output", "type": "path", "flags": ["-o", "--output"], "required": True},
                {"name": "precision", "type": "string", "flags": ["--precision"], "choices": ["fp16", "fp32"]},
                {"name": "tile_tokens", "type": "int", "flags": ["--owner-tile-size"], "default": 96},
                {"name": "retained_layers", "type": "int", "flags": ["--owner-layer"], "action": "append"},
                {"name": "paged", "type": "bool", "flags": ["--owner-paged"], "action": "store_true"},
            ],
        }],
    }
    (owner / "cli.json").write_text(json.dumps(descriptor))
    # No model.py exists: catalog/command construction must not import the owner.
    monkeypatch.setattr(family_cli, "_root", lambda: root)
    manifest = tmp_path / "example.json"
    manifest.write_text(json.dumps({
        "name": "example", "bundle": "example.bundle", "family": "example_owner",
        "task": "text_generation", "precision": "fp16", "hf_id": "example/checkpoint",
        "testcases": [{"name": "smoke"}],
        "build": {"retained_layers": [2, 7], "paged": True},
    }))
    return manifest


def test_declared_build_uses_owner_names_and_types(
    owner_manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ManifestCatalog(owner_manifest.parent).resolve(str(owner_manifest))
    command = _build_command(model, tmp_path / "checkpoint", tmp_path / "result.bundle", ())
    assert command[:5] == (sys.executable, "-m", "tensorrt_model_connect", "example_owner", "build")
    assert not any(value.startswith("--owner-tile-size") for value in command)
    assert "--tile-tokens" not in command
    assert "--task" not in command
    received = []

    def build(**values):
        received.append(values)
        return 0

    monkeypatch.setitem(sys.modules, "families.example_owner.model", SimpleNamespace(build=build))
    assert family_cli.main(command[3:]) == 0
    assert received == [{
        "model": tmp_path / "checkpoint", "output": tmp_path / "result.bundle",
        "precision": "fp16", "tile_tokens": 96, "retained_layers": [2, 7], "paged": True,
    }]
    declaration = family_cli._root() / "example_owner/cli.json"
    descriptor = json.loads(declaration.read_text())
    descriptor["commands"][0]["arguments"][3]["default"] = 128
    declaration.write_text(json.dumps(descriptor))
    updated = _build_command(model, tmp_path / "checkpoint", tmp_path / "result.bundle", ())
    assert updated == command
    assert family_cli.main(updated[3:]) == 0
    assert received[-1]["tile_tokens"] == 128


@pytest.mark.parametrize("update", [
    {"build": {"typo_tokens": 128}},
    {"typo_tokens": 128},
    {"build": {"tile_tokens": "128"}},
    {"build": {"paged": 1}},
    {"build": {"retained_layers": [1, "2"]}},
    {"build": {"output": "elsewhere.bundle"}},
    {"build": {"tensor_parallel_size": 1}},
    {"tile_tokens": 128, "build": {"tile_tokens": 256}},
])
def test_declared_build_rejects_unowned_or_invalid_values(owner_manifest: Path, update: dict) -> None:
    payload = json.loads(owner_manifest.read_text())
    payload.update(update)
    owner_manifest.write_text(json.dumps(payload))
    with pytest.raises(BenchmarkError):
        ManifestCatalog(owner_manifest.parent).resolve(str(owner_manifest))


def test_declared_build_rejects_unknown_programmatic_settings(owner_manifest: Path, tmp_path: Path) -> None:
    model = ManifestCatalog(owner_manifest.parent).resolve(str(owner_manifest))
    model = replace(model, build_settings={**model.build_settings, "undeclared": True})
    with pytest.raises(BenchmarkError, match="unknown command arguments"):
        _build_command(model, tmp_path / "checkpoint", tmp_path / "out.bundle", ())


def test_family_parallel_default_is_checked_without_freezing_cli_defaults(owner_manifest: Path) -> None:
    declaration = family_cli._root() / "example_owner/cli.json"
    descriptor = json.loads(declaration.read_text())
    descriptor["commands"][0]["arguments"].append({
        "name": "tensor_parallel_size", "type": "int", "flags": ["--tensor-parallel-size"], "default": 2,
    })
    declaration.write_text(json.dumps(descriptor))
    with pytest.raises(BenchmarkError, match="distributed"):
        ManifestCatalog(owner_manifest.parent).resolve(str(owner_manifest))


def test_parallel_provenance_does_not_create_an_undeclared_cli_option(owner_manifest: Path, tmp_path: Path) -> None:
    raw = json.loads(owner_manifest.read_text())
    raw.update(tensor_parallel_size=1, context_parallel_size=1)
    owner_manifest.write_text(json.dumps(raw))
    model = ManifestCatalog(owner_manifest.parent).resolve(str(owner_manifest))
    command = _build_command(model, tmp_path / "checkpoint", tmp_path / "result.bundle", ())
    assert model.parallelism == (1, 1)
    assert model.summary()["parallelism"] == {"tensor_parallel_size": 1, "context_parallel_size": 1}
    assert not any("parallel-size" in argument for argument in command)
    raw["tensor_parallel_size"] = 2
    owner_manifest.write_text(json.dumps(raw))
    with pytest.raises(BenchmarkError, match="distributed"):
        ManifestCatalog(owner_manifest.parent).resolve(str(owner_manifest))


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, None])
def test_parallel_provenance_requires_positive_integers(owner_manifest: Path, value: object) -> None:
    raw = json.loads(owner_manifest.read_text())
    raw["tensor_parallel_size"] = value
    owner_manifest.write_text(json.dumps(raw))
    with pytest.raises(BenchmarkError, match="positive integer"):
        ManifestCatalog(owner_manifest.parent).resolve(str(owner_manifest))


def test_family_without_build_command_keeps_legacy_entrypoint(owner_manifest: Path, tmp_path: Path) -> None:
    root = family_cli._root()
    descriptor = root / "example_owner/cli.json"
    payload = json.loads(descriptor.read_text())
    payload["commands"][0]["name"] = "prepare"
    descriptor.write_text(json.dumps(payload))
    raw = json.loads(owner_manifest.read_text())
    raw.pop("build")
    raw["max_sequence_length"] = 128
    owner_manifest.write_text(json.dumps(raw))
    model = ManifestCatalog(owner_manifest.parent).resolve(str(owner_manifest))
    command = _build_command(model, tmp_path / "checkpoint", tmp_path / "result.bundle", ())
    assert command[:4] == (sys.executable, "-m", "tensorrt_model_connect", "build")
    assert command[command.index("--max-sequence-length") + 1] == "128"
