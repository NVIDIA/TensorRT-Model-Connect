# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark build arguments follow the selected owner's declaration."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tensorrt_model_connect import family_cli
import trtmc_benchmark.builder as benchmark_builder
from trtmc_benchmark.builder import BundleBuilder, _build_command
from trtmc_benchmark.catalog import ManifestCatalog, resolve_case
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


@pytest.mark.parametrize("output_flags", [["-o", "--output"], ["--artifact"], None])
def test_declared_build_subprocess_publishes_atomically(
    owner_manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output_flags,
) -> None:
    owner = family_cli._root() / "example_owner"
    (owner.parent / "__init__.py").write_text('"""Isolated test families."""\n')
    (owner / "__init__.py").write_text('"""CPU command fixture."""\n')
    (owner / "model.py").write_text('''import json
from pathlib import Path

def build(*, model, output, precision, tile_tokens=96, retained_layers=(), paged=False, fail=False):
    values = {"model": str(model), "output": str(output), "precision": precision,
              "tile_tokens": tile_tokens, "retained_layers": list(retained_layers), "paged": paged}
    Path(output).write_text(json.dumps(values))
    if fail:
        raise RuntimeError("requested owner failure after writing temporary output")
    return 0
''')
    declaration = owner / "cli.json"
    descriptor = json.loads(declaration.read_text())
    output = descriptor["commands"][0]["arguments"][1]
    if output_flags is None:
        output.pop("flags")
    else:
        output["flags"] = output_flags
    descriptor["commands"][0]["arguments"].append({
        "name": "fail", "type": "bool", "flags": ["--fail"], "action": "store_true", "default": False,
    })
    declaration.write_text(json.dumps(descriptor))
    checkpoint = tmp_path / "snapshots" / ("a" * 40)
    checkpoint.mkdir(parents=True)
    manifest = json.loads(owner_manifest.read_text())
    manifest["hf_id"] = str(checkpoint)
    manifest["testcases"] = [{"name": "smoke", "prompt": "Hello", "max_new_tokens": 1}]
    owner_manifest.write_text(json.dumps(manifest))
    find_spec = benchmark_builder.importlib.util.find_spec

    def fixture_spec(name, *arguments, **keywords):
        if name == "families.example_owner":
            return SimpleNamespace(origin=str(owner / "__init__.py"))
        return find_spec(name, *arguments, **keywords)

    monkeypatch.setattr(benchmark_builder.importlib.util, "find_spec", fixture_spec)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((
        str(owner.parent.parent), str(Path(family_cli.__file__).resolve().parents[1]),
    )))
    checked_outputs = []

    def validate_output(path, _model, _runtime_root):
        # This fixture tests process execution and publication; native bundle
        # inspection has independent format/identity coverage.
        payload = json.loads(path.read_text())
        assert payload["tile_tokens"] == 96
        assert payload["retained_layers"] == [2, 7]
        assert payload["paged"] is True
        checked_outputs.append(path)

    monkeypatch.setattr(benchmark_builder, "_validate_bundle", validate_output)
    model = ManifestCatalog().resolve(str(owner_manifest))
    builder = BundleBuilder(tmp_path / "cache")
    case = resolve_case(model, builder.provisional_path(model))
    _, built = builder.prepare([case], allow_build=True, rebuild=False, dry_run=False)
    assert built[0].status == "built"
    assert checked_outputs[0] != case.bundle_path
    payload = json.loads(case.bundle_path.read_text())
    assert Path(payload["output"]) == checked_outputs[0]
    assert not checked_outputs[0].exists()
    assert not list(case.bundle_path.parent.glob(".trtmc-bench-*.bundle"))
    original = case.bundle_path.read_bytes()
    receipt = case.bundle_path.with_suffix(".bundle.benchmark.json")
    original_receipt = receipt.read_bytes()
    _, reused = builder.prepare([case], allow_build=False, rebuild=False, dry_run=False)
    assert reused[0].status == "reused"

    failed = replace(case, model=replace(model, build_settings={**model.build_settings, "fail": True}))
    with pytest.raises(BenchmarkError, match="failed with exit code"):
        builder.prepare([failed], allow_build=True, rebuild=False, dry_run=False)
    assert case.bundle_path.read_bytes() == original
    assert receipt.read_bytes() == original_receipt
    assert "requested owner failure" in (case.bundle_path.parent / "build.stderr.log").read_text()
    assert not list(case.bundle_path.parent.glob(".trtmc-bench-*.bundle"))
    with pytest.raises(BenchmarkError, match="no matching immutable build identity"):
        builder.prepare([failed], allow_build=False, rebuild=False, dry_run=False)


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


def test_invalid_owner_parallelism_does_not_hide_healthy_catalog_entries(owner_manifest: Path) -> None:
    root = family_cli._root()
    declaration = root / "example_owner/cli.json"
    descriptor = json.loads(declaration.read_text())
    descriptor["commands"][0]["arguments"].append({
        "name": "tensor_parallel_size", "type": "string", "flags": ["--parallel"],
    })
    declaration.write_text(json.dumps(descriptor))
    invalid = json.loads(owner_manifest.read_text())
    invalid["build"]["tensor_parallel_size"] = "auto"
    invalid_path = root / "example_owner/tests/manifests/example.json"
    invalid_path.parent.mkdir(parents=True)
    invalid_path.write_text(json.dumps(invalid))
    healthy = {**invalid, "name": "healthy", "family": "healthy_owner"}
    healthy.pop("build")
    healthy_path = root / "healthy_owner/tests/manifests/healthy.json"
    healthy_path.parent.mkdir(parents=True)
    healthy_path.write_text(json.dumps(healthy))

    catalog = ManifestCatalog(root)
    entries = {entry.name: entry for entry in catalog.entries()}
    assert set(entries) == {"example", "healthy"}
    assert entries["example"].status == "invalid"
    assert "invalid benchmark parallelism" in entries["example"].reason
    assert entries["healthy"].status == "ready"
    with pytest.raises(BenchmarkError, match="invalid benchmark parallelism"):
        catalog.resolve(str(invalid_path))


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
