# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned source and compatibility contracts for the MiniMax-H3 reference."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from tensorrt_model_connect.families.minimax_h3.provenance import file_record
from tests.e2e.models.minimax_h3 import hf_reference
from tests.e2e.models.minimax_h3.e2e_plugins import reference
from tests.e2e_harness.contracts import RunContext
from tools.ci.model_reference_cache import parse_model_reference_contract


MODEL_DIR = Path(__file__).resolve().parent


def _git_checkout(tmp_path: Path) -> tuple[Path, str]:
    checkout = tmp_path / "transformers"
    entrypoint = checkout / "src" / "transformers" / "__init__.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text('__version__ = "test"\n', encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "initial"], check=True)
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return entrypoint, revision


def test_minimax_h3_declares_exact_diffusers_reference_cache() -> None:
    manifest_path = MODEL_DIR / "MODEL.toml"
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["model_reference_cache"] == {
        "repository": "https://github.com/huggingface/diffusers.git",
        "revision": "abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc",
        "relative_path": "minimax_h3/reference/diffusers-abc5e9bf71fd",
        "entrypoint": "src/diffusers/__init__.py",
        "environment_variable": "TRTMC_MINIMAX_H3_DIFFUSERS_REPO",
    }
    contract = parse_model_reference_contract(
        manifest,
        "minimax_h3",
        manifest_path,
        "premerge",
    )
    assert contract is not None
    assert contract.as_payload() == manifest["model_reference_cache"]


def test_reference_environment_prioritizes_pinned_diffusers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "diffusers"
    entrypoint = source / "src" / "diffusers" / "__init__.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text('__version__ = "test"\n', encoding="utf-8")
    monkeypatch.setenv("TRTMC_MINIMAX_H3_DIFFUSERS_REPO", str(source))
    monkeypatch.setenv("PYTHONPATH", "/existing/python/path")

    environment = reference._reference_environment(RunContext(case=SimpleNamespace()))

    assert environment["PYTHONPATH"].split(":")[0] == str(source / "src")
    assert environment["PYTHONPATH"].endswith(":/existing/python/path")
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_reference_evidence_is_resolved_from_context_and_passed_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts_root = tmp_path / "artifacts"
    e2e_root = artifacts_root / "e2e"
    e2e_root.mkdir(parents=True)
    evidence = artifacts_root / "model-reference-cache.json"
    evidence.write_text("{}\n", encoding="utf-8")

    output_dir = e2e_root / "case" / "hf_reference"
    output_dir.mkdir(parents=True)
    model_path = tmp_path / "model"
    model_path.mkdir()
    prompt = tmp_path / "prompt.json"
    prompt.write_text("{}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    case = SimpleNamespace(
        name="case",
        hf_id="unused",
        inputs={"num_inference_steps": 50, "prompt_file": str(prompt)},
        metadata={},
    )
    ctx = RunContext(case=case, artifacts_dir=str(e2e_root))
    monkeypatch.setattr(reference, "validate_fixed_profile", lambda _case: None)
    monkeypatch.setattr(reference, "artifact_dir", lambda *_args: output_dir)
    monkeypatch.setattr(reference, "_model_snapshot", lambda _case: model_path)
    monkeypatch.setattr(reference, "resolve_owned_file", lambda _path: prompt)
    monkeypatch.setattr(reference, "source_revision", lambda *_args: "a" * 40)
    monkeypatch.setattr(reference, "_reference_environment", lambda _ctx: {"PINNED": "1"})
    monkeypatch.setattr(reference.subprocess, "run", fake_run)

    reference.reference.run_stage(
        case,
        SimpleNamespace(name="end_to_end"),
        ctx,
    )

    command = captured["command"]
    assert command[command.index("--diffusers-evidence") + 1] == str(evidence.resolve())
    assert captured["environment"] == {"PINNED": "1"}


def test_reference_evidence_preserves_symlink_for_provenance_rejection(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    e2e_root = artifacts_root / "e2e"
    e2e_root.mkdir(parents=True)
    target = tmp_path / "evidence-target.json"
    target.write_text("{}\n", encoding="utf-8")
    evidence = artifacts_root / "model-reference-cache.json"
    evidence.symlink_to(target)

    selected = reference._reference_evidence_path(
        RunContext(case=SimpleNamespace(), artifacts_dir=str(e2e_root))
    )

    assert selected == evidence.absolute()
    assert selected.is_symlink()


def test_transformers_git_qualification_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entrypoint, revision = _git_checkout(tmp_path)
    monkeypatch.setattr(hf_reference, "TRANSFORMERS_COMPAT_REVISION", revision)
    monkeypatch.setattr(hf_reference, "BASE_TRANSFORMERS_ENTRYPOINT", entrypoint)
    monkeypatch.setattr(
        hf_reference, "BASE_TRANSFORMERS_ENTRYPOINT_RECORD", file_record(entrypoint)
    )

    record = hf_reference.qualified_transformers_source(entrypoint, "test")
    assert record["qualification"] == "clean_git_checkout"
    assert record["revision"] == revision

    monkeypatch.setattr(hf_reference, "TRANSFORMERS_COMPAT_REVISION", "f" * 40)
    with pytest.raises(ValueError, match="revision mismatch"):
        hf_reference.qualified_transformers_source(entrypoint, "test")


def test_transformers_git_qualification_rejects_dirty_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entrypoint, revision = _git_checkout(tmp_path)
    monkeypatch.setattr(hf_reference, "TRANSFORMERS_COMPAT_REVISION", revision)
    entrypoint.write_text('__version__ = "dirty"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="tracked modifications"):
        hf_reference.qualified_transformers_source(entrypoint, "test")


def test_transformers_base_qualification_is_exact_and_honest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entrypoint = tmp_path / "immutable" / "transformers" / "__init__.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text('__version__ = "5.2.0"\n', encoding="utf-8")
    monkeypatch.setattr(hf_reference, "BASE_TRANSFORMERS_ENTRYPOINT", entrypoint)
    monkeypatch.setattr(
        hf_reference, "BASE_TRANSFORMERS_ENTRYPOINT_RECORD", file_record(entrypoint)
    )

    record = hf_reference.qualified_transformers_source(entrypoint, "5.2.0")

    assert record == {
        "qualification": "immutable_base_5_2_plus_local_shim",
        "version": "5.2.0",
        "entrypoint": str(entrypoint),
        "entrypoint_record": file_record(entrypoint),
    }
    assert "revision" not in record
    with pytest.raises(ValueError, match="version mismatch"):
        hf_reference.qualified_transformers_source(entrypoint, "5.2.1")

    entrypoint.write_text('__version__ = "5.2.x"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="entrypoint mismatch"):
        hf_reference.qualified_transformers_source(entrypoint, "5.2.0")


def test_local_processor_compat_is_bound_and_stable() -> None:
    processor = SimpleNamespace(
        image_token_ids=[10, 11],
        video_token_id=20,
        audio_token_ids=[30],
    )
    source = {"qualification": "immutable_base_5_2_plus_local_shim"}

    label, identity = hf_reference.prepare_processor_compat(processor, source)

    assert label == "local-create-mm-token-type-ids-for-transformers-5.2.0"
    assert processor.create_mm_token_type_ids([[0, 10, 20, 30, 11]]) == [[0, 1, 2, 3, 1]]
    hf_reference.validate_processor_method_unchanged(processor, identity)

    processor.create_mm_token_type_ids = lambda _inputs: []
    with pytest.raises(ValueError, match="changed during the run"):
        hf_reference.validate_processor_method_unchanged(processor, identity)


def test_git_transformers_requires_its_own_processor_helper() -> None:
    processor = SimpleNamespace()

    with pytest.raises(ValueError, match="no callable create_mm_token_type_ids"):
        hf_reference.prepare_processor_compat(processor, {"qualification": "clean_git_checkout"})
