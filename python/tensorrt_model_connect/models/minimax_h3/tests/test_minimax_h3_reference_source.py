# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned source and compatibility contracts for the MiniMax-H3 reference."""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from tensorrt_model_connect.models.minimax_h3.provenance import file_record
from tensorrt_model_connect.models.minimax_h3.tests import hf_reference
from tensorrt_model_connect.models.minimax_h3.tests.e2e_plugins import reference
from tests.e2e_harness import orchestrator
from tests.e2e_harness.artifact_sink import FileArtifactSink
from tests.e2e_harness.contracts import E2EResult, RunContext, StageOutput
from tests.e2e_harness.manifest_loader import load_model_manifest
from tools.ci.model_reference_cache import parse_model_reference_contract


MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parents[4]


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
    manifest_path = MODEL_DIR.parent / "MODEL.toml"
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


def test_model_plugin_reference_uses_validation_revision_without_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = SimpleNamespace(
        metadata={
            "validation_sample_id": "fixed-profile",
            "reference_source_revision": "a" * 40,
        }
    )
    ctx = RunContext(case=case)
    monkeypatch.setattr(
        reference,
        "source_revision",
        lambda *_args: pytest.fail("validation reference must not require a bundle"),
    )

    revision = reference._reference_source_revision(case, ctx)

    assert revision == "a" * 40


def test_model_plugin_reference_requires_exact_validation_revision() -> None:
    case = SimpleNamespace(metadata={"validation_sample_id": "fixed-profile"})

    with pytest.raises(ValueError, match="lowercase 40-character Git SHA"):
        reference._reference_source_revision(case, RunContext(case=case))


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
    emit_frames = True

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        if emit_frames:
            frames_dir = output_dir / "frames"
            frames_dir.mkdir()
            for index in range(2):
                (frames_dir / f"frame_{index:04d}.png").touch()
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

    output = reference.reference.run_stage(
        case,
        SimpleNamespace(name="end_to_end"),
        ctx,
    )

    command = captured["command"]
    assert command[command.index("--diffusers-evidence") + 1] == str(evidence.resolve())
    assert captured["environment"] == {"PINNED": "1"}
    assert output.data["num_frames"] == 2
    assert output.data["frames_dir"] == str(output_dir / "frames")
    assert output.data["frame_paths"] == [
        str(output_dir / "frames" / f"frame_{index:04d}.png") for index in range(2)
    ]

    for path in (output_dir / "frames").iterdir():
        path.unlink()
    (output_dir / "frames").rmdir()
    emit_frames = False
    output = reference.reference.run_stage(
        case,
        SimpleNamespace(name="end_to_end"),
        ctx,
    )
    assert "num_frames" not in output.data
    assert "frames_dir" not in output.data
    assert "frame_paths" not in output.data


def test_hf_report_frames_are_complete_clipped_png_evidence(tmp_path: Path) -> None:
    frames = np.linspace(
        -0.25,
        1.25,
        num=hf_reference.EXPECTED_NUM_FRAMES * 2 * 3 * 3,
        dtype=np.float32,
    ).reshape(hf_reference.EXPECTED_NUM_FRAMES, 2, 3, 3)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "stale.png").write_bytes(b"stale")

    report_frames = hf_reference._materialize_report_frames(
        frames,
        frames_dir,
        output_type="np",
    )
    paths = sorted(frames_dir.glob("frame_*.png"))

    assert report_frames is not None
    assert report_frames["count"] == hf_reference.EXPECTED_NUM_FRAMES
    assert report_frames["directory"] == "frames"
    assert report_frames["write_s"] >= 0.0
    assert report_frames["included_in_median_request_s"] is False
    assert len(paths) == hf_reference.EXPECTED_NUM_FRAMES
    assert paths[0].name == "frame_0000.png"
    assert paths[-1].name == "frame_0123.png"
    assert not (frames_dir / "stale.png").exists()
    first = np.asarray(Image.open(paths[0]))
    last = np.asarray(Image.open(paths[-1]))
    assert first.min() == 0
    assert last.max() == 255

    with pytest.raises(ValueError, match="123 frames instead of 124"):
        hf_reference._write_report_frames(frames[:-1], frames_dir)
    non_finite = frames.copy()
    non_finite[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite pixels"):
        hf_reference._write_report_frames(non_finite, frames_dir)


def test_hf_latent_output_does_not_claim_or_write_report_media(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    latents = np.zeros((1, 16, 31, 96, 168), dtype=np.float32)

    report_frames = hf_reference._materialize_report_frames(
        latents,
        frames_dir,
        output_type="latent",
    )

    assert report_frames is None
    assert not frames_dir.exists()


def test_hf_report_frames_register_and_satisfy_html_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(PROJECT_DIR / "scripts"))
    report = importlib.import_module("generate_e2e_report")
    case = load_model_manifest(MODEL_DIR / "manifests" / "minimax-h3-768p.json").testcases[0]
    sink = FileArtifactSink(tmp_path / "e2e", case)
    trt_frames = sink.base_dir / "trt_native" / "frames"
    ref_frames = sink.base_dir / "hf_reference" / "frames"
    for frames_dir, color in ((trt_frames, (8, 16, 24)), (ref_frames, (24, 16, 8))):
        frames_dir.mkdir(parents=True)
        Image.new("RGB", (4, 4), color).save(frames_dir / "frame_0000.png")

    orchestrator._auto_register_artifacts(
        sink,
        StageOutput(stage_name="end_to_end", data={"frames_dir": str(trt_frames)}),
        "trt",
    )
    orchestrator._auto_register_artifacts(
        sink,
        StageOutput(stage_name="end_to_end", data={"frames_dir": str(ref_frames)}),
        "ref",
    )
    result_path = sink.finalize(E2EResult(case_name=case.name))
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    result["_artifact_dir"] = str(sink.base_dir)

    assert result["artifacts"] == {
        "trt_frames": "trt_native/frames",
        "ref_frames": "hf_reference/frames",
    }
    assert report.validate_evidence([result], project_dir=PROJECT_DIR) == []


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
