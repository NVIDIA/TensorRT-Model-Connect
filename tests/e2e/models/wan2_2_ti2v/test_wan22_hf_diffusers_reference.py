# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WAN2.2 nightly Hugging Face Diffusers reference contract tests."""

from __future__ import annotations

import subprocess
import struct
import sys
import zlib
from pathlib import Path
from types import ModuleType

import pytest

from tensorrt_model_connect.families import family_hf_warm_dependencies
from tensorrt_model_connect.hf_snapshot import hf_snapshot_allow_patterns
from tests.e2e.models.wan2_2_ti2v.e2e_plugins import reference as reference_plugins
from tests.e2e.models.wan2_2_ti2v.e2e_plugins.references import (
    hf_diffusers as wan22_hf_diffusers,
)
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec
from tests.e2e_harness.manifest_loader import load_manifest

_MODEL_DIR = Path(__file__).resolve().parent
_FULL_MANIFEST = _MODEL_DIR / "manifests/wan22-ti2v-5b.json"
_L0_MANIFEST = _MODEL_DIR / "manifests/wan22-ti2v-5b-l0.json"


def _write_png(path: Path, width: int, height: int, value: int = 0) -> None:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes([value, value, value]) * width
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(row * height))
        + chunk(b"IEND", b"")
    )


def _case() -> E2ECase:
    return E2ECase(
        name="wan22-unit",
        hf_id="Wan-AI/Wan2.2-TI2V-5B",
        family="wan2_2_ti2v",
        runtime_strategy="diffusion_wan2_2_ti2v",
        inputs={
            "prompt": "A cat wearing boxing gloves",
            "video_num_frames": 3,
            "video_height": 16,
            "video_width": 32,
            "num_inference_steps": 7,
            "guidance_scale": 5.0,
            "flow_shift": 5.0,
            "text_max_length": 512,
            "seed": 42,
        },
        metadata={"runtime_timeout_s": 60},
    )


def _context(case: E2ECase, tmp_path: Path) -> RunContext:
    return RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        reference_python="/reference/python",
    )


def _script(command: list[str]) -> str:
    return command[command.index("-c") + 1]


def test_only_nightly_selects_the_external_diffusers_reference() -> None:
    full = load_manifest(_FULL_MANIFEST)
    l0 = load_manifest(_L0_MANIFEST)

    assert (full.reference_backend, full.oracle_level) == (
        "hf_diffusers",
        "L1_external_reference",
    )
    assert (l0.reference_backend, l0.oracle_level) == (
        "invariant_only",
        "L4_invariants",
    )
    assert full.inputs["video_num_frames"] == 121
    assert full.inputs["num_inference_steps"] == 50
    assert l0.inputs["video_num_frames"] == 5
    assert l0.inputs["num_inference_steps"] == 15


def test_both_reference_backends_are_registered() -> None:
    assert {plugin.backend_name for plugin in reference_plugins.reference} == {
        "hf_diffusers",
        "invariant_only",
    }


def test_diffusers_reference_is_present_in_the_offline_cache_contract() -> None:
    assert (
        dict(family_hf_warm_dependencies("wan2_2_ti2v"))["wan22-ti2v-5b-diffusers-reference"]
        == wan22_hf_diffusers.HF_REFERENCE_ID
    )


def test_cached_reference_uses_the_selective_offline_snapshot_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_snapshot_download(repo_id: str, **kwargs: object) -> str:
        calls.append((repo_id, kwargs))
        return str(snapshot)

    huggingface_hub = ModuleType("huggingface_hub")
    huggingface_hub.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)

    resolved = wan22_hf_diffusers._resolve_cached_model_ref()

    assert resolved == str(snapshot)
    assert calls == [
        (
            wan22_hf_diffusers.HF_REFERENCE_ID,
            {
                "allow_patterns": hf_snapshot_allow_patterns(),
                "local_files_only": True,
            },
        )
    ]
    assert wan22_hf_diffusers._snapshot_revision(resolved) == "a" * 40


def test_reference_uses_official_diffusers_configuration_and_writes_hf_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case()
    ctx = _context(case, tmp_path)
    snapshot = tmp_path / "snapshots" / ("b" * 40)
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(wan22_hf_diffusers, "_resolve_cached_model_ref", lambda: str(snapshot))
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        captured["command"] = command
        captured["kwargs"] = kwargs
        frames_dir = tmp_path / case.name / "hf_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for index in range(3):
            _write_png(frames_dir / f"frame_{index:04d}.png", 32, 16, index * 40)
        return subprocess.CompletedProcess(command, 0, stdout="Generated 3 frames\n", stderr="")

    monkeypatch.setattr(wan22_hf_diffusers.subprocess, "run", fake_run)
    output = wan22_hf_diffusers.Wan22HfDiffusersReference().run_stage(
        case, StageSpec(name="end_to_end"), ctx
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[:2] == ["/reference/python", "-c"]
    script = _script(command)
    assert "AutoencoderKLWan.from_pretrained" in script
    assert 'subfolder="vae"' in script
    assert "torch_dtype=torch.float32" in script
    assert "WanPipeline.from_pretrained" in script
    assert "torch_dtype=torch.bfloat16" in script
    assert "UMT5 shared and encoder input embedding shapes do not match" in script
    assert "pipe.text_encoder.set_input_embeddings(pipe.text_encoder.shared)" in script
    assert "pipe.text_encoder.encoder.embed_tokens is not pipe.text_encoder.shared" in script
    assert "UMT5 encoder input embedding is not tied to shared.weight" in script
    assert script.index("shared_embedding_shape") < script.index("set_input_embeddings")
    assert script.index("set_input_embeddings") < script.index('pipe.to("cuda")')
    assert 'pipe.to("cuda")' in script
    assert "height=16" in script
    assert "width=32" in script
    assert "num_frames=3" in script
    assert "num_inference_steps=7" in script
    assert "guidance_scale=5.0" in script
    assert "max_sequence_length=512" in script
    assert 'torch.Generator(device="cuda").manual_seed(42)' in script
    assert "local_files_only=True" in script
    assert str(tmp_path / case.name / "hf_frames") in script
    assert output.data["num_frames"] == 3
    assert output.data["frames_dir"] == str(tmp_path / case.name / "hf_frames")
    assert output.metadata["model_id"] == wan22_hf_diffusers.HF_REFERENCE_ID
    assert output.metadata["model_revision"] == "b" * 40
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["timeout"] == 60
    assert kwargs["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_reference_fails_closed_on_subprocess_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case()
    ctx = _context(case, tmp_path)
    monkeypatch.setattr(wan22_hf_diffusers, "_resolve_cached_model_ref", lambda: "/model")
    monkeypatch.setattr(
        wan22_hf_diffusers.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, stdout="", stderr="offline cache miss"
        ),
    )

    with pytest.raises(RuntimeError, match="offline cache miss"):
        wan22_hf_diffusers.Wan22HfDiffusersReference().run_stage(
            case, StageSpec(name="end_to_end"), ctx
        )
    assert (tmp_path / case.name / "hf_diffusers_end_to_end_stderr.log").read_text(
        encoding="utf-8"
    ) == "offline cache miss"


def test_reference_rejects_partial_or_wrong_sized_outputs(tmp_path: Path) -> None:
    frames_dir = tmp_path / "hf_frames"
    frames_dir.mkdir()
    _write_png(frames_dir / "frame_0000.png", 32, 16)

    with pytest.raises(RuntimeError, match="1 frames; expected 2"):
        wan22_hf_diffusers._validate_frames(
            frames_dir,
            expected_count=2,
            expected_width=32,
            expected_height=16,
        )

    with pytest.raises(RuntimeError, match=r"size \(32, 16\); expected \(16, 32\)"):
        wan22_hf_diffusers._validate_frames(
            frames_dir,
            expected_count=1,
            expected_width=16,
            expected_height=32,
        )
