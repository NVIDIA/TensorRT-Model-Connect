# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WAN2.2 pinned official-Wan Nightly reference contract tests."""

from __future__ import annotations

import subprocess
import struct
import sys
import tomllib
import zlib
from pathlib import Path
from types import ModuleType

import pytest

from tensorrt_model_connect.families import family_hf_warm_dependencies
from tensorrt_model_connect.hf_snapshot import hf_snapshot_allow_patterns
from tests.e2e.models.wan2_2_ti2v.e2e_plugins import reference as reference_plugins
from tests.e2e.models.wan2_2_ti2v.e2e_plugins.references import (
    official_wan as wan22_official,
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
            "video_num_frames": 5,
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


def _write_official_source(storage_root: Path) -> Path:
    source = storage_root / wan22_official.OFFICIAL_RELATIVE_PATH
    entrypoint = source / wan22_official.OFFICIAL_ENTRYPOINT
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# pinned official Wan entrypoint\n", encoding="utf-8")
    return source


def test_only_nightly_selects_the_external_official_wan_reference() -> None:
    full = load_manifest(_FULL_MANIFEST)
    l0 = load_manifest(_L0_MANIFEST)

    assert (full.reference_backend, full.oracle_level) == (
        "wan_official",
        "L1_external_reference",
    )
    assert (l0.reference_backend, l0.oracle_level) == (
        "invariant_only",
        "L4_invariants",
    )
    assert (full.inputs["video_width"], full.inputs["video_height"]) == (1280, 704)
    assert full.inputs["video_num_frames"] == 121
    assert full.inputs["num_inference_steps"] == 50
    assert (l0.inputs["video_width"], l0.inputs["video_height"]) == (672, 384)
    assert l0.inputs["video_num_frames"] == 5
    assert l0.inputs["num_inference_steps"] == 15


def test_both_reference_backends_are_registered() -> None:
    assert {plugin.backend_name for plugin in reference_plugins.reference} == {
        "wan_official",
        "invariant_only",
    }


def test_model_declares_the_exact_pinned_official_wan_source() -> None:
    owner = tomllib.loads((_MODEL_DIR / "MODEL.toml").read_text(encoding="utf-8"))
    assert owner["model_reference_cache"] == {
        "repository": wan22_official.OFFICIAL_REPOSITORY,
        "revision": wan22_official.OFFICIAL_REVISION,
        "relative_path": wan22_official.OFFICIAL_RELATIVE_PATH,
        "entrypoint": wan22_official.OFFICIAL_ENTRYPOINT,
    }


def test_converted_diffusers_reference_is_not_warmed() -> None:
    warm = dict(family_hf_warm_dependencies("wan2_2_ti2v"))
    assert "wan22-ti2v-5b-diffusers-reference" not in warm
    assert "Wan-AI/Wan2.2-TI2V-5B-Diffusers" not in warm.values()


def test_cached_reference_uses_the_raw_selective_offline_snapshot_contract(
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

    resolved = wan22_official._resolve_cached_model_ref()

    assert resolved == str(snapshot)
    assert calls == [
        (
            wan22_official.HF_REFERENCE_ID,
            {
                "revision": wan22_official.HF_REFERENCE_REVISION,
                "allow_patterns": hf_snapshot_allow_patterns(),
                "local_files_only": True,
            },
        )
    ]
    assert wan22_official._snapshot_revision(resolved) == "a" * 40


def test_reference_uses_official_wan_configuration_and_writes_hf_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case()
    ctx = _context(case, tmp_path)
    snapshot = tmp_path / "snapshots" / wan22_official.HF_REFERENCE_REVISION
    snapshot.mkdir(parents=True)
    storage_root = tmp_path / "reference-private"
    source = _write_official_source(storage_root)
    monkeypatch.setenv("TRTMC_STORAGE_ROOT", str(storage_root))
    monkeypatch.setattr(wan22_official, "_resolve_cached_model_ref", lambda: str(snapshot))
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        captured["command"] = command
        captured["kwargs"] = kwargs
        frames_dir = tmp_path / case.name / "hf_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for index in range(5):
            _write_png(frames_dir / f"frame_{index:04d}.png", 32, 16, index * 40)
        return subprocess.CompletedProcess(command, 0, stdout="Generated 5 frames\n", stderr="")

    monkeypatch.setattr(wan22_official.subprocess, "run", fake_run)
    output = wan22_official.Wan22OfficialWanReference().run_stage(
        case, StageSpec(name="end_to_end"), ctx
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[:2] == ["/reference/python", "-c"]
    script = _script(command)
    assert str(source) in script
    assert "from wan.textimage2video import WanTI2V" in script
    assert "from wan.configs.wan_ti2v_5B import ti2v_5B" in script
    assert "from wan.modules.attention import attention as wan_attention" in script
    assert "wan_model_module.flash_attention = wan_attention" in script
    assert 'sys.modules["wan"] = wan' in script
    assert 'sys.modules["wan.configs"] = wan_configs' in script
    assert 'sys.modules["easydict"] = easydict' in script
    assert 'sys.modules["imageio"] = imageio' in script
    assert "checkpoint_dir=model_ref" in script
    assert "t5_cpu=False" in script
    assert "init_on_cpu=True" in script
    assert "convert_model_dtype=False" in script
    assert "size=(32, 16)" in script
    assert "max_area=32 * 16" in script
    assert "frame_num=5" in script
    assert "sampling_steps=7" in script
    assert "guide_scale=5.0" in script
    assert "shift=5.0" in script
    assert "sample_solver=\"unipc\"" in script
    assert "seed=42" in script
    assert "offload_model=False" in script
    assert "((video.clamp(-1.0, 1.0) + 1.0) * 127.5)" in script
    assert str(tmp_path / case.name / "hf_frames") in script
    assert output.data["num_frames"] == 5
    assert output.data["frames_dir"] == str(tmp_path / case.name / "hf_frames")
    assert output.data["reference_implementation"] == "official_wan"
    assert output.metadata["model_id"] == wan22_official.HF_REFERENCE_ID
    assert output.metadata["model_revision"] == wan22_official.HF_REFERENCE_REVISION
    assert output.metadata["expected_model_revision"] == wan22_official.HF_REFERENCE_REVISION
    assert output.metadata["official_revision"] == wan22_official.OFFICIAL_REVISION
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["timeout"] == 60
    assert kwargs["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"


def test_reference_requires_the_pinned_official_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TRTMC_STORAGE_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="TRTMC_STORAGE_ROOT is required"):
        wan22_official._resolve_official_source()

    with pytest.raises(RuntimeError, match="Pinned official Wan reference is unavailable"):
        wan22_official._resolve_official_source(str(tmp_path))


def test_reference_fails_closed_on_subprocess_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case()
    ctx = _context(case, tmp_path)
    storage_root = tmp_path / "reference-private"
    _write_official_source(storage_root)
    monkeypatch.setenv("TRTMC_STORAGE_ROOT", str(storage_root))
    monkeypatch.setattr(wan22_official, "_resolve_cached_model_ref", lambda: "/model")
    monkeypatch.setattr(
        wan22_official.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, stdout="", stderr="official source failure"
        ),
    )

    with pytest.raises(RuntimeError, match="official source failure"):
        wan22_official.Wan22OfficialWanReference().run_stage(
            case, StageSpec(name="end_to_end"), ctx
        )
    assert (tmp_path / case.name / "official_wan_end_to_end_stderr.log").read_text(
        encoding="utf-8"
    ) == "official source failure"


def test_reference_rejects_partial_noncontiguous_or_wrong_sized_outputs(tmp_path: Path) -> None:
    frames_dir = tmp_path / "hf_frames"
    frames_dir.mkdir()
    _write_png(frames_dir / "frame_0000.png", 32, 16)

    with pytest.raises(RuntimeError, match="1 frames; expected 2"):
        wan22_official._validate_frames(
            frames_dir,
            expected_count=2,
            expected_width=32,
            expected_height=16,
        )

    _write_png(frames_dir / "frame_0002.png", 32, 16)
    with pytest.raises(RuntimeError, match="non-contiguous frame sequence"):
        wan22_official._validate_frames(
            frames_dir,
            expected_count=2,
            expected_width=32,
            expected_height=16,
        )

    (frames_dir / "frame_0002.png").unlink()
    with pytest.raises(RuntimeError, match=r"size \(32, 16\); expected \(16, 32\)"):
        wan22_official._validate_frames(
            frames_dir,
            expected_count=1,
            expected_width=16,
            expected_height=32,
        )
