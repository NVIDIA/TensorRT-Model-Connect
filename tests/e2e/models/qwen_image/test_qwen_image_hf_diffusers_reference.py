# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen Image model-owned HF diffusers reference tests."""

from __future__ import annotations

import subprocess
import sys
from types import ModuleType

import pytest

from tensorrt_model_connect.hf_snapshot import hf_snapshot_allow_patterns
from tests.e2e.models.qwen_image.e2e_plugins.references import (
    hf_diffusers as qwen_image_hf_diffusers,
)
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec


def _make_qwen_image_case(inputs: dict | None = None) -> E2ECase:
    return E2ECase(
        name="qwen-image-case",
        hf_id="Qwen/Qwen-Image-2512",
        family="qwen_image",
        runtime_strategy="diffusion_qwen_image",
        bundle="qwen-image-case.trtfb",
        inputs=inputs or {},
    )


def _make_ctx(case: E2ECase, tmp_path) -> RunContext:
    binary_path = tmp_path / "trtmc"
    binary_path.write_text("", encoding="utf-8")
    return RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        binary_path=str(binary_path),
        engine_dir=str(tmp_path),
    )


def _capture_subprocess(monkeypatch, tmp_path):
    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        frames_dir = tmp_path / "qwen-image-case" / "hf_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        (frames_dir / "frame_0000.png").write_bytes(b"")
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Generated 1 frames\n", stderr="")

    monkeypatch.setattr(qwen_image_hf_diffusers.subprocess, "run", _fake_run)
    return captured


def _extract_script(cmd: list[str]) -> str:
    assert "-c" in cmd, f"expected python -c invocation, got {cmd!r}"
    idx = cmd.index("-c")
    return cmd[idx + 1]


def test_qwen_image_reference_uses_qwen_image_pipeline(monkeypatch, tmp_path):
    case = _make_qwen_image_case(
        inputs={
            "prompt": "A red apple on a wooden table",
            "negative_prompt": " ",
            "num_inference_steps": 20,
            "cfg_scale": 4.0,
            "height": 1024,
            "width": 1024,
            "seed": 42,
        },
    )
    ctx = _make_ctx(case, tmp_path)
    captured = _capture_subprocess(monkeypatch, tmp_path)

    qwen_image_hf_diffusers.HfDiffusersReference().run_stage(
        case, StageSpec(name="end_to_end"), ctx)

    script = _extract_script(captured["cmd"])
    assert "QwenImagePipeline" in script
    assert 'family in ("qwen_image",)' in script
    assert "true_cfg_scale=qi_cfg_scale" in script
    assert "qi_cfg_scale = 4.0" in script
    assert "torch.bfloat16" in script
    assert "qi_height = 1024" in script
    assert "qi_width = 1024" in script
    assert "num_steps = 20" in script
    assert "seed = 42" in script
    assert "qi_negative_prompt = ' '" in script
    assert "prompt = 'A red apple on a wooden table'" in script


def test_qwen_image_reference_uses_cpu_offload_for_48gb_gpus(monkeypatch, tmp_path):
    case = _make_qwen_image_case(inputs={"prompt": "scene"})
    ctx = _make_ctx(case, tmp_path)
    captured = _capture_subprocess(monkeypatch, tmp_path)

    qwen_image_hf_diffusers.HfDiffusersReference().run_stage(
        case, StageSpec(name="end_to_end"), ctx)

    script = _extract_script(captured["cmd"])
    assert "pipe.enable_model_cpu_offload()" in script
    assert 'pipe.to("cuda")' not in script


def test_qwen_image_reference_falls_back_to_guidance_scale(monkeypatch, tmp_path):
    case = _make_qwen_image_case(
        inputs={
            "prompt": "scene",
            "guidance_scale": 3.5,
            "num_inference_steps": 8,
        },
    )
    ctx = _make_ctx(case, tmp_path)
    captured = _capture_subprocess(monkeypatch, tmp_path)

    qwen_image_hf_diffusers.HfDiffusersReference().run_stage(
        case, StageSpec(name="end_to_end"), ctx)

    script = _extract_script(captured["cmd"])
    assert "true_cfg_scale=qi_cfg_scale" in script
    assert "qi_cfg_scale = 3.5" in script


def test_qwen_image_reference_image_height_width_alias(monkeypatch, tmp_path):
    case = _make_qwen_image_case(
        inputs={
            "prompt": "scene",
            "image_height": 768,
            "image_width": 512,
            "num_inference_steps": 8,
        },
    )
    ctx = _make_ctx(case, tmp_path)
    captured = _capture_subprocess(monkeypatch, tmp_path)

    qwen_image_hf_diffusers.HfDiffusersReference().run_stage(
        case, StageSpec(name="end_to_end"), ctx)

    script = _extract_script(captured["cmd"])
    assert "qi_height = 768" in script
    assert "qi_width = 512" in script


def test_qwen_image_reference_writes_frames_dir(monkeypatch, tmp_path):
    case = _make_qwen_image_case(
        inputs={
            "prompt": "scene",
            "num_inference_steps": 4,
            "seed": 7,
        },
    )
    ctx = _make_ctx(case, tmp_path)
    captured = _capture_subprocess(monkeypatch, tmp_path)

    qwen_image_hf_diffusers.HfDiffusersReference().run_stage(
        case, StageSpec(name="end_to_end"), ctx)

    script = _extract_script(captured["cmd"])
    assert 'frame_{i:04d}.png' in script or 'frame_{{i:04d}}.png' in script
    assert "hf_frames" in script


def test_qwen_image_reference_forward_compat_edit_variants(monkeypatch, tmp_path):
    case_no_image = _make_qwen_image_case(
        inputs={"prompt": "scene", "num_inference_steps": 4},
    )
    ctx = _make_ctx(case_no_image, tmp_path)
    captured = _capture_subprocess(monkeypatch, tmp_path)
    qwen_image_hf_diffusers.HfDiffusersReference().run_stage(
        case_no_image, StageSpec(name="end_to_end"), ctx)
    script = _extract_script(captured["cmd"])
    assert "QwenImageEditPlusPipeline" in script
    assert "QwenImageEditPipeline" in script
    assert "QwenImagePipeline" in script
    assert 'if "Edit" in cls_name and not bool(qi_image_path)' in script
    assert "qi_image_path = ''" in script

    case_with_image = _make_qwen_image_case(
        inputs={"prompt": "scene", "num_inference_steps": 4,
                "image": "/tmp/x.png"},
    )
    ctx2 = _make_ctx(case_with_image, tmp_path)
    captured2 = _capture_subprocess(monkeypatch, tmp_path)
    qwen_image_hf_diffusers.HfDiffusersReference().run_stage(
        case_with_image, StageSpec(name="end_to_end"), ctx2)
    script2 = _extract_script(captured2["cmd"])
    assert "qi_image_path = '/tmp/x.png'" in script2
    assert "qi_input_image = Image.open(qi_image_path).convert(\"RGB\")" in script2
    assert 'qi_call_kwargs["image"] = qi_input_image' in script2


def test_cached_model_ref_uses_the_selective_snapshot_contract(
    tmp_path, monkeypatch
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    calls: list[tuple[str, dict[str, object]]] = []
    expected_kwargs = {
        "allow_patterns": hf_snapshot_allow_patterns(),
        "local_files_only": True,
    }

    def fake_snapshot_download(repo_id: str, **kwargs: object) -> str:
        calls.append((repo_id, kwargs))
        if kwargs != expected_kwargs:
            raise RuntimeError("selective snapshot rejected without its allowlist")
        return str(snapshot)

    huggingface_hub = ModuleType("huggingface_hub")
    huggingface_hub.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)

    resolved = qwen_image_hf_diffusers._resolve_cached_model_ref("Qwen/Qwen-Image")

    assert resolved == str(snapshot)
    assert calls == [
        (
            "Qwen/Qwen-Image",
            expected_kwargs,
        )
    ]


def test_qwen_image_reference_fails_closed_on_subprocess_error(monkeypatch, tmp_path):
    case = _make_qwen_image_case(inputs={"prompt": "scene"})
    ctx = _make_ctx(case, tmp_path)

    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            1,
            stdout="",
            stderr="offline cache miss",
        )

    monkeypatch.setattr(qwen_image_hf_diffusers.subprocess, "run", _fake_run)

    with pytest.raises(
        RuntimeError,
        match=r"Qwen Image HF reference failed \(rc=1\): offline cache miss",
    ):
        qwen_image_hf_diffusers.HfDiffusersReference().run_stage(
            case, StageSpec(name="end_to_end"), ctx
        )


def test_qwen_image_reference_fails_closed_without_frames(monkeypatch, tmp_path):
    case = _make_qwen_image_case(inputs={"prompt": "scene"})
    ctx = _make_ctx(case, tmp_path)

    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="Generated 0 frames\n",
            stderr="",
        )

    monkeypatch.setattr(qwen_image_hf_diffusers.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="Qwen Image HF reference produced no frames"):
        qwen_image_hf_diffusers.HfDiffusersReference().run_stage(
            case, StageSpec(name="end_to_end"), ctx
        )
