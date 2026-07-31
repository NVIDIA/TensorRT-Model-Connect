# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Z-Image HF reference precision and output-health contract tests."""

from __future__ import annotations

import subprocess

from tests.e2e.models.z_image.e2e_plugins.references import hf_diffusers
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec


def test_hf_reference_uses_stable_precision_and_rejects_invalid_output(
    tmp_path, monkeypatch
) -> None:
    case = E2ECase(
        name="z-image-turbo",
        hf_id="Tongyi-MAI/Z-Image-Turbo",
        family="z_image",
        runtime_strategy="diffusion_zimage",
        inputs={
            "image_height": 512,
            "image_width": 512,
            "seed": 42,
        },
        metadata={},
    )
    context = RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        reference_python="/opt/venv/bin/python",
    )
    captured: dict[str, object] = {}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(
        hf_diffusers, "_resolve_cached_model_ref", lambda _id: "/model"
    )
    monkeypatch.setattr(hf_diffusers.subprocess, "run", fake_run)

    hf_diffusers.HfDiffusersReference().run_stage(
        case, StageSpec(name="end_to_end"), context
    )

    script = captured["cmd"][2]
    assert "torch_dtype=torch.bfloat16" in script
    assert "pipeline_torch_dtypes" not in script
    assert "mixed_fp32_modules" not in script
    assert "torch.isfinite(latents).all()" in script
    assert "callback_on_step_end=_require_finite_latents" in script
    assert "np.isfinite(frame_array).all()" in script
    assert "float(frame_array.std()) == 0.0" in script
    assert 'device="cuda", dtype=torch.bfloat16' in script
