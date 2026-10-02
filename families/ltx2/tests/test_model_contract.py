# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-request contract of the LTX-2.5 family (no TensorRT, torch or checkpoint needed)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from families.ltx2 import model


def _request(**overrides) -> SimpleNamespace:
    fields = dict(task="text_to_audio_video", dynamic_kv_cache=False, tensor_parallel_size=1,
                  max_batch_size=1, quantization=None, fp32_layers=(), precision="bf16", context_parallel_size=1,
                  backend="trt", model_dir="/nonexistent/ltx2")
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("version", ["1.7.1", "1.7.1.107", "1.8.0", "2.0"])
def test_rtx_context_parallel_gate_accepts_multi_device_releases(version: str) -> None:
    model.require_rtx_context_parallel(version, 2)


@pytest.mark.parametrize("version", [None, "1.6.0.0", "1.6.1.4", "1.7.0.32"])
def test_rtx_context_parallel_gate_rejects_older_releases(version: str | None) -> None:
    with pytest.raises(RuntimeError) as error:
        model.require_rtx_context_parallel(version, 2)
    message = str(error.value)
    assert "requires TensorRT-RTX >= 1.7.1" in message
    assert f"(found {version or 'not installed'})" in message
    assert "1.6.x ships without multi-device support" in message


@pytest.mark.parametrize("version", [None, "1.6.0.0"])
def test_rtx_gate_does_not_apply_to_a_single_device(version: str | None) -> None:
    model.require_rtx_context_parallel(version, 1)


def test_rtx_backend_build_checks_the_installed_version(monkeypatch) -> None:
    monkeypatch.setattr(model, "installed_rtx_version", lambda: "1.6.1.4")
    with pytest.raises(RuntimeError, match="TensorRT-RTX >= 1.7.1"):
        model.build(_request(backend="trt_rtx", context_parallel_size=2), writer=None)


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"task": "text_to_video"}, ValueError),
        ({"precision": "fp16"}, ValueError),
        ({"tensor_parallel_size": 2}, NotImplementedError),
        ({"max_batch_size": 2}, NotImplementedError),
        ({"quantization": "fp8"}, NotImplementedError),
        ({"fp32_layers": (3,)}, NotImplementedError),
        ({"dynamic_kv_cache": True}, NotImplementedError),
        ({"context_parallel_size": 4}, ValueError),
    ],
)
def test_build_rejects_unsupported_requests(overrides: dict, error: type) -> None:
    with pytest.raises(error):
        model.build(_request(**overrides), writer=None)


def test_stage_two_schedule_is_the_distilled_refinement_tail() -> None:
    # diffusers STAGE_2_DISTILLED_SIGMA_VALUES: the last three distilled sigmas.
    assert model.STAGE_2_DISTILLED_SIGMAS == (0.909375, 0.725, 0.421875)
    assert model.STAGE_2_DISTILLED_SIGMAS == model.DISTILLED_SIGMAS[-3:]


def test_two_stage_grid_needs_64_aligned_sizes_and_the_upsampler(tmp_path) -> None:
    shape = SimpleNamespace(latent_height=22, latent_width=40)
    with pytest.raises(ValueError, match="divisible by 64"):
        model._stage1_shape(tmp_path, shape, 672, 1280, 32)
    with pytest.raises(FileNotFoundError, match="latent_upsampler"):
        model._stage1_shape(tmp_path, shape, 704, 1280, 32)


def test_distilled_schedule_is_the_eight_step_checkpoint_schedule() -> None:
    assert len(model.DISTILLED_SIGMAS) == 8
    assert model.DISTILLED_SIGMAS[0] == 1.0
    assert list(model.DISTILLED_SIGMAS) == sorted(model.DISTILLED_SIGMAS, reverse=True)
