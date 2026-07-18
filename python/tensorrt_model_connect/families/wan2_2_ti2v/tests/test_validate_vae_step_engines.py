# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualification gate contracts for recurrent Wan2.2 VAE validation."""

from __future__ import annotations

import gc
import math
import weakref
from pathlib import Path

import pytest
import torch

from tensorrt_model_connect.families.wan2_2_ti2v.reference.validate_vae_step_engines import (
    _CACHE_ALIGNMENT,
    _DEFAULT_MAX_VIDEO_RMSE,
    _DEFAULT_MAX_RELATIVE_L2_ERROR,
    _EngineRunner,
    _MappedCacheSlice,
    _MappedHostCacheBank,
    _cache_address,
    _cuda_device_can_map_host_memory,
    _file_identity,
    _metrics,
    _qualification,
)
from tensorrt_model_connect.families.wan2_2_ti2v.vae_step_builder import (
    SMALL_VAE_STEP_PROFILE,
    vae_step_cache_bytes,
)


def _metric_record(cosine: float = 1.0, rmse: float = 0.0) -> dict:
    return {"cosine_similarity": cosine, "relative_l2_error": 0.0, "rmse": rmse}


def _report(cosine: float = 1.0) -> dict:
    return {
        "video_metrics": _metric_record(cosine),
        "initializer_video_metrics": _metric_record(cosine),
        "recurrent_video_metrics": _metric_record(cosine),
        "per_frame_metrics": [{"frame": 0, **_metric_record(cosine)}],
        "final_cache_metrics": [{"index": 0, **_metric_record(cosine)}],
    }


def test_qualification_passes_expected_outputs() -> None:
    qualification = _qualification(_report(), 0.998, _DEFAULT_MAX_VIDEO_RMSE)

    assert qualification["comparisons_checked"] == 5
    assert qualification["worst_cosine_similarity"] == pytest.approx(1.0)
    assert qualification["passed"] is True


def test_qualification_gates_worst_output() -> None:
    report = _report()
    report["final_cache_metrics"][0]["cosine_similarity"] = 0.9979

    qualification = _qualification(report, 0.998, _DEFAULT_MAX_VIDEO_RMSE)

    assert qualification["worst_comparison"] == "final_cache_0"
    assert qualification["passed"] is False


@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
def test_qualification_rejects_non_finite_cosines(non_finite: float) -> None:
    qualification = _qualification(_report(non_finite), 0.998, _DEFAULT_MAX_VIDEO_RMSE)

    assert len(qualification["non_finite_comparisons"]) == 5
    assert qualification["worst_cosine_similarity"] is None
    assert qualification["passed"] is False


def test_qualification_rejects_scaled_visible_output_despite_perfect_cosine() -> None:
    reference = torch.tensor([-0.5, -0.25, 0.25, 0.5], dtype=torch.float32)
    scaled = reference * 1.5
    scaled_metrics = _metrics(reference, scaled)
    assert scaled_metrics["cosine_similarity"] == pytest.approx(1.0)

    report = _report()
    report["video_metrics"] = scaled_metrics
    qualification = _qualification(report, 0.998, _DEFAULT_MAX_VIDEO_RMSE)

    assert qualification["worst_video_rmse_comparison"] == "video"
    assert qualification["worst_video_rmse"] > _DEFAULT_MAX_VIDEO_RMSE
    assert qualification["passed"] is False


def test_qualification_rejects_scaled_final_cache_despite_perfect_cosine() -> None:
    reference = torch.tensor([-0.5, -0.25, 0.25, 0.5], dtype=torch.float32)
    scaled_metrics = _metrics(reference, reference * 1.5)
    assert scaled_metrics["cosine_similarity"] == pytest.approx(1.0)
    assert scaled_metrics["relative_l2_error"] == pytest.approx(0.5)

    report = _report()
    report["final_cache_metrics"][0] = {"index": 0, **scaled_metrics}
    qualification = _qualification(report, 0.998, _DEFAULT_MAX_VIDEO_RMSE)

    assert qualification["worst_relative_l2_comparison"] == "final_cache_0"
    assert qualification["worst_relative_l2_error"] > _DEFAULT_MAX_RELATIVE_L2_ERROR
    assert qualification["passed"] is False


def test_metrics_zero_reference_semantics_fail_closed() -> None:
    zeros = torch.zeros(4, dtype=torch.float32)
    exact_zero_metrics = _metrics(zeros, zeros)
    mismatch_metrics = _metrics(zeros, torch.ones_like(zeros))

    assert exact_zero_metrics["cosine_similarity"] == pytest.approx(1.0)
    assert exact_zero_metrics["relative_l2_error"] == 0.0
    assert mismatch_metrics["cosine_similarity"] == pytest.approx(0.0)
    assert math.isinf(mismatch_metrics["relative_l2_error"])


@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
def test_qualification_rejects_non_finite_relative_l2(non_finite: float) -> None:
    report = _report()
    report["final_cache_metrics"][0]["relative_l2_error"] = non_finite

    qualification = _qualification(report, 0.998, _DEFAULT_MAX_VIDEO_RMSE)

    assert qualification["non_finite_relative_l2_comparisons"] == ["final_cache_0"]
    assert qualification["worst_relative_l2_error"] is None
    assert qualification["passed"] is False


@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
def test_qualification_rejects_non_finite_video_rmse(non_finite: float) -> None:
    report = _report()
    report["per_frame_metrics"][0]["rmse"] = non_finite

    qualification = _qualification(report, 0.998, _DEFAULT_MAX_VIDEO_RMSE)

    assert qualification["non_finite_video_rmse_comparisons"] == ["frame_0"]
    assert qualification["worst_video_rmse"] is None
    assert qualification["passed"] is False


def test_engine_runner_requires_exact_input_cache_count() -> None:
    runner = object.__new__(_EngineRunner)

    with pytest.raises(ValueError, match="Expected 32 cache inputs, got 0"):
        runner.run(latent_frame=torch.empty(0), caches=[])


def test_file_identity_records_resolved_plugin_provenance(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin.so"
    plugin.write_bytes(b"plugin-bytes")

    identity = _file_identity(plugin)

    assert identity["path"] == str(plugin.resolve())
    assert identity["size_bytes"] == len(b"plugin-bytes")
    assert len(identity["sha256"]) == 64


def test_mapped_cache_slice_rejects_closed_or_released_owner_without_cycle() -> None:
    class Owner:
        closed = False

    owner = Owner()
    cache = _MappedCacheSlice(shape=(1,), host_address=1, device_address=2, owner=owner)
    owner.slices = [cache]
    owner_reference = weakref.ref(owner)

    owner.closed = True
    with pytest.raises(RuntimeError, match="no longer valid"):
        _ = cache.device_address
    with pytest.raises(RuntimeError, match="no longer valid"):
        cache.cpu_tensor()

    del owner
    gc.collect()
    assert owner_reference() is None
    with pytest.raises(RuntimeError, match="no longer valid"):
        _cache_address(cache)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mapped_host_cache_bank_is_aligned_and_zeroable() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    if not torch.cuda.get_device_properties(device).is_integrated:
        pytest.skip("Mapped-host VAE caches are used only on integrated CUDA devices")
    if not _cuda_device_can_map_host_memory(device):
        pytest.skip("Integrated CUDA device cannot map host memory")

    bank = _MappedHostCacheBank(SMALL_VAE_STEP_PROFILE)
    first_slice = bank.slices[0]
    try:
        assert bank.total_bytes >= vae_step_cache_bytes(SMALL_VAE_STEP_PROFILE)
        assert all(cache.device_address % _CACHE_ALIGNMENT == 0 for cache in bank.slices)
        stream = torch.cuda.current_stream()
        bank.zero_async(stream.cuda_stream)
        stream.synchronize()
        assert all(torch.count_nonzero(cache.cpu_tensor()) == 0 for cache in bank.slices)
    finally:
        bank.close()

    with pytest.raises(RuntimeError, match="no longer valid"):
        _ = first_slice.device_address
    with pytest.raises(RuntimeError, match="no longer valid"):
        first_slice.cpu_tensor()
    with pytest.raises(RuntimeError, match="no longer valid"):
        _cache_address(first_slice)
