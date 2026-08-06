# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned E2E entrypoint and static contract checks for MiniMax-H3."""

from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tests.e2e.models.minimax_h3 import e2e_plugins as minimax_h3_e2e_plugins
from tests.e2e.models.minimax_h3.e2e_plugins.comparator import comparator
from tests.e2e.models.minimax_h3.e2e_plugins.reference import (
    _model_snapshot,
    _reference_allow_patterns,
)
from tensorrt_model_connect.families import find_diffusion_plugin, load_plugin_by_id
from tensorrt_model_connect.families.minimax_h3.provenance import file_record
from tests.e2e_harness.contracts import RunContext, StageOutput, StageSpec, ThresholdProfile
from tests.e2e_harness.manifest_loader import load_model_manifest
from tests.e2e_harness.registry import (
    activate_model_plugins,
    get_comparator,
    get_reference,
    get_runner,
)


_MODEL_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _MODEL_DIR.parents[3]
_MANIFEST_PATH = _MODEL_DIR / "manifests" / "minimax-h3-768p.json"
_RUNNER_PATH = _MODEL_DIR / "runner.py"
_SPEC = importlib.util.spec_from_file_location(
    f"{_MODEL_DIR.name}_e2e_runner",
    _RUNNER_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_runner)


def pytest_generate_tests(metafunc):
    if "case_name" in metafunc.fixturenames:
        metafunc.parametrize("case_name", _runner.model_case_names(metafunc.config))


def test_minimax_h3_manifest_is_truthful_single_device_contract() -> None:
    model = load_model_manifest(_MANIFEST_PATH)
    assert model.name == "minimax-h3-768p"
    assert model.family == "minimax_h3"
    assert len(model.testcases) == 1

    case = model.testcases[0]
    assert case.runtime_strategy == "diffusion_minimax_h3"
    assert case.task_strategy == "diffusion_media_generation"
    assert "ci_tier" not in case.metadata
    assert case.metadata["build_args"]["parallel"] == {
        "mode": "single",
        "cp_size": 1,
    }
    assert any(
        requirement.kind == "gpu_count_min" and requirement.args.get("count") == 1
        for requirement in case.preflight
    )
    assert case.inputs["video_num_frames"] == 124
    assert case.inputs["video_height"] == 768
    assert case.inputs["video_width"] == 1344
    assert case.inputs["num_inference_steps"] == 50
    assert case.threshold_overrides["low_frequency_block_size"] == 16
    assert case.threshold_overrides["minimum_frame_low_frequency_correlation"] == 0.8
    assert case.threshold_overrides["minimum_mean_low_frequency_correlation"] == 0.9
    assert "minimum_psnr_db" not in case.threshold_overrides
    assert "maximum_mean_absolute_error" not in case.threshold_overrides


def test_minimax_h3_plugins_cover_native_reference_and_comparison() -> None:
    activate_model_plugins(_MODEL_DIR)
    assert get_runner("diffusion_media_generation") is not None
    assert get_reference("hf_diffusers") is not None
    assert get_comparator("diffusion_media_generation") is not None


def test_family_registry_loads_native_plugin_for_public_pipelines() -> None:
    plugin = load_plugin_by_id("minimax_h3")
    assert plugin is not None
    assert plugin.name == "minimax_h3"
    for pipeline_class in ("MiniMaxH3ModularPipeline", "MiniMaxH3Pipeline"):
        assert find_diffusion_plugin(pipeline_class) is plugin


def test_minimax_h3_reference_resolves_complete_family_snapshot_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    case = load_model_manifest(_MANIFEST_PATH).testcases[0]
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_snapshot_download(repo_id: str, **kwargs: object) -> str:
        calls.append((repo_id, kwargs))
        return str(snapshot)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    assert _model_snapshot(case) == snapshot.resolve()
    assert calls == [
        (
            case.hf_id,
            {
                "revision": case.hf_revision,
                "allow_patterns": _reference_allow_patterns(),
                "local_files_only": True,
            },
        )
    ]
    patterns = set(_reference_allow_patterns())
    assert len(patterns) == 11
    assert not patterns & {"FL2VA/**", "Ref2VA/**", "assets/**"}


@pytest.mark.parametrize("nested", [True, False])
def test_minimax_h3_model_plugin_dir_accepts_proof_and_direct_layouts(
    tmp_path: Path,
    nested: bool,
) -> None:
    root = tmp_path / "model-plugins"
    plugin_dir = root / "minimax_h3" if nested else root
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "libtrtmc_model_minimax_h3.so").write_bytes(b"model plugin")
    case = load_model_manifest(_MANIFEST_PATH).testcases[0]

    resolved = minimax_h3_e2e_plugins.model_plugin_dir(
        RunContext(case=case, model_plugin_dir=str(root))
    )

    assert resolved == plugin_dir.resolve()


def test_minimax_h3_model_plugin_dir_prefers_isolated_proof_layout(tmp_path: Path) -> None:
    root = tmp_path / "model-plugins"
    nested = root / "minimax_h3"
    nested.mkdir(parents=True)
    for plugin_dir in (root, nested):
        (plugin_dir / "libtrtmc_model_minimax_h3.so").write_bytes(b"model plugin")
    case = load_model_manifest(_MANIFEST_PATH).testcases[0]

    resolved = minimax_h3_e2e_plugins.model_plugin_dir(
        RunContext(case=case, model_plugin_dir=str(root))
    )

    assert resolved == nested.resolve()


def test_minimax_h3_model_plugin_dir_preserves_local_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "source"
    plugin_dir = project_dir / "build" / "models" / "minimax_h3"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "libtrtmc_model_minimax_h3.so").write_bytes(b"model plugin")
    monkeypatch.setattr(minimax_h3_e2e_plugins, "PROJECT_DIR", project_dir)
    case = load_model_manifest(_MANIFEST_PATH).testcases[0]

    resolved = minimax_h3_e2e_plugins.model_plugin_dir(RunContext(case=case))

    assert resolved == plugin_dir.resolve()


def test_minimax_h3_model_plugin_dir_fails_closed_without_dso(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(minimax_h3_e2e_plugins, "PROJECT_DIR", tmp_path / "source")
    case = load_model_manifest(_MANIFEST_PATH).testcases[0]

    with pytest.raises(FileNotFoundError, match="libtrtmc_model_minimax_h3.so"):
        minimax_h3_e2e_plugins.model_plugin_dir(
            RunContext(case=case, model_plugin_dir=str(tmp_path / "model-plugins"))
        )


def _visual_thresholds(
    frames: int,
    height: int,
    width: int,
    *,
    block_size: int = 16,
) -> dict[str, float]:
    return {
        "exact_num_frames": frames,
        "exact_video_height": height,
        "exact_video_width": width,
        "low_frequency_block_size": block_size,
        "minimum_frame_low_frequency_correlation": 0.8,
        "minimum_mean_low_frequency_correlation": 0.9,
        "minimum_brightness_profile_correlation": 0.95,
        "maximum_frame_brightness_absolute_error": 0.08,
        "minimum_temporal_activity_correlation": 0.9,
        "maximum_temporal_activity_absolute_error": 0.05,
        "minimum_temporal_activity_ratio": 0.5,
        "maximum_temporal_activity_ratio": 1.5,
        "minimum_frame_std_ratio": 0.7,
        "maximum_frame_std_ratio": 1.4,
    }


def _synthetic_video() -> np.ndarray:
    frames, height, width = 12, 64, 64
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x /= width - 1
    y /= height - 1
    video = []
    for index in range(frames):
        phase = index / (frames - 1)
        center_x = 0.12 + 0.74 * phase**1.7
        center_y = 0.5 + 0.18 * np.sin(phase * np.pi * 2.5)
        radius = 0.075 + 0.035 * np.sin(phase * np.pi) ** 2
        subject = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2.0 * radius**2))
        pulse = 0.035 * np.sin(phase**1.4 * np.pi * 3.0)
        base = 0.24 + 0.16 * x + 0.09 * y + 0.34 * subject + pulse
        video.append(
            np.stack(
                (
                    base,
                    base * 0.82 + 0.055 * y,
                    base * 0.64 + 0.095 * x,
                ),
                axis=-1,
            )
        )
    return np.asarray(video, dtype=np.float32)


def _compare_arrays(
    tmp_path: Path,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    thresholds: dict[str, float] | None = None,
):
    reference_path = tmp_path / "reference.npy"
    candidate_path = tmp_path / "candidate.npy"
    np.save(reference_path, reference)
    np.save(candidate_path, candidate)
    revision = "1" * 40
    inventory = "a" * 64
    return comparator.compare(
        StageOutput(
            stage_name="end_to_end",
            data={
                "returncode": 0,
                "frames_path": str(candidate_path),
                "source_revision": revision,
                "receipt": {
                    "status": "passed",
                    "backend": "tensorrt_native_single_device",
                    "world_size": 1,
                    "collective_transport": "none",
                    "source_revision": revision,
                    "checkpoint_inventory_sha256": inventory,
                },
            },
        ),
        StageOutput(
            stage_name="end_to_end",
            data={
                "returncode": 0,
                "frames_path": str(reference_path),
                "source_revision": revision,
                "receipt": {
                    "status": "passed",
                    "source_revision": revision,
                    "checkpoint_snapshot": {"inventory_sha256": inventory},
                },
            },
        ),
        ThresholdProfile(
            task_strategy="diffusion_media_generation",
            metrics=thresholds or _visual_thresholds(*reference.shape[:3]),
        ),
        StageSpec(name="end_to_end"),
    )


def test_minimax_h3_comparator_accepts_high_frequency_texture_drift(
    tmp_path: Path,
) -> None:
    reference = _synthetic_video()
    yy, xx = np.mgrid[: reference.shape[1], : reference.shape[2]]
    checkerboard = (((xx + yy) % 2) * 2 - 1).astype(np.float32)
    candidate = reference + 0.06 * checkerboard[None, :, :, None]

    result = _compare_arrays(tmp_path, reference, candidate)

    assert result.status == "passed"
    assert result.metrics["psnr_db"].value < 40.0
    assert result.metrics["mean_absolute_error"].value > 1.0 / 255.0
    assert result.metrics["psnr_db"].operator == "diagnostic"
    assert result.metrics["psnr_db"].threshold is None
    assert result.metrics["mean_absolute_error"].operator == "diagnostic"
    assert result.metrics["frame_low_frequency_correlation_minimum"].value == pytest.approx(1.0)
    assert result.metrics["temporal_activity_correlation"].value == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("failure_mode", "expected_failed_metric"),
    [
        ("collapse", "frame_std_ratio_minimum"),
        ("freeze", "temporal_activity_ratio_minimum"),
        ("timing_shift", "temporal_activity_correlation"),
    ],
)
def test_minimax_h3_comparator_rejects_visible_failure_modes(
    tmp_path: Path,
    failure_mode: str,
    expected_failed_metric: str,
) -> None:
    reference = _synthetic_video()
    if failure_mode == "collapse":
        frame_means = reference.mean(axis=(1, 2, 3), keepdims=True)
        candidate = np.broadcast_to(frame_means, reference.shape).copy()
    elif failure_mode == "freeze":
        candidate = np.repeat(reference[:1], reference.shape[0], axis=0)
    else:
        candidate = np.roll(reference, shift=3, axis=0)

    result = _compare_arrays(tmp_path, reference, candidate)

    assert result.status == "failed"
    assert not result.metrics[expected_failed_metric].passed


def test_minimax_h3_comparator_requires_exact_shape_and_finite_pixels(
    tmp_path: Path,
) -> None:
    reference = _synthetic_video()
    result = _compare_arrays(
        tmp_path,
        reference,
        reference,
        thresholds=_visual_thresholds(
            reference.shape[0] + 1,
            reference.shape[1],
            reference.shape[2],
        ),
    )
    assert result.status == "failed"
    assert not result.metrics["num_frames"].passed

    non_finite = reference.copy()
    non_finite[3, 4, 5, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite pixels"):
        _compare_arrays(tmp_path, reference, non_finite)

    out_of_range = reference.copy()
    out_of_range[3, 4, 5, 1] = 1.01
    with pytest.raises(ValueError, match=r"pixels outside \[0, 1\]"):
        _compare_arrays(tmp_path, reference, out_of_range)


def test_compare_video_cli_binds_threshold_schema_and_run_receipts(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.npy"
    candidate_path = tmp_path / "candidate.npy"
    frames = np.zeros((1, 16, 16, 3), dtype=np.float32)
    np.save(reference_path, frames)
    np.save(candidate_path, frames)
    revision = "1" * 40
    workload = {
        "prompt": "test",
        "seed": 0,
        "height": 16,
        "width": 16,
        "num_frames": 1,
        "num_inference_steps": 1,
    }
    reference_receipt_path = tmp_path / "reference.json"
    candidate_receipt_path = tmp_path / "candidate.json"
    reference_receipt_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "source_revision": revision,
                "checkpoint_inventory_sha256": "a" * 64,
                "workload": workload,
                "frames": file_record(reference_path),
            }
        )
    )
    candidate_receipt = {
        "status": "passed",
        "backend": "tensorrt_native_single_device",
        "world_size": 1,
        "collective_transport": "none",
        "plan_sha256": {"denoiser.plan": "b" * 64},
        "source_revision": revision,
        "checkpoint_inventory_sha256": "a" * 64,
        "request": workload,
        "frames": file_record(candidate_path),
    }
    candidate_receipt_path.write_text(json.dumps(candidate_receipt))
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(
        json.dumps(
            {
                "threshold_overrides": {
                    **_visual_thresholds(1, 16, 16),
                }
            }
        )
    )
    output_path = tmp_path / "comparison.json"
    command = [
        sys.executable,
        str(_MODEL_DIR / "compare_video.py"),
        str(reference_path),
        str(candidate_path),
        "--reference-receipt",
        str(reference_receipt_path),
        "--candidate-receipt",
        str(candidate_receipt_path),
        "--thresholds",
        str(thresholds_path),
        "--output",
        str(output_path),
        "--source-revision",
        revision,
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_PROJECT_DIR / "python")
    result = subprocess.run(
        command, cwd=_PROJECT_DIR, env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    comparison = json.loads(output_path.read_text())
    assert comparison["passed"] is True
    assert comparison["pixel_metrics_gating"] is False
    assert comparison["metrics"]["psnr_db"]["operator"] == "diagnostic"

    candidate_receipt["source_revision"] = "2" * 40
    candidate_receipt_path.write_text(json.dumps(candidate_receipt))
    result = subprocess.run(
        command, cwd=_PROJECT_DIR, env=environment, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "different source revision" in result.stderr


def test_model_e2e(case_name: str, request) -> None:
    _runner.run_model_e2e(case_name, request)
