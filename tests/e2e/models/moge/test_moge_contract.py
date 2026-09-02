# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only registration and measured-parity comparator tests for MoGe."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tests.e2e.models.moge.e2e_plugins.comparator import MonocularGeometryComparator
from tests.e2e.models.moge.e2e_plugins import reference as reference_module
from tests.e2e.models.moge.e2e_plugins.runner import _load_geometry_output
from tests.e2e_harness.contracts import (
    E2ECase,
    RunContext,
    StageOutput,
    StageSpec,
    ThresholdProfile,
)

_ROOT = Path(__file__).resolve().parent


def _geometry() -> dict:
    mask = np.asarray([[1, 0], [1, 1]], dtype=np.uint8)
    depth = np.asarray([[1.0, np.inf], [2.0, 3.0]], dtype=np.float32)
    points = np.zeros((2, 2, 3), dtype=np.float32)
    points[..., 2] = depth
    points[0, 1] = np.inf
    return {
        "points": points,
        "depth": depth,
        "mask": mask,
        "intrinsics": np.asarray(
            [[1.2, 0.0, 0.5], [0.0, 1.2, 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        "height": 2,
        "width": 2,
        "num_tokens": 1800,
    }


def _thresholds() -> ThresholdProfile:
    payload = json.loads((_ROOT / "thresholds/moge-2-vitl.json").read_text(encoding="utf-8"))
    return ThresholdProfile(
        task_strategy="monocular_geometry",
        metrics=payload["threshold_overrides"],
    )


def test_descriptor_and_manifest_pin_exact_identity() -> None:
    with (_ROOT / "MODEL.toml").open("rb") as source:
        descriptor = tomllib.load(source)
    manifest = json.loads((_ROOT / "manifests/moge-2-vitl.json").read_text(encoding="utf-8"))
    assert descriptor["id"] == descriptor["plugin"] == "moge"
    assert descriptor["model_reference_cache"]["revision"] == (
        "74fbce054ebed49800de42d0ad0e83495065719a"
    )
    assert (manifest["hf_id"], manifest["hf_revision"]) == (
        "Ruicheng/moge-2-vitl",
        "39c4d5e957afe587e04eec59dc2bcc3be5ecd968",
    )
    assert manifest["runtime_strategy"] == "moge_monocular_geometry"
    assert manifest["task_strategy"] == "monocular_geometry"
    assert manifest["e2e_parallel_resource"] == "exclusive_gpu"
    assert manifest["e2e_min_free_gpu_memory_mib"] == 16384
    testcase = manifest["testcases"][0]
    assert testcase["num_tokens"] == 1800
    assert "reference_backend" not in testcase
    assert "oracle_level" not in testcase
    assert "preflight_requirements" not in testcase
    assert "threshold_overrides" not in manifest


def test_measured_parity_comparator_passes_matching_geometry() -> None:
    actual = _geometry()
    reference = _geometry()
    result = MonocularGeometryComparator().compare(
        StageOutput("full_inference", data=actual),
        StageOutput("full_inference", data=reference),
        _thresholds(),
        StageSpec("full_inference"),
    )
    assert result.status == "passed"
    assert set(result.metrics) == set(_thresholds().metrics)
    assert "FP32 geometry parity passed" in result.message


def test_measured_parity_comparator_rejects_numerically_unrelated_geometry() -> None:
    actual = _geometry()
    reference = _geometry()
    reference["depth"] = reference["depth"].copy()
    reference["depth"][reference["mask"].astype(bool)] *= 10.0
    reference["points"] = reference["points"].copy()
    reference["points"][reference["mask"].astype(bool)] *= 10.0
    result = MonocularGeometryComparator().compare(
        StageOutput("full_inference", data=actual),
        StageOutput("full_inference", data=reference),
        _thresholds(),
        StageSpec("full_inference"),
    )
    assert result.status == "failed"
    assert not result.metrics["depth_absrel_mean"].passed
    assert not result.metrics["depth_rel_l2"].passed
    assert not result.metrics["points_rel_l2"].passed


def test_runner_loads_the_public_geometry_artifact_contract(tmp_path: Path) -> None:
    height, width = 2, 3
    np.arange(height * width * 3, dtype="<f4").tofile(tmp_path / "points.f32")
    np.arange(1, height * width + 1, dtype="<f4").tofile(tmp_path / "depth.f32")
    np.ones((height, width), dtype=np.uint8).tofile(tmp_path / "mask.u8")
    (tmp_path / "intrinsics.json").write_text(
        json.dumps(
            {
                "height": height,
                "width": width,
                "intrinsics": [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
                "normalized": True,
            }
        ),
        encoding="utf-8",
    )
    payload = _load_geometry_output(tmp_path, {"height": height, "width": width})
    assert payload["points"].shape == (height, width, 3)
    assert payload["depth"].shape == payload["mask"].shape == (height, width)
    assert payload["intrinsics"].shape == (3, 3)
    assert payload["num_tokens"] == 1800


def test_reference_uses_safe_checkpoint_load_and_explicit_tokens() -> None:
    source = (_ROOT / "official_reference.py").read_text(encoding="utf-8")
    assert "weights_only=True" in source
    assert "num_tokens=arguments.num_tokens" in source
    assert "requires --num-tokens 1800" in source
    assert "use_fp16=False" in source


def test_reference_reports_a_missing_output_artifact(
    tmp_path: Path, monkeypatch,
) -> None:
    source_root = tmp_path / "source"
    (source_root / "moge/model").mkdir(parents=True)
    (source_root / "moge/model/v2.py").write_text("# pinned source\n", encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.pt").write_bytes(b"checkpoint")
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    monkeypatch.setenv("TRTMC_MOGE_SOURCE_DIR", str(source_root))
    monkeypatch.setattr(
        reference_module,
        "_checkpoint_snapshot",
        lambda _case, **_kwargs: snapshot,
    )
    monkeypatch.setattr(
        reference_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    case = E2ECase(
        name="moge-2-vitl",
        hf_id="Ruicheng/moge-2-vitl",
        family="moge",
        runtime_strategy="moge_monocular_geometry",
        hf_revision="a" * 40,
        inputs={"image": str(image), "num_tokens": 1800},
    )
    ctx = RunContext(case=case, artifacts_dir=str(tmp_path / "artifacts"))

    output = reference_module.MoGeTorchReference().run_stage(
        case, StageSpec("full_inference"), ctx
    )

    assert "exited 0 but did not create" in output.data["output_error"]
