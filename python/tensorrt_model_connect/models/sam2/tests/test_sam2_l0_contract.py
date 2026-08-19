# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contracts for the secretless SAM2 premerge smoke."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import pytest

from tensorrt_model_connect.benchmark.catalog import ManifestCatalog
from tensorrt_model_connect.benchmark.types import BenchmarkError
import tensorrt_model_connect.models.sam2.tests.e2e_plugins.reference as sam2_reference
from tensorrt_model_connect.models.sam2.tests.e2e_plugins.comparator import comparator
from tensorrt_model_connect.models.sam2.tests.e2e_plugins.runner import _png, _runtime_library
from tests.e2e_harness.contracts import (
    E2ECase,
    RunContext,
    StageOutput,
    StageSpec,
    ThresholdProfile,
)
from tests.e2e_harness.threshold_policy import requires_threshold_sidecar


def _receipt(
    *,
    repeat_exact: bool = True,
    counts: list[int] | None = None,
    hashes: list[str] | None = None,
) -> StageOutput:
    return StageOutput(
        stage_name="five_frame_tracking",
        data={
            "schema_version": 2,
            "runtime_invariants": {
                "plan_sections": [
                    "engine_plan",
                    "sam2_prompt_engine_plan",
                    "sam2_recurrent_h1_engine_plan",
                    "sam2_recurrent_h2_engine_plan",
                    "sam2_recurrent_h3_engine_plan",
                    "sam2_recurrent_h4_engine_plan",
                ],
                "checkpoint_variant": "public_sam2_1_small_with_synthetic_bbox_v1",
                "same_session_repeat_exact": repeat_exact,
                "bbox_xyxy": [136.0, 160.0, 952.0, 1120.0],
                "detector_score": 1.0,
                "label": 1,
                "binary_masks": True,
                "mask_foreground_pixels": counts or [2000, 3000, 4000, 5000, 6000],
                "mask_sha256": hashes or [f"{index:064x}" for index in range(5)],
            },
        },
    )


def test_l0_manifest_is_threshold_free_and_keeps_nightly_separate() -> None:
    model_dir = Path(__file__).resolve().parent
    l0 = json.loads((model_dir / "manifests/sam2-public-core-l0.json").read_text())
    nightly = json.loads((model_dir / "manifests/sam2-l4-local.json").read_text())
    testcase = l0["testcases"][0]
    assert l0["hf_id"] == "facebook/sam2.1-hiera-small"
    assert l0["hf_revision"] == "6c381d9c16faed5e8a7c4a2cd99918bdca8316e4"
    assert l0["inputs"] == {"fixture_kind": "deterministic_synthetic_rgb8"}
    assert l0["task_strategy"] == nightly["task_strategy"] == "prompted_segmentation"
    assert l0["benchmark_exclusion_reason"] == nightly["benchmark_exclusion_reason"]
    assert testcase["ci_tier"] == "l0_only"
    assert testcase["oracle_level"] == "L4_invariants"
    assert not requires_threshold_sidecar({**l0, **testcase})
    assert nightly["ci_tier"] == "nightly_only"
    assert nightly["testcases"][0]["test_category"] == "regression"


def test_sam2_profiles_are_visible_but_not_generic_benchmarks() -> None:
    catalog = ManifestCatalog()
    entries = {entry.name: entry for entry in catalog.entries() if entry.family == "sam2"}
    assert set(entries) == {"sam2-l4-local", "sam2-public-core-l0"}
    assert {entry.status for entry in entries.values()} == {"e2e_only"}
    for name in entries:
        with pytest.raises(BenchmarkError, match="five-frame public C ABI"):
            catalog.resolve(name)


def test_runtime_library_accepts_model_proof_parent_or_leaf(tmp_path: Path) -> None:
    parent = tmp_path / "model-plugins"
    leaf = parent / "sam2"
    leaf.mkdir(parents=True)
    nested = leaf / "libtrtmc_model_sam2.so"
    nested.touch()
    assert _runtime_library(parent) == nested
    assert _runtime_library(leaf) == nested
    (parent / nested.name).touch()
    with pytest.raises(RuntimeError, match="resolve exactly once"):
        _runtime_library(parent)


def test_report_png_has_valid_header_and_dimensions() -> None:
    encoded = _png(3, 2, 3, bytes(range(18)))
    assert encoded.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", encoded[16:24]) == (3, 2)


def test_golden_reference_writes_exact_first_mask(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sam2_reference, "_WIDTH", 3)
    monkeypatch.setattr(sam2_reference, "_HEIGHT", 2)
    monkeypatch.setattr(sam2_reference, "_FRAME_PIXELS", 6)
    monkeypatch.setattr(
        sam2_reference,
        "_load_golden",
        lambda _root: (((0.0, 0.0, 1.0, 1.0), 1.0, 1), bytes((0, 1, 0, 1, 1, 0)), "hash"),
    )
    case = E2ECase(
        "sam2-reference", "custom", "sam2", "sam2_bbox_video_tracking", inputs={"fixture_dir": "f"}
    )
    stage = StageSpec("five_frame_tracking")
    ctx = RunContext(case=case, engine_dir=str(tmp_path), artifacts_dir=str(tmp_path))
    output = sam2_reference.Sam2LocalGoldenReference().run_stage(case, stage, ctx)
    encoded = Path(output.data["viz_path"]).read_bytes()
    compressed_size = struct.unpack(">I", encoded[33:37])[0]

    assert output.data["golden_manifest_sha256"] == "hash"
    assert Path(output.data["viz_path"]).parent == tmp_path / case.name
    assert struct.unpack(">II", encoded[16:24]) == (3, 2)
    assert zlib.decompress(encoded[41 : 41 + compressed_size]) == bytes((0, 0, 255, 0, 0, 255, 255, 0))
    with pytest.raises(RuntimeError, match="artifacts are unavailable"):
        sam2_reference.Sam2LocalGoldenReference().run_stage(
            case, stage, RunContext(case=case, engine_dir=str(tmp_path))
        )


def test_l0_comparator_requires_same_session_exact_repeat() -> None:
    reference = StageOutput(stage_name="five_frame_tracking", data={"_invariant_only": True})
    stage = StageSpec("five_frame_tracking")
    threshold = ThresholdProfile("sam2_bbox_video_tracking")
    assert comparator.compare(_receipt(), reference, threshold, stage).status == "passed"
    assert (
        comparator.compare(_receipt(repeat_exact=False), reference, threshold, stage).status
        == "failed"
    )


def test_l0_comparator_rejects_degenerate_masks() -> None:
    reference = StageOutput(stage_name="five_frame_tracking", data={"_invariant_only": True})
    stage = StageSpec("five_frame_tracking")
    threshold = ThresholdProfile("sam2_bbox_video_tracking")
    frame_pixels = 1280 * 1088
    cases = (
        [0] * 5,
        [frame_pixels] * 5,
        [5] * 5,
    )
    for counts in cases:
        assert comparator.compare(_receipt(counts=counts), reference, threshold, stage).status == (
            "failed"
        )
    assert (
        comparator.compare(_receipt(hashes=["0" * 64] * 5), reference, threshold, stage).status
        == "failed"
    )
