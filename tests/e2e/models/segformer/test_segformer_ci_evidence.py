# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused checks for SegFormer acceptance thresholds and report evidence."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from tests.e2e.models.segformer.e2e_plugins.contract import (
    SegformerSegmentationPlugin,
)
from tests.e2e.models.segformer.e2e_plugins.runners.segmentation import (
    SegmentationRunner,
    _save_segmentation_visualization,
)
from tests.e2e_harness.contracts import (
    E2ECase,
    RunContext,
    StageOutput,
    ThresholdProfile,
)
from tests.e2e_harness.orchestrator import _auto_register_artifacts


def _output(class_map: np.ndarray) -> StageOutput:
    return StageOutput(stage_name="full_inference", data={"class_map": class_map})


def test_contract_consumes_declared_segmentation_thresholds() -> None:
    trt = np.array([[0, 0], [0, 1]], dtype=np.int32)
    ref = np.array([[0, 1], [1, 1]], dtype=np.int32)
    thresholds = ThresholdProfile(
        task_strategy="segmentation",
        metrics={
            "mIoU": 0.94,
            "pixel_accuracy": 0.99,
            # Legacy names must not weaken the declared acceptance criteria.
            "contract_miou_threshold": 0.10,
            "contract_pixel_accuracy": 0.10,
        },
    )

    result = SegformerSegmentationPlugin().verify(_output(trt), _output(ref), None, thresholds)

    assert result.status == "failed"
    assert result.metrics["mIoU"].threshold == 0.94
    assert result.metrics["pixel_accuracy"].threshold == 0.99
    assert not result.metrics["mIoU"].passed
    assert not result.metrics["pixel_accuracy"].passed


def test_contract_requires_both_declared_thresholds() -> None:
    class_map = np.zeros((2, 2), dtype=np.int32)
    thresholds = ThresholdProfile(
        task_strategy="segmentation",
        metrics={"pixel_accuracy": 0.85},
    )

    result = SegformerSegmentationPlugin().verify(
        _output(class_map), _output(class_map), None, thresholds
    )

    assert result.status == "error"
    assert "mIoU" in result.message


def test_contract_rejects_mask_shape_mismatch() -> None:
    thresholds = ThresholdProfile(
        task_strategy="segmentation",
        metrics={"mIoU": 0.94, "pixel_accuracy": 0.99},
    )

    result = SegformerSegmentationPlugin().verify(
        _output(np.zeros((128, 128), dtype=np.int32)),
        _output(np.zeros((382, 640), dtype=np.int32)),
        None,
        thresholds,
    )

    assert result.status == "error"
    assert "shape mismatch" in result.message


def test_single_gpu_thresholds_reject_the_reproduced_regression() -> None:
    threshold_path = Path(__file__).parent / "thresholds" / "segformer-b0-ade.json"
    thresholds = json.loads(threshold_path.read_text())["threshold_overrides"]

    assert thresholds["mIoU"] == 0.94
    assert thresholds["pixel_accuracy"] == 0.99
    assert thresholds["min_pixel_agreement"] == 0.99


def test_trt_visualization_matches_hf_palette_and_input_size(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.png"
    visualization_path = tmp_path / "seg_output_viz.png"
    Image.new("RGB", (6, 4), color="white").save(input_path)
    class_map = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int32)

    returned = _save_segmentation_visualization(
        class_map,
        image_path=str(input_path),
        visualization_path=str(visualization_path),
    )

    assert returned == str(visualization_path)
    rendered = np.asarray(Image.open(visualization_path).convert("RGB"))
    assert rendered.shape == (4, 6, 3)

    random_state = np.random.RandomState(42)
    hf_palette = random_state.randint(0, 255, (3, 3), dtype=np.uint8)
    hf_palette[0] = [0, 0, 0]
    expected_ids = np.asarray(
        Image.fromarray(class_map.astype(np.uint8)).resize(
            (6, 4),
            resample=getattr(Image, "Resampling", Image).NEAREST,
        )
    )
    np.testing.assert_array_equal(rendered, hf_palette[expected_ids])


def test_runner_registers_only_human_readable_segmentation_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.png"
    binary_path = tmp_path / "trtmc"
    Image.new("RGB", (6, 4), color="white").save(input_path)
    binary_path.touch()
    raw_class_map = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)

    def fake_run(command, **_kwargs):
        output_path = Path(command[command.index("--output") + 1])
        Image.fromarray(raw_class_map).save(output_path)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "tests.e2e.models.segformer.e2e_plugins.runners.segmentation.subprocess.run",
        fake_run,
    )
    case = E2ECase(
        name="segformer-report-test",
        hf_id="nvidia/segformer-b0-finetuned-ade-512-512",
        family="segformer",
        runtime_strategy="segformer_segmentation",
        task_strategy="segmentation",
        bundle="segformer-report-test.bundle",
        inputs={"image": str(input_path)},
    )
    artifacts_dir = tmp_path / "artifacts"
    output = SegmentationRunner()._run_full_inference(
        case,
        RunContext(
            case=case,
            artifacts_dir=str(artifacts_dir),
            binary_path=str(binary_path),
            engine_dir=str(tmp_path),
        ),
    )

    assert output.data["class_map_path"].endswith("seg_output.png")
    assert output.data["viz_path"].endswith("seg_output_viz.png")
    assert "output_path" not in output.data
    assert "segmentation_map_path" not in output.data

    class Sink:
        base_dir = artifacts_dir / case.name

        def __init__(self) -> None:
            self.artifacts = {}

        def register_artifact(self, key, value) -> None:
            self.artifacts[key] = value

    sink = Sink()
    _auto_register_artifacts(sink, output, "trt")
    assert sink.artifacts == {"trt_segmentation_map": "seg_output_viz.png"}
