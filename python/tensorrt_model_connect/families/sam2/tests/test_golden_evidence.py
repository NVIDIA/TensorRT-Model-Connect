# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused CPU tests for the SAM2 exact-workload golden evidence seam."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect.families.sam2 import golden_evidence
from tensorrt_model_connect.families.sam2.archive_contract import (
    REFERENCE_CHECKPOINT_SHA256,
    REFERENCE_CONFIG_SHA256,
)
from tensorrt_model_connect.families.sam2.golden_evidence import (
    COMPATIBLE_SOURCE_COMMIT,
    COMPATIBLE_SOURCE_FILES_SHA256,
    COMPATIBLE_SOURCE_OVERLAY_FILES_SHA256,
    INPUT_IMAGES_DECODED_RGB_UINT8_SHA256,
    INPUT_IMAGES_RESIZED_1024_RGB_UINT8_SHA256,
    INPUT_IMAGES_SHA256,
    MASK_SHAPE,
    PUBLIC_SAM2_BASE_COMMIT,
    PUBLIC_SAM2_BASE_FILES_SHA256,
    FrameZeroBBox,
    Provenance,
    Sam2GoldenEvidenceError,
    WorkloadCapture,
    compare_captures_exact,
    compare_evidence,
    load_evidence,
    repository_qualification_state,
    write_evidence,
)


_CHECKED_IN_REFERENCE = (
    Path(__file__).resolve().parents[5]
    / "tests/cpp/models/sam2/data/golden/compatible_source_pytorch_bf16"
)


def _masks() -> np.ndarray:
    masks = np.zeros(MASK_SHAPE, dtype=np.uint8)
    for frame_index in range(5):
        masks[frame_index, 0, 100 + frame_index : 200 + frame_index, 300:400] = 1
    return masks


def _bbox() -> FrameZeroBBox:
    return FrameZeroBBox(
        original_xyxy=(106.25, 250.0, 318.75, 500.0),
        model_xyxy_1024=(100.0, 200.0, 300.0, 400.0),
        score=0.5,
        label=1,
    )


def _capture(*, masks: np.ndarray | None = None, bbox: FrameZeroBBox | None = None):
    return WorkloadCapture(
        masks=_masks() if masks is None else masks,
        frame_zero_bbox=_bbox() if bbox is None else bbox,
    )


def _source_environment() -> dict[str, object]:
    return {
        "python": "3.12.3",
        "antlr4_python3_runtime": "4.9.3",
        "numpy": "2.5.2",
        "pillow": "12.3.0",
        "pillow_jpeg_codec": "6.2",
        "libjpeg_turbo": "3.1.4.1",
        "input_images_decoded_rgb_uint8_sha256": dict(INPUT_IMAGES_DECODED_RGB_UINT8_SHA256),
        "input_images_resized_1024_rgb_uint8_sha256": dict(
            INPUT_IMAGES_RESIZED_1024_RGB_UINT8_SHA256
        ),
        "hydra_core": "1.3.2",
        "iopath": "0.1.10",
        "omegaconf": "2.3.1",
        "portalocker": "4.1.0",
        "pyyaml": "6.0.3",
        "torch": "2.7.1+cu128",
        "torchvision": "0.22.1+cu128",
        "tqdm": "4.67.1",
        "torch_cuda": "12.8",
        "cuda_driver": "595.58.03",
        "gpu": "NVIDIA L4",
        "cuda_capability": [8, 9],
        "sam2_optional_extension_present": False,
        "autocast": "cuda bfloat16",
        "tf32_matmul": True,
        "tf32_cudnn": True,
        "cudnn_deterministic": False,
        "cudnn_benchmark": False,
        "cudnn_version": 90701,
        "deterministic_algorithms": False,
        "python_isolated": True,
        "python_safe_path": True,
        "python_no_user_site": True,
        "python_no_site": True,
        "controlled_site_packages": ("/workspace/ref-work/.venv/lib/python3.12/site-packages"),
        "venv_pyvenv_cfg_sha256": (
            "16529e11b2fe1e50d7bca13c16b18bdd5ff478ae2db7750e483aba6e3733d858"
        ),
        "capture_input_isolation": "private_read_only_verified_snapshot_v1",
        "capture_runs": 3,
        "async_loading_frames": True,
        "apply_postprocessing": True,
        "config_name": "configs/sam2.1/trtmc_delivery_bbox_59488bb78c7c.yaml",
        "dependency_origins": {
            name: (f"/workspace/ref-work/.venv/lib/python3.12/site-packages/{name}/__init__.py")
            for name in (
                "antlr4",
                "hydra",
                "iopath",
                "numpy",
                "omegaconf",
                "pillow",
                "portalocker",
                "pyyaml",
                "torch",
                "torchvision",
                "tqdm",
            )
        },
        "video_res_logits_dtypes": [["torch.bfloat16"] * 5 for _ in range(3)],
    }


def _capture_code_artifacts() -> dict[str, str]:
    return {
        "capture_code/tensorrt_model_connect.families.sam2.archive_contract": (
            golden_evidence.AUTHORITATIVE_ARCHIVE_CONTRACT_SHA256
        ),
        "capture_code/tensorrt_model_connect.families.sam2.golden_evidence.normalized": (
            golden_evidence.AUTHORITATIVE_GOLDEN_EVIDENCE_NORMALIZED_SHA256
        ),
        "capture_code/tensorrt_model_connect.families.sam2.capture_golden": "a" * 64,
    }


def _source_provenance(**changes) -> Provenance:
    values = {
        "source_commit": PUBLIC_SAM2_BASE_COMMIT,
        "source_overlay_declared_commit": COMPATIBLE_SOURCE_COMMIT,
        "source_files_sha256": dict(COMPATIBLE_SOURCE_FILES_SHA256),
        "checkpoint_sha256": REFERENCE_CHECKPOINT_SHA256,
        "config_sha256": REFERENCE_CONFIG_SHA256,
        "image_files_sha256": dict(INPUT_IMAGES_SHA256),
        "capture_tool_sha256": "a" * 64,
        "environment": _source_environment(),
        "artifacts_sha256": _capture_code_artifacts(),
    }
    values.update(changes)
    return Provenance(**values)


def _candidate_provenance(**changes) -> Provenance:
    values = {
        "source_commit": "c" * 40,
        "source_overlay_declared_commit": "d" * 40,
        "source_files_sha256": {"golden_evidence.py": "d" * 64},
        "checkpoint_sha256": REFERENCE_CHECKPOINT_SHA256,
        "config_sha256": REFERENCE_CONFIG_SHA256,
        "image_files_sha256": dict(INPUT_IMAGES_SHA256),
        "capture_tool_sha256": "e" * 64,
        "environment": {
            "python": "3.12.3",
            "gpu": "NVIDIA L4",
            "tensorrt": "11.1.0.106",
            "precision": "candidate",
        },
        "artifacts_sha256": {"sam2.bundle": "f" * 64},
    }
    values.update(changes)
    return Provenance(**values)


def _write_reference(
    path: Path,
    pin: pytest.MonkeyPatch,
    capture: WorkloadCapture | None = None,
) -> None:
    selected = _capture() if capture is None else capture
    write_evidence(
        path,
        capture=selected,
        provenance=_source_provenance(),
        producer="compatible_source_pytorch_bf16",
        authoritative_source_run=True,
        replay_captures=(selected, selected),
    )
    manifest_sha256 = hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
    pin.setattr(
        golden_evidence,
        "AUTHORITATIVE_REFERENCE_MANIFEST_SHA256",
        manifest_sha256,
    )


def _write_candidate(path: Path, capture: WorkloadCapture | None = None) -> None:
    write_evidence(
        path,
        capture=_capture() if capture is None else capture,
        provenance=_candidate_provenance(),
        producer="candidate",
    )


@pytest.fixture
def pinned_capture_tool(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.setattr(golden_evidence, "AUTHORITATIVE_CAPTURE_TOOL_SHA256", "a" * 64)
    return monkeypatch


def test_checked_in_reference_is_pinned_while_runtime_remains_unqualified() -> None:
    assert golden_evidence.AUTHORITATIVE_CAPTURE_TOOL_SHA256 is not None
    assert golden_evidence.AUTHORITATIVE_CAPTURE_TOOL_SHA256 != (
        "0c28cdf957bfa949bd0ad0099980266cf6927c62d98e33b96026bdc44244f3a8"
    )
    assert golden_evidence.AUTHORITATIVE_CAPTURE_TOOL_SHA256 != "0" * 64
    assert golden_evidence.AUTHORITATIVE_GOLDEN_EVIDENCE_NORMALIZED_SHA256 != "0" * 64
    assert golden_evidence.AUTHORITATIVE_REFERENCE_MANIFEST_SHA256 == (
        "c25251ee27da05afd75adc3c6869cbc2944b80c05c5d6e703b6ebbbba697a4f0"
    )
    assert repository_qualification_state() == {
        "status": "unqualified",
        "reason": "authoritative_compatible_source_golden_pending",
        "required_reference_producer": "compatible_source_pytorch_bf16",
        "required_deterministic_runs": 3,
        "required_frames": [0, 1, 2, 3, 4],
        "required_post_nms_detection_count": 1,
        "required_selected_object_count": 1,
        "capture_tool_sha256_pinned": True,
        "reference_manifest_sha256_pinned": True,
    }


def test_checked_in_reference_loads_as_authoritative() -> None:
    loaded = load_evidence(_CHECKED_IN_REFERENCE)

    assert loaded.manifest_sha256 == golden_evidence.AUTHORITATIVE_REFERENCE_MANIFEST_SHA256
    assert loaded.authoritative_reference is True


def test_zero_hash_placeholders_are_not_reviewed_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(golden_evidence, "AUTHORITATIVE_CAPTURE_TOOL_SHA256", "0" * 64)
    monkeypatch.setattr(
        golden_evidence,
        "AUTHORITATIVE_REFERENCE_MANIFEST_SHA256",
        "0" * 64,
    )

    state = repository_qualification_state()
    assert state["capture_tool_sha256_pinned"] is False
    assert state["reference_manifest_sha256_pinned"] is False


def test_round_trip_is_lossless_bit_packed_and_path_independent(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    capture = _capture()
    first_manifest = write_evidence(
        first,
        capture=capture,
        provenance=_candidate_provenance(),
        producer="candidate",
    )
    second_manifest = write_evidence(
        second,
        capture=capture,
        provenance=_candidate_provenance(),
        producer="candidate",
    )

    assert first_manifest == second_manifest
    loaded = load_evidence(first)
    np.testing.assert_array_equal(loaded.masks, capture.masks)
    assert first_manifest["masks"]["encoding"] == "numpy_packbits_v1"
    assert first_manifest["masks"]["bitorder"] == "little"
    assert first_manifest["workload"]["mask_materialization"] == {
        "source": "propagate_in_video.video_res_logits",
        "source_shape": [5, 1, 1280, 1088],
        "comparison": "greater_than",
        "threshold": 0.0,
        "upstream_resize_contract": {
            "source": "per_frame_pred_mask_logits",
            "source_shape": [1, 1, 256, 256],
            "resize_mode": "bilinear",
            "resize_size_hw": [1280, 1088],
            "align_corners": False,
        },
    }
    assert (first / "masks.bitpack").stat().st_size == math.prod(MASK_SHAPE) // 8
    assert first_manifest["qualification"] == {
        "status": "unqualified",
        "reason": "authoritative_reference_comparison_pending",
    }


@pytest.mark.parametrize(
    ("capture", "message"),
    [
        (
            WorkloadCapture(
                masks=np.zeros((4, 1, 1280, 1088), dtype=np.uint8),
                frame_zero_bbox=_bbox(),
            ),
            "exact shape",
        ),
        (
            WorkloadCapture(
                masks=np.full(MASK_SHAPE, 2, dtype=np.uint8),
                frame_zero_bbox=_bbox(),
            ),
            "only 0 and 1",
        ),
        (replace(_capture(), post_nms_detection_count=2), "top-1 selection is forbidden"),
        (replace(_capture(), post_nms_detection_count=1.0), "top-1 selection is forbidden"),
        (replace(_capture(), selected_object_id=1), "selected object id 0"),
        (replace(_capture(), selected_object_id=0.0), "selected object id 0"),
    ],
)
def test_exact_workload_shape_binary_and_single_detection_are_mandatory(
    tmp_path: Path, capture: WorkloadCapture, message: str
) -> None:
    with pytest.raises(Sam2GoldenEvidenceError, match=message):
        write_evidence(
            tmp_path / "evidence",
            capture=capture,
            provenance=_candidate_provenance(),
            producer="candidate",
        )


def test_bbox_coordinate_spaces_must_describe_the_same_unclipped_box(tmp_path: Path) -> None:
    inconsistent = replace(_bbox(), original_xyxy=(100.0, 250.0, 318.75, 500.0))
    with pytest.raises(Sam2GoldenEvidenceError, match="inconsistent scaling"):
        _write_candidate(tmp_path / "candidate", _capture(bbox=inconsistent))


def test_bbox_label_must_be_one_of_the_two_head_classes(tmp_path: Path) -> None:
    with pytest.raises(Sam2GoldenEvidenceError, match="two bbox classes"):
        _write_candidate(tmp_path / "candidate", _capture(bbox=replace(_bbox(), label=2)))


def test_authoritative_reference_requires_exact_delivered_config_not_adjacent_source_yaml(
    tmp_path: Path,
) -> None:
    capture = _capture()
    with pytest.raises(Sam2GoldenEvidenceError, match="config hash mismatch"):
        write_evidence(
            tmp_path / "reference",
            capture=capture,
            provenance=_source_provenance(config_sha256="b" * 64),
            producer="compatible_source_pytorch_bf16",
            authoritative_source_run=True,
            replay_captures=(capture, capture),
        )


def test_authoritative_reference_binds_public_base_and_exact_bbox_overlay(
    tmp_path: Path,
    pinned_capture_tool: pytest.MonkeyPatch,
) -> None:
    assert len(PUBLIC_SAM2_BASE_FILES_SHA256) == 25
    assert len(COMPATIBLE_SOURCE_OVERLAY_FILES_SHA256) == 7
    assert len(COMPATIBLE_SOURCE_FILES_SHA256) == 28
    assert COMPATIBLE_SOURCE_FILES_SHA256["sam2/modeling/backbones/__init__.py"] == (
        "34bd8069c54764e7b8d73a78905dbe6467140a2f73170875128f6ca4d8cdd0aa"
    )
    assert not any("hoi" in path for path in COMPATIBLE_SOURCE_FILES_SHA256)

    capture = _capture()
    changed_files = dict(COMPATIBLE_SOURCE_FILES_SHA256)
    changed_files["sam2/modeling/sam2_base.py"] = "0" * 64
    provenance_variants = (
        _source_provenance(source_commit="0" * 40),
        _source_provenance(source_overlay_declared_commit="1" * 40),
        _source_provenance(source_files_sha256=changed_files),
    )
    for index, provenance in enumerate(provenance_variants):
        with pytest.raises(Sam2GoldenEvidenceError, match="mismatch"):
            write_evidence(
                tmp_path / f"reference-{index}",
                capture=capture,
                provenance=provenance,
                producer="compatible_source_pytorch_bf16",
                authoritative_source_run=True,
                replay_captures=(capture, capture),
            )


def test_authoritative_reference_rejects_decoded_rgb_drift(
    tmp_path: Path,
    pinned_capture_tool: pytest.MonkeyPatch,
) -> None:
    capture = _capture()
    environment = _source_environment()
    decoded = dict(INPUT_IMAGES_DECODED_RGB_UINT8_SHA256)
    decoded["000003.jpg"] = "0" * 64
    environment["input_images_decoded_rgb_uint8_sha256"] = decoded

    with pytest.raises(
        Sam2GoldenEvidenceError,
        match="environment input_images_decoded_rgb_uint8_sha256 mismatch",
    ):
        write_evidence(
            tmp_path / "reference",
            capture=capture,
            provenance=_source_provenance(environment=environment),
            producer="compatible_source_pytorch_bf16",
            authoritative_source_run=True,
            replay_captures=(capture, capture),
        )


def test_authoritative_reference_rejects_source_resize_rgb_drift(
    tmp_path: Path,
    pinned_capture_tool: pytest.MonkeyPatch,
) -> None:
    capture = _capture()
    environment = _source_environment()
    resized = dict(INPUT_IMAGES_RESIZED_1024_RGB_UINT8_SHA256)
    resized["000003.jpg"] = "0" * 64
    environment["input_images_resized_1024_rgb_uint8_sha256"] = resized

    with pytest.raises(
        Sam2GoldenEvidenceError,
        match="environment input_images_resized_1024_rgb_uint8_sha256 mismatch",
    ):
        write_evidence(
            tmp_path / "reference",
            capture=capture,
            provenance=_source_provenance(environment=environment),
            producer="compatible_source_pytorch_bf16",
            authoritative_source_run=True,
            replay_captures=(capture, capture),
        )


def test_authoritative_reference_requires_a_pinned_capture_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = _capture()
    monkeypatch.setattr(golden_evidence, "AUTHORITATIVE_CAPTURE_TOOL_SHA256", None)
    with pytest.raises(Sam2GoldenEvidenceError, match="capture-tool contract is not pinned"):
        write_evidence(
            tmp_path / "reference",
            capture=capture,
            provenance=_source_provenance(),
            producer="compatible_source_pytorch_bf16",
            authoritative_source_run=True,
            replay_captures=(capture, capture),
        )


def test_authoritative_reference_requires_three_exact_source_runs(
    tmp_path: Path, pinned_capture_tool: pytest.MonkeyPatch
) -> None:
    capture = _capture()
    with pytest.raises(Sam2GoldenEvidenceError, match="at least three"):
        write_evidence(
            tmp_path / "too-few",
            capture=capture,
            provenance=_source_provenance(),
            producer="compatible_source_pytorch_bf16",
            authoritative_source_run=True,
            replay_captures=(capture,),
        )

    changed_masks = capture.masks.copy()
    changed_masks[0, 0, 100, 300] = 0
    with pytest.raises(Sam2GoldenEvidenceError, match="not bitwise deterministic"):
        write_evidence(
            tmp_path / "different",
            capture=capture,
            provenance=_source_provenance(),
            producer="compatible_source_pytorch_bf16",
            authoritative_source_run=True,
            replay_captures=(capture, _capture(masks=changed_masks)),
        )


def test_manifest_status_and_fabricated_replays_cannot_self_promote(
    tmp_path: Path,
    pinned_capture_tool: pytest.MonkeyPatch,
) -> None:
    pinned_capture_tool.setattr(
        golden_evidence,
        "AUTHORITATIVE_REFERENCE_MANIFEST_SHA256",
        "0" * 64,
    )
    root = tmp_path / "source"
    capture = _capture()
    write_evidence(
        root,
        capture=capture,
        provenance=_source_provenance(),
        producer="compatible_source_pytorch_bf16",
    )
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    capture_hash = manifest["capture_sha256"]
    manifest["qualification"] = {
        "status": "authoritative_reference_candidate",
        "reason": "requires_checked_in_exact_manifest_sha256_pin",
    }
    manifest["determinism"] = {
        "run_count": 3,
        "all_exact": True,
        "capture_sha256": [capture_hash, capture_hash, capture_hash],
        "replays": [{"exact": True}, {"exact": True}],
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    loaded = load_evidence(root)
    assert loaded.authoritative_reference is False


def test_authority_requires_the_byte_exact_reviewed_manifest(
    tmp_path: Path,
    pinned_capture_tool: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reference"
    _write_reference(root, pinned_capture_tool)
    assert load_evidence(root).authoritative_reference is True

    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["qualification"]["reason"] = "edited-after-review"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    assert load_evidence(root).authoritative_reference is False


def test_exact_capture_comparison_covers_bbox_and_each_mask() -> None:
    reference = _capture()
    changed_masks = reference.masks.copy()
    changed_masks[3, 0, 100, 300] = 1
    result = compare_captures_exact(reference, _capture(masks=changed_masks))

    assert result["exact"] is False
    assert result["frame_zero_bbox_exact"] is True
    assert [item["exact"] for item in result["frame_masks_exact"]] == [
        True,
        True,
        True,
        False,
        True,
    ]


def test_identical_candidate_passes_accuracy_only_not_runtime_qualification(
    tmp_path: Path, pinned_capture_tool: pytest.MonkeyPatch
) -> None:
    _write_reference(tmp_path / "reference", pinned_capture_tool)
    _write_candidate(tmp_path / "candidate")

    result = compare_evidence(tmp_path / "reference", tmp_path / "candidate")

    assert result["status"] == "accuracy_parity_passed"
    assert result["passed"] is True
    assert result["runtime_qualified"] is False
    assert all(result["gates"].values())
    assert result["metrics"]["masks"]["minimum_frame_iou"] == 1.0
    assert result["metrics"]["masks"]["macro_iou"] == 1.0
    assert (
        result["comparison_sha256"]
        == compare_evidence(tmp_path / "reference", tmp_path / "candidate")["comparison_sha256"]
    )


def test_comparison_stays_unqualified_without_authoritative_source_reference(
    tmp_path: Path,
) -> None:
    capture = _capture()
    write_evidence(
        tmp_path / "reference",
        capture=capture,
        provenance=_source_provenance(),
        producer="compatible_source_pytorch_bf16",
        replay_captures=(capture, capture),
    )
    _write_candidate(tmp_path / "candidate")

    result = compare_evidence(tmp_path / "reference", tmp_path / "candidate")

    assert result["status"] == "unqualified"
    assert result["passed"] is False
    assert result["gates"]["reference_authoritative"] is False


@pytest.mark.parametrize(
    ("candidate", "failed_gate"),
    [
        (_capture(bbox=replace(_bbox(), label=0)), "label_exact"),
        (_capture(bbox=replace(_bbox(), score=0.52)), "score_error"),
        (
            _capture(
                bbox=FrameZeroBBox(
                    original_xyxy=(107.3125, 250.0, 319.8125, 500.0),
                    model_xyxy_1024=(101.0, 200.0, 301.0, 400.0),
                    score=0.5,
                    label=1,
                )
            ),
            "original_box_coordinate_error",
        ),
    ],
)
def test_bbox_gates_reject_material_candidate_drift(
    tmp_path: Path,
    candidate: WorkloadCapture,
    failed_gate: str,
    pinned_capture_tool: pytest.MonkeyPatch,
) -> None:
    _write_reference(tmp_path / "reference", pinned_capture_tool)
    _write_candidate(tmp_path / "candidate", candidate)

    result = compare_evidence(tmp_path / "reference", tmp_path / "candidate")

    assert result["status"] == "accuracy_parity_rejected"
    assert result["passed"] is False
    assert result["gates"][failed_gate] is False


def test_every_frame_macro_and_global_mask_iou_are_strict_gates(
    tmp_path: Path, pinned_capture_tool: pytest.MonkeyPatch
) -> None:
    candidate_masks = _masks()
    for frame_index in range(5):
        candidate_masks[frame_index, 0, 100 + frame_index : 102 + frame_index, 300:400] = 0
    _write_reference(tmp_path / "reference", pinned_capture_tool)
    _write_candidate(tmp_path / "candidate", _capture(masks=candidate_masks))

    result = compare_evidence(tmp_path / "reference", tmp_path / "candidate")

    assert result["status"] == "accuracy_parity_rejected"
    assert result["gates"]["minimum_frame_mask_iou"] is False
    assert result["gates"]["macro_mask_iou"] is False
    assert result["gates"]["global_mask_iou"] is False
    assert result["metrics"]["masks"]["frames"][2]["iou"] == pytest.approx(0.98)
    assert result["metrics"]["masks"]["global_iou"] == pytest.approx(0.98)


def test_candidate_asset_provenance_must_match_authoritative_reference(
    tmp_path: Path, pinned_capture_tool: pytest.MonkeyPatch
) -> None:
    _write_reference(tmp_path / "reference", pinned_capture_tool)
    wrong_images = dict(INPUT_IMAGES_SHA256)
    wrong_images["000004.jpg"] = "0" * 64
    write_evidence(
        tmp_path / "candidate",
        capture=_capture(),
        provenance=_candidate_provenance(image_files_sha256=wrong_images),
        producer="candidate",
    )

    result = compare_evidence(tmp_path / "reference", tmp_path / "candidate")

    assert result["passed"] is False
    assert result["gates"]["asset_identity_exact"] is False


def test_candidate_side_requires_candidate_producer(
    tmp_path: Path, pinned_capture_tool: pytest.MonkeyPatch
) -> None:
    _write_reference(tmp_path / "reference", pinned_capture_tool)
    capture = _capture()
    write_evidence(
        tmp_path / "candidate",
        capture=capture,
        provenance=_source_provenance(),
        producer="compatible_source_pytorch_bf16",
        replay_captures=(capture, capture),
    )

    result = compare_evidence(tmp_path / "reference", tmp_path / "candidate")

    assert result["passed"] is False
    assert result["gates"]["candidate_producer"] is False


def test_payload_tampering_is_detected_before_comparison(tmp_path: Path) -> None:
    destination = tmp_path / "candidate"
    _write_candidate(destination)
    payload = bytearray((destination / "masks.bitpack").read_bytes())
    payload[0] ^= 1
    (destination / "masks.bitpack").write_bytes(payload)

    with pytest.raises(Sam2GoldenEvidenceError, match="does not match its receipt"):
        load_evidence(destination)


def test_manifest_capture_identity_tampering_is_detected(tmp_path: Path) -> None:
    destination = tmp_path / "candidate"
    _write_candidate(destination)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frames"][0]["logical_uint8_sha256"] = hashlib.sha256(b"wrong").hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(Sam2GoldenEvidenceError, match="frame mask hash"):
        load_evidence(destination)
