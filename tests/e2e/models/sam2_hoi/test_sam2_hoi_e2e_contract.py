# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import ctypes
import json
import os
from pathlib import Path
import tomllib

import numpy as np
import pytest

from tensorrt_model_connect.families.sam2_hoi.validation import AccuracyThresholds
from tests.e2e.models.sam2_hoi.e2e_plugins._schema import (
    FRAME_COUNT,
    expected_keys,
    frame_key,
    load_npz_arrays,
    normalize_runtime_json,
)
from tests.e2e.models.sam2_hoi.e2e_plugins.comparators.video_tracking import (
    HoiVideoTrackingComparator,
)
from tests.e2e.models.sam2_hoi.e2e_plugins.references.archive import (
    Sam2HoiArchiveReference,
)
from tests.e2e.models.sam2_hoi.e2e_plugins.runners import video_tracking
from tests.e2e.models.sam2_hoi.e2e_plugins.runners.video_tracking import (
    HoiVideoTrackingRunner,
)
from tests.e2e_harness.contracts import (
    E2ECase,
    RunContext,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)
from tests.e2e_harness.manifest_loader import load_model_manifest
from tests.e2e_harness.registry import (
    activate_model_plugins,
    get_comparator,
    get_reference,
    get_runner,
    reset,
)


FAMILY_DIR = Path(__file__).resolve().parent
SOURCE_COMMIT = "79ab25d6bd5535bcb748de9f3b90e16b1a24e58d"


def _arrays(*, height: int = 64, width: int = 64) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for frame in range(FRAME_COUNT):
        masks = np.zeros((2, 1, height, width), dtype=np.uint8)
        masks[0, 0, 4:20, 5:21] = 1
        masks[1, 0, 24:54, 20:56] = 1
        arrays.update(
            {
                frame_key(frame, "object_ids"): np.asarray([0, 1], dtype=np.int64),
                frame_key(frame, "binary_masks"): masks,
                frame_key(frame, "det_bboxes"): np.asarray(
                    [[5.0, 4.0, 21.0, 20.0], [20.0, 24.0, 56.0, 54.0]],
                    dtype=np.float32,
                ),
                frame_key(frame, "det_labels"): np.asarray([1, 2], dtype=np.int64),
                frame_key(frame, "det_scores"): np.asarray([0.49, 0.41], dtype=np.float32),
                frame_key(frame, "interaction_pairs"): np.asarray([[0, 1]], dtype=np.int64),
            }
        )
    return arrays


def _write_npz(path: Path, arrays: dict[str, np.ndarray] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **(arrays or _arrays()))
    return path


def _threshold_profile() -> ThresholdProfile:
    defaults = AccuracyThresholds()
    return ThresholdProfile(
        task_strategy="hoi_video_tracking",
        metrics={
            "detection_score_max_abs": defaults.detection_score_max_abs,
            "detection_box_max_abs_pixels": defaults.detection_box_max_abs_pixels,
            "detection_box_min_iou": defaults.detection_box_min_iou,
            "mask_min_iou": defaults.mask_min_iou,
            "mask_min_dice": defaults.mask_min_dice,
            "mask_min_pixel_agreement": defaults.mask_min_pixel_agreement,
            "exact_object_ids": 1.0,
            "exact_det_labels": 1.0,
            "exact_interaction_pairs": 1.0,
            "required_frame_count": 5.0,
        },
    )


def _case(
    model_root: Path,
    reference_npz: Path,
    frames_dir: Path | None = None,
) -> E2ECase:
    return E2ECase(
        name="sam2-hoi-tracking",
        hf_id=str(model_root),
        family="sam2_hoi",
        runtime_strategy="sam2_hoi_video_tracking",
        task_strategy="hoi_video_tracking",
        reference_backend="sam2_hoi_archive_reference",
        bundle="sam2-hoi-tracking.bundle",
        inputs={
            "frames_dir": str(frames_dir or model_root / "inputs"),
            "reference_npz": str(reference_npz),
            "expected_frame_count": 5,
            "expected_height": 64,
            "expected_width": 64,
        },
        metadata={"source_commit": SOURCE_COMMIT},
    )


def _runtime_json(
    path: Path,
    arrays: dict[str, np.ndarray],
    masks_dir: Path | None = None,
) -> Path:
    masks_dir = masks_dir or path.parent / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame in range(FRAME_COUNT):
        mask_path = masks_dir / f"frame_{frame:06d}.npy"
        np.save(mask_path, arrays[frame_key(frame, "binary_masks")])
        frames.append(
            {
                "frame_index": frame,
                "object_ids": arrays[frame_key(frame, "object_ids")].tolist(),
                "binary_masks_path": str(mask_path.relative_to(path.parent)),
                "det_bboxes": arrays[frame_key(frame, "det_bboxes")].tolist(),
                "det_labels": arrays[frame_key(frame, "det_labels")].tolist(),
                "det_scores": arrays[frame_key(frame, "det_scores")].tolist(),
                "interaction_pairs": arrays[frame_key(frame, "interaction_pairs")].tolist(),
            }
        )
    path.write_text(json.dumps({"schema_version": 1, "frames": frames}), encoding="utf-8")
    return path


class _FakeFunction:
    def __init__(self, implementation):
        self.implementation = implementation
        self.argtypes = None
        self.restype = None

    def __call__(self, *arguments):
        return self.implementation(*arguments)


class _FakeSam2HoiLibrary:
    def __init__(self, captured: dict[str, object], *, run_status: int = 0) -> None:
        self.trtmc_sam2_hoi_video_abi_version = _FakeFunction(lambda: 1)
        self.trtmc_sam2_hoi_video_last_error = _FakeFunction(lambda: b"fixture C ABI error")
        self.trtmc_sam2_hoi_video_create_from_bundle_v1 = _FakeFunction(
            lambda bundle, plugins, backend: self._create(captured, bundle, plugins, backend)
        )
        self.trtmc_sam2_hoi_video_session_destroy = _FakeFunction(
            lambda session: captured.update(destroyed_session=session)
        )
        self.trtmc_sam2_hoi_video_run_jpeg_files_v1 = _FakeFunction(
            lambda *arguments: self._run(captured, run_status, *arguments)
        )

    @staticmethod
    def _create(captured: dict[str, object], bundle, plugins, backend) -> int:
        captured["create_paths"] = tuple(os.fsdecode(value) for value in (bundle, plugins, backend))
        return 17

    @staticmethod
    def _run(captured: dict[str, object], status: int, *arguments) -> int:
        captured["frame_paths"] = tuple(os.fsdecode(value) for value in arguments[1:6])
        captured["output_json"] = os.fsdecode(arguments[6])
        captured["output_masks_dir"] = os.fsdecode(arguments[7])
        captured["result_size"] = arguments[9]
        if status != 0:
            return status
        result = ctypes.cast(arguments[8], ctypes.POINTER(video_tracking._RunResult)).contents
        result.struct_size = ctypes.sizeof(result)
        result.abi_version = 1
        result.produced_frame_count = FRAME_COUNT
        _runtime_json(
            Path(captured["output_json"]),
            _arrays(),
            Path(captured["output_masks_dir"]),
        )
        return 0


def test_manifest_declares_distinct_archive_backed_tracking_contract() -> None:
    manifest = FAMILY_DIR / "manifests" / "sam2-hoi-tracking.json"
    raw_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    model = load_model_manifest(manifest)
    case = model.testcases[0]
    assert model.family == "sam2_hoi"
    assert model.hf_id == "artifacts/sam2_hoi/hoi"
    assert case.task_strategy == "hoi_video_tracking"
    assert case.runtime_strategy == "sam2_hoi_video_tracking"
    assert case.reference_backend == "sam2_hoi_archive_reference"
    assert case.oracle_level == "L3_snapshot_regression"
    assert raw_manifest["model_source_kind"] == "local_source_package"
    assert raw_manifest["runtime_api"] == {
        "kind": "model_owned_c_abi",
        "library": "libtrtmc_model_sam2_hoi.so",
        "header": "trtmc/models/sam2_hoi_video.h",
        "entrypoint": "trtmc_sam2_hoi_video_run_jpeg_files_v1",
    }
    assert "fixed five-JPEG public C ABI" in raw_manifest["benchmark_exclusion_reason"]
    assert "ci_tier" not in raw_manifest["testcases"][0]
    assert [stage.name for stage in case.stages] == ["full_tracking"]
    assert case.inputs["expected_frame_count"] == 5
    assert case.threshold_overrides == _threshold_profile().metrics


def test_owner_pins_the_real_source_package_for_both_proof_suites() -> None:
    owner = tomllib.loads((FAMILY_DIR / "MODEL.toml").read_text(encoding="utf-8"))

    assert owner["model_source_package"] == {
        "suites": ["premerge", "nightly"],
        "cache_file": (
            "sam2_hoi/hoi_infer-"
            "891c1729686f93fafab7ac6d4994db6cf3c3f27595ff9e5f97c9d6ee406f12b0.tar.gz"
        ),
        "sha256": "891c1729686f93fafab7ac6d4994db6cf3c3f27595ff9e5f97c9d6ee406f12b0",
        "project_path": "artifacts/sam2_hoi/hoi",
        "entrypoint": "SOURCE_COMMIT",
    }


def test_model_plugins_are_discovered_only_from_the_sam2_hoi_owner() -> None:
    try:
        activate_model_plugins(FAMILY_DIR)
        plugins = (
            get_runner("hoi_video_tracking"),
            get_reference("sam2_hoi_archive_reference"),
            get_comparator("hoi_video_tracking"),
        )
        assert all(plugin is not None for plugin in plugins)
        assert all(
            type(plugin).__module__.startswith("tests.e2e.models.sam2_hoi.e2e_plugins.")
            for plugin in plugins
        )
    finally:
        reset()


def test_model_owned_plugins_do_not_import_sibling_segmentation_families() -> None:
    forbidden = {
        "tests.e2e.models.sam",
        "tests.e2e.models.sam3",
        "tensorrt_model_connect.families.sam",
        "tensorrt_model_connect.families.sam3",
    }
    imports: set[str] = set()
    for path in sorted((FAMILY_DIR / "e2e_plugins").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
    assert not (imports & forbidden)


def test_no_real_model_or_reference_artifacts_are_checked_in() -> None:
    forbidden_suffixes = {".pt", ".pth", ".jpg", ".jpeg", ".npz", ".npy"}
    assert not [
        path
        for path in FAMILY_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]


def test_runtime_json_normalizes_to_exact_thirty_array_schema(tmp_path: Path) -> None:
    runtime_json = _runtime_json(tmp_path / "tracking.json", _arrays())
    output_npz = normalize_runtime_json(runtime_json, tmp_path / "tracking.npz")
    loaded = load_npz_arrays(output_npz)
    assert set(loaded) == expected_keys()
    assert len(loaded) == 30
    assert loaded["frame_000004_binary_masks"].shape == (2, 1, 64, 64)


def test_runtime_json_rejects_less_than_all_five_frames(tmp_path: Path) -> None:
    runtime_json = _runtime_json(tmp_path / "tracking.json", _arrays())
    payload = json.loads(runtime_json.read_text(encoding="utf-8"))
    payload["frames"].pop()
    runtime_json.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly five frames"):
        normalize_runtime_json(runtime_json, tmp_path / "tracking.npz")


def test_comparator_accepts_exact_five_frame_parity(tmp_path: Path) -> None:
    reference = _write_npz(tmp_path / "reference.npz")
    candidate = _write_npz(tmp_path / "candidate.npz")
    result = HoiVideoTrackingComparator().compare(
        StageOutput("full_tracking", data={"output_npz": str(candidate)}),
        StageOutput("full_tracking", data={"output_npz": str(reference)}),
        _threshold_profile(),
        StageSpec(name="full_tracking"),
    )
    assert result.status == StageStatus.PASSED.value
    assert all(metric.passed for metric in result.metrics.values())


@pytest.mark.parametrize(
    ("field", "mutate", "metric"),
    [
        (
            "object_ids",
            lambda value: np.asarray([1, 0], dtype=value.dtype),
            "exact_object_ids",
        ),
        (
            "det_labels",
            lambda value: np.asarray([2, 1], dtype=value.dtype),
            "exact_det_labels",
        ),
        (
            "interaction_pairs",
            lambda value: np.asarray([[1, 0]], dtype=value.dtype),
            "exact_interaction_pairs",
        ),
        (
            "det_scores",
            lambda value: value + np.float32(0.0101),
            "detection_score_max_abs",
        ),
        (
            "det_bboxes",
            lambda value: value + np.float32(2.1),
            "detection_box_max_abs_pixels",
        ),
    ],
)
def test_comparator_rejects_identity_and_detection_drift(
    tmp_path: Path,
    field: str,
    mutate,
    metric: str,
) -> None:
    reference_arrays = _arrays()
    candidate_arrays = {key: value.copy() for key, value in reference_arrays.items()}
    key = frame_key(2, field)
    candidate_arrays[key] = mutate(candidate_arrays[key])
    reference = _write_npz(tmp_path / "reference.npz", reference_arrays)
    candidate = _write_npz(tmp_path / "candidate.npz", candidate_arrays)
    result = HoiVideoTrackingComparator().compare(
        StageOutput("full_tracking", data={"output_npz": str(candidate)}),
        StageOutput("full_tracking", data={"output_npz": str(reference)}),
        _threshold_profile(),
        StageSpec(name="full_tracking"),
    )
    assert result.status == StageStatus.FAILED.value
    assert not result.metrics[metric].passed


def test_comparator_rejects_mask_drift(tmp_path: Path) -> None:
    reference_arrays = _arrays()
    candidate_arrays = {key: value.copy() for key, value in reference_arrays.items()}
    masks = candidate_arrays[frame_key(3, "binary_masks")]
    masks[0, 0, 4:10, 5:15] = 0
    reference = _write_npz(tmp_path / "reference.npz", reference_arrays)
    candidate = _write_npz(tmp_path / "candidate.npz", candidate_arrays)
    result = HoiVideoTrackingComparator().compare(
        StageOutput("full_tracking", data={"output_npz": str(candidate)}),
        StageOutput("full_tracking", data={"output_npz": str(reference)}),
        _threshold_profile(),
        StageSpec(name="full_tracking"),
    )
    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["mask_min_iou"].passed
    assert not result.metrics["mask_min_dice"].passed
    assert not result.metrics["mask_min_pixel_agreement"].passed


def test_comparator_fails_closed_when_threshold_sidecar_is_incomplete(
    tmp_path: Path,
) -> None:
    reference = _write_npz(tmp_path / "reference.npz")
    candidate = _write_npz(tmp_path / "candidate.npz")
    result = HoiVideoTrackingComparator().compare(
        StageOutput("full_tracking", data={"output_npz": str(candidate)}),
        StageOutput("full_tracking", data={"output_npz": str(reference)}),
        ThresholdProfile(task_strategy="hoi_video_tracking", metrics={}),
        StageSpec(name="full_tracking"),
    )
    assert result.status == StageStatus.ERROR.value
    assert "threshold sidecar is incomplete" in result.message


def test_archive_reference_requires_exact_source_commit(tmp_path: Path) -> None:
    model_root = tmp_path / "hoi"
    model_root.mkdir()
    (model_root / "SOURCE_COMMIT").write_text(SOURCE_COMMIT + "\n", encoding="utf-8")
    reference_npz = _write_npz(model_root / "reference" / "torch_reference.npz")
    output = Sam2HoiArchiveReference().run_stage(
        _case(model_root, reference_npz),
        StageSpec(name="full_tracking"),
        RunContext(case=_case(model_root, reference_npz)),
    )
    assert output.data["frame_count"] == 5
    assert len(output.data["frames"]) == 5

    (model_root / "SOURCE_COMMIT").write_text("0" * 40 + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source commit mismatch"):
        Sam2HoiArchiveReference().run_stage(
            _case(model_root, reference_npz),
            StageSpec(name="full_tracking"),
            RunContext(case=_case(model_root, reference_npz)),
        )


def test_runtime_library_accepts_model_proof_parent_or_leaf(tmp_path: Path) -> None:
    parent = tmp_path / "model-plugins"
    leaf = parent / "sam2_hoi"
    leaf.mkdir(parents=True)
    nested = leaf / "libtrtmc_model_sam2_hoi.so"
    nested.touch()
    assert video_tracking._runtime_library(parent) == nested
    assert video_tracking._runtime_library(leaf) == nested
    (parent / nested.name).touch()
    with pytest.raises(RuntimeError, match="resolve exactly once"):
        video_tracking._runtime_library(parent)


def test_runner_invokes_only_model_owned_c_abi_and_exposes_all_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "hoi"
    frames_dir = model_root / "inputs"
    frames_dir.mkdir(parents=True)
    for frame in range(FRAME_COUNT):
        (frames_dir / f"{frame:06d}.jpg").write_bytes(b"synthetic")
    reference_npz = _write_npz(model_root / "reference" / "torch_reference.npz")
    case = _case(model_root, reference_npz, frames_dir)
    engine_dir = tmp_path / "engines"
    engine_dir.mkdir()
    (engine_dir / case.bundle).write_bytes(b"synthetic bundle")
    backend_dir = tmp_path / "backends"
    backend_dir.mkdir()
    binary = backend_dir / "trtmc"
    binary.write_bytes(b"synthetic binary")
    plugin_dir = tmp_path / "model-plugins"
    plugin_dir.mkdir()
    runtime_library = plugin_dir / "libtrtmc_model_sam2_hoi.so"
    runtime_library.write_bytes(b"synthetic DSO")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        video_tracking,
        "_load_library",
        lambda path: _FakeSam2HoiLibrary(captured),
    )
    output = HoiVideoTrackingRunner().run_stage(
        case,
        StageSpec(name="full_tracking"),
        RunContext(
            case=case,
            binary_path=str(binary),
            engine_dir=str(engine_dir),
            artifacts_dir=str(tmp_path / "artifacts"),
            model_plugin_dir=str(plugin_dir),
        ),
    )
    assert captured["create_paths"] == (
        str(engine_dir / case.bundle),
        str(plugin_dir),
        str(backend_dir),
    )
    assert captured["frame_paths"] == tuple(
        str(frames_dir / f"{frame:06d}.jpg") for frame in range(FRAME_COUNT)
    )
    assert captured["output_json"] == str(tmp_path / "artifacts" / case.name / "trt_tracking.json")
    assert captured["output_masks_dir"] == str(tmp_path / "artifacts" / case.name / "trt_masks")
    assert captured["result_size"] == 64
    assert captured["destroyed_session"] == 17
    assert output.data["frame_count"] == 5
    assert [frame["frame_index"] for frame in output.data["frames"]] == list(range(5))
    assert len(list(Path(captured["output_masks_dir"]).glob("*.npy"))) == FRAME_COUNT
    assert output.metadata == {
        "runtime_library": str(runtime_library),
        "runtime_entrypoint": "trtmc_sam2_hoi_video_run_jpeg_files_v1",
    }


def test_runner_fails_closed_when_model_owned_c_abi_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "hoi"
    frames_dir = model_root / "inputs"
    frames_dir.mkdir(parents=True)
    for frame in range(FRAME_COUNT):
        (frames_dir / f"{frame:06d}.jpg").write_bytes(b"synthetic")
    reference_npz = _write_npz(model_root / "reference" / "torch_reference.npz")
    case = _case(model_root, reference_npz, frames_dir)
    engine_dir = tmp_path / "engines"
    engine_dir.mkdir()
    (engine_dir / case.bundle).write_bytes(b"synthetic bundle")
    backend_dir = tmp_path / "backends"
    backend_dir.mkdir()
    binary = backend_dir / "trtmc"
    binary.write_bytes(b"synthetic binary")
    plugin_dir = tmp_path / "model-plugins"
    plugin_dir.mkdir()
    (plugin_dir / "libtrtmc_model_sam2_hoi.so").write_bytes(b"synthetic DSO")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        video_tracking,
        "_load_library",
        lambda path: _FakeSam2HoiLibrary(captured, run_status=3),
    )
    with pytest.raises(RuntimeError, match="status 3: fixture C ABI error"):
        HoiVideoTrackingRunner().run_stage(
            case,
            StageSpec(name="full_tracking"),
            RunContext(
                case=case,
                binary_path=str(binary),
                engine_dir=str(engine_dir),
                artifacts_dir=str(tmp_path / "artifacts"),
                model_plugin_dir=str(plugin_dir),
            ),
        )
    assert captured["destroyed_session"] == 17


def test_runtime_strategy_matrix_declares_model_owned_c_abi_exemption() -> None:
    matrix = json.loads(
        (FAMILY_DIR.parents[2] / "runtime_strategy_matrix.yaml").read_text(encoding="utf-8")
    )
    entry = matrix["runtime_strategies"]["sam2_hoi_video_tracking"]
    assert entry["task_strategy"] == "hoi_video_tracking"
    assert entry["cli_commands"] == []
    assert "model-owned" in entry["cli_exemption"]
    assert "C ABI" in entry["cli_exemption"]
    assert entry["runner_class"].endswith("HoiVideoTrackingRunner")
    assert entry["comparator_class"].endswith("HoiVideoTrackingComparator")
