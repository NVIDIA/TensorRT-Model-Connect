# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU controls for actual YOLOv10 recording, pixels and failure boundaries."""

from copy import deepcopy
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from families.yolov10.tests import reporting, test_e2e as e2e
from tools import e2e_evidence
from tools.e2e_report import render_case


def _native():
    return {"boxes": [20.0, 25.0, 80.0, 70.0], "classes": [2], "scores": [0.9]}


def _recorder(tmp_path):
    return e2e_evidence.Evidence(
        tmp_path / "evidence",
        family="yolov10",
        case="report",
        source_revision="a" * 40,
        roots=(tmp_path,),
    )


def _record(tmp_path, native, reference, *, size=(120, 100)):
    source = tmp_path / "source.png"
    Image.new("RGB", size, "white").save(source)
    recorder = _recorder(tmp_path)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        reporting.begin_detection_operation(source, "CLI")
        e2e_evidence.record_evidence("native", native)
        e2e_evidence.record_evidence("reference", reference)
        reporting.record_detection_views(native, reference, source, tmp_path / "views")
    finally:
        e2e_evidence._ACTIVE.reset(token)
    return recorder


def test_real_boxes_classes_and_scores_are_paired_without_coordinate_conversion(tmp_path):
    native = _native()
    reference = {"boxes": [30.0, 35.0, 90.0, 80.0], "classes": [7], "scores": [0.8]}
    originals = deepcopy((native, reference))
    recorder = _record(tmp_path, native, reference)
    for role, x, y in (("Native", 20, 50), ("Reference", 30, 60)):
        view = next(v for v in recorder.data["views"] if v["title"].endswith(role))
        with Image.open(recorder.directory / view["image"]["artifact"]) as picture:
            assert picture.getpixel((x, y)) == reporting._COLORS[role]
            assert picture.getpixel((110, 95)) == (255, 255, 255)
    recorder.finish("passed")
    primary = render_case(recorder.data, recorder.directory).split(
        '<details class="case-details">', 1
    )[0]
    assert 'data-media-paired="true"' in primary
    assert 'data-media-role="native"' in primary
    assert 'data-media-role="reference"' in primary
    assert "class ID 2, score 0.9, XYXY [20, 25, 80, 70]" in primary
    assert "class ID 7, score 0.8, XYXY [30, 35, 90, 80]" in primary
    assert (native, reference) == originals


def test_reference_conversion_is_reused_once_and_unclipped(tmp_path):
    rows = np.array([[0.0, 0.0, 30.0, 40.0, 0.75, 2.0]])
    original = rows.copy()
    recorder = _recorder(tmp_path)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        reference = e2e._record_reference_boxes(rows, 2.0, 10.0, 20.0)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert reference == {"boxes": [-5.0, -10.0, 10.0, 10.0], "scores": [0.75], "classes": [2]}
    recorder = _record(tmp_path, _native(), reference)
    assert recorder.data["detection_view_summary"]["reference"]["detections"][0]["box"] == [
        -5,
        -10,
        10,
        10,
    ]
    np.testing.assert_array_equal(rows, original)


def test_display_cap_and_resize_preserve_complete_outputs(tmp_path):
    native = {"boxes": [200.0, 250.0, 800.0, 700.0] * 40, "scores": [0.9] * 40, "classes": [2] * 40}
    original = deepcopy(native)
    recorder = _record(tmp_path, native, native, size=(1536, 1024))
    for role in ("native", "reference"):
        assert len(recorder.data[role]["scores"]) == 40
        summary = recorder.data["detection_view_summary"][role]
        assert summary["displayed"] == 32 and summary["total"] == 40
        assert summary["source_size"] == [1536, 1024]
        assert summary["display_size"] == [768, 512]
        view = next(v for v in recorder.data["views"] if v["title"].endswith(role.title()))
        with Image.open(recorder.directory / view["image"]["artifact"]) as picture:
            assert picture.getpixel((100, 250)) == reporting._COLORS[role.title()]
        assert "first 32 of 40" in view["caption"]
        assert "original-image pixels" in view["caption"]
        assert "resized to at most 768 x 768" in view["caption"]
    assert native == original


def test_disabled_recording_does_not_read_images_or_convert_reference(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("recording performed I/O while disabled")

    monkeypatch.setattr(Image, "open", forbidden)
    reporting.begin_detection_operation(tmp_path / "missing.png", "CLI")
    reporting.record_detection_views(None, None, tmp_path / "missing.png", tmp_path / "views")
    assert e2e._record_reference_boxes(None, 0, 0, 0) is None
    assert list(tmp_path.iterdir()) == []


def test_malformed_native_preserves_reference_overlay(tmp_path):
    recorder = _record(tmp_path, {"boxes": [], "classes": [2], "scores": [0.9]}, _native())
    assert "unavailable" in recorder.data["detection_view_summary"]["native"]
    assert recorder.data["detection_view_summary"]["reference"]["displayed"] == 1
    assert "image" not in recorder.data["views"][0]
    assert "image" in recorder.data["views"][1]


def _official(tmp_path, monkeypatch, *, reference_failure=None, sdk_failure=None, actual=None):
    source = tmp_path / "source.png"
    Image.new("RGB", (120, 100), "white").save(source)
    actual = actual or _native()
    actual = {**actual, "task": "image_to_boxes", "image_height": 100, "image_width": 120}
    manifest = {
        "name": "report",
        "family": "yolov10",
        "hf_id": "example/model",
        "hf_revision": "fixed",
        "task": "image_to_boxes",
        "bundle": "model.bundle",
        "precision": "fp32",
        "max_sequence_length": 1,
        "tensor_parallel_size": 1,
    }
    monkeypatch.setattr(e2e, "CASES", {"report": (manifest, {"name": "report"})})
    monkeypatch.setattr(
        e2e,
        "_required_path",
        lambda value, label: tmp_path / "binary" if label == "TRTMC_BINARY" else tmp_path,
    )
    monkeypatch.setattr(e2e, "_model_dir", lambda *args: tmp_path)
    monkeypatch.setattr(e2e, "_asset", lambda *args: source)
    monkeypatch.setattr(e2e, "build", lambda *args: None)
    (tmp_path / "config.json").write_text("{}")
    for name in (
        "libtrtmc_backend_trt.so",
        "libtrtmc_model_yolov10.so",
        "test_yolov10_sdk_c",
        "test_yolov10_sdk_cpp",
    ):
        (tmp_path / name).touch()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == sys.executable:
            if reference_failure is not None:
                raise reference_failure
            np.savez(
                command[-1],
                expected=np.array([[50, 70, 170, 160, 0.9, 2]]),
                scale=2.0,
                pad_x=10.0,
                pad_y=20.0,
            )
        if str(command[0]).endswith("_sdk_cpp") and sdk_failure is not None:
            raise sdk_failure
        return SimpleNamespace(stdout=json.dumps(actual), stderr="", args=command)

    monkeypatch.setattr(e2e.subprocess, "run", run)
    return _recorder(tmp_path), calls


def _run_official(recorder, tmp_path):
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        e2e.test_official_checkpoint_e2e("report", tmp_path)
    finally:
        e2e_evidence._ACTIVE.reset(token)


def test_failed_reference_keeps_native_and_original_exception(tmp_path, monkeypatch):
    error = subprocess.CalledProcessError(9, ["reference"])
    recorder, calls = _official(tmp_path, monkeypatch, reference_failure=error)
    with pytest.raises(subprocess.CalledProcessError) as raised:
        _run_official(recorder, tmp_path)
    assert raised.value is error and len(calls) == 2
    assert recorder.data["native"]["boxes"] == _native()["boxes"]
    assert recorder.data["reference"] is None
    assert [v["title"] for v in recorder.data["views"]] == ["Sample 0 / Native"]
    recorder.finish("failed")
    primary = render_case(recorder.data, recorder.directory).split(
        '<details class="case-details">', 1
    )[0]
    assert 'data-media-role="native"' in primary and "data:image/png;base64," in primary
    assert 'data-media-role="reference"' not in primary


def test_preview_disk_failure_does_not_replace_original_count_assertion(tmp_path, monkeypatch):
    actual = {key: value * 2 for key, value in _native().items()}
    recorder, calls = _official(tmp_path, monkeypatch, actual=actual)

    def fail(*args, **kwargs):
        raise OSError("preview disk full")

    monkeypatch.setattr(Image.Image, "save", fail)
    with pytest.raises(AssertionError, match="engine returned 2 detections, reference 1"):
        _run_official(recorder, tmp_path)
    assert len(calls) == 2
    assert recorder.data["reference"]["boxes"] == [20, 25, 80, 70]
    assert all("preview disk full" in view["caption"] for view in recorder.data["views"])
    assert all("image" not in view for view in recorder.data["views"])


def test_official_success_preserves_runtime_order_and_sdk_pairing(tmp_path, monkeypatch):
    recorder, calls = _official(tmp_path, monkeypatch)
    _run_official(recorder, tmp_path)
    assert len(calls) == 4
    assert calls[0][0].endswith("binary") and calls[1][0] == sys.executable
    assert calls[2][0].endswith("_sdk_c") and calls[3][0].endswith("_sdk_cpp")
    assert recorder.data["inputs"]["operation"] == "SDK C++"
    for role in ("native", "reference"):
        assert recorder.data[role]["boxes"] == [20, 25, 80, 70]
        assert recorder.data["detection_view_summary"][role]["operation"] == "SDK C++"
    assert all(v["title"].startswith("Sample 2 /") for v in recorder.data["views"])
    assert all(v["caption"].startswith("SDK C++ / source.png:") for v in recorder.data["views"])
    recorder.finish("passed")
    primary = render_case(recorder.data, recorder.directory).split(
        '<details class="case-details">', 1
    )[0]
    assert 'data-media-index="2" data-media-paired="true"' in primary
    assert 'data-media-index="0"' not in primary


def test_failed_second_sdk_clears_prior_results_and_previews(tmp_path, monkeypatch):
    error = subprocess.CalledProcessError(8, ["sdk_cpp"])
    recorder, calls = _official(tmp_path, monkeypatch, sdk_failure=error)
    with pytest.raises(subprocess.CalledProcessError) as raised:
        _run_official(recorder, tmp_path)
    assert raised.value is error and len(calls) == 4
    assert recorder.data["inputs"]["operation"] == "SDK C++"
    assert recorder.data["native"] is recorder.data["reference"] is None
    assert recorder.data["views"] == [] and recorder.data["detection_view_summary"] is None
    recorder.finish("failed")
    document = render_case(recorder.data, recorder.directory)
    primary = document.split('<details class="case-details">', 1)[0]
    assert 'data-media-role="native"' not in primary
    assert 'data-media-role="reference"' not in primary
    assert "data:image/png;base64," in document
