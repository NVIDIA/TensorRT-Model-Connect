# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real PNG and HTML controls for the family-owned detection previews."""

from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from families.yolov5.tests import reporting
from tools import e2e_evidence
from tools.e2e_report import render_case


def _inputs():
    return (
        {"boxes": [20.0, 25.0, 80.0, 70.0], "scores": [0.9], "classes": [0]},
        np.array([[50.0, 70.0, 170.0, 160.0, 0.9, 0.0]]),
        {"scale": 2.0, "pad_x": 10.0, "pad_y": 20.0},
    )


def _record(tmp_path, native, reference, options, *, sample_index=0, recorder=None):
    image = tmp_path / "source.png"
    if not image.exists():
        Image.new("RGB", (120, 100), "white").save(image)
    recorder = recorder or e2e_evidence.Evidence(
        tmp_path / "evidence",
        family="yolov5",
        case="detection",
        source_revision="a" * 40,
        roots=(tmp_path,),
    )
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        e2e_evidence.record_evidence("inputs", {"image": image})
        e2e_evidence.record_evidence("native", native)
        e2e_evidence.record_evidence("reference", reference)
        reporting.record_detection_views(
            native,
            reference,
            image,
            tmp_path / "views",
            sample_index=sample_index,
            **options,
        )
    finally:
        e2e_evidence._ACTIVE.reset(token)
    return recorder


def test_native_and_reference_boxes_are_visible_before_details(tmp_path):
    native, reference, options = _inputs()
    before_native, before_reference = deepcopy(native), deepcopy(reference)
    recorder = _record(tmp_path, native, reference, options)
    summary = recorder.data["detection_view_summary"]
    for role, color in reporting._COLORS.items():
        side = summary[role.lower()]
        assert side["source_size"] == [120, 100]
        assert side["detections"][0] == {
            "box": [20.0, 25.0, 80.0, 70.0],
            "label": str(0),
            "score": 0.9,
        }
        view = next(
            item for item in recorder.data["views"] if item["title"] == f"Sample 0 / {role}"
        )
        with Image.open(recorder.directory / view["image"]["artifact"]) as picture:
            assert picture.getpixel((20, 50)) == color
            assert picture.getpixel((50, 75)) == (255, 255, 255)
        assert str(0) + " (0.900)" in view["caption"]
    # A failed comparison still exposes its actual paired images.
    recorder.finish("failed", failure="original model comparison failed")
    document = render_case(recorder.data, recorder.directory)
    primary = document.split('<details class="case-details">', 1)[0]
    assert 'data-media-paired="true"' in primary
    assert 'data-media-role="native"' in primary
    assert 'data-media-role="reference"' in primary
    assert primary.count("data:image/png;base64,") >= 2
    assert native == before_native
    np.testing.assert_equal(reference, before_reference)


@pytest.mark.parametrize("different_output", [False, True])
def test_repeated_samples_keep_distinct_native_reference_pairs(tmp_path, different_output):
    import base64

    native, reference, options = _inputs()
    recorder = _record(tmp_path, native, reference, options)
    first = next(view for view in recorder.data["views"] if view["title"] == "Sample 0 / Native")
    first_path = recorder.directory / first["image"]["artifact"]
    first_bytes = first_path.read_bytes()
    second_native = deepcopy(native)
    if different_output:
        if isinstance(second_native, dict):
            second_native["boxes"][0] += 5.0
        else:
            second_native[0]["box"][0] += 5.0
    _record(tmp_path, second_native, reference, options, sample_index=1, recorder=recorder)
    second = next(view for view in recorder.data["views"] if view["title"] == "Sample 1 / Native")
    second_bytes = (recorder.directory / second["image"]["artifact"]).read_bytes()
    assert first_path.read_bytes() == first_bytes
    assert (first_bytes != second_bytes) is different_output
    recorder.finish("passed")
    document = render_case(recorder.data, recorder.directory)
    top = document.split('<details class="case-details">', 1)[0]
    assert 'data-media-index="0" data-media-paired="true"' in document
    assert 'data-media-index="1" data-media-paired="true"' in top
    assert 'data-media-index="0"' not in top
    if different_output:
        assert base64.b64encode(second_bytes).decode() in top
        assert base64.b64encode(first_bytes).decode() not in top


def test_previews_are_opt_in_and_do_not_require_valid_outputs(tmp_path):
    _, _, options = _inputs()
    reporting.record_detection_views(
        None, None, tmp_path / "absent.png", tmp_path / "views", **options
    )
    assert not (tmp_path / "views").exists()


def test_malformed_native_view_does_not_hide_valid_reference(tmp_path):
    _, reference, options = _inputs()
    recorder = _record(tmp_path, None, reference, options)
    summary = recorder.data["detection_view_summary"]
    assert "unavailable" in summary["native"]
    assert summary["reference"]["displayed"] == 1
    assert "image" not in recorder.data["views"][0]
    assert "image" in recorder.data["views"][1]


def test_preview_write_failure_does_not_raise_or_claim_images(tmp_path, monkeypatch):
    native, reference, options = _inputs()
    Image.new("RGB", (120, 100), "white").save(tmp_path / "source.png")

    def fail(*args, **kwargs):
        raise OSError("preview disk full")

    monkeypatch.setattr(Image.Image, "save", fail)
    recorder = _record(tmp_path, native, reference, options)
    assert all("preview disk full" in view["caption"] for view in recorder.data["views"])
    assert all("image" not in view for view in recorder.data["views"])


def test_reference_letterbox_coordinates_are_unscaled_without_clipping():
    kept = np.array([[0.0, 0.0, 30.0, 40.0, 0.75, 0.0]])
    before = kept.copy()
    rows, count = reporting._reference_rows(kept, scale=2.0, pad_x=10.0, pad_y=20.0)
    assert rows == [{"box": [-5.0, -10.0, 10.0, 10.0], "score": 0.75, "label": "0"}]
    assert count == 1
    np.testing.assert_array_equal(kept, before)


def test_preview_bound_is_explicit_and_preserves_full_inputs(tmp_path):
    native, reference, options = _inputs()
    if isinstance(native, list):
        native *= 40
    else:
        for key in native:
            native[key] *= 40
    reference = np.tile(reference, (40, 1)) if isinstance(reference, np.ndarray) else reference * 40
    before_native, before_reference = deepcopy(native), deepcopy(reference)
    recorder = _record(tmp_path, native, reference, options)
    for side in recorder.data["detection_view_summary"].values():
        assert side["displayed"] == 32
        assert side["total"] == 40
    assert all("first 32 of 40" in view["caption"] for view in recorder.data["views"])
    assert native == before_native
    np.testing.assert_equal(reference, before_reference)


def test_failed_second_sdk_clears_previous_overlays_and_results(tmp_path, monkeypatch):
    import base64
    import json
    from types import SimpleNamespace
    import pytest
    from families.yolov5.tests import test_e2e as e2e

    native, reference, options = _inputs()
    recorder = _record(tmp_path, native, reference, options)
    old_urls = ["data:image/png;base64," + base64.b64encode(
        (recorder.directory / view["image"]["artifact"]).read_bytes()).decode()
        for view in recorder.data["views"]]
    for language in ("c", "cpp"):
        (tmp_path / f"test_yolov5_sdk_{language}").touch()
    monkeypatch.setattr(e2e, "_required_path", lambda *args: tmp_path)
    result = {**native, "task": "image_to_boxes", "image_height": 100, "image_width": 120}
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0].endswith("_cpp"):
            raise RuntimeError("second SDK failed")
        return SimpleNamespace(stdout=json.dumps(result))

    monkeypatch.setattr(e2e.subprocess, "run", run)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        with pytest.raises(RuntimeError, match="second SDK failed"):
            e2e._sdk_detections(tmp_path, tmp_path / "model.bundle", tmp_path / "source.png", tmp_path)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert len(calls) == 2
    assert recorder.data["inputs"]["sdk_language"] == "cpp"
    assert recorder.data["native"] is recorder.data["reference"] is None
    assert recorder.data["views"] == []
    assert recorder.data["detection_view_summary"] is None
    assert any(row["name"] == "detection_view_summary" and row["value"] for row in recorder.data["observations"])
    recorder.data["status"] = "failed"
    document = render_case(recorder.data, recorder.directory)
    top = document.split('<details class="case-details">', 1)[0]
    for url in old_urls:
        assert url not in top and url in document


@pytest.mark.parametrize("operation", ["CLI", "SDK C", "SDK C++"])
def test_detection_captions_name_the_operation_and_source_image(tmp_path, operation):
    native, reference, options = _inputs()
    recorder = _record(tmp_path, native, reference, {**options, "operation": operation})
    for view in recorder.data["views"]:
        assert view["caption"].startswith(operation + " / source.png:")
    for role in ("native", "reference"):
        assert recorder.data["detection_view_summary"][role]["operation"] == operation


def test_outer_sdk_results_keep_the_actual_consumer_language(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    from families.yolov5.tests import test_e2e as e2e

    native, kept, options = _inputs()
    native = {**native, "task": "image_to_boxes", "image_height": 100, "image_width": 120}
    image = tmp_path / "source.png"
    Image.new("RGB", (120, 100), "white").save(image)
    manifest = {"name": "report", "family": "yolov5", "hf_id": "example/model", "task": "image_to_boxes",
                "bundle": "model.bundle", "precision": "fp32", "max_sequence_length": 1, "tensor_parallel_size": 1}
    monkeypatch.setattr(e2e, "CASES", {"report": (manifest, {"name": "report"})})
    monkeypatch.setattr(e2e, "_required_path", lambda value, label: tmp_path / "binary" if label == "TRTMC_BINARY" else tmp_path)
    monkeypatch.setattr(e2e, "_model_dir", lambda *args: tmp_path)
    monkeypatch.setattr(e2e, "_asset", lambda *args: image)
    monkeypatch.setattr(e2e, "build", lambda *args: None)
    for name in ("libtrtmc_backend_trt.so", "libtrtmc_model_yolov5.so", "test_yolov5_sdk_c", "test_yolov5_sdk_cpp"):
        (tmp_path / name).touch()
    calls, captions = [], []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == sys.executable:
            np.savez(command[-1], kept=kept, **options)
        return SimpleNamespace(stdout=json.dumps(native))

    monkeypatch.setattr(e2e.subprocess, "run", run)
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="yolov5", case="report",
                                    source_revision="a" * 40, roots=(tmp_path,))
    original_views = e2e.record_detection_views

    def views(*args, **kwargs):
        captions.append((kwargs.get("operation", "CLI"), recorder.data["inputs"].get("sdk_language")))
        original_views(*args, **kwargs)

    monkeypatch.setattr(e2e, "record_detection_views", views)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        e2e.test_official_checkpoint_e2e("report", tmp_path)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert len(calls) == 4
    assert calls[2][0].endswith("_sdk_c") and calls[3][0].endswith("_sdk_cpp")
    assert captions == [("CLI", None), ("SDK C", "c"), ("SDK C++", "cpp")]
    assert recorder.data["inputs"]["sdk_language"] == "cpp"
    assert all(view["caption"].startswith("SDK C++ / source.png:") for view in recorder.data["views"])
