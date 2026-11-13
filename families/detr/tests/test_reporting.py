# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real PNG and HTML controls for the family-owned detection previews."""

from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from families.detr.tests import reporting
from tools import e2e_evidence
from tools.e2e_report import render_case


def _inputs():
    return (
        {"boxes": [20.0, 25.0, 80.0, 70.0], "scores": [0.9], "classes": [0]},
        [{"box": [20.0, 25.0, 80.0, 70.0], "score": 0.9, "label": 0}],
        {},
    )


def _record(tmp_path, native, reference, options, *, sample_index=0, recorder=None):
    image = tmp_path / "source.png"
    if not image.exists():
        Image.new("RGB", (120, 100), "white").save(image)
    recorder = recorder or e2e_evidence.Evidence(
        tmp_path / "evidence",
        family="detr",
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
    from families.detr.tests import test_e2e as e2e

    native, reference, options = _inputs()
    recorder = _record(tmp_path, native, reference, options)
    old_urls = ["data:image/png;base64," + base64.b64encode(
        (recorder.directory / view["image"]["artifact"]).read_bytes()).decode()
        for view in recorder.data["views"]]
    for language in ("c", "cpp"):
        (tmp_path / f"test_detr_sdk_{language}").touch()
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


@pytest.mark.parametrize("failed_stage", ["native", "reference", "compare"])
def test_variant_failures_keep_their_actual_stage_and_consumer_identity(tmp_path, monkeypatch, failed_stage):
    from pathlib import Path
    from families.detr.tests import test_e2e as e2e
    from tools.e2e_report import _assessment

    Image.new("RGB", (120, 100), "white").save(tmp_path / "source.png")
    case = {"name": "report", "test_image": "source.png"}
    manifest = {"name": "report", "family": "detr", "task": "image_to_boxes", "precision": "fp32",
                "bundle": "model.bundle", "hf_id": "example/model"}
    monkeypatch.setattr(e2e, "CASES", {"report": (tmp_path / "manifest.json", manifest, case)})
    monkeypatch.setattr(e2e, "TEST_ROOT", tmp_path)
    monkeypatch.setattr(e2e, "_model_dir", lambda *args: tmp_path)
    monkeypatch.setattr(e2e, "_runtime", lambda *args: (tmp_path / "binary", tmp_path))
    monkeypatch.setattr(e2e, "_build", lambda *args: None)
    monkeypatch.setattr(e2e, "_asset", lambda value: tmp_path / value)
    native, reference, _ = _inputs()
    native = {**native, "image_height": 100, "image_width": 120}
    calls, captions = [], []

    def run_native(binary, runtime, bundle, current):
        name = Path(current["test_image"]).name
        calls.append(("native", name))
        if name == "portrait.png" and failed_stage == "native":
            raise RuntimeError("portrait native failed")
        return {**native, "boxes": [100.0, 100.0, 120.0, 120.0]} if name == "portrait.png" and failed_stage == "compare" else deepcopy(native)

    def run_reference(model, current):
        name = Path(current["test_image"]).name
        calls.append(("reference", name))
        if name == "portrait.png" and failed_stage == "reference":
            raise RuntimeError("portrait reference failed")
        return deepcopy(reference)

    def sdk(*args):
        calls.append(("sdk", "source.png"))
        return [deepcopy(native), deepcopy(native)]

    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="detr", case="report",
                                    source_revision="a" * 40, roots=(tmp_path,))
    original_views = e2e.record_detection_views

    def views(*args, **kwargs):
        captions.append((kwargs.get("operation", "CLI"), recorder.data["inputs"].get("sdk_language")))
        original_views(*args, **kwargs)

    monkeypatch.setattr(e2e, "_native", run_native)
    monkeypatch.setattr(e2e, "_official_reference", run_reference)
    monkeypatch.setattr(e2e, "_sdk_detections", sdk)
    monkeypatch.setattr(e2e, "record_detection_views", views)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        with pytest.raises(AssertionError if failed_stage == "compare" else RuntimeError):
            e2e.test_official_checkpoint_e2e("report", tmp_path)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert captions[:3] == [("CLI", None), ("SDK C", "c"), ("SDK C++", "cpp")]
    assert calls[:3] == [("native", "source.png"), ("reference", "source.png"), ("sdk", "source.png")]
    assert calls.count(("native", "portrait.png")) == 1
    assert calls.count(("reference", "portrait.png")) == (0 if failed_stage == "native" else 1)
    assert recorder.data["failure_stage"] == failed_stage
    assert recorder.data["inputs"]["variant"] == "portrait"
    if failed_stage != "compare":
        assert recorder.data["detection_view_summary"] is None
    recorder.data["status"] = "failed"
    if failed_stage == "reference":
        assert _assessment(recorder.data)["label"] == "Reference run failed"
