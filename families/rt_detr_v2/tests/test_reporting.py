# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real PNG and HTML controls for the family-owned detection previews."""

from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from families.rt_detr_v2.tests import reporting
from tools import e2e_evidence
from tools.e2e_report import render_case


@pytest.mark.parametrize("failure", ["iou", "score", "label"])
def test_failed_match_keeps_attempted_metrics_without_a_success_claim(tmp_path, failure):
    from families.rt_detr_v2.tests import test_e2e as e2e

    native, reference, _ = _inputs()
    native = {**native, "image_height": 100, "image_width": 120}
    reference = deepcopy(reference)
    if failure == "iou":
        reference[0]["box"] = [60.0, 25.0, 120.0, 70.0]
    elif failure == "score":
        reference[0]["score"] = 0.1
    else:
        reference[0]["label"] = 9
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="rt_detr_v2", case="failed-match",
                                    source_revision="a" * 40, roots=(tmp_path,))
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        with pytest.raises(AssertionError):
            e2e._assert_parity(native, reference)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    measured = recorder.data["diagnostics"]["attempted_match"]
    assert "passed" not in measured and "reference_comparison" not in recorder.data
    assert measured["min_box_iou"] == 0.5 and measured["max_score_error"] == 0.05
    if failure == "label":
        assert measured["reference"] is None and measured["score_error"] is None
    elif failure == "iou":
        assert measured["box_iou"] == 0.2
    else:
        assert measured["box_iou"] == 1.0
        assert measured["score_error"] == abs(0.9 - 0.1)


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
        family="rt_detr_v2",
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
