# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real PNG and HTML controls for the family-owned detection previews."""

from copy import deepcopy

import numpy as np
from PIL import Image

from families.yolos.tests import reporting
from tools import e2e_evidence
from tools.e2e_report import render_case


def _inputs():
    return (
        [{"box": [20.0, 25.0, 80.0, 70.0], "score": 0.9, "label": 0}],
        [{"box": [20.0, 25.0, 80.0, 70.0], "score": 0.9, "label": 0}],
        {},
    )


def _record(tmp_path, native, reference, options, *, sample_index=0, recorder=None):
    image = tmp_path / "source.png"
    if not image.exists():
        Image.new("RGB", (120, 100), "white").save(image)
    recorder = recorder or e2e_evidence.Evidence(
        tmp_path / "evidence",
        family="yolos",
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


def test_repeated_samples_keep_distinct_native_reference_pairs(tmp_path):
    native, reference, options = _inputs()
    recorder = _record(tmp_path, native, reference, options)
    _record(tmp_path, native, reference, options, sample_index=1, recorder=recorder)
    recorder.finish("passed")
    document = render_case(recorder.data, recorder.directory)
    assert 'data-media-index="0" data-media-paired="true"' in document
    assert 'data-media-index="1" data-media-paired="true"' in document


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
