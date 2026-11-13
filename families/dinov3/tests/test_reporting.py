# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence must retain paired features when the existing comparison fails."""

from pathlib import Path

import numpy as np
import pytest

from families.dinov3.tests import test_e2e as e2e
from families.dinov3.tests.reporting import query_patch_maps
from families.dinov3 import validation
from tools import e2e_evidence
from tools.e2e_report import _assessment


def _features():
    hidden = np.asarray([[[1, 0], [1, 0], [0, 1], [1, 1], [1, -1]]], dtype=np.float64)
    return ({"last_hidden_state": hidden.copy(), "last_hidden_state_shape": hidden.shape,
             "pooler_output": hidden[:, 0].copy(), "pooler_output_shape": (1, 2)},
            {"last_hidden_state": hidden.copy(), "pooler_output": hidden[:, 0].copy(),
             "num_register_tokens": 0})


def test_query_maps_keep_inputs_unchanged_and_expose_local_difference():
    actual, expected = _features()
    original = expected["last_hidden_state"].copy()
    maps = query_patch_maps(actual, expected)
    assert all(np.max(item["absolute_error"]) == 0 for item in maps)
    np.testing.assert_array_equal(expected["last_hidden_state"], original)
    np.testing.assert_array_equal(actual["last_hidden_state"], original)
    actual["last_hidden_state"][0, 2] = [1, 0]
    different = query_patch_maps(actual, expected)
    assert max(float(np.max(item["absolute_error"])) for item in different) > 0
    assert all(item["native"].shape == item["reference"].shape == (2, 2) for item in different)


def test_failed_parity_retains_native_reference_and_thresholds(monkeypatch, tmp_path: Path):
    name = next(iter(e2e.CASES))
    actual, expected = _features()
    actual["last_hidden_state"][0, 2] = [-1, 0]
    thresholds = {"full_cosine": 0.999, "cls_cosine": 0.999, "pooler_cosine": 0.999,
                  "relative_frobenius": 0.001, "register_cosine": 0.999,
                  "mean_patch_cosine": 0.999, "p01_patch_cosine": 0.999}
    monkeypatch.setattr(e2e, "_model_dir", lambda manifest: tmp_path)
    monkeypatch.setattr(e2e, "_runtime", lambda manifest: (tmp_path / "trtmc", tmp_path))
    monkeypatch.setattr(e2e, "_build", lambda *args: None)
    monkeypatch.setattr(e2e, "_native", lambda *args: actual)
    monkeypatch.setattr(e2e, "_official_reference", lambda *args: expected)
    monkeypatch.setattr(e2e, "_thresholds", lambda case: thresholds)
    monkeypatch.setattr(e2e, "record_report_views", lambda *args: None)
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="dinov3", case=name,
                                   source_revision="a" * 40, roots=(tmp_path,))
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        with pytest.raises(AssertionError):
            e2e.test_official_checkpoint_e2e(name, tmp_path)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert recorder.data["failure_stage"] == "compare"
    assert recorder.data["native"]["last_hidden_state"]["shape"] == [1, 5, 2]
    assert recorder.data["reference"]["last_hidden_state"]["shape"] == [1, 5, 2]
    assert recorder.data["thresholds"] == thresholds
    assert [event["stage"] for event in recorder.data["timing"]] == ["build", "native", "reference", "compare"]
    assert "reference_comparison" not in recorder.data


@pytest.mark.parametrize("register_count", [0, 2])
def test_successful_original_validator_records_its_exact_metrics(monkeypatch, tmp_path, register_count):
    hidden = np.arange(1, (5 + register_count) * 3 + 1, dtype=np.float64).reshape(
        1, 5 + register_count, 3
    )
    actual = {
        "last_hidden_state": hidden.copy(), "last_hidden_state_shape": hidden.shape,
        "pooler_output": hidden[:, 0].copy(), "pooler_output_shape": (1, 3),
    }
    expected = {
        "last_hidden_state": hidden.copy(), "pooler_output": hidden[:, 0].copy(),
        "num_register_tokens": register_count,
    }
    actual["last_hidden_state"][0, -1, -1] += 1e-6
    original_values = actual["last_hidden_state"].copy()
    thresholds = {
        "full_cosine": 0.999, "cls_cosine": 0.999, "pooler_cosine": 0.999,
        "relative_frobenius": 0.001, "register_cosine": 0.999,
        "mean_patch_cosine": 0.999, "p01_patch_cosine": 0.999,
    }
    manifest = {"task": "image_features", "bundle": "model.bundle", "hf_id": "test/dinov3"}
    case = {"name": "recorded", "num_register_tokens": register_count}
    monkeypatch.setattr(e2e, "CASES", {"recorded": (tmp_path, manifest, case)})
    monkeypatch.setattr(e2e, "_model_dir", lambda *args: tmp_path)
    monkeypatch.setattr(e2e, "_runtime", lambda *args: (tmp_path, tmp_path))
    monkeypatch.setattr(e2e, "_build", lambda *args: None)
    monkeypatch.setattr(e2e, "_native", lambda *args: actual)
    monkeypatch.setattr(e2e, "_official_reference", lambda *args: expected)
    monkeypatch.setattr(e2e, "_thresholds", lambda *args: thresholds)
    observed = {}
    original_check = validation._assert_at_least

    def capture(name, value, limits, key):
        observed[key] = value
        original_check(name, value, limits, key)

    monkeypatch.setattr(validation, "_assert_at_least", capture)
    recorder = e2e_evidence.Evidence(
        tmp_path / "evidence", family="dinov3", case="recorded",
        source_revision="a" * 40, roots=(tmp_path,),
    )
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        e2e.test_official_checkpoint_e2e("recorded", tmp_path)
        recorder.finish("passed")
    finally:
        e2e_evidence._ACTIVE.reset(token)
    checks = {item["name"]: item for item in recorder.data["reference_comparison"]["checks"]}
    assert set(checks) == set(observed) | {"relative_frobenius"}
    for name, value in observed.items():
        assert checks[name]["actual"] == value
        assert checks[name]["expected"] == thresholds[name]
        assert checks[name]["operator"] == ">="
    relative = np.linalg.norm(original_values - hidden) / max(float(np.linalg.norm(hidden)), 1e-12)
    assert checks["relative_frobenius"]["actual"] == float(relative)
    assert checks["relative_frobenius"]["operator"] == "<="
    assert recorder.data["comparison_provenance"]["validator_completed"] is True
    assert _assessment(recorder.data)["kind"] == "reference"
    np.testing.assert_array_equal(actual["last_hidden_state"], original_values)
    np.testing.assert_array_equal(expected["last_hidden_state"], hidden)


def test_display_measurement_failure_does_not_replace_original_validation(tmp_path, monkeypatch):
    from families.dinov3.tests.reporting import record_validated_comparison

    actual, expected = _features()
    thresholds = {
        "full_cosine": 0.999, "cls_cosine": 0.999, "pooler_cosine": 0.999,
        "relative_frobenius": 0.001, "mean_patch_cosine": 0.999, "p01_patch_cosine": 0.999,
    }
    validation._assert_parity(actual, expected, {"task": "image_features"},
                              {"num_register_tokens": 0}, thresholds)
    recorder = e2e_evidence.Evidence(
        tmp_path / "evidence", family="dinov3", case="recorded",
        source_revision="a" * 40, roots=(tmp_path,),
    )
    def unavailable(*args):
        raise OSError("display metric unavailable")
    monkeypatch.setattr(validation, "_cosine", unavailable)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        record_validated_comparison(actual, expected, thresholds)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert "reference_comparison" not in recorder.data
    assert "display metric unavailable" in recorder.data["comparison_unavailable"]
