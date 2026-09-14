# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence must retain paired features when the existing comparison fails."""

from pathlib import Path

import numpy as np
import pytest

from families.dinov3.tests import test_e2e as e2e
from families.dinov3.tests.reporting import query_patch_maps
from tools import e2e_evidence


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
