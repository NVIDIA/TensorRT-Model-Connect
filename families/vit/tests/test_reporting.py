# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Counterfactual activation controls stay diagnostics, not independent references."""

from copy import deepcopy

import pytest

from families.vit.tests import test_e2e as e2e
from tools import e2e_evidence
from tools.e2e_report import _assess_reference_comparisons, _assessment


def test_successful_tanh_control_does_not_add_an_independent_reference(tmp_path):
    native = [1.0, 2.0]
    reference = deepcopy(native)
    counterfactual = [1.001, 2.001]
    original = deepcopy((native, reference, counterfactual))
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="vit", case="counterfactual",
                                    source_revision="a" * 40, roots=(tmp_path,))
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        e2e._check_logits(native, reference, "fp16", counterfactual)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert (native, reference, counterfactual) == original
    assert len(_assess_reference_comparisons(recorder.data)) == 1
    control = recorder.data["diagnostics"]["counterfactual_control"]
    assert control["tanh_reference"] == counterfactual
    assert control["actual"] == 0.0 and control["operator"] == "<"
    assert control["expected"] == e2e._TANH_SHARE_LIMIT and control["passed"] is True


def test_original_counterfactual_gate_still_rejects_the_defect(tmp_path):
    reference = [1.0, 2.0]
    counterfactual = [1.001, 2.001]
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="vit", case="counterfactual",
                                    source_revision="a" * 40, roots=(tmp_path,))
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        with pytest.raises(AssertionError, match="tanh-GELU defect"):
            e2e._check_logits(counterfactual, reference, "fp16", counterfactual)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    comparisons = _assess_reference_comparisons(recorder.data)
    assert len(comparisons) == 1
    assert all(check["passed"] for check in comparisons[0]["checks"])
    control = recorder.data["diagnostics"]["counterfactual_control"]
    assert control["actual"] == 1.0 and control["passed"] is False
    assert control["expected"] == e2e._TANH_SHARE_LIMIT
    assert recorder.data["thresholds"]["max_tanh_defect_share"] == e2e._TANH_SHARE_LIMIT
    assert control["native"]["scores"] == counterfactual
    assert control["reference"]["scores"] == reference
    assert _assessment({**recorder.data, "status": "failed"})["kind"] == "failed"


def test_recording_error_does_not_replace_the_original_counterfactual_failure(tmp_path, monkeypatch):
    reference = [1.0, 2.0]
    counterfactual = [1.001, 2.001]
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="vit", case="counterfactual",
                                    source_revision="a" * 40, roots=(tmp_path,))

    def unavailable(*args):
        raise OSError("report storage unavailable")

    monkeypatch.setattr(recorder, "record", unavailable)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        with pytest.raises(AssertionError, match="tanh-GELU defect"):
            e2e._check_logits(counterfactual, reference, "fp16", counterfactual)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert recorder.data["issues"]
