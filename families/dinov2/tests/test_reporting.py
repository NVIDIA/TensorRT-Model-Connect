# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Display complete token features without changing their comparison semantics."""

from copy import deepcopy
import json

import numpy as np
import pytest

from families.dinov2.tests import test_e2e as e2e
from tools import e2e_evidence
from tools.e2e_report import _assess_reference_comparisons, render_case


@pytest.fixture
def recorded_checkpoint(tmp_path, monkeypatch):
    """Exercise the official function with full-sized CPU-only model outputs."""
    rows, columns, grid = 257, 768, 16
    reference = (np.arange(rows * columns, dtype=np.float64).reshape(rows, columns) + 1) / 1024
    # A nonzero error exposes both curves while remaining inside the existing gate.
    native = reference + 0.001
    payload = {
        "last_hidden_state_shape": [1, rows, columns],
        "last_hidden_state": native.ravel().tolist(),
        "pooler_output_shape": [1, columns],
        "pooler_output": native[0].tolist(),
        "pooling": "class_token",
        "normalization": "none",
        "tokens": [{"role": "class"}] + [
            {"role": "patch", "grid_row": row, "grid_column": column,
             "source_normalized_box": [column / grid, row / grid,
                                       (column + 1) / grid, (row + 1) / grid]}
            for row in range(grid) for column in range(grid)
        ],
    }
    original_payload = deepcopy(payload)
    calls = []
    monkeypatch.setattr(e2e, "_model_dir", lambda _: tmp_path)
    monkeypatch.setattr(e2e, "_runtime", lambda: (tmp_path / "trtmc", tmp_path))
    monkeypatch.setattr(e2e, "_build", lambda *args: calls.append("build"))

    def reference_call(model_dir, case, hidden_act=None):
        calls.append(hidden_act or "reference")
        return (reference + 0.1 if hidden_act else reference).tolist()

    def native_call(*args):
        calls.append("native")
        return payload

    monkeypatch.setattr(e2e, "_reference", reference_call)
    monkeypatch.setattr(e2e, "_run_json", native_call)
    recorders = []

    def run(precision):
        case = "dinov2-base" + ("-fp16" if precision == "fp16" else "")
        recorder = e2e_evidence.Evidence(
            tmp_path / "evidence" / "dinov2" / case,
            family="dinov2", case=case, source_revision="a" * 40,
            roots=(tmp_path, e2e.TEST_ROOT),
        )
        recorders.append(recorder)
        token = e2e_evidence._ACTIVE.set(recorder)
        try:
            e2e.test_official_checkpoint_e2e(case, tmp_path)
            recorder.finish("passed")
        finally:
            e2e_evidence._ACTIVE.reset(token)
        return recorder

    return run, payload, original_payload, native, reference, calls, recorders


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_full_checkpoint_features_are_paired_in_default_display(recorded_checkpoint, precision):
    run, payload, original, native, reference, calls, _ = recorded_checkpoint
    recorder = run(precision)
    data = json.loads((recorder.directory / "evidence.json").read_text())
    assert payload == original
    assert calls == ["build", "reference", *(["gelu_pytorch_tanh"] if precision == "fp16" else []), "native"]
    for role, expected in (("native", native), ("reference", reference)):
        tensor = data[role]["values"]
        assert tensor["shape"] == [257, 768]
        saved = np.load(recorder.directory / tensor["artifact"], allow_pickle=False)
        assert saved.size == 197376
        np.testing.assert_array_equal(saved, expected)
    raw = data["native"]["last_hidden_state"]
    np.testing.assert_array_equal(
        np.load(recorder.directory / raw["artifact"], allow_pickle=False), native.ravel()
    )
    assert data["native"]["last_hidden_state_shape"] == [1, 257, 768]
    assert data["native"]["normalization"] == "none"
    comparison = data["reference_comparison"]
    assert comparison["scope"] == "independent_reference" and comparison["enforced"] is True
    assert [check["name"] for check in comparison["checks"]] == ["cosine", "max_abs_error"]
    assert all(check["passed"] for check in comparison["checks"])
    top = render_case(data, recorder.directory).split('<details class="case-details">', 1)[0]
    assert "257 × 768" in top and "197376 values" in top
    assert '<polyline aria-label="Native"' in top
    assert '<polyline aria-label="Reference"' in top
    assert 'Last hidden state · Dtype' not in top


def test_truncated_native_preview_cannot_become_full_display_values(recorded_checkpoint):
    run, payload, _, _, _, calls, recorders = recorded_checkpoint
    payload["last_hidden_state"] = payload["last_hidden_state"][:32] + [{"omitted_items": 197344}]
    with pytest.raises(TypeError):
        run("fp32")
    data = recorders[-1].data
    assert "values" not in data["native"]
    assert data["native"]["last_hidden_state"][-1] == {"omitted_items": 197344}
    assert calls == ["build", "reference", "native"]
    assert not _assess_reference_comparisons(data)


def test_tail_feature_mismatch_still_fails_the_original_gate(recorded_checkpoint):
    run, payload, _, _, reference, calls, recorders = recorded_checkpoint
    payload["last_hidden_state"][-1] = reference[-1, -1] + 1.0
    with pytest.raises(AssertionError, match="features differ from the reference"):
        run("fp32")
    recorder = recorders[-1]
    saved = np.load(
        recorder.directory / recorder.data["native"]["values"]["artifact"], allow_pickle=False
    )
    assert saved[-1, -1] == payload["last_hidden_state"][-1]
    assert calls == ["build", "reference", "native"]
    assert not _assess_reference_comparisons(recorder.data)


def test_successful_tanh_control_does_not_add_an_independent_reference(tmp_path):
    native = [[1.0, 2.0]]
    reference = deepcopy(native)
    counterfactual = [[1.001, 2.001]]
    original = deepcopy((native, reference, counterfactual))
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="dinov2", case="counterfactual",
                                    source_revision="a" * 40, roots=(tmp_path,))
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        e2e._check_features(native, reference, "fp16", counterfactual)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert (native, reference, counterfactual) == original
    assert len(_assess_reference_comparisons(recorder.data)) == 1
    control = recorder.data["diagnostics"]["counterfactual_control"]
    assert control["tanh_reference"] == counterfactual
    assert control["actual"] == 0.0 and control["operator"] == "<"
    assert control["expected"] == e2e._TANH_SHARE_LIMIT and control["passed"] is True


def test_original_counterfactual_gate_still_rejects_the_defect(tmp_path):
    reference = [[1.0, 2.0]]
    counterfactual = [[1.001, 2.001]]
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="dinov2", case="counterfactual",
                                    source_revision="a" * 40, roots=(tmp_path,))
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        with pytest.raises(AssertionError, match="tanh-GELU defect"):
            e2e._check_features(counterfactual, reference, "fp16", counterfactual)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert not _assess_reference_comparisons(recorder.data)
