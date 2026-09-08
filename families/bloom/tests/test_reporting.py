# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token diagnostics describe the first difference without judging correctness."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from families.bloom.tests import test_e2e as e2e
from tools import e2e_evidence


@pytest.mark.parametrize("native,reference,index,native_token,reference_token", [
    ([1, 2], [1, 3], 1, 2, 3),
    ([1], [1, 3], 1, None, 3),
    ([1, 2], [1], 1, 2, None),
])
def test_first_token_difference_includes_early_termination(monkeypatch, native, reference, index, native_token, reference_token):
    observations = []
    monkeypatch.setattr(e2e, "record_evidence", lambda name, value: observations.append(value))
    e2e._record_text_diagnostics(native, reference)
    assert observations[0]["matching_prefix_tokens"] == index
    assert observations[0]["first_difference"] == {"index": index, "native_token_id": native_token, "reference_token_id": reference_token}


def test_text_failure_keeps_named_reference_and_effective_default(monkeypatch, tmp_path: Path):
    name = "report-text"
    manifest = {"bundle": "model.bundle", "tensor_parallel_size": 1, "hf_id": "test/model"}
    case = {"max_new_tokens": 2, "prompt": "prompt", "top_k": 1}
    monkeypatch.setattr(e2e, "_CASES", {name: (manifest, case)})
    monkeypatch.setattr(e2e, "_require_selected", lambda *args: None)
    monkeypatch.setattr(e2e, "_required_environment", lambda *args: (tmp_path / "trtmc", tmp_path, None))
    monkeypatch.setattr(e2e, "_checkpoint", lambda *args: tmp_path)
    monkeypatch.setattr(e2e, "_prompt", lambda *args: "prompt")
    monkeypatch.setattr(e2e, "_build_bundle", lambda *args: None)
    monkeypatch.setattr(e2e, "_assert_rank_sections", lambda *args: None)
    monkeypatch.setattr(e2e, "_run_native", lambda *args: {"token_ids": [1, 2], "text": "wrong"})
    monkeypatch.setattr(e2e, "_hf_reference", lambda *args: ([1, 3], "right", None, "wrong"))
    monkeypatch.setattr(e2e, "_thresholds", lambda *args: {})
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="bloom", case=name,
                                   source_revision="a" * 40, roots=(tmp_path,))
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        with pytest.raises(AssertionError):
            e2e.test_e2e(name, SimpleNamespace(config=None), tmp_path)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert recorder.data["reference"]["reference_text"] == "right"
    assert recorder.data["diagnostics"]["first_difference"]["index"] == 1
    assert recorder.data["thresholds"] == {"contract_ned_threshold": 0.25}
    assert recorder.data["failure_stage"] == "compare"
