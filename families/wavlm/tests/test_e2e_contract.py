# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the E2E's own rules: case selection, parity inputs and evidence."""

from __future__ import annotations

import pytest

from tools import e2e_evidence

from . import test_e2e as e2e

_LARGE = "wavlm-libri-clean-100h-large"
_LARGE_FP16 = "wavlm-libri-clean-100h-large-fp16"
_BASE_PLUS = "wavlm-libri-clean-100h-base-plus"
_CASE = {"test_input_audio": "data/librispeech-test-clean-6930-75918-0003.wav"}


class _Config:
    def __init__(self, **options):
        self.options = options

    def getoption(self, name):
        return self.options.get(name)


class _Metafunc:
    fixturenames = ("tmp_path", "case_name")

    def __init__(self, **options):
        self.config = _Config(**options)
        self.parametrized = []

    def parametrize(self, name, values, ids):
        self.parametrized.append((name, ids))


def _generate(**options):
    metafunc = _Metafunc(**options)
    e2e.pytest_generate_tests(metafunc)
    return metafunc.parametrized


def test_the_model_dir_override_is_refused_for_several_checkpoints(monkeypatch, tmp_path):
    # --e2e-model wavlm selects both checkpoints; one directory cannot stand for both,
    # or the base-plus case would build and pass on the large checkpoint.
    monkeypatch.setenv(e2e.MODEL_DIR_OVERRIDE, str(tmp_path))
    with pytest.raises(pytest.UsageError, match="patrickvonplaten/wavlm-libri-clean-100h-base-plus"):
        _generate(**{"--e2e-model": ["wavlm"]})


def test_the_model_dir_override_serves_the_cases_of_one_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv(e2e.MODEL_DIR_OVERRIDE, str(tmp_path))
    selected = _generate(**{"--e2e-testcase": [f"{_LARGE},{_LARGE_FP16}"]})
    assert selected == [("case_name", [_LARGE, _LARGE_FP16])]


def test_every_checkpoint_resolves_by_itself_without_the_override(monkeypatch):
    monkeypatch.delenv(e2e.MODEL_DIR_OVERRIDE, raising=False)
    selected = _generate(**{"--e2e-model": ["wavlm"]})
    assert selected == [("case_name", [_BASE_PLUS, _LARGE, _LARGE_FP16])]


@pytest.mark.parametrize("ids", [[1.9, 2], [1.0, 2.0], [True, 2], "12", (1, 2), None])
def test_token_ids_are_json_integers_only(ids):
    with pytest.raises(AssertionError, match="list of integers"):
        e2e._token_ids(ids, "native")


def test_the_parity_gate_does_not_truncate_token_ids():
    # int() would turn 1.9 into 1 and pass the exact fp32 gate on matching text.
    native = {"text": "a b", "token_ids": [1.9, 2]}
    with pytest.raises(AssertionError, match="native token_ids"):
        e2e._assert_parity(native, "a b", [1, 2], {})
    e2e._assert_parity({"text": "a b", "token_ids": [1, 2]}, "a b", [1, 2], {})


def _record(tmp_path, native, reference_text, reference_ids, thresholds):
    tmp_path.mkdir(parents=True, exist_ok=True)
    recorder = e2e_evidence.Evidence(
        tmp_path / "evidence", family=e2e.FAMILY, case="contract-check",
        source_revision="a" * 40, roots=(tmp_path,),
    )
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        e2e._record_reference_comparison(
            native, reference_text, reference_ids, thresholds, _CASE, tmp_path)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    return recorder.data


@pytest.mark.parametrize("thresholds", [{}, {"max_character_error_rate": 0.01}])
def test_a_blank_reference_records_the_failed_guard_and_nothing_after_it(tmp_path, thresholds):
    # The gate stops at a blank reference; the evidence shows that check failing
    # instead of an error from the character error rate's division.
    data = _record(tmp_path, {"text": "", "token_ids": []}, "", [], thresholds)
    assert "comparison_preview" not in data
    checks = data["reference_comparison"]["checks"]
    assert [(check["name"], check["actual"], check["passed"]) for check in checks] == [
        ("reference_characters", 0, False)]


def test_the_recorded_checks_follow_the_parity_gate(tmp_path):
    exact = _record(tmp_path, {"text": "a b", "token_ids": [1, 0, 2]}, "a b", [1, 0, 2], {})
    assert [(check["name"], check["actual"], check["operator"], check["expected"], check["passed"])
            for check in exact["reference_comparison"]["checks"]] == [
        ("reference_characters", 3, ">=", 1, True),
        ("transcript_character_edits", 0, "==", 0, True),
        ("token_id_mismatches", 0, "==", 0, True),
    ]
    bounded = _record(tmp_path / "fp16", {"text": "a c", "token_ids": [1, 0, 3]}, "a b",
                      [1, 0, 2], {"max_character_error_rate": 0.5})
    rate = bounded["reference_comparison"]["checks"][1]
    assert (rate["name"], rate["actual"], rate["expected"], rate["passed"]) == (
        "character_error_rate", 1 / 3, 0.5, True)


def test_the_report_shows_the_input_with_the_text_its_speaker_read(tmp_path):
    view = e2e._input_view(e2e.TEST_ROOT / _CASE["test_input_audio"])
    assert view["title"] == "Input audio" and view["audio"].is_file()
    assert view["caption"].startswith("Read text (dataset transcript): FROM THE RESPECT PAID HER")
    # A clip without a stored transcript is shown without a caption.
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"")
    assert "caption" not in e2e._input_view(clip)
