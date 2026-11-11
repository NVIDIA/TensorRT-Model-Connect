# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Negative controls must fail even when an incorrect observation appears plausible."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from families.lighton_ocr.tests.oracle import compare_generation, compare_observations
from families.lighton_ocr.tests import test_e2e as e2e


CASE_NAME = next(iter(e2e.CASES))
LIMITS = json.loads((Path(__file__).parent / "thresholds" / (CASE_NAME + ".json")).read_text())["threshold_overrides"]


def _arrays():
    first = np.arange(16, dtype=np.float32).reshape(1, 16) / 16
    second = first.copy()
    first[0, 5], second[0, 6] = 3, 3
    return {
        "pixel_values": np.linspace(-0.5, 0.5, 288, dtype=np.float32).reshape(1, 3, 8, 12),
        "image_features": np.linspace(-0.5, 0.5, 24, dtype=np.float32).reshape(6, 4),
        "step_0_logits": first,
        "step_1_logits": second,
    }


def _write(directory, arrays):
    directory.mkdir()
    metadata = {
        "precision": "fp32", "prompt_ids": [1, *([3] * 6), 2], "cache_prefix_unchanged": True,
        "checkpoint_config": {"vision_config": {"patch_size": 2}, "spatial_merge_size": 2,
                              "image_token_id": 3, "text_config": {"hidden_size": 4, "vocab_size": 16}},
        "tensors": {},
    }
    for name, array in arrays.items():
        filename = name + ".f32"
        array.tofile(directory / filename)
        metadata["tensors"][name] = {"file": filename, "shape": list(array.shape), "dtype": "float32"}
    (directory / "metadata.json").write_text(json.dumps(metadata))


def _pair(tmp_path, expected=None, actual=None):
    expected = _arrays() if expected is None else expected
    native, reference = tmp_path / "native", tmp_path / "reference"
    _write(reference, expected)
    _write(native, expected if actual is None else actual)
    return native, reference


def _metadata(directory, mutate):
    path = directory / "metadata.json"
    metadata = json.loads(path.read_text())
    mutate(metadata)
    path.write_text(json.dumps(metadata))


def _generation():
    return {"text": "LIGHTON OCR 1042 Total 30.00", "token_ids": [5, 6, 2],
            "eos_token_ids": [2], "reached_eos": True, "prefill_ms": 1.0, "decode_ms": 2.0}


def _compare_generation(actual, expected):
    compare_generation(actual, expected, LIMITS, max_new_tokens=1024,
                       expected_fragments=["LIGHTON OCR", "1042", "30.00"], vocab_size=16)


def test_exact_whole_observations_and_generation_pass(tmp_path):
    native, reference = _pair(tmp_path)
    metrics = compare_observations(native, reference, LIMITS)
    assert len(metrics) == 4
    assert all(value["relative_l2"] == value["maximum_absolute_error"] == 0 for value in metrics.values())
    assert metrics["step_0_logits"]["argmax_agreement"] == metrics["step_1_logits"]["argmax_agreement"] == 1
    _compare_generation(_generation(), _generation())


@pytest.mark.parametrize("mutation", ["shape", "matching_wrong_shape", "dtype", "missing", "byte_count"])
def test_incomplete_or_misdescribed_observations_fail(tmp_path, mutation):
    native, reference = _pair(tmp_path)
    if mutation in {"shape", "matching_wrong_shape"}:
        def change(value):
            value["tensors"]["image_features"]["shape"] = [4, 6]
        _metadata(native, change)
        if mutation == "matching_wrong_shape":
            _metadata(reference, change)
    elif mutation == "dtype":
        _metadata(native, lambda value: value["tensors"]["pixel_values"].update(dtype="float64"))
    elif mutation == "missing":
        _metadata(native, lambda value: value["tensors"].pop("step_1_logits"))
    else:
        path = native / "image_features.f32"
        path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(AssertionError):
        compare_observations(native, reference, LIMITS)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf])
def test_matching_nonfinite_values_never_establish_parity(tmp_path, nonfinite):
    arrays = _arrays()
    arrays["pixel_values"][0, 0, 0, 0] = nonfinite
    native, reference = _pair(tmp_path, arrays)
    with pytest.raises(AssertionError, match="finite"):
        compare_observations(native, reference, LIMITS)


@pytest.mark.parametrize("gate", ["relative", "maximum", "argmax"])
def test_logit_value_and_choice_gates_are_independent(tmp_path, gate):
    expected = _arrays()
    if gate == "maximum":
        expected["step_0_logits"][:] = 1000
        expected["step_0_logits"][0, 5] = 3000
    elif gate == "argmax":
        expected["step_0_logits"][0, 6] = np.nextafter(np.float32(3), np.float32(0))
    actual = deepcopy(expected)
    if gate == "relative":
        actual["step_0_logits"] *= np.float32(1.0002)
        assert np.max(np.abs(actual["step_0_logits"] - expected["step_0_logits"])) < LIMITS["logits_maximum_absolute_error"]
    elif gate == "maximum":
        actual["step_0_logits"][0, 0] += np.float32(0.02)
        assert np.linalg.norm(actual["step_0_logits"] - expected["step_0_logits"]) / np.linalg.norm(expected["step_0_logits"]) < LIMITS["logits_relative_l2"]
    else:
        actual["step_0_logits"][0, [5, 6]] = expected["step_0_logits"][0, [6, 5]]
        assert np.max(np.abs(actual["step_0_logits"] - expected["step_0_logits"])) < LIMITS["logits_maximum_absolute_error"]
    native, reference = _pair(tmp_path, expected, actual)
    with pytest.raises(AssertionError, match="step_0_logits"):
        compare_observations(native, reference, LIMITS)


@pytest.mark.parametrize("mutation", ["prompt", "cache"])
def test_prompt_and_cache_prefix_are_part_of_the_numerical_contract(tmp_path, mutation):
    native, reference = _pair(tmp_path)
    if mutation == "prompt":
        _metadata(native, lambda value: value["prompt_ids"].__setitem__(0, 0))
    else:
        _metadata(native, lambda value: value.update(cache_prefix_unchanged=False))
    with pytest.raises(AssertionError):
        compare_observations(native, reference, LIMITS)


@pytest.mark.parametrize("mutation", ["ids", "text", "empty", "missing_eos", "false_eos_claim", "missing_fragment", "invalid_id"])
def test_ocr_semantics_and_exact_output_cannot_be_replaced_by_partial_matches(mutation):
    expected = _generation()
    actual = deepcopy(expected)
    if mutation == "ids":
        actual["token_ids"][1] = 7
    elif mutation == "text":
        actual["text"] += "\n"
    elif mutation == "empty":
        actual.update(text="", token_ids=[2])
        expected = deepcopy(actual)
    elif mutation in {"missing_eos", "false_eos_claim"}:
        actual["token_ids"][-1] = 7
        actual["reached_eos"] = mutation == "false_eos_claim"
        expected = deepcopy(actual)
    elif mutation == "missing_fragment":
        actual["text"] = "Some other document"
        expected = deepcopy(actual)
    else:
        actual["token_ids"][0] = 16
        expected = deepcopy(actual)
    with pytest.raises(AssertionError):
        _compare_generation(actual, expected)


@pytest.mark.parametrize("mutation", ["missing_timing", "nonfinite_timing"])
def test_native_timing_fields_must_be_present_and_finite(mutation):
    actual = _generation()
    if mutation == "missing_timing":
        del actual["prefill_ms"]
    else:
        actual["decode_ms"] = float("nan")
    with pytest.raises(AssertionError):
        _compare_generation(actual, _generation())


class Selection:
    def __init__(self, **values):
        self.values = values

    def getoption(self, name, default=None):
        return self.values.get(name, default)


@pytest.mark.parametrize("selector", ["--e2e-model", "--e2e-testcase", "--e2e-models-file"])
def test_all_three_selectors_enable_the_exact_case(tmp_path, selector):
    if selector == "--e2e-models-file":
        path = tmp_path / "models.txt"
        path.write_text("# Exact checkpoint case\n" + CASE_NAME + "\n")
        config = Selection(**{selector: str(path)})
    else:
        value = e2e.FAMILY if selector == "--e2e-model" else CASE_NAME
        config = Selection(**{selector: [value]})
    e2e._require_selected(CASE_NAME, config)


@pytest.mark.parametrize("selector", ["--e2e-model", "--e2e-testcase", "--e2e-models-file"])
def test_unknown_lighton_selection_fails_instead_of_becoming_a_skip(tmp_path, selector):
    value = "lightonocr-unknown-case"
    if selector == "--e2e-models-file":
        path = tmp_path / "models.txt"
        path.write_text(value + "\n")
        config = Selection(**{selector: str(path)})
    else:
        config = Selection(**{selector: [value]})
    with pytest.raises(AssertionError, match="Unknown LightOnOCR"):
        e2e._require_selected(CASE_NAME, config)


@pytest.mark.parametrize("use_models_file", [False, True])
def test_matching_model_does_not_override_an_invalid_testcase(tmp_path, use_models_file):
    options = {"--e2e-testcase": ["wrong"]}
    if use_models_file:
        path = tmp_path / "models.txt"
        path.write_text(e2e.FAMILY + "\n")
        options["--e2e-models-file"] = str(path)
    else:
        options["--e2e-model"] = [e2e.FAMILY]
    with pytest.raises(AssertionError, match="Unknown LightOnOCR testcase"):
        e2e._require_selected(CASE_NAME, Selection(**options))


def test_matching_testcase_does_not_override_a_different_model():
    config = Selection(**{"--e2e-model": ["qwen"], "--e2e-testcase": [CASE_NAME]})
    with pytest.raises(pytest.skip.Exception):
        e2e._require_selected(CASE_NAME, config)


def test_unselected_e2e_does_not_require_models_or_gpu(monkeypatch):
    monkeypatch.delenv("TRTMC_E2E", raising=False)
    with pytest.raises(pytest.skip.Exception):
        e2e._require_selected(CASE_NAME, Selection())
    with pytest.raises(pytest.skip.Exception):
        e2e._require_selected(CASE_NAME, Selection(**{"--e2e-model": ["qwen"]}))


def test_selected_e2e_rejects_missing_required_artifact(monkeypatch):
    monkeypatch.delenv("TRTMC_LIGHTON_OCR_MODEL_DIR", raising=False)
    with pytest.raises(AssertionError, match="TRTMC_LIGHTON_OCR_MODEL_DIR"):
        e2e._path("TRTMC_LIGHTON_OCR_MODEL_DIR", directory=True)


def test_failed_command_retains_logs_without_exception_context_or_environment_values(tmp_path, monkeypatch):
    recorded = []
    monkeypatch.setattr(e2e, "record_evidence", lambda name, value: recorded.append((name, value)))
    output = tmp_path / "failed-command.log"
    program = "import sys; print('before failure'); print('actual failure', file=sys.stderr); sys.exit(3)"
    marker = "FAKE_ONLY_LIGHTON_ENVIRONMENT_SECRET_82741"
    environment = {"PATH": os.defpath, "LIGHTON_TEST_SECRET": marker}
    with pytest.raises(pytest.fail.Exception) as failure:
        e2e._run([sys.executable, "-c", program], output, environment)
    assert "exited 3" in str(failure.value)
    assert failure.value.pytrace is False
    assert failure.value.__context__ is None and failure.value.__cause__ is None
    assert marker not in str(failure.value) + repr(recorded)
    assert output.read_text().strip() == "before failure"
    assert recorded[0][1]["stderr"].read_text().strip() == "actual failure"


@pytest.mark.parametrize("kind", ["timeout", "os_error"])
def test_other_process_failures_do_not_retain_sensitive_exception_details(tmp_path, monkeypatch, kind):
    marker = "FAKE_ONLY_LIGHTON_ENVIRONMENT_SECRET_82741"
    def fail_process(command, **_kwargs):
        if kind == "timeout":
            raise subprocess.TimeoutExpired(command, 1, output=marker, stderr=marker)
        raise FileNotFoundError(2, marker, marker)
    monkeypatch.setattr(e2e.subprocess, "run", fail_process)
    with pytest.raises(pytest.fail.Exception) as failure:
        e2e._run([sys.executable, "-c", "pass"], tmp_path / "failed.log", {"LIGHTON_TEST_SECRET": marker})
    assert failure.value.pytrace is False
    assert failure.value.__context__ is None and failure.value.__cause__ is None
    assert marker not in str(failure.value)
