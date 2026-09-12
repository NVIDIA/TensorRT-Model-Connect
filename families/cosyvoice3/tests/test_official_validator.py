# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light tests for pinned-source reference selection."""

import subprocess

import pytest
import numpy as np

from families.cosyvoice3.config import SOURCE_REVISION
from families.cosyvoice3.tests.reference_helpers import (
    _ieee_fp32_reference,
    _official_dit,
    _cases,
    ATOL,
    INTEGRATED_ATOL,
    INTEGRATED_RTOL,
    RTOL,
)
from families.cosyvoice3.tests.reference_helpers import compare_outputs


def test_rejects_wrong_source_revision(tmp_path, monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="0" * 40 + "\n")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ValueError, match=SOURCE_REVISION):
        _official_dit(tmp_path)


@pytest.mark.parametrize("status", [" M cosyvoice/flow/DiT/dit.py", "?? cosyvoice/flow/new.py"])
def test_rejects_modified_pinned_source(tmp_path, monkeypatch, status):
    def fake_run(command, **kwargs):
        result = SOURCE_REVISION if "rev-parse" in command else status
        return subprocess.CompletedProcess(command, 0, stdout=result + "\n")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="local changes"):
        _official_dit(tmp_path)


def test_stress_suite_and_calibrated_thresholds_are_pinned():
    # Calibrated against the FP64 oracle on 2026-09-08; see reference_helpers.
    assert ATOL == RTOL == 2e-2
    assert INTEGRATED_ATOL == INTEGRATED_RTOL == 1e-1
    cases = _cases()
    assert [(n, m) for n, m, _ in cases] == [(n, m) for n in (4, 17, 64, 128) for m in (False, True)]
    for (_, _, first), (_, _, second) in zip(cases, _cases()):
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        np.testing.assert_array_equal(first["t"], np.array([0, .7], dtype=np.float32))


@pytest.mark.parametrize("actual,expected", [
    (np.ones((2, 1)), np.ones((2, 3))),  # np.allclose would broadcast.
    (np.array([[np.inf]]), np.array([[np.inf]])),  # allclose accepts equal infinities.
    (np.array([[np.nan]]), np.ones((1, 1))),
    (np.empty((2, 0)), np.empty((2, 0))),
])
def test_parity_rejects_invalid_outputs(actual, expected):
    with pytest.raises(ValueError):
        compare_outputs(actual, expected, atol=ATOL, rtol=RTOL)


def test_parity_reports_elementwise_failures_without_weakening_gate():
    expected = np.array([[0, 0, 10]], dtype=np.float32)
    # Ratios 3 and 0.5 of the calibrated gate: exactly one element fails.
    actual = np.array([[0, 3 * ATOL, 10 + .5 * (ATOL + 10 * RTOL)]], dtype=np.float32)
    result = compare_outputs(actual, expected, atol=ATOL, rtol=RTOL)
    assert result["passed"] is False
    assert result["elements_over_tolerance"] == 1
    assert result["total_elements"] == 3
    assert result["max_tolerance_ratio"] == pytest.approx(3)
    assert result["passed"] == bool(np.allclose(actual, expected, atol=ATOL, rtol=RTOL))


def test_supplemental_conditions_are_reproducible_and_labelled_separately():
    from families.cosyvoice3.tests.reference_helpers import _cfg_cases
    cases = list(_cfg_cases())
    assert len(cases) == 8
    for (frames, masked, values), (_, _, repeated) in zip(cases, _cfg_cases()):
        assert values["mu"].shape == (1, 80, frames)
        assert values["spks"].shape == (1, 80)
        assert values["mask"].shape == (1, 1, frames)
        assert not values["cond"][:, :, frames // 2:].any()
        assert bool((values["mask"] == 0).any()) == masked
        for key in values:
            np.testing.assert_array_equal(values[key], repeated[key])


def test_ieee_fp32_reference_restores_backend_settings():
    class Backend:
        allow_tf32 = True

    class FakeTorch:
        class backends:
            cudnn = Backend()

            class cuda:
                matmul = Backend()

        precision = "medium"

        @classmethod
        def get_float32_matmul_precision(cls):
            return cls.precision

        @classmethod
        def set_float32_matmul_precision(cls, value):
            cls.precision = value

    with _ieee_fp32_reference(FakeTorch):
        assert FakeTorch.backends.cudnn.allow_tf32 is False
        assert FakeTorch.backends.cuda.matmul.allow_tf32 is False
        assert FakeTorch.precision == "highest"
    assert FakeTorch.backends.cudnn.allow_tf32 is True
    assert FakeTorch.backends.cuda.matmul.allow_tf32 is True
    assert FakeTorch.precision == "medium"
