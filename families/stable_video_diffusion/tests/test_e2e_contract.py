# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the E2E module's case selection."""

from __future__ import annotations

import pytest

from . import test_e2e as e2e


class _Config:
    def __init__(self, **options):
        self.options = options

    def getoption(self, name):
        return self.options.get(name)


class _Metafunc:
    fixturenames = ("case_name", "tmp_path")

    def __init__(self, **options):
        self.config = _Config(**options)
        self.parametrized = []

    def parametrize(self, name, values, ids):
        self.parametrized.append((name, list(ids)))


def _generate(**options):
    metafunc = _Metafunc(**options)
    e2e.pytest_generate_tests(metafunc)
    return metafunc.parametrized


def test_an_explicit_selection_fails_when_no_testcase_matches(tmp_path):
    models_file = tmp_path / "models.txt"
    models_file.write_text("stable_video_diffusion\n", encoding="utf-8")
    # The family, one of its manifests or one of its cases, each with a case it selects.
    selections = [
        ({"--e2e-model": ["stable_video_diffusion"]}, "svd-xt-fp32"),
        ({"--e2e-model": ["svd-xt"]}, "svd-xt-fp16"),
        ({"--e2e-model": ["svd-xt-fp16-l0"]}, "svd-xt-fp16-l0"),
        ({"--e2e-models-file": str(models_file)}, "svd-xt-fp32"),
    ]
    for selection, case in selections:
        # Pytest would skip the empty parametrization, and the run would pass without a test.
        with pytest.raises(pytest.UsageError,
                           match="its testcases are svd-xt-fp16, svd-xt-fp16-l0, svd-xt-fp32"):
            _generate(**selection, **{"--e2e-testcase": ["svd-xt-fp32-typo"]})
        assert _generate(**selection, **{"--e2e-testcase": [case]}) == [("case_name", [case])]
    # A selection of another family still skips this module.
    assert _generate(**{"--e2e-model": ["wavlm"],
                        "--e2e-testcase": ["wavlm-libri-clean-100h-large"]}) == [("case_name", [])]
