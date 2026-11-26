# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the E2E module's case selection and frame comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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


class _Item:
    """A collected test item with the given parameters."""

    def __init__(self, originalname="test_e2e", **params):
        self.originalname = originalname
        self.callspec = type("Callspec", (), {"params": params})()


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


def test_a_testcase_only_selection_matching_no_case_fails_at_run_time():
    # Without --e2e-model another family of the session may own the testcase: one item decides at
    # run time instead of an empty parametrization that pytest would skip.
    assert _generate(**{"--e2e-testcase": ["svd-xt-fp32-typo"]}) == [
        ("case_name", [e2e.UNMATCHED_TESTCASES])]
    assert _generate(**{"--e2e-testcase": ["svd-xt-fp32"]}) == [("case_name", ["svd-xt-fp32"])]
    session = [_Item(case_name=e2e.UNMATCHED_TESTCASES),
               _Item(case_name="wavlm-libri-clean-100h-large"),
               _Item(manifest={}, case={"name": "clef-flash-invoice"})]
    typo = _Config(**{"--e2e-testcase": ["svd-xt-fp32-typo"]})
    with pytest.raises(pytest.fail.Exception, match="matches no E2E case collected in this"):
        e2e._require_collected_testcases(typo, session)
    # The placeholder item does not count as a case.
    placeholder = _Config(**{"--e2e-testcase": [e2e.UNMATCHED_TESTCASES]})
    with pytest.raises(pytest.fail.Exception, match="matches no E2E case collected in this"):
        e2e._require_collected_testcases(placeholder, session)
    # Another family of the session owns the testcase, as a case name or a case's name: nothing
    # to run here.
    for owned_case in ("wavlm-libri-clean-100h-large", "clef-flash-invoice"):
        owned = _Config(**{"--e2e-testcase": [owned_case]})
        with pytest.raises(pytest.skip.Exception, match="no stable_video_diffusion testcase"):
            e2e._require_collected_testcases(owned, session)
    # A test that is not an E2E does not own a case.
    with pytest.raises(pytest.fail.Exception):
        e2e._require_collected_testcases(
            owned, [_Item(originalname="test_support", case={"name": "clef-flash-invoice"})])


def test_the_official_run_restores_the_tf32_settings(monkeypatch):
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    during = []

    class _Refused:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            during.append((torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32))
            raise RuntimeError("no checkpoint in this test")

    monkeypatch.setattr(diffusers, "StableVideoDiffusionPipeline", _Refused)
    _, manifest, case = e2e.CASES["svd-xt-fp32"]
    saved = torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32
    try:
        for matmul in ("high", "medium"):
            torch.set_float32_matmul_precision(matmul)
            torch.backends.cudnn.allow_tf32 = True
            with pytest.raises(RuntimeError, match="no checkpoint"):
                e2e._official(Path("unused"), manifest, case, Path("unused.png"), "fp32")
            assert torch.get_float32_matmul_precision() == matmul
            assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.set_float32_matmul_precision(saved[0])
        torch.backends.cudnn.allow_tf32 = saved[1]
    assert during == [("highest", False), ("highest", False)]


def test_frame_comparisons_refuse_clips_of_different_shapes():
    clip = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    # A truncated reference would be compared on its frames only.
    with pytest.raises(AssertionError, match="clip shapes differ"):
        e2e._frame_psnrs(clip, clip[:2])
    # One row would be broadcast over the whole image.
    with pytest.raises(AssertionError, match="image shapes differ"):
        e2e._psnr(clip[0], clip[0, :1])
    assert e2e._frame_psnrs(clip, clip.copy()) == [99.0, 99.0, 99.0]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_quantization_refuses_nonfinite_frames(value):
    frames = np.full((1, 2, 2, 3), 0.5, dtype=np.float32)
    frames[0, 0, 0, 0] = value
    with pytest.raises(AssertionError, match="finite"):
        e2e._quantize(frames)
