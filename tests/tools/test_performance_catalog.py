# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from tools.performance import catalog as performance_catalog


REPOSITORY = Path(__file__).resolve().parents[2]
SUITE = REPOSITORY / "benchmarks" / "performance" / "release.yaml"


def test_release_suite_loads_and_selects_models_in_request_order() -> None:
    suite = performance_catalog.load_suite(SUITE)

    selected = suite.select(models=["distilgpt2", "gpt2-125m"])

    assert [case["model"] for case in selected] == ["distilgpt2", "gpt2-125m"]
    assert len(suite.cases) == 111


def test_release_suite_includes_fast_foundation_stereo() -> None:
    suite = performance_catalog.load_suite(SUITE)

    assert "fast-foundation-stereo" not in suite.excluded_profiles
    case = next(case for case in suite.cases if case["model"] == "fast-foundation-stereo")
    assert case["id"] == "fast_foundation_stereo.disparity"


def test_release_suite_uses_model_owned_dinov3_parity_thresholds() -> None:
    suite = performance_catalog.load_suite(SUITE)
    cases = {case["model"]: case for case in suite.cases if case["family"] == "dinov3"}

    assert cases["dinov3-vits16-pretrain-lvd1689m"]["baseline"][
        "max_image_feature_relative_frobenius"
    ] == 0.01
    assert cases["dinov3-convnext-tiny-pretrain-lvd1689m"]["baseline"][
        "max_image_feature_relative_frobenius"
    ] == 0.015


def test_selection_rejects_multiple_modes() -> None:
    suite = performance_catalog.load_suite(SUITE)

    with pytest.raises(
        performance_catalog.PerformanceSuiteError,
        match="entry, model, and family selections are mutually exclusive",
    ):
        suite.select(entries=["gpt2.generate"], models=["gpt2-125m"])


@pytest.mark.parametrize(
    ("name", "expected"),
    [("gpt2-l0-fp32", True), ("gpt2-125m", False)],
)
def test_l0_profile_classification(name: str, expected: bool) -> None:
    assert performance_catalog.is_l0_profile(name) is expected
