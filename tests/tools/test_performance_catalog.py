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


def test_release_suite_includes_minimax_h3_video_only_performance() -> None:
    suite = performance_catalog.load_suite(SUITE)

    assert "minimax-h3-768p" not in suite.excluded_profiles
    case = next(case for case in suite.cases if case["model"] == "minimax-h3-768p")
    assert case["id"] == "minimax_h3.generate_image"
    assert case["operation"] == "generate_image"
    assert case["measurement"] == {"warmup": 3, "iterations": 10}
    assert case["baseline"]["adapter_options"] == {
        "diffusers_revision": "abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc",
        "generator_device": "cpu",
        "output_fields": ["videos"],
        "require_pinned_diffusers_source": True,
        "require_pinned_transformers_source": True,
        "transformers_compat_revision": "bed02e1faee69e866e382f835b4f7b0a3c7b8431",
    }


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
