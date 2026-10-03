# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

import pytest

from trtmc_aiperf_qual import noninferiority

numpy = pytest.importorskip("numpy")


def test_the_score_test_passes_fails_or_stays_inconclusive():
    assert noninferiority.binary(0, 0, 100, 3.0)["outcome"] == "pass"  # no disagreement on 100 problems
    assert noninferiority.binary(1, 0, 100, 3.0)["outcome"] == "inconclusive"
    # v7's defects: gpt2 on LAMBADA (53 vs 0 of 5,153) and gemma-3-4b on MMLU (65 vs 3 of 1,140)
    assert noninferiority.binary(53, 0, 5153, 0.5)["outcome"] == "fail"
    assert noninferiority.binary(65, 3, 1140, 1.0)["outcome"] == "fail"
    # v7's healthy and noisy-but-unbiased models
    assert noninferiority.binary(8, 5, 1140, 1.0)["outcome"] == "pass"
    assert noninferiority.binary(46, 48, 2280, 1.0)["outcome"] == "pass"


def test_cutoffs_are_exact_for_the_count_and_margin():
    # Values the design states (exact enumeration of both tails at the margin).
    assert noninferiority.critical(100, 0.05) == pytest.approx(1.84)
    assert noninferiority.critical(287, 0.01) == pytest.approx(1.79)
    assert noninferiority.critical(2280, 0.01) == pytest.approx(1.73)


def test_sampled_models_use_the_t_bound_over_problems():
    steady = noninferiority.paired_means([0.0] * 50 + [50.0, -50.0] * 5, 5.0)
    assert steady["outcome"] == "pass" and steady["regression_points"] == 0.0
    assert noninferiority.paired_means([100.0] * 30, 5.0)["outcome"] == "fail"
    assert noninferiority.paired_means([1.0], 5.0)["outcome"] == "inconclusive"
    # The exact t(0.95, 123) quantile: this upper bound is 2.0005, just above the margin of 2
    edge = noninferiority.paired_means([100 / 3, -100 / 3] * 10 + [0.0] * 104, 2.0)
    assert edge["outcome"] == "inconclusive" and edge["upper"] == pytest.approx(2.000511, abs=1e-5)


def test_moving_blocks_keep_consecutive_windows_and_the_sample_size():
    import random

    order = list(range(10))
    picked = noninferiority._moving_blocks(order, 3, random.Random(0))
    assert len(picked) == 10
    assert all(b == a + 1 for start in range(0, 9, 3) for a, b in zip(picked[start:start + 2], picked[start + 1:start + 3]))
    # Positions given out of order are resampled in series order
    scores = [float(position) for position in (4, 0, 3, 1, 2)]
    result = noninferiority.bootstrap(lambda indices: (sum(scores[i] for i in indices), sum(scores[i] for i in indices)),
                                      [4, 0, 3, 1, 2], {"margin": 1.0}, True, 50, block=2)
    assert result["outcome"] == "pass" and result["block"] == 2


def test_relative_margins_and_their_zero_native_score():
    assert noninferiority.margin({"margin": 0.3, "relative_margin": 0.05}, 10.0) == pytest.approx(0.5)
    assert noninferiority.margin({"margin": 0.3, "relative_margin": 0.05}, 2.0) == pytest.approx(0.3)
    zero = noninferiority.bootstrap(lambda indices: (0.0, 0.0), [0, 1], {"margin": 0.0, "relative_margin": 0.01},
                                    False, 10)
    assert zero["outcome"] == "not-comparable"


def test_the_harness_does_not_depend_on_benchmark_qualification():
    """DESIGN.md Section 3: no import of qualification_tests, no read of families/*/tests/benchmark."""
    apps = Path(__file__).resolve().parents[2]
    offenders = []
    for root in (apps / "aiperf_qual", apps / "perf_serving"):
        for path in root.rglob("*.py"):
            if "tests" in path.relative_to(root).parts:
                continue
            text = path.read_text()
            if re.search(r"qualification_tests|tests/benchmark", text):
                offenders.append(str(path.relative_to(apps)))
    assert offenders == []
