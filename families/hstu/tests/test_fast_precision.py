# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in fast-policy quality and graph freshness checks on prepared bundles.

Strict score-parity gates remain in test_e2e.py. Fast mode deliberately uses
ranking/mean-error gates, and reports rather than hides individual score drift.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from families.hstu.checkpoint import load_weights
from families.hstu.config import Config
from families.hstu.reference import Reference, example_inputs
from families.hstu.triton.native_client import NativeClient

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def fast_root():
    root = os.environ.get("HSTU_FAST_E2E_ROOT")
    runtime = os.environ.get("TRTMC_RUNTIME_ROOT")
    if not root or not runtime:
        pytest.skip("set HSTU_FAST_E2E_ROOT and TRTMC_RUNTIME_ROOT for fast-policy checks")
    return Path(root), runtime


@pytest.mark.parametrize("candidates", [50, 64])
@pytest.mark.parametrize("batch", [1, 4])
def test_fast_policy_ranking_and_mean_score_error(fast_root, candidates, batch):
    root, runtime = fast_root
    directory = root / f"hstu{candidates}-bf16"
    c = Config.load(directory)
    assert c.bf16_policy == "fast"
    reference = Reference(c, load_weights(directory, c))
    with NativeClient(directory / "model.bundle", runtime) as task:
        assert task.metadata["bf16_policy"] == "fast"
        for seed in (42, *range(20261125, 20261134)):
            inputs = example_inputs(c, batch, seed)
            expected = reference.score(inputs)
            actual = task.score(inputs)
            assert np.isfinite(actual["scores"]).all()
            assert np.mean(np.abs(actual["scores"] - expected["scores"])) <= 0.02
            overlap = np.mean(
                [
                    len(set(a[:10]) & set(b[:10])) / 10
                    for a, b in zip(actual["picked_indices"], expected["picked_indices"])
                ]
            )
            assert overlap >= 0.95
            for picks in actual["picked_indices"]:
                assert len(set(picks)) == 50 and np.all((picks >= 0) & (picks < candidates))


@pytest.mark.parametrize("candidates", [50, 64])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_graph_rebinds_shapes_and_refreshes_inputs(fast_root, candidates, precision):
    root, runtime = fast_root
    directory = root / f"hstu{candidates}-{precision}"
    c = Config.load(directory)
    assert c.cuda_graphs
    with NativeClient(directory / "model.bundle", runtime) as task:
        assert task.metadata["cuda_graphs"] is True
        for batch, seed in ((1, 42), (4, 43), (1, 44), (4, 45), (1, 42)):
            inputs = example_inputs(c, batch, seed)
            with NativeClient(directory / "model.bundle", runtime) as fresh:
                expected = fresh.score(inputs)
            for _ in range(2):
                actual = task.score(inputs)
                for name in expected:
                    np.testing.assert_array_equal(actual[name], expected[name])
