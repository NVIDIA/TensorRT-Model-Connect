# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify explicit fixed-batch bundles without assuming TensorRT kernel choices.

Prepare fixed-b1/output and fixed-b4/output with tests.triton.prepare, the
strict policy and BF16 attention epilogue. Set HSTU_FIXED_E2E_ROOT to their
parent and TRTMC_RUNTIME_ROOT to the matching SDK. Fusion requires separate
executed-kernel evidence; correctness does not depend on GPU or SDK version.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from families.hstu.checkpoint import load_weights
from families.hstu.config import Config
from families.hstu.reference import Reference, example_inputs
from families.hstu.triton.native_client import NativeClient, snapshot_metadata

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("candidates", [50, 64])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_fixed_profile_accuracy_and_history(candidates, batch, precision):
    root, runtime = os.environ.get("HSTU_FIXED_E2E_ROOT"), os.environ.get("TRTMC_RUNTIME_ROOT")
    if not root or not runtime:
        pytest.skip("set HSTU_FIXED_E2E_ROOT and TRTMC_RUNTIME_ROOT")
    directory = Path(root) / f"fixed-b{batch}/output/hstu{candidates}-{precision}"
    c = Config.load(directory)
    assert c.batching == "fixed" and c.bf16_attention_epilogue and c.bf16_policy == "strict"
    reference = Reference(c, load_weights(directory, c))
    gate = json.loads((Path(__file__).parent / "thresholds/correctness.json").read_text())[
        precision
    ]
    with NativeClient(
        directory / "model.bundle", runtime, hydration=True, kv_cache_size_bytes=9437184
    ) as task:
        assert task.metadata["min_batch_size"] == task.metadata["max_batch_size"] == batch
        for seed in (42, *range(20261125, 20261134)):
            inputs = example_inputs(c, batch, seed)
            expected, actual = reference.score(inputs), task.score(inputs)
            np.testing.assert_allclose(
                actual["scores"],
                expected["scores"],
                atol=gate["score_atol"],
                rtol=gate["score_rtol"],
            )
            assert (
                np.mean(actual["picked_indices"] == expected["picked_indices"])
                >= gate["greedy_position_agreement"]
            )
            overlap = np.mean(
                [
                    len(set(a[:10]) & set(b[:10])) / 10
                    for a, b in zip(actual["picked_indices"], expected["picked_indices"])
                ]
            )
            assert overlap >= gate.get("top10_overlap", 1.0)
        inputs = example_inputs(c, batch)
        history_names = [s["name"] for s in task.metadata["stages"][0]["inputs"]]
        snapshot = task.export_history({n: inputs[n] for n in history_names}, ["r1"] * batch)
        handles = task.restore_history(snapshot)
        cold = task.score(inputs)
        before = task.stats()
        cached = task.score_with_history(
            handles,
            snapshot_metadata(snapshot)["revisions"],
            {n: v for n, v in inputs.items() if n not in history_names},
        )
        for name in cold:
            np.testing.assert_array_equal(cached[name], cold[name])
        after = task.stats()
        assert after["history_prepare_calls_total"] == before["history_prepare_calls_total"]
        assert after["decode_step_calls_total"] == before["decode_step_calls_total"] + 50
        assert after["used_bytes"] == after["active_sequences"] == 0
        wrong_batch = 1 if batch == 4 else 2
        wrong = example_inputs(c, wrong_batch)
        with pytest.raises(RuntimeError, match="batch"):
            task.score(wrong)
        with pytest.raises(RuntimeError, match="batch"):
            task.export_history({n: wrong[n] for n in history_names}, ["bad"] * wrong_batch)
        with pytest.raises(RuntimeError, match="batch"):
            task.score_with_history(
                [handles[0]] * wrong_batch,
                ["r1"] * wrong_batch,
                {n: v for n, v in wrong.items() if n not in history_names},
            )
        assert task.stats()["decode_step_calls_total"] == after["decode_step_calls_total"]
