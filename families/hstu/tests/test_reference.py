# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
import pytest
from families.hstu.config import Config
from families.hstu.checkpoint import create_fixture, load_weights
from families.hstu.reference import Reference, example_inputs


@pytest.mark.parametrize("candidates", [50, 64])
def test_public_fixture_reproducible_and_complete(tmp_path, candidates):
    a, b = tmp_path / "a", tmp_path / "b"
    c = create_fixture(a, candidates)
    create_fixture(b, candidates)
    wa, wb = load_weights(a, c), load_weights(b, c)
    for name in wa:
        np.testing.assert_array_equal(wa[name], wb[name])
    assert json.loads((a / "checkpoint_provenance.json").read_text())["pretrained"] is False
    runtime = c.runtime(4, "bf16")
    history, request, step = runtime["stages"]
    assert {s["name"] for s in history["inputs"] + request["inputs"]} == {
        s["name"] for s in runtime["inputs"]
    }
    assert {s["name"] for s in history["inputs"]}.isdisjoint(s["name"] for s in request["inputs"])
    assert [s["dtype"] for s in history["outputs"]] == ["float32", "float32", "int32"]
    assert {s["name"] for s in step["outputs"]} == {
        "scores_out",
        "placed_mask_out",
        "frozen_raw_out",
        "placed_count_out",
        "picked",
    }
    assert 2 * c.num_heads * c.history_length * (c.hidden_size // c.num_heads) * 4 == 262144


@pytest.mark.parametrize("candidates", [50, 64])
def test_padding_counts_empty_history_stable_ties_and_nonrepeat(tmp_path, candidates):
    c = create_fixture(tmp_path / "fixture", candidates)
    w = load_weights(tmp_path / "fixture", c)
    ref = Reference(c, w)
    x = example_inputs(c, 4)
    x["num_candidates"][:, 0] = [0, 1, 17, candidates]
    y = ref.score(x)
    for row, count in enumerate(x["num_candidates"][:, 0]):
        valid = y["picked_indices"][row]
        valid = valid[valid >= 0]
        assert len(valid) == min(count, 50) and len(set(valid.tolist())) == len(valid)
        assert np.all(valid < count)
    padded = {k: v.copy() for k, v in x.items()}
    padded["history_ids"][0] = c.vocab_size - 1
    np.testing.assert_array_equal(ref.score(padded)["scores"][0], y["scores"][0])
    ties = example_inputs(c, 1)
    ties["candidate_ids"].fill(1)
    np.testing.assert_array_equal(ref.score(ties)["picked_indices"][0], np.arange(50))
    changed = {k: v.copy() for k, v in x.items()}
    changed["history_ids"][3] = 1
    assert not np.array_equal(ref.score(changed)["scores"][3], y["scores"][3])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_size": 128},
        {"num_heads": 4},
        {"history_length": 64},
        {"candidates": 51},
        {"epsilon": 1e-5},
        {"execution_method": "unrolled"},
        {"model_type": "hstu"},
    ],
)
def test_reject_unimplemented_profiles(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_bad_checkpoint_rejected(tmp_path):
    c = create_fixture(tmp_path / "fixture")
    p = tmp_path / "fixture" / "hstu_weights.npz"
    w = load_weights(p.parent, c)
    w["uvqk"][0, 0] = np.nan
    np.savez(p, **w)
    with pytest.raises(ValueError, match="uvqk"):
        load_weights(p.parent, c)
