# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
import os
from pathlib import Path
import numpy as np
import pytest
from families.hstu_research.config import Architecture, Config, ServingProfile
from families.hstu_research.checkpoint import expected_shapes, load_weights, parse_gin
from families.hstu_research.reference import Reference, example_inputs


def fixture():
    a = Architecture(
        hidden_size=16,
        num_heads=2,
        attention_dim=4,
        value_dim=8,
        num_layers=3,
        vocab_size=40,
        max_sequence_length=17,
        max_history_length=16,
    )
    c = Config(architecture=a, serving=ServingProfile(history_length=12, candidates=9))
    rng = np.random.default_rng(7)
    w = {
        key: rng.normal(0, 0.02, shape).astype(np.float32)
        for key, shape in expected_shapes(a).items()
    }
    w["causal_mask"] = np.triu(np.ones((17, 17), np.float32), 1)
    return c, w


def test_shape_profile_and_masked_candidates(tmp_path):
    c, w = fixture()
    c.save(tmp_path)
    np.savez(tmp_path / "hstu_research_weights.npz", **w)
    assert Config.load(tmp_path) == c
    assert len(load_weights(tmp_path, c)) == 3 + 5 * c.architecture.num_layers
    x = example_inputs(c, 3)
    x["num_candidates"][:, 0] = [0, 1, 9]
    y = Reference(c, w).score(x)
    assert y["picked_indices"][0, 0] == -1 and y["picked_indices"][1, 0] == 0
    assert not y["scores"][0].any()
    with pytest.raises(ValueError):
        replace(c, serving=replace(c.serving, history_length=17))


def test_profile_padding_preserves_encoder_and_scores():
    c, w = fixture()
    small = replace(c, serving=replace(c.serving, history_length=6, candidates=5))
    x = example_inputs(small, 1)
    large = {name: value.copy() for name, value in x.items()}
    large["history_ids"] = np.pad(x["history_ids"], ((0, 0), (0, 6)))
    large["history_timestamps"] = np.pad(x["history_timestamps"], ((0, 0), (0, 6)))
    large["candidate_ids"] = np.pad(x["candidate_ids"], ((0, 0), (0, 4)))
    a, b = Reference(small, w).score(x), Reference(c, w).score(large)
    np.testing.assert_allclose(a["scores"], b["scores"][:, :5], atol=1e-6, rtol=1e-5)
    np.testing.assert_array_equal(a["picked_indices"], b["picked_indices"])


def test_batch_profile_and_training_config():
    c, _ = fixture()
    c = replace(c, serving=replace(c.serving, min_batch_size=2, opt_batch_size=4))
    r = c.runtime(8, "bf16")
    assert (r["min_batch_size"], r["opt_batch_size"], r["max_batch_size"]) == (2, 4, 8)
    assert r["stages"][0]["outputs"][0]["shape"] == [-1, 3, 2, 12, 4]
    assert parse_gin("# ignored\nx = False\ny = {'a': 1}\n") == dict(x=False, y={"a": 1})
    with pytest.raises(ValueError):
        c.runtime(3, "fp32")


def test_precision_boundaries():
    c, _ = fixture()
    optimized = replace(c, bf16_projections="all", fp32_block_indices=(0,))
    r = optimized.runtime(4, "bf16")
    assert r["bf16_projections"] == "all" and r["fp32_block_indices"] == [0]
    for indices in ((3,), (-1,), (0, 0)):
        with pytest.raises(ValueError):
            replace(c, fp32_block_indices=indices)
    with pytest.raises(ValueError):
        replace(c, bf16_projections="invalid")
    with pytest.raises(ValueError, match="at least one"):
        replace(c, fp32_block_indices=(0, 1, 2)).runtime(4, "bf16")


def test_public_checkpoint_against_pinned_upstream():
    directory = os.environ.get("HSTU_RESEARCH_CHECKPOINT")
    source = os.environ.get("HSTU_UPSTREAM_SOURCE")
    if not directory or not source:
        pytest.skip("select the downloaded public checkpoint and pinned upstream source")
    from .upstream import upstream_scores

    c = Config.load(directory)
    weights = load_weights(directory, c)
    x = example_inputs(c, 2)
    expected = upstream_scores(c, weights, x, Path(source))
    actual = Reference(c, weights).score(x)
    np.testing.assert_allclose(actual["scores"], expected, atol=1e-5, rtol=1e-4)
