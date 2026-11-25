# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, replace
import json
import numpy as np
import pytest
from families.hstu.config import Config
from families.hstu.checkpoint import create_fixture, load_weights
from families.hstu.reference import Reference, example_inputs


def test_checkpoint_capacity_is_independent_of_serving(tmp_path):
    c = Config(
        hidden_size=128,
        num_heads=4,
        history_length=32,
        candidates=17,
        decode_steps=7,
        max_history_length=128,
        max_decode_steps=50,
        normalization_length=256,
        min_batch_size=2,
        opt_batch_size=4,
    )
    create_fixture(tmp_path, config=c)
    assert load_weights(tmp_path, c)["position_embeddings"].shape == (179, 128)
    data = json.loads((tmp_path / "config.json").read_text())
    assert data["architecture"]["max_history_length"] == 128
    assert data["serving"]["decode_steps"] == 7
    assert Config.load(tmp_path) == c
    runtime = c.runtime(8, "bf16")
    assert (runtime["min_batch_size"], runtime["opt_batch_size"], runtime["max_batch_size"]) == (
        2,
        4,
        8,
    )
    with pytest.raises(ValueError):
        replace(c, history_length=129)
    with pytest.raises(ValueError):
        c.runtime(3, "bf16")


def test_padding_capacity_does_not_change_model_math(tmp_path):
    small = Config(
        history_length=32,
        candidates=17,
        decode_steps=7,
        max_history_length=64,
        normalization_length=256,
    )
    create_fixture(tmp_path, config=small)
    weights = load_weights(tmp_path, small)
    large = replace(small, history_length=64, candidates=31)
    x = example_inputs(small, 1)
    x["history_lengths"].fill(20)
    expanded = {key: value.copy() for key, value in x.items()}
    expanded["history_ids"] = np.pad(x["history_ids"], ((0, 0), (0, 32)))
    expanded["candidate_ids"] = np.pad(x["candidate_ids"], ((0, 0), (0, 14)))
    a, b = Reference(small, weights).score(x), Reference(large, weights).score(expanded)
    np.testing.assert_allclose(a["scores"], b["scores"][:, :17], atol=1e-6, rtol=1e-5)
    np.testing.assert_array_equal(a["picked_indices"], b["picked_indices"])


def test_legacy_flat_configuration_remains_readable(tmp_path):
    values = asdict(Config())
    for key in (
        "max_history_length",
        "max_decode_steps",
        "normalization_length",
        "min_batch_size",
        "opt_batch_size",
        "refinement_candidates",
        "selection_margin",
    ):
        values.pop(key)
    (tmp_path / "config.json").write_text(json.dumps(values))
    c = Config.load(tmp_path)
    assert c.normalization_length == 228 and c.max_history_length == 128
