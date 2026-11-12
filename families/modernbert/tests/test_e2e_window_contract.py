# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def reference(tmp_path, monkeypatch):
    e2e = importlib.import_module("families.modernbert.tests.test_e2e")
    config = {
        "model_type": "modernbert",
        "num_hidden_layers": 2,
        "global_attn_every_n_layers": 2,
        "local_attention": 128,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    state = {"tokens": 130, "truncation": None, "device_calls": 0}

    def tokenize(text, **kwargs):
        state["truncation"] = kwargs["truncation"]
        return {"input_ids": np.zeros((1, state["tokens"]), dtype=np.int64)}

    def device(name):
        state["device_calls"] += 1
        raise RuntimeError("reference validation reached device setup")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(device=device))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenize),
            AutoModel=SimpleNamespace(),
        ),
    )
    return e2e, tmp_path, state, config


@pytest.mark.parametrize("tokens", [1, 64, 129])
def test_long_reference_rejects_inputs_without_full_window_span(reference, tokens):
    e2e, path, state, _ = reference
    state["tokens"] = tokens
    with pytest.raises(AssertionError, match="span both sides of the local window"):
        e2e._official_reference(
            path,
            {"task": "encoding", "max_sequence_length": 256},
            {"prompt": "short", "exercise_local_window": True},
            path,
        )
    assert state["device_calls"] == 0


def test_long_reference_rejects_input_outside_native_profile(reference):
    e2e, path, state, _ = reference
    state["tokens"] = 257
    with pytest.raises(AssertionError, match="exceeds the native profile"):
        e2e._official_reference(
            path,
            {"task": "encoding", "max_sequence_length": 256},
            {"prompt": "long", "exercise_local_window": True},
            path,
        )
    assert state["device_calls"] == 0


@pytest.mark.parametrize("layout", ["legacy", "materialized"])
def test_long_reference_accepts_boundary_workload_without_truncation(reference, layout):
    e2e, path, state, config = reference
    if layout == "materialized":
        config.update(layer_types=["global_attention", "local_attention"], sliding_window=64)
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="reached device setup"):
        e2e._official_reference(
            path,
            {"task": "encoding", "max_sequence_length": 256},
            {"prompt": "long", "exercise_local_window": True},
            path,
        )
    assert state["truncation"] is False


@pytest.mark.parametrize(
    "name", ["modernbert-base", "modernbert-base-tp4", "granite-embedding-small-english-r2"]
)
def test_registered_long_window_cases_keep_existing_precision_and_threshold(name):
    e2e = importlib.import_module("families.modernbert.tests.test_e2e")
    _, manifest, case = e2e.CASES[name + "-local-window"]
    assert case["exercise_local_window"] is True
    assert case["reference_precision"] == "fp32"
    assert manifest["precision"] == "fp32"
    assert e2e._thresholds(case["name"]) == e2e._thresholds(name)
