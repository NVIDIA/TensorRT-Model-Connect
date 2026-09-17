# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the family's benchmark timing and reference receipt."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from families.timm_densenet.tests import performance_reference as reference


def arguments(**updates):
    values = {
        "family": "timm_densenet", "operation": "classify",
        "selected_task": "image_to_class_scores", "model": "fixture/model",
        "case_name": "timm_densenet.classify", "precision": "fp16", "mode": "hf-eager",
        "warmup": 1, "iterations": 2, "timing_contract_json": json.dumps(reference.TIMING),
        "adapter_options_json": "{}", "trust_remote_code": False,
        "request_json": '{"image_path":"image.jpeg"}',
    }
    return SimpleNamespace(**(values | updates))


def test_timing_stops_before_classification_and_receipt(monkeypatch):
    events = []
    clocks = iter((1.0, 1.01, 2.0, 2.02))
    scores = np.array([[1.0, 4.0, -2.0]], dtype=np.float32)

    def clock():
        events.append("clock")
        return next(clocks)

    def invoke():
        events.append("model-and-host-scores")
        return scores.copy()

    def synchronize():
        events.append("synchronize")

    summarize = reference._summary

    def summary(value):
        events.append("argmax-finite-summary")
        return summarize(value)

    monkeypatch.setattr(reference.time, "perf_counter", clock)
    monkeypatch.setattr(reference, "_load_reference", lambda *_: (invoke, synchronize, "fixture"))
    monkeypatch.setattr(reference, "_summary", summary)
    result = json.loads(json.dumps(reference.run(arguments()), allow_nan=False))
    assert events == ["model-and-host-scores", "synchronize"] + 2 * [
        "synchronize", "clock", "model-and-host-scores", "synchronize", "clock",
    ] + ["argmax-finite-summary"]
    assert result["schema_version"] == "trtmc.perf-baseline/v1"
    assert result["status"] == "completed"
    for field in ("family", "operation", "selected_task", "model", "case_name", "precision", "mode"):
        assert result[field] == getattr(arguments(), field)
    assert result["measurement"] == {"warmup": 1, "iterations": 2}
    assert result["measurement_policy"] == reference.TIMING
    assert all(result[key] == value for key, value in reference.TIMING.items())
    assert result["samples_ms"] == pytest.approx([10.0, 20.0])
    assert result["metrics"]["latency_ms"]["p50"] == pytest.approx(15.0)
    assert result["output_summary"] == {
        "top_class": 1, "shape": [1, 3], "element_count": 3, "finite": True,
        "scores": [1.0, 4.0, -2.0],
    }


@pytest.mark.parametrize("scores", [[], [[]], [[1.0], [2.0]], [[float("nan")]], [[float("inf")]]])
def test_invalid_scores_do_not_produce_a_success_receipt(scores, monkeypatch):
    monkeypatch.setattr(reference, "_load_reference", lambda *_: (
        lambda: np.asarray(scores, dtype=np.float32), lambda: None, "fixture",
    ))
    with pytest.raises(ValueError, match="class-score|class scores"):
        reference.run(arguments(warmup=0, iterations=1))


@pytest.mark.parametrize("updates", [
    {"selected_task": "classification"}, {"family": "other"}, {"iterations": 0},
    {"warmup": -1}, {"request_json": '{"image_path":"image.jpeg","top_k":1}'},
    {"request_json": '{"image_path":"image.jpeg","batch_size":2}'},
    {"adapter_options_json": '{"extra":true}'}, {"trust_remote_code": True},
    {"timing_contract_json": json.dumps(reference.TIMING | {"input_preparation_included": True})},
])
def test_invalid_contract_fails_before_model_loading(updates, monkeypatch):
    monkeypatch.setattr(reference, "_load_reference", lambda *_: pytest.fail("unexpected model load"))
    with pytest.raises(ValueError):
        reference.run(arguments(**updates))
