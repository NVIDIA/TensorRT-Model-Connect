# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the family's benchmark timing and reference receipt."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from families.dinov2.tests import performance_reference as reference


def arguments(**updates):
    values = {
        "family": "dinov2",
        "operation": "extract_features",
        "selected_task": "image_to_token_and_pooled_features",
        "model": "fixture/model",
        "case_name": "dinov2.extract_features",
        "precision": "fp16",
        "mode": "hf-eager",
        "warmup": 1,
        "iterations": 2,
        "timing_contract_json": json.dumps(reference.TIMING),
        "adapter_options_json": "{}",
        "trust_remote_code": False,
        "request_json": '{"image_path":"image.png"}',
    }
    return SimpleNamespace(**(values | updates))


def features(tokens=5, width=3):
    hidden = np.arange(tokens * width, dtype=np.float32).reshape(1, tokens, width)
    return hidden, hidden[:, 0, :].copy()


def test_timing_stops_before_summary_and_receipt(monkeypatch):
    events = []
    clocks = iter((1.0, 1.01, 2.0, 2.02))

    def clock():
        events.append("clock")
        return next(clocks)

    def invoke():
        events.append("model-and-host-features")
        return features()

    def synchronize():
        events.append("synchronize")

    summarize = reference._summary

    def summary(value):
        events.append("shape-finite-summary")
        return summarize(value)

    monkeypatch.setattr(reference.time, "perf_counter", clock)
    monkeypatch.setattr(reference, "_load_reference", lambda *_: (invoke, synchronize, "fixture"))
    monkeypatch.setattr(reference, "_summary", summary)
    result = json.loads(json.dumps(reference.run(arguments()), allow_nan=False))
    assert events == ["model-and-host-features", "synchronize"] + 2 * [
        "synchronize",
        "clock",
        "model-and-host-features",
        "synchronize",
        "clock",
    ] + ["shape-finite-summary"]
    assert result["schema_version"] == "trtmc.perf-baseline/v1"
    assert result["status"] == "completed"
    for field in (
        "family",
        "operation",
        "selected_task",
        "model",
        "case_name",
        "precision",
        "mode",
    ):
        assert result[field] == getattr(arguments(), field)
    assert result["measurement"] == {"warmup": 1, "iterations": 2}
    assert result["measurement_policy"] == reference.TIMING
    assert result["samples_ms"] == pytest.approx([10.0, 20.0])
    assert result["metrics"]["latency_ms"]["p50"] == pytest.approx(15.0)
    # These are the two shapes the image-features-shape contract compares natively.
    assert result["output_summary"] == {
        "last_hidden_state_shape": [1, 5, 3],
        "pooler_output_shape": [1, 3],
        "element_count": 18,
        "finite": True,
    }


@pytest.mark.parametrize(
    "value",
    [
        (np.zeros((5, 3), np.float32), np.zeros((1, 3), np.float32)),
        (np.zeros((1, 0, 3), np.float32), np.zeros((1, 3), np.float32)),
        (np.zeros((1, 5, 3), np.float32), np.zeros((1, 4), np.float32)),
        (np.full((1, 5, 3), np.nan, np.float32), np.zeros((1, 3), np.float32)),
        (np.zeros((1, 5, 3), np.float32), np.full((1, 3), np.inf, np.float32)),
    ],
)
def test_invalid_features_do_not_produce_a_success_receipt(value, monkeypatch):
    monkeypatch.setattr(
        reference, "_load_reference", lambda *_: (lambda: value, lambda: None, "fixture")
    )
    with pytest.raises(ValueError, match="DINOv2"):
        reference.run(arguments(warmup=0, iterations=1))


@pytest.mark.parametrize(
    "updates",
    [
        {"selected_task": "image_features"},
        {"family": "dinov3"},
        {"operation": "classify"},
        {"iterations": 0},
        {"warmup": -1},
        {"request_json": '{"image_path":"image.png","pooling":"mean"}'},
        {"request_json": '{"image_path":"image.png","batch_size":2}'},
        {"adapter_options_json": '{"extra":true}'},
        {"trust_remote_code": True},
        {
            "timing_contract_json": json.dumps(
                reference.TIMING | {"input_preparation_included": True}
            )
        },
    ],
)
def test_invalid_contract_fails_before_model_loading(updates, monkeypatch):
    monkeypatch.setattr(
        reference, "_load_reference", lambda *_: pytest.fail("unexpected model load")
    )
    with pytest.raises(ValueError):
        reference.run(arguments(**updates))
