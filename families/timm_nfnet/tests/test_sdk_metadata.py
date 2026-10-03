# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The builder publishes complete checkpoint-owned Task metadata."""

from types import SimpleNamespace

import pytest

from families.timm_nfnet import model
from tensorrt_model_connect import BuildRequest


@pytest.mark.parametrize("metadata,invalid", [
    ({}, False),
    ({"vocabulary_id": "fixture:five", "label_names": ["a", "b", "c", "d", "e"]}, False),
    ({"label_names": ["only-one"]}, True),
    ({"label_names": ["a", "b", "", "d", "e"]}, True),
    ({"label_names": 5}, True),
    ({"vocabulary_id": 5}, True),
])
def test_builder_task_and_metadata(tmp_path, monkeypatch, metadata, invalid):
    raw = {"num_classes": 5, **metadata}
    runtime = {"image_height": 2, "image_width": 2, "num_classes": 5,
               "crop_pct": 1.0, "interpolation": "bilinear", "crop_mode": "squash",
               "mean": [0.5] * 3, "std": [0.25] * 3}
    monkeypatch.setattr(model, "_read_config", lambda _: raw)
    monkeypatch.setattr(model.Checkpoint, "open", lambda _: object())
    engine_calls = []

    def build_engine(*args):
        engine_calls.append(args)
        return b"plan", runtime

    monkeypatch.setattr(model, "_build_engine", build_engine)
    sections = {}
    headers = []
    writer = SimpleNamespace(set_header=lambda **value: headers.append(value),
                             add_bytes=lambda key, value: sections.update({key: value}),
                             add_json=lambda key, value: sections.update({key: value}))
    request = BuildRequest(model_dir=tmp_path, output_path=tmp_path / "model.bundle",
                           family="timm_nfnet", task="image_to_class_scores", precision="fp32")
    if invalid:
        with pytest.raises(ValueError, match="vocabulary_id|label_names"):
            model.build(request, writer)
        assert not sections and not headers
        assert not engine_calls
        return
    model.build(request, writer)
    assert len(engine_calls) == 1
    assert headers == [{"family": "timm_nfnet", "task": "image_to_class_scores", "backend": "trt"}]
    assert sections["engine.plan"] == b"plan"
    assert sections["runtime.json"]["num_classes"] == 5
    assert sections["runtime.json"]["vocabulary_id"] == metadata.get("vocabulary_id", "")
    assert sections["runtime.json"]["labels"] == metadata.get("label_names", [])
    assert sections["runtime.json"]["crop_mode"] == "squash"
