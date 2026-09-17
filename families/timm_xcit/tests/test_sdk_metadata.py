# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The builder publishes the class contract owned by its checkpoint."""

from types import SimpleNamespace

import pytest

from families.timm_xcit import model


@pytest.mark.parametrize("named", [False, True])
def test_bundle_metadata_preserves_class_order(tmp_path, monkeypatch, named):
    raw = {"num_classes": 5}
    if named:
        raw.update(vocabulary_id="fixture:five", label_names=["a", "b", "c", "d", "e"])
    runtime = {"image_height": 2, "image_width": 2, "num_classes": 5,
               "crop_pct": 1.0, "interpolation": "bilinear",
               "mean": [0.5] * 3, "std": [0.25] * 3}
    monkeypatch.setattr(model, "_read_config", lambda _: raw)
    monkeypatch.setattr(model.Checkpoint, "open", lambda _: object())
    monkeypatch.setattr(model, "_build_engine", lambda *_: (b"plan", runtime))
    sections = {}
    headers = []
    writer = SimpleNamespace(set_header=lambda **value: headers.append(value),
                             add_bytes=lambda key, value: sections.update({key: value}),
                             add_json=lambda key, value: sections.update({key: value}))
    request = SimpleNamespace(dynamic_kv_cache=False, image_height=None, image_width=None,
                              video_num_frames=None, max_batch_size=1, tensor_parallel_size=1,
                              context_parallel_size=1, task="image_to_class_scores",
                              quantization=None, fp32_layers=(), max_sequence_length=1,
                              model_dir=tmp_path, precision="fp16", verbose=False, backend="trt")
    model.build(request, writer)
    assert headers == [{"family": "timm_xcit", "task": "image_to_class_scores", "backend": "trt"}]
    assert sections["runtime.json"]["num_classes"] == 5
    assert sections["runtime.json"]["vocabulary_id"] == raw.get("vocabulary_id", "")
    assert sections["runtime.json"]["labels"] == raw.get("label_names", [])
    assert sections["engine.plan"] == b"plan"
