# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference independence and class-aware suppression contracts."""

import importlib.abc
import io
import sys
import types

import pytest
import torch

from families.yolo11.tests.reference.model import _Unpickler, detections, load_model


class BlockUpstream(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"ultralytics", "models"}:
            raise AssertionError(f"reference imported upstream code: {fullname}")
        return None


def test_reference_runs_serialized_graph_without_upstream(tmp_path, monkeypatch):
    namespace = types.ModuleType("ultralytics.nn.tasks")
    cls = type("DetectionModel", (torch.nn.Module,), {"__module__": namespace.__name__})
    setattr(namespace, cls.__name__, cls)
    instance = cls()
    layer = torch.nn.Conv2d(1, 1, 1, bias=False)
    layer.weight.data.fill_(2)
    layer.f = -1
    instance.model = torch.nn.Sequential(layer)
    with monkeypatch.context() as patch:
        parts = namespace.__name__.split(".")
        for end in range(1, len(parts)):
            name = ".".join(parts[:end])
            patch.setitem(sys.modules, name, types.ModuleType(name))
        patch.setitem(sys.modules, namespace.__name__, namespace)
        torch.save({"model": instance}, tmp_path / "yolo11n.pt")
    before = {
        name: module
        for name, module in sys.modules.items()
        if name.split(".")[0] in {"models", "ultralytics"}
    }
    monkeypatch.setattr(sys, "meta_path", [BlockUpstream(), *sys.meta_path])
    reference = load_model(tmp_path)
    inputs = torch.arange(6.0).reshape(1, 1, 2, 3)
    torch.testing.assert_close(reference(inputs), inputs * 2, rtol=0, atol=0)
    assert before == {
        name: module
        for name, module in sys.modules.items()
        if name.split(".")[0] in {"models", "ultralytics"}
    }


def test_reference_rejects_unknown_checkpoint_operation():
    with pytest.raises(ValueError, match="unsupported reference operation"):
        _Unpickler(io.BytesIO()).find_class("ultralytics.nn.tasks", "UnknownOperation")


def test_suppression_keeps_different_classes_and_rejects_low_scores():
    # Same-class duplicate, overlapping different class, and low-confidence box.
    boxes = torch.tensor(
        [[10, 10, 8, 8], [10, 10, 8, 8], [10, 10, 8, 8], [30, 30, 4, 4]], dtype=torch.float32
    )
    scores = torch.tensor([[0.9, 0.1], [0.8, 0.1], [0.1, 0.7], [0.1, 0.2]])
    raw = torch.cat((boxes, scores), dim=1).transpose(0, 1)[None]
    actual = detections(raw)
    expected = torch.tensor([[6, 6, 14, 14, 0.9, 0], [6, 6, 14, 14, 0.7, 1]])
    torch.testing.assert_close(actual, expected)
    assert detections(raw, threshold=1.0).shape == (0, 6)
