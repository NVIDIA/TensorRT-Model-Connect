# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint data stays exact without importing the upstream model package."""

import importlib.abc
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from families.yolo11.checkpoint import Checkpoint


class _BlockUpstream(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ultralytics" or fullname.startswith("ultralytics."):
            raise AssertionError(f"builder tried to import {fullname}")
        return None


@pytest.mark.parametrize("zip_archive", [True, False])
@pytest.mark.parametrize("upstream_available", [True, False])
def test_archive_preserves_tensor_data_without_upstream(tmp_path: Path, monkeypatch, zip_archive, upstream_available):
    modules = {name: types.ModuleType(name) for name in
               ("ultralytics", "ultralytics.nn", "ultralytics.nn.tasks")}
    cls = type("DetectionModel", (torch.nn.Module,), {"__module__": "ultralytics.nn.tasks"})
    modules["ultralytics.nn.tasks"].DetectionModel = cls
    model = cls()
    model.model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4))
    model.names = {0: "person", 1: "bicycle"}
    model.register_buffer("transient", torch.ones(1), persistent=False)
    model.half()
    expected = {name: value.detach().float().numpy().copy()
                for name, value in model.state_dict().items()
                if not name.endswith("num_batches_tracked")}
    archive = tmp_path / Checkpoint.ARCHIVE
    with monkeypatch.context() as patch:
        for name, module in modules.items():
            patch.setitem(sys.modules, name, module)
        torch.save({"model": model, "train_args": {"imgsz": 96}}, archive,
                   _use_new_zipfile_serialization=zip_archive)
    if upstream_available:
        def forbidden_restore(self, state):
            raise AssertionError("upstream implementation was executed")
        cls.__setstate__ = forbidden_restore
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)
    before = {name: value for name, value in sys.modules.items() if name.startswith("ultralytics")}
    monkeypatch.setattr(sys, "meta_path", [_BlockUpstream(), *sys.meta_path])
    actual = Checkpoint.from_archive(archive)
    assert actual.class_names == ("person", "bicycle")
    assert actual.image_size == 96
    assert actual.names == frozenset(expected)
    for name, value in expected.items():
        assert actual.tensor(name).dtype == np.float32
        np.testing.assert_array_equal(actual.tensor(name), value)
    after = {name: value for name, value in sys.modules.items() if name.startswith("ultralytics")}
    assert before == after
