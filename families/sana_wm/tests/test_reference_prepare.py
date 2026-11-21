# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint storage regressions for both native-reference preparation entry points."""

from __future__ import annotations

import errno
from importlib import import_module
import os
from pathlib import Path
import shutil

import pytest


@pytest.fixture(params=[
    "families.sana_wm.reference.prepare",
    "families.sana_wm.tests.benchmark.prepare_environment",
])
def prepare(request):
    return import_module(request.param)


def _snapshot(tmp_path: Path) -> tuple[Path, Path]:
    cache = tmp_path / "cache"
    blob = cache / "blobs/weights"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"checkpoint")
    snapshot = cache / "snapshots/revision"
    (snapshot / "dit").mkdir(parents=True)
    weight = snapshot / "dit/model.safetensors"
    weight.symlink_to(os.path.relpath(blob, weight.parent))
    return snapshot, blob


def _no_hardlinks(monkeypatch):
    def cross_device(*_args, **_kwargs):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(os, "link", cross_device)


def test_snapshot_reuses_cache_across_mounts(prepare, monkeypatch, tmp_path):
    snapshot, blob = _snapshot(tmp_path)
    _no_hardlinks(monkeypatch)
    destination = tmp_path / "SANA-model"

    prepare._materialize_snapshot(snapshot, destination)

    weight = destination / "dit/model.safetensors"
    assert weight.is_symlink()
    assert weight.samefile(blob)
    assert weight.read_bytes() == b"checkpoint"


def test_retried_environments_share_weights_and_preserve_cache(prepare, monkeypatch, tmp_path):
    snapshot, blob = _snapshot(tmp_path)
    _no_hardlinks(monkeypatch)
    first, retry = tmp_path / "environment", tmp_path / "environment-r1"
    for environment in (first, retry):
        environment.mkdir()
        prepare._materialize_snapshot(snapshot, environment / "SANA-model")

    first_weight = first / "SANA-model/dit/model.safetensors"
    retry_weight = retry / "SANA-model/dit/model.safetensors"
    assert first_weight.samefile(blob)
    assert retry_weight.samefile(blob)
    shutil.rmtree(first)
    assert retry_weight.read_bytes() == b"checkpoint"
    assert blob.read_bytes() == b"checkpoint"


def test_failed_link_does_not_leave_a_prepared_directory(prepare, monkeypatch, tmp_path):
    snapshot, blob = _snapshot(tmp_path)
    _no_hardlinks(monkeypatch)

    def denied(*_args, **_kwargs):
        raise PermissionError("link creation denied")

    monkeypatch.setattr(Path, "symlink_to", denied)
    destination = tmp_path / "SANA-model"
    with pytest.raises(PermissionError, match="link creation denied"):
        prepare._materialize_snapshot(snapshot, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob("SANA-model.*"))
    assert blob.read_bytes() == b"checkpoint"
