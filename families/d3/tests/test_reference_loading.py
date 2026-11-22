# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ensure the reference always loads code from its selected checkpoint."""
import sys
from types import SimpleNamespace

import pytest

from families.d3.tests.reference import load_model


def test_reference_rejects_cached_module_from_another_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="5.17.0"))
    monkeypatch.setitem(sys.modules, "d3_runtime", SimpleNamespace(__file__=str(tmp_path / "other.py")))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    with pytest.raises(RuntimeError, match="outside the checkpoint"):
        load_model(checkpoint)


def test_reference_does_not_execute_module_outside_checkpoint(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = tmp_path / "executed"
    (outside / "d3_runtime.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="5.17.0"))
    monkeypatch.delitem(sys.modules, "d3_runtime", raising=False)
    monkeypatch.setattr(sys, "path", [str(outside), *sys.path])
    with pytest.raises(RuntimeError, match="outside the checkpoint"):
        load_model(checkpoint)
    assert not marker.exists()
