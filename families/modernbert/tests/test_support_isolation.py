# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("preloaded", ["cold", "graph", "parallel", "model"])
def test_builder_fixture_preserves_later_backend_import(tmp_path, preloaded):
    test = tmp_path / "test_sequential_builds.py"
    test.write_text(
        '''import importlib
import sys
from types import ModuleType

from families.modernbert.tests.test_support import builder

backend = ModuleType("tensorrt")
backend.Builder = object()
sys.modules["tensorrt"] = backend
preloaded = PRELOADED
if preloaded == "graph":
    original = importlib.import_module("families.modernbert.graph.model")
elif preloaded == "parallel":
    original = importlib.import_module("families.modernbert.parallel")
elif preloaded == "model":
    original = importlib.import_module("families.modernbert.model")


def test_00_cpu_bundle_fixture(builder):
    assert builder[0]._ModernBertModel().build_engine() == b"plan"


def test_01_subsequent_backend_build_import():
    assert sys.modules["tensorrt"] is backend
    family = importlib.import_module("families.modernbert")
    graph = importlib.import_module("families.modernbert.graph")
    if preloaded != "model":
        assert "families.modernbert.model" not in sys.modules
        assert not hasattr(family, "model")
    if preloaded == "cold":
        assert "families.modernbert.graph.model" not in sys.modules
        assert not hasattr(graph, "model")
    if preloaded in ("cold", "graph"):
        assert "families.modernbert.parallel" not in sys.modules
        assert not hasattr(family, "parallel")
    model = importlib.import_module("families.modernbert.model")
    parallel = importlib.import_module("families.modernbert.parallel")
    assert model.trt is backend
    assert model.graph_ops.trt is backend
    assert parallel.trt is backend
    assert parallel.graph_ops.trt is backend
    assert model.ParallelConfig is parallel.ParallelConfig
    if preloaded == "model":
        assert model is original
    elif preloaded == "graph":
        assert model.graph_ops is original
    elif preloaded == "parallel":
        assert parallel is original
'''.replace("PRELOADED", repr(preloaded)),
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(root / "core/builder"), environment.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test), "--confcutdir", str(tmp_path),
         "-q", "-p", "no:cacheprovider"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
