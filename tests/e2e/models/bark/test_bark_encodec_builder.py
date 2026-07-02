# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bark-owned encodec builder tests using deterministic TRT stubs."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest


_PKG_ROOT = Path(__file__).resolve().parents[4] / "python"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))


def _make_fake_trt_base() -> types.SimpleNamespace:
    class _Logger:
        VERBOSE = 2
        WARNING = 1

        def __init__(self, _level):
            pass

    return types.SimpleNamespace(
        Logger=_Logger,
        ElementWiseOperation=types.SimpleNamespace(SUM="sum", SUB="sub", PROD="prod"),
        ReduceOperation=types.SimpleNamespace(AVG="avg"),
        UnaryOperation=types.SimpleNamespace(SQRT="sqrt", RECIP="recip"),
        MatrixOperation=types.SimpleNamespace(NONE="none", TRANSPOSE="transpose"),
        ActivationType=types.SimpleNamespace(SIGMOID="sigmoid", TANH="tanh"),
        AttentionNormalizationOp=types.SimpleNamespace(SOFTMAX="softmax"),
        MemoryPoolType=types.SimpleNamespace(WORKSPACE="workspace"),
        BuilderFlag=types.SimpleNamespace(TF32="tf32"),
        NetworkDefinitionCreationFlag=types.SimpleNamespace(EXPLICIT_BATCH=0, STRONGLY_TYPED=1),
        Permutation=lambda dims: tuple(dims),
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
        int32="int32",
    )


def _drop_imported_module(module_name: str) -> None:
    sys.modules.pop(module_name, None)
    package_name, _, attribute_name = module_name.rpartition(".")
    package = sys.modules.get(package_name)
    if package is not None and hasattr(package, attribute_name):
        delattr(package, attribute_name)


def _import_with_fake_trt(module_name: str):
    _drop_imported_module(module_name)
    previous_trt = sys.modules.get("tensorrt")
    sys.modules["tensorrt"] = _make_fake_trt_base()
    try:
        return importlib.import_module(module_name)
    finally:
        if previous_trt is None:
            sys.modules.pop("tensorrt", None)
        else:
            sys.modules["tensorrt"] = previous_trt


@pytest.mark.unit
def test_encodec_fuse_weight_norm_matches_manual_formula() -> None:
    """Bark encodec weight-norm fusion is deterministic."""
    mod = _import_with_fake_trt("tensorrt_model_connect.families.bark.encodec_builder")

    g = np.array([[[2.0]], [[4.0]]], dtype=np.float32)
    v = np.array(
        [
            [[3.0, 4.0]],
            [[6.0, 8.0]],
        ],
        dtype=np.float32,
    )

    fused = mod._fuse_weight_norm(g, v)
    expected = np.array([[[1.2, 1.6]], [[2.4, 3.2]]], dtype=np.float32)

    np.testing.assert_allclose(fused, expected, atol=1e-6)
    assert fused.dtype == np.float32
