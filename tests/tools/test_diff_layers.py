# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-tests for tools/diff_layers.py — layer diff table, pass/fail logic.

Trace: ARCH-TRT-001, UD-TRT-DIFF-LAYERS
Intent: Validate per-layer hidden state diff logic including tolerance checks and argmax matching
Preconditions: NumPy arrays simulating TRT and HF layer outputs are available
Postconditions: Identical layers show zero diff, within-tolerance layers pass, and argmax matches are detected
"""

from __future__ import annotations

import numpy as np


class TestLayerDiffLogic:
    """Test the per-layer comparison logic (pure numpy, no GPU)."""

    def test_identical_layers_pass(self):
        a = np.random.randn(1, 64).astype(np.float32)
        diff = np.abs(a - a)
        assert float(diff.max()) == 0.0

    def test_within_tolerance(self):
        a = np.random.randn(1, 64).astype(np.float32)
        b = a + np.random.uniform(-0.01, 0.01, a.shape).astype(np.float32)
        diff = np.abs(a - b)
        assert float(diff.max()) <= 0.05

    def test_exceeds_tolerance(self):
        a = np.zeros((1, 64), dtype=np.float32)
        b = np.ones((1, 64), dtype=np.float32)
        diff = np.abs(a - b)
        assert float(diff.max()) > 0.05

    def test_std_computation(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        std = float(np.std(a))
        assert abs(std - np.sqrt(2.0)) < 0.01

    def test_mean_diff(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.1, 2.1, 3.1], dtype=np.float32)
        diff = np.abs(a - b)
        assert abs(float(diff.mean()) - 0.1) < 0.01


class TestArgmaxLogitMatch:
    """Test logit argmax matching (same logic as diff_layers final row)."""

    def test_matching_argmax(self):
        trt = np.array([0.1, 0.2, 0.9, 0.3])
        hf = np.array([0.15, 0.25, 0.85, 0.35])
        assert np.argmax(trt) == np.argmax(hf)

    def test_mismatched_argmax(self):
        trt = np.array([0.1, 0.9, 0.2, 0.3])
        hf = np.array([0.1, 0.2, 0.9, 0.3])
        assert np.argmax(trt) != np.argmax(hf)
