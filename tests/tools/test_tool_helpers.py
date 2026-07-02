# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-tests for tools/tool_helpers.py — cosine_sim, compare_arrays.

Trace: ARCH-DIFF-001, UD-DIFF-HELPERS
Intent: Validate tool_helpers cosine similarity computation and array comparison for edge cases
Preconditions: tool_helpers module is importable; numpy available
Postconditions: Cosine similarity matches mathematical expectations including zero/opposite vector edge cases
"""

from __future__ import annotations

import numpy as np


def _import_tool_helpers():
    import importlib
    return importlib.import_module("tool_helpers")


# ---------------------------------------------------------------------------
# cosine_sim
# ---------------------------------------------------------------------------

class TestCosineSim:
    """Tests for cosine_sim(a, b) — cosine similarity between two arrays."""

    def test_identical_vectors(self):
        mod = _import_tool_helpers()
        a = np.array([1.0, 2.0, 3.0])
        result = mod.cosine_sim(a, a)
        assert abs(result - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        mod = _import_tool_helpers()
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        result = mod.cosine_sim(a, b)
        assert abs(result) < 1e-6

    def test_opposite_vectors(self):
        mod = _import_tool_helpers()
        a = np.array([1.0, 2.0, 3.0])
        b = -a
        result = mod.cosine_sim(a, b)
        assert result < -0.99

    def test_zero_vector_graceful(self):
        """Zero vector should return ~0 due to the 1e-8 epsilon."""
        mod = _import_tool_helpers()
        a = np.array([1.0, 2.0, 3.0])
        b = np.zeros(3)
        result = mod.cosine_sim(a, b)
        assert abs(result) < 1e-3

    def test_both_zero_vectors(self):
        mod = _import_tool_helpers()
        a = np.zeros(3)
        b = np.zeros(3)
        result = mod.cosine_sim(a, b)
        assert abs(result) < 1e-3

    def test_different_shapes_same_flattened(self):
        """2D and 1D arrays with the same flattened content give same result."""
        mod = _import_tool_helpers()
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        result = mod.cosine_sim(a, b)
        assert abs(result - 1.0) < 1e-6

    def test_negative_identical_vectors(self):
        mod = _import_tool_helpers()
        a = np.array([-1.0, -2.0, -3.0])
        result = mod.cosine_sim(a, a)
        assert abs(result - 1.0) < 1e-6

    def test_unit_vectors(self):
        mod = _import_tool_helpers()
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        result = mod.cosine_sim(a, b)
        assert abs(result) < 1e-6

    def test_scalar_arrays(self):
        """Single-element arrays: parallel → 1.0."""
        mod = _import_tool_helpers()
        a = np.array([5.0])
        b = np.array([3.0])
        result = mod.cosine_sim(a, b)
        assert abs(result - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# compare_arrays
# ---------------------------------------------------------------------------

class TestCompareArrays:
    """Tests for compare_arrays(name, ours, ref, atol)."""

    def test_identical_arrays_pass(self, capsys):
        mod = _import_tool_helpers()
        a = np.array([1.0, 2.0, 3.0])
        result = mod.compare_arrays("test", a, a, atol=1e-3)
        assert result is True
        captured = capsys.readouterr().out
        assert "PASS" in captured

    def test_identical_arrays_zero_diff(self, capsys):
        mod = _import_tool_helpers()
        a = np.array([1.0, 2.0, 3.0])
        mod.compare_arrays("test", a, a, atol=1e-3)
        captured = capsys.readouterr().out
        assert "max_diff=0.000000" in captured

    def test_within_tolerance_pass(self, capsys):
        mod = _import_tool_helpers()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0001, 2.0001, 3.0001])
        result = mod.compare_arrays("test", a, b, atol=1e-3)
        assert result is True
        captured = capsys.readouterr().out
        assert "PASS" in captured

    def test_exceeding_tolerance_fail(self, capsys):
        mod = _import_tool_helpers()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 5.0])
        result = mod.compare_arrays("test", a, b, atol=1e-3)
        assert result is False
        captured = capsys.readouterr().out
        assert "FAIL" in captured

    def test_exactly_at_tolerance_pass(self, capsys):
        """max_diff == atol should pass (uses <= check)."""
        mod = _import_tool_helpers()
        a = np.array([0.0])
        b = np.array([0.5])
        result = mod.compare_arrays("test", a, b, atol=0.5)
        assert result is True
        captured = capsys.readouterr().out
        assert "PASS" in captured

    def test_output_contains_cosine_sim(self, capsys):
        mod = _import_tool_helpers()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        mod.compare_arrays("test", a, b, atol=1e-3)
        captured = capsys.readouterr().out
        assert "cosine_sim=" in captured

    def test_output_contains_mean_diff(self, capsys):
        mod = _import_tool_helpers()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 3.1])
        mod.compare_arrays("test", a, b, atol=1.0)
        captured = capsys.readouterr().out
        assert "mean_diff=" in captured

    def test_2d_arrays(self, capsys):
        """2D arrays are flattened for comparison."""
        mod = _import_tool_helpers()
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = mod.compare_arrays("test_2d", a, b, atol=1e-3)
        assert result is True
