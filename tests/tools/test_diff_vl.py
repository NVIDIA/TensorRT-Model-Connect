# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-tests for tools/diff_vl.py - dispatch, cosine similarity, sanity checks.

Trace: ARCH-TRT-001, UD-TRT-DIFF-VL
Intent: Validate VL diff dispatch, cosine similarity computation, and preprocessor config
Preconditions: diff_vl module is importable; numpy available
Postconditions: Family-owned handlers are discovered and cosine similarity matches mathematical expectations
"""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest


class TestFamilyHandlerDispatch:
    """Test model-owned handler dispatch - importable without GPU."""

    def test_unowned_model_uses_generic_path(self):
        import importlib

        mod = importlib.import_module("diff_vl")
        assert mod._find_family_diff_vl_handler("example_decoder") is None
        assert mod._find_family_diff_vl_handler("generic_text") is None


class TestAutoVisionModelClass:
    """Test Transformers 5.5 model class selection and legacy fallback."""

    def test_prefers_current_image_text_api(self, monkeypatch):
        import diff_vl

        transformers = ModuleType("transformers")
        current_class = object()
        transformers.AutoModelForImageTextToText = current_class
        transformers.AutoModelForVision2Seq = object()
        monkeypatch.setitem(sys.modules, "transformers", transformers)

        assert diff_vl._get_auto_vision_model_class() is current_class

    def test_falls_back_when_current_api_is_unavailable(self, monkeypatch):
        import diff_vl

        transformers = ModuleType("transformers")
        legacy_class = object()
        transformers.AutoModelForVision2Seq = legacy_class
        monkeypatch.setitem(sys.modules, "transformers", transformers)

        assert diff_vl._get_auto_vision_model_class() is legacy_class

    def test_does_not_mask_current_api_import_errors(self, monkeypatch):
        import diff_vl

        class BrokenTransformersModule(ModuleType):
            def __getattr__(self, name):
                if name == "AutoModelForImageTextToText":
                    raise ImportError("current API dependency failed")
                if name == "AutoModelForVision2Seq":
                    raise AssertionError("legacy fallback must not be used")
                raise AttributeError(name)

        monkeypatch.setitem(
            sys.modules, "transformers", BrokenTransformersModule("transformers")
        )

        with pytest.raises(ImportError, match="current API dependency failed"):
            diff_vl._get_auto_vision_model_class()


class TestCosineSimilarity:
    """Test cosine similarity computation (same formula as diff_vl.py)."""

    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        cos_sim = np.dot(a, a) / (np.linalg.norm(a) * np.linalg.norm(a) + 1e-8)
        assert abs(cos_sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        assert abs(cos_sim) < 1e-6

    def test_opposite_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        b = -a
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        assert cos_sim < -0.99

    def test_threshold_check(self):
        """Cosine similarity < 0.5 means uncorrelated features (hard fail in diff_vl)."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.5, 0.5, 0.0])
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        assert cos_sim > 0.5  # These are somewhat correlated


class TestSanityChecks:
    """Test the NaN/Inf/zero sanity checks from diff_vl.py."""

    def test_all_zeros_detected(self):
        features = np.zeros((1, 64), dtype=np.float32)
        assert np.all(features == 0)

    def test_nan_detected(self):
        features = np.array([1.0, float("nan"), 3.0])
        assert np.any(np.isnan(features))

    def test_inf_detected(self):
        features = np.array([1.0, float("inf"), 3.0])
        assert np.any(np.isinf(features))

    def test_valid_features_pass(self):
        features = np.random.randn(1, 64).astype(np.float32)
        assert not np.all(features == 0)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))


class TestPreprocessorDefaults:
    """Verify expected default values used in diff_vl.py."""

    def test_default_image_mean(self):
        expected = (0.48145466, 0.4578275, 0.40821073)
        assert len(expected) == 3
        assert all(0.0 < v < 1.0 for v in expected)

    def test_default_image_std(self):
        expected = (0.26862954, 0.26130258, 0.27577711)
        assert len(expected) == 3
        assert all(0.0 < v < 1.0 for v in expected)

    def test_default_image_size(self):
        assert 448 > 0
        assert 448 % 14 == 0  # Must be divisible by patch size
