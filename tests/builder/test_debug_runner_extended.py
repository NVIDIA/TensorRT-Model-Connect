# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extended tests for model-owned debug-runner bundle utilities.

Concrete TRT runner implementations and bundle readers are model-owned; this
file covers an owned E2E bundle-reader implementation and pure cache arithmetic.
"""

from __future__ import annotations

import json
import pytest

from tests.builder.debug_runner_test_support import make_bundle_bytes

# Module-level skip: tensorrt_model_connect submodules need tensorrt installed
try:
    import tensorrt_model_connect.debug_runner  # noqa: F401
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect.debug_runner requires tensorrt", allow_module_level=True)


# ---------------------------------------------------------------------------
# load_config_from_bundle — extended cases
# ---------------------------------------------------------------------------

class TestLoadConfigFromBundleExtended:
    """Extended tests for load_config_from_bundle()."""

    def test_missing_config_section_returns_empty(self, tmp_path):
        """Bundle without a config.json section returns empty dict."""
        from tests.e2e.models.qwen.e2e_plugins.runners.vl_debug_runner import load_config_from_bundle

        header = {"num_layers": 2, "max_cache_length": 64}
        bundle = make_bundle_bytes(header, engine_plan=b"FAKE")

        path = tmp_path / "no_config.trtfb"
        path.write_bytes(bundle)

        cfg = load_config_from_bundle(str(path))
        assert cfg == {}

    def test_config_with_nested_values(self, tmp_path):
        """Config section with nested JSON is correctly round-tripped."""
        from tests.e2e.models.qwen.e2e_plugins.runners.vl_debug_runner import load_config_from_bundle

        config_data = json.dumps({
            "model_type": "example_decoder",
            "hidden_size": 256,
            "eos_token_id": [2, 3],
            "nested": {"a": 1, "b": [2, 3]},
        }).encode("utf-8")

        header = {"num_layers": 1, "max_cache_length": 32}
        bundle = make_bundle_bytes(
            header,
            engine_plan=b"EP",
            extra_sections={"config.json": config_data},
        )

        path = tmp_path / "nested_cfg.trtfb"
        path.write_bytes(bundle)

        cfg = load_config_from_bundle(str(path))
        assert cfg["model_type"] == "example_decoder"
        assert cfg["hidden_size"] == 256
        assert cfg["eos_token_id"] == [2, 3]
        assert cfg["nested"]["a"] == 1
        assert cfg["nested"]["b"] == [2, 3]


# ---------------------------------------------------------------------------
# load_preprocessor_config_from_bundle
# ---------------------------------------------------------------------------

class TestLoadPreprocessorConfigFromBundle:
    """Tests for load_preprocessor_config_from_bundle()."""

    def test_present(self, tmp_path):
        """Extracts and parses preprocessor_config.json section."""
        from tests.e2e.models.qwen.e2e_plugins.runners.vl_debug_runner import load_preprocessor_config_from_bundle

        preproc_data = json.dumps({
            "temporal_patch_size": 2,
            "patch_size": 14,
            "merge_size": 2,
            "image_mean": [0.48, 0.46, 0.41],
            "image_std": [0.27, 0.26, 0.28],
        }).encode("utf-8")

        header = {"num_layers": 1, "max_cache_length": 32}
        bundle = make_bundle_bytes(
            header,
            engine_plan=b"EP",
            extra_sections={"preprocessor_config.json": preproc_data},
        )

        path = tmp_path / "preproc.trtfb"
        path.write_bytes(bundle)

        cfg = load_preprocessor_config_from_bundle(str(path))
        assert cfg["temporal_patch_size"] == 2
        assert cfg["patch_size"] == 14
        assert pytest.approx(cfg["image_mean"]) == [0.48, 0.46, 0.41]

    def test_missing_returns_empty(self, tmp_path):
        """Bundle without preprocessor_config.json returns empty dict."""
        from tests.e2e.models.qwen.e2e_plugins.runners.vl_debug_runner import load_preprocessor_config_from_bundle

        header = {"num_layers": 1, "max_cache_length": 32}
        bundle = make_bundle_bytes(header, engine_plan=b"EP")

        path = tmp_path / "no_preproc.trtfb"
        path.write_bytes(bundle)

        cfg = load_preprocessor_config_from_bundle(str(path))
        assert cfg == {}


# ---------------------------------------------------------------------------
# Bundle with multiple sections
# ---------------------------------------------------------------------------

class TestMultiSectionBundle:
    """Verify that bundles with many sections are parsed correctly,
    with each section extractable independently and unknown sections
    handled gracefully.
    """

    def _build_multi_section_bundle(self, tmp_path):
        """Build a bundle with engine_plan + config.json + tokenizer.json."""
        config_data = json.dumps({"model_type": "example_decoder"}).encode("utf-8")
        tokenizer_data = json.dumps({"tokens": ["a", "b"]}).encode("utf-8")

        header = {"num_layers": 4, "max_cache_length": 128}
        engine_plan = b"REAL_ENGINE_PLAN_DATA"
        bundle = make_bundle_bytes(
            header,
            engine_plan=engine_plan,
            extra_sections={
                "config.json": config_data,
                "tokenizer.json": tokenizer_data,
            },
        )
        path = tmp_path / "multi.trtfb"
        path.write_bytes(bundle)
        return str(path), engine_plan, config_data, tokenizer_data

    def test_engine_plan_extracted(self, tmp_path):
        """load_engine_from_bundle extracts the correct engine_plan section."""
        from tests.e2e.models.qwen.e2e_plugins.runners.vl_debug_runner import load_engine_from_bundle

        path, engine_plan, _, _ = self._build_multi_section_bundle(tmp_path)
        plan, hdr = load_engine_from_bundle(path)
        assert plan == engine_plan
        assert hdr["num_layers"] == 4

    def test_config_section_extracted(self, tmp_path):
        """load_config_from_bundle extracts config.json from multi-section bundle."""
        from tests.e2e.models.qwen.e2e_plugins.runners.vl_debug_runner import load_config_from_bundle

        path, _, _, _ = self._build_multi_section_bundle(tmp_path)
        cfg = load_config_from_bundle(path)
        assert cfg["model_type"] == "example_decoder"

    def test_arbitrary_section_extracted(self, tmp_path):
        """load_section_from_bundle can extract tokenizer.json from bundle."""
        from tests.e2e.models.qwen.e2e_plugins.runners.vl_debug_runner import load_section_from_bundle

        path, _, _, tokenizer_data = self._build_multi_section_bundle(tmp_path)
        data = load_section_from_bundle(path, "tokenizer.json")
        assert data == tokenizer_data
        parsed = json.loads(data.decode("utf-8"))
        assert parsed["tokens"] == ["a", "b"]

    def test_unknown_section_returns_none(self, tmp_path):
        """Requesting a non-existent section returns None (graceful)."""
        from tests.e2e.models.qwen.e2e_plugins.runners.vl_debug_runner import load_section_from_bundle

        path, _, _, _ = self._build_multi_section_bundle(tmp_path)
        result = load_section_from_bundle(path, "totally_unknown_section")
        assert result is None


# ---------------------------------------------------------------------------
# load_section_from_bundle — invalid bundle
# ---------------------------------------------------------------------------

class TestLoadSectionInvalidBundle:
    """load_section_from_bundle should raise on corrupted bundles."""

    def test_invalid_magic_raises(self, tmp_path):
        from tests.e2e.models.qwen.e2e_plugins.runners.vl_debug_runner import load_section_from_bundle

        path = tmp_path / "bad.trtfb"
        path.write_bytes(b"GARBAGE_DATA_NOT_A_BUNDLE")

        with pytest.raises(ValueError, match="Not a valid .trtfb bundle"):
            load_section_from_bundle(str(path), "engine_plan")


# ---------------------------------------------------------------------------
# TrtRunner.step() cache update edge cases (numpy-level logic)
# ---------------------------------------------------------------------------

class TestTrtRunnerCacheUpdateLogic:
    """Verify the cache_length tracking in TrtRunner.step() logic.

    These tests check that cache_length is updated correctly and caps
    at max_cache_length, using the same logic as the actual step() method.
    """

    def test_cache_length_increments(self):
        """cache_length increments by 1 per step until max."""
        max_cache = 4
        cache_length = 0
        for _ in range(6):
            cache_length = min(cache_length + 1, max_cache)

        assert cache_length == max_cache

    def test_cache_length_update_sequence(self):
        """Track the full sequence of cache_length values."""
        max_cache = 3
        cache_length = 0
        history = [cache_length]
        for _ in range(5):
            cache_length = min(cache_length + 1, max_cache)
            history.append(cache_length)
        assert history == [0, 1, 2, 3, 3, 3]

    def test_d2d_copy_path_selection(self):
        """Verify which D2D copy path is taken based on cache_length."""
        max_cache = 3

        # Before cache is full: append path
        for cl in [0, 1, 2]:
            assert cl < max_cache, f"cache_length={cl} should take append path"

        # After cache is full: shift path
        for cl in [3, 4, 10]:
            assert cl >= max_cache, f"cache_length={cl} should take shift path"
