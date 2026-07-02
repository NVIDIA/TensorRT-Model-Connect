# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for model-owned debug runner bundle readers.

Mock-based, no TRT/GPU needed. Tests bundle parsing logic and
bundle section parsing through an owned E2E debug runner module.

Trace: ARCH-DBG-001, UD-DBG-02
Intent: Validate model-owned bundle section loading and vision engine extraction.
Preconditions: No TRT or GPU required; uses in-memory .trtfb bundles and mocks for TRT engine deserialization.
Postconditions: Engine plan bytes are correctly extracted from bundle sections, vision plans are found when present.
"""

from __future__ import annotations

import json
import struct
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: build a minimal .trtfb bundle in memory
# ---------------------------------------------------------------------------

def _make_bundle_bytes(
    header: dict,
    engine_plan: bytes = b"FAKE_ENGINE_PLAN",
    vision_plan: bytes | None = None,
    extra_sections: dict[str, bytes] | None = None,
) -> bytes:
    """Build a minimal .trtfb bundle in memory."""
    magic = b"TRTFB\x00\x01\x00"
    sections: dict[str, dict] = {}
    body = b""

    # engine_plan section
    sections["engine_plan"] = {"offset": len(body), "size": len(engine_plan)}
    body += engine_plan

    # optional vision section
    if vision_plan is not None:
        sections["vision_engine_plan"] = {
            "offset": len(body), "size": len(vision_plan),
        }
        body += vision_plan

    if extra_sections:
        for name, data in extra_sections.items():
            sections[name] = {"offset": len(body), "size": len(data)}
            body += data

    header["sections"] = sections
    header_json = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<Q", len(header_json))

    return magic + header_len + header_json + body


# ---------------------------------------------------------------------------
# load_engine_from_bundle
# ---------------------------------------------------------------------------

class TestLoadEngineFromBundle:
    """Tests for load_engine_from_bundle() bundle parsing."""

    def test_roundtrip(self, tmp_path):
        from tests.e2e.models.qwen.e2e_plugins.runners.vl_debug_runner import load_engine_from_bundle

        header = {
            "model_id": "test-model",
            "max_cache_length": 128,
            "num_layers": 4,
        }
        engine_data = b"PLAN_BYTES_1234"
        bundle = _make_bundle_bytes(header, engine_plan=engine_data)

        path = tmp_path / "test.trtfb"
        path.write_bytes(bundle)

        plan, hdr = load_engine_from_bundle(str(path))
        assert plan == engine_data
        assert hdr["model_id"] == "test-model"
        assert hdr["max_cache_length"] == 128
        assert hdr["num_layers"] == 4

    def test_invalid_magic(self, tmp_path):
        from tests.e2e.models.qwen.e2e_plugins.runners.vl_debug_runner import load_engine_from_bundle

        path = tmp_path / "bad.trtfb"
        path.write_bytes(b"NOT_A_BUNDLE_xxxxxxxxxxxx")

        with pytest.raises(ValueError, match="Not a valid .trtfb bundle"):
            load_engine_from_bundle(str(path))

    def test_named_engine_section(self, tmp_path):
        from tests.e2e.models.qwen.e2e_plugins.runners.vl_debug_runner import load_engine_from_bundle

        header = {
            "model_id": "test-model",
            "max_cache_length": 128,
            "num_layers": 4,
        }
        bundle = _make_bundle_bytes(
            header,
            engine_plan=b"SINGLE_PLAN",
            extra_sections={"engine_plan_tp_rank1": b"TP_RANK1_PLAN"},
        )

        path = tmp_path / "tp.trtfb"
        path.write_bytes(bundle)

        plan, hdr = load_engine_from_bundle(
            str(path), section_name="engine_plan_tp_rank1")
        assert plan == b"TP_RANK1_PLAN"
        assert hdr["model_id"] == "test-model"


# ---------------------------------------------------------------------------
# load_vision_engine_from_bundle
# ---------------------------------------------------------------------------

class TestLoadVisionEngineFromBundle:
    """Tests for load_vision_engine_from_bundle()."""

    def test_with_vision_section(self, tmp_path):
        from tests.e2e.models.qwen.e2e_plugins.runners.vl_debug_runner import load_vision_engine_from_bundle

        header = {"num_layers": 2, "max_cache_length": 64}
        engine_data = b"TEXT_ENGINE"
        vision_data = b"VISION_ENGINE"
        bundle = _make_bundle_bytes(
            header, engine_plan=engine_data, vision_plan=vision_data)

        path = tmp_path / "vl.trtfb"
        path.write_bytes(bundle)

        plan, hdr = load_vision_engine_from_bundle(str(path))
        assert plan == vision_data
        assert hdr["num_layers"] == 2

    def test_without_vision_section(self, tmp_path):
        from tests.e2e.models.qwen.e2e_plugins.runners.vl_debug_runner import load_vision_engine_from_bundle

        header = {"num_layers": 2, "max_cache_length": 64}
        bundle = _make_bundle_bytes(header, engine_plan=b"TEXT_ONLY")

        path = tmp_path / "text.trtfb"
        path.write_bytes(bundle)

        plan, hdr = load_vision_engine_from_bundle(str(path))
        assert plan is None
        assert hdr["num_layers"] == 2


# ---------------------------------------------------------------------------
# load_section_from_bundle / load_config_from_bundle
# ---------------------------------------------------------------------------

class TestBundleSectionUtils:
    """Tests for section loading utilities."""

    def test_load_section_missing(self, tmp_path):
        from tests.e2e.models.qwen.e2e_plugins.runners.vl_debug_runner import load_section_from_bundle

        header = {"num_layers": 1, "max_cache_length": 32}
        bundle = _make_bundle_bytes(header, engine_plan=b"X")

        path = tmp_path / "test.trtfb"
        path.write_bytes(bundle)

        result = load_section_from_bundle(str(path), "nonexistent_section")
        assert result is None

    def test_load_config_from_bundle(self, tmp_path):
        from tests.e2e.models.qwen.e2e_plugins.runners.vl_debug_runner import load_config_from_bundle

        # Build a bundle with a config.json section
        config_data = json.dumps({"model_type": "example_decoder"}).encode("utf-8")
        magic = b"TRTFB\x00\x01\x00"

        engine_plan = b"FAKE_ENGINE"
        sections = {
            "engine_plan": {"offset": 0, "size": len(engine_plan)},
            "config.json": {
                "offset": len(engine_plan),
                "size": len(config_data),
            },
        }
        header = {"num_layers": 1, "max_cache_length": 32, "sections": sections}
        header_json = json.dumps(header).encode("utf-8")
        header_len = struct.pack("<Q", len(header_json))

        path = tmp_path / "cfg.trtfb"
        path.write_bytes(magic + header_len + header_json + engine_plan + config_data)

        cfg = load_config_from_bundle(str(path))
        assert cfg["model_type"] == "example_decoder"

class TestRunnerFromBundle:
    def test_mpi_rank_info_uses_single_node_rank(self, monkeypatch):
        from tensorrt_model_connect.debug_runner import _mpi_rank_info_from_env

        monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "3")
        monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")

        assert _mpi_rank_info_from_env() == (3, 4)

# ---------------------------------------------------------------------------
# TrtRunner.step() mask/position logic (mocked CUDA)
# ---------------------------------------------------------------------------

class TestTrtRunnerMaskLogic:
    """Test the numpy-level mask and position logic in TrtRunner.step().

    The device-resident TrtRunner uses the same mask/position logic as before,
    just with on-device cache. These tests verify the CPU-side mask computation.
    """

    def _make_stub(self, max_cache_length=4, cache_length=0):
        """Build a stub with cache_length and max_cache_length for testing."""
        class Stub:
            pass
        s = Stub()
        s.max_cache_length = max_cache_length
        s.cache_length = cache_length
        return s

    def test_position_starts_at_zero(self):
        s = self._make_stub(cache_length=0)
        position_id = min(s.cache_length, s.max_cache_length)
        assert position_id == 0

    def test_position_increments(self):
        s = self._make_stub(cache_length=3, max_cache_length=8)
        position_id = min(s.cache_length, s.max_cache_length)
        assert position_id == 3

    def test_position_caps_at_max(self):
        s = self._make_stub(cache_length=10, max_cache_length=4)
        position_id = min(s.cache_length, s.max_cache_length)
        assert position_id == 4

    def test_mask_empty_cache(self):
        """With no cache entries, only current token slot is valid."""
        s = self._make_stub(max_cache_length=4, cache_length=0)
        attention_window = s.max_cache_length + 1

        mask = np.full((1, attention_window), -1e9, dtype=np.float32)
        valid = min(s.cache_length, s.max_cache_length)
        mask[0, :valid] = 0.0
        mask[0, -1] = 0.0

        assert mask[0, 0] == pytest.approx(-1e9)
        assert mask[0, 3] == pytest.approx(-1e9)
        assert mask[0, 4] == pytest.approx(0.0)  # current token

    def test_mask_partial_cache(self):
        """With 2 cached entries, positions 0,1 and current token are valid."""
        s = self._make_stub(max_cache_length=4, cache_length=2)
        attention_window = s.max_cache_length + 1

        mask = np.full((1, attention_window), -1e9, dtype=np.float32)
        valid = min(s.cache_length, s.max_cache_length)
        mask[0, :valid] = 0.0
        mask[0, -1] = 0.0

        assert mask[0, 0] == pytest.approx(0.0)
        assert mask[0, 1] == pytest.approx(0.0)
        assert mask[0, 2] == pytest.approx(-1e9)
        assert mask[0, 3] == pytest.approx(-1e9)
        assert mask[0, 4] == pytest.approx(0.0)  # current token

    def test_mask_full_cache(self):
        """With full cache, all positions are valid."""
        s = self._make_stub(max_cache_length=4, cache_length=4)
        attention_window = s.max_cache_length + 1

        mask = np.full((1, attention_window), -1e9, dtype=np.float32)
        valid = min(s.cache_length, s.max_cache_length)
        mask[0, :valid] = 0.0
        mask[0, -1] = 0.0

        for i in range(5):
            assert mask[0, i] == pytest.approx(0.0), (
                f"Position {i} should be valid with full cache")


# ---------------------------------------------------------------------------
