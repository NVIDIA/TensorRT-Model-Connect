# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-tests for tools/diffusion_helpers.py — silu, gelu_tanh, bundle I/O, timestep embedding.

Trace: ARCH-PIP-DIFF-001, UD-DIFF-HELPERS
Intent: Validate diffusion helper functions (silu, gelu_tanh), bundle config/weight I/O, and timestep embedding
Preconditions: diffusion_helpers module is importable; numpy available
Postconditions: Activation functions match mathematical definitions and bundle I/O correctly round-trips data
"""

from __future__ import annotations

import json
import struct

import numpy as np


def _import_diffusion_helpers():
    import importlib
    return importlib.import_module("diffusion_helpers")


# ---------------------------------------------------------------------------
# silu
# ---------------------------------------------------------------------------

class TestSilu:
    """Tests for silu(x) = x * sigmoid(x)."""

    def test_zero(self):
        mod = _import_diffusion_helpers()
        result = mod.silu(np.array([0.0]))
        assert abs(float(result[0])) < 1e-7

    def test_large_positive(self):
        """silu(x) ~ x for large positive x (sigmoid -> 1)."""
        mod = _import_diffusion_helpers()
        x = np.array([50.0])
        result = mod.silu(x)
        assert abs(float(result[0]) - 50.0) < 1e-3

    def test_large_negative(self):
        """silu(x) ~ 0 for large negative x (sigmoid -> 0)."""
        mod = _import_diffusion_helpers()
        x = np.array([-50.0])
        result = mod.silu(x)
        assert abs(float(result[0])) < 1e-3

    def test_one(self):
        """silu(1) = 1 * sigmoid(1) ~ 0.7311."""
        mod = _import_diffusion_helpers()
        result = mod.silu(np.array([1.0]))
        expected = 1.0 / (1.0 + np.exp(-1.0))  # sigmoid(1) ~ 0.7311
        assert abs(float(result[0]) - expected) < 1e-6

    def test_vectorized(self):
        mod = _import_diffusion_helpers()
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = mod.silu(x)
        # silu is monotonically increasing for x > ~-0.278
        assert result[-1] > result[-2] > result[2]
        # silu(0) = 0
        assert abs(result[2]) < 1e-7

    def test_clipping_prevents_overflow(self):
        """Values beyond +/-88 are clipped to prevent exp overflow."""
        mod = _import_diffusion_helpers()
        x = np.array([200.0, -200.0])
        result = mod.silu(x)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# gelu_tanh
# ---------------------------------------------------------------------------

class TestGeluTanh:
    """Tests for gelu_tanh(x) — GELU with tanh approximation."""

    def test_zero(self):
        mod = _import_diffusion_helpers()
        result = mod.gelu_tanh(np.array([0.0]))
        assert abs(float(result[0])) < 1e-7

    def test_large_positive(self):
        """gelu_tanh(x) ~ x for large positive x."""
        mod = _import_diffusion_helpers()
        x = np.array([10.0])
        result = mod.gelu_tanh(x)
        assert abs(float(result[0]) - 10.0) < 0.01

    def test_large_negative(self):
        """gelu_tanh(x) ~ 0 for large negative x."""
        mod = _import_diffusion_helpers()
        x = np.array([-10.0])
        result = mod.gelu_tanh(x)
        assert abs(float(result[0])) < 0.01

    def test_known_value_at_one(self):
        """gelu_tanh(1) ~ 0.8412 (standard reference value)."""
        mod = _import_diffusion_helpers()
        result = mod.gelu_tanh(np.array([1.0]))
        assert abs(float(result[0]) - 0.8412) < 0.001

    def test_negative_one(self):
        """gelu_tanh(-1) ~ -0.1588 (standard reference value)."""
        mod = _import_diffusion_helpers()
        result = mod.gelu_tanh(np.array([-1.0]))
        assert abs(float(result[0]) - (-0.1588)) < 0.001

    def test_vectorized(self):
        mod = _import_diffusion_helpers()
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        result = mod.gelu_tanh(x)
        # gelu is approximately monotonic for x > ~-0.17
        assert result[4] > result[3] > result[2]
        assert abs(result[2]) < 1e-7


# ---------------------------------------------------------------------------
# load_pp_weights / load_bundle_config — synthetic bundle
# ---------------------------------------------------------------------------

def _write_synthetic_bundle(path, config_dict, pp_weights):
    """Create a minimal .trtfb bundle with config.json and preprocessor_weights.

    Bundle format:
      8 bytes: magic "TRTFB\\x00\\x01\\x00"
      8 bytes: uint64 LE header JSON length
      N bytes: header JSON (with sections)
      body bytes: section data concatenated
    """
    # Build preprocessor_weights blob: 4-byte index JSON len + index JSON + weight data
    pp_index = {}
    blobs = b""
    for key, arr in pp_weights.items():
        arr = arr.astype(np.float32)
        pp_index[key] = {
            "shape": list(arr.shape),
            "offset": len(blobs),
        }
        blobs += arr.tobytes()
    index_json = json.dumps(pp_index).encode("utf-8")
    pp_data = struct.pack("<I", len(index_json)) + index_json + blobs

    # Build config.json blob
    config_data = json.dumps(config_dict).encode("utf-8")

    # Compute section offsets (body starts after magic + header length + header)
    sections = {
        "config.json": {"offset": 0, "size": len(config_data)},
        "preprocessor_weights": {
            "offset": len(config_data),
            "size": len(pp_data),
        },
    }

    header = json.dumps({"sections": sections}).encode("utf-8")
    magic = b"TRTFB\x00\x01\x00"

    with open(path, "wb") as f:
        f.write(magic)
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(config_data)
        f.write(pp_data)


class TestLoadBundleConfig:
    """Tests for load_bundle_config(bundle_path)."""

    def test_round_trip(self, tmp_path):
        mod = _import_diffusion_helpers()
        config = {"model_type": "test_dit", "hidden_size": 128}
        bundle_path = str(tmp_path / "test.trtfb")
        _write_synthetic_bundle(bundle_path, config, {})

        loaded = mod.load_bundle_config(bundle_path)
        assert loaded["model_type"] == "test_dit"
        assert loaded["hidden_size"] == 128

    def test_complex_config(self, tmp_path):
        mod = _import_diffusion_helpers()
        config = {
            "model_type": "synthetic_diffusion",
            "hidden_size": 3072,
            "num_layers": 24,
            "runtime_strategy": "diffusion",
            "nested": {"key": "value"},
        }
        bundle_path = str(tmp_path / "complex.trtfb")
        _write_synthetic_bundle(bundle_path, config, {})

        loaded = mod.load_bundle_config(bundle_path)
        assert loaded["num_layers"] == 24
        assert loaded["nested"]["key"] == "value"


class TestLoadPpWeights:
    """Tests for load_pp_weights(bundle_path)."""

    def test_round_trip_single_weight(self, tmp_path):
        mod = _import_diffusion_helpers()
        w = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        pp_weights = {"condition_embedder.time_embedding.0.weight": w}
        bundle_path = str(tmp_path / "test.trtfb")
        _write_synthetic_bundle(bundle_path, {"model_type": "test"}, pp_weights)

        loaded = mod.load_pp_weights(bundle_path)
        assert "condition_embedder.time_embedding.0.weight" in loaded
        np.testing.assert_allclose(
            loaded["condition_embedder.time_embedding.0.weight"], w, atol=1e-7)

    def test_round_trip_multiple_weights(self, tmp_path):
        mod = _import_diffusion_helpers()
        pp_weights = {
            "w1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "w2": np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32),
            "b1": np.array([0.1], dtype=np.float32),
        }
        bundle_path = str(tmp_path / "multi.trtfb")
        _write_synthetic_bundle(bundle_path, {"model_type": "test"}, pp_weights)

        loaded = mod.load_pp_weights(bundle_path)
        assert len(loaded) == 3
        np.testing.assert_allclose(loaded["w1"], pp_weights["w1"], atol=1e-7)
        np.testing.assert_allclose(loaded["w2"], pp_weights["w2"], atol=1e-7)
        np.testing.assert_allclose(loaded["b1"], pp_weights["b1"], atol=1e-7)

    def test_weight_shapes_preserved(self, tmp_path):
        mod = _import_diffusion_helpers()
        w = np.random.randn(8, 16).astype(np.float32)
        pp_weights = {"matrix": w}
        bundle_path = str(tmp_path / "shape.trtfb")
        _write_synthetic_bundle(bundle_path, {"model_type": "test"}, pp_weights)

        loaded = mod.load_pp_weights(bundle_path)
        assert loaded["matrix"].shape == (8, 16)


# ---------------------------------------------------------------------------
# compute_timestep_embedding
# ---------------------------------------------------------------------------

class TestComputeTimestepEmbedding:
    """Tests for compute_timestep_embedding(timestep, pp_weights, freq_dim).

    Uses synthetic preprocessor weights to validate shapes and basic
    properties of the timestep embedding computation.
    """

    @staticmethod
    def _make_pp_weights(freq_dim=256, hidden_dim=64):
        """Build minimal preprocessor weight dict for timestep embedding."""
        return {
            "condition_embedder.time_embedding.0.weight":
                np.random.randn(freq_dim, hidden_dim).astype(np.float32) * 0.02,
            "condition_embedder.time_embedding.0.bias":
                np.zeros(hidden_dim, dtype=np.float32),
            "condition_embedder.time_embedding.2.weight":
                np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02,
            "condition_embedder.time_embedding.2.bias":
                np.zeros(hidden_dim, dtype=np.float32),
            "condition_embedder.time_proj.weight":
                np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02,
            "condition_embedder.time_proj.bias":
                np.zeros(hidden_dim, dtype=np.float32),
        }

    def test_output_shapes(self):
        mod = _import_diffusion_helpers()
        pp = self._make_pp_weights(freq_dim=256, hidden_dim=64)
        temb_6d, time_embed = mod.compute_timestep_embedding(
            500.0, pp, freq_dim=256)
        assert temb_6d.shape == (1, 64)
        assert time_embed.shape == (1, 64)

    def test_timestep_zero(self):
        mod = _import_diffusion_helpers()
        pp = self._make_pp_weights()
        temb_6d, time_embed = mod.compute_timestep_embedding(0.0, pp)
        assert np.all(np.isfinite(temb_6d))
        assert np.all(np.isfinite(time_embed))

    def test_timestep_500(self):
        mod = _import_diffusion_helpers()
        pp = self._make_pp_weights()
        temb_6d, time_embed = mod.compute_timestep_embedding(500.0, pp)
        assert np.all(np.isfinite(temb_6d))
        assert np.all(np.isfinite(time_embed))

    def test_timestep_999(self):
        mod = _import_diffusion_helpers()
        pp = self._make_pp_weights()
        temb_6d, time_embed = mod.compute_timestep_embedding(999.0, pp)
        assert np.all(np.isfinite(temb_6d))
        assert np.all(np.isfinite(time_embed))

    def test_different_timesteps_produce_different_embeddings(self):
        mod = _import_diffusion_helpers()
        pp = self._make_pp_weights()
        temb_0, _ = mod.compute_timestep_embedding(0.0, pp)
        temb_500, _ = mod.compute_timestep_embedding(500.0, pp)
        temb_999, _ = mod.compute_timestep_embedding(999.0, pp)

        # Different timesteps should produce different embeddings
        assert not np.allclose(temb_0, temb_500, atol=1e-6)
        assert not np.allclose(temb_500, temb_999, atol=1e-6)
        assert not np.allclose(temb_0, temb_999, atol=1e-6)

    def test_deterministic(self):
        """Same timestep + weights should produce identical results."""
        mod = _import_diffusion_helpers()
        pp = self._make_pp_weights()
        temb_a, te_a = mod.compute_timestep_embedding(42.0, pp)
        temb_b, te_b = mod.compute_timestep_embedding(42.0, pp)
        np.testing.assert_array_equal(temb_a, temb_b)
        np.testing.assert_array_equal(te_a, te_b)

    def test_custom_freq_dim(self):
        mod = _import_diffusion_helpers()
        pp = self._make_pp_weights(freq_dim=128, hidden_dim=32)
        temb_6d, time_embed = mod.compute_timestep_embedding(
            100.0, pp, freq_dim=128)
        assert temb_6d.shape == (1, 32)
        assert time_embed.shape == (1, 32)
