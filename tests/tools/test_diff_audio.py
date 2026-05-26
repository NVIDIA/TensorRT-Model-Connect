"""Self-tests for tools/diff_audio.py — compute_energy, WAV I/O, token_stats, arg parsing.

Trace: ARCH-PIP-AUD-001, UD-AUD-DIFF
Intent: Validate audio diff tool compute_energy, WAV read/write round-trip, token statistics, and arg parsing
Preconditions: diff_audio module is importable via tools/ path; numpy available
Postconditions: Energy computation is correct for edge cases, WAV I/O round-trips preserve sample data
"""

from __future__ import annotations

import os
import struct
import tempfile

import numpy as np
import pytest


def _import_diff_audio():
    import importlib
    return importlib.import_module("diff_audio")


# ---------------------------------------------------------------------------
# compute_energy
# ---------------------------------------------------------------------------

class TestComputeEnergy:
    """Tests for compute_energy(waveform) — RMS energy."""

    def test_zero_waveform(self):
        mod = _import_diff_audio()
        result = mod.compute_energy(np.zeros(100))
        assert result == 0.0

    def test_empty_waveform(self):
        mod = _import_diff_audio()
        result = mod.compute_energy(np.array([]))
        assert result == 0.0

    def test_dc_offset(self):
        """Constant signal [1,1,1,1] has RMS = 1.0."""
        mod = _import_diff_audio()
        waveform = np.ones(4, dtype=np.float32)
        result = mod.compute_energy(waveform)
        assert abs(result - 1.0) < 1e-6

    def test_sine_wave(self):
        """Sine wave has RMS ~ 1/sqrt(2) ~ 0.707."""
        mod = _import_diff_audio()
        t = np.linspace(0, 2 * np.pi, 10000, endpoint=False)
        waveform = np.sin(t).astype(np.float32)
        result = mod.compute_energy(waveform)
        assert abs(result - 1.0 / np.sqrt(2)) < 0.01

    def test_negative_dc(self):
        """RMS of [-2, -2, -2, -2] should be 2.0."""
        mod = _import_diff_audio()
        waveform = np.full(4, -2.0, dtype=np.float32)
        result = mod.compute_energy(waveform)
        assert abs(result - 2.0) < 1e-6

    def test_single_sample(self):
        mod = _import_diff_audio()
        waveform = np.array([0.5], dtype=np.float32)
        result = mod.compute_energy(waveform)
        assert abs(result - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# read_wav_f32 / write_wav_f32 — round-trip
# ---------------------------------------------------------------------------

class TestWavRoundTrip:
    """Tests for write_wav_f32 and read_wav_f32 — WAV I/O."""

    def test_round_trip(self, tmp_path):
        mod = _import_diff_audio()
        samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        wav_path = str(tmp_path / "test.wav")

        mod.write_wav_f32(wav_path, samples, sample_rate=24000)
        read_samples, sr = mod.read_wav_f32(wav_path)

        assert sr == 24000
        assert len(read_samples) == len(samples)
        np.testing.assert_allclose(read_samples, samples, atol=1e-7)

    def test_round_trip_different_sample_rate(self, tmp_path):
        mod = _import_diff_audio()
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        wav_path = str(tmp_path / "test_sr.wav")

        mod.write_wav_f32(wav_path, samples, sample_rate=44100)
        read_samples, sr = mod.read_wav_f32(wav_path)

        assert sr == 44100
        np.testing.assert_allclose(read_samples, samples, atol=1e-7)

    def test_round_trip_empty(self, tmp_path):
        mod = _import_diff_audio()
        samples = np.array([], dtype=np.float32)
        wav_path = str(tmp_path / "empty.wav")

        mod.write_wav_f32(wav_path, samples, sample_rate=24000)
        read_samples, sr = mod.read_wav_f32(wav_path)

        assert sr == 24000
        assert len(read_samples) == 0

    def test_round_trip_large(self, tmp_path):
        """Round-trip a 1-second waveform at 24kHz."""
        mod = _import_diff_audio()
        samples = np.random.randn(24000).astype(np.float32)
        wav_path = str(tmp_path / "large.wav")

        mod.write_wav_f32(wav_path, samples, sample_rate=24000)
        read_samples, sr = mod.read_wav_f32(wav_path)

        assert sr == 24000
        assert len(read_samples) == 24000
        np.testing.assert_allclose(read_samples, samples, atol=1e-7)

    def test_invalid_file_not_riff(self, tmp_path):
        """Non-RIFF file should raise ValueError."""
        mod = _import_diff_audio()
        bad_path = str(tmp_path / "bad.wav")
        with open(bad_path, "wb") as f:
            f.write(b"NOT_RIFF_DATA_HERE")

        with pytest.raises(ValueError, match="Not a RIFF file"):
            mod.read_wav_f32(bad_path)

    def test_invalid_file_riff_not_wave(self, tmp_path):
        """RIFF file without WAVE tag should raise ValueError."""
        mod = _import_diff_audio()
        bad_path = str(tmp_path / "bad_wave.wav")
        with open(bad_path, "wb") as f:
            f.write(b"RIFF")
            f.write(struct.pack("<I", 4))
            f.write(b"AVI ")

        with pytest.raises(ValueError, match="Not a WAVE file"):
            mod.read_wav_f32(bad_path)


# ---------------------------------------------------------------------------
# token_stats
# ---------------------------------------------------------------------------

class TestTokenStats:
    """Tests for token_stats(tokens, label) — basic token statistics."""

    def test_empty_tokens(self, capsys):
        mod = _import_diff_audio()
        stats = mod.token_stats(np.array([], dtype=np.int32), "test")
        assert stats["count"] == 0
        captured = capsys.readouterr().out
        assert "EMPTY" in captured

    def test_basic_stats(self, capsys):
        mod = _import_diff_audio()
        tokens = np.array([10, 20, 30, 20], dtype=np.int32)
        stats = mod.token_stats(tokens, "test")
        assert stats["count"] == 4
        assert stats["min"] == 10
        assert stats["max"] == 30
        assert stats["unique"] == 3
        assert stats["entropy"] > 0

    def test_single_token(self, capsys):
        mod = _import_diff_audio()
        tokens = np.array([42], dtype=np.int32)
        stats = mod.token_stats(tokens, "test")
        assert stats["count"] == 1
        assert stats["min"] == 42
        assert stats["max"] == 42
        assert stats["unique"] == 1

    def test_uniform_tokens_high_entropy(self, capsys):
        """Many unique tokens should have higher entropy than few."""
        mod = _import_diff_audio()
        uniform = np.arange(100, dtype=np.int32)
        stats_uniform = mod.token_stats(uniform, "uniform")
        _ = capsys.readouterr()

        repeated = np.zeros(100, dtype=np.int32)
        stats_repeated = mod.token_stats(repeated, "repeated")
        _ = capsys.readouterr()

        assert stats_uniform["entropy"] > stats_repeated["entropy"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

class TestArgParsing:
    """Tests for diff_audio.py argument parser via main() structure."""

    def test_module_has_main(self):
        mod = _import_diff_audio()
        assert callable(mod.main)

    def test_module_has_stage_functions(self):
        mod = _import_diff_audio()
        assert callable(mod.stage1_cpp_smoke_test)
        assert callable(mod.stage2_token_comparison)
        assert callable(mod.stage3_codec_comparison)
        assert callable(mod.stage4_greedy_parity)
        assert callable(mod.run_as_diff_test)

    def test_module_has_helper_functions(self):
        mod = _import_diff_audio()
        assert callable(mod.compute_energy)
        assert callable(mod.read_wav_f32)
        assert callable(mod.write_wav_f32)
        assert callable(mod.read_token_file)
        assert callable(mod.token_stats)


# ---------------------------------------------------------------------------
# read_token_file
# ---------------------------------------------------------------------------

class TestReadTokenFile:
    """Tests for read_token_file(path) — newline-delimited token file reader."""

    def test_basic_tokens(self, tmp_path):
        mod = _import_diff_audio()
        path = str(tmp_path / "tokens.txt")
        with open(path, "w") as f:
            f.write("10\n20\n30\n")

        tokens = mod.read_token_file(path)
        np.testing.assert_array_equal(tokens, [10, 20, 30])
        assert tokens.dtype == np.int32

    def test_empty_file(self, tmp_path):
        mod = _import_diff_audio()
        path = str(tmp_path / "empty.txt")
        with open(path, "w") as f:
            pass

        tokens = mod.read_token_file(path)
        assert len(tokens) == 0

    def test_blank_lines_skipped(self, tmp_path):
        mod = _import_diff_audio()
        path = str(tmp_path / "blanks.txt")
        with open(path, "w") as f:
            f.write("5\n\n10\n\n15\n")

        tokens = mod.read_token_file(path)
        np.testing.assert_array_equal(tokens, [5, 10, 15])


class TestFrameworkAdapter:
    """Tests for diff framework adapter behavior that does not require GPU."""

    def test_run_as_diff_test_skips_without_bundle(self):
        from diff_framework.protocol import TestContext

        mod = _import_diff_audio()
        result = mod.run_as_diff_test(TestContext(
            model="suno/bark-small",
            runtime_strategy="text_to_audio_bark",
            bundle_path=None,
            binary_path="./build/trtmc",
        ))
        assert result.status == "SKIP"
        assert result.test_name == "bark_audio_pipeline"
        assert "bundle" in result.message
