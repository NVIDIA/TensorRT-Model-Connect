"""Unit tests for tools/cpu_profile.py — CPU phase breakdown tool."""

from __future__ import annotations

import importlib


def _import():
    return importlib.import_module("cpu_profile")


# ---------------------------------------------------------------------------
# Phase constants
# ---------------------------------------------------------------------------

class TestPhaseConstants:
    def test_decoder_phases(self):
        mod = _import()
        phases = mod.DECODER_PHASES
        assert "mask_build" in phases
        assert "h2d" in phases
        assert "tensor_bind" in phases
        assert "execute" in phases
        assert "d2d_cache" in phases
        assert "d2h" in phases
        assert "argmax" in phases

    def test_decoder_phases_ordered_tuple(self):
        mod = _import()
        # Phases should be a tuple (immutable, ordered)
        assert isinstance(mod.DECODER_PHASES, tuple)


# ---------------------------------------------------------------------------
# _aggregate()
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_single_phase_single_sample(self):
        mod = _import()
        phase_times = {"execute": [5.0]}
        rows = mod._aggregate(phase_times)
        assert len(rows) == 1
        assert rows[0]["phase"] == "execute"
        assert rows[0]["mean_ms"] == 5.0
        assert rows[0]["std_ms"] == 0.0
        assert rows[0]["pct"] == 100.0
        assert rows[0]["samples"] == 1

    def test_two_phases_percentages(self):
        mod = _import()
        phase_times = {"a": [1.0, 1.0], "b": [3.0, 3.0]}
        rows = mod._aggregate(phase_times)
        by_phase = {r["phase"]: r for r in rows}
        assert abs(by_phase["a"]["pct"] - 25.0) < 0.1
        assert abs(by_phase["b"]["pct"] - 75.0) < 0.1

    def test_percentages_sum_to_100(self):
        mod = _import()
        phase_times = {
            "mask_build": [0.1] * 5,
            "h2d": [0.2] * 5,
            "execute": [1.5] * 5,
            "d2h": [0.3] * 5,
        }
        rows = mod._aggregate(phase_times)
        total_pct = sum(r["pct"] for r in rows)
        assert abs(total_pct - 100.0) < 0.2

    def test_std_computed_for_multiple_samples(self):
        mod = _import()
        phase_times = {"x": [1.0, 3.0]}
        rows = mod._aggregate(phase_times)
        assert rows[0]["std_ms"] > 0

    def test_empty_phase_excluded(self):
        mod = _import()
        phase_times = {"a": [2.0], "b": []}
        rows = mod._aggregate(phase_times)
        phases = [r["phase"] for r in rows]
        assert "a" in phases
        assert "b" not in phases

    def test_mean_ms_rounded(self):
        mod = _import()
        phase_times = {"a": [1.123456789]}
        rows = mod._aggregate(phase_times)
        # Should be rounded to 4 decimal places
        assert len(str(rows[0]["mean_ms"]).split(".")[-1]) <= 4

    def test_samples_count_correct(self):
        mod = _import()
        phase_times = {"execute": [1.0] * 10}
        rows = mod._aggregate(phase_times)
        assert rows[0]["samples"] == 10


# ---------------------------------------------------------------------------
# _print_table()
# ---------------------------------------------------------------------------

class TestPrintTable:
    def _make_rows(self):
        return [
            {"phase": "execute", "mean_ms": 5.0, "std_ms": 0.1, "pct": 83.3, "samples": 20},
            {"phase": "h2d", "mean_ms": 0.8, "std_ms": 0.05, "pct": 13.3, "samples": 20},
            {"phase": "argmax", "mean_ms": 0.2, "std_ms": 0.01, "pct": 3.4, "samples": 20},
        ]

    def test_model_name_in_output(self, capsys):
        mod = _import()
        mod._print_table(self._make_rows(), "decoder", "example-org/example-decoder",
                         28, 6, 10)
        out = capsys.readouterr().out
        assert "example-org/example-decoder" in out

    def test_runner_type_in_output(self, capsys):
        mod = _import()
        mod._print_table(self._make_rows(), "decoder", "test-model", 28, 6, 10)
        out = capsys.readouterr().out
        assert "decoder" in out

    def test_all_phases_printed(self, capsys):
        mod = _import()
        mod._print_table(self._make_rows(), "decoder", "m", 28, 6, 10)
        out = capsys.readouterr().out
        assert "execute" in out
        assert "h2d" in out
        assert "argmax" in out

    def test_bottleneck_identified(self, capsys):
        mod = _import()
        mod._print_table(self._make_rows(), "decoder", "m", 28, 6, 10)
        out = capsys.readouterr().out
        assert "Bottleneck" in out
        assert "execute" in out  # execute is the largest phase

    def test_total_printed(self, capsys):
        mod = _import()
        mod._print_table(self._make_rows(), "decoder", "m", 28, 6, 10)
        out = capsys.readouterr().out
        assert "TOTAL" in out

    def test_layer_count_in_output(self, capsys):
        mod = _import()
        mod._print_table(self._make_rows(), "decoder", "m", 28, 6, 10)
        out = capsys.readouterr().out
        assert "28" in out  # num_layers

    def test_num_tokens_in_output(self, capsys):
        mod = _import()
        mod._print_table(self._make_rows(), "decoder", "m", 28, 6, 10)
        out = capsys.readouterr().out
        assert "6" in out   # prompt_tokens
        assert "10" in out  # max_new_tokens


# ---------------------------------------------------------------------------
# gpu / trt version helpers (no GPU needed — mocked)
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_get_gpu_name_returns_string(self):
        mod = _import()
        result = mod._get_gpu_name()
        assert isinstance(result, str)

    def test_get_trt_version_returns_string(self):
        mod = _import()
        result = mod._get_trt_version()
        assert isinstance(result, str)
