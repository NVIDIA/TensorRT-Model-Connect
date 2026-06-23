"""Self-tests for tools/diff_logits.py — battery prompts, tolerance, compare_logits.

Trace: ARCH-TRT-001, UD-TRT-DIFF-LOGITS
Intent: Validate logit comparison tool battery prompts, tolerance thresholds, and per-step diff metrics
Preconditions: diff_logits module is importable; synthetic logit arrays are available
Postconditions: Identical logits produce zero diff, within-tolerance logits pass, and battery prompts are well-formed
"""

from __future__ import annotations

import numpy as np
import pytest


def _import_diff_logits():
    import importlib
    return importlib.import_module("diff_logits")


class TestStandardPrompts:
    """Battery prompt list sanity checks."""

    def test_battery_has_entries(self):
        mod = _import_diff_logits()
        assert len(mod.STANDARD_PROMPTS) >= 3

    def test_battery_entries_are_tuples(self):
        mod = _import_diff_logits()
        for entry in mod.STANDARD_PROMPTS:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            label, prompt = entry
            assert isinstance(label, str) and len(label) > 0
            assert isinstance(prompt, str) and len(prompt) > 0

    def test_battery_labels_unique(self):
        mod = _import_diff_logits()
        labels = [label for label, _ in mod.STANDARD_PROMPTS]
        assert len(labels) == len(set(labels))


class TestFamilyHandlerDispatch:
    """Model-owned logit diff hook dispatch sanity checks."""

    def test_unowned_model_uses_default_paths(self):
        mod = _import_diff_logits()
        assert mod._find_family_diff_logits_handler("example_decoder") is None
        assert mod._find_family_diff_logits_handler("generic_text") is None


class TestCompareLogits:
    """Tests for compare_logits() — pure numpy, no GPU."""

    def test_identical_logits_zero_diff(self):
        mod = _import_diff_logits()
        logits = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        max_diff, lines, step_metrics = mod.compare_logits(
            logits, logits, atol=1e-3)
        assert max_diff == 0.0
        assert len(lines) == 2

    def test_small_diff_within_tolerance(self):
        mod = _import_diff_logits()
        trt = [np.array([1.0, 2.0, 3.0])]
        hf = [np.array([1.0001, 2.0001, 3.0001])]
        max_diff, lines, step_metrics = mod.compare_logits(trt, hf, atol=1e-3)
        assert max_diff < 1e-3
        assert "argmax_match=Y" in lines[0]

    def test_large_diff_exceeds_tolerance(self):
        mod = _import_diff_logits()
        trt = [np.array([1.0, 2.0, 3.0])]
        hf = [np.array([1.0, 2.0, 5.0])]
        max_diff, lines, step_metrics = mod.compare_logits(trt, hf, atol=1e-3)
        assert max_diff > 1e-3
        assert "argmax_match=Y" in lines[0]  # argmax still 2 for both

    def test_argmax_mismatch(self):
        mod = _import_diff_logits()
        trt = [np.array([10.0, 1.0, 1.0])]
        hf = [np.array([1.0, 1.0, 10.0])]
        max_diff, lines, step_metrics = mod.compare_logits(trt, hf, atol=1e-3)
        assert "argmax_match=N" in lines[0]

    def test_shape_mismatch_reported(self):
        mod = _import_diff_logits()
        trt = [np.array([1.0, 2.0])]
        hf = [np.array([1.0, 2.0, 3.0])]
        max_diff, lines, step_metrics = mod.compare_logits(trt, hf, atol=1e-3)
        assert "shape mismatch" in lines[0]

    def test_top_k_overlap(self):
        mod = _import_diff_logits()
        vocab = 100
        logits = np.random.randn(vocab).astype(np.float32)
        trt = [logits]
        hf = [logits.copy()]
        max_diff, lines, step_metrics = mod.compare_logits(
            trt, hf, atol=1e-3, top_k=5)
        assert "top5_overlap=5/5" in lines[0]

    def test_different_length_sequences(self):
        mod = _import_diff_logits()
        trt = [np.array([1.0, 2.0])] * 3
        hf = [np.array([1.0, 2.0])] * 5
        max_diff, lines, step_metrics = mod.compare_logits(trt, hf, atol=1e-3)
        assert len(lines) == 3  # min(3, 5)

    def test_step_metrics_returned(self):
        mod = _import_diff_logits()
        trt = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        hf = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        _, _, step_metrics = mod.compare_logits(trt, hf, atol=1e-3)
        assert len(step_metrics) == 2
        for sm in step_metrics:
            assert "cosine_sim" in sm
            assert "argmax_match" in sm
            assert "mean_abs_diff" in sm
            assert "max_abs_diff" in sm

    def test_step_metrics_skips_shape_mismatch(self):
        mod = _import_diff_logits()
        trt = [np.array([1.0, 2.0])]
        hf = [np.array([1.0, 2.0, 3.0])]
        _, _, step_metrics = mod.compare_logits(trt, hf, atol=1e-3)
        assert len(step_metrics) == 0


class TestCosimeSimilarity:
    """Tests for _cosine_similarity()."""

    def test_identical_vectors(self):
        mod = _import_diff_logits()
        a = np.array([1.0, 2.0, 3.0])
        assert mod._cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        mod = _import_diff_logits()
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert mod._cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_zero_vector_returns_zero(self):
        mod = _import_diff_logits()
        a = np.array([1.0, 2.0, 3.0])
        z = np.zeros(3)
        assert mod._cosine_similarity(a, z) == 0.0
        assert mod._cosine_similarity(z, a) == 0.0


class TestJsonOutput:
    """Tests for _build_json_report() — machine-readable accuracy output."""

    def _make_prompt_result(self, label, passed, max_diff,
                            step_metrics, trt_text="hello", hf_text="hello"):
        return {
            "label": label,
            "passed": passed,
            "max_diff": max_diff,
            "step_metrics": step_metrics,
            "trt_text": trt_text,
            "hf_text": hf_text,
        }

    def _make_step_metric(self, cosine_sim=0.99, argmax_match=True,
                          mean_abs_diff=0.001, max_abs_diff=0.005):
        return {
            "step": 0,
            "cosine_sim": cosine_sim,
            "argmax_match": argmax_match,
            "mean_abs_diff": mean_abs_diff,
            "max_abs_diff": max_abs_diff,
        }

    def test_json_has_required_fields(self):
        mod = _import_diff_logits()
        steps = [self._make_step_metric()]
        results = [self._make_prompt_result("p1", True, 0.0001, steps)]
        report = mod._build_json_report(results, atol=1e-3)
        required = {"pass", "cosine_p5", "top1_match_rate",
                     "token_agreement", "mean_abs_diff"}
        assert required.issubset(report.keys())

    def test_pass_true_when_all_prompts_pass(self):
        mod = _import_diff_logits()
        steps = [self._make_step_metric()]
        results = [
            self._make_prompt_result("p1", True, 0.0001, steps),
            self._make_prompt_result("p2", True, 0.0002, steps),
        ]
        report = mod._build_json_report(results, atol=1e-3)
        assert report["pass"] is True

    def test_pass_false_when_any_prompt_fails(self):
        mod = _import_diff_logits()
        steps = [self._make_step_metric()]
        results = [
            self._make_prompt_result("p1", True, 0.0001, steps),
            self._make_prompt_result("p2", False, 5.0, steps),
        ]
        report = mod._build_json_report(results, atol=1e-3)
        assert report["pass"] is False

    def test_cosine_p5_computed_correctly(self):
        """cosine_p5 is the 5th percentile of all per-step cosine sims."""
        mod = _import_diff_logits()
        # 20 steps with cosine 0.99, then 1 step with cosine 0.80.
        # The 5th percentile of [0.80, 0.99, 0.99, ..., 0.99] (21 values)
        # should be close to 0.80 (the low outlier).
        high_steps = [self._make_step_metric(cosine_sim=0.99)
                      for _ in range(20)]
        low_step = [self._make_step_metric(cosine_sim=0.80)]
        all_steps = low_step + high_steps
        results = [self._make_prompt_result("p1", True, 0.001, all_steps)]
        report = mod._build_json_report(results, atol=1e-3)
        # 5th percentile of 21 values: index 1.0 (interpolated), so it is
        # between 0.80 and 0.99 but should be exactly at/near 0.80.
        assert report["cosine_p5"] == pytest.approx(
            float(np.percentile([0.80] + [0.99] * 20, 5)), abs=1e-9)

    def test_top1_match_rate(self):
        mod = _import_diff_logits()
        steps = [
            self._make_step_metric(argmax_match=True),
            self._make_step_metric(argmax_match=True),
            self._make_step_metric(argmax_match=False),
            self._make_step_metric(argmax_match=True),
        ]
        results = [self._make_prompt_result("p1", True, 0.001, steps)]
        report = mod._build_json_report(results, atol=1e-3)
        assert report["top1_match_rate"] == pytest.approx(0.75)

    def test_token_agreement_text_match(self):
        mod = _import_diff_logits()
        steps = [self._make_step_metric()]
        results = [
            self._make_prompt_result("p1", True, 0.001, steps,
                                     trt_text="hello", hf_text="hello"),
            self._make_prompt_result("p2", True, 0.001, steps,
                                     trt_text="world", hf_text="nope"),
        ]
        report = mod._build_json_report(results, atol=1e-3)
        assert report["token_agreement"] == pytest.approx(0.5)

    def test_empty_results(self):
        mod = _import_diff_logits()
        report = mod._build_json_report([], atol=1e-3)
        assert report["pass"] is True  # vacuously true, no failures
        assert report["cosine_p5"] == 0.0
        assert report["top1_match_rate"] == 0.0
        assert report["token_agreement"] == 0.0
        assert report["mean_abs_diff"] == 0.0
