"""Contract test plugin for base causal LM continuation and code completion."""
from __future__ import annotations

from ..contracts import MetricResult
from .base import normalize_text, strip_prompt_echo, levenshtein_ned, make_pass, make_fail

class CausalContinuationPlugin:
    reference_families = ["causal_base_continuation", "code_base_completion", "seq2seq_base_weak"]
    user_contract = "continuation_parity"

    def configure_reference(self, case):
        # Base models: raw tokenization, no chat template
        return {}

    def verify(self, trt_output, ref_output, case, threshold):
        prompt = case.inputs.get("prompt", "")
        trt_text = normalize_text(strip_prompt_echo(trt_output.text or "", prompt))
        if case.reference_family == "seq2seq_base_weak":
            ref_text = normalize_text(ref_output.text or "")
        else:
            ref_text = normalize_text(strip_prompt_echo(ref_output.text or "", prompt))

        if not trt_text and not ref_text:
            metrics = {
                "non_empty_continuation": MetricResult(
                    value=0.0, threshold=1.0, operator="==", passed=False,
                    note="empty TRT and reference text do not validate parity"),
            }
            return make_fail(
                "full_generation", metrics,
                "non-empty continuation required",
                "Both TRT and reference produced empty continuation")

        ned = levenshtein_ned(trt_text, ref_text)
        ned_threshold = threshold.metrics.get("contract_ned_threshold", 0.25)

        # For base models, also check prefix match (first N chars)
        prefix_len = min(50, min(len(trt_text), len(ref_text)))
        prefix_match = (trt_text[:prefix_len] == ref_text[:prefix_len]) if prefix_len > 0 else True

        metrics = {
            "ned": MetricResult(value=ned, threshold=ned_threshold, operator="<=", passed=ned <= ned_threshold),
            "prefix_match": MetricResult(value=1.0 if prefix_match else 0.0, threshold=1.0, operator="==", passed=prefix_match, note=f"first {prefix_len} chars"),
        }

        passed = ned <= ned_threshold
        rule = "ned <= threshold (continuation parity)"
        if passed:
            return make_pass("full_generation", metrics, rule)
        return make_fail("full_generation", metrics, rule, f"Continuation diverged: NED={ned:.3f}")

plugin = CausalContinuationPlugin()
