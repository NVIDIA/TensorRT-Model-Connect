"""Qwen3.5-owned multimodal chat contract plugin."""

from __future__ import annotations

from tests.e2e_harness.contracts import MetricResult
from tests.e2e_harness.plugins.base import (
    extract_answer,
    levenshtein_ned,
    make_fail,
    make_pass,
    normalize_text,
)


class Qwen35MultimodalChatPlugin:
    reference_families = ["multimodal_chat_qwen35"]
    user_contract = "chat_response"

    def configure_reference(self, case):
        return {
            "use_chat_template": True,
            "use_processor": True,
            "enable_thinking": False,
        }

    def verify(self, trt_output, ref_output, case, threshold):
        prompt = case.inputs.get("prompt", "")
        trt_answer = normalize_text(extract_answer(trt_output, prompt))
        ref_answer = normalize_text(extract_answer(ref_output, prompt))

        if not trt_answer:
            return make_fail("full_generation", {}, message="TRT produced empty response")

        exact_match = trt_answer == ref_answer
        ned = levenshtein_ned(trt_answer, ref_answer)
        ned_threshold = threshold.metrics.get("contract_ned_threshold", 0.15)
        metrics = {
            "exact_match": MetricResult(
                value=1.0 if exact_match else 0.0,
                threshold=1.0,
                operator="==",
                passed=exact_match,
            ),
            "ned": MetricResult(
                value=ned,
                threshold=ned_threshold,
                operator="<=",
                passed=ned <= ned_threshold,
            ),
        }

        rule = "exact_match OR ned <= threshold"
        if exact_match or ned <= ned_threshold:
            return make_pass("full_generation", metrics, rule)
        return make_fail(
            "full_generation",
            metrics,
            rule,
            f"Qwen3.5 multimodal chat response diverged: NED={ned:.3f}",
        )


plugin = Qwen35MultimodalChatPlugin()
