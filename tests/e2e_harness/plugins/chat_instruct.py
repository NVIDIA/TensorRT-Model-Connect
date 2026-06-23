"""Contract test plugin for chat/instruct models."""
from __future__ import annotations
from ..contracts import MetricResult
from .base import (
    contract_config,
    normalize_text,
    extract_answer,
    levenshtein_ned,
    make_pass,
    make_fail,
)

class ChatInstructPlugin:
    reference_families = ["chat_instruct_template"]
    user_contract = "chat_response"

    def configure_reference(self, case):
        return contract_config(case)

    def verify(self, trt_output, ref_output, case, threshold):
        prompt = case.inputs.get("prompt", "")
        config = contract_config(case)
        if config.get("enable_thinking") is False:
            raw_trt = trt_output.text or ""
            if "<think>" in raw_trt:
                return make_fail(
                    "full_generation",
                    {
                        "thinking_suppressed": MetricResult(
                            value=0.0,
                            threshold=1.0,
                            operator="==",
                            passed=False,
                            note="no-thinking output must not contain <think>",
                        )
                    },
                    message="TRT emitted a thinking block with thinking disabled",
                )

        trt_answer = normalize_text(extract_answer(trt_output, prompt))
        ref_answer = normalize_text(extract_answer(ref_output, prompt))

        if not trt_answer:
            return make_fail("full_generation", {}, message="TRT produced empty response")

        exact_match = (trt_answer == ref_answer)
        ned = levenshtein_ned(trt_answer, ref_answer)
        ned_threshold = threshold.metrics.get("contract_ned_threshold", 0.15)

        metrics = {
            "exact_match": MetricResult(value=1.0 if exact_match else 0.0, threshold=1.0, operator="==", passed=exact_match),
            "ned": MetricResult(value=ned, threshold=ned_threshold, operator="<=", passed=ned <= ned_threshold),
        }

        passed = exact_match or ned <= ned_threshold
        rule = "exact_match OR ned <= threshold"
        if passed:
            return make_pass("full_generation", metrics, rule)
        return make_fail("full_generation", metrics, rule, f"Chat response diverged: NED={ned:.3f}")

plugin = ChatInstructPlugin()
