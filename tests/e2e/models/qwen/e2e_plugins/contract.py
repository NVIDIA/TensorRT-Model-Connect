"""Qwen-owned post-trained chat contract plugin."""

from __future__ import annotations

from tests.e2e_harness.contracts import MetricResult
from tests.e2e_harness.plugins.base import (
    extract_answer,
    levenshtein_ned,
    make_fail,
    make_pass,
    normalize_text,
)


class QwenPostTrainedChatPlugin:
    reference_families = ["chat_qwen3_posttrained"]
    user_contract = "chat_response"

    _PRE_FORMATTED_MARKERS = (
        "<|im_start|>",
        "[INST]",
        "<|start|>",
        "<|user|>",
        "<start_of_turn>",
        "<|start_header_id|>",
        "<extra_id_0>",
        "<SPECIAL_10>",
    )

    def configure_reference(self, case):
        prompt = case.inputs.get("prompt", "")
        already_formatted = any(marker in prompt for marker in self._PRE_FORMATTED_MARKERS)
        return {"use_chat_template": not already_formatted, "enable_thinking": False}

    def verify(self, trt_output, ref_output, case, threshold):
        prompt = case.inputs.get("prompt", "")
        contract_config = case.metadata.get("contract_config", {})
        if contract_config.get("enable_thinking") is False:
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
            f"Qwen chat response diverged: NED={ned:.3f}",
        )


plugin = QwenPostTrainedChatPlugin()
