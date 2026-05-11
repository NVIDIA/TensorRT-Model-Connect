"""Contract plugin for the Qwen2.5 base-model model-card example."""
from __future__ import annotations

from ..contracts import MetricResult
from .base import (
    extract_answer,
    levenshtein_ned,
    make_fail,
    make_pass,
    normalize_text,
)


class Qwen25ModelCardPlugin:
    reference_families = ["qwen25_base_model_card"]
    user_contract = "chat_response"

    def configure_reference(self, case):
        return {"use_chat_template": True, "enable_thinking": True}

    def verify(self, trt_output, ref_output, case, threshold):
        prompt = case.inputs.get("prompt", "")
        command = trt_output.metadata.get("cpp", {}).get("command", [])
        uses_model_card_template = "--chat-template" in command and "--no-thinking" not in command

        trt_answer = normalize_text(extract_answer(trt_output, prompt))
        ref_answer = normalize_text(extract_answer(ref_output, prompt))
        exact_match = bool(trt_answer) and trt_answer == ref_answer
        ned = levenshtein_ned(trt_answer, ref_answer)
        ned_threshold = threshold.metrics.get("contract_ned_threshold", 0.15)

        metrics = {
            "model_card_chat_template": MetricResult(
                value=1.0 if uses_model_card_template else 0.0,
                threshold=1.0,
                operator="==",
                passed=uses_model_card_template,
                note="requires --chat-template without --no-thinking",
            ),
            "non_empty_response": MetricResult(
                value=1.0 if trt_answer else 0.0,
                threshold=1.0,
                operator="==",
                passed=bool(trt_answer),
            ),
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
        if uses_model_card_template and trt_answer and (exact_match or ned <= ned_threshold):
            return make_pass("full_generation", metrics, "model-card chat template AND text parity")
        return make_fail(
            "full_generation",
            metrics,
            "model-card chat template AND text parity",
            f"Qwen2.5 model-card response diverged: NED={ned:.3f}",
        )


plugin = Qwen25ModelCardPlugin()
