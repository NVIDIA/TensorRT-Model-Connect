"""Model-card contract for Nemotron-H no-thinking chat generation."""

from __future__ import annotations

import re

from ..contracts import E2ECase, MetricResult, StageOutput, ThresholdProfile
from .base import extract_answer, make_fail, make_pass, normalize_text


def _cpp_command(output: StageOutput) -> list[str]:
    command = output.metadata.get("cpp", {}).get("command", [])
    return [str(part) for part in command] if isinstance(command, list) else []


def _strip_empty_think_blocks(text: str) -> str:
    return re.sub(r"<think>\s*</think>", "", text, flags=re.IGNORECASE)


class NemotronHModelCardPlugin:
    reference_families = ["nemotron_h_no_think_model_card"]
    user_contract = "chat_response"

    def configure_reference(self, case: E2ECase) -> dict:
        return {"use_chat_template": True, "enable_thinking": False}

    def verify(
        self,
        trt_output: StageOutput,
        ref_output: StageOutput,
        case: E2ECase,
        threshold: ThresholdProfile,
    ):
        raw_text = trt_output.text or ""
        answer = _strip_empty_think_blocks(extract_answer(trt_output, case.inputs.get("prompt", "")))
        normalized = normalize_text(answer)
        lowered_raw = _strip_empty_think_blocks(raw_text).lower()

        command = _cpp_command(trt_output)
        flags_forwarded = "--chat-template" in command and "--no-thinking" in command
        no_visible_thinking = "<think>" not in lowered_raw and "</think>" not in lowered_raw
        topic_keywords = ("gpu", "gpus", "cuda", "silicon", "cores", "parallel")
        mentions_gpu_topic = any(keyword in normalized for keyword in topic_keywords)
        non_empty = len(normalized) >= int(
            threshold.metrics.get("contract_min_output_chars", 12))
        min_lines = int(threshold.metrics.get("contract_min_haiku_lines", 2))
        line_count = len([line for line in answer.splitlines() if line.strip()])
        haiku_shape = line_count >= min_lines
        cpp_ok = int(trt_output.data.get("cpp_returncode", 0)) == 0

        metrics = {
            "cpp_returncode_ok": MetricResult(
                value=1.0 if cpp_ok else 0.0, threshold=1.0,
                operator="==", passed=cpp_ok),
            "chat_no_thinking_flags": MetricResult(
                value=1.0 if flags_forwarded else 0.0, threshold=1.0,
                operator="==", passed=flags_forwarded,
                note="requires --chat-template and --no-thinking"),
            "no_visible_thinking_trace": MetricResult(
                value=1.0 if no_visible_thinking else 0.0, threshold=1.0,
                operator="==", passed=no_visible_thinking),
            "non_empty_response": MetricResult(
                value=float(len(normalized)), threshold=float(
                    threshold.metrics.get("contract_min_output_chars", 12)),
                operator=">=", passed=non_empty),
            "mentions_gpu_topic": MetricResult(
                value=1.0 if mentions_gpu_topic else 0.0, threshold=1.0,
                operator="==", passed=mentions_gpu_topic),
            "haiku_line_shape": MetricResult(
                value=float(line_count), threshold=float(min_lines),
                operator=">=", passed=haiku_shape),
        }

        passed = (
            cpp_ok
            and flags_forwarded
            and no_visible_thinking
            and non_empty
            and mentions_gpu_topic
            and haiku_shape
        )
        rule = (
            "cpp_returncode_ok AND chat_no_thinking_flags AND "
            "no_visible_thinking_trace AND non_empty_response AND "
            "mentions_gpu_topic AND haiku_line_shape"
        )
        if passed:
            return make_pass("full_generation", metrics, rule)
        return make_fail(
            "full_generation", metrics, rule,
            "Nemotron-H model-card no-thinking haiku contract failed",
        )


plugin = NemotronHModelCardPlugin()
