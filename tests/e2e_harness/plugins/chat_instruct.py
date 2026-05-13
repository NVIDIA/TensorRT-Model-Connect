"""Contract test plugin for chat/instruct models."""
from __future__ import annotations
from ..contracts import MetricResult
from .base import normalize_text, extract_answer, levenshtein_ned, make_pass, make_fail


def _normalize_chat_answer(text: str) -> str:
    """Normalize model text after removing SentencePiece word-boundary markers."""
    return normalize_text(text.replace("▁", " "))

class ChatInstructPlugin:
    reference_families = ["chat_instruct_template", "chat_qwen3_posttrained"]
    user_contract = "chat_response"

    # Markers that indicate the prompt already contains chat formatting.
    _PRE_FORMATTED_MARKERS = (
        "<|im_start|>", "[INST]", "<|start|>", "<|user|>",
        "<start_of_turn>", "<|start_header_id|>", "<extra_id_0>", "<SPECIAL_10>",
    )

    def configure_reference(self, case):
        prompt = case.inputs.get("prompt", "")
        # Skip chat template if the prompt already contains chat tokens
        already_formatted = any(m in prompt for m in self._PRE_FORMATTED_MARKERS)
        # InternLM2's ChatML template has no thinking-control field. Passing
        # --no-thinking would append a Qwen-style thinking block in the C++
        # helper, making TRT and HF prompts diverge before the model runs.
        enable_thinking = True if case.name == "internlm2-1.8b" else False
        config = {
            "use_chat_template": not already_formatted,
            "enable_thinking": enable_thinking,
        }
        if case.name == "internlm2-1.8b":
            config["reference_generation_mode"] = "hf_generate"
        return config

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

        trt_answer = _normalize_chat_answer(extract_answer(trt_output, prompt))
        ref_answer = _normalize_chat_answer(extract_answer(ref_output, prompt))

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
