"""Contract test plugin for translation and seq2seq text-to-text models."""
from __future__ import annotations
from ..contracts import MetricResult
from .base import (normalize_text, extract_answer, levenshtein_ned, make_pass, make_fail)


class TranslationPlugin:
    reference_families = ["translation_chat_template", "seq2seq_text2text", "seq2seq_translation"]
    user_contract = "translation"

    # Markers indicating the prompt already contains chat/template formatting.
    _PRE_FORMATTED_MARKERS = (
        "<|im_start|>", "[INST]", "<|start|>", "<|user|>",
        "<start_of_turn>", "<|start_header_id|>", "<extra_id_0>",
        "<SPECIAL_10>", "<s>System", "<s>User",
    )

    def configure_reference(self, case):
        config = dict(case.metadata.get("contract_config", {}))
        for src_key, dst_key in (
            ("translation_source_lang", "src_lang"),
            ("translation_target_lang", "tgt_lang"),
            ("translation_forced_bos_token", "forced_bos_token"),
        ):
            value = case.metadata.get(src_key)
            if value:
                config[dst_key] = value

        if case.reference_family == "translation_chat_template":
            prompt = case.inputs.get("prompt", "")
            already_formatted = any(m in prompt for m in self._PRE_FORMATTED_MARKERS)
            config["use_chat_template"] = not already_formatted
            return config
        if case.reference_family == "seq2seq_translation":
            config["auto_class"] = "AutoModelForSeq2SeqLM"
            return config
        if case.reference_family == "seq2seq_text2text":
            config["auto_class"] = "AutoModelForSeq2SeqLM"
            return config
        return config

    def verify(self, trt_output, ref_output, case, threshold):
        prompt = case.inputs.get("prompt", "")
        trt_text = normalize_text(extract_answer(trt_output, prompt))
        ref_text = normalize_text(extract_answer(ref_output, prompt))

        if not trt_text:
            return make_fail("full_generation", {}, message="TRT produced empty translation")

        exact = (trt_text == ref_text)
        ned = levenshtein_ned(trt_text, ref_text)
        ned_threshold = threshold.metrics.get("contract_ned_threshold", 0.15)

        metrics = {
            "exact_match": MetricResult(value=1.0 if exact else 0.0, threshold=1.0, operator="==", passed=exact),
            "ned": MetricResult(value=ned, threshold=ned_threshold, operator="<=", passed=ned <= ned_threshold),
        }

        passed = exact or ned <= ned_threshold
        if passed:
            return make_pass("full_generation", metrics, "exact OR ned <= threshold")
        return make_fail("full_generation", metrics, "exact OR ned <= threshold",
                        f"Translation diverged: NED={ned:.3f}")


plugin = TranslationPlugin()
