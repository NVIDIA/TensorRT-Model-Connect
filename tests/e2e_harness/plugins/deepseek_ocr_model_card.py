"""Model-card OCR contract for DeepSeek-OCR-2."""

from __future__ import annotations

import re

from ..contracts import E2ECase, MetricResult, StageOutput, ThresholdProfile
from .base import make_fail, make_pass, normalize_text


def _normalized_contains(text: str, fragment: str) -> bool:
    def compact(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", normalize_text(value)).strip()

    return compact(fragment) in compact(text)


class DeepSeekOCRModelCardPlugin:
    reference_families = ["deepseek_ocr_model_card"]
    user_contract = "ocr_text"

    def configure_reference(self, case: E2ECase) -> dict:
        return {"ocr_mode": True}

    def verify(
        self,
        trt_output: StageOutput,
        ref_output: StageOutput,
        case: E2ECase,
        threshold: ThresholdProfile,
    ):
        stage = trt_output.stage_name
        if stage == "vision_encode":
            passed = bool(trt_output.data.get("passed", False))
            metrics = {
                "vision_encode_ok": MetricResult(
                    value=1.0 if passed else 0.0, threshold=1.0,
                    operator="==", passed=passed,
                    note="vision encoder subprocess returned success"),
            }
            if passed:
                return make_pass("vision_encode", metrics, "vision encoder health")
            return make_fail(
                "vision_encode", metrics, "vision encoder health",
                "DeepSeek-OCR vision encoder failed")

        text = trt_output.data.get("generated_text", trt_output.text or "")
        normalized = normalize_text(text)
        expected_fragments = case.metadata.get("ocr_expected_fragments", [])
        if not isinstance(expected_fragments, list):
            expected_fragments = []

        min_chars = int(threshold.metrics.get("contract_min_output_chars", 24))
        non_empty = len(normalized) >= min_chars
        matched = [
            fragment for fragment in expected_fragments
            if isinstance(fragment, str) and _normalized_contains(text, fragment)
        ]
        min_fragments = int(threshold.metrics.get(
            "contract_min_expected_fragments", len(expected_fragments)))
        fragments_ok = len(matched) >= min_fragments
        cpp_ok = int(trt_output.metadata.get("returncode", 0)) == 0
        command = trt_output.metadata.get("command", [])
        image_flag_forwarded = isinstance(command, list) and "--image" in command

        metrics = {
            "cpp_returncode_ok": MetricResult(
                value=1.0 if cpp_ok else 0.0, threshold=1.0,
                operator="==", passed=cpp_ok),
            "image_flag_forwarded": MetricResult(
                value=1.0 if image_flag_forwarded else 0.0, threshold=1.0,
                operator="==", passed=image_flag_forwarded),
            "non_empty_ocr_text": MetricResult(
                value=float(len(normalized)), threshold=float(min_chars),
                operator=">=", passed=non_empty),
            "expected_fragment_matches": MetricResult(
                value=float(len(matched)), threshold=float(min_fragments),
                operator=">=", passed=fragments_ok,
                note=", ".join(matched)),
        }
        passed = cpp_ok and image_flag_forwarded and non_empty and fragments_ok
        rule = (
            "cpp_returncode_ok AND image_flag_forwarded AND "
            "non_empty_ocr_text AND expected_fragment_matches"
        )
        if passed:
            return make_pass("full_generation", metrics, rule)
        return make_fail(
            "full_generation", metrics, rule,
            "DeepSeek-OCR model-card OCR contract failed")


plugin = DeepSeekOCRModelCardPlugin()
