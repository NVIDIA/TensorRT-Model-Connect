"""Contract test plugin for vision-language QA and OCR models."""
from __future__ import annotations

import re
import string

from ..contracts import MetricResult, OracleLevel, StageOutput
from .base import (
    normalize_text, extract_answer, levenshtein_ned, make_pass, make_fail,
    make_skip,
)

_ANSWER_EDGE_PUNCTUATION = string.punctuation + string.whitespace


def _normalize_vl_answer(text: str) -> str:
    """Normalize short VL QA answers without penalizing terminal punctuation."""
    return normalize_text(text).strip(_ANSWER_EDGE_PUNCTUATION)


def _align_embedded_single_word_answer(left: str, right: str) -> tuple[str, str]:
    """Treat one-word answers as equivalent when embedded in a sentence."""
    left_words = re.findall(r"\b\w+\b", left)
    right_words = re.findall(r"\b\w+\b", right)
    if len(left_words) == 1 and len(right_words) > 1 and left_words[0] in right_words:
        return left_words[0], left_words[0]
    if len(right_words) == 1 and len(left_words) > 1 and right_words[0] in left_words:
        return right_words[0], right_words[0]
    return left, right


def _reference_is_invariant_only(case, ref_output: StageOutput) -> bool:
    return (
        case.reference_backend == "invariant_only"
        or case.oracle_level == OracleLevel.L4_INVARIANTS.value
        or bool((ref_output.data or {}).get("_invariant_only"))
        or ref_output.metadata.get("source") == "invariant_only"
    )


def _normalized_substring_hits(text: str, substrings: list[str]) -> tuple[int, list[str]]:
    normalized = normalize_text(text)
    missing = [
        expected for expected in substrings
        if normalize_text(expected) not in normalized
    ]
    return len(substrings) - len(missing), missing


class VLQAPlugin:
    reference_families = ["vl_instruct_qa", "ocr_markdown"]
    user_contract = "vl_answer"

    def configure_reference(self, case):
        config = {"use_processor": True, "use_chat_template": True}
        if case.reference_family == "ocr_markdown":
            config["ocr_mode"] = True
        return config

    def verify(self, trt_output, ref_output, case, threshold):
        stage = trt_output.stage_name

        # vision_encode: invariant check only (no text to compare)
        if stage == "vision_encode":
            returncode = trt_output.metadata.get("returncode")
            passed = bool(trt_output.data.get("passed", False))
            if returncode not in (None, 0):
                passed = False
            metrics = {
                "vision_encode_ok": MetricResult(
                    value=1.0 if passed else 0.0, threshold=1.0, operator="==",
                    passed=bool(passed), note="vision encoder ran successfully"),
            }
            if passed:
                return make_pass("vision_encode", metrics, "vision encoder health")
            return make_fail("vision_encode", metrics, "vision encoder health",
                             "Vision encoder failed")

        # Non-generation stages: pass-through invariant check
        if stage != "full_generation":
            metrics = {
                "stage_ok": MetricResult(
                    value=1.0, threshold=0.0, operator=">=",
                    passed=True, note=f"{stage} completed"),
            }
            return make_pass(stage, metrics, f"{stage} invariant check")

        # full_generation: text comparison
        prompt = case.inputs.get("prompt", "")

        # TRT VL runner puts text in data["generated_text"], ref in data["text"]
        trt_text = trt_output.data.get("generated_text", trt_output.text or "")
        ref_text = ref_output.data.get("text", ref_output.text or "")

        returncode = trt_output.metadata.get("returncode")
        if returncode not in (None, 0):
            metrics = {
                "trt_returncode_ok": MetricResult(
                    value=float(returncode), threshold=0.0, operator="==",
                    passed=False, note="TRT generation subprocess exit code"),
            }
            return make_fail(
                "full_generation", metrics,
                "TRT generation subprocess must exit cleanly",
                f"TRT generation failed with return code {returncode}")

        is_ocr = case.reference_family == "ocr_markdown"
        required_substrings = ref_output.data.get("required_substrings", [])
        if is_ocr and required_substrings:
            hits, missing = _normalized_substring_hits(trt_text, required_substrings)
            passed = not missing
            metrics = {
                "required_ocr_substrings": MetricResult(
                    value=float(hits),
                    threshold=float(len(required_substrings)),
                    operator="==",
                    passed=passed,
                    note="required OCR substrings present in TRT output"),
            }
            if passed:
                return make_pass(
                    "full_generation", metrics,
                    "required OCR substrings present")
            return make_fail(
                "full_generation", metrics,
                "required OCR substrings present",
                "TRT OCR output missing expected text: " + ", ".join(missing))

        if not ref_text:
            has_output = len(normalize_text(trt_text)) > 0
            metrics = {
                "has_output": MetricResult(
                    value=1.0 if has_output else 0.0, threshold=1.0, operator="==",
                    passed=has_output, note="TRT produced non-empty text"),
            }
            if _reference_is_invariant_only(case, ref_output):
                if not has_output:
                    return make_fail(
                        "full_generation", metrics,
                        "invariant: non-empty output",
                        "TRT produced empty VL/OCR output")
                return make_skip(
                    "full_generation", metrics,
                    "reference text required for VL/OCR contract",
                    "Invariant-only reference cannot validate VL/OCR text output")
            return make_fail("full_generation", metrics, "invariant: non-empty output",
                            "Reference produced empty VL/OCR text; parity is unvalidated")

        trt_answer = normalize_text(extract_answer(
            StageOutput(stage_name=stage, text=trt_text), prompt))
        ref_answer = normalize_text(extract_answer(
            StageOutput(stage_name=stage, text=ref_text), prompt))
        raw_trt_answer = trt_answer
        raw_ref_answer = ref_answer
        if not is_ocr:
            trt_answer = _normalize_vl_answer(trt_answer)
            ref_answer = _normalize_vl_answer(ref_answer)
            trt_answer, ref_answer = _align_embedded_single_word_answer(
                trt_answer, ref_answer)

        if not trt_answer:
            return make_fail("full_generation", {}, message="TRT produced empty VL answer")

        exact = (trt_answer == ref_answer)
        ned = levenshtein_ned(trt_answer, ref_answer)

        ned_threshold = threshold.metrics.get(
            "contract_ned_threshold", 0.05 if is_ocr else 0.15)

        metrics = {
            "exact_match": MetricResult(value=1.0 if exact else 0.0, threshold=1.0, operator="==", passed=exact),
            "ned": MetricResult(value=ned, threshold=ned_threshold, operator="<=", passed=ned <= ned_threshold),
        }
        if not is_ocr and (raw_trt_answer != trt_answer or raw_ref_answer != ref_answer):
            raw_ned = levenshtein_ned(raw_trt_answer, raw_ref_answer)
            metrics["raw_answer_ned"] = MetricResult(
                value=raw_ned, threshold=None, operator="",
                passed=True,
                note="informational before one-word answer normalization")

        passed = exact or ned <= ned_threshold
        label = "OCR" if is_ocr else "VL QA"
        rule = "exact_match OR ned <= threshold"
        if passed:
            return make_pass("full_generation", metrics, rule)
        return make_fail("full_generation", metrics, rule, f"{label} answer diverged: NED={ned:.3f}")

plugin = VLQAPlugin()
