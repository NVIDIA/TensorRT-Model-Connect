# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""internvl-owned E2E contract plugins."""
from __future__ import annotations

import re
import string

from tests.e2e_harness.contracts import (
    MetricResult,
    OracleLevel,
    StageOutput,
)
# Model-owned contract helpers. Keep behavior here so contract semantics do not
# drift across model families through shared harness code.
def contract_config(case):
    config = case.metadata.get("contract_config", {})
    return dict(config) if isinstance(config, dict) else {}


def normalize_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.split()).strip().lower()


def strip_prompt_echo(text: str, prompt: str) -> str:
    if not text or not prompt:
        return text
    idx = text.find(prompt)
    if 0 <= idx <= 2048:
        return text[idx + len(prompt):].lstrip()
    norm_text = normalize_text(text)
    norm_prompt = normalize_text(prompt)
    if norm_prompt and norm_text.startswith(norm_prompt):
        return text[len(prompt):].lstrip() if text.startswith(prompt) else text
    return text


_CHAT_ROLE_PREFIXES = (
    "### response:", "### assistant:", "assistant:",
    "<|assistant|>", "<|im_start|>assistant\n",
)

_CHAT_TURN_MARKERS = (
    "### response:", "### instruction:", "### assistant:",
    "### user:", "<|assistant|>", "<|user|>",
    "<|im_start|>", "<|im_end|>",
)


def strip_chat_markup(text: str) -> str:
    if not text:
        return ""
    out = text.lstrip()
    while True:
        lowered = out.lower()
        matched = False
        for prefix in _CHAT_ROLE_PREFIXES:
            if lowered.startswith(prefix):
                out = out[len(prefix):].lstrip()
                matched = True
                break
        if not matched:
            break
    lowered = out.lower()
    cut = len(out)
    for marker in _CHAT_TURN_MARKERS:
        idx = lowered.find(marker)
        if idx > 0:
            cut = min(cut, idx)
    if cut < len(out):
        out = out[:cut]
    import re
    out = re.sub(r"(?:\s*#{2,}\s*)+$", "", out).strip()
    return out


def extract_answer(output, prompt: str = "") -> str:
    raw = output.text or ""
    if prompt:
        raw = strip_prompt_echo(raw, prompt)
    raw = strip_chat_markup(raw)
    return raw.strip()


def levenshtein_ned(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, c1 in enumerate(a):
        curr = [i + 1]
        for j, c2 in enumerate(b):
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (0 if c1 == c2 else 1),
            ))
        prev = curr
    return prev[-1] / max_len


def make_pass(stage_name: str, metrics, rule: str = ""):
    from tests.e2e_harness.contracts import CompareResult
    return CompareResult(
        stage_name=stage_name,
        status="passed",
        metrics=metrics,
        composite_rule=rule,
        message="Contract verified",
    )


def make_fail(stage_name: str, metrics, rule: str = "", message: str = ""):
    from tests.e2e_harness.contracts import CompareResult
    return CompareResult(
        stage_name=stage_name,
        status="failed",
        metrics=metrics,
        composite_rule=rule,
        message=message or "Contract verification failed",
    )


def make_skip(stage_name: str, metrics, rule: str = "", message: str = ""):
    from tests.e2e_harness.contracts import CompareResult
    return CompareResult(
        stage_name=stage_name,
        status="skipped",
        metrics=metrics,
        composite_rule=rule,
        message=message or "Contract validation skipped",
    )


def make_error(stage_name: str, error: str):
    from tests.e2e_harness.contracts import CompareResult
    return CompareResult(
        stage_name=stage_name,
        status="error",
        message=f"Contract verification error: {error}",
    )

_ANSWER_EDGE_PUNCTUATION = string.punctuation + string.whitespace

def _normalize_vl_answer(text: str) -> str:
    return normalize_text(text).strip(_ANSWER_EDGE_PUNCTUATION)

def _align_embedded_single_word_answer(left: str, right: str) -> tuple[str, str]:
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

class InternvlVLQAPlugin:
    reference_families = ["vl_instruct_qa"]
    user_contract = "vl_answer"

    def configure_reference(self, case):
        return contract_config(case)

    def verify(self, trt_output, ref_output, case, threshold):
        stage = trt_output.stage_name
        if stage == "vision_encode":
            returncode = trt_output.metadata.get("returncode")
            passed = bool(trt_output.data.get("passed", False))
            if returncode not in (None, 0):
                passed = False
            metrics = {
                "vision_encode_ok": MetricResult(
                    value=1.0 if passed else 0.0,
                    threshold=1.0,
                    operator="==",
                    passed=bool(passed),
                    note="vision encoder ran successfully",
                ),
            }
            if passed:
                return make_pass("vision_encode", metrics, "vision encoder health")
            return make_fail(
                "vision_encode",
                metrics,
                "vision encoder health",
                "Vision encoder failed",
            )

        if stage != "full_generation":
            metrics = {
                "stage_ok": MetricResult(
                    value=1.0,
                    threshold=0.0,
                    operator=">=",
                    passed=True,
                    note=f"{stage} completed",
                ),
            }
            return make_pass(stage, metrics, f"{stage} invariant check")

        prompt = case.inputs.get("prompt", "")
        trt_text = trt_output.data.get("generated_text", trt_output.text or "")
        ref_text = ref_output.data.get("text", ref_output.text or "")
        returncode = trt_output.metadata.get("returncode")
        if returncode not in (None, 0):
            metrics = {
                "trt_returncode_ok": MetricResult(
                    value=float(returncode),
                    threshold=0.0,
                    operator="==",
                    passed=False,
                    note="TRT generation subprocess exit code",
                ),
            }
            return make_fail(
                "full_generation",
                metrics,
                "TRT generation subprocess must exit cleanly",
                f"TRT generation failed with return code {returncode}",
            )

        is_ocr = bool(contract_config(case).get("ocr_mode"))
        required_substrings = ref_output.data.get("required_substrings", [])
        if is_ocr and required_substrings:
            if not ref_text:
                metrics = {
                    "reference_text_present": MetricResult(
                        value=0.0,
                        threshold=1.0,
                        operator="==",
                        passed=False,
                        note="OCR golden snapshot exposes human-readable reference text",
                    ),
                }
                return make_fail(
                    "full_generation",
                    metrics,
                    "OCR golden snapshot includes visible reference text",
                    "OCR golden snapshot must include human-readable reference text",
                )

            ref_hits, ref_missing = _normalized_substring_hits(ref_text, required_substrings)
            if ref_missing:
                metrics = {
                    "reference_contract_substrings": MetricResult(
                        value=float(ref_hits),
                        threshold=float(len(required_substrings)),
                        operator="==",
                        passed=False,
                        note="required OCR substrings visible in reference text",
                    ),
                }
                return make_fail(
                    "full_generation",
                    metrics,
                    "OCR required substrings are visible in reference text",
                    "OCR golden snapshot hides required text from the report: "
                    + ", ".join(ref_missing),
                )

            hits, missing = _normalized_substring_hits(trt_text, required_substrings)
            passed = not missing
            metrics = {
                "reference_contract_substrings": MetricResult(
                    value=float(len(required_substrings)),
                    threshold=float(len(required_substrings)),
                    operator="==",
                    passed=True,
                    note="required OCR substrings visible in reference text",
                ),
                "required_ocr_substrings": MetricResult(
                    value=float(hits),
                    threshold=float(len(required_substrings)),
                    operator="==",
                    passed=passed,
                    note="required OCR substrings present in TRT output",
                ),
            }
            if passed:
                return make_pass("full_generation", metrics, "required OCR substrings present")
            return make_fail(
                "full_generation",
                metrics,
                "required OCR substrings present",
                "TRT OCR output missing expected text: " + ", ".join(missing),
            )

        if not ref_text:
            has_output = len(normalize_text(trt_text)) > 0
            metrics = {
                "has_output": MetricResult(
                    value=1.0 if has_output else 0.0,
                    threshold=1.0,
                    operator="==",
                    passed=has_output,
                    note="TRT produced non-empty text",
                ),
            }
            if _reference_is_invariant_only(case, ref_output):
                if not has_output:
                    return make_fail(
                        "full_generation",
                        metrics,
                        "invariant: non-empty output",
                        "TRT produced empty VL/OCR output",
                    )
                return make_skip(
                    "full_generation",
                    metrics,
                    "reference text required for VL/OCR contract",
                    "Invariant-only reference cannot validate VL/OCR text output",
                )
            return make_fail(
                "full_generation",
                metrics,
                "invariant: non-empty output",
                "Reference produced empty VL/OCR text; parity is unvalidated",
            )

        trt_answer = normalize_text(extract_answer(StageOutput(stage_name=stage, text=trt_text), prompt))
        ref_answer = normalize_text(extract_answer(StageOutput(stage_name=stage, text=ref_text), prompt))
        raw_trt_answer = trt_answer
        raw_ref_answer = ref_answer
        if not is_ocr:
            trt_answer = _normalize_vl_answer(trt_answer)
            ref_answer = _normalize_vl_answer(ref_answer)
            trt_answer, ref_answer = _align_embedded_single_word_answer(
                trt_answer, ref_answer)

        if not trt_answer:
            return make_fail("full_generation", {}, message="TRT produced empty VL answer")

        exact = trt_answer == ref_answer
        ned = levenshtein_ned(trt_answer, ref_answer)
        ned_threshold = threshold.metrics.get("contract_ned_threshold", 0.05 if is_ocr else 0.15)
        metrics = {
            "exact_match": MetricResult(
                value=1.0 if exact else 0.0,
                threshold=1.0,
                operator="==",
                passed=exact,
            ),
            "ned": MetricResult(
                value=ned,
                threshold=ned_threshold,
                operator="<=",
                passed=ned <= ned_threshold,
            ),
        }
        if not is_ocr and (raw_trt_answer != trt_answer or raw_ref_answer != ref_answer):
            raw_ned = levenshtein_ned(raw_trt_answer, raw_ref_answer)
            metrics["raw_answer_ned"] = MetricResult(
                value=raw_ned,
                threshold=None,
                operator="",
                passed=True,
                note="informational before one-word answer normalization",
            )

        passed = exact or ned <= ned_threshold
        label = "OCR" if is_ocr else "VL QA"
        rule = "exact_match OR ned <= threshold"
        if passed:
            return make_pass("full_generation", metrics, rule)
        return make_fail(
            "full_generation",
            metrics,
            rule,
            f"{label} answer diverged: NED={ned:.3f}",
        )

plugin = InternvlVLQAPlugin()
