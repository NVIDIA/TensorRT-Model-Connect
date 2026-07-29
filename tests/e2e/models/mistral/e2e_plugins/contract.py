# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mistral-owned E2E contract plugins."""
from __future__ import annotations

from tests.e2e_harness.contracts import (
    MetricResult,
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


def generated_token_parity(trt_output, ref_output, threshold, metrics) -> bool:
    token_threshold = threshold.metrics.get("contract_token_agreement_rate")
    if token_threshold is None:
        return True

    outputs = (("TRT", trt_output), ("HF reference", ref_output))
    tokens: list[list[int]] = []
    missing: list[str] = []
    for label, output in outputs:
        raw = (output.data or {}).get("token_ids")
        if not isinstance(raw, list):
            missing.append(label)
            continue
        try:
            tokens.append([int(token) for token in raw])
        except (TypeError, ValueError):
            missing.append(label)

    if missing:
        metrics["generated_token_ids_available"] = MetricResult(
            value=0.0,
            threshold=1.0,
            operator="==",
            passed=False,
            note=f"missing generated token IDs from {' and '.join(missing)}",
        )
        return False

    trt_tokens, ref_tokens = tokens
    total = max(len(trt_tokens), len(ref_tokens))
    matches = sum(
        trt_tokens[index] == ref_tokens[index]
        for index in range(min(len(trt_tokens), len(ref_tokens)))
    )
    agreement = matches / total if total else 1.0
    exact = trt_tokens == ref_tokens
    exact_required = float(token_threshold) >= 1.0
    metrics["generated_token_agreement_rate"] = MetricResult(
        value=agreement,
        threshold=float(token_threshold),
        operator=">=",
        passed=agreement >= float(token_threshold),
        note=f"TRT tokens={len(trt_tokens)}, HF reference tokens={len(ref_tokens)}",
    )
    metrics["generated_token_exact"] = MetricResult(
        value=1.0 if exact else 0.0,
        threshold=1.0 if exact_required else None,
        operator="==" if exact_required else "",
        passed=exact if exact_required else True,
    )
    return agreement >= float(token_threshold) and (exact or not exact_required)


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

class MistralChatInstructPlugin:
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

        token_parity = generated_token_parity(
            trt_output, ref_output, threshold, metrics
        )
        strict_tokens = threshold.metrics.get(
            "contract_token_agreement_rate"
        ) is not None
        passed = (
            exact_match and token_parity
            if strict_tokens
            else exact_match or ned <= ned_threshold
        )
        rule = (
            "exact normalized text AND exact configured token parity"
            if strict_tokens
            else "exact_match OR ned <= threshold"
        )
        if passed:
            return make_pass("full_generation", metrics, rule)
        return make_fail(
            "full_generation",
            metrics,
            rule,
            f"Chat response diverged: NED={ned:.3f}",
        )

class MistralTranslationPlugin:
    reference_families = ["translation_chat_template"]
    user_contract = "translation"

    def configure_reference(self, case):
        return contract_config(case)

    def verify(self, trt_output, ref_output, case, threshold):
        prompt = case.inputs.get("prompt", "")
        trt_text = normalize_text(extract_answer(trt_output, prompt))
        ref_text = normalize_text(extract_answer(ref_output, prompt))

        if not trt_text:
            return make_fail("full_generation", {}, message="TRT produced empty translation")

        exact = trt_text == ref_text
        ned = levenshtein_ned(trt_text, ref_text)
        ned_threshold = threshold.metrics.get("contract_ned_threshold", 0.15)
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

        token_parity = generated_token_parity(
            trt_output, ref_output, threshold, metrics
        )
        strict_tokens = threshold.metrics.get(
            "contract_token_agreement_rate"
        ) is not None
        passed = (
            exact and token_parity
            if strict_tokens
            else exact or ned <= ned_threshold
        )
        rule = (
            "exact normalized text AND exact configured token parity"
            if strict_tokens
            else "exact OR ned <= threshold"
        )
        if passed:
            return make_pass("full_generation", metrics, rule)
        return make_fail(
            "full_generation",
            metrics,
            rule,
            f"Translation diverged: NED={ned:.3f}",
        )

plugin = [MistralChatInstructPlugin(), MistralTranslationPlugin()]
