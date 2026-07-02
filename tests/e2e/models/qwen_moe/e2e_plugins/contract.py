# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-MoE-owned post-trained chat contract plugin."""

from __future__ import annotations

from tests.e2e_harness.contracts import CompareResult, MetricResult


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
    "### response:",
    "### assistant:",
    "assistant:",
    "<|assistant|>",
    "<|im_start|>assistant\n",
)

_CHAT_TURN_MARKERS = (
    "### response:",
    "### instruction:",
    "### assistant:",
    "### user:",
    "<|assistant|>",
    "<|user|>",
    "<|im_start|>",
    "<|im_end|>",
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
            curr.append(
                min(
                    prev[j + 1] + 1,
                    curr[j] + 1,
                    prev[j] + (0 if c1 == c2 else 1),
                )
            )
        prev = curr
    return prev[-1] / max_len


def make_pass(stage_name: str, metrics, rule: str = "") -> CompareResult:
    return CompareResult(
        stage_name=stage_name,
        status="passed",
        metrics=metrics,
        composite_rule=rule,
        message="Qwen-MoE chat contract verified",
    )


def make_fail(stage_name: str, metrics, rule: str = "", message: str = "") -> CompareResult:
    return CompareResult(
        stage_name=stage_name,
        status="failed",
        metrics=metrics,
        composite_rule=rule,
        message=message or "Qwen-MoE chat contract failed",
    )


class QwenMoePostTrainedChatPlugin:
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
        config = case.metadata.get("contract_config", {})
        if isinstance(config, dict) and config.get("enable_thinking") is False:
            raw_trt = trt_output.text or ""
            if "<think>" in raw_trt:
                metrics = {
                    "thinking_suppressed": MetricResult(
                        value=0.0,
                        threshold=1.0,
                        operator="==",
                        passed=False,
                        note="no-thinking output must not contain <think>",
                    )
                }
                return make_fail(
                    "full_generation",
                    metrics,
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
            f"Qwen-MoE chat response diverged: NED={ned:.3f}",
        )


plugin = QwenMoePostTrainedChatPlugin()
