# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PersonaPlex-owned speech-to-speech contract plugin."""

from __future__ import annotations

from tests.e2e_harness.contracts import MetricResult
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

class PersonaPlexSpeechToSpeechPlugin:
    reference_families = ["s2s_personaplex"]
    user_contract = "speech_response"

    def configure_reference(self, case):
        return {}

    def verify(self, trt_output, ref_output, case, threshold):
        import numpy as np

        trt_wav = trt_output.data.get("wav_path")
        trt_rms = trt_output.data.get("rms")
        trt_duration = trt_output.data.get("duration_s")
        min_rms = threshold.metrics.get("contract_min_rms", 0.001)

        metrics = {}
        has_wav = trt_wav is not None and isinstance(trt_wav, str) and len(trt_wav) > 0
        metrics["has_audio"] = MetricResult(
            value=1.0 if has_wav else 0.0,
            threshold=1.0,
            operator="==",
            passed=has_wav,
        )

        if trt_rms is not None:
            rms_ok = float(trt_rms) >= min_rms
            metrics["rms"] = MetricResult(
                value=float(trt_rms),
                threshold=min_rms,
                operator=">=",
                passed=rms_ok,
            )

        if trt_duration is not None:
            dur_ok = float(trt_duration) >= 0.1
            metrics["duration_s"] = MetricResult(
                value=float(trt_duration),
                threshold=0.1,
                operator=">=",
                passed=dur_ok,
            )

        trt_tokens = trt_output.data.get("output_tokens")
        ref_tokens = ref_output.data.get("reference_tokens")
        tokens_available = trt_tokens is not None and ref_tokens is not None
        metrics["reference_tokens_available"] = MetricResult(
            value=1.0 if tokens_available else 0.0,
            threshold=1.0,
            operator="==",
            passed=tokens_available,
        )
        if tokens_available:
            trt_arr = np.asarray(trt_tokens).reshape(-1)
            ref_arr = np.asarray(ref_tokens).reshape(-1)
            token_count = min(trt_arr.size, ref_arr.size)
            if token_count > 0:
                token_match = float(np.mean(
                    trt_arr[:token_count] == ref_arr[:token_count]))
                token_threshold = threshold.metrics.get(
                    "contract_token_match", 0.5)
                metrics["token_match"] = MetricResult(
                    value=token_match,
                    threshold=token_threshold,
                    operator=">=",
                    passed=token_match >= token_threshold,
                )
            else:
                metrics["non_empty_reference_overlap"] = MetricResult(
                    value=0.0,
                    threshold=1.0,
                    operator="==",
                    passed=False,
                )

        rule = "audio health + token match"
        if all(metric.passed for metric in metrics.values()):
            return make_pass("full_generation", metrics, rule)
        return make_fail(
            "full_generation",
            metrics,
            rule,
            "PersonaPlex speech response health check failed",
        )

plugin = PersonaPlexSpeechToSpeechPlugin()
