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


def _strictest_threshold(metrics, *names: str, default: float) -> float:
    """Return the strictest configured minimum across equivalent keys."""
    configured = [float(metrics[name]) for name in names if name in metrics]
    return max(configured) if configured else default


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
        min_rms = _strictest_threshold(
            threshold.metrics,
            "speech_min_rms",
            "contract_min_rms",
            default=0.001,
        )
        token_threshold = _strictest_threshold(
            threshold.metrics,
            "speech_min_token_match",
            "contract_token_match",
            default=0.8,
        )
        depth_threshold = _strictest_threshold(
            threshold.metrics,
            "depth_token_match_rate",
            default=0.7,
        )
        audio_threshold = _strictest_threshold(
            threshold.metrics,
            "audio_token_match_rate",
            default=0.7,
        )
        frame_threshold = _strictest_threshold(
            threshold.metrics,
            "frame_exact_match_rate",
            "speech_min_frame_exact",
            default=0.7,
        )

        metrics = {}
        has_wav = trt_wav is not None and isinstance(trt_wav, str) and len(trt_wav) > 0
        metrics["has_audio"] = MetricResult(
            value=1.0 if has_wav else 0.0,
            threshold=1.0,
            operator="==",
            passed=has_wav,
        )

        rms_value = float(trt_rms) if trt_rms is not None else 0.0
        rms_ok = trt_rms is not None and rms_value >= min_rms
        metrics["rms"] = MetricResult(
            value=rms_value,
            threshold=min_rms,
            operator=">=",
            passed=rms_ok,
            note="configured by speech_min_rms",
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
            trt_arr = np.asarray(trt_tokens)
            ref_arr = np.asarray(ref_tokens)
            layout_compatible = (
                trt_arr.ndim == 2
                and ref_arr.ndim == 2
                and trt_arr.shape[1] == ref_arr.shape[1]
                and trt_arr.shape[1] >= 2
            )
            metrics["token_layout_compatible"] = MetricResult(
                value=1.0 if layout_compatible else 0.0,
                threshold=1.0,
                operator="==",
                passed=layout_compatible,
                note=(
                    f"TRT shape={trt_arr.shape}, reference shape={ref_arr.shape}; "
                    "expected [frames, depth+audio codebooks]"
                ),
            )

            frame_count_match = layout_compatible and trt_arr.shape[0] == ref_arr.shape[0]
            metrics["frame_count_match"] = MetricResult(
                value=1.0 if frame_count_match else 0.0,
                threshold=1.0,
                operator="==",
                passed=frame_count_match,
                note=(f"TRT frames={trt_arr.shape[0]}, reference frames={ref_arr.shape[0]}"),
            )

            frame_count = (
                min(trt_arr.shape[0], ref_arr.shape[0])
                if layout_compatible
                else 0
            )
            if frame_count > 0:
                trt_aligned = trt_arr[:frame_count]
                ref_aligned = ref_arr[:frame_count]

                token_match = float(np.mean(trt_aligned == ref_aligned))
                metrics["token_match"] = MetricResult(
                    value=token_match,
                    threshold=token_threshold,
                    operator=">=",
                    passed=token_match >= token_threshold,
                    note="configured by speech_min_token_match",
                )

                depth_match = float(np.mean(trt_aligned[:, 0] == ref_aligned[:, 0]))
                metrics["depth_token_match_rate"] = MetricResult(
                    value=depth_match,
                    threshold=depth_threshold,
                    operator=">=",
                    passed=depth_match >= depth_threshold,
                )

                audio_match = float(np.mean(trt_aligned[:, 1:] == ref_aligned[:, 1:]))
                metrics["audio_token_match_rate"] = MetricResult(
                    value=audio_match,
                    threshold=audio_threshold,
                    operator=">=",
                    passed=audio_match >= audio_threshold,
                )

                frame_exact = float(np.mean(np.all(
                    trt_aligned == ref_aligned,
                    axis=1,
                )))
                metrics["frame_exact_match_rate"] = MetricResult(
                    value=frame_exact,
                    threshold=frame_threshold,
                    operator=">=",
                    passed=frame_exact >= frame_threshold,
                    note=(
                        "strictest of frame_exact_match_rate and "
                        "speech_min_frame_exact"
                    ),
                )
            elif layout_compatible:
                metrics["non_empty_reference_overlap"] = MetricResult(
                    value=0.0,
                    threshold=1.0,
                    operator="==",
                    passed=False,
                )

        rule = (
            "audio health + frame count + aggregate token match + "
            "depth token match + audio token match + frame exact match"
        )
        if all(metric.passed for metric in metrics.values()):
            return make_pass("full_generation", metrics, rule)
        return make_fail(
            "full_generation",
            metrics,
            rule,
            "PersonaPlex speech response health check failed",
        )

plugin = PersonaPlexSpeechToSpeechPlugin()
