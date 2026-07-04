# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Magpie-owned text-to-audio contract plugin."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap

from tests.e2e_harness.contracts import (
    CompareResult,
    E2ECase,
    MetricResult,
    PluginRuntimeContext,
    StageOutput,
    ThresholdProfile,
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

logger = logging.getLogger(__name__)

_DEFAULT_HF_ASR_MODEL = "openai/whisper-large-v3-turbo"
_HF_ASR_TIMEOUT_S = 600

def _run_asr_roundtrip(
    wav_path: str,
    python: str,
    model_id: str,
    max_new_tokens: int = 256,
) -> dict[str, str] | None:
    script = textwrap.dedent(
        """
        import json

        import numpy as np
        import scipy.io.wavfile as wavfile
        import torch
        from scipy.signal import resample
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        wav_path = %(wav_path)r
        model_id = %(model_id)r
        max_new_tokens = %(max_new_tokens)d

        sample_rate, audio = wavfile.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if np.issubdtype(audio.dtype, np.integer):
            info = np.iinfo(audio.dtype)
            audio = audio.astype(np.float32) / max(abs(info.min), info.max)
        else:
            audio = audio.astype(np.float32)

        processor = AutoProcessor.from_pretrained(model_id)
        target_sample_rate = getattr(processor.feature_extractor, "sampling_rate", 16000)
        if sample_rate != target_sample_rate:
            target_len = int(round(len(audio) * float(target_sample_rate) / float(sample_rate)))
            audio = resample(audio, target_len).astype(np.float32)
            sample_rate = target_sample_rate

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )
        model.to(device)
        model.eval()

        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs = {
            k: (v.to(torch_dtype) if torch.is_floating_point(v) else v)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(json.dumps({
            "transcript": transcript,
            "backend": "hf_transformers",
            "model": model_id,
            "device": device,
        }))
        """
        % {
            "wav_path": wav_path,
            "model_id": model_id,
            "max_new_tokens": max_new_tokens,
        }
    )

    try:
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True,
            text=True,
            timeout=_HF_ASR_TIMEOUT_S,
        )
        if result.returncode != 0:
            logger.warning(
                "HF ASR failed (rc=%d, model=%s): %s",
                result.returncode,
                model_id,
                result.stderr[-1000:],
            )
            return None

        for line in reversed(result.stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            transcript = data.get("transcript")
            if isinstance(transcript, str):
                return {k: str(v) for k, v in data.items()}
        logger.warning("HF ASR produced no parseable transcript (model=%s)", model_id)
        return None
    except Exception as e:
        logger.warning("HF ASR subprocess failed (model=%s): %s", model_id, e)
        return None

class MagpieTTSPlugin:
    reference_families = ["tts_magpie"]
    user_contract = "tts_audio"

    def configure_reference(self, case: E2ECase) -> dict:
        return {}

    def verify(
        self,
        trt_output: StageOutput,
        ref_output: StageOutput,
        case: E2ECase,
        threshold: ThresholdProfile,
        *,
        runtime_context: PluginRuntimeContext | None = None,
    ) -> CompareResult:
        trt_wav = trt_output.data.get("wav_path")
        trt_rms = trt_output.data.get("rms")
        trt_duration = trt_output.data.get("duration_s")
        ref_wav = ref_output.data.get("wav_path")
        ref_rms = ref_output.data.get("rms")
        ref_duration = ref_output.data.get("duration_s")
        ref_returncode = ref_output.data.get(
            "returncode", ref_output.metadata.get("returncode"))

        min_rms = threshold.metrics.get("contract_min_rms", 0.001)
        min_duration = threshold.metrics.get("contract_min_duration_s", 0.1)
        max_duration = threshold.metrics.get("contract_max_duration_s", 30.0)

        metrics: dict[str, MetricResult] = {}

        ref_completed = ref_returncode == 0
        metrics["reference_completed"] = MetricResult(
            value=1.0 if ref_completed else 0.0,
            threshold=1.0,
            operator="==",
            passed=ref_completed,
            note="original NeMo reference completed successfully",
        )

        has_ref_wav = (
            ref_completed
            and isinstance(ref_wav, str)
            and os.path.isfile(ref_wav)
        )
        metrics["reference_has_audio"] = MetricResult(
            value=1.0 if has_ref_wav else 0.0,
            threshold=1.0,
            operator="==",
            passed=has_ref_wav,
            note="original NeMo reference WAV file produced",
        )

        if ref_rms is not None:
            ref_rms_ok = float(ref_rms) >= min_rms
            metrics["reference_rms"] = MetricResult(
                value=float(ref_rms),
                threshold=min_rms,
                operator=">=",
                passed=ref_rms_ok,
                note="original NeMo reference non-silence check",
            )

        if ref_duration is not None:
            ref_dur = float(ref_duration)
            ref_dur_ok = min_duration <= ref_dur <= max_duration
            metrics["reference_duration_s"] = MetricResult(
                value=ref_dur,
                threshold=min_duration,
                operator=">=",
                passed=ref_dur_ok,
                note=(
                    "original NeMo reference range "
                    f"[{min_duration}, {max_duration}]"
                ),
            )

        has_wav = trt_wav is not None and isinstance(trt_wav, str) and os.path.isfile(trt_wav)
        metrics["has_audio"] = MetricResult(
            value=1.0 if has_wav else 0.0,
            threshold=1.0,
            operator="==",
            passed=has_wav,
            note="WAV file produced",
        )

        if trt_rms is not None:
            rms_ok = float(trt_rms) >= min_rms
            metrics["rms"] = MetricResult(
                value=float(trt_rms),
                threshold=min_rms,
                operator=">=",
                passed=rms_ok,
                note="non-silence check",
            )

        if trt_duration is not None:
            dur = float(trt_duration)
            dur_ok = min_duration <= dur <= max_duration
            metrics["duration_s"] = MetricResult(
                value=dur,
                threshold=min_duration,
                operator=">=",
                passed=dur_ok,
                note=f"range [{min_duration}, {max_duration}]",
            )

        input_prompt = case.inputs.get("prompt", "")
        asr_transcript = None
        asr_info: dict[str, str] | None = None
        asr_model = str(
            case.metadata.get("tts_asr_model")
            or os.environ.get("TRTMC_TTS_ASR_MODEL")
            or _DEFAULT_HF_ASR_MODEL
        )
        asr_python = (
            (runtime_context.reference_python if runtime_context else "")
            or (runtime_context.hf_python if runtime_context else "")
            or (runtime_context.runtime_python if runtime_context else "")
            or sys.executable
        )

        if has_wav and input_prompt:
            asr_info = _run_asr_roundtrip(trt_wav, asr_python, asr_model)
            if asr_info:
                asr_transcript = asr_info.get("transcript")

        if asr_transcript is not None:
            norm_transcript = normalize_text(asr_transcript)
            norm_prompt = normalize_text(input_prompt)

            ned = levenshtein_ned(norm_transcript, norm_prompt)
            ned_threshold = threshold.metrics.get("contract_asr_ned_threshold", 0.15)

            exact = norm_transcript == norm_prompt
            metrics["asr_exact_match"] = MetricResult(
                value=1.0 if exact else 0.0,
                threshold=None,
                operator="==",
                passed=True,
                note="informational; NED is the gate",
            )
            metrics["asr_ned"] = MetricResult(
                value=ned,
                threshold=ned_threshold,
                operator="<=",
                passed=ned <= ned_threshold,
                note=(
                    f"backend={asr_info.get('backend', 'hf_transformers')} "
                    f"model={asr_info.get('model', asr_model)} "
                    f"device={asr_info.get('device', 'unknown')}; "
                    f"transcript: '{asr_transcript[:80]}...'"
                    if len(asr_transcript) > 80
                    else
                    f"backend={asr_info.get('backend', 'hf_transformers')} "
                    f"model={asr_info.get('model', asr_model)} "
                    f"device={asr_info.get('device', 'unknown')}; "
                    f"transcript: '{asr_transcript}'"
                ),
            )
        elif has_wav and input_prompt:
            metrics["asr_roundtrip"] = MetricResult(
                value=0.0,
                threshold=1.0,
                operator="==",
                passed=False,
                note=f"HF ASR round-trip failed for model={asr_model}",
            )

        ref_asr_info = None
        if has_ref_wav and input_prompt:
            ref_asr_info = _run_asr_roundtrip(
                ref_wav, asr_python, asr_model)

        if ref_asr_info and ref_asr_info.get("transcript") is not None:
            ref_transcript = str(ref_asr_info["transcript"])
            ref_ned = levenshtein_ned(
                normalize_text(ref_transcript), normalize_text(input_prompt))
            ned_threshold = threshold.metrics.get(
                "contract_asr_ned_threshold", 0.15)
            metrics["reference_asr_ned"] = MetricResult(
                value=ref_ned,
                threshold=ned_threshold,
                operator="<=",
                passed=ref_ned <= ned_threshold,
                note=(
                    "original NeMo reference; "
                    f"backend={ref_asr_info.get('backend', 'hf_transformers')} "
                    f"model={ref_asr_info.get('model', asr_model)}; "
                    f"transcript: '{ref_transcript[:80]}'"
                ),
            )
        elif has_ref_wav and input_prompt:
            metrics["reference_asr_roundtrip"] = MetricResult(
                value=0.0,
                threshold=1.0,
                operator="==",
                passed=False,
                note=(
                    "HF ASR round-trip failed for the original NeMo "
                    f"reference with model={asr_model}"
                ),
            )

        gated = [m for m in metrics.values() if m.threshold is not None]
        passed = all(m.passed for m in gated)

        rule = "TRT and original NeMo reference audio exist AND health checks pass"
        if input_prompt:
            rule += " AND both ASR transcript NED values <= threshold"

        if passed:
            return make_pass("full_generation", metrics, rule)
        return make_fail(
            "full_generation",
            metrics,
            rule,
            "TTS audio contract failed",
        )

plugin = MagpieTTSPlugin()
