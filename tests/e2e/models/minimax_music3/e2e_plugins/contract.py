# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-Music3's text_to_audio contract.

The comparator beside this file scores what the runner measures directly --
that a waveform exists, carries signal and runs about as long as asked. None of
those say the audio contains the lyrics, and that is the whole point of the
onboarding request. The transcription round-trip is what says it, and it lives
here because the contract owns it: the runner produces a WAV, this reads it
back through an ASR model and turns the transcript into a distance.

Held here rather than in shared harness code, per the architecture's rule, so
the semantics cannot drift across families through a common helper.

Two numbers differ from the speech families this borrows from. The threshold is
0.55 rather than 0.15, because these are sung lyrics over twenty seconds that do
not reach the end of the text they are scored against -- the reference pipeline
itself scores 0.3571 on this case. And the transcript is compared against the
lyrics, which travel in the prompt, not against the caption.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import textwrap

from .contracts import (
    CompareResult,
    E2ECase,
    MetricResult,
    PluginRuntimeContext,
    StageOutput,
    ThresholdProfile,
)

logger = logging.getLogger(__name__)

#: Whisper large-v3-turbo is what the repository's other audio contracts use.
DEFAULT_ASR_MODEL = "openai/whisper-large-v3-turbo"

#: Twenty seconds of music takes longer to transcribe than a spoken sentence,
#: and the model is downloaded on first use.
ASR_TIMEOUT_S = 900

#: Structure tags are directions, not words to sing. Scoring them against a
#: transcript that will never contain them only inflates the distance.
_TAG = "["


def normalize_text(text: str) -> str:
    """Lowercase, drop structure tags and punctuation, collapse whitespace."""

    import re

    without_tags = re.sub(r"\[[^\]]*\]", " ", text.lower())
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", without_tags).split())


def levenshtein_ned(hypothesis: str, reference: str) -> float:
    """Normalised edit distance in [0, 1]; 0 is identical."""

    if not hypothesis and not reference:
        return 0.0
    previous = list(range(len(reference) + 1))
    for i, left in enumerate(hypothesis, start=1):
        current = [i]
        for j, right in enumerate(reference, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (left != right))
            )
        previous = current
    return previous[-1] / max(len(hypothesis), len(reference), 1)


def _asr_script(wav_path: str, model_id: str) -> str:
    """Return the transcription program run in the reference interpreter.

    It runs out of process because the ASR model and the bundle would otherwise
    share a GPU that has just held five engines.
    """

    return textwrap.dedent(
        """
        import json, struct
        import numpy as np
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        wav_path = %(wav_path)r
        model_id = %(model_id)r

        # Read the header directly: this model writes IEEE float32, which the
        # standard library's wave module refuses.
        raw = open(wav_path, "rb").read()
        audio_format, channels, sample_rate = struct.unpack("<HHI", raw[20:28])
        bits = struct.unpack("<H", raw[34:36])[0]
        payload = raw[44:]
        if audio_format == 3 and bits == 32:
            count = len(payload) // 4
            audio = np.frombuffer(payload[: count * 4], dtype="<f4").astype(np.float32)
        elif bits == 16:
            count = len(payload) // 2
            audio = np.frombuffer(payload[: count * 2], dtype="<i2").astype(np.float32) / 32768.0
        else:
            raise SystemExit("unsupported wav encoding")

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        processor = AutoProcessor.from_pretrained(model_id)
        target = getattr(processor.feature_extractor, "sampling_rate", 16000)
        if sample_rate != target:
            index = np.linspace(0, len(audio) - 1, int(len(audio) * target / sample_rate))
            audio = audio[index.astype(np.int64)]

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=dtype)
        model.to(device).eval()

        inputs = processor(audio, sampling_rate=target, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs = {
            k: (v.to(dtype) if torch.is_floating_point(v) else v) for k, v in inputs.items()
        }
        with torch.no_grad():
            ids = model.generate(
                **inputs, max_new_tokens=440, language="en", task="transcribe"
            )
        transcript = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        print(json.dumps({
            "transcript": transcript,
            "backend": "hf_transformers",
            "model": model_id,
            "device": device,
        }))
        """
        % {"wav_path": wav_path, "model_id": model_id}
    )


def run_asr_roundtrip(wav_path: str, python: str, model_id: str) -> dict | None:
    """Transcribe one WAV, or return None with the reason logged."""

    try:
        result = subprocess.run(
            [python, "-c", _asr_script(wav_path, model_id)],
            capture_output=True,
            text=True,
            timeout=ASR_TIMEOUT_S,
        )
    except Exception as error:
        logger.warning("ASR subprocess failed (model=%s): %s", model_id, error)
        return None

    if result.returncode != 0:
        logger.warning(
            "ASR failed (rc=%d, model=%s): %s", result.returncode, model_id,
            result.stderr[-1000:],
        )
        return None

    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload.get("transcript"), str):
            return {key: str(value) for key, value in payload.items()}
    logger.warning("ASR produced no parseable transcript (model=%s)", model_id)
    return None


class MinimaxMusic3Contract:
    reference_families = ["music_minimax_music3"]
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
        """Add the transcription metric the comparator alone cannot produce."""

        metrics: dict[str, MetricResult] = {}
        wav_path = trt_output.data.get("wav_path")
        if not wav_path:
            return CompareResult(
                stage_name="full_generation",
                status="failed",
                metrics=metrics,
                composite_rule="asr_ned <= threshold",
                message="no waveform to transcribe",
            )

        model_id = str(case.metadata.get("tts_asr_model") or DEFAULT_ASR_MODEL)
        python = sys.executable
        if runtime_context is not None:
            python = str(getattr(runtime_context, "reference_python", None) or python)

        info = run_asr_roundtrip(str(wav_path), python, model_id)
        if info is None:
            metrics["asr_ned"] = MetricResult(
                value=float("nan"),
                threshold=float(threshold.metrics["contract_asr_ned_threshold"]),
                operator="<=",
                passed=False,
                note=f"ASR round-trip did not run for model={model_id}",
            )
            return CompareResult(
                stage_name="full_generation",
                status="failed",
                metrics=metrics,
                composite_rule="asr_ned <= threshold",
                message="the lyric check could not run",
            )

        # The lyrics travel in the prompt; the caption is the music description
        # and is deliberately not scored.
        lyrics = (getattr(case, "inputs", {}) or {}).get("prompt", "")
        transcript = info["transcript"]
        ned = levenshtein_ned(normalize_text(transcript), normalize_text(lyrics))
        limit = float(threshold.metrics["contract_asr_ned_threshold"])
        excerpt = transcript if len(transcript) <= 80 else transcript[:80] + "..."
        metrics["asr_ned"] = MetricResult(
            value=ned,
            threshold=limit,
            operator="<=",
            passed=ned <= limit,
            note=f"model={info.get('model')} device={info.get('device')}; "
                 f"transcript: '{excerpt}'",
        )
        passed = ned <= limit
        return CompareResult(
            stage_name="full_generation",
            status="passed" if passed else "failed",
            metrics=metrics,
            composite_rule="asr_ned <= threshold",
            message="Lyrics transcribed within the contract's distance"
            if passed
            else f"the audio does not carry the lyrics (NED {ned:.4f})",
        )


plugin = MinimaxMusic3Contract()
