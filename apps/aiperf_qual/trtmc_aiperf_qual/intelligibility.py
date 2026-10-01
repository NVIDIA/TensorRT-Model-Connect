# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Task-level check for generated speech: ASR round trip (``tts_intelligibility``).

Text-to-speech models that sample (Bark, MagpieTTS) produce a different waveform on every stream
of random numbers, so per-sample parity with the native model cannot judge them. Both sides speak
the same sentences; an ASR model transcribes both, and TRTMC's word error rate against the text
must stay within ``max_wer_increase`` of the native model's.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .config import Environment
from .generation import generate, generate_native
from .services import _serve_env
from .suites import build_suite

TRANSCRIBE = r"""
import json, sys
import numpy as np, soundfile, torch
from transformers import pipeline
items = json.loads(sys.argv[2])
asr = pipeline("automatic-speech-recognition", model=sys.argv[1], device=0 if torch.cuda.is_available() else -1)
texts = []
for path, rate in items:
    if path.endswith(".npy"):
        audio = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    else:
        audio, rate = soundfile.read(path, dtype="float32", always_2d=False)
        audio = audio.mean(axis=1) if audio.ndim == 2 else audio
    texts.append(asr({"raw": audio, "sampling_rate": int(rate)})["text"])
print(json.dumps(texts))
"""


def _audio(workdir: Path, record: Mapping[str, Any]) -> tuple[str, int] | None:
    """The generated audio of one request: the WAV the server wrote, else the saved sample array."""
    wavs = sorted(workdir.glob("output*.wav"))
    if wavs:
        return str(wavs[0]), 0
    observation = record.get("observation") or {}
    artifact = observation.get("artifact")
    if isinstance(artifact, str) and artifact.endswith(".npy") and Path(artifact).is_file():
        return artifact, int(observation.get("sample_rate") or 24000)
    return None


def run(environment: Environment, model: dict[str, Any], check: Mapping[str, Any], python: str,
        out: Path) -> dict[str, Any]:
    from trtmc_aiperf_plugins.accuracy import word_error_rate

    from .models import _suite

    suite = build_suite(_suite(check["suite"], model["catalog_profile"], environment.path("repo")), environment)
    native, _, _ = generate_native(environment, model, suite, python, out, "tts")
    audio = {side: [_audio(workdir, record) for workdir, record in outputs] for side, outputs in (
        ("candidate", generate(environment, model, "trtmc", out / "tts-candidate", suite)), ("native", native))}
    texts = {}
    for side, items in audio.items():
        present = [item for item in items if item]
        completed = subprocess.run([str(environment["serve_python"]), "-c", TRANSCRIBE, str(check.get(
            "asr_model", "openai/whisper-tiny")), json.dumps(present)], capture_output=True, text=True,
            env={key: value for key, value in _serve_env(environment).items() if key != "HF_HUB_OFFLINE"},
            timeout=3600)
        if completed.returncode:
            raise RuntimeError(f"ASR failed: {completed.stderr[-400:]}")
        transcripts = iter(json.loads(completed.stdout.strip().splitlines()[-1]))
        texts[side] = [next(transcripts) if item else "" for item in items]
    words = [str(sample["request"].get("prompt", "")) for sample in suite.samples]
    wer = {side: [word_error_rate(text, truth) for text, truth in zip(texts[side], words)] for side in texts}
    mean = {side: sum(values) / len(values) for side, values in wer.items()}
    limit = float(check.get("max_wer_increase", 0.1))
    passed = sum(c <= n + limit for c, n in zip(wer["candidate"], wer["native"]))
    failures = [{"sample_id": sample["sample_id"], "explanation": f"WER {c:.2f} vs native {n:.2f}",
                 "actual": texts["candidate"][index], "expected": texts["native"][index]}
                for index, (sample, c, n) in enumerate(zip(suite.samples, wer["candidate"], wer["native"]))
                if c > n + limit]
    return {"suite": "tts-intelligibility", "source": "task", "benchmark": f"ASR round trip ({suite.name})",
            "status": "pass" if mean["candidate"] <= mean["native"] + limit else "fail",
            "samples": len(suite.samples), "expected_samples": len(suite.samples), "passed": passed,
            "required_passes": None, "gate": {"max_mean_wer_increase": limit},
            "metrics": {"candidate_wer": mean["candidate"], "native_wer": mean["native"]}, "failures": failures[:10]}
