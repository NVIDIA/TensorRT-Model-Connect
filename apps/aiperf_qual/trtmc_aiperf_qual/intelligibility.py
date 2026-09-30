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
from .services import _serve_env, serving
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


def _audio(scratch: Path, record: Mapping[str, Any]) -> tuple[str, int] | None:
    """The generated audio of one request: the WAV the server wrote, else the saved sample array."""
    workdir = scratch / str(record["request_id"])
    wavs = sorted(workdir.glob("output*.wav"))
    if wavs:
        return str(wavs[0]), 0
    observation = record.get("observation") or {}
    artifact = observation.get("artifact")
    if isinstance(artifact, str) and artifact.endswith(".npy") and Path(artifact).is_file():
        return artifact, int(observation.get("sample_rate") or 24000)
    return None


def _generate(environment: Environment, model: dict[str, Any], backend: str, out: Path, suite: Any,
              python: str | None, precision: str | None) -> list[tuple[str, int] | None]:
    from .runner import _observations

    kwargs = {"precision": precision, "python": python} if backend != "trtmc" else {}
    with serving(environment, model, backend, out, keep_artifacts=True, **kwargs) as service:
        _observations(environment, service, model, suite, out / "aiperf")
    # AIPerf sends the samples in order, one at a time: the last records are the suite's.
    records = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines() if line.strip()]
    ordered = [record for record in records if record.get("route", "").startswith("/v1/tasks/")]
    return [_audio(out / "scratch", record) for record in ordered[-len(suite.samples):]]


def _native_audio(environment: Environment, model: dict[str, Any], suite: Any, python: str,
                  out: Path) -> list[tuple[str, int] | None]:
    """The native model's audio: the generic adapter, else the family's declared reference, at the
    Perf precisions in order."""
    from .runner import timing_precisions

    reference, errors = model["reference"], []
    for backend in dict.fromkeys([reference["backend"], reference.get("fallback") or reference["backend"]]):
        for precision in timing_precisions(reference):
            try:
                return _generate(environment, model, backend, out / f"tts-native-{backend}-{precision}", suite,
                                 python, precision)
            except Exception as error:  # noqa: BLE001 - try the next precision, then the fallback
                errors.append(f"{backend} {precision}: {type(error).__name__}: {str(error)[-200:]}")
    raise RuntimeError("; ".join(errors)[-1500:])


def run(environment: Environment, model: dict[str, Any], check: Mapping[str, Any], python: str,
        out: Path) -> dict[str, Any]:
    from trtmc_aiperf_plugins.accuracy import word_error_rate

    from .models import _suite

    suite = build_suite(_suite(check["suite"], model["catalog_profile"], environment.path("repo")), environment)
    audio = {"candidate": _generate(environment, model, "trtmc", out / "tts-candidate", suite, None, None),
             "native": _native_audio(environment, model, suite, python, out)}
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
