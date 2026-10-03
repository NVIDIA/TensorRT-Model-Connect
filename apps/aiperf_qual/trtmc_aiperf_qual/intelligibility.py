# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Task-level check for generated speech: ASR round trip (``tts_intelligibility``).

Text-to-speech models that sample (Bark, MagpieTTS) produce a different waveform on every stream
of random numbers, so per-sample parity with the native model cannot judge them. Both sides speak
the same sentences; an ASR model transcribes both, and TRTMC's corpus word error rate against the text
is judged against the native model's by the paired bootstrap (``noninferiority``, the check's gate).
Every TRTMC output must also be valid audio (``tts-validity``): finite, not silent, and of a duration
within VALID_DURATION_RATIO of the native output. Voice identity and prosody are not covered.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from . import absolute
from .config import Environment
from .generation import generate, generate_native
from .services import _serve_env
from .suites import build_suite

TRANSCRIBE = r"""
import json, sys
import numpy as np, soundfile, torch
from transformers import pipeline
items = json.loads(sys.argv[2])
asr = pipeline("automatic-speech-recognition", model=sys.argv[1], revision=sys.argv[3] or None,
               device=0 if torch.cuda.is_available() else -1)
rows = []
for path, rate in items:
    if path.endswith(".npy"):
        audio = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    else:
        audio, rate = soundfile.read(path, dtype="float32", always_2d=False)
        audio = audio.mean(axis=1) if audio.ndim == 2 else audio
    finite = bool(np.isfinite(audio).all())
    rms = float(np.sqrt(np.mean(np.square(np.nan_to_num(audio))))) if audio.size else 0.0
    text = asr({"raw": np.nan_to_num(audio), "sampling_rate": int(rate)})["text"] if audio.size else ""
    rows.append({"text": text, "seconds": audio.size / max(int(rate), 1), "finite": finite,
                 "dbfs": 20.0 * np.log10(rms) if rms > 0 else -1000.0})
print(json.dumps(rows))
"""
SILENCE_DBFS = -50.0
VALID_DURATION_RATIO = (0.5, 2.0)


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

    from .models import _suite

    suite = build_suite(_suite(check["suite"], model["catalog_profile"]), environment)
    native, _, _ = generate_native(environment, model, suite, python, out, "tts")
    audio = {side: [_audio(workdir, record) for workdir, record in outputs] for side, outputs in (
        ("candidate", generate(environment, model, "trtmc", out / "tts-candidate", suite)), ("native", native))}
    heard = {}
    for side, items in audio.items():
        present = [item for item in items if item]
        completed = subprocess.run([str(environment["serve_python"]), "-c", TRANSCRIBE, str(check.get(
            "asr_model", "openai/whisper-tiny")), json.dumps(present), str(check.get("asr_revision") or "")],
            capture_output=True, text=True,
            env={key: value for key, value in _serve_env(environment).items() if key != "HF_HUB_OFFLINE"},
            timeout=3600)
        if completed.returncode:
            raise RuntimeError(f"ASR failed: {completed.stderr[-400:]}")
        rows = iter(json.loads(completed.stdout.strip().splitlines()[-1]))
        heard[side] = [next(rows) if item else None for item in items]
    problems = [{"task": "tts", "gold": str(sample["request"].get("prompt", "")), "sample_id": sample["sample_id"]}
                for sample in suite.samples]
    sides = {side: {"observations": {"greedy": {index: {"text": row["text"]} for index, row in enumerate(rows) if row}},
                    "exit": {"greedy": 0}, "timings": {"greedy": {}}} for side, rows in heard.items()}
    item = {"suite": "tts-intelligibility", "metric": "wer",
            "gate": dict(check.get("gate") or {"margin": 2.0, "relative_margin": 0.10})}
    entry = absolute.judge(item, problems, sides["candidate"], sides["native"])
    entry.update(source="task", benchmark=f"ASR round trip ({check.get('asr_model')}, {suite.name})")
    return [entry, validity(problems, heard["candidate"], heard["native"])]


def validity(problems: list[dict[str, Any]], candidate: list[Mapping[str, Any] | None],
             native: list[Mapping[str, Any] | None]) -> dict[str, Any]:
    """Every TRTMC output is finite, not silent, and lasts VALID_DURATION_RATIO of the native output."""
    low, high = VALID_DURATION_RATIO
    failures = []
    for problem, mine, theirs in zip(problems, candidate, native):
        reason = ("no audio" if not mine else "non-finite samples" if not mine["finite"]
                  else f"silent ({mine['dbfs']:.1f} dBFS)" if mine["dbfs"] < SILENCE_DBFS
                  else f"{mine['seconds']:.2f} s vs native {theirs['seconds']:.2f} s"
                  if theirs and not low * theirs["seconds"] <= mine["seconds"] <= high * theirs["seconds"] else None)
        if reason:
            failures.append({"sample_id": problem["sample_id"], "explanation": reason})
    count = len(problems)
    return {"suite": "tts-validity", "source": "task", "benchmark": "audio validity (finite, not silent, duration)",
            "samples": count, "expected_samples": count, "passed": count - len(failures), "required_passes": count,
            "status": "pass" if not failures else "fail", "failures": failures[:10],
            "reasons": [f"{len(failures)} of {count} outputs invalid"] if failures else [],
            "gate": {"min_dbfs": SILENCE_DBFS, "duration_ratio": list(VALID_DURATION_RATIO)}}
