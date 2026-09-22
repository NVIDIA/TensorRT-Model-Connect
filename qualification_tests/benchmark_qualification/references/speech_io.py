#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-agnostic request and PCM mechanics for family-owned ASR references."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SpeechSample:
    sample_id: str
    gold_text: str
    waveform: Any
    sample_rate: int
    wav_path: Path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--request", required=True, type=Path)
    value.add_argument("--output", required=True, type=Path)
    return value


def load_request(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("speech reference request must be an object")
    return value


def prepare_samples(request: Mapping[str, Any], output: Path) -> list[SpeechSample]:
    import numpy as np
    from scipy.signal import resample_poly
    import soundfile as sf

    configured = request.get("samples")
    if not isinstance(configured, list) or not configured:
        raise ValueError("speech reference requires a non-empty samples list")
    audio_root = output.parent / "reference-audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    prepared = []
    for index, value in enumerate(configured):
        if not isinstance(value, Mapping):
            raise ValueError(f"speech reference sample {index} must be an object")
        sample_id = required_string(value.get("sample_id"), "sample_id")
        gold_text = required_string(value.get("gold_text"), "gold_text")
        source = Path(required_string(value.get("audio_path"), "audio_path"))
        if not source.is_file():
            raise FileNotFoundError(source)
        waveform, sample_rate = sf.read(source, dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)
        target_rate = 16_000
        if int(sample_rate) != target_rate:
            divisor = math.gcd(int(sample_rate), target_rate)
            waveform = resample_poly(
                waveform, target_rate // divisor, int(sample_rate) // divisor
            ).astype(np.float32)
        wav_path = audio_root / f"{index:04d}.wav"
        sf.write(wav_path, waveform, target_rate, subtype="PCM_16")
        prepared.append(SpeechSample(sample_id, gold_text, waveform, target_rate, wav_path))
    return prepared


def write_result(output: Path, samples: Sequence[SpeechSample], texts: Sequence[str]) -> None:
    if len(samples) != len(texts):
        raise ValueError("speech reference returned a different sample count")
    rows = []
    for sample, text in zip(samples, texts, strict=True):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"speech reference returned an empty transcript for {sample.sample_id}")
        rows.append(
            {
                "sample_id": sample.sample_id,
                "text": text.strip(),
                "audio_path": str(sample.wav_path),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"samples": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def model_id(request: Mapping[str, Any]) -> tuple[str, str | None]:
    model = required_string(request.get("model"), "model")
    revision = request.get("revision")
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise ValueError("revision must be a non-empty string when set")
    return model, revision


def torch_dtype(torch_module: Any, request: Mapping[str, Any]) -> Any:
    return {
        "fp16": torch_module.float16,
        "fp32": torch_module.float32,
        "bf16": torch_module.bfloat16,
    }[required_string(request.get("precision"), "precision")]
