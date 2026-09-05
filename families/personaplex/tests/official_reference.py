#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the pinned official PersonaPlex greedy pipeline for one WAV."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import wave
from pathlib import Path
from typing import Sequence

import numpy as np


SOURCE_ENVIRONMENT = "TRTMC_REFERENCE_SOURCE_DIR"
SOURCE_ENTRYPOINT = "moshi/moshi/offline.py"
REFERENCE_SAMPLE_RATE = 24_000
_AUDIO_COMPAT = Path(__file__).with_name("personaplex_audio_compat")
_PRECISIONS = {"fp16", "bf16", "fp32"}


def _source() -> Path:
    value = os.environ.get(SOURCE_ENVIRONMENT)
    if not value:
        raise RuntimeError(
            f"{SOURCE_ENVIRONMENT} is required for the official PersonaPlex reference"
        )
    source = Path(value).resolve()
    if not (source / SOURCE_ENTRYPOINT).is_file():
        raise RuntimeError(f"official PersonaPlex source is missing {SOURCE_ENTRYPOINT}")
    return source


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "").strip()
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(_AUDIO_COMPAT), existing) if value
    )
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def _read_outputs(tokens_path: Path, audio_path: Path) -> dict:
    tokens = np.load(tokens_path, allow_pickle=False)
    with wave.open(str(audio_path), "rb") as audio:
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        sample_rate = audio.getframerate()
        num_samples = audio.getnframes()
        samples = np.frombuffer(audio.readframes(num_samples), dtype="<i2").astype(np.float32)
    if (
        tokens.ndim != 2
        or tokens.shape[0] < 1
        or tokens.shape[1] != 8
        or channels != 1
        or sample_width != 2
        or sample_rate != REFERENCE_SAMPLE_RATE
        or num_samples < 1
    ):
        raise RuntimeError(
            "official PersonaPlex emitted invalid artifacts: "
            f"tokens={tokens.shape}, channels={channels}, width={sample_width}, "
            f"rate={sample_rate}, samples={num_samples}"
        )
    samples *= 1.0 / 32768.0
    return {
        "speech_tokens": tokens.astype(np.int32, copy=False),
        "audio": str(audio_path),
        "sample_rate": sample_rate,
        "rms": float(np.sqrt(np.mean(samples * samples))),
    }


def generate(
    model_dir: Path,
    input_wav: Path,
    output_root: Path,
    *,
    max_frames: int,
    precision: str,
    timeout_s: int,
) -> dict:
    """Run the checked-out official source against the materialized checkpoint."""
    if precision not in _PRECISIONS:
        raise ValueError(f"unsupported PersonaPlex reference precision: {precision}")
    output_root.mkdir(parents=True, exist_ok=True)
    tokens_path = output_root / "official_tokens.npy"
    audio_path = output_root / "official_speech.wav"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--official-repo",
        str(_source()),
        "--model-dir",
        str(model_dir.resolve()),
        "--input-wav",
        str(input_wav.resolve()),
        "--max-frames",
        str(max_frames),
        "--precision",
        precision,
        "--tokens-output",
        str(tokens_path),
        "--audio-output",
        str(audio_path),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=_environment(),
    )
    if result.returncode:
        detail = (result.stderr or result.stdout or "no subprocess output")[-4000:]
        raise RuntimeError(f"official PersonaPlex reference failed: {detail}")
    return _read_outputs(tokens_path, audio_path)


def _write_pcm16(path: Path, audio, sample_rate: int) -> None:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise RuntimeError("official PersonaPlex produced invalid audio")
    pcm = np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def _run(arguments: argparse.Namespace) -> None:
    source = arguments.official_repo.resolve()
    if not (source / SOURCE_ENTRYPOINT).is_file():
        raise RuntimeError(f"official PersonaPlex source is missing {SOURCE_ENTRYPOINT}")
    sys.path[:0] = [str(source / "moshi"), str(source)]

    import torch
    from moshi.models import LMGen, loaders
    from moshi.models.lm import _iterate_audio, encode_from_sphn, load_audio
    from moshi.offline import warmup

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        arguments.precision
    ]
    mimi_weights = arguments.model_dir / loaders.MIMI_NAME
    model_weights = arguments.model_dir / loaders.MOSHI_NAME
    if not mimi_weights.is_file() or not model_weights.is_file():
        raise RuntimeError("materialized PersonaPlex checkpoint is missing official weights")
    device = "cuda"
    mimi = loaders.get_mimi(str(mimi_weights), device)
    other_mimi = loaders.get_mimi(str(mimi_weights), device)
    language_model = loaders.get_moshi_lm(str(model_weights), device=device, dtype=dtype).eval()
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    generator = LMGen(
        language_model,
        audio_silence_frame_cnt=0,
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
        use_sampling=False,
        temp=0.8,
        temp_text=0.7,
        top_k=250,
        top_k_text=25,
    )
    for streaming_model in (mimi, other_mimi, generator):
        streaming_model.streaming_forever(1)
    warmup(mimi, other_mimi, generator, device, frame_size)
    for streaming_model in (mimi, other_mimi, generator):
        streaming_model.reset_streaming()

    source_audio = load_audio(str(arguments.input_wav), mimi.sample_rate)
    output_tokens = []
    output_audio = []
    with torch.inference_mode():
        for encoded in encode_from_sphn(
            mimi,
            _iterate_audio(source_audio, sample_interval_size=frame_size, pad=True),
            max_batch=1,
        ):
            for index in range(encoded.shape[-1]):
                tokens = generator.step(encoded[:, :, index : index + 1])
                if tokens is None:
                    continue
                output_tokens.append(tokens[0, 1:9, 0].detach().cpu().numpy())
                decoded = mimi.decode(tokens[:, 1:9])
                other_mimi.decode(tokens[:, 1:9])
                output_audio.append(decoded[0, 0].detach().cpu().to(torch.float32).numpy())
                if len(output_tokens) >= arguments.max_frames:
                    break
            if len(output_tokens) >= arguments.max_frames:
                break
    if not output_tokens or not output_audio:
        raise RuntimeError("official PersonaPlex produced no speech frames")
    tokens = np.stack(output_tokens).astype(np.int32, copy=False)
    audio = np.concatenate(output_audio).astype(np.float32, copy=False)
    arguments.tokens_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(arguments.tokens_output, tokens, allow_pickle=False)
    _write_pcm16(arguments.audio_output, audio, int(mimi.sample_rate))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-wav", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--precision", choices=sorted(_PRECISIONS), required=True)
    parser.add_argument("--tokens-output", type=Path, required=True)
    parser.add_argument("--audio-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _run(_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
