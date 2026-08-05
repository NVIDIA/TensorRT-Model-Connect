#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official PersonaPlex greedy speech pipeline for one WAV."""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path
from typing import Sequence


def _snapshot_revision(path: str) -> str:
    parts = Path(path).resolve().parts
    try:
        index = parts.index("snapshots")
    except ValueError:
        return ""
    return parts[index + 1] if index + 1 < len(parts) else ""


def _write_pcm16(path: Path, audio, sample_rate: int) -> None:
    import numpy as np

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


def run(arguments: argparse.Namespace) -> None:
    official_repo = arguments.official_repo.resolve()
    sys.path[:0] = [str(official_repo / "moshi"), str(official_repo)]

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from moshi.models import LMGen, loaders
    from moshi.models.lm import _iterate_audio, encode_from_sphn, load_audio
    from moshi.offline import warmup

    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[arguments.precision]
    download_options = {"local_files_only": arguments.local_files_only}
    if arguments.revision:
        download_options["revision"] = arguments.revision
    mimi_weights = hf_hub_download(
        arguments.model,
        loaders.MIMI_NAME,
        **download_options,
    )
    model_weights = hf_hub_download(
        arguments.model,
        loaders.MOSHI_NAME,
        **download_options,
    )
    device = "cuda"
    mimi = loaders.get_mimi(mimi_weights, device)
    other_mimi = loaders.get_mimi(mimi_weights, device)
    language_model = loaders.get_moshi_lm(
        model_weights,
        device=device,
        dtype=dtype,
    ).eval()
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
    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    generator.streaming_forever(1)
    warmup(mimi, other_mimi, generator, device, frame_size)
    mimi.reset_streaming()
    other_mimi.reset_streaming()
    generator.reset_streaming()

    source_audio = load_audio(str(arguments.input_wav), mimi.sample_rate)
    input_tokens = []
    output_text_tokens = []
    output_tokens = []
    output_audio = []
    with torch.inference_mode():
        for encoded in encode_from_sphn(
            mimi,
            _iterate_audio(
                source_audio,
                sample_interval_size=frame_size,
                pad=True,
            ),
            max_batch=1,
        ):
            for index in range(encoded.shape[-1]):
                input_tokens.append(
                    encoded[0, :, index].detach().cpu().numpy()
                )
                tokens = generator.step(encoded[:, :, index : index + 1])
                if tokens is None:
                    continue
                output_text_tokens.append(
                    tokens[0, 0, 0].detach().cpu().numpy()
                )
                output_tokens.append(
                    tokens[0, 1:9, 0].detach().cpu().numpy()
                )
                decoded = mimi.decode(tokens[:, 1:9])
                other_mimi.decode(tokens[:, 1:9])
                output_audio.append(
                    decoded[0, 0].detach().cpu().to(torch.float32).numpy()
                )
                if len(output_tokens) >= arguments.max_frames:
                    break
            if len(output_tokens) >= arguments.max_frames:
                break

    if not output_tokens or not output_audio:
        raise RuntimeError("official PersonaPlex produced no speech frames")
    input_tokens_array = np.stack(input_tokens).astype(np.int32, copy=False)
    text_tokens_array = np.stack(output_text_tokens).astype(np.int32, copy=False)
    tokens_array = np.stack(output_tokens).astype(np.int32, copy=False)
    audio_array = np.concatenate(output_audio).astype(np.float32, copy=False)
    arguments.tokens_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(arguments.tokens_output, tokens_array, allow_pickle=False)
    np.save(arguments.input_tokens_output, input_tokens_array, allow_pickle=False)
    np.save(arguments.text_tokens_output, text_tokens_array, allow_pickle=False)
    _write_pcm16(arguments.audio_output, audio_array, int(mimi.sample_rate))
    arguments.metadata_output.write_text(
        json.dumps(
            {
                "model_id": arguments.model,
                "resolved_revision": (
                    arguments.revision
                    or _snapshot_revision(model_weights)
                    or "unresolved"
                ),
                "reference_source_revision": arguments.source_revision,
                "decoding": "greedy",
                "precision": arguments.precision,
                "sample_rate": int(mimi.sample_rate),
                "num_samples": int(audio_array.size),
                "num_frames": int(tokens_array.shape[0]),
                "token_shape": list(tokens_array.shape),
                "input_token_shape": list(input_tokens_array.shape),
                "text_token_shape": list(text_tokens_array.shape),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="")
    parser.add_argument("--input-wav", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument(
        "--precision",
        choices=("fp16", "bf16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--tokens-output", type=Path, required=True)
    parser.add_argument("--input-tokens-output", type=Path, required=True)
    parser.add_argument("--text-tokens-output", type=Path, required=True)
    parser.add_argument("--audio-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    run(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
