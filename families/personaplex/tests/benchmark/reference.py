#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the family-owned PersonaPlex reference."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from qualification_tests.benchmark_qualification.performance import reference_harness


def _materialize(summary: dict[str, Any], output: Path) -> None:
    import soundfile as sound

    audio = summary.pop("_audio_f32")
    artifact = output.with_suffix(".audio.wav").resolve()
    artifact.parent.mkdir(parents=True, exist_ok=True)
    sound.write(artifact, audio, int(summary["sample_rate"]), subtype="FLOAT")
    summary["audio_artifact"] = str(artifact)


def _load(
    arguments: Any,
    request: Mapping[str, Any],
    options: Mapping[str, Any],
) -> reference_harness.Session:
    if arguments.mode != "pytorch-eager":
        raise ValueError("PersonaPlex reference requires pytorch-eager mode")
    official_repo = Path(str(options.get("official_repo", ""))).resolve()
    if not official_repo.is_dir():
        raise ValueError("PersonaPlex reference requires official_repo")
    if importlib.util.find_spec("sphn") is None:
        raise RuntimeError("PersonaPlex reference requires the official sphn package")
    sys.path[:0] = [str(official_repo / "moshi"), str(official_repo)]
    import torch
    from huggingface_hub import hf_hub_download
    from moshi.models import LMGen, loaders
    from moshi.models.lm import _iterate_audio, encode_from_sphn, load_audio
    from moshi.offline import warmup

    device = "cuda"
    common = {
        "local_files_only": arguments.local_files_only,
    }
    if arguments.revision:
        common["revision"] = arguments.revision
    mimi_weights = hf_hub_download(arguments.model, loaders.MIMI_NAME, **common)
    model_weights = hf_hub_download(arguments.model, loaders.MOSHI_NAME, **common)
    mimi = loaders.get_mimi(mimi_weights, device)
    other_mimi = loaders.get_mimi(mimi_weights, device)
    language_model = loaders.get_moshi_lm(model_weights, device=device).eval()
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
    source_audio = load_audio(str(request["audio_path"]), mimi.sample_rate)
    max_frames = int(request.get("max_new_tokens", options.get("max_frames", 100)))

    def invoke() -> Mapping[str, Any]:
        mimi.reset_streaming()
        other_mimi.reset_streaming()
        generator.reset_streaming()
        generated_frames = 0
        decoded_chunks = []

        def summary() -> Mapping[str, Any]:
            audio = (
                torch.cat(decoded_chunks, dim=-1).detach().float().cpu().reshape(-1).numpy()
                if decoded_chunks
                else torch.empty(0, dtype=torch.float32).numpy()
            )
            return {
                "audio_frames": generated_frames,
                "audio_samples": int(audio.size),
                "sample_rate": mimi.sample_rate,
                "_audio_f32": audio,
            }

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
                    decoded_chunks.append(mimi.decode(tokens[:, 1:9]))
                    other_mimi.decode(tokens[:, 1:9])
                    generated_frames += 1
                    if generated_frames >= max_frames:
                        return summary()
        return summary()

    return reference_harness.Session(
        invoke,
        "moshi",
        timing_scope="task-pipeline-call-wall",
        input_preparation_included=True,
        materialize=_materialize,
    )


def main(argv: Sequence[str] | None = None) -> int:
    return reference_harness.run(
        list(sys.argv[1:] if argv is None else argv), description=__doc__, load=_load
    )


if __name__ == "__main__":
    raise SystemExit(main())
