# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Magpie TTS owner reference."""

from __future__ import annotations

import gc
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

from tools.reference.speech import (
    _safe_sample_filename,
    _transcribe_tts,
    _wav_metrics,
    _write_wav_pcm16,
)


SPEAKER_ENCODER_REPO = "Edresson/Speaker_Encoder_H_ASP"
SPEAKER_ENCODER_FILENAME = "pytorch_model.bin"
SPEAKER_ENCODER_URL = (
    "https://huggingface.co/Edresson/Speaker_Encoder_H_ASP/resolve/main/"
    "pytorch_model.bin"
)


def _resolve_archive(arguments: Any) -> Path:
    model_path = Path(arguments.model)
    if model_path.is_file() and model_path.suffix == ".nemo":
        return model_path
    if model_path.is_dir():
        archives = sorted(model_path.glob("*.nemo"))
        if not archives:
            raise FileNotFoundError(f"Magpie NeMo archive is missing under {arguments.model}")
        return archives[0]
    from huggingface_hub import hf_hub_download

    kwargs: dict[str, Any] = {
        "repo_id": arguments.model,
        "filename": "magpie_tts_multilingual_357m.nemo",
        "local_files_only": arguments.local_files_only,
    }
    if arguments.model_revision:
        kwargs["revision"] = arguments.model_revision
    return Path(hf_hub_download(**kwargs))


def _load_runtime(arguments: Any, torch: Any) -> Any:
    import fsspec
    from huggingface_hub import hf_hub_download
    from nemo.collections.tts.models import MagpieTTSModel

    speaker_checkpoint = hf_hub_download(
        repo_id=SPEAKER_ENCODER_REPO,
        filename=SPEAKER_ENCODER_FILENAME,
        local_files_only=arguments.local_files_only,
    )
    original_open = fsspec.open

    def offline_open(path: Any, *args: Any, **kwargs: Any) -> Any:
        if str(path).split("?", 1)[0] == SPEAKER_ENCODER_URL:
            path = speaker_checkpoint
        return original_open(path, *args, **kwargs)

    fsspec.open = offline_open
    model = MagpieTTSModel.restore_from(restore_path=str(_resolve_archive(arguments)))
    return model.eval().to(torch.device(arguments.device))


def run(
    arguments: Any,
    manifest: Mapping[str, Any],
    prompts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    import numpy as np
    import torch

    generation = manifest.get("generation", {})
    generation = generation if isinstance(generation, Mapping) else {}
    scoring = manifest.get("scoring", {})
    scoring = scoring if isinstance(scoring, Mapping) else {}
    seed = arguments.seed if arguments.seed is not None else int(generation.get("seed", 42))
    model = _load_runtime(arguments, torch)
    output_dir = arguments.predictions.parent / "hf_audio"
    responses = []
    for prompt in prompts:
        sample_seed = seed + int(prompt.get("eval_index", 0))
        random.seed(sample_seed)
        np.random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)
        started = time.perf_counter()
        with torch.inference_mode():
            tensor, length = model.do_tts(
                transcript=str(prompt.get("prompt", "")),
                language=str(prompt.get("language", "en") or "en"),
                use_cfg=True,
            )
        audio = tensor.detach().cpu().numpy().reshape(-1)
        audio = audio[: int(length.item()) if length.numel() else len(audio)]
        wav_path = output_dir / _safe_sample_filename(str(prompt.get("sample_id", "")), ".wav")
        _write_wav_pcm16(wav_path, audio, 22050)
        metrics = _wav_metrics(wav_path)
        responses.append(
            {
                "sample_id": str(prompt.get("sample_id", "")),
                "output_text": "",
                "wav_path": str(wav_path),
                "wav_exists": True,
                "rms": metrics["rms"],
                "duration_s": metrics["duration_s"],
                "sample_rate": metrics["sample_rate"],
                "wall_ms": (time.perf_counter() - started) * 1000.0,
                "source": "hf",
            }
        )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    transcripts = _transcribe_tts(
        arguments,
        [Path(row["wav_path"]) for row in responses],
        str(scoring.get("asr_model", "openai/whisper-large-v3-turbo")),
    )
    for row, transcript in zip(responses, transcripts, strict=True):
        row["output_text"] = transcript
        row["asr_transcript"] = transcript
    return responses
