# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bark-owned text-to-audio reference."""

from __future__ import annotations

import gc
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

from tools.reference.speech import (
    _model_dtype,
    _safe_sample_filename,
    _transcribe_tts,
    _wav_metrics,
    _write_wav_pcm16,
)


def _load_runtime(arguments: Any, torch: Any) -> tuple[Any, Any]:
    from transformers import AutoProcessor, BarkModel, logging

    logging.set_verbosity_error()
    kwargs: dict[str, Any] = {
        "trust_remote_code": arguments.trust_remote_code,
        "local_files_only": arguments.local_files_only,
    }
    if arguments.model_revision:
        kwargs["revision"] = arguments.model_revision
    processor = AutoProcessor.from_pretrained(arguments.model, **kwargs)
    model = BarkModel.from_pretrained(
        arguments.model,
        torch_dtype=_model_dtype(torch, arguments.dtype),
        **kwargs,
    )
    return processor, model.eval().to(torch.device(arguments.device))


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
    device = torch.device(arguments.device)
    processor, model = _load_runtime(arguments, torch)
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
        inputs = processor(str(prompt.get("prompt", "")), return_tensors="pt")
        inputs = inputs.to(device) if hasattr(inputs, "to") else {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            audio = model.generate(**inputs).detach().cpu().numpy().reshape(-1)
        sample_rate = int(model.generation_config.sample_rate)
        wav_path = output_dir / _safe_sample_filename(str(prompt.get("sample_id", "")), ".wav")
        _write_wav_pcm16(wav_path, audio, sample_rate)
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
