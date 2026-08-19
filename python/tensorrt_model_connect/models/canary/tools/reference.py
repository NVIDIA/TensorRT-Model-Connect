# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canary-owned NeMo ASR reference."""

from __future__ import annotations

import gc
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from tools.reference.speech import (
    _asr_row,
    _audio_for_prompt,
    _safe_sample_filename,
    _transcription_text,
    _write_wav_pcm16,
)


def _resolve_archive(arguments: Any) -> Path:
    model_path = Path(arguments.model)
    if model_path.is_file() and model_path.suffix == ".nemo":
        return model_path
    if model_path.is_dir():
        archives = sorted(model_path.glob("*.nemo"))
    else:
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id=arguments.model,
                allow_patterns=["*.nemo"],
                local_files_only=arguments.local_files_only,
                **({"revision": arguments.model_revision} if arguments.model_revision else {}),
            )
        )
        archives = sorted(snapshot.glob("*.nemo"))
    if not archives:
        raise FileNotFoundError(f"Canary NeMo archive is missing for {arguments.model}")
    return archives[0]


def _load_model(arguments: Any) -> Any:
    import nemo.collections.asr as nemo_asr

    if arguments.local_files_only:
        return nemo_asr.models.ASRModel.restore_from(
            restore_path=str(_resolve_archive(arguments)),
            map_location=arguments.device,
        )
    return nemo_asr.models.ASRModel.from_pretrained(
        arguments.model,
        map_location=arguments.device,
    )


def run(
    arguments: Any,
    manifest: Mapping[str, Any],
    prompts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Run Canary through its upstream NeMo runtime."""

    generation = manifest.get("generation", {})
    generation = generation if isinstance(generation, Mapping) else {}
    target_rate = int(generation.get("sample_rate", 16000) or 16000)
    model = _load_model(arguments)
    if arguments.device != "cpu" and hasattr(model, "to"):
        model = model.to(arguments.device)
    model.eval()
    output_dir = arguments.predictions.parent / "hf_canary_audio"
    responses = []
    for prompt in prompts:
        audio, _source = _audio_for_prompt(prompt, target_rate)
        wav_path = output_dir / _safe_sample_filename(
            str(prompt.get("sample_id", "")), ".wav"
        )
        _write_wav_pcm16(wav_path, audio, target_rate)
        started = time.perf_counter()
        transcription = model.transcribe([str(wav_path)], batch_size=1)
        responses.append(
            _asr_row(
                prompt,
                _transcription_text(transcription),
                (time.perf_counter() - started) * 1000.0,
            )
        )
    del model
    gc.collect()
    return responses
