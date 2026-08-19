# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron streaming ASR owner reference implementations."""

from __future__ import annotations

import gc
import json
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


OPTIONAL_CTC_STATE_KEYS = frozenset(
    {
        "ctc_decoder.decoder_layers.0.bias",
        "ctc_decoder.decoder_layers.0.weight",
    }
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
        raise FileNotFoundError(f"Nemotron ASR NeMo archive is missing for {arguments.model}")
    return archives[0]


def _extend_prompt_for_forward(
    torch_module: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    prompt = kwargs.get("prompt")
    if prompt is None or prompt.shape[1] == 0:
        return args, kwargs
    updated = dict(kwargs)
    updated["prompt"] = torch_module.cat((prompt, prompt[:, -1:, :]), dim=1)
    return args, updated


def _load_nemotron35_model(arguments: Any, archive: Path) -> tuple[Any, Any]:
    import torch
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModelWithPrompt
    from nemo.core.connectors.save_restore_connector import SaveRestoreConnector

    class Connector(SaveRestoreConnector):
        def load_instance_with_state_dict(
            self,
            instance: Any,
            state_dict: Mapping[str, Any],
            strict: bool,
        ) -> None:
            del strict
            incompatible = instance.load_state_dict(state_dict, strict=False)
            missing = frozenset(incompatible.missing_keys)
            unexpected = frozenset(incompatible.unexpected_keys)
            if missing not in (frozenset(), OPTIONAL_CTC_STATE_KEYS) or unexpected:
                raise RuntimeError(
                    "Nemotron 3.5 ASR archive state_dict mismatch: "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
            instance._set_model_restore_state(is_being_restored=False)

    device = torch.device(arguments.device)
    model = EncDecHybridRNNTCTCBPEModelWithPrompt.restore_from(
        str(archive),
        map_location=device,
        strict=False,
        save_restore_connector=Connector(),
    )
    model.eval()
    if hasattr(model, "to"):
        model.to(device)
    return torch, model


def _run_nemotron35(
    arguments: Any,
    prompts: Sequence[Mapping[str, Any]],
    generation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    torch, model = _load_nemotron35_model(arguments, _resolve_archive(arguments))
    target_rate = int(generation.get("sample_rate", 16000) or 16000)
    output_dir = arguments.predictions.parent / "hf_canary_audio"
    original_forward = model.forward

    def extended_forward(*args: Any, **kwargs: Any) -> Any:
        args, kwargs = _extend_prompt_for_forward(torch, args, kwargs)
        return original_forward(*args, **kwargs)

    model.forward = extended_forward
    responses = []
    try:
        for prompt in prompts:
            audio, _source = _audio_for_prompt(prompt, target_rate)
            sample_id = str(prompt.get("sample_id", ""))
            wav_path = output_dir / _safe_sample_filename(sample_id, ".wav")
            _write_wav_pcm16(wav_path, audio, target_rate)
            manifest_path = output_dir / _safe_sample_filename(
                sample_id, ".manifest.jsonl"
            )
            language = str(prompt.get("language") or generation.get("language") or "auto")
            manifest_path.write_text(
                json.dumps(
                    {
                        "audio_filepath": str(wav_path),
                        "duration": len(audio) / target_rate,
                        "text": "",
                        "lang": language,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            started = time.perf_counter()
            transcription = model.transcribe(str(manifest_path), batch_size=1, verbose=False)
            responses.append(
                _asr_row(
                    prompt,
                    _transcription_text(transcription),
                    (time.perf_counter() - started) * 1000.0,
                )
            )
    finally:
        model.forward = original_forward
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return responses


def _run_standard(
    arguments: Any,
    prompts: Sequence[Mapping[str, Any]],
    generation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import nemo.collections.asr as nemo_asr

    if arguments.local_files_only:
        model = nemo_asr.models.ASRModel.restore_from(
            restore_path=str(_resolve_archive(arguments)),
            map_location=arguments.device,
        )
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(
            arguments.model, map_location=arguments.device
        )
    if arguments.device != "cpu" and hasattr(model, "to"):
        model = model.to(arguments.device)
    model.eval()
    target_rate = int(generation.get("sample_rate", 16000) or 16000)
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
            _asr_row(prompt, _transcription_text(transcription), (time.perf_counter() - started) * 1000.0)
        )
    return responses


def run(
    arguments: Any,
    manifest: Mapping[str, Any],
    prompts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    generation = manifest.get("generation", {})
    generation = generation if isinstance(generation, Mapping) else {}
    if "nemotron-3.5-asr-streaming" in arguments.model.lower():
        return _run_nemotron35(arguments, prompts, generation)
    return _run_standard(arguments, prompts, generation)
