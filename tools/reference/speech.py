#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run ASR and TTS references directly through their upstream runtimes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import re
import struct
import sys
import time
from typing import Any, Mapping, Sequence
import wave


SCHEMA_VERSION = "trtmc.native-reference-reproduction/v1"


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(row)
    return rows


def _selected_rows(
    rows: Sequence[Mapping[str, Any]],
    sample_id: str,
) -> list[dict[str, Any]]:
    selected = [
        dict(row)
        for row in rows
        if not sample_id or str(row.get("sample_id", "")) == sample_id
    ]
    if sample_id and not selected:
        raise ValueError(f"sample_id {sample_id!r} is not present in the prepared prompts")
    return selected


def _generation(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    value = manifest.get("generation", {})
    return value if isinstance(value, Mapping) else {}


def _scoring(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    value = manifest.get("scoring", {})
    return value if isinstance(value, Mapping) else {}


def _model_dtype(torch_module: Any, name: str) -> str | Any:
    if name == "float16":
        return torch_module.float16
    if name == "bfloat16":
        return torch_module.bfloat16
    return "auto"


def _safe_sample_filename(sample_id: str, suffix: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._")
    return f"{stem or 'sample'}{suffix}"


def _to_device(batch: Any, device: Any) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _read_wav_float32(path: str) -> tuple[Any, int]:
    import numpy as np

    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    dtype_scale = {
        1: (np.uint8, 128.0),
        2: ("<i2", 32768.0),
        4: ("<i4", 2147483648.0),
    }
    if sample_width not in dtype_scale:
        raise RuntimeError(f"Unsupported WAV sample width {sample_width} bytes for {path}")
    dtype, scale = dtype_scale[sample_width]
    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    audio = (audio - 128.0) / scale if sample_width == 1 else audio / scale
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def _resample_audio(audio: Any, source_rate: int, target_rate: int) -> Any:
    if source_rate == target_rate or len(audio) == 0:
        return audio
    import numpy as np

    target_length = max(1, int(len(audio) * target_rate / source_rate))
    source_x = np.arange(len(audio), dtype=np.float32)
    target_x = np.linspace(0, len(audio) - 1, target_length, dtype=np.float32)
    return np.interp(target_x, source_x, audio).astype(np.float32)


def _write_wav_pcm16(path: Path, audio: Any, sample_rate: int) -> None:
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1.0:
        samples = samples / peak
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _wav_metrics(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        frame_count = wav_file.getnframes()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(frame_count)
    rms = 0.0
    if frames and sample_width == 2:
        count = len(frames) // 2
        samples = struct.unpack(f"<{count}h", frames)
        rms = math.sqrt(
            sum((float(value) / 32768.0) ** 2 for value in samples) / count
        )
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_s": frame_count / sample_rate if sample_rate else 0.0,
        "rms": rms,
    }


def _write_predictions(
    arguments: argparse.Namespace,
    responses: Sequence[Mapping[str, Any]],
) -> None:
    arguments.predictions.parent.mkdir(parents=True, exist_ok=True)
    arguments.predictions.write_text(
        json.dumps({"responses": list(responses)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with arguments.raw_output.open("w", encoding="utf-8") as raw_file:
        for row in responses:
            raw_file.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _is_nemo_asr(arguments: argparse.Namespace) -> bool:
    values = (
        arguments.reference_family.lower(),
        arguments.family.lower(),
        arguments.model.lower(),
    )
    return (
        values[0] in {"asr_canary", "asr_nemo"}
        or values[1] in {"canary", "nemotron_speech_streaming"}
        or "canary" in values[2]
        or "nemotron-speech-streaming" in values[2]
    )


def _transcription_text(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    if hasattr(value, "text"):
        return str(value.text)
    if isinstance(value, Mapping):
        return str(value.get("text", ""))
    return str(value)


def _asr_row(
    prompt: Mapping[str, Any],
    output_text: str,
    wall_ms: float,
    token_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    tokens = None if token_ids is None else [int(value) for value in token_ids]
    return {
        "sample_id": str(prompt.get("sample_id", "")),
        "output_text": output_text,
        "generated_tokens": None if tokens is None else len(tokens),
        "generated_token_ids": tokens,
        "wall_ms": wall_ms,
        "source": "hf",
    }


def _audio_for_prompt(
    prompt: Mapping[str, Any],
    target_rate: int,
) -> tuple[Any, Path]:
    audio_path = str(prompt.get("audio", ""))
    if not audio_path:
        raise ValueError(
            f"ASR reference expects an audio path for {prompt.get('sample_id', '')}"
        )
    audio, source_rate = _read_wav_float32(audio_path)
    return _resample_audio(audio, source_rate, target_rate), Path(audio_path)


def _load_whisper_runtime(
    arguments: argparse.Namespace,
) -> tuple[Any, Any, Any, Any]:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, logging

    logging.set_verbosity_error()
    processor = AutoProcessor.from_pretrained(
        arguments.model,
        trust_remote_code=arguments.trust_remote_code,
        local_files_only=arguments.local_files_only,
    )
    model_kwargs = {
        "torch_dtype": _model_dtype(torch, arguments.dtype),
        "trust_remote_code": arguments.trust_remote_code,
        "local_files_only": arguments.local_files_only,
    }
    if arguments.device_map:
        model_kwargs["device_map"] = arguments.device_map
    if arguments.attn_impl:
        model_kwargs["attn_implementation"] = arguments.attn_impl
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        arguments.model,
        **model_kwargs,
    ).eval()
    device = model.device if arguments.device_map else torch.device(arguments.device)
    if not arguments.device_map:
        model.to(device)
    return torch, processor, model, device


def _whisper_inputs(
    processor: Any,
    audio: Any,
    target_rate: int,
    device: Any,
    model_dtype: Any,
) -> dict[str, Any]:
    inputs = processor(audio, sampling_rate=target_rate, return_tensors="pt")
    return {
        key: (
            value.to(device=device, dtype=model_dtype)
            if hasattr(value, "is_floating_point") and value.is_floating_point()
            else value.to(device)
            if hasattr(value, "to")
            else value
        )
        for key, value in inputs.items()
    }


def _seed_torch(torch: Any, seed: int) -> None:
    if seed < 0:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_whisper_asr(
    arguments: argparse.Namespace,
    prompts: Sequence[Mapping[str, Any]],
    generation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    torch, processor, model, device = _load_whisper_runtime(arguments)
    model_dtype = next(model.parameters()).dtype
    target_rate = int(getattr(processor.feature_extractor, "sampling_rate", 16000))
    max_new_tokens = arguments.max_new_tokens or int(
        generation.get("max_new_tokens", 100)
    )
    seed = arguments.seed
    if seed is None:
        seed = int(generation.get("seed", -1))
    responses = []
    for prompt in prompts:
        audio, _source = _audio_for_prompt(prompt, target_rate)
        eval_index = int(prompt.get("eval_index", 0))
        _seed_torch(torch, seed + eval_index if seed >= 0 else -1)
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **_whisper_inputs(
                    processor,
                    audio,
                    target_rate,
                    device,
                    model_dtype,
                ),
                max_new_tokens=max_new_tokens,
            )
        responses.append(
            _asr_row(
                prompt,
                processor.batch_decode(output_ids, skip_special_tokens=True)[0],
                (time.perf_counter() - started) * 1000.0,
                output_ids[0].tolist(),
            )
        )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return responses


def _run_nemotron35_asr(
    arguments: argparse.Namespace,
    prompts: Sequence[Mapping[str, Any]],
    generation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModel, AutoProcessor

    device = torch.device(arguments.device)
    common = {
        "trust_remote_code": arguments.trust_remote_code,
        "local_files_only": arguments.local_files_only,
    }
    processor = AutoProcessor.from_pretrained(arguments.model, **common)
    model = AutoModel.from_pretrained(
        arguments.model,
        torch_dtype=_model_dtype(torch, arguments.dtype),
        **common,
    ).eval().to(device)
    target_rate = int(generation.get("sample_rate", 16000) or 16000)
    max_new_tokens = arguments.max_new_tokens or int(
        generation.get("max_new_tokens", 256)
    )
    output_dir = arguments.predictions.parent / "hf_canary_audio"
    responses = []
    for prompt in prompts:
        audio, _source = _audio_for_prompt(prompt, target_rate)
        wav_path = output_dir / _safe_sample_filename(
            str(prompt.get("sample_id", "")),
            ".wav",
        )
        _write_wav_pcm16(wav_path, audio, target_rate)
        inputs = processor(
            audio,
            sampling_rate=target_rate,
            language=str(prompt.get("language") or generation.get("language", "en-US")),
            return_tensors="pt",
        )
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **_to_device(inputs, device),
                max_new_tokens=max_new_tokens,
            )
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        responses.append(
            _asr_row(
                prompt,
                processor.batch_decode(sequences, skip_special_tokens=True)[0],
                (time.perf_counter() - started) * 1000.0,
                sequences[0].detach().cpu().tolist(),
            )
        )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return responses


def _run_nemo_asr(
    arguments: argparse.Namespace,
    prompts: Sequence[Mapping[str, Any]],
    generation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if "nemotron-3.5-asr-streaming" in arguments.model.lower():
        return _run_nemotron35_asr(arguments, prompts, generation)
    target_rate = int(generation.get("sample_rate", 16000) or 16000)
    output_dir = arguments.predictions.parent / "hf_canary_audio"
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        return _run_pipeline_asr(
            arguments,
            prompts,
            target_rate,
            output_dir,
        )
    model = nemo_asr.models.ASRModel.from_pretrained(
        arguments.model,
        map_location=arguments.device,
    )
    if arguments.device != "cpu" and hasattr(model, "to"):
        model = model.to(arguments.device)
    model.eval()
    responses = []
    for prompt in prompts:
        audio, _source = _audio_for_prompt(prompt, target_rate)
        wav_path = output_dir / _safe_sample_filename(
            str(prompt.get("sample_id", "")),
            ".wav",
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


def _run_pipeline_asr(
    arguments: argparse.Namespace,
    prompts: Sequence[Mapping[str, Any]],
    target_rate: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    import torch
    from transformers import pipeline

    device = (
        0
        if arguments.device.startswith("cuda") and torch.cuda.is_available()
        else -1
    )
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=arguments.model,
        torch_dtype=_model_dtype(torch, arguments.dtype),
        device=device,
        trust_remote_code=arguments.trust_remote_code,
        model_kwargs={"local_files_only": True}
        if arguments.local_files_only
        else {},
    )
    responses = []
    for prompt in prompts:
        audio, _source = _audio_for_prompt(prompt, target_rate)
        wav_path = output_dir / _safe_sample_filename(
            str(prompt.get("sample_id", "")),
            ".wav",
        )
        _write_wav_pcm16(wav_path, audio, target_rate)
        started = time.perf_counter()
        result = transcriber({"raw": audio, "sampling_rate": target_rate})
        responses.append(
            _asr_row(
                prompt,
                _transcription_text(result),
                (time.perf_counter() - started) * 1000.0,
            )
        )
    return responses


def _load_tts_runtime(arguments: argparse.Namespace, torch: Any) -> tuple[Any, Any]:
    if "magpie" in arguments.model.lower():
        from nemo.collections.tts.models import MagpieTTSModel

        model = MagpieTTSModel.from_pretrained(arguments.model)
        return None, model.eval().to(torch.device(arguments.device))
    from transformers import AutoProcessor, BarkModel, logging

    logging.set_verbosity_error()
    processor = AutoProcessor.from_pretrained(
        arguments.model,
        trust_remote_code=arguments.trust_remote_code,
        local_files_only=arguments.local_files_only,
    )
    model = BarkModel.from_pretrained(
        arguments.model,
        trust_remote_code=arguments.trust_remote_code,
        local_files_only=arguments.local_files_only,
        torch_dtype=_model_dtype(torch, arguments.dtype),
    )
    return processor, model.eval().to(torch.device(arguments.device))


def _generate_tts_audio(
    model: Any,
    processor: Any,
    prompt: Mapping[str, Any],
    device: Any,
) -> tuple[Any, int]:
    if processor is None:
        audio_tensor, audio_length = model.do_tts(
            transcript=str(prompt.get("prompt", "")),
            language=str(prompt.get("language", "en") or "en"),
            use_cfg=True,
        )
        audio = audio_tensor.detach().cpu().numpy().reshape(-1)
        length = int(audio_length.item()) if audio_length.numel() else len(audio)
        return audio[:length], 22050
    inputs = _to_device(
        processor(str(prompt.get("prompt", "")), return_tensors="pt"),
        device,
    )
    audio = model.generate(**inputs).detach().cpu().numpy().reshape(-1)
    return audio, int(model.generation_config.sample_rate)


def _transcribe_tts(
    arguments: argparse.Namespace,
    wav_paths: Sequence[Path],
    model_id: str,
) -> list[str]:
    import torch
    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        pipeline,
    )

    device = (
        0
        if arguments.device.startswith("cuda") and torch.cuda.is_available()
        else -1
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        local_files_only=arguments.local_files_only,
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=arguments.local_files_only,
    )
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
    )
    target_rate = int(transcriber.feature_extractor.sampling_rate)
    waveforms = []
    for path in wav_paths:
        audio, sample_rate = _read_wav_float32(str(path))
        waveforms.append(_resample_audio(audio, sample_rate, target_rate))
    outputs = transcriber(waveforms, batch_size=min(8, len(waveforms)))
    if isinstance(outputs, Mapping):
        outputs = [outputs]
    return [_transcription_text(output).strip() for output in outputs]


def _run_tts(
    arguments: argparse.Namespace,
    prompts: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import numpy as np
    import torch

    generation = _generation(manifest)
    seed = arguments.seed
    if seed is None:
        seed = int(generation.get("seed", 42))
    device = torch.device(arguments.device)
    processor, model = _load_tts_runtime(arguments, torch)
    output_dir = arguments.predictions.parent / "hf_audio"
    responses = []
    for prompt in prompts:
        eval_index = int(prompt.get("eval_index", 0))
        sample_seed = seed + eval_index
        random.seed(sample_seed)
        np.random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)
        started = time.perf_counter()
        with torch.inference_mode():
            audio, sample_rate = _generate_tts_audio(
                model,
                processor,
                prompt,
                device,
            )
        wav_path = output_dir / _safe_sample_filename(
            str(prompt.get("sample_id", "")),
            ".wav",
        )
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
    scoring = _scoring(manifest)
    transcripts = _transcribe_tts(
        arguments,
        [Path(row["wav_path"]) for row in responses],
        str(scoring.get("asr_model", "openai/whisper-large-v3-turbo")),
    )
    for row, transcript in zip(responses, transcripts, strict=True):
        row["output_text"] = transcript
        row["asr_transcript"] = transcript
    return responses


def _reproduction_command(arguments: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        arguments.model,
        "--family",
        arguments.family,
        "--reference-family",
        arguments.reference_family,
        "--prompts",
        "{work_dir}/prompts.jsonl",
        "--answers",
        "{work_dir}/answers.json",
        "--manifest",
        "{work_dir}/manifest.json",
        "--predictions",
        "{reference_predictions_json}",
        "--raw-output",
        "{reference_raw_jsonl}",
        "--dtype",
        arguments.dtype,
        "--device",
        arguments.device,
        "--sample-id",
        "{sample_id}",
    ]
    for flag, value in (
        ("--device-map", arguments.device_map),
        ("--attn-impl", arguments.attn_impl),
        ("--max-new-tokens", arguments.max_new_tokens),
        ("--seed", arguments.seed),
    ):
        if value not in (None, ""):
            command.extend([flag, str(value)])
    for enabled, flag in (
        (arguments.trust_remote_code, "--trust-remote-code"),
        (arguments.local_files_only, "--local-files-only"),
    ):
        if enabled:
            command.append(flag)
    return command


def _write_reproduction_metadata(arguments: argparse.Namespace) -> None:
    if arguments.repro_metadata is None:
        return
    arguments.repro_metadata.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "backend": "upstream_speech",
                "entrypoint": str(Path(__file__).resolve()),
                "entrypoint_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "command": _reproduction_command(arguments),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run(arguments: argparse.Namespace) -> None:
    manifest = _load_json(arguments.manifest)
    prompts = _selected_rows(
        _load_jsonl(arguments.prompts),
        arguments.sample_id,
    )
    dataset_kind = str(manifest.get("dataset_kind", ""))
    if dataset_kind == "seedtts_json":
        responses = _run_tts(arguments, prompts, manifest)
    elif dataset_kind == "asr_chat_json":
        generation = _generation(manifest)
        if _is_nemo_asr(arguments):
            responses = _run_nemo_asr(arguments, prompts, generation)
        else:
            responses = _run_whisper_asr(arguments, prompts, generation)
    else:
        raise ValueError(f"unsupported speech dataset kind: {dataset_kind!r}")
    _write_predictions(arguments, responses)
    _write_reproduction_metadata(arguments)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ASR or TTS directly through its upstream runtime."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--family", default="")
    parser.add_argument("--reference-family", default="")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--repro-metadata", type=Path)
    parser.add_argument("--sample-id", default="")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="")
    parser.add_argument("--attn-impl", default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
