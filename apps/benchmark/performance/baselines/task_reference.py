#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark task-aligned non-text model reference operations."""

from __future__ import annotations

import argparse
import atexit
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[4]
for source_root in (
    REPOSITORY,
    REPOSITORY / "core/builder",
    REPOSITORY / "apps/benchmark",
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from apps.benchmark.performance.baselines.timing_contracts import timing_contract  # noqa: E402

SYSTEM_PROMPT_QWEN3_OMNI = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)
QWEN3_OMNI_TEXT_CHAT_TEMPLATE = """{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\n' }}
{%- if message.content is string %}
{{- message.content }}
{%- else %}
{%- for item in message.content %}
{%- if item.type == 'text' %}{{- item.text }}{%- endif %}
{%- endfor %}
{%- endif %}
{{- '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"""
MAGPIE_SPEAKER_ENCODER_REPO = "Edresson/Speaker_Encoder_H_ASP"
MAGPIE_SPEAKER_ENCODER_FILENAME = "pytorch_model.bin"
MAGPIE_SPEAKER_ENCODER_URL = (
    "https://huggingface.co/Edresson/Speaker_Encoder_H_ASP/resolve/main/pytorch_model.bin"
)
ADAPTERS = (
    "hf-diffusers",
    "hf-qwen3-omni",
    "hf-transformers-asr",
    "hf-transformers-embedding",
    "hf-transformers-reranking",
    "hf-transformers-tts",
    "hf-transformers-vision",
    "hf-transformers-vlm",
    "nemo-asr",
    "nemo-tts",
    "pytorch-lerobot-act",
    "pytorch-personaplex",
    "pytorch-timeseries",
    "upstream-elf",
    "upstream-fast-foundation-stereo",
    "upstream-lance",
    "upstream-sana-wm",
)
PYTORCH_ADAPTERS = {
    "nemo-asr",
    "nemo-tts",
    "pytorch-lerobot-act",
    "pytorch-personaplex",
    "pytorch-timeseries",
    "upstream-elf",
    "upstream-fast-foundation-stereo",
    "upstream-lance",
    "upstream-sana-wm",
}


@dataclass(frozen=True)
class Session:
    """One loaded reference model and its repeatable timed operation."""

    invoke: Callable[[], Mapping[str, Any]]
    framework: str
    timing_scope: str = "task-model-call-wall"
    input_preparation_included: bool = False
    asset_loading_included: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, choices=ADAPTERS)
    parser.add_argument("--family", required=True)
    parser.add_argument("--operation", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--adapter-options-json", default="{}")
    parser.add_argument("--timing-contract-json", default="{}")
    parser.add_argument("--precision", required=True, choices=("fp16", "fp32", "bf16"))
    parser.add_argument("--mode", required=True, choices=("hf-eager", "pytorch-eager"))
    parser.add_argument("--padding", default="longest")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _json_object(raw: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value


def _reference_checkout(checkout: str, *, repository: str) -> str:
    path = Path(checkout).resolve()
    if not path.is_dir():
        raise ValueError(f"{repository} checkout does not exist: {path}")
    return str(path)


def _torch_dtype(torch_module: Any, precision: str) -> Any:
    return {
        "fp16": torch_module.float16,
        "fp32": torch_module.float32,
        "bf16": torch_module.bfloat16,
    }[precision]


def _seed_all(torch_module: Any, seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _request_seed(request: Mapping[str, Any], default: Any = 42) -> int:
    value = request.get("seed", default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("reference request seed must be an integer")
    return value


def _bark_generation_options(request: Mapping[str, Any]) -> dict[str, int]:
    max_new_tokens = request.get("max_new_tokens", 0)
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError("Bark request max_new_tokens must be an integer")
    if max_new_tokens <= 0:
        return {}
    return {"semantic_max_new_tokens": max_new_tokens}


def _apply_magpie_generation_options(model: Any, request: Mapping[str, Any]) -> None:
    max_new_tokens = request.get("max_new_tokens", 0)
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError("Magpie request max_new_tokens must be an integer")
    if max_new_tokens > 0:
        model.inference_parameters.max_decoder_steps = max_new_tokens


def _load_kwargs(arguments: argparse.Namespace, torch_module: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "trust_remote_code": arguments.trust_remote_code,
        "local_files_only": arguments.local_files_only,
        "torch_dtype": _torch_dtype(torch_module, arguments.precision),
    }
    if arguments.revision:
        values["revision"] = arguments.revision
    return values


def _processor_kwargs(arguments: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {
        "trust_remote_code": arguments.trust_remote_code,
        "local_files_only": arguments.local_files_only,
    }
    if arguments.revision:
        values["revision"] = arguments.revision
    return values


def _load_sam3_processor(
    transformers: Any,
    model: str,
    processor_kwargs: Mapping[str, Any],
) -> Any:
    return transformers.Sam3Processor.from_pretrained(model, **processor_kwargs)


def _cached_snapshot_path(repo_id: str, requested: str | None, marker_file: str) -> Path | None:
    if Path(repo_id).exists():
        return Path(repo_id).resolve()
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(
            repo_id=repo_id,
            filename=marker_file,
            revision=requested or "main",
        )
    except (ImportError, OSError, ValueError):
        return None
    if isinstance(cached, str) and Path(cached).is_file():
        return Path(cached).parent.resolve()
    return None


def _to_device(value: Any, device: Any, dtype: Any = None) -> Any:
    if isinstance(value, Mapping):
        return {name: _to_device(item, device, dtype) for name, item in value.items()}
    if hasattr(value, "to"):
        if dtype is not None and getattr(value, "is_floating_point", lambda: False)():
            return value.to(device=device, dtype=dtype)
        return value.to(device)
    return value


def _asset_path(arguments: argparse.Namespace, request: Mapping[str, Any], key: str) -> Path:
    raw = str(request.get(key, "") or "")
    if not raw:
        raise ValueError(f"{arguments.family} reference requires request.{key}")
    path = Path(raw)
    if not path.is_absolute():
        path = arguments.manifest.resolve().parent.parent / path
    if not path.is_file():
        raise FileNotFoundError(f"reference input does not exist: {path}")
    return path


def _tensor_summary(value: Any) -> dict[str, Any]:
    shape = [int(dim) for dim in value.shape]
    return {
        "shape": shape,
        "element_count": int(value.numel()),
        "finite": bool(value.isfinite().all().item()),
    }


def _task_value(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    name: str,
    default: Any = None,
) -> Any:
    if name in request:
        return request[name]
    return default


def _load_tts(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    options: Mapping[str, Any],
) -> Session:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    prompt = str(request.get("prompt", ""))
    if arguments.family == "magpie_tts":
        import fsspec
        from huggingface_hub import hf_hub_download
        from nemo.collections.tts.models import MagpieTTSModel

        speaker_revision = str(options.get("speaker_encoder_revision", "") or "")
        if not speaker_revision:
            raise ValueError("Magpie TTS requires adapter_options.speaker_encoder_revision")
        speaker_checkpoint = hf_hub_download(
            repo_id=MAGPIE_SPEAKER_ENCODER_REPO,
            filename=MAGPIE_SPEAKER_ENCODER_FILENAME,
            revision=speaker_revision,
            local_files_only=arguments.local_files_only,
        )
        original_fsspec_open = fsspec.open

        def offline_fsspec_open(path: Any, *args: Any, **kwargs: Any) -> Any:
            normalized = str(path).split("?", 1)[0]
            if normalized == MAGPIE_SPEAKER_ENCODER_URL:
                path = speaker_checkpoint
            return original_fsspec_open(path, *args, **kwargs)

        fsspec.open = offline_fsspec_open
        model_archive = hf_hub_download(
            repo_id=arguments.model,
            filename="magpie_tts_multilingual_357m.nemo",
            revision=arguments.revision,
            local_files_only=arguments.local_files_only,
        )
        model = MagpieTTSModel.restore_from(restore_path=model_archive).eval().to(device)
        _apply_magpie_generation_options(model, request)
        seed = _request_seed(
            request,
            42,
        )

        def invoke() -> Mapping[str, Any]:
            _seed_all(torch, seed)
            with torch.inference_mode():
                audio, length = model.do_tts(transcript=prompt, language="en", use_cfg=True)
            sample_count = int(length.item()) if length.numel() else int(audio.numel())
            return {"audio_samples": sample_count, "sample_rate": 22_050}

    else:
        from transformers import AutoProcessor, BarkModel

        processor = AutoProcessor.from_pretrained(arguments.model, **_processor_kwargs(arguments))
        model = (
            BarkModel.from_pretrained(arguments.model, **_load_kwargs(arguments, torch))
            .eval()
            .to(device)
        )
        inputs = _to_device(processor(prompt, return_tensors="pt"), device)
        seed = _request_seed(
            request,
            42,
        )
        generation_options = _bark_generation_options(request)

        def invoke() -> Mapping[str, Any]:
            _seed_all(torch, seed)
            with torch.inference_mode():
                audio = model.generate(**inputs, **generation_options)
            return {
                "audio_samples": int(audio.numel()),
                "sample_rate": int(model.generation_config.sample_rate),
            }

    if arguments.family == "magpie_tts":
        return Session(
            invoke,
            "nemo",
            timing_scope="task-pipeline-call-wall",
            input_preparation_included=True,
        )
    return Session(invoke, "transformers")


def _load_nemo_asr_reference_model(
    arguments: argparse.Namespace,
    *,
    device: Any,
) -> Any:
    if (
        arguments.family == "nemotron_speech_streaming"
        and "nemotron-3.5-asr-streaming" in arguments.model.lower()
    ):
        from apps.benchmark.performance.baselines.audio_reference import (
            load_nemotron35_asr_model,
        )

        return load_nemotron35_asr_model(
            model=arguments.model,
            revision=arguments.revision or "",
            local_files_only=arguments.local_files_only,
            device=str(device),
        )

    import nemo.collections.asr as nemo_asr

    return nemo_asr.models.ASRModel.from_pretrained(
        arguments.model,
        map_location="cpu",
    )


def _disable_nemo_asr_cuda_graphs(model: Any) -> bool:
    """Disable NeMo's optional RNNT CUDA Graph decoder when available."""
    decoding = getattr(getattr(model, "decoding", None), "decoding", None)
    disable_cuda_graphs = getattr(decoding, "disable_cuda_graphs", None)
    if not callable(disable_cuda_graphs):
        return False
    return bool(disable_cuda_graphs())


def _load_asr(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    _options: Mapping[str, Any],
) -> Session:
    import torch
    from apps.benchmark.performance.baselines.audio_reference import (
        read_wav_float32,
        resample_audio,
        transcription_text,
        write_wav_pcm16,
    )

    audio, sample_rate = read_wav_float32(str(_asset_path(arguments, request, "audio_path")))
    target_rate = 16_000
    audio = resample_audio(audio, sample_rate, target_rate)
    max_new_tokens = int(request.get("max_new_tokens", 100))
    device = torch.device("cuda")

    if arguments.family in {"canary", "nemotron_speech_streaming"}:
        model = _load_nemo_asr_reference_model(arguments, device=device).eval().to(device)
        _disable_nemo_asr_cuda_graphs(model)
        reference_dtype = _torch_dtype(torch, arguments.precision)
        autocast_dtype = reference_dtype if arguments.precision != "fp32" else torch.float16
        temporary = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temporary.close()
        atexit.register(Path(temporary.name).unlink, missing_ok=True)
        write_wav_pcm16(Path(temporary.name), audio, target_rate)

        transcription_input: str | list[str]
        if arguments.family == "nemotron_speech_streaming":
            manifest = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
            manifest.close()
            atexit.register(Path(manifest.name).unlink, missing_ok=True)
            record: dict[str, Any] = {
                "audio_filepath": temporary.name,
                "duration": float(len(audio)) / target_rate,
                "text": "",
            }
            language = str(request.get("language", "") or "")
            if language and language != "auto":
                record["lang"] = language
            Path(manifest.name).write_text(json.dumps(record) + "\n", encoding="utf-8")
            transcription_input = manifest.name
        else:
            transcription_input = [temporary.name]

        def invoke() -> Mapping[str, Any]:
            with torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
                enabled=arguments.precision != "fp32",
            ):
                result = model.transcribe(transcription_input, batch_size=1)
            return {"text": transcription_text(result), "output_tokens": None}

    else:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        processor = AutoProcessor.from_pretrained(arguments.model, **_processor_kwargs(arguments))
        model = (
            AutoModelForSpeechSeq2Seq.from_pretrained(
                arguments.model, **_load_kwargs(arguments, torch)
            )
            .eval()
            .to(device)
        )
        processor_options: dict[str, Any] = {
            "sampling_rate": target_rate,
            "return_tensors": "pt",
        }
        language = str(request.get("language", "") or "")
        if language:
            processor_options["language"] = language
        inputs = _to_device(
            processor(audio, **processor_options), device, next(model.parameters()).dtype
        )

        def invoke() -> Mapping[str, Any]:
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
            sequences = generated.sequences if hasattr(generated, "sequences") else generated
            token_ids = [int(token) for token in sequences[0].detach().cpu().tolist()]
            return {
                "text": processor.batch_decode(sequences, skip_special_tokens=True)[0],
                "token_ids": token_ids,
                "output_tokens": len(token_ids),
            }

    framework = (
        "nemo" if arguments.family in {"canary", "nemotron_speech_streaming"} else "transformers"
    )
    if arguments.family in {"canary", "nemotron_speech_streaming"}:
        return Session(
            invoke,
            framework,
            timing_scope="task-pipeline-call-wall",
            input_preparation_included=True,
            asset_loading_included=True,
        )
    return Session(invoke, framework)


def _load_vlm_model(
    transformers_module: Any, family: str, model_id: str, kwargs: dict[str, Any]
) -> Any:
    class_name = {
        "internvl": "AutoModel",
        "phi4_multimodal": "AutoModelForCausalLM",
        "qwen_vl": "AutoModelForImageTextToText",
    }[family]
    model_class = getattr(transformers_module, class_name)
    return model_class.from_pretrained(model_id, **kwargs)


def _load_deepseek_ocr(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
) -> Session:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if arguments.precision != "bf16":
        raise ValueError("DeepSeek-OCR-2 official reference requires bf16 precision")
    tokenizer = AutoTokenizer.from_pretrained(arguments.model, **_processor_kwargs(arguments))
    load_options = _load_kwargs(arguments, torch)
    load_options.update(
        {
            "attn_implementation": "eager",
            "use_safetensors": True,
        }
    )
    model = AutoModel.from_pretrained(arguments.model, **load_options).eval().cuda()
    image_path = str(_asset_path(arguments, request, "image_path"))
    prompt = str(request.get("prompt", ""))
    if "<image>" not in prompt:
        prompt = f"<image>\n{prompt}"
    output_directory = Path(tempfile.mkdtemp(prefix="trtmc-perf-deepseek-ocr-"))
    atexit.register(shutil.rmtree, output_directory, ignore_errors=True)

    def invoke() -> Mapping[str, Any]:
        text = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=str(output_directory),
            base_size=1024,
            image_size=768,
            crop_mode=True,
            save_results=False,
            eval_mode=True,
        )
        decoded = str(text or "")
        token_ids = tokenizer(decoded, add_special_tokens=False).input_ids
        return {"text": decoded, "token_ids": token_ids, "output_tokens": len(token_ids)}

    return Session(
        invoke,
        "transformers",
        timing_scope="task-pipeline-call-wall",
        input_preparation_included=True,
        asset_loading_included=True,
    )


def _locateanything_config(arguments: argparse.Namespace, transformers_module: Any) -> Any:
    options = _processor_kwargs(arguments)
    return transformers_module.AutoConfig.from_pretrained(arguments.model, **options)


def _locateanything_tokenizer(arguments: argparse.Namespace, _torch_module: Any) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(arguments.model, **_processor_kwargs(arguments))


def _load_locateanything(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
) -> Session:
    import torch
    import transformers
    from apps.benchmark.performance.baselines.reference_support import (
        preprocess_locateanything,
    )

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
    device = torch.device("cuda")
    config = _locateanything_config(arguments, transformers)
    tokenizer = _locateanything_tokenizer(arguments, torch)
    load_options = _load_kwargs(arguments, torch)
    load_options["config"] = config
    model = (
        transformers.AutoModel.from_pretrained(arguments.model, **load_options).eval().to(device)
    )
    image_inputs = preprocess_locateanything(str(_asset_path(arguments, request, "image_path")))
    pixel_values = torch.from_numpy(image_inputs["pixel_values"]).to(device)
    image_grid_hws = torch.from_numpy(image_inputs["image_grid_hws"]).to(
        device=device, dtype=torch.int32
    )
    prompt = str(request.get("prompt", ""))
    prompt_text = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<img>"
        + "<IMG_CONTEXT>" * 256
        + f"</img>{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generate_options: dict[str, Any] = {
        "pixel_values": pixel_values,
        "image_grid_hws": image_grid_hws,
        "input_ids": input_ids,
        "tokenizer": tokenizer,
        "max_new_tokens": int(request.get("max_new_tokens", 32)),
        "use_cache": True,
        "generation_mode": "slow",
        "do_sample": False,
    }
    if inputs.get("attention_mask") is not None:
        generate_options["attention_mask"] = inputs["attention_mask"].to(device)

    def invoke() -> Mapping[str, Any]:
        with torch.inference_mode():
            output = model.generate(**generate_options)
        if isinstance(output, str):
            text = output
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        elif isinstance(output, (list, tuple)) and output and isinstance(output[0], str):
            text = output[0]
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        else:
            sequence = output[0] if output.ndim > 1 else output
            generated = sequence[input_ids.shape[-1] :]
            token_ids = [int(token) for token in generated.detach().cpu().tolist()]
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
        if not text.strip():
            raise RuntimeError("LocateAnything reference produced empty text")
        return {"text": text, "token_ids": token_ids, "output_tokens": len(token_ids)}

    return Session(invoke, "transformers")


def _vl_prompt_has_image_placeholder(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "<|image_pad|>",
            "<|vision_start|>",
            "<|image_1|>",
            "<image>",
            "<IMG_CONTEXT>",
        )
    )


def _load_vlm(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    _options: Mapping[str, Any],
) -> Session:
    if arguments.family == "deepseek_ocr":
        return _load_deepseek_ocr(arguments, request)
    if arguments.family == "locateanything":
        return _load_locateanything(arguments, request)

    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor

    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(arguments.model, **_processor_kwargs(arguments))
    load_options = _load_kwargs(arguments, torch)
    if arguments.family == "phi4_multimodal":
        config = transformers.AutoConfig.from_pretrained(
            arguments.model, **_processor_kwargs(arguments)
        )
        config._attn_implementation = "eager"
        config._attn_implementation_internal = "eager"
        load_options.update({"config": config, "attn_implementation": "eager"})
    model = (
        _load_vlm_model(transformers, arguments.family, arguments.model, load_options)
        .eval()
        .to(device)
    )
    image = Image.open(_asset_path(arguments, request, "image_path")).convert("RGB")
    prompt = str(request.get("prompt", ""))
    max_new_tokens = int(request.get("max_new_tokens", 16))

    if arguments.family == "phi4_multimodal":
        messages = [{"role": "user", "content": f"<|image_1|>{prompt}"}]
        template_owner = getattr(processor, "tokenizer", processor)
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        template_owner = processor
    rendered = template_owner.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if not isinstance(rendered, str) or not _vl_prompt_has_image_placeholder(rendered):
        raise ValueError("vision-language chat template lost the image placeholder")
    inputs = processor(text=rendered, images=image, padding=True, return_tensors="pt")
    inputs = _to_device(inputs, device, next(model.parameters()).dtype)

    def invoke() -> Mapping[str, Any]:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        input_length = int(inputs["input_ids"].shape[-1])
        sequence = output_ids[0]
        generated = sequence[input_length:] if sequence.shape[0] > input_length else sequence
        token_ids = [int(token) for token in generated.detach().cpu().tolist()]
        text = processor.decode(generated, skip_special_tokens=True)
        return {"text": text, "token_ids": token_ids, "output_tokens": len(token_ids)}

    return Session(invoke, "transformers")


def _load_embedding(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    _options: Mapping[str, Any],
) -> Session:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(arguments.model, **_processor_kwargs(arguments))
    model = (
        AutoModel.from_pretrained(arguments.model, **_load_kwargs(arguments, torch))
        .eval()
        .to(device)
    )
    prompt = str(request.get("prompt", ""))
    declared_timing = timing_contract(runner="task-reference", family=arguments.family)

    def prepare_inputs() -> Mapping[str, Any]:
        return _to_device(
            tokenizer(prompt, return_tensors="pt", truncation=True),
            device,
        )

    prepared_inputs = None
    if not declared_timing["input_preparation_included"]:
        prepared_inputs = prepare_inputs()

    def invoke() -> Mapping[str, Any]:
        inputs = prepare_inputs() if prepared_inputs is None else prepared_inputs
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs.hidden_states[-1]
        mask = inputs.get("attention_mask", torch.ones(hidden.shape[:2], device=device))
        mask = mask.unsqueeze(-1).to(hidden.dtype)
        vector = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        vector = torch.nn.functional.normalize(vector, p=2, dim=-1)
        summary = _tensor_summary(vector)
        summary.update(
            {
                "embedding_vectors": 1,
                "embedding_elements": int(vector.numel()),
                "dim": int(vector.shape[-1]),
            }
        )
        return summary

    return Session(
        invoke,
        "transformers",
        timing_scope=str(declared_timing["timing_scope"]),
        input_preparation_included=bool(declared_timing["input_preparation_included"]),
        asset_loading_included=bool(declared_timing["asset_loading_included"]),
    )


def _load_reranking(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    _options: Mapping[str, Any],
) -> Session:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoProcessor

    device = torch.device("cuda")
    documents = [str(document) for document in request.get("documents", [])]
    query = str(request.get("query", ""))
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            arguments.model, **_load_kwargs(arguments, torch)
        )
        .eval()
        .to(device)
    )
    processor = AutoProcessor.from_pretrained(
        arguments.model,
        max_input_tiles=6,
        use_thumbnail=True,
        rerank_max_length=8192,
        **_processor_kwargs(arguments),
    )
    examples = [
        {"question": query, "doc_text": document, "doc_image": ""} for document in documents
    ]
    inputs = processor.process_queries_documents_crossencoder(examples)
    inputs = _to_device(inputs, device)

    def invoke() -> Mapping[str, Any]:
        with torch.inference_mode():
            logits = model(**inputs).logits.detach().float().cpu()
        if logits.ndim == 2:
            logits = logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1]
        scores = [float(score) for score in logits.reshape(-1).tolist()]
        return {"scores": scores, "document_count": len(documents)}

    return Session(invoke, "transformers")


_PIXART_TRTMC_MIXED_PRECISION = "pixart_fp16_dit_fp32_t5"


def _diffusion_component_precision_contract(
    arguments: argparse.Namespace,
    options: Mapping[str, Any],
) -> str:
    contract = str(options.get("component_precision_contract", "") or "")
    if not contract:
        return ""
    if contract != _PIXART_TRTMC_MIXED_PRECISION:
        raise ValueError(f"unsupported diffusion component precision contract: {contract}")
    if arguments.family != "pixart" or arguments.precision != "fp16":
        raise ValueError(f"{contract} requires family=pixart and precision=fp16")
    return contract


def _cast_floating_tensors(value: Any, dtype: Any, torch_module: Any) -> Any:
    if isinstance(value, torch_module.Tensor):
        return value.to(dtype=dtype) if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(_cast_floating_tensors(item, dtype, torch_module) for item in value)
    if isinstance(value, list):
        return [_cast_floating_tensors(item, dtype, torch_module) for item in value]
    if isinstance(value, dict):
        return {
            key: _cast_floating_tensors(item, dtype, torch_module) for key, item in value.items()
        }
    return value


def _configure_diffusion_component_precision(
    pipeline: Any,
    arguments: argparse.Namespace,
    options: Mapping[str, Any],
    torch_module: Any,
) -> None:
    contract = _diffusion_component_precision_contract(arguments, options)
    if not contract:
        return

    def fp16_transformer_inputs(_module: Any, args: Any, kwargs: Any) -> Any:
        return (
            _cast_floating_tensors(args, torch_module.float16, torch_module),
            _cast_floating_tensors(kwargs, torch_module.float16, torch_module),
        )

    def fp32_transformer_output(_module: Any, _args: Any, output: Any) -> Any:
        return _cast_floating_tensors(output, torch_module.float32, torch_module)

    pipeline.transformer.register_forward_pre_hook(fp16_transformer_inputs, with_kwargs=True)
    pipeline.transformer.register_forward_hook(fp32_transformer_output)


def _diffusion_pipeline(
    arguments: argparse.Namespace,
    torch_module: Any,
    options: Mapping[str, Any],
) -> Any:
    import diffusers

    model_id = str(options.get("model_id", arguments.model))
    class_name = {
        "flux": "Flux2Pipeline" if "FLUX.2" in model_id.upper() else "FluxPipeline",
        "ltx_video": "LTXPipeline",
        "pixart": "PixArtSigmaPipeline",
        "qwen_image": "QwenImagePipeline",
        "sana_wm": "SanaVideoPipeline",
        "wan_t2v": "WanPipeline",
        "wan2_2_ti2v": "WanPipeline",
        "z_image": "ZImagePipeline",
    }[arguments.family]
    pipeline_class = getattr(diffusers, class_name)
    requested_revision = (
        str(options.get("model_revision", getattr(arguments, "revision", None) or "")) or None
    )
    model_source: str | Path = model_id
    if arguments.local_files_only:
        cached = _cached_snapshot_path(model_id, requested_revision, "model_index.json")
        if cached is None:
            raise FileNotFoundError(f"cached Diffusers snapshot is missing: {model_id}")
        model_source = cached

    load_options: dict[str, Any] = {}
    component_contract = _diffusion_component_precision_contract(arguments, options)
    if component_contract == _PIXART_TRTMC_MIXED_PRECISION:
        from transformers import T5EncoderModel

        load_options["text_encoder"] = T5EncoderModel.from_pretrained(
            model_source,
            subfolder="text_encoder",
            torch_dtype=torch_module.float32,
            **(
                {"revision": requested_revision}
                if requested_revision and model_source == model_id
                else {}
            ),
            local_files_only=arguments.local_files_only,
        )
    if arguments.family == "wan2_2_ti2v":
        load_options["vae"] = diffusers.AutoencoderKLWan.from_pretrained(
            model_source,
            subfolder="vae",
            torch_dtype=torch_module.float32,
            **(
                {"revision": requested_revision}
                if requested_revision and model_source == model_id
                else {}
            ),
            local_files_only=arguments.local_files_only,
        )
    return pipeline_class.from_pretrained(
        model_source,
        torch_dtype=_torch_dtype(torch_module, arguments.precision),
        **load_options,
        **(
            {"revision": requested_revision}
            if requested_revision and model_source == model_id
            else {}
        ),
        trust_remote_code=bool(options.get("trust_remote_code", arguments.trust_remote_code)),
        local_files_only=arguments.local_files_only,
    )


def _load_diffusers(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    options: Mapping[str, Any],
) -> Session:
    import inspect
    import torch
    from PIL import Image

    pipeline = _diffusion_pipeline(arguments, torch, options)
    _configure_diffusion_component_precision(pipeline, arguments, options, torch)
    if bool(options.get("cpu_offload", False)) and hasattr(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to("cuda")
    signature = inspect.signature(pipeline.__call__)
    accepted = set(signature.parameters)
    accepts_extra = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    batch_size = int(request.get("batch_size", 1))
    prompt = str(request.get("prompt", ""))
    raw_prompts = request.get("prompts")
    if raw_prompts is not None:
        if (
            not isinstance(raw_prompts, list)
            or len(raw_prompts) != batch_size
            or any(not isinstance(value, str) for value in raw_prompts)
        ):
            raise ValueError("prompts must contain one string per batch item")
        prompt_value: str | list[str] = list(raw_prompts)
    elif batch_size > 1:
        prompt_value = [prompt] * batch_size
    else:
        prompt_value = prompt
    values: dict[str, Any] = {"prompt": prompt_value}
    negative_prompt = str(request.get("negative_prompt", ""))
    if negative_prompt:
        values["negative_prompt"] = negative_prompt
    steps = int(request.get("num_steps", -1))
    if steps > 0:
        values.update({"num_inference_steps": steps, "step": steps})
    height = int(request.get("height", request.get("video_height", 0)))
    width = int(request.get("width", request.get("video_width", 0)))
    if height > 0:
        values["height"] = height
    if width > 0:
        values["width"] = width
    num_frames = int(
        request.get(
            "video_num_frames",
            _task_value(arguments, request, "num_frames", 1),
        )
    )
    if num_frames > 0:
        values["num_frames"] = num_frames
    for name in ("action", "intrinsics"):
        value = str(_task_value(arguments, request, name, "") or "")
        if value:
            values[name] = value
    fps = int(_task_value(arguments, request, "fps", 0))
    if fps > 0:
        values["fps"] = fps
    for name in ("flow_shift", "translation_speed", "rotation_speed_deg"):
        value = _task_value(arguments, request, name)
        if value is not None:
            values[name] = float(value)
    cfg_scale = float(request.get("cfg_scale", -1.0))
    if cfg_scale >= 0:
        values["cfg_scale"] = cfg_scale
    if bool(request.get("no_refiner", False)):
        values["no_refiner"] = True
    guidance = float(request.get("guidance_scale", -1.0))
    if guidance >= 0:
        values["guidance_scale"] = guidance
    if arguments.family == "qwen_image" and cfg_scale >= 0:
        values["true_cfg_scale"] = cfg_scale
    values["output_type"] = "np"
    image_path = str(request.get("image_path", "") or "")
    if image_path and ("image" in accepted or accepts_extra):
        values["image"] = Image.open(_asset_path(arguments, request, "image_path")).convert("RGB")
    call_values = {
        name: value for name, value in values.items() if name in accepted or accepts_extra
    }
    required = [str(name) for name in options.get("required_call_arguments", [])]
    missing = [name for name in required if name not in call_values]
    if missing:
        raise ValueError(
            f"{arguments.family} reference is missing required call arguments: "
            + ", ".join(missing)
        )
    seed = int(request.get("seed", 42))
    raw_seeds = request.get("seeds")
    if raw_seeds is not None:
        if not isinstance(raw_seeds, list) or len(raw_seeds) != batch_size:
            raise ValueError("seeds must contain one integer per batch item")
        seeds: int | list[int] = [int(value) for value in raw_seeds]
    elif batch_size > 1:
        seeds = [seed] * batch_size
    else:
        seeds = seed
    if "generator" in accepted or accepts_extra:
        if isinstance(seeds, list):
            call_values["generator"] = [
                torch.Generator("cuda").manual_seed(value) for value in seeds
            ]
        else:
            call_values["generator"] = torch.Generator("cuda").manual_seed(seeds)

    def invoke() -> Mapping[str, Any]:
        if "generator" in call_values:
            generators = call_values["generator"]
            if isinstance(generators, list):
                for generator, value in zip(generators, seeds, strict=True):
                    generator.manual_seed(value)
            else:
                generators.manual_seed(seeds)
        result = pipeline(**call_values)
        media = getattr(result, "images", None)
        if media is None:
            media = getattr(result, "frames", None)
        media_type = str(request.get("media_type", "image"))
        return _media_summary(media, media_type)

    return Session(
        invoke,
        "diffusers",
        timing_scope="task-pipeline-call-wall",
        input_preparation_included=True,
    )


def _media_count(media: Any, media_type: str) -> int:
    """Count image batches or video frames without coercing arrays to bool."""
    if media is None:
        return 0
    try:
        outer_count = len(media)
    except TypeError:
        return 1
    if outer_count == 0 or media_type != "video":
        return outer_count
    try:
        return len(media[0])
    except TypeError:
        return outer_count


def _media_summary(media: Any, media_type: str) -> dict[str, Any]:
    """Describe materialized image/video geometry without copying its pixels."""
    count = _media_count(media, media_type)
    item = media
    try:
        item = item[0]
        if media_type == "video":
            item = item[0]
    except (IndexError, KeyError, TypeError):
        pass
    width = height = channels = None
    size = getattr(item, "size", None)
    bands = getattr(item, "getbands", None)
    if isinstance(size, tuple) and len(size) == 2 and callable(bands):
        width, height = (int(value) for value in size)
        channels = len(bands())
    else:
        shape = tuple(int(value) for value in getattr(item, "shape", ()))
        if len(shape) >= 3:
            if shape[-1] in {1, 3, 4}:
                height, width, channels = shape[-3:]
            elif shape[-3] in {1, 3, 4}:
                channels, height, width = shape[-3:]
    summary = {
        "media_type": media_type,
        "media_count": count,
        "height": height,
        "width": width,
        "channels": channels,
    }
    try:
        import numpy as np

        array = np.asarray(media)
        if np.issubdtype(array.dtype, np.number):
            finite = bool(np.isfinite(array).all())
            if not finite:
                raise RuntimeError("reference returned non-finite media values")
            summary["finite"] = True
    except (TypeError, ValueError):
        pass
    return summary


def _numeric_values(request: Mapping[str, Any], key: str) -> list[float]:
    raw = request.get(key)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"time-series reference requires non-empty request.{key}")
    return [float(value) for value in raw]


def _align(values: Sequence[float], length: int, fill: float) -> list[float]:
    result = [fill] * length
    count = min(len(values), length)
    result[-count:] = values[-count:]
    return result


def _patchtst_task(config: Any) -> str:
    for attribute in ("patchtst_task", "task_type", "problem_type"):
        value = str(getattr(config, attribute, "") or "").lower()
        if "class" in value:
            return "classification"
        if "regress" in value:
            return "regression"
        if "forecast" in value or "predict" in value:
            return "forecast"

    architectures = getattr(config, "architectures", []) or []
    if isinstance(architectures, str):
        architectures = [architectures]
    for architecture in architectures:
        value = str(architecture).lower()
        if "class" in value:
            return "classification"
        if "regress" in value:
            return "regression"
        if "forecast" in value or "predict" in value:
            return "forecast"
    return "regression"


def _load_timeseries(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    _options: Mapping[str, Any],
) -> Session:
    import torch
    import transformers

    device = torch.device("cuda")
    dtype = _torch_dtype(torch, arguments.precision)
    if arguments.family == "chronos_bolt":
        from chronos import ChronosBoltPipeline

        chronos_options = _processor_kwargs(arguments)
        chronos_options.update({"device_map": str(device), "dtype": dtype})
        model = ChronosBoltPipeline.from_pretrained(arguments.model, **chronos_options)
        context = torch.tensor(_numeric_values(request, "past_values"), dtype=dtype, device=device)

        def invoke() -> Mapping[str, Any]:
            with torch.inference_mode():
                value = model.predict(
                    context,
                    prediction_length=model.model_prediction_length,
                    limit_prediction_length=True,
                )
            return _tensor_summary(value)

        return Session(invoke, "chronos")

    config = transformers.AutoConfig.from_pretrained(
        arguments.model, **_processor_kwargs(arguments)
    )
    if arguments.family == "timesfm":
        model = (
            transformers.TimesFmModelForPrediction.from_pretrained(
                arguments.model, **_load_kwargs(arguments, torch)
            )
            .eval()
            .to(device)
        )
        length = int(model.config.context_length)
        raw = _numeric_values(request, "past_values")
        series = torch.tensor(_align(raw, length, 0.0), dtype=dtype, device=device).reshape(
            1, length
        )
        padding = [1] * length
        padding[-min(len(raw), length) :] = [0] * min(len(raw), length)
        padding_tensor = torch.tensor(padding, dtype=torch.int32, device=device).reshape(1, length)
        frequency = int(request.get("frequency", 0))
        frequency_tensor = torch.tensor([[frequency]], dtype=torch.long, device=device)

        def invoke() -> Mapping[str, Any]:
            with torch.inference_mode():
                decoder = model.decoder(
                    past_values=series,
                    past_values_padding=padding_tensor,
                    freq=frequency_tensor,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                output = model._postprocess_output(
                    decoder.last_hidden_state, (decoder.loc, decoder.scale)
                )[:, -1, : model.config.horizon_length, 0]
            return _tensor_summary(output)

    else:
        is_mixer = arguments.family == "patchtsmixer"
        if is_mixer:
            model_class = transformers.PatchTSMixerForPrediction
            output_name = "prediction_outputs"
        else:
            task = _patchtst_task(config)
            class_name, output_name = {
                "classification": (
                    "PatchTSTForClassification",
                    "prediction_logits",
                ),
                "forecast": ("PatchTSTForPrediction", "prediction_outputs"),
                "regression": ("PatchTSTForRegression", "regression_outputs"),
            }[task]
            model_class = getattr(transformers, class_name)
        if not hasattr(model_class, "all_tied_weights_keys"):
            model_class.all_tied_weights_keys = {}
        model = (
            model_class.from_pretrained(arguments.model, **_load_kwargs(arguments, torch))
            .eval()
            .to(device)
        )
        length = int(config.context_length)
        channels = int(config.num_input_channels)
        raw = _numeric_values(request, "past_values")
        values = torch.tensor(
            _align(raw, length * channels, 0.0), dtype=dtype, device=device
        ).reshape(1, length, channels)
        raw_mask = request.get("observed_mask")
        if is_mixer and raw_mask is not None:
            mask_values = _numeric_values(request, "observed_mask")
            aligned_mask = _align(mask_values, length * channels, 1.0)
        elif is_mixer:
            aligned_mask = [1.0] * (length * channels)
        else:
            aligned_mask = _align([1.0] * len(raw), length * channels, 0.0)
        observed = torch.tensor(
            aligned_mask,
            dtype=dtype,
            device=device,
        ).reshape(1, length, channels)

        def invoke() -> Mapping[str, Any]:
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type="cuda",
                    dtype=dtype,
                    enabled=dtype != torch.float32,
                ),
            ):
                if is_mixer:
                    outputs = model(
                        past_values=values * observed,
                        observed_mask=observed,
                        return_loss=False,
                        return_dict=True,
                    )
                    output = outputs.prediction_outputs
                else:
                    outputs = model(
                        past_values=values,
                        past_observed_mask=observed.gt(0.5),
                        return_dict=True,
                    )
                    output = getattr(outputs, output_name)
            if isinstance(output, (tuple, list)):
                output = torch.stack(list(output), dim=-1)
            return _tensor_summary(output)

    return Session(invoke, "transformers")


def _load_vision(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    _options: Mapping[str, Any],
) -> Session:
    import torch
    from PIL import Image
    import transformers

    device = torch.device("cuda")
    image = Image.open(_asset_path(arguments, request, "image_path")).convert("RGB")
    width, height = image.size
    kwargs = _load_kwargs(arguments, torch)
    processor_kwargs = _processor_kwargs(arguments)

    if arguments.family in {
        "timm_densenet",
        "timm_efficientnet",
        "timm_inception",
        "timm_mnasnet",
        "timm_mobilenetv3",
        "timm_repvgg",
        "timm_resnet",
        "timm_vgg",
        "timm_vit",
    }:
        import timm
        from timm.data import create_transform, resolve_model_data_config

        model = timm.create_model(f"hf-hub:{arguments.model}", pretrained=True)
        dtype = _torch_dtype(torch, arguments.precision)
        model = model.eval().to(device=device, dtype=dtype)
        transform = create_transform(**resolve_model_data_config(model), is_training=False)
        inputs = transform(image).unsqueeze(0).to(device=device, dtype=dtype)

        def invoke() -> Mapping[str, Any]:
            with torch.inference_mode():
                logits = model(inputs)
            return {"top_class": int(logits.argmax(dim=-1)[0]), **_tensor_summary(logits)}

    elif arguments.family == "dinov3":
        processor = transformers.AutoImageProcessor.from_pretrained(
            arguments.model, **processor_kwargs
        )
        model = transformers.AutoModel.from_pretrained(arguments.model, **kwargs).eval().to(device)
        inputs = _to_device(
            processor(images=image, return_tensors="pt"),
            device,
            next(model.parameters()).dtype,
        )

        def invoke() -> Mapping[str, Any]:
            with torch.inference_mode():
                outputs = model(**inputs)
            return {
                "last_hidden_state_shape": _tensor_summary(outputs.last_hidden_state)["shape"],
                "pooler_output_shape": _tensor_summary(outputs.pooler_output)["shape"],
            }

    elif arguments.family == "segformer":
        processor = transformers.AutoImageProcessor.from_pretrained(
            arguments.model, **processor_kwargs
        )
        model = (
            transformers.AutoModelForSemanticSegmentation.from_pretrained(arguments.model, **kwargs)
            .eval()
            .to(device)
        )
        inputs = _to_device(
            processor(images=image, return_tensors="pt"),
            device,
            next(model.parameters()).dtype,
        )

        def invoke() -> Mapping[str, Any]:
            with torch.inference_mode():
                logits = model(**inputs).logits
            masks = logits.argmax(dim=1)
            return {"mask_count": int(masks.shape[0]), **_tensor_summary(masks)}

    elif arguments.family == "sam3":
        processor = _load_sam3_processor(transformers, arguments.model, processor_kwargs)
        model = transformers.Sam3Model.from_pretrained(arguments.model, **kwargs).eval().to(device)
        inputs = _to_device(
            processor(
                images=image,
                text=str(request.get("prompt", "")),
                return_tensors="pt",
            ),
            device,
            next(model.parameters()).dtype,
        )

        def invoke() -> Mapping[str, Any]:
            with torch.inference_mode():
                outputs = model(**inputs)
            original_sizes = inputs.get("original_sizes")
            target_sizes = (
                original_sizes.cpu().tolist() if hasattr(original_sizes, "cpu") else original_sizes
            )
            result = processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=target_sizes,
            )[0]
            masks = result.get("masks")
            num_masks = 0 if masks is None else int(masks.shape[0] if masks.ndim > 2 else 1)
            return {
                "num_masks": num_masks,
                "height": height,
                "width": width,
            }

    else:
        processor = transformers.SamProcessor.from_pretrained(arguments.model, **processor_kwargs)
        model = transformers.SamModel.from_pretrained(arguments.model, **kwargs).eval().to(device)
        points = [
            [
                [
                    int(float(request.get("point_x", 0.5)) * width),
                    int(float(request.get("point_y", 0.5)) * height),
                ]
            ]
        ]
        inputs = _to_device(
            processor(image, input_points=points, return_tensors="pt"),
            device,
            next(model.parameters()).dtype,
        )

        def invoke() -> Mapping[str, Any]:
            with torch.inference_mode():
                outputs = model(**inputs)
            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0]
            height, width = (int(value) for value in masks.shape[-2:])
            num_masks = int(masks.numel() // (height * width))
            return {
                "num_masks": num_masks,
                "height": height,
                "width": width,
            }

    return Session(invoke, "transformers")


def _qwen3_omni_chat_inputs(processor: Any, conversation: Sequence[Mapping[str, Any]]) -> Any:
    processor_template = getattr(processor, "chat_template", None)
    tokenizer_template = getattr(getattr(processor, "tokenizer", None), "chat_template", None)
    template = (
        processor_template
        if isinstance(processor_template, str) and processor_template.strip()
        else tokenizer_template
        if isinstance(tokenizer_template, str) and tokenizer_template.strip()
        else QWEN3_OMNI_TEXT_CHAT_TEMPLATE
    )
    return processor.apply_chat_template(
        conversation,
        chat_template=template,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )


def _load_qwen3_omni(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    options: Mapping[str, Any],
) -> Session:
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    processor = Qwen3OmniMoeProcessor.from_pretrained(
        arguments.model, **_processor_kwargs(arguments)
    )
    load_options = _load_kwargs(arguments, torch)
    load_options.update(
        {
            "device_map": str(options.get("device_map", "cuda:0")),
            "enable_audio_output": True,
        }
    )
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        arguments.model, **load_options
    ).eval()
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_QWEN3_OMNI}]},
        {"role": "user", "content": [{"type": "text", "text": str(request.get("prompt", ""))}]},
    ]
    inputs = _qwen3_omni_chat_inputs(processor, conversation).to(model.device)
    thinker_tokens = int(request.get("max_new_tokens", 16))
    talker_tokens = int(request.get("talker_max_new_tokens", 32))
    seed = _request_seed(request)

    def invoke() -> Mapping[str, Any]:
        _seed_all(torch, seed)
        with torch.inference_mode():
            text_ids, audio = model.generate(
                **inputs,
                thinker_max_new_tokens=thinker_tokens,
                talker_max_new_tokens=talker_tokens,
                thinker_do_sample=False,
                talker_do_sample=False,
                speaker=str(options.get("speaker", "Ethan")),
            )
        return {
            "text": processor.batch_decode(text_ids, skip_special_tokens=True)[0],
            "audio_samples": int(audio.numel()),
            "sample_rate": 24_000,
        }

    return Session(invoke, "transformers")


def _load_personaplex(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    options: Mapping[str, Any],
) -> Session:
    official_repo = str(options.get("official_repo", ""))
    if not official_repo:
        raise ValueError("pytorch-personaplex requires baseline.adapter_options.official_repo")
    official_repo = _reference_checkout(
        official_repo, repository="https://github.com/NVIDIA/personaplex"
    )
    import importlib.util

    if importlib.util.find_spec("sphn") is None:
        raise RuntimeError("PersonaPlex reference requires the official sphn package")
    sys.path[:0] = [str(Path(official_repo) / "moshi"), official_repo]
    import torch
    from huggingface_hub import hf_hub_download
    from moshi.models import LMGen, loaders
    from moshi.models.lm import _iterate_audio, encode_from_sphn, load_audio
    from moshi.offline import warmup

    device = "cuda"
    mimi_weights = hf_hub_download(
        arguments.model,
        loaders.MIMI_NAME,
        revision=arguments.revision,
        local_files_only=arguments.local_files_only,
    )
    model_weights = hf_hub_download(
        arguments.model,
        loaders.MOSHI_NAME,
        revision=arguments.revision,
        local_files_only=arguments.local_files_only,
    )
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
    source_audio = load_audio(str(_asset_path(arguments, request, "audio_path")), mimi.sample_rate)
    max_frames = int(request.get("max_new_tokens", options.get("max_frames", 100)))

    def invoke() -> Mapping[str, Any]:
        mimi.reset_streaming()
        other_mimi.reset_streaming()
        generator.reset_streaming()
        generated_frames = 0
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
                    mimi.decode(tokens[:, 1:9])
                    other_mimi.decode(tokens[:, 1:9])
                    generated_frames += 1
                    if generated_frames >= max_frames:
                        return {
                            "audio_frames": generated_frames,
                            "audio_samples": generated_frames * frame_size,
                            "sample_rate": mimi.sample_rate,
                        }
        return {
            "audio_frames": generated_frames,
            "audio_samples": generated_frames * frame_size,
            "sample_rate": mimi.sample_rate,
        }

    return Session(
        invoke,
        "moshi",
        timing_scope="task-pipeline-call-wall",
        input_preparation_included=True,
    )


def _load_fast_foundation_stereo(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    options: Mapping[str, Any],
) -> Session:
    import numpy as np
    import torch
    from PIL import Image

    from apps.benchmark.performance.baselines.reference_support import (
        configure_official_model_args,
        fast_foundation_stereo_model_dir,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Fast Foundation Stereo reference")
    max_disp = int(request.get("max_disp", 192))
    valid_iters = int(request.get("valid_iters", 8))
    height = int(request.get("height", 700))
    width = int(request.get("width", 700))
    if (height, width, max_disp, valid_iters) != (700, 700, 192, 8):
        raise ValueError(
            "Fast Foundation Stereo performance requires 700x700, max_disp=192, and valid_iters=8"
        )
    model_dir = options.get("model_dir")
    if not isinstance(model_dir, str) or not model_dir:
        raise ValueError("Fast Foundation Stereo reference requires adapter_options.model_dir")
    model_root = fast_foundation_stereo_model_dir(model_dir)
    previous_cwd = Path.cwd()
    try:
        os.chdir(model_root)
        sys.path.insert(0, str(model_root))
        from core.utils.utils import InputPadder
        from Utils import AMP_DTYPE

        checkpoint = model_root / "weights/23-36-37/model_best_bp2_serialize.pth"
        model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    finally:
        os.chdir(previous_cwd)
    configure_official_model_args(
        model,
        max_disparity=max_disp,
        valid_iters=valid_iters,
    )
    model = model.cuda().eval()

    def load_image(key: str) -> np.ndarray:
        with Image.open(_asset_path(arguments, request, key)) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if pixels.shape != (height, width, 3):
            raise ValueError(f"{key} must be 700x700 RGB, got {pixels.shape}")
        return pixels

    left = load_image("left_image_path")
    right = load_image("right_image_path")

    def invoke() -> Mapping[str, Any]:
        left_tensor = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
        right_tensor = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(left_tensor.shape, divis_by=32, force_square=False)
        left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)
        with (
            torch.inference_mode(),
            torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE),
        ):
            disparity = model.forward(
                left_tensor,
                right_tensor,
                iters=valid_iters,
                test_mode=True,
                optimize_build_volume="pytorch1",
            )
        output = padder.unpad(disparity.float()).cpu().numpy().reshape(height, width)
        output = np.clip(output, 0, None).astype(np.float32, copy=False)
        return {
            "height": height,
            "width": width,
            "element_count": int(output.size),
            "finite_fraction": float(np.isfinite(output).mean()),
            "nonnegative_fraction": float((output >= 0).mean()),
            "_disparity_f32": output,
        }

    return Session(
        invoke,
        f"torch-{torch.__version__}",
        timing_scope="task-pipeline-call-wall",
        input_preparation_included=True,
        asset_loading_included=False,
    )


LOADERS: dict[
    str, Callable[[argparse.Namespace, Mapping[str, Any], Mapping[str, Any]], Session]
] = {
    "hf-diffusers": _load_diffusers,
    "hf-qwen3-omni": _load_qwen3_omni,
    "hf-transformers-asr": _load_asr,
    "hf-transformers-embedding": _load_embedding,
    "hf-transformers-reranking": _load_reranking,
    "hf-transformers-tts": _load_tts,
    "hf-transformers-vision": _load_vision,
    "hf-transformers-vlm": _load_vlm,
    "nemo-asr": _load_asr,
    "nemo-tts": _load_tts,
    "pytorch-personaplex": _load_personaplex,
    "pytorch-timeseries": _load_timeseries,
    "upstream-fast-foundation-stereo": _load_fast_foundation_stereo,
}


def _synchronize() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        return


def _measure(session: Session, warmup: int, iterations: int) -> tuple[list[float], dict[str, Any]]:
    output: Mapping[str, Any] = {}
    for _ in range(warmup):
        output = session.invoke()
        _synchronize()
    samples = []
    for _ in range(iterations):
        _synchronize()
        started = time.perf_counter()
        output = session.invoke()
        _synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples, dict(output)


def _run_elf(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    options: Mapping[str, Any],
) -> tuple[list[float], dict[str, Any], str, str, bool]:
    reference_repo = str(options.get("reference_repo", ""))
    if not reference_repo:
        raise ValueError("upstream-elf requires baseline.adapter_options.reference_repo")
    reference_repo = _reference_checkout(
        reference_repo, repository="https://github.com/lillian039/ELF"
    )
    count = arguments.warmup + arguments.iterations
    prompt = str(request.get("prompt", ""))

    def repository_input(name: str) -> str:
        value = str(request.get(name, "") or "")
        if not value:
            return ""
        path = Path(value)
        model_root = arguments.manifest.resolve().parent.parent
        return str(path if path.is_absolute() else (model_root / path).resolve())

    with tempfile.TemporaryDirectory(prefix="trtmc-perf-elf-") as temporary:
        root = Path(temporary)
        dataset = root / "dataset.jsonl"
        rows = [
            {"id": f"perf_{index:04d}", "input": prompt, "output": ""} for index in range(count)
        ]
        dataset.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        output = root / "predictions.json"
        command = [
            sys.executable,
            str(REPOSITORY / "apps/benchmark/performance/baselines/elf_reference.py"),
            "--reference-repo",
            reference_repo,
            "--config",
            str(options["config"]),
            "--checkpoint",
            str(options["checkpoint"]),
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--shared-inputs-dir",
            str(root / "shared"),
            "--generation-mode",
            str(request.get("generation_mode", "conditional")),
            "--sampling-method",
            str(options.get("sampling_method", "ode")),
            "--num-steps",
            str(request.get("num_steps", 64)),
            "--cfg-scale",
            str(request.get("cfg_scale", 2.0)),
            "--self-cond-cfg-scale",
            str(options.get("self_cond_cfg_scale", 1.0)),
            "--sde-gamma",
            str(options.get("sde_gamma", 0.0)),
            "--seed",
            str(options.get("seed", 42)),
            "--precision",
            arguments.precision,
        ]
        replay_inputs = (
            ("initial_latents_path", "--initial-latents"),
            ("sampling_steps_path", "--sampling-steps"),
            ("condition_latents_path", "--condition-latents"),
            ("condition_mask_path", "--condition-mask"),
            ("sde_noises_path", "--sde-noises"),
        )
        for request_name, flag in replay_inputs:
            value = repository_input(request_name)
            if value:
                command.extend([flag, value])
        if arguments.local_files_only:
            command.append("--local-files-only")
        completed = subprocess.run(
            command,
            cwd=reference_repo,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"ELF reference failed with rc={completed.returncode}: {completed.stderr[-2000:]}"
            )
        payload = json.loads(output.read_text(encoding="utf-8"))
    responses = payload.get("responses", [])
    if len(responses) != count:
        raise RuntimeError(f"ELF reference returned {len(responses)} responses; expected {count}")
    samples = [float(row["wall_ms"]) for row in responses[arguments.warmup :]]
    summary_row = responses[arguments.warmup]
    summary = {
        "text": str(summary_row.get("output_text", "")),
        "token_ids": summary_row.get("generated_token_ids", []),
        "output_tokens": len(summary_row.get("generated_token_ids", [])),
    }
    return (
        samples,
        summary,
        "elf-pytorch",
        "task-model-call-wall",
        False,
    )


def _run_lance(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    options: Mapping[str, Any],
) -> tuple[list[float], dict[str, Any], str, str, bool]:
    reference_repo = str(options.get("reference_repo", ""))
    if not reference_repo:
        raise ValueError("upstream-lance requires baseline.adapter_options.reference_repo")
    if arguments.precision != "bf16":
        raise ValueError("upstream-lance requires bf16 precision")
    with tempfile.TemporaryDirectory(prefix="trtmc-perf-lance-adapter-") as temporary:
        output = Path(temporary) / "result.json"
        command = [
            sys.executable,
            str(REPOSITORY / "apps/benchmark/performance/baselines/lance_reference.py"),
            "--reference-repo",
            reference_repo,
            "--model",
            arguments.model,
            "--model-subdir",
            str(options.get("model_subdir", "Lance_3B")),
            "--vit-subdir",
            str(options.get("vit_subdir", "Qwen2.5-VL-ViT")),
            "--image",
            str(_asset_path(arguments, request, "image_path")),
            "--prompt",
            str(request.get("prompt", "")),
            "--instruction",
            str(
                options.get(
                    "instruction",
                    "Look at the image carefully and answer the question.",
                )
            ),
            "--max-new-tokens",
            str(request.get("max_new_tokens", 16)),
            "--warmup",
            str(arguments.warmup),
            "--iterations",
            str(arguments.iterations),
            "--resolution",
            str(options.get("resolution", "image_768res")),
            "--height",
            str(options.get("height", 768)),
            "--width",
            str(options.get("width", 768)),
            "--output",
            str(output),
        ]
        if arguments.revision:
            command.extend(["--revision", arguments.revision])
        if arguments.local_files_only:
            command.append("--local-files-only")
        completed = subprocess.run(
            command,
            cwd=reference_repo,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Lance reference failed with rc={completed.returncode}: {completed.stderr[-2000:]}"
            )
        payload = json.loads(output.read_text(encoding="utf-8"))
    samples = [float(value) for value in payload.get("samples_ms", [])]
    if len(samples) != arguments.iterations:
        raise RuntimeError(
            f"Lance reference returned {len(samples)} samples; expected {arguments.iterations}"
        )
    return (
        samples,
        {"text": str(payload.get("text", "")), "output_tokens": None},
        "lance-pytorch",
        "task-pipeline-call-wall",
        True,
    )


def _run_lerobot_act(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    options: Mapping[str, Any],
) -> tuple[list[float], dict[str, Any], str, str, bool]:
    source_root = str(options.get("source_root", ""))
    if not source_root:
        raise ValueError("LeRobot ACT reference requires adapter_options.source_root")
    source_root = _reference_checkout(
        source_root, repository="https://github.com/huggingface/lerobot"
    )
    if arguments.operation != "control" or arguments.precision != "fp32":
        raise ValueError("LeRobot ACT performance requires control in fp32")

    with tempfile.TemporaryDirectory(prefix="trtmc-perf-lerobot-act-") as temporary:
        output = Path(temporary) / "result.json"
        command = [
            sys.executable,
            str(REPOSITORY / "families/lerobot_act/tests/performance_reference.py"),
            "--source-root",
            source_root,
            "--model",
            arguments.model,
            "--image",
            str(_asset_path(arguments, request, "image_path")),
            "--state",
            str(_asset_path(arguments, request, "state_path")),
            "--warmup",
            str(arguments.warmup),
            "--iterations",
            str(arguments.iterations),
            "--output",
            str(output),
        ]
        if arguments.revision:
            command.extend(("--revision", arguments.revision))
        if arguments.local_files_only:
            command.append("--local-files-only")
        completed = subprocess.run(
            command,
            cwd=REPOSITORY,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                "LeRobot ACT reference failed with "
                f"rc={completed.returncode}: {completed.stderr[-2000:]}"
            )
        payload = json.loads(output.read_text(encoding="utf-8"))
    samples = [float(value) for value in payload.get("samples_ms", [])]
    if len(samples) != arguments.iterations:
        raise RuntimeError(
            f"LeRobot ACT reference returned {len(samples)} samples; "
            f"expected {arguments.iterations}"
        )
    summary = payload.get("output_summary")
    if not isinstance(summary, Mapping):
        raise RuntimeError("LeRobot ACT reference returned no output summary")
    return (
        samples,
        dict(summary),
        str(payload.get("framework", "lerobot-pytorch")),
        "task-pipeline-call-wall",
        True,
    )


def _run_sana_wm(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    options: Mapping[str, Any],
) -> tuple[list[float], dict[str, Any], str, str, bool]:
    reference_repo = str(options.get("reference_repo", ""))
    if not reference_repo:
        raise ValueError("upstream-sana-wm requires baseline.adapter_options.reference_repo")
    reference_repo = _reference_checkout(
        reference_repo, repository="https://github.com/NVlabs/Sana"
    )

    model_root = arguments.manifest.resolve().parent.parent

    def model_input(value: object) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else (model_root / path).resolve()

    image = model_input(request["image_path"])
    prompt_text = str(request.get("prompt", ""))
    if not prompt_text:
        raise ValueError("SANA-WM request requires prompt")
    action = request.get("action")
    if not isinstance(action, str) or not action:
        raise ValueError("SANA-WM request requires a non-empty action")
    intrinsics = model_input(options.get("intrinsics", "assets/demo_0_intrinsics.npy"))
    for label, path in (("image", image), ("intrinsics", intrinsics)):
        if not path.is_file():
            raise FileNotFoundError(f"SANA-WM {label} input does not exist: {path}")

    with tempfile.TemporaryDirectory(prefix="trtmc-perf-sana-wm-") as temporary:
        root = Path(temporary)
        output = root / "benchmark.json"
        prompt = root / "prompt.txt"
        prompt.write_text(prompt_text, encoding="utf-8")
        command = [
            sys.executable,
            str(REPOSITORY / "apps/benchmark/performance/baselines/sana_wm_reference.py"),
            "--reference-repo",
            reference_repo,
            "--image",
            str(image),
            "--model-dir",
            str(options["model_dir"]),
            "--prompt",
            str(prompt),
            "--action",
            action,
            "--intrinsics",
            str(intrinsics),
            "--translation_speed",
            str(request["translation_speed"]),
            "--rotation_speed_deg",
            str(request["rotation_speed_deg"]),
            "--num_frames",
            str(request["num_frames"]),
            "--fps",
            str(request["fps"]),
            "--step",
            str(request["num_steps"]),
            "--cfg_scale",
            str(request["cfg_scale"]),
            "--flow_shift",
            str(request["flow_shift"]),
            "--seed",
            str(request["seed"]),
            "--refiner_seed",
            str(request["seed"]),
            "--warmup",
            str(arguments.warmup),
            "--iterations",
            str(arguments.iterations),
            "--output",
            str(output),
        ]
        if request["no_action_overlay"]:
            command.append("--no_action_overlay")
        completed = subprocess.run(
            command,
            cwd=reference_repo,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                "SANA-WM reference failed with "
                f"rc={completed.returncode}: {completed.stderr[-4000:]}"
            )
        payload = json.loads(output.read_text(encoding="utf-8"))
    samples = [float(value) for value in payload.get("samples_ms", [])]
    if len(samples) != arguments.iterations:
        raise RuntimeError(
            f"SANA-WM reference returned {len(samples)} samples; expected {arguments.iterations}"
        )
    summary = dict(payload.get("output_summary", {}))
    return (
        samples,
        summary,
        "sana-wm-pytorch",
        "task-pipeline-call-wall",
        True,
    )


def _environment() -> dict[str, Any]:
    value: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
    }
    try:
        import torch

        value["torch"] = torch.__version__
        if torch.cuda.is_available():
            value["gpu"] = torch.cuda.get_device_name(0)
            value["cuda"] = torch.version.cuda
    except ImportError:
        pass
    for module_name in ("transformers", "diffusers", "nemo", "chronos", "moshi"):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        value[module_name] = str(getattr(module, "__version__", "unknown"))
    return value


def run(arguments: argparse.Namespace) -> int:
    if arguments.warmup < 0 or arguments.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    expected_mode = "pytorch-eager" if arguments.adapter in PYTORCH_ADAPTERS else "hf-eager"
    if arguments.mode != expected_mode:
        raise ValueError(f"adapter {arguments.adapter} requires mode {expected_mode}")
    request = _json_object(arguments.request_json, "--request-json")
    options = _json_object(arguments.adapter_options_json, "--adapter-options-json")
    configured_timing = _json_object(arguments.timing_contract_json, "--timing-contract-json")
    declared_timing = timing_contract(
        runner="task-reference",
        family=arguments.family,
    )
    expected_timing = {
        name: declared_timing[name]
        for name in (
            "timing_scope",
            "input_preparation_included",
            "asset_loading_included",
        )
    }
    if configured_timing and configured_timing != expected_timing:
        raise ValueError(
            f"configured timing contract does not match {arguments.family} reference: "
            f"configured={configured_timing}, reference={expected_timing}"
        )
    load_started = time.perf_counter()
    load_seconds: float | None = None
    if arguments.adapter == "upstream-elf":
        samples, output_summary, framework, timing_scope, input_included = _run_elf(
            arguments, request, options
        )
    elif arguments.adapter == "upstream-lance":
        (
            samples,
            output_summary,
            framework,
            timing_scope,
            input_included,
        ) = _run_lance(arguments, request, options)
    elif arguments.adapter == "pytorch-lerobot-act":
        (
            samples,
            output_summary,
            framework,
            timing_scope,
            input_included,
        ) = _run_lerobot_act(arguments, request, options)
    elif arguments.adapter == "upstream-sana-wm":
        (
            samples,
            output_summary,
            framework,
            timing_scope,
            input_included,
        ) = _run_sana_wm(arguments, request, options)
    else:
        session = LOADERS[arguments.adapter](arguments, request, options)
        load_seconds = time.perf_counter() - load_started
        framework = session.framework
        timing_scope = session.timing_scope
        input_included = session.input_preparation_included
        asset_included = session.asset_loading_included
        actual_timing = {
            "timing_scope": timing_scope,
            "input_preparation_included": input_included,
            "asset_loading_included": asset_included,
        }
        if actual_timing != expected_timing:
            raise RuntimeError(
                f"{arguments.family} reference implementation timing drifted: "
                f"actual={actual_timing}, declared={expected_timing}"
            )
        samples, output_summary = _measure(session, arguments.warmup, arguments.iterations)
        disparity = output_summary.pop("_disparity_f32", None)
        if disparity is not None:
            artifact_path = arguments.output.with_suffix(".disparity.f32").resolve()
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            disparity.tofile(artifact_path)
            output_summary["disparity_artifact"] = str(artifact_path)
    if arguments.adapter in {
        "pytorch-lerobot-act",
        "upstream-elf",
        "upstream-lance",
        "upstream-sana-wm",
    }:
        asset_included = bool(expected_timing["asset_loading_included"])
        actual_timing = {
            "timing_scope": timing_scope,
            "input_preparation_included": input_included,
            "asset_loading_included": asset_included,
        }
        if actual_timing != expected_timing:
            raise RuntimeError(
                f"{arguments.family} reference implementation timing drifted: "
                f"actual={actual_timing}, declared={expected_timing}"
            )
    result = {
        "schema_version": "trtmc.perf-baseline/v1",
        "status": "completed",
        "backend": arguments.adapter,
        "adapter": arguments.adapter,
        "framework": framework,
        "mode": arguments.mode,
        "precision": arguments.precision,
        "padding": arguments.padding,
        "experts_implementation": None,
        "compile_scope": None,
        "compile_evidence": None,
        "timing_scope": timing_scope,
        "input_preparation_included": input_included,
        "asset_loading_included": asset_included,
        "model_load_included": False,
        "model_load_seconds": load_seconds,
        "model": arguments.model,
        "case_name": arguments.case_name,
        "measurement": {
            "warmup": arguments.warmup,
            "iterations": arguments.iterations,
        },
        "measurement_policy": {
            "timing_scope": timing_scope,
            "input_preparation_included": input_included,
            "asset_loading_included": asset_included,
            "model_load_excluded": True,
            "warmup_excluded": True,
            "output_materialization_included": True,
        },
        "samples_ms": samples,
        "metrics": {
            "sample_count": len(samples),
            "latency_ms": {
                "p50": statistics.median(samples),
                "min": min(samples),
                "max": max(samples),
                "mean": statistics.fmean(samples),
            },
        },
        "output_summary": output_summary,
        "environment": _environment(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in samples):
        raise RuntimeError("reference produced an invalid timing sample")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = build_parser().parse_args(argv)
        if arguments.local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        return run(arguments)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
