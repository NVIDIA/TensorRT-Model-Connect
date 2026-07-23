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


REPOSITORY = Path(__file__).resolve().parents[3]
for source_root in (REPOSITORY, REPOSITORY / "python"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
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
    "pytorch-personaplex",
    "pytorch-timeseries",
    "upstream-elf",
    "upstream-lance",
)
PYTORCH_ADAPTERS = {
    "nemo-asr",
    "nemo-tts",
    "pytorch-personaplex",
    "pytorch-timeseries",
    "upstream-elf",
    "upstream-lance",
}


@dataclass(frozen=True)
class Session:
    """One loaded reference model and its repeatable timed operation."""

    invoke: Callable[[], Mapping[str, Any]]
    resolved_revision: str
    framework: str
    timing_scope: str = "task-model-call-wall"
    input_preparation_included: bool = False
    reference_dependencies: Mapping[str, str] | None = None
    reference_source: Mapping[str, str] | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, choices=ADAPTERS)
    parser.add_argument("--family", required=True)
    parser.add_argument("--operation", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--runtime-json", default="{}")
    parser.add_argument("--adapter-options-json", default="{}")
    parser.add_argument("--precision", required=True, choices=("fp16", "fp32", "bf16"))
    parser.add_argument("--mode", required=True, choices=("hf-eager", "pytorch-eager"))
    parser.add_argument("--padding", default="longest")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--workload-digest", required=True)
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


def _pinned_checkout_revision(
    checkout: str, expected_revision: str, *, repository: str
) -> str:
    path = Path(checkout).resolve()
    if not path.is_dir():
        raise ValueError(f"{repository} checkout does not exist: {path}")
    if not expected_revision:
        raise ValueError(f"{repository} requires a pinned reference_commit")
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={path}",
            "-C",
            str(path),
            "rev-parse",
            "HEAD",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if completed.returncode or not revision:
        detail = completed.stderr.strip() or "git rev-parse returned no revision"
        raise ValueError(f"cannot resolve {repository} checkout revision: {detail}")
    if revision != expected_revision:
        raise ValueError(
            f"{repository} checkout revision mismatch: expected "
            f"{expected_revision}, got {revision}"
        )
    return revision


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


def _model_revision(model: Any, requested: str | None) -> str:
    config = getattr(model, "config", None)
    return str(getattr(config, "_commit_hash", None) or requested or "unresolved")


def _snapshot_revision(path: str | Path) -> str:
    parts = Path(path).resolve().parts
    try:
        index = parts.index("snapshots")
    except ValueError:
        return ""
    return parts[index + 1] if index + 1 < len(parts) else ""


def _cached_snapshot_revision(repo_id: str, requested: str | None) -> str:
    try:
        from huggingface_hub import scan_cache_dir

        repository = next(
            (repo for repo in scan_cache_dir().repos if repo.repo_id == repo_id), None
        )
    except (ImportError, OSError, ValueError):
        return ""
    if repository is None:
        return ""
    revisions = list(repository.revisions)
    if requested:
        matches = [
            revision
            for revision in revisions
            if revision.commit_hash == requested or requested in revision.refs
        ]
    else:
        matches = [revision for revision in revisions if "main" in revision.refs]
        if not matches and len(revisions) == 1:
            matches = revisions
    return str(matches[0].commit_hash) if len(matches) == 1 else ""


def _cached_snapshot_path(
    repo_id: str, requested: str | None, marker_file: str
) -> Path | None:
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


def _resolved_revision(arguments: argparse.Namespace, model: Any) -> str:
    revision = _model_revision(model, arguments.revision)
    if revision != "unresolved":
        return revision
    if Path(arguments.model).exists():
        return "local-path"
    try:
        from huggingface_hub import hf_hub_download, snapshot_download

        for filename in ("config.json", "model_index.json"):
            try:
                path = hf_hub_download(
                    repo_id=arguments.model,
                    filename=filename,
                    revision=arguments.revision,
                    local_files_only=True,
                )
            except (OSError, ValueError):
                continue
            revision = _snapshot_revision(path)
            if revision:
                return revision
        try:
            return _snapshot_revision(
                snapshot_download(
                    repo_id=arguments.model,
                    revision=arguments.revision,
                    local_files_only=True,
                )
            )
        except (OSError, ValueError):
            pass
    except ImportError:
        pass
    return _cached_snapshot_revision(arguments.model, arguments.revision) or "unresolved"


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


def _runtime_config(arguments: argparse.Namespace) -> Mapping[str, Any]:
    runtime = getattr(arguments, "resolved_runtime", {})
    config = runtime.get("config", {}) if isinstance(runtime, Mapping) else {}
    return config if isinstance(config, Mapping) else {}


def _task_value(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    name: str,
    default: Any = None,
) -> Any:
    if name in request:
        return request[name]
    return _runtime_config(arguments).get(f"{arguments.family}.{name}", default)


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

        try:
            from nemo.collections.tts.models import MagpieTTSModel
        except ImportError:
            from nemo.collections.tts.models import MagpieTTS_Model as MagpieTTSModel

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
        seed = int(_runtime_config(arguments).get("audio_magpie.seed", 42))

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

        def invoke() -> Mapping[str, Any]:
            _seed_all(torch, 42)
            with torch.inference_mode():
                audio = model.generate(**inputs)
            return {
                "audio_samples": int(audio.numel()),
                "sample_rate": int(model.generation_config.sample_rate),
            }

    if arguments.family == "magpie_tts":
        revision = arguments.revision or _snapshot_revision(model_archive) or "unresolved"
        framework = "nemo"
        dependencies = {
            MAGPIE_SPEAKER_ENCODER_REPO: _snapshot_revision(speaker_checkpoint) or speaker_revision
        }
    else:
        revision = _resolved_revision(arguments, model)
        framework = "transformers"
        dependencies = None
    return Session(invoke, revision, framework, reference_dependencies=dependencies)


def _load_asr(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    _options: Mapping[str, Any],
) -> Session:
    import torch
    from tools.task_eval import _read_wav_float32, _resample_audio

    audio, sample_rate = _read_wav_float32(str(_asset_path(arguments, request, "audio_path")))
    target_rate = 16_000
    audio = _resample_audio(audio, sample_rate, target_rate)
    max_new_tokens = int(request.get("max_new_tokens", 100))
    device = torch.device("cuda")

    if arguments.family in {"canary", "nemotron_speech_streaming"}:
        import nemo.collections.asr as nemo_asr
        from tools.task_eval import _transcription_text

        model = (
            nemo_asr.models.ASRModel.from_pretrained(arguments.model, map_location="cpu")
            .eval()
            .to(device)
        )
        reference_dtype = _torch_dtype(torch, arguments.precision)
        autocast_dtype = reference_dtype if arguments.precision != "fp32" else torch.float16
        temporary = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temporary.close()
        atexit.register(Path(temporary.name).unlink, missing_ok=True)
        from tools.task_eval import _write_wav_pcm16

        _write_wav_pcm16(Path(temporary.name), audio, target_rate)

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
            return {"text": _transcription_text(result), "output_tokens": None}

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
    return Session(invoke, _resolved_revision(arguments, model), framework)


def _load_vlm_model(transformers_module: Any, model_id: str, kwargs: dict[str, Any]) -> Any:
    errors = []
    for name in (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
        "AutoModel",
    ):
        model_class = getattr(transformers_module, name, None)
        if model_class is None:
            continue
        try:
            return model_class.from_pretrained(model_id, **kwargs)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError(
        "no Transformers vision-language AutoClass accepted the model: " + "; ".join(errors)
    )


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
        _resolved_revision(arguments, model),
        "transformers",
        timing_scope="task-pipeline-call-wall",
        input_preparation_included=True,
    )


def _install_locateanything_compat() -> None:
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(DynamicCache, "to_legacy_cache"):

        def to_legacy_cache(self: Any) -> tuple[Any, ...]:
            return tuple((layer.keys, layer.values) for layer in self.layers)

        DynamicCache.to_legacy_cache = to_legacy_cache
    if not hasattr(DynamicCache, "from_legacy_cache"):

        @classmethod
        def from_legacy_cache(cls: Any, past_key_values: Any = None) -> Any:
            cache = cls()
            for layer_index, values in enumerate(past_key_values or ()):
                cache.update(values[0], values[1], layer_index)
            return cache

        DynamicCache.from_legacy_cache = from_legacy_cache
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}
    original = getattr(PreTrainedModel, "get_expanded_tied_weights_keys", None)
    if original is None or getattr(original, "_trtmc_locateanything_compat", False):
        return

    def get_expanded_tied_weights_keys(self: Any, all_submodels: bool = False) -> Any:
        tied = getattr(self, "_tied_weights_keys", None)
        if not isinstance(tied, list):
            return original(self, all_submodels=all_submodels)
        if all_submodels:
            expanded = {}
            for prefix, submodule in self.named_modules(remove_duplicate=False):
                if not isinstance(submodule, PreTrainedModel):
                    continue
                values = submodule.get_expanded_tied_weights_keys(all_submodels=False)
                if prefix:
                    values = {
                        f"{prefix}.{key}": f"{prefix}.{value}" for key, value in values.items()
                    }
                expanded.update(values)
            return expanded
        if not getattr(self.config, "tie_word_embeddings", False):
            return {}
        if "lm_head.weight" in tied:
            return {"lm_head.weight": "model.embed_tokens.weight"}
        return {}

    get_expanded_tied_weights_keys._trtmc_locateanything_compat = True
    PreTrainedModel.get_expanded_tied_weights_keys = get_expanded_tied_weights_keys


def _locateanything_config(arguments: argparse.Namespace, transformers_module: Any) -> Any:
    from transformers import PretrainedConfig

    options = _processor_kwargs(arguments)
    config = transformers_module.AutoConfig.from_pretrained(arguments.model, **options)
    raw_path = Path(arguments.model) / "config.json"
    if raw_path.is_file():
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        raw, _ = PretrainedConfig.get_config_dict(arguments.model, **options)
    if hasattr(config, "text_config") and not hasattr(config.text_config, "rope_theta"):
        text_config = raw.get("text_config", {}) if isinstance(raw, Mapping) else {}
        rope_theta = text_config.get("rope_theta")
        for key in ("rope_parameters", "rope_scaling"):
            nested = text_config.get(key, {})
            if rope_theta is None and isinstance(nested, Mapping):
                rope_theta = nested.get("rope_theta")
        if rope_theta is None and isinstance(raw, Mapping):
            rope_theta = raw.get("rope_theta", 10_000.0)
        config.text_config.rope_theta = float(rope_theta)
    return config


def _locateanything_tokenizer(arguments: argparse.Namespace, torch_module: Any) -> Any:
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(arguments.model, **_processor_kwargs(arguments))
    except (KeyError, TypeError, ValueError, OSError) as auto_error:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        model_path = Path(arguments.model)
        if model_path.is_dir():
            tokenizer_json = model_path / "tokenizer.json"
            tokenizer_config_path = model_path / "tokenizer_config.json"
        else:
            download_options = {
                "repo_id": arguments.model,
                "revision": arguments.revision,
                "local_files_only": arguments.local_files_only,
            }
            tokenizer_json = Path(hf_hub_download(filename="tokenizer.json", **download_options))
            try:
                tokenizer_config_path = Path(
                    hf_hub_download(filename="tokenizer_config.json", **download_options)
                )
            except (OSError, ValueError):
                tokenizer_config_path = Path()
        raw_tokenizer = Tokenizer.from_file(str(tokenizer_json))
        tokenizer_config = (
            json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
            if tokenizer_config_path.is_file()
            else {}
        )

        class TokenizersWrapper:
            def __init__(self) -> None:
                self.model_max_length = int(tokenizer_config.get("model_max_length", 16_384))

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return raw_tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

            def __call__(self, text: str, return_tensors: str | None = None) -> Any:
                ids = self.encode(text)
                if return_tensors == "pt":
                    input_ids = torch_module.tensor([ids], dtype=torch_module.long)
                    return {
                        "input_ids": input_ids,
                        "attention_mask": torch_module.ones_like(input_ids),
                    }
                return {"input_ids": ids}

            def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
                if torch_module.is_tensor(ids):
                    ids = ids.detach().cpu().tolist()
                if isinstance(ids, int):
                    ids = [ids]
                return raw_tokenizer.decode(
                    [int(token) for token in ids],
                    skip_special_tokens=skip_special_tokens,
                )

            def batch_decode(
                self, batch_ids: Any, skip_special_tokens: bool = True
            ) -> list[str]:
                if torch_module.is_tensor(batch_ids):
                    batch_ids = batch_ids.detach().cpu().tolist()
                return [
                    self.decode(ids, skip_special_tokens=skip_special_tokens)
                    for ids in batch_ids
                ]

        print(
            f"warning: AutoTokenizer failed ({auto_error}); using tokenizer.json",
            file=sys.stderr,
        )
        return TokenizersWrapper()


def _repair_locateanything_rotary_buffers(model: Any, torch_module: Any) -> None:
    repaired = 0
    model_device = next(model.parameters()).device
    for module in model.language_model.modules():
        if not all(hasattr(module, name) for name in ("_set_cos_sin_cache", "base", "dim")):
            continue
        device = getattr(getattr(module, "inv_freq", None), "device", model_device)
        inv_freq = 1.0 / (
            float(module.base)
            ** (
                torch_module.arange(
                    0,
                    int(module.dim),
                    2,
                    device=device,
                    dtype=torch_module.float32,
                )
                / float(module.dim)
            )
        )
        module.register_buffer("inv_freq", inv_freq, persistent=False)
        sequence_length = int(
            getattr(
                module,
                "max_position_embeddings",
                getattr(model.config.text_config, "max_position_embeddings", 32_768),
            )
        )
        module._set_cos_sin_cache(
            seq_len=sequence_length,
            device=inv_freq.device,
            dtype=torch_module.float32,
        )
        repaired += 1
    if repaired == 0:
        raise RuntimeError("LocateAnything reference did not find rotary buffers to repair")


def _load_locateanything(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
) -> Session:
    import torch
    import transformers
    from tensorrt_model_connect.families.locateanything.vl_debug_runner import (
        preprocess_image_inputs_for_trt,
    )

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
    _install_locateanything_compat()
    device = torch.device("cuda")
    config = _locateanything_config(arguments, transformers)
    tokenizer = _locateanything_tokenizer(arguments, torch)
    load_options = _load_kwargs(arguments, torch)
    load_options["config"] = config
    model = (
        transformers.AutoModel.from_pretrained(arguments.model, **load_options).eval().to(device)
    )
    _repair_locateanything_rotary_buffers(model, torch)

    image_inputs = preprocess_image_inputs_for_trt(
        str(_asset_path(arguments, request, "image_path")),
        preprocessor_type="patchify_chw",
        fixed_image_size=448,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        patch_size=14,
        interpolation="bicubic",
    )
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

    return Session(invoke, _resolved_revision(arguments, model), "transformers")


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
    model = _load_vlm_model(transformers, arguments.model, load_options).eval().to(device)
    image = Image.open(_asset_path(arguments, request, "image_path")).convert("RGB")
    prompt = str(request.get("prompt", ""))
    max_new_tokens = int(request.get("max_new_tokens", 16))

    fallback_text = {
        "internvl": f"<IMG_CONTEXT>\n{prompt}",
        "phi4_multimodal": f"<|image_1|>{prompt}",
        "qwen_vl": f"<|vision_start|><|image_pad|><|vision_end|>{prompt}",
    }.get(arguments.family, prompt)
    rendered = ""
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
    try:
        rendered = template_owner.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not isinstance(rendered, str) or not _vl_prompt_has_image_placeholder(rendered):
            raise ValueError("vision-language chat template lost the image placeholder")
        inputs = processor(text=rendered, images=image, padding=True, return_tensors="pt")
    except (AttributeError, KeyError, TypeError, ValueError):
        rendered = fallback_text
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

    return Session(invoke, _resolved_revision(arguments, model), "transformers")


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
    inputs = _to_device(
        tokenizer(str(request.get("prompt", "")), return_tensors="pt", truncation=True),
        device,
    )

    def invoke() -> Mapping[str, Any]:
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

    return Session(invoke, _resolved_revision(arguments, model), "transformers")


def _load_reranking(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    _options: Mapping[str, Any],
) -> Session:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoProcessor, AutoTokenizer

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
    try:
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
    except (AttributeError, KeyError, TypeError, ValueError, OSError):
        tokenizer = AutoTokenizer.from_pretrained(arguments.model, **_processor_kwargs(arguments))
        inputs = tokenizer(
            [f"question:{query}   passage:{document}" for document in documents],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
    inputs = _to_device(inputs, device)

    def invoke() -> Mapping[str, Any]:
        with torch.inference_mode():
            logits = model(**inputs).logits.detach().float().cpu()
        if logits.ndim == 2:
            logits = logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1]
        scores = [float(score) for score in logits.reshape(-1).tolist()]
        return {"scores": scores, "document_count": len(documents)}

    return Session(invoke, _resolved_revision(arguments, model), "transformers")


def _diffusion_pipeline(
    arguments: argparse.Namespace,
    torch_module: Any,
    options: Mapping[str, Any],
) -> Any:
    import diffusers

    classes = {
        "flux": ("FluxPipeline",),
        "ltx_video": ("LTXPipeline", "LTXVideoPipeline", "DiffusionPipeline"),
        "pixart": ("PixArtSigmaPipeline", "DiffusionPipeline"),
        "qwen_image": ("QwenImagePipeline", "DiffusionPipeline"),
        "sana_wm": ("SanaVideoPipeline", "DiffusionPipeline"),
        "wan_t2v": ("WanPipeline", "DiffusionPipeline"),
        "z_image": ("ZImagePipeline", "DiffusionPipeline"),
    }[arguments.family]
    model_source: str | Path = arguments.model
    if arguments.local_files_only:
        model_source = (
            _cached_snapshot_path(arguments.model, arguments.revision, "model_index.json")
            or model_source
        )
    errors = []
    for name in classes:
        pipeline_class = getattr(diffusers, name, None)
        if pipeline_class is None:
            continue
        try:
            return pipeline_class.from_pretrained(
                model_source,
                torch_dtype=_torch_dtype(torch_module, arguments.precision),
                **(
                    {"revision": arguments.revision}
                    if arguments.revision and model_source == arguments.model
                    else {}
                ),
                trust_remote_code=bool(
                    options.get("trust_remote_code", arguments.trust_remote_code)
                ),
                local_files_only=arguments.local_files_only,
            )
        except (TypeError, ValueError, OSError) as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("no Diffusers pipeline accepted the model: " + "; ".join(errors))


def _load_diffusers(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    options: Mapping[str, Any],
) -> Session:
    import inspect
    import torch
    from PIL import Image

    pipeline = _diffusion_pipeline(arguments, torch, options)
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
    values: dict[str, Any] = {"prompt": str(request.get("prompt", ""))}
    negative_prompt = str(request.get("negative_prompt", ""))
    if negative_prompt:
        values["negative_prompt"] = negative_prompt
    steps = int(request.get("num_inference_steps", -1))
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
    if "generator" in accepted or accepts_extra:
        call_values["generator"] = torch.Generator("cuda").manual_seed(seed)

    def invoke() -> Mapping[str, Any]:
        if "generator" in call_values:
            call_values["generator"].manual_seed(seed)
        result = pipeline(**call_values)
        media = getattr(result, "images", None)
        if media is None:
            media = getattr(result, "frames", None)
        media_type = str(request.get("media_type", "image"))
        return {"media_type": media_type, "media_count": _media_count(media, media_type)}

    revision = _resolved_revision(arguments, getattr(pipeline, "transformer", pipeline))
    return Session(
        invoke,
        revision,
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
        context = torch.tensor(_numeric_values(request, "branch_input"), dtype=dtype, device=device)

        def invoke() -> Mapping[str, Any]:
            with torch.inference_mode():
                value = model.predict(
                    context,
                    prediction_length=model.model_prediction_length,
                    limit_prediction_length=True,
                )
            return _tensor_summary(value)

        revision = _resolved_revision(arguments, getattr(model, "model", model))
        return Session(invoke, revision, "chronos")

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
        raw = _numeric_values(request, "branch_input")
        series = torch.tensor(_align(raw, length, 0.0), dtype=dtype, device=device).reshape(
            1, length
        )
        padding = [1] * length
        padding[-min(len(raw), length) :] = [0] * min(len(raw), length)
        padding_tensor = torch.tensor(padding, dtype=torch.int32, device=device).reshape(1, length)
        frequency = int(round(_numeric_values(request, "trunk_input")[0]))
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
        model_class = (
            transformers.PatchTSMixerForPrediction
            if is_mixer
            else transformers.PatchTSTForRegression
        )
        if not hasattr(model_class, "all_tied_weights_keys"):
            model_class.all_tied_weights_keys = {}
        model = (
            model_class.from_pretrained(arguments.model, **_load_kwargs(arguments, torch))
            .eval()
            .to(device)
        )
        length = int(config.context_length)
        channels = int(config.num_input_channels)
        raw = _numeric_values(request, "field_input")
        values = torch.tensor(
            _align(raw, length * channels, 0.0), dtype=dtype, device=device
        ).reshape(1, length, channels)
        raw_mask = request.get("trunk_input")
        if is_mixer and raw_mask is not None:
            mask_values = _numeric_values(request, "trunk_input")
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
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=dtype != torch.float32,
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
                    output = outputs.regression_outputs
            if isinstance(output, (tuple, list)):
                output = torch.stack(list(output), dim=-1)
            return _tensor_summary(output)

    return Session(invoke, _resolved_revision(arguments, model), "transformers")


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
    kwargs = _load_kwargs(arguments, torch)
    processor_kwargs = _processor_kwargs(arguments)

    if arguments.family == "timm_vit":
        import timm
        from timm.data import create_transform, resolve_model_data_config

        try:
            model = timm.create_model(f"hf-hub:{arguments.model}", pretrained=True)
        except (RuntimeError, ValueError):
            model = timm.create_model(f"hf_hub:{arguments.model}", pretrained=True)
        dtype = _torch_dtype(torch, arguments.precision)
        model = model.eval().to(device=device, dtype=dtype)
        transform = create_transform(**resolve_model_data_config(model), is_training=False)
        inputs = transform(image).unsqueeze(0).to(device=device, dtype=dtype)

        def invoke() -> Mapping[str, Any]:
            with torch.inference_mode():
                logits = model(inputs)
            return {"top_class": int(logits.argmax(dim=-1)[0]), **_tensor_summary(logits)}

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
        processor = transformers.Sam3Processor.from_pretrained(arguments.model, **processor_kwargs)
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
            return _tensor_summary(outputs.pred_masks)

    else:
        processor = transformers.SamProcessor.from_pretrained(arguments.model, **processor_kwargs)
        model = transformers.SamModel.from_pretrained(arguments.model, **kwargs).eval().to(device)
        width, height = image.size
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
            return _tensor_summary(outputs.pred_masks)

    return Session(invoke, _resolved_revision(arguments, model), "transformers")


def _qwen3_omni_chat_inputs(
    processor: Any, conversation: Sequence[Mapping[str, Any]]
) -> Any:
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
    load_options.update({"device_map": "auto", "enable_audio_output": True})
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        arguments.model, **load_options
    ).eval()
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_QWEN3_OMNI}]},
        {"role": "user", "content": [{"type": "text", "text": str(request.get("prompt", ""))}]},
    ]
    inputs = _qwen3_omni_chat_inputs(processor, conversation).to(model.device)
    thinker_tokens = int(request.get("max_new_tokens", 16))
    talker_tokens = int(options.get("talker_max_new_tokens", 32))

    def invoke() -> Mapping[str, Any]:
        _seed_all(torch, 42)
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

    return Session(invoke, _resolved_revision(arguments, model), "transformers")


def _load_personaplex(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    options: Mapping[str, Any],
) -> Session:
    official_repo = str(
        options.get("official_repo", "") or os.environ.get("PERSONAPLEX_OFFICIAL_REPO", "")
    )
    if not official_repo:
        raise ValueError(
            "pytorch-personaplex requires baseline.adapter_options.official_repo or "
            "PERSONAPLEX_OFFICIAL_REPO"
        )
    reference_revision = _pinned_checkout_revision(
        official_repo,
        str(options.get("reference_commit", "")),
        repository="https://github.com/NVIDIA/personaplex",
    )
    # The official checkout vendors the importable package below ``moshi/``.
    # Keep the checkout root too so sibling repository modules remain visible.
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
                        return {"audio_frames": generated_frames, "sample_rate": mimi.sample_rate}
        return {"audio_frames": generated_frames, "sample_rate": mimi.sample_rate}

    revision = arguments.revision or _snapshot_revision(model_weights) or "unresolved"
    return Session(
        invoke,
        revision,
        "moshi",
        timing_scope="task-pipeline-call-wall",
        input_preparation_included=True,
        reference_source={
            "repository": "https://github.com/NVIDIA/personaplex",
            "revision": reference_revision,
        },
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
) -> tuple[list[float], dict[str, Any], str, str, str, bool]:
    reference_repo = str(
        options.get("reference_repo", "") or os.environ.get("TRTMC_ELF_REFERENCE_REPO", "")
    )
    if not reference_repo:
        raise ValueError(
            "upstream-elf requires baseline.adapter_options.reference_repo or "
            "TRTMC_ELF_REFERENCE_REPO"
        )
    count = arguments.warmup + arguments.iterations
    prompt = str(request.get("prompt", ""))
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
            str(REPOSITORY / "tools/elf_hf_reference.py"),
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
            str(request.get("num_inference_steps", 64)),
            "--cfg-scale",
            str(request.get("cfg_scale", 2.0)),
            "--self-cond-cfg-scale",
            str(options.get("self_cond_cfg_scale", 1.0)),
            "--sde-gamma",
            str(options.get("sde_gamma", 0.0)),
            "--seed",
            str(options.get("seed", 42)),
        ]
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
    reference_path = Path(reference_repo).resolve()
    revision = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={reference_path}",
            "-C",
            str(reference_path),
            "rev-parse",
            "HEAD",
        ],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return (
        samples,
        summary,
        revision or "unresolved",
        "elf-pytorch",
        "task-pipeline-call-wall",
        False,
    )


def _run_lance(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    options: Mapping[str, Any],
) -> tuple[list[float], dict[str, Any], str, str, str, bool, dict[str, str]]:
    reference_repo = str(
        options.get("reference_repo", "") or os.environ.get("TRTMC_LANCE_REFERENCE_REPO", "")
    )
    if not reference_repo:
        raise ValueError(
            "upstream-lance requires baseline.adapter_options.reference_repo or "
            "TRTMC_LANCE_REFERENCE_REPO"
        )
    reference_commit = str(options.get("reference_commit", ""))
    if not reference_commit:
        raise ValueError("upstream-lance requires adapter_options.reference_commit")
    if arguments.precision != "bf16":
        raise ValueError("upstream-lance requires bf16 precision")
    with tempfile.TemporaryDirectory(prefix="trtmc-perf-lance-adapter-") as temporary:
        output = Path(temporary) / "result.json"
        command = [
            sys.executable,
            str(REPOSITORY / "tools/lance_reference.py"),
            "--reference-repo",
            reference_repo,
            "--reference-commit",
            reference_commit,
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
        str(payload.get("model_revision", "unresolved")),
        "lance-pytorch",
        "task-pipeline-call-wall",
        True,
        {
            "repository": "https://github.com/bytedance/Lance",
            "revision": str(payload.get("reference_revision", reference_commit)),
        },
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
    arguments.resolved_runtime = _json_object(arguments.runtime_json, "--runtime-json")
    options = _json_object(arguments.adapter_options_json, "--adapter-options-json")
    load_started = time.perf_counter()
    load_seconds: float | None = None
    reference_source: dict[str, str] | None = None
    reference_dependencies: Mapping[str, str] | None = None
    if arguments.adapter == "upstream-elf":
        samples, output_summary, revision, framework, timing_scope, input_included = _run_elf(
            arguments, request, options
        )
    elif arguments.adapter == "upstream-lance":
        (
            samples,
            output_summary,
            revision,
            framework,
            timing_scope,
            input_included,
            reference_source,
        ) = _run_lance(arguments, request, options)
    else:
        session = LOADERS[arguments.adapter](arguments, request, options)
        load_seconds = time.perf_counter() - load_started
        samples, output_summary = _measure(session, arguments.warmup, arguments.iterations)
        revision = session.resolved_revision
        framework = session.framework
        timing_scope = session.timing_scope
        input_included = session.input_preparation_included
        reference_dependencies = session.reference_dependencies
        reference_source = session.reference_source
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
        "model_load_included": False,
        "model_load_seconds": load_seconds,
        "model": arguments.model,
        "resolved_revision": revision,
        "workload_digest": arguments.workload_digest,
        "measurement": {
            "warmup": arguments.warmup,
            "iterations": arguments.iterations,
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
    if reference_source is not None:
        result["reference_source"] = reference_source
    if reference_dependencies:
        result["reference_dependencies"] = dict(reference_dependencies)
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in samples):
        raise RuntimeError("reference produced an invalid timing sample")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
