#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark task-aligned non-text model reference operations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
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

from qualification_tests.benchmark_qualification.performance.references.timing_contracts import timing_contract  # noqa: E402
from qualification_tests.benchmark_qualification.performance.references.hf_transformers import _batch_prompt, flatten_config  # noqa: E402

ADAPTERS = (
    "hf-diffusers",
    "hf-diffusers-modular",
    "hf-transformers-asr",
    "hf-transformers-embedding",
    "hf-transformers-reranking",
    "hf-transformers-tts",
    "hf-transformers-vision",
    "hf-transformers-vlm",
    "pytorch-timeseries",
    "timm-classification",
)
PYTORCH_ADAPTERS = {"pytorch-timeseries"}


@dataclass(frozen=True)
class Session:
    """One loaded reference model and its repeatable timed operation."""

    invoke: Callable[[], Any]
    framework: str
    timing_scope: str = "task-model-call-wall"
    input_preparation_included: bool = False
    asset_loading_included: bool = False
    compile_evidence: dict[str, Any] | None = None
    summarize: Callable[[Any], Mapping[str, Any]] | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, choices=ADAPTERS)
    parser.add_argument("--operation", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--selected-task", help="Semantic Task selected independently of bundle identity"
    )
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--adapter-options-json", default="{}")
    parser.add_argument("--timing-contract-json", default="{}")
    parser.add_argument("--precision", required=True, choices=("fp16", "fp32", "bf16"))
    parser.add_argument(
        "--mode", required=True, choices=("hf-eager", "pytorch-eager", "torch-compile")
    )
    parser.add_argument("--padding", default="longest")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--case-name", required=True)
    parser.add_argument(
        "--testcase-name", help="Selected manifest testcase, not the performance entry ID"
    )
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


def _selected_task(arguments: argparse.Namespace) -> str:
    selected = getattr(arguments, "selected_task", None)
    if selected is None:
        return json.loads(arguments.manifest.read_text(encoding="utf-8"))["task"]
    if not isinstance(selected, str) or not selected or selected != selected.strip():
        raise ValueError("selected_task must be a nonempty Task ID without surrounding whitespace")
    return selected


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
        raise ValueError(f"{arguments.adapter} reference requires request.{key}")
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


def _flatten_tensor_values(value: Any) -> list[float]:
    flattened: list[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            flattened.append(float(item))

    visit(value.detach().float().cpu().tolist())
    return flattened


def _forecast_summary(
    value: Any, task: str, quantile_levels: Sequence[float] = ()
) -> dict[str, Any]:
    summary = _tensor_summary(value)
    summary["values"] = _flatten_tensor_values(value)
    if task not in {"series_to_point_forecast", "series_to_quantile_forecast"}:
        return summary
    shape = summary["shape"]
    if len(shape) not in {2, 3} or shape[0] != 1:
        raise ValueError("forecast reference requires an explicit one-series batch axis")
    if task == "series_to_point_forecast":
        horizon, channels = shape[1], shape[2] if len(shape) == 3 else 1
        summary.update(shape=[horizon, channels], axes=["horizon", "channel"])
    else:
        if len(shape) != 3 or len(quantile_levels) != shape[1]:
            raise ValueError(
                "quantile forecast reference requires exact checkpoint quantile levels"
            )
        horizon = shape[2]
        summary.update(
            shape=[shape[1], horizon, 1],
            axes=["quantile", "horizon", "channel"],
            quantile_levels=list(quantile_levels),
        )
    summary.update(
        forecast_elements=summary["element_count"], horizon_steps=list(range(1, horizon + 1))
    )
    return summary


def _regression_values_summary(value: Any) -> dict[str, Any]:
    summary = _tensor_summary(value)
    shape = summary["shape"]
    if len(shape) != 2 or shape[0] != 1 or shape[1] <= 0:
        raise ValueError("deterministic regression requires one batch with a nonempty target axis")
    if not summary["finite"]:
        raise ValueError("deterministic regression target values must be finite")
    return {
        "kind": "regression_values",
        "target_count": shape[1],
        "regression_targets": shape[1],
        "parameter_elements": 0,
        "values": value[0].detach().float().cpu().tolist(),
        "axes": ["target"],
        "target_names": [],
        "target_units": [],
    }


def _regression_summary(
    output: Any, distribution: str, parameter_names: Sequence[str]
) -> dict[str, Any]:
    if distribution not in {"normal", "student_t", "negative_binomial"}:
        raise ValueError("regression reference requires a declared probability distribution")
    if not isinstance(output, (tuple, list)) or len(output) != len(parameter_names) or not output:
        raise ValueError("regression reference parameters do not match the declared distribution")
    # Transformers names distribution parameters loc/df. The public Task
    # contract names them location/degrees_of_freedom; never infer tuple order.
    aliases = {"loc": "location", "df": "degrees_of_freedom"}
    names = [aliases.get(name, name) for name in parameter_names]
    required = {
        "normal": {"location", "scale"},
        "student_t": {"degrees_of_freedom", "location", "scale"},
        "negative_binomial": {"total_count", "logits"},
    }[distribution]
    if len(names) != len(required) or set(names) != required:
        raise ValueError(
            "regression reference parameter names do not match the declared distribution"
        )
    parameters = []
    targets = None
    for name, value in zip(names, output):
        shape = tuple(int(size) for size in value.shape)
        if (
            len(shape) != 2
            or shape[0] != 1
            or shape[1] <= 0
            or (targets is not None and shape[1] != targets)
        ):
            raise ValueError(
                "regression parameters must each have one batch and the same target axis"
            )
        targets = shape[1]
        if not _tensor_summary(value)["finite"]:
            raise ValueError("regression reference returned nonfinite parameters")
        parameters.append({"name": name, "values": value[0].detach().float().cpu().tolist()})
    return {
        "distribution": distribution,
        "target_count": targets,
        "regression_targets": targets,
        "parameter_elements": targets * len(parameters),
        "axes": ["target"],
        "parameters": parameters,
        "target_names": [],
        "target_units": [],
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
    _options: Mapping[str, Any],
) -> Session:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoProcessor, BarkModel

    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    prompt = str(request.get("prompt", ""))
    processor = AutoProcessor.from_pretrained(arguments.model, **_processor_kwargs(arguments))
    model = (
        BarkModel.from_pretrained(arguments.model, **_load_kwargs(arguments, torch))
        .eval()
        .to(device)
    )
    inputs = _to_device(processor(prompt, return_tensors="pt"), device)
    seed = _request_seed(request, 42)
    generation_options = _bark_generation_options(request)

    def invoke() -> Mapping[str, Any]:
        _seed_all(torch, seed)
        with torch.inference_mode():
            audio = model.generate(**inputs, **generation_options)
        return {
            "audio_samples": int(audio.numel()),
            "sample_rate": int(model.generation_config.sample_rate),
            "_audio_f32": audio.detach().float().cpu().reshape(-1).numpy(),
        }

    return Session(invoke, "transformers")


def _load_asr(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    _options: Mapping[str, Any],
) -> Session:
    import torch
    from qualification_tests.benchmark_qualification.performance.references.audio_reference import (
        read_wav_float32,
        resample_audio,
    )

    audio, sample_rate = read_wav_float32(str(_asset_path(arguments, request, "audio_path")))
    target_rate = 16_000
    audio = resample_audio(audio, sample_rate, target_rate)
    max_new_tokens = int(request.get("max_new_tokens", 100))
    device = torch.device("cuda")

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

    return Session(invoke, "transformers")


def _load_vlm_model(
    transformers_module: Any, class_name: str, model_id: str, kwargs: dict[str, Any]
) -> Any:
    if class_name not in {
        "AutoModel",
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
    }:
        raise ValueError(f"unsupported vision-language model_class: {class_name}")
    model_class = getattr(transformers_module, class_name)
    return model_class.from_pretrained(model_id, **kwargs)


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
    options: Mapping[str, Any],
) -> Session:
    workflow = str(options.get("workflow", "chat"))
    if workflow not in {"chat", "single-image-token-chat"}:
        raise ValueError(f"unsupported vision-language workflow: {workflow}")

    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor

    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(arguments.model, **_processor_kwargs(arguments))
    load_options = _load_kwargs(arguments, torch)
    if bool(options.get("eager_attention", False)):
        config = transformers.AutoConfig.from_pretrained(
            arguments.model, **_processor_kwargs(arguments)
        )
        config._attn_implementation = "eager"
        config._attn_implementation_internal = "eager"
        load_options.update({"config": config, "attn_implementation": "eager"})
    model = (
        _load_vlm_model(
            transformers,
            str(options.get("model_class", "AutoModelForImageTextToText")),
            arguments.model,
            load_options,
        )
        .eval()
        .to(device)
    )
    image = Image.open(_asset_path(arguments, request, "image_path")).convert("RGB")
    prompt = str(request.get("prompt", ""))
    max_new_tokens = int(request.get("max_new_tokens", 16))

    if workflow == "single-image-token-chat":
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
    configured = _json_object(
        getattr(arguments, "timing_contract_json", "{}"), "--timing-contract-json"
    )
    if not configured:
        raise ValueError("embedding reference requires explicit --timing-contract-json")
    declared_timing = timing_contract(runner="task-reference", declared=configured)
    if declared_timing["asset_loading_included"]:
        raise ValueError(
            "embedding reference preloads its assets; asset_loading_included must be false"
        )
    prompts = _batch_prompt(request)
    if len(prompts) != 1:
        raise ValueError("embedding reference requires exactly one text input")
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(arguments.model, **_processor_kwargs(arguments))
    model = (
        AutoModel.from_pretrained(arguments.model, **_load_kwargs(arguments, torch))
        .eval()
        .to(device)
    )
    compile_evidence = _compile_forward(model) if arguments.mode == "torch-compile" else None
    prompt = prompts[0]

    def prepare_inputs() -> Mapping[str, Any]:
        return _to_device(
            tokenizer(prompt, return_tensors="pt", truncation=True),
            device,
        )

    prepared_inputs = None
    if not declared_timing["input_preparation_included"]:
        prepared_inputs = prepare_inputs()

    def invoke() -> Any:
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
        return vector.detach().to("cpu", dtype=torch.float32)

    def summarize(vector: Any) -> Mapping[str, Any]:
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
        asset_loading_included=False,
        compile_evidence=compile_evidence,
        summarize=summarize,
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
    if arguments.precision != "fp16":
        raise ValueError(f"{contract} requires precision=fp16")
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
    vae_class_name = options.get("vae_class")
    if vae_class_name is not None:
        if not isinstance(vae_class_name, str) or not vae_class_name.startswith("Autoencoder"):
            raise ValueError("diffusion adapter_options.vae_class must name an Autoencoder class")
        try:
            vae_class = getattr(diffusers, vae_class_name)
        except AttributeError as error:
            raise ValueError(f"unsupported Diffusers VAE class: {vae_class_name}") from error
        vae_precision = str(options.get("vae_precision", arguments.precision))
        load_options["vae"] = vae_class.from_pretrained(
            model_source,
            subfolder="vae",
            torch_dtype=_torch_dtype(torch_module, vae_precision),
            **(
                {"revision": requested_revision}
                if requested_revision and model_source == model_id
                else {}
            ),
            local_files_only=arguments.local_files_only,
        )
    return diffusers.DiffusionPipeline.from_pretrained(
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
    if "prompt" in request and "prompts" in request:
        raise ValueError("reference request must not provide both prompt and prompts")
    prompt = request.get("prompts", request.get("prompt", ""))
    if "prompts" in request and not isinstance(prompt, list):
        raise ValueError("prompts must contain one string per batch item")
    if isinstance(prompt, list):
        if not prompt or any(not isinstance(value, str) for value in prompt):
            raise ValueError("prompt list must contain one string per batch item")
        count = len(prompt)
    else:
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string or a non-empty list of strings")
        count = 1
    batch_size = request.get("batch_size", count)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(prompt, list):
        if count != batch_size:
            raise ValueError("prompt list must contain one string per batch item")
        prompt_value: str | list[str] = list(prompt)
    else:
        prompt_value = [prompt] * batch_size if batch_size > 1 else prompt
    values: dict[str, Any] = {"prompt": prompt_value}
    negative_prompt = str(request.get("negative_prompt", ""))
    if (
        negative_prompt
        or "negative_prompt" in request
        or bool(options.get("always_negative_prompt", False))
    ):
        values["negative_prompt"] = negative_prompt
    steps = int(request.get("num_steps", -1))
    if steps > 0:
        values.update({"num_inference_steps": steps, "step": steps})
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    height = int(
        request.get("height", request.get("video_height", manifest.get("image_height", 0)))
    )
    width = int(request.get("width", request.get("video_width", manifest.get("image_width", 0))))
    if height > 0:
        values["height"] = height
    if width > 0:
        values["width"] = width
    num_frames = int(
        request.get(
            "video_num_frames",
            _task_value(arguments, request, "num_frames", manifest.get("video_num_frames", 1)),
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
    guidance_parameter = str(options.get("guidance_parameter", "guidance_scale"))
    if guidance_parameter not in {"guidance_scale", "true_cfg_scale"}:
        raise ValueError(f"unsupported diffusion guidance_parameter: {guidance_parameter}")
    if cfg_scale >= 0 and guidance_parameter != "true_cfg_scale":
        values["cfg_scale"] = cfg_scale
    if bool(request.get("no_refiner", False)):
        values["no_refiner"] = True
    guidance = float(request.get("guidance_scale", -1.0))
    if guidance_parameter == "true_cfg_scale":
        true_cfg_scale = guidance if guidance >= 0 else cfg_scale
        if true_cfg_scale >= 0:
            values["true_cfg_scale"] = true_cfg_scale
    elif guidance >= 0:
        values["guidance_scale"] = guidance
    values["output_type"] = "np"
    if "image_path" in request and "image_paths" in request:
        raise ValueError("reference request must not provide both image_path and image_paths")
    image_path = request.get("image_path", "")
    if "image_paths" in request:
        image_paths = request["image_paths"]
        if (
            not isinstance(image_paths, list)
            or len(image_paths) != 1
            or not isinstance(image_paths[0], str)
            or not image_paths[0]
        ):
            raise ValueError("this reference supports exactly one conditioning image")
        image_path = image_paths[0]
    task = getattr(arguments, "selected_task", None) or manifest["task"]
    if task in {"images_text_to_image_edit", "image_edit"} and not image_path:
        raise ValueError("image edit reference requires one conditioning image")
    if image_path:
        if "image" not in accepted and not accepts_extra:
            raise ValueError("selected reference pipeline does not accept conditioning image")
        values["image"] = Image.open(
            _asset_path(arguments, {"image_path": image_path}, "image_path")
        ).convert("RGB")
    call_values = {
        name: value for name, value in values.items() if name in accepted or accepts_extra
    }
    required = [str(name) for name in options.get("required_call_arguments", [])]
    missing = [name for name in required if name not in call_values]
    if missing:
        raise ValueError(
            f"{arguments.adapter} reference is missing required call arguments: "
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
        return {**_media_summary(media, media_type), "_media": media}

    return Session(
        invoke,
        "diffusers",
        timing_scope="task-pipeline-call-wall",
        input_preparation_included=True,
    )


def _load_modular_diffusers(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    options: Mapping[str, Any],
) -> Session:
    """Run a ModularPipeline whose workflow is declared by the family profile."""
    import torch
    from diffusers import ModularPipeline

    workflow = str(options.get("workflow", ""))
    output_name = str(options.get("output", "videos"))
    if not workflow or not output_name:
        raise ValueError("modular Diffusers requires adapter_options.workflow and output")
    load_options = {
        "workflow": workflow,
        "local_files_only": arguments.local_files_only,
    }
    if arguments.revision:
        load_options["revision"] = arguments.revision
    pipeline = ModularPipeline.from_pretrained(arguments.model, **load_options)
    pipeline.load_components(
        dtype=_torch_dtype(torch, arguments.precision),
        pretrained_model_name_or_path=arguments.model,
        local_files_only=arguments.local_files_only,
    )
    pipeline = pipeline.to("cuda")
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    seed = int(request.get("seed", 42))
    generator = torch.Generator().manual_seed(seed)
    values: dict[str, Any] = {
        "prompt": str(request.get("prompt", "")),
        "height": int(request.get("height", manifest.get("image_height", 0))),
        "width": int(request.get("width", manifest.get("image_width", 0))),
        "num_inference_steps": int(request.get("num_steps", 0)),
        "generator": generator,
        "output": output_name,
        "output_type": "np",
    }
    frames = int(request.get("video_num_frames", manifest.get("video_num_frames", 1)))
    if frames > 1:
        values["num_frames"] = frames
    for source, target in (
        ("negative_prompt", "negative_prompt"),
        ("guidance_scale", "guidance_scale"),
        ("cfg_scale", "cfg_scale"),
    ):
        if source in request:
            values[target] = request[source]

    def invoke() -> Mapping[str, Any]:
        generator.manual_seed(seed)
        media = pipeline(**values)
        media_type = "video" if frames > 1 else "image"
        return {**_media_summary(media, media_type), "_media": media}

    return Session(
        invoke,
        "diffusers-modular",
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


def _media_items(media: Any, media_type: str) -> list[Any]:
    """Return image items, or frames from the first generated video."""
    import numpy as np

    if media_type == "video":
        if isinstance(media, np.ndarray):
            values = media[0] if media.ndim == 5 else media
        else:
            values = media[0] if isinstance(media, (list, tuple)) and media else media
    else:
        values = media
    if isinstance(values, np.ndarray):
        if values.ndim == 3:
            return [values]
        return [values[index] for index in range(len(values))]
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


def _write_media_artifacts(summary: dict[str, Any], output: Path) -> None:
    """Materialize deterministic sample frames after reference timing completes."""
    import numpy as np
    from PIL import Image

    media = summary.pop("_media")
    items = _media_items(media, str(summary.get("media_type", "image")))
    declared = int(summary.get("media_count", 0))
    if declared != len(items) or declared < 1:
        raise RuntimeError("reference media count differs from its materialized output")
    indices = sorted({0, declared // 2, declared - 1})
    directory = output.with_suffix(".media").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for index in indices:
        item = items[index]
        values = np.asarray(item)
        scale = 255.0 if np.issubdtype(values.dtype, np.floating) else 1.0
        image = (
            item
            if isinstance(item, Image.Image)
            else Image.fromarray(np.clip(values * scale, 0, 255).astype(np.uint8))
        )
        path = directory / f"{index:06d}.png"
        image.convert("RGB").save(path)
        artifacts.append(str(path))
    summary["artifact_indices"] = indices
    summary["frame_artifacts" if summary.get("media_type") == "video" else "image_artifacts"] = (
        artifacts
    )


def _numeric_values(request: Mapping[str, Any], key: str) -> list[float]:
    raw = request.get(key)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"time-series reference requires non-empty request.{key}")
    return [float(value) for value in raw]


def _observed_values(request: Mapping[str, Any], count: int) -> list[float]:
    raw = request.get("observed_mask")
    if raw is None or raw == []:
        return [1.0] * count
    values = _numeric_values(request, "observed_mask")
    if len(values) != count:
        raise ValueError("time-series observed_mask must match past_values")
    return values


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
    options: Mapping[str, Any],
) -> Session:
    import torch
    import transformers

    device = torch.device("cuda")
    dtype = _torch_dtype(torch, arguments.precision)
    task_id = _selected_task(arguments)
    reference_type = str(options.get("reference_type", ""))
    if reference_type == "chronos-bolt":
        from chronos import ChronosBoltPipeline

        chronos_options = _processor_kwargs(arguments)
        chronos_options.update({"device_map": str(device), "dtype": dtype})
        model = ChronosBoltPipeline.from_pretrained(arguments.model, **chronos_options)
        compile_evidence = None
        if arguments.mode == "torch-compile":
            compile_evidence = _compile_forward(model.model)
        raw = _numeric_values(request, "past_values")
        observed = _observed_values(request, len(raw))
        context = torch.tensor(
            [value if mask > 0 else float("nan") for value, mask in zip(raw, observed)],
            dtype=dtype,
            device=device,
        )

        def invoke() -> Mapping[str, Any]:
            with torch.inference_mode():
                value = model.predict(
                    context,
                    prediction_length=model.model_prediction_length,
                    limit_prediction_length=True,
                )
            quantiles = model.model.config.chronos_config["quantiles"]
            return _forecast_summary(value, task_id, quantiles)

        return Session(invoke, "chronos", compile_evidence=compile_evidence)

    config = transformers.AutoConfig.from_pretrained(
        arguments.model, **_processor_kwargs(arguments)
    )
    if reference_type == "timesfm":
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
        observed = _align(_observed_values(request, len(raw)), length, 0.0)
        padding = [0 if mask > 0 else 1 for mask in observed]
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
            return _forecast_summary(output, task_id)

    else:
        if reference_type not in {"patchtsmixer", "patchtst"}:
            raise ValueError(f"unsupported time-series reference_type: {reference_type}")
        is_mixer = reference_type == "patchtsmixer"
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
        aligned_mask = _align(_observed_values(request, len(raw)), length * channels, 0.0)
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
            if task_id == "series_to_regression_distribution":
                return _regression_summary(
                    output, config.distribution_output, tuple(model.distribution_output.args_dim)
                )
            if task_id == "series_to_regression_values":
                return _regression_values_summary(output)
            if isinstance(output, (tuple, list)):
                output = torch.stack(list(output), dim=-1)
            return _forecast_summary(output, task_id)

    return Session(invoke, "transformers")


def _load_vision(
    arguments: argparse.Namespace,
    request: Mapping[str, Any],
    options: Mapping[str, Any],
) -> Session:
    import torch
    from PIL import Image
    import transformers

    device = torch.device("cuda")
    image = Image.open(_asset_path(arguments, request, "image_path")).convert("RGB")
    width, height = image.size
    kwargs = _load_kwargs(arguments, torch)
    processor_kwargs = _processor_kwargs(arguments)
    vision_task = str(options.get("vision_task", ""))

    if arguments.adapter == "timm-classification":
        import timm
        from timm.data import create_transform, resolve_model_data_config

        model_id = arguments.model
        if arguments.revision:
            model_id += "@" + arguments.revision
        model = timm.create_model(f"hf-hub:{model_id}", pretrained=True)
        dtype = _torch_dtype(torch, arguments.precision)
        model = model.eval().to(device=device, dtype=dtype)
        transform = create_transform(**resolve_model_data_config(model), is_training=False)
        inputs = transform(image).unsqueeze(0).to(device=device, dtype=dtype)

        def invoke() -> Mapping[str, Any]:
            with torch.inference_mode():
                logits = model(inputs)
            return {"top_class": int(logits.argmax(dim=-1)[0]), **_tensor_summary(logits)}

    elif vision_task == "object-detection":
        processor = transformers.AutoImageProcessor.from_pretrained(
            arguments.model, **processor_kwargs
        )
        model = (
            transformers.AutoModelForObjectDetection.from_pretrained(arguments.model, **kwargs)
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
                outputs = model(**inputs)
            target_sizes = torch.tensor([[height, width]], device=device)
            results = processor.post_process_object_detection(
                outputs, threshold=0.5, target_sizes=target_sizes
            )[0]
            return {
                "detected_images": 1,
                "detections": int(results["scores"].shape[0]),
                "image_height": height,
                "image_width": width,
                "boxes": results["boxes"].float().cpu().reshape(-1).tolist(),
                "scores": results["scores"].float().cpu().tolist(),
                "class_ids": results["labels"].cpu().tolist(),
                "shape": [int(results["scores"].shape[0]), 4],
                "coordinates": "xyxy",
                "units": "pixels",
            }

    elif vision_task == "image-features":
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

    elif vision_task == "semantic-segmentation":
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
                outputs = model(**inputs)
            mask = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(height, width)]
            )[0]
            return {
                "num_masks": 1,
                "height": int(mask.shape[0]),
                "width": int(mask.shape[1]),
            }

    else:
        raise ValueError(f"unsupported Transformers vision_task: {vision_task}")

    framework = "timm" if arguments.adapter == "timm-classification" else "transformers"
    return Session(invoke, framework)



LOADERS: dict[
    str, Callable[[argparse.Namespace, Mapping[str, Any], Mapping[str, Any]], Session]
] = {
    "hf-diffusers": _load_diffusers,
    "hf-diffusers-modular": _load_modular_diffusers,
    "hf-transformers-asr": _load_asr,
    "hf-transformers-embedding": _load_embedding,
    "hf-transformers-reranking": _load_reranking,
    "hf-transformers-tts": _load_tts,
    "hf-transformers-vision": _load_vision,
    "hf-transformers-vlm": _load_vlm,
    "pytorch-timeseries": _load_timeseries,
    "timm-classification": _load_vision,
}


def _synchronize() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        return


def _compile_forward(model: Any) -> dict[str, Any]:
    import torch
    from torch._dynamo.backends.registry import lookup_backend

    evidence = {"compiled_graph_count": 0}
    inductor = lookup_backend("inductor")

    def compile_graph(graph: Any, inputs: Any, **options: Any) -> Any:
        compiled = inductor(graph, inputs, **options)
        evidence["compiled_graph_count"] += 1
        return compiled

    model.forward = torch.compile(
        model.forward,
        backend=compile_graph,
        fullgraph=False,
        dynamic=False,
    )
    evidence.update(
        {
            "api": "torch.compile",
            "target": "model.forward",
            "backend": "inductor",
            "mode": "default",
            "fullgraph": False,
            "dynamic": False,
            "applied": True,
        }
    )
    return evidence


def _measure(session: Session, warmup: int, iterations: int) -> tuple[list[float], dict[str, Any]]:
    output: Any = {}
    untimed_iterations = max(warmup, int(session.compile_evidence is not None))
    for _ in range(untimed_iterations):
        output = session.invoke()
        _synchronize()
    compiled_graphs = None
    if session.compile_evidence is not None:
        compiled_graphs = int(session.compile_evidence["compiled_graph_count"])
        if compiled_graphs < 1:
            raise RuntimeError("warmup did not execute a compiled graph")
    samples = []
    for _ in range(iterations):
        _synchronize()
        started = time.perf_counter()
        output = session.invoke()
        _synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    if (
        compiled_graphs is not None
        and int(session.compile_evidence["compiled_graph_count"]) != compiled_graphs
    ):
        raise RuntimeError("model compilation occurred inside timed samples")
    summary = session.summarize(output) if session.summarize is not None else output
    return samples, dict(summary)


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
    options = _json_object(arguments.adapter_options_json, "--adapter-options-json")
    expected_mode = "pytorch-eager" if arguments.adapter in PYTORCH_ADAPTERS else "hf-eager"
    supported_modes = {expected_mode}
    if (
        arguments.adapter == "pytorch-timeseries"
        and options.get("reference_type") == "chronos-bolt"
    ):
        supported_modes.add("torch-compile")
    if arguments.adapter == "hf-transformers-embedding":
        supported_modes.add("torch-compile")
    if arguments.mode not in supported_modes:
        raise ValueError(f"adapter {arguments.adapter} requires one of {sorted(supported_modes)}")
    request = flatten_config(_json_object(arguments.request_json, "--request-json"))
    configured_timing = _json_object(arguments.timing_contract_json, "--timing-contract-json")
    fields = ("timing_scope", "input_preparation_included", "asset_loading_included")
    expected_timing = None
    if configured_timing:
        if set(configured_timing) != set(fields):
            raise ValueError(
                "--timing-contract-json must declare exactly the three reference timing fields"
            )
        declared = timing_contract(runner="task-reference", declared=configured_timing)
        expected_timing = {name: declared[name] for name in fields}
    load_started = time.perf_counter()
    load_seconds: float | None = None
    compile_evidence: dict[str, Any] | None = None
    session = LOADERS[arguments.adapter](arguments, request, options)
    compile_evidence = session.compile_evidence
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
    timing_contract(runner="task-reference", declared=actual_timing)
    if expected_timing is not None and actual_timing != expected_timing:
        raise RuntimeError(
            f"{arguments.adapter} reference implementation timing drifted: "
            f"actual={actual_timing}, declared={expected_timing}"
        )
    samples, output_summary = _measure(session, arguments.warmup, arguments.iterations)
    if "_media" in output_summary:
        _write_media_artifacts(output_summary, arguments.output)
    disparity = output_summary.pop("_disparity_f32", None)
    if disparity is not None:
        artifact_path = arguments.output.with_suffix(".disparity.f32").resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        disparity.tofile(artifact_path)
        output_summary["disparity_artifact"] = str(artifact_path)
    audio = output_summary.pop("_audio_f32", None)
    if audio is not None:
        import soundfile as sf

        artifact_path = arguments.output.with_suffix(".audio.wav").resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(artifact_path, audio, output_summary["sample_rate"], subtype="FLOAT")
        output_summary["audio_artifact"] = str(artifact_path)
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
        "compile_scope": "model.forward" if arguments.mode == "torch-compile" else None,
        "compile_evidence": compile_evidence,
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
            "compile_excluded": True,
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
    if compile_evidence is not None:
        compile_evidence["warmup_completed"] = True
        compile_evidence["timed_callable_uses_compiled_target"] = True
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
