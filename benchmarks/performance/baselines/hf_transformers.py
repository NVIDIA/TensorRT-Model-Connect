#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark one Transformers model at the same public task boundary as trtmc-bench."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Callable, Mapping, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument(
        "--task",
        required=True,
        choices=("encoder", "causal-lm", "seq2seq-lm"),
    )
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--precision", required=True, choices=("fp16", "fp32", "bf16"))
    parser.add_argument("--max-length", required=True, type=int)
    parser.add_argument("--padding", default="longest", choices=("longest", "max-length"))
    parser.add_argument("--mode", required=True, choices=("torch-compile", "hf-eager"))
    parser.add_argument(
        "--model-class",
        default="task",
        choices=("task", "auto"),
        help="Use the task AutoClass or the model repository's generic AutoModel registration.",
    )
    parser.add_argument(
        "--generation-method",
        default="generate",
        choices=("generate", "ar-generate"),
        help="Public generation method to time after loading the model.",
    )
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--compile-dynamic", action="store_true")
    parser.add_argument(
        "--experts-implementation",
        choices=("eager", "batched_mm", "grouped_mm"),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--workload-digest", required=True)
    parser.add_argument(
        "--output-token-policy",
        default="new-tokens",
        choices=("new-tokens", "strip-start", "strip-start-and-eos"),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _request(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--request-json is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("--request-json must contain an object")
    return value


def _dtype(torch_module: Any, precision: str) -> Any:
    return {
        "fp16": torch_module.float16,
        "fp32": torch_module.float32,
        "bf16": torch_module.bfloat16,
    }[precision]


def _ensure_dynamic_cache_api(dynamic_cache: type[Any]) -> None:
    """Bridge the Transformers DynamicCache method rename for remote model code."""
    if not hasattr(dynamic_cache, "get_max_cache_shape") and hasattr(
        dynamic_cache, "get_max_length"
    ):
        setattr(dynamic_cache, "get_max_cache_shape", dynamic_cache.get_max_length)


def _load(arguments: argparse.Namespace) -> tuple[Any, Any, str]:
    import torch
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )
    from transformers.cache_utils import DynamicCache

    _ensure_dynamic_cache_api(DynamicCache)

    common: dict[str, Any] = {
        "trust_remote_code": arguments.trust_remote_code,
        "local_files_only": arguments.local_files_only,
    }
    if arguments.revision:
        common["revision"] = arguments.revision
    tokenizer = AutoTokenizer.from_pretrained(arguments.model, **common)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model_class = AutoModel
    if arguments.model_class == "task":
        model_class = {
            "encoder": AutoModel,
            "causal-lm": AutoModelForCausalLM,
            "seq2seq-lm": AutoModelForSeq2SeqLM,
        }[arguments.task]
    model_options = {
        "torch_dtype": _dtype(torch, arguments.precision),
        "low_cpu_mem_usage": True,
        **common,
    }
    if arguments.experts_implementation:
        model_options["experts_implementation"] = arguments.experts_implementation
    model = model_class.from_pretrained(arguments.model, **model_options)
    model.eval().to("cuda")
    resolved_revision = str(
        getattr(model.config, "_commit_hash", None) or arguments.revision or "unresolved"
    )
    return tokenizer, model, resolved_revision


def _compile(model: Any, arguments: argparse.Namespace) -> dict[str, Any] | None:
    if arguments.mode != "torch-compile":
        return None
    import torch

    original_forward = model.forward
    compiled_forward = torch.compile(
        original_forward,
        mode=arguments.compile_mode,
        fullgraph=arguments.compile_fullgraph,
        dynamic=arguments.compile_dynamic,
    )
    model.forward = compiled_forward
    return {
        "api": "torch.compile",
        "target": "model.forward",
        "backend": "inductor",
        "mode": arguments.compile_mode,
        "fullgraph": arguments.compile_fullgraph,
        "dynamic": arguments.compile_dynamic,
        "applied": True,
    }


def _batch_prompt(request: Mapping[str, Any]) -> list[str]:
    prompt = request.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError("Transformers baseline requires request.prompt")
    batch_size = request.get("batch_size", 1)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("request.batch_size must be a positive integer")
    return [prompt] * batch_size


def _encoder_call(
    tokenizer: Any,
    model: Any,
    request: Mapping[str, Any],
    max_length: int,
    padding: str,
    precision: str,
) -> tuple[Callable[[], Any], Callable[[Any], dict[str, Any]]]:
    import torch

    prompts = _batch_prompt(request)

    def invoke() -> Any:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length" if padding == "max-length" else True,
            truncation=True,
            max_length=max_length,
        )
        gpu_inputs = {name: value.to("cuda") for name, value in encoded.items()}
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type="cuda",
                dtype=_dtype(torch, precision),
                enabled=precision != "fp32",
            ),
        ):
            outputs = model(**gpu_inputs)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = getattr(outputs, "pooler_output", None)
        if hidden is None:
            hidden = outputs[0]
        if hidden.ndim == 3:
            hidden = hidden[:, 0, :]
        if hidden.ndim != 2:
            raise RuntimeError(f"encoder returned unsupported shape {tuple(hidden.shape)}")
        return hidden.detach().to("cpu", dtype=torch.float32)

    def summarize(hidden: Any) -> dict[str, Any]:
        shape = [int(value) for value in hidden.shape]
        return {
            "shape": shape,
            "embedding_vectors": len(prompts),
            "embedding_elements": int(hidden.numel()),
            "dim": int(shape[-1]),
            "finite": bool(torch.isfinite(hidden).all().item()),
        }

    return invoke, summarize


def _generation_call(
    tokenizer: Any,
    model: Any,
    request: Mapping[str, Any],
    task: str,
    output_token_policy: str,
    precision: str,
    generation_method: str,
) -> tuple[Callable[[], Any], Callable[[Any], dict[str, Any]]]:
    import torch

    prompts = _batch_prompt(request)
    if len(prompts) != 1:
        raise ValueError("the release generation baseline currently requires batch_size=1")
    prompt = prompts[0]
    use_chat_template = bool(request.get("use_chat_template", False))
    max_new_tokens = int(request.get("max_new_tokens", 20))
    if max_new_tokens <= 0:
        raise ValueError("request.max_new_tokens must be positive")

    def encode_prompt() -> Any:
        if use_chat_template:
            if not getattr(tokenizer, "chat_template", None):
                raise ValueError("request requires a chat template but tokenizer has none")
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=bool(request.get("enable_thinking", True)),
            )
        return tokenizer(prompt, return_tensors="pt")

    generation: dict[str, Any] = {
        "do_sample": float(request.get("temperature", 0.0)) > 0.0,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if generation["do_sample"]:
        generation.update(
            {
                "temperature": float(request.get("temperature", 1.0)),
                "top_k": int(request.get("top_k", 0)),
                "top_p": float(request.get("top_p", 1.0)),
            }
        )
        seed = int(request.get("seed", -1))
        if seed >= 0:
            torch.manual_seed(seed)

    def invoke() -> dict[str, Any]:
        encoded = encode_prompt()
        gpu_inputs = {name: value.to("cuda") for name, value in encoded.items()}
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type="cuda",
                dtype=_dtype(torch, precision),
                enabled=precision != "fp32",
            ),
        ):
            if generation_method == "ar-generate":
                generated = model.ar_generate(
                    gpu_inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    temperature=float(request.get("temperature", 0.0)),
                    eos_token_id=tokenizer.eos_token_id,
                )
                if isinstance(generated, tuple):
                    generated = generated[0]
            else:
                generated = model.generate(**gpu_inputs, **generation)
        input_length = 0 if task == "seq2seq-lm" else int(gpu_inputs["input_ids"].shape[-1])
        token_ids = generated[0, input_length:].detach().to("cpu", dtype=torch.int64).tolist()
        if task == "seq2seq-lm":
            token_ids = _normalize_seq2seq_tokens(
                token_ids,
                getattr(model.config, "decoder_start_token_id", None),
                getattr(model.config, "eos_token_id", None),
                output_token_policy,
            )
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        return {"token_ids": token_ids, "text": text}

    def summarize(output: Mapping[str, Any]) -> dict[str, Any]:
        token_ids = [int(value) for value in output["token_ids"]]
        return {
            "token_ids": token_ids,
            "output_tokens": len(token_ids),
            "text": str(output["text"]),
        }

    return invoke, summarize


def _normalize_seq2seq_tokens(
    token_ids: Sequence[int],
    decoder_start_token_id: int | None,
    eos_token_id: int | Sequence[int] | None,
    policy: str,
) -> list[int]:
    normalized = [int(value) for value in token_ids]
    if policy == "new-tokens":
        return normalized
    normalized = _strip_first_token(normalized, decoder_start_token_id)
    if policy != "strip-start-and-eos" or not normalized or eos_token_id is None:
        return normalized
    eos_ids = _token_id_set(eos_token_id)
    if normalized[-1] in eos_ids:
        normalized.pop()
    return normalized


def _strip_first_token(token_ids: list[int], expected: int | None) -> list[int]:
    if token_ids and expected is not None and token_ids[0] == int(expected):
        return token_ids[1:]
    return token_ids


def _token_id_set(value: int | Sequence[int]) -> set[int]:
    if isinstance(value, int):
        return {int(value)}
    return {int(item) for item in value}


def _measure(
    invoke: Callable[[], Any],
    summarize: Callable[[Any], dict[str, Any]],
    warmup: int,
    iterations: int,
) -> tuple[list[float], dict[str, Any]]:
    import torch

    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    last = None
    for _ in range(warmup):
        last = invoke()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        last = invoke()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    if last is None or not all(math.isfinite(value) and value > 0 for value in samples):
        raise RuntimeError("baseline produced no finite positive timing observations")
    return samples, summarize(last)


def _summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": max(values),
    }


def _environment(torch_module: Any, transformers_module: Any) -> dict[str, Any]:
    device = torch_module.cuda.current_device()
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "python_executable": os.path.realpath(os.sys.executable),
        "torch": torch_module.__version__,
        "transformers": transformers_module.__version__,
        "cuda": torch_module.version.cuda,
        "gpu": torch_module.cuda.get_device_name(device),
        "gpu_capability": list(torch_module.cuda.get_device_capability(device)),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the performance baseline")
    request = _request(arguments.request_json)
    tokenizer, model, resolved_revision = _load(arguments)
    compile_evidence = _compile(model, arguments)
    if arguments.task == "encoder":
        invoke, summarize = _encoder_call(
            tokenizer,
            model,
            request,
            arguments.max_length,
            arguments.padding,
            arguments.precision,
        )
    else:
        invoke, summarize = _generation_call(
            tokenizer,
            model,
            request,
            arguments.task,
            arguments.output_token_policy,
            arguments.precision,
            arguments.generation_method,
        )
    samples, output_summary = _measure(invoke, summarize, arguments.warmup, arguments.iterations)
    if compile_evidence is not None:
        compile_evidence["warmup_completed"] = True
        compile_evidence["timed_callable_uses_compiled_target"] = True
    result = {
        "schema_version": "trtmc.perf-baseline/v1",
        "status": "completed",
        "backend": "hf-transformers",
        "mode": arguments.mode,
        "model_class": arguments.model_class,
        "generation_method": arguments.generation_method,
        "compile_scope": "model.forward" if arguments.mode == "torch-compile" else None,
        "compile_evidence": compile_evidence,
        "model": arguments.model,
        "requested_revision": arguments.revision,
        "resolved_revision": resolved_revision,
        "task": arguments.task,
        "precision": arguments.precision,
        "padding": arguments.padding,
        "experts_implementation": arguments.experts_implementation,
        "workload_digest": arguments.workload_digest,
        "output_token_policy": arguments.output_token_policy,
        "measurement_policy": {
            "timing_scope": "public_operation_call_wall",
            "model_load_excluded": True,
            "compile_excluded": True,
            "warmup_excluded": True,
            "tokenization_included": True,
            "device_transfers_included": True,
            "output_materialization_included": True,
            "autocast_enabled": arguments.precision != "fp32",
        },
        "samples_ms": samples,
        "metrics": {"latency_ms": _summary(samples)},
        "output_summary": output_summary,
        "environment": _environment(torch, transformers),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "request_sha256": hashlib.sha256(
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    _write_json(arguments.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
