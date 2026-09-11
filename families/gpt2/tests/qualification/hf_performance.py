#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-2 HF performance reference; imports no TRTMC builder dependencies."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
    parser.add_argument("--task", required=True, choices=("causal-lm",))
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--precision", required=True, choices=("fp16", "fp32", "bf16"))
    parser.add_argument("--max-length", required=True, type=int)
    parser.add_argument("--padding", default="longest", choices=("longest",))
    parser.add_argument("--mode", required=True, choices=("torch-compile",))
    parser.add_argument("--compile-mode", default="default", choices=("default",))
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--compile-dynamic", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--output-token-policy", default="new-tokens", choices=("new-tokens",))
    parser.add_argument("--output", required=True, type=Path)
    parser.set_defaults(
        model_class="task", generation_method="generate", experts_implementation=None
    )
    return parser


def _load(arguments: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from transformers import AutoTokenizer, GPT2LMHeadModel

    common = {"local_files_only": arguments.local_files_only}
    if arguments.revision:
        common["revision"] = arguments.revision
    tokenizer = AutoTokenizer.from_pretrained(arguments.model, **common)
    tokenizer.pad_token = tokenizer.eos_token
    model = (
        GPT2LMHeadModel.from_pretrained(
            arguments.model, torch_dtype=_dtype(torch, arguments.precision), **common
        )
        .eval()
        .to("cuda")
    )
    return tokenizer, model


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


def _compile(model: Any, arguments: argparse.Namespace) -> dict[str, Any] | None:
    if arguments.mode != "torch-compile":
        return None
    import torch
    from torch._dynamo.backends.registry import lookup_backend

    original_forward = model.forward
    evidence = {"compiled_graph_count": 0}
    inductor = lookup_backend("inductor")

    def compile_graph(graph, inputs, **options):
        compiled = inductor(graph, inputs, **options)
        evidence["compiled_graph_count"] += 1
        return compiled

    compiled_forward = torch.compile(
        original_forward,
        backend=compile_graph,
        fullgraph=arguments.compile_fullgraph,
        dynamic=arguments.compile_dynamic,
    )
    model.forward = compiled_forward
    evidence.update(
        {
            "api": "torch.compile",
            "target": "model.forward",
            "backend": "inductor",
            "mode": arguments.compile_mode,
            "fullgraph": arguments.compile_fullgraph,
            "dynamic": arguments.compile_dynamic,
            "applied": True,
        }
    )
    return evidence


def _batch_prompt(request: Mapping[str, Any]) -> list[str]:
    prompt = request.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError("Transformers baseline requires request.prompt")
    batch_size = request.get("batch_size", 1)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("request.batch_size must be a positive integer")
    return [prompt] * batch_size


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
            return _chat_prompt_inputs(
                tokenizer,
                prompt,
                enable_thinking=bool(request.get("enable_thinking", True)),
            )
        return tokenizer(prompt, return_tensors="pt")

    do_sample = request.get("do_sample")
    if do_sample is None:
        do_sample = float(request.get("temperature", 0.0)) > 0.0
    if not isinstance(do_sample, bool):
        raise ValueError("request.do_sample must be a boolean")
    generation: dict[str, Any] = {
        "do_sample": do_sample,
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
            generated = model.generate(**gpu_inputs, **generation)
        input_length = int(gpu_inputs["input_ids"].shape[-1])
        token_ids = generated[0, input_length:].detach().to("cpu", dtype=torch.int64).tolist()
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


def _chat_prompt_inputs(tokenizer: Any, prompt: str, *, enable_thinking: bool) -> Any:
    messages = [{"role": "user", "content": prompt}]
    if enable_thinking:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=True,
        )
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    if rendered.endswith("<think>\n"):
        rendered = rendered[: -len("<think>\n")] + "<think></think>"
    return tokenizer(rendered, return_tensors="pt", add_special_tokens=False)


def _measure(
    invoke: Callable[[], Any],
    summarize: Callable[[Any], dict[str, Any]],
    warmup: int,
    iterations: int,
    compile_evidence: Mapping[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    import torch

    if warmup <= 0 or iterations <= 0:
        raise ValueError("warmup and iterations must be positive")
    last = None
    for _ in range(warmup):
        last = invoke()
    torch.cuda.synchronize()
    compiled_graphs = compile_evidence["compiled_graph_count"]
    if compiled_graphs < 1:
        raise RuntimeError("warmup did not execute any compiled graphs")
    samples: list[float] = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        last = invoke()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    if last is None or not all(math.isfinite(value) and value > 0 for value in samples):
        raise RuntimeError("baseline produced no finite positive timing observations")
    if compile_evidence["compiled_graph_count"] != compiled_graphs:
        raise RuntimeError("model compilation occurred inside timed samples")
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
        "python_executable": os.sys.executable,
        "gpu_uuid": str(torch_module.cuda.get_device_properties(device).uuid),
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
    tokenizer, model = _load(arguments)
    compile_evidence = _compile(model, arguments)
    invoke, summarize = _generation_call(
        tokenizer,
        model,
        request,
        arguments.task,
        arguments.output_token_policy,
        arguments.precision,
        arguments.generation_method,
    )
    samples, output_summary = _measure(
        invoke, summarize, arguments.warmup, arguments.iterations, compile_evidence
    )
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
        "revision": (
            getattr(getattr(model, "config", None), "_commit_hash", None) or arguments.revision
        ),
        "case_name": arguments.case_name,
        "task": arguments.task,
        "precision": arguments.precision,
        "padding": arguments.padding,
        "experts_implementation": arguments.experts_implementation,
        "output_token_policy": arguments.output_token_policy,
        "measurement_policy": {
            "timing_scope": "public_operation_call_wall",
            "input_preparation_included": True,
            "asset_loading_included": False,
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
    }
    _write_json(arguments.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
