#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run text generation directly through the upstream Transformers API."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "trtmc.native-reference-reproduction/v1"


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)
    return rows


def _generation_value(
    arguments: argparse.Namespace,
    generation: Mapping[str, Any],
    name: str,
    default: Any,
) -> Any:
    configured = getattr(arguments, name)
    return generation.get(name, default) if configured is None else configured


def _model_dtype(torch_module: Any, name: str) -> str | Any:
    if name == "float16":
        return torch_module.float16
    if name == "bfloat16":
        return torch_module.bfloat16
    if name == "float32":
        return torch_module.float32
    return "auto"


def _load_model(
    transformers_module: Any,
    model_id: str,
    *,
    model_kwargs: dict[str, Any],
    model_revision: str,
    trust_remote_code: bool,
    local_files_only: bool,
) -> tuple[Any, bool]:
    config_kwargs = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if model_revision:
        config_kwargs["revision"] = model_revision
    config = transformers_module.AutoConfig.from_pretrained(model_id, **config_kwargs)
    is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
    model_class = (
        transformers_module.AutoModelForSeq2SeqLM
        if is_encoder_decoder
        else transformers_module.AutoModelForCausalLM
    )
    return model_class.from_pretrained(model_id, **model_kwargs).eval(), is_encoder_decoder


def _request_prompt(request: Mapping[str, Any]) -> str:
    messages = request.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if (
                isinstance(message, dict)
                and message.get("role") == "user"
                and isinstance(message.get("content"), str)
            ):
                return str(message["content"])
    prompt = request.get("prompt")
    if isinstance(prompt, str):
        return prompt
    return ""


def _selected_indexes(
    prompts: Sequence[Mapping[str, Any]],
    sample_id: str,
) -> list[int]:
    if not sample_id:
        return list(range(len(prompts)))
    selected = [
        index
        for index, prompt in enumerate(prompts)
        if str(prompt.get("sample_id", "")) == sample_id
    ]
    if not selected:
        raise ValueError(f"sample_id {sample_id!r} is not present in the prepared prompts")
    return selected


def _entrypoint_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _reproduction_command(arguments: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        arguments.model,
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
    if arguments.reference_family:
        command.extend(["--reference-family", arguments.reference_family])
    for flag, value in (
        ("--model-revision", arguments.model_revision),
        ("--device-map", arguments.device_map),
        ("--attn-impl", arguments.attn_impl),
        ("--experts-implementation", arguments.experts_implementation),
        ("--max-new-tokens", arguments.max_new_tokens),
        ("--temperature", arguments.temperature),
        ("--top-k", arguments.top_k),
        ("--top-p", arguments.top_p),
        ("--seed", arguments.seed),
    ):
        if value not in (None, ""):
            command.extend([flag, str(value)])
    for enabled, flag in (
        (arguments.trust_remote_code, "--trust-remote-code"),
        (arguments.local_files_only, "--local-files-only"),
        (arguments.do_sample, "--do-sample"),
        (arguments.apply_chat_template, "--apply-chat-template"),
    ):
        if enabled:
            command.append(flag)
    return command


def _write_reproduction_metadata(arguments: argparse.Namespace) -> None:
    if arguments.repro_metadata is None:
        return
    payload = {
        "schema_version": SCHEMA_VERSION,
        "backend": "transformers",
        "entrypoint": str(Path(__file__).resolve()),
        "entrypoint_sha256": _entrypoint_sha256(),
        "command": _reproduction_command(arguments),
    }
    arguments.repro_metadata.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _runtime_dependencies() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("native text reference requires torch and transformers") from exc
    return torch, transformers


def _generation_settings(
    arguments: argparse.Namespace,
    manifest: Mapping[str, Any],
    answers: Mapping[str, Any],
) -> dict[str, Any]:
    generation = manifest.get("generation", {})
    generation = generation if isinstance(generation, dict) else {}
    task_config = manifest.get("task_eval", {})
    task_config = task_config if isinstance(task_config, dict) else {}
    settings = {
        "max_new_tokens": int(
            _generation_value(arguments, generation, "max_new_tokens", 1)
        ),
        "temperature": float(
            _generation_value(arguments, generation, "temperature", 1.0)
        ),
        "top_k": int(_generation_value(arguments, generation, "top_k", 1)),
        "top_p": float(_generation_value(arguments, generation, "top_p", 1.0)),
        "seed": int(_generation_value(arguments, generation, "seed", -1)),
        "do_sample": arguments.do_sample
        or bool(generation.get("do_sample", False)),
        "apply_chat_template": arguments.apply_chat_template
        or bool(
            generation.get(
                "apply_chat_template",
                answers.get("apply_chat_template", False),
            )
        ),
        "generation_overrides": {},
    }
    if "hf_use_cache" in task_config:
        settings["generation_overrides"]["use_cache"] = bool(
            task_config["hf_use_cache"]
        )
    hf_generation_overrides = task_config.get("hf_generation_overrides", {})
    if not isinstance(hf_generation_overrides, dict):
        raise ValueError("task_eval.hf_generation_overrides must be a mapping")
    settings["generation_overrides"].update(hf_generation_overrides)
    return settings


def _load_runtime(
    arguments: argparse.Namespace,
    torch_module: Any,
    transformers_module: Any,
) -> tuple[Any, Any, Any, bool]:
    transformers_module.logging.set_verbosity_error()
    tokenizer_kwargs = {
        "trust_remote_code": arguments.trust_remote_code,
        "local_files_only": arguments.local_files_only,
    }
    if arguments.model_revision:
        tokenizer_kwargs["revision"] = arguments.model_revision
    tokenizer = transformers_module.AutoTokenizer.from_pretrained(
        arguments.model,
        **tokenizer_kwargs,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "torch_dtype": _model_dtype(torch_module, arguments.dtype),
        "trust_remote_code": arguments.trust_remote_code,
        "local_files_only": arguments.local_files_only,
    }
    if arguments.device_map:
        model_kwargs["device_map"] = arguments.device_map
    if arguments.attn_impl:
        model_kwargs["attn_implementation"] = arguments.attn_impl
    if arguments.experts_implementation:
        model_kwargs["experts_implementation"] = arguments.experts_implementation
    if arguments.model_revision:
        model_kwargs["revision"] = arguments.model_revision
    model, is_encoder_decoder = _load_model(
        transformers_module,
        arguments.model,
        model_kwargs=model_kwargs,
        model_revision=arguments.model_revision,
        trust_remote_code=arguments.trust_remote_code,
        local_files_only=arguments.local_files_only,
    )
    if arguments.device_map:
        device = model.device
    else:
        device = torch_module.device(arguments.device)
        model.to(device)
    return tokenizer, model, device, is_encoder_decoder


def _seed_runtime(torch_module: Any, seed: int, source_index: int) -> None:
    if seed < 0:
        return
    torch_module.manual_seed(seed + source_index)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed + source_index)


def _generated_sequence(
    output_ids: Any,
    *,
    input_length: int,
    model: Any,
    is_encoder_decoder: bool,
) -> tuple[Any, bool]:
    if not is_encoder_decoder:
        return output_ids[0, input_length:], False
    generated = output_ids[0]
    decoder_start_token_id = getattr(
        model.config,
        "decoder_start_token_id",
        None,
    )
    if (
        decoder_start_token_id is not None
        and generated.numel() > 0
        and int(generated[0]) == int(decoder_start_token_id)
    ):
        generated = generated[1:]
    return generated, True


def _generate_sample(
    *,
    torch_module: Any,
    tokenizer: Any,
    model: Any,
    device: Any,
    is_encoder_decoder: bool,
    request: Mapping[str, Any],
    prompt_row: Mapping[str, Any],
    source_index: int,
    settings: Mapping[str, Any],
) -> tuple[str, Any, float]:
    prompt = str(prompt_row.get("prompt") or _request_prompt(request))
    if settings["apply_chat_template"]:
        prompt = tokenizer.apply_chat_template(
            request["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {name: value.to(device) for name, value in encoded.items()}
    _seed_runtime(torch_module, int(settings["seed"]), source_index)
    start = time.perf_counter()
    with torch_module.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=settings["max_new_tokens"],
            do_sample=settings["do_sample"],
            temperature=settings["temperature"],
            top_k=settings["top_k"],
            top_p=settings["top_p"],
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **settings["generation_overrides"],
        )
    wall_ms = (time.perf_counter() - start) * 1000.0
    generated, skip_special_tokens = _generated_sequence(
        output_ids,
        input_length=encoded["input_ids"].shape[1],
        model=model,
        is_encoder_decoder=is_encoder_decoder,
    )
    return (
        tokenizer.decode(generated, skip_special_tokens=skip_special_tokens),
        generated,
        wall_ms,
    )


def run(arguments: argparse.Namespace) -> None:
    torch, transformers = _runtime_dependencies()

    manifest = _load_json(arguments.manifest)
    answers = _load_json(arguments.answers)
    prompt_rows = _load_jsonl(arguments.prompts)
    requests = answers.get("requests", [])
    if not isinstance(requests, list) or len(requests) != len(prompt_rows):
        raise ValueError("answers and prepared prompts must contain the same number of samples")
    selected = _selected_indexes(prompt_rows, arguments.sample_id)
    settings = _generation_settings(arguments, manifest, answers)
    tokenizer, model, device, is_encoder_decoder = _load_runtime(
        arguments,
        torch,
        transformers,
    )
    responses = []
    arguments.raw_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.predictions.parent.mkdir(parents=True, exist_ok=True)
    with arguments.raw_output.open("w", encoding="utf-8") as raw_file:
        for run_index, source_index in enumerate(selected):
            prompt_row = prompt_rows[source_index]
            request = requests[source_index]
            if not isinstance(request, dict):
                raise ValueError(f"request {source_index} must be a JSON object")
            output_text, generated, wall_ms = _generate_sample(
                torch_module=torch,
                tokenizer=tokenizer,
                model=model,
                device=device,
                is_encoder_decoder=is_encoder_decoder,
                request=request,
                prompt_row=prompt_row,
                source_index=source_index,
                settings=settings,
            )
            row = {
                "sample_id": prompt_row.get("sample_id", f"sample_{source_index:06d}"),
                "output_text": output_text,
                "generated_tokens": int(generated.shape[0]),
                "generated_token_ids": [
                    int(token_id) for token_id in generated.tolist()
                ],
                "wall_ms": wall_ms,
                "source": "hf",
            }
            responses.append(row)
            raw_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_file.flush()
            print(
                f"[transformers.native_reference] sample={run_index + 1}/{len(selected)}",
                file=sys.stderr,
            )
    arguments.predictions.write_text(
        json.dumps({"responses": responses}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_reproduction_metadata(arguments)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a text model directly through Transformers."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default="")
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
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="")
    parser.add_argument("--attn-impl", default="")
    parser.add_argument("--experts-implementation", default="")
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
