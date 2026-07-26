#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run vision-language references directly through Transformers."""

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


def _selected_indices(
    rows: Sequence[Mapping[str, Any]],
    sample_id: str,
) -> list[int]:
    selected = [
        index
        for index, row in enumerate(rows)
        if not sample_id or str(row.get("sample_id", "")) == sample_id
    ]
    if sample_id and not selected:
        raise ValueError(f"sample_id {sample_id!r} is not present in the prepared prompts")
    return selected


def _model_dtype(torch_module: Any, name: str) -> str | Any:
    if name == "float16":
        return torch_module.float16
    if name == "bfloat16":
        return torch_module.bfloat16
    return "auto"


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
    for flag, value in (
        ("--model-revision", arguments.model_revision),
        ("--reference-family", arguments.reference_family),
        ("--device-map", arguments.device_map),
        ("--attn-impl", arguments.attn_impl),
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
                "backend": "transformers_vlm",
                "entrypoint": str(Path(__file__).resolve()),
                "entrypoint_sha256": _entrypoint_sha256(),
                "command": _reproduction_command(arguments),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _runtime_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        import transformers
        from transformers import AutoProcessor
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "native VLM reference requires torch and transformers"
        ) from exc
    return torch, transformers, AutoProcessor


def _model_classes(transformers_module: Any) -> list[Any]:
    classes = [
        getattr(transformers_module, name, None)
        for name in (
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
            "AutoModel",
        )
    ]
    available = [model_class for model_class in classes if model_class is not None]
    if not available:
        raise RuntimeError(
            "Transformers installation does not expose a VLM-capable AutoModel class"
        )
    return available


def _load_model(
    transformers_module: Any,
    model_id: str,
    model_kwargs: Mapping[str, Any],
) -> Any:
    errors = []
    for model_class in _model_classes(transformers_module):
        try:
            return model_class.from_pretrained(
                model_id,
                **dict(model_kwargs),
            ).eval()
        except ValueError as exc:
            errors.append(f"{model_class.__name__}: {exc}")
    raise RuntimeError(
        f"Could not load VLM reference model {model_id!r}: "
        + " | ".join(errors)
    )


def _load_images(image_paths: Sequence[str]) -> list[Any]:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("native VLM reference requires Pillow") from exc
    images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    return images


def _has_image_placeholder(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "<|image_pad|>",
            "<|vision_start|>",
            "<image>",
            "<IMG_CONTEXT>",
        )
    )


def _apply_chat_template(obj: Any, messages: Sequence[Any]) -> str:
    if not hasattr(obj, "apply_chat_template"):
        return ""
    try:
        return str(
            obj.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    except ValueError as exc:
        if "chat_template" in str(exc):
            return ""
        raise


def _chat_text(
    processor: Any,
    request: Mapping[str, Any],
    fallback_prompt: str,
    fallback_template: str,
) -> str:
    messages = request.get("messages")
    messages = messages if isinstance(messages, list) else []
    rendered = _apply_chat_template(processor, messages)
    tokenizer = getattr(processor, "tokenizer", None)
    if not rendered and tokenizer is not None:
        rendered = _apply_chat_template(tokenizer, messages)
    if rendered or _has_image_placeholder(fallback_prompt):
        return rendered or fallback_prompt
    return (
        fallback_template.replace("{prompt}", fallback_prompt)
        if fallback_template
        else fallback_prompt
    )


def _to_device(batch: Any, device: Any) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _generation_settings(
    arguments: argparse.Namespace,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    defaults = manifest.get("generation", {})
    defaults = defaults if isinstance(defaults, dict) else {}
    return {
        "max_new_tokens": (
            arguments.max_new_tokens
            if arguments.max_new_tokens is not None
            else int(defaults.get("max_new_tokens", 8))
        ),
        "temperature": (
            arguments.temperature
            if arguments.temperature is not None
            else float(defaults.get("temperature", 1.0))
        ),
        "top_k": (
            arguments.top_k
            if arguments.top_k is not None
            else int(defaults.get("top_k", 1))
        ),
        "top_p": (
            arguments.top_p
            if arguments.top_p is not None
            else float(defaults.get("top_p", 1.0))
        ),
        "seed": (
            arguments.seed
            if arguments.seed is not None
            else int(defaults.get("seed", -1))
        ),
        "do_sample": arguments.do_sample
        or bool(defaults.get("do_sample", False)),
    }


def _load_runtime(
    arguments: argparse.Namespace,
    torch_module: Any,
    transformers_module: Any,
    processor_class: Any,
) -> tuple[Any, Any, Any]:
    transformers_module.logging.set_verbosity_error()
    processor = processor_class.from_pretrained(
        arguments.model,
        trust_remote_code=arguments.trust_remote_code,
        local_files_only=arguments.local_files_only,
        **(
            {"revision": arguments.model_revision}
            if arguments.model_revision
            else {}
        ),
    )
    model_kwargs = {
        "torch_dtype": _model_dtype(torch_module, arguments.dtype),
        "trust_remote_code": arguments.trust_remote_code,
        "local_files_only": arguments.local_files_only,
    }
    if arguments.device_map:
        model_kwargs["device_map"] = arguments.device_map
    if arguments.attn_impl:
        model_kwargs["attn_implementation"] = arguments.attn_impl
    if arguments.model_revision:
        model_kwargs["revision"] = arguments.model_revision
    model = _load_model(
        transformers_module,
        arguments.model,
        model_kwargs,
    )
    if arguments.device_map:
        device = model.device
    else:
        device = torch_module.device(arguments.device)
        model.to(device)
    return processor, model, device


def _is_deepseek_ocr(model_id: str, model: Any) -> bool:
    return "deepseek-ocr" in model_id.lower() and hasattr(model, "infer")


def _deepseek_prompt(prompt: str) -> str:
    return prompt if "<image>" in prompt else f"<image>\n{prompt}"


def _deepseek_response(
    *,
    model: Any,
    processor: Any,
    prompt_row: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    images = [str(path) for path in prompt_row.get("images", [])]
    if len(images) != 1:
        raise ValueError("DeepSeek-OCR reference expects exactly one image")
    sample_id = str(prompt_row.get("sample_id", "vlm"))
    start = time.perf_counter()
    output_text = model.infer(
        processor,
        prompt=_deepseek_prompt(str(prompt_row.get("prompt", ""))),
        image_file=images[0],
        output_path=str(output_root / sample_id),
        save_results=False,
        eval_mode=True,
    )
    wall_ms = (time.perf_counter() - start) * 1000.0
    output_text = "" if output_text is None else str(output_text)
    try:
        token_ids = [
            int(token_id)
            for token_id in processor(
                output_text,
                add_special_tokens=False,
            ).input_ids
        ]
    except Exception:
        token_ids = []
    return {
        "sample_id": sample_id,
        "output_text": output_text,
        "generated_tokens": len(token_ids),
        "generated_token_ids": token_ids,
        "wall_ms": wall_ms,
        "source": "hf",
    }


def _generate_kwargs(
    processor: Any,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if tokenizer is not None and pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.pad_token_id
    kwargs = {
        "max_new_tokens": settings["max_new_tokens"],
        "do_sample": settings["do_sample"],
        "temperature": settings["temperature"],
        "top_k": settings["top_k"],
        "top_p": settings["top_p"],
        "num_beams": 1,
    }
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    return kwargs


def _generated_text(processor: Any, generated: Any) -> str:
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(
            generated.unsqueeze(0),
            skip_special_tokens=False,
        )[0]
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer.decode(generated, skip_special_tokens=False)
    return str(generated.tolist())


def _vlm_response(
    *,
    torch_module: Any,
    processor: Any,
    model: Any,
    device: Any,
    request: Mapping[str, Any],
    prompt_row: Mapping[str, Any],
    source_index: int,
    settings: Mapping[str, Any],
    fallback_template: str,
) -> dict[str, Any]:
    image_paths = [str(path) for path in prompt_row.get("images", [])]
    if len(image_paths) != 1:
        raise ValueError("VLM reference expects exactly one image")
    prompt = _chat_text(
        processor,
        request,
        str(prompt_row.get("prompt", "")),
        fallback_template,
    )
    inputs = processor(
        text=[prompt],
        images=_load_images(image_paths),
        padding=True,
        return_tensors="pt",
    )
    inputs = _to_device(inputs, device)
    seed = int(settings["seed"])
    if seed >= 0:
        torch_module.manual_seed(seed + source_index)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed + source_index)
    start = time.perf_counter()
    with torch_module.inference_mode():
        output_ids = model.generate(
            **inputs,
            **_generate_kwargs(processor, settings),
        )
    wall_ms = (time.perf_counter() - start) * 1000.0
    generated = output_ids[0, int(inputs["input_ids"].shape[1]) :]
    return {
        "sample_id": prompt_row.get("sample_id", f"vlm_{source_index:06d}"),
        "output_text": _generated_text(processor, generated),
        "generated_tokens": int(generated.shape[0]),
        "generated_token_ids": [
            int(token_id) for token_id in generated.tolist()
        ],
        "wall_ms": wall_ms,
        "source": "hf",
    }


def run(arguments: argparse.Namespace) -> None:
    torch, transformers, processor_class = _runtime_dependencies()
    manifest = _load_json(arguments.manifest)
    answers = _load_json(arguments.answers)
    prompt_rows = _load_jsonl(arguments.prompts)
    requests = answers.get("requests", [])
    if not isinstance(requests, list) or len(requests) != len(prompt_rows):
        raise ValueError("answers and prepared prompts must contain the same samples")
    selected = _selected_indices(prompt_rows, arguments.sample_id)
    processor, model, device = _load_runtime(
        arguments,
        torch,
        transformers,
        processor_class,
    )
    settings = _generation_settings(arguments, manifest)
    task_config = manifest.get("task_eval", {})
    task_config = task_config if isinstance(task_config, dict) else {}
    fallback_template = str(
        task_config.get("vlm_fallback_prompt_template", "") or ""
    )
    responses = []
    for run_index, source_index in enumerate(selected):
        prompt_row = prompt_rows[source_index]
        request = requests[source_index]
        if not isinstance(request, dict):
            raise ValueError(f"request {source_index} must be a JSON object")
        if _is_deepseek_ocr(arguments.model, model):
            response = _deepseek_response(
                model=model,
                processor=processor,
                prompt_row=prompt_row,
                output_root=arguments.predictions.parent
                / "hf_deepseek_ocr_outputs",
            )
        else:
            response = _vlm_response(
                torch_module=torch,
                processor=processor,
                model=model,
                device=device,
                request=request,
                prompt_row=prompt_row,
                source_index=source_index,
                settings=settings,
                fallback_template=fallback_template,
            )
        responses.append(response)
        print(
            f"[transformers.native_vlm] sample={run_index + 1}/{len(selected)}",
            file=sys.stderr,
        )
    arguments.raw_output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.raw_output.open("w", encoding="utf-8") as raw_file:
        for response in responses:
            raw_file.write(json.dumps(response, ensure_ascii=False) + "\n")
    arguments.predictions.write_text(
        json.dumps({"responses": responses}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_reproduction_metadata(arguments)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a VLM directly through Transformers."
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
