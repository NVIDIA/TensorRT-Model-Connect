#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run encoder and embedding references directly through Transformers."""

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


def _model_dtype(torch_module: Any, name: str) -> str | Any:
    if name == "float16":
        return torch_module.float16
    if name == "bfloat16":
        return torch_module.bfloat16
    if name == "float32":
        return torch_module.float32
    return "auto"


def _reference_classes(
    transformers_module: Any,
    reference_family: str,
) -> tuple[Any, Any]:
    if reference_family == "dpr_context_embed":
        return (
            transformers_module.DPRContextEncoder,
            transformers_module.DPRContextEncoderTokenizerFast,
        )
    return transformers_module.AutoModel, transformers_module.AutoTokenizer


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
    ):
        if value:
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
        raise RuntimeError(
            "native encoder reference requires torch and transformers"
        ) from exc
    return torch, transformers


def _last_hidden_state(outputs: Any) -> Any:
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states:
        return hidden_states[-1]
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    return None


def _result_vector(
    torch_module: Any,
    hidden: Any,
    encoded: Mapping[str, Any],
    vector_mode: str,
) -> list[float]:
    if vector_mode != "embedding":
        return hidden[0, 0].float().cpu().numpy().tolist()
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch_module.ones(
            hidden.shape[:2],
            device=hidden.device,
        )
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    vector = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    vector = torch_module.nn.functional.normalize(vector, p=2, dim=-1)[0]
    return vector.float().cpu().numpy().tolist()


def _load_runtime(
    arguments: argparse.Namespace,
    torch_module: Any,
    transformers_module: Any,
) -> tuple[Any, Any, Any]:
    model_class, tokenizer_class = _reference_classes(
        transformers_module,
        arguments.reference_family,
    )
    transformers_module.logging.set_verbosity_error()
    tokenizer_kwargs = {
        "trust_remote_code": arguments.trust_remote_code,
        "local_files_only": arguments.local_files_only,
    }
    if arguments.model_revision:
        tokenizer_kwargs["revision"] = arguments.model_revision
    tokenizer = tokenizer_class.from_pretrained(
        arguments.model,
        **tokenizer_kwargs,
    )
    model_kwargs = {
        "torch_dtype": _model_dtype(torch_module, arguments.dtype),
        "trust_remote_code": arguments.trust_remote_code,
        "local_files_only": arguments.local_files_only,
    }
    if arguments.device_map:
        model_kwargs["device_map"] = arguments.device_map
    if arguments.model_revision:
        model_kwargs["revision"] = arguments.model_revision
    model = model_class.from_pretrained(arguments.model, **model_kwargs).eval()
    requested_dtype = _model_dtype(torch_module, arguments.dtype)
    model_type = str(getattr(getattr(model, "config", None), "model_type", ""))
    if model_type == "xlnet" and requested_dtype != "auto":
        # Transformers leaves XLNet's directly registered relative-attention
        # parameters (q/k/v/r and biases) in FP32 even when from_pretrained()
        # receives a reduced dtype.  Cast the complete model explicitly so
        # those tensors match the FP16/BF16 hidden states used by einsum.
        model.to(dtype=requested_dtype)
    if arguments.device_map:
        device = model.device
    else:
        device = torch_module.device(arguments.device)
        model.to(device)
    return tokenizer, model, device


def run(arguments: argparse.Namespace) -> None:
    torch, transformers = _runtime_dependencies()
    manifest = _load_json(arguments.manifest)
    prompt_rows = _selected_rows(
        _load_jsonl(arguments.prompts),
        arguments.sample_id,
    )
    task_config = manifest.get("task_eval", {})
    task_config = task_config if isinstance(task_config, dict) else {}
    vector_mode = (
        "embedding"
        if task_config.get("task_strategy") == "embedding"
        else "cls"
    )
    tokenizer, model, device = _load_runtime(
        arguments,
        torch,
        transformers,
    )
    responses = []
    arguments.raw_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.predictions.parent.mkdir(parents=True, exist_ok=True)
    with arguments.raw_output.open("w", encoding="utf-8") as raw_file:
        for index, prompt_row in enumerate(prompt_rows):
            encoded = tokenizer(
                str(prompt_row["prompt"]),
                return_tensors="pt",
                truncation=True,
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            start = time.perf_counter()
            with torch.inference_mode():
                outputs = model(**encoded, output_hidden_states=True)
            wall_ms = (time.perf_counter() - start) * 1000.0
            hidden = _last_hidden_state(outputs)
            if hidden is None or hidden.ndim != 3:
                raise RuntimeError(
                    f"encoder output for {prompt_row['sample_id']} "
                    "has no rank-3 hidden state"
                )
            row = {
                "sample_id": str(prompt_row["sample_id"]),
                "pair_id": str(prompt_row["pair_id"]),
                "pair_side": str(prompt_row["pair_side"]),
                "score": float(prompt_row["score"]),
                "vector_mode": vector_mode,
                "vector": _result_vector(
                    torch,
                    hidden,
                    encoded,
                    vector_mode,
                ),
                "wall_ms": wall_ms,
                "source": "hf",
            }
            responses.append(row)
            raw_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_file.flush()
            print(
                f"[transformers.native_encoder] "
                f"sample={index + 1}/{len(prompt_rows)}",
                file=sys.stderr,
            )
    arguments.predictions.write_text(
        json.dumps({"responses": responses}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_reproduction_metadata(arguments)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an encoder model directly through Transformers."
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
