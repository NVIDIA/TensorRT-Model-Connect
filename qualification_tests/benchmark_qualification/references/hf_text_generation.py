#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Produce deterministic Hugging Face text-generation qualification references."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    request = json.loads(arguments.request.read_text(encoding="utf-8"))

    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("the HF Accuracy reference requires CUDA")
    precision = str(request.get("precision", "fp32"))
    dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if precision not in dtypes:
        raise ValueError(f"unsupported reference precision {precision!r}")
    model_options = _model_load_options(request)
    tokenizer_options = {
        name: model_options[name]
        for name in ("local_files_only", "revision", "trust_remote_code")
        if name in model_options
    }
    tokenizer = AutoTokenizer.from_pretrained(str(request["model"]), **tokenizer_options)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    task = str(request.get("task", "causal-lm"))
    model_type = _model_class(task, AutoModelForCausalLM, AutoModelForSeq2SeqLM)
    model = model_type.from_pretrained(
        str(request["model"]), **_precision_load_options(dtypes[precision]), **model_options
    ).eval()
    model.to("cuda")
    generation = request.get("generation", {})
    if not isinstance(generation, Mapping):
        raise ValueError("generation must be an object")
    translation, source_language_token_id = _translation_controls(tokenizer, generation)
    results = []
    with torch.inference_mode():
        for sample in request["samples"]:
            prompt = _truncate(
                tokenizer,
                str(sample["prompt"]),
                int(request["prompt_token_limit"]),
                str(request.get("truncation_side", "left")),
            )
            encoded = tokenizer(prompt, return_tensors="pt")
            _apply_source_language(encoded, source_language_token_id, tokenizer)
            encoded = {name: value.to("cuda") for name, value in encoded.items()}
            options: dict[str, Any] = {
                "max_new_tokens": int(generation.get("max_new_tokens", 64)),
                "do_sample": bool(generation.get("do_sample", False)),
                "num_beams": 1,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "repetition_penalty": float(generation.get("repetition_penalty", 1.0)),
                **translation,
            }
            if options["do_sample"]:
                options.update(
                    temperature=float(generation.get("temperature", 1.0)),
                    top_k=int(generation.get("top_k", 0)),
                    top_p=float(generation.get("top_p", 1.0)),
                )
                seed = int(generation.get("seed", -1))
                if seed >= 0:
                    torch.manual_seed(seed)
            generated = model.generate(**encoded, **options)
            prompt_length = 0 if task == "seq2seq-lm" else int(encoded["input_ids"].shape[-1])
            token_ids = generated[0, prompt_length:].detach().cpu().tolist()
            if task == "seq2seq-lm":
                generation_config = getattr(model, "generation_config", model.config)
                token_ids = _normalize_seq2seq_tokens(
                    token_ids,
                    getattr(generation_config, "decoder_start_token_id", None),
                    getattr(generation_config, "eos_token_id", None),
                    str(request.get("output_token_policy", "new-tokens")),
                )
            results.append(
                {
                    "sample_id": str(sample["sample_id"]),
                    "prompt": prompt,
                    "token_ids": [int(value) for value in token_ids],
                    "text": tokenizer.decode(token_ids, skip_special_tokens=True),
                }
            )
    output = {
        "schema_version": "trtmc.accuracy-reference/v1",
        "model": request["model"],
        "revision": getattr(model.config, "_commit_hash", None) or request.get("revision"),
        "task": task,
        "precision": precision,
        "output_token_policy": str(request.get("output_token_policy", "new-tokens")),
        "samples": results,
    }
    arguments.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def _model_load_options(request: Mapping[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {
        "local_files_only": os.environ.get("TRTMC_QUALIFICATION_LOCAL_FILES_ONLY") == "1"
    }
    if request.get("revision"):
        options["revision"] = request["revision"]
    trust_remote_code = request.get("trust_remote_code")
    if trust_remote_code is not None:
        if not isinstance(trust_remote_code, bool):
            raise ValueError("trust_remote_code must be a boolean")
        options["trust_remote_code"] = trust_remote_code
    experts_implementation = request.get("experts_implementation")
    if experts_implementation is not None:
        if not isinstance(experts_implementation, str) or not experts_implementation:
            raise ValueError("experts_implementation must be a non-empty string")
        options["experts_implementation"] = experts_implementation
    return options


def _model_class(task: str, causal: Any, seq2seq: Any) -> Any:
    if task == "causal-lm":
        return causal
    if task == "seq2seq-lm":
        return seq2seq
    raise ValueError(f"unsupported reference task {task!r}")


def _translation_controls(
    tokenizer: Any, request: Mapping[str, Any]
) -> tuple[dict[str, int], int | None]:
    source = request.get("source_language")
    target = request.get("target_language")
    source_placement = request.get("source_language_placement")
    for name, value in (("source_language", source), ("target_language", target)):
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"generation.{name} must be a nonempty language identifier")
    if source_placement is not None and source_placement != "replace-final-unk":
        raise ValueError(
            "generation.source_language_placement must be replace-final-unk when set"
        )

    def token_id(language: str) -> int:
        lookup = getattr(tokenizer, "get_lang_id", None)
        if lookup is None:
            lookup = tokenizer.convert_tokens_to_ids
        value = lookup(language)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value == getattr(tokenizer, "unk_token_id", None)
        ):
            raise ValueError(f"tokenizer does not recognize language {language!r}")
        return value

    def explicit_id(name: str) -> int | None:
        value = request.get(name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < -1:
            raise ValueError(f"generation.{name} must be a nonnegative integer or -1")
        return None if value == -1 else value

    source_id = explicit_id("source_language_token_id")
    target_id = explicit_id("forced_bos_token_id")
    if source_placement is not None and source_id is None:
        raise ValueError(
            "generation.source_language_placement requires source_language_token_id"
        )
    manual_source_id = None
    if source_id is not None:
        if source is not None and token_id(source) != source_id:
            raise ValueError("source language disagrees with source_language_token_id")
        source = tokenizer.convert_ids_to_tokens(source_id)
    if source is not None:
        if hasattr(tokenizer, "src_lang"):
            token_id(source)
            tokenizer.src_lang = source
        elif source == getattr(tokenizer, "source_lang", None):
            pass
        elif source_id is not None and token_id(source) == source_id:
            if source_placement != "replace-final-unk":
                raise ValueError(
                    "generic translation tokenizer requires explicit "
                    "source_language_placement=replace-final-unk"
                )
            manual_source_id = source_id
        else:
            raise ValueError("reference tokenizer does not support the requested source language")
    if target is not None:
        if hasattr(tokenizer, "src_lang"):
            resolved = token_id(target)
            if target_id is not None and target_id != resolved:
                raise ValueError("target language disagrees with forced_bos_token_id")
            target_id = resolved
        elif target == getattr(tokenizer, "target_lang", None):
            pass
        elif target_id is not None and token_id(target) == target_id:
            pass
        else:
            raise ValueError("reference tokenizer does not support the requested target language")
    controls = {} if target_id is None else {"forced_bos_token_id": target_id}
    return controls, manual_source_id


def _apply_source_language(
    encoded: Mapping[str, Any], source_token_id: int | None, tokenizer: Any
) -> None:
    if source_token_id is None:
        return
    input_ids = encoded.get("input_ids")
    if input_ids is None or getattr(input_ids, "ndim", None) != 2:
        raise ValueError("tokenized translation input must contain rank-2 input_ids")
    attention_mask = encoded.get("attention_mask")
    for row in range(int(input_ids.shape[0])):
        if attention_mask is None:
            index = int(input_ids.shape[1]) - 1
        else:
            positions = attention_mask[row].nonzero(as_tuple=False)
            if int(positions.numel()) == 0:
                raise ValueError("tokenized translation input cannot be empty")
            index = int(positions[-1].item())
        current = int(input_ids[row, index].item())
        if current == source_token_id:
            continue
        if current != getattr(tokenizer, "unk_token_id", None):
            raise ValueError("generic translation tokenizer did not emit a language placeholder")
        input_ids[row, index] = source_token_id


def _normalize_seq2seq_tokens(
    token_ids: Sequence[int],
    decoder_start_token_id: int | None,
    eos_token_id: int | Sequence[int] | None,
    policy: str,
) -> list[int]:
    if policy not in {"new-tokens", "strip-start", "strip-start-and-eos"}:
        raise ValueError(f"unsupported output token policy {policy!r}")
    normalized = [int(value) for value in token_ids]
    if policy == "new-tokens":
        return normalized
    if (
        normalized
        and decoder_start_token_id is not None
        and normalized[0] == int(decoder_start_token_id)
    ):
        normalized.pop(0)
    if policy != "strip-start-and-eos" or not normalized or eos_token_id is None:
        return normalized
    eos_ids = (
        {int(eos_token_id)}
        if isinstance(eos_token_id, int)
        else {int(value) for value in eos_token_id}
    )
    if normalized[-1] in eos_ids:
        normalized.pop()
    return normalized


def _precision_load_options(dtype: Any) -> dict[str, Any]:
    return {"torch_dtype": dtype}


def _truncate(tokenizer: Any, prompt: str, limit: int, side: str) -> str:
    if limit < 1 or side not in {"left", "right"}:
        raise ValueError("prompt token limit and truncation side are invalid")
    token_ids = [int(value) for value in tokenizer.encode(prompt, add_special_tokens=False)]
    if len(token_ids) <= limit:
        return prompt
    selected = token_ids[-limit:] if side == "left" else token_ids[:limit]
    return str(
        tokenizer.decode(
            selected,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
