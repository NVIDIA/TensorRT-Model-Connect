#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Produce deterministic Hugging Face text-generation qualification references."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    request = json.loads(arguments.request.read_text(encoding="utf-8"))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("the HF Accuracy reference requires CUDA")
    precision = str(request.get("precision", "fp32"))
    dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if precision not in dtypes:
        raise ValueError(f"unsupported reference precision {precision!r}")
    common: dict[str, Any] = {
        "local_files_only": os.environ.get("TRTMC_QUALIFICATION_LOCAL_FILES_ONLY") == "1"
    }
    if request.get("revision"):
        common["revision"] = request["revision"]
    tokenizer = AutoTokenizer.from_pretrained(str(request["model"]), **common)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(request["model"]), dtype=dtypes[precision], **common
    ).eval()
    model.to("cuda")
    generation = request.get("generation", {})
    if not isinstance(generation, Mapping):
        raise ValueError("generation must be an object")
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
            encoded = {name: value.to("cuda") for name, value in encoded.items()}
            options: dict[str, Any] = {
                "max_new_tokens": int(generation.get("max_new_tokens", 64)),
                "do_sample": bool(generation.get("do_sample", False)),
                "num_beams": 1,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "repetition_penalty": float(generation.get("repetition_penalty", 1.0)),
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
            prompt_length = int(encoded["input_ids"].shape[-1])
            token_ids = generated[0, prompt_length:].detach().cpu().tolist()
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
        "precision": precision,
        "samples": results,
    }
    arguments.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


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
