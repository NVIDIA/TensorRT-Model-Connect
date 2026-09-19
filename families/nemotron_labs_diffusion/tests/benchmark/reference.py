#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official Nemotron Labs Diffusion autoregressive reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _dtype(torch: Any, precision: str) -> Any:
    try:
        return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    except KeyError as error:
        raise ValueError(f"unsupported reference precision {precision!r}") from error


def _inputs(tokenizer: Any, prompt: str, generation: dict[str, Any]) -> Any:
    if not generation.get("use_chat_template", False):
        return tokenizer(prompt, return_tensors="pt")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(generation.get("enable_thinking", False)),
    )
    return tokenizer(rendered, return_tensors="pt", add_special_tokens=False)


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


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    request = json.loads(arguments.request.read_text(encoding="utf-8"))
    samples = request.get("samples")
    generation = request.get("generation")
    if not isinstance(samples, list) or not samples or not isinstance(generation, dict):
        raise ValueError("reference request must contain samples and generation")
    mode = generation.get("text_generation_mode", generation.get("generation_mode", "ar"))
    if mode != "ar":
        raise ValueError("this qualification reference supports text_generation_mode=ar")

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_id = str(request["model"])
    revision = request.get("revision")
    options = {
        "trust_remote_code": bool(request.get("trust_remote_code", True)),
        **({"revision": revision} if revision else {}),
    }
    tokenizer = AutoTokenizer.from_pretrained(model_id, **options)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=_dtype(torch, str(request.get("precision", "fp32"))),
        **options,
    ).eval().to("cuda")
    max_new_tokens = int(generation.get("max_new_tokens", 20))
    prompt_token_limit = int(request.get("prompt_token_limit", 192))
    truncation_side = str(request.get("truncation_side", "left"))
    results = []
    for sample in samples:
        prompt = _truncate(
            tokenizer,
            str(sample["prompt"]),
            prompt_token_limit,
            truncation_side,
        )
        encoded = _inputs(tokenizer, prompt, generation)
        input_ids = encoded["input_ids"].to(model.device)
        with torch.inference_mode():
            generated = model.ar_generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=float(generation.get("temperature", 0.0)),
                eos_token_id=tokenizer.eos_token_id,
            )
        if isinstance(generated, tuple):
            generated = generated[0]
        token_ids = generated[0, input_ids.shape[-1] :][:max_new_tokens].cpu().tolist()
        results.append(
            {
                "sample_id": str(sample["sample_id"]),
                "prompt": prompt,
                "token_ids": token_ids,
                "text": tokenizer.decode(token_ids, skip_special_tokens=True),
            }
        )
    arguments.output.write_text(
        json.dumps({"samples": results}, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
