#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hugging Face local model generation for trtmc backend.")
    parser.add_argument("--model-dir", required=True, help="Local model directory containing config + safetensors")
    parser.add_argument("--prompt-file", help="Path to prompt text file")
    parser.add_argument("--prompt", default="", help="Prompt text (ignored when --prompt-file is set)")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--do-sample", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--check", action="store_true", help="Only validate loadability")
    return parser.parse_args()


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"model dir not found: {model_dir}", file=sys.stderr)
        return 2

    logging.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        use_safetensors=True,
    )
    model.eval()

    if args.check:
        print("ok")
        return 0

    prompt = load_prompt(args)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max(args.max_new_tokens, 0),
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
        )

    text = tokenizer.decode(output[0], skip_special_tokens=False)
    # No trailing newline so C++ wrapper does not need to strip final blank line.
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
