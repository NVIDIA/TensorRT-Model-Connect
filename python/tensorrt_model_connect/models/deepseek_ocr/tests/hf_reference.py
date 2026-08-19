#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official DeepSeek-OCR-2 Hugging Face inference path."""

from __future__ import annotations

import argparse
import json
import tempfile

import torch
from transformers import AutoModel, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval().cuda()

    prompt = args.prompt
    if "<image>" not in prompt:
        prompt = f"<image>\n{prompt}"

    with tempfile.TemporaryDirectory(prefix="deepseek_ocr_hf_") as output_dir:
        text = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=args.image,
            output_path=output_dir,
            base_size=1024,
            image_size=768,
            crop_mode=True,
            save_results=False,
            eval_mode=True,
        )

    print(json.dumps({"text": str(text or "")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
