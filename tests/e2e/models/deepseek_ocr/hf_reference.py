#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official DeepSeek-OCR-2 Hugging Face inference path."""

from __future__ import annotations

import argparse
import json
import re
import tempfile

import torch
from transformers import AutoModel, AutoTokenizer

from tensorrt_model_connect.deepseek_ocr_reference_compat import (
    DEEPSEEK_OCR_REFERENCE_REVISION,
    assert_deepseek_ocr_eager_attention,
    configure_deepseek_ocr_legacy_generation_cache,
    configure_deepseek_ocr_rotary_embeddings,
    install_deepseek_ocr_transformers_compat,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be an exact 40-character commit SHA")
    if args.revision != DEEPSEEK_OCR_REFERENCE_REVISION:
        parser.error(
            "--revision must match the qualified DeepSeek-OCR-2 reference "
            f"{DEEPSEEK_OCR_REFERENCE_REVISION}"
        )

    compat = install_deepseek_ocr_transformers_compat()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        code_revision=args.revision,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        code_revision=args.revision,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    eager_attention_layers = assert_deepseek_ocr_eager_attention(model, compat)
    rotary_language_layers, rotary_visual_encoders = configure_deepseek_ocr_rotary_embeddings(
        model, compat
    )
    if rotary_language_layers != eager_attention_layers or rotary_visual_encoders != 1:
        raise RuntimeError("DeepSeek-OCR rotary compatibility coverage mismatch")
    configure_deepseek_ocr_legacy_generation_cache(model)
    model = model.cuda()

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

    print(
        json.dumps(
            {
                "text": str(text or ""),
                "eager_attention_layers": eager_attention_layers,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
