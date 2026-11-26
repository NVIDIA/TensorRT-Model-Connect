#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export Phi4's standard tokenizer without its model-only remote configuration."""

import argparse
import importlib.metadata
import json
from pathlib import Path


def export_tokenizer(model_dir: Path, output_dir: Path) -> None:
    from transformers import AutoTokenizer, GPT2TokenizerFast, PreTrainedTokenizerFast

    config = json.loads((model_dir / "config.json").read_text())
    if config.get("model_type") not in {"phi4mm", "phi4_multimodal", "phi4multimodal"}:
        raise ValueError("model directory must contain the Phi4 multimodal checkpoint")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir, local_files_only=True)
    probes = ("hello", "What is the main object in this image? Answer in one word.",
              "<|user|>\n<|image_1|>\nQuestion?<|end|>\n<|assistant|>\n", "找到图片中的狗。")
    identities = [
        {"text": text, "add_special_tokens": special,
         "token_ids": tokenizer.encode(text, add_special_tokens=special)}
        for text in probes for special in (False, True)
    ]
    vocabulary = tokenizer.get_vocab()
    # Keep the checkpoint's GPT-4o regex/backend unchanged. A model-specific
    # GPT2 wrapper in a newer client can rebuild it with a different regex.
    exported = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer.backend_tokenizer,
        model_max_length=tokenizer.model_max_length,
        padding_side=tokenizer.padding_side,
        truncation_side=tokenizer.truncation_side,
        clean_up_tokenization_spaces=tokenizer.clean_up_tokenization_spaces,
        chat_template=tokenizer.chat_template,
        **tokenizer.special_tokens_map,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    exported.save_pretrained(output_dir)
    restored = AutoTokenizer.from_pretrained(output_dir, use_fast=True, local_files_only=True)
    if restored.get_vocab() != vocabulary:
        raise RuntimeError("exported tokenizer changed the vocabulary")
    if restored.chat_template != tokenizer.chat_template:
        raise RuntimeError("exported tokenizer changed the chat template")
    for probe in identities:
        if restored.encode(probe["text"], add_special_tokens=probe["add_special_tokens"]) != probe["token_ids"]:
            raise RuntimeError("exported tokenizer changed token IDs or framing")
    audit = {"source": str(model_dir.resolve()),
             "transformers": importlib.metadata.version("transformers"),
             "tokenizer_class": type(tokenizer).__name__, "probes": identities}
    (output_dir / "trtmc-tokenizer-export.json").write_text(json.dumps(audit, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="local Phi4 multimodal checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for client tokenizer files")
    args = parser.parse_args()
    export_tokenizer(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
