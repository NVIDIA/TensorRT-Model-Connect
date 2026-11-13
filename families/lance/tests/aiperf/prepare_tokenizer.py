#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export Lance's tokenizer as a standard local directory for the AIPerf client."""

import argparse
import importlib.metadata
import json
from pathlib import Path


def export_tokenizer(model_dir: Path, output_dir: Path) -> None:
    from transformers import AutoConfig, AutoTokenizer

    source = model_dir / "Lance_3B"
    config = AutoConfig.from_pretrained(str(source / "llm_config.json"), local_files_only=True)
    if config.model_type != "qwen2_5_vl":
        raise ValueError("Lance_3B/llm_config.json must declare qwen2_5_vl")
    tokenizer = AutoTokenizer.from_pretrained(source, config=config, use_fast=True, local_files_only=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(output_dir)
    restored = AutoTokenizer.from_pretrained(output_dir, use_fast=True, local_files_only=True)
    probes = ("hello", "What is the main object in this image? Answer in one word.",
              "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>")
    identities = {}
    for text in probes:
        expected = tokenizer.encode(text, add_special_tokens=False)
        if restored.encode(text, add_special_tokens=False) != expected:
            raise RuntimeError("exported tokenizer changed token IDs")
        identities[text] = expected
    audit = {"source": str(source.resolve()),
             "transformers": importlib.metadata.version("transformers"),
             "tokenizer_class": type(tokenizer).__name__, "probes": identities}
    (output_dir / "trtmc-tokenizer-export.json").write_text(json.dumps(audit, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="local checkpoint root containing Lance_3B")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for exported tokenizer files")
    args = parser.parse_args()
    export_tokenizer(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
