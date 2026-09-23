# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create a reproducible raw-text/ID fixture for speculative decoding checks."""

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--input-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.input_tokens < 1:
        parser.error("--input-tokens must be positive")
    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    paragraph = (
        "A compiler transforms a mathematical program into instructions for a processor. "
        "The runtime supplies input tensors and manages the state used by successive calls. "
        "For a language model, cached keys and values represent the processed prefix. "
        "Speculative decoding proposes several tokens and verifies them with the target model. "
    )
    text = paragraph * 80 + "Explain how the compiler and runtime cooperate."
    ids = tokenizer.encode(text, add_special_tokens=False).ids[:args.input_tokens]
    if len(ids) != args.input_tokens:
        raise ValueError("requested fixture exceeds the source text length")
    text = tokenizer.decode(ids, skip_special_tokens=False)
    if tokenizer.encode(text, add_special_tokens=False).ids != ids:
        raise ValueError("fixture does not round-trip through the tokenizer")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(ids).encode()
    (args.output_dir / "input_ids.json").write_bytes(encoded)
    (args.output_dir / "input.txt").write_text(text)
    (args.output_dir / "fixture.json").write_text(json.dumps({
        "input_tokens": len(ids), "input_ids_sha256": hashlib.sha256(encoded).hexdigest(),
        "add_special_tokens": False, "apply_chat_template": False,
    }, indent=2))


if __name__ == "__main__":
    main()
