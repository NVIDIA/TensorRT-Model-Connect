#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import pathlib
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF tokenizer bridge for trtmc C++ runtime")
    parser.add_argument("--model-dir", required=True, help="Local Hugging Face model directory")
    parser.add_argument("--check", action="store_true", help="Validate tokenizer can be loaded")
    parser.add_argument(
        "--op",
        choices=["encode", "decode", "id-for-token", "token-for-id"],
        default="",
        help="Tokenizer operation",
    )
    parser.add_argument("--text-file", default="", help="Input text file for encode")
    parser.add_argument("--ids", default="", help="Comma-separated token IDs for decode")
    parser.add_argument("--token", default="", help="Token string for id-for-token")
    parser.add_argument("--id", type=int, default=0, help="Token ID for token-for-id")
    parser.add_argument("--add-special-tokens", action="store_true",
                        help="Include special tokens (e.g. EOS) during encode")
    return parser.parse_args()


def load_tokenizer(model_dir: str):
    import os
    from transformers import AutoTokenizer

    tokenizer_json = os.path.join(model_dir, "tokenizer.json")

    tok = None
    try:
        tok = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True,
        )
    except Exception:
        try:
            tok = AutoTokenizer.from_pretrained(
                model_dir,
                local_files_only=True,
                trust_remote_code=True,
                use_fast=False,
            )
        except Exception:
            # AutoTokenizer failed entirely (e.g. custom tokenizer class not
            # available). Fall back to PreTrainedTokenizerFast from tokenizer.json.
            if os.path.exists(tokenizer_json):
                from transformers import PreTrainedTokenizerFast
                return PreTrainedTokenizerFast(tokenizer_file=tokenizer_json)
            raise

    # Verify byte-level BPE decode round-trips correctly. Some tokenizers
    # (e.g. a mismatched slow tokenizer loading tokenizer.json) lose spaces or
    # produce raw byte tokens (Ġ for space). Fall back to PreTrainedTokenizerFast.
    if os.path.exists(tokenizer_json):
        test_ids = tok.encode("Hello world")
        test_dec = tok.decode(test_ids, skip_special_tokens=True)
        if "Hello world" not in test_dec:
            from transformers import PreTrainedTokenizerFast
            tok = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json)
    return tok


def parse_ids_csv(ids_text: str) -> list[int]:
    ids_text = ids_text.strip()
    if not ids_text:
        return []
    out: list[int] = []
    for part in ids_text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> int:
    args = parse_args()

    model_dir = pathlib.Path(args.model_dir)
    if not model_dir.exists():
        print(f"Model directory does not exist: {model_dir}", file=sys.stderr)
        return 2

    try:
        tokenizer = load_tokenizer(str(model_dir))
    except Exception as exc:
        print(f"Failed to load tokenizer: {exc}", file=sys.stderr)
        return 3

    if args.check:
        return 0

    if args.op == "encode":
        if not args.text_file:
            print("--text-file is required for encode", file=sys.stderr)
            return 4
        text = pathlib.Path(args.text_file).read_text(encoding="utf-8")
        ids = tokenizer.encode(text, add_special_tokens=args.add_special_tokens)
        print(" ".join(str(i) for i in ids))
        return 0

    if args.op == "decode":
        ids = parse_ids_csv(args.ids)
        decoded = tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print(decoded)
        return 0

    if args.op == "id-for-token":
        if args.token == "":
            print("--token is required for id-for-token", file=sys.stderr)
            return 5
        token_id = tokenizer.convert_tokens_to_ids(args.token)
        if token_id is None:
            token_id = tokenizer.unk_token_id
        if token_id is None:
            token_id = 0
        print(int(token_id))
        return 0

    if args.op == "token-for-id":
        token = tokenizer.convert_ids_to_tokens(int(args.id))
        if token is None:
            token = tokenizer.unk_token or ""
        print(token)
        return 0

    print("--op is required unless --check is set", file=sys.stderr)
    return 6


if __name__ == "__main__":
    raise SystemExit(main())
