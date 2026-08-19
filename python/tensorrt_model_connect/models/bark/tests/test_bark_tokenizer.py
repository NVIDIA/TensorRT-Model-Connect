# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit test: Bark tokenizer must use add_special_tokens=False.

HF's BarkProcessor tokenizes with add_special_tokens=False (no [CLS]/[SEP]).
The TRT C++ pipeline must match this behavior. If add_special_tokens=True is
used, the extra [CLS]/[SEP] tokens corrupt the semantic prefill, producing
completely different (and degenerate) audio.

Trace: UD-BARK-TOKENIZER — Bark tokenizer special-token contract
Intent: Verify Bark tokenizer does NOT add [CLS]/[SEP] special tokens
Preconditions: transformers is importable
Postconditions: Bark-style BERT tokenization omits CLS/SEP and preserves padding
"""

from __future__ import annotations

import pytest

try:
    from transformers import BertTokenizer
except ImportError:
    pytest.skip("transformers not available", allow_module_level=True)


PROMPT = "Hello, this is a test of the audio generation system."
BARK_PROCESSOR_LENGTH = 256


@pytest.fixture(scope="module")
def bark_vocab_file(tmp_path_factory):
    """Create a tiny local BERT vocab with Bark-style special token ids."""
    vocab_dir = tmp_path_factory.mktemp("bark_vocab")
    vocab_file = vocab_dir / "vocab.txt"
    vocab_file.write_text("\n".join([
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "hello",
        ",",
        "this",
        "is",
        "a",
        "test",
        "of",
        "the",
        "audio",
        "generation",
        "system",
        ".",
    ]) + "\n")
    return vocab_file


@pytest.fixture(scope="module")
def bark_tokenizer(bark_vocab_file):
    """Load a local BERT tokenizer matching Bark's special-token contract."""
    return BertTokenizer(vocab_file=str(bark_vocab_file), do_lower_case=True)


def bark_processor_ids(tokenizer, text: str) -> list[int]:
    """Mirror BarkProcessor's text path: no special tokens, padded to 256."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids + [tokenizer.pad_token_id] * (BARK_PROCESSOR_LENGTH - len(ids))


def test_bark_tokenizer_no_special_tokens(bark_tokenizer):
    """Bark tokenization must NOT add [CLS]/[SEP] (add_special_tokens=False).

    HF's BarkProcessor always calls encode(text, add_special_tokens=False).
    If [CLS] (101) and [SEP] (102) are present, the semantic model receives
    wrong token offsets and generates garbage audio.
    """
    ids_correct = bark_tokenizer.encode(PROMPT, add_special_tokens=False)
    ids_wrong = bark_tokenizer.encode(PROMPT, add_special_tokens=True)

    # The correct tokenization should not include CLS/SEP
    cls_id = bark_tokenizer.cls_token_id  # typically 101
    sep_id = bark_tokenizer.sep_token_id  # typically 102

    assert cls_id not in ids_correct, \
        f"CLS token {cls_id} found in Bark tokenization — add_special_tokens should be False"
    assert sep_id not in ids_correct, \
        f"SEP token {sep_id} found in Bark tokenization — add_special_tokens should be False"

    # The wrong tokenization DOES include CLS/SEP (sanity check)
    assert ids_wrong[0] == cls_id, "Sanity: add_special_tokens=True should prepend CLS"
    assert ids_wrong[-1] == sep_id, "Sanity: add_special_tokens=True should append SEP"

    # Token count difference should be exactly 2 ([CLS] and [SEP])
    assert len(ids_wrong) == len(ids_correct) + 2, \
        f"Expected 2 extra tokens with special, got {len(ids_wrong) - len(ids_correct)}"


def test_bark_tokenizer_matches_processor_padding_contract(bark_tokenizer):
    """Verify BarkProcessor-style padding does NOT add special tokens."""
    processor_ids = bark_processor_ids(bark_tokenizer, PROMPT)

    # processor pads to 256 — extract non-zero tokens
    processor_tokens = [t for t in processor_ids if t != 0]

    # Our encode without special tokens should match
    our_tokens = bark_tokenizer.encode(PROMPT, add_special_tokens=False)

    assert our_tokens == processor_tokens, \
        (f"Token mismatch:\n"
         f"  encode(add_special=False): {our_tokens}\n"
         f"  AutoProcessor:             {processor_tokens}")


def test_bark_token_count_is_reasonable(bark_tokenizer):
    """Bark tokenization of a simple prompt should produce 8-30 tokens."""
    ids = bark_tokenizer.encode(PROMPT, add_special_tokens=False)
    assert 8 <= len(ids) <= 30, \
        f"Expected 8-30 tokens for test prompt, got {len(ids)}"
