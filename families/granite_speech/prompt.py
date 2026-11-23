# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt token ids and the detokenisation table, taken from the checkpoint's own tokenizer.

Transcription uses the instruction of the model card. Its chat-templated prompt is tokenised
once, at build time, with the official tokenizer; the bundle stores the ids before and after the
audio placeholder, and the runtime inserts one placeholder per projector output, as the official
processor does before it tokenises. Generated ids become text through the byte-level table,
skipping special tokens, which is how the reference decodes with ``skip_special_tokens=True``.
"""

from __future__ import annotations

from pathlib import Path

INSTRUCTION = "can you transcribe the speech into a written format?"
AUDIO_TOKEN = "<|audio|>"


def byte_decoder() -> dict[str, int]:
    """Inverse of the GPT-2 byte-to-unicode table used by byte-level BPE vocabularies."""
    printable = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
                 + list(range(ord("®"), ord("ÿ") + 1)))
    codes = printable[:]
    extra = 0
    for value in range(256):
        if value not in printable:
            printable.append(value)
            codes.append(256 + extra)
            extra += 1
    return {chr(code): value for value, code in zip(printable, codes)}


def _tokenizer(model_dir: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir))


def prompt_ids(model_dir: Path, audio_token_id: int) -> tuple[str, list[int], list[int]]:
    """The templated prompt and the ids before and after its single audio placeholder."""
    tokenizer = _tokenizer(model_dir)
    if tokenizer.convert_tokens_to_ids(AUDIO_TOKEN) != audio_token_id:
        raise NotImplementedError(f"granite_speech needs {AUDIO_TOKEN} to be token {audio_token_id}")
    chat = [{"role": "user", "content": AUDIO_TOKEN + INSTRUCTION}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    ids = list(tokenizer(prompt)["input_ids"])
    if ids.count(audio_token_id) != 1:
        raise NotImplementedError("granite_speech needs exactly one audio placeholder in the prompt")
    at = ids.index(audio_token_id)
    prefix, suffix = ids[:at], ids[at + 1:]
    # The runtime inserts any number of placeholders between the fixed ids; that is only the
    # official expansion if the tokenizer never merges text across the placeholder.
    for count in (2, 7):
        expanded = list(tokenizer(prompt.replace(AUDIO_TOKEN, AUDIO_TOKEN * count))["input_ids"])
        if expanded != prefix + [audio_token_id] * count + suffix:
            raise NotImplementedError("granite_speech tokenizer merges text across the audio placeholder")
    return prompt, prefix, suffix


def vocabulary(model_dir: Path, vocab_size: int) -> dict:
    """Byte strings of every id below ``vocab_size`` (hex) and the ids decoding skips."""
    tokenizer = _tokenizer(model_dir)
    backend = tokenizer.backend_tokenizer
    if type(backend.decoder).__name__ != "ByteLevel":
        raise NotImplementedError("granite_speech decodes byte-level BPE vocabularies only")
    table = byte_decoder()
    added = tokenizer.added_tokens_decoder
    tokens: list[str] = []
    skipped: list[int] = []
    for token_id in range(vocab_size):
        if token_id in added:
            entry = added[token_id]
            if entry.special:
                skipped.append(token_id)
                tokens.append("")
            else:
                tokens.append(entry.content.encode("utf-8").hex())
            continue
        piece = backend.id_to_token(token_id)
        if piece is None:
            skipped.append(token_id)
            tokens.append("")
            continue
        try:
            raw = bytes(table[char] for char in piece)
        except KeyError as error:
            raise NotImplementedError(f"granite_speech token {token_id} is not byte-level: {piece!r}") from error
        tokens.append(raw.hex())
    return {"tokens": tokens, "skip": skipped}
