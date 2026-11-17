# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time transcription prompt and vocabulary tables from the checkpoint's Tekken tokenizer.

The prompt framing is taken from ``mistral_common``'s own transcription encoder,
so the native runtime only repeats the audio placeholder and selects a suffix:

    prefix + [AUDIO] * (tokens per chunk * chunks) + suffix(language)
"""

from __future__ import annotations

import base64
import io
import json
import struct
import wave
from pathlib import Path

from .config import SAMPLE_RATE, VoxtralConfig

# Languages the checkpoint's model card documents for transcription.
LANGUAGES = ("en", "es", "fr", "pt", "hi", "de", "nl", "it")


def _silence(seconds: float) -> io.BytesIO:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(b"\0\0" * int(seconds * SAMPLE_RATE))
    buffer.seek(0)
    return buffer


def _split(tokens: list[int], audio_token_id: int) -> tuple[list[int], int, list[int]]:
    positions = [index for index, token in enumerate(tokens) if token == audio_token_id]
    if not positions or positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError("the transcription prompt does not hold one contiguous audio span")
    return tokens[: positions[0]], len(positions), tokens[positions[-1] + 1 :]


def prompt_contract(model_dir: Path, config: VoxtralConfig) -> dict:
    """Prompt framing, stop token and language suffixes as encoded by mistral_common."""
    from mistral_common.protocol.transcription.request import TranscriptionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tokenizer = MistralTokenizer.from_file(str(Path(model_dir) / "tekken.json"))
    instruct = tokenizer.instruct_tokenizer

    def encode(seconds: float, language: str | None) -> list[int]:
        request = TranscriptionRequest.from_openai(
            {"model": "voxtral", "file": _silence(seconds), "language": language})
        return list(instruct.encode_transcription(request).tokens)

    prefix, one_chunk, suffix = _split(encode(1.0, None), config.audio_token_id)
    prefix_two, two_chunks, suffix_two = _split(encode(31.0, None), config.audio_token_id)
    if (one_chunk != config.audio_tokens_per_chunk or two_chunks != 2 * one_chunk
            or prefix_two != prefix or suffix_two != suffix):
        raise ValueError(
            f"Tekken prompt places {one_chunk} / {two_chunks} audio tokens for one / two chunks; "
            f"the encoder produces {config.audio_tokens_per_chunk} per chunk")
    languages = {}
    for language in LANGUAGES:
        language_prefix, count, language_suffix = _split(encode(1.0, language),
                                                         config.audio_token_id)
        if language_prefix != prefix or count != one_chunk:
            raise ValueError(f"the {language!r} transcription prompt changes the prompt prefix")
        languages[language] = language_suffix
    inner = instruct.tokenizer
    eos = int(inner.eos_id)
    if inner.n_words != config.text.vocab_size:
        raise ValueError(f"Tekken has {inner.n_words} tokens, the decoder {config.text.vocab_size}")
    return {
        "prompt_prefix_ids": prefix,
        "prompt_suffix_ids": suffix,
        "language_suffix_ids": languages,
        "eos_token_id": eos,
        "num_special_tokens": int(inner.num_special_tokens),
    }


def vocabulary_table(model_dir: Path, vocab_size: int, num_special_tokens: int) -> bytes:
    """Byte strings of every token id, special ids empty.

    Layout (little endian): ``u32 count``, ``u32 offsets[count + 1]``, then the
    concatenated bytes; token ``i`` is ``bytes[offsets[i]:offsets[i + 1]]``.
    """
    tekken = json.loads((Path(model_dir) / "tekken.json").read_text(encoding="utf-8"))
    vocabulary = tekken["vocab"]
    if len(vocabulary) < vocab_size - num_special_tokens:
        raise ValueError("tekken.json holds fewer regular tokens than the decoder vocabulary")
    pieces = [b""] * num_special_tokens
    for rank in range(vocab_size - num_special_tokens):
        entry = vocabulary[rank]
        if entry["rank"] != rank:
            raise ValueError("tekken.json vocabulary is not ordered by rank")
        pieces.append(base64.b64decode(entry["token_bytes"]))
    offsets = [0]
    for piece in pieces:
        offsets.append(offsets[-1] + len(piece))
    return (struct.pack("<I", vocab_size) + struct.pack(f"<{vocab_size + 1}I", *offsets)
            + b"".join(pieces))
