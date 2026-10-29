# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The PaliGemma tokenizer asset.

LeRobot names ``google/paligemma-3b-pt-224`` as the tokenizer, which is a gated
repository, and checkpoints do not ship it. The SentencePiece model inside it is
the one the public big_vision bucket serves (checked byte for byte when this
family was added), so the builder takes either a local copy or that public file.
The runtime tokenizer is only valid for this exact vocabulary, so a file of any
other size is refused here and the converted vocabulary is checked in ``model.py``.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path


TOKENIZER_URL = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
TOKENIZER_BYTES = 4264023
# The converted vocabulary: 257,152 pieces with pad, eos, bos and unk at 0..3.
VOCABULARY_PIECES = 257152
SPECIAL_IDS = {"pad_id": 0, "eos_id": 1, "bos_id": 2, "unknown_id": 3}
_TIMEOUT_SECONDS = 120


def _checked(data: bytes, origin: str) -> bytes:
    if len(data) != TOKENIZER_BYTES:
        raise ValueError(
            f"PaliGemma tokenizer from {origin} is {len(data)} bytes, "
            f"expected {TOKENIZER_BYTES}; it is not the supported vocabulary"
        )
    return data


def check_vocabulary(metadata) -> None:
    """Refuse a converted tokenizer whose vocabulary is not PaliGemma's."""
    found = {name: getattr(metadata, name) for name in SPECIAL_IDS}
    if metadata.piece_count != VOCABULARY_PIECES or found != SPECIAL_IDS:
        raise ValueError(
            f"PaliGemma tokenizer has {metadata.piece_count} pieces and special ids {found}; "
            f"expected {VOCABULARY_PIECES} and {SPECIAL_IDS}"
        )


def load_tokenizer_model(model_dir: str | Path) -> bytes:
    """Return the size-checked SentencePiece model: ``model_dir/tokenizer.model`` or the public file."""
    local = Path(model_dir) / "tokenizer.model"
    if local.is_file():
        return _checked(local.read_bytes(), str(local))
    try:
        with urllib.request.urlopen(TOKENIZER_URL, timeout=_TIMEOUT_SECONDS) as response:  # noqa: S310
            return _checked(response.read(), TOKENIZER_URL)
    except OSError as error:
        raise RuntimeError(
            "The PaliGemma tokenizer is not in the checkpoint directory and could not be "
            f"downloaded from {TOKENIZER_URL}; place the file at {local}"
        ) from error
