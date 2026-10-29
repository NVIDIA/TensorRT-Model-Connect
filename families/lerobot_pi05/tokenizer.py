# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The PaliGemma tokenizer asset.

LeRobot names ``google/paligemma-3b-pt-224`` as the tokenizer, which is a gated
repository, and checkpoints do not ship it. The SentencePiece model inside it is
byte-identical to the one the public big_vision bucket serves, so the builder
takes either a local copy or that public file, and accepts nothing whose hash
differs: the runtime tokenizer is only valid for this exact vocabulary.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


TOKENIZER_URL = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
TOKENIZER_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
TOKENIZER_BYTES = 4264023
_TIMEOUT_SECONDS = 120


def _verified(data: bytes, origin: str) -> bytes:
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != TOKENIZER_BYTES or digest != TOKENIZER_SHA256:
        raise ValueError(
            f"PaliGemma tokenizer from {origin} does not match the supported vocabulary "
            f"(sha256 {digest}, expected {TOKENIZER_SHA256})"
        )
    return data


def load_tokenizer_model(model_dir: str | Path) -> bytes:
    """Return the verified SentencePiece model: ``model_dir/tokenizer.model`` or the public file."""
    local = Path(model_dir) / "tokenizer.model"
    if local.is_file():
        return _verified(local.read_bytes(), str(local))
    try:
        with urllib.request.urlopen(TOKENIZER_URL, timeout=_TIMEOUT_SECONDS) as response:  # noqa: S310
            return _verified(response.read(), TOKENIZER_URL)
    except OSError as error:
        raise RuntimeError(
            "The PaliGemma tokenizer is not in the checkpoint directory and could not be "
            f"downloaded from {TOKENIZER_URL}; place the file at {local}"
        ) from error
