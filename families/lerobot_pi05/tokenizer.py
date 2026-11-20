# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The PaliGemma tokenizer asset.

LeRobot's pi05 processor loads its tokenizer from ``google/paligemma-3b-pt-224`` and
checkpoints do not ship it. The builder takes ``tokenizer.model`` from the checkpoint
directory if present, otherwise that file at a pinned revision of the repository through
the Hugging Face cache, so a build is offline once it is cached. The repository is gated,
so CI stages the public big_vision copy that the manifests declare in ``external_files``.
That bucket serves the same file (identical size and SHA-256 at the pinned revision).
The runtime tokenizer is only valid for this exact vocabulary, so a file of any other size
is refused here and the converted vocabulary is checked in ``model.py``.
"""

from __future__ import annotations

from pathlib import Path


TOKENIZER_REPO = "google/paligemma-3b-pt-224"
TOKENIZER_REVISION = "35e4f46485b4d07967e7e9935bc3786aad50687c"
TOKENIZER_FILE = "tokenizer.model"
PUBLIC_COPY = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
TOKENIZER_BYTES = 4264023
# The converted vocabulary: 257,152 pieces with pad, eos, bos and unk at 0..3.
VOCABULARY_PIECES = 257152
SPECIAL_IDS = {"pad_id": 0, "eos_id": 1, "bos_id": 2, "unknown_id": 3}


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
    """Return the size-checked SentencePiece model: the checkpoint's copy or the pinned Hub file."""
    local = Path(model_dir) / TOKENIZER_FILE
    if local.is_file():
        return _checked(local.read_bytes(), str(local))
    pinned = f"{TOKENIZER_REPO}@{TOKENIZER_REVISION}"
    try:
        from huggingface_hub import hf_hub_download

        cached = Path(hf_hub_download(TOKENIZER_REPO, TOKENIZER_FILE, revision=TOKENIZER_REVISION))
    except Exception as error:  # noqa: BLE001 - offline, gated and network errors share one remedy
        raise FileNotFoundError(
            f"the PaliGemma tokenizer is not at {local}, and {TOKENIZER_FILE} of {pinned} is "
            f"neither in the Hugging Face cache nor downloadable ({type(error).__name__}). The "
            f"repository is gated: log in with access to it, or copy the identical public file "
            f"{PUBLIC_COPY} to {local}"
        ) from error
    return _checked(cached.read_bytes(), pinned)
