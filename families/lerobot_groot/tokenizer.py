# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Qwen3-VL tokenizer asset of the GR00T N1.7 backbone.

LeRobot's groot processor loads its tokenizer from ``nvidia/Cosmos-Reason2-2B`` and
checkpoints do not ship it. The builder takes ``tokenizer.json`` from the checkpoint
directory if present, otherwise that file at a pinned revision of the repository through
the Hugging Face cache, so a build is offline once it is cached. The repository is gated,
so CI stages the public ``Qwen/Qwen3-VL-2B-Instruct`` copy that the manifests declare in
``external_files``: at the pinned revisions both repositories serve the same tokenizer and
processor files (identical Git blob ids). The runtime prompt is only valid for this exact
vocabulary, so a file of any other size, or with other template tokens, is refused.
"""

from __future__ import annotations

import json
from pathlib import Path


TOKENIZER_REPO = "nvidia/Cosmos-Reason2-2B"
TOKENIZER_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
TOKENIZER_FILE = "tokenizer.json"
PUBLIC_COPY = "Qwen/Qwen3-VL-2B-Instruct"
PUBLIC_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
TOKENIZER_BYTES = 7032403
# The chat-template and image tokens the runtime prompt is built from.
TEMPLATE_TOKENS = {
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|vision_start|>": 151652,
    "<|vision_end|>": 151653,
    "<|image_pad|>": 151655,
}


def _checked(data: bytes, origin: str) -> bytes:
    if len(data) != TOKENIZER_BYTES:
        raise ValueError(
            f"Qwen3-VL tokenizer from {origin} is {len(data)} bytes, expected {TOKENIZER_BYTES}; "
            "it is not the supported vocabulary"
        )
    added = {token["content"]: token["id"] for token in json.loads(data).get("added_tokens", [])}
    found = {token: added.get(token) for token in TEMPLATE_TOKENS}
    if found != TEMPLATE_TOKENS:
        raise ValueError(f"Qwen3-VL tokenizer from {origin} has template tokens {found}, expected {TEMPLATE_TOKENS}")
    return data


def load_tokenizer_json(model_dir: str | Path) -> bytes:
    """Return the checked ``tokenizer.json``: the checkpoint's copy or the pinned Hub file."""
    local = Path(model_dir) / TOKENIZER_FILE
    if local.is_file():
        return _checked(local.read_bytes(), str(local))
    pinned = f"{TOKENIZER_REPO}@{TOKENIZER_REVISION}"
    try:
        from huggingface_hub import hf_hub_download

        cached = Path(hf_hub_download(TOKENIZER_REPO, TOKENIZER_FILE, revision=TOKENIZER_REVISION))
    except Exception as error:  # noqa: BLE001 - offline, gated and network errors share one remedy
        raise FileNotFoundError(
            f"the Qwen3-VL tokenizer is not at {local}, and {TOKENIZER_FILE} of {pinned} is neither in "
            f"the Hugging Face cache nor downloadable ({type(error).__name__}). The repository is gated: "
            f"log in with access to it, or copy the identical public {TOKENIZER_FILE} of "
            f"{PUBLIC_COPY}@{PUBLIC_REVISION} to {local}"
        ) from error
    return _checked(cached.read_bytes(), pinned)
