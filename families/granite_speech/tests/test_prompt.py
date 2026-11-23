# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt ids and the detokenisation table against the official tokenizer (CPU).

The checkpoint-backed tests read the pinned snapshot from the local Hugging Face cache and skip
when it is absent; the E2E repeats the prompt check against the official processor's input ids.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from families.granite_speech import prompt

_MANIFEST = Path(__file__).resolve().parent / "manifests" / "granite-4.0-1b-speech.json"


def _cached_checkpoint() -> Path:
    hub = pytest.importorskip("huggingface_hub")
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    try:
        return Path(hub.snapshot_download(manifest["hf_id"], revision=manifest["hf_revision"],
                                          local_files_only=True))
    except Exception:  # noqa: BLE001 - any cache miss means the checkpoint is not provisioned
        pytest.skip("the pinned checkpoint is not in the local Hugging Face cache")


def test_byte_decoder_inverts_the_gpt2_byte_table():
    table = prompt.byte_decoder()
    assert sorted(table.values()) == list(range(256))
    assert table["A"] == ord("A") and table["~"] == ord("~")
    assert table["Ġ"] == ord(" ")  # GPT-2 renders a space as U+0120
    assert table["Ċ"] == ord("\n")


def test_prompt_ids_split_around_one_placeholder():
    model_dir = _cached_checkpoint()
    text, prefix, suffix = prompt.prompt_ids(model_dir, 100352)
    assert text == "USER: <|audio|>can you transcribe the speech into a written format?\n ASSISTANT:"
    # The official processor's input ids for the checkpoint sample (252 placeholders) start with
    # these three ids and end with these fifteen.
    assert prefix == [6584, 25, 220]
    assert suffix == [4919, 499, 1380, 3191, 279, 8982, 1139, 264, 5439, 3645, 5380, 36660, 3931, 2891, 25]


def test_vocabulary_decodes_like_the_official_tokenizer():
    transformers = pytest.importorskip("transformers")
    model_dir = _cached_checkpoint()
    table = prompt.vocabulary(model_dir, 100353)
    assert len(table["tokens"]) == 100353
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_dir))
    skipped = set(table["skip"])
    assert 100257 in skipped and 100352 in skipped  # EOS and the audio placeholder are special

    def decode(ids):
        raw = b"".join(bytes.fromhex(table["tokens"][i]) for i in ids if i not in skipped)
        return raw.decode("utf-8", errors="replace")

    samples = [
        tokenizer("le conte du pêcheur, dit-elle — 東京 🙂", add_special_tokens=False)["input_ids"] + [100257],
        [2000, 6935, 29671, 574, 264, 72524, 8415, 100257],
        list(range(100200, 100353)),
    ]
    import random

    rng = random.Random(0)
    samples.append([rng.randrange(100353) for _ in range(400)])
    for ids in samples:
        assert decode(ids) == tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
