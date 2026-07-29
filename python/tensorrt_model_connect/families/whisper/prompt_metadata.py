# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whisper decoder prompt metadata derived from released checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

from .config import ModelConfig


def _read_json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _forced_decoder_token_map(*configs: dict) -> dict[int, int]:
    tokens: dict[int, int] = {}
    for config in configs:
        forced_ids = config.get("forced_decoder_ids")
        if not isinstance(forced_ids, list):
            continue
        for item in forced_ids:
            if (
                isinstance(item, list)
                and len(item) == 2
                and isinstance(item[0], int)
                and not isinstance(item[0], bool)
                and isinstance(item[1], int)
                and not isinstance(item[1], bool)
            ):
                tokens[int(item[0])] = int(item[1])
    return tokens


def whisper_decoder_prompt_metadata(config: ModelConfig) -> dict:
    """Resolve the model-specific English transcription decoder prefix."""
    raw = config.raw
    model_dir_value = raw.get("_model_dir")
    generation = {}
    if isinstance(model_dir_value, str) and model_dir_value:
        generation = _read_json_object(
            Path(model_dir_value) / "generation_config.json"
        )
    forced = _forced_decoder_token_map(raw, generation)

    decoder_start = generation.get(
        "decoder_start_token_id", raw.get("decoder_start_token_id")
    )
    language_token = forced.get(1)
    if language_token is None:
        lang_to_id = generation.get("lang_to_id")
        if isinstance(lang_to_id, dict):
            language_token = lang_to_id.get("<|en|>")

    task_token = None
    task_to_id = generation.get("task_to_id")
    if isinstance(task_to_id, dict):
        task_token = task_to_id.get("transcribe")
    if task_token is None:
        task_token = forced.get(2)

    no_timestamps_token = generation.get("no_timestamps_token_id")
    if no_timestamps_token is None:
        no_timestamps_token = forced.get(3)

    prompt = [
        decoder_start,
        language_token,
        task_token,
        no_timestamps_token,
    ]
    if not all(
        isinstance(token, int) and not isinstance(token, bool)
        for token in prompt
    ):
        return {}
    return {"decoder_start_token_ids": [int(token) for token in prompt]}
