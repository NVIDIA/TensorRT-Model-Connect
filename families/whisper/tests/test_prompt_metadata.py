# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Released-checkpoint Whisper decoder prompt contracts."""

from __future__ import annotations

import json
from pathlib import Path

from families.whisper.config import ModelConfig
from families.whisper.prompt_metadata import whisper_decoder_prompt_metadata


def _model_config(model_dir: Path, **raw) -> ModelConfig:
    return ModelConfig(
        model_type="whisper",
        raw={"_model_dir": str(model_dir), **raw},
    )


def test_tiny_decoder_prompt_uses_released_checkpoint_ids(tmp_path: Path) -> None:
    (tmp_path / "generation_config.json").write_text(
        json.dumps(
            {
                "decoder_start_token_id": 50258,
                "forced_decoder_ids": [[1, None], [2, 50359]],
                "lang_to_id": {"<|en|>": 50259},
                "task_to_id": {"transcribe": 50359},
                "no_timestamps_token_id": 50363,
            }
        ),
        encoding="utf-8",
    )
    config = _model_config(
        tmp_path,
        decoder_start_token_id=50258,
        forced_decoder_ids=[[1, 50259], [2, 50359], [3, 50363]],
    )

    assert whisper_decoder_prompt_metadata(config) == {
        "decoder_start_token_ids": [50258, 50259, 50359, 50363]
    }


def test_large_v3_turbo_decoder_prompt_does_not_use_tiny_ids(tmp_path: Path) -> None:
    (tmp_path / "generation_config.json").write_text(
        json.dumps(
            {
                "decoder_start_token_id": 50258,
                "forced_decoder_ids": [[1, None], [2, 50360]],
                "lang_to_id": {"<|en|>": 50259},
                "task_to_id": {"transcribe": 50360},
                "no_timestamps_token_id": 50364,
            }
        ),
        encoding="utf-8",
    )
    config = _model_config(tmp_path, decoder_start_token_id=50258)

    assert whisper_decoder_prompt_metadata(config) == {
        "decoder_start_token_ids": [50258, 50259, 50360, 50364]
    }
