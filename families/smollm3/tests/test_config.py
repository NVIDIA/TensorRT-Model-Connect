# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for SmolLM3 checkpoint configuration."""

import json
from pathlib import Path

import pytest

from families.smollm3.config import ModelConfig


@pytest.mark.parametrize("name", ["bos_token_id", "eos_token_id", "pad_token_id"])
@pytest.mark.parametrize("token_id", [0, 128012, -1, None])
def test_special_token_ids_from_checkpoint(
    tmp_path: Path, name: str, token_id: int | None
) -> None:
    checkpoint = {"model_type": "smollm3", name: token_id}
    (tmp_path / "config.json").write_text(json.dumps(checkpoint), encoding="utf-8")

    config = ModelConfig.from_dir(tmp_path)

    assert getattr(config, name) == (-1 if token_id is None else token_id)
    assert config.raw == checkpoint


def test_missing_special_token_ids_remain_unset() -> None:
    config = ModelConfig.from_json('{"model_type": "smollm3"}')

    assert (config.bos_token_id, config.eos_token_id, config.pad_token_id) == (-1, -1, -1)
