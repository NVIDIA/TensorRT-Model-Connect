# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stop-token handling for checkpoints that name more than one EOS id."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ..config import ModelConfig
from ..model import _eos_token_ids, _runtime_config


def _config(eos: object) -> ModelConfig:
    return ModelConfig.from_json(
        json.dumps(
            {
                "model_type": "llama",
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 32,
                "bos_token_id": 0,
                "eos_token_id": eos,
                "pad_token_id": 0,
            }
        )
    )


def test_a_single_stop_token_is_normalised_to_one_entry() -> None:
    assert _eos_token_ids(2) == [2]
    assert _eos_token_ids([2]) == [2]


def test_several_stop_tokens_keep_their_order() -> None:
    """MiniCPM5 names two; the second is the one it actually emits."""
    assert _eos_token_ids([1, 130073]) == [1, 130073]


def test_a_boolean_is_not_a_token_id() -> None:
    """`bool` is an `int` subclass, so it would otherwise pass silently."""
    with pytest.raises(ValueError, match="must be an integer or a list of integers"):
        _eos_token_ids(True)
    with pytest.raises(ValueError, match="must be an integer or a list of integers"):
        _eos_token_ids([2, False])


def test_a_non_integer_stop_token_is_refused() -> None:
    with pytest.raises(ValueError, match="must be an integer or a list of integers"):
        _eos_token_ids("</s>")


def test_an_empty_stop_token_list_is_refused() -> None:
    with pytest.raises(ValueError, match="must name at least one token"):
        _eos_token_ids([])


def test_one_stop_token_writes_only_the_scalar(tmp_path: Path) -> None:
    """A single-stop bundle keeps the field set it had before multi-EOS."""
    runtime = _runtime_config(tmp_path, _config(2))
    assert runtime["eos_token_id"] == 2
    assert "eos_token_ids" not in runtime


def test_several_stop_tokens_write_both_fields(tmp_path: Path) -> None:
    """The scalar stays readable by a runtime that predates the list."""
    runtime = _runtime_config(tmp_path, _config([1, 130073]))
    assert runtime["eos_token_id"] == 1
    assert runtime["eos_token_ids"] == [1, 130073]


def test_generation_config_overrides_the_model_config(tmp_path: Path) -> None:
    """A checkpoint may widen its stop set in generation_config.json."""
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [7, 8, 9]}), encoding="utf-8"
    )
    runtime = _runtime_config(tmp_path, _config(2))
    assert runtime["eos_token_id"] == 7
    assert runtime["eos_token_ids"] == [7, 8, 9]


def test_generation_config_must_hold_one_object(tmp_path: Path) -> None:
    (tmp_path / "generation_config.json").write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain one JSON object"):
        _runtime_config(tmp_path, _config(2))
