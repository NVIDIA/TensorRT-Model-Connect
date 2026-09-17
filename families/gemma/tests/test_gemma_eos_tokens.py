# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma names two stop tokens, and the second is the one turns end on."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")

try:
    from families.gemma.config import ModelConfig
    from families.gemma.model import _eos_token_ids, _runtime_config
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires TensorRT", allow_module_level=True)


def _config(eos: object) -> ModelConfig:
    return ModelConfig.from_json(
        json.dumps(
            {
                "model_type": "gemma2",
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "intermediate_size": 16,
                "vocab_size": 32,
                "max_position_embeddings": 256,
                "eos_token_id": eos,
            }
        )
    )


def test_a_single_stop_token_is_normalised() -> None:
    assert _eos_token_ids(1) == [1]
    assert _eos_token_ids([1]) == [1]


def test_both_published_gemma_stop_sets_survive() -> None:
    """Gemma 2 names [1, 107] and Gemma 3 names [1, 106].

    In each the second id is <end_of_turn>. Keeping only the first is what the
    runtime used to do, and a chat turn then never terminated: on the shipped
    gemma-2-2b case the engine emitted 107 and carried on to 1, one token past
    the reference. The extra token decoded to nothing, because both ids are
    special, which is why the manifest passed anyway.
    """
    assert _eos_token_ids([1, 107]) == [1, 107]
    assert _eos_token_ids([1, 106]) == [1, 106]


def test_a_boolean_is_not_a_token_id() -> None:
    with pytest.raises(ValueError, match="must be an integer or a list of integers"):
        _eos_token_ids(True)
    with pytest.raises(ValueError, match="must be an integer or a list of integers"):
        _eos_token_ids([1, False])


def test_an_empty_stop_token_list_is_refused() -> None:
    with pytest.raises(ValueError, match="must name at least one token"):
        _eos_token_ids([])


def test_one_stop_token_writes_only_the_scalar(tmp_path: Path) -> None:
    runtime = _runtime_config(tmp_path, _config(1))
    assert runtime["eos_token_id"] == 1
    assert "eos_token_ids" not in runtime


def test_several_stop_tokens_write_both_fields(tmp_path: Path) -> None:
    runtime = _runtime_config(tmp_path, _config([1, 107]))
    assert runtime["eos_token_id"] == 1
    assert runtime["eos_token_ids"] == [1, 107]


def test_generation_config_widens_the_stop_set(tmp_path: Path) -> None:
    """Gemma ships the full set in generation_config.json."""
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [1, 106]}), encoding="utf-8"
    )
    runtime = _runtime_config(tmp_path, _config(106))
    assert runtime["eos_token_id"] == 1
    assert runtime["eos_token_ids"] == [1, 106]
