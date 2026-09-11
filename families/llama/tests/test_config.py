# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from families.llama.config import ModelConfig


def test_scalar_eos_token_id() -> None:
    config = ModelConfig.create_tiny(model_type="llama", eos_token_id=1)

    assert config.eos_token_id == 1
    assert config.eos_token_ids == (1,)


def test_list_eos_token_id_preserves_every_id() -> None:
    # HF configs allow eos_token_id to be a list of stop tokens (e.g. Llama
    # 3.1+, MiniCPM5). Every id must reach the runtime bundle: the model may
    # naturally emit any one of them as its real per-turn stop token (for
    # Llama 3.1-Instruct, that's <|eot_id|>, the *last* id in the list, not
    # the first), so truncating to one id would silently break stopping.
    config = ModelConfig.create_tiny(model_type="llama", eos_token_id=[1, 130073])

    assert config.eos_token_id == 1
    assert config.eos_token_ids == (1, 130073)


def test_empty_eos_token_id_list_falls_back() -> None:
    config = ModelConfig.create_tiny(model_type="llama", eos_token_id=[])

    assert config.eos_token_id == -1
    assert config.eos_token_ids == ()


def test_missing_eos_token_id_falls_back() -> None:
    config = ModelConfig.create_tiny(model_type="llama")

    assert config.eos_token_id == -1
    assert config.eos_token_ids == ()
