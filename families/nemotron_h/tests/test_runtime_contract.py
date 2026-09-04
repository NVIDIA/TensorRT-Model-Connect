# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from families.nemotron_h.model import _runtime_config, _stop_token_ids


def test_runtime_keeps_a_scalar_eos_and_every_declared_stop_token(tmp_path) -> None:
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [2, 11, 12]}), encoding="utf-8"
    )
    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        bos_token_id=1,
        eos_token_id=12,
        pad_token_id=0,
    )
    family_model = SimpleNamespace(get_bundle_config_overrides=lambda _config: {})

    runtime = _runtime_config(tmp_path, config, family_model)

    assert runtime["eos_token_id"] == 2
    assert runtime["stop_token_ids"] == [2, 11, 12]


@pytest.mark.parametrize("value", [[], [2, 2], [True], [32], "2"])
def test_stop_token_contract_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError, match="eos_token_id"):
        _stop_token_ids(value, 32)
