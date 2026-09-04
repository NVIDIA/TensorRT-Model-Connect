# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from families.t5.config import ModelConfig


def test_zero_token_ids_are_valid(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "t5",
                "vocab_size": 32,
                "d_model": 16,
                "d_ff": 32,
                "num_layers": 1,
                "num_heads": 1,
                "bos_token_id": 0,
                "eos_token_id": 1,
                "pad_token_id": 0,
            }
        ),
        encoding="utf-8",
    )

    config = ModelConfig.from_dir(tmp_path)

    assert config.bos_token_id == 0
    assert config.eos_token_id == 1
    assert config.pad_token_id == 0
