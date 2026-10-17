# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from families.embeddinggemma2.config import Config


def test_all_text_weight_shapes_match_pinned_official_safetensors_header():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/checkpoint-metadata.json").read_text(encoding="utf-8")
    )
    config = Config.from_dict(fixture["config"])
    assert config.hidden_size == 512 and config.embedding_dim == 768
    assert config.layer(5)[:2] == (512, 1)
    assert config.layer(0)[:2] == (256, 2)
    assert {name: list(shape) for name, shape in config.weight_shapes().items()} == fixture[
        "text_weight_shapes"
    ]
