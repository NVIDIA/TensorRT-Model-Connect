# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned LFM2.5-VL decoder geometry and position encoding contract."""

import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from families.lfm2_vl.config import validate_dense_lfm2_vl_config


def raw_config():
    return json.loads((Path(__file__).parent / "fixtures/config.json").read_text())["text_config"]


def test_checkpoint_geometry():
    cfg = validate_dense_lfm2_vl_config(SimpleNamespace(raw=raw_config()))
    assert (cfg.hidden_size, cfg.vocab_size, cfg.intermediate_size) == (2048, 65536, 8192)
    assert (cfg.num_attention_layers, cfg.num_conv_layers) == (6, 10)
    assert cfg.rope_theta == 1_000_000
    assert cfg.default_cache_length == 32768


@pytest.mark.parametrize("field", ["rope_theta", "theta"])
def test_rejects_inconsistent_rope(field):
    raw = raw_config()
    raw[field] = 10000
    with pytest.raises(ValueError, match="must agree"):
        validate_dense_lfm2_vl_config(SimpleNamespace(raw=raw))
