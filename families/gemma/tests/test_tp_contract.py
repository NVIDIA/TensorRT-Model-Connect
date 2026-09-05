# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from families.gemma.checkpoint_mapper import WeightDict
from families.gemma.parallel import ParallelConfig, shard_standard_decoder_weights


def test_swiglu_weights_are_sharded_for_tp4() -> None:
    config = SimpleNamespace(
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=16,
        head_dim=2,
    )
    weights = WeightDict(
        {
            "layer.0.w_q": np.arange(32, dtype=np.float32).reshape(4, 8),
            "layer.0.w_o": np.arange(32, dtype=np.float32).reshape(8, 4),
            "layer.0.q_norm": np.arange(8, dtype=np.float32),
            "layer.0.w_gate": np.arange(64, dtype=np.float32).reshape(4, 16),
            "layer.0.w_up": np.arange(64, dtype=np.float32).reshape(4, 16),
            "layer.0.w_down": np.arange(64, dtype=np.float32).reshape(16, 4),
            "_attention_size": 8,
            "_kv_attention_size": 8,
            "_mlp_size": 16,
        }
    )

    sharded = shard_standard_decoder_weights(config, weights, ParallelConfig(4, 2))

    assert sharded["layer.0.w_q"].shape == (4, 2)
    assert sharded["layer.0.w_o"].shape == (2, 4)
    assert sharded["layer.0.q_norm"].shape == (2,)
    assert sharded["layer.0.w_gate"].shape == (4, 4)
    assert sharded["layer.0.w_up"].shape == (4, 4)
    assert sharded["layer.0.w_down"].shape == (4, 4)
    assert sharded["_attention_size"] == 2
    assert sharded["_kv_attention_size"] == 2
    assert sharded["_mlp_size"] == 4
