# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from families.smollm3 import checkpoint_mapper
from families.smollm3.config import ModelConfig


class _Reader:
    def get_tensor(self, _name: str) -> np.ndarray:
        return np.zeros((3, 4), dtype=np.float32)


def test_checkpoint_embedding_shape_validation_is_always_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    reader = _Reader()
    readers = SimpleNamespace(tensor_map={"model.embed_tokens.weight": reader})
    monkeypatch.setattr(checkpoint_mapper, "_open_safetensors", lambda _path: readers)
    config = ModelConfig.create_tiny(
        "smollm3",
        hidden_size=4,
        vocab_size=5,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=16,
    )

    with pytest.raises(ValueError, match=r"Embedding shape \(3, 4\) != \(5, 4\)"):
        checkpoint_mapper.load_standard_weights(tmp_path, config)
