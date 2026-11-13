# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM checkpoint loading memory regression coverage."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")

from families.glm import model
from families.glm.config import ModelConfig


def test_layers_are_converted_one_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Large temporary projection arrays cannot overlap across decoder layers."""
    hidden, vocab, intermediate = 4, 7, 6
    lock = threading.Lock()
    active = 0
    maximum = 0

    def load(_readers, name: str) -> np.ndarray:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(0.005)
            if name == "model.embed_tokens.weight":
                shape = (vocab, hidden)
            elif name.endswith("gate_up_proj.weight"):
                shape = (2 * intermediate, hidden)
            elif name.endswith("down_proj.weight"):
                shape = (hidden, intermediate)
            elif name.endswith(".weight") and ".self_attn." in name:
                shape = (hidden, hidden)
            else:
                shape = (hidden,)
            return np.ones(shape, dtype=np.float32)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(model, "_open_safetensors", lambda _path: object())
    monkeypatch.setattr(model, "_load_tensor", load)
    monkeypatch.setattr(
        model,
        "_has_tensor",
        lambda _readers, name: name != "lm_head.weight",
    )
    config = ModelConfig(
        model_type="glm",
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
    )

    weights = model._GlmModel().load_weights("/unused", config, precision="fp16")

    assert weights["layer.1.w_down"].shape == (intermediate, hidden)
    assert maximum == 1
