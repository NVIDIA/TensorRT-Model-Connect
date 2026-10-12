# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS checkpoint conversion memory regression coverage."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")

from families.gpt_oss import model
from families.gpt_oss.config import ModelConfig


class _Tensor:
    def __init__(self, values: np.ndarray):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def to(self, *, device: str, dtype):
        assert device == "cpu"
        return _Tensor(self.values.astype(dtype))

    def numpy(self) -> np.ndarray:
        return self.values


def test_fp16_conversion_releases_source_state_without_fp32_mirror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden, vocab, experts, intermediate = 2, 3, 2, 2
    prefix = "model.layers.0"

    def tensor(*shape: int) -> _Tensor:
        return _Tensor(np.arange(np.prod(shape), dtype=np.float32).reshape(shape))

    state = {
        "model.embed_tokens.weight": tensor(vocab, hidden),
        f"{prefix}.input_layernorm.weight": tensor(hidden),
        f"{prefix}.post_attention_layernorm.weight": tensor(hidden),
        f"{prefix}.self_attn.q_proj.weight": tensor(hidden, hidden),
        f"{prefix}.self_attn.k_proj.weight": tensor(hidden, hidden),
        f"{prefix}.self_attn.v_proj.weight": tensor(hidden, hidden),
        f"{prefix}.self_attn.o_proj.weight": tensor(hidden, hidden),
        f"{prefix}.self_attn.q_proj.bias": tensor(hidden),
        f"{prefix}.self_attn.k_proj.bias": tensor(hidden),
        f"{prefix}.self_attn.v_proj.bias": tensor(hidden),
        f"{prefix}.self_attn.o_proj.bias": tensor(hidden),
        f"{prefix}.mlp.router.weight": tensor(experts, hidden),
        f"{prefix}.mlp.router.bias": tensor(experts),
        f"{prefix}.mlp.experts.gate_up_proj": tensor(experts, hidden, 2 * intermediate),
        f"{prefix}.mlp.experts.gate_up_proj_bias": tensor(experts, 2 * intermediate),
        f"{prefix}.mlp.experts.down_proj": tensor(experts, intermediate, hidden),
        f"{prefix}.mlp.experts.down_proj_bias": tensor(experts, hidden),
        "model.norm.weight": tensor(hidden),
    }

    class FakeModel:
        def state_dict(self):
            return state

    loader = SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: FakeModel())
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(bfloat16="bfloat16", float16=np.float16, float32=np.float32),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=loader),
    )
    config = ModelConfig(
        model_type="gpt_oss",
        vocab_size=vocab,
        hidden_size=hidden,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        raw={"num_local_experts": experts, "num_experts_per_tok": 1},
    )

    weights = model._GptOssModel().load_weights("/unused", config, precision="fp16")

    assert state == {}
    arrays = [value for value in weights.values() if isinstance(value, np.ndarray)]
    assert arrays
    assert {value.dtype for value in arrays} == {np.dtype(np.float16)}
