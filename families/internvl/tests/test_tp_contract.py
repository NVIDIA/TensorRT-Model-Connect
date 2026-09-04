# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from families.internvl import model
from families.internvl.checkpoint_mapper import WeightDict
from families.internvl.parallel import ParallelConfig, shard_standard_decoder_weights
from families.internvl.tests.test_e2e import CASES, _official_prompt


def test_swiglu_weights_are_sharded_for_each_rank() -> None:
    config = SimpleNamespace(
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=8,
    )
    weights = WeightDict(
        {
            "layer.0.w_q": np.arange(32, dtype=np.float32).reshape(4, 8),
            "layer.0.w_o": np.arange(32, dtype=np.float32).reshape(8, 4),
            "layer.0.w_gate": np.arange(32, dtype=np.float32).reshape(4, 8),
            "layer.0.w_up": np.arange(32, dtype=np.float32).reshape(4, 8),
            "layer.0.w_down": np.arange(32, dtype=np.float32).reshape(8, 4),
            "_attention_size": 8,
            "_kv_attention_size": 8,
            "_mlp_size": 8,
        }
    )
    sharded = shard_standard_decoder_weights(config, weights, ParallelConfig(2, 1))
    assert sharded["layer.0.w_q"].shape == (4, 4)
    assert sharded["layer.0.w_o"].shape == (4, 4)
    assert sharded["layer.0.w_gate"].shape == (4, 4)
    assert sharded["layer.0.w_up"].shape == (4, 4)
    assert sharded["layer.0.w_down"].shape == (4, 4)
    assert sharded["_attention_size"] == 4
    assert sharded["_kv_attention_size"] == 4
    assert sharded["_mlp_size"] == 4


def test_build_emits_one_dual_profile_plan_per_rank(monkeypatch, tmp_path) -> None:
    config = SimpleNamespace(
        model_type="internvl",
        max_position_embeddings=4096,
        num_hidden_layers=2,
        vocab_size=32,
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=8,
        raw={},
    )
    ranks = []

    class FakeModel:
        @staticmethod
        def load_weights(_model_dir, _config):
            return WeightDict()

        @staticmethod
        def build_engine(_config, _weights, _length, **kwargs):
            parallel = kwargs["parallel_config"]
            ranks.append(parallel.rank)
            return f"rank-{parallel.rank}".encode()

        @staticmethod
        def build_vision_engine(*_args, **_kwargs):
            return b"vision"

        @staticmethod
        def get_vl_config(_config):
            return {"image_token_id": 7, "vision_output_dim": 8, "prefill_max_length": 16}

    class Writer:
        def __init__(self):
            self.sections = {}

        @staticmethod
        def set_header(**_kwargs):
            return None

        def add_bytes(self, name, value):
            self.sections[name] = value

        def add_json(self, name, value):
            self.sections[name] = value

    monkeypatch.setattr(model.ModelConfig, "from_dir", lambda _path: config)
    monkeypatch.setattr(model, "_InternVLModel", FakeModel)
    monkeypatch.setattr(model, "_tokenizer_runtime_contract", lambda _path: {})
    writer = Writer()
    request = SimpleNamespace(
        model_dir=tmp_path,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        task="vision_language_generation",
        tensor_parallel_size=2,
        quantization=None,
        fp32_layers=(),
        precision="fp16",
        max_sequence_length=16,
        verbose=False,
    )

    model.build(request, writer)

    assert ranks == [0, 1]
    assert writer.sections["engine.rank0.plan"] == b"rank-0"
    assert writer.sections["engine.rank1.plan"] == b"rank-1"
    assert "engine.plan" not in writer.sections
    assert "prefill.plan" not in writer.sections
    assert writer.sections["runtime.json"]["tensor_parallel_size"] == 2
    assert writer.sections["vision.plan"] == b"vision"

    ranks.clear()
    request.tensor_parallel_size = 1
    writer = Writer()
    model.build(request, writer)
    assert ranks == [-1, -1]
    assert writer.sections["engine.plan"] == b"rank--1"
    assert writer.sections["prefill.plan"] == b"rank--1"
    assert not any(name.startswith("engine.rank") for name in writer.sections)
    assert writer.sections["runtime.json"]["tensor_parallel_size"] == 1


def test_tp_manifests_remain_active() -> None:
    assert CASES["internvl3-2b-tp2"][1]["tensor_parallel_size"] == 2
    assert CASES["internvl3-8b-tp4"][1]["tensor_parallel_size"] == 4


def test_official_prompt_adds_image_placeholder_without_changing_user_text() -> None:
    user_prompt = "What color is the vehicle?"

    class Processor:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            assert messages[0]["content"][1] == {"type": "text", "text": user_prompt}
            return f"<IMG_CONTEXT>\n{user_prompt}\nassistant"

    prompt = _official_prompt(Processor(), user_prompt)
    assert prompt.count(user_prompt) == 1
    assert "<IMG_CONTEXT>" in prompt
