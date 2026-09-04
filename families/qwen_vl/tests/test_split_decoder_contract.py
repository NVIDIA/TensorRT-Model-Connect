# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

from families.qwen_vl import model


class RecordingWriter:
    def __init__(self) -> None:
        self.sections = {}

    def set_header(self, **_header) -> None:
        pass

    def add_bytes(self, name, value) -> None:
        self.sections[name] = value

    def add_json(self, name, value) -> None:
        self.sections[name] = value


def test_build_marks_both_decoder_plans_as_one_active_split_build(monkeypatch, tmp_path) -> None:
    config = SimpleNamespace(
        model_type="qwen2_5_vl",
        max_position_embeddings=128,
        raw={},
        num_hidden_layers=2,
        vocab_size=32,
        hidden_size=16,
        bos_token_id=1,
        eos_token_id=2,
    )
    roles = []

    class FamilyModel:
        @staticmethod
        def load_weights(_model_dir, _config):
            return {}

        @staticmethod
        def build_engine(active_config, *_args, **_kwargs):
            roles.append(
                (
                    active_config.raw["_decoder_engine_role"],
                    active_config.raw["_active_split_decoder_build"],
                )
            )
            return roles[-1][0].encode()

        @staticmethod
        def build_vision_engine(*_args, **_kwargs):
            return b"vision"

        @staticmethod
        def get_vl_config(_config):
            return {}

    monkeypatch.setattr(model.ModelConfig, "from_dir", lambda _path: config)
    monkeypatch.setattr(model, "_QwenVLModel", FamilyModel)
    monkeypatch.setattr(model, "_tokenizer_runtime_contract", lambda _path: {})
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"bos_token_id": 1, "eos_token_id": [2, 3]}), encoding="utf-8"
    )
    request = SimpleNamespace(
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        task="vision_language_generation",
        tensor_parallel_size=1,
        quantization=None,
        fp32_layers=(),
        model_dir=tmp_path,
        precision="bf16",
        max_sequence_length=64,
        verbose=False,
    )
    writer = RecordingWriter()

    model.build(request, writer)

    assert roles == [("prefill", True), ("decode", True)]
    assert "_decoder_engine_role" not in config.raw
    assert "_active_split_decoder_build" not in config.raw
    assert writer.sections["prefill.plan"] == b"prefill"
    assert writer.sections["engine.plan"] == b"decode"
    assert writer.sections["runtime.json"]["id_eos_ids"] == [2, 3]
