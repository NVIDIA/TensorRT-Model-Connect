# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")

from families.mixtral import model


def test_active_case_routes_fp32_layers_to_builder(monkeypatch, tmp_path) -> None:
    manifest = json.loads(
        (Path(__file__).parent / "manifests" / "mixtral-stories-15m.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["fp32_layers"] == [3, 4, 5]

    config = SimpleNamespace(
        model_type="mixtral",
        max_position_embeddings=512,
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=6,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        raw={},
    )
    observed = []

    class FakeModel:
        @staticmethod
        def load_weights(_model_dir, loaded_config):
            observed.append(tuple(loaded_config.raw["_fp32_layers"]))
            return {}

        @staticmethod
        def build_engine(loaded_config, _weights, _length, **_kwargs):
            observed.append(tuple(loaded_config.raw["_fp32_layers"]))
            return b"mixtral-plan"

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
    monkeypatch.setattr(model, "_MixtralModel", FakeModel)
    writer = Writer()
    request = SimpleNamespace(
        model_dir=tmp_path,
        backend="trt",
        dynamic_kv_cache=False,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        task="text_generation",
        tensor_parallel_size=1,
        quantization=None,
        fp32_layers=tuple(manifest["fp32_layers"]),
        precision=manifest["precision"],
        max_sequence_length=manifest["max_sequence_length"],
        verbose=False,
    )

    model.build(request, writer)

    assert observed == [(3, 4, 5), (3, 4, 5)]
    assert writer.sections["engine.plan"] == b"mixtral-plan"
