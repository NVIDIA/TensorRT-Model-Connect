# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from families.qwen_vl import model


class RecordingWriter:
    def __init__(self) -> None:
        self.sections = {}
        self.events = []

    def set_header(self, **header) -> None:
        self.events.append("write:header")
        self.header = header

    def add_bytes(self, name, value) -> None:
        self.events.append(f"write:{name}")
        self.sections[name] = value

    def add_json(self, name, value) -> None:
        self.events.append(f"write:{name}")
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
        backend="trt",
        dynamic_kv_cache=False,
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


@pytest.mark.parametrize(("model_type", "tp_size"), [("qwen2_5_vl", 2), ("qwen3_vl", 4)])
def test_build_streams_tp_rank_plans_and_builds_vision_once(
    monkeypatch, tmp_path, model_type, tp_size
) -> None:
    config = SimpleNamespace(
        model_type=model_type,
        max_position_embeddings=512,
        raw={},
        num_hidden_layers=2,
        vocab_size=32,
        hidden_size=16,
        bos_token_id=1,
        eos_token_id=2,
    )
    writer = RecordingWriter()
    engine_calls = []
    vision_calls = []

    class FamilyModel:
        @staticmethod
        def load_weights(_model_dir, active_config):
            assert active_config.raw["_fp32_layers"] == (28, 30, 53)
            return {"weights": True}

        @staticmethod
        def build_engine(active_config, weights, max_length, **options):
            assert active_config.raw["_fp32_layers"] == (28, 30, 53)
            assert weights == {"weights": True}
            assert max_length == 256
            assert options["precision"] == "fp32"
            parallel = options["parallel_config"]
            writer.events.append(f"build:rank{parallel.rank}")
            engine_calls.append(parallel)
            return f"rank-{parallel.rank}".encode()

        @staticmethod
        def build_vision_engine(_model_dir, active_config, weights, **options):
            assert active_config.raw["_fp32_layers"] == (28, 30, 53)
            assert weights == {"weights": True}
            assert options == {"precision": "fp32", "verbose": False}
            writer.events.append("build:vision")
            vision_calls.append(True)
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
        backend="trt",
        dynamic_kv_cache=False,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        task="vision_language_generation",
        tensor_parallel_size=tp_size,
        quantization=None,
        fp32_layers=(28, 30, 53),
        model_dir=tmp_path,
        precision="fp32",
        max_sequence_length=256,
        verbose=False,
    )

    model.build(request, writer)

    assert [(parallel.tp_size, parallel.rank) for parallel in engine_calls] == [
        (tp_size, rank) for rank in range(tp_size)
    ]
    assert len(vision_calls) == 1
    assert {name: value for name, value in writer.sections.items() if isinstance(value, bytes)} == {
        **{f"engine.rank{rank}.plan": f"rank-{rank}".encode() for rank in range(tp_size)},
        "vision.plan": b"vision",
    }
    expected_events = ["write:header"]
    for rank in range(tp_size):
        expected_events.extend([f"build:rank{rank}", f"write:engine.rank{rank}.plan"])
    expected_events.extend(["build:vision", "write:vision.plan", "write:runtime.json"])
    assert writer.events == expected_events
    assert writer.sections["runtime.json"]["tensor_parallel_size"] == tp_size
    assert "_decoder_engine_role" not in config.raw
    assert "_active_split_decoder_build" not in config.raw


@pytest.mark.parametrize(
    ("raw", "tp_size", "precision", "expected_precision", "deepstack_levels"),
    [
        ({}, 2, "fp32", "fp32", 0),
        (
            {
                "vision_config": {"deepstack_visual_indexes": [5, 11, 17]},
                "_fp32_layers": [model._TEXT_DECODER_COMPONENT],
            },
            4,
            "fp16",
            "fp32",
            3,
        ),
    ],
)
def test_family_model_routes_tp_decoder_with_component_precision(
    monkeypatch, raw, tp_size, precision, expected_precision, deepstack_levels
) -> None:
    calls = {}

    def build_tp(config, weights, max_length, **options):
        calls.update(
            config=config,
            weights=weights,
            max_length=max_length,
            options=options,
        )
        return b"tp-plan"

    monkeypatch.setattr(model, "build_qwen_vl_tp_decoder_engine", build_tp)
    config = SimpleNamespace(raw=raw)
    parallel = model.ParallelConfig(tp_size=tp_size, rank=1)

    result = model._QwenVLModel().build_engine(
        config,
        {"weights": True},
        256,
        precision=precision,
        parallel_config=parallel,
    )

    assert result == b"tp-plan"
    assert calls["config"] is config
    assert calls["weights"] == {"weights": True}
    assert calls["max_length"] == 256
    assert calls["options"]["parallel_config"] == parallel
    assert calls["options"]["precision"] == expected_precision
    assert calls["options"]["deepstack_num_levels"] == deepstack_levels
