# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks for the family-owned DeepSeek-OCR build route."""

from __future__ import annotations

from types import SimpleNamespace

from families.deepseek_ocr import model as model_module


class _Writer:
    def __init__(self) -> None:
        self.header: dict[str, str] = {}
        self.bytes: dict[str, bytes] = {}
        self.json: dict[str, dict] = {}
        self.events: list[str] = []

    def set_header(self, **values: str) -> None:
        self.events.append("write:header")
        self.header = values

    def add_bytes(self, name: str, value: bytes) -> None:
        assert name not in self.bytes
        self.events.append(f"write:{name}")
        self.bytes[name] = value

    def add_json(self, name: str, value: dict) -> None:
        assert name not in self.json
        self.json[name] = value


def test_tp_build_writes_rank_plans_and_builds_vision_once(monkeypatch, tmp_path) -> None:
    config = SimpleNamespace(
        model_type="deepseek_vl_v2",
        max_position_embeddings=4096,
        num_hidden_layers=2,
        vocab_size=32,
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=8,
        raw={},
    )
    engine_calls: list[object] = []
    vision_calls: list[object] = []

    class FakeModel:
        def load_weights(self, model_dir, loaded_config, *, precision):
            assert model_dir == str(tmp_path)
            assert loaded_config is config
            assert precision == "fp32"
            return {"weights": True}

        def build_engine(self, loaded_config, weights, max_length, **options):
            assert loaded_config is config
            assert weights == {"weights": True}
            assert max_length == 4096
            assert options["precision"] == "fp32"
            assert options["quant_ctx"] is None
            assert options["verbose"] is False
            parallel = options["parallel_config"]
            writer.events.append(f"build:rank{parallel.rank}")
            engine_calls.append(parallel)
            return f"rank-{parallel.rank}".encode()

        def build_vision_engine(self, model_dir, loaded_config, weights, **options):
            assert model_dir == str(tmp_path)
            assert loaded_config is config
            assert weights == {"weights": True}
            assert options == {"precision": "fp32", "verbose": False}
            writer.events.append("build:vision")
            vision_calls.append((model_dir, loaded_config, weights, options))
            return b"vision"

        def get_vl_config(self, loaded_config):
            assert loaded_config is config
            return {
                "image_token_id": 7,
                "vision_output_dim": 8,
                "prefill_max_length": 64,
            }

    monkeypatch.setattr(
        model_module, "ModelConfig", SimpleNamespace(from_dir=lambda _model_dir: config)
    )
    monkeypatch.setattr(model_module, "_DeepseekOcrModel", FakeModel)
    request = SimpleNamespace(
        model_dir=tmp_path,
        backend="trt",
        dynamic_kv_cache=False,
        task="vision_language_generation",
        precision="fp32",
        max_sequence_length=4096,
        tensor_parallel_size=2,
        context_parallel_size=1,
        quantization=None,
        fp32_layers=(),
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        verbose=False,
    )
    writer = _Writer()

    model_module.build(request, writer)

    assert [(parallel.tp_size, parallel.rank) for parallel in engine_calls] == [(2, 0), (2, 1)]
    assert len(vision_calls) == 1
    assert writer.header == {
        "family": "deepseek_ocr",
        "task": "vision_language_generation",
        "backend": "trt",
    }
    assert writer.bytes == {
        "engine.rank0.plan": b"rank-0",
        "engine.rank1.plan": b"rank-1",
        "vision.plan": b"vision",
    }
    assert writer.events == [
        "write:header",
        "build:rank0",
        "write:engine.rank0.plan",
        "build:rank1",
        "write:engine.rank1.plan",
        "build:vision",
        "write:vision.plan",
    ]
    assert writer.json["runtime.json"]["tensor_parallel_size"] == 2
    assert "_decoder_engine_role" not in config.raw
