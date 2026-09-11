# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned tests for the plain SmolLM3 build entry point."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("tensorrt")

from families.smollm3 import model  # noqa: E402
from tensorrt_model_connect import BuildRequest  # noqa: E402


class RecordingWriter:
    def __init__(self) -> None:
        self.header: dict[str, object] | None = None
        self.sections: dict[str, bytes] = {}
        self.json_sections: dict[str, object] = {}

    def set_header(self, **header: object) -> None:
        self.header = header

    def add_bytes(self, name: str, value: bytes) -> None:
        assert name not in self.sections
        self.sections[name] = value

    def add_json(self, name: str, value: object) -> None:
        assert name not in self.json_sections
        self.json_sections[name] = value


def _checkpoint(root: Path) -> Path:
    root.mkdir()
    config = {
        "model_type": "smollm3",
        "architectures": ["SmolLM3ForCausalLM"],
        "vocab_size": 128256,
        "hidden_size": 2048,
        "intermediate_size": 11008,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-6,
        "rope_theta": 5_000_000.0,
        "bos_token_id": 128000,
        "eos_token_id": 128012,
        "pad_token_id": 128004,
        "tie_word_embeddings": True,
        "max_position_embeddings": 65536,
        "hidden_act": "silu",
        "no_rope_layer_interval": 4,
        "no_rope_layers": [int((index + 1) % 4 != 0) for index in range(36)],
        "pretraining_tp": 2,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer.json").write_bytes(b"tokenizer")
    return root


def _request(model_dir: Path, **changes: object) -> BuildRequest:
    request = BuildRequest(
        model_dir=model_dir,
        output_path=model_dir.parent / "smollm3.bundle",
        family="smollm3",
        task="text_generation",
        precision="bf16",
        max_sequence_length=256,
    )
    return replace(request, **changes)


def test_build_emits_the_family_owned_split_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    calls: list[tuple[str, int, str]] = []
    monkeypatch.setattr(model, "load_standard_weights", lambda *args, **kwargs: {})

    def build_engine(config, weights, length: int, *, precision: str, verbose: bool) -> bytes:
        role = str(config.raw["_decoder_engine_role"])
        calls.append((role, length, precision))
        return role.encode()

    monkeypatch.setattr(model, "_build_engine", build_engine)
    writer = RecordingWriter()

    model.build(_request(checkpoint), writer)

    assert writer.header == {
        "family": "smollm3",
        "task": "text_generation",
        "backend": "trt",
    }
    assert writer.sections == {
        "prefill.plan": b"prefill",
        "engine.plan": b"decode",
        "tokenizer.json": b"tokenizer",
    }
    assert calls == [("prefill", 256, "bf16"), ("decode", 256, "bf16")]
    runtime = writer.json_sections["runtime.json"]
    assert isinstance(runtime, dict)
    assert runtime["decoder_engine_layout"] == "split"
    assert runtime["max_cache_length"] == 256
    assert runtime["precision"] == "bf16"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"task": "classification"}, "task=text_generation"),
        ({"dynamic_kv_cache": True}, "dynamic_kv_cache"),
        ({"precision": "fp8"}, "fp32, fp16, or bf16"),
        ({"max_batch_size": 2}, "max_batch_size"),
        ({"tensor_parallel_size": 2}, "tensor-parallel"),
        ({"context_parallel_size": 2}, "context parallelism"),
        ({"quantization": "fp8"}, "quantized"),
        ({"image_height": 256}, "image_height"),
        ({"image_width": 256}, "image_width"),
        ({"video_num_frames": 8}, "video_num_frames"),
    ],
)
def test_build_rejects_unsupported_profiles(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    with pytest.raises((ValueError, NotImplementedError), match=message):
        model.build(_request(checkpoint, **changes), RecordingWriter())
