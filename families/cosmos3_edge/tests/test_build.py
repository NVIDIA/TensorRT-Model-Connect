# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family bundle contracts; these mocked graphs do not qualify model accuracy."""

from contextlib import contextmanager
from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest

from tensorrt_model_connect.build import BuildRequest
from families.cosmos3_edge import model
from families.cosmos3_edge.tests.test_native_math import _config


class Writer:
    def __init__(self):
        self.sections = {}
        self.header = None

    def set_header(self, **header):
        self.header = header

    def add_bytes(self, name, value):
        assert name not in self.sections
        self.sections[name] = value

    add_json = add_bytes

    @contextmanager
    def open_section(self, name):
        stream = BytesIO()
        yield stream
        self.add_bytes(name, stream.getvalue())


@pytest.fixture
def build_request(tmp_path, monkeypatch):
    from families.cosmos3_edge.edge_llm import builder
    monkeypatch.setattr(builder, "package_present", lambda: False)
    return BuildRequest(model_dir=tmp_path, output_path=tmp_path / "model.bundle",
                        family="cosmos3_edge", task="image_video_text_to_text",
                        precision="fp16", max_sequence_length=64)


@pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
def test_native_bundle_contains_complete_reasoner_contract(build_request, monkeypatch, precision):
    config = _config()
    monkeypatch.setattr(model.ModelConfig, "from_dir", lambda _: config)
    embeddings = np.arange(config.vocab_size * config.hidden_size, dtype=np.float32)
    monkeypatch.setattr(model, "CheckpointWeights", lambda _: {"embed_tokens.weight": embeddings})
    calls = []

    def decoder(c, w, capacity, **kwargs):
        calls.append(("decoder", capacity, kwargs))
        return b"decoder"

    def visual(c, w, patches, **kwargs):
        calls.append(("visual", patches, kwargs))
        return b"visual"

    monkeypatch.setattr(model, "build_decoder", decoder)
    monkeypatch.setattr(model, "build_visual", visual)
    for name in ("tokenizer.json", "chat_template.jinja", "preprocessor_config.json",
                 "video_preprocessor_config.json", "generation_config.json"):
        (build_request.model_dir / name).write_bytes(name.encode())
    writer = Writer()
    model.build(replace(build_request, precision=precision), writer)
    assert writer.header == {"family": "cosmos3_edge", "task": build_request.task, "backend": "trt"}
    assert writer.sections["embeddings.f32"] == embeddings.astype("<f4").tobytes()
    assert writer.sections["runtime.json"]["config"] == config.raw
    assert writer.sections["runtime.json"]["max_cache_length"] == 64
    assert writer.sections["runtime.json"]["max_input_length"] == 64
    assert "video_preprocessor_config.json" in writer.sections
    assert calls[0] == ("decoder", 64, {"precision": precision, "max_input_length": 64, "verbose": False})
    assert calls[1] == ("visual", 256, {"precision": precision, "verbose": False})


@pytest.mark.parametrize("updates", [
    {"task": "image_generation"}, {"tensor_parallel_size": 2}, {"context_parallel_size": 2},
    {"max_batch_size": 2}, {"dynamic_kv_cache": True}, {"image_height": 224},
    {"image_width": 224}, {"video_num_frames": 4}, {"mtp_seq_len": 2},
    {"fp32_layers": (0,)}, {"quantization": "nvfp4"}, {"precision": "int8"},
])
def test_unsupported_requests_fail_before_writing(build_request, updates):
    writer = Writer()
    error = NotImplementedError if updates.get("dynamic_kv_cache") else ValueError
    with pytest.raises(error):
        model.build(replace(build_request, **updates), writer)
    assert writer.header is None
    assert not writer.sections


def test_context_capacity_is_not_silently_clamped(build_request, monkeypatch):
    monkeypatch.setattr(model.ModelConfig, "from_dir", lambda _: _config())
    with pytest.raises(ValueError, match="context capacity"):
        model.build(replace(build_request, max_sequence_length=129), Writer())
