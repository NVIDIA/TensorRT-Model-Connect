# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import importlib
import struct

import numpy as np
import pytest

from tensorrt_model_connect.families.qwen3_omni.audio_runtime import (
    TalkerRequest,
    _WORKER_ERROR,
    _WORKER_MAGIC,
    _WORKER_OK,
    _WORKER_READY,
    _WORKER_REQUEST_HEADER,
    _WORKER_RESPONSE_HEADER,
    _chatml,
    _read_request,
    _serve_worker,
    _thinker_forward_input_ids,
)
from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.families.qwen3_omni.plugin import (
    Qwen3OmniPlugin,
    _talker_model_locator,
)


def _payload(prompt: str, assistant: str) -> bytes:
    prompt_bytes = prompt.encode("utf-8")
    assistant_bytes = assistant.encode("utf-8")
    return (
        struct.pack("<II", len(prompt_bytes), len(assistant_bytes)) + prompt_bytes + assistant_bytes
    )


def _worker_input(*requests: bytes) -> io.BytesIO:
    framed = bytearray()
    for request in requests:
        framed.extend(_WORKER_REQUEST_HEADER.pack(_WORKER_MAGIC, len(request)))
        framed.extend(request)
    framed.extend(_WORKER_REQUEST_HEADER.pack(_WORKER_MAGIC, 0))
    return io.BytesIO(framed)


def _worker_responses(payload: bytes) -> list[tuple[int, bytes, float]]:
    stream = io.BytesIO(payload)
    responses = []
    while header := stream.read(_WORKER_RESPONSE_HEADER.size):
        magic, status, size, talker_ms = _WORKER_RESPONSE_HEADER.unpack(header)
        assert magic == _WORKER_MAGIC
        body = stream.read(size)
        assert len(body) == size
        responses.append((status, body, talker_ms))
    return responses


def test_talker_request_preserves_prompt_and_trims_generated_stop_marker() -> None:
    request = _read_request(_payload("Say hello.", "Hello from Qwen-Omni!<|im_end|>ignored"))

    assert request == TalkerRequest(prompt="Say hello.", assistant_text="Hello from Qwen-Omni!")


def test_talker_request_rejects_empty_assistant_text() -> None:
    with pytest.raises(ValueError, match="no speakable assistant text"):
        _read_request(_payload("Say hello.", "<|im_end|>"))


def test_talker_request_rejects_truncated_payload() -> None:
    with pytest.raises(ValueError, match="expected"):
        _read_request(struct.pack("<II", 3, 4) + b"abc")


def test_talker_chatml_contains_model_roles_and_exact_text() -> None:
    rendered = _chatml(TalkerRequest(prompt="question", assistant_text="answer"))

    assert "<|im_start|>system\n" in rendered
    assert "<|im_start|>user\nquestion<|im_end|>" in rendered
    assert "<|im_start|>assistant\nanswer<|im_end|>" in rendered
    assert rendered.endswith("answer<|im_end|>")


def test_talker_does_not_forward_selected_thinker_eos() -> None:
    sequence_ids = np.array([[10, 11, 151645]])

    assert _thinker_forward_input_ids(sequence_ids).tolist() == [[10, 11]]


def test_persistent_talker_worker_initializes_once_for_multiple_requests() -> None:
    lifecycle = {"initializations": 0, "requests": 0}

    class FakeTalker:
        def __init__(self, model_id: str, revision: str, max_frames: int) -> None:
            assert (model_id, revision, max_frames) == ("model", "revision", 4)
            lifecycle["initializations"] += 1

        def generate_codes(self, request: TalkerRequest) -> np.ndarray:
            lifecycle["requests"] += 1
            value = lifecycle["requests"]
            assert request.assistant_text in {"first", "second"}
            return np.full((1, 2), value, dtype="<i4")

    output = io.BytesIO()
    _serve_worker(
        "model",
        "revision",
        4,
        _worker_input(_payload("prompt", "first"), _payload("prompt", "second")),
        output,
        FakeTalker,
    )

    responses = _worker_responses(output.getvalue())
    assert [response[0] for response in responses] == [_WORKER_READY, _WORKER_OK, _WORKER_OK]
    assert np.frombuffer(responses[1][1], dtype="<i4").tolist() == [1, 1]
    assert np.frombuffer(responses[2][1], dtype="<i4").tolist() == [2, 2]
    assert lifecycle == {"initializations": 1, "requests": 2}


def test_persistent_talker_worker_reports_request_error_and_continues() -> None:
    lifecycle = {"initializations": 0, "requests": 0}

    class RecoveringTalker:
        def __init__(self, _model_id: str, _revision: str, _max_frames: int) -> None:
            lifecycle["initializations"] += 1

        def generate_codes(self, _request: TalkerRequest) -> np.ndarray:
            lifecycle["requests"] += 1
            if lifecycle["requests"] == 1:
                raise RuntimeError("intentional request failure")
            return np.array([[7, 8]], dtype="<i4")

    output = io.BytesIO()
    _serve_worker(
        "model",
        "",
        4,
        _worker_input(_payload("prompt", "first"), _payload("prompt", "second")),
        output,
        RecoveringTalker,
    )

    responses = _worker_responses(output.getvalue())
    assert [response[0] for response in responses] == [
        _WORKER_READY,
        _WORKER_ERROR,
        _WORKER_OK,
    ]
    assert b"intentional request failure" in responses[1][1]
    assert np.frombuffer(responses[2][1], dtype="<i4").tolist() == [7, 8]
    assert lifecycle == {"initializations": 1, "requests": 2}


def test_talker_model_locator_pins_hugging_face_snapshot(tmp_path) -> None:
    snapshot = tmp_path / "models--Qwen--Qwen3-Omni-30B-A3B-Instruct" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)

    assert _talker_model_locator(snapshot) == (
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "abc123",
    )


def test_talker_model_locator_preserves_deliberate_local_directory(tmp_path) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()

    assert _talker_model_locator(model_dir) == (str(model_dir.resolve()), "")


def test_bundle_config_persists_portable_talker_locator() -> None:
    plugin = Qwen3OmniPlugin()
    plugin._talker_model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    plugin._talker_model_revision = "abc123"

    overrides = plugin.get_bundle_config_overrides(ModelConfig.create_tiny("qwen3_omni"))

    assert overrides["omni_talker_model_id"] == "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    assert overrides["omni_talker_model_revision"] == "abc123"


def test_thinker_load_weights_preserves_bf16_storage(monkeypatch, tmp_path) -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.qwen3_omni.plugin")
    config = ModelConfig.create_tiny("qwen3_omni")

    class FakeReader:
        @staticmethod
        def keys() -> list[str]:
            return []

    def has_tensor(_readers, key: str) -> bool:
        return (
            key == "model.thinker.embed_tokens.weight"
            or key == "model.thinker.norm.weight"
            or key == "lm_head.weight"
            or key.startswith("model.thinker.layers.")
        )

    def load_tensor(_readers, key: str) -> np.ndarray:
        if key == "model.thinker.embed_tokens.weight":
            shape = (config.vocab_size, config.hidden_size)
        elif key == "lm_head.weight":
            shape = (config.vocab_size, config.hidden_size)
        elif key.endswith(("input_layernorm.weight", "post_attention_layernorm.weight")):
            shape = (config.hidden_size,)
        elif key == "model.thinker.norm.weight":
            shape = (config.hidden_size,)
        elif key.endswith("mlp.gate.weight"):
            shape = (8, config.hidden_size)
        elif key.endswith(("gate_proj.weight", "up_proj.weight")):
            shape = (config.intermediate_size, config.hidden_size)
        elif key.endswith("down_proj.weight"):
            shape = (config.hidden_size, config.intermediate_size)
        else:
            shape = (config.hidden_size, config.hidden_size)
        return np.ones(shape, dtype=np.float32)

    monkeypatch.setattr(plugin_module, "_open_safetensors", lambda _path: [FakeReader()])
    monkeypatch.setattr(plugin_module, "_has_tensor", has_tensor)
    monkeypatch.setattr(plugin_module, "_load_tensor", load_tensor)

    weights = plugin_module.Qwen3OmniPlugin().load_weights(
        str(tmp_path), config, precision="bf16")

    assert weights["embedding"].dtype.name == "bfloat16"
    assert weights["layer.0.w_q"].dtype.name == "bfloat16"
    assert weights["layer.0.router"].dtype.name == "bfloat16"
    assert weights["layer.0.experts.w_gate"].dtype.name == "bfloat16"
    assert weights["layer.0.experts.w_gate"].shape == (8, 16, 32)
    assert weights["layer.0.experts.w_down"].shape == (8, 32, 16)
    assert weights["w_out"].dtype.name == "bfloat16"
    assert weights["final_norm"].dtype == np.float32


def test_thinker_moe_batches_only_routed_expert_multiplies(monkeypatch) -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.qwen3_omni.plugin")
    constants = []
    matrix_multiplies = []
    gathers = []

    class Tensor:
        def __init__(self, name: str, shape=(), dtype="bf16"):
            self.name = name
            self.shape = tuple(shape)
            self.dtype = dtype

    class Layer:
        def __init__(self, output: Tensor):
            self.output = output

        def get_output(self, _index):
            return self.output

        @property
        def reshape_dims(self):
            return self.output.shape

        @reshape_dims.setter
        def reshape_dims(self, shape):
            self.output.shape = tuple(shape)

    class Network:
        def add_shuffle(self, tensor):
            return Layer(Tensor(f"shuffle({tensor.name})", tensor.shape, tensor.dtype))

        def add_gather(self, data, indices, axis):
            gathers.append((data.name, indices.name, axis))
            return Layer(Tensor(f"gather({data.name})", dtype=data.dtype))

        def add_matrix_multiply(self, lhs, _lhs_op, rhs, _rhs_op):
            matrix_multiplies.append((lhs.name, rhs.name))
            return Layer(Tensor(f"mm({lhs.name},{rhs.name})", dtype=lhs.dtype))

        def add_activation(self, tensor, _operation):
            return Layer(Tensor(f"activation({tensor.name})", dtype=tensor.dtype))

        def add_elementwise(self, lhs, rhs, _operation):
            return Layer(Tensor(f"elementwise({lhs.name},{rhs.name})", dtype=lhs.dtype))

        def add_reduce(self, tensor, _operation, _axes, keep_dims):
            del keep_dims
            return Layer(Tensor(f"reduce({tensor.name})", dtype=tensor.dtype))

    def add_constant(_network, shape, values, dtype=np.float32):
        del values, dtype
        tensor = Tensor(f"weight{len(constants)}", shape)
        constants.append(tensor)
        return tensor

    monkeypatch.setattr(plugin_module.graph_ops, "add_constant", add_constant)
    network = Network()
    output = plugin_module._add_routed_swiglu_experts(
        network,
        Tensor("input", (-1, 16)),
        Tensor("top_indices", (-1, 2), "int32"),
        Tensor("routing_weights", (-1, 2)),
        hidden_size=16,
        top_k=2,
        w_gate=np.ones((8, 16, 32), dtype=np.float32),
        w_up=np.ones((8, 16, 32), dtype=np.float32),
        w_down=np.ones((8, 32, 16), dtype=np.float32),
    )

    assert output.dtype == "bf16"
    assert len(matrix_multiplies) == 3
    assert all(rhs.startswith("gather(weight") for _lhs, rhs in matrix_multiplies)
    assert [entry for entry in gathers if entry[0].startswith("weight")] == [
        ("weight0", "top_indices", 0),
        ("weight1", "top_indices", 0),
        ("weight2", "top_indices", 0),
    ]
