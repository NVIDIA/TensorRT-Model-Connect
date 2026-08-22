# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib

import numpy as np
from tensorrt_model_connect.config import ModelConfig


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

    plugin = plugin_module.Qwen3OmniPlugin()

    def unexpected_multimodal_probe(*_args, **_kwargs):
        raise AssertionError("text-only weight loading must not inspect multimodal components")

    monkeypatch.setattr(plugin, "_detect_audio_encoder", unexpected_multimodal_probe)
    monkeypatch.setattr(plugin, "_detect_talker", unexpected_multimodal_probe)
    monkeypatch.setattr(plugin, "_detect_code2wav", unexpected_multimodal_probe)

    weights = plugin.load_weights(str(tmp_path), config, precision="bf16")

    assert weights["embedding"].dtype.name == "bfloat16"
    assert weights["layer.0.w_q"].dtype.name == "bfloat16"
    assert weights["layer.0.router"].dtype.name == "bfloat16"
    assert weights["layer.0.experts.w_gate"].dtype.name == "bfloat16"
    assert weights["layer.0.experts.w_gate"].shape == (8, 16, 32)
    assert weights["layer.0.experts.w_down"].shape == (8, 32, 16)
    assert weights["w_out"].dtype.name == "bfloat16"
    assert weights["final_norm"].dtype == np.float32
    assert "_audio_encoder_cfg" not in weights
    assert "_talker_cfg" not in weights
    assert "_code2wav_cfg" not in weights
    assert not any(key.startswith(("audio.", "vision.", "code2wav.")) for key in weights)


def test_bundle_config_overrides_are_thinker_only() -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.qwen3_omni.plugin")
    config = ModelConfig.create_tiny("qwen3_omni")
    plugin = plugin_module.Qwen3OmniPlugin()
    plugin._thinker_cfg = {"num_experts": 8, "num_experts_per_tok": 2}

    overrides = plugin.get_bundle_config_overrides(config)

    assert overrides["num_local_experts"] == 8
    assert overrides["num_experts_per_tok"] == 2
    assert not any(key.startswith(("audio_", "omni_")) for key in overrides)


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
