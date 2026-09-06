# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import ml_dtypes
import numpy as np
import tensorrt as trt

from .. import model as model_module
from ..config import ModelConfig


def test_build_writes_only_the_thinker_text_bundle(monkeypatch, tmp_path) -> None:
    config = SimpleNamespace(
        model_type="qwen3_omni_moe",
        architectures=["Qwen3OmniMoeForConditionalGeneration"],
        max_position_embeddings=512,
        num_hidden_layers=48,
        num_key_value_heads=4,
        head_dim=128,
        vocab_size=151936,
        raw={"im_end_token_id": 151645},
    )

    class FakeModel:
        @staticmethod
        def load_weights(model_dir, loaded_config, *, precision):
            assert model_dir == str(tmp_path)
            assert loaded_config is config
            assert precision == "bf16"
            return {"thinker": True}

        @staticmethod
        def build_engine(loaded_config, weights, max_cache_length, **options):
            assert loaded_config is config
            assert weights == {"thinker": True}
            assert max_cache_length == 256
            assert options == {"precision": "bf16", "verbose": False}
            return b"thinker"

    class Writer:
        def __init__(self):
            self.header = None
            self.sections = {}

        def set_header(self, **header):
            self.header = header

        def add_bytes(self, name, value):
            self.sections[name] = value

        def add_json(self, name, value):
            self.sections[name] = value

    monkeypatch.setattr(model_module.ModelConfig, "from_dir", lambda _path: config)
    monkeypatch.setattr(model_module, "_Qwen3OmniModel", FakeModel)
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    request = SimpleNamespace(
        model_dir=tmp_path,
        backend="trt",
        dynamic_kv_cache=False,
        task="text_generation",
        precision="bf16",
        max_sequence_length=256,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        tensor_parallel_size=1,
        context_parallel_size=1,
        quantization=None,
        fp32_layers=(),
        verbose=False,
    )
    writer = Writer()

    model_module.build(request, writer)

    assert writer.header == {
        "family": "qwen3_omni",
        "task": "text_generation",
        "backend": "trt",
    }
    assert writer.sections == {
        "thinker.plan": b"thinker",
        "runtime.json": {
            "precision": "bf16",
            "thinker_num_layers": 48,
            "thinker_num_key_value_heads": 4,
            "thinker_head_dim": 128,
            "thinker_vocab_size": 151936,
            "thinker_max_cache_length": 256,
            "thinker_eos_token_id": 151645,
        },
        "tokenizer.json": b"{}",
    }


def test_thinker_load_weights_preserves_bf16_storage(monkeypatch, tmp_path) -> None:
    config = ModelConfig.create_tiny(
        num_hidden_layers=1,
        raw={
            "thinker_config": {
                "text_config": {
                    "num_experts": 8,
                    "num_experts_per_tok": 2,
                    "moe_intermediate_size": 32,
                }
            }
        },
    )

    class FakeReaders:
        pass

    def load_tensor(_readers, key: str) -> np.ndarray:
        hidden = config.hidden_size
        if key == "thinker.model.embed_tokens.weight":
            shape = (config.vocab_size, hidden)
        elif key == "thinker.lm_head.weight":
            shape = (config.vocab_size, hidden)
        elif key == "thinker.model.norm.weight" or key.endswith(
            ("input_layernorm.weight", "post_attention_layernorm.weight")
        ):
            shape = (hidden,)
        elif key.endswith(("q_norm.weight", "k_norm.weight")):
            shape = (config.head_dim,)
        elif key.endswith("q_proj.weight"):
            shape = (config.attention_size, hidden)
        elif key.endswith(("k_proj.weight", "v_proj.weight")):
            shape = (config.num_key_value_heads * config.head_dim, hidden)
        elif key.endswith("o_proj.weight"):
            shape = (hidden, config.attention_size)
        elif key.endswith("mlp.gate.weight"):
            shape = (8, hidden)
        elif key.endswith(("gate_proj.weight", "up_proj.weight")):
            shape = (32, hidden)
        elif key.endswith("down_proj.weight"):
            shape = (hidden, 32)
        else:
            raise AssertionError(f"unexpected tensor request: {key}")
        return np.ones(shape, dtype=np.float32)

    monkeypatch.setattr(model_module, "_open_safetensors", lambda _path: FakeReaders())
    monkeypatch.setattr(model_module, "_load_tensor", load_tensor)

    weights = model_module._Qwen3OmniModel().load_weights(str(tmp_path), config, precision="bf16")

    assert weights["embedding"].dtype.name == "bfloat16"
    assert weights["layer.0.w_q"].dtype.name == "bfloat16"
    assert weights["layer.0.router"].dtype.name == "bfloat16"
    assert weights["layer.0.experts.w_gate"].dtype.name == "bfloat16"
    assert weights["layer.0.experts.w_gate"].shape == (8, 16, 32)
    assert weights["layer.0.experts.w_down"].shape == (8, 32, 16)
    assert weights["w_out"].dtype.name == "bfloat16"
    assert weights["final_norm"].dtype == np.float32


def test_thinker_moe_batches_only_routed_expert_multiplies(monkeypatch) -> None:
    constants = []
    matrix_multiplies = []
    gathers = []

    class Tensor:
        def __init__(self, name: str, shape=(), dtype=trt.bfloat16):
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

        def add_cast(self, tensor, dtype):
            return Layer(Tensor(f"cast({tensor.name})", tensor.shape, dtype))

        def add_activation(self, tensor, _operation):
            return Layer(Tensor(f"activation({tensor.name})", dtype=tensor.dtype))

        def add_elementwise(self, lhs, rhs, _operation):
            return Layer(Tensor(f"elementwise({lhs.name},{rhs.name})", dtype=lhs.dtype))

        def add_reduce(self, tensor, _operation, _axes, keep_dims):
            del keep_dims
            return Layer(Tensor(f"reduce({tensor.name})", dtype=tensor.dtype))

    def add_constant(_network, shape, values, dtype=np.float32):
        del values
        tensor_dtype = (
            trt.bfloat16 if np.dtype(dtype) == np.dtype(ml_dtypes.bfloat16) else trt.float32
        )
        tensor = Tensor(f"weight{len(constants)}", shape, tensor_dtype)
        constants.append(tensor)
        return tensor

    monkeypatch.setattr(model_module.graph_ops, "add_constant", add_constant)
    network = Network()
    output = model_module._add_routed_swiglu_experts(
        network,
        Tensor("input", (-1, 16)),
        Tensor("top_indices", (-1, 2), trt.int32),
        Tensor("routing_weights", (-1, 2), trt.float32),
        hidden_size=16,
        top_k=2,
        w_gate=np.ones((8, 16, 32), dtype=ml_dtypes.bfloat16),
        w_up=np.ones((8, 16, 32), dtype=ml_dtypes.bfloat16),
        w_down=np.ones((8, 32, 16), dtype=ml_dtypes.bfloat16),
        dtype=np.dtype(ml_dtypes.bfloat16),
    )

    assert output.dtype == trt.bfloat16
    assert len(matrix_multiplies) == 3
    assert all(rhs.startswith("gather(weight") for _lhs, rhs in matrix_multiplies)
    assert [entry for entry in gathers if entry[0].startswith("weight")] == [
        ("weight0", "top_indices", 0),
        ("weight1", "top_indices", 0),
        ("weight2", "top_indices", 0),
    ]
