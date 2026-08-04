# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contract tests for InternVL's TensorRT native KV route."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tensorrt_model_connect.families.internvl.build_routing import (
    validate_native_kv_architecture,
    validate_native_kv_build,
)
from tensorrt_model_connect.families.internvl.config import ModelConfig
from tensorrt_model_connect.families.internvl.native_kv_contract import (
    validate_native_kv_weights,
)
from tensorrt_model_connect.parallel_config import ParallelConfig


def _config(*, size: str = "2b", **updates) -> ModelConfig:
    sizes = {
        "2b": (1536, 8960, 28, 12, 2),
        "8b": (3584, 18944, 28, 28, 4),
    }
    hidden, mlp, layers, heads, kv_heads = sizes[size]
    values = dict(
        model_type="internvl",
        architectures=["InternVLForConditionalGeneration"],
        vocab_size=151674,
        hidden_size=hidden,
        intermediate_size=mlp,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_position_embeddings=32768,
        hidden_act="silu",
        _head_dim=128,
        raw={
            "text_config": {
                "model_type": "qwen2",
                "rope_scaling": {
                    "factor": 2.0,
                    "rope_type": "dynamic",
                    "type": "dynamic",
                },
            },
            "_decoder_engine_layout": "split",
        },
    )
    values.update(updates)
    return ModelConfig(**values)


@pytest.mark.parametrize(
    ("size", "parallel"),
    [
        ("2b", ParallelConfig()),
        ("2b", ParallelConfig(mode="tensor_parallel", tp_size=2)),
        ("8b", ParallelConfig()),
        ("8b", ParallelConfig(mode="tensor_parallel", tp_size=4)),
    ],
)
def test_registered_internvl_sizes_share_full_context_native_contract(
    size: str, parallel: ParallelConfig,
) -> None:
    config = _config(size=size)
    validate_native_kv_build(
        config, precision="bf16", max_cache_length=32768,
        parallel=parallel, quantized=False, debug_layer_outputs=False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"precision": "fp16"}, "BF16"),
        ({"max_cache_length": 384}, "model context"),
        ({"quantized": True}, "quantized"),
        ({"debug_layer_outputs": True}, "debug"),
    ],
)
def test_unsupported_build_modes_fail_closed(kwargs, message: str) -> None:
    options = dict(
        precision="bf16", max_cache_length=32768,
        parallel=ParallelConfig(), quantized=False,
        debug_layer_outputs=False,
    )
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        validate_native_kv_build(_config(), **options)


def test_tp_requires_rank_local_head_divisibility() -> None:
    with pytest.raises(ValueError, match="KV heads"):
        validate_native_kv_build(
            _config(size="2b"), precision="bf16", max_cache_length=32768,
            parallel=ParallelConfig(mode="tensor_parallel", tp_size=4),
            quantized=False, debug_layer_outputs=False)


def test_generic_dynamic_kv_request_fails_closed() -> None:
    with pytest.raises(ValueError, match="fixed full-context capacity"):
        validate_native_kv_build(
            _config(raw={
                "text_config": {"model_type": "qwen2"},
                "_decoder_engine_layout": "split",
                "_runtime_dynamic_kv_requested": True,
            }),
            precision="bf16", max_cache_length=32768,
            parallel=ParallelConfig(),
            quantized=False, debug_layer_outputs=False)


def test_non_qwen2_text_backbone_fails_closed() -> None:
    config = _config()
    config.raw["text_config"]["model_type"] = "llama"
    with pytest.raises(ValueError, match="Qwen2"):
        validate_native_kv_architecture(config)


@dataclass
class _Tensor:
    shape: tuple[int, ...]


def test_weight_contract_accepts_qwen2_biases_and_rejects_bad_shapes() -> None:
    config = _config(
        vocab_size=32, hidden_size=128, intermediate_size=256,
        num_hidden_layers=1, num_attention_heads=1,
        num_key_value_heads=1,
    )
    weights: dict[str, object] = {
        "embedding": _Tensor((32, 128)),
        "final_norm": _Tensor((128,)),
        "w_out": _Tensor((128, 32)),
        "_attention_size": 128,
        "_kv_attention_size": 128,
        "_mlp_size": 256,
    }
    for name, shape in (
        ("input_norm", (128,)), ("w_q", (128, 128)),
        ("w_k", (128, 128)), ("w_v", (128, 128)),
        ("w_o", (128, 128)), ("post_attn_norm", (128,)),
        ("w_gate", (128, 256)), ("w_up", (128, 256)),
        ("w_down", (256, 128)), ("q_bias", (128,)),
        ("k_bias", (128,)), ("v_bias", (128,)),
    ):
        weights[f"layer.0.{name}"] = _Tensor(shape)
    validate_native_kv_weights(config, weights)

    weights["layer.0.w_k"] = _Tensor((127, 128))
    with pytest.raises(ValueError, match="must have shape"):
        validate_native_kv_weights(config, weights)
