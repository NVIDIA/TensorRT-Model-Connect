# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT engine-contract tests for the two qualified native families."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorrt")

from tests.builder.conftest import requires_trt

pytestmark = pytest.mark.dynamic_memory


def _weights(
    *,
    hidden: int,
    vocab: int,
    layers: int,
    kv_width: int,
) -> dict:
    rng = np.random.RandomState(17)
    values: dict[str, np.ndarray | int] = {
        "embedding": rng.randn(vocab, hidden).astype(np.float32),
        "final_norm": rng.randn(hidden).astype(np.float32),
        "w_out": rng.randn(hidden, vocab).astype(np.float32),
        "_attention_size": hidden,
        "_kv_attention_size": kv_width,
        "_mlp_size": hidden * 2,
    }
    for layer in range(layers):
        prefix = f"layer.{layer}"
        values[f"{prefix}.input_norm"] = rng.randn(
            hidden).astype(np.float32)
        values[f"{prefix}.post_attn_norm"] = rng.randn(
            hidden).astype(np.float32)
        values[f"{prefix}.w_q"] = rng.randn(
            hidden, hidden).astype(np.float32)
        values[f"{prefix}.w_k"] = rng.randn(
            hidden, kv_width).astype(np.float32)
        values[f"{prefix}.w_v"] = rng.randn(
            hidden, kv_width).astype(np.float32)
        values[f"{prefix}.w_o"] = rng.randn(
            hidden, hidden).astype(np.float32)
        values[f"{prefix}.w_gate"] = rng.randn(
            hidden, hidden * 2).astype(np.float32)
        values[f"{prefix}.w_up"] = rng.randn(
            hidden, hidden * 2).astype(np.float32)
        values[f"{prefix}.w_down"] = rng.randn(
            hidden * 2, hidden).astype(np.float32)
    return values


def _contract(family: str) -> dict:
    return {
        "contract_version": 1,
        "qualified_model_id": (
            "Qwen/Qwen3-0.6B"
            if family == "qwen"
            else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        ),
        "qualified_model_revision": "1" * 40,
        "qualified_config_sha256": "2" * 64,
        "qualified_target": "gb300-trt-11.2",
        "qualified_runtime_stack": {
            "sm": "sm103",
            "tensorrt": "11.2.0.113",
            "cuda_runtime": "13.3",
            "cudnn_backend": "9.20.0",
            "cudnn_frontend_revision":
                "7b9b711c22b6823e87150213ecd8449260db8610",
            "nvrtc": "13.3",
            "driver": "580.105.08",
        },
        "native_kv_plugin_abi": 2,
        "model_context_limit": 8,
        "prefill_chunk_limit": 2,
        "kv_layout": "contiguous_runtime_v1",
        "kv_dtype": "bfloat16",
        "kv_bytes_per_token": 128,
        "active_kv_profile_limits": [2, 4, 8],
        "runtime_owned": True,
        "precision": "bf16",
    }


def _build(family: str, role: str):
    import tensorrt as trt

    if family == "qwen":
        from tensorrt_model_connect.families.qwen.config import (
            ModelConfig,
        )
        from tensorrt_model_connect.families.qwen.standard_decoder_builder import (
            build_standard_decoder_engine,
        )
    else:
        from tensorrt_model_connect.families.llama.config import (
            ModelConfig,
        )
        from tensorrt_model_connect.families.llama.standard_decoder_builder import (
            build_standard_decoder_engine,
        )

    hidden, vocab, layers = 64, 32, 1
    query_heads, kv_heads = 4, 2
    head_dim = hidden // query_heads
    config = ModelConfig(
        model_type="qwen3" if family == "qwen" else "llama",
        hidden_size=hidden,
        vocab_size=vocab,
        num_hidden_layers=layers,
        num_attention_heads=query_heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=8,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )
    config.raw["_runtime_memory_contract"] = _contract(family)
    config.raw["_decoder_engine_role"] = role
    plan = build_standard_decoder_engine(
        config,
        _weights(
            hidden=hidden,
            vocab=vocab,
            layers=layers,
            kv_width=kv_heads * head_dim,
        ),
        8,
        precision="bf16",
    )
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    return runtime, engine


def _io_names(engine) -> tuple[set[str], set[str]]:
    import tensorrt as trt

    inputs: set[str] = set()
    outputs: set[str] = set()
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.add(name)
        else:
            outputs.add(name)
    return inputs, outputs


@requires_trt
@pytest.mark.parametrize("family", ("qwen", "llama"))
def test_qualified_split_engines_have_runtime_owned_row_major_kv(
    family: str,
) -> None:
    prefill_runtime, prefill = _build(family, "prefill")
    decode_runtime, decode = _build(family, "decode")
    assert prefill_runtime is not None
    assert decode_runtime is not None

    expected_inputs = {
        "token_id",
        "position_id",
        "history_length",
        "cache_k_0",
        "cache_v_0",
    }
    for engine in (prefill, decode):
        inputs, outputs = _io_names(engine)
        assert inputs == expected_inputs
        assert outputs == {
            "logits",
            "present_k_0",
            "present_v_0",
        }
        assert tuple(engine.get_tensor_shape("cache_k_0")) == (
            -1, 32)
        assert tuple(engine.get_tensor_shape("cache_v_0")) == (
            -1, 32)
        assert "attention_mask" not in inputs
        assert tuple(
            engine.get_tensor_shape("present_k_0")
        )[1:] == (32,)
        assert tuple(
            engine.get_tensor_shape("present_v_0")
        )[1:] == (32,)

    assert prefill.num_optimization_profiles == 1
    assert tuple(prefill.get_tensor_shape("present_k_0")) == (
        -1, 32)
    assert prefill.get_tensor_profile_shape("token_id", 0) == [
        (1,), (2,), (2,)]
    assert prefill.get_tensor_profile_shape("cache_k_0", 0) == [
        (1, 32),
        (2, 32),
        (8, 32),
    ]
    assert decode.num_optimization_profiles == 3
    assert tuple(decode.get_tensor_shape("present_k_0")) == (
        1, 32)
    for profile_index, bucket in enumerate((2, 4, 8)):
        assert decode.get_tensor_profile_shape(
            "token_id", profile_index) == [
                (1,), (1,), (1,)]
        assert decode.get_tensor_profile_shape(
            "cache_k_0", profile_index) == [
                (1, 32),
                (bucket, 32),
                (bucket, 32),
            ]
