# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL text-builder contracts for multimodal rotary positions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tests.builder.conftest import requires_trt

pytest.importorskip(
    "tensorrt_model_connect",
    reason="Qwen3-VL builder tests require TensorRT",
)


def _tiny_qwen3_vl_config() -> SimpleNamespace:
    return SimpleNamespace(
        raw={
            "text_config": {
                "rope_scaling": {
                    "mrope_section": [2, 1, 1],
                    "mrope_interleaved": True,
                    "rope_type": "default",
                },
            },
            "_decoder_engine_role": "dual_profile",
        },
        hidden_size=16,
        vocab_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        attention_size=16,
        intermediate_size=16,
        rms_norm_eps=1e-6,
        rope_theta=5000000.0,
    )


def _tiny_qwen3_vl_weights() -> dict[str, np.ndarray | int]:
    rng = np.random.default_rng(20260729)

    def rand(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float32) * 0.02

    hidden = 16
    kv_attention = 8
    mlp = 16
    return {
        "_attention_size": hidden,
        "_kv_attention_size": kv_attention,
        "_mlp_size": mlp,
        "embedding": rand(16, hidden),
        "layer.0.input_norm": np.ones(hidden, dtype=np.float32),
        "layer.0.post_attn_norm": np.ones(hidden, dtype=np.float32),
        "layer.0.w_q": rand(hidden, hidden),
        "layer.0.w_k": rand(hidden, kv_attention),
        "layer.0.w_v": rand(hidden, kv_attention),
        "layer.0.w_o": rand(hidden, hidden),
        "layer.0.w_gate": rand(hidden, mlp),
        "layer.0.w_up": rand(hidden, mlp),
        "layer.0.w_down": rand(mlp, hidden),
        "final_norm": np.ones(hidden, dtype=np.float32),
        "w_out": rand(hidden, 16),
    }


@requires_trt
def test_qwen3_vl_decoder_declares_rank_two_mrope_profiles() -> None:
    import tensorrt as trt

    from tensorrt_model_connect.models.qwen_vl.model import (
        _build_qwen3_vl_decoder,
    )

    plan = _build_qwen3_vl_decoder(
        _tiny_qwen3_vl_config(),
        _tiny_qwen3_vl_weights(),
        4,
        precision="fp32",
    )

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    assert engine.get_tensor_shape("mrope_position_ids") == (3, -1)
    assert engine.num_optimization_profiles == 2
    assert tuple(engine.get_tensor_profile_shape(
        "mrope_position_ids", 0)) == ((3, 1), (3, 4), (3, 4))
    assert tuple(engine.get_tensor_profile_shape(
        "mrope_position_ids", 1)) == ((3, 1), (3, 1), (3, 1))
