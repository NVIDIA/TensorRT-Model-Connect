# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical feature-tap probe: python -m families.llama.tests.speculative_features_gpu."""

import json

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cuda

from ..config import ModelConfig
from ..speculative.contract import EngineContract
from ..speculative.features import target_feature_indices
from ..speculative.graph import build_target
from .selection_gpu import checked


def check_features(draft_config, expected):
    # Each layer adds exactly one: the embedding is one, RMSNorm has unit
    # weights and zero epsilon, Q/K and MLP are zero, V/O are identity. Thus
    # layer i's input is i+1 and its output is i+2, independently of the taps.
    config = ModelConfig(model_type="llama", vocab_size=4, hidden_size=128,
                         intermediate_size=128, num_hidden_layers=32,
                         num_attention_heads=1, num_key_value_heads=1, rms_norm_eps=0.0)
    contract = EngineContract("target", 32, 128, 1, 128, 4, 2,
                              max_query=1, feature_width=384)
    zero = np.zeros((128, 128), np.float16)
    identity = np.eye(128, dtype=np.float16)
    norm = np.ones(128, np.float32)
    weights = {"embedding": np.ones((4, 128), np.float16), "final_norm": norm,
               "w_out": np.zeros((128, 4), np.float16)}
    for layer in range(32):
        for name in ("input_norm", "post_attn_norm"):
            weights[f"layer.{layer}.{name}"] = norm
        for name in ("w_q", "w_k", "w_gate", "w_up", "w_down"):
            weights[f"layer.{layer}.{name}"] = zero
        for name in ("w_v", "w_o"):
            weights[f"layer.{layer}.{name}"] = identity
    plan = build_target(config, weights, contract,
                        feature_indices=target_feature_indices(32, draft_config))
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    stream = checked(cuda.cudaStreamCreate())
    buffers = {}
    try:
        inputs = {name: np.array([0], np.int32)
                  for name in ("token_id", "position_id", "logits_indices", "cache_write_indices")}
        inputs["key_value_lengths"] = np.array([1], np.int32)
        inputs["attention_mask"] = np.array([[1, 0]], np.int32)
        for layer in range(32):
            for kind in ("k", "v"):
                inputs[f"cache_{kind}_{layer}"] = np.zeros((1, 1, 2, 128), np.float16)
        for name, value in inputs.items():
            if -1 in engine.get_tensor_shape(name):
                assert context.set_input_shape(name, value.shape)
            buffers[name] = checked(cuda.cudaMalloc(value.nbytes))
            assert context.set_tensor_address(name, buffers[name])
            checked(cuda.cudaMemcpyAsync(buffers[name], value.ctypes.data, value.nbytes,
                                         cuda.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
        outputs = {"features": np.empty((1, 384), np.float16),
                   "logits": np.empty((1, 4), np.float32)}
        for name, value in outputs.items():
            buffers[name] = checked(cuda.cudaMalloc(value.nbytes))
            assert context.set_tensor_address(name, buffers[name])
        for layer in range(32):
            for kind in ("k", "v"):
                assert context.set_tensor_address(f"present_{kind}_{layer}", buffers[f"cache_{kind}_{layer}"])
        assert context.execute_async_v3(stream)
        actual = outputs["features"]
        checked(cuda.cudaMemcpyAsync(actual.ctypes.data, buffers["features"], actual.nbytes,
                                     cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
        checked(cuda.cudaStreamSynchronize(stream))
        np.testing.assert_array_equal(actual.reshape(3, 128),
                                      np.repeat(np.array(expected)[:, None], 128, axis=1))
    finally:
        checked(cuda.cudaStreamSynchronize(stream))
        for pointer in buffers.values():
            checked(cuda.cudaFree(pointer))
        checked(cuda.cudaStreamDestroy(stream))


if __name__ == "__main__":
    check_features({}, [3, 17, 30])
    check_features({"eagle_aux_hidden_state_layer_ids": [29, 2, 16]}, [30, 3, 17])
    print(json.dumps({"feature_tap_cases": 2, "passed": True}))
