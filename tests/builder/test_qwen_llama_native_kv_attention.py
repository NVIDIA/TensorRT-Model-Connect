# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Topology checks for correctness-first fixed-KV decoder attention."""

from __future__ import annotations

import importlib

import pytest

from tests.builder.conftest import requires_trt

trt = pytest.importorskip("tensorrt")


def _qwen_builder_module():
    importlib.import_module("tensorrt_model_connect.families.qwen.plugin")
    return importlib.import_module(
        "tensorrt_model_connect.families.qwen.dual_profile_decoder_builder"
    )


@pytest.mark.parametrize(
    (
        "native_kv_cache",
        "profile_mode",
        "cache_length",
        "opt_length",
        "requested",
        "expected",
    ),
    [
        (True, "prefill", 16384, 64, None, (64, 64)),
        (True, "prefill", 40960, 128, None, (64, 64)),
        (True, "prefill", 40960, 64, 32, (32, 32)),
        (True, "dual_profile", 40960, 128, None, (64, 64)),
        (True, "decode", 40960, 64, None, (64, 40960)),
        (False, "prefill", 40960, 64, None, (64, 40960)),
    ],
)
def test_explicit_attention_prefill_limit(
    native_kv_cache,
    profile_mode,
    cache_length,
    opt_length,
    requested,
    expected,
):
    builder = _qwen_builder_module()

    assert builder._resolve_prefill_lengths(
        cache_length,
        opt_length,
        requested,
        native_kv_cache=native_kv_cache,
        profile_mode=profile_mode,
    ) == expected


class _PrimitiveRecordingNetwork:
    def __init__(self, network):
        self._network = network
        self.attention_calls = 0
        self.matmul_calls = 0

    def __getattr__(self, name):
        return getattr(self._network, name)

    def add_attention_v2(self, *args, **kwargs):
        self.attention_calls += 1
        return self._network.add_attention_v2(*args, **kwargs)

    def add_matrix_multiply(self, *args, **kwargs):
        self.matmul_calls += 1
        return self._network.add_matrix_multiply(*args, **kwargs)


def _add_real_native_attention(module_name: str, dtype):
    from tensorrt_model_connect.native_kv_attention_builder import (
        add_active_prefix_causal_masks,
    )

    graph_ops = importlib.import_module(module_name)
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    raw_network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    network = _PrimitiveRecordingNetwork(raw_network)
    q = network.add_input("q", dtype, (1, 512))
    k = network.add_input("k", dtype, (1, 256))
    v = network.add_input("v", dtype, (1, 256))
    cache_k = network.add_input("cache_k", dtype, (1, 2, 8, 128))
    cache_v = network.add_input("cache_v", dtype, (1, 2, 8, 128))
    write_indices = network.add_input(
        "cache_write_indices", trt.int32, (1,)
    )
    lengths = network.add_input("key_value_lengths", trt.int32, (1,))
    masks = add_active_prefix_causal_masks(
        network,
        q,
        write_indices,
        lengths,
        8,
    )
    result = graph_ops.add_native_kv_cache_attention_from_rows(
        network,
        q,
        k,
        v,
        cache_k,
        cache_v,
        write_indices,
        masks,
        num_heads=4,
        num_kv_heads=2,
        head_dim=128,
        q_seq=1,
    )
    return network, masks, result


@requires_trt
@pytest.mark.parametrize(
    ("module_name", "dtype"),
    [
        (
            "tensorrt_model_connect.families.qwen.graph_ops",
            trt.float16,
        ),
        (
            "tensorrt_model_connect.families.qwen.graph_ops",
            trt.bfloat16,
        ),
        (
            "tensorrt_model_connect.families.llama.graph_ops",
            trt.bfloat16,
        ),
    ],
)
def test_native_kv_uses_explicit_primitive_attention(
    module_name,
    dtype,
):
    network, masks, result = _add_real_native_attention(module_name, dtype)

    assert masks.attention.dtype == trt.bool
    assert masks.active_prefix.dtype == trt.bool
    assert network.attention_calls == 0
    assert network.matmul_calls == 2
    assert result["present_k"].dtype == dtype
    assert result["present_v"].dtype == dtype
    assert result["context"].dtype == dtype


@requires_trt
def test_llama_unqualified_fp16_graph_fails_closed():
    with pytest.raises(ValueError, match="requires BF16"):
        _add_real_native_attention(
            "tensorrt_model_connect.families.llama.graph_ops",
            trt.float16,
        )
