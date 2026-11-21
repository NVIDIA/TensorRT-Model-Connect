# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


FAMILY = Path(__file__).resolve().parents[1]


@pytest.fixture
def parallel(monkeypatch):
    name = "_phi_parallel_cpu_test"
    spec = importlib.util.spec_from_file_location(name, FAMILY / "parallel.py")
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, name, module)
        patch.setitem(sys.modules, "tensorrt", SimpleNamespace())
        spec.loader.exec_module(module)
    return module


def mapped_weights():
    rng = np.random.default_rng(42)
    return {
        "layer.0.w_gate": rng.normal(0, 0.1, (16, 32)).astype(np.float32),
        "layer.0.w_up": rng.normal(0, 0.1, (16, 32)).astype(np.float32),
        "layer.0.w_down": rng.normal(0, 0.1, (32, 16)).astype(np.float32),
        "layer.0.w_q": rng.normal(0, 0.1, (16, 16)).astype(np.float32),
        "layer.0.w_o": rng.normal(0, 0.1, (16, 16)).astype(np.float32),
        "embedding": rng.normal(0, 0.1, (40, 16)).astype(np.float32),
        "_attention_size": 16,
        "_kv_attention_size": 16,
        "_mlp_size": 32,
    }


def config():
    return SimpleNamespace(num_attention_heads=8, num_key_value_heads=8, intermediate_size=32)


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_each_rank_gets_matched_swiglu_slices_without_mutating_weights(parallel, tp):
    weights = mapped_weights()
    original = {
        key: value.copy() for key, value in weights.items() if isinstance(value, np.ndarray)
    }
    for rank in range(tp):
        shard = parallel.shard_standard_decoder_weights(
            config(), weights, parallel.ParallelConfig(tp, rank)
        )
        for name in ("w_gate", "w_up"):
            value = shard[f"layer.0.{name}"]
            assert value.shape == (16, 32 // tp)
            assert value.flags.c_contiguous
            np.testing.assert_array_equal(
                value, weights[f"layer.0.{name}"][:, rank * (32 // tp) : (rank + 1) * (32 // tp)]
            )
        assert shard["layer.0.w_down"].shape == (32 // tp, 16)
        assert shard["layer.0.w_down"].flags.c_contiguous
        assert shard["_mlp_size"] == 32 // tp
        assert shard["_attention_size"] == shard["_kv_attention_size"] == 16 // tp
        assert shard["_tensor_parallel_rank"] == rank
        assert shard["embedding"] is weights["embedding"]
    for name, value in original.items():
        np.testing.assert_array_equal(weights[name], value)
    assert weights["_mlp_size"] == 32


class Network:
    def add_activation(self, value, kind):
        return SimpleNamespace(get_output=lambda index: 1 / (1 + np.exp(-value)))

    def add_elementwise(self, lhs, rhs, kind):
        return SimpleNamespace(get_output=lambda index: lhs * rhs)


def builder_swiglu():
    tree = ast.parse((FAMILY / "dual_profile_decoder_tp_builder.py").read_text(encoding="utf-8"))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_swiglu_mlp"
    )
    tree.body = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        helper,
    ]
    ast.fix_missing_locations(tree)
    namespace = {
        "trt": SimpleNamespace(
            ActivationType=SimpleNamespace(SIGMOID=0), ElementWiseOperation=SimpleNamespace(PROD=0)
        )
    }
    exec(compile(tree, "phi_tp_swiglu", "exec"), namespace)
    return namespace["_swiglu_mlp"]


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_rank_local_builder_outputs_sum_to_unsharded_swiglu(parallel, tp):
    weights = mapped_weights()
    x = np.random.default_rng(43).normal(0, 0.1, (3, 16))
    gate = x @ weights["layer.0.w_gate"].astype(np.float64)
    reference = (
        gate / (1 + np.exp(-gate)) * (x @ weights["layer.0.w_up"].astype(np.float64))
    ) @ weights["layer.0.w_down"].astype(np.float64)
    outputs = []
    for rank in range(tp):
        shard = parallel.shard_standard_decoder_weights(
            config(), weights, parallel.ParallelConfig(tp, rank)
        )

        def matmul(value, rows, columns, weight, name):
            return value @ weight.astype(np.float64).reshape(rows, columns)

        outputs.append(
            builder_swiglu()(
                Network(),
                x,
                matmul=matmul,
                weights=shard,
                prefix="layer.0",
                hidden=16,
                mlp_size=shard["_mlp_size"],
            )
        )
    np.testing.assert_allclose(np.sum(outputs, axis=0), reference, rtol=1e-12, atol=1e-12)


def test_tp1_keeps_the_original_weights(parallel):
    weights = mapped_weights()
    assert (
        parallel.shard_standard_decoder_weights(config(), weights, parallel.ParallelConfig())
        is weights
    )
