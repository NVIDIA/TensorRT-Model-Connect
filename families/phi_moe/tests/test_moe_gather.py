# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phi-MoE expert blocks (decode gather, batched prefill): the engines must match a dense reference."""

from __future__ import annotations

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt", reason="TensorRT is required for Phi-MoE builder tests")
torch = pytest.importorskip("torch", reason="torch is required to run the Phi-MoE test engines")

from families.phi_moe import model  # noqa: E402
from families.phi_moe.checkpoint_mapper import WeightDict  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA device is required")

HIDDEN = 32
INTERMEDIATE = 48
EXPERTS = 16
JITTER = 0.01
TOKENS = 24
PREFIX = "layer.0"


def _round_fp16(array: np.ndarray) -> np.ndarray:
    """Values exactly representable in fp16, so every engine dtype sees the same numbers."""
    return array.astype(np.float16).astype(np.float32)


def _weights(seed: int = 0) -> WeightDict:
    rng = np.random.RandomState(seed)
    weights = WeightDict()
    weights[f"{PREFIX}.router"] = _round_fp16(rng.randn(HIDDEN, EXPERTS) * 0.5)
    for expert in range(EXPERTS):
        name = f"{PREFIX}.expert.{expert}"
        weights[f"{name}.w_gate"] = _round_fp16(rng.randn(HIDDEN, INTERMEDIATE) * 0.2)
        weights[f"{name}.w_up"] = _round_fp16(rng.randn(HIDDEN, INTERMEDIATE) * 0.2)
        weights[f"{name}.w_down"] = _round_fp16(rng.randn(INTERMEDIATE, HIDDEN) * 0.2)
    return weights


def _tokens(seed: int = 1) -> np.ndarray:
    return _round_fp16(np.random.RandomState(seed).randn(TOKENS, HIDDEN))


def _compile(define, dtype, rows: int = 1):
    """Build a one-input network ``x [rows, HIDDEN]`` whose outputs ``define`` marks; return an engine."""
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    x = network.add_input("x", trt.float16 if dtype == np.float16 else trt.float32, (rows, HIDDEN))
    for name, tensor in define(network, x).items():
        tensor.name = name
        network.mark_output(tensor)
    plan = builder.build_serialized_network(network, builder.create_builder_config())
    assert plan is not None
    return trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan), network


def _run(engine, tokens: np.ndarray, dtype) -> dict[str, np.ndarray]:
    context = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    outputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    torch_dtype = torch.float16 if dtype == np.float16 else torch.float32
    rows = engine.get_tensor_shape("x")[0]  # tokens per call
    x = torch.zeros(rows, HIDDEN, dtype=torch_dtype, device="cuda")
    buffers = {
        name: torch.zeros(
            tuple(engine.get_tensor_shape(name)),
            dtype={
                trt.DataType.INT32: torch.int32,
                trt.DataType.HALF: torch.float16,
            }.get(engine.get_tensor_dtype(name), torch.float32),
            device="cuda",
        )
        for name in outputs
    }
    context.set_tensor_address("x", x.data_ptr())
    for name, buffer in buffers.items():
        context.set_tensor_address(name, buffer.data_ptr())
    results: dict[str, list[np.ndarray]] = {name: [] for name in outputs}
    for start in range(0, len(tokens), rows):
        x.copy_(torch.from_numpy(tokens[start : start + rows]).to(torch_dtype))
        assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        for name, buffer in buffers.items():
            results[name].append(buffer.float().cpu().numpy().copy())
    return {name: np.concatenate(values) for name, values in results.items()}


def _sparsemixer_reference(scores: torch.Tensor) -> tuple[list[int], list[float]]:
    """SparseMixer top-2 (inference mode) over router logits ``[experts]``, in fp32."""
    indices, weights = [], []
    masked_scores = scores.clone()
    for _ in range(2):
        max_val, max_ind = masked_scores.max(dim=-1)
        factor = scores.abs().clamp(min=max_val)
        mask = (max_val - scores) / factor > 2 * JITTER
        probs = torch.softmax(masked_scores.masked_fill(mask, float("-inf")), dim=-1)
        indices.append(int(max_ind))
        weights.append(float(probs[max_ind]))
        masked_scores = masked_scores.scatter(-1, max_ind[None], float("-inf"))
    return indices, weights


def _dense_reference(weights: WeightDict, tokens: np.ndarray):
    """Every expert evaluated for every token, then the two routed ones weighted and summed."""
    router = torch.from_numpy(weights[f"{PREFIX}.router"]).double()
    selected, outputs = [], []
    for row in tokens:
        x = torch.from_numpy(row).double()
        indices, routing = _sparsemixer_reference((x @ router).float())
        expert_outputs = []
        for expert in range(EXPERTS):
            name = f"{PREFIX}.expert.{expert}"
            gate = x @ torch.from_numpy(weights[f"{name}.w_gate"]).double()
            up = x @ torch.from_numpy(weights[f"{name}.w_up"]).double()
            hidden = torch.nn.functional.silu(gate) * up
            expert_outputs.append(hidden @ torch.from_numpy(weights[f"{name}.w_down"]).double())
        outputs.append(sum(w * expert_outputs[i] for i, w in zip(indices, routing)).numpy())
        selected.append(indices)
    return np.array(selected), np.stack(outputs)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _moe_block(weights: WeightDict, dtype):
    work = trt.float16 if dtype == np.float16 else trt.float32

    def define(network, x):
        out = model._add_moe_block(
            network, x, weights, PREFIX, HIDDEN, EXPERTS, INTERMEDIATE, 2, JITTER, dtype, work
        )
        return {"moe": out}

    return define


def _prefill_moe_block(weights: WeightDict, dtype):
    work = trt.float16 if dtype == np.float16 else trt.float32

    def define(network, x):
        out = model._add_prefill_moe_block(
            network, x, weights, PREFIX, HIDDEN, EXPERTS, 2, JITTER, dtype, work
        )
        return {"moe": out}

    return define


# One token per call (decode) or every token in one call (prefill).
@pytest.mark.parametrize("rows", [1, TOKENS])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_router_selects_the_same_two_experts(dtype, rows: int) -> None:
    weights, tokens = _weights(), _tokens()
    work = trt.float16 if dtype == np.float16 else trt.float32

    def define(network, x):
        indices, routing = model._add_sparsemixer_top2(
            network, x, weights, PREFIX, HIDDEN, EXPERTS, JITTER, dtype, work
        )
        return {"indices": indices, "routing": routing}

    engine, _ = _compile(define, dtype, rows)
    got = _run(engine, tokens, dtype)
    expected_indices, _ = _dense_reference(weights, tokens)
    assert (got["indices"].astype(np.int64) == expected_indices).all()
    assert len({tuple(row) for row in expected_indices}) > 4, "routing should vary across tokens"


@pytest.mark.parametrize(("dtype", "minimum"), [(np.float32, 0.99999), (np.float16, 0.999)])
def test_gathered_experts_match_dense_reference(dtype, minimum: float) -> None:
    weights, tokens = _weights(), _tokens()
    engine, _ = _compile(_moe_block(weights, dtype), dtype)
    got = _run(engine, tokens, dtype)["moe"]
    _, expected = _dense_reference(weights, tokens)
    assert got.shape == expected.shape
    assert _cosine(got, expected) > minimum
    assert all(_cosine(g, e) > minimum for g, e in zip(got, expected))


def test_block_evaluates_two_experts_per_token() -> None:
    """The network holds the router and one batched matmul per projection, not one per expert."""
    _, network = _compile(_moe_block(_weights(), np.float16), np.float16)
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    matmuls = [layer for layer in layers if layer.type == trt.LayerType.MATRIX_MULTIPLY]
    assert len(matmuls) == 4  # router, gate, up, down


@pytest.mark.parametrize(("dtype", "minimum"), [(np.float32, 0.99999), (np.float16, 0.999)])
def test_prefill_block_matches_dense_reference(dtype, minimum: float) -> None:
    """Every token in one call, each keeping its own two routed experts."""
    weights, tokens = _weights(), _tokens()
    engine, _ = _compile(_prefill_moe_block(weights, dtype), dtype, rows=TOKENS)
    got = _run(engine, tokens, dtype)["moe"]
    _, expected = _dense_reference(weights, tokens)
    assert got.shape == expected.shape
    assert _cosine(got, expected) > minimum
    assert all(_cosine(g, e) > minimum for g, e in zip(got, expected))


def test_prefill_block_reads_every_expert_once() -> None:
    """The router, then gate, up and down each as one batched matmul over all expert weights."""
    _, network = _compile(_prefill_moe_block(_weights(), np.float16), np.float16, rows=TOKENS)
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    matmuls = [layer for layer in layers if layer.type == trt.LayerType.MATRIX_MULTIPLY]
    over_all_experts = [m for m in matmuls if tuple(m.get_input(1).shape)[0] == EXPERTS]
    assert len(matmuls) == 4 and len(over_all_experts) == 3
