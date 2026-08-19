# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel Nemotron Speech Streaming RNNT predictor builder.

This mirrors the single-device predictor graph in ``model.py`` while keeping
the RNNT runtime contract unchanged:

* predictor embeddings stay replicated,
* each LSTM gate projection is column-sharded across the hidden dimension,
* rank-local hidden/cell slices are zero-padded back to full hidden size,
* TensorRT distributed ALL_REDUCE sums the padded slices so every rank produces
  the same full predictor state and ``pred_output`` tensors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import sys

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops
from ...parallel_config import add_all_reduce_sum, normalize_parallel_config

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from ...parallel_config import ParallelConfig


def _slice_lstm_gate_columns(
    arr: np.ndarray, hidden: int, parallel: "ParallelConfig"
) -> np.ndarray:
    """Slice i/f/g/o gate columns for this rank while preserving gate order."""
    rank = parallel.rank
    tp = parallel.tp_size
    local_hidden = hidden // tp
    start = rank * local_hidden
    end = start + local_hidden
    parts = [
        arr[..., start:end],
        arr[..., hidden + start : hidden + end],
        arr[..., 2 * hidden + start : 2 * hidden + end],
        arr[..., 3 * hidden + start : 3 * hidden + end],
    ]
    return np.ascontiguousarray(np.concatenate(parts, axis=-1))


def _validate_predictor_tp(weights: "WeightDict", parallel: "ParallelConfig") -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError("Nemotron Speech Streaming TP predictor requires a concrete rank")
    hidden = int(weights["_pred_hidden"])
    if hidden % parallel.tp_size != 0:
        raise ValueError(
            "Nemotron Speech Streaming TP predictor requires pred_hidden divisible by "
            f"tp_size ({hidden} vs {parallel.tp_size})"
        )
    for layer in range(int(weights["_pred_layers"])):
        pfx = f"pred.{layer}"
        expected = (hidden, 4 * hidden)
        for key in (f"{pfx}.w_ih_t", f"{pfx}.w_hh_t"):
            if tuple(weights[key].shape) != expected:
                raise ValueError(f"{key} shape must be {expected}; got {weights[key].shape}")
        if tuple(weights[f"{pfx}.bias"].shape) != (1, 4 * hidden):
            raise ValueError(
                f"{pfx}.bias shape must be {(1, 4 * hidden)}; got {weights[f'{pfx}.bias'].shape}"
            )


def _add_rank_full_state(network, local, hidden: int, parallel: "ParallelConfig"):
    local_hidden = hidden // parallel.tp_size
    start = parallel.rank * local_hidden
    end = start + local_hidden
    parts = []
    if start > 0:
        parts.append(
            graph_ops.add_constant(network, (1, start), np.zeros((1, start), dtype=np.float32))
        )
    parts.append(local)
    if end < hidden:
        suffix = hidden - end
        parts.append(
            graph_ops.add_constant(network, (1, suffix), np.zeros((1, suffix), dtype=np.float32))
        )
    cat = network.add_concatenation(parts)
    cat.axis = 1
    return add_all_reduce_sum(network, cat.get_output(0), parallel.tp_size)


def _add_lstm_cell_tp(
    network, x, h_prev, c_prev, weights, pfx: str, hidden: int, parallel: "ParallelConfig"
):
    local_hidden = hidden // parallel.tp_size
    w_ih = graph_ops.add_constant(
        network,
        (hidden, 4 * local_hidden),
        _slice_lstm_gate_columns(weights[f"{pfx}.w_ih_t"], hidden, parallel),
    )
    w_hh = graph_ops.add_constant(
        network,
        (hidden, 4 * local_hidden),
        _slice_lstm_gate_columns(weights[f"{pfx}.w_hh_t"], hidden, parallel),
    )
    bias = graph_ops.add_constant(
        network,
        (1, 4 * local_hidden),
        _slice_lstm_gate_columns(weights[f"{pfx}.bias"], hidden, parallel),
    )

    xw = network.add_matrix_multiply(x, trt.MatrixOperation.NONE, w_ih, trt.MatrixOperation.NONE)
    hw = network.add_matrix_multiply(
        h_prev, trt.MatrixOperation.NONE, w_hh, trt.MatrixOperation.NONE
    )
    gates = network.add_elementwise(
        xw.get_output(0), hw.get_output(0), trt.ElementWiseOperation.SUM
    )
    gates = network.add_elementwise(gates.get_output(0), bias, trt.ElementWiseOperation.SUM)

    gate_i = network.add_slice(
        gates.get_output(0), start=(0, 0), shape=(1, local_hidden), stride=(1, 1)
    )
    gate_f = network.add_slice(
        gates.get_output(0), start=(0, local_hidden), shape=(1, local_hidden), stride=(1, 1)
    )
    gate_g = network.add_slice(
        gates.get_output(0), start=(0, 2 * local_hidden), shape=(1, local_hidden), stride=(1, 1)
    )
    gate_o = network.add_slice(
        gates.get_output(0), start=(0, 3 * local_hidden), shape=(1, local_hidden), stride=(1, 1)
    )

    i_t = network.add_activation(gate_i.get_output(0), trt.ActivationType.SIGMOID).get_output(0)
    f_t = network.add_activation(gate_f.get_output(0), trt.ActivationType.SIGMOID).get_output(0)
    g_t = network.add_activation(gate_g.get_output(0), trt.ActivationType.TANH).get_output(0)
    o_t = network.add_activation(gate_o.get_output(0), trt.ActivationType.SIGMOID).get_output(0)

    c_local = network.add_slice(
        c_prev, start=(0, parallel.rank * local_hidden), shape=(1, local_hidden), stride=(1, 1)
    )
    forget = network.add_elementwise(
        f_t, c_local.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)
    update = network.add_elementwise(i_t, g_t, trt.ElementWiseOperation.PROD).get_output(0)
    c_local_new = network.add_elementwise(forget, update, trt.ElementWiseOperation.SUM).get_output(
        0
    )
    tanh_c = network.add_activation(c_local_new, trt.ActivationType.TANH).get_output(0)
    h_local_new = network.add_elementwise(o_t, tanh_c, trt.ElementWiseOperation.PROD).get_output(0)

    h_new = _add_rank_full_state(network, h_local_new, hidden, parallel)
    c_new = _add_rank_full_state(network, c_local_new, hidden, parallel)
    return h_new, c_new


def build_nemotron_streaming_tp_predictor(
    weights: "WeightDict",
    *,
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError(
            "build_nemotron_streaming_tp_predictor requires parallel.mode=tensor_parallel "
            "and tp_size > 1"
        )
    _validate_predictor_tp(weights, parallel)

    pred_hidden = int(weights["_pred_hidden"])
    pred_layers = int(weights["_pred_layers"])
    vocab_total = int(weights["_vocab_total"])

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)

    token_id = network.add_input("token_id", trt.int32, (1,))
    embedding = graph_ops.add_constant(
        network, (vocab_total, pred_hidden), weights["pred_embedding"]
    )
    hidden = network.add_gather(embedding, token_id, 0).get_output(0)

    next_h = []
    next_c = []
    for layer in range(pred_layers):
        h_in = network.add_input(f"state_h_{layer}", trt.float32, (1, pred_hidden))
        c_in = network.add_input(f"state_c_{layer}", trt.float32, (1, pred_hidden))
        hidden, c_new = _add_lstm_cell_tp(
            network, hidden, h_in, c_in, weights, f"pred.{layer}", pred_hidden, parallel
        )
        next_h.append(hidden)
        next_c.append(c_new)

    pred_output = network.add_identity(hidden).get_output(0)
    pred_output.name = "pred_output"
    network.mark_output(pred_output)
    for layer in range(pred_layers):
        next_h[layer].name = f"next_h_{layer}"
        next_c[layer].name = f"next_c_{layer}"
        network.mark_output(next_h[layer])
        network.mark_output(next_c[layer])

    if verbose:
        print(
            "[trtmc build] Building RNNT TP predictor "
            f"({pred_layers}L, h={pred_hidden}, tp={parallel.tp_size}, rank={parallel.rank})",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("RNNT TP predictor build failed")
    return bytes(plan)
