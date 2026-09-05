# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenFold3 sequence-local atom attention expressed with TensorRT APIs."""

from __future__ import annotations

from typing import Any

import numpy as np

from .graph_ops import Graph


ATOM_CHANNELS = 128
ATOM_PAIR_CHANNELS = 16
ATOM_HEADS = 4
ATOM_HEAD_WIDTH = 32
ATOM_BLOCKS = 3
ATOM_WINDOW_QUERIES = 32
ATOM_WINDOW_KEYS = 128


def padded_atom_count(atom_count: int) -> int:
    """Return the query-window-padded atom count used in a static bundle."""

    if atom_count <= 0:
        raise ValueError("OpenFold3 atom count must be positive")
    return ((atom_count + ATOM_WINDOW_QUERIES - 1) // ATOM_WINDOW_QUERIES) * (ATOM_WINDOW_QUERIES)


def _key_indices(atom_count: int, padded_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    if padded_atoms != padded_atom_count(atom_count):
        raise ValueError("OpenFold3 padded atom count does not match the real atom count")
    windows = padded_atoms // ATOM_WINDOW_QUERIES
    centers = ATOM_WINDOW_QUERIES // 2 + np.arange(windows) * ATOM_WINDOW_QUERIES
    initial = centers[:, None] + np.arange(-ATOM_WINDOW_KEYS // 2, ATOM_WINDOW_KEYS // 2)[None, :]
    underflow = np.maximum(-initial[:, 0], 0)
    overflow = np.maximum(initial[:, -1] - (atom_count - 1), 0)
    shift = np.where(underflow > 0, underflow, -overflow)
    final = initial + shift[:, None]
    invalid = (final < 0) | (final >= atom_count)
    return (
        np.clip(final, 0, atom_count - 1).astype(np.int32),
        invalid.astype(np.float32),
    )


def _query_blocks(graph: Graph, tensor: Any, padded_atoms: int):
    shape = tuple(int(dimension) for dimension in tensor.shape)
    return graph.reshape(
        tensor,
        (*shape[:-2], padded_atoms // ATOM_WINDOW_QUERIES, ATOM_WINDOW_QUERIES, shape[-1]),
    )


def _key_blocks(
    graph: Graph,
    tensor: Any,
    *,
    atom_count: int,
    padded_atoms: int,
):
    indices, invalid = _key_indices(atom_count, padded_atoms)
    gathered = graph.gather(
        tensor,
        graph.constant(indices, dtype=np.int32),
        len(tensor.shape) - 2,
    )
    mask_shape = (1,) * (len(gathered.shape) - 3) + (*invalid.shape, 1)
    valid = graph.sub(
        graph.scalar_like(1.0, gathered),
        graph.cast(graph.constant(invalid, mask_shape), gathered.dtype),
    )
    return graph.mul(gathered, valid)


def atom_pair_mask(
    graph: Graph,
    atom_mask: Any,
    *,
    atom_count: int,
    padded_atoms: int,
):
    windows = padded_atoms // ATOM_WINDOW_QUERIES
    query = graph.reshape(atom_mask, (1, windows, ATOM_WINDOW_QUERIES, 1))
    key = _key_blocks(
        graph,
        graph.reshape(atom_mask, (1, padded_atoms, 1)),
        atom_count=atom_count,
        padded_atoms=padded_atoms,
    )
    key = graph.reshape(key, (1, windows, 1, ATOM_WINDOW_KEYS))
    return graph.mul(query, key)


def _adaln(graph: Graph, a: Any, s: Any, prefix: str, compute_dtype: Any):
    a_norm = graph.unit_layer_norm(a)
    s_norm = graph.layer_norm(s, f"{prefix}.layer_norm_s")
    s_compute = graph.cast(s_norm, compute_dtype)
    scale = graph.sigmoid(graph.linear(s_compute, f"{prefix}.linear_g"))
    shift = graph.linear(s_compute, f"{prefix}.linear_s")
    return graph.add(
        graph.mul(graph.cast(scale, a_norm.dtype), a_norm),
        graph.cast(shift, a_norm.dtype),
    )


def _cross_attention(
    graph: Graph,
    a: Any,
    s: Any,
    z: Any,
    atom_mask: Any,
    prefix: str,
    *,
    atom_count: int,
    padded_atoms: int,
    compute_dtype: Any,
):
    windows = padded_atoms // ATOM_WINDOW_QUERIES
    pair_mask = atom_pair_mask(
        graph,
        atom_mask,
        atom_count=atom_count,
        padded_atoms=padded_atoms,
    )
    a_q = _query_blocks(graph, a, padded_atoms)
    a_k = _key_blocks(graph, a, atom_count=atom_count, padded_atoms=padded_atoms)
    s_q = _query_blocks(graph, s, padded_atoms)
    s_k = _key_blocks(graph, s, atom_count=atom_count, padded_atoms=padded_atoms)
    a_q = _adaln(graph, a_q, s_q, f"{prefix}.layer_norm_a_q", compute_dtype)
    a_k = _adaln(graph, a_k, s_k, f"{prefix}.layer_norm_a_k", compute_dtype)
    q_compute = graph.cast(a_q, compute_dtype)
    k_compute = graph.cast(a_k, compute_dtype)

    def project(source: Any, name: str, length: int):
        projected = graph.linear(source, f"{prefix}.mha.linear_{name}")
        projected = graph.reshape(projected, (1, windows, length, ATOM_HEADS, ATOM_HEAD_WIDTH))
        return graph.transpose(projected, (0, 1, 3, 2, 4))

    query = project(q_compute, "q", ATOM_WINDOW_QUERIES)
    key = project(k_compute, "k", ATOM_WINDOW_KEYS)
    value = project(k_compute, "v", ATOM_WINDOW_KEYS)
    scores = graph.network.add_matrix_multiply(
        graph.cast(query, graph.stable_attention_dtype),
        graph.trt.MatrixOperation.NONE,
        graph.cast(key, graph.stable_attention_dtype),
        graph.trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    scores = graph.mul(scores, graph.scalar_like(1.0 / np.sqrt(ATOM_HEAD_WIDTH), scores))
    pair_bias = graph.linear(graph.cast(z, compute_dtype), f"{prefix}.linear_z")
    pair_bias = graph.transpose(pair_bias, (0, 1, 4, 2, 3))
    scores = graph.add(scores, graph.cast(pair_bias, scores.dtype))
    mask_bias = graph.mul(
        graph.sub(pair_mask, graph.scalar_like(1.0, pair_mask)),
        graph.scalar_like(1.0e9, pair_mask),
    )
    scores = graph.add(
        scores,
        graph.cast(graph.reshape(mask_bias, (1, windows, 1, 32, 128)), scores.dtype),
    )
    probabilities = graph.softmax_last(scores)
    attended = graph.network.add_matrix_multiply(
        probabilities,
        graph.trt.MatrixOperation.NONE,
        graph.cast(value, probabilities.dtype),
        graph.trt.MatrixOperation.NONE,
    ).get_output(0)
    attended = graph.cast(attended, compute_dtype)
    attended = graph.transpose(attended, (0, 1, 3, 2, 4))
    attended = graph.reshape(attended, (1, windows, ATOM_WINDOW_QUERIES, ATOM_CHANNELS))
    gate = graph.sigmoid(graph.linear(q_compute, f"{prefix}.mha.linear_g"))
    attended = graph.linear(graph.mul(gate, attended), f"{prefix}.mha.linear_o")
    attended = graph.reshape(attended, (1, padded_atoms, ATOM_CHANNELS))
    output_gate = graph.sigmoid(
        graph.linear(graph.cast(s, compute_dtype), f"{prefix}.linear_ada_out")
    )
    return graph.mul(attended, output_gate)


def _conditioned_transition(
    graph: Graph,
    a: Any,
    s: Any,
    atom_mask: Any,
    prefix: str,
    compute_dtype: Any,
):
    adapted = _adaln(graph, a, s, f"{prefix}.layer_norm", compute_dtype)
    adapted = graph.cast(adapted, compute_dtype)
    hidden = graph.mul(
        graph.silu(graph.linear(adapted, f"{prefix}.swiglu.linear_a")),
        graph.linear(adapted, f"{prefix}.swiglu.linear_b"),
    )
    update = graph.linear(hidden, f"{prefix}.linear_out")
    gate = graph.sigmoid(graph.linear(graph.cast(s, compute_dtype), f"{prefix}.linear_g"))
    mask = graph.reshape(atom_mask, (1, int(atom_mask.shape[-1]), 1))
    return graph.mul(graph.mul(update, gate), graph.cast(mask, update.dtype))


def atom_transformer(
    graph: Graph,
    q: Any,
    condition: Any,
    pair: Any,
    atom_mask: Any,
    prefix: str,
    *,
    atom_count: int,
    padded_atoms: int,
    compute_dtype: Any,
):
    """Add one exact three-block sequence-local atom transformer."""

    pair = graph.layer_norm(pair, f"{prefix}.layer_norm_z")
    for block in range(ATOM_BLOCKS):
        block_prefix = f"{prefix}.blocks.{block}"
        update = _cross_attention(
            graph,
            q,
            condition,
            pair,
            atom_mask,
            f"{block_prefix}.attention_pair_bias",
            atom_count=atom_count,
            padded_atoms=padded_atoms,
            compute_dtype=compute_dtype,
        )
        q = graph.add(q, graph.cast(update, q.dtype))
        update = _conditioned_transition(
            graph,
            q,
            condition,
            atom_mask,
            f"{block_prefix}.conditioned_transition",
            compute_dtype,
        )
        q = graph.add(q, graph.cast(update, q.dtype))
    return q


def reference_atom_features(
    graph: Graph,
    features: dict[str, Any],
    prefix: str,
    *,
    atom_count: int,
    padded_atoms: int,
    finalize_pair: bool = True,
):
    """Implement AF3 Algorithm 5 lines 1–14 for pinned atom features."""

    ref_pos = features["ref_pos"]
    ref_mask = features["ref_mask"]
    charge = graph.reshape(features["ref_charge"], (1, padded_atoms, 1))
    charge = graph.unary(charge, graph.trt.UnaryOperation.ASINH)
    mask_column = graph.reshape(ref_mask, (1, padded_atoms, 1))
    name_chars = graph.reshape(features["ref_atom_name_chars"], (1, padded_atoms, 256))
    cl = graph.linear(ref_pos, f"{prefix}.ref_atom_feature_embedder.linear_ref_pos")
    for tensor, name in (
        (charge, "linear_ref_charge"),
        (mask_column, "linear_ref_mask"),
        (features["ref_element"], "linear_ref_element"),
        (name_chars, "linear_ref_atom_chars"),
    ):
        update = graph.linear(
            graph.cast(tensor, ref_pos.dtype),
            f"{prefix}.ref_atom_feature_embedder.{name}",
        )
        cl = graph.add(cl, graph.cast(update, cl.dtype))

    q_pos = _query_blocks(graph, ref_pos, padded_atoms)
    k_pos = _key_blocks(graph, ref_pos, atom_count=atom_count, padded_atoms=padded_atoms)
    displacement = graph.sub(
        graph.reshape(q_pos, (1, padded_atoms // 32, 32, 1, 3)),
        graph.reshape(k_pos, (1, padded_atoms // 32, 1, 128, 3)),
    )
    pair_mask = atom_pair_mask(
        graph,
        features["atom_mask"],
        atom_count=atom_count,
        padded_atoms=padded_atoms,
    )
    q_uid = _query_blocks(
        graph,
        graph.reshape(features["ref_space_uid"], (1, padded_atoms, 1)),
        padded_atoms,
    )
    k_uid = _key_blocks(
        graph,
        graph.reshape(features["ref_space_uid"], (1, padded_atoms, 1)),
        atom_count=atom_count,
        padded_atoms=padded_atoms,
    )
    same_uid = graph.equal(
        graph.reshape(q_uid, (1, padded_atoms // 32, 32, 1, 1)),
        graph.reshape(k_uid, (1, padded_atoms // 32, 1, 128, 1)),
    )
    valid = graph.mul(
        graph.reshape(pair_mask, (1, padded_atoms // 32, 32, 128, 1)),
        graph.cast(same_uid, pair_mask.dtype),
    )
    plm = graph.mul(
        graph.linear(displacement, f"{prefix}.ref_atom_feature_embedder.linear_ref_offset"),
        valid,
    )
    squared = graph.reduce_sum(graph.mul(displacement, displacement), 4, keep_dims=True)
    inverse_squared = graph.div(
        graph.scalar_like(1.0, squared),
        graph.add(graph.scalar_like(1.0, squared), squared),
    )
    for tensor, name in (
        (inverse_squared, "linear_inv_sq_dists"),
        (valid, "linear_valid_mask"),
    ):
        update = graph.mul(
            graph.linear(tensor, f"{prefix}.ref_atom_feature_embedder.{name}"), valid
        )
        plm = graph.add(plm, graph.cast(update, plm.dtype))

    if not finalize_pair:
        return cl, plm
    return cl, finalize_atom_pair(
        graph,
        cl,
        plm,
        pair_mask,
        prefix,
        atom_count=atom_count,
        padded_atoms=padded_atoms,
    )


def finalize_atom_pair(
    graph: Graph,
    cl: Any,
    plm: Any,
    pair_mask: Any,
    prefix: str,
    *,
    atom_count: int,
    padded_atoms: int,
):
    """Apply Algorithm 5 lines 13–14 after optional noisy/trunk conditioning."""

    q_cl = _query_blocks(graph, cl, padded_atoms)
    k_cl = _key_blocks(graph, cl, atom_count=atom_count, padded_atoms=padded_atoms)
    q_update = graph.linear(graph.relu(q_cl), f"{prefix}.linear_l")
    k_update = graph.linear(graph.relu(k_cl), f"{prefix}.linear_m")
    plm = graph.add(
        plm,
        graph.reshape(q_update, (1, padded_atoms // 32, 32, 1, 16)),
    )
    plm = graph.add(
        plm,
        graph.reshape(k_update, (1, padded_atoms // 32, 1, 128, 16)),
    )
    update = graph.linear(graph.relu(plm), f"{prefix}.pair_mlp.1")
    update = graph.linear(graph.relu(update), f"{prefix}.pair_mlp.3")
    update = graph.linear(graph.relu(update), f"{prefix}.pair_mlp.5")
    valid = graph.reshape(pair_mask, (1, padded_atoms // 32, 32, 128, 1))
    plm = graph.mul(graph.add(plm, update), valid)
    return plm
