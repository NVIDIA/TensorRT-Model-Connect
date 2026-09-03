# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct strongly typed TensorRT builder for the Boltz-2 input embedder."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tensorrt_model_connect import trt_compat

from .checkpoint import load_weight_prefixes
from .graph_ops import Graph


ATOM_COUNT = 928
ATOM_CHANNELS = 128
ATOM_PAIR_CHANNELS = 16
ATOM_FEATURES = 388
ATOM_WINDOW_QUERIES = 32
ATOM_WINDOW_KEYS = 128
ATOM_WINDOWS = ATOM_COUNT // ATOM_WINDOW_QUERIES
ATOM_HEADS = 4
ATOM_LAYERS = 3


def atom_windows(atom_count: int) -> int:
    """Return the number of upstream 32-query atom windows for a padded shape."""

    if atom_count <= 0 or atom_count % ATOM_WINDOW_QUERIES:
        raise ValueError(
            "Boltz-2 padded atom count must be a positive multiple of "
            f"{ATOM_WINDOW_QUERIES}, got {atom_count}"
        )
    return atom_count // ATOM_WINDOW_QUERIES


def atom_attention_detail_shapes(atom_count: int) -> dict[str, tuple[int, ...]]:
    windows = atom_windows(atom_count)
    return {
        "query": (windows, ATOM_HEADS, ATOM_WINDOW_QUERIES, ATOM_CHANNELS // ATOM_HEADS),
        "key": (windows, ATOM_HEADS, ATOM_WINDOW_KEYS, ATOM_CHANNELS // ATOM_HEADS),
        "value": (windows, ATOM_HEADS, ATOM_WINDOW_KEYS, ATOM_CHANNELS // ATOM_HEADS),
        "logits": (windows, ATOM_HEADS, ATOM_WINDOW_QUERIES, ATOM_WINDOW_KEYS),
        "scores": (windows, ATOM_HEADS, ATOM_WINDOW_QUERIES, ATOM_WINDOW_KEYS),
        "probabilities": (windows, ATOM_HEADS, ATOM_WINDOW_QUERIES, ATOM_WINDOW_KEYS),
        "attended": (
            windows,
            ATOM_HEADS,
            ATOM_WINDOW_QUERIES,
            ATOM_CHANNELS // ATOM_HEADS,
        ),
        "gate": (windows, ATOM_WINDOW_QUERIES, ATOM_CHANNELS),
        "gated_output": (windows, ATOM_WINDOW_QUERIES, ATOM_CHANNELS),
        "projected_output": (windows, ATOM_WINDOW_QUERIES, ATOM_CHANNELS),
        "output_projection": (windows, ATOM_WINDOW_QUERIES, ATOM_CHANNELS),
        "result": (windows, ATOM_WINDOW_QUERIES, ATOM_CHANNELS),
    }


ATOM_ATTENTION_DETAIL_SHAPES = atom_attention_detail_shapes(ATOM_COUNT)


@dataclass(frozen=True)
class InputEmbedderBuildResult:
    engine_path: str
    engine_sha256: str
    engine_size_bytes: int
    build_seconds: float
    token_count: int
    atom_count: int
    precision: str


def _indexing_matrix(windows: int = ATOM_WINDOWS) -> np.ndarray:
    half_windows = 2 * windows
    halves_per_key_window = ATOM_WINDOW_KEYS // (ATOM_WINDOW_QUERIES // 2)
    indices = np.arange(half_windows)
    buckets = np.clip(
        indices[None, :] - indices[:, None] + halves_per_key_window // 2,
        0,
        halves_per_key_window + 1,
    )
    buckets = buckets.reshape(windows, 2, half_windows)[:, 0, :]
    one_hot = np.eye(halves_per_key_window + 2, dtype=np.float32)[buckets]
    return (
        one_hot[..., 1:-1]
        .transpose(1, 0, 2)
        .reshape(
            half_windows,
            halves_per_key_window * windows,
        )
    )


def _to_keys(graph: Graph, tensor: Any):
    batch, atoms, channels = (int(dim) for dim in tensor.shape)
    windows = atom_windows(atoms)
    halves = graph.reshape(
        tensor,
        (batch, 2 * windows, ATOM_WINDOW_QUERIES // 2, channels),
    )
    matrix = graph.constant(_indexing_matrix(windows))
    matrix = graph.cast(matrix, halves.dtype)
    keys = graph.einsum((halves, matrix), "bjid,jk->bkid")
    return graph.reshape(keys, (batch, windows, ATOM_WINDOW_KEYS, channels))


def _adaln(
    graph: Graph,
    tensor: Any,
    condition: Any,
    prefix: str,
    compute_dtype: Any,
):
    normalized = graph.unit_layer_norm(tensor)
    normalized_condition = graph.layer_norm(condition, f"{prefix}.s_norm")
    compute_condition = graph.cast(normalized_condition, compute_dtype)
    scale = graph.sigmoid(graph.linear(compute_condition, f"{prefix}.s_scale"))
    bias = graph.linear(compute_condition, f"{prefix}.s_bias")
    scaled = graph.mul(graph.cast(scale, normalized.dtype), normalized)
    return graph.add(scaled, graph.cast(bias, scaled.dtype))


def _attention(
    graph: Graph,
    tensor: Any,
    condition: Any,
    bias: Any,
    mask: Any,
    prefix: str,
    compute_dtype: Any,
    debug_adapted_outputs: list[Any] | None = None,
    debug_detail_outputs: list[dict[str, Any]] | None = None,
):
    windows = int(tensor.shape[0])
    atoms = windows * ATOM_WINDOW_QUERIES
    adapted = _adaln(graph, tensor, condition, f"{prefix}.adaln", compute_dtype)
    if debug_adapted_outputs is not None:
        debug_adapted_outputs.append(adapted)
    compute = graph.cast(adapted, compute_dtype)

    def projected(name: str):
        value = graph.linear(compute, f"{prefix}.pair_bias_attn.proj_{name}")
        value = graph.reshape(
            value,
            (windows, ATOM_WINDOW_QUERIES, ATOM_HEADS, ATOM_CHANNELS // ATOM_HEADS),
        )
        return graph.transpose(value, (0, 2, 1, 3))

    query = projected("q")
    key_source = graph.reshape(adapted, (1, atoms, ATOM_CHANNELS))
    key_source = _to_keys(graph, key_source)
    key_source = graph.reshape(
        key_source,
        (windows, ATOM_WINDOW_KEYS, ATOM_CHANNELS),
    )
    key_source = graph.cast(key_source, compute_dtype)

    def projected_key(name: str):
        value = graph.linear(key_source, f"{prefix}.pair_bias_attn.proj_{name}")
        value = graph.reshape(
            value,
            (windows, ATOM_WINDOW_KEYS, ATOM_HEADS, ATOM_CHANNELS // ATOM_HEADS),
        )
        return graph.transpose(value, (0, 2, 1, 3))

    key = projected_key("k")
    value = projected_key("v")
    logits = graph.einsum(
        (
            graph.cast(query, graph.trt.float32),
            graph.cast(key, graph.trt.float32),
        ),
        "bhid,bhjd->bhij",
    )
    logits = graph.div(
        logits,
        graph.scalar_like(np.sqrt(ATOM_CHANNELS // ATOM_HEADS), logits),
    )
    scores = graph.add(
        logits,
        graph.transpose(graph.cast(bias, logits.dtype), (0, 3, 1, 2)),
    )
    key_mask = graph.reshape(mask, (1, atoms, 1))
    key_mask = _to_keys(graph, key_mask)
    key_mask = graph.reshape(key_mask, (windows, 1, 1, ATOM_WINDOW_KEYS))
    mask_bias = graph.mul(
        graph.sub(graph.scalar_like(1.0, key_mask), key_mask),
        graph.scalar_like(-1.0e6, key_mask),
    )
    scores = graph.add(scores, graph.cast(mask_bias, scores.dtype))
    probabilities = graph.softmax_last(scores)
    attended = graph.einsum(
        (probabilities, graph.cast(value, probabilities.dtype)),
        "bhij,bhjd->bhid",
    )
    output = graph.transpose(attended, (0, 2, 1, 3))
    output = graph.reshape(output, (windows, ATOM_WINDOW_QUERIES, ATOM_CHANNELS))
    gate = graph.sigmoid(graph.linear(compute, f"{prefix}.pair_bias_attn.proj_g"))
    gated_output = graph.mul(gate, graph.cast(output, gate.dtype))
    output = graph.linear(
        gated_output,
        f"{prefix}.pair_bias_attn.proj_o",
    )
    projection = graph.sigmoid(
        graph.linear(
            graph.cast(condition, compute_dtype),
            f"{prefix}.output_projection.0",
        )
    )
    result = graph.mul(projection, output)
    if debug_detail_outputs is not None:
        debug_detail_outputs.append(
            {
                "query": query,
                "key": key,
                "value": value,
                "logits": logits,
                "scores": scores,
                "probabilities": probabilities,
                "attended": attended,
                "gate": gate,
                "gated_output": gated_output,
                "projected_output": output,
                "output_projection": projection,
                "result": result,
            }
        )
    return result


def _conditioned_transition(
    graph: Graph,
    tensor: Any,
    condition: Any,
    prefix: str,
    compute_dtype: Any,
):
    adapted = _adaln(graph, tensor, condition, f"{prefix}.adaln", compute_dtype)
    compute = graph.cast(adapted, compute_dtype)
    gate_and_value = graph.linear(compute, f"{prefix}.swish_gate.0")
    shape = tuple(int(dim) for dim in gate_and_value.shape)
    half = shape[-1] // 2
    value = graph.slice(gate_and_value, (0, 0, 0), (*shape[:-1], half))
    gate = graph.slice(gate_and_value, (0, 0, half), (*shape[:-1], half))
    gated = graph.mul(value, graph.silu(gate))
    second = graph.linear(compute, f"{prefix}.a_to_b")
    update = graph.linear(graph.mul(gated, second), f"{prefix}.b_to_a")
    projection = graph.sigmoid(
        graph.linear(
            graph.cast(condition, compute_dtype),
            f"{prefix}.output_projection.0",
        )
    )
    return graph.mul(projection, update)


def _atom_transformer(
    graph: Graph,
    q: Any,
    c: Any,
    bias: Any,
    atom_mask: Any,
    *,
    layer_prefix: str,
    compute_dtype: Any,
    debug_outputs: list[Any] | None = None,
    debug_attention_outputs: list[Any] | None = None,
    debug_adapted_outputs: list[Any] | None = None,
    debug_attention_detail_outputs: list[dict[str, Any]] | None = None,
):
    atom_count = int(q.shape[1])
    windows = atom_windows(atom_count)
    q = graph.reshape(q, (windows, ATOM_WINDOW_QUERIES, ATOM_CHANNELS))
    c = graph.reshape(c, (windows, ATOM_WINDOW_QUERIES, ATOM_CHANNELS))
    atom_mask = graph.cast(atom_mask, graph.trt.float32)
    bias = graph.reshape(
        bias,
        (windows, ATOM_WINDOW_QUERIES, ATOM_WINDOW_KEYS, ATOM_LAYERS, ATOM_HEADS),
    )
    for layer in range(ATOM_LAYERS):
        prefix = f"{layer_prefix}.{layer}"
        layer_bias = graph.slice(
            bias,
            (0, 0, 0, layer, 0),
            (windows, ATOM_WINDOW_QUERIES, ATOM_WINDOW_KEYS, 1, ATOM_HEADS),
        )
        layer_bias = graph.reshape(
            layer_bias,
            (windows, ATOM_WINDOW_QUERIES, ATOM_WINDOW_KEYS, ATOM_HEADS),
        )
        q = graph.add(
            q,
            graph.cast(
                _attention(
                    graph,
                    q,
                    c,
                    layer_bias,
                    atom_mask,
                    prefix,
                    compute_dtype,
                    debug_adapted_outputs,
                    debug_attention_detail_outputs,
                ),
                q.dtype,
            ),
        )
        if debug_attention_outputs is not None:
            debug_attention_outputs.append(q)
        update = _conditioned_transition(
            graph,
            q,
            c,
            f"{prefix}.transition",
            compute_dtype,
        )
        q = graph.add(q, graph.cast(update, q.dtype))
        if debug_outputs is not None:
            debug_outputs.append(q)
    return graph.reshape(q, (1, atom_count, ATOM_CHANNELS))


def define_input_embedder_network(
    network: Any,
    trt: Any,
    weights: dict[str, np.ndarray],
    *,
    token_count: int,
    atom_count: int,
):
    """Define the exact pinned no-affinity Boltz-2 input embedding graph."""

    windows = atom_windows(atom_count)
    bf16 = getattr(trt, "bfloat16", None)
    if bf16 is None:
        raise RuntimeError("Boltz-2 requires TensorRT with strongly typed BF16 support")
    graph = Graph(network, trt, weights)
    ref_pos = network.add_input("ref_pos", trt.float32, (1, atom_count, 3))
    ref_space_uid = network.add_input("ref_space_uid", trt.int32, (1, atom_count))
    ref_charge = network.add_input("ref_charge", trt.float32, (1, atom_count))
    ref_element = network.add_input("ref_element", trt.int32, (1, atom_count, 128))
    ref_atom_name_chars = network.add_input(
        "ref_atom_name_chars", trt.int32, (1, atom_count, 4, 64)
    )
    atom_to_token = network.add_input("atom_to_token", trt.int32, (1, atom_count, token_count))
    atom_pad_mask = network.add_input("atom_pad_mask", trt.float32, (1, atom_count))
    res_type = network.add_input("res_type", trt.int32, (1, token_count, 33))
    profile = network.add_input("profile", trt.float32, (1, token_count, 33))
    deletion_mean = network.add_input("deletion_mean", trt.float32, (1, token_count))
    method_feature = network.add_input("method_feature", trt.int32, (1, token_count))
    modified = network.add_input("modified", trt.int32, (1, token_count))
    cyclic_period = network.add_input("cyclic_period", trt.float32, (1, token_count))
    mol_type = network.add_input("mol_type", trt.int32, (1, token_count))

    atom_features = graph.concatenate(
        (
            ref_pos,
            graph.reshape(ref_charge, (1, atom_count, 1)),
            graph.cast(ref_element, trt.float32),
            graph.cast(graph.reshape(ref_atom_name_chars, (1, atom_count, 256)), trt.float32),
        ),
        2,
    )
    c = graph.linear(atom_features, "input_embedder.atom_encoder.embed_atom_features")
    q = c

    ref_queries = graph.reshape(ref_pos, (1, windows, ATOM_WINDOW_QUERIES, 1, 3))
    ref_keys = graph.reshape(_to_keys(graph, ref_pos), (1, windows, 1, ATOM_WINDOW_KEYS, 3))
    displacement = graph.sub(ref_keys, ref_queries)
    squared = graph.mul(displacement, displacement)
    squared = graph.reduce_sum(squared, 4, keep_dims=True)
    reciprocal_distance = graph.div(
        graph.scalar_like(1.0, squared),
        graph.add(graph.scalar_like(1.0, squared), squared),
    )

    atom_mask = graph.cast(atom_pad_mask, trt.bool)
    mask_queries = graph.reshape(atom_mask, (1, windows, ATOM_WINDOW_QUERIES, 1))
    mask_keys = graph.reshape(
        _to_keys(graph, graph.reshape(atom_pad_mask, (1, atom_count, 1))),
        (1, windows, 1, ATOM_WINDOW_KEYS),
    )
    uid_queries = graph.reshape(ref_space_uid, (1, windows, ATOM_WINDOW_QUERIES, 1))
    uid_keys = graph.reshape(
        _to_keys(graph, graph.cast(graph.reshape(ref_space_uid, (1, atom_count, 1)), trt.float32)),
        (1, windows, 1, ATOM_WINDOW_KEYS),
    )
    same_uid = graph.equal(uid_queries, graph.cast(uid_keys, uid_queries.dtype))
    valid = graph.elementwise(
        mask_queries, graph.cast(mask_keys, trt.bool), trt.ElementWiseOperation.AND
    )
    valid = graph.elementwise(valid, same_uid, trt.ElementWiseOperation.AND)
    valid = graph.cast(
        graph.reshape(valid, (1, windows, ATOM_WINDOW_QUERIES, ATOM_WINDOW_KEYS, 1)),
        trt.float32,
    )

    def masked_linear(tensor: Any, prefix: str):
        return graph.mul(graph.linear(tensor, prefix), valid)

    p = masked_linear(displacement, "input_embedder.atom_encoder.embed_atompair_ref_pos")
    p = graph.add(
        p,
        masked_linear(reciprocal_distance, "input_embedder.atom_encoder.embed_atompair_ref_dist"),
    )
    p = graph.add(p, masked_linear(valid, "input_embedder.atom_encoder.embed_atompair_mask"))
    c_queries = graph.reshape(c, (1, windows, ATOM_WINDOW_QUERIES, 1, ATOM_CHANNELS))
    c_keys = graph.reshape(
        _to_keys(graph, c), (1, windows, 1, ATOM_WINDOW_KEYS, ATOM_CHANNELS)
    )
    p = graph.add(
        p,
        graph.linear(graph.relu(c_queries), "input_embedder.atom_encoder.c_to_p_trans_q.1"),
    )
    p = graph.add(
        p,
        graph.linear(graph.relu(c_keys), "input_embedder.atom_encoder.c_to_p_trans_k.1"),
    )
    p_update = graph.relu(p)
    p_update = graph.linear(p_update, "input_embedder.atom_encoder.p_mlp.1")
    p_update = graph.linear(graph.relu(p_update), "input_embedder.atom_encoder.p_mlp.3")
    p_update = graph.linear(graph.relu(p_update), "input_embedder.atom_encoder.p_mlp.5")
    p = graph.add(p, p_update)

    normalized_p = graph.layer_norm(p, "input_embedder.atom_enc_proj_z.0")
    atom_bias = graph.linear(graph.cast(normalized_p, bf16), "input_embedder.atom_enc_proj_z.1")
    q = _atom_transformer(
        graph,
        q,
        c,
        atom_bias,
        atom_pad_mask,
        layer_prefix=(
            "input_embedder.atom_attention_encoder.atom_encoder.diffusion_transformer.layers"
        ),
        compute_dtype=bf16,
    )

    # The token pooling block explicitly disables autocast and therefore stays FP32.
    q_to_a = graph.relu(
        graph.linear(q, "input_embedder.atom_attention_encoder.atom_to_token_trans.0")
    )
    token_map = graph.cast(atom_to_token, trt.float32)
    token_counts = graph.reduce_sum(token_map, 1, keep_dims=True)
    token_map = graph.div(
        token_map, graph.add(token_counts, graph.scalar_like(1.0e-6, token_counts))
    )
    token_map = graph.transpose(token_map, (0, 2, 1))
    pooled = graph.network.add_matrix_multiply(
        token_map,
        trt.MatrixOperation.NONE,
        q_to_a,
        trt.MatrixOperation.NONE,
    ).get_output(0)

    token_embedding = graph.linear(graph.cast(res_type, bf16), "input_embedder.res_type_encoding")
    profile_input = graph.concatenate(
        (profile, graph.reshape(deletion_mean, (1, token_count, 1))), 2
    )
    profile_embedding = graph.linear(
        graph.cast(profile_input, bf16), "input_embedder.msa_profile_encoding"
    )
    s_inputs = graph.add(pooled, graph.cast(token_embedding, pooled.dtype))
    s_inputs = graph.add(s_inputs, graph.cast(profile_embedding, s_inputs.dtype))
    for indices, prefix in (
        (method_feature, "input_embedder.method_conditioning_init"),
        (modified, "input_embedder.modified_conditioning_init"),
        (mol_type, "input_embedder.mol_type_conditioning_init"),
    ):
        embedded = graph.embedding(indices, prefix, bf16)
        s_inputs = graph.add(s_inputs, graph.cast(embedded, s_inputs.dtype))
    cyclic = graph.minimum(cyclic_period, graph.scalar_like(1.0, cyclic_period))
    cyclic = graph.reshape(cyclic, (1, token_count, 1))
    cyclic = graph.linear(graph.cast(cyclic, bf16), "input_embedder.cyclic_conditioning_init")
    s_inputs = graph.add(s_inputs, graph.cast(cyclic, s_inputs.dtype))
    s_inputs.name = "s_inputs"
    network.mark_output(s_inputs)
    return s_inputs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_input_embedder_engine(
    checkpoint_path: Path,
    engine_path: Path,
    *,
    token_count: int = 117,
    atom_count: int = ATOM_COUNT,
    workspace_bytes: int = 16 << 30,
    avg_timing_iterations: int = 8,
    verbose: bool = False,
    verify_checkpoint: bool = True,
) -> InputEmbedderBuildResult:
    """Build the direct static-profile Boltz-2 atom/token input engine."""

    _, weights = load_weight_prefixes(
        checkpoint_path,
        ("input_embedder.",),
        verify=verify_checkpoint,
    )
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(strongly_typed=True, explicit_batch=True)
    )
    define_input_embedder_network(
        network,
        trt,
        weights,
        token_count=token_count,
        atom_count=atom_count,
    )
    config = builder.create_builder_config()
    config.avg_timing_iterations = avg_timing_iterations
    config.max_aux_streams = 0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    started = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if plan is None:
        raise RuntimeError("TensorRT failed to build the Boltz-2 input embedder engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(plan)
    return InputEmbedderBuildResult(
        engine_path=str(engine_path),
        engine_sha256=_sha256(engine_path),
        engine_size_bytes=engine_path.stat().st_size,
        build_seconds=build_seconds,
        token_count=token_count,
        atom_count=atom_count,
        precision="bf16-mixed",
    )
