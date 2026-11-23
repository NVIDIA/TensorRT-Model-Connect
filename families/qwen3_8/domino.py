# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Qwen3.6 Domino correction. The target owns its token embedding."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorrt as trt

from . import draft, graph_ops


@dataclass(frozen=True)
class DominoConfig:
    raw: dict
    feature_layers: tuple[int, ...]
    hidden: int
    vocab: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    mask_token: int
    epsilon: float
    rank: int
    state: int
    shift_label: bool
    pool_size: int
    query_rows: int = 16

    @classmethod
    def from_dict(cls, raw, target):
        policy = raw.get("dflash_config") or {}
        h, v, layers = raw["hidden_size"], raw["vocab_size"], raw["num_hidden_layers"]
        taps = tuple(policy.get("target_layer_ids", ()))
        if (raw.get("architectures") != ["DFlashDraftModel"] or raw.get("block_size") != 16
                or policy.get("projector_type") != "domino" or policy.get("pure_draft_prefix_len") != 1
                or type(policy.get("shift_label")) is not bool
                or (h, v, raw.get("num_target_layers")) !=
                   (target["hidden_size"], target["vocab_size"], target["num_hidden_layers"])
                or not taps or len(set(taps)) != len(taps)
                or any(type(i) is not int or not 0 <= i < target["num_hidden_layers"] for i in taps)
                or raw.get("quantization_config") or target.get("quantization_config")
                or raw.get("attention_bias", False) or raw.get("hidden_act") != "silu"
                or raw.get("sliding_window") or raw.get("use_sliding_window")
                or raw.get("layer_types") != ["full_attention"] * layers):
            raise ValueError("Unsupported Qwen3.6 Domino companion or target contract")
        heads, kv, dim = raw["num_attention_heads"], raw["num_key_value_heads"], raw["head_dim"]
        width, rank, state = raw["intermediate_size"], policy["emb_dim"], policy["gru_hidden_dim"]
        mask, eps = policy["mask_token_id"], raw["rms_norm_eps"]
        if (min(h, v, layers, heads, kv, dim, width, rank, state) <= 0 or heads % kv
                or dim % 2 or state % 2 or rank % 2 or not 0 <= mask < v
                or not np.isfinite(eps) or eps <= 0):
            raise ValueError("Invalid Qwen3.6 Domino dimensions")
        return cls(raw, taps, h, v, layers, heads, kv, dim, width, mask, eps, rank, state,
                   policy["shift_label"], min(v, 2048))


def load_draft_weights(model_dir, config):
    specs = list(draft.block_weight_shapes(config))
    h, state, rank = config.hidden, config.state, config.rank
    specs.extend([
        ("prefix_gru.weight_ih_l0", (3 * state, h), False),
        ("prefix_gru.weight_hh_l0", (3 * state, state), False),
        ("embed_proj.0.weight", (rank, h + state), False),
        ("embed_proj.2.weight", (config.vocab, rank), False),
    ])
    return draft.load_block_weights(model_dir, specs)


def build_draft_engine(config, weights, target_lm_head, max_cache_length, *, quant_ctx=None,
                       verbose=False, debug_layer_outputs=False):
    if quant_ctx is not None or target_lm_head is None:
        raise ValueError("The Qwen3.6 Domino target needs its original dense LM head")

    def outputs(net, hidden):
        tensors = add_candidate_pool(net, hidden, target_lm_head, weights["embed_proj.0.weight"],
                                     weights["embed_proj.2.weight"], config.pool_size,
                                     shift_label=config.shift_label)
        return dict(zip(("pool_ids", "pool_base_logits", "pool_codebook", "proposal_zproj"), tensors))

    return draft.build_block_engine(config, weights, max_cache_length, outputs, verbose=verbose,
                                    debug_layer_outputs=debug_layer_outputs)


def build_correction_engine(weight_ih, weight_hh, proj_state, pool_size, *, verbose=False):
    """Advance the bias-free GRU by one token and score the next proposal pool.

    Both greedy schedules seed an all-zero state with the confirmed anchor and
    the first, uncorrected proposal. The released sidecar stores each target
    embedding's input projection in FP16; project just the requested token and
    round at the same boundary instead of duplicating that vocabulary table.
    """
    if any(w.ndim != 2 or w.dtype != np.float16 for w in (weight_ih, weight_hh, proj_state)):
        raise ValueError("Domino correction weights must be FP16 matrices")
    gates, hidden = weight_hh.shape
    width, emb = weight_ih.shape
    rank, state = proj_state.shape
    if (min(hidden, emb, rank, pool_size) <= 0 or width != gates or gates != 3 * hidden
            or state != hidden or hidden % 2 or rank % 2):
        raise ValueError("Domino correction dimensions do not match the trained GRU and projection")
    log = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(log)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    config.clear_flag(trt.BuilderFlag.TF32)

    def const(value):
        a = np.ascontiguousarray(value)
        return graph_ops.add_constant(net, a.shape, a, dtype=a.dtype.type)

    def cast(x, dtype):
        return x if x.dtype == dtype else net.add_cast(x, dtype).get_output(0)

    def fp32(x):
        return cast(x, trt.float32)

    def linear(x, weights):
        return net.add_matrix_multiply(fp32(x), trt.MatrixOperation.NONE, fp32(weights),
                                       trt.MatrixOperation.TRANSPOSE).get_output(0)

    def op(x, y, kind):
        return net.add_elementwise(x, y, kind).get_output(0)

    def gate(x, i):
        return net.add_slice(x, (0, i * hidden), (1, hidden), (1, 1)).get_output(0)

    def act(x, kind):
        return net.add_activation(x, kind).get_output(0)

    half = trt.float16
    ew, ac = trt.ElementWiseOperation, trt.ActivationType
    embedding = net.add_input("token_embedding", half, (1, emb))
    prev = net.add_input("previous_state", half, (1, hidden))
    zproj = net.add_input("z_projection", half, (1, rank))
    book = net.add_input("pool_codebook", half, (pool_size, rank))
    base = net.add_input("base_scores", trt.float32, (1, pool_size))
    x = fp32(cast(linear(embedding, const(weight_ih)), half))
    hx = linear(prev, const(weight_hh))
    reset = act(op(gate(x, 0), gate(hx, 0), ew.SUM), ac.SIGMOID)
    update = act(op(gate(x, 1), gate(hx, 1), ew.SUM), ac.SIGMOID)
    new = act(op(gate(x, 2), op(reset, gate(hx, 2), ew.PROD), ew.SUM), ac.TANH)
    one = const(np.ones((1, 1), dtype=np.float32))
    next_state = cast(op(op(op(one, update, ew.SUB), new, ew.PROD),
                         op(update, fp32(prev), ew.PROD), ew.SUM), half)
    z = op(fp32(zproj), linear(next_state, const(proj_state)), ew.SUM)
    correction = op(z, act(z, ac.SIGMOID), ew.PROD)
    corrected = op(base, linear(correction, book), ew.SUM)
    for name, tensor in (("next_state", next_state), ("corrected_pool_scores", corrected)):
        tensor.name = name
        net.mark_output(tensor)
    engine = builder.build_serialized_network(net, config)
    if engine is None:
        raise RuntimeError("Failed to build native Qwen3.6 Domino correction engine")
    return bytes(engine)


def add_candidate_pool(net, block_hidden, lm_head, embed_proj, codebook, pool_size, *, shift_label):
    """Build the released shared candidate pool; never take independent top-k at each position.

    block_hidden is post-normalization FP16. The original Qwen3.6 target LM head
    is unquantized. z uses the first, hidden-size slice of embed_proj.0; its
    state slice is used by the separate recurrent correction engine.
    """
    block, hidden = map(int, block_hidden.shape)
    head_dim, vocab = lm_head.shape
    rank, combined = embed_proj.shape
    if (block != 16 or block_hidden.dtype != trt.float16 or hidden != head_dim
            or combined <= hidden or codebook.shape != (vocab, rank)
            or any(w.dtype != np.float16 for w in (lm_head, embed_proj, codebook))
            or not 0 < pool_size <= min(vocab, 3840)):
        raise ValueError("Invalid Qwen Domino block, shared pool or projection geometry")

    def constant(a):
        a = np.ascontiguousarray(a)
        return graph_ops.add_constant(net, a.shape, a, dtype=a.dtype.type)

    def matmul(x, w):
        return net.add_matrix_multiply(x, trt.MatrixOperation.NONE, constant(w),
                                       trt.MatrixOperation.TRANSPOSE).get_output(0)

    rows = block - 1
    proposal = net.add_slice(block_hidden, (0 if shift_label else 1, 0),
                             (rows, hidden), (1, 1)).get_output(0)
    # The target mapper already stored [hidden, vocab] contiguously.
    scores = net.add_matrix_multiply(proposal, trt.MatrixOperation.NONE, constant(lm_head),
                                     trt.MatrixOperation.NONE).get_output(0)
    logits = net.add_cast(scores, trt.float32).get_output(0)
    peaks = net.add_reduce(logits, trt.ReduceOperation.MAX, 1, True).get_output(0)
    pool = net.add_topk(peaks, trt.TopKOperation.MAX, pool_size, 2).get_output(1)
    flatten = net.add_shuffle(pool)
    flatten.reshape_dims = (pool_size,)
    ids = flatten.get_output(0)
    scores = net.add_gather(logits, ids, 1).get_output(0)
    pool_book = net.add_gather(constant(codebook), ids, 0).get_output(0)
    z_rows = net.add_slice(proposal, (1, 0), (rows - 1, hidden), (1, 1)).get_output(0)
    return ids, scores, pool_book, matmul(z_rows, embed_proj[:, :hidden])
