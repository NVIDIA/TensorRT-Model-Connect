"""Tensor-parallel MagpieTTS decoder builder.

This mirrors the decoder graph in ``plugin.py`` and only changes the
rank-local projection widths. Self-attention Q/K/V and FC1 are
column-parallel, self-attention output and FC2 are row-parallel and joined
with TensorRT distributed ALL_REDUCE. Magpie's asymmetric cross-attention has
one head in the current checkpoint, so it stays replicated on every rank.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import model as graph_ops
from ....parallel_config import (
    add_all_reduce_sum,
    normalize_parallel_config,
)

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ..weights import WeightDict
    from ..config import ModelConfig
    from ....parallel_config import ParallelConfig


def _mark_debug_output(network, tensor, name):
    identity = network.add_identity(tensor)
    cast = network.add_cast(identity.get_output(0), trt.float32)
    out = cast.get_output(0)
    out.name = name
    network.mark_output(out)


def _slice_first_dim(value: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    if value.shape[0] % tp_size != 0:
        raise ValueError(f"Cannot shard first dimension {value.shape[0]} over TP{tp_size}")
    chunk = value.shape[0] // tp_size
    return np.ascontiguousarray(value[rank * chunk : (rank + 1) * chunk])


def _slice_last_dim(value: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    if value.shape[-1] % tp_size != 0:
        raise ValueError(f"Cannot shard last dimension {value.shape[-1]} over TP{tp_size}")
    chunk = value.shape[-1] // tp_size
    slc = [slice(None)] * value.ndim
    slc[-1] = slice(rank * chunk, (rank + 1) * chunk)
    return np.ascontiguousarray(value[tuple(slc)])


def validate_magpie_decoder_tp(weights: "WeightDict", parallel: "ParallelConfig") -> None:
    parallel.validate()
    if not parallel.enabled:
        raise ValueError("MagpieTTS tensor-parallel builder requires an enabled parallel config")
    if parallel.rank < 0:
        raise ValueError("MagpieTTS tensor-parallel engine build requires a concrete rank")

    tp = parallel.tp_size
    hidden = int(weights["_hidden_size"])
    dec_heads = int(weights["_dec_heads"])
    dec_ffn = int(weights["_dec_ffn"])
    if hidden % tp != 0:
        raise ValueError(f"MagpieTTS hidden_size={hidden} must be divisible by TP{tp}")
    if dec_heads % tp != 0:
        raise ValueError(f"MagpieTTS decoder heads={dec_heads} must be divisible by TP{tp}")
    if dec_ffn % tp != 0:
        raise ValueError(f"MagpieTTS decoder FFN size={dec_ffn} must be divisible by TP{tp}")


def shard_magpie_decoder_weights(weights: "WeightDict", parallel_config=None) -> "WeightDict":
    """Return rank-local weights for the Magpie decoder."""
    parallel = normalize_parallel_config(parallel_config)
    validate_magpie_decoder_tp(weights, parallel)

    rank = parallel.rank
    tp = parallel.tp_size
    dec_layers = int(weights["_dec_layers"])
    out = type(weights)()

    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
            continue

        handled = False
        for layer_idx in range(dec_layers):
            pfx = f"layer.{layer_idx}"
            if key in {f"{pfx}.w_q", f"{pfx}.w_k", f"{pfx}.w_v", f"{pfx}.w_fc1"}:
                out[key] = _slice_last_dim(value, rank, tp)
                handled = True
                break
            if key in {f"{pfx}.w_o", f"{pfx}.w_fc2"}:
                out[key] = _slice_first_dim(value, rank, tp)
                handled = True
                break
        if not handled:
            out[key] = value

    out["_tensor_parallel_size"] = tp
    out["_tensor_parallel_rank"] = rank
    return out


def build_magpie_tp_decoder_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx=None,
    verbose: bool = False,
    debug_layer_outputs: bool = False,
    parallel_config=None,
) -> bytes:
    del config
    if quant_ctx is not None:
        raise ValueError("MagpieTTS tensor-parallel builds do not support quantization")

    parallel = normalize_parallel_config(parallel_config)
    validate_magpie_decoder_tp(weights, parallel)

    dec_layers = int(weights["_dec_layers"])
    dec_heads = int(weights["_dec_heads"])
    dec_ffn = int(weights["_dec_ffn"])
    hidden = int(weights["_hidden_size"])
    num_codebooks = int(weights["_num_codebooks"])
    codebook_size = int(weights["_codebook_size"])
    max_source_positions = int(weights["_max_source_positions"])
    xa_n_heads = int(weights["_xa_n_heads"])
    xa_d_head = int(weights["_xa_d_head"])
    head_dim = hidden // dec_heads
    local_heads = dec_heads // parallel.tp_size
    local_attention_size = local_heads * head_dim
    local_ffn = dec_ffn // parallel.tp_size
    output_size = num_codebooks * codebook_size
    rank_weights = shard_magpie_decoder_weights(weights, parallel)

    ctx_len = 1
    if "baked_context_lengths" in weights:
        ctx_lengths = np.asarray(weights["baked_context_lengths"], dtype=np.int32)
        if ctx_lengths.size > 0:
            ctx_len = max(int(ctx_lengths.max()), 1)

    W = max_cache_length

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    trt_config.clear_flag(trt.BuilderFlag.TF32)

    input_embed = network.add_input("input_embed", trt.float32, (-1, hidden))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    attention_mask = network.add_input("attention_mask", trt.float32, (1, -1, -1))

    cache_k_inputs, cache_v_inputs = [], []
    for i in range(dec_layers):
        cache_k_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("cache_k", i),
                trt.float32,
                (max_cache_length, local_attention_size),
            )
        )
        cache_v_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("cache_v", i),
                trt.float32,
                (max_cache_length, local_attention_size),
            )
        )

    cross_kv_dtype = trt.float16 if precision == "fp16" else trt.float32
    cross_k_inputs, cross_v_inputs = [], []
    for i in range(dec_layers):
        cross_k_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("cross_k", i),
                cross_kv_dtype,
                (max_source_positions, hidden),
            )
        )
        cross_v_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("cross_v", i),
                cross_kv_dtype,
                (max_source_positions, hidden),
            )
        )

    cross_attn_prior = network.add_input(
        "cross_attn_prior", trt.float32, (1, 1, max_source_positions)
    )
    prior_layers = set(range(3, 10))

    ar_profile = builder.create_optimization_profile()
    ar_profile.set_shape("input_embed", (1, hidden), (1, hidden), (ctx_len, hidden))
    ar_profile.set_shape("position_id", (1,), (1,), (ctx_len,))
    ar_profile.set_shape("attention_mask", (1, 1, W + 1), (1, 1, W + 1), (1, ctx_len, W + ctx_len))
    trt_config.add_optimization_profile(ar_profile)

    pf_profile = builder.create_optimization_profile()
    pf_profile.set_shape("input_embed", (1, hidden), (ctx_len, hidden), (ctx_len, hidden))
    pf_profile.set_shape("position_id", (1,), (ctx_len,), (ctx_len,))
    pf_profile.set_shape(
        "attention_mask",
        (1, 1, W + 1),
        (1, ctx_len, W + ctx_len),
        (1, ctx_len, W + ctx_len),
    )
    trt_config.add_optimization_profile(pf_profile)

    dec_pos_np = rank_weights["dec_pos_embedding"]
    pos_table = graph_ops.add_constant(network, dec_pos_np.shape, dec_pos_np)
    pos_embed = network.add_gather(pos_table, position_id, 0)
    hidden_state = network.add_elementwise(
        input_embed, pos_embed.get_output(0), trt.ElementWiseOperation.SUM
    ).get_output(0)

    eps_tensor = graph_ops.add_constant(network, (1, 1), np.array([1e-5], dtype=np.float32))
    xa_scale_tensor = graph_ops.add_constant(
        network, (1, 1, 1), np.array([1.0 / np.sqrt(max(xa_d_head, 1))], dtype=np.float32)
    )

    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")

    present_k_outputs, present_v_outputs = [], []
    alignment_layers = [3, 4, 5, 6]
    alignment_weights = []
    for layer_idx in range(dec_layers):
        prefix = f"layer.{layer_idx}"
        layer_prior = cross_attn_prior if layer_idx in prior_layers else None
        result = _add_magpie_tp_decoder_layer(
            network=network,
            hidden=hidden_state,
            cache_k=cache_k_inputs[layer_idx],
            cache_v=cache_v_inputs[layer_idx],
            cross_k=cross_k_inputs[layer_idx],
            cross_v=cross_v_inputs[layer_idx],
            attention_mask=attention_mask,
            xa_scale_tensor=xa_scale_tensor,
            eps_tensor=eps_tensor,
            weights=rank_weights,
            prefix=prefix,
            hidden_size=hidden,
            local_attention_size=local_attention_size,
            local_heads=local_heads,
            head_dim=head_dim,
            local_ffn=local_ffn,
            max_source_positions=max_source_positions,
            xa_n_heads=xa_n_heads,
            xa_d_head=xa_d_head,
            tp_size=parallel.tp_size,
            cross_attn_prior=layer_prior,
        )
        hidden_state = result["hidden"]
        present_k_outputs.append(result["present_k"])
        present_v_outputs.append(result["present_v"])
        if layer_idx in alignment_layers:
            alignment_weights.append(result["cross_attn_weights"])
        if layer_idx == dec_layers - 1:
            _mark_debug_output(network, result["cross_attn_weights"], "cross_attn_weights")
        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

    if len(alignment_weights) >= 2:
        avg = alignment_weights[0]
        for aw in alignment_weights[1:]:
            avg = network.add_elementwise(avg, aw, trt.ElementWiseOperation.SUM).get_output(0)
        n_align = graph_ops.add_constant(
            network, (1, 1, 1), np.array([1.0 / len(alignment_weights)], dtype=np.float32)
        )
        avg = network.add_elementwise(avg, n_align, trt.ElementWiseOperation.PROD).get_output(0)
        avg_over_heads = network.add_reduce(avg, trt.ReduceOperation.AVG, 1 << 0, True)
        _mark_debug_output(network, avg_over_heads.get_output(0), "alignment_weights")

    hidden_state = graph_ops.add_layer_norm(
        network,
        hidden_state,
        hidden,
        rank_weights["final_norm"],
        np.zeros(hidden, dtype=np.float32),
        eps_tensor,
    )
    _mark_debug_output(network, hidden_state, "decoder_hidden")

    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, output_size, rank_weights["w_out"]
    )
    logits = graph_ops.add_bias_sum(network, logits, output_size, rank_weights["w_out_bias"])
    logits.name = "logits"
    network.mark_output(logits)

    for i in range(dec_layers):
        present_k_outputs[i].name = graph_ops.layer_tensor_name("present_k", i)
        present_v_outputs[i].name = graph_ops.layer_tensor_name("present_v", i)
        network.mark_output(present_k_outputs[i])
        network.mark_output(present_v_outputs[i])

    if verbose:
        print(
            f"[trtmc build] Building MagpieTTS TP decoder rank "
            f"{parallel.rank}/{parallel.tp_size} ({dec_layers}L, h={hidden}, "
            f"local_sa_heads={local_heads}, local_attn={local_attention_size}, "
            f"xa_heads={xa_n_heads} d_head={xa_d_head}, local_ffn={local_ffn}, "
            f"cache={max_cache_length}, output={num_codebooks}x{codebook_size}, "
            f"prefill_ctx_len={ctx_len})",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT MagpieTTS tensor-parallel decoder engine build failed")
    return bytes(plan)


def _add_magpie_tp_decoder_layer(
    *,
    network,
    hidden,
    cache_k,
    cache_v,
    cross_k,
    cross_v,
    attention_mask,
    xa_scale_tensor,
    eps_tensor,
    weights,
    prefix,
    hidden_size,
    local_attention_size,
    local_heads,
    head_dim,
    local_ffn,
    max_source_positions,
    xa_n_heads,
    xa_d_head,
    tp_size,
    cross_attn_prior=None,
):
    xa_attention_size = xa_n_heads * xa_d_head

    normed = graph_ops.add_layer_norm(
        network,
        hidden,
        hidden_size,
        weights[f"{prefix}.input_norm"],
        np.zeros(hidden_size, dtype=np.float32),
        eps_tensor,
    )

    q = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, local_attention_size, weights[f"{prefix}.w_q"]
    )
    k = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, local_attention_size, weights[f"{prefix}.w_k"]
    )
    v = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, local_attention_size, weights[f"{prefix}.w_v"]
    )

    present_k = k
    present_v = v

    ak = network.add_concatenation([cache_k, k])
    ak.axis = 0
    av = network.add_concatenation([cache_v, v])
    av.axis = 0

    mask_4d = graph_ops.add_3d_mask_to_4d(network, attention_mask)
    cf = graph_ops.add_attention_from_rows(
        network,
        q,
        ak.get_output(0),
        av.get_output(0),
        num_heads=local_heads,
        head_dim=head_dim,
        q_seq=None,
        kv_seq=None,
        mask=mask_4d,
    )

    sa = graph_ops.add_matmul_rhs_constant(
        network, cf, local_attention_size, hidden_size, weights[f"{prefix}.w_o"]
    )
    sa = add_all_reduce_sum(network, sa, tp_size)
    psa = network.add_elementwise(hidden, sa, trt.ElementWiseOperation.SUM).get_output(0)

    cn_query = graph_ops.add_layer_norm(
        network,
        psa,
        hidden_size,
        weights[f"{prefix}.norm_xattn_query"],
        np.zeros(hidden_size, dtype=np.float32),
        eps_tensor,
    )

    cn_memory = graph_ops.add_layer_norm(
        network,
        cross_k,
        hidden_size,
        weights[f"{prefix}.norm_xattn_memory"],
        np.zeros(hidden_size, dtype=np.float32),
        eps_tensor,
    )

    cq = graph_ops.add_matmul_rhs_constant(
        network, cn_query, hidden_size, xa_attention_size, weights[f"{prefix}.cross_w_q"]
    )
    ck_proj = graph_ops.add_matmul_rhs_constant(
        network, cn_memory, hidden_size, xa_attention_size, weights[f"{prefix}.cross_w_k"]
    )
    cv_proj = graph_ops.add_matmul_rhs_constant(
        network, cn_memory, hidden_size, xa_attention_size, weights[f"{prefix}.cross_w_v"]
    )

    cqh = network.add_shuffle(cq)
    cqh.reshape_dims = (-1, xa_n_heads, xa_d_head)
    cqh.second_transpose = trt.Permutation([1, 0, 2])
    ckh = network.add_shuffle(ck_proj)
    ckh.reshape_dims = (max_source_positions, xa_n_heads, xa_d_head)
    ckh.second_transpose = trt.Permutation([1, 0, 2])
    cvh = network.add_shuffle(cv_proj)
    cvh.reshape_dims = (max_source_positions, xa_n_heads, xa_d_head)
    cvh.second_transpose = trt.Permutation([1, 0, 2])

    cs = network.add_elementwise(
        network.add_matrix_multiply(
            cqh.get_output(0),
            trt.MatrixOperation.NONE,
            ckh.get_output(0),
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0),
        xa_scale_tensor,
        trt.ElementWiseOperation.PROD,
    )
    csm = network.add_softmax(cs.get_output(0))
    csm.axes = 1 << 2

    if cross_attn_prior is not None:
        attn_weighted = network.add_elementwise(
            csm.get_output(0), cross_attn_prior, trt.ElementWiseOperation.PROD
        )
        sum_layer = network.add_reduce(
            attn_weighted.get_output(0), trt.ReduceOperation.SUM, 1 << 2, True
        )
        eps_norm = graph_ops.add_constant(network, (1, 1, 1), np.array([1e-8], dtype=np.float32))
        sum_safe = network.add_elementwise(
            sum_layer.get_output(0), eps_norm, trt.ElementWiseOperation.SUM
        )
        csm_final = network.add_elementwise(
            attn_weighted.get_output(0), sum_safe.get_output(0), trt.ElementWiseOperation.DIV
        )
    else:
        csm_final = csm

    cc = network.add_matrix_multiply(
        csm_final.get_output(0),
        trt.MatrixOperation.NONE,
        cvh.get_output(0),
        trt.MatrixOperation.NONE,
    )
    ccf = network.add_shuffle(cc.get_output(0))
    ccf.first_transpose = trt.Permutation([1, 0, 2])
    ccf.reshape_dims = (-1, xa_attention_size)

    ca = graph_ops.add_matmul_rhs_constant(
        network, ccf.get_output(0), xa_attention_size, hidden_size, weights[f"{prefix}.cross_w_o"]
    )
    pca = network.add_elementwise(psa, ca, trt.ElementWiseOperation.SUM).get_output(0)

    fn = graph_ops.add_layer_norm(
        network,
        pca,
        hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        np.zeros(hidden_size, dtype=np.float32),
        eps_tensor,
    )

    fc1 = graph_ops.add_matmul_rhs_constant(
        network, fn, hidden_size, local_ffn, weights[f"{prefix}.w_fc1"]
    )
    act = graph_ops.add_activation(network, fc1, "gelu_new")
    fc2 = graph_ops.add_matmul_rhs_constant(
        network, act, local_ffn, hidden_size, weights[f"{prefix}.w_fc2"]
    )
    fc2 = add_all_reduce_sum(network, fc2, tp_size)

    out = network.add_elementwise(pca, fc2, trt.ElementWiseOperation.SUM).get_output(0)

    return {
        "hidden": out,
        "present_k": present_k,
        "present_v": present_v,
        "cross_attn_weights": csm.get_output(0),
    }
