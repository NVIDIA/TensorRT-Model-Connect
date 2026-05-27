"""Tensor-parallel Marian decoder builder.

This duplicates the single-device Marian decoder graph and changes only the
rank-local projection pieces:

* self-attention and cross-attention Q/K/V projections are column-sharded,
* attention output and FFN output projections are row-sharded,
* row-parallel joins use TensorRT distributed ALL_REDUCE,
* embeddings, norms, encoder outputs, and the LM head stay replicated.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from ... import graph_ops
from ...parallel_config import add_all_reduce_sum, normalize_parallel_config
from .plugin import _mark_debug_output

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict
    from ...config import ModelConfig
    from ...parallel_config import ParallelConfig


def _slice_last_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=-1)[rank])


def _slice_first_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=0)[rank])


def _validate_marian_tp(
    config: "ModelConfig",
    weights: "WeightDict",
    parallel: "ParallelConfig",
) -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError("Marian tensor-parallel decoder build requires a concrete rank")

    tp = parallel.tp_size
    hidden = int(config.hidden_size)
    dec_heads = int(weights["_dec_heads"])
    dec_ffn = int(weights["_dec_ffn"])
    if hidden % tp != 0:
        raise ValueError(
            "Marian tensor parallel requires hidden size divisible by tp_size "
            f"({hidden} vs {tp})")
    if dec_heads % tp != 0:
        raise ValueError(
            "Marian tensor parallel requires decoder_attention_heads divisible by tp_size "
            f"({dec_heads} vs {tp})")
    if dec_ffn % tp != 0:
        raise ValueError(
            "Marian tensor parallel requires decoder_ffn_dim divisible by tp_size "
            f"({dec_ffn} vs {tp})")

    column_keys = (
        ".w_q", ".w_k", ".w_v",
        ".cross_w_q", ".cross_w_k", ".cross_w_v",
        ".w_fc1",
    )
    column_biases = (
        ".q_bias", ".k_bias", ".v_bias",
        ".cross_b_q", ".cross_b_k", ".cross_b_v",
        ".fc1_bias",
    )
    row_keys = (".w_o", ".cross_w_o", ".w_fc2")
    for layer_idx in range(int(weights["_dec_layers"])):
        prefix = f"layer.{layer_idx}"
        for suffix in column_keys:
            key = f"{prefix}{suffix}"
            if weights[key].shape[-1] % tp != 0:
                raise ValueError(f"{key} output dim is not divisible by tp_size={tp}")
        for suffix in column_biases:
            key = f"{prefix}{suffix}"
            if weights[key].shape[-1] % tp != 0:
                raise ValueError(f"{key} output dim is not divisible by tp_size={tp}")
        for suffix in row_keys:
            key = f"{prefix}{suffix}"
            if weights[key].shape[0] % tp != 0:
                raise ValueError(f"{key} input dim is not divisible by tp_size={tp}")


def shard_marian_decoder_weights(
    weights: "WeightDict",
    *,
    parallel: "ParallelConfig",
) -> "WeightDict":
    """Return rank-local decoder weights for the Marian TP builder."""
    if not parallel.enabled:
        return weights

    rank = parallel.rank
    tp = parallel.tp_size
    out = type(weights)()
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
            continue
        if key.endswith((
            ".w_q", ".w_k", ".w_v",
            ".cross_w_q", ".cross_w_k", ".cross_w_v",
            ".w_fc1",
            ".q_bias", ".k_bias", ".v_bias",
            ".cross_b_q", ".cross_b_k", ".cross_b_v",
            ".fc1_bias",
        )):
            out[key] = _slice_last_dim(value, rank, tp)
        elif key.endswith((".w_o", ".cross_w_o", ".w_fc2")):
            out[key] = _slice_first_dim(value, rank, tp)
        else:
            out[key] = value

    out["_tensor_parallel_size"] = tp
    out["_tensor_parallel_rank"] = rank
    return out


def _add_row_parallel_bias(network, tensor, hidden_size: int, bias: np.ndarray, tp_size: int):
    joined = add_all_reduce_sum(network, tensor, tp_size)
    return graph_ops.add_bias_sum(network, joined, hidden_size, bias)


def _add_marian_tp_decoder_layer(
    *,
    network,
    hidden,
    cache_k,
    cache_v,
    cross_k,
    cross_v,
    attention_mask,
    encoder_mask,
    eps,
    weights,
    prefix: str,
    hidden_size: int,
    local_attention_size: int,
    local_heads: int,
    head_dim: int,
    local_ffn_dim: int,
    max_cache_length: int,
    max_enc_seq_len: int,
    tp_size: int,
):
    attention_window = max_cache_length + 1

    q = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, hidden, hidden_size, local_attention_size, weights[f"{prefix}.w_q"]),
        local_attention_size,
        weights[f"{prefix}.q_bias"],
    )
    k = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, hidden, hidden_size, local_attention_size, weights[f"{prefix}.w_k"]),
        local_attention_size,
        weights[f"{prefix}.k_bias"],
    )
    v = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, hidden, hidden_size, local_attention_size, weights[f"{prefix}.w_v"]),
        local_attention_size,
        weights[f"{prefix}.v_bias"],
    )
    present_k, present_v = k, v

    kr = network.add_shuffle(k)
    kr.reshape_dims = (1, local_attention_size)
    vr = network.add_shuffle(v)
    vr.reshape_dims = (1, local_attention_size)
    ak = network.add_concatenation([cache_k, kr.get_output(0)])
    ak.axis = 0
    av = network.add_concatenation([cache_v, vr.get_output(0)])
    av.axis = 0

    m4 = network.add_shuffle(attention_mask)
    m4.reshape_dims = (1, 1, 1, attention_window)
    cf = graph_ops.add_attention_from_rows(
        network, q, ak.get_output(0), av.get_output(0),
        num_heads=local_heads, head_dim=head_dim,
        q_seq=1, kv_seq=attention_window,
        mask=m4.get_output(0))
    sa = graph_ops.add_matmul_rhs_constant(
        network, cf, local_attention_size, hidden_size, weights[f"{prefix}.w_o"])
    sa = _add_row_parallel_bias(
        network, sa, hidden_size, weights[f"{prefix}.o_bias"], tp_size)

    psa = network.add_elementwise(hidden, sa, trt.ElementWiseOperation.SUM).get_output(0)
    psa = graph_ops.add_layer_norm_native(
        network, psa, hidden_size,
        weights[f"{prefix}.input_norm"], weights[f"{prefix}.input_norm_beta"],
        eps)

    cq = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, psa, hidden_size, local_attention_size,
            weights[f"{prefix}.cross_w_q"]),
        local_attention_size,
        weights[f"{prefix}.cross_b_q"],
    )
    ck_proj = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, cross_k, hidden_size, local_attention_size,
            weights[f"{prefix}.cross_w_k"]),
        local_attention_size,
        weights[f"{prefix}.cross_b_k"],
    )
    cv_proj = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, cross_v, hidden_size, local_attention_size,
            weights[f"{prefix}.cross_w_v"]),
        local_attention_size,
        weights[f"{prefix}.cross_b_v"],
    )

    enc_mask_4d = network.add_shuffle(encoder_mask)
    enc_mask_4d.reshape_dims = (1, 1, 1, max_enc_seq_len)
    ccf = graph_ops.add_attention_from_rows(
        network, cq, ck_proj, cv_proj,
        num_heads=local_heads, head_dim=head_dim,
        q_seq=1, kv_seq=max_enc_seq_len,
        mask=enc_mask_4d.get_output(0))
    ca = graph_ops.add_matmul_rhs_constant(
        network, ccf, local_attention_size, hidden_size, weights[f"{prefix}.cross_w_o"])
    ca = _add_row_parallel_bias(
        network, ca, hidden_size, weights[f"{prefix}.cross_b_o"], tp_size)

    pca = network.add_elementwise(psa, ca, trt.ElementWiseOperation.SUM).get_output(0)
    pca = graph_ops.add_layer_norm_native(
        network, pca, hidden_size,
        weights[f"{prefix}.cross_attn_norm"],
        weights[f"{prefix}.cross_attn_norm_beta"], eps)

    fc1 = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, pca, hidden_size, local_ffn_dim, weights[f"{prefix}.w_fc1"]),
        local_ffn_dim,
        weights[f"{prefix}.fc1_bias"],
    )
    act = graph_ops.add_activation(network, fc1, "silu")
    fc2 = graph_ops.add_matmul_rhs_constant(
        network, act, local_ffn_dim, hidden_size, weights[f"{prefix}.w_fc2"])
    fc2 = _add_row_parallel_bias(
        network, fc2, hidden_size, weights[f"{prefix}.fc2_bias"], tp_size)

    out = network.add_elementwise(pca, fc2, trt.ElementWiseOperation.SUM).get_output(0)
    out = graph_ops.add_layer_norm_native(
        network, out, hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        weights[f"{prefix}.post_attn_norm_beta"], eps)

    return {"hidden": out, "present_k": present_k, "present_v": present_v}


def build_marian_tp_decoder_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    verbose: bool = False,
    debug_layer_outputs: bool = False,
    parallel_config=None,
) -> bytes:
    """Build one rank-local Marian decoder with tensor-parallel joins."""
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError(
            "build_marian_tp_decoder_engine requires tensor_parallel mode with tp_size > 1")
    _validate_marian_tp(config, weights, parallel)

    rank_weights = shard_marian_decoder_weights(weights, parallel=parallel)
    dec_layers = int(weights["_dec_layers"])
    dec_heads = int(weights["_dec_heads"])
    dec_ffn = int(weights["_dec_ffn"])
    max_pos = int(weights["_max_position_embeddings"])
    hidden = int(config.hidden_size)
    vocab = int(config.vocab_size)
    head_dim = hidden // dec_heads
    local_heads = dec_heads // parallel.tp_size
    local_attention_size = hidden // parallel.tp_size
    local_ffn_dim = dec_ffn // parallel.tp_size
    attention_window = max_cache_length + 1
    max_enc_seq_len = max_pos

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    trt_config.clear_flag(trt.BuilderFlag.TF32)

    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input("attention_mask", trt.float32, (attention_window,))

    cache_k_inputs, cache_v_inputs = [], []
    for layer_idx in range(dec_layers):
        cache_k_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cache_k", layer_idx),
            trt.float32, (max_cache_length, local_attention_size)))
        cache_v_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cache_v", layer_idx),
            trt.float32, (max_cache_length, local_attention_size)))

    cross_k_inputs, cross_v_inputs = [], []
    for layer_idx in range(dec_layers):
        cross_k_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cross_k", layer_idx),
            trt.float32, (max_enc_seq_len, hidden)))
        cross_v_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cross_v", layer_idx),
            trt.float32, (max_enc_seq_len, hidden)))

    encoder_mask = network.add_input("encoder_mask", trt.float32, (max_enc_seq_len,))

    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), rank_weights["dec_embedding"])
    pos_embed_np = rank_weights["dec_pos_embedding"]
    pos_embedding_table = graph_ops.add_constant(network, pos_embed_np.shape, pos_embed_np)

    token_embed = network.add_gather(embedding_table, token_id, 0).get_output(0)
    scale_val = np.sqrt(float(hidden))
    scale_const = graph_ops.add_constant(
        network, (1, 1), np.array([scale_val], dtype=np.float32))
    token_embed = network.add_elementwise(
        token_embed, scale_const, trt.ElementWiseOperation.PROD).get_output(0)
    pos_embed = network.add_gather(pos_embedding_table, position_id, 0).get_output(0)
    hidden_state = network.add_elementwise(
        token_embed, pos_embed, trt.ElementWiseOperation.SUM).get_output(0)

    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")

    present_k_outputs, present_v_outputs = [], []
    for layer_idx in range(dec_layers):
        prefix = f"layer.{layer_idx}"
        result = _add_marian_tp_decoder_layer(
            network=network,
            hidden=hidden_state,
            cache_k=cache_k_inputs[layer_idx],
            cache_v=cache_v_inputs[layer_idx],
            cross_k=cross_k_inputs[layer_idx],
            cross_v=cross_v_inputs[layer_idx],
            attention_mask=attention_mask,
            encoder_mask=encoder_mask,
            eps=config.rms_norm_eps,
            weights=rank_weights,
            prefix=prefix,
            hidden_size=hidden,
            local_attention_size=local_attention_size,
            local_heads=local_heads,
            head_dim=head_dim,
            local_ffn_dim=local_ffn_dim,
            max_cache_length=max_cache_length,
            max_enc_seq_len=max_enc_seq_len,
            tp_size=parallel.tp_size,
        )
        hidden_state = result["hidden"]
        present_k_outputs.append(result["present_k"])
        present_v_outputs.append(result["present_v"])
        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, vocab, rank_weights["w_out"])
    logits = graph_ops.add_bias_sum(
        network, logits, vocab, rank_weights["final_logits_bias"])
    logits.name = "logits"
    network.mark_output(logits)

    for layer_idx in range(dec_layers):
        present_k_outputs[layer_idx].name = graph_ops.layer_tensor_name(
            "present_k", layer_idx)
        present_v_outputs[layer_idx].name = graph_ops.layer_tensor_name(
            "present_v", layer_idx)
        network.mark_output(present_k_outputs[layer_idx])
        network.mark_output(present_v_outputs[layer_idx])

    if verbose:
        print(
            "[trtmc build] Marian TP decoder "
            f"(rank={parallel.rank}/{parallel.tp_size}, {dec_layers}L, "
            f"h={hidden}, local_heads={local_heads}, cache={max_cache_length})",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT Marian TP decoder build failed")
    return bytes(plan)
