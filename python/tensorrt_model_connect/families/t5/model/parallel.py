"""Tensor-parallel T5 decoder builder.

This mirrors the single-device decoder graph in ``plugin.py`` while applying
tensor parallelism only to the decoder projections:

* self-attention and cross-attention Q/K/V projections are column-sharded,
* attention output and FFN output projections are row-sharded,
* TensorRT distributed ALL_REDUCE restores full hidden states after row joins,
* embeddings, norms, encoder outputs, and the LM head stay replicated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import sys

import numpy as np
from tensorrt_model_connect import trt_compat

from . import model as graph_ops
from ....parallel_config import add_all_reduce_sum, normalize_parallel_config
from ..plugin import _make_t5_causal_buckets, _make_t5_cross_buckets, _mark_debug_output

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ..weights import WeightDict
    from ..config import ModelConfig
    from ....parallel_config import ParallelConfig


def _slice_last_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=-1)[rank])


def _slice_first_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=0)[rank])


def _validate_t5_tp(weights: "WeightDict", parallel: "ParallelConfig") -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError("T5 tensor-parallel decoder build requires a concrete rank")

    tp = parallel.tp_size
    num_heads = int(weights["_num_heads"])
    attention_size = num_heads * int(weights["_d_kv"])
    ffn_dim = int(weights["_d_ff"])
    if num_heads % tp != 0:
        raise ValueError(
            f"T5 tensor parallel requires num_heads divisible by tp_size ({num_heads} vs {tp})"
        )
    if attention_size % tp != 0:
        raise ValueError(
            "T5 tensor parallel requires attention size divisible by tp_size "
            f"({attention_size} vs {tp})"
        )
    if ffn_dim % tp != 0:
        raise ValueError(
            f"T5 tensor parallel requires d_ff divisible by tp_size ({ffn_dim} vs {tp})"
        )

    for layer_idx in range(int(weights["_dec_layers"])):
        prefix = f"layer.{layer_idx}"
        for key in (
            f"{prefix}.w_q",
            f"{prefix}.w_k",
            f"{prefix}.w_v",
            f"{prefix}.cross_w_q",
            f"{prefix}.cross_w_k",
            f"{prefix}.cross_w_v",
            f"{prefix}.w_fc1",
        ):
            if weights[key].shape[-1] % tp != 0:
                raise ValueError(f"{key} output dim is not divisible by tp_size={tp}")
        for key in (f"{prefix}.w_o", f"{prefix}.cross_w_o", f"{prefix}.w_fc2"):
            if weights[key].shape[0] % tp != 0:
                raise ValueError(f"{key} input dim is not divisible by tp_size={tp}")


def shard_t5_decoder_weights(
    weights: "WeightDict",
    *,
    parallel: "ParallelConfig",
) -> "WeightDict":
    """Return rank-local decoder weights for the T5 TP builder."""
    if not parallel.enabled:
        return weights

    rank = parallel.rank
    tp = parallel.tp_size
    out = type(weights)()
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
            continue
        if key.endswith(
            (
                ".w_q",
                ".w_k",
                ".w_v",
                ".cross_w_q",
                ".cross_w_k",
                ".cross_w_v",
                ".w_fc1",
            )
        ):
            out[key] = _slice_last_dim(value, rank, tp)
        elif key.endswith((".w_o", ".cross_w_o", ".w_fc2")):
            out[key] = _slice_first_dim(value, rank, tp)
        elif key in {"dec_self_rel_attn_bias", "dec_cross_rel_attn_bias"}:
            out[key] = _slice_last_dim(value, rank, tp)
        else:
            out[key] = value

    out["_tensor_parallel_size"] = tp
    out["_tensor_parallel_rank"] = rank
    return out


def _add_t5_tp_decoder_layer(
    *,
    network,
    hidden,
    cache_k,
    cache_v,
    cross_k,
    cross_v,
    attention_mask,
    position_id,
    eps_tensor,
    weights,
    prefix: str,
    hidden_size: int,
    local_attention_size: int,
    local_heads: int,
    head_dim: int,
    local_ffn_dim: int,
    max_cache_length: int,
    max_source_positions: int,
    dec_self_rel_bias,
    dec_cross_rel_bias,
    num_buckets: int,
    max_distance: int,
    enc_mask=None,
    tp_size: int = 1,
):
    attention_window = max_cache_length + 1

    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size, weights[f"{prefix}.input_norm"], eps_tensor
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
    present_k, present_v = k, v

    kr = network.add_shuffle(k)
    kr.reshape_dims = (1, local_attention_size)
    vr = network.add_shuffle(v)
    vr.reshape_dims = (1, local_attention_size)
    ak = network.add_concatenation([cache_k, kr.get_output(0)])
    ak.axis = 0
    av = network.add_concatenation([cache_v, vr.get_output(0)])
    av.axis = 0

    if dec_self_rel_bias is not None:
        bucket_indices = _make_t5_causal_buckets(attention_window, num_buckets, max_distance)
        bias = (
            dec_self_rel_bias[bucket_indices.flatten()]
            .reshape(attention_window, attention_window, local_heads)
            .transpose(2, 0, 1)
        )
        bias_const = graph_ops.add_constant(network, bias.shape, bias.astype(np.float32))
        bias_row = network.add_gather(bias_const, position_id, 1)
        mask_3d = network.add_shuffle(attention_mask)
        mask_3d.reshape_dims = (1, 1, attention_window)
        self_mask_3d = network.add_elementwise(
            bias_row.get_output(0), mask_3d.get_output(0), trt.ElementWiseOperation.SUM
        )
        self_mask_4d = network.add_shuffle(self_mask_3d.get_output(0))
        self_mask_4d.reshape_dims = (1, local_heads, 1, attention_window)
        self_mask = self_mask_4d.get_output(0)
    else:
        self_mask = graph_ops.add_2d_mask_to_4d(network, attention_mask)

    context = graph_ops.add_attention_from_rows(
        network,
        q,
        ak.get_output(0),
        av.get_output(0),
        num_heads=local_heads,
        head_dim=head_dim,
        q_seq=1,
        kv_seq=attention_window,
        mask=self_mask,
        scale=1.0,
    )
    sa = graph_ops.add_matmul_rhs_constant(
        network, context, local_attention_size, hidden_size, weights[f"{prefix}.w_o"]
    )
    sa = add_all_reduce_sum(network, sa, tp_size)
    psa = network.add_elementwise(hidden, sa, trt.ElementWiseOperation.SUM).get_output(0)

    cross_normed = graph_ops.add_rms_norm(
        network, psa, hidden_size, weights[f"{prefix}.cross_attn_norm"], eps_tensor
    )
    cross_q = graph_ops.add_matmul_rhs_constant(
        network, cross_normed, hidden_size, local_attention_size, weights[f"{prefix}.cross_w_q"]
    )
    cross_k_proj = graph_ops.add_matmul_rhs_constant(
        network, cross_k, hidden_size, local_attention_size, weights[f"{prefix}.cross_w_k"]
    )
    cross_v_proj = graph_ops.add_matmul_rhs_constant(
        network, cross_v, hidden_size, local_attention_size, weights[f"{prefix}.cross_w_v"]
    )

    if dec_cross_rel_bias is not None:
        cross_buckets = _make_t5_cross_buckets(
            attention_window, max_source_positions, num_buckets, max_distance
        )
        cross_bias = (
            dec_cross_rel_bias[cross_buckets.flatten()]
            .reshape(attention_window, max_source_positions, local_heads)
            .transpose(2, 0, 1)
        )
        cross_bias_const = graph_ops.add_constant(
            network, cross_bias.shape, cross_bias.astype(np.float32)
        )
        cross_bias_row = network.add_gather(cross_bias_const, position_id, 1)
        cross_mask = cross_bias_row.get_output(0)
        if enc_mask is not None:
            cross_mask = network.add_elementwise(
                cross_mask, enc_mask, trt.ElementWiseOperation.SUM
            ).get_output(0)
        cross_mask_4d = network.add_shuffle(cross_mask)
        cross_mask_4d.reshape_dims = (1, local_heads, 1, max_source_positions)
        cross_mask = cross_mask_4d.get_output(0)
    elif enc_mask is not None:
        cross_mask_4d = network.add_shuffle(enc_mask)
        cross_mask_4d.reshape_dims = (1, 1, 1, max_source_positions)
        cross_mask = cross_mask_4d.get_output(0)
    else:
        cross_mask = None

    cross_context = graph_ops.add_attention_from_rows(
        network,
        cross_q,
        cross_k_proj,
        cross_v_proj,
        num_heads=local_heads,
        head_dim=head_dim,
        q_seq=1,
        kv_seq=max_source_positions,
        mask=cross_mask,
        scale=1.0,
    )
    cross_out = graph_ops.add_matmul_rhs_constant(
        network, cross_context, local_attention_size, hidden_size, weights[f"{prefix}.cross_w_o"]
    )
    cross_out = add_all_reduce_sum(network, cross_out, tp_size)
    pca = network.add_elementwise(psa, cross_out, trt.ElementWiseOperation.SUM).get_output(0)

    ffn_normed = graph_ops.add_rms_norm(
        network, pca, hidden_size, weights[f"{prefix}.post_attn_norm"], eps_tensor
    )
    fc1 = graph_ops.add_matmul_rhs_constant(
        network, ffn_normed, hidden_size, local_ffn_dim, weights[f"{prefix}.w_fc1"]
    )
    relu = network.add_activation(fc1, trt.ActivationType.RELU)
    fc2 = graph_ops.add_matmul_rhs_constant(
        network, relu.get_output(0), local_ffn_dim, hidden_size, weights[f"{prefix}.w_fc2"]
    )
    fc2 = add_all_reduce_sum(network, fc2, tp_size)
    out = network.add_elementwise(pca, fc2, trt.ElementWiseOperation.SUM).get_output(0)
    return {"hidden": out, "present_k": present_k, "present_v": present_v}


def build_t5_tp_decoder_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    verbose: bool = False,
    debug_layer_outputs: bool = False,
    parallel_config=None,
) -> bytes:
    """Build one rank-local T5 decoder with tensor-parallel projection joins."""
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError(
            "build_t5_tp_decoder_engine requires tensor_parallel mode with tp_size > 1"
        )
    _validate_t5_tp(weights, parallel)

    rank_weights = shard_t5_decoder_weights(weights, parallel=parallel)
    decoder_layers = int(weights["_dec_layers"])
    num_heads = int(weights["_num_heads"])
    head_dim = int(weights["_d_kv"])
    ffn_dim = int(weights["_d_ff"])
    hidden_size = int(weights["_hidden"])
    vocab_size = int(weights["_vocab_size"])
    num_buckets = int(weights["_num_buckets"])
    max_distance = int(weights["_max_distance"])
    eps = float(weights["_layer_norm_eps"])
    attention_size = num_heads * head_dim
    local_heads = num_heads // parallel.tp_size
    local_attention_size = attention_size // parallel.tp_size
    local_ffn_dim = ffn_dim // parallel.tp_size
    attention_window = max_cache_length + 1
    max_source_positions = max_cache_length

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    trt_config.clear_flag(trt.BuilderFlag.TF32)

    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input("attention_mask", trt.float32, (1, attention_window))
    cache_k_inputs, cache_v_inputs = [], []
    for layer_idx in range(decoder_layers):
        cache_k_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("cache_k", layer_idx),
                trt.float32,
                (max_cache_length, local_attention_size),
            )
        )
        cache_v_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("cache_v", layer_idx),
                trt.float32,
                (max_cache_length, local_attention_size),
            )
        )

    cross_k_inputs, cross_v_inputs = [], []
    for layer_idx in range(decoder_layers):
        cross_k_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("cross_k", layer_idx),
                trt.float32,
                (max_source_positions, hidden_size),
            )
        )
        cross_v_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("cross_v", layer_idx),
                trt.float32,
                (max_source_positions, hidden_size),
            )
        )

    encoder_mask = network.add_input("encoder_mask", trt.float32, (max_source_positions,))
    encoder_mask_3d = network.add_shuffle(encoder_mask)
    encoder_mask_3d.reshape_dims = (1, 1, max_source_positions)
    encoder_mask = encoder_mask_3d.get_output(0)

    eps_tensor = graph_ops.add_constant(network, (1, 1), np.array([eps], dtype=np.float32))
    embedding = graph_ops.add_constant(
        network, (vocab_size, hidden_size), rank_weights["shared_embedding"]
    )
    hidden_state = network.add_gather(embedding, token_id, 0).get_output(0)
    self_rel_bias = rank_weights.get("dec_self_rel_attn_bias")
    cross_rel_bias = rank_weights.get("dec_cross_rel_attn_bias")
    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")

    present_k_outputs, present_v_outputs = [], []
    for layer_idx in range(decoder_layers):
        prefix = f"layer.{layer_idx}"
        result = _add_t5_tp_decoder_layer(
            network=network,
            hidden=hidden_state,
            cache_k=cache_k_inputs[layer_idx],
            cache_v=cache_v_inputs[layer_idx],
            cross_k=cross_k_inputs[layer_idx],
            cross_v=cross_v_inputs[layer_idx],
            attention_mask=attention_mask,
            position_id=position_id,
            eps_tensor=eps_tensor,
            weights=rank_weights,
            prefix=prefix,
            hidden_size=hidden_size,
            local_attention_size=local_attention_size,
            local_heads=local_heads,
            head_dim=head_dim,
            local_ffn_dim=local_ffn_dim,
            max_cache_length=max_cache_length,
            max_source_positions=max_source_positions,
            dec_self_rel_bias=self_rel_bias,
            dec_cross_rel_bias=cross_rel_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            enc_mask=encoder_mask,
            tp_size=parallel.tp_size,
        )
        hidden_state = result["hidden"]
        present_k_outputs.append(result["present_k"])
        present_v_outputs.append(result["present_v"])
        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

    hidden_state = graph_ops.add_rms_norm(
        network, hidden_state, hidden_size, rank_weights["final_norm"], eps_tensor
    )
    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden_size, vocab_size, rank_weights["w_out"]
    )
    logits.name = "logits"
    network.mark_output(logits)

    for layer_idx in range(decoder_layers):
        present_k_outputs[layer_idx].name = graph_ops.layer_tensor_name("present_k", layer_idx)
        present_v_outputs[layer_idx].name = graph_ops.layer_tensor_name("present_v", layer_idx)
        network.mark_output(present_k_outputs[layer_idx])
        network.mark_output(present_v_outputs[layer_idx])

    if verbose:
        print(
            "[trtmc build] T5 TP decoder "
            f"(rank={parallel.rank}/{parallel.tp_size}, {decoder_layers}L, "
            f"h={hidden_size}, local_heads={local_heads}, cache={max_cache_length})",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT T5 TP decoder build failed")
    return bytes(plan)
