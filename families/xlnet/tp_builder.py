# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XLNet encoder engine builder with relative positional attention.

Builds a TensorRT engine for XLNet's content-stream inference:
  - Sinusoidal relative positional encoding (computed at build time)
  - Relative attention: (ac + bd + ef) * scale
  - Segment-relative encoding with learned seg_embed
  - POST-norm architecture (same as BERT)
  - No KV cache (full bidirectional attention)

Tensor names for the C++ runtime:
  Inputs:  input_ids [seq_len], token_type_ids [seq_len]
  Outputs: hidden_states [seq_len, hidden_size]

Trace IDs: ARCH-XLNET, UD-XLNET-002
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from . import graph_ops
from .config import ModelConfig
from .parallel import add_all_reduce_sum, normalize_parallel_config


if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from .parallel import ParallelConfig


def _compute_sinusoidal_pos_emb(seq_len: int, d_model: int) -> np.ndarray:
    """Compute sinusoidal relative positional embeddings.

    For bidirectional (attn_type="bi"), positions go from klen to -qlen.
    With no mems, klen=qlen=seq_len, so positions: [seq_len, seq_len-1, ..., 1-seq_len].
    Total positions: 2*seq_len - 1 (but HF uses 2*seq_len due to range(klen, -qlen, -1)).

    Returns: [2*seq_len, d_model] positional embedding matrix.
    """
    # For bi-directional: pos_seq = arange(klen, -qlen, -1) = arange(seq_len, -seq_len, -1)
    # That gives 2*seq_len positions
    2 * seq_len
    pos_seq = np.arange(seq_len, -seq_len, -1, dtype=np.float32)

    freq_seq = np.arange(0, d_model, 2.0, dtype=np.float32)
    inv_freq = 1.0 / np.power(10000.0, freq_seq / d_model)

    # sinusoid_inp: [n_pos, d_model//2]
    sinusoid_inp = np.outer(pos_seq, inv_freq)

    # pos_emb: [n_pos, d_model] = [sin, cos] concatenated
    pos_emb = np.concatenate([np.sin(sinusoid_inp), np.cos(sinusoid_inp)], axis=-1)

    return pos_emb.astype(np.float32)


def _add_seq_layer_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float,
) -> trt.ITensor:
    """LayerNorm over [seq_len, hidden] using TRT native normalization."""
    return graph_ops.add_layer_norm_native(
        network, inp, hidden_size, gamma, beta, eps)


def _slice_last_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=-1)[rank])


def _slice_first_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=0)[rank])


def _validate_xlnet_tp(
    config: ModelConfig,
    weights: "WeightDict",
    parallel: "ParallelConfig",
) -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError("XLNet tensor-parallel build requires a concrete rank")

    tp = parallel.tp_size
    if config.num_attention_heads % tp != 0:
        raise ValueError(
            "XLNet tensor parallel requires num_attention_heads divisible by "
            f"tp_size ({config.num_attention_heads} vs {tp})")
    if config.intermediate_size % tp != 0:
        raise ValueError(
            "XLNet tensor parallel requires intermediate_size divisible by "
            f"tp_size ({config.intermediate_size} vs {tp})")

    for layer_idx in range(config.num_hidden_layers):
        prefix = f"layer.{layer_idx}"
        for key in (
            f"{prefix}.w_q", f"{prefix}.w_k", f"{prefix}.w_v",
            f"{prefix}.w_o", f"{prefix}.w_r", f"{prefix}.w_fc1",
        ):
            if weights[key].shape[-1] % tp != 0:
                raise ValueError(f"{key} output dim must be divisible by tp_size")
        if weights[f"{prefix}.w_fc2"].shape[0] % tp != 0:
            raise ValueError(f"{prefix}.w_fc2 input dim must be divisible by tp_size")
        for key in (f"{prefix}.r_w_bias", f"{prefix}.r_r_bias", f"{prefix}.r_s_bias"):
            if weights[key].shape[0] % tp != 0:
                raise ValueError(f"{key} head dim must be divisible by tp_size")
        if weights[f"{prefix}.seg_embed"].shape[1] % tp != 0:
            raise ValueError(f"{prefix}.seg_embed head dim must be divisible by tp_size")


def shard_xlnet_weights(
    config: ModelConfig,
    weights: "WeightDict",
    *,
    parallel: "ParallelConfig",
) -> "WeightDict":
    """Return rank-local XLNet weights for the TP builder."""
    _validate_xlnet_tp(config, weights, parallel)
    if not parallel.enabled:
        return weights

    out = type(weights)()
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
            continue

        if key.endswith((".w_q", ".w_k", ".w_v", ".w_o", ".w_r", ".w_fc1")):
            out[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith(".w_fc2"):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith((".r_w_bias", ".r_r_bias", ".r_s_bias", ".fc1_bias")):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith(".seg_embed"):
            out[key] = np.ascontiguousarray(
                np.array_split(value, parallel.tp_size, axis=1)[parallel.rank])
        else:
            out[key] = value

    out["_attention_size"] = config.attention_size // parallel.tp_size
    out["_intermediate_size"] = config.intermediate_size // parallel.tp_size
    out["_tensor_parallel_size"] = parallel.tp_size
    out["_tensor_parallel_rank"] = parallel.rank
    return out


def _add_rel_shift(network, bd, num_heads, qlen, klen):
    """Implement rel_shift_bnij for TRT (no batch dim).

    Input bd: [num_heads, qlen, 2*qlen] (from position attention)
    Output: [num_heads, qlen, klen]

    Algorithm: reshape to [N, 2*qlen, qlen], remove first row,
    reshape to [N, qlen, 2*qlen-1], slice to [:, :, :klen].
    """
    shuf1 = network.add_shuffle(bd)
    shuf1.reshape_dims = (num_heads, 2 * qlen, qlen)

    slice_l = network.add_slice(
        shuf1.get_output(0),
        start=(0, 1, 0),
        shape=(num_heads, 2 * qlen - 1, qlen),
        stride=(1, 1, 1))

    shuf2 = network.add_shuffle(slice_l.get_output(0))
    shuf2.reshape_dims = (num_heads, qlen, 2 * qlen - 1)

    slice2 = network.add_slice(
        shuf2.get_output(0),
        start=(0, 0, 0),
        shape=(num_heads, qlen, klen),
        stride=(1, 1, 1))

    return slice2.get_output(0)


def build_tp_xlnet_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_seq_length: int,
    *,
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    """Build a rank-local TRT engine plan for XLNet encoder.

    Args:
        config: Model architecture from config.json.
        weights: Loaded weight dict from XLNet plugin.
        max_seq_length: Maximum sequence length the engine is compiled for.
        verbose: Print TRT builder logs.

    Returns:
        Serialized engine plan bytes.
    """
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError(
            "build_tp_xlnet_engine requires tensor_parallel mode and tp_size > 1")
    weights = shard_xlnet_weights(config, weights, parallel=parallel)

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    full_num_heads = config.num_attention_heads
    num_heads = config.num_attention_heads // parallel.tp_size
    d_head = config.raw.get("d_head", hidden // full_num_heads)
    attn_size = num_heads * d_head
    intermediate = config.intermediate_size // parallel.tp_size
    eps = config.rms_norm_eps
    ff_activation = config.raw.get("ff_activation", "gelu")

    qlen = max_seq_length
    scale = 1.0 / (d_head ** 0.5)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.clear_flag(trt.BuilderFlag.TF32)

    # -------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------
    input_ids = network.add_input("input_ids", trt.int32, (max_seq_length,))
    attention_mask_input = network.add_input("attention_mask", trt.int32, (max_seq_length,))

    # token_type_ids: constant zeros (all segment-0) — the C++ encoder
    # pipeline doesn't provide this input, and inference is single-segment.
    tt_zeros_layer = network.add_constant(
        (max_seq_length,), trt.Weights(np.zeros(max_seq_length, dtype=np.int32)))
    token_type_ids = tt_zeros_layer.get_output(0)

    # -------------------------------------------------------------------
    # Constants
    # -------------------------------------------------------------------
    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["embedding"])
    scale_t = graph_ops.add_constant(
        network, (1, 1, 1), np.array([scale], dtype=np.float32))

    # Precompute sinusoidal relative positional embeddings: [2*qlen, hidden]
    pos_emb_np = _compute_sinusoidal_pos_emb(qlen, hidden)
    pos_emb_const = graph_ops.add_constant(
        network, pos_emb_np.shape, pos_emb_np)

    # Build additive attention mask from attention_mask input
    mask_float = network.add_cast(attention_mask_input, trt.float32)
    ones_mask = graph_ops.add_constant(network, (1,), np.array([1.0], dtype=np.float32))
    neg_large = graph_ops.add_constant(network, (1,), np.array([-1e30], dtype=np.float32))
    inv_mask = network.add_elementwise(
        ones_mask, mask_float.get_output(0), trt.ElementWiseOperation.SUB)
    pad_penalty = network.add_elementwise(
        inv_mask.get_output(0), neg_large, trt.ElementWiseOperation.PROD)
    # Reshape to [1, 1, seq_len] for broadcasting
    pad_mask_reshape = network.add_shuffle(pad_penalty.get_output(0))
    pad_mask_reshape.reshape_dims = (1, 1, max_seq_length)

    # -------------------------------------------------------------------
    # Segment matrix computation from token_type_ids
    # seg_mat[i,j] = one_hot(token_type_ids[i] \!= token_type_ids[j], 2)
    # Shape: [qlen, klen, 2]  (for segment attention ef term)
    #
    # We precompute a default same-segment matrix and compute the actual
    # one dynamically from token_type_ids.
    # -------------------------------------------------------------------
    # Cast token_type_ids to float: [seq_len]
    tt_float = network.add_cast(token_type_ids, trt.float32)

    # Reshape for broadcasting: [seq_len, 1] and [1, seq_len]
    tt_col = network.add_shuffle(tt_float.get_output(0))
    tt_col.reshape_dims = (max_seq_length, 1)
    tt_row = network.add_shuffle(tt_float.get_output(0))
    tt_row.reshape_dims = (1, max_seq_length)

    # diff = |tt_col - tt_row| (0 if same segment, >0 if different)
    tt_diff = network.add_elementwise(
        tt_col.get_output(0), tt_row.get_output(0),
        trt.ElementWiseOperation.SUB)
    tt_abs = network.add_unary(tt_diff.get_output(0), trt.UnaryOperation.ABS)

    # seg_diff: [seq_len, seq_len] with 0=same, 1=different
    # Clamp to [0,1] (in case of more than 2 segments)
    one_t = graph_ops.add_constant(network, (1, 1), np.array([1.0], dtype=np.float32))
    seg_diff_raw = network.add_elementwise(
        tt_abs.get_output(0), one_t, trt.ElementWiseOperation.MIN)

    # One-hot: seg_mat_same = 1 - seg_diff, seg_mat_diff = seg_diff
    # Stack to [seq_len, seq_len, 2]: dim0=same, dim1=different
    seg_same = network.add_elementwise(
        one_t, seg_diff_raw.get_output(0), trt.ElementWiseOperation.SUB)

    # Reshape both to [seq_len, seq_len, 1] and concat to [seq_len, seq_len, 2]
    seg_same_3d = network.add_shuffle(seg_same.get_output(0))
    seg_same_3d.reshape_dims = (max_seq_length, max_seq_length, 1)
    seg_diff_3d = network.add_shuffle(seg_diff_raw.get_output(0))
    seg_diff_3d.reshape_dims = (max_seq_length, max_seq_length, 1)

    seg_mat = network.add_concatenation(
        [seg_same_3d.get_output(0), seg_diff_3d.get_output(0)])
    seg_mat.axis = 2  # [seq_len, seq_len, 2]

    # -------------------------------------------------------------------
    # Word embedding
    # -------------------------------------------------------------------
    word_embed = network.add_gather(embedding_table, input_ids, 0)
    # hidden_state: [seq_len, hidden]
    hidden_state = word_embed.get_output(0)

    # -------------------------------------------------------------------
    # Encoder layers
    # -------------------------------------------------------------------
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"

        hidden_state = _add_xlnet_layer(
            network=network,
            hidden=hidden_state,
            weights=weights,
            prefix=prefix,
            pos_emb=pos_emb_const,
            seg_mat=seg_mat.get_output(0),
            attn_mask=pad_mask_reshape.get_output(0),
            hidden_size=hidden,
            attn_size=attn_size,
            intermediate_size=intermediate,
            num_heads=num_heads,
            d_head=d_head,
            seq_length=max_seq_length,
            scale_tensor=scale_t,
            ff_activation=ff_activation,
            eps=eps,
            tp_size=parallel.tp_size,
        )

    # -------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------
    hidden_state.name = "hidden_states"
    network.mark_output(hidden_state)

    if verbose:
        print(f"[trtmc build] Building XLNet encoder TRT engine "
              f"({num_layers} layers, hidden={hidden}, tp={parallel.tp_size}, "
              f"seq_len={max_seq_length}) ...", file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")

    return bytes(plan)


def _add_xlnet_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    pos_emb: trt.ITensor,
    seg_mat: trt.ITensor,
    attn_mask: trt.ITensor,
    hidden_size: int,
    attn_size: int,
    intermediate_size: int,
    num_heads: int,
    d_head: int,
    seq_length: int,
    scale_tensor: trt.ITensor,
    ff_activation: str,
    eps: float,
    tp_size: int,
) -> trt.ITensor:
    """Add one XLNet encoder layer with relative positional attention.

    XLNet layer (content-stream, bidirectional):
        q = h @ W_q, k = h @ W_k, v = h @ W_v
        k_r = pos_emb @ W_r
        ac = (q + r_w_bias) @ k^T          # content attention
        bd = (q + r_r_bias) @ k_r^T        # position attention (+ rel_shift)
        ef = segment attention              # segment-relative
        attn_score = (ac + bd + ef) * scale
        attn_out = softmax(attn_score) @ v
        output = LayerNorm(h + O(attn_out))  # post-norm
        output = LayerNorm(output + FFN(output))  # post-norm
    """
    klen = seq_length  # No mems

    # --- QKV projections: [seq_len, hidden] @ [hidden, attn_size] -> [seq_len, attn_size] ---
    q = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, attn_size,
        weights[f"{prefix}.w_q"])
    k = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, attn_size,
        weights[f"{prefix}.w_k"])
    v = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, attn_size,
        weights[f"{prefix}.w_v"])

    # Position key: [2*qlen, hidden] @ [hidden, attn_size] -> [2*qlen, attn_size]
    k_r = graph_ops.add_matmul_rhs_constant(
        network, pos_emb, hidden_size, attn_size,
        weights[f"{prefix}.w_r"])

    # Reshape QKV to multi-head: [seq_len, attn_size] -> [num_heads, seq_len, d_head]
    q_heads = network.add_shuffle(q)
    q_heads.reshape_dims = (seq_length, num_heads, d_head)
    q_heads.second_transpose = trt.Permutation([1, 0, 2])

    k_heads = network.add_shuffle(k)
    k_heads.reshape_dims = (seq_length, num_heads, d_head)
    k_heads.second_transpose = trt.Permutation([1, 0, 2])

    v_heads = network.add_shuffle(v)
    v_heads.reshape_dims = (seq_length, num_heads, d_head)
    v_heads.second_transpose = trt.Permutation([1, 0, 2])

    # Position key heads: [2*qlen, attn_size] -> [num_heads, 2*qlen, d_head]
    kr_heads = network.add_shuffle(k_r)
    kr_heads.reshape_dims = (2 * seq_length, num_heads, d_head)
    kr_heads.second_transpose = trt.Permutation([1, 0, 2])

    # --- Content attention: ac = (q + r_w_bias) @ k^T ---
    # r_w_bias: [num_heads, d_head] -> [num_heads, 1, d_head]
    r_w_bias = graph_ops.add_constant(
        network, (num_heads, 1, d_head), weights[f"{prefix}.r_w_bias"].reshape(num_heads, 1, d_head))
    q_plus_rw = network.add_elementwise(
        q_heads.get_output(0), r_w_bias,
        trt.ElementWiseOperation.SUM)

    # ac: [N, qlen, d_head] @ [N, klen, d_head]^T -> [N, qlen, klen]
    ac = network.add_matrix_multiply(
        q_plus_rw.get_output(0), trt.MatrixOperation.NONE,
        k_heads.get_output(0), trt.MatrixOperation.TRANSPOSE)

    # --- Position attention: bd = (q + r_r_bias) @ k_r^T ---
    r_r_bias = graph_ops.add_constant(
        network, (num_heads, 1, d_head), weights[f"{prefix}.r_r_bias"].reshape(num_heads, 1, d_head))
    q_plus_rr = network.add_elementwise(
        q_heads.get_output(0), r_r_bias,
        trt.ElementWiseOperation.SUM)

    # bd_raw: [N, qlen, d_head] @ [N, 2*qlen, d_head]^T -> [N, qlen, 2*qlen]
    bd_raw = network.add_matrix_multiply(
        q_plus_rr.get_output(0), trt.MatrixOperation.NONE,
        kr_heads.get_output(0), trt.MatrixOperation.TRANSPOSE)

    # Apply relative shift to bd
    bd = _add_rel_shift(network, bd_raw.get_output(0), num_heads, seq_length, klen)

    # --- Segment attention: ef ---
    # ef = einsum("ibnd,snd->ibns", q + r_s_bias, seg_embed)
    # ef = einsum("ijbs,ibns->bnij", seg_mat, ef)
    # With batch=1 removed:
    # q_plus_rs: [N, qlen, d_head], seg_embed: [2, N, d_head]
    # step 1: for each position i, compute dot of q[i] with seg_embed[s] -> [qlen, N, 2]
    # step 2: gather from seg_mat and sum -> [N, qlen, klen]

    r_s_bias = graph_ops.add_constant(
        network, (num_heads, 1, d_head), weights[f"{prefix}.r_s_bias"].reshape(num_heads, 1, d_head))
    q_plus_rs = network.add_elementwise(
        q_heads.get_output(0), r_s_bias,
        trt.ElementWiseOperation.SUM)
    # q_plus_rs: [N, qlen, d_head]

    # seg_embed: [2, N, d_head]
    seg_embed = graph_ops.add_constant(
        network, (2, num_heads, d_head), weights[f"{prefix}.seg_embed"])

    # Compute q_plus_rs @ seg_embed^T for each segment type
    # seg_embed[0]: [N, d_head] - same segment
    # seg_embed[1]: [N, d_head] - different segment
    # Reshape seg_embed to [2, N, d_head]
    # We need: for each head, for each query position, dot product with seg_embed[s]
    # = [N, qlen, d_head] @ [N, d_head, 2] -> [N, qlen, 2]

    # Transpose seg_embed: [2, N, d_head] -> [N, d_head, 2]
    seg_t = network.add_shuffle(seg_embed)
    seg_t.first_transpose = trt.Permutation([1, 2, 0])  # [N, d_head, 2]

    # ef_per_pos: [N, qlen, 2] = q_plus_rs @ seg_embed_t
    ef_per_pos = network.add_matrix_multiply(
        q_plus_rs.get_output(0), trt.MatrixOperation.NONE,
        seg_t.get_output(0), trt.MatrixOperation.NONE)

    # Now we need to select from ef_per_pos based on seg_mat
    # seg_mat: [qlen, klen, 2] (one-hot)
    # For each (i, j), ef[n, i, j] = sum_s(seg_mat[i, j, s] * ef_per_pos[n, i, s])
    # This is: ef_per_pos: [N, qlen, 2] matmul seg_mat_transposed: [qlen, 2, klen] -> won't work directly
    # Better: ef_per_pos[n,i,s] * seg_mat[i,j,s] summed over s
    # = einsum("nis,ijs->nij")
    # Equivalent to: for each i, ef_per_pos[n,i,:] @ seg_mat[i,:,:]^T
    # This is a batched matmul over the query dimension.
    #
    # Reshape ef_per_pos: [N, qlen, 2] -> [N*qlen, 1, 2]
    # Reshape seg_mat: [qlen, klen, 2] -> [qlen, 2, klen] -> broadcast to [N*qlen, 2, klen]
    # Result: [N*qlen, 1, klen] -> reshape to [N, qlen, klen]

    # Alternative: since seg_mat is [qlen, klen, 2], transpose to [qlen, 2, klen]
    seg_mat_t = network.add_shuffle(seg_mat)
    seg_mat_t.first_transpose = trt.Permutation([0, 2, 1])  # [qlen, 2, klen]

    # For each head n and position i:
    # ef[n,i,:] = ef_per_pos[n,i,:] @ seg_mat_t[i,:,:]
    # = [1, 2] @ [2, klen] -> [1, klen]
    # This requires a per-position batched matmul which TRT doesn't support directly.
    # Instead, decompose:
    # ef = ef_same * seg_mat_same + ef_diff * seg_mat_diff
    # where seg_mat_same[i,j] = seg_mat[i,j,0] and seg_mat_diff[i,j] = seg_mat[i,j,1]

    # Split ef_per_pos into same/diff: [N, qlen, 2] -> [N, qlen, 1] each
    ef_same_slice = network.add_slice(
        ef_per_pos.get_output(0),
        start=(0, 0, 0),
        shape=(num_heads, seq_length, 1),
        stride=(1, 1, 1))
    ef_diff_slice = network.add_slice(
        ef_per_pos.get_output(0),
        start=(0, 0, 1),
        shape=(num_heads, seq_length, 1),
        stride=(1, 1, 1))

    # seg_mat_same: seg_mat[:,:,0] -> [qlen, klen]
    seg_same_slice = network.add_slice(
        seg_mat,
        start=(0, 0, 0),
        shape=(seq_length, seq_length, 1),
        stride=(1, 1, 1))
    # Reshape to [1, qlen, klen]
    seg_same_2d = network.add_shuffle(seg_same_slice.get_output(0))
    seg_same_2d.reshape_dims = (1, seq_length, seq_length)

    seg_diff_slice = network.add_slice(
        seg_mat,
        start=(0, 0, 1),
        shape=(seq_length, seq_length, 1),
        stride=(1, 1, 1))
    seg_diff_2d = network.add_shuffle(seg_diff_slice.get_output(0))
    seg_diff_2d.reshape_dims = (1, seq_length, seq_length)

    # ef = ef_same * seg_same + ef_diff * seg_diff
    # ef_same_slice: [N, qlen, 1] broadcasts with seg_same_2d: [1, qlen, klen]
    ef_same_term = network.add_elementwise(
        ef_same_slice.get_output(0), seg_same_2d.get_output(0),
        trt.ElementWiseOperation.PROD)
    ef_diff_term = network.add_elementwise(
        ef_diff_slice.get_output(0), seg_diff_2d.get_output(0),
        trt.ElementWiseOperation.PROD)
    ef = network.add_elementwise(
        ef_same_term.get_output(0), ef_diff_term.get_output(0),
        trt.ElementWiseOperation.SUM)

    # --- Combine attention scores ---
    # attn_score = (ac + bd + ef) * scale
    ac_bd = network.add_elementwise(
        ac.get_output(0), bd,
        trt.ElementWiseOperation.SUM)
    ac_bd_ef = network.add_elementwise(
        ac_bd.get_output(0), ef.get_output(0),
        trt.ElementWiseOperation.SUM)
    scaled = network.add_elementwise(
        ac_bd_ef.get_output(0), scale_tensor,
        trt.ElementWiseOperation.PROD)

    # Apply attention mask (padding)
    masked = network.add_elementwise(
        scaled.get_output(0), attn_mask,
        trt.ElementWiseOperation.SUM)

    # For bidirectional XLNet, we do NOT apply causal mask (attn_type="bi")
    # No non_tgt_mask either (that's only used with perm_mask/target_mapping)

    # XLNet adds content, relative-position, and segment logits before
    # softmax. TRT native IAttention cannot express these extra logits as a
    # query-independent additive mask, so this attention remains decomposed.
    softmax = network.add_softmax(masked.get_output(0))
    softmax.axes = 1 << 2  # last dim (klen)

    # Context: softmax @ V -> [N, qlen, d_head]
    context_heads = network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE,
        v_heads.get_output(0), trt.MatrixOperation.NONE)

    # Reshape: [N, qlen, d_head] -> [qlen, N, d_head] -> [qlen, attn_size]
    context_flat = network.add_shuffle(context_heads.get_output(0))
    context_flat.first_transpose = trt.Permutation([1, 0, 2])
    context_flat.reshape_dims = (seq_length, attn_size)

    # Output projection: [qlen, attn_size] @ [attn_size, hidden] -> [qlen, hidden]
    # Note: XLNet's O weight is [d_model, n_head, d_head] reshaped to [d_model, attn_size]
    # The einsum is "ibnd,hnd->ibh" which is [qlen, N, d_head] @ [hidden, N, d_head]
    # This means O^T maps from [attn_size] to [hidden], so we need to transpose our weight.
    # Our w_o is already [hidden, attn_size] from reshape, so we need [attn_size, hidden]
    # for context_flat @ w_o^T.
    # Actually, the einsum "ibnd,hnd->ibh" contracts over n,d producing [i,b,h].
    # With our flattened version: context_flat @ W_o^T where W_o is [hidden, attn_size]
    # So we need to use TRANSPOSE on the rhs.
    w_o = graph_ops.add_constant(
        network, (hidden_size, attn_size), weights[f"{prefix}.w_o"])
    attn_out = network.add_matrix_multiply(
        context_flat.get_output(0), trt.MatrixOperation.NONE,
        w_o, trt.MatrixOperation.TRANSPOSE)
    attn_out_sum = add_all_reduce_sum(network, attn_out.get_output(0), tp_size)

    # POST-norm: LayerNorm(h + attn_out)
    residual1 = network.add_elementwise(
        hidden, attn_out_sum,
        trt.ElementWiseOperation.SUM)
    normed1 = _add_seq_layer_norm(
        network, residual1.get_output(0), hidden_size,
        weights[f"{prefix}.attn_norm"],
        weights[f"{prefix}.attn_norm_beta"], eps)

    # --- FFN ---
    fc1 = graph_ops.add_matmul_rhs_constant(
        network, normed1, hidden_size, intermediate_size,
        weights[f"{prefix}.w_fc1"])
    fc1 = graph_ops.add_bias_sum(network, fc1, intermediate_size,
                                  weights[f"{prefix}.fc1_bias"])
    activated = graph_ops.add_activation(network, fc1, ff_activation)
    fc2 = graph_ops.add_matmul_rhs_constant(
        network, activated, intermediate_size, hidden_size,
        weights[f"{prefix}.w_fc2"])
    fc2 = add_all_reduce_sum(network, fc2, tp_size)
    fc2 = graph_ops.add_bias_sum(network, fc2, hidden_size,
                                  weights[f"{prefix}.fc2_bias"])

    # POST-norm: LayerNorm(normed1 + ffn_out)
    residual2 = network.add_elementwise(
        normed1, fc2, trt.ElementWiseOperation.SUM)
    normed2 = _add_seq_layer_norm(
        network, residual2.get_output(0), hidden_size,
        weights[f"{prefix}.ff_norm"],
        weights[f"{prefix}.ff_norm_beta"], eps)

    return normed2
