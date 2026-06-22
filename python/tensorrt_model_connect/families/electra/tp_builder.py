"""Tensor-parallel encoder-only engine builder (ELECTRA-style).

Builds one rank-local TensorRT engine for an encoder-only transformer. Unlike
the standard decoder builder, this has:
  - No KV cache (processes entire sequence at once)
  - No causal attention mask (full bidirectional attention)
  - POST-norm architecture (residual + LayerNorm after attention/FFN)
  - Token type embeddings (segment A/B)
  - Output: hidden_states [seq_len, hidden_size]

Tensor parallelism only changes the encoder projections:
  - Q/K/V and FFN fc1 are column-parallel
  - Attention output and FFN fc2 are row-parallel
  - Row-parallel outputs are joined with TensorRT distributed ALL_REDUCE

Tensor names for the C++ runtime:
  Inputs:  input_ids [seq_len], attention_mask [seq_len]
  Outputs: hidden_states [seq_len, hidden_size]

token_type_ids is baked as constant zeros (single-segment inference).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops
from .config import ModelConfig
from ...parallel_config import add_all_reduce_sum, normalize_parallel_config

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from ...parallel_config import ParallelConfig


def _slice_last_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=-1)[rank])


def _slice_first_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=0)[rank])


def _validate_encoder_tp(
    config: ModelConfig,
    weights: "WeightDict",
    parallel: "ParallelConfig",
) -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError("ELECTRA tensor-parallel build requires a concrete rank")

    tp = parallel.tp_size
    if config.num_attention_heads % tp != 0:
        raise ValueError(
            "ELECTRA tensor parallel requires num_attention_heads divisible by "
            f"tp_size ({config.num_attention_heads} vs {tp})")
    if config.intermediate_size % tp != 0:
        raise ValueError(
            "ELECTRA tensor parallel requires intermediate_size divisible by "
            f"tp_size ({config.intermediate_size} vs {tp})")

    for layer_idx in range(config.num_hidden_layers):
        prefix = f"layer.{layer_idx}"
        for key in (f"{prefix}.w_q", f"{prefix}.w_k", f"{prefix}.w_v"):
            if weights[key].shape[-1] % tp != 0:
                raise ValueError(f"{key} output dim must be divisible by tp_size")
        if weights[f"{prefix}.w_o"].shape[0] % tp != 0:
            raise ValueError(f"{prefix}.w_o input dim must be divisible by tp_size")
        if weights[f"{prefix}.w_fc1"].shape[-1] % tp != 0:
            raise ValueError(f"{prefix}.w_fc1 output dim must be divisible by tp_size")
        if weights[f"{prefix}.w_fc2"].shape[0] % tp != 0:
            raise ValueError(f"{prefix}.w_fc2 input dim must be divisible by tp_size")

    if "relative_position_bias" in weights and weights["relative_position_bias"].shape[0] % tp != 0:
        raise ValueError(
            "relative_position_bias head dim must be divisible by tp_size")


def shard_encoder_weights(
    config: ModelConfig,
    weights: "WeightDict",
    *,
    parallel: "ParallelConfig",
) -> "WeightDict":
    """Return rank-local ELECTRA-style encoder weights for the TP builder."""
    _validate_encoder_tp(config, weights, parallel)
    if not parallel.enabled:
        return weights

    out = type(weights)()
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
            continue

        if key.endswith((".w_q", ".w_k", ".w_v", ".w_fc1")):
            out[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith((".q_bias", ".k_bias", ".v_bias", ".fc1_bias")):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith((".w_o", ".w_fc2")):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        elif key == "relative_position_bias":
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        else:
            out[key] = value

    out["_attention_size"] = config.attention_size // parallel.tp_size
    out["_intermediate_size"] = config.intermediate_size // parallel.tp_size
    out["_tensor_parallel_size"] = parallel.tp_size
    out["_tensor_parallel_rank"] = parallel.rank
    return out


def build_tp_encoder_engine(
    config: ModelConfig,
    weights: "WeightDict",
    max_seq_length: int,
    *,
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    """Build a rank-local TRT engine plan for an ELECTRA-style encoder.

    Args:
        config: Model architecture from config.json.
        weights: Loaded weight dict from ELECTRA plugin.
        max_seq_length: Maximum sequence length the engine is compiled for.
        verbose: Print TRT builder logs.

    Returns:
        Serialized engine plan bytes.
    """
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError(
            "build_tp_encoder_engine requires tensor_parallel mode and tp_size > 1")
    weights = shard_encoder_weights(config, weights, parallel=parallel)

    hidden = config.hidden_size
    num_layers = config.num_hidden_layers
    full_num_heads = config.num_attention_heads
    num_heads = config.num_attention_heads // parallel.tp_size
    head_dim = hidden // full_num_heads
    intermediate = config.intermediate_size // parallel.tp_size
    eps = config.rms_norm_eps  # Actually layer_norm_eps for ELECTRA
    type_vocab_size = config.raw.get("type_vocab_size", 2)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    # Disable TF32 to ensure full FP32 precision. TF32 uses 10-bit mantissa
    # which causes significant accuracy loss across 12+ encoder layers.
    trt_config.clear_flag(trt.BuilderFlag.TF32)

    # Resolve GELU variant from config
    hidden_act = config.hidden_act or config.raw.get("activation", "") or "gelu"

    # -------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------
    input_ids = network.add_input("input_ids", trt.int32, (max_seq_length,))
    attention_mask_input = network.add_input("attention_mask", trt.int32, (max_seq_length,))

    # token_type_ids: constant zeros (all segment-0) — the C++ encoder
    # pipeline doesn't provide this input, and inference is single-segment.
    tt_zeros = network.add_constant(
        (max_seq_length,), trt.Weights(np.zeros(max_seq_length, dtype=np.int32)))
    token_type_ids = tt_zeros.get_output(0)

    # -------------------------------------------------------------------
    # Shared constants (support embedding factorization: embedding_size != hidden)
    # -------------------------------------------------------------------
    embedding_size = weights["embedding"].shape[1]  # may differ from hidden (e.g. ALELECTRA 128 vs 768)
    embedding_table = graph_ops.add_constant(
        network, weights["embedding"].shape, weights["embedding"])
    position_embed_table = graph_ops.add_constant(
        network, weights["position_embedding"].shape, weights["position_embedding"])
    token_type_table = graph_ops.add_constant(
        network, (type_vocab_size, embedding_size), weights["token_type_embedding"])

    graph_ops.add_constant(
        network, (1, 1), np.array([eps], dtype=np.float32))
    attn_scale_tensor = graph_ops.add_constant(
        network, (1, 1, 1), np.array([1.0 / np.sqrt(max(head_dim, 1))], dtype=np.float32))

    # Build additive attention mask from attention_mask input:
    # attention_mask is [seq_len] with 1=real, 0=padding.
    # Convert to [1, 1, seq_len] additive mask: 0.0 for real, -1e10 for padding.
    mask_float = network.add_cast(attention_mask_input, trt.float32)
    ones_mask = graph_ops.add_constant(
        network, (1,), np.array([1.0], dtype=np.float32))
    neg_large = graph_ops.add_constant(
        network, (1,), np.array([-1e10], dtype=np.float32))
    inv_mask = network.add_elementwise(
        ones_mask, mask_float.get_output(0),
        trt.ElementWiseOperation.SUB)  # 0 for real, 1 for pad
    pad_penalty = network.add_elementwise(
        inv_mask.get_output(0), neg_large,
        trt.ElementWiseOperation.PROD)  # 0.0 for real, -1e10 for pad
    # Reshape to [1, 1, seq_len] for broadcasting: [num_heads, seq_len, seq_len]
    pad_mask_reshape = network.add_shuffle(pad_penalty.get_output(0))
    pad_mask_reshape.reshape_dims = (1, 1, max_seq_length)

    # Position indices: [0, 1, 2, ..., max_seq_length-1]
    position_indices = graph_ops.add_constant(
        network, (max_seq_length,),
        np.arange(max_seq_length, dtype=np.int32).astype(np.float32))
    # Cast back to int32 for gather
    pos_int = network.add_cast(position_indices, trt.int32)

    # -------------------------------------------------------------------
    # Embedding: word + position + token_type + LayerNorm
    # -------------------------------------------------------------------
    word_embed = network.add_gather(embedding_table, input_ids, 0)
    pos_embed = network.add_gather(position_embed_table, pos_int.get_output(0), 0)
    tt_embed = network.add_gather(token_type_table, token_type_ids, 0)

    # Sum all three embedding types
    embed_sum1 = network.add_elementwise(
        word_embed.get_output(0), pos_embed.get_output(0),
        trt.ElementWiseOperation.SUM)
    embed_sum2 = network.add_elementwise(
        embed_sum1.get_output(0), tt_embed.get_output(0),
        trt.ElementWiseOperation.SUM)

    # Embedding LayerNorm (over embedding_size, which may differ from hidden)
    hidden_state = _add_seq_layer_norm(
        network, embed_sum2.get_output(0), embedding_size, max_seq_length,
        weights["embed_norm"], weights["embed_norm_beta"], eps)

    # Optional embedding projection: embedding_size -> hidden_size (ALELECTRA, FNet)
    if "embed_projection" in weights:
        hidden_state = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, embedding_size, hidden,
            weights["embed_projection"])
        if "embed_projection_bias" in weights:
            hidden_state = graph_ops.add_bias_sum(
                network, hidden_state, hidden,
                weights["embed_projection_bias"])

    # -------------------------------------------------------------------
    # Optional relative position bias (MPNet, T5-style)
    # -------------------------------------------------------------------
    rel_pos_bias_tensor = None
    if "relative_position_bias" in weights:
        # Pre-computed [num_heads, seq_len, seq_len] bias matrix
        rpb = weights["relative_position_bias"]
        rel_pos_bias_tensor = graph_ops.add_constant(
            network, rpb.shape, rpb)

    # -------------------------------------------------------------------
    # Encoder layers
    # -------------------------------------------------------------------
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"

        hidden_state = _add_encoder_layer(
            network=network,
            hidden=hidden_state,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_heads=num_heads,
            head_dim=head_dim,
            seq_length=max_seq_length,
            attn_scale_tensor=attn_scale_tensor,
            attn_mask=pad_mask_reshape.get_output(0),
            rel_pos_bias=rel_pos_bias_tensor,
            hidden_act=hidden_act,
            eps=eps,
            tp_size=parallel.tp_size,
        )

    # -------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------
    hidden_state.name = "hidden_states"
    network.mark_output(hidden_state)

    # -------------------------------------------------------------------
    # Build engine
    # -------------------------------------------------------------------
    if verbose:
        print(f"[trtmc build] Building ELECTRA encoder TRT engine "
              f"({num_layers} layers, hidden={hidden}, tp={parallel.tp_size}, "
              f"seq_len={max_seq_length}) ...", file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")

    return bytes(plan)


def _add_seq_layer_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    seq_length: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float,
) -> trt.ITensor:
    """LayerNorm over sequence of vectors: [seq_len, hidden] -> [seq_len, hidden].

    Uses the shared TRT native normalization helper.
    """
    return graph_ops.add_layer_norm_native(
        network, inp, hidden_size, gamma, beta, eps)


def _add_encoder_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    head_dim: int,
    seq_length: int,
    attn_scale_tensor: trt.ITensor,
    attn_mask: trt.ITensor,
    rel_pos_bias: trt.ITensor | None = None,
    hidden_act: str = "gelu",
    eps: float,
    tp_size: int,
) -> trt.ITensor:
    """Add one ELECTRA encoder layer with POST-norm.

    ELECTRA architecture per layer:
        attn_out = MultiHeadSelfAttention(hidden)
        hidden = LayerNorm(hidden + attn_out)  # post-norm
        ffn_out = FFN(hidden)
        hidden = LayerNorm(hidden + ffn_out)   # post-norm
    """
    attention_size = num_heads * head_dim

    # --- Self-attention (no causal mask, bidirectional) ---
    # QKV projections
    q = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, attention_size,
        weights[f"{prefix}.w_q"])
    k = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, attention_size,
        weights[f"{prefix}.w_k"])
    v = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, attention_size,
        weights[f"{prefix}.w_v"])

    # QKV biases
    q = graph_ops.add_bias_sum(network, q, attention_size, weights[f"{prefix}.q_bias"])
    k = graph_ops.add_bias_sum(network, k, attention_size, weights[f"{prefix}.k_bias"])
    v = graph_ops.add_bias_sum(network, v, attention_size, weights[f"{prefix}.v_bias"])

    # Add relative position bias if present (MPNet/T5-style). IAttention
    # applies scaling internally through graph_ops.add_attention_from_rows, so
    # the mask contains only additive bias/masking terms.
    additive_mask = attn_mask
    if rel_pos_bias is not None:
        additive_mask = network.add_elementwise(
            rel_pos_bias, attn_mask, trt.ElementWiseOperation.SUM).get_output(0)

    mask_4d = network.add_shuffle(additive_mask)
    mask_4d.reshape_dims = (
        (1, num_heads, seq_length, seq_length)
        if rel_pos_bias is not None
        else (1, 1, 1, seq_length)
    )

    context_flat = graph_ops.add_attention_from_rows(
        network, q, k, v,
        num_heads=num_heads, head_dim=head_dim,
        q_seq=seq_length, kv_seq=seq_length,
        mask=mask_4d.get_output(0),
        tag=prefix + ".attn")

    # Output projection
    attn_out = graph_ops.add_matmul_rhs_constant(
        network, context_flat,
        attention_size, hidden_size,
        weights[f"{prefix}.w_o"])
    attn_out = add_all_reduce_sum(network, attn_out, tp_size)
    attn_out = graph_ops.add_bias_sum(
        network, attn_out, hidden_size, weights[f"{prefix}.o_bias"])

    # POST-norm: LayerNorm(hidden + attn_out)
    residual1 = network.add_elementwise(
        hidden, attn_out, trt.ElementWiseOperation.SUM)
    normed1 = _add_seq_layer_norm(
        network, residual1.get_output(0), hidden_size, seq_length,
        weights[f"{prefix}.post_attn_norm"],
        weights[f"{prefix}.post_attn_norm_beta"], eps)

    # --- FFN: fc1 -> GELU -> fc2 ---
    fc1 = graph_ops.add_matmul_rhs_constant(
        network, normed1, hidden_size, intermediate_size,
        weights[f"{prefix}.w_fc1"])
    fc1 = graph_ops.add_bias_sum(network, fc1, intermediate_size,
                                  weights[f"{prefix}.fc1_bias"])
    activated = graph_ops.add_activation(network, fc1, hidden_act)
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
        network, residual2.get_output(0), hidden_size, seq_length,
        weights[f"{prefix}.output_norm"],
        weights[f"{prefix}.output_norm_beta"], eps)

    return normed2
