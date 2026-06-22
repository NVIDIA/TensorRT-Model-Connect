"""Qwen3 non-autoregressive text encoder builder.

Builds a TRT engine for the Qwen3 model used as a text encoder in Z-Image.
Unlike the standard decoder builder, this:
  - Processes the entire sequence at once (no KV cache)
  - Uses bidirectional causal attention (full attention mask)
  - Returns hidden_states from a configurable layer (default: layer -2)

Engine I/O:
    Inputs:  input_ids [seq_len] int32, attention_mask [seq_len] float32
    Outputs: text_embeddings [seq_len, hidden_size] float32
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops
from .checkpoint_mapper import WeightDict, _open_safetensors, _load_tensor, _has_tensor
from ...engine_builder import add_dynamic_batch_profile


trt = trt_compat.get_trt()

def load_qwen3_encoder_weights(
    model_dir: str,
    *,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    intermediate_size: int,
    vocab_size: int,
) -> WeightDict:
    """Load Qwen3 encoder weights from HF safetensors."""
    model_path = Path(model_dir)
    readers = _open_safetensors(model_path)
    weights = WeightDict()

    def _t(name: str) -> np.ndarray:
        w = _load_tensor(readers, name)
        return np.ascontiguousarray(w.T, dtype=np.float32)

    def _f(name: str) -> np.ndarray:
        return _load_tensor(readers, name).astype(np.float32)

    # Embedding
    weights["embed_tokens"] = _f("model.embed_tokens.weight")

    for i in range(num_layers):
        p = f"model.layers.{i}"

        # Self-attention projections (transposed for matmul)
        weights[f"layer.{i}.q_proj"] = _t(f"{p}.self_attn.q_proj.weight")
        weights[f"layer.{i}.k_proj"] = _t(f"{p}.self_attn.k_proj.weight")
        weights[f"layer.{i}.v_proj"] = _t(f"{p}.self_attn.v_proj.weight")
        weights[f"layer.{i}.o_proj"] = _t(f"{p}.self_attn.o_proj.weight")

        # QK norms
        weights[f"layer.{i}.q_norm"] = _f(f"{p}.self_attn.q_norm.weight")
        weights[f"layer.{i}.k_norm"] = _f(f"{p}.self_attn.k_norm.weight")

        # RMSNorm
        weights[f"layer.{i}.input_layernorm"] = _f(f"{p}.input_layernorm.weight")
        weights[f"layer.{i}.post_attn_norm"] = _f(f"{p}.post_attention_layernorm.weight")

        # SwiGLU MLP
        weights[f"layer.{i}.gate_proj"] = _t(f"{p}.mlp.gate_proj.weight")
        weights[f"layer.{i}.up_proj"] = _t(f"{p}.mlp.up_proj.weight")
        weights[f"layer.{i}.down_proj"] = _t(f"{p}.mlp.down_proj.weight")

    # Final norm (only needed if output_layer < num_layers)
    if _has_tensor(readers, "model.norm.weight"):
        weights["final_norm"] = _f("model.norm.weight")

    return weights


def build_qwen3_encoder_engine(
    weights: WeightDict,
    *,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    intermediate_size: int,
    vocab_size: int,
    max_seq_len: int,
    rope_theta: float = 1000000.0,
    eps: float = 1e-6,
    output_layer: int = -2,
    verbose: bool = False,
    max_batch_size: int = 1,
    opt_batch_size: int | None = None,
) -> bytes:
    """Build Qwen3 text encoder TRT engine.

    Args:
        output_layer: Which layer's output to return. -2 means second-to-last.
        max_batch_size: Maximum batch size for the dynamic batch profile.
            When ``max_batch_size == 1`` (default), behavior is identical to
            the previous static-shape build (no batch dim, no profile).
            When > 1, a leading batch dim is added to the inputs and a single
            wide optimization profile ``kMIN=1, kOPT=opt_batch_size,
            kMAX=max_batch_size`` is attached (design Decisions A and C).
        opt_batch_size: kOPT for the dynamic batch profile. Defaults to
            ``min(max_batch_size, 4)`` per design Decision C.
        All other args describe the Qwen3 architecture.
    """
    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be >= 1 (got {max_batch_size})")
    if max_batch_size > 1:
        return _build_qwen3_encoder_engine_batched(
            weights,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            eps=eps,
            output_layer=output_layer,
            max_batch_size=max_batch_size,
            opt_batch_size=opt_batch_size,
            verbose=verbose,
        )

    del opt_batch_size  # unused on the static (B=1) path
    if output_layer < 0:
        output_layer = num_layers + output_layer  # e.g., 36 + (-2) = 34

    kv_dim = num_kv_heads * head_dim

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    # Inputs — static-shape legacy path (max_batch_size==1).
    input_ids = network.add_input("input_ids", trt.int32, (max_seq_len,))
    attn_mask = network.add_input(
        "attention_mask", trt.float32, (max_seq_len,))

    # Constants
    eps_t = graph_ops.add_constant(network, (1, 1), np.array([eps], dtype=np.float32))

    # Embedding
    embed_table = graph_ops.add_constant(
        network, (vocab_size, hidden_size), weights["embed_tokens"])
    hidden = network.add_gather(embed_table, input_ids, 0).get_output(0)

    graph_ops.validate_native_rope_dim(head_dim, field_name="head_dim")
    rope_cos_half_np = graph_ops.make_rope_table_half_dim(
        max_seq_len, head_dim, rope_theta, True)
    rope_sin_half_np = graph_ops.make_rope_table_half_dim(
        max_seq_len, head_dim, rope_theta, False)
    rope_cos_half = graph_ops.add_constant(
        network, rope_cos_half_np.shape, rope_cos_half_np)
    rope_sin_half = graph_ops.add_constant(
        network, rope_sin_half_np.shape, rope_sin_half_np)
    rope_position_ids = graph_ops.add_constant(
        network, (max_seq_len,), np.arange(max_seq_len, dtype=np.int32),
        dtype=np.int32)

    # Build attention mask for native IAttention. attn_mask input is 0.0 for
    # valid tokens and -1e9 for padding; [1, 1, 1, S] broadcasts across heads
    # and query positions.
    # We mask padded positions with -1e9
    mask_reshape = network.add_shuffle(attn_mask)
    mask_reshape.reshape_dims = (1, 1, 1, max_seq_len)

    for layer_idx in range(num_layers):
        if layer_idx == output_layer:
            # Save this hidden state as output
            output_hidden = hidden

        # RMSNorm
        normed = graph_ops.add_rms_norm(
            network, hidden, hidden_size,
            weights[f"layer.{layer_idx}.input_layernorm"], eps_t)

        # QKV projections
        q = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, num_heads * head_dim,
            weights[f"layer.{layer_idx}.q_proj"])
        k = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, kv_dim,
            weights[f"layer.{layer_idx}.k_proj"])
        v = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, kv_dim,
            weights[f"layer.{layer_idx}.v_proj"])

        # QK norms (per-head RMSNorm)
        q_norm_w = weights[f"layer.{layer_idx}.q_norm"]
        k_norm_w = weights[f"layer.{layer_idx}.k_norm"]
        # Tile per-head norm weights for all heads
        q_norm_tiled = np.tile(q_norm_w.reshape(1, head_dim), (num_heads, 1))
        k_norm_tiled = np.tile(k_norm_w.reshape(1, head_dim), (num_kv_heads, 1))

        q = _add_per_head_rms_norm(network, q, num_heads, head_dim, q_norm_tiled, eps_t, max_seq_len)
        k = _add_per_head_rms_norm(network, k, num_kv_heads, head_dim, k_norm_tiled, eps_t, max_seq_len)

        q = graph_ops.add_apply_rope_native(
            network, q, num_heads, head_dim,
            rope_cos_half, rope_sin_half, rope_position_ids,
            head_dim, sequence_length=max_seq_len)
        k = graph_ops.add_apply_rope_native(
            network, k, num_kv_heads, head_dim,
            rope_cos_half, rope_sin_half, rope_position_ids,
            head_dim, sequence_length=max_seq_len)

        ctx_flat = graph_ops.add_attention_from_rows(
            network, q, k, v,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_seq=max_seq_len, kv_seq=max_seq_len,
            mask=mask_reshape.get_output(0),
            tag=f"layer.{layer_idx}.attn")

        # Output projection
        attn_out = graph_ops.add_matmul_rhs_constant(
            network, ctx_flat, num_heads * head_dim, hidden_size,
            weights[f"layer.{layer_idx}.o_proj"])

        # Residual
        hidden = network.add_elementwise(
            hidden, attn_out, trt.ElementWiseOperation.SUM).get_output(0)

        # Post-attention RMSNorm
        normed2 = graph_ops.add_rms_norm(
            network, hidden, hidden_size,
            weights[f"layer.{layer_idx}.post_attn_norm"], eps_t)

        # SwiGLU MLP
        gate = graph_ops.add_matmul_rhs_constant(
            network, normed2, hidden_size, intermediate_size,
            weights[f"layer.{layer_idx}.gate_proj"])
        up = graph_ops.add_matmul_rhs_constant(
            network, normed2, hidden_size, intermediate_size,
            weights[f"layer.{layer_idx}.up_proj"])

        # SiLU(gate) * up
        sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
        silu = network.add_elementwise(
            gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
        gated = network.add_elementwise(
            silu.get_output(0), up, trt.ElementWiseOperation.PROD)

        down = graph_ops.add_matmul_rhs_constant(
            network, gated.get_output(0), intermediate_size, hidden_size,
            weights[f"layer.{layer_idx}.down_proj"])

        # Residual
        hidden = network.add_elementwise(
            hidden, down, trt.ElementWiseOperation.SUM).get_output(0)

    # Use the output from the target layer
    if output_layer >= num_layers:
        output_hidden = hidden
    elif output_layer < 0:
        output_hidden = hidden

    cast_out = network.add_cast(output_hidden, trt.float32)
    out_final = cast_out.get_output(0)
    out_final.name = "text_embeddings"
    network.mark_output(out_final)

    print(f"[qwen3-encoder] Building TRT engine "
          f"(layers={num_layers}, hidden={hidden_size}, output_layer={output_layer}, "
          f"seq_len={max_seq_len}) ...", file=sys.stderr)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("Qwen3 encoder TRT engine build failed")
    return bytes(plan)


def _add_per_head_rms_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    gamma: np.ndarray,
    eps_t: trt.ITensor,
    seq_len: int,
) -> trt.ITensor:
    """Per-head RMSNorm for sequence input [seq_len, num_heads * head_dim]."""
    return graph_ops.add_rms_norm_per_head(
        network, inp, num_heads, head_dim, gamma, eps_t,
        sequence_length=seq_len)


# ---------------------------------------------------------------------------
# Dynamic-batch path (max_batch_size > 1)
# ---------------------------------------------------------------------------


def _dynamic_batch_shape(network, reference, tail: tuple[int, ...]):
    """Shape tensor ``[B, *tail]`` using dim 0 from a dynamic-batch tensor."""
    ref_shape = network.add_shape(reference).get_output(0)
    batch = network.add_slice(ref_shape, start=(0,), shape=(1,), stride=(1,))
    tail_t = graph_ops.add_constant(
        network, (len(tail),), np.asarray(tail, dtype=np.int64), dtype=np.int64)
    target = network.add_concatenation([batch.get_output(0), tail_t])
    target.axis = 0
    return target.get_output(0)


def _slice_batched_last_dim(network, x, seq_len: int, num_heads: int,
                            start: int, width: int):
    """Slice ``[B, S, H, *]`` along the last axis preserving runtime B."""
    s = network.add_slice(
        x, start=(0, 0, 0, start), shape=(0, 0, 0, 0), stride=(1, 1, 1, 1))
    s.set_input(2, _dynamic_batch_shape(network, x, (seq_len, num_heads, width)))
    return s.get_output(0)


def _reshape_batched_rows_to_heads_4d(network, x, num_heads: int, head_dim: int,
                                      seq_len: int):
    """``[B, S, H*D]`` -> ``[B, H, S, D]``."""
    r = network.add_shuffle(x)
    r.reshape_dims = (-1, seq_len, num_heads, head_dim)
    r.second_transpose = trt.Permutation([0, 2, 1, 3])
    return r.get_output(0)


def _reshape_heads_4d_to_batched_rows(network, x, num_heads: int, head_dim: int,
                                      seq_len: int):
    """``[B, H, S, D]`` -> ``[B, S, H*D]``."""
    r = network.add_shuffle(x)
    r.first_transpose = trt.Permutation([0, 2, 1, 3])
    r.reshape_dims = (-1, seq_len, num_heads * head_dim)
    return r.get_output(0)


def _add_apply_rope_native_batched(network, inp, cos_cache_2d, sin_cache_2d,
                                   num_heads: int, head_dim: int, seq_len: int):
    """Apply rotate-half RoPE to ``[B, S, H*D]`` using a static per-position cache."""
    half = head_dim // 2
    x = network.add_shuffle(inp)
    x.reshape_dims = (-1, seq_len, num_heads, head_dim)
    x_4d = x.get_output(0)

    x1 = _slice_batched_last_dim(network, x_4d, seq_len, num_heads, 0, half)
    x2 = _slice_batched_last_dim(network, x_4d, seq_len, num_heads, half, half)

    cos = network.add_shuffle(cos_cache_2d)
    cos.reshape_dims = (1, seq_len, 1, half)
    sin = network.add_shuffle(sin_cache_2d)
    sin.reshape_dims = (1, seq_len, 1, half)

    x1_cos = network.add_elementwise(x1, cos.get_output(0), trt.ElementWiseOperation.PROD)
    x2_sin = network.add_elementwise(x2, sin.get_output(0), trt.ElementWiseOperation.PROD)
    first = network.add_elementwise(
        x1_cos.get_output(0), x2_sin.get_output(0), trt.ElementWiseOperation.SUB)

    x2_cos = network.add_elementwise(x2, cos.get_output(0), trt.ElementWiseOperation.PROD)
    x1_sin = network.add_elementwise(x1, sin.get_output(0), trt.ElementWiseOperation.PROD)
    second = network.add_elementwise(
        x2_cos.get_output(0), x1_sin.get_output(0), trt.ElementWiseOperation.SUM)

    rope = network.add_concatenation([first.get_output(0), second.get_output(0)])
    rope.axis = 3
    out = network.add_shuffle(rope.get_output(0))
    out.reshape_dims = (-1, seq_len, num_heads * head_dim)
    return out.get_output(0)


def _add_attention_from_batched_rows(network, q, k, v, *,
                                     num_heads: int, num_kv_heads: int,
                                     head_dim: int, q_seq: int, kv_seq: int,
                                     mask):
    """Native IAttention for ``[B, S, H*D]`` Q and ``[B, S, KVH*D]`` K/V."""
    q_4d = _reshape_batched_rows_to_heads_4d(network, q, num_heads, head_dim, q_seq)
    k_4d = _reshape_batched_rows_to_heads_4d(network, k, num_kv_heads, head_dim, kv_seq)
    v_4d = _reshape_batched_rows_to_heads_4d(network, v, num_kv_heads, head_dim, kv_seq)
    ctx_4d = graph_ops.add_attention_core(
        network, q_4d, k_4d, v_4d,
        causal=False, mask=mask,
        scale=float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0,
    )
    return _reshape_heads_4d_to_batched_rows(
        network, ctx_4d, num_heads, head_dim, q_seq)


def _build_qwen3_encoder_engine_batched(
    weights: WeightDict,
    *,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    intermediate_size: int,
    vocab_size: int,
    max_seq_len: int,
    rope_theta: float,
    eps: float,
    output_layer: int,
    max_batch_size: int,
    opt_batch_size: int | None,
    verbose: bool,
) -> bytes:
    """Build a dynamic-leading-batch Qwen3 encoder TRT engine."""
    if opt_batch_size is None:
        opt_batch_size = min(max_batch_size, 4)
    if output_layer < 0:
        output_layer = num_layers + output_layer

    kv_dim = num_kv_heads * head_dim

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    input_ids = network.add_input("input_ids", trt.int32, (-1, max_seq_len))
    attn_mask = network.add_input(
        "attention_mask", trt.float32, (-1, max_seq_len))

    add_dynamic_batch_profile(
        builder, config, network,
        input_names=["input_ids", "attention_mask"],
        max_batch=max_batch_size,
        opt_batch=opt_batch_size,
        static_shape={
            "input_ids": (max_seq_len,),
            "attention_mask": (max_seq_len,),
        },
    )

    # eps tensor shaped (1, 1, 1) so it broadcasts with [B, S, D] RMSNorms.
    eps_t = graph_ops.add_constant(
        network, (1, 1, 1), np.array([eps], dtype=np.float32))

    embed_table = graph_ops.add_constant(
        network, (vocab_size, hidden_size), weights["embed_tokens"])
    hidden = network.add_gather(embed_table, input_ids, 0).get_output(0)

    graph_ops.validate_native_rope_dim(head_dim, field_name="head_dim")
    rope_cos_half_np = graph_ops.make_rope_table_half_dim(
        max_seq_len, head_dim, rope_theta, True)
    rope_sin_half_np = graph_ops.make_rope_table_half_dim(
        max_seq_len, head_dim, rope_theta, False)
    rope_cos_half = graph_ops.add_constant(
        network, rope_cos_half_np.shape, rope_cos_half_np)
    rope_sin_half = graph_ops.add_constant(
        network, rope_sin_half_np.shape, rope_sin_half_np)

    mask_reshape = network.add_shuffle(attn_mask)
    mask_reshape.reshape_dims = (-1, 1, 1, max_seq_len)
    mask_4d = mask_reshape.get_output(0)

    output_hidden = hidden
    for layer_idx in range(num_layers):
        if layer_idx == output_layer:
            output_hidden = hidden

        normed = graph_ops.add_rms_norm_last_dim(
            network, hidden, hidden_size,
            weights[f"layer.{layer_idx}.input_layernorm"], eps_t)

        q = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, num_heads * head_dim,
            weights[f"layer.{layer_idx}.q_proj"])
        k = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, kv_dim,
            weights[f"layer.{layer_idx}.k_proj"])
        v = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, kv_dim,
            weights[f"layer.{layer_idx}.v_proj"])

        q_norm_w = weights[f"layer.{layer_idx}.q_norm"]
        k_norm_w = weights[f"layer.{layer_idx}.k_norm"]
        q_norm_tiled = np.tile(q_norm_w.reshape(1, head_dim), (num_heads, 1))
        k_norm_tiled = np.tile(k_norm_w.reshape(1, head_dim), (num_kv_heads, 1))

        q = graph_ops.add_rms_norm_per_head_batched(
            network, q, num_heads, head_dim, q_norm_tiled, eps_t,
            sequence_length=max_seq_len)
        k = graph_ops.add_rms_norm_per_head_batched(
            network, k, num_kv_heads, head_dim, k_norm_tiled, eps_t,
            sequence_length=max_seq_len)

        q = _add_apply_rope_native_batched(
            network, q, rope_cos_half, rope_sin_half,
            num_heads, head_dim, max_seq_len)
        k = _add_apply_rope_native_batched(
            network, k, rope_cos_half, rope_sin_half,
            num_kv_heads, head_dim, max_seq_len)

        ctx_flat = _add_attention_from_batched_rows(
            network, q, k, v,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, q_seq=max_seq_len, kv_seq=max_seq_len,
            mask=mask_4d)

        attn_out = graph_ops.add_matmul_rhs_constant(
            network, ctx_flat, num_heads * head_dim, hidden_size,
            weights[f"layer.{layer_idx}.o_proj"])

        hidden = network.add_elementwise(
            hidden, attn_out, trt.ElementWiseOperation.SUM).get_output(0)

        normed2 = graph_ops.add_rms_norm_last_dim(
            network, hidden, hidden_size,
            weights[f"layer.{layer_idx}.post_attn_norm"], eps_t)

        gate = graph_ops.add_matmul_rhs_constant(
            network, normed2, hidden_size, intermediate_size,
            weights[f"layer.{layer_idx}.gate_proj"])
        up = graph_ops.add_matmul_rhs_constant(
            network, normed2, hidden_size, intermediate_size,
            weights[f"layer.{layer_idx}.up_proj"])
        sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
        silu = network.add_elementwise(
            gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
        gated = network.add_elementwise(
            silu.get_output(0), up, trt.ElementWiseOperation.PROD)
        down = graph_ops.add_matmul_rhs_constant(
            network, gated.get_output(0), intermediate_size, hidden_size,
            weights[f"layer.{layer_idx}.down_proj"])

        hidden = network.add_elementwise(
            hidden, down, trt.ElementWiseOperation.SUM).get_output(0)

    if output_layer >= num_layers:
        output_hidden = hidden

    cast_out = network.add_cast(output_hidden, trt.float32)
    out_final = cast_out.get_output(0)
    out_final.name = "text_embeddings"
    network.mark_output(out_final)

    print(f"[qwen3-encoder] Building dynamic-batch TRT engine "
          f"(B=1..{max_batch_size}, opt={opt_batch_size}, layers={num_layers}, "
          f"hidden={hidden_size}, output_layer={output_layer}, "
          f"seq_len={max_seq_len}) ...", file=sys.stderr)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("Qwen3 encoder TRT engine build failed")
    return bytes(plan)
