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

from ... import graph_ops
from ...checkpoint_mapper import WeightDict, _open_safetensors, _load_tensor, _has_tensor


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
) -> bytes:
    """Build Qwen3 text encoder TRT engine.

    Args:
        output_layer: Which layer's output to return. -2 means second-to-last.
        All other args describe the Qwen3 architecture.
    """
    if output_layer < 0:
        output_layer = num_layers + output_layer  # e.g., 36 + (-2) = 34

    kv_dim = num_kv_heads * head_dim

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    # Inputs
    input_ids = network.add_input("input_ids", trt.int32, (max_seq_len,))
    attn_mask = network.add_input("attention_mask", trt.float32, (max_seq_len,))

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
