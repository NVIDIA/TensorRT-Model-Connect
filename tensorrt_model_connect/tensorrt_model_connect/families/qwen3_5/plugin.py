"""Qwen3.5 family plugin -- Hybrid Gated DeltaNet + self-attention decoder.

Qwen3.5 uses a heterogeneous layer stack with two layer types defined by
text_config.layer_types (list of strings):
  "linear" = Gated DeltaNet layer (linear attention with delta rule)
  "FULL"   = Standard self-attention layer (GQA, partial RoPE, output gating)

Qwen3.5-9B: 32 layers (24 DeltaNet "linear" + 8 self-attention "FULL").
FULL layers appear every 4th layer (indices 3, 7, 11, ...).

Key architecture details:

  DeltaNet layers (24 layers):
    - in_proj_qkv -> conv1d step -> SiLU -> split Q[nkv×dim], K[nkv×dim], V[nheads×dim]
    - L2-norm Q and K
    - keep compact Q,K from num_kv_heads -> num_heads
    - Delta rule state update: state [nheads, head_dim, head_dim]
    - Gated RMSNorm with separate gate projection (in_proj_z)
    - Decay: -exp(A_log) * softplus(a_proj(x) + dt_bias) per head
    - Beta (write strength): sigmoid(b_proj(x)) per head

  Full attention layers (8 layers):
    - q_proj [2*attn_size, hidden] -> split query + gate
    - QK-norm with (1+weight) centering
    - Partial RoPE (partial_rotary_factor=0.25, 64 of 256 dims)
    - KV cache + scaled dot-product attention
    - Output gating: attn_out * sigmoid(gate)

Weight key mapping (HF -> engine):
  model.language_model.embed_tokens.weight               -> embedding
  model.language_model.layers.{i}.input_layernorm.weight  -> layer.{i}.input_norm
  --- DeltaNet (linear) layers ---
  model.language_model.layers.{i}.linear_attn.in_proj.weight      -> deltanet_in_proj_qkv
  model.language_model.layers.{i}.linear_attn.g_proj.weight       -> deltanet_z_proj (gate)
  model.language_model.layers.{i}.linear_attn.a_proj.weight       -> deltanet_a_proj (decay)
  model.language_model.layers.{i}.linear_attn.b_proj.weight       -> deltanet_b_proj (beta)
  model.language_model.layers.{i}.linear_attn.A_log               -> A
  model.language_model.layers.{i}.linear_attn.dt_bias             -> dt_bias
  model.language_model.layers.{i}.linear_attn.conv1d.weight/bias  -> conv1d
  model.language_model.layers.{i}.linear_attn.norm.weight         -> deltanet_norm
  model.language_model.layers.{i}.linear_attn.o_proj.weight       -> deltanet_out_proj
  --- Full attention layers ---
  model.language_model.layers.{i}.self_attn.q_proj.weight         -> split: w_q + w_gate_attn
  model.language_model.layers.{i}.self_attn.k_proj.weight         -> w_k (keep compacted)
  model.language_model.layers.{i}.self_attn.v_proj.weight         -> w_v (keep compacted)
  model.language_model.layers.{i}.self_attn.o_proj.weight         -> w_o
  model.language_model.layers.{i}.self_attn.q_norm.weight         -> q_norm ((1+w) tiled)
  model.language_model.layers.{i}.self_attn.k_norm.weight         -> k_norm ((1+w) tiled)
  --- SwiGLU MLP (both layer types) ---
  model.language_model.layers.{i}.mlp.gate_proj.weight            -> w_gate
  model.language_model.layers.{i}.mlp.up_proj.weight              -> w_up
  model.language_model.layers.{i}.mlp.down_proj.weight            -> w_down
  model.language_model.layers.{i}.post_attention_layernorm.weight  -> post_attn_norm
  --- Final ---
  model.language_model.norm.weight                                -> final_norm
  lm_head.weight                                                  -> w_lm_head
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tensorrt_model_connect import trt_compat

from ...config import ModelConfig
from ...checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from ... import graph_ops
from ... import graph_blocks


trt = trt_compat.get_trt()

def _parse_layer_types(raw_types: list[str]) -> list[str]:
    """Normalize layer type strings to 'deltanet' or 'attention'."""
    mapping = {
        "linear": "deltanet",
        "linear_attention": "deltanet",
        "full": "attention",
        "full_attention": "attention",
    }
    return [mapping.get(t.lower(), t.lower()) for t in raw_types]


class Qwen35Plugin:
    name = "qwen3_5"
    runtime_strategy = "hybrid_mamba_attention"

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return mt in {"qwen3_5", "qwen3.5"}

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        raw = config.raw

        # Text config may be nested under text_config
        text_cfg = raw.get("text_config", raw)

        # Parse layer types
        raw_layer_types = text_cfg.get("layer_types", ["linear"] * num_layers)
        layer_types = _parse_layer_types(raw_layer_types)
        assert len(layer_types) == num_layers, (
            f"layer_types length {len(layer_types)} != num_hidden_layers {num_layers}")

        # Full attention dimensions
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        attn_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim

        # DeltaNet dimensions (from text_config linear_* fields)
        deltanet_num_heads = text_cfg.get("linear_num_value_heads", 32)
        deltanet_num_kv_heads = text_cfg.get("linear_num_key_heads", 16)
        deltanet_head_dim = text_cfg.get("linear_value_head_dim",
                                         text_cfg.get("linear_key_head_dim", 128))
        d_inner = deltanet_num_heads * deltanet_head_dim
        deltanet_qk_dim = deltanet_num_kv_heads * deltanet_head_dim
        conv_dim = deltanet_qk_dim + deltanet_qk_dim + d_inner  # Q + K + V
        d_conv = text_cfg.get("linear_conv_kernel_dim", 4)

        # MLP dimensions
        mlp_size = config.intermediate_size

        # RoPE config for full attention layers
        # rope_parameters may be nested in text_config
        rope_params = text_cfg.get("rope_parameters", {})
        partial_rotary_factor = rope_params.get(
            "partial_rotary_factor",
            text_cfg.get("partial_rotary_factor", 0.25))
        rope_theta = rope_params.get(
            "rope_theta",
            text_cfg.get("rope_theta", config.rope_theta))

        weights = WeightDict()

        # Embedding
        embed_key = "model.language_model.embed_tokens.weight"
        if not _has_tensor(readers, embed_key):
            embed_key = "model.embed_tokens.weight"
        embedding = _load_tensor(readers, embed_key)
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
        weights["embedding"] = embedding.astype(np.float32)

        deltanet_count = 0
        attn_count = 0

        for layer_idx in range(num_layers):
            lt = layer_types[layer_idx]
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.language_model.layers.{layer_idx}"

            # Input layernorm (all layer types)
            # Qwen3.5 uses (1+weight) centering in RMSNorm
            norm_key = f"{hf_prefix}.input_layernorm.weight"
            if _has_tensor(readers, norm_key):
                weights[f"{prefix}.input_norm"] = (
                    1.0 + _load_tensor(readers, norm_key).astype(np.float32))
            else:
                weights[f"{prefix}.input_norm"] = np.ones(
                    hidden, dtype=np.float32)

            # Post-attention layernorm (all layer types)
            post_norm_key = f"{hf_prefix}.post_attention_layernorm.weight"
            if _has_tensor(readers, post_norm_key):
                weights[f"{prefix}.post_attn_norm"] = (
                    1.0 + _load_tensor(readers, post_norm_key).astype(np.float32))
            else:
                weights[f"{prefix}.post_attn_norm"] = np.ones(
                    hidden, dtype=np.float32)

            if lt == "deltanet":
                self._load_deltanet_weights(
                    readers, weights, prefix, hf_prefix,
                    hidden, d_inner, conv_dim, d_conv,
                    deltanet_num_heads, deltanet_num_kv_heads,
                    deltanet_head_dim)
                deltanet_count += 1

            elif lt == "attention":
                self._load_attention_weights(
                    readers, weights, prefix, hf_prefix,
                    hidden, attn_size, kv_size,
                    num_heads, num_kv_heads, head_dim)
                attn_count += 1

            # SwiGLU MLP (all layer types)
            self._load_mlp_weights(
                readers, weights, prefix, hf_prefix,
                hidden, mlp_size)

        # Final norm (also uses (1+weight) centering)
        final_norm_key = "model.language_model.norm.weight"
        if not _has_tensor(readers, final_norm_key):
            final_norm_key = "model.norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = (
                1.0 + _load_tensor(readers, final_norm_key).astype(np.float32))
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head
        lm_head_key = "lm_head.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_lm_head"] = _transpose_2d(
                _load_tensor(readers, lm_head_key), "lm_head")
        else:
            weights["w_lm_head"] = _transpose_2d(
                embedding.copy(), "embedding_tied")

        # Metadata for engine builder
        weights["_layer_types"] = layer_types
        weights["_d_inner"] = d_inner
        weights["_d_conv"] = d_conv
        weights["_conv_dim"] = conv_dim
        weights["_deltanet_num_heads"] = deltanet_num_heads
        weights["_deltanet_num_kv_heads"] = deltanet_num_kv_heads
        weights["_deltanet_head_dim"] = deltanet_head_dim
        weights["_num_mamba_layers"] = deltanet_count
        weights["_num_attention_layers"] = attn_count
        weights["_attn_size"] = attn_size
        weights["_mlp_size"] = mlp_size
        weights["_partial_rotary_factor"] = partial_rotary_factor
        weights["_rope_theta"] = rope_theta

        return weights

    def _load_deltanet_weights(
        self, readers, weights, prefix, hf_prefix,
        hidden, d_inner, conv_dim, d_conv,
        num_heads, num_kv_heads, head_dim,
    ):
        """Load DeltaNet (linear attention) layer weights."""
        attn_prefix = f"{hf_prefix}.linear_attn"

        # in_proj_qkv (QKV combined): [conv_dim, hidden] -> transpose
        in_proj_raw = _load_tensor(readers, f"{attn_prefix}.in_proj_qkv.weight")
        weights[f"{prefix}.deltanet_in_proj_qkv"] = _transpose_2d(
            in_proj_raw, "deltanet_in_proj_qkv")

        # Gate projection (z): [d_inner, hidden] -> transpose
        z_proj_raw = _load_tensor(readers, f"{attn_prefix}.in_proj_z.weight")
        weights[f"{prefix}.deltanet_z_proj"] = _transpose_2d(
            z_proj_raw, "deltanet_z_proj")

        # Decay projection (a): [num_heads, hidden] -> transpose
        a_proj_raw = _load_tensor(readers, f"{attn_prefix}.in_proj_a.weight")
        weights[f"{prefix}.deltanet_a_proj"] = _transpose_2d(
            a_proj_raw, "deltanet_a_proj")

        # Beta projection (b): [num_heads, hidden] -> transpose
        b_proj_raw = _load_tensor(readers, f"{attn_prefix}.in_proj_b.weight")
        weights[f"{prefix}.deltanet_b_proj"] = _transpose_2d(
            b_proj_raw, "deltanet_b_proj")

        # A_log: [num_heads] -> precompute -exp(A_log)
        A_log = _load_tensor(readers, f"{attn_prefix}.A_log")
        weights[f"{prefix}.A"] = -np.exp(A_log.astype(np.float32))

        # dt_bias: [num_heads]
        dt_bias = _load_tensor(readers, f"{attn_prefix}.dt_bias")
        weights[f"{prefix}.dt_bias"] = dt_bias.astype(np.float32)

        # conv1d: [conv_dim, 1, d_conv] -> reshape to [conv_dim, d_conv]
        conv_w = _load_tensor(readers, f"{attn_prefix}.conv1d.weight")
        weights[f"{prefix}.conv1d_weight"] = conv_w.reshape(
            conv_dim, d_conv).astype(np.float32)

        conv_b_key = f"{attn_prefix}.conv1d.bias"
        if _has_tensor(readers, conv_b_key):
            weights[f"{prefix}.conv1d_bias"] = _load_tensor(
                readers, conv_b_key).astype(np.float32)
        else:
            weights[f"{prefix}.conv1d_bias"] = np.zeros(
                conv_dim, dtype=np.float32)

        # Gated RMSNorm weight: [head_dim] -> tile to [d_inner]
        norm_key = f"{attn_prefix}.norm.weight"
        if _has_tensor(readers, norm_key):
            norm_raw = _load_tensor(readers, norm_key).astype(np.float32)
            # If weight is per-head (head_dim), tile to d_inner
            if norm_raw.shape[0] == head_dim and head_dim < d_inner:
                norm_raw = np.tile(norm_raw, num_heads)
            weights[f"{prefix}.deltanet_norm"] = norm_raw
        else:
            weights[f"{prefix}.deltanet_norm"] = np.ones(
                d_inner, dtype=np.float32)

        # Output projection: [hidden, d_inner] -> transpose
        out_raw = _load_tensor(readers, f"{attn_prefix}.out_proj.weight")
        weights[f"{prefix}.deltanet_out_proj"] = _transpose_2d(
            out_raw, "deltanet_out_proj")

    def _load_attention_weights(
        self, readers, weights, prefix, hf_prefix,
        hidden, attn_size, kv_size,
        num_heads, num_kv_heads, head_dim,
    ):
        """Load full self-attention layer weights."""
        attn_prefix = f"{hf_prefix}.self_attn"

        # q_proj: [2*attn_size, hidden] -> split per head into query + gate
        # HF does: q_proj(x).view(B, seq, num_heads, 2*head_dim).chunk(2, dim=-1)
        # This interleaves: for each head, first head_dim dims are query, next are gate
        q_raw = _load_tensor(readers, f"{attn_prefix}.q_proj.weight")
        # q_raw: [num_heads * 2 * head_dim, hidden] = [8192, 4096]
        # Reshape to [num_heads, 2*head_dim, hidden], split, reshape back
        q_reshaped = q_raw.reshape(num_heads, 2 * head_dim, hidden)
        q_part = q_reshaped[:, :head_dim, :].reshape(attn_size, hidden)
        gate_part = q_reshaped[:, head_dim:, :].reshape(attn_size, hidden)
        weights[f"{prefix}.w_q"] = _transpose_2d(q_part, "q_proj")
        weights[f"{prefix}.w_gate_attn"] = _transpose_2d(gate_part, "gate_proj")

        # k_proj: [kv_size, hidden] -> keep compact
        k_raw = _load_tensor(readers, f"{attn_prefix}.k_proj.weight")
        k_t = _transpose_2d(k_raw, "k_proj")
        weights[f"{prefix}.w_k"] = k_t

        # v_proj: [kv_size, hidden] -> keep compact
        v_raw = _load_tensor(readers, f"{attn_prefix}.v_proj.weight")
        v_t = _transpose_2d(v_raw, "v_proj")
        weights[f"{prefix}.w_v"] = v_t

        # o_proj: [hidden, attn_size] -> transpose
        o_raw = _load_tensor(readers, f"{attn_prefix}.o_proj.weight")
        weights[f"{prefix}.w_o"] = _transpose_2d(o_raw, "o_proj")

        # QK-norm with (1+weight) centering, tiled to num_heads
        q_norm_key = f"{attn_prefix}.q_norm.weight"
        if _has_tensor(readers, q_norm_key):
            q_norm_raw = _load_tensor(readers, q_norm_key).astype(np.float32)
            q_norm_centered = 1.0 + q_norm_raw  # (1+weight) centering
            weights[f"{prefix}.q_norm"] = np.tile(
                q_norm_centered, num_heads)
        k_norm_key = f"{attn_prefix}.k_norm.weight"
        if _has_tensor(readers, k_norm_key):
            k_norm_raw = _load_tensor(readers, k_norm_key).astype(np.float32)
            k_norm_centered = 1.0 + k_norm_raw
            weights[f"{prefix}.k_norm"] = np.tile(
                k_norm_centered, num_kv_heads)

    def _load_mlp_weights(
        self, readers, weights, prefix, hf_prefix,
        hidden, mlp_size,
    ):
        """Load SwiGLU MLP weights."""
        gate_key = f"{hf_prefix}.mlp.gate_proj.weight"
        up_key = f"{hf_prefix}.mlp.up_proj.weight"
        down_key = f"{hf_prefix}.mlp.down_proj.weight"

        if _has_tensor(readers, gate_key):
            weights[f"{prefix}.w_gate"] = _transpose_2d(
                _load_tensor(readers, gate_key), "gate_proj")
            weights[f"{prefix}.w_up"] = _transpose_2d(
                _load_tensor(readers, up_key), "up_proj")
            weights[f"{prefix}.w_down"] = _transpose_2d(
                _load_tensor(readers, down_key), "down_proj")

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        debug_layer_outputs: bool = False,
    ) -> bytes:
        """Build hybrid TRT engine with DeltaNet + attention layers."""
        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers

        layer_types: list[str] = weights["_layer_types"]
        d_inner: int = weights["_d_inner"]
        d_conv: int = weights["_d_conv"]
        conv_dim: int = weights["_conv_dim"]
        deltanet_num_heads: int = weights["_deltanet_num_heads"]
        deltanet_num_kv_heads: int = weights["_deltanet_num_kv_heads"]
        deltanet_head_dim: int = weights["_deltanet_head_dim"]
        num_mamba: int = weights["_num_mamba_layers"]
        num_attn: int = weights["_num_attention_layers"]
        attn_size: int = weights["_attn_size"]
        mlp_size: int = weights["_mlp_size"]
        partial_rotary_factor: float = weights["_partial_rotary_factor"]

        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = attn_size // num_heads
        kv_attention_size = graph_blocks.infer_kv_attention_size(
            weights, num_kv_heads=num_kv_heads, head_dim=head_dim)
        attention_window = max_cache_length + 1

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        # --- Inputs ---
        token_id = network.add_input("token_id", trt.int32, (1,))
        position_id = network.add_input("position_id", trt.int32, (1,))
        attention_mask = network.add_input(
            "attention_mask", trt.float32, (1, attention_window))

        # DeltaNet state inputs (conv + ssm per DeltaNet layer)
        conv_state_inputs = []
        ssm_state_inputs = []
        for mi in range(num_mamba):
            cs = network.add_input(
                graph_ops.layer_tensor_name("conv_state", mi),
                trt.float32, (conv_dim, d_conv))
            ss = network.add_input(
                graph_ops.layer_tensor_name("ssm_state", mi),
                trt.float32, (deltanet_num_heads, deltanet_head_dim, deltanet_head_dim))
            conv_state_inputs.append(cs)
            ssm_state_inputs.append(ss)

        # Attention KV cache inputs
        cache_k_inputs = []
        cache_v_inputs = []
        for ai in range(num_attn):
            ck = network.add_input(
                graph_ops.layer_tensor_name("cache_k", ai),
                trt.float32, (max_cache_length, kv_attention_size))
            cv = network.add_input(
                graph_ops.layer_tensor_name("cache_v", ai),
                trt.float32, (max_cache_length, kv_attention_size))
            cache_k_inputs.append(ck)
            cache_v_inputs.append(cv)

        # --- Shared constants ---
        embedding_table = graph_ops.add_constant(
            network, (vocab, hidden), weights["embedding"])
        eps_tensor = graph_ops.add_constant(
            network, (1, 1),
            np.array([config.rms_norm_eps], dtype=np.float32))

        rope_theta: float = weights["_rope_theta"]
        rotary_embedding_dim = int(head_dim * partial_rotary_factor)

        # RoPE tables for full attention layers (partial rotary)
        cos_half = graph_ops.make_rope_table_half_dim(
            attention_window, head_dim, rope_theta,
            cosine=True, partial_rotary_factor=partial_rotary_factor)
        sin_half = graph_ops.make_rope_table_half_dim(
            attention_window, head_dim, rope_theta,
            cosine=False, partial_rotary_factor=partial_rotary_factor)

        cos_half_tensor = graph_ops.add_constant(
            network, cos_half.shape, cos_half)
        sin_half_tensor = graph_ops.add_constant(
            network, sin_half.shape, sin_half)

        # --- Embedding ---
        gather = network.add_gather(embedding_table, token_id, 0)
        hidden_state = gather.get_output(0)

        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, "debug_embed")

        # --- Layer stack ---
        present_conv_outputs = []
        present_ssm_outputs = []
        present_k_outputs = []
        present_v_outputs = []
        mamba_counter = 0
        attn_counter = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            lt = layer_types[layer_idx]

            if lt == "deltanet":
                result = _add_deltanet_layer(
                    network=network,
                    hidden=hidden_state,
                    conv_state_in=conv_state_inputs[mamba_counter],
                    ssm_state_in=ssm_state_inputs[mamba_counter],
                    eps_tensor=eps_tensor,
                    weights=weights,
                    prefix=prefix,
                    hidden_size=hidden,
                    d_inner=d_inner,
                    d_conv=d_conv,
                    conv_dim=conv_dim,
                    num_heads=deltanet_num_heads,
                    num_kv_heads=deltanet_num_kv_heads,
                    head_dim=deltanet_head_dim,
                    mlp_size=mlp_size,
                )
                hidden_state = result["hidden"]
                present_conv_outputs.append(result["present_conv"])
                present_ssm_outputs.append(result["present_ssm"])
                mamba_counter += 1

            elif lt == "attention":
                result = _add_full_attention_layer(
                    network=network,
                    hidden=hidden_state,
                    cache_k=cache_k_inputs[attn_counter],
                    cache_v=cache_v_inputs[attn_counter],
                    attention_mask=attention_mask,
                    position_id=position_id,
                    cos_half_tensor=cos_half_tensor,
                    sin_half_tensor=sin_half_tensor,
                    eps_tensor=eps_tensor,
                    weights=weights,
                    prefix=prefix,
                    hidden_size=hidden,
                    attn_size=attn_size,
                    kv_attention_size=kv_attention_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    rotary_embedding_dim=rotary_embedding_dim,
                    max_cache_length=max_cache_length,
                    mlp_size=mlp_size,
                )
                hidden_state = result["hidden"]
                present_k_outputs.append(result["present_k"])
                present_v_outputs.append(result["present_v"])
                attn_counter += 1

            if debug_layer_outputs:
                _mark_debug_output(
                    network, hidden_state, f"debug_hidden_{layer_idx}")

        # --- Final norm ---
        final_norm = weights.get("final_norm")
        if final_norm is not None and len(final_norm) > 0:
            hidden_state = graph_ops.add_rms_norm(
                network, hidden_state, hidden, final_norm, eps_tensor)

        # --- LM head ---
        logits = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden, vocab, weights["w_lm_head"])
        b_out = np.zeros(vocab, dtype=np.float32)
        logits = graph_ops.add_bias_sum(network, logits, vocab, b_out)
        logits.name = "logits"
        network.mark_output(logits)

        # --- Present state outputs ---
        for mi in range(num_mamba):
            pc = present_conv_outputs[mi]
            ps = present_ssm_outputs[mi]
            pc.name = graph_ops.layer_tensor_name("present_conv", mi)
            ps.name = graph_ops.layer_tensor_name("present_ssm", mi)
            network.mark_output(pc)
            network.mark_output(ps)

        for ai in range(num_attn):
            pk = present_k_outputs[ai]
            pv = present_v_outputs[ai]
            pk.name = graph_ops.layer_tensor_name("present_k", ai)
            pv.name = graph_ops.layer_tensor_name("present_v", ai)
            network.mark_output(pk)
            network.mark_output(pv)

        # --- Build ---
        if verbose:
            print(f"[trtmc-build] Building Qwen3.5 hybrid TRT engine "
                  f"({num_layers} layers: {num_mamba} deltanet + "
                  f"{num_attn} attention, "
                  f"hidden={hidden}, d_inner={d_inner}, "
                  f"nheads_dn={deltanet_num_heads}, "
                  f"head_dim_dn={deltanet_head_dim}, "
                  f"cache={max_cache_length}) ...",
                  file=sys.stderr)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed")

        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        """Inject hybrid-specific config fields into the bundle."""
        raw = config.raw
        text_cfg = raw.get("text_config", raw)

        raw_layer_types = text_cfg.get("layer_types", [])
        layer_types = _parse_layer_types(raw_layer_types)

        deltanet_num_heads = text_cfg.get("linear_num_value_heads", 32)
        deltanet_head_dim = text_cfg.get("linear_value_head_dim",
                                         text_cfg.get("linear_key_head_dim", 128))
        deltanet_num_kv_heads = text_cfg.get("linear_num_key_heads", 16)
        d_inner = deltanet_num_heads * deltanet_head_dim
        d_conv = text_cfg.get("linear_conv_kernel_dim", 4)
        deltanet_qk_dim = deltanet_num_kv_heads * deltanet_head_dim
        conv_dim = deltanet_qk_dim + deltanet_qk_dim + d_inner

        num_mamba = sum(1 for lt in layer_types if lt == "deltanet")
        num_attn = sum(1 for lt in layer_types if lt == "attention")

        return {
            "layer_types": layer_types,
            "num_mamba_layers": num_mamba,
            "num_attention_layers": num_attn,
            "d_inner": d_inner,
            "mamba_d_state": deltanet_head_dim,
            "mamba_d_conv": d_conv,
            "mamba_nheads": deltanet_num_heads,
            "mamba_head_dim": deltanet_head_dim,
            "conv_dim": conv_dim,
        }


def _mark_debug_output(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    name: str,
) -> None:
    identity = network.add_identity(tensor)
    cast = network.add_cast(identity.get_output(0), trt.float32)
    out = cast.get_output(0)
    out.name = name
    network.mark_output(out)


def _add_deltanet_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    conv_state_in: trt.ITensor,
    ssm_state_in: trt.ITensor,
    eps_tensor: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    d_inner: int,
    d_conv: int,
    conv_dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    mlp_size: int,
) -> dict[str, trt.ITensor]:
    """Add one Gated DeltaNet layer (single-step decode).

    DeltaNet uses delta-rule linear attention with:
      - Conv1d on QKV projection output
      - L2-normalized Q,K with Compact GQA/MQA K/V
      - Per-head decay (A_log + softplus(a + dt_bias))
      - Per-head beta (write strength via sigmoid)
      - Delta rule state update: S' = decay*S + outer(k, (v - S@k)*beta)
      - Gated output: norm(S'@q) * silu(z) * norm_weight

    Returns: {hidden, present_conv, present_ssm}
    """
    qk_dim = num_kv_heads * head_dim  # Q and K dimension before Compact GQA/MQA K/V

    # ===== 1. RMSNorm =====
    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size,
        weights[f"{prefix}.input_norm"], eps_tensor)

    # ===== 2. Input projections =====
    # QKV combined: [1, hidden] -> [1, conv_dim]
    qkv = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, conv_dim,
        weights[f"{prefix}.deltanet_in_proj_qkv"])

    # Gate (z): [1, hidden] -> [1, d_inner]
    z = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, d_inner,
        weights[f"{prefix}.deltanet_z_proj"])

    # Decay projection (a): [1, hidden] -> [1, num_heads]
    a_raw = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, num_heads,
        weights[f"{prefix}.deltanet_a_proj"])

    # Beta projection (b): [1, hidden] -> [1, num_heads]
    b_raw = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, num_heads,
        weights[f"{prefix}.deltanet_b_proj"])

    # ===== 3. Conv1d step on QKV =====
    # conv_state_in: [conv_dim, d_conv]
    # qkv: [1, conv_dim] -> [conv_dim, 1]
    qkv_col = network.add_shuffle(qkv)
    qkv_col.reshape_dims = (conv_dim, 1)

    if d_conv > 1:
        slice_layer = network.add_slice(
            conv_state_in,
            start=(0, 1),
            shape=(conv_dim, d_conv - 1),
            stride=(1, 1))
        new_conv_state = network.add_concatenation(
            [slice_layer.get_output(0), qkv_col.get_output(0)])
        new_conv_state.axis = 1
        present_conv = new_conv_state.get_output(0)
    else:
        present_conv = qkv_col.get_output(0)

    conv_w = graph_ops.add_constant(
        network, (conv_dim, d_conv), weights[f"{prefix}.conv1d_weight"])
    conv_prod = network.add_elementwise(
        present_conv, conv_w, trt.ElementWiseOperation.PROD)
    conv_sum = network.add_reduce(
        conv_prod.get_output(0), trt.ReduceOperation.SUM,
        1 << 1, keep_dims=True)
    conv_flat = network.add_shuffle(conv_sum.get_output(0))
    conv_flat.reshape_dims = (1, conv_dim)
    conv_out = graph_ops.add_bias_sum(
        network, conv_flat.get_output(0), conv_dim,
        weights[f"{prefix}.conv1d_bias"])
    qkv_activated = graph_ops.add_activation(network, conv_out, "silu")

    # ===== 4. Split Q, K, V from activated output =====
    offset = 0
    q_slice = network.add_slice(
        qkv_activated, start=(0, offset), shape=(1, qk_dim), stride=(1, 1))
    q_raw_t = q_slice.get_output(0)
    offset += qk_dim

    k_slice = network.add_slice(
        qkv_activated, start=(0, offset), shape=(1, qk_dim), stride=(1, 1))
    k_raw_t = k_slice.get_output(0)
    offset += qk_dim

    v_slice = network.add_slice(
        qkv_activated, start=(0, offset), shape=(1, d_inner), stride=(1, 1))
    v_raw = v_slice.get_output(0)

    # ===== 5. L2-normalize Q and K =====
    # Reshape to [num_kv_heads, head_dim], normalize per-head, reshape back
    q_heads_in = network.add_shuffle(q_raw_t)
    q_heads_in.reshape_dims = (num_kv_heads, head_dim)
    q_normed = graph_ops.add_l2_norm(network, q_heads_in.get_output(0), 1, eps=1e-6)

    k_heads_in = network.add_shuffle(k_raw_t)
    k_heads_in.reshape_dims = (num_kv_heads, head_dim)
    k_normed = graph_ops.add_l2_norm(network, k_heads_in.get_output(0), 1, eps=1e-6)

    # ===== 6. keep compact Q,K from num_kv_heads -> num_heads =====
    heads_per_group = num_heads // num_kv_heads

    if heads_per_group > 1:
        # Q: [num_kv_heads, head_dim] -> [num_kv_heads, 1, head_dim] ->
        #    tile -> [num_kv_heads, heads_per_group, head_dim] ->
        #    [num_heads, head_dim]
        q_3d = network.add_shuffle(q_normed)
        q_3d.reshape_dims = (num_kv_heads, 1, head_dim)
        tile_ones = graph_ops.add_constant(
            network, (1, heads_per_group, 1),
            np.ones((1, heads_per_group, 1), dtype=np.float32))
        q_tiled = network.add_elementwise(
            q_3d.get_output(0), tile_ones, trt.ElementWiseOperation.PROD)
        q_expanded_s = network.add_shuffle(q_tiled.get_output(0))
        q_expanded_s.reshape_dims = (num_heads, head_dim)
        q_expanded = q_expanded_s.get_output(0)

        k_3d = network.add_shuffle(k_normed)
        k_3d.reshape_dims = (num_kv_heads, 1, head_dim)
        k_tiled = network.add_elementwise(
            k_3d.get_output(0), tile_ones, trt.ElementWiseOperation.PROD)
        k_t_s = network.add_shuffle(k_tiled.get_output(0))
        k_t_s.reshape_dims = (num_heads, head_dim)
        k_t = k_t_s.get_output(0)
    else:
        q_expanded = q_normed
        k_t = k_normed

    # V: [1, d_inner] -> [num_heads, head_dim]
    v_heads = network.add_shuffle(v_raw)
    v_heads.reshape_dims = (num_heads, head_dim)
    v_t = v_heads.get_output(0)

    # ===== 7. Compute decay: -exp(A_log) * softplus(a + dt_bias) per head =====
    # A: [num_heads] (precomputed as -exp(A_log))
    A_const = graph_ops.add_constant(
        network, (1, num_heads), weights[f"{prefix}.A"])

    # dt_bias: [num_heads]
    dt_bias_const = graph_ops.add_constant(
        network, (1, num_heads), weights[f"{prefix}.dt_bias"])
    a_biased = network.add_elementwise(
        a_raw, dt_bias_const, trt.ElementWiseOperation.SUM)

    # softplus(a + dt_bias): log(1 + exp(x))
    a_exp = network.add_unary(a_biased.get_output(0), trt.UnaryOperation.EXP)
    one = graph_ops.add_constant(
        network, (1, 1), np.array([1.0], dtype=np.float32))
    a_exp_p1 = network.add_elementwise(
        a_exp.get_output(0), one, trt.ElementWiseOperation.SUM)
    a_softplus = network.add_unary(
        a_exp_p1.get_output(0), trt.UnaryOperation.LOG)

    # decay = A * softplus(...) per head: [1, num_heads]
    decay_flat = network.add_elementwise(
        A_const, a_softplus.get_output(0), trt.ElementWiseOperation.PROD)
    # exp(decay) for the state update: [1, num_heads] -> [num_heads, 1, 1]
    decay_reshaped = network.add_shuffle(decay_flat.get_output(0))
    decay_reshaped.reshape_dims = (num_heads, 1, 1)
    decay_exp = network.add_unary(
        decay_reshaped.get_output(0), trt.UnaryOperation.EXP)

    # ===== 8. Compute beta: sigmoid(b) per head =====
    # b_raw: [1, num_heads]
    beta = network.add_activation(b_raw, trt.ActivationType.SIGMOID)
    # [1, num_heads] -> [num_heads, 1]
    beta_reshaped = network.add_shuffle(beta.get_output(0))
    beta_reshaped.reshape_dims = (num_heads, 1)

    # ===== 9. Delta rule state update =====
    # HF state layout: [H, K_dim, V_dim]
    # ssm_state_in: [num_heads, head_dim, head_dim]  (K on axis -2, V on axis -1)
    # k: [num_heads, head_dim], q: [num_heads, head_dim], v: [num_heads, head_dim]

    # 9a. Decay state first: state = state * exp(g)
    decayed_state = network.add_elementwise(
        decay_exp.get_output(0), ssm_state_in,
        trt.ElementWiseOperation.PROD)

    # 9b. kv_mem = state^T @ k: read old value for this key
    # [H, V, K] @ [H, K, 1] = [H, V, 1]  (transpose state to swap K/V axes)
    k_col = network.add_shuffle(k_t)
    k_col.reshape_dims = (num_heads, head_dim, 1)
    kv_old_3d = network.add_matrix_multiply(
        decayed_state.get_output(0), trt.MatrixOperation.TRANSPOSE,
        k_col.get_output(0), trt.MatrixOperation.NONE)
    kv_old = network.add_shuffle(kv_old_3d.get_output(0))
    kv_old.reshape_dims = (num_heads, head_dim)

    # 9c. delta = (v - kv_mem) * beta
    v_minus_old = network.add_elementwise(
        v_t, kv_old.get_output(0), trt.ElementWiseOperation.SUB)
    v_delta = network.add_elementwise(
        v_minus_old.get_output(0), beta_reshaped.get_output(0),
        trt.ElementWiseOperation.PROD)

    # 9d. state_new = decayed_state + outer(k, delta)
    # outer: k[:, :, None] * delta[:, None, :] = [H, K, 1] @ [H, 1, V] = [H, K, V]
    k_col2 = network.add_shuffle(k_t)
    k_col2.reshape_dims = (num_heads, head_dim, 1)
    v_delta_row = network.add_shuffle(v_delta.get_output(0))
    v_delta_row.reshape_dims = (num_heads, 1, head_dim)
    outer_prod = network.add_matrix_multiply(
        k_col2.get_output(0), trt.MatrixOperation.NONE,
        v_delta_row.get_output(0), trt.MatrixOperation.NONE)

    new_state = network.add_elementwise(
        decayed_state.get_output(0), outer_prod.get_output(0),
        trt.ElementWiseOperation.SUM)
    present_ssm = new_state.get_output(0)

    # 9e. output = state_new^T @ (q * scale)
    # HF applies: query *= 1/sqrt(k_dim)
    q_scale = graph_ops.add_constant(
        network, (1, 1),
        np.array([1.0 / np.sqrt(head_dim)], dtype=np.float32))
    q_scaled = network.add_elementwise(
        q_expanded, q_scale, trt.ElementWiseOperation.PROD)
    # [H, V, K] @ [H, K, 1] = [H, V, 1]
    q_col = network.add_shuffle(q_scaled.get_output(0))
    q_col.reshape_dims = (num_heads, head_dim, 1)
    output_3d = network.add_matrix_multiply(
        present_ssm, trt.MatrixOperation.TRANSPOSE,
        q_col.get_output(0), trt.MatrixOperation.NONE)
    output_flat = network.add_shuffle(output_3d.get_output(0))
    output_flat.reshape_dims = (1, d_inner)

    # ===== 10. Gated RMSNorm per-head: weight * norm(output) * silu(z) =====
    # HF norm operates per head_v_dim: reshape to [num_heads, head_dim], norm, reshape back
    deltanet_norm_w = weights[f"{prefix}.deltanet_norm"]
    # Use same eps as HF Qwen3_5RMSNormGated (config.rms_norm_eps = 1e-6)
    eps_small = graph_ops.add_constant(
        network, (1, 1),
        np.array([1e-6], dtype=np.float32))

    # Reshape output and z to [num_heads, head_dim] for per-head norm
    output_heads = network.add_shuffle(output_flat.get_output(0))
    output_heads.reshape_dims = (num_heads, head_dim)

    # Per-head RMSNorm: norm each head independently
    sq = network.add_elementwise(
        output_heads.get_output(0), output_heads.get_output(0),
        trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom_in = network.add_elementwise(
        mean.get_output(0), eps_small, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        output_heads.get_output(0), recip.get_output(0),
        trt.ElementWiseOperation.PROD)

    # Reshape back and apply weight
    norm_flat = network.add_shuffle(normalized.get_output(0))
    norm_flat.reshape_dims = (1, d_inner)
    gamma_t = graph_ops.add_constant(network, (1, d_inner), deltanet_norm_w)
    normed_output = network.add_elementwise(
        norm_flat.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)

    # Gate: multiply by silu(z)
    z_activated = graph_ops.add_activation(network, z, "silu")
    gated = network.add_elementwise(
        normed_output.get_output(0), z_activated,
        trt.ElementWiseOperation.PROD)

    # ===== 11. Output projection + residual =====
    out = graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), d_inner, hidden_size,
        weights[f"{prefix}.deltanet_out_proj"])

    residual = network.add_elementwise(
        hidden, out, trt.ElementWiseOperation.SUM)
    hidden_after_attn = residual.get_output(0)

    # ===== 12. Post-attention norm + SwiGLU MLP + residual =====
    post_normed = graph_ops.add_rms_norm(
        network, hidden_after_attn, hidden_size,
        weights[f"{prefix}.post_attn_norm"], eps_tensor)

    mlp_out = graph_blocks.add_swiglu_mlp(
        network, post_normed,
        weights=weights,
        prefix=prefix,
        hidden_size=hidden_size,
        mlp_size=mlp_size,
    )

    mlp_residual = network.add_elementwise(
        hidden_after_attn, mlp_out,
        trt.ElementWiseOperation.SUM)

    return {
        "hidden": mlp_residual.get_output(0),
        "present_conv": present_conv,
        "present_ssm": present_ssm,
    }


def _add_full_attention_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    eps_tensor: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    attn_size: int,
    kv_attention_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_embedding_dim: int,
    max_cache_length: int,
    mlp_size: int,
) -> dict[str, trt.ITensor]:
    """Add one full self-attention layer with output gating.

    Qwen3.5 full attention has:
      - QK-norm with (1+weight) centering
      - Partial RoPE (25% of dims)
      - Output gating: context * sigmoid(gate) BEFORE o_proj
      - SwiGLU MLP after attention

    Returns: {hidden, present_k, present_v}
    """
    attention_window = max_cache_length + 1

    # Pre-attention norm
    normed = graph_blocks.apply_norm(
        network, hidden, hidden_size,
        weights[f"{prefix}.input_norm"],
        weights.get(f"{prefix}.input_norm_beta"),
        eps_tensor, "rmsnorm")

    # QKV projections
    q = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, attn_size,
        weights[f"{prefix}.w_q"])
    k = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, kv_attention_size,
        weights[f"{prefix}.w_k"])
    v = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, kv_attention_size,
        weights[f"{prefix}.w_v"])

    # Per-head QK norm
    q_norm = weights.get(f"{prefix}.q_norm")
    if q_norm is not None:
        q = graph_ops.add_rms_norm_per_head(
            network, q, num_heads, head_dim, q_norm, eps_tensor)
    k_norm = weights.get(f"{prefix}.k_norm")
    if k_norm is not None:
        k = graph_ops.add_rms_norm_per_head(
            network, k, num_kv_heads, head_dim, k_norm, eps_tensor)

    # Native RoPE
    q = graph_ops.add_apply_rope_native(
        network, q, num_heads, head_dim, cos_half_tensor, sin_half_tensor,
        position_id, rotary_embedding_dim)
    k = graph_ops.add_apply_rope_native(
        network, k, num_kv_heads, head_dim, cos_half_tensor, sin_half_tensor,
        position_id, rotary_embedding_dim)

    # Save present K/V
    present_k = k
    present_v = v

    # Reshape K, V for concatenation
    k_reshape = network.add_shuffle(k)
    k_reshape.reshape_dims = (1, kv_attention_size)
    v_reshape = network.add_shuffle(v)
    v_reshape.reshape_dims = (1, kv_attention_size)

    # Concatenate with cache
    all_k = network.add_concatenation(
        [cache_k, k_reshape.get_output(0)])
    all_k.axis = 0
    all_v = network.add_concatenation(
        [cache_v, v_reshape.get_output(0)])
    all_v.axis = 0

    mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask)
    context_flat = graph_ops.add_attention_from_rows(
        network, q, all_k.get_output(0), all_v.get_output(0),
        num_heads=num_heads, head_dim=head_dim, num_kv_heads=num_kv_heads,
        q_seq=1, kv_seq=attention_window,
        mask=mask_4d)

    # Gate: applied BEFORE o_proj (HF order)
    gate_attn_w = weights.get(f"{prefix}.w_gate_attn")
    attn_out = context_flat
    if gate_attn_w is not None:
        gate = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, attn_size, gate_attn_w)
        gate_sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
        gated = network.add_elementwise(
            attn_out, gate_sigmoid.get_output(0),
            trt.ElementWiseOperation.PROD)
        attn_out = gated.get_output(0)

    # Output projection (AFTER gate)
    attn_out = graph_ops.add_matmul_rhs_constant(
        network, attn_out, attn_size, hidden_size,
        weights[f"{prefix}.w_o"])

    # Residual after attention
    residual = network.add_elementwise(
        hidden, attn_out, trt.ElementWiseOperation.SUM)
    hidden_after_attn = residual.get_output(0)

    # Post-attention norm + SwiGLU MLP + residual
    post_normed = graph_ops.add_rms_norm(
        network, hidden_after_attn, hidden_size,
        weights[f"{prefix}.post_attn_norm"], eps_tensor)

    mlp_out = graph_blocks.add_swiglu_mlp(
        network, post_normed,
        weights=weights,
        prefix=prefix,
        hidden_size=hidden_size,
        mlp_size=mlp_size,
    )

    mlp_residual = network.add_elementwise(
        hidden_after_attn, mlp_out,
        trt.ElementWiseOperation.SUM)

    return {
        "hidden": mlp_residual.get_output(0),
        "present_k": present_k,
        "present_v": present_v,
    }


plugin = Qwen35Plugin()
