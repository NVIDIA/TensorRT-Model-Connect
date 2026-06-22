"""Eagle VLM family plugin — embedding and reranking models.

Supports NVIDIA Llama-Nemotron embed-vl-1b-v2 and rerank-vl-1b-v2:
  - Text backbone: Llama 3.2 1B (16 layers, 2048 hidden, GQA 32/8)
  - Vision encoder: SigLIP-2 (27 layers, 1152 hidden, 16 heads)
  - Embedding mode: encode text/image -> mean pool -> L2 normalize -> float vector
  - Reranking mode: cross-encode query+doc -> relevance score

Detection:
  - model_type starts with "llama_nemotron_vl"
  - Architectures: LlamaNemotronVLModel, LlamaNemotronVLForSequenceClassification
  - Reranker detected from: model_type contains "rerank", or
    architectures contain "SequenceClassification"

Weight prefix: language_model.* (not model.layers.*)
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)

if TYPE_CHECKING:
    pass


def _is_reranker(config: ModelConfig) -> bool:
    """Detect reranking mode from config."""
    if config.raw.get("is_reranker", False):
        return True
    mt = config.model_type.lower()
    if "rerank" in mt:
        return True
    archs = config.raw.get("architectures", [])
    for arch in archs:
        if "rerank" in arch.lower() or "sequenceclassification" in arch.lower():
            return True
    return False


class EagleVLMPlugin:
    name = "eagle_vlm"

    @property
    def runtime_strategy(self):
        # Set dynamically based on config during build; default to embedding
        return getattr(self, '_runtime_strategy', 'embedding')

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return mt.startswith("llama_nemotron_vl")

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        return _load_eagle_weights(model_dir, config)

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False, parallel_config=None,
    ) -> bytes:
        is_rerank = _is_reranker(config)
        # Store for runtime_strategy property
        self._runtime_strategy = "reranking" if is_rerank else "embedding"
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="Eagle VLM tensor-parallel builds")
            from .tp_builder import build_eagle_vlm_tp_engine
            return build_eagle_vlm_tp_engine(
                config, weights, max_cache_length,
                is_reranker=is_rerank,
                verbose=verbose,
                parallel_config=parallel)
        return _build_eagle_engine(
            config, weights, max_cache_length,
            is_reranker=is_rerank, verbose=verbose)

    def build_vision_engine(
        self, model_dir: str, config: ModelConfig, weights: WeightDict,
        *, precision: str = "fp32", verbose: bool = False,
    ) -> bytes | None:
        vision_config = config.raw.get("vision_config")
        if vision_config is None:
            return None
        vision_weights = _load_vision_weights(model_dir, config)
        return _build_siglip_vision_engine(
            vision_config, vision_weights, verbose=verbose)

    def get_vl_config(self, config: ModelConfig) -> dict | None:
        """Return embedding/reranking config for the bundle."""
        is_rerank = _is_reranker(config)
        vc = config.raw.get("vision_config", {})
        image_size = vc.get("image_size", 384)
        patch_size = vc.get("patch_size", 14)

        # After 2x2 pixel_shuffle merge, the num_vision_tokens is reduced by 4
        merge_size = 2
        num_merged = (image_size // patch_size // merge_size) ** 2

        cfg = {
            "is_reranker": is_rerank,
            "embedding_dim": config.hidden_size,
            "vision_image_size": image_size,
            "vision_patch_size": patch_size,
            "num_vision_tokens": num_merged,
            "vision_output_dim": config.hidden_size,  # MLP projector output = text hidden
            "preprocessor_type": "simple_chw",
        }
        return cfg

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict | None:
        is_rerank = _is_reranker(config)
        return {
            "runtime_strategy": "reranking" if is_rerank else "embedding",
        }


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _load_eagle_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    """Load Eagle text backbone weights (Llama architecture)."""
    from pathlib import Path

    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    kv_attention_size = num_kv_heads * config.head_dim

    weights = WeightDict()

    # Embedding — detect prefix
    embed_key = "language_model.embed_tokens.weight"
    if not _has_tensor(readers, embed_key):
        embed_key = "model.language_model.embed_tokens.weight"
    if not _has_tensor(readers, embed_key):
        embed_key = "language_model.model.embed_tokens.weight"
    if not _has_tensor(readers, embed_key):
        embed_key = "model.embed_tokens.weight"
    embedding = _load_tensor(readers, embed_key)
    assert embedding.shape == (vocab, hidden), (
        f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
    weights["embedding"] = embedding.astype(np.float32)

    # Detect layer prefix
    if _has_tensor(readers, "language_model.layers.0.input_layernorm.weight"):
        layer_prefix = "language_model.layers"
    elif _has_tensor(readers, "model.language_model.layers.0.input_layernorm.weight"):
        layer_prefix = "model.language_model.layers"
    elif _has_tensor(readers, "language_model.model.layers.0.input_layernorm.weight"):
        layer_prefix = "language_model.model.layers"
    elif _has_tensor(readers, "model.layers.0.input_layernorm.weight"):
        layer_prefix = "model.layers"
    else:
        raise RuntimeError("Cannot find text decoder layer weights")

    attention_size = 0
    mlp_size = 0

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        hf_prefix = f"{layer_prefix}.{layer_idx}"

        # Norms
        input_norm = _load_tensor(readers, f"{hf_prefix}.input_layernorm.weight")
        post_norm = _load_tensor(readers, f"{hf_prefix}.post_attention_layernorm.weight")
        weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32)
        weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

        # Q/K/V/O projections
        q_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.q_proj.weight")
        k_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.k_proj.weight")
        v_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.v_proj.weight")
        o_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.o_proj.weight")

        q_hidden = q_raw.shape[0]
        if attention_size == 0:
            attention_size = q_hidden

        q_t = _transpose_2d(q_raw, "q_proj")
        k_t = _transpose_2d(k_raw, "k_proj")
        v_t = _transpose_2d(v_raw, "v_proj")
        o_t = _transpose_2d(o_raw, "o_proj")


        weights[f"{prefix}.w_q"] = q_t
        weights[f"{prefix}.w_k"] = k_t
        weights[f"{prefix}.w_v"] = v_t
        weights[f"{prefix}.w_o"] = o_t

        # SwiGLU MLP
        gate_raw = _load_tensor(readers, f"{hf_prefix}.mlp.gate_proj.weight")
        up_raw = _load_tensor(readers, f"{hf_prefix}.mlp.up_proj.weight")
        down_raw = _load_tensor(readers, f"{hf_prefix}.mlp.down_proj.weight")

        if mlp_size == 0:
            mlp_size = gate_raw.shape[0]

        weights[f"{prefix}.w_gate"] = _transpose_2d(gate_raw, "gate")
        weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up")
        weights[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down")

    # Final norm — derive from layer_prefix
    final_norm_key = f"{layer_prefix.rsplit('.layers', 1)[0]}.norm.weight"
    if not _has_tensor(readers, final_norm_key):
        final_norm_key = "language_model.model.norm.weight"
    if not _has_tensor(readers, final_norm_key):
        final_norm_key = "model.norm.weight"
    if _has_tensor(readers, final_norm_key):
        weights["final_norm"] = _load_tensor(readers, final_norm_key).astype(np.float32)

    # Pooling head / score head (if present)
    # Eagle embedding uses the last hidden state with mean pooling + L2 norm
    # Eagle reranking uses a score head: linear projection from hidden -> 1
    for key_name in ("score.weight", "model.score.weight",
                     "classifier.weight", "language_model.score.weight"):
        if _has_tensor(readers, key_name):
            score_w = _load_tensor(readers, key_name)
            weights["score_weight"] = _transpose_2d(score_w, "score")
            break

    for key_name in ("score.bias", "model.score.bias",
                     "classifier.bias", "language_model.score.bias"):
        if _has_tensor(readers, key_name):
            weights["score_bias"] = _load_tensor(readers, key_name).astype(np.float32)
            break

    # No LM head needed for embedding/reranking (no token generation)
    # But we need w_out as a placeholder for the standard builder check
    weights["w_out"] = np.zeros((hidden, 1), dtype=np.float32)

    weights["_attention_size"] = attention_size
    weights["_kv_attention_size"] = kv_attention_size
    weights["_mlp_size"] = mlp_size

    return weights


def _load_vision_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    """Load SigLIP-2 vision encoder weights."""
    from pathlib import Path

    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    weights = WeightDict()
    for reader in readers:
        for key in reader.keys():
            if key.startswith("vision_model.") or key.startswith("visual."):
                weights[key] = _load_tensor([reader], key)
            elif key.startswith("model.vision_model."):
                # Rerank model uses model.vision_model.* prefix — strip model.
                canon = key[len("model."):]
                weights[canon] = _load_tensor([reader], key)
            elif key.startswith("vision_tower."):
                canon = key.replace("vision_tower.", "vision_model.")
                weights[canon] = _load_tensor([reader], key)
            elif key.startswith("multi_modal_projector."):
                weights[key] = _load_tensor([reader], key)
            elif key.startswith("mlp1.") or key.startswith("model.mlp1."):
                canon = key.replace("model.", "", 1) if key.startswith("model.mlp1.") else key
                weights[canon] = _load_tensor([reader], key)

    return weights


# ---------------------------------------------------------------------------
# TRT engine builders
# ---------------------------------------------------------------------------

def _build_eagle_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    is_reranker: bool = False,
    verbose: bool = False,
) -> bytes:
    """Build Eagle text backbone engine for embedding or reranking.

    For embedding: output is [1, hidden] (last hidden state at position 0,
    to be mean-pooled and L2-normalized in C++).

    For reranking: output is [1, 1] relevance score (via score head).

    Unlike the standard decoder, this is a SINGLE forward pass encoder
    (no autoregressive loop, no KV cache).
    """
    from tensorrt_model_connect import trt_compat
    trt = trt_compat.get_trt()
    from . import graph_ops
    from . import graph_blocks

    attention_size: int = weights.get("_attention_size", config.attention_size)
    mlp_size: int = weights.get("_mlp_size", config.intermediate_size)
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = attention_size // num_heads
    kv_attention_size = graph_blocks.infer_kv_attention_size(
        weights, num_kv_heads=num_kv_heads, head_dim=head_dim)
    seq_length = max_cache_length  # max sequence length for the encoder

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # --- Inputs ---
    # input_ids: [seq_length] token IDs (padded to max length)
    input_ids = network.add_input("input_ids", trt.int32, (seq_length,))
    # attention_mask: [seq_length] — 1 for real tokens, 0 for padding.
    # Used to mask out padding positions in bidirectional attention.
    attention_mask_input = network.add_input("attention_mask", trt.int32, (seq_length,))
    # input_embed: [seq_length, hidden] — pre-computed features for image tokens
    input_embed = network.add_input("input_embed", trt.float32, (seq_length, hidden))
    # use_input_embed: [seq_length] — per-position flag (0.0=use token lookup, 1.0=use input_embed)
    use_input_embed = network.add_input("use_input_embed", trt.float32, (seq_length,))

    # --- Embedding with input_embed bypass ---
    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["embedding"])
    # Gather: [seq_length] -> [seq_length, hidden]
    gather = network.add_gather(embedding_table, input_ids, 0)
    token_embed = gather.get_output(0)

    # Select: hidden[i] = input_embed[i] * use[i] + token_embed[i] * (1 - use[i])
    # Reshape use_input_embed [seq_length] -> [seq_length, 1] for broadcasting
    use_reshape = network.add_shuffle(use_input_embed)
    use_reshape.reshape_dims = (seq_length, 1)
    ones_bcast = graph_ops.add_constant(
        network, (1, 1), np.array([1.0], dtype=np.float32))
    inv_use = network.add_elementwise(
        ones_bcast, use_reshape.get_output(0),
        trt.ElementWiseOperation.SUB)  # [seq_length, 1]: 1 where token, 0 where embed
    embed_part = network.add_elementwise(
        input_embed, use_reshape.get_output(0),
        trt.ElementWiseOperation.PROD)  # input_embed * use
    token_part = network.add_elementwise(
        token_embed, inv_use.get_output(0),
        trt.ElementWiseOperation.PROD)  # token_embed * (1 - use)
    merged = network.add_elementwise(
        embed_part.get_output(0), token_part.get_output(0),
        trt.ElementWiseOperation.SUM)
    hidden_state = merged.get_output(0)

    # --- RoPE tables ---
    graph_ops.validate_native_rope_dim(head_dim, field_name="head_dim")
    # Check for Llama3 RoPE scaling (from rope_parameters in llm_config)
    rope_params = config.raw.get("llm_config", {}).get("rope_parameters", {})
    rope_type = rope_params.get("rope_type", "")
    if rope_type == "llama3":
        cos_half_np = _make_llama3_rope_table_half_dim(
            seq_length, head_dim, config.rope_theta, True,
            factor=rope_params.get("factor", 1.0),
            low_freq_factor=rope_params.get("low_freq_factor", 1.0),
            high_freq_factor=rope_params.get("high_freq_factor", 1.0),
            original_max_position_embeddings=rope_params.get(
                "original_max_position_embeddings", 8192))
        sin_half_np = _make_llama3_rope_table_half_dim(
            seq_length, head_dim, config.rope_theta, False,
            factor=rope_params.get("factor", 1.0),
            low_freq_factor=rope_params.get("low_freq_factor", 1.0),
            high_freq_factor=rope_params.get("high_freq_factor", 1.0),
            original_max_position_embeddings=rope_params.get(
                "original_max_position_embeddings", 8192))
    else:
        cos_half_np = graph_ops.make_rope_table_half_dim(
            seq_length, head_dim, config.rope_theta, True)
        sin_half_np = graph_ops.make_rope_table_half_dim(
            seq_length, head_dim, config.rope_theta, False)
    cos_half_tensor = graph_ops.add_constant(
        network, cos_half_np.shape, cos_half_np)
    sin_half_tensor = graph_ops.add_constant(
        network, sin_half_np.shape, sin_half_np)
    rope_position_ids = graph_ops.add_constant(
        network, (seq_length,), np.arange(seq_length, dtype=np.int32),
        dtype=np.int32)
    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=np.float32))
    attn_scale = 1.0 / np.sqrt(max(head_dim, 1))

    # --- Build padding attention mask from attention_mask input ---
    # attention_mask is [seq_length] with 1=real, 0=padding.
    # We need a [1, seq_length, seq_length] additive mask where
    # positions attending TO padding get -1e10.
    # Convert int32 mask to float: 0 -> -1e10, 1 -> 0.0
    mask_float = network.add_cast(
        attention_mask_input, trt.float32)
    # (1 - mask) * -1e10: padding positions get -1e10
    ones_const = graph_ops.add_constant(
        network, (1,), np.array([1.0], dtype=np.float32))
    neg_large = graph_ops.add_constant(
        network, (1,), np.array([-1e10], dtype=np.float32))
    inv_mask = network.add_elementwise(
        ones_const, mask_float.get_output(0),
        trt.ElementWiseOperation.SUB)  # [seq_length]: 0 for real, 1 for pad
    pad_penalty = network.add_elementwise(
        inv_mask.get_output(0), neg_large,
        trt.ElementWiseOperation.PROD)  # [seq_length]: 0.0 for real, -1e10 for pad
    pad_mask_row = network.add_shuffle(pad_penalty.get_output(0))
    pad_mask_row.reshape_dims = (1, seq_length)
    query_zeros = graph_ops.add_constant(
        network, (seq_length, 1), np.zeros((seq_length, 1), dtype=np.float32))
    pad_mask_2d = network.add_elementwise(
        query_zeros, pad_mask_row.get_output(0), trt.ElementWiseOperation.SUM)
    pad_mask_4d = graph_ops.add_2d_mask_to_4d(network, pad_mask_2d.get_output(0))

    # --- Encoder layers (no KV cache -- full self-attention over seq_length) ---
    # Single-pass encoder: process all positions at once.
    # Eagle uses LlamaBidirectionalModel with is_causal=False, so we use
    # bidirectional (non-causal) attention -- no upper-triangular mask.
    # The padding mask prevents real tokens from attending to padding positions.
    # Output: hidden_states [seq_length, hidden_size]

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"

        # Pre-norm
        norm1 = graph_blocks.apply_norm(
            network, hidden_state, hidden,
            weights[f"{prefix}.input_norm"],
            None, eps_tensor, "rmsnorm")

        # Self-attention: Q, K, V projections
        q = graph_ops.add_matmul_rhs_constant(
            network, norm1, hidden, attention_size, weights[f"{prefix}.w_q"])
        k = graph_ops.add_matmul_rhs_constant(
            network, norm1, hidden, kv_attention_size, weights[f"{prefix}.w_k"])
        v = graph_ops.add_matmul_rhs_constant(
            network, norm1, hidden, kv_attention_size, weights[f"{prefix}.w_v"])

        q_rope = graph_ops.add_apply_rope_native(
            network, q, num_heads, head_dim,
            cos_half_tensor, sin_half_tensor, rope_position_ids,
            head_dim, sequence_length=seq_length)
        k_rope = graph_ops.add_apply_rope_native(
            network, k, num_kv_heads, head_dim,
            cos_half_tensor, sin_half_tensor, rope_position_ids,
            head_dim, sequence_length=seq_length)

        attn_concat = graph_ops.add_attention_from_rows(
            network, q_rope, k_rope, v,
            num_heads=num_heads, head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            q_seq=seq_length, kv_seq=seq_length,
            mask=pad_mask_4d,
            scale=attn_scale)

        # Output projection
        proj_out = graph_ops.add_matmul_rhs_constant(
            network, attn_concat, attention_size, hidden,
            weights[f"{prefix}.w_o"])

        # Residual
        residual1 = network.add_elementwise(
            hidden_state, proj_out,
            trt.ElementWiseOperation.SUM)

        # Post-attention norm + MLP
        norm2 = graph_blocks.apply_norm(
            network, residual1.get_output(0), hidden,
            weights[f"{prefix}.post_attn_norm"],
            None, eps_tensor, "rmsnorm")

        mlp_out = graph_blocks.add_swiglu_mlp(
            network, norm2, weights=weights, prefix=prefix,
            hidden_size=hidden, mlp_size=mlp_size)

        # Final residual
        residual2 = network.add_elementwise(
            residual1.get_output(0), mlp_out,
            trt.ElementWiseOperation.SUM)
        hidden_state = residual2.get_output(0)

    # --- Final norm ---
    final_norm = weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = graph_blocks.apply_norm(
            network, hidden_state, hidden, final_norm, None,
            eps_tensor, "rmsnorm")

    # --- Output ---
    if is_reranker and "score_weight" in weights:
        # Reranking: apply score head to last token's hidden state
        # Take last position: hidden_state[seq_length-1, :]
        # For now, output all positions; C++ will pick the right one
        score = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden,
            weights["score_weight"].shape[1], weights["score_weight"])
        if "score_bias" in weights:
            score = graph_ops.add_bias_sum(
                network, score, weights["score_weight"].shape[1],
                weights["score_bias"])
        score.name = "score"
        network.mark_output(score)
    else:
        # Embedding: output all hidden states [seq_length, hidden]
        # C++ will do mean pooling + L2 normalization
        hidden_state.name = "hidden_states"
        network.mark_output(hidden_state)

    # --- Build ---
    mode_str = "reranking" if is_reranker else "embedding"
    if verbose:
        print(f"[trtmc build] Building Eagle {mode_str} engine "
              f"({num_layers} layers, hidden={hidden}, "
              f"attn={attention_size}, mlp={mlp_size}, "
              f"seq_len={seq_length}) ...",
              file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")

    return bytes(plan)


def _build_siglip_vision_engine(
    vision_config: dict,
    weights: WeightDict,
    *,
    verbose: bool = False,
) -> bytes:
    """Build SigLIP-2 vision encoder TRT engine.

    Input: pixel_values [3, H, W]
    Output: vision_features [num_patches, vision_hidden_size]
    """
    from tensorrt_model_connect import trt_compat
    trt = trt_compat.get_trt()
    from . import graph_ops

    image_size = vision_config.get("image_size", 384)
    patch_size = vision_config.get("patch_size", 14)
    vision_hidden = vision_config.get("hidden_size", 1152)
    num_vision_layers = vision_config.get("num_hidden_layers", 27)
    num_vision_heads = vision_config.get("num_attention_heads", 16)
    layer_norm_eps = vision_config.get("layer_norm_eps", 1e-6)

    num_patches_h = image_size // patch_size
    num_patches_w = image_size // patch_size
    num_patches = num_patches_h * num_patches_w
    head_dim = vision_hidden // num_vision_heads

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # Input: pixel_values [3, image_size, image_size]
    pixel_values = network.add_input(
        "pixel_values", trt.float32, (3, image_size, image_size))

    # --- Patch embedding (Conv2D) ---
    patch_proj_key = None
    for prefix in ("vision_model.vision_model.embeddings.patch_embedding.weight",
                    "vision_model.embeddings.patch_embedding.weight",
                    "visual.patch_embed.proj.weight"):
        if prefix in weights:
            patch_proj_key = prefix
            break

    if patch_proj_key is not None:
        patch_w = weights[patch_proj_key]
        # Conv2D weight shape: [out_channels, in_channels, kH, kW]
        if patch_w.ndim == 4:
            patch_w_flat = patch_w.astype(np.float32)
        else:
            patch_w_flat = patch_w.reshape(
                vision_hidden, 3, patch_size, patch_size).astype(np.float32)

        # Reshape input for Conv2D: [1, 3, H, W]
        input_4d = network.add_shuffle(pixel_values)
        input_4d.reshape_dims = (1, 3, image_size, image_size)

        conv = network.add_convolution_nd(
            input_4d.get_output(0),
            vision_hidden,
            (patch_size, patch_size),
            trt.Weights(np.ascontiguousarray(patch_w_flat)))
        conv.stride_nd = (patch_size, patch_size)

        # Look for bias (TRT conv expects rank-1 or rank-4 bias)
        for bias_key in ("vision_model.vision_model.embeddings.patch_embedding.bias",
                         "vision_model.embeddings.patch_embedding.bias",
                         "visual.patch_embed.proj.bias"):
            if bias_key in weights:
                bias_data = weights[bias_key].astype(np.float32).reshape(
                    1, vision_hidden, 1, 1)
                conv.set_input(2, graph_ops.add_constant(
                    network, (1, vision_hidden, 1, 1), bias_data))
                break

        # Conv output: [1, vision_hidden, num_patches_h, num_patches_w]
        # Reshape to [num_patches, vision_hidden]
        flatten = network.add_shuffle(conv.get_output(0))
        flatten.reshape_dims = (vision_hidden, num_patches)
        flatten.second_transpose = (1, 0)  # [num_patches, vision_hidden]
        hidden_state = flatten.get_output(0)
    else:
        # Fallback: identity (should not happen with valid weights)
        raise RuntimeError("Missing patch embedding weights for SigLIP-2")

    # --- Position embedding (if present) ---
    for pos_key in ("vision_model.vision_model.embeddings.position_embedding.weight",
                     "vision_model.embeddings.position_embedding.weight",
                     "visual.pos_embed"):
        if pos_key in weights:
            pos_w = weights[pos_key].astype(np.float32)
            if pos_w.shape[0] >= num_patches:
                pos_w = pos_w[:num_patches, :]
            pos_const = graph_ops.add_constant(
                network, (num_patches, vision_hidden), pos_w)
            pos_add = network.add_elementwise(
                hidden_state, pos_const, trt.ElementWiseOperation.SUM)
            hidden_state = pos_add.get_output(0)
            break

    eps_const = graph_ops.add_constant(
        network, (1, 1),
        np.array([layer_norm_eps], dtype=np.float32))

    # --- Transformer layers ---
    for layer_idx in range(num_vision_layers):
        # Find layer prefix
        lp = None
        for prefix in (f"vision_model.vision_model.encoder.layers.{layer_idx}",
                       f"vision_model.encoder.layers.{layer_idx}",
                       f"visual.blocks.{layer_idx}"):
            # Check if any weight starts with this prefix
            test_key = f"{prefix}.layer_norm1.weight"
            alt_key = f"{prefix}.norm1.weight"
            if test_key in weights or alt_key in weights:
                lp = prefix
                break

        if lp is None:
            continue  # skip missing layers

        # Layer norm 1
        ln1_w_key = f"{lp}.layer_norm1.weight"
        if ln1_w_key not in weights:
            ln1_w_key = f"{lp}.norm1.weight"
        ln1_b_key = f"{lp}.layer_norm1.bias"
        if ln1_b_key not in weights:
            ln1_b_key = f"{lp}.norm1.bias"

        ln1_w = weights.get(ln1_w_key)
        ln1_b = weights.get(ln1_b_key)
        if ln1_w is not None:
            normed = _add_layer_norm_vision(
                network, hidden_state, vision_hidden,
                ln1_w.astype(np.float32),
                ln1_b.astype(np.float32) if ln1_b is not None else np.zeros(vision_hidden, dtype=np.float32),
                eps_const)
        else:
            normed = hidden_state

        # Self-attention (simplified: fused QKV or separate Q/K/V)
        qkv_key = f"{lp}.self_attn.qkv.weight"
        if qkv_key not in weights:
            qkv_key = f"{lp}.attn.qkv.weight"

        if qkv_key in weights:
            # Fused QKV
            qkv_w = weights[qkv_key].astype(np.float32)
            if qkv_w.shape[0] == 3 * vision_hidden:
                qkv_w_t = np.ascontiguousarray(qkv_w.T)
                qkv = graph_ops.add_matmul_rhs_constant(
                    network, normed, vision_hidden, 3 * vision_hidden, qkv_w_t)
            else:
                qkv = graph_ops.add_matmul_rhs_constant(
                    network, normed, vision_hidden, 3 * vision_hidden,
                    np.ascontiguousarray(qkv_w.T))

            # Split Q, K, V
            q_slice = network.add_slice(
                qkv, (0, 0), (num_patches, vision_hidden), (1, 1))
            k_slice = network.add_slice(
                qkv, (0, vision_hidden), (num_patches, vision_hidden), (1, 1))
            v_slice = network.add_slice(
                qkv, (0, 2 * vision_hidden), (num_patches, vision_hidden), (1, 1))
            q_out = q_slice.get_output(0)
            k_out = k_slice.get_output(0)
            v_out = v_slice.get_output(0)
        else:
            # Separate Q, K, V
            q_key = f"{lp}.self_attn.q_proj.weight"
            if q_key not in weights:
                q_key = f"{lp}.attn.q_proj.weight"
            k_key = f"{lp}.self_attn.k_proj.weight"
            if k_key not in weights:
                k_key = f"{lp}.attn.k_proj.weight"
            v_key = f"{lp}.self_attn.v_proj.weight"
            if v_key not in weights:
                v_key = f"{lp}.attn.v_proj.weight"

            q_w = weights[q_key].astype(np.float32)
            k_w = weights[k_key].astype(np.float32)
            v_w = weights[v_key].astype(np.float32)

            q_out = graph_ops.add_matmul_rhs_constant(
                network, normed, vision_hidden, vision_hidden,
                np.ascontiguousarray(q_w.T))
            k_out = graph_ops.add_matmul_rhs_constant(
                network, normed, vision_hidden, vision_hidden,
                np.ascontiguousarray(k_w.T))
            v_out = graph_ops.add_matmul_rhs_constant(
                network, normed, vision_hidden, vision_hidden,
                np.ascontiguousarray(v_w.T))

        concat = graph_ops.add_attention_from_rows(
            network, q_out, k_out, v_out,
            num_heads=num_vision_heads, head_dim=head_dim,
            q_seq=num_patches, kv_seq=num_patches)

        # Output projection
        out_key = f"{lp}.self_attn.out_proj.weight"
        if out_key not in weights:
            out_key = f"{lp}.attn.out_proj.weight"
            if out_key not in weights:
                out_key = f"{lp}.attn.proj.weight"

        if out_key in weights:
            out_w = weights[out_key].astype(np.float32)
            proj = graph_ops.add_matmul_rhs_constant(
                network, concat, vision_hidden, vision_hidden,
                np.ascontiguousarray(out_w.T))
        else:
            proj = concat

        # Residual
        res1 = network.add_elementwise(
            hidden_state, proj, trt.ElementWiseOperation.SUM)

        # Layer norm 2
        ln2_w_key = f"{lp}.layer_norm2.weight"
        if ln2_w_key not in weights:
            ln2_w_key = f"{lp}.norm2.weight"
        ln2_b_key = f"{lp}.layer_norm2.bias"
        if ln2_b_key not in weights:
            ln2_b_key = f"{lp}.norm2.bias"

        ln2_w = weights.get(ln2_w_key)
        ln2_b = weights.get(ln2_b_key)
        if ln2_w is not None:
            normed2 = _add_layer_norm_vision(
                network, res1.get_output(0), vision_hidden,
                ln2_w.astype(np.float32),
                ln2_b.astype(np.float32) if ln2_b is not None else np.zeros(vision_hidden, dtype=np.float32),
                eps_const)
        else:
            normed2 = res1.get_output(0)

        # MLP: fc1 -> GELU -> fc2
        fc1_key = f"{lp}.mlp.fc1.weight"
        if fc1_key not in weights:
            fc1_key = f"{lp}.mlp.fc1.weight"
        fc2_key = f"{lp}.mlp.fc2.weight"
        if fc2_key not in weights:
            fc2_key = f"{lp}.mlp.fc2.weight"

        if fc1_key in weights and fc2_key in weights:
            fc1_w = weights[fc1_key].astype(np.float32)
            fc2_w = weights[fc2_key].astype(np.float32)

            mlp_hidden = fc1_w.shape[0]
            fc1_out = graph_ops.add_matmul_rhs_constant(
                network, normed2, vision_hidden, mlp_hidden,
                np.ascontiguousarray(fc1_w.T))

            # GELU activation
            fc1_b_key = f"{lp}.mlp.fc1.bias"
            if fc1_b_key in weights:
                fc1_out = graph_ops.add_bias_sum(
                    network, fc1_out, mlp_hidden,
                    weights[fc1_b_key].astype(np.float32))

            # Approximate GELU via TRT
            gelu_layer = network.add_activation(
                fc1_out, trt.ActivationType.GELU_ERF)

            fc2_out = graph_ops.add_matmul_rhs_constant(
                network, gelu_layer.get_output(0), mlp_hidden, vision_hidden,
                np.ascontiguousarray(fc2_w.T))

            fc2_b_key = f"{lp}.mlp.fc2.bias"
            if fc2_b_key in weights:
                fc2_out = graph_ops.add_bias_sum(
                    network, fc2_out, vision_hidden,
                    weights[fc2_b_key].astype(np.float32))
        else:
            fc2_out = normed2

        # Residual
        res2 = network.add_elementwise(
            res1.get_output(0), fc2_out, trt.ElementWiseOperation.SUM)
        hidden_state = res2.get_output(0)

    # --- Pixel shuffle (2x2 merge) + MLP projector (mlp1) ---
    # Eagle VLM merges 2x2 adjacent patches before the MLP projector:
    # [num_patches, vision_hidden] -> [num_patches/4, 4*vision_hidden] -> MLP -> [num_patches/4, text_hidden]
    mlp1_ln_w = weights.get("mlp1.0.weight")
    mlp1_fc1_w = weights.get("mlp1.1.weight")
    mlp1_fc2_w = weights.get("mlp1.3.weight")

    if mlp1_ln_w is not None and mlp1_fc1_w is not None and mlp1_fc2_w is not None:
        mlp1_ln_b = weights.get("mlp1.0.bias")
        mlp1_fc1_b = weights.get("mlp1.1.bias")
        mlp1_fc2_b = weights.get("mlp1.3.bias")

        text_hidden_size = mlp1_fc2_w.shape[0]  # output dim of projector
        mlp1_in_dim = mlp1_ln_w.shape[0]  # typically 4*vision_hidden (4608 = 4*1152)
        merge_factor = mlp1_in_dim // vision_hidden  # e.g. 4 for 2x2 merge

        if merge_factor > 1:
            # Pixel shuffle: reshape [H*W, C] -> [H, W, C] -> [H//m, m, W//m, m, C] -> [H//m * W//m, m*m*C]
            m = int(np.sqrt(merge_factor))  # merge_size per spatial dim
            # Truncate spatial dims to be divisible by m
            h_trunc = (num_patches_h // m) * m
            w_trunc = (num_patches_w // m) * m
            out_h = num_patches_h // m
            out_w = num_patches_w // m
            num_merged = out_h * out_w

            # Reshape: [num_patches, vision_hidden] -> [num_patches_h, num_patches_w, vision_hidden]
            reshape_hw = network.add_shuffle(hidden_state)
            reshape_hw.reshape_dims = (num_patches_h, num_patches_w, vision_hidden)

            # Slice to truncated size if needed (discard edge patches)
            if h_trunc != num_patches_h or w_trunc != num_patches_w:
                slice_layer = network.add_slice(
                    reshape_hw.get_output(0),
                    (0, 0, 0), (h_trunc, w_trunc, vision_hidden), (1, 1, 1))
                truncated = slice_layer.get_output(0)
            else:
                truncated = reshape_hw.get_output(0)

            # Reshape: [h_trunc, w_trunc, C] -> [out_h, m, out_w, m, C]
            reshape_merge = network.add_shuffle(truncated)
            reshape_merge.reshape_dims = (out_h, m, out_w, m, vision_hidden)
            # Transpose to [out_h, out_w, m, m, C]
            reshape_merge.second_transpose = (0, 2, 1, 3, 4)

            # Flatten: [out_h, out_w, m, m, C] -> [out_h*out_w, m*m*C]
            flatten = network.add_shuffle(reshape_merge.get_output(0))
            flatten.reshape_dims = (num_merged, mlp1_in_dim)
            hidden_state = flatten.get_output(0)

            if verbose:
                print(f"[trtmc build] Pixel shuffle: [{num_patches_h}x{num_patches_w}, {vision_hidden}] -> "
                      f"[{num_merged}, {mlp1_in_dim}]", file=sys.stderr)
        else:
            num_merged = num_patches

        # LayerNorm over mlp1_in_dim
        ln_eps = graph_ops.add_constant(
            network, (1, 1),
            np.array([layer_norm_eps], dtype=np.float32))
        hidden_state = _add_layer_norm_vision(
            network, hidden_state, mlp1_in_dim,
            mlp1_ln_w.astype(np.float32),
            mlp1_ln_b.astype(np.float32) if mlp1_ln_b is not None
                else np.zeros(mlp1_in_dim, dtype=np.float32),
            ln_eps)

        # FC1: mlp1_in_dim -> intermediate
        mlp1_inter = mlp1_fc1_w.shape[0]
        fc1 = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, mlp1_in_dim, mlp1_inter,
            np.ascontiguousarray(mlp1_fc1_w.astype(np.float32).T))
        if mlp1_fc1_b is not None:
            fc1 = graph_ops.add_bias_sum(
                network, fc1, mlp1_inter,
                mlp1_fc1_b.astype(np.float32))

        # GELU activation
        gelu = network.add_activation(fc1, trt.ActivationType.GELU_ERF)

        # FC2: intermediate -> text_hidden_size
        fc2 = graph_ops.add_matmul_rhs_constant(
            network, gelu.get_output(0), mlp1_inter, text_hidden_size,
            np.ascontiguousarray(mlp1_fc2_w.astype(np.float32).T))
        if mlp1_fc2_b is not None:
            fc2 = graph_ops.add_bias_sum(
                network, fc2, text_hidden_size,
                mlp1_fc2_b.astype(np.float32))

        hidden_state = fc2
        # Update num_patches for output annotation
        num_patches = num_merged if merge_factor > 1 else num_patches
        if verbose:
            print(f"[trtmc build] Added MLP projector: {mlp1_in_dim} -> "
                  f"{mlp1_inter} -> {text_hidden_size}", file=sys.stderr)

    # Output: vision_features [num_patches, output_dim]
    hidden_state.name = "vision_features"
    network.mark_output(hidden_state)

    if verbose:
        print(f"[trtmc build] Building SigLIP-2 vision engine "
              f"({num_vision_layers} layers, hidden={vision_hidden}, "
              f"patches={num_patches}) ...", file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("SigLIP-2 vision engine build failed")

    return bytes(plan)


def _add_layer_norm_vision(
    network, inp, hidden_size, gamma, beta, eps_tensor):
    """Add LayerNorm for vision transformer."""
    from tensorrt_model_connect import trt_compat
    trt = trt_compat.get_trt()
    from . import graph_ops

    gamma_const = graph_ops.add_constant(
        network, (1, hidden_size), gamma.reshape(1, hidden_size))
    beta_const = graph_ops.add_constant(
        network, (1, hidden_size), beta.reshape(1, hidden_size))

    # Mean
    mean_layer = network.add_reduce(inp, trt.ReduceOperation.AVG, 1 << 1, True)
    # Subtract mean
    sub = network.add_elementwise(
        inp, mean_layer.get_output(0), trt.ElementWiseOperation.SUB)
    # Variance
    sq = network.add_elementwise(
        sub.get_output(0), sub.get_output(0), trt.ElementWiseOperation.PROD)
    var = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, True)
    # Add eps
    var_eps = network.add_elementwise(
        var.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM)
    # Sqrt
    sqrt = network.add_unary(var_eps.get_output(0), trt.UnaryOperation.SQRT)
    # Divide
    normed = network.add_elementwise(
        sub.get_output(0), sqrt.get_output(0), trt.ElementWiseOperation.DIV)
    # Scale + shift
    scaled = network.add_elementwise(
        normed.get_output(0), gamma_const, trt.ElementWiseOperation.PROD)
    shifted = network.add_elementwise(
        scaled.get_output(0), beta_const, trt.ElementWiseOperation.SUM)
    return shifted.get_output(0)


def _make_llama3_rope_table_half_dim(
    max_seq_length: int,
    head_dim: int,
    rope_theta: float,
    cosine: bool,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
) -> np.ndarray:
    """Build Llama3-style native RoPE table with frequency scaling.

    Llama3 RoPE applies frequency-dependent scaling to inv_freq:
    - High-frequency dims (short wavelength): no scaling
    - Low-frequency dims (long wavelength): scale by 1/factor
    - Mid-frequency dims: smooth interpolation
    """
    half_dim = head_dim // 2

    # Standard inv_freq
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))

    # Llama3 wavelength-based scaling
    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor

    scaled_inv_freq = np.empty_like(inv_freq)
    for i, freq in enumerate(inv_freq):
        wavelen = 2.0 * np.pi / freq
        if wavelen < high_freq_wavelen:
            scaled_inv_freq[i] = freq
        elif wavelen > low_freq_wavelen:
            scaled_inv_freq[i] = freq / factor
        else:
            smooth = (original_max_position_embeddings / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor)
            scaled_inv_freq[i] = (1 - smooth) * freq / factor + smooth * freq

    # Build table [max_seq_length, head_dim // 2] for IRotaryEmbeddingLayer.
    table = np.full(
        (max_seq_length, half_dim),
        1.0 if cosine else 0.0,
        dtype=np.float32,
    )

    for pos in range(max_seq_length):
        for dim in range(half_dim):
            angle = pos * scaled_inv_freq[dim]
            table[pos, dim] = float(np.cos(angle) if cosine else np.sin(angle))

    return table


plugin = EagleVLMPlugin()
