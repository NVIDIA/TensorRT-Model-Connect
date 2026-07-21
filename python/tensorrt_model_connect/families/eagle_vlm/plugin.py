# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from pathlib import Path

import numpy as np

from .config import ModelConfig
from .weights import (
    WeightDict,
    _has_tensor,
    _load_tensor,
    _open_safetensors,
    _transpose_2d,
)
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)

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
        return getattr(self, "_runtime_strategy", "eagle_vlm_embedding")

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
        self._runtime_strategy = "eagle_vlm_reranking" if is_rerank else "eagle_vlm_embedding"
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="Eagle VLM tensor-parallel builds")
            from .model.parallel import build_eagle_vlm_tp_engine
            return build_eagle_vlm_tp_engine(
                config, weights, max_cache_length,
                is_reranker=is_rerank,
                precision=precision,
                verbose=verbose,
                parallel_config=parallel)
        return _build_eagle_engine(
            config, weights, max_cache_length,
            is_reranker=is_rerank, precision=precision, verbose=verbose)

    def build_vision_engine(
        self, model_dir: str, config: ModelConfig, weights: WeightDict,
        *, precision: str = "fp32", verbose: bool = False,
    ) -> bytes | None:
        vision_config = config.raw.get("vision_config")
        if vision_config is None:
            return None
        vision_weights = _load_vision_weights(model_dir)
        return _build_siglip_vision_engine(
            vision_config, vision_weights, precision=precision, verbose=verbose)

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
            "runtime_strategy": "eagle_vlm_reranking" if is_rerank else "eagle_vlm_embedding",
        }


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _load_eagle_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    """Load Eagle text backbone weights (Llama architecture)."""
    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    kv_attention_size = num_kv_heads * config.head_dim

    weights = WeightDict()

    # Embedding and reranking releases differ only by the outer ``model.``
    # wrapper. The flat ``model`` root is retained for the family unit fixture.
    for model_root in ("language_model", "model.language_model", "model"):
        embed_key = f"{model_root}.embed_tokens.weight"
        if _has_tensor(readers, embed_key):
            break
    else:
        raise RuntimeError("Cannot find Eagle text embedding weights")
    embedding = _load_tensor(readers, embed_key)
    assert embedding.shape == (vocab, hidden), (
        f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
    weights["embedding"] = embedding.astype(np.float32)

    layer_prefix = f"{model_root}.layers"

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

        weights[f"{prefix}.w_q"] = _transpose_2d(q_raw, "q_proj")
        weights[f"{prefix}.w_k"] = _transpose_2d(k_raw, "k_proj")
        weights[f"{prefix}.w_v"] = _transpose_2d(v_raw, "v_proj")
        weights[f"{prefix}.w_o"] = _transpose_2d(o_raw, "o_proj")

        # SwiGLU MLP
        gate_raw = _load_tensor(readers, f"{hf_prefix}.mlp.gate_proj.weight")
        up_raw = _load_tensor(readers, f"{hf_prefix}.mlp.up_proj.weight")
        down_raw = _load_tensor(readers, f"{hf_prefix}.mlp.down_proj.weight")

        if mlp_size == 0:
            mlp_size = gate_raw.shape[0]

        weights[f"{prefix}.w_gate"] = _transpose_2d(gate_raw, "gate")
        weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up")
        weights[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down")

    final_norm_key = f"{model_root}.norm.weight"
    if _has_tensor(readers, final_norm_key):
        weights["final_norm"] = _load_tensor(readers, final_norm_key).astype(np.float32)

    # Pooling head / score head (if present)
    # Eagle embedding uses the last hidden state with mean pooling + L2 norm
    # Eagle reranking uses a score head: linear projection from hidden -> 1
    if _has_tensor(readers, "score.weight"):
        weights["score_weight"] = _transpose_2d(
            _load_tensor(readers, "score.weight"), "score"
        )
    if _has_tensor(readers, "score.bias"):
        weights["score_bias"] = _load_tensor(readers, "score.bias").astype(np.float32)

    # No LM head needed for embedding/reranking (no token generation)
    # But we need w_out as a placeholder for the standard builder check
    weights["w_out"] = np.zeros((hidden, 1), dtype=np.float32)

    weights["_attention_size"] = attention_size
    weights["_kv_attention_size"] = kv_attention_size
    weights["_mlp_size"] = mlp_size

    return weights


def _load_vision_weights(model_dir: str) -> WeightDict:
    """Load SigLIP-2 vision encoder weights."""
    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    weights = WeightDict()
    for reader in readers:
        for key in reader.keys():
            if key.startswith("vision_model."):
                weights[key] = _load_tensor(readers, key)
            elif key.startswith("model.vision_model."):
                # Rerank model uses model.vision_model.* prefix — strip model.
                canon = key[len("model."):]
                weights[canon] = _load_tensor(readers, key)
            elif key.startswith("mlp1.") or key.startswith("model.mlp1."):
                canon = key.replace("model.", "", 1) if key.startswith("model.mlp1.") else key
                weights[canon] = _load_tensor(readers, key)

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
    precision: str = "fp32",
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
    from .model import model as graph_ops
    from .model import model as graph_blocks

    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
    else:
        raise ValueError(
            f"Unsupported Eagle VLM precision {precision!r}; expected fp32 or fp16")

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
    if work_trt_dtype != trt.float32:
        input_embed = network.add_cast(input_embed, work_trt_dtype).get_output(0)
        use_input_embed = network.add_cast(
            use_input_embed, work_trt_dtype).get_output(0)

    # --- Embedding with input_embed bypass ---
    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype)
    # Gather: [seq_length] -> [seq_length, hidden]
    gather = network.add_gather(embedding_table, input_ids, 0)
    token_embed = gather.get_output(0)

    # Select: hidden[i] = input_embed[i] * use[i] + token_embed[i] * (1 - use[i])
    # Reshape use_input_embed [seq_length] -> [seq_length, 1] for broadcasting
    use_reshape = network.add_shuffle(use_input_embed)
    use_reshape.reshape_dims = (seq_length, 1)
    ones_bcast = graph_ops.add_constant(
        network, (1, 1), np.array([1.0], dtype=work_np_dtype),
        dtype=work_np_dtype)
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
    # The released checkpoints use ``rope_scaling``; retain the earlier
    # ``rope_parameters`` spelling for compatible Eagle checkpoint revisions.
    llm_config = config.raw.get("llm_config", config.raw)
    rope_params = llm_config.get("rope_scaling") or llm_config.get("rope_parameters", {})
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
        network, cos_half_np.shape, cos_half_np, dtype=work_np_dtype)
    sin_half_tensor = graph_ops.add_constant(
        network, sin_half_np.shape, sin_half_np, dtype=work_np_dtype)
    rope_position_ids = graph_ops.add_constant(
        network, (seq_length,), np.arange(seq_length, dtype=np.int32),
        dtype=np.int32)
    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=work_np_dtype),
        dtype=work_np_dtype)
    attn_scale = 1.0 / np.sqrt(max(head_dim, 1))

    # --- Build padding attention mask from attention_mask input ---
    # attention_mask is [seq_length] with 1=real, 0=padding.
    # We need a [1, seq_length, seq_length] additive mask where
    # positions attending TO padding get -1e10.
    # Convert int32 mask to float: 0 -> -1e10, 1 -> 0.0
    mask_float = network.add_cast(
        attention_mask_input, work_trt_dtype)
    # (1 - mask) * -1e10: padding positions get -1e10
    ones_const = graph_ops.add_constant(
        network, (1,), np.array([1.0], dtype=work_np_dtype),
        dtype=work_np_dtype)
    mask_penalty = -1e4 if precision == "fp16" else -1e10
    neg_large = graph_ops.add_constant(
        network, (1,), np.array([mask_penalty], dtype=work_np_dtype),
        dtype=work_np_dtype)
    inv_mask = network.add_elementwise(
        ones_const, mask_float.get_output(0),
        trt.ElementWiseOperation.SUB)  # [seq_length]: 0 for real, 1 for pad
    pad_penalty = network.add_elementwise(
        inv_mask.get_output(0), neg_large,
        trt.ElementWiseOperation.PROD)  # [seq_length]: 0.0 for real, -1e10 for pad
    pad_mask_row = network.add_shuffle(pad_penalty.get_output(0))
    pad_mask_row.reshape_dims = (1, seq_length)
    query_zeros = graph_ops.add_constant(
        network, (seq_length, 1),
        np.zeros((seq_length, 1), dtype=work_np_dtype),
        dtype=work_np_dtype)
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
        norm1 = graph_blocks.add_rms_norm(
            network, hidden_state, hidden,
            weights[f"{prefix}.input_norm"],
            eps_tensor, dtype=work_np_dtype)

        # Self-attention: Q, K, V projections
        q = graph_ops.add_matmul_rhs_constant(
            network, norm1, hidden, attention_size, weights[f"{prefix}.w_q"],
            dtype=work_np_dtype)
        k = graph_ops.add_matmul_rhs_constant(
            network, norm1, hidden, kv_attention_size, weights[f"{prefix}.w_k"],
            dtype=work_np_dtype)
        v = graph_ops.add_matmul_rhs_constant(
            network, norm1, hidden, kv_attention_size, weights[f"{prefix}.w_v"],
            dtype=work_np_dtype)

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
            weights[f"{prefix}.w_o"], dtype=work_np_dtype)

        # Residual
        residual1 = network.add_elementwise(
            hidden_state, proj_out,
            trt.ElementWiseOperation.SUM)

        # Post-attention norm + MLP
        norm2 = graph_blocks.add_rms_norm(
            network, residual1.get_output(0), hidden,
            weights[f"{prefix}.post_attn_norm"],
            eps_tensor, dtype=work_np_dtype)

        mlp_out = graph_blocks.add_swiglu_mlp(
            network, norm2, weights=weights, prefix=prefix,
            hidden_size=hidden, mlp_size=mlp_size,
            dtype=work_np_dtype)

        # Final residual
        residual2 = network.add_elementwise(
            residual1.get_output(0), mlp_out,
            trt.ElementWiseOperation.SUM)
        hidden_state = residual2.get_output(0)

    # --- Final norm ---
    final_norm = weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = graph_blocks.add_rms_norm(
            network, hidden_state, hidden, final_norm,
            eps_tensor, dtype=work_np_dtype)

    # --- Output ---
    if is_reranker and "score_weight" in weights:
        # Reranking: apply score head to last token's hidden state
        # Take last position: hidden_state[seq_length-1, :]
        # For now, output all positions; C++ will pick the right one
        score = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden,
            weights["score_weight"].shape[1], weights["score_weight"],
            dtype=work_np_dtype)
        if "score_bias" in weights:
            score = graph_ops.add_bias_sum(
                network, score, weights["score_weight"].shape[1],
                weights["score_bias"], dtype=work_np_dtype)
        if score.dtype != trt.float32:
            score = network.add_cast(score, trt.float32).get_output(0)
        score.name = "score"
        network.mark_output(score)
    else:
        # Embedding: output all hidden states [seq_length, hidden]
        # C++ will do mean pooling + L2 normalization
        output = hidden_state
        if output.dtype != trt.float32:
            output = network.add_cast(output, trt.float32).get_output(0)
        output.name = "hidden_states"
        network.mark_output(output)

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
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    """Build SigLIP-2 vision encoder TRT engine.

    Input: pixel_values [3, H, W]
    Output: vision_features [num_patches, vision_hidden_size]
    """
    from tensorrt_model_connect import trt_compat
    trt = trt_compat.get_trt()
    from .model import model as graph_ops

    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
    else:
        raise ValueError(
            f"Unsupported Eagle VLM precision {precision!r}; expected fp32 or fp16")

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
    if work_trt_dtype != trt.float32:
        pixel_values = network.add_cast(pixel_values, work_trt_dtype).get_output(0)

    # The supported releases both use the SigLIP-2 checkpoint namespace.
    vision_root = "vision_model.vision_model"
    patch_root = f"{vision_root}.embeddings.patch_embedding"
    patch_w = weights[f"{patch_root}.weight"].astype(work_np_dtype)
    input_4d = network.add_shuffle(pixel_values)
    input_4d.reshape_dims = (1, 3, image_size, image_size)
    conv = network.add_convolution_nd(
        input_4d.get_output(0),
        vision_hidden,
        (patch_size, patch_size),
        trt.Weights(np.ascontiguousarray(patch_w)),
    )
    conv.stride_nd = (patch_size, patch_size)
    bias = weights[f"{patch_root}.bias"].astype(work_np_dtype).reshape(
        1, vision_hidden, 1, 1
    )
    conv.set_input(
        2,
        graph_ops.add_constant(
            network, (1, vision_hidden, 1, 1), bias, dtype=work_np_dtype
        ),
    )
    flatten = network.add_shuffle(conv.get_output(0))
    flatten.reshape_dims = (vision_hidden, num_patches)
    flatten.second_transpose = (1, 0)
    hidden_state = flatten.get_output(0)

    position = weights[f"{vision_root}.embeddings.position_embedding.weight"][
        :num_patches
    ].astype(work_np_dtype)
    position = graph_ops.add_constant(
        network, (num_patches, vision_hidden), position, dtype=work_np_dtype
    )
    hidden_state = network.add_elementwise(
        hidden_state, position, trt.ElementWiseOperation.SUM
    ).get_output(0)

    eps_const = graph_ops.add_constant(
        network, (1, 1),
        np.array([layer_norm_eps], dtype=np.float32))

    # --- Transformer layers ---
    for layer_idx in range(num_vision_layers):
        lp = f"{vision_root}.encoder.layers.{layer_idx}"
        ln1_w_key = f"{lp}.layer_norm1.weight"
        ln1_b_key = f"{lp}.layer_norm1.bias"
        normed = _add_layer_norm_vision(
            network,
            hidden_state,
            vision_hidden,
            weights[ln1_w_key].astype(np.float32),
            weights[ln1_b_key].astype(np.float32),
            eps_const,
            dtype=work_np_dtype,
        )

        projections = []
        for name in ("q_proj", "k_proj", "v_proj"):
            weight = weights[f"{lp}.self_attn.{name}.weight"].astype(work_np_dtype)
            projections.append(
                graph_ops.add_matmul_rhs_constant(
                    network,
                    normed,
                    vision_hidden,
                    vision_hidden,
                    np.ascontiguousarray(weight.T),
                    dtype=work_np_dtype,
                )
            )
        q_out, k_out, v_out = projections

        concat = graph_ops.add_attention_from_rows(
            network, q_out, k_out, v_out,
            num_heads=num_vision_heads, head_dim=head_dim,
            q_seq=num_patches, kv_seq=num_patches)

        out_key = f"{lp}.self_attn.out_proj.weight"
        out_w = weights[out_key].astype(work_np_dtype)
        proj = graph_ops.add_matmul_rhs_constant(
            network,
            concat,
            vision_hidden,
            vision_hidden,
            np.ascontiguousarray(out_w.T),
            dtype=work_np_dtype,
        )

        # Residual
        res1 = network.add_elementwise(
            hidden_state, proj, trt.ElementWiseOperation.SUM)

        ln2_w_key = f"{lp}.layer_norm2.weight"
        ln2_b_key = f"{lp}.layer_norm2.bias"
        normed2 = _add_layer_norm_vision(
            network,
            res1.get_output(0),
            vision_hidden,
            weights[ln2_w_key].astype(np.float32),
            weights[ln2_b_key].astype(np.float32),
            eps_const,
            dtype=work_np_dtype,
        )

        fc1_key = f"{lp}.mlp.fc1.weight"
        fc2_key = f"{lp}.mlp.fc2.weight"
        fc1_w = weights[fc1_key].astype(work_np_dtype)
        fc2_w = weights[fc2_key].astype(work_np_dtype)
        mlp_hidden = fc1_w.shape[0]
        fc1_out = graph_ops.add_matmul_rhs_constant(
            network,
            normed2,
            vision_hidden,
            mlp_hidden,
            np.ascontiguousarray(fc1_w.T),
            dtype=work_np_dtype,
        )
        fc1_out = graph_ops.add_bias_sum(
            network,
            fc1_out,
            mlp_hidden,
            weights[f"{lp}.mlp.fc1.bias"].astype(work_np_dtype),
            dtype=work_np_dtype,
        )
        gelu_layer = network.add_activation(fc1_out, trt.ActivationType.GELU_ERF)
        fc2_out = graph_ops.add_matmul_rhs_constant(
            network,
            gelu_layer.get_output(0),
            mlp_hidden,
            vision_hidden,
            np.ascontiguousarray(fc2_w.T),
            dtype=work_np_dtype,
        )
        fc2_out = graph_ops.add_bias_sum(
            network,
            fc2_out,
            vision_hidden,
            weights[f"{lp}.mlp.fc2.bias"].astype(work_np_dtype),
            dtype=work_np_dtype,
        )

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
            ln_eps, dtype=work_np_dtype)

        # FC1: mlp1_in_dim -> intermediate
        mlp1_inter = mlp1_fc1_w.shape[0]
        fc1 = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, mlp1_in_dim, mlp1_inter,
            np.ascontiguousarray(mlp1_fc1_w.astype(work_np_dtype).T),
            dtype=work_np_dtype)
        if mlp1_fc1_b is not None:
            fc1 = graph_ops.add_bias_sum(
                network, fc1, mlp1_inter,
                mlp1_fc1_b.astype(work_np_dtype), dtype=work_np_dtype)

        # GELU activation
        gelu = network.add_activation(fc1, trt.ActivationType.GELU_ERF)

        # FC2: intermediate -> text_hidden_size
        fc2 = graph_ops.add_matmul_rhs_constant(
            network, gelu.get_output(0), mlp1_inter, text_hidden_size,
            np.ascontiguousarray(mlp1_fc2_w.astype(work_np_dtype).T),
            dtype=work_np_dtype)
        if mlp1_fc2_b is not None:
            fc2 = graph_ops.add_bias_sum(
                network, fc2, text_hidden_size,
                mlp1_fc2_b.astype(work_np_dtype), dtype=work_np_dtype)

        hidden_state = fc2
        # Update num_patches for output annotation
        num_patches = num_merged if merge_factor > 1 else num_patches
        if verbose:
            print(f"[trtmc build] Added MLP projector: {mlp1_in_dim} -> "
                  f"{mlp1_inter} -> {text_hidden_size}", file=sys.stderr)

    # Output: vision_features [num_patches, output_dim]
    output = hidden_state
    if output.dtype != trt.float32:
        output = network.add_cast(output, trt.float32).get_output(0)
    output.name = "vision_features"
    network.mark_output(output)

    if verbose:
        print(f"[trtmc build] Building SigLIP-2 vision engine "
              f"({num_vision_layers} layers, hidden={vision_hidden}, "
              f"patches={num_patches}) ...", file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("SigLIP-2 vision engine build failed")

    return bytes(plan)


def _add_layer_norm_vision(
    network, inp, hidden_size, gamma, beta, eps_tensor,
    dtype: np.dtype = np.float32,
):
    """Add LayerNorm for vision transformer."""
    from tensorrt_model_connect import trt_compat
    trt = trt_compat.get_trt()
    from .model import model as graph_ops

    output_dtype = inp.dtype
    if dtype != np.float32:
        inp = network.add_cast(inp, trt.float32).get_output(0)
    if eps_tensor.dtype != trt.float32:
        eps_tensor = network.add_cast(eps_tensor, trt.float32).get_output(0)
    gamma_const = graph_ops.add_constant(
        network, (1, hidden_size), gamma.reshape(1, hidden_size),
        dtype=np.float32)
    beta_const = graph_ops.add_constant(
        network, (1, hidden_size), beta.reshape(1, hidden_size),
        dtype=np.float32)

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
    output = shifted.get_output(0)
    if output.dtype != output_dtype:
        output = network.add_cast(output, output_dtype).get_output(0)
    return output


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
    """Build the frequency-scaled Llama 3 RoPE table used by Eagle."""
    inv_freq = np.power(
        rope_theta, -np.arange(0, head_dim, 2, dtype=np.float64) / head_dim
    )
    wavelength = 2.0 * np.pi / inv_freq
    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor
    smooth = (original_max_position_embeddings / wavelength - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    scaled = np.where(
        wavelength < high_freq_wavelen,
        inv_freq,
        np.where(
            wavelength > low_freq_wavelen,
            inv_freq / factor,
            (1.0 - smooth) * inv_freq / factor + smooth * inv_freq,
        ),
    )
    angles = np.arange(max_seq_length, dtype=np.float64)[:, None] * scaled[None, :]
    return np.asarray(np.cos(angles) if cosine else np.sin(angles), dtype=np.float32)


plugin = EagleVLMPlugin()
