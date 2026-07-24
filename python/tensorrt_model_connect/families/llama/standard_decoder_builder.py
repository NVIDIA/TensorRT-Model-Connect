# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standard decoder engine builder (parameterized).

Builds a TensorRT engine for a decoder-only transformer. Supports multiple
norm, MLP, and position embedding strategies via parameters:

  norm_type:     "rmsnorm" | "layernorm"
  mlp_type:      "swiglu"  | "gelu_fc"
  position_type: "rope"    | "learned" | "alibi"
  activation:    "silu" | "gelu_new" | "gelu" | "relu" | "relu2" (used by gelu_fc MLP)

Tensor names MUST match what the C++ runtime expects:
  Inputs:  token_id, position_id, attention_mask, cache_k_0..N, cache_v_0..N
  Outputs: logits, present_k_0..N, present_v_0..N
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops
from . import graph_blocks
from .config import ModelConfig
from .dual_profile_decoder_builder import build_dual_profile_decoder_engine
from .utils import const_in_work_dtype

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from ...quantization.context import QuantContext


def _mark_debug_output(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    name: str,
) -> None:
    """Mark a tensor as a network output for debug inspection."""
    # Use an identity layer to avoid aliasing issues with existing outputs.
    cast = network.add_cast(tensor, trt.float32)
    out = cast.get_output(0)
    out.name = name
    network.mark_output(out)


def build_standard_decoder_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx: QuantContext | None = None,
    norm_type: str = "rmsnorm",
    mlp_type: str = "swiglu",
    position_type: str = "rope",
    activation: str = "silu",
    partial_rotary_factor: float = 1.0,
    interleaved_rope: bool = False,
    parallel_residual: bool = False,
    scale_attn_weights: bool = True,
    alibi_bias_scale: float = 1.0,
    embed_input: bool = False,
    verbose: bool = False,
    debug_layer_outputs: bool = False,
    hidden_state_output: bool = False,
) -> bytes:
    """Build a TRT engine plan (serialized bytes) for a standard decoder.

    Args:
        config: Model architecture from config.json.
        weights: Loaded weight dict from checkpoint_mapper.
        max_cache_length: KV cache length (engine is compiled for this value).
        precision: Compute precision ("fp32", "fp16", or "bf16").
        norm_type: "rmsnorm" or "layernorm".
        mlp_type: "swiglu" (3 projections: gate/up/down) or
                  "gelu_fc" (2 projections: fc1/fc2 with activation).
        position_type: "rope" (rotary), "learned" (absolute position embeddings),
            or "alibi" (attention with linear biases, no position embeddings).
        activation: Activation function for gelu_fc MLP ("gelu_new", "gelu", "relu", "relu2").
        partial_rotary_factor: Fraction of head dims that get RoPE (default 1.0).
        interleaved_rope: If True, use interleaved RoPE (CodeGen/GPT-J) where
            adjacent dims (d, d+1) share frequencies. Default False uses
            rotated-half (LLaMA/Qwen) where (d, d+half) share frequencies.
        scale_attn_weights: Whether to scale attention scores by 1/sqrt(head_dim).
            Most models use this (True, default). GPT-Neo does NOT scale (False).
        alibi_bias_scale: Extra scale applied to ALiBi slopes before building
            the additive attention mask. BLOOM leaves ALiBi unscaled while
            Falcon scales ALiBi with the same factor as QK scores.
        embed_input: If True, add input_embed [1, hidden] and use_input_embed [1]
            engine inputs. When use_input_embed==1, the decoder uses input_embed
            directly instead of the embedding lookup. Used for VL models where
            the vision encoder provides fused embeddings during prefill.
        verbose: Print TRT builder logs.
        debug_layer_outputs: If True, mark per-layer hidden states as network
            outputs for diff testing.

    Returns:
        Serialized engine plan bytes.
    """
    import os as _os
    # Mark the graph as honoring the internal decoder role contract. This is
    # embedded in the mutable config for family helpers that need to branch on
    # the active engine layout while building.
    config.raw["_decoder_engine_layout_supported"] = True
    decoder_engine_role = str(config.raw.get("_decoder_engine_role", "dual_profile"))
    runtime_memory_contract = config.raw.get(
        "_runtime_memory_contract")
    if isinstance(runtime_memory_contract, dict):
        if any((
            embed_input,
            debug_layer_outputs,
            hidden_state_output,
        )):
            raise NotImplementedError(
                "qualified runtime-memory graph does not support alternate "
                "decoder outputs or embedding inputs")
        if decoder_engine_role not in (
            "prefill", "decode", "dual_profile"
        ):
            raise ValueError(
                "qualified runtime-memory decoder role must be prefill, "
                "decode, or dual_profile")
        return build_dual_profile_decoder_engine(
            config, weights, max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            norm_type=norm_type,
            mlp_type=mlp_type,
            position_type=position_type,
            activation=activation,
            partial_rotary_factor=partial_rotary_factor,
            interleaved_rope=interleaved_rope,
            parallel_residual=parallel_residual,
            scale_attn_weights=scale_attn_weights,
            alibi_bias_scale=alibi_bias_scale,
            verbose=verbose,
            dynamic_kv_profile_rows=list(
                runtime_memory_contract[
                    "active_kv_profile_limits"]),
            profile_mode=decoder_engine_role,
        )

    # Dispatch to the dynamic-Sq builder for dual-profile and split-prefill
    # engines. Quantized builds (``quant_ctx``) thread Q/DQ insertion through
    # every projection matmul via
    # ``QuantContext.maybe_quantized_matmul``, so they share the dispatch.
    #
    # The legacy single-profile graph below stays in place for paths the
    # dual-profile builder does not yet cover:
    #
    #   - embed_input=True             (VL prefill replacement, Bark sub-engines)
    #   - debug_layer_outputs=True     (per-layer hidden-state dumps)
    #   - hidden_state_output=True     (speech / Bark hidden output)
    #   - dynamic-KV decode           (multi-bucket, fixed Sq=1 graph)
    #
    # ``TRTMC_NO_DUAL_PROFILE=1`` is an internal escape hatch (perf A/B,
    # bisects against the legacy graph). It is *not* intended as a
    # supported user-facing flag.
    dynamic_kv_cache = bool(config.raw.get("dynamic_kv_cache", False))
    if dynamic_kv_cache and position_type == "alibi":
        raise ValueError("dynamic_kv_cache is not supported for ALiBi decoder builds")

    requested_fp32_layers = tuple(config.raw.get("_fp32_layers", ()))
    _dual_profile_disabled_for = (
        embed_input
        or debug_layer_outputs
        or hidden_state_output
        or bool(requested_fp32_layers)
        or _os.environ.get("TRTMC_NO_DUAL_PROFILE") == "1"
    )
    if decoder_engine_role == "prefill" and _dual_profile_disabled_for:
        raise NotImplementedError(
            "split prefill engine is not supported for this standard decoder "
            "configuration")
    use_batched_builder = (
        decoder_engine_role == "prefill"
        or (decoder_engine_role == "dual_profile" and not dynamic_kv_cache)
    )
    if not _dual_profile_disabled_for and use_batched_builder:
        return build_dual_profile_decoder_engine(
            config, weights, max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            norm_type=norm_type,
            mlp_type=mlp_type,
            position_type=position_type,
            activation=activation,
            partial_rotary_factor=partial_rotary_factor,
            interleaved_rope=interleaved_rope,
            parallel_residual=parallel_residual,
            scale_attn_weights=scale_attn_weights,
            alibi_bias_scale=alibi_bias_scale,
            verbose=verbose,
            dynamic_kv_profile_rows=(
                (
                    config.raw.get("_dynamic_kv_profile_rows")
                    or [max_cache_length]
                )
                if dynamic_kv_cache
                else None
            ),
            profile_mode=("prefill" if decoder_engine_role == "prefill" else "dual_profile"),
        )

    attention_size: int = weights.get("_attention_size", config.attention_size)
    mlp_size: int = weights.get("_mlp_size", config.intermediate_size)
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    fp32_layers = frozenset(int(layer) for layer in requested_fp32_layers)
    invalid_fp32_layers = sorted(
        layer for layer in fp32_layers if layer < 0 or layer >= num_layers)
    if invalid_fp32_layers:
        raise ValueError(
            f"fp32_layers contains out-of-range indices: {invalid_fp32_layers}")
    if precision == "fp32":
        fp32_layers = frozenset()
    if fp32_layers and quant_ctx is not None:
        raise ValueError("fp32_layers is not supported with quantized builds")
    head_dim = attention_size // num_heads
    kv_attention_size = graph_blocks.infer_kv_attention_size(
        weights, num_kv_heads=num_kv_heads, head_dim=head_dim)
    attention_window = max_cache_length + 1
    dynamic_kv_opt_rows = int(config.raw.get("_dynamic_kv_opt_length", max_cache_length))
    dynamic_kv_opt_rows = max(1, min(dynamic_kv_opt_rows, max_cache_length))
    raw_profile_rows = config.raw.get("_dynamic_kv_profile_rows")
    if raw_profile_rows:
        dynamic_kv_profile_rows = []
        for row in raw_profile_rows:
            clamped = max(1, min(int(row), max_cache_length))
            if clamped not in dynamic_kv_profile_rows:
                dynamic_kv_profile_rows.append(clamped)
        dynamic_kv_profile_rows.sort()
    else:
        dynamic_kv_profile_rows = []

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    # Dynamic-KV model-only bundles use a hidden, family-certified engine
    # envelope. Give TensorRT enough build-only tactic workspace for that
    # envelope without exposing another user flag.
    workspace_bytes = (8 << 30) if dynamic_kv_cache else (1 << 30)
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

    # Precision configuration
    if precision == "fp16":
        work_np_dtype = np.float16
        work_trt_dtype = trt.float16
    elif precision == "bf16":
        work_np_dtype = np.float16  # stored as float16, TRT uses bfloat16
        work_trt_dtype = trt.bfloat16
    else:
        work_np_dtype = np.float32
        work_trt_dtype = trt.float32

    # ---------------------------------------------------------------
    # Inputs
    # ---------------------------------------------------------------
    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input(
        "attention_mask", trt.float32,
        (1, -1) if dynamic_kv_cache else (1, attention_window))

    # Optional VL inputs: when embed_input=True, the decoder can accept
    # a pre-computed embedding vector instead of a token ID.
    input_embed_tensor = None
    use_input_embed_tensor = None
    if embed_input:
        input_embed_tensor = network.add_input(
            "input_embed", trt.float32, (1, hidden))
        use_input_embed_tensor = network.add_input(
            "use_input_embed", trt.float32, (1,))

    cache_k_inputs = []
    cache_v_inputs = []
    for i in range(num_layers):
        ck = network.add_input(
            graph_ops.layer_tensor_name("cache_k", i),
            work_trt_dtype,
            (-1, kv_attention_size) if dynamic_kv_cache else (
                max_cache_length, kv_attention_size))
        cv = network.add_input(
            graph_ops.layer_tensor_name("cache_v", i),
            work_trt_dtype,
            (-1, kv_attention_size) if dynamic_kv_cache else (
                max_cache_length, kv_attention_size))
        cache_k_inputs.append(ck)
        cache_v_inputs.append(cv)

    if dynamic_kv_cache:
        if dynamic_kv_profile_rows:
            for profile_rows in dynamic_kv_profile_rows:
                profile = builder.create_optimization_profile()
                # Keep all profiles valid for short prompts / early decode steps.
                # The profile-specific value is the opt/max row budget, not a
                # lower bound on the live cache length.
                min_rows = 1
                profile.set_shape("attention_mask",
                                  (1, min_rows + 1),
                                  (1, profile_rows + 1),
                                  (1, profile_rows + 1))
                for i in range(num_layers):
                    min_cache_shape = (min_rows, kv_attention_size)
                    cache_shape = (profile_rows, kv_attention_size)
                    profile.set_shape(graph_ops.layer_tensor_name("cache_k", i),
                                      min_cache_shape, cache_shape, cache_shape)
                    profile.set_shape(graph_ops.layer_tensor_name("cache_v", i),
                                      min_cache_shape, cache_shape, cache_shape)
                trt_config.add_optimization_profile(profile)
        else:
            profile = builder.create_optimization_profile()
            profile.set_shape("attention_mask", (1, 2), (1, dynamic_kv_opt_rows + 1),
                              (1, attention_window))
            for i in range(num_layers):
                profile.set_shape(graph_ops.layer_tensor_name("cache_k", i),
                                  (1, kv_attention_size),
                                  (dynamic_kv_opt_rows, kv_attention_size),
                                  (max_cache_length, kv_attention_size))
                profile.set_shape(graph_ops.layer_tensor_name("cache_v", i),
                                  (1, kv_attention_size),
                                  (dynamic_kv_opt_rows, kv_attention_size),
                                  (max_cache_length, kv_attention_size))
            trt_config.add_optimization_profile(profile)

    # Cast attention mask to work dtype for elementwise compatibility
    if work_trt_dtype != trt.float32:
        mask_cast = network.add_cast(attention_mask, work_trt_dtype)
        attention_mask = mask_cast.get_output(0)

    def _cast_work_dtype(tensor: trt.ITensor) -> trt.ITensor:
        if tensor.dtype == work_trt_dtype:
            return tensor
        return network.add_cast(tensor, work_trt_dtype).get_output(0)

    # ---------------------------------------------------------------
    # Shared constants
    # ---------------------------------------------------------------
    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype)

    # RoPE tables (only needed when position_type == "rope")
    position_embed_table = None
    alibi_slopes_tensor = None
    alibi_indices_tensor = None

    # Native RoPE tensors for IRotaryEmbeddingLayer (TRT 10+).
    # Shape: [attention_window, rotary_ndims // 2].
    cos_half_tensor = None
    sin_half_tensor = None
    rotary_embedding_dim = int(head_dim * partial_rotary_factor)

    if position_type == "rope":
        graph_ops.validate_native_rope_dim(rotary_embedding_dim)
        cos_half_np = graph_ops.make_rope_table_half_dim(
            attention_window, head_dim, config.rope_theta, True,
            partial_rotary_factor, interleaved=interleaved_rope)
        sin_half_np = graph_ops.make_rope_table_half_dim(
            attention_window, head_dim, config.rope_theta, False,
            partial_rotary_factor, interleaved=interleaved_rope)
        cos_half_tensor = graph_ops.add_constant(
            network, cos_half_np.shape, cos_half_np, dtype=work_np_dtype)
        cos_half_tensor = _cast_work_dtype(cos_half_tensor)
        sin_half_tensor = graph_ops.add_constant(
            network, sin_half_np.shape, sin_half_np, dtype=work_np_dtype)
        sin_half_tensor = _cast_work_dtype(sin_half_tensor)
    elif position_type == "learned":
        pos_embed_np = weights["position_embedding"]
        position_embed_table = graph_ops.add_constant(
            network, pos_embed_np.shape, pos_embed_np, dtype=work_np_dtype)
    elif position_type == "alibi":
        alibi_slopes_np = graph_ops.compute_alibi_slopes(num_heads) * float(alibi_bias_scale)
        alibi_slopes_tensor = graph_ops.add_constant(
            network, (num_heads, 1, 1),
            alibi_slopes_np.reshape(num_heads, 1, 1), dtype=np.float32)
        # Cache position indices [0, 1, ..., max_cache_length-1].
        # The current token's position (position_id) is appended at runtime.
        alibi_indices_tensor = graph_ops.add_constant(
            network, (max_cache_length,),
            np.arange(max_cache_length, dtype=np.float32),
            dtype=np.float32)

    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=work_np_dtype),
        dtype=work_np_dtype)
    attn_scale = (1.0 / np.sqrt(max(head_dim, 1))) if scale_attn_weights else 1.0
    # ---------------------------------------------------------------
    # Embedding lookup (with optional embed_input override for VL)
    # ---------------------------------------------------------------
    gather = network.add_gather(embedding_table, token_id, 0)
    token_embed = gather.get_output(0)

    if embed_input and input_embed_tensor is not None and use_input_embed_tensor is not None:
        # Conditional embedding: (1 - flag) * token_embed + flag * input_embed
        # use_input_embed is [1] scalar (FP32), broadcast to [1, hidden]
        flag_broadcast = network.add_shuffle(use_input_embed_tensor)
        flag_broadcast.reshape_dims = (1, 1)
        # Cast flag to work dtype for elementwise compatibility
        flag_for_math = flag_broadcast.get_output(0)
        if work_trt_dtype != trt.float32:
            flag_for_math = network.add_cast(flag_for_math, work_trt_dtype).get_output(0)
        one_const = const_in_work_dtype(
            network, (1, 1), np.array([1.0], dtype=work_np_dtype),
            work_np_dtype, work_trt_dtype)
        token_embed = _cast_work_dtype(token_embed)
        inv_flag = network.add_elementwise(
            one_const, flag_for_math,
            trt.ElementWiseOperation.SUB)
        # (1 - flag) * token_embed
        tok_part = network.add_elementwise(
            inv_flag.get_output(0), token_embed,
            trt.ElementWiseOperation.PROD)
        # flag * input_embed
        embed_part = network.add_elementwise(
            flag_for_math, _cast_work_dtype(input_embed_tensor),
            trt.ElementWiseOperation.PROD)
        # sum
        hidden_state_sum = network.add_elementwise(
            tok_part.get_output(0), embed_part.get_output(0),
            trt.ElementWiseOperation.SUM)
        hidden_state = hidden_state_sum.get_output(0)
    else:
        hidden_state = token_embed

    # Add learned position embedding if applicable
    if position_type == "learned" and position_embed_table is not None:
        pos_gather = network.add_gather(position_embed_table, position_id, 0)
        pos_add = network.add_elementwise(
            hidden_state, pos_gather.get_output(0),
            trt.ElementWiseOperation.SUM)
        hidden_state = pos_add.get_output(0)

    # In BF16 mode many embedding/position constants are still materialized from
    # float16 storage. Normalize the decoder's main hidden stream back to the
    # requested runtime dtype before entering the layer stack.
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)

    # Optional embedding LayerNorm (e.g. BLOOM) — use native INormalizationLayer
    embed_norm = weights.get("embedding_norm")
    if embed_norm is not None:
        embed_norm_beta = weights.get("embedding_norm_beta")
        if embed_norm_beta is None:
            embed_norm_beta = np.zeros(hidden, dtype=work_np_dtype)
        hidden_state = graph_ops.add_layer_norm_native(
            network, hidden_state, hidden, embed_norm, embed_norm_beta,
            config.rms_norm_eps, dtype=work_np_dtype)

    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")

    # FFI attention kernel: set by the perf agent on their branch.
    # Default: None (use native TRT attention).
    ffi_attention_kernel = None

    # ---------------------------------------------------------------
    # Decoder layers
    # ---------------------------------------------------------------
    present_k_outputs = []
    present_v_outputs = []

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        layer_is_fp32 = layer_idx in fp32_layers
        layer_np_dtype = np.float32 if layer_is_fp32 else work_np_dtype
        layer_trt_dtype = trt.float32 if layer_is_fp32 else work_trt_dtype

        def _cast_layer_dtype(tensor: trt.ITensor | None) -> trt.ITensor | None:
            if tensor is None or tensor.dtype == layer_trt_dtype:
                return tensor
            return network.add_cast(tensor, layer_trt_dtype).get_output(0)

        result = _add_decoder_layer(
            network=network,
            hidden=_cast_layer_dtype(hidden_state),
            cache_k=_cast_layer_dtype(cache_k_inputs[layer_idx]),
            cache_v=_cast_layer_dtype(cache_v_inputs[layer_idx]),
            attention_mask=_cast_layer_dtype(attention_mask),
            position_id=position_id,
            attention_scale=attn_scale,
            eps_tensor=_cast_layer_dtype(eps_tensor),
            eps=config.rms_norm_eps,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden,
            attention_size=attention_size,
            kv_attention_size=kv_attention_size,
            mlp_size=mlp_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_cache_length=max_cache_length,
            norm_type=norm_type,
            mlp_type=mlp_type,
            position_type=position_type,
            activation=activation,
            parallel_residual=parallel_residual,
            alibi_slopes_tensor=alibi_slopes_tensor,
            alibi_indices_tensor=alibi_indices_tensor,
            dtype=layer_np_dtype,
            quant_ctx=quant_ctx,
            cos_half_tensor=_cast_layer_dtype(cos_half_tensor),
            sin_half_tensor=_cast_layer_dtype(sin_half_tensor),
            rotary_embedding_dim=rotary_embedding_dim,
            interleaved_rope=interleaved_rope,
            ffi_attention_kernel=ffi_attention_kernel,
            dynamic_kv_cache=dynamic_kv_cache,
        )

        hidden_state = _cast_work_dtype(result["hidden"])
        present_k_outputs.append(_cast_work_dtype(result["present_k"]))
        present_v_outputs.append(_cast_work_dtype(result["present_v"]))

        if debug_layer_outputs:
            _mark_debug_output(network, result["post_attn"], f"debug_post_attn_{layer_idx}")
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

    # ---------------------------------------------------------------
    # Final norm
    # ---------------------------------------------------------------
    final_norm = weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = _apply_norm(
            network, hidden_state, hidden, final_norm,
            weights.get("final_norm_beta"), eps_tensor, norm_type,
            dtype=work_np_dtype, eps=config.rms_norm_eps)

    # Optional: mark hidden state as extra output for speech pipelines
    if hidden_state_output:
        hs_out = network.add_identity(hidden_state).get_output(0)
        hs_out.name = "hidden_state"
        network.mark_output(hs_out)

    # ---------------------------------------------------------------
    # LM head (logits)
    # ---------------------------------------------------------------
    # Output vocab may differ from input vocab (e.g. Bark semantic: 129600 in, 10048 out).
    # Derive from w_out shape if available.
    out_vocab = weights["w_out"].shape[1] if isinstance(weights["w_out"], np.ndarray) else vocab
    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, out_vocab, weights["w_out"],
        dtype=work_np_dtype)
    # LM head bias (if present, e.g. CodeGen) or zero bias for C++ parity
    lm_bias = weights.get("lm_head_bias")
    if lm_bias is not None:
        logits = graph_ops.add_bias_sum(network, logits, out_vocab, lm_bias,
                                        dtype=work_np_dtype)
    else:
        b_out = np.zeros(out_vocab, dtype=work_np_dtype)
        logits = graph_ops.add_bias_sum(network, logits, out_vocab, b_out,
                                        dtype=work_np_dtype)

    # Logits output: always FP32 for accurate argmax/sampling
    if work_trt_dtype != trt.float32:
        logits_cast = network.add_cast(logits, trt.float32)
        logits = logits_cast.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    # ---------------------------------------------------------------
    # Present K/V outputs
    # ---------------------------------------------------------------
    for i in range(num_layers):
        pk = present_k_outputs[i]
        pv = present_v_outputs[i]
        pk.name = graph_ops.layer_tensor_name("present_k", i)
        pv.name = graph_ops.layer_tensor_name("present_v", i)
        network.mark_output(pk)
        network.mark_output(pv)

    # ---------------------------------------------------------------
    # Build engine
    # ---------------------------------------------------------------
    if verbose:
        print(f"[trtmc build] Building TRT engine ({num_layers} layers, "
              f"hidden={hidden}, attn={attention_size}, kv={kv_attention_size}, "
              f"mlp={mlp_size}, "
              f"cache={max_cache_length}, precision={precision}) ...",
              file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")

    return bytes(plan)


def _apply_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray | None,
    eps_tensor: trt.ITensor,
    norm_type: str,
    dtype: np.dtype = np.float32,
    eps: float | None = None,
) -> trt.ITensor:
    """Dispatch to RMSNorm or LayerNorm based on norm_type."""
    return graph_blocks.apply_norm(
        network, inp, hidden_size, gamma, beta, eps_tensor, norm_type,
        dtype=dtype, eps=eps)


def _add_decoder_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    attention_scale: float | None,
    eps_tensor: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    attention_size: int,
    kv_attention_size: int,
    mlp_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_cache_length: int,
    norm_type: str = "rmsnorm",
    mlp_type: str = "swiglu",
    position_type: str = "rope",
    activation: str = "silu",
    parallel_residual: bool = False,
    alibi_slopes_tensor: trt.ITensor | None = None,
    alibi_indices_tensor: trt.ITensor | None = None,
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    cos_half_tensor: trt.ITensor | None = None,
    sin_half_tensor: trt.ITensor | None = None,
    rotary_embedding_dim: int = 0,
    interleaved_rope: bool = False,
    ffi_attention_kernel: str | None = None,
    dynamic_kv_cache: bool = False,
    eps: float | None = None,
) -> dict[str, trt.ITensor]:
    """Add one standard decoder layer block. Returns hidden, present_k, present_v."""

    # Attention block (pre-norm -> QKV -> RoPE -> cache -> attn -> out proj)
    attn = graph_blocks.add_attention_block(
        network, hidden, cache_k, cache_v, attention_mask, position_id,
        weights=weights, prefix=prefix,
        hidden_size=hidden_size, attention_size=attention_size,
        kv_attention_size=kv_attention_size,
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
        max_cache_length=max_cache_length,
        attention_scale=attention_scale,
        eps_tensor=eps_tensor, eps=eps,
        norm_type=norm_type, position_type=position_type,
        alibi_slopes_tensor=alibi_slopes_tensor,
        alibi_indices_tensor=alibi_indices_tensor,
        dtype=dtype,
        quant_ctx=quant_ctx,
        layer_prefix=prefix,
        cos_half_tensor=cos_half_tensor,
        sin_half_tensor=sin_half_tensor,
        rotary_embedding_dim=rotary_embedding_dim,
        interleaved_rope=interleaved_rope,
        ffi_attention_kernel=ffi_attention_kernel,
        dynamic_kv_cache=dynamic_kv_cache,
    )
    attn_out = attn["attn_out"]
    present_k = attn["present_k"]
    present_v = attn["present_v"]

    # --- Parallel vs sequential residual ---
    if parallel_residual:
        post_attn_norm_w = weights.get(f"{prefix}.post_attn_norm")
        if post_attn_norm_w is not None:
            norm2 = _apply_norm(
                network, hidden, hidden_size,
                post_attn_norm_w,
                weights.get(f"{prefix}.post_attn_norm_beta"),
                eps_tensor, norm_type, dtype=dtype, eps=eps)
        else:
            norm2 = attn["normed"]
    else:
        residual1 = network.add_elementwise(
            hidden, attn_out, trt.ElementWiseOperation.SUM)
        norm2 = _apply_norm(
            network, residual1.get_output(0), hidden_size,
            weights[f"{prefix}.post_attn_norm"],
            weights.get(f"{prefix}.post_attn_norm_beta"),
            eps_tensor, norm_type, dtype=dtype, eps=eps)

    # MLP
    if mlp_type == "gelu_fc":
        mlp_out = graph_blocks.add_gelu_fc_mlp(
            network, norm2, weights=weights, prefix=prefix,
            hidden_size=hidden_size, mlp_size=mlp_size,
            activation=activation, dtype=dtype,
            quant_ctx=quant_ctx, layer_prefix=prefix)
    else:
        mlp_out = graph_blocks.add_swiglu_mlp(
            network, norm2, weights=weights, prefix=prefix,
            hidden_size=hidden_size, mlp_size=mlp_size, dtype=dtype,
            quant_ctx=quant_ctx, layer_prefix=prefix)

    # Final residual connection
    if parallel_residual:
        sum_attn = network.add_elementwise(
            hidden, attn_out, trt.ElementWiseOperation.SUM)
        residual2 = network.add_elementwise(
            sum_attn.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)
        post_attn_tensor = sum_attn.get_output(0)
    else:
        residual2 = network.add_elementwise(
            residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)
        post_attn_tensor = residual1.get_output(0)

    return {
        "hidden": residual2.get_output(0),
        "post_attn": post_attn_tensor,
        "present_k": present_k,
        "present_v": present_v,
    }
