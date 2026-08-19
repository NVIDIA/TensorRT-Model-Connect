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
from .default_dual_profile_decoder import build_dual_profile_decoder_engine
from .utils import const_in_work_dtype, create_builder_context

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


def build_standard_decoder_engine(config: ModelConfig, weights: WeightDict, max_cache_length: int, *, precision: str='fp32', quant_ctx: QuantContext | None=None, activation: str='silu', partial_rotary_factor: float=1.0, interleaved_rope: bool=False, parallel_residual: bool=False, scale_attn_weights: bool=True, embed_input: bool=False, verbose: bool=False, debug_layer_outputs: bool=False, hidden_state_output: bool=False) -> bytes:
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
    config.raw['_decoder_engine_layout_supported'] = True
    decoder_engine_role = str(config.raw.get('_decoder_engine_role', 'dual_profile'))
    _dual_profile_disabled_for = embed_input or debug_layer_outputs or hidden_state_output or bool(config.raw.get('dynamic_kv_cache', False)) or (_os.environ.get('TRTMC_NO_DUAL_PROFILE') == '1')
    if decoder_engine_role == 'prefill' and _dual_profile_disabled_for:
        raise NotImplementedError('split prefill engine is not supported for this standard decoder configuration')
    if not _dual_profile_disabled_for and decoder_engine_role in ('dual_profile', 'prefill'):
        return build_dual_profile_decoder_engine(config, weights, max_cache_length, precision=precision, quant_ctx=quant_ctx, activation=activation, partial_rotary_factor=partial_rotary_factor, interleaved_rope=interleaved_rope, parallel_residual=parallel_residual, scale_attn_weights=scale_attn_weights, verbose=verbose, profile_mode='prefill' if decoder_engine_role == 'prefill' else 'dual_profile')
    attention_size: int = weights.get('_attention_size', config.attention_size)
    mlp_size: int = weights.get('_mlp_size', config.intermediate_size)
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = attention_size // num_heads
    kv_attention_size = graph_blocks.infer_kv_attention_size(weights, num_kv_heads=num_kv_heads, head_dim=head_dim)
    attention_window = max_cache_length + 1
    dynamic_kv_cache = bool(config.raw.get('dynamic_kv_cache', False))
    dynamic_kv_opt_rows = int(config.raw.get('_dynamic_kv_opt_length', max_cache_length))
    dynamic_kv_opt_rows = max(1, min(dynamic_kv_opt_rows, max_cache_length))
    raw_profile_rows = config.raw.get('_dynamic_kv_profile_rows')
    if raw_profile_rows:
        dynamic_kv_profile_rows = []
        for row in raw_profile_rows:
            clamped = max(1, min(int(row), max_cache_length))
            if clamped not in dynamic_kv_profile_rows:
                dynamic_kv_profile_rows.append(clamped)
        dynamic_kv_profile_rows.sort()
    else:
        dynamic_kv_profile_rows = []
    builder_context = create_builder_context(verbose=verbose, workspace_bytes=1 << 30)
    builder = builder_context.builder
    network = builder_context.network
    trt_config = builder_context.config
    if precision == 'fp16':
        work_np_dtype = np.float16
        work_trt_dtype = trt.float16
    elif precision == 'bf16':
        work_np_dtype = np.float16
        work_trt_dtype = trt.bfloat16
    else:
        work_np_dtype = np.float32
        work_trt_dtype = trt.float32
    token_id = network.add_input('token_id', trt.int32, (1,))
    position_id = network.add_input('position_id', trt.int32, (1,))
    attention_mask = network.add_input('attention_mask', trt.float32, (1, -1) if dynamic_kv_cache else (1, attention_window))
    input_embed_tensor = None
    use_input_embed_tensor = None
    if embed_input:
        input_embed_tensor = network.add_input('input_embed', trt.float32, (1, hidden))
        use_input_embed_tensor = network.add_input('use_input_embed', trt.float32, (1,))
    cache_k_inputs = []
    cache_v_inputs = []
    for i in range(num_layers):
        ck = network.add_input(graph_ops.layer_tensor_name('cache_k', i), work_trt_dtype, (-1, kv_attention_size) if dynamic_kv_cache else (max_cache_length, kv_attention_size))
        cv = network.add_input(graph_ops.layer_tensor_name('cache_v', i), work_trt_dtype, (-1, kv_attention_size) if dynamic_kv_cache else (max_cache_length, kv_attention_size))
        cache_k_inputs.append(ck)
        cache_v_inputs.append(cv)
    if dynamic_kv_cache:
        if dynamic_kv_profile_rows:
            for profile_rows in dynamic_kv_profile_rows:
                profile = builder.create_optimization_profile()
                min_rows = 1
                profile.set_shape('attention_mask', (1, min_rows + 1), (1, profile_rows + 1), (1, profile_rows + 1))
                for i in range(num_layers):
                    min_cache_shape = (min_rows, kv_attention_size)
                    cache_shape = (profile_rows, kv_attention_size)
                    profile.set_shape(graph_ops.layer_tensor_name('cache_k', i), min_cache_shape, cache_shape, cache_shape)
                    profile.set_shape(graph_ops.layer_tensor_name('cache_v', i), min_cache_shape, cache_shape, cache_shape)
                trt_config.add_optimization_profile(profile)
        else:
            profile = builder.create_optimization_profile()
            profile.set_shape('attention_mask', (1, 2), (1, dynamic_kv_opt_rows + 1), (1, attention_window))
            for i in range(num_layers):
                profile.set_shape(graph_ops.layer_tensor_name('cache_k', i), (1, kv_attention_size), (dynamic_kv_opt_rows, kv_attention_size), (max_cache_length, kv_attention_size))
                profile.set_shape(graph_ops.layer_tensor_name('cache_v', i), (1, kv_attention_size), (dynamic_kv_opt_rows, kv_attention_size), (max_cache_length, kv_attention_size))
            trt_config.add_optimization_profile(profile)
    if work_trt_dtype != trt.float32:
        mask_cast = network.add_cast(attention_mask, work_trt_dtype)
        attention_mask = mask_cast.get_output(0)

    def _cast_work_dtype(tensor: trt.ITensor) -> trt.ITensor:
        if tensor.dtype == work_trt_dtype:
            return tensor
        return network.add_cast(tensor, work_trt_dtype).get_output(0)
    embedding_table = graph_ops.add_constant(network, (vocab, hidden), weights['embedding'], dtype=work_np_dtype)
    alibi_slopes_tensor = None
    alibi_indices_tensor = None
    cos_half_tensor = None
    sin_half_tensor = None
    rotary_embedding_dim = int(head_dim * partial_rotary_factor)
    graph_ops.validate_native_rope_dim(rotary_embedding_dim)
    cos_half_np = graph_ops.make_rope_table_half_dim(attention_window, head_dim, config.rope_theta, True, partial_rotary_factor, interleaved=interleaved_rope)
    sin_half_np = graph_ops.make_rope_table_half_dim(attention_window, head_dim, config.rope_theta, False, partial_rotary_factor, interleaved=interleaved_rope)
    cos_half_tensor = graph_ops.add_constant(network, cos_half_np.shape, cos_half_np, dtype=work_np_dtype)
    cos_half_tensor = _cast_work_dtype(cos_half_tensor)
    sin_half_tensor = graph_ops.add_constant(network, sin_half_np.shape, sin_half_np, dtype=work_np_dtype)
    sin_half_tensor = _cast_work_dtype(sin_half_tensor)
    eps_tensor = graph_ops.add_constant(network, (1, 1), np.array([config.rms_norm_eps], dtype=work_np_dtype), dtype=work_np_dtype)
    attn_scale = 1.0 / np.sqrt(max(head_dim, 1)) if scale_attn_weights else 1.0
    gather = network.add_gather(embedding_table, token_id, 0)
    token_embed = gather.get_output(0)
    if embed_input and input_embed_tensor is not None and (use_input_embed_tensor is not None):
        token_embed_for_math = _cast_work_dtype(token_embed)
        flag_broadcast = network.add_shuffle(use_input_embed_tensor)
        flag_broadcast.reshape_dims = (1, 1)
        flag_for_math = flag_broadcast.get_output(0)
        if work_trt_dtype != trt.float32:
            flag_for_math = network.add_cast(flag_for_math, work_trt_dtype).get_output(0)
        one_const = const_in_work_dtype(network, (1, 1), np.array([1.0], dtype=work_np_dtype), work_np_dtype, work_trt_dtype)
        inv_flag = network.add_elementwise(one_const, flag_for_math, trt.ElementWiseOperation.SUB)
        tok_part = network.add_elementwise(inv_flag.get_output(0), token_embed_for_math, trt.ElementWiseOperation.PROD)
        input_embed_for_math = input_embed_tensor
        if input_embed_for_math.dtype != work_trt_dtype:
            input_embed_for_math = network.add_cast(input_embed_for_math, work_trt_dtype).get_output(0)
        embed_part = network.add_elementwise(flag_for_math, input_embed_for_math, trt.ElementWiseOperation.PROD)
        hidden_state_sum = network.add_elementwise(tok_part.get_output(0), embed_part.get_output(0), trt.ElementWiseOperation.SUM)
        hidden_state = hidden_state_sum.get_output(0)
    else:
        hidden_state = token_embed
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)
    embed_norm = weights.get('embedding_norm')
    if embed_norm is not None:
        embed_norm_beta = weights.get('embedding_norm_beta')
        if embed_norm_beta is None:
            embed_norm_beta = np.zeros(hidden, dtype=work_np_dtype)
        hidden_state = graph_ops.add_layer_norm_native(network, hidden_state, hidden, embed_norm, embed_norm_beta, config.rms_norm_eps, dtype=work_np_dtype)
    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, 'debug_embed')
    ffi_attention_kernel = None
    present_k_outputs = []
    present_v_outputs = []
    for layer_idx in range(num_layers):
        prefix = f'layer.{layer_idx}'
        result = _add_decoder_layer(network=network, hidden=hidden_state, cache_k=cache_k_inputs[layer_idx], cache_v=cache_v_inputs[layer_idx], attention_mask=attention_mask, position_id=position_id, attention_scale=attn_scale, eps_tensor=eps_tensor, eps=config.rms_norm_eps, weights=weights, prefix=prefix, hidden_size=hidden, attention_size=attention_size, kv_attention_size=kv_attention_size, mlp_size=mlp_size, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim, max_cache_length=max_cache_length, activation=activation, parallel_residual=parallel_residual, alibi_slopes_tensor=alibi_slopes_tensor, alibi_indices_tensor=alibi_indices_tensor, dtype=work_np_dtype, quant_ctx=quant_ctx, cos_half_tensor=cos_half_tensor, sin_half_tensor=sin_half_tensor, rotary_embedding_dim=rotary_embedding_dim, interleaved_rope=interleaved_rope, ffi_attention_kernel=ffi_attention_kernel, dynamic_kv_cache=dynamic_kv_cache)
        hidden_state = result['hidden']
        present_k_outputs.append(result['present_k'])
        present_v_outputs.append(result['present_v'])
        if debug_layer_outputs:
            _mark_debug_output(network, result['post_attn'], f'debug_post_attn_{layer_idx}')
            _mark_debug_output(network, hidden_state, f'debug_hidden_{layer_idx}')
    final_norm = weights.get('final_norm')
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = _apply_norm(network, hidden_state, hidden, final_norm, weights.get('final_norm_beta'), eps_tensor, 'layernorm', dtype=work_np_dtype, eps=config.rms_norm_eps)
    if hidden_state_output:
        hs_out = network.add_identity(hidden_state).get_output(0)
        hs_out.name = 'hidden_state'
        network.mark_output(hs_out)
    out_vocab = weights['w_out'].shape[1] if isinstance(weights['w_out'], np.ndarray) else vocab
    logits = graph_ops.add_matmul_rhs_constant(network, hidden_state, hidden, out_vocab, weights['w_out'], dtype=work_np_dtype)
    lm_bias = weights.get('lm_head_bias')
    if lm_bias is not None:
        logits = graph_ops.add_bias_sum(network, logits, out_vocab, lm_bias, dtype=work_np_dtype)
    else:
        b_out = np.zeros(out_vocab, dtype=work_np_dtype)
        logits = graph_ops.add_bias_sum(network, logits, out_vocab, b_out, dtype=work_np_dtype)
    if work_trt_dtype != trt.float32:
        logits_cast = network.add_cast(logits, trt.float32)
        logits = logits_cast.get_output(0)
    logits.name = 'logits'
    network.mark_output(logits)
    for i in range(num_layers):
        pk = present_k_outputs[i]
        pv = present_v_outputs[i]
        pk.name = graph_ops.layer_tensor_name('present_k', i)
        pv.name = graph_ops.layer_tensor_name('present_v', i)
        network.mark_output(pk)
        network.mark_output(pv)
    if verbose:
        print(f'[trtmc build] Building TRT engine ({num_layers} layers, hidden={hidden}, attn={attention_size}, kv={kv_attention_size}, mlp={mlp_size}, cache={max_cache_length}, precision={precision}) ...', file=sys.stderr)
    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError('TensorRT engine build failed')
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


def _add_decoder_layer(*, network: trt.INetworkDefinition, hidden: trt.ITensor, cache_k: trt.ITensor, cache_v: trt.ITensor, attention_mask: trt.ITensor, position_id: trt.ITensor, attention_scale: float | None, eps_tensor: trt.ITensor, weights: WeightDict, prefix: str, hidden_size: int, attention_size: int, kv_attention_size: int, mlp_size: int, num_heads: int, num_kv_heads: int, head_dim: int, max_cache_length: int, activation: str='silu', parallel_residual: bool=False, alibi_slopes_tensor: trt.ITensor | None=None, alibi_indices_tensor: trt.ITensor | None=None, dtype: np.dtype=np.float32, quant_ctx: QuantContext | None=None, cos_half_tensor: trt.ITensor | None=None, sin_half_tensor: trt.ITensor | None=None, rotary_embedding_dim: int=0, interleaved_rope: bool=False, ffi_attention_kernel: str | None=None, dynamic_kv_cache: bool=False, eps: float | None=None) -> dict[str, trt.ITensor]:
    """Add one standard decoder layer block. Returns hidden, present_k, present_v."""
    attn = graph_blocks.add_attention_block(network, hidden, cache_k, cache_v, attention_mask, position_id, weights=weights, prefix=prefix, hidden_size=hidden_size, attention_size=attention_size, kv_attention_size=kv_attention_size, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim, max_cache_length=max_cache_length, attention_scale=attention_scale, eps_tensor=eps_tensor, eps=eps, alibi_slopes_tensor=alibi_slopes_tensor, alibi_indices_tensor=alibi_indices_tensor, dtype=dtype, quant_ctx=quant_ctx, layer_prefix=prefix, cos_half_tensor=cos_half_tensor, sin_half_tensor=sin_half_tensor, rotary_embedding_dim=rotary_embedding_dim, interleaved_rope=interleaved_rope, ffi_attention_kernel=ffi_attention_kernel, dynamic_kv_cache=dynamic_kv_cache)
    attn_out = attn['attn_out']
    present_k = attn['present_k']
    present_v = attn['present_v']
    if parallel_residual:
        post_attn_norm_w = weights.get(f'{prefix}.post_attn_norm')
        if post_attn_norm_w is not None:
            norm2 = _apply_norm(network, hidden, hidden_size, post_attn_norm_w, weights.get(f'{prefix}.post_attn_norm_beta'), eps_tensor, 'layernorm', dtype=dtype, eps=eps)
        else:
            norm2 = attn['normed']
    else:
        residual1 = network.add_elementwise(hidden, attn_out, trt.ElementWiseOperation.SUM)
        norm2 = _apply_norm(network, residual1.get_output(0), hidden_size, weights[f'{prefix}.post_attn_norm'], weights.get(f'{prefix}.post_attn_norm_beta'), eps_tensor, 'layernorm', dtype=dtype, eps=eps)
    mlp_out = graph_blocks.add_swiglu_mlp(network, norm2, weights=weights, prefix=prefix, hidden_size=hidden_size, mlp_size=mlp_size, dtype=dtype, quant_ctx=quant_ctx, layer_prefix=prefix)
    if parallel_residual:
        sum_attn = network.add_elementwise(hidden, attn_out, trt.ElementWiseOperation.SUM)
        residual2 = network.add_elementwise(sum_attn.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)
        post_attn_tensor = sum_attn.get_output(0)
    else:
        residual2 = network.add_elementwise(residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)
        post_attn_tensor = residual1.get_output(0)
    return {'hidden': residual2.get_output(0), 'post_attn': post_attn_tensor, 'present_k': present_k, 'present_v': present_v}
