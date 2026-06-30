"""Tensor-parallel PersonaPlex temporal decoder builder.

This mirrors the single-device standard decoder path used by PersonaPlex's
temporal transformer. Tensor parallelism only changes the decoder projections:
Q/K/V and SwiGLU gate/up are column-parallel, output/down are row-parallel and
joined with TensorRT distributed ALL_REDUCE.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import model as graph_blocks
from . import model as graph_ops
from ....parallel_config import (
    add_all_reduce_sum,
    normalize_parallel_config,
    shard_standard_decoder_weights,
)

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ..weights import WeightDict
    from ..config import ModelConfig


def _mark_debug_output(network: trt.INetworkDefinition, tensor: trt.ITensor, name: str) -> None:
    cast = network.add_cast(tensor, trt.float32)
    out = cast.get_output(0)
    out.name = name
    network.mark_output(out)


def _ensure_tp_metadata(config: "ModelConfig", weights: "WeightDict") -> "WeightDict":
    if "_kv_attention_size" in weights:
        return weights
    copied = type(weights)(weights)
    copied["_kv_attention_size"] = int(weights.get("_attention_size", config.attention_size))
    return copied


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
    return graph_blocks.apply_norm(
        network, inp, hidden_size, gamma, beta, eps_tensor, norm_type, dtype=dtype, eps=eps
    )


def build_personaplex_tp_decoder_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx=None,
    activation: str = "silu",
    partial_rotary_factor: float = 1.0,
    parallel_residual: bool = False,
    scale_attn_weights: bool = True,
    verbose: bool = False,
    debug_layer_outputs: bool = False,
    parallel_config=None,
) -> bytes:
    """Build one rank-local PersonaPlex temporal decoder engine."""
    if quant_ctx is not None:
        raise ValueError("PersonaPlex tensor-parallel builds do not support quantization")
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError("PersonaPlex tensor-parallel builder requires an enabled parallel config")
    if parallel.rank < 0:
        raise ValueError("PersonaPlex tensor-parallel engine build requires a concrete rank")
    weights = _ensure_tp_metadata(config, weights)
    weights = shard_standard_decoder_weights(config, weights, parallel)
    attention_size = int(weights.get("_attention_size", config.attention_size))
    mlp_size = int(weights.get("_mlp_size", config.intermediate_size))
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads // parallel.tp_size
    num_kv_heads = config.num_key_value_heads // parallel.tp_size
    head_dim = attention_size // num_heads
    kv_attention_size = graph_blocks.infer_kv_attention_size(
        weights, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    attention_window = max_cache_length + 1
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if precision == "fp16":
        work_np_dtype = np.float16
        work_trt_dtype = trt.float16
    elif precision == "bf16":
        work_np_dtype = np.float16
        work_trt_dtype = trt.bfloat16
    else:
        work_np_dtype = np.float32
        work_trt_dtype = trt.float32
    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input("attention_mask", trt.float32, (1, attention_window))
    input_embed_tensor = None
    use_input_embed_tensor = None
    input_embed_tensor = network.add_input("input_embed", work_trt_dtype, (1, hidden))
    use_input_embed_tensor = network.add_input("use_input_embed", trt.float32, (1,))
    cache_k_inputs = []
    cache_v_inputs = []
    for i in range(num_layers):
        cache_k_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("cache_k", i),
                work_trt_dtype,
                (max_cache_length, kv_attention_size),
            )
        )
        cache_v_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("cache_v", i),
                work_trt_dtype,
                (max_cache_length, kv_attention_size),
            )
        )
    if work_trt_dtype != trt.float32:
        attention_mask = network.add_cast(attention_mask, work_trt_dtype).get_output(0)

    def _cast_work_dtype(tensor: trt.ITensor) -> trt.ITensor:
        if tensor.dtype == work_trt_dtype:
            return tensor
        return network.add_cast(tensor, work_trt_dtype).get_output(0)

    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype
    )
    cos_half_tensor = None
    sin_half_tensor = None
    rotary_embedding_dim = int(head_dim * partial_rotary_factor)
    graph_ops.validate_native_rope_dim(rotary_embedding_dim)
    cos_half_np = graph_ops.make_rope_table_half_dim(
        attention_window, head_dim, config.rope_theta, True, partial_rotary_factor, interleaved=True
    )
    sin_half_np = graph_ops.make_rope_table_half_dim(
        attention_window,
        head_dim,
        config.rope_theta,
        False,
        partial_rotary_factor,
        interleaved=True,
    )
    cos_half_tensor = _cast_work_dtype(
        graph_ops.add_constant(network, cos_half_np.shape, cos_half_np, dtype=work_np_dtype)
    )
    sin_half_tensor = _cast_work_dtype(
        graph_ops.add_constant(network, sin_half_np.shape, sin_half_np, dtype=work_np_dtype)
    )
    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=work_np_dtype), dtype=work_np_dtype
    )
    attn_scale = 1.0 / np.sqrt(max(head_dim, 1)) if scale_attn_weights else 1.0
    token_embed = network.add_gather(embedding_table, token_id, 0).get_output(0)
    if input_embed_tensor is not None and use_input_embed_tensor is not None:
        flag_broadcast = network.add_shuffle(use_input_embed_tensor)
        flag_broadcast.reshape_dims = (1, 1)
        flag_for_math = flag_broadcast.get_output(0)
        if work_trt_dtype != trt.float32:
            flag_for_math = network.add_cast(flag_for_math, work_trt_dtype).get_output(0)
        one_const = graph_ops.add_constant(
            network, (1, 1), np.array([1.0], dtype=work_np_dtype), dtype=work_np_dtype
        )
        inv_flag = network.add_elementwise(
            one_const, flag_for_math, trt.ElementWiseOperation.SUB
        ).get_output(0)
        tok_part = network.add_elementwise(
            inv_flag, token_embed, trt.ElementWiseOperation.PROD
        ).get_output(0)
        embed_part = network.add_elementwise(
            flag_for_math, input_embed_tensor, trt.ElementWiseOperation.PROD
        ).get_output(0)
        hidden_state = network.add_elementwise(
            tok_part, embed_part, trt.ElementWiseOperation.SUM
        ).get_output(0)
    else:
        hidden_state = token_embed
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)
    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")
    present_k_outputs = []
    present_v_outputs = []
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        result = _add_tp_decoder_layer(
            network=network,
            hidden=hidden_state,
            cache_k=cache_k_inputs[layer_idx],
            cache_v=cache_v_inputs[layer_idx],
            attention_mask=attention_mask,
            position_id=position_id,
            attention_scale=attn_scale,
            eps_tensor=eps_tensor,
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
            activation=activation,
            parallel_residual=parallel_residual,
            dtype=work_np_dtype,
            quant_ctx=quant_ctx,
            cos_half_tensor=cos_half_tensor,
            sin_half_tensor=sin_half_tensor,
            rotary_embedding_dim=rotary_embedding_dim,
            tp_size=parallel.tp_size,
        )
        hidden_state = result["hidden"]
        present_k_outputs.append(result["present_k"])
        present_v_outputs.append(result["present_v"])
        if debug_layer_outputs:
            _mark_debug_output(network, result["post_attn"], f"debug_post_attn_{layer_idx}")
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")
    final_norm = weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = _apply_norm(
            network,
            hidden_state,
            hidden,
            final_norm,
            weights.get("final_norm_beta"),
            eps_tensor,
            "rmsnorm",
            dtype=work_np_dtype,
            eps=config.rms_norm_eps,
        )
    hs_out = network.add_identity(hidden_state).get_output(0)
    hs_out.name = "hidden_state"
    network.mark_output(hs_out)
    out_vocab = weights["w_out"].shape[1] if isinstance(weights["w_out"], np.ndarray) else vocab
    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, out_vocab, weights["w_out"], dtype=work_np_dtype
    )
    lm_bias = weights.get("lm_head_bias")
    if lm_bias is not None:
        logits = graph_ops.add_bias_sum(network, logits, out_vocab, lm_bias, dtype=work_np_dtype)
    else:
        logits = graph_ops.add_bias_sum(
            network,
            logits,
            out_vocab,
            np.zeros(out_vocab, dtype=work_np_dtype),
            dtype=work_np_dtype,
        )
    if work_trt_dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    for i in range(num_layers):
        present_k_outputs[i].name = graph_ops.layer_tensor_name("present_k", i)
        present_v_outputs[i].name = graph_ops.layer_tensor_name("present_v", i)
        network.mark_output(present_k_outputs[i])
        network.mark_output(present_v_outputs[i])
    if verbose:
        print(
            f"[trtmc build] Building PersonaPlex temporal TP rank {parallel.rank}/{parallel.tp_size} ({num_layers}L, h={hidden}, local_heads={num_heads}, local_attn={attention_size}, local_mlp={mlp_size}, cache={max_cache_length}, precision={precision}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT PersonaPlex tensor-parallel temporal build failed")
    return bytes(plan)


def _add_tp_decoder_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    attention_scale: float | None,
    eps_tensor: trt.ITensor,
    weights: "WeightDict",
    prefix: str,
    hidden_size: int,
    attention_size: int,
    kv_attention_size: int,
    mlp_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_cache_length: int,
    activation: str,
    parallel_residual: bool,
    dtype: np.dtype,
    quant_ctx,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    rotary_embedding_dim: int,
    tp_size: int,
    eps: float | None = None,
) -> dict[str, trt.ITensor]:
    attn = graph_blocks.add_attention_block(
        network,
        hidden,
        cache_k,
        cache_v,
        attention_mask,
        position_id,
        weights=weights,
        prefix=prefix,
        hidden_size=hidden_size,
        attention_size=attention_size,
        kv_attention_size=kv_attention_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_cache_length=max_cache_length,
        attention_scale=attention_scale,
        eps_tensor=eps_tensor,
        eps=eps,
        position_type="rope",
        alibi_slopes_tensor=None,
        alibi_indices_tensor=None,
        dtype=dtype,
        quant_ctx=quant_ctx,
        layer_prefix=prefix,
        cos_half_tensor=cos_half_tensor,
        sin_half_tensor=sin_half_tensor,
        rotary_embedding_dim=rotary_embedding_dim,
        interleaved_rope=True,
        ffi_attention_kernel=None,
        dynamic_kv_cache=False,
    )
    attn_out = add_all_reduce_sum(network, attn["attn_out"], tp_size)
    if parallel_residual:
        post_attn_norm_w = weights.get(f"{prefix}.post_attn_norm")
        if post_attn_norm_w is not None:
            norm2 = _apply_norm(
                network,
                hidden,
                hidden_size,
                post_attn_norm_w,
                weights.get(f"{prefix}.post_attn_norm_beta"),
                eps_tensor,
                "rmsnorm",
                dtype=dtype,
                eps=eps,
            )
        else:
            norm2 = attn["normed"]
    else:
        residual1 = network.add_elementwise(
            hidden, attn_out, trt.ElementWiseOperation.SUM
        ).get_output(0)
        norm2 = _apply_norm(
            network,
            residual1,
            hidden_size,
            weights[f"{prefix}.post_attn_norm"],
            weights.get(f"{prefix}.post_attn_norm_beta"),
            eps_tensor,
            "rmsnorm",
            dtype=dtype,
            eps=eps,
        )
    mlp_out = graph_blocks.add_swiglu_mlp(
        network,
        norm2,
        weights=weights,
        prefix=prefix,
        hidden_size=hidden_size,
        mlp_size=mlp_size,
        dtype=dtype,
        quant_ctx=quant_ctx,
        layer_prefix=prefix,
    )
    mlp_out = add_all_reduce_sum(network, mlp_out, tp_size)
    if parallel_residual:
        sum_attn = network.add_elementwise(
            hidden, attn_out, trt.ElementWiseOperation.SUM
        ).get_output(0)
        residual2 = network.add_elementwise(
            sum_attn, mlp_out, trt.ElementWiseOperation.SUM
        ).get_output(0)
        post_attn_tensor = sum_attn
    else:
        residual2 = network.add_elementwise(
            residual1, mlp_out, trt.ElementWiseOperation.SUM
        ).get_output(0)
        post_attn_tensor = residual1
    return {
        "hidden": residual2,
        "post_attn": post_attn_tensor,
        "present_k": attn["present_k"],
        "present_v": attn["present_v"],
    }
