# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-2 tensor-parallel prefill/decode engine builder."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from ....parallel_config import (
    add_all_reduce_sum,
    normalize_parallel_config,
    shard_standard_decoder_weights,
)
from .model import (
    add_2d_mask_to_4d,
    add_attention_from_rows,
    add_bias_sum,
    add_constant,
    add_gelu_fc_projection,
    add_last_token_logits,
    const_in_work_dtype,
    create_builder_context,
    layer_tensor_name,
    make_matmul_fn,
    norm_multi,
    work_dtypes,
)

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ....quantization.context import QuantContext
    from ..config import ModelConfig
    from ..weights import WeightDict


def _validate_tp_quantization(quant_ctx: QuantContext | None) -> None:
    if quant_ctx is None:
        return
    format_name = getattr(getattr(quant_ctx, "profile", None), "format", None)
    if getattr(format_name, "name", None) != "fp8":
        raise ValueError("Tensor-parallel decoder quantization currently supports fp8 only")


def build_dual_profile_tp_decoder_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "fp16",
    opt_prefill_length: int = 64,
    max_prefill_length: int | None = None,
    quant_ctx: QuantContext | None = None,
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    """Build one rank-local GPT-2 engine with prefill/decode profiles."""
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError(
            "GPT-2 tensor-parallel builds require mode=tensor_parallel and tp_size > 1"
        )
    _validate_tp_quantization(quant_ctx)
    weights = shard_standard_decoder_weights(config, weights, parallel)
    if max_prefill_length is None:
        max_prefill_length = max_cache_length
    max_prefill_length = max(1, min(max_prefill_length, max_cache_length))
    opt_prefill_length = max(1, min(opt_prefill_length, max_prefill_length))

    attention_size = weights["_attention_size"]
    mlp_size = weights["_mlp_size"]
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads // parallel.tp_size
    head_dim = attention_size // num_heads
    builder_context = create_builder_context(verbose=verbose)
    builder = builder_context.builder
    network = builder_context.network
    trt_config = builder_context.config
    work_np_dtype, work_trt_dtype = work_dtypes(precision)

    token_id = network.add_input("token_id", trt.int32, (-1,))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    attention_mask = network.add_input("attention_mask", trt.float32, (-1, -1))
    cache_shape = (max_cache_length, attention_size)
    cache_k_inputs = [
        network.add_input(layer_tensor_name("cache_k", i), work_trt_dtype, cache_shape)
        for i in range(num_layers)
    ]
    cache_v_inputs = [
        network.add_input(layer_tensor_name("cache_v", i), work_trt_dtype, cache_shape)
        for i in range(num_layers)
    ]
    attention_mask_work = attention_mask
    if work_trt_dtype != trt.float32:
        attention_mask_work = network.add_cast(attention_mask, work_trt_dtype).get_output(0)

    def _add_profile(opt_sq: int, max_sq: int, *, fixed: bool = False) -> None:
        profile = builder.create_optimization_profile()
        min_sq = opt_sq if fixed else 1
        profile.set_shape("token_id", (min_sq,), (opt_sq,), (max_sq,))
        profile.set_shape("position_id", (min_sq,), (opt_sq,), (max_sq,))
        profile.set_shape(
            "attention_mask",
            (min_sq, max_cache_length + min_sq),
            (opt_sq, max_cache_length + opt_sq),
            (max_sq, max_cache_length + max_sq),
        )
        trt_config.add_optimization_profile(profile)

    _add_profile(opt_prefill_length, max_prefill_length)
    _add_profile(1, 1, fixed=True)

    embedding_table = const_in_work_dtype(
        network,
        (vocab, hidden),
        weights["embedding"],
        work_np_dtype,
        work_trt_dtype,
    )
    position_weights = weights["position_embedding"]
    position_table = const_in_work_dtype(
        network,
        position_weights.shape,
        position_weights,
        work_np_dtype,
        work_trt_dtype,
    )
    eps_tensor = add_constant(
        network,
        (1, 1),
        np.array([[config.rms_norm_eps]], dtype=np.float32),
        dtype=np.float32,
    )
    matmul = make_matmul_fn(network, work_np_dtype, quant_ctx)
    token_embedding = network.add_gather(embedding_table, token_id, 0).get_output(0)
    position_embedding = network.add_gather(position_table, position_id, 0).get_output(0)
    hidden_state = network.add_elementwise(
        token_embedding, position_embedding, trt.ElementWiseOperation.SUM
    ).get_output(0)
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)

    mask_4d = add_2d_mask_to_4d(network, attention_mask_work)
    attention_scale = 1.0 / np.sqrt(max(head_dim, 1))
    present_k_outs: list[trt.ITensor] = []
    present_v_outs: list[trt.ITensor] = []
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        normed = norm_multi(
            network,
            hidden_state,
            hidden,
            weights[f"{prefix}.input_norm"],
            weights[f"{prefix}.input_norm_beta"],
            eps_tensor,
            work_np_dtype,
        )
        q = matmul(
            normed,
            hidden,
            attention_size,
            weights[f"{prefix}.w_q"],
            f"{prefix}.w_q",
        )
        k = matmul(
            normed,
            hidden,
            attention_size,
            weights[f"{prefix}.w_k"],
            f"{prefix}.w_k",
        )
        v = matmul(
            normed,
            hidden,
            attention_size,
            weights[f"{prefix}.w_v"],
            f"{prefix}.w_v",
        )
        q = add_bias_sum(
            network,
            q,
            attention_size,
            weights[f"{prefix}.q_bias"],
            dtype=work_np_dtype,
        )
        k = add_bias_sum(
            network,
            k,
            attention_size,
            weights[f"{prefix}.k_bias"],
            dtype=work_np_dtype,
        )
        v = add_bias_sum(
            network,
            v,
            attention_size,
            weights[f"{prefix}.v_bias"],
            dtype=work_np_dtype,
        )
        present_k_outs.append(k)
        present_v_outs.append(v)
        all_k = network.add_concatenation([cache_k_inputs[layer_idx], k])
        all_k.axis = 0
        all_v = network.add_concatenation([cache_v_inputs[layer_idx], v])
        all_v.axis = 0
        context = add_attention_from_rows(
            network,
            q,
            all_k.get_output(0),
            all_v.get_output(0),
            num_heads=num_heads,
            head_dim=head_dim,
            q_seq=None,
            kv_seq=None,
            mask=mask_4d,
            scale=attention_scale,
            tag=f"{prefix}.attn",
        )
        attn_out = matmul(
            context,
            attention_size,
            hidden,
            weights[f"{prefix}.w_o"],
            f"{prefix}.w_o",
        )
        attn_out = add_all_reduce_sum(network, attn_out, parallel.tp_size)
        attn_out = add_bias_sum(
            network,
            attn_out,
            hidden,
            weights[f"{prefix}.o_bias"],
            dtype=work_np_dtype,
        )
        residual = network.add_elementwise(
            hidden_state, attn_out, trt.ElementWiseOperation.SUM
        ).get_output(0)
        normed = norm_multi(
            network,
            residual,
            hidden,
            weights[f"{prefix}.post_attn_norm"],
            weights[f"{prefix}.post_attn_norm_beta"],
            eps_tensor,
            work_np_dtype,
        )
        mlp_out = add_gelu_fc_projection(
            network,
            normed,
            matmul=matmul,
            weights=weights,
            prefix=prefix,
            hidden=hidden,
            mlp_size=mlp_size,
            dtype=work_np_dtype,
        )
        mlp_out = add_all_reduce_sum(network, mlp_out, parallel.tp_size)
        mlp_out = add_bias_sum(
            network,
            mlp_out,
            hidden,
            weights[f"{prefix}.fc2_bias"],
            dtype=work_np_dtype,
        )
        hidden_state = network.add_elementwise(
            residual, mlp_out, trt.ElementWiseOperation.SUM
        ).get_output(0)

    hidden_state = norm_multi(
        network,
        hidden_state,
        hidden,
        weights["final_norm"],
        weights["final_norm_beta"],
        eps_tensor,
        work_np_dtype,
    )
    logits = add_last_token_logits(network, hidden_state, hidden, weights["w_out"], work_np_dtype)
    if work_trt_dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    for layer_idx, (present_k, present_v) in enumerate(zip(present_k_outs, present_v_outs)):
        present_k.name = layer_tensor_name("present_k", layer_idx)
        present_v.name = layer_tensor_name("present_v", layer_idx)
        network.mark_output(present_k)
        network.mark_output(present_v)
    if verbose:
        print(
            f"[trtmc build] Building GPT-2 TP engine (layers={num_layers}, "
            f"hidden={hidden}, local_attn={attention_size}, "
            f"local_mlp={mlp_size}, cache={max_cache_length}, "
            f"precision={precision}, tp={parallel.tp_size}, rank={parallel.rank}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("GPT-2 tensor-parallel engine build failed")
    return bytes(plan)
