# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-native GLM decoder builder.

GLM emits two single-profile engines with the same graph and weights:

* ``prefill`` accepts a dynamic token chunk up to 32K tokens.
* ``decode`` accepts exactly one token.

Both engines use TensorRT's native KV update layer. The cache capacity is the
checkpoint's full advertised context and every cache/present pair aliases one
runtime-owned BF16 buffer.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_blocks, graph_ops
from .build_routing import resolved_head_dim, resolved_partial_rotary_factor
from .utils import (
    const_in_work_dtype as _const_in_work_dtype,
    create_builder_context,
    rms_norm as _rms_norm,
)

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from .config import ModelConfig


_NATIVE_PREFILL_CHUNK_TOKENS = 32768


def _swiglu_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    matmul,
    weights: "WeightDict",
    prefix: str,
    hidden: int,
    mlp_size: int,
) -> trt.ITensor:
    gate = matmul(
        inp,
        hidden,
        mlp_size,
        weights[f"{prefix}.w_gate"],
    )
    up = matmul(
        inp,
        hidden,
        mlp_size,
        weights[f"{prefix}.w_up"],
    )
    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(
        gate,
        sigmoid.get_output(0),
        trt.ElementWiseOperation.PROD,
    )
    gated = network.add_elementwise(
        swish.get_output(0),
        up,
        trt.ElementWiseOperation.PROD,
    )
    return matmul(
        gated.get_output(0),
        mlp_size,
        hidden,
        weights[f"{prefix}.w_down"],
    )


def build_native_decoder_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    profile_mode: str,
    opt_prefill_length: int = 64,
    max_prefill_length: int | None = None,
    verbose: bool = False,
) -> bytes:
    """Build one native-KV GLM prefill or decode engine."""

    if profile_mode not in ("prefill", "decode"):
        raise ValueError(
            f"native GLM profile_mode must be 'prefill' or 'decode', got {profile_mode!r}"
        )
    if max_prefill_length is None:
        max_prefill_length = min(
            max_cache_length,
            _NATIVE_PREFILL_CHUNK_TOKENS,
        )
    max_prefill_length = max(1, min(max_prefill_length, max_cache_length))
    opt_prefill_length = max(1, min(opt_prefill_length, max_prefill_length))

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = resolved_head_dim(config)
    attention_size = num_heads * head_dim
    kv_attention_size = num_kv_heads * head_dim
    mlp_size = config.intermediate_size
    rotary_embedding_dim = int(head_dim * resolved_partial_rotary_factor(config))
    inv_freq = graph_ops.make_native_active_rope_inv_freq(
        head_dim,
        config.rope_theta,
        resolved_partial_rotary_factor(config),
    )

    builder_context = create_builder_context(
        verbose=verbose,
        workspace_bytes=1 << 30,
    )
    builder = builder_context.builder
    network = builder_context.network
    trt_config = builder_context.config
    work_np_dtype = np.float16
    work_trt_dtype = trt.bfloat16

    token_id = network.add_input("token_id", trt.int32, (-1,))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    cache_write_indices = network.add_input(
        "cache_write_indices",
        trt.int32,
        (1,),
    )
    key_value_lengths = network.add_input(
        "key_value_lengths",
        trt.int32,
        (1,),
    )

    cache_shape = (1, num_kv_heads, max_cache_length, head_dim)
    cache_k_inputs: list[trt.ITensor] = []
    cache_v_inputs: list[trt.ITensor] = []
    for layer_idx in range(num_layers):
        cache_k_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("cache_k", layer_idx),
                work_trt_dtype,
                cache_shape,
            )
        )
        cache_v_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("cache_v", layer_idx),
                work_trt_dtype,
                cache_shape,
            )
        )

    profile = builder.create_optimization_profile()
    if profile_mode == "prefill":
        profile.set_shape(
            "token_id",
            (1,),
            (opt_prefill_length,),
            (max_prefill_length,),
        )
        profile.set_shape(
            "position_id",
            (1,),
            (opt_prefill_length,),
            (max_prefill_length,),
        )
    else:
        profile.set_shape("token_id", (1,), (1,), (1,))
        profile.set_shape("position_id", (1,), (1,), (1,))
    trt_config.add_optimization_profile(profile)

    embedding_table = _const_in_work_dtype(
        network,
        (vocab, hidden),
        weights["embedding"],
        work_np_dtype,
        work_trt_dtype,
    )
    cos_active, sin_active = graph_ops.add_active_rope_cache(
        network,
        position_id,
        inv_freq,
        work_trt_dtype,
    )
    eps_tensor = graph_ops.add_constant(
        network,
        (1, 1),
        np.array([[config.rms_norm_eps]], dtype=np.float32),
        dtype=np.float32,
    )
    matmul = graph_blocks.make_matmul_fn(
        network,
        work_np_dtype,
    )

    embedding = network.add_gather(embedding_table, token_id, 0)
    hidden_state = embedding.get_output(0)
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(
            hidden_state,
            work_trt_dtype,
        ).get_output(0)

    present_k_outputs: list[trt.ITensor] = []
    present_v_outputs: list[trt.ITensor] = []
    attention_scale = 1.0 / np.sqrt(head_dim)

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        normed = _rms_norm(
            network,
            hidden_state,
            hidden,
            weights[f"{prefix}.input_norm"],
            eps_tensor,
            work_np_dtype,
        )

        q = matmul(
            normed,
            hidden,
            attention_size,
            weights[f"{prefix}.w_q"],
        )
        k = matmul(
            normed,
            hidden,
            kv_attention_size,
            weights[f"{prefix}.w_k"],
        )
        v = matmul(
            normed,
            hidden,
            kv_attention_size,
            weights[f"{prefix}.w_v"],
        )
        q = graph_ops.add_bias_sum(
            network,
            q,
            attention_size,
            weights[f"{prefix}.q_bias"],
            dtype=work_np_dtype,
        )
        k = graph_ops.add_bias_sum(
            network,
            k,
            kv_attention_size,
            weights[f"{prefix}.k_bias"],
            dtype=work_np_dtype,
        )
        v = graph_ops.add_bias_sum(
            network,
            v,
            kv_attention_size,
            weights[f"{prefix}.v_bias"],
            dtype=work_np_dtype,
        )
        q = graph_ops.add_apply_active_rope(
            network,
            q,
            num_heads,
            head_dim,
            cos_active,
            sin_active,
            rotary_embedding_dim,
            interleaved=True,
        )
        k = graph_ops.add_apply_active_rope(
            network,
            k,
            num_kv_heads,
            head_dim,
            cos_active,
            sin_active,
            rotary_embedding_dim,
            interleaved=True,
        )

        native_attention = graph_ops.add_native_kv_cache_attention_from_rows(
            network,
            q,
            k,
            v,
            cache_k_inputs[layer_idx],
            cache_v_inputs[layer_idx],
            cache_write_indices,
            key_value_lengths,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            scale=attention_scale,
            tag=f"{prefix}.attn",
        )
        present_k_outputs.append(native_attention["present_k"])
        present_v_outputs.append(native_attention["present_v"])

        attention_output = matmul(
            native_attention["context"],
            attention_size,
            hidden,
            weights[f"{prefix}.w_o"],
        )
        attention_residual = network.add_elementwise(
            hidden_state,
            attention_output,
            trt.ElementWiseOperation.SUM,
        )
        post_attention = _rms_norm(
            network,
            attention_residual.get_output(0),
            hidden,
            weights[f"{prefix}.post_attn_norm"],
            eps_tensor,
            work_np_dtype,
        )
        mlp_output = _swiglu_mlp(
            network,
            post_attention,
            matmul=matmul,
            weights=weights,
            prefix=prefix,
            hidden=hidden,
            mlp_size=mlp_size,
        )
        hidden_state = network.add_elementwise(
            attention_residual.get_output(0),
            mlp_output,
            trt.ElementWiseOperation.SUM,
        ).get_output(0)

    hidden_state = _rms_norm(
        network,
        hidden_state,
        hidden,
        weights["final_norm"],
        eps_tensor,
        work_np_dtype,
    )

    shape_tensor = network.add_shape(hidden_state).get_output(0)
    one_hidden = graph_ops.add_constant(
        network,
        (2,),
        np.array([1, hidden], dtype=np.int64),
        dtype=np.int64,
    )
    start = network.add_elementwise(
        shape_tensor,
        one_hidden,
        trt.ElementWiseOperation.SUB,
    ).get_output(0)
    last_row = network.add_slice(
        hidden_state,
        start=(0, 0),
        shape=(0, 0),
        stride=(1, 1),
    )
    last_row.set_input(1, start)
    last_row.set_input(2, one_hidden)

    logits = graph_ops.add_matmul_rhs_constant(
        network,
        last_row.get_output(0),
        hidden,
        vocab,
        weights["w_out"],
        dtype=work_np_dtype,
    )
    logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    for layer_idx in range(num_layers):
        present_k = present_k_outputs[layer_idx]
        present_v = present_v_outputs[layer_idx]
        present_k.name = graph_ops.layer_tensor_name(
            "present_k",
            layer_idx,
        )
        present_v.name = graph_ops.layer_tensor_name(
            "present_v",
            layer_idx,
        )
        network.mark_output(present_k)
        network.mark_output(present_v)

    if verbose:
        token_max = max_prefill_length if profile_mode == "prefill" else 1
        print(
            "[trtmc build] Building GLM native-KV "
            f"{profile_mode} engine (layers={num_layers}, hidden={hidden}, "
            f"cache={max_cache_length}, token_max={token_max}) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError(f"GLM native-KV {profile_mode} engine build failed")
    return bytes(plan)
