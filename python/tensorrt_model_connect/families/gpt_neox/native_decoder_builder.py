# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native-only split prefill/decode builder for dense GPT-NeoX models."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops
from .build_routing import resolved_head_dim, resolved_rotary_dim
from .utils import const_in_work_dtype, create_builder_context, norm_multi

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from .config import ModelConfig


_NATIVE_PREFILL_CHUNK_TOKENS = 32768


def _gelu_fc_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    matmul,
    weights: "WeightDict",
    prefix: str,
    hidden: int,
    mlp_size: int,
    activation: str,
    work_np_dtype: np.dtype,
) -> trt.ITensor:
    fc1 = matmul(
        inp,
        hidden,
        mlp_size,
        weights[f"{prefix}.w_fc1"],
        f"{prefix}.w_fc1",
    )
    fc1 = graph_ops.add_bias_sum(
        network,
        fc1,
        mlp_size,
        weights[f"{prefix}.fc1_bias"],
        dtype=work_np_dtype,
    )
    activated = graph_ops.add_activation(
        network,
        fc1,
        activation,
        dtype=work_np_dtype,
    )
    fc2 = matmul(
        activated,
        mlp_size,
        hidden,
        weights[f"{prefix}.w_fc2"],
        f"{prefix}.w_fc2",
    )
    return graph_ops.add_bias_sum(
        network,
        fc2,
        hidden,
        weights[f"{prefix}.fc2_bias"],
        dtype=work_np_dtype,
    )


def build_native_decoder_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "fp16",
    opt_prefill_length: int = 64,
    max_prefill_length: int | None = None,
    profile_mode: str,
    verbose: bool = False,
) -> bytes:
    """Build one native-KV prefill or decode engine at full capacity."""

    if precision != "fp16":
        raise ValueError("native GPT-NeoX decoder requires FP16")
    if profile_mode not in ("prefill", "decode"):
        raise ValueError(
            "native GPT-NeoX profile_mode must be 'prefill' or 'decode', "
            f"got {profile_mode!r}"
        )

    if max_prefill_length is None:
        max_prefill_length = min(
            max_cache_length,
            _NATIVE_PREFILL_CHUNK_TOKENS,
        )
    max_prefill_length = max(
        1,
        min(max_prefill_length, max_cache_length),
    )
    opt_prefill_length = max(
        1,
        min(opt_prefill_length, max_prefill_length),
    )

    hidden = config.hidden_size
    vocab = config.vocab_size
    mlp_size = config.intermediate_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = resolved_head_dim(config)
    rotary_dim = resolved_rotary_dim(config)
    attention_size = num_heads * head_dim

    inv_freq = graph_ops.make_native_active_rope_inv_freq(
        head_dim,
        config.rope_theta,
        rotary_dim=rotary_dim,
    )

    builder_context = create_builder_context(
        verbose=verbose,
        workspace_bytes=16 << 30,
    )
    builder = builder_context.builder
    network = builder_context.network
    trt_config = builder_context.config
    work_np_dtype = np.float16
    work_trt_dtype = trt.float16

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

    cache_shape = (
        1,
        num_kv_heads,
        max_cache_length,
        head_dim,
    )
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

    embedding_table = const_in_work_dtype(
        network,
        (vocab, hidden),
        weights["embedding"],
        work_np_dtype,
        work_trt_dtype,
    )
    cos_half, sin_half = graph_ops.add_active_rope_cache(
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
    attn_scale = float(1.0 / np.sqrt(head_dim))

    def matmul(
        lhs: trt.ITensor,
        lhs_width: int,
        rhs_width: int,
        rhs_weights: np.ndarray,
        _weight_name: str,
    ) -> trt.ITensor:
        return graph_ops.add_matmul_rhs_constant(
            network,
            lhs,
            lhs_width,
            rhs_width,
            rhs_weights,
            dtype=work_np_dtype,
        )

    hidden_state = network.add_gather(
        embedding_table,
        token_id,
        0,
    ).get_output(0)
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(
            hidden_state,
            work_trt_dtype,
        ).get_output(0)

    present_k_outs: list[trt.ITensor] = []
    present_v_outs: list[trt.ITensor] = []
    parallel_residual = bool(
        config.raw.get("use_parallel_residual", True)
    )
    activation = str(config.hidden_act).lower()

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        normed = norm_multi(
            network,
            hidden_state,
            hidden,
            weights[f"{prefix}.input_norm"],
            weights[f"{prefix}.input_norm_beta"],
            eps_tensor,
            "layernorm",
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
            attention_size,
            weights[f"{prefix}.k_bias"],
            dtype=work_np_dtype,
        )
        v = graph_ops.add_bias_sum(
            network,
            v,
            attention_size,
            weights[f"{prefix}.v_bias"],
            dtype=work_np_dtype,
        )

        q = graph_ops.add_apply_rope_native(
            network,
            q,
            num_heads,
            head_dim,
            cos_half,
            sin_half,
            None,
            rotary_dim,
            False,
            sequence_length=None,
        )
        k = graph_ops.add_apply_rope_native(
            network,
            k,
            num_kv_heads,
            head_dim,
            cos_half,
            sin_half,
            None,
            rotary_dim,
            False,
            sequence_length=None,
        )

        native_attention = (
            graph_ops.add_native_kv_cache_attention_from_rows(
                network,
                q,
                k,
                v,
                cache_k_inputs[layer_idx],
                cache_v_inputs[layer_idx],
                cache_write_indices,
                key_value_lengths,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_seq=None,
                scale=attn_scale,
                tag=f"{prefix}.attn",
            )
        )
        present_k_outs.append(native_attention["present_k"])
        present_v_outs.append(native_attention["present_v"])

        attn_out = matmul(
            native_attention["context"],
            attention_size,
            hidden,
            weights[f"{prefix}.w_o"],
            f"{prefix}.w_o",
        )
        attn_out = graph_ops.add_bias_sum(
            network,
            attn_out,
            hidden,
            weights[f"{prefix}.o_bias"],
            dtype=work_np_dtype,
        )

        if parallel_residual:
            norm2 = norm_multi(
                network,
                hidden_state,
                hidden,
                weights[f"{prefix}.post_attn_norm"],
                weights[f"{prefix}.post_attn_norm_beta"],
                eps_tensor,
                "layernorm",
                work_np_dtype,
            )
        else:
            residual1 = network.add_elementwise(
                hidden_state,
                attn_out,
                trt.ElementWiseOperation.SUM,
            ).get_output(0)
            norm2 = norm_multi(
                network,
                residual1,
                hidden,
                weights[f"{prefix}.post_attn_norm"],
                weights[f"{prefix}.post_attn_norm_beta"],
                eps_tensor,
                "layernorm",
                work_np_dtype,
            )

        mlp_out = _gelu_fc_mlp(
            network,
            norm2,
            matmul=matmul,
            weights=weights,
            prefix=prefix,
            hidden=hidden,
            mlp_size=mlp_size,
            activation=activation,
            work_np_dtype=work_np_dtype,
        )
        if parallel_residual:
            hidden_plus_attention = network.add_elementwise(
                hidden_state,
                attn_out,
                trt.ElementWiseOperation.SUM,
            ).get_output(0)
            hidden_state = network.add_elementwise(
                hidden_plus_attention,
                mlp_out,
                trt.ElementWiseOperation.SUM,
            ).get_output(0)
        else:
            hidden_state = network.add_elementwise(
                residual1,
                mlp_out,
                trt.ElementWiseOperation.SUM,
            ).get_output(0)

    hidden_state = norm_multi(
        network,
        hidden_state,
        hidden,
        weights["final_norm"],
        weights["final_norm_beta"],
        eps_tensor,
        "layernorm",
        work_np_dtype,
    )

    shape_t = network.add_shape(hidden_state).get_output(0)
    one_hidden = graph_ops.add_constant(
        network,
        (2,),
        np.array([1, hidden], dtype=np.int64),
        dtype=np.int64,
    )
    start_t = network.add_elementwise(
        shape_t,
        one_hidden,
        trt.ElementWiseOperation.SUB,
    ).get_output(0)
    last_hidden_slice = network.add_slice(
        hidden_state,
        start=(0, 0),
        shape=(0, 0),
        stride=(1, 1),
    )
    last_hidden_slice.set_input(1, start_t)
    last_hidden_slice.set_input(2, one_hidden)
    last_hidden = last_hidden_slice.get_output(0)

    # Pythia logits share a large common offset.  Emitting the raw LM-head
    # result in FP16 gives that offset a one-point ULP and collapses legitimate
    # sub-point top-1 margins into ties.  Subtracting one vocabulary column
    # from every weight column removes the common offset before FP16 rounding;
    # adding its one-column dot product back in FP32 preserves the public
    # logits while keeping the full-vocabulary matmul on the fast FP16 path.
    lm_weights = np.asarray(weights["w_out"], dtype=np.float32)
    reference_weight = lm_weights[:, :1]
    centered_weights = lm_weights - reference_weight
    centered_logits = graph_ops.add_matmul_rhs_constant(
        network,
        last_hidden,
        hidden,
        vocab,
        centered_weights,
        dtype=work_np_dtype,
    )
    centered_logits = network.add_cast(
        centered_logits,
        trt.float32,
    ).get_output(0)
    last_hidden_fp32 = network.add_cast(
        last_hidden,
        trt.float32,
    ).get_output(0)
    reference_logit = graph_ops.add_matmul_rhs_constant(
        network,
        last_hidden_fp32,
        hidden,
        1,
        reference_weight,
        dtype=np.float32,
    )
    logits = network.add_elementwise(
        centered_logits,
        reference_logit,
        trt.ElementWiseOperation.SUM,
    )
    logits = logits.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    for layer_idx, (present_k, present_v) in enumerate(
        zip(present_k_outs, present_v_outs)
    ):
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
        print(
            f"[trtmc build] Building native GPT-NeoX {profile_mode} engine "
            f"(layers={num_layers}, hidden={hidden}, heads={num_heads}, "
            f"head_dim={head_dim}, rotary_dim={rotary_dim}, "
            f"cache={max_cache_length}, max_prefill={max_prefill_length}, "
            "precision=fp16) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError(
            f"native GPT-NeoX {profile_mode} engine build failed"
        )
    return bytes(plan)
