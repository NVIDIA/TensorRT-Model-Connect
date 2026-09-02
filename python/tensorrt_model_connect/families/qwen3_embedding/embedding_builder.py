# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Embedding-owned cacheless causal backbone builder."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

from tensorrt_model_connect import trt_compat

from . import graph_ops
from .checkpoint_mapper import _target_np_dtype

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from .config import ModelConfig


trt = trt_compat.get_trt()


def _const_in_work_dtype(
    network,
    shape: tuple,
    values: np.ndarray,
    work_np_dtype: np.dtype,
    work_trt_dtype,
):
    const = graph_ops.add_constant(network, shape, values, dtype=work_np_dtype)
    if const.dtype != work_trt_dtype:
        const = network.add_cast(const, work_trt_dtype).get_output(0)
    return const


def _make_matmul_fn(network, dtype: np.dtype):
    def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
        del weight_name
        return graph_ops.add_matmul_rhs_constant(
            network, lhs, lhs_w, rhs_w, rhs_weights, dtype=dtype
        )

    return matmul


def _norm_multi(
    network,
    inp,
    hidden: int,
    gamma: np.ndarray,
    beta: np.ndarray | None,
    eps_tensor,
    norm_type: str,
    dtype: np.dtype,
):
    if norm_type == "layernorm":
        if beta is None:
            beta = np.zeros(hidden, dtype=np.float32)
        return graph_ops.add_layer_norm(
            network, inp, hidden, gamma, beta, eps_tensor, dtype=dtype
        )
    return graph_ops.add_rms_norm(
        network, inp, hidden, gamma, eps_tensor, dtype=dtype
    )


def _swiglu_mlp(
    network,
    inp,
    *,
    matmul,
    weights: "WeightDict",
    prefix: str,
    hidden: int,
    mlp_size: int,
):
    gate = matmul(
        inp, hidden, mlp_size, weights[f"{prefix}.w_gate"], f"{prefix}.w_gate"
    )
    up = matmul(
        inp, hidden, mlp_size, weights[f"{prefix}.w_up"], f"{prefix}.w_up"
    )
    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(
        gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD
    )
    gated = network.add_elementwise(
        swish.get_output(0), up, trt.ElementWiseOperation.PROD
    )
    return matmul(
        gated.get_output(0),
        mlp_size,
        hidden,
        weights[f"{prefix}.w_down"],
        f"{prefix}.w_down",
    )


def _infer_kv_attention_size(
    weights: "WeightDict", *, num_kv_heads: int, head_dim: int
) -> int:
    expected = int(num_kv_heads * head_dim)
    explicit = weights.get("_kv_attention_size")
    if explicit is not None and int(explicit) != expected:
        raise ValueError(
            "Compact K/V width must equal num_key_value_heads * head_dim "
            f"({expected}), got {int(explicit)}"
        )
    first_k = weights.get("layer.0.w_k")
    if isinstance(first_k, np.ndarray) and first_k.ndim == 2:
        actual = int(first_k.shape[1])
        if actual != expected:
            raise ValueError(
                f"layer.0.w_k must use compact K/V width {expected}, got {actual}"
            )
    return expected


def build_qwen3_embedding_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_sequence_length: int,
    *,
    precision: str = "bf16",
    verbose: bool = False,
) -> bytes:
    """Build the causal Qwen3 backbone and expose every final hidden row."""

    if str(config.model_type).lower() != "qwen3" or tuple(config.architectures) != (
        "Qwen3ForCausalLM",
    ):
        raise ValueError(
            "Qwen3-Embedding requires the dense Qwen3ForCausalLM backbone"
        )
    if max_sequence_length < 1 or max_sequence_length > config.max_position_embeddings:
        raise ValueError(
            "Qwen3-Embedding max sequence length must be within the model context; "
            f"got {max_sequence_length}, model maximum={config.max_position_embeddings}"
        )
    precision = str(precision).lower()
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "bf16":
        work_np_dtype, work_trt_dtype = _target_np_dtype(precision), trt.bfloat16
    else:
        raise ValueError(f"Qwen3-Embedding precision must be fp16 or bf16, got {precision!r}")

    hidden = config.hidden_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    attention_size = num_heads * head_dim
    kv_attention_size = _infer_kv_attention_size(
        weights, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    mlp_size = config.intermediate_size

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()

    token_id = network.add_input("token_id", trt.int32, (-1,))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    profile = builder.create_optimization_profile()
    opt_length = min(128, max_sequence_length)
    profile.set_shape("token_id", (1,), (opt_length,), (max_sequence_length,))
    profile.set_shape("position_id", (1,), (opt_length,), (max_sequence_length,))
    trt_config.add_optimization_profile(profile)

    embedding_table = _const_in_work_dtype(
        network,
        (config.vocab_size, hidden),
        weights["embedding"],
        work_np_dtype,
        work_trt_dtype,
    )
    hidden_state = network.add_gather(embedding_table, token_id, 0).get_output(0)
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)

    graph_ops.validate_native_rope_dim(head_dim)
    cos_half = graph_ops.make_rope_table_half_dim(
        max_sequence_length, head_dim, config.rope_theta, True
    )
    sin_half = graph_ops.make_rope_table_half_dim(
        max_sequence_length, head_dim, config.rope_theta, False
    )
    cos_tensor = _const_in_work_dtype(
        network, cos_half.shape, cos_half, work_np_dtype, work_trt_dtype
    )
    sin_tensor = _const_in_work_dtype(
        network, sin_half.shape, sin_half, work_np_dtype, work_trt_dtype
    )
    eps_tensor = graph_ops.add_constant(
        network,
        (1, 1),
        np.array([[config.rms_norm_eps]], dtype=np.float32),
        dtype=np.float32,
    )
    eps_per_head = graph_ops.add_constant(
        network,
        (1, 1, 1),
        np.array([[[config.rms_norm_eps]]], dtype=np.float32),
        dtype=np.float32,
    )
    matmul = _make_matmul_fn(network, work_np_dtype)
    attention_scale = 1.0 / np.sqrt(max(head_dim, 1))

    for layer_index in range(num_layers):
        prefix = f"layer.{layer_index}"
        normed = _norm_multi(
            network,
            hidden_state,
            hidden,
            weights[f"{prefix}.input_norm"],
            weights.get(f"{prefix}.input_norm_beta"),
            eps_tensor,
            "rmsnorm",
            work_np_dtype,
        )
        q = matmul(normed, hidden, attention_size, weights[f"{prefix}.w_q"], f"{prefix}.w_q")
        k = matmul(
            normed,
            hidden,
            kv_attention_size,
            weights[f"{prefix}.w_k"],
            f"{prefix}.w_k",
        )
        v = matmul(
            normed,
            hidden,
            kv_attention_size,
            weights[f"{prefix}.w_v"],
            f"{prefix}.w_v",
        )

        q_norm = weights.get(f"{prefix}.q_norm")
        if q_norm is not None:
            q = graph_ops.add_rms_norm_per_head(
                network,
                q,
                num_heads,
                head_dim,
                q_norm,
                eps_per_head,
                dtype=work_np_dtype,
                sequence_length=None,
            )
        k_norm = weights.get(f"{prefix}.k_norm")
        if k_norm is not None:
            k = graph_ops.add_rms_norm_per_head(
                network,
                k,
                num_kv_heads,
                head_dim,
                k_norm,
                eps_per_head,
                dtype=work_np_dtype,
                sequence_length=None,
            )

        q = graph_ops.add_apply_rope_native(
            network,
            q,
            num_heads,
            head_dim,
            cos_tensor,
            sin_tensor,
            position_id,
            head_dim,
            sequence_length=None,
        )
        k = graph_ops.add_apply_rope_native(
            network,
            k,
            num_kv_heads,
            head_dim,
            cos_tensor,
            sin_tensor,
            position_id,
            head_dim,
            sequence_length=None,
        )
        context = graph_ops.add_attention_from_rows(
            network,
            q,
            k,
            v,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_seq=None,
            kv_seq=None,
            causal=True,
            scale=attention_scale,
            tag=f"{prefix}.embedding_attention",
        )
        attention_output = matmul(
            context,
            attention_size,
            hidden,
            weights[f"{prefix}.w_o"],
            f"{prefix}.w_o",
        )
        residual = network.add_elementwise(
            hidden_state, attention_output, trt.ElementWiseOperation.SUM
        ).get_output(0)
        post_attention = _norm_multi(
            network,
            residual,
            hidden,
            weights[f"{prefix}.post_attn_norm"],
            weights.get(f"{prefix}.post_attn_norm_beta"),
            eps_tensor,
            "rmsnorm",
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
            residual, mlp_output, trt.ElementWiseOperation.SUM
        ).get_output(0)

    hidden_state = _norm_multi(
        network,
        hidden_state,
        hidden,
        weights["final_norm"],
        weights.get("final_norm_beta"),
        eps_tensor,
        "rmsnorm",
        work_np_dtype,
    )
    if hidden_state.dtype != trt.float32:
        hidden_state = network.add_cast(hidden_state, trt.float32).get_output(0)
    hidden_state.name = "hidden_states"
    network.mark_output(hidden_state)

    if verbose:
        print(
            "[trtmc build] Building Qwen3-Embedding engine "
            f"(layers={num_layers}, hidden={hidden}, max_sequence={max_sequence_length}, "
            f"precision={precision}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("Qwen3-Embedding TensorRT engine build failed")
    return bytes(plan)
