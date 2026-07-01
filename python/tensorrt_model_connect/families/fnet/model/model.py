# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned TensorRT model graph and utility implementation."""

from __future__ import annotations


import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from ....parallel_config import add_all_reduce_sum, normalize_parallel_config
from ..config import ModelConfig


trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ....parallel_config import ParallelConfig
    from ..weights import WeightDict


def _cast_back_to_trt_dtype(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    target_dtype: trt.DataType,
) -> trt.ITensor:
    """Cast a tensor back to the original TRT runtime dtype after FP32 compute."""
    if tensor.dtype == target_dtype:
        return tensor
    return network.add_cast(tensor, target_dtype).get_output(0)


def add_constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Add a constant tensor in the given *dtype* (default float32)."""
    weights = trt.Weights(np.ascontiguousarray(values, dtype=dtype))
    layer = network.add_constant(shape, weights)
    return layer.get_output(0)


def add_matmul_rhs_constant(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    lhs_width: int,
    rhs_width: int,
    rhs_weights: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Matrix multiply: lhs @ rhs_constant.  rhs is [lhs_width, rhs_width]."""
    rank = len(tuple(lhs.shape))
    rhs_shape = (lhs_width, rhs_width) if rank <= 2 else (1,) * (rank - 2) + (lhs_width, rhs_width)
    rhs = add_constant(
        network,
        rhs_shape,
        np.asarray(rhs_weights).reshape(rhs_shape),
        dtype=dtype,
    )
    rhs = _cast_back_to_trt_dtype(network, rhs, lhs.dtype)
    mm = network.add_matrix_multiply(
        lhs,
        trt.MatrixOperation.NONE,
        rhs,
        trt.MatrixOperation.NONE,
    )
    return _cast_back_to_trt_dtype(network, mm.get_output(0), lhs.dtype)


def add_bias_sum(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    width: int,
    bias: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Element-wise add a bias broadcast over all non-feature axes."""
    rank = len(tuple(inp.shape))
    bias_shape = (width,) if rank <= 1 else (1,) * (rank - 1) + (width,)
    bias_t = add_constant(network, bias_shape, np.asarray(bias).reshape(bias_shape), dtype=dtype)
    bias_t = _cast_back_to_trt_dtype(network, bias_t, inp.dtype)
    s = network.add_elementwise(inp, bias_t, trt.ElementWiseOperation.SUM)
    return _cast_back_to_trt_dtype(network, s.get_output(0), inp.dtype)


def add_gelu_new(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """GELU (tanh approximation): 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).

    Constants are cast to ``inp.dtype`` so the elementwise ops are valid in
    a STRONGLY_TYPED network when ``inp`` is bf16 (storage np_dtype is
    fp16, runtime trt_dtype is bfloat16) or any other non-matching combo.
    """
    target_dtype = inp.dtype
    const_shape = (1,) * max(1, len(tuple(inp.shape)))

    def _const(value):
        c = add_constant(network, const_shape, np.array([value], dtype=np.float32), dtype=dtype)
        return _cast_back_to_trt_dtype(network, c, target_dtype)

    # x^3
    x_sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    x_cu = network.add_elementwise(x_sq.get_output(0), inp, trt.ElementWiseOperation.PROD)
    # 0.044715 * x^3
    coeff = _const(0.044715)
    scaled_cube = network.add_elementwise(x_cu.get_output(0), coeff, trt.ElementWiseOperation.PROD)
    # x + 0.044715 * x^3
    inner_sum = network.add_elementwise(
        inp, scaled_cube.get_output(0), trt.ElementWiseOperation.SUM
    )
    # sqrt(2/pi) * (x + 0.044715 * x^3)
    sqrt_2_over_pi = _const(np.sqrt(2.0 / np.pi))
    tanh_arg = network.add_elementwise(
        sqrt_2_over_pi, inner_sum.get_output(0), trt.ElementWiseOperation.PROD
    )
    # tanh(...)
    tanh_l = network.add_activation(tanh_arg.get_output(0), trt.ActivationType.TANH)
    # 1 + tanh(...)
    one = _const(1.0)
    one_plus_tanh = network.add_elementwise(one, tanh_l.get_output(0), trt.ElementWiseOperation.SUM)
    # 0.5 * x
    half = _const(0.5)
    half_x = network.add_elementwise(half, inp, trt.ElementWiseOperation.PROD)
    # 0.5 * x * (1 + tanh(...))
    result = network.add_elementwise(
        half_x.get_output(0), one_plus_tanh.get_output(0), trt.ElementWiseOperation.PROD
    )
    return result.get_output(0)


def add_activation(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    activation_type: str,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Add FNet's GELU activation."""
    if activation_type in ("gelu_new", "gelu"):
        return add_gelu_new(network, inp, dtype=dtype)
    raise ValueError(f"Unsupported FNet activation: {activation_type}")


# ---------------------------------------------------------------------------
# TRT native normalization.
# ---------------------------------------------------------------------------


def add_layer_norm_native(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """LayerNorm via TRT native INormalizationLayer (add_normalization_v2).

    Replaces the manual reduce/elementwise chain in add_layer_norm with a
    single fused layer that TRT can optimize end-to-end. In strongly typed
    networks, input/scale/bias must have identical tensor types; compute
    precision is set to FP32 for numerical stability when the TensorRT Python
    layer exposes that control.

    Note: INormalizationLayer computes (x - mean) / sqrt(var + eps) * gamma + beta.
    This is LayerNorm, NOT RMSNorm.  Use add_rms_norm for RMSNorm models.

    Args:
        inp:         Input tensor [*, hidden_size].
        hidden_size: Size of the normalized dimension (last axis).
        gamma:       Scale weights [hidden_size].
        beta:        Bias weights [hidden_size].
        eps:         Numerical stability epsilon (scalar, not a tensor).
        dtype:       Storage dtype for gamma/beta constants before TRT cast.
    """
    inp_shape = getattr(inp, "shape", None)
    rank = len(tuple(inp_shape)) if inp_shape is not None else 2
    param_shape = (hidden_size,) if rank <= 1 else (1,) * (rank - 1) + (hidden_size,)
    gamma_t = add_constant(
        network, param_shape, np.asarray(gamma).reshape(param_shape), dtype=dtype
    )
    beta_t = add_constant(network, param_shape, np.asarray(beta).reshape(param_shape), dtype=dtype)
    gamma_t = _cast_back_to_trt_dtype(network, gamma_t, inp.dtype)
    beta_t = _cast_back_to_trt_dtype(network, beta_t, inp.dtype)
    # axesMask bit i selects axis i as a reduction axis. The normalized
    # hidden dimension is always the last axis for [*, hidden_size] tensors.
    norm = network.add_normalization_v2(inp, gamma_t, beta_t, 1 << (rank - 1))
    norm.epsilon = eps
    # TensorRT 11 removed the Python INormalizationLayer.compute_precision
    # attribute. Keep the TRT 10 hint, and let TRT 11 infer the precision.
    if hasattr(norm, "compute_precision"):
        norm.compute_precision = trt.float32
    return norm.get_output(0)


def _compute_dft_matrices(n: int):
    """Pre-compute real and imaginary DFT matrices for dimension n.

    Returns (cos_mat, sin_mat) each of shape [n, n].
    DFT definition: X_k = sum_j x_j * exp(-2pi*i*j*k/n)
    real part: cos(2*pi*j*k/n), imag part: -sin(2*pi*j*k/n)
    """
    j = np.arange(n, dtype=np.float64)
    k = np.arange(n, dtype=np.float64)
    jk = np.outer(k, j)  # [n, n]
    angle = 2.0 * np.pi * jk / n
    return np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)


def _add_seq_layer_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float,
    dtype: np.dtype,
) -> trt.ITensor:
    return add_layer_norm_native(network, inp, hidden_size, gamma, beta, eps, dtype=dtype)


def _slice_last_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=-1)[rank])


def _slice_first_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=0)[rank])


def _validate_fnet_tp(
    config: ModelConfig,
    weights: WeightDict,
    parallel: ParallelConfig,
) -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError("FNet tensor-parallel build requires a concrete rank")
    if config.intermediate_size % parallel.tp_size != 0:
        raise ValueError(
            "FNet tensor parallel requires intermediate_size divisible by "
            f"tp_size ({config.intermediate_size} vs {parallel.tp_size})"
        )
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"layer.{layer_idx}"
        if weights[f"{prefix}.w_fc1"].shape[-1] % parallel.tp_size != 0:
            raise ValueError(f"{prefix}.w_fc1 output dim must be divisible by tp_size")
        if weights[f"{prefix}.w_fc2"].shape[0] % parallel.tp_size != 0:
            raise ValueError(f"{prefix}.w_fc2 input dim must be divisible by tp_size")


def shard_fnet_encoder_weights(
    config: ModelConfig,
    weights: WeightDict,
    *,
    parallel: ParallelConfig,
) -> WeightDict:
    """Return rank-local FNet FFN weights."""
    _validate_fnet_tp(config, weights, parallel)
    if not parallel.enabled:
        return weights

    out = type(weights)()
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
        elif key.endswith(".w_fc1"):
            out[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith((".fc1_bias", ".w_fc2")):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        else:
            out[key] = value

    out["_intermediate_size"] = config.intermediate_size // parallel.tp_size
    out["_tensor_parallel_size"] = parallel.tp_size
    out["_tensor_parallel_rank"] = parallel.rank
    return out


def build_fnet_encoder_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_seq_length: int,
    *,
    precision: str = "fp32",
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    """Build a single-device or rank-local tensor-parallel FNet engine."""
    parallel = normalize_parallel_config(parallel_config)
    if parallel.enabled:
        weights = shard_fnet_encoder_weights(config, weights, parallel=parallel)
    tp_size = parallel.tp_size if parallel.enabled else 1

    hidden = config.hidden_size
    num_layers = config.num_hidden_layers
    intermediate = config.intermediate_size // tp_size
    eps = config.rms_norm_eps
    type_vocab_size = config.raw.get("type_vocab_size", 4)
    hidden_act = config.hidden_act or config.raw.get("activation", "") or "gelu_new"

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    trt_config.clear_flag(trt.BuilderFlag.TF32)

    S = max_seq_length
    H = hidden
    if precision not in {"fp32", "fp16"}:
        raise ValueError(f"FNet supports fp32 or fp16 precision, got {precision!r}")
    work_np_dtype = np.float16 if precision == "fp16" else np.float32

    # Inputs
    input_ids = network.add_input("input_ids", trt.int32, (S,))
    network.add_input("attention_mask", trt.int32, (S,))

    # token_type_ids: constant zeros
    tt_zeros = network.add_constant((S,), trt.Weights(np.zeros(S, dtype=np.int32)))
    token_type_ids = tt_zeros.get_output(0)

    # Embedding tables (may use embedding_size != hidden for factorized embeddings)
    embedding_size = weights["embedding"].shape[1]
    embedding_table = add_constant(
        network, weights["embedding"].shape, weights["embedding"], dtype=work_np_dtype
    )
    position_embed_table = add_constant(
        network,
        weights["position_embedding"].shape,
        weights["position_embedding"],
        dtype=work_np_dtype,
    )
    token_type_table = add_constant(
        network,
        (type_vocab_size, embedding_size),
        weights["token_type_embedding"],
        dtype=work_np_dtype,
    )

    # Position indices
    position_indices = network.add_constant((S,), trt.Weights(np.arange(S, dtype=np.int32)))

    # Embedding: word + position + token_type
    word_embed = network.add_gather(embedding_table, input_ids, 0)
    pos_embed = network.add_gather(position_embed_table, position_indices.get_output(0), 0)
    tt_embed = network.add_gather(token_type_table, token_type_ids, 0)

    embed_sum1 = network.add_elementwise(
        word_embed.get_output(0), pos_embed.get_output(0), trt.ElementWiseOperation.SUM
    )
    embed_sum2 = network.add_elementwise(
        embed_sum1.get_output(0), tt_embed.get_output(0), trt.ElementWiseOperation.SUM
    )

    # Embedding LayerNorm (over embedding_size)
    hidden_state = _add_seq_layer_norm(
        network,
        embed_sum2.get_output(0),
        embedding_size,
        weights["embed_norm"],
        weights["embed_norm_beta"],
        eps,
        work_np_dtype,
    )

    # Optional embedding projection: embedding_size -> hidden_size
    if "embed_projection" in weights:
        hidden_state = add_matmul_rhs_constant(
            network,
            hidden_state,
            embedding_size,
            hidden,
            weights["embed_projection"],
            dtype=work_np_dtype,
        )
        if "embed_projection_bias" in weights:
            hidden_state = add_bias_sum(
                network,
                hidden_state,
                hidden,
                weights["embed_projection_bias"],
                dtype=work_np_dtype,
            )

    # Pre-compute 2D DFT matrices as constants
    # real(DFT2D(X)) = cos_S @ X @ cos_H - sin_S @ X @ sin_H
    cos_s, sin_s = _compute_dft_matrices(S)
    cos_h, sin_h = _compute_dft_matrices(H)

    cos_s_const = add_constant(network, (S, S), cos_s, dtype=work_np_dtype)
    sin_s_const = add_constant(network, (S, S), sin_s, dtype=work_np_dtype)
    cos_h_const = add_constant(network, (H, H), cos_h, dtype=work_np_dtype)
    sin_h_const = add_constant(network, (H, H), sin_h, dtype=work_np_dtype)

    # Encoder layers
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"

        # --- 2D DFT (replaces self-attention) ---
        # real(DFT2D(X)) = cos_S @ X @ cos_H - sin_S @ X @ sin_H
        # Term 1: cos_S @ X @ cos_H
        cx = network.add_matrix_multiply(
            cos_s_const, trt.MatrixOperation.NONE, hidden_state, trt.MatrixOperation.NONE
        )
        cxch = network.add_matrix_multiply(
            cx.get_output(0), trt.MatrixOperation.NONE, cos_h_const, trt.MatrixOperation.NONE
        )

        # Term 2: sin_S @ X @ sin_H
        sx = network.add_matrix_multiply(
            sin_s_const, trt.MatrixOperation.NONE, hidden_state, trt.MatrixOperation.NONE
        )
        sxsh = network.add_matrix_multiply(
            sx.get_output(0), trt.MatrixOperation.NONE, sin_h_const, trt.MatrixOperation.NONE
        )

        # real = term1 - term2
        dft_out = network.add_elementwise(
            cxch.get_output(0), sxsh.get_output(0), trt.ElementWiseOperation.SUB
        ).get_output(0)

        # POST-norm: residual + LayerNorm after DFT
        residual1 = network.add_elementwise(hidden_state, dft_out, trt.ElementWiseOperation.SUM)
        normed1 = _add_seq_layer_norm(
            network,
            residual1.get_output(0),
            hidden,
            weights[f"{prefix}.post_attn_norm"],
            weights[f"{prefix}.post_attn_norm_beta"],
            eps,
            work_np_dtype,
        )

        # --- FFN ---
        fc1 = add_matmul_rhs_constant(
            network,
            normed1,
            hidden,
            intermediate,
            weights[f"{prefix}.w_fc1"],
            dtype=work_np_dtype,
        )
        fc1 = add_bias_sum(
            network,
            fc1,
            intermediate,
            weights[f"{prefix}.fc1_bias"],
            dtype=work_np_dtype,
        )
        activated = add_activation(network, fc1, hidden_act, dtype=work_np_dtype)
        fc2 = add_matmul_rhs_constant(
            network,
            activated,
            intermediate,
            hidden,
            weights[f"{prefix}.w_fc2"],
            dtype=work_np_dtype,
        )
        if tp_size > 1:
            fc2 = add_all_reduce_sum(network, fc2, tp_size)
        fc2 = add_bias_sum(network, fc2, hidden, weights[f"{prefix}.fc2_bias"], dtype=work_np_dtype)

        # POST-norm: residual + LayerNorm after FFN
        residual2 = network.add_elementwise(normed1, fc2, trt.ElementWiseOperation.SUM)
        hidden_state = _add_seq_layer_norm(
            network,
            residual2.get_output(0),
            hidden,
            weights[f"{prefix}.output_norm"],
            weights[f"{prefix}.output_norm_beta"],
            eps,
            work_np_dtype,
        )

    # Output
    if hidden_state.dtype != trt.float32:
        hidden_state = network.add_cast(hidden_state, trt.float32).get_output(0)
    hidden_state.name = "hidden_states"
    network.mark_output(hidden_state)

    if verbose:
        print(
            f"[trtmc build] Building FNet encoder TRT engine "
            f"({num_layers} layers, hidden={hidden}, tp={tp_size}, "
            f"seq_len={S}) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")

    return bytes(plan)


def build_tp_fnet_encoder_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_seq_length: int,
    *,
    precision: str = "fp32",
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    """Build one rank-local FNet engine."""
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError(
            "build_tp_fnet_encoder_engine requires tensor_parallel mode and tp_size > 1"
        )
    return build_fnet_encoder_engine(
        config,
        weights,
        max_seq_length,
        precision=precision,
        verbose=verbose,
        parallel_config=parallel,
    )
