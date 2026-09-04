# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned BERT encoder graph implementation."""

from __future__ import annotations


import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import tensorrt as trt
from .parallel import ParallelConfig, add_all_reduce_sum
from .config import ModelConfig

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter
    from .weights import WeightDict


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

    def _const(name, value):
        c = add_constant(network, const_shape, np.array([value], dtype=np.float32), dtype=dtype)
        return _cast_back_to_trt_dtype(network, c, target_dtype)

    # x^3
    x_sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    x_cu = network.add_elementwise(x_sq.get_output(0), inp, trt.ElementWiseOperation.PROD)
    # 0.044715 * x^3
    coeff = _const("coeff", 0.044715)
    scaled_cube = network.add_elementwise(x_cu.get_output(0), coeff, trt.ElementWiseOperation.PROD)
    # x + 0.044715 * x^3
    inner_sum = network.add_elementwise(
        inp, scaled_cube.get_output(0), trt.ElementWiseOperation.SUM
    )
    # sqrt(2/pi) * (x + 0.044715 * x^3)
    sqrt_2_over_pi = _const("sqrt_2_over_pi", np.sqrt(2.0 / np.pi))
    tanh_arg = network.add_elementwise(
        sqrt_2_over_pi, inner_sum.get_output(0), trt.ElementWiseOperation.PROD
    )
    # tanh(...)
    tanh_l = network.add_activation(tanh_arg.get_output(0), trt.ActivationType.TANH)
    # 1 + tanh(...)
    one = _const("one", 1.0)
    one_plus_tanh = network.add_elementwise(one, tanh_l.get_output(0), trt.ElementWiseOperation.SUM)
    # 0.5 * x
    half = _const("half", 0.5)
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
    """Add an activation supported by the family-owned graph contract."""
    if activation_type in ("gelu_new", "gelu"):
        return add_gelu_new(network, inp, dtype=dtype)
    if activation_type == "relu":
        return network.add_activation(inp, trt.ActivationType.RELU).get_output(0)
    if activation_type in ("relu2", "squared_relu"):
        relu = network.add_activation(inp, trt.ActivationType.RELU).get_output(0)
        return network.add_elementwise(relu, relu, trt.ElementWiseOperation.PROD).get_output(0)
    if activation_type == "silu":
        sigmoid = network.add_activation(inp, trt.ActivationType.SIGMOID).get_output(0)
        return network.add_elementwise(inp, sigmoid, trt.ElementWiseOperation.PROD).get_output(0)
    raise ValueError(f"Unsupported activation: {activation_type}")


# ---------------------------------------------------------------------------
# TRT 10 native attention APIs (TRT 10.x)
#
# Three primitives replace manual primitive chains:
#   add_layer_norm_native  → INormalizationLayer  (replaces add_layer_norm)
#   add_apply_rope_native  → IRotaryEmbeddingLayer
#   add_attention_core     → IAttention           (replaces score+softmax+V)
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
    return norm.get_output(0)


def reshape_rows_to_heads_4d(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    num_heads: int,
    head_dim: int,
    sequence_length: int | None = None,
    tag: str | None = None,
) -> trt.ITensor:
    """Reshape [S, H * D] rows into [1, H, S, D].

    The transpose is required for S > 1 because each input row contains all
    heads for one token. ``sequence_length=None`` means runtime-dynamic S.
    """
    seq_dim = -1 if sequence_length is None else sequence_length
    r1 = network.add_shuffle(x)
    if tag:
        r1.name = tag + "_s_h_d"
    r1.reshape_dims = (seq_dim, num_heads, head_dim)
    r1.second_transpose = trt.Permutation([1, 0, 2])

    r2 = network.add_shuffle(r1.get_output(0))
    if tag:
        r2.name = tag + "_1_h_s_d"
    r2.reshape_dims = (1, num_heads, seq_dim, head_dim)
    return r2.get_output(0)


def reshape_heads_4d_to_rows(
    network: trt.INetworkDefinition,
    x_4d: trt.ITensor,
    attention_size: int,
    sequence_length: int | None = None,
    tag: str | None = None,
) -> trt.ITensor:
    """Reshape [1, H, S, D] back to [S, H * D]."""
    seq_dim = -1 if sequence_length is None else sequence_length
    out = network.add_shuffle(x_4d)
    if tag:
        out.name = tag + "_s_h_d"
    out.first_transpose = trt.Permutation([0, 2, 1, 3])
    out.reshape_dims = (seq_dim, attention_size)
    return out.get_output(0)


def add_attention_core(
    network: trt.INetworkDefinition,
    q_4d: trt.ITensor,
    k_4d: trt.ITensor,
    v_4d: trt.ITensor,
    mask: trt.ITensor | None = None,
    scale: float | None = None,
) -> trt.ITensor:
    """Add BERT's bidirectional scaled dot-product attention."""
    output_dtype = q_4d.dtype
    if scale is None:
        head_dim = q_4d.shape[-1]
        scale = float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0
    scale_np_dtype = np.float16 if q_4d.dtype == trt.float16 else np.float32
    scale_t = add_constant(network, (1, 1, 1, 1), np.array([[[[scale]]]]), dtype=scale_np_dtype)
    if q_4d.dtype == trt.bfloat16:
        scale_t = network.add_cast(scale_t, trt.bfloat16).get_output(0)
    q_scaled = network.add_elementwise(q_4d, scale_t, trt.ElementWiseOperation.PROD)

    attn = network.add_attention(
        q_scaled.get_output(0),
        k_4d,
        v_4d,
        trt.AttentionNormalizationOp.SOFTMAX,
        False,
    )
    attn.decomposable = True
    if mask is not None:
        attn.mask = mask
    return _cast_back_to_trt_dtype(network, attn.get_output(0), output_dtype)


def add_attention_from_rows(
    network: trt.INetworkDefinition,
    q: trt.ITensor,
    k: trt.ITensor,
    v: trt.ITensor,
    *,
    num_heads: int,
    head_dim: int,
    sequence_length: int,
    mask: trt.ITensor | None = None,
    tag: str | None = None,
) -> trt.ITensor:
    """Add BERT attention for row-major Q/K/V tensors."""
    attention_size = num_heads * head_dim
    q_4d = reshape_rows_to_heads_4d(
        network,
        q,
        num_heads,
        head_dim,
        sequence_length=sequence_length,
        tag=None if tag is None else tag + ".q",
    )
    k_4d = reshape_rows_to_heads_4d(
        network,
        k,
        num_heads,
        head_dim,
        sequence_length=sequence_length,
        tag=None if tag is None else tag + ".k",
    )
    v_4d = reshape_rows_to_heads_4d(
        network,
        v,
        num_heads,
        head_dim,
        sequence_length=sequence_length,
        tag=None if tag is None else tag + ".v",
    )
    scale = float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0
    ctx_4d = add_attention_core(network, q_4d, k_4d, v_4d, mask=mask, scale=scale)
    return reshape_heads_4d_to_rows(
        network,
        ctx_4d,
        attention_size,
        sequence_length=sequence_length,
        tag=None if tag is None else tag + ".ctx",
    )


def _slice_last_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=-1)[rank])


def _slice_first_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=0)[rank])


def _validate_encoder_tp(
    config: ModelConfig,
    weights: WeightDict,
    parallel: ParallelConfig,
) -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError("BERT tensor-parallel build requires a concrete rank")

    tp_size = parallel.tp_size
    if config.num_attention_heads % tp_size != 0:
        raise ValueError(
            "BERT tensor parallel requires num_attention_heads divisible by "
            f"tp_size ({config.num_attention_heads} vs {tp_size})"
        )
    if config.intermediate_size % tp_size != 0:
        raise ValueError(
            "BERT tensor parallel requires intermediate_size divisible by "
            f"tp_size ({config.intermediate_size} vs {tp_size})"
        )

    for layer_idx in range(config.num_hidden_layers):
        prefix = f"layer.{layer_idx}"
        for key in (f"{prefix}.w_q", f"{prefix}.w_k", f"{prefix}.w_v"):
            if weights[key].shape[-1] % tp_size != 0:
                raise ValueError(f"{key} output dim must be divisible by tp_size")
        if weights[f"{prefix}.w_o"].shape[0] % tp_size != 0:
            raise ValueError(f"{prefix}.w_o input dim must be divisible by tp_size")
        if weights[f"{prefix}.w_fc1"].shape[-1] % tp_size != 0:
            raise ValueError(f"{prefix}.w_fc1 output dim must be divisible by tp_size")
        if weights[f"{prefix}.w_fc2"].shape[0] % tp_size != 0:
            raise ValueError(f"{prefix}.w_fc2 input dim must be divisible by tp_size")


def shard_encoder_weights(
    config: ModelConfig,
    weights: WeightDict,
    *,
    parallel: ParallelConfig,
) -> WeightDict:
    """Return the rank-local BERT encoder weights."""
    _validate_encoder_tp(config, weights, parallel)
    if not parallel.enabled:
        return weights

    out = type(weights)()
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
        elif key.endswith((".w_q", ".w_k", ".w_v", ".w_fc1")):
            out[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith((".q_bias", ".k_bias", ".v_bias", ".fc1_bias")):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith((".w_o", ".w_fc2")):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        else:
            out[key] = value

    out["_attention_size"] = config.attention_size // parallel.tp_size
    out["_intermediate_size"] = config.intermediate_size // parallel.tp_size
    out["_tensor_parallel_size"] = parallel.tp_size
    out["_tensor_parallel_rank"] = parallel.rank
    return out


def build_encoder_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_seq_length: int,
    *,
    precision: str = "fp32",
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    """Build a single-device or rank-local tensor-parallel BERT engine."""
    parallel = parallel_config or ParallelConfig()
    parallel.validate()
    if parallel.enabled:
        weights = shard_encoder_weights(config, weights, parallel=parallel)
    tp_size = parallel.tp_size if parallel.enabled else 1

    hidden = config.hidden_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads // tp_size
    head_dim = hidden // config.num_attention_heads
    intermediate = config.intermediate_size // tp_size
    eps = config.rms_norm_eps
    type_vocab_size = config.raw.get("type_vocab_size", 2)
    requested_fp32_layers = frozenset(int(layer) for layer in config.raw.get("_fp32_layers", ()))
    invalid_fp32_layers = sorted(
        layer for layer in requested_fp32_layers if layer < 0 or layer >= num_layers
    )
    if invalid_fp32_layers:
        raise ValueError(f"fp32_layers contains out-of-range indices: {invalid_fp32_layers}")
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "bf16":
        work_np_dtype, work_trt_dtype = np.float16, trt.bfloat16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
        requested_fp32_layers = frozenset()
    else:
        raise ValueError(f"Unsupported BERT precision: {precision}")

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    # Disable TF32 to ensure full FP32 precision. TF32 uses 10-bit mantissa
    # which causes significant accuracy loss across 12+ encoder layers.
    trt_config.clear_flag(trt.BuilderFlag.TF32)

    # Resolve GELU variant from config
    hidden_act = config.hidden_act or config.raw.get("activation", "") or "gelu"

    # -------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------
    input_ids = network.add_input("input_ids", trt.int32, (max_seq_length,))
    attention_mask_input = network.add_input("attention_mask", trt.int32, (max_seq_length,))

    # token_type_ids: constant zeros (all segment-0) — the C++ encoder
    # pipeline doesn't provide this input, and inference is single-segment.
    tt_zeros = network.add_constant(
        (max_seq_length,), trt.Weights(np.zeros(max_seq_length, dtype=np.int32))
    )
    token_type_ids = tt_zeros.get_output(0)

    # -------------------------------------------------------------------
    # Shared constants
    # -------------------------------------------------------------------
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
        network, (type_vocab_size, hidden), weights["token_type_embedding"], dtype=work_np_dtype
    )

    # Build additive attention mask from attention_mask input:
    # attention_mask is [seq_len] with 1=real, 0=padding.
    # Convert to [1, 1, seq_len] additive mask: 0.0 for real, -1e10 for padding.
    mask_float = network.add_cast(attention_mask_input, work_trt_dtype)
    ones_mask = add_constant(
        network, (1,), np.array([1.0], dtype=work_np_dtype), dtype=work_np_dtype
    )
    mask_penalty = -1e4 if precision in {"fp16", "bf16"} else -1e10
    neg_large = add_constant(
        network, (1,), np.array([mask_penalty], dtype=work_np_dtype), dtype=work_np_dtype
    )
    inv_mask = network.add_elementwise(
        ones_mask, mask_float.get_output(0), trt.ElementWiseOperation.SUB
    )  # 0 for real, 1 for pad
    pad_penalty = network.add_elementwise(
        inv_mask.get_output(0), neg_large, trt.ElementWiseOperation.PROD
    )  # 0.0 for real, -1e10 for pad
    # Reshape to [1, 1, seq_len] for broadcasting: [num_heads, seq_len, seq_len]
    pad_mask_reshape = network.add_shuffle(pad_penalty.get_output(0))
    pad_mask_reshape.reshape_dims = (1, 1, max_seq_length)

    # Position indices: [0, 1, 2, ..., max_seq_length-1]
    position_indices = network.add_constant(
        (max_seq_length,), trt.Weights(np.arange(max_seq_length, dtype=np.int32))
    )

    # -------------------------------------------------------------------
    # Embedding: word + position + token_type + LayerNorm
    # -------------------------------------------------------------------
    word_embed = network.add_gather(embedding_table, input_ids, 0)
    pos_embed = network.add_gather(position_embed_table, position_indices.get_output(0), 0)
    tt_embed = network.add_gather(token_type_table, token_type_ids, 0)

    # Sum all three embedding types
    embed_sum1 = network.add_elementwise(
        word_embed.get_output(0), pos_embed.get_output(0), trt.ElementWiseOperation.SUM
    )
    embed_sum2 = network.add_elementwise(
        embed_sum1.get_output(0), tt_embed.get_output(0), trt.ElementWiseOperation.SUM
    )

    # Embedding LayerNorm
    hidden_state = _add_seq_layer_norm(
        network,
        embed_sum2.get_output(0),
        hidden,
        weights["embed_norm"],
        weights["embed_norm_beta"],
        eps,
        dtype=work_np_dtype,
    )

    # -------------------------------------------------------------------
    # Encoder layers
    # -------------------------------------------------------------------
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        layer_is_fp32 = layer_idx in requested_fp32_layers
        layer_np_dtype = np.float32 if layer_is_fp32 else work_np_dtype
        layer_trt_dtype = trt.float32 if layer_is_fp32 else work_trt_dtype

        def _cast_layer_dtype(tensor: trt.ITensor) -> trt.ITensor:
            if tensor.dtype == layer_trt_dtype:
                return tensor
            return network.add_cast(tensor, layer_trt_dtype).get_output(0)

        hidden_state = _add_encoder_layer(
            network=network,
            hidden=_cast_layer_dtype(hidden_state),
            weights=weights,
            prefix=prefix,
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_heads=num_heads,
            head_dim=head_dim,
            seq_length=max_seq_length,
            attn_mask=_cast_layer_dtype(pad_mask_reshape.get_output(0)),
            hidden_act=hidden_act,
            eps=eps,
            tp_size=tp_size,
            dtype=layer_np_dtype,
        )
        if hidden_state.dtype != work_trt_dtype:
            hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)

    # -------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------
    public_output = hidden_state
    if public_output.dtype != trt.float32:
        public_output = network.add_cast(public_output, trt.float32).get_output(0)
    public_output.name = "hidden_states"
    network.mark_output(public_output)

    # -------------------------------------------------------------------
    # Build engine
    # -------------------------------------------------------------------
    if verbose:
        print(
            f"[trtmc build] Building BERT encoder TRT engine "
            f"({num_layers} layers, hidden={hidden}, tp={tp_size}, "
            f"seq_len={max_seq_length}, precision={precision}) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")

    return bytes(plan)


def build_tp_encoder_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_seq_length: int,
    *,
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    """Build one rank-local BERT tensor-parallel engine."""
    parallel = parallel_config or ParallelConfig()
    parallel.validate()
    if not parallel.enabled:
        raise ValueError("build_tp_encoder_engine requires tensor_parallel mode and tp_size > 1")
    return build_encoder_engine(
        config,
        weights,
        max_seq_length,
        verbose=verbose,
        parallel_config=parallel,
    )


def _add_seq_layer_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """LayerNorm over the hidden dimension of each sequence row."""
    return add_layer_norm_native(network, inp, hidden_size, gamma, beta, eps, dtype=dtype)


def _add_encoder_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    head_dim: int,
    seq_length: int,
    attn_mask: trt.ITensor,
    hidden_act: str = "gelu",
    eps: float,
    tp_size: int = 1,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Add one BERT encoder layer with POST-norm.

    BERT architecture per layer:
        attn_out = MultiHeadSelfAttention(hidden)
        hidden = LayerNorm(hidden + attn_out)  # post-norm
        ffn_out = FFN(hidden)
        hidden = LayerNorm(hidden + ffn_out)   # post-norm
    """
    attention_size = num_heads * head_dim

    # --- Self-attention (no causal mask, bidirectional) ---
    # QKV projections
    q = add_matmul_rhs_constant(
        network, hidden, hidden_size, attention_size, weights[f"{prefix}.w_q"], dtype=dtype
    )
    k = add_matmul_rhs_constant(
        network, hidden, hidden_size, attention_size, weights[f"{prefix}.w_k"], dtype=dtype
    )
    v = add_matmul_rhs_constant(
        network, hidden, hidden_size, attention_size, weights[f"{prefix}.w_v"], dtype=dtype
    )

    # QKV biases
    q = add_bias_sum(network, q, attention_size, weights[f"{prefix}.q_bias"], dtype=dtype)
    k = add_bias_sum(network, k, attention_size, weights[f"{prefix}.k_bias"], dtype=dtype)
    v = add_bias_sum(network, v, attention_size, weights[f"{prefix}.v_bias"], dtype=dtype)

    mask_4d = network.add_shuffle(attn_mask)
    mask_4d.reshape_dims = (1, 1, 1, seq_length)

    context_flat = add_attention_from_rows(
        network,
        q,
        k,
        v,
        num_heads=num_heads,
        head_dim=head_dim,
        sequence_length=seq_length,
        mask=mask_4d.get_output(0),
        tag=prefix + ".attn",
    )

    # Output projection
    attn_out = add_matmul_rhs_constant(
        network, context_flat, attention_size, hidden_size, weights[f"{prefix}.w_o"], dtype=dtype
    )
    if tp_size > 1:
        attn_out = add_all_reduce_sum(network, attn_out, tp_size)
    attn_out = add_bias_sum(
        network, attn_out, hidden_size, weights[f"{prefix}.o_bias"], dtype=dtype
    )

    # POST-norm: LayerNorm(hidden + attn_out)
    residual1 = network.add_elementwise(hidden, attn_out, trt.ElementWiseOperation.SUM)
    normed1 = _add_seq_layer_norm(
        network,
        residual1.get_output(0),
        hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        weights[f"{prefix}.post_attn_norm_beta"],
        eps,
        dtype=dtype,
    )

    # --- FFN: fc1 -> GELU -> fc2 ---
    fc1 = add_matmul_rhs_constant(
        network, normed1, hidden_size, intermediate_size, weights[f"{prefix}.w_fc1"], dtype=dtype
    )
    fc1 = add_bias_sum(network, fc1, intermediate_size, weights[f"{prefix}.fc1_bias"], dtype=dtype)
    activated = add_activation(network, fc1, hidden_act, dtype=dtype)
    fc2 = add_matmul_rhs_constant(
        network, activated, intermediate_size, hidden_size, weights[f"{prefix}.w_fc2"], dtype=dtype
    )
    if tp_size > 1:
        fc2 = add_all_reduce_sum(network, fc2, tp_size)
    fc2 = add_bias_sum(network, fc2, hidden_size, weights[f"{prefix}.fc2_bias"], dtype=dtype)

    # POST-norm: LayerNorm(normed1 + ffn_out)
    residual2 = network.add_elementwise(normed1, fc2, trt.ElementWiseOperation.SUM)
    normed2 = _add_seq_layer_norm(
        network,
        residual2.get_output(0),
        hidden_size,
        weights[f"{prefix}.output_norm"],
        weights[f"{prefix}.output_norm_beta"],
        eps,
        dtype=dtype,
    )

    return normed2


name = "bert"


def load_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    """Load BERT checkpoint weights through the owner-local mapper."""
    from .weights import load_bert_weights

    return load_bert_weights(model_dir, config)


def build_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx=None,
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    """Build one single-device or rank-local BERT encoder engine."""
    if quant_ctx is not None:
        raise ValueError("BERT builds do not support quantization")

    parallel = parallel_config or ParallelConfig()
    parallel.validate()
    if parallel.enabled:
        parallel.validate()
        if precision != "fp32":
            raise ValueError("BERT tensor-parallel builds currently support only fp32")
        return build_tp_encoder_engine(
            config,
            weights,
            max_seq_length=max_cache_length,
            verbose=verbose,
            parallel_config=parallel,
        )

    return build_encoder_engine(
        config,
        weights,
        max_seq_length=max_cache_length,
        precision=precision,
        verbose=verbose,
    )


def _detect_tokenizer_frame(source: str) -> tuple[list[int], list[int]] | None:
    """Return the exact BERT special-token prefix and suffix, when detectable."""
    from transformers import AutoTokenizer

    kwargs = {"trust_remote_code": True}
    if not Path(source).is_dir():
        kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    default_ids = list(tokenizer.encode("hello"))
    plain_ids = list(tokenizer.encode("hello", add_special_tokens=False))

    if default_ids == plain_ids:
        return [], []
    if not plain_ids:
        return default_ids, []
    for start in range(len(default_ids) - len(plain_ids) + 1):
        if default_ids[start : start + len(plain_ids)] == plain_ids:
            return default_ids[:start], default_ids[start + len(plain_ids) :]
    return None


def _wordpiece_tokenizer_needs_rebuild(model_dir: Path) -> bool:
    """Detect an existing WordPiece tokenizer that omits source vocabulary IDs."""
    tokenizer_path = model_dir / "tokenizer.json"
    vocab_path = model_dir / "vocab.txt"
    config_path = model_dir / "config.json"
    if not (tokenizer_path.is_file() and vocab_path.is_file() and config_path.is_file()):
        return False
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer_model = tokenizer.get("model", {})
    vocab = tokenizer_model.get("vocab", {})
    expected_vocab_size = int(config.get("vocab_size", 0))
    if tokenizer_model.get("type") != "WordPiece":
        return False
    if not isinstance(vocab, dict) or not vocab or expected_vocab_size <= 0:
        return False
    tokenizer_id_space = max(int(token_id) for token_id in vocab.values()) + 1
    source_id_space = len(vocab_path.read_text(encoding="utf-8").splitlines())
    return tokenizer_id_space < expected_vocab_size and source_id_space >= expected_vocab_size


def _ensure_tokenizer_json(model_dir: Path) -> None:
    """Create or repair BERT tokenizer.json without rewriting sibling metadata."""
    import tempfile

    tokenizer_path = model_dir / "tokenizer.json"
    rebuild = _wordpiece_tokenizer_needs_rebuild(model_dir)
    if tokenizer_path.is_file() and not rebuild:
        return

    temporary_path: Path | None = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        with tempfile.TemporaryDirectory(prefix="trtmc-bert-tokenizer-") as temporary:
            generated = Path(temporary) / "tokenizer.json"
            tokenizer.backend_tokenizer.save(str(generated))
            if not generated.is_file():
                raise RuntimeError("tokenizer conversion did not create tokenizer.json")
            with tempfile.NamedTemporaryFile(
                dir=model_dir,
                prefix=".trtmc-bert-tokenizer-",
                suffix=".json",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                output.write(generated.read_bytes())
            temporary_path.replace(tokenizer_path)
            temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one BERT bundle without shared model orchestration."""
    if request.image_height is not None:
        raise NotImplementedError("bert does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("bert does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("bert does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("bert does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task not in {"encoding", "embedding", "reranking"}:
        raise ValueError("BERT supports encoding, embedding, and reranking")
    model_path = Path(request.model_dir)
    config = ModelConfig.from_dir(model_path)
    if config.model_type != "bert":
        raise ValueError(f"BERT requires model_type='bert', got {config.model_type!r}")
    precision = request.precision.lower()
    if precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError(f"BERT does not support precision {precision!r}")
    if request.quantization is not None:
        raise ValueError("BERT does not support quantization")
    if request.tensor_parallel_size not in {1, 2, 4, 8}:
        raise ValueError("BERT tensor_parallel_size must be 1, 2, 4, or 8")
    if request.tensor_parallel_size > 1 and precision != "fp32":
        raise ValueError("BERT tensor parallel currently requires precision='fp32'")

    sequence_length = request.max_sequence_length or 256
    if sequence_length > config.max_position_embeddings:
        raise ValueError(
            "BERT max_sequence_length exceeds checkpoint capacity: "
            f"{sequence_length} > {config.max_position_embeddings}"
        )
    config.raw["_fp32_layers"] = sorted(set(request.fp32_layers))
    weights = load_weights(str(model_path), config)
    parallel = ParallelConfig(tp_size=request.tensor_parallel_size)
    parallel.validate()

    writer.set_header(
        family="bert",
        task=request.task,
        backend="trt",
    )
    if parallel.enabled:
        for rank in range(parallel.tp_size):
            plan = build_engine(
                config,
                weights,
                sequence_length,
                precision=precision,
                verbose=request.verbose,
                parallel_config=parallel.for_rank(rank),
            )
            writer.add_bytes(f"engine.rank{rank}.plan", plan)
    else:
        plan = build_engine(
            config,
            weights,
            sequence_length,
            precision=precision,
            verbose=request.verbose,
            parallel_config=parallel,
        )
        writer.add_bytes("engine.plan", plan)

    tokenizer_frame = _detect_tokenizer_frame(str(model_path))
    _ensure_tokenizer_json(model_path)
    if tokenizer_frame is None:
        tokenizer_frame = _detect_tokenizer_frame(str(model_path))
    if tokenizer_frame is None:
        raise RuntimeError("BERT tokenizer special-token framing could not be determined")
    prefix_ids, suffix_ids = tokenizer_frame
    runtime_config = {
        "tensor_parallel_size": parallel.tp_size,
        "tokenizer_add_special_tokens": False,
        "tokenizer_prefix_ids": prefix_ids,
        "tokenizer_suffix_ids": suffix_ids,
    }
    writer.add_json("runtime.json", runtime_config)
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        path = model_path / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
