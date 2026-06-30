"""Family-owned TensorRT model graph and utility implementation."""

from __future__ import annotations


import numpy as np
from tensorrt_model_connect import trt_compat
import sys
from typing import TYPE_CHECKING
from ..config import ModelConfig
# Graph Ops


trt = trt_compat.get_trt()


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
    """Dispatch activation by name: 'silu', 'gelu_new', 'gelu', 'relu', 'relu2'/'squared_relu'."""
    if activation_type in ("gelu_new", "gelu"):
        return add_gelu_new(network, inp, dtype=dtype)
    elif activation_type == "relu":
        act = network.add_activation(inp, trt.ActivationType.RELU)
        return act.get_output(0)
    elif activation_type in ("relu2", "squared_relu"):
        relu = network.add_activation(inp, trt.ActivationType.RELU)
        sq = network.add_elementwise(
            relu.get_output(0), relu.get_output(0), trt.ElementWiseOperation.PROD
        )
        return sq.get_output(0)
    elif activation_type == "silu":
        sigmoid = network.add_activation(inp, trt.ActivationType.SIGMOID)
        swish = network.add_elementwise(inp, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
        return swish.get_output(0)
    else:
        raise ValueError(f"Unsupported activation: {activation_type}")


# Alias: add_gelu_tanh is the same as add_gelu_new (tanh approximation)
add_gelu_tanh = add_gelu_new


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
    # TensorRT 11 removed the Python INormalizationLayer.compute_precision
    # attribute. Keep the TRT 10 hint, and let TRT 11 infer the precision.
    if hasattr(norm, "compute_precision"):
        norm.compute_precision = trt.float32
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


def add_2d_mask_to_4d(
    network: trt.INetworkDefinition,
    mask_2d: trt.ITensor,
) -> trt.ITensor:
    """Reshape additive attention mask [Sq, K] to [1, 1, Sq, K]."""
    mask_shape = network.add_shape(mask_2d).get_output(0)
    ones = add_constant(network, (2,), np.array([1, 1], dtype=np.int64), dtype=np.int64)
    target = network.add_concatenation([ones, mask_shape])
    target.axis = 0
    mask_4d = network.add_shuffle(mask_2d)
    mask_4d.set_input(1, target.get_output(0))
    return mask_4d.get_output(0)


def add_attention_core(
    network: trt.INetworkDefinition,
    q_4d: trt.ITensor,
    k_4d: trt.ITensor,
    v_4d: trt.ITensor,
    causal: bool = False,
    mask: trt.ITensor | None = None,
    scale: float | None = None,
    fp32_accumulation: bool = False,
) -> trt.ITensor:
    """Scaled dot-product attention via TRT native IAttention layer.

    Replaces the manual Q@K^T → scale → softmax → @V chain.  TRT 10 fuses
    this into a single kernel when a compatible implementation is available;
    decomposable=True ensures a correct fallback to primitives otherwise.

    NOTE: TRT IAttention computes raw BMM1 = Q @ K^T without any built-in
    1/sqrt(D) scaling.  We pre-scale Q by 1/sqrt(D) so that the fused kernel
    computes the standard scaled dot-product attention formula.

    Args:
        q_4d:    Query  [B, H, q_seq, D].
        k_4d:    Key    [B, H, kv_seq, D].
        v_4d:    Value  [B, H, kv_seq, D].
        causal:  Apply causal (autoregressive) mask.  Mutually exclusive
                 with ``mask``.
        mask:    Optional additive float mask [B, H, q_seq, kv_seq] added
                 to scaled logits before softmax.  Cannot be used with
                 causal=True.
        scale:   Optional Q pre-scale factor.  Defaults to 1/sqrt(D).
        fp32_accumulation:
                 Cast Q/K/V to FP32 before IAttention, then cast the context
                 back to the original Q dtype.  TRT may still select a
                 Half-input fused MHA tactic after optimizing the casts, while
                 keeping the IAttention accumulation/output boundary in FP32.

    Returns:
        Context tensor [B, H, q_seq, D].
    """
    output_dtype = q_4d.dtype
    if fp32_accumulation and output_dtype != trt.float32:
        q_4d = network.add_cast(q_4d, trt.float32).get_output(0)
        k_4d = network.add_cast(k_4d, trt.float32).get_output(0)
        v_4d = network.add_cast(v_4d, trt.float32).get_output(0)
        if mask is not None and mask.dtype != trt.float32:
            mask = network.add_cast(mask, trt.float32).get_output(0)

    # Pre-scale Q: TRT IAttention does not apply score scaling itself.
    # Match the scale constant's dtype to Q's dtype: in strongly-typed networks
    # a FP32 constant mixed with a FP16/BF16 Q causes add_elementwise to emit
    # a type-mismatch error and produce a tensor with corrupted dimensions,
    # which makes add_attention return None.
    if scale is None:
        head_dim = q_4d.shape[-1]
        scale = float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0
    # Use FP16 weights directly for FP16; BF16 has no numpy native type so
    # create as FP32 and cast; FP32 falls through to the default.
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
        causal,
    )
    # Allow TRT to decompose into primitive ops when no fused kernel is
    # available (e.g. unsupported head-dim or dtype).  This guarantees
    # correctness on any configuration at the cost of potential performance.
    attn.decomposable = True
    if mask is not None and not causal:
        attn.mask = mask
    return _cast_back_to_trt_dtype(network, attn.get_output(0), output_dtype)


def _scalar_constant_for_trt_dtype(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    value: float,
    dtype: trt.DataType,
) -> trt.ITensor:
    np_dtype = np.float16 if dtype == trt.float16 else np.float32
    const = add_constant(network, shape, np.full(shape, value, dtype=np_dtype), dtype=np_dtype)
    if dtype == trt.bfloat16:
        const = network.add_cast(const, trt.bfloat16).get_output(0)
    return const


def add_tanh_softcap(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    cap: float,
    *,
    scalar_shape: tuple[int, ...],
) -> trt.ITensor:
    """Apply ``tanh(tensor / cap) * cap`` using scalar broadcasting."""
    cap_t = _scalar_constant_for_trt_dtype(network, scalar_shape, float(cap), tensor.dtype)
    scaled = network.add_elementwise(tensor, cap_t, trt.ElementWiseOperation.DIV).get_output(0)
    capped = network.add_activation(scaled, trt.ActivationType.TANH).get_output(0)
    return network.add_elementwise(capped, cap_t, trt.ElementWiseOperation.PROD).get_output(0)


def _repeat_kv_heads_4d(
    network: trt.INetworkDefinition,
    x_4d: trt.ITensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> trt.ITensor:
    if num_kv_heads == num_heads:
        return x_4d
    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}")

    repeat = num_heads // num_kv_heads
    if num_kv_heads == 1:
        concat = network.add_concatenation([x_4d] * repeat)
        concat.axis = 1
        return concat.get_output(0)

    x_shape = network.add_shape(x_4d).get_output(0)
    one = add_constant(network, (1,), np.array([1], dtype=np.int64), dtype=np.int64)
    seq = network.add_slice(x_shape, start=(2,), shape=(1,), stride=(1,))
    dim = add_constant(network, (1,), np.array([head_dim], dtype=np.int64), dtype=np.int64)
    slice_shape = network.add_concatenation([one, one, seq.get_output(0), dim])
    slice_shape.axis = 0

    repeated = []
    for head_idx in range(num_kv_heads):
        head_slice = network.add_slice(
            x_4d, start=(0, head_idx, 0, 0), shape=(1, 1, 1, head_dim), stride=(1, 1, 1, 1)
        )
        head_slice.set_input(2, slice_shape.get_output(0))
        repeated.extend([head_slice.get_output(0)] * repeat)

    concat = network.add_concatenation(repeated)
    concat.axis = 1
    return concat.get_output(0)


def _add_attention_core_with_logit_softcap(
    network: trt.INetworkDefinition,
    q_4d: trt.ITensor,
    k_4d: trt.ITensor,
    v_4d: trt.ITensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    mask: trt.ITensor | None,
    scale: float,
    logit_softcap: float,
) -> trt.ITensor:
    output_dtype = q_4d.dtype
    k_4d = _repeat_kv_heads_4d(
        network, k_4d, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    v_4d = _repeat_kv_heads_4d(
        network, v_4d, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
    )

    score_q = q_4d
    score_k = k_4d
    score_mask = mask
    if output_dtype != trt.float32:
        score_q = network.add_cast(score_q, trt.float32).get_output(0)
        score_k = network.add_cast(score_k, trt.float32).get_output(0)
        if score_mask is not None and score_mask.dtype != trt.float32:
            score_mask = network.add_cast(score_mask, trt.float32).get_output(0)

    scale_t = _scalar_constant_for_trt_dtype(network, (1, 1, 1, 1), scale, score_q.dtype)
    scores = network.add_matrix_multiply(
        score_q, trt.MatrixOperation.NONE, score_k, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    scores = network.add_elementwise(scores, scale_t, trt.ElementWiseOperation.PROD).get_output(0)

    scores = add_tanh_softcap(network, scores, logit_softcap, scalar_shape=(1, 1, 1, 1))

    if score_mask is not None:
        scores = network.add_elementwise(
            scores, score_mask, trt.ElementWiseOperation.SUM
        ).get_output(0)

    probs = network.add_softmax(scores)
    probs.axes = 1 << 3
    probs_t = probs.get_output(0)
    if probs_t.dtype != output_dtype:
        probs_t = network.add_cast(probs_t, output_dtype).get_output(0)

    context = network.add_matrix_multiply(
        probs_t, trt.MatrixOperation.NONE, v_4d, trt.MatrixOperation.NONE
    ).get_output(0)
    return _cast_back_to_trt_dtype(network, context, output_dtype)


def add_attention_from_rows(
    network: trt.INetworkDefinition,
    q: trt.ITensor,
    k: trt.ITensor,
    v: trt.ITensor,
    *,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int | None = None,
    q_seq: int | None,
    kv_seq: int | None,
    causal: bool = False,
    mask: trt.ITensor | None = None,
    scale: float | None = None,
    logit_softcap: float | None = None,
    fp32_accumulation: bool = False,
    tag: str | None = None,
) -> trt.ITensor:
    """Native IAttention for row-major [S, H * D] Q/K/V tensors.

    ``num_kv_heads`` can be smaller than ``num_heads`` for GQA/MQA. TRT
    native IAttention supports this directly, so callers should not expand K/V
    heads unless the model semantics require per-query-head K/V values.
    """
    attention_size = num_heads * head_dim
    kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    q_4d = reshape_rows_to_heads_4d(
        network,
        q,
        num_heads,
        head_dim,
        sequence_length=q_seq,
        tag=None if tag is None else tag + ".q",
    )
    k_4d = reshape_rows_to_heads_4d(
        network,
        k,
        kv_heads,
        head_dim,
        sequence_length=kv_seq,
        tag=None if tag is None else tag + ".k",
    )
    v_4d = reshape_rows_to_heads_4d(
        network,
        v,
        kv_heads,
        head_dim,
        sequence_length=kv_seq,
        tag=None if tag is None else tag + ".v",
    )
    if scale is None:
        scale = float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0
    if logit_softcap is not None and float(logit_softcap) > 0.0:
        if causal:
            raise NotImplementedError("logit_softcap attention requires an explicit additive mask")
        ctx_4d = _add_attention_core_with_logit_softcap(
            network,
            q_4d,
            k_4d,
            v_4d,
            num_heads=num_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            mask=mask,
            scale=scale,
            logit_softcap=float(logit_softcap),
        )
    else:
        ctx_4d = add_attention_core(
            network,
            q_4d,
            k_4d,
            v_4d,
            causal=causal,
            mask=mask,
            scale=scale,
            fp32_accumulation=fp32_accumulation,
        )
    return reshape_heads_4d_to_rows(
        network,
        ctx_4d,
        attention_size,
        sequence_length=q_seq,
        tag=None if tag is None else tag + ".ctx",
    )


# Backward-compatible name used by existing tests and call sites.
_add_attention_core = add_attention_core


# Builder


trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ..weights import WeightDict


def build_convbert_encoder_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_seq_length: int,
    *,
    verbose: bool = False,
) -> bytes:
    """Build a TRT engine plan for ConvBERT encoder."""
    hidden = config.hidden_size
    embedding_size = config.raw.get("embedding_size", hidden)
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    eps = config.rms_norm_eps  # layer_norm_eps
    type_vocab_size = config.raw.get("type_vocab_size", 2)
    hidden_act = config.hidden_act or "gelu"
    intermediate = config.intermediate_size

    # ConvBERT specific
    new_num_heads = int(weights["_convbert_new_num_heads"][0])
    head_size = int(weights["_convbert_head_size"][0])
    all_head_size = int(weights["_convbert_all_head_size"][0])
    conv_kernel_size = int(weights["_convbert_conv_kernel_size"][0])

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    trt_config.clear_flag(trt.BuilderFlag.TF32)

    S = max_seq_length  # alias for brevity

    # -------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------
    input_ids = network.add_input("input_ids", trt.int32, (S,))
    attention_mask_input = network.add_input("attention_mask", trt.int32, (S,))

    # token_type_ids: constant zeros (all segment-0) — the C++ encoder
    # pipeline doesn't provide this input, and inference is single-segment.
    tt_zeros = network.add_constant((S,), trt.Weights(np.zeros(S, dtype=np.int32)))
    token_type_ids = tt_zeros.get_output(0)

    # -------------------------------------------------------------------
    # Shared constants
    # -------------------------------------------------------------------
    embedding_table = add_constant(network, (vocab, embedding_size), weights["embedding"])
    position_embed_table = add_constant(
        network, weights["position_embedding"].shape, weights["position_embedding"]
    )
    token_type_table = add_constant(
        network, (type_vocab_size, embedding_size), weights["token_type_embedding"]
    )

    # Additive attention mask: [1, 1, S]
    mask_float = network.add_cast(attention_mask_input, trt.float32)
    ones_mask = add_constant(network, (1,), np.array([1.0], dtype=np.float32))
    neg_large = add_constant(network, (1,), np.array([-1e10], dtype=np.float32))
    inv_mask = network.add_elementwise(
        ones_mask, mask_float.get_output(0), trt.ElementWiseOperation.SUB
    )
    pad_penalty = network.add_elementwise(
        inv_mask.get_output(0), neg_large, trt.ElementWiseOperation.PROD
    )
    pad_mask_reshape = network.add_shuffle(pad_penalty.get_output(0))
    pad_mask_reshape.reshape_dims = (1, 1, S)
    attn_mask = pad_mask_reshape.get_output(0)

    # Position indices
    position_indices = add_constant(network, (S,), np.arange(S, dtype=np.int32).astype(np.float32))
    pos_int = network.add_cast(position_indices, trt.int32)

    # -------------------------------------------------------------------
    # Embedding: word + position + token_type + LayerNorm
    # -------------------------------------------------------------------
    word_embed = network.add_gather(embedding_table, input_ids, 0)
    pos_embed = network.add_gather(position_embed_table, pos_int.get_output(0), 0)
    tt_embed = network.add_gather(token_type_table, token_type_ids, 0)

    embed_sum1 = network.add_elementwise(
        word_embed.get_output(0), pos_embed.get_output(0), trt.ElementWiseOperation.SUM
    )
    embed_sum2 = network.add_elementwise(
        embed_sum1.get_output(0), tt_embed.get_output(0), trt.ElementWiseOperation.SUM
    )

    hidden_state = _add_seq_layer_norm(
        network,
        embed_sum2.get_output(0),
        embedding_size,
        S,
        weights["embed_norm"],
        weights["embed_norm_beta"],
        eps,
    )

    # -------------------------------------------------------------------
    # Encoder layers
    # -------------------------------------------------------------------
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"

        hidden_state = _add_convbert_layer(
            network=network,
            hidden=hidden_state,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden,
            intermediate_size=intermediate,
            new_num_heads=new_num_heads,
            head_size=head_size,
            all_head_size=all_head_size,
            conv_kernel_size=conv_kernel_size,
            seq_length=S,
            attn_mask=attn_mask,
            hidden_act=hidden_act,
            eps=eps,
        )

    # -------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------
    hidden_state.name = "hidden_states"
    network.mark_output(hidden_state)

    # -------------------------------------------------------------------
    # Build engine
    # -------------------------------------------------------------------
    if verbose:
        print(
            f"[trtmc build] Building ConvBERT encoder TRT engine "
            f"({num_layers} layers, hidden={hidden}, "
            f"seq_len={S}, conv_kernel={conv_kernel_size}) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")

    return bytes(plan)


def _add_seq_layer_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    seq_length: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float,
) -> trt.ITensor:
    """LayerNorm over [seq_len, hidden] using TRT native normalization."""
    return add_layer_norm_native(network, inp, hidden_size, gamma, beta, eps)


def _add_separable_conv1d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    all_head_size: int,
    conv_kernel_size: int,
    seq_length: int,
    dw_weight: np.ndarray,
    pw_weight: np.ndarray,
    bias: np.ndarray,
) -> trt.ITensor:
    """SeparableConv1D: depthwise conv1d + pointwise conv1d + bias.

    Input: [seq_len, hidden_size] (our 2D layout)
    Output: [seq_len, all_head_size]
    """
    pad = conv_kernel_size // 2

    # Reshape: [seq, hidden] -> [1, hidden, seq, 1] for TRT Conv2d
    shuf_in = network.add_shuffle(inp)
    shuf_in.first_transpose = trt.Permutation([1, 0])  # [hidden, seq]
    shuf_in.reshape_dims = (1, hidden_size, seq_length, 1)

    # Depthwise convolution: kernel [hidden, 1, kernel_size, 1]
    dw_w_4d = dw_weight.reshape(hidden_size, 1, conv_kernel_size, 1)
    dw_trt = trt.Weights(np.ascontiguousarray(dw_w_4d, dtype=np.float32))
    dw_conv = network.add_convolution_nd(
        shuf_in.get_output(0),
        num_output_maps=hidden_size,
        kernel_shape=(conv_kernel_size, 1),
        kernel=dw_trt,
    )
    dw_conv.padding_nd = (pad, 0)
    dw_conv.num_groups = hidden_size

    # Pointwise conv: kernel [all_head_size, hidden, 1, 1]
    pw_w_4d = pw_weight.reshape(all_head_size, hidden_size, 1, 1)
    pw_trt = trt.Weights(np.ascontiguousarray(pw_w_4d, dtype=np.float32))
    pw_conv = network.add_convolution_nd(
        dw_conv.get_output(0),
        num_output_maps=all_head_size,
        kernel_shape=(1, 1),
        kernel=pw_trt,
    )

    # Add bias: [1, all_head_size, 1, 1]
    bias_4d = add_constant(network, (1, all_head_size, 1, 1), bias.reshape(1, all_head_size, 1, 1))
    biased = network.add_elementwise(pw_conv.get_output(0), bias_4d, trt.ElementWiseOperation.SUM)

    # Reshape: [1, all_head_size, seq, 1] -> [seq, all_head_size]
    squeeze = network.add_shuffle(biased.get_output(0))
    squeeze.reshape_dims = (all_head_size, seq_length)
    squeeze.second_transpose = trt.Permutation([1, 0])

    return squeeze.get_output(0)


def _add_unfold(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    channels: int,
    seq_length: int,
    kernel_size: int,
) -> trt.ITensor:
    """Unfold (im2col) for 1D signal.

    Input: [channels, seq_length] (channel-first)
    Output: [channels * kernel_size, seq_length]

    For each position p, gathers values at positions [p-pad, ..., p+pad]
    for each channel, with zero-padding at boundaries.

    Uses TRT slice layers with zero-fill mode for out-of-bounds positions.
    """
    pad = kernel_size // 2
    shifts = []

    # Expand to 4D: [1, channels, seq_length, 1] for TRT padding (requires 4D)
    expand = network.add_shuffle(inp)
    expand.reshape_dims = (1, channels, seq_length, 1)
    inp_4d = expand.get_output(0)

    for k in range(kernel_size):
        offset = k - pad  # shift amount: -pad to +pad

        if offset == 0:
            # No shift needed
            identity = network.add_shuffle(inp_4d)
            identity.reshape_dims = (channels, seq_length)
            shifts.append(identity.get_output(0))
        elif offset < 0:
            # Pad left (prepend zeros along H=seq dim), slice from start
            abs_off = -offset
            # For 4D [N,C,H,W], padding is 2D: (H_pad, W_pad)
            pad_layer = network.add_padding_nd(
                inp_4d,
                pre_padding=(abs_off, 0),
                post_padding=(0, 0),
            )
            # Slice from start: [1, channels, seq_length, 1]
            sl = network.add_slice(
                pad_layer.get_output(0),
                start=(0, 0, 0, 0),
                shape=(1, channels, seq_length, 1),
                stride=(1, 1, 1, 1),
            )
            reshape = network.add_shuffle(sl.get_output(0))
            reshape.reshape_dims = (channels, seq_length)
            shifts.append(reshape.get_output(0))
        else:
            # Pad right (append zeros along H=seq dim), slice from offset
            pad_layer = network.add_padding_nd(
                inp_4d,
                pre_padding=(0, 0),
                post_padding=(offset, 0),
            )
            # Slice from offset: [1, channels, seq_length, 1]
            sl = network.add_slice(
                pad_layer.get_output(0),
                start=(0, 0, offset, 0),
                shape=(1, channels, seq_length, 1),
                stride=(1, 1, 1, 1),
            )
            reshape = network.add_shuffle(sl.get_output(0))
            reshape.reshape_dims = (channels, seq_length)
            shifts.append(reshape.get_output(0))

    # Concatenate along channel dim: [channels * kernel_size, seq_length]
    if len(shifts) == 1:
        return shifts[0]
    cat = network.add_concatenation(shifts)
    cat.axis = 0  # channel dim
    return cat.get_output(0)


def _add_convbert_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    intermediate_size: int,
    new_num_heads: int,
    head_size: int,
    all_head_size: int,
    conv_kernel_size: int,
    seq_length: int,
    attn_mask: trt.ITensor,
    hidden_act: str,
    eps: float,
) -> trt.ITensor:
    """Add one ConvBERT encoder layer with mixed attention + dynamic convolution."""
    S = seq_length

    # === Branch 1: Standard multi-head self-attention ===
    q = add_matmul_rhs_constant(
        network, hidden, hidden_size, all_head_size, weights[f"{prefix}.w_q"]
    )
    k = add_matmul_rhs_constant(
        network, hidden, hidden_size, all_head_size, weights[f"{prefix}.w_k"]
    )
    v = add_matmul_rhs_constant(
        network, hidden, hidden_size, all_head_size, weights[f"{prefix}.w_v"]
    )

    q = add_bias_sum(network, q, all_head_size, weights[f"{prefix}.q_bias"])
    k = add_bias_sum(network, k, all_head_size, weights[f"{prefix}.k_bias"])
    v = add_bias_sum(network, v, all_head_size, weights[f"{prefix}.v_bias"])

    mixed_query = q  # save for conv branch

    # Key padding mask broadcasts across every query row.
    mask_row = network.add_shuffle(attn_mask)
    mask_row.reshape_dims = (1, S)
    zero_col = add_constant(network, (S, 1), np.zeros((S, 1), dtype=np.float32))
    mask_2d = network.add_elementwise(
        zero_col, mask_row.get_output(0), trt.ElementWiseOperation.SUM
    )
    mask_4d = add_2d_mask_to_4d(network, mask_2d.get_output(0))

    context = add_attention_from_rows(
        network,
        q,
        k,
        v,
        num_heads=new_num_heads,
        head_dim=head_size,
        q_seq=S,
        kv_seq=S,
        mask=mask_4d,
    )

    context_perm = network.add_shuffle(context)
    context_perm.reshape_dims = (S, new_num_heads, head_size)

    # === Branch 2: Span-based dynamic convolution ===

    # SeparableConv1D on hidden_states
    key_conv_attn = _add_separable_conv1d(
        network,
        hidden,
        hidden_size,
        all_head_size,
        conv_kernel_size,
        S,
        weights[f"{prefix}.sep_conv_dw"],
        weights[f"{prefix}.sep_conv_pw"],
        weights[f"{prefix}.sep_conv_bias"],
    )

    # conv_attn = key_conv_attn * query (element-wise)
    conv_attn = network.add_elementwise(key_conv_attn, mixed_query, trt.ElementWiseOperation.PROD)

    # conv_kernel = linear(conv_attn) -> [seq, num_heads * kernel_size]
    conv_kernel = add_matmul_rhs_constant(
        network,
        conv_attn.get_output(0),
        all_head_size,
        new_num_heads * conv_kernel_size,
        weights[f"{prefix}.conv_kernel_w"],
    )
    conv_kernel = add_bias_sum(
        network,
        conv_kernel,
        new_num_heads * conv_kernel_size,
        weights[f"{prefix}.conv_kernel_bias"],
    )

    # Reshape to [seq * num_heads, kernel_size, 1], softmax on kernel_size dim
    ck_reshape = network.add_shuffle(conv_kernel)
    ck_reshape.reshape_dims = (S * new_num_heads, conv_kernel_size, 1)
    ck_softmax = network.add_softmax(ck_reshape.get_output(0))
    ck_softmax.axes = 1 << 1

    # conv_out = linear(hidden) -> [seq, all_head_size]
    conv_out = add_matmul_rhs_constant(
        network, hidden, hidden_size, all_head_size, weights[f"{prefix}.conv_out_w"]
    )
    conv_out = add_bias_sum(network, conv_out, all_head_size, weights[f"{prefix}.conv_out_bias"])

    # Transpose to [all_head_size, seq] for unfold
    conv_out_t = network.add_shuffle(conv_out)
    conv_out_t.first_transpose = trt.Permutation([1, 0])

    # Unfold: [kernel_size * all_head_size, seq] (kernel-major ordering)
    unfolded = _add_unfold(network, conv_out_t.get_output(0), all_head_size, S, conv_kernel_size)

    # Rearrange from kernel-major [K*C, seq] to channel-major [C*K, seq]
    # Reshape to [K, C, seq], permute to [C, K, seq], reshape to [C*K, seq]
    unf_reorder = network.add_shuffle(unfolded)
    unf_reorder.reshape_dims = (conv_kernel_size, all_head_size, S)
    unf_reorder.second_transpose = trt.Permutation([1, 0, 2])
    # Now [all_head_size, kernel_size, seq]

    # Transpose to [seq, all_head_size, kernel_size]
    unf_to_seq_first = network.add_shuffle(unf_reorder.get_output(0))
    unf_to_seq_first.first_transpose = trt.Permutation([2, 0, 1])
    # [seq, all_head_size, kernel_size]

    # Reshape to [seq * num_heads, head_size, kernel_size]
    unf_reshape = network.add_shuffle(unf_to_seq_first.get_output(0))
    unf_reshape.reshape_dims = (S * new_num_heads, head_size, conv_kernel_size)

    # Matmul: [S*H, head_size, K] @ [S*H, K, 1] -> [S*H, head_size, 1]
    conv_result = network.add_matrix_multiply(
        unf_reshape.get_output(0),
        trt.MatrixOperation.NONE,
        ck_softmax.get_output(0),
        trt.MatrixOperation.NONE,
    )

    # Reshape to [S, new_num_heads, head_size]
    conv_reshaped = network.add_shuffle(conv_result.get_output(0))
    conv_reshaped.reshape_dims = (S, new_num_heads, head_size)

    # === Concatenate: [S, 2*num_heads, head_size] ===
    cat = network.add_concatenation([context_perm.get_output(0), conv_reshaped.get_output(0)])
    cat.axis = 1

    # Flatten: [S, hidden_size]
    cat_flat = network.add_shuffle(cat.get_output(0))
    cat_flat.reshape_dims = (S, 2 * new_num_heads * head_size)

    # === Output projection ===
    attn_out = add_matmul_rhs_constant(
        network, cat_flat.get_output(0), hidden_size, hidden_size, weights[f"{prefix}.w_o"]
    )
    attn_out = add_bias_sum(network, attn_out, hidden_size, weights[f"{prefix}.o_bias"])

    # POST-norm
    residual1 = network.add_elementwise(hidden, attn_out, trt.ElementWiseOperation.SUM)
    normed1 = _add_seq_layer_norm(
        network,
        residual1.get_output(0),
        hidden_size,
        S,
        weights[f"{prefix}.post_attn_norm"],
        weights[f"{prefix}.post_attn_norm_beta"],
        eps,
    )

    # === FFN ===
    fc1 = add_matmul_rhs_constant(
        network, normed1, hidden_size, intermediate_size, weights[f"{prefix}.w_fc1"]
    )
    fc1 = add_bias_sum(network, fc1, intermediate_size, weights[f"{prefix}.fc1_bias"])
    activated = add_activation(network, fc1, hidden_act)
    fc2 = add_matmul_rhs_constant(
        network, activated, intermediate_size, hidden_size, weights[f"{prefix}.w_fc2"]
    )
    fc2 = add_bias_sum(network, fc2, hidden_size, weights[f"{prefix}.fc2_bias"])

    # POST-norm
    residual2 = network.add_elementwise(normed1, fc2, trt.ElementWiseOperation.SUM)
    normed2 = _add_seq_layer_norm(
        network,
        residual2.get_output(0),
        hidden_size,
        S,
        weights[f"{prefix}.output_norm"],
        weights[f"{prefix}.output_norm_beta"],
        eps,
    )

    return normed2
