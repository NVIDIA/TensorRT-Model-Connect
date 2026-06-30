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


# Backward-compatible name used by existing tests and call sites.
_add_attention_core = add_attention_core


# Fnet Encoder Builder


trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ..weights import WeightDict


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
) -> trt.ITensor:
    """LayerNorm over [seq_len, hidden] using TRT native normalization."""
    return add_layer_norm_native(network, inp, hidden_size, gamma, beta, eps)


def build_fnet_encoder_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_seq_length: int,
    *,
    verbose: bool = False,
) -> bytes:
    """Build a TRT engine plan for FNet encoder with 2D DFT."""
    hidden = config.hidden_size
    num_layers = config.num_hidden_layers
    intermediate = config.intermediate_size
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

    # Inputs
    input_ids = network.add_input("input_ids", trt.int32, (S,))
    network.add_input("attention_mask", trt.int32, (S,))

    # token_type_ids: constant zeros
    tt_zeros = network.add_constant((S,), trt.Weights(np.zeros(S, dtype=np.int32)))
    token_type_ids = tt_zeros.get_output(0)

    # Embedding tables (may use embedding_size != hidden for factorized embeddings)
    embedding_size = weights["embedding"].shape[1]
    embedding_table = add_constant(network, weights["embedding"].shape, weights["embedding"])
    position_embed_table = add_constant(
        network, weights["position_embedding"].shape, weights["position_embedding"]
    )
    token_type_table = add_constant(
        network, (type_vocab_size, embedding_size), weights["token_type_embedding"]
    )

    # Position indices
    position_indices = add_constant(network, (S,), np.arange(S, dtype=np.int32).astype(np.float32))
    pos_int = network.add_cast(position_indices, trt.int32)

    # Embedding: word + position + token_type
    word_embed = network.add_gather(embedding_table, input_ids, 0)
    pos_embed = network.add_gather(position_embed_table, pos_int.get_output(0), 0)
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
    )

    # Optional embedding projection: embedding_size -> hidden_size
    if "embed_projection" in weights:
        hidden_state = add_matmul_rhs_constant(
            network, hidden_state, embedding_size, hidden, weights["embed_projection"]
        )
        if "embed_projection_bias" in weights:
            hidden_state = add_bias_sum(
                network, hidden_state, hidden, weights["embed_projection_bias"]
            )

    # Pre-compute 2D DFT matrices as constants
    # real(DFT2D(X)) = cos_S @ X @ cos_H - sin_S @ X @ sin_H
    cos_s, sin_s = _compute_dft_matrices(S)
    cos_h, sin_h = _compute_dft_matrices(H)

    cos_s_const = add_constant(network, (S, S), cos_s)
    sin_s_const = add_constant(network, (S, S), sin_s)
    cos_h_const = add_constant(network, (H, H), cos_h)
    sin_h_const = add_constant(network, (H, H), sin_h)

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
        )

        # --- FFN ---
        fc1 = add_matmul_rhs_constant(
            network, normed1, hidden, intermediate, weights[f"{prefix}.w_fc1"]
        )
        fc1 = add_bias_sum(network, fc1, intermediate, weights[f"{prefix}.fc1_bias"])
        activated = add_activation(network, fc1, hidden_act)
        fc2 = add_matmul_rhs_constant(
            network, activated, intermediate, hidden, weights[f"{prefix}.w_fc2"]
        )
        fc2 = add_bias_sum(network, fc2, hidden, weights[f"{prefix}.fc2_bias"])

        # POST-norm: residual + LayerNorm after FFN
        residual2 = network.add_elementwise(normed1, fc2, trt.ElementWiseOperation.SUM)
        hidden_state = _add_seq_layer_norm(
            network,
            residual2.get_output(0),
            hidden,
            weights[f"{prefix}.output_norm"],
            weights[f"{prefix}.output_norm_beta"],
            eps,
        )

    # Output
    hidden_state.name = "hidden_states"
    network.mark_output(hidden_state)

    if verbose:
        print(
            f"[trtmc build] Building FNet encoder TRT engine "
            f"({num_layers} layers, hidden={hidden}, "
            f"seq_len={S}) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")

    return bytes(plan)
