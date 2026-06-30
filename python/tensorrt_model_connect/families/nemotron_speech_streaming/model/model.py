"""Family-owned TensorRT model graph and utility implementation."""

from __future__ import annotations


import numpy as np
from tensorrt_model_connect import trt_compat
from typing import TYPE_CHECKING
import io
import json
import math
import sys
import tarfile
from pathlib import Path
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


def add_layer_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """LayerNorm: gamma * ((x - mean) / sqrt(var + eps)) + beta.

    FP32 precision boundary: when dtype != float32, casts to FP32 before
    norm computation for numerical stability, then casts back.
    """
    need_cast = dtype != np.float32
    output_dtype = inp.dtype
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
        eps_tensor = network.add_cast(eps_tensor, trt.float32).get_output(0)
    # mean = reduce_mean(x)
    mean = network.add_reduce(inp, trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    # x - mean
    centered = network.add_elementwise(inp, mean.get_output(0), trt.ElementWiseOperation.SUB)
    # variance = mean((x - mean)^2)
    sq = network.add_elementwise(
        centered.get_output(0), centered.get_output(0), trt.ElementWiseOperation.PROD
    )
    var = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    # sqrt(var + eps)
    denom_in = network.add_elementwise(var.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    # normalized = (x - mean) / sqrt(var + eps)
    normalized = network.add_elementwise(
        centered.get_output(0), recip.get_output(0), trt.ElementWiseOperation.PROD
    )
    # gamma * normalized + beta
    gamma_t = add_constant(network, (1, hidden_size), gamma, dtype=np.float32)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD
    )
    beta_t = add_constant(network, (1, hidden_size), beta, dtype=np.float32)
    result = network.add_elementwise(scaled.get_output(0), beta_t, trt.ElementWiseOperation.SUM)
    result = result.get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    return result


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


# ---------------------------------------------------------------------------
# Conv / Norm / Resize ops for segmentation and audio models
# ---------------------------------------------------------------------------


def add_conv2d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    out_channels: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    groups: int = 1,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """2D convolution wrapper.

    Input: [N, C_in, H, W]
    Weight: [C_out, C_in/groups, kH, kW]
    Output: [N, C_out, H', W']
    """
    conv_w = trt.Weights(np.ascontiguousarray(weight, dtype=dtype))
    conv_b = trt.Weights()
    if bias is not None:
        conv_b = trt.Weights(np.ascontiguousarray(bias, dtype=dtype))

    conv = network.add_convolution_nd(
        inp,
        num_output_maps=out_channels,
        kernel_shape=kernel_size,
        kernel=conv_w,
        bias=conv_b,
    )
    conv.stride_nd = stride
    conv.padding_nd = padding
    conv.num_groups = groups
    return conv.get_output(0)


def add_batch_norm_2d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_channels: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float = 1e-5,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Fused BatchNorm2d: gamma * (x - mean) / sqrt(var + eps) + beta.

    Input: [N, C, H, W]
    Output: same shape

    FP32 precision boundary: when dtype != float32, casts to FP32 before
    norm computation for numerical stability, then casts back.
    """
    need_cast = dtype != np.float32
    output_dtype = inp.dtype
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
    # Fuse into scale + shift
    scale = gamma / np.sqrt(running_var + eps)
    shift = beta - running_mean * scale

    scale_t = add_constant(
        network,
        (1, num_channels, 1, 1),
        scale.reshape(1, -1, 1, 1).astype(np.float32),
        dtype=np.float32,
    )
    shift_t = add_constant(
        network,
        (1, num_channels, 1, 1),
        shift.reshape(1, -1, 1, 1).astype(np.float32),
        dtype=np.float32,
    )

    scaled = network.add_elementwise(inp, scale_t, trt.ElementWiseOperation.PROD)
    result = network.add_elementwise(
        scaled.get_output(0), shift_t, trt.ElementWiseOperation.SUM
    ).get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    return result


def add_conv1d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """1D convolution via 2D convolution with height=1.

    Input: [N, C_in, L]
    Weight: [C_out, C_in/groups, K]
    Output: [N, C_out, L']
    """
    # Reshape to [N, C_in, 1, L]
    n, c_in, length = inp.shape
    reshape_in = network.add_shuffle(inp)
    reshape_in.reshape_dims = (n, c_in, 1, length)

    # Weight: [C_out, C_in/groups, K] -> [C_out, C_in/groups, 1, K]
    w_4d = weight.reshape(out_channels, -1, 1, kernel_size)
    result = add_conv2d(
        network,
        reshape_in.get_output(0),
        w_4d,
        bias,
        out_channels,
        kernel_size=(1, kernel_size),
        stride=(1, stride),
        padding=(0, padding),
        groups=groups,
        dtype=dtype,
    )

    # Reshape back to [N, C_out, L']
    out_length = result.shape[3]
    reshape_out = network.add_shuffle(result)
    reshape_out.reshape_dims = (n, out_channels, out_length)
    return reshape_out.get_output(0)


# Alias: add_gelu_tanh is the same as add_gelu_new (tanh approximation)
add_gelu_tanh = add_gelu_new


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


# Graph Blocks


trt = trt_compat.get_trt()

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Precision boundary helpers (used by standard_decoder_builder, not inside
# blocks themselves).
# ---------------------------------------------------------------------------


def make_matmul_fn(network, dtype, quant_ctx):
    """Create a matmul callable that routes through quant_ctx if present.

    Returns a function: (lhs, lhs_w, rhs_w, rhs_weights, weight_name) -> ITensor
    """
    if quant_ctx is None:

        def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
            return add_matmul_rhs_constant(network, lhs, lhs_w, rhs_w, rhs_weights, dtype=dtype)

        return matmul
    else:

        def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
            return quant_ctx.maybe_quantized_matmul(
                network, lhs, lhs_w, rhs_w, rhs_weights, weight_name, dtype=dtype
            )

        return matmul


_make_matmul_fn = make_matmul_fn


# Canary Encoder Helpers


trt = trt_compat.get_trt()


def _to_np(t) -> np.ndarray:
    if hasattr(t, "numpy"):
        return t.detach().cpu().numpy().astype(np.float32)
    return np.asarray(t, dtype=np.float32)


def _load_nemo_archive(path: str):
    import torch
    import yaml

    nemo_path = Path(path)
    if nemo_path.is_dir():
        nemo_files = sorted(nemo_path.glob("*.nemo"))
        if nemo_files:
            nemo_path = nemo_files[0]
        else:
            raise FileNotFoundError(f"No .nemo file found in {path}")
    state_dict = config_dict = None
    with tarfile.open(str(nemo_path), "r") as tar:
        for member in tar.getmembers():
            bn = Path(member.name).name
            if bn == "model_weights.ckpt":
                f = tar.extractfile(member)
                if f:
                    state_dict = torch.load(
                        io.BytesIO(f.read()), map_location="cpu", weights_only=False
                    )
            elif bn == "model_config.yaml":
                f = tar.extractfile(member)
                if f:
                    config_dict = yaml.safe_load(f.read())
    if state_dict is None:
        raise FileNotFoundError(f"model_weights.ckpt not found in {nemo_path}")
    if config_dict is None:
        raise FileNotFoundError(f"model_config.yaml not found in {nemo_path}")
    return state_dict, config_dict


def _extract_tokenizer_from_nemo(nemo_path: str, dest_dir: Path) -> None:
    nemo = Path(nemo_path)
    if nemo.is_dir():
        nemo_files = sorted(nemo.glob("*.nemo"))
        if nemo_files:
            nemo = nemo_files[0]
    with tarfile.open(str(nemo), "r") as tar:
        for member in tar.getmembers():
            bn = Path(member.name).name
            if bn.endswith(".model") and "tokenizer" in bn.lower():
                f = tar.extractfile(member)
                if f:
                    (dest_dir / "tokenizer.model").write_bytes(f.read())
                    break
    # Generate a fast tokenizer.json from the SentencePiece model.
    # Use the tokenizers library directly to avoid HF warnings in stdout.
    tok_model_path = dest_dir / "tokenizer.model"
    tok_json_path = dest_dir / "tokenizer.json"
    if tok_model_path.exists() and not tok_json_path.exists():
        try:
            import sentencepiece as spm

            sp = spm.SentencePieceProcessor()
            sp.Load(str(tok_model_path))
            # Build a minimal tokenizer.json compatible with HF fast tokenizer
            vocab = {sp.IdToPiece(i): i for i in range(sp.GetPieceSize())}
            tok_json = {
                "version": "1.0",
                "model": {
                    "type": "Unigram",
                    "unk_id": sp.unk_id(),
                    "vocab": [[piece, 0.0] for piece in vocab],
                },
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": None,
                "post_processor": None,
                "decoder": {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": True},
            }
            tok_json_path.write_text(json.dumps(tok_json))
        except Exception:
            pass

    tok_cfg = dest_dir / "tokenizer_config.json"
    if not tok_cfg.exists():
        tok_cfg.write_text(
            json.dumps(
                {
                    "tokenizer_class": "PreTrainedTokenizerFast",
                },
                indent=2,
            )
        )


def _relative_pe(seq_len: int, d_model: int, max_len: int = 5000) -> np.ndarray:
    """Compute relative PE for Transformer-XL attention.

    Matches NeMo RelPositionalEncoding: builds table with positive and
    negative position encodings, where negative uses sin(-k*d) = -sin(k*d).
    """
    pos = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
    div = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model))

    pe_pos = np.zeros((max_len, d_model), dtype=np.float32)
    pe_pos[:, 0::2] = np.sin(pos * div)
    pe_pos[:, 1::2] = np.cos(pos * div)
    pe_pos = pe_pos[::-1].copy()

    pe_neg = np.zeros((max_len, d_model), dtype=np.float32)
    pe_neg[:, 0::2] = np.sin(-pos * div)
    pe_neg[:, 1::2] = np.cos(-pos * div)
    pe_neg = pe_neg[1:]

    pe_full = np.concatenate([pe_pos, pe_neg], axis=0)
    start = max_len - seq_len
    end = max_len + seq_len - 1
    return pe_full[start:end]


def _compute_enc_seq_len(mel_length: int) -> int:
    """Encoder time output after 3 CausalConv2D stride-2 stages.

    Time dim uses symmetric padding (left=1, right=1): (t+2-3)//2+1 = (t-1)//2+1
    """
    t = mel_length
    for _ in range(3):
        t = (t + 2 - 3) // 2 + 1
    return t


def _compute_causal_enc_seq_len(mel_length: int) -> int:
    t = mel_length
    for _ in range(3):
        t = t // 2 + 1
    return t


# ---------------------------------------------------------------------------
# Encoder TRT graph helpers
# ---------------------------------------------------------------------------


def _build_subsampling(network, mel_input, weights, sub_ch, hidden, num_mel_bins, mel_length):
    causal_downsampling = bool(weights.get("_causal_downsampling", False))

    def add_subsample_conv(inp, weight, bias, out_channels, *, groups=1):
        if causal_downsampling:
            pad = network.add_padding_nd(inp, pre_padding=(2, 2), post_padding=(1, 1))
            inp = pad.get_output(0)
            padding = (0, 0)
        else:
            padding = (1, 1)
        return add_conv2d(
            network,
            inp,
            weight=weight,
            bias=bias,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=padding,
            groups=groups,
        )

    # NeMo ConformerEncoder passes audio as [B, T, F] to pre_encode.
    # MaskedConvSequential.forward unsqueezes to [B, 1, T, F].
    # So Conv2d input is [1, 1, mel_length, mel_bins] (time=H, features=W).
    # Our mel_input is [mel_bins, mel_length] = [F, T]. Transpose to [T, F].
    tr_mel = network.add_shuffle(mel_input)
    tr_mel.first_transpose = trt.Permutation([1, 0])  # [F,T] → [T,F]
    ri = network.add_shuffle(tr_mel.get_output(0))
    ri.reshape_dims = (1, 1, mel_length, num_mel_bins)  # [1, 1, T, F]
    x = ri.get_output(0)
    # Standard Conv2d with symmetric padding + ReLU (NOT SiLU, NOT causal)
    x = add_subsample_conv(x, weights["enc_sub_conv0_w"], weights["enc_sub_conv0_b"], sub_ch)
    x = add_activation(network, x, "relu")
    for s in range(2):
        x = add_subsample_conv(
            x, weights[f"enc_sub_dw{s}_w"], weights[f"enc_sub_dw{s}_b"], sub_ch, groups=sub_ch
        )
        x = add_conv2d(
            network,
            x,
            weight=weights[f"enc_sub_pw{s}_w"],
            bias=weights[f"enc_sub_pw{s}_b"],
            out_channels=sub_ch,
            kernel_size=(1, 1),
        )
        x = add_activation(network, x, "relu")
    # After convs: [1, C, T_out, F_out] where T=time, F=features
    time_out = int(weights.get("_enc_seq", _compute_enc_seq_len(mel_length)))
    sub_out_in = int(weights["enc_sub_out_w"].shape[0])
    feat_out = sub_out_in // sub_ch
    # NeMo: x.transpose(1,2).reshape(B,T,-1) on [B,C,T,F]
    # = permute(0,2,1,3) → [B,T,C,F], reshape → [T, C*F]
    tr = network.add_shuffle(x)
    tr.first_transpose = trt.Permutation([0, 2, 1, 3])  # [B,C,T,F] → [B,T,C,F]
    tr.reshape_dims = (time_out, sub_ch * feat_out)  # [T, C*F]
    out = add_matmul_rhs_constant(
        network, tr.get_output(0), sub_ch * feat_out, hidden, weights["enc_sub_out_w"]
    )
    return add_bias_sum(network, out, hidden, weights["enc_sub_out_b"])


def _rel_shift(network, x, H, S):
    zeros = add_constant(network, (H, S, 1), np.zeros((H, S, 1), dtype=np.float32))
    padded = network.add_concatenation([zeros, x])
    padded.axis = 2
    rs1 = network.add_shuffle(padded.get_output(0))
    rs1.reshape_dims = (H, 2 * S, S)
    sl1 = network.add_slice(
        rs1.get_output(0), start=(0, 1, 0), shape=(H, 2 * S - 1, S), stride=(1, 1, 1)
    )
    rs2 = network.add_shuffle(sl1.get_output(0))
    rs2.reshape_dims = (H, S, 2 * S - 1)
    sl2 = network.add_slice(rs2.get_output(0), start=(0, 0, 0), shape=(H, S, S), stride=(1, 1, 1))
    return sl2.get_output(0)


def _add_rel_pos_attention(
    network, hs, weights, pfx, hidden, H, D, S, rel_pe_proj, eps, enc_mask=None
):
    normed = add_layer_norm(
        network, hs, hidden, weights[f"{pfx}.norm_sa"], weights[f"{pfx}.norm_sa_b"], eps
    )
    q = add_bias_sum(
        network,
        add_matmul_rhs_constant(network, normed, hidden, hidden, weights[f"{pfx}.w_q"]),
        hidden,
        weights[f"{pfx}.b_q"],
    )
    k = add_bias_sum(
        network,
        add_matmul_rhs_constant(network, normed, hidden, hidden, weights[f"{pfx}.w_k"]),
        hidden,
        weights[f"{pfx}.b_k"],
    )
    v = add_bias_sum(
        network,
        add_matmul_rhs_constant(network, normed, hidden, hidden, weights[f"{pfx}.w_v"]),
        hidden,
        weights[f"{pfx}.b_v"],
    )
    qr = network.add_shuffle(q)
    qr.reshape_dims = (S, H, D)
    kr = network.add_shuffle(k)
    kr.reshape_dims = (S, H, D)
    vr = network.add_shuffle(v)
    vr.reshape_dims = (S, H, D)
    bu = add_constant(network, (1, H, D), weights[f"{pfx}.pos_bias_u"])
    bv = add_constant(network, (1, H, D), weights[f"{pfx}.pos_bias_v"])
    qu = network.add_elementwise(qr.get_output(0), bu, trt.ElementWiseOperation.SUM).get_output(0)
    qv = network.add_elementwise(qr.get_output(0), bv, trt.ElementWiseOperation.SUM).get_output(0)
    qu_t = network.add_shuffle(qu)
    qu_t.first_transpose = trt.Permutation([1, 0, 2])
    qv_t = network.add_shuffle(qv)
    qv_t.first_transpose = trt.Permutation([1, 0, 2])
    k_t = network.add_shuffle(kr.get_output(0))
    k_t.first_transpose = trt.Permutation([1, 0, 2])
    v_t = network.add_shuffle(vr.get_output(0))
    v_t.first_transpose = trt.Permutation([1, 0, 2])
    cs = network.add_matrix_multiply(
        qu_t.get_output(0),
        trt.MatrixOperation.NONE,
        k_t.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    rp_t = network.add_shuffle(rel_pe_proj)
    rp_t.first_transpose = trt.Permutation([1, 0, 2])
    ps_raw = network.add_matrix_multiply(
        qv_t.get_output(0),
        trt.MatrixOperation.NONE,
        rp_t.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    ps = _rel_shift(network, ps_raw, H, S)
    total = network.add_elementwise(cs, ps, trt.ElementWiseOperation.SUM).get_output(0)
    sc = add_constant(network, (1, 1, 1), np.array([1.0 / math.sqrt(D)], dtype=np.float32))
    scaled = network.add_elementwise(total, sc, trt.ElementWiseOperation.PROD).get_output(0)
    # Apply encoder sequence mask: [1, 1, S] added to scores [H, S, S]
    if enc_mask is not None:
        scaled = network.add_elementwise(scaled, enc_mask, trt.ElementWiseOperation.SUM).get_output(
            0
        )
    # Conformer relative-position attention uses a rel-shifted Q*R term in
    # the logits, which native IAttention cannot represent as a plain mask.
    sm = network.add_softmax(scaled)
    sm.axes = 1 << 2
    ao = network.add_matrix_multiply(
        sm.get_output(0), trt.MatrixOperation.NONE, v_t.get_output(0), trt.MatrixOperation.NONE
    ).get_output(0)
    at = network.add_shuffle(ao)
    at.first_transpose = trt.Permutation([1, 0, 2])
    af = network.add_shuffle(at.get_output(0))
    af.reshape_dims = (S, hidden)
    return add_bias_sum(
        network,
        add_matmul_rhs_constant(network, af.get_output(0), hidden, hidden, weights[f"{pfx}.w_o"]),
        hidden,
        weights[f"{pfx}.b_o"],
    )


def _add_causal_depthwise_conv1d(network, x, weights, pfx, hidden, kern):
    pad = kern - 1
    if pad > 0:
        zeros = add_constant(
            network, (1, hidden, pad), np.zeros((1, hidden, pad), dtype=np.float32)
        )
        cat = network.add_concatenation([zeros, x])
        cat.axis = 2
        x = cat.get_output(0)
    return add_conv1d(
        network,
        x,
        weight=weights[f"{pfx}.cdw_w"],
        bias=weights[f"{pfx}.cdw_b"],
        out_channels=hidden,
        kernel_size=kern,
        groups=hidden,
    )


def _add_conv_norm(network, x, weights, pfx, hidden, S, eps, conv_norm_type):
    if conv_norm_type == "layer_norm":
        r1 = network.add_shuffle(x)
        r1.reshape_dims = (hidden, S)
        r2 = network.add_shuffle(r1.get_output(0))
        r2.first_transpose = trt.Permutation([1, 0])
        normed = add_layer_norm(
            network, r2.get_output(0), hidden, weights[f"{pfx}.bn_w"], weights[f"{pfx}.bn_b"], eps
        )
        r3 = network.add_shuffle(normed)
        r3.first_transpose = trt.Permutation([1, 0])
        r4 = network.add_shuffle(r3.get_output(0))
        r4.reshape_dims = (1, hidden, S)
        return r4.get_output(0)

    bn = network.add_shuffle(x)
    bn.reshape_dims = (1, hidden, 1, S)
    x = add_batch_norm_2d(
        network,
        bn.get_output(0),
        hidden,
        gamma=weights[f"{pfx}.bn_w"],
        beta=weights[f"{pfx}.bn_b"],
        running_mean=weights[f"{pfx}.bn_m"],
        running_var=weights[f"{pfx}.bn_v"],
    )
    bo = network.add_shuffle(x)
    bo.reshape_dims = (1, hidden, S)
    return bo.get_output(0)


def _add_conv_module(
    network,
    hs,
    weights,
    pfx,
    hidden,
    kern,
    S,
    eps,
    conv_norm_type="batch_norm",
    conv_context_size="symmetric",
):
    normed = add_layer_norm(
        network, hs, hidden, weights[f"{pfx}.norm_conv"], weights[f"{pfx}.norm_conv_b"], eps
    )
    r1 = network.add_shuffle(normed)
    r1.first_transpose = trt.Permutation([1, 0])
    r2 = network.add_shuffle(r1.get_output(0))
    r2.reshape_dims = (1, hidden, S)
    x = add_conv1d(
        network,
        r2.get_output(0),
        weight=weights[f"{pfx}.cpw1_w"],
        bias=weights[f"{pfx}.cpw1_b"],
        out_channels=2 * hidden,
        kernel_size=1,
    )
    xa = network.add_slice(x, start=(0, 0, 0), shape=(1, hidden, S), stride=(1, 1, 1)).get_output(0)
    xb = network.add_slice(
        x, start=(0, hidden, 0), shape=(1, hidden, S), stride=(1, 1, 1)
    ).get_output(0)
    gate = network.add_activation(xb, trt.ActivationType.SIGMOID).get_output(0)
    x = network.add_elementwise(xa, gate, trt.ElementWiseOperation.PROD).get_output(0)
    if conv_context_size == "causal":
        x = _add_causal_depthwise_conv1d(network, x, weights, pfx, hidden, kern)
    else:
        x = add_conv1d(
            network,
            x,
            weight=weights[f"{pfx}.cdw_w"],
            bias=weights[f"{pfx}.cdw_b"],
            out_channels=hidden,
            kernel_size=kern,
            padding=kern // 2,
            groups=hidden,
        )
    x = _add_conv_norm(network, x, weights, pfx, hidden, S, eps, conv_norm_type)
    x = add_activation(network, x, "silu")
    x = add_conv1d(
        network,
        x,
        weight=weights[f"{pfx}.cpw2_w"],
        bias=weights[f"{pfx}.cpw2_b"],
        out_channels=hidden,
        kernel_size=1,
    )
    r3 = network.add_shuffle(x)
    r3.reshape_dims = (hidden, S)
    r4 = network.add_shuffle(r3.get_output(0))
    r4.first_transpose = trt.Permutation([1, 0])
    return r4.get_output(0)


def _add_half_ffn(network, hs, weights, pfx, hidden, ffn, eps):
    normed = add_layer_norm(
        network, hs, hidden, weights[f"{pfx}.norm"], weights[f"{pfx}.norm_b"], eps
    )
    fc1 = add_bias_sum(
        network,
        add_matmul_rhs_constant(network, normed, hidden, ffn, weights[f"{pfx}.w1"]),
        ffn,
        weights[f"{pfx}.b1"],
    )
    act = add_activation(network, fc1, "silu")
    fc2 = add_bias_sum(
        network,
        add_matmul_rhs_constant(network, act, ffn, hidden, weights[f"{pfx}.w2"]),
        hidden,
        weights[f"{pfx}.b2"],
    )
    half = add_constant(network, (1, 1), np.array([0.5], dtype=np.float32))
    return network.add_elementwise(fc2, half, trt.ElementWiseOperation.PROD).get_output(0)


def _add_conformer_block(
    network,
    hs,
    weights,
    pfx,
    hidden,
    H,
    D,
    ffn,
    kern,
    S,
    rpe,
    eps,
    enc_mask=None,
    conv_norm_type="batch_norm",
    conv_context_size="symmetric",
):
    ffn1 = _add_half_ffn(network, hs, weights, f"{pfx}.ff1", hidden, ffn, eps)
    hs = network.add_elementwise(hs, ffn1, trt.ElementWiseOperation.SUM).get_output(0)
    attn = _add_rel_pos_attention(network, hs, weights, pfx, hidden, H, D, S, rpe, eps, enc_mask)
    hs = network.add_elementwise(hs, attn, trt.ElementWiseOperation.SUM).get_output(0)
    conv = _add_conv_module(
        network,
        hs,
        weights,
        pfx,
        hidden,
        kern,
        S,
        eps,
        conv_norm_type=conv_norm_type,
        conv_context_size=conv_context_size,
    )
    hs = network.add_elementwise(hs, conv, trt.ElementWiseOperation.SUM).get_output(0)
    ffn2 = _add_half_ffn(network, hs, weights, f"{pfx}.ff2", hidden, ffn, eps)
    hs = network.add_elementwise(hs, ffn2, trt.ElementWiseOperation.SUM).get_output(0)
    return add_layer_norm(
        network, hs, hidden, weights[f"{pfx}.norm_out"], weights[f"{pfx}.norm_out_b"], eps
    )


def _build_encoder(config, weights, *, verbose=False):
    el = weights["_enc_layers"]
    eh = weights["_enc_heads"]
    h = weights["_hidden"]
    hd = weights["_head_dim"]
    ef = weights["_enc_ffn"]
    k = weights["_kern"]
    mb = weights["_mel_bins"]
    ml = weights["_mel_length"]
    es = weights["_enc_seq"]
    sc = weights["_sub_ch"]
    conv_norm_type = str(weights.get("_conv_norm_type", "batch_norm")).lower()
    conv_context_size = str(weights.get("_conv_context_size", "symmetric")).lower()

    log = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    b = trt.Builder(log)
    net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    tc = b.create_builder_config()
    tc.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    eps = add_constant(net, (1, 1), np.array([1e-5], dtype=np.float32))
    mel = net.add_input("mel_features", trt.float32, (mb, ml))
    # Encoder attention mask: [1, 1, enc_seq] — 0.0 for valid, -10000.0 for padded.
    # Applied additively to self-attention scores before softmax.
    mask_shape = (
        (1, es, es) if bool(weights.get("_encoder_attention_mask_2d", False)) else (1, 1, es)
    )
    enc_mask = net.add_input("encoder_mask", trt.float32, mask_shape)

    hs = _build_subsampling(net, mel, weights, sc, h, mb, ml)
    for li in range(el):
        pfx = f"el.{li}"
        rpe = add_constant(net, (2 * es - 1, eh, hd), weights[f"{pfx}.rpe_proj"])
        hs = _add_conformer_block(
            net,
            hs,
            weights,
            pfx,
            h,
            eh,
            hd,
            ef,
            k,
            es,
            rpe,
            eps,
            enc_mask,
            conv_norm_type=conv_norm_type,
            conv_context_size=conv_context_size,
        )

    hs.name = "encoder_output"
    net.mark_output(hs)
    if verbose:
        print(
            f"[trtmc build] Building Canary encoder ({el}L, h={h}, heads={eh}, seq={es})",
            file=sys.stderr,
        )
    plan = b.build_serialized_network(net, tc)
    if plan is None:
        raise RuntimeError("Canary encoder build failed")
    return bytes(plan)
