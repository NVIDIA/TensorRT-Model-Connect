"""EnCodec decoder TRT graph builder.

Builds a TRT engine for the EnCodec neural audio codec decoder.
Architecture:
  Input: audio_codes [1, 8, T] (8 codebooks, T timesteps)
  -> Codebook lookup + sum
  -> Conv1d input
  -> LSTM (unrolled for TRT)
  -> 4 upsample stages (ConvTranspose1d + 2x residual blocks with ELU)
  -> Conv1d output
  -> Tanh
  Output: waveform [1, 1, T*320]

Weight norm fusion: v * (g / ||v||_2) => fused_weight
"""

from __future__ import annotations

import sys

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops


trt = trt_compat.get_trt()

def _fuse_weight_norm(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Fuse weight_norm: weight = g * v / ||v||_2.

    g: [out_channels, 1, 1] or [out_channels]
    v: [out_channels, in_channels, kernel_size]
    """
    g = g.astype(np.float32).flatten()
    v = v.astype(np.float32)
    # Compute L2 norm over (in_channels, kernel_size) dims
    norm = np.sqrt(np.sum(v ** 2, axis=tuple(range(1, v.ndim)), keepdims=True) + 1e-12)
    # g shape: broadcast to match v
    g_shaped = g.reshape(-1, *([1] * (v.ndim - 1)))
    return (g_shaped * v / norm).astype(np.float32)


def build_encodec_decoder_engine(
    state_dict: dict,
    prefix: str = "codec_model.decoder.",
    num_codebooks: int = 8,
    codebook_size: int = 1024,
    codebook_dim: int = 128,
    seq_length: int = 512,
    *,
    verbose: bool = False,
) -> bytes:
    """Build TRT engine for EnCodec decoder.

    Args:
        state_dict: Full model state dict with numpy arrays.
        prefix: Key prefix for decoder weights.
        num_codebooks: Number of VQ codebooks (default 8).
        codebook_size: Codebook vocabulary size (default 1024).
        codebook_dim: Codebook embedding dimension (default 128).
        seq_length: Input sequence length (default 512).
        verbose: Enable verbose TRT logging.

    Returns:
        Serialized TRT engine plan bytes.
    """
    def _to_np(key):
        t = state_dict[key]
        if hasattr(t, 'numpy'):
            return t.numpy().astype(np.float32)
        return np.asarray(t, dtype=np.float32)

    def _has_key(key):
        return key in state_dict

    def _get_fused_conv_weight(w_g_key, w_v_key):
        if _has_key(w_g_key) and _has_key(w_v_key):
            return _fuse_weight_norm(_to_np(w_g_key), _to_np(w_v_key))
        # Fallback: try direct weight key
        fallback = w_v_key.replace("weight_v", "weight")
        if _has_key(fallback):
            return _to_np(fallback)
        raise KeyError(f"Cannot find conv weight: {w_g_key} or {fallback}")

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    # Input: audio_codes [1, num_codebooks, seq_length] (int32)
    audio_codes = network.add_input(
        "audio_codes", trt.int32, (1, num_codebooks, seq_length))

    # Codebook lookup and sum
    # Each codebook: [codebook_size, codebook_dim]
    # Gather per codebook, then sum across codebooks
    quantizer_prefix = "codec_model.quantizer.layers."
    codebook_embeds = []
    for cb in range(num_codebooks):
        cb_key = f"{quantizer_prefix}{cb}._codebook.embed"
        if not _has_key(cb_key):
            cb_key = f"{quantizer_prefix}{cb}.codebook.embed"
        embed_table = graph_ops.add_constant(
            network, (codebook_size, codebook_dim), _to_np(cb_key))

        # Extract codes for this codebook: [1, seq_length]
        slice_layer = network.add_slice(
            audio_codes,
            start=(0, cb, 0),
            shape=(1, 1, seq_length),
            stride=(1, 1, 1))
        codes_flat = network.add_shuffle(slice_layer.get_output(0))
        codes_flat.reshape_dims = (seq_length,)

        # Gather: [seq_length] -> [seq_length, codebook_dim]
        gathered = network.add_gather(embed_table, codes_flat.get_output(0), 0)
        codebook_embeds.append(gathered.get_output(0))

    # Sum all codebook embeddings
    summed = codebook_embeds[0]
    for i in range(1, num_codebooks):
        add_layer = network.add_elementwise(
            summed, codebook_embeds[i], trt.ElementWiseOperation.SUM)
        summed = add_layer.get_output(0)

    # summed: [seq_length, codebook_dim] -> [1, codebook_dim, seq_length] for Conv1d
    reshape_3d = network.add_shuffle(summed)
    reshape_3d.reshape_dims = (1, seq_length, codebook_dim)
    transpose_3d = network.add_shuffle(reshape_3d.get_output(0))
    transpose_3d.first_transpose = trt.Permutation([0, 2, 1])
    x = transpose_3d.get_output(0)  # [1, codebook_dim, seq_length]

    # === Input Conv1d (model.0): codebook_dim -> 512, k=7, causal ===
    input_conv_w = _get_fused_conv_weight(
        f"{prefix}layers.0.conv.weight_g", f"{prefix}layers.0.conv.weight_v")
    input_conv_b = _to_np(f"{prefix}layers.0.conv.bias") if _has_key(
        f"{prefix}layers.0.conv.bias") else None
    x = graph_ops.add_causal_pad_1d(network, x, 6)  # (7-1)*1 = 6
    x = graph_ops.add_conv1d(network, x, input_conv_w, input_conv_b, 512, 7)

    # === LSTM (model.1): 2 layers, hidden_size=512, with residual ===
    # Permute [1, 512, T] -> [1, T, 512] for LSTM
    lstm_perm_in = network.add_shuffle(x)
    lstm_perm_in.first_transpose = trt.Permutation([0, 2, 1])
    lstm_x = lstm_perm_in.get_output(0)
    lstm_residual = lstm_x

    for layer_i in range(2):
        w_ih = _to_np(f"{prefix}layers.1.lstm.weight_ih_l{layer_i}")
        w_hh = _to_np(f"{prefix}layers.1.lstm.weight_hh_l{layer_i}")
        b_ih = _to_np(f"{prefix}layers.1.lstm.bias_ih_l{layer_i}")
        b_hh = _to_np(f"{prefix}layers.1.lstm.bias_hh_l{layer_i}")
        lstm_x = graph_ops.add_lstm_unrolled(
            network, lstm_x, w_ih, w_hh, b_ih, b_hh, 512, seq_length)

    # Residual: lstm_output + lstm_input
    lstm_sum = network.add_elementwise(
        lstm_x, lstm_residual, trt.ElementWiseOperation.SUM)
    # Permute back [1, T, 512] -> [1, 512, T]
    lstm_perm_out = network.add_shuffle(lstm_sum.get_output(0))
    lstm_perm_out.first_transpose = trt.Permutation([0, 2, 1])
    x = lstm_perm_out.get_output(0)

    # === 4 Upsample stages ===
    # Each stage: ELU -> ConvTranspose1d -> causal trim -> ResBlock
    # EnCodec upsampling_ratios = [8, 5, 4, 2] (already in decoder order)
    upsample_stages = [
        # (deconv_layer_idx, in_ch, out_ch, kernel, stride, resblock_layer_idx)
        (3, 512, 256, 16, 8, 4),
        (6, 256, 128, 10, 5, 7),
        (9, 128, 64, 8, 4, 10),
        (12, 64, 32, 4, 2, 13),
    ]

    for deconv_idx, in_ch, out_ch, kernel, stride, res_idx in upsample_stages:
        # ELU activation (model.{deconv_idx-1} is ELU, no weights)
        x = graph_ops.add_elu(network, x)

        # ConvTranspose1d with weight_norm
        deconv_w = _get_fused_conv_weight(
            f"{prefix}layers.{deconv_idx}.conv.weight_g",
            f"{prefix}layers.{deconv_idx}.conv.weight_v")
        deconv_b = _to_np(f"{prefix}layers.{deconv_idx}.conv.bias") if _has_key(
            f"{prefix}layers.{deconv_idx}.conv.bias") else None
        x = graph_ops.add_conv1d_transpose(
            network, x, deconv_w, deconv_b, out_ch, kernel, stride)

        # Causal trim: trim padding_total from right
        padding_total = kernel - stride
        if padding_total > 0:
            x = graph_ops.add_slice_trim_right(network, x, padding_total)

        # ResBlock: ELU -> Conv1d(out_ch -> hidden, k=3, causal) ->
        #           ELU -> Conv1d(hidden -> out_ch, k=1) + shortcut(out_ch -> out_ch, k=1)
        hidden_ch = out_ch // 2  # compress=2
        res_in = x

        # block.0: ELU, block.1: SConv1d(out_ch -> hidden_ch, k=3, d=1, causal)
        x = graph_ops.add_elu(network, x)
        conv1_w = _get_fused_conv_weight(
            f"{prefix}layers.{res_idx}.block.1.conv.weight_g",
            f"{prefix}layers.{res_idx}.block.1.conv.weight_v")
        conv1_b = _to_np(f"{prefix}layers.{res_idx}.block.1.conv.bias") if _has_key(
            f"{prefix}layers.{res_idx}.block.1.conv.bias") else None
        x = graph_ops.add_causal_pad_1d(network, x, 2)  # (3-1)*1 = 2
        x = graph_ops.add_conv1d(network, x, conv1_w, conv1_b, hidden_ch, 3)

        # block.2: ELU, block.3: SConv1d(hidden_ch -> out_ch, k=1)
        x = graph_ops.add_elu(network, x)
        conv2_w = _get_fused_conv_weight(
            f"{prefix}layers.{res_idx}.block.3.conv.weight_g",
            f"{prefix}layers.{res_idx}.block.3.conv.weight_v")
        conv2_b = _to_np(f"{prefix}layers.{res_idx}.block.3.conv.bias") if _has_key(
            f"{prefix}layers.{res_idx}.block.3.conv.bias") else None
        x = graph_ops.add_conv1d(network, x, conv2_w, conv2_b, out_ch, 1)

        # Shortcut: SConv1d(out_ch -> out_ch, k=1)
        short_w = _get_fused_conv_weight(
            f"{prefix}layers.{res_idx}.shortcut.conv.weight_g",
            f"{prefix}layers.{res_idx}.shortcut.conv.weight_v")
        short_b = _to_np(f"{prefix}layers.{res_idx}.shortcut.conv.bias") if _has_key(
            f"{prefix}layers.{res_idx}.shortcut.conv.bias") else None
        shortcut = graph_ops.add_conv1d(network, res_in, short_w, short_b, out_ch, 1)

        # Residual add
        x = network.add_elementwise(
            x, shortcut, trt.ElementWiseOperation.SUM).get_output(0)

    # === Output: ELU + Conv1d(32 -> 1, k=7, causal) + Tanh ===
    x = graph_ops.add_elu(network, x)
    out_conv_w = _get_fused_conv_weight(
        f"{prefix}layers.15.conv.weight_g", f"{prefix}layers.15.conv.weight_v")
    out_conv_b = _to_np(f"{prefix}layers.15.conv.bias") if _has_key(
        f"{prefix}layers.15.conv.bias") else None
    x = graph_ops.add_causal_pad_1d(network, x, 6)  # (7-1)*1 = 6
    x = graph_ops.add_conv1d(network, x, out_conv_w, out_conv_b, 1, 7)

    # Tanh activation
    x = network.add_activation(x, trt.ActivationType.TANH).get_output(0)

    x.name = "waveform"
    network.mark_output(x)

    if verbose:
        print(f"[trtmc build] Building EnCodec decoder engine "
              f"(codebooks={num_codebooks}, dim={codebook_dim}, seq={seq_length}) ...",
              file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed for EnCodec decoder")
    return bytes(plan)
