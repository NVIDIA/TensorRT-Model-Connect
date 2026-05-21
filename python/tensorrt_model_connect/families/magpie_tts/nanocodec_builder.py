"""NanoCodec HiFi-GAN decoder TRT engine builder.

Builds a TRT engine for the NanoCodec decoder (NeMo nemo-nano-codec).
Architecture (from actual NanoCodec 22kHz checkpoint):
  Input: codec_tokens [8, T] (8 GroupFSQ groups, T frames, int32)
         input_len [1] (int32 scalar: actual number of valid frames)
  -> GroupFSQ dequantize + length mask (zero padded positions)
  -> CausalHiFiGAN decoder with per-conv length masking:
     pre_conv(32->864, k=7, causal) + mask
     -> 5 upsample stages: CausalConvT(grouped) + mask -> Snake -> ResBlocks + mask
     -> HalfSnake -> post_conv(27->1, k=3, causal) + mask -> clamp(-1, 1)
  Output: waveform [1, T*1024] (float32)

Per-conv masking matches NeMo's mask_sequence_tensor: after every causal conv,
positions >= current_len are zeroed. current_len scales by stride at each upsample.
"""

from __future__ import annotations

import sys

import numpy as np
from tensorrt_model_connect import trt_compat

from ... import graph_ops

trt = trt_compat.get_trt()

# ---------------------------------------------------------------------------
# Architecture constants (NanoCodec 22kHz / 1.89kbps / 21.5fps)
# ---------------------------------------------------------------------------

FSQ_LEVELS = [8, 7, 6, 6]
FSQ_TOTAL_CODES = 8 * 7 * 6 * 6     # 2016
NUM_FSQ_GROUPS = 8
FSQ_DIM_PER_GROUP = len(FSQ_LEVELS)  # 4
FSQ_TOTAL_DIM = NUM_FSQ_GROUPS * FSQ_DIM_PER_GROUP  # 32

UPSAMPLE_RATES = [8, 8, 4, 2, 2]
TOTAL_UPSAMPLE = 8 * 8 * 4 * 2 * 2  # 1024
NUM_RES_SUBBLOCKS = 3
RES_DILATIONS = [1, 3, 5]


def _fuse_weight_norm_parametrized(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    g = g.astype(np.float32)
    v = v.astype(np.float32)
    norm = np.sqrt(
        np.sum(v ** 2, axis=tuple(range(1, v.ndim)), keepdims=True) + 1e-12)
    return (g * v / norm).astype(np.float32)


def _make_fsq_lookup_table(levels: list[int]) -> np.ndarray:
    """NeMo FSQ: val = (idx - L//2) / (L//2), basis = cumprod([1, L0, ...])."""
    total = 1
    for lv in levels:
        total *= lv
    basis = [1]
    for lv in levels[:-1]:
        basis.append(basis[-1] * lv)
    ndim = len(levels)
    table = np.zeros((total, ndim), dtype=np.float32)
    for code in range(total):
        for d in range(ndim):
            idx = (code // basis[d]) % levels[d]
            half = levels[d] // 2
            table[code, d] = (idx - half) / half
    return table


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------

def _build_length_mask(network, current_len, max_len):
    """[1,1,max_len] float mask: 1 where pos < current_len, else 0."""
    arange = graph_ops.add_constant(
        network, (1, 1, max_len),
        np.arange(max_len, dtype=np.float32).reshape(1, 1, max_len))
    len_f = network.add_cast(current_len, trt.float32)
    len_r = network.add_shuffle(len_f.get_output(0))
    len_r.reshape_dims = (1, 1, 1)
    diff = network.add_elementwise(
        len_r.get_output(0), arange,
        trt.ElementWiseOperation.SUB).get_output(0)
    clip = network.add_activation(diff, trt.ActivationType.CLIP)
    clip.alpha = 0.0
    clip.beta = 1.0
    return network.add_unary(
        clip.get_output(0), trt.UnaryOperation.FLOOR).get_output(0)


def _apply_mask(network, x, mask):
    return network.add_elementwise(
        x, mask, trt.ElementWiseOperation.PROD).get_output(0)


def _scale_len(network, current_len, factor):
    fc = network.add_constant(
        (1,), trt.Weights(np.array([factor], dtype=np.int32)))
    return network.add_elementwise(
        current_len, fc.get_output(0),
        trt.ElementWiseOperation.PROD).get_output(0)


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def _apply_snake_core(network, x, alpha):
    ax = network.add_elementwise(
        alpha, x, trt.ElementWiseOperation.PROD).get_output(0)
    sin_ax = network.add_unary(ax, trt.UnaryOperation.SIN).get_output(0)
    sin2 = network.add_elementwise(
        sin_ax, sin_ax, trt.ElementWiseOperation.PROD).get_output(0)
    inv_a = network.add_unary(
        alpha, trt.UnaryOperation.RECIP).get_output(0)
    scaled = network.add_elementwise(
        inv_a, sin2, trt.ElementWiseOperation.PROD).get_output(0)
    return network.add_elementwise(
        x, scaled, trt.ElementWiseOperation.SUM).get_output(0)


def _add_snake(network, x, alpha_np, channels):
    """Snake (direct alpha, no exp). HalfSnake if alpha_ch < channels."""
    c_groups = alpha_np.shape[1]
    alpha = graph_ops.add_constant(
        network, tuple(alpha_np.shape), alpha_np.astype(np.float32))
    if c_groups == channels:
        return _apply_snake_core(network, x, alpha)
    half = c_groups
    _, _, length = x.shape
    x1 = network.add_slice(
        x, start=(0, 0, 0), shape=(1, half, length), stride=(1, 1, 1))
    rest = channels - half
    x2 = network.add_slice(
        x, start=(0, half, 0), shape=(1, rest, length), stride=(1, 1, 1))
    x1_s = _apply_snake_core(network, x1.get_output(0), alpha)
    lrelu = network.add_activation(
        x2.get_output(0), trt.ActivationType.LEAKY_RELU)
    lrelu.alpha = 0.01
    cat = network.add_concatenation([x1_s, lrelu.get_output(0)])
    cat.axis = 1
    return cat.get_output(0)


# ---------------------------------------------------------------------------
# Conv helpers (with mask)
# ---------------------------------------------------------------------------

def _add_causal_conv1d_wn(network, x, sd, prefix, out_ch, kernel, mask,
                           dilation=1):
    """Causal Conv1d + length mask."""
    g = sd[f"{prefix}.parametrizations.weight.original0"].astype(np.float32)
    v = sd[f"{prefix}.parametrizations.weight.original1"].astype(np.float32)
    weight = _fuse_weight_norm_parametrized(g, v)
    bk = f"{prefix}.bias"
    bias = sd[bk].astype(np.float32) if bk in sd else None
    pad_left = (kernel - 1) * dilation
    _, c_in, length = x.shape
    ri = network.add_shuffle(x)
    ri.reshape_dims = (1, c_in, 1, length)
    inp = ri.get_output(0)
    if pad_left > 0:
        p = network.add_padding_nd(inp, pre_padding=(0, pad_left),
                                   post_padding=(0, 0))
        inp = p.get_output(0)
    w4 = np.ascontiguousarray(
        weight.reshape(weight.shape[0], weight.shape[1], 1, kernel))
    cw = trt.Weights(w4)
    cb = (trt.Weights(np.ascontiguousarray(bias))
          if bias is not None else trt.Weights())
    conv = network.add_convolution_nd(
        inp, num_output_maps=out_ch,
        kernel_shape=(1, kernel), kernel=cw, bias=cb)
    conv.stride_nd = (1, 1)
    conv.dilation_nd = (1, dilation)
    ol = conv.get_output(0).shape[3]
    ro = network.add_shuffle(conv.get_output(0))
    ro.reshape_dims = (1, out_ch, ol)
    return _apply_mask(network, ro.get_output(0), mask)


def _add_causal_conv_t1d_wn(network, x, sd, prefix, out_ch, kernel, stride,
                             groups, mask):
    """Causal ConvTranspose1d + length mask."""
    g = sd[f"{prefix}.parametrizations.weight.original0"].astype(np.float32)
    v = sd[f"{prefix}.parametrizations.weight.original1"].astype(np.float32)
    weight = _fuse_weight_norm_parametrized(g, v)
    bk = f"{prefix}.bias"
    bias = sd[bk].astype(np.float32) if bk in sd else None
    _, c_in, length = x.shape
    ri = network.add_shuffle(x)
    ri.reshape_dims = (1, c_in, 1, length)
    opg = out_ch // groups
    w4 = np.ascontiguousarray(
        weight.reshape(c_in, opg, 1, kernel), dtype=np.float32)
    cw = trt.Weights(w4)
    cb = (trt.Weights(np.ascontiguousarray(bias))
          if bias is not None else trt.Weights())
    dc = network.add_deconvolution_nd(
        ri.get_output(0), num_output_maps=out_ch,
        kernel_shape=(1, kernel), kernel=cw, bias=cb)
    dc.stride_nd = (1, stride)
    dc.num_groups = groups
    trim = kernel - stride
    dco = dc.get_output(0)
    olf = dco.shape[3]
    if trim > 0:
        ol = olf - trim
        sl = network.add_slice(dco, start=(0, 0, 0, 0),
                               shape=(1, out_ch, 1, ol), stride=(1, 1, 1, 1))
        r4 = sl.get_output(0)
    else:
        r4 = dco
        ol = olf
    ro = network.add_shuffle(r4)
    ro.reshape_dims = (1, out_ch, ol)
    return _apply_mask(network, ro.get_output(0), mask)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_nanocodec_decoder_engine(
    codec_state_dict: dict,
    max_frames: int = 512,
    *,
    verbose: bool = False,
) -> bytes:
    """Build TRT engine for NanoCodec HiFi-GAN decoder.

    Inputs:
        codec_tokens [8, max_frames] int32
        input_len    [1]             int32 (actual valid frames)
    Output:
        waveform [1, max_frames*1024] float32
    """
    sd = {}
    for k, v in codec_state_dict.items():
        arr = v.numpy() if hasattr(v, 'numpy') else np.asarray(v)
        sd[k] = arr.astype(np.float32) if arr.dtype != np.float32 else arr

    max_frames = ((max_frames + 63) // 64) * 64

    num_res_groups = 0
    while f"audio_decoder.res_layers.0.res_blocks.{num_res_groups}.res_blocks.0.input_conv.conv.bias" in sd:
        num_res_groups += 1

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    tc = builder.create_builder_config()
    tc.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    # === Inputs ===
    codec_tokens = network.add_input(
        "codec_tokens", trt.int32, (NUM_FSQ_GROUPS, max_frames))
    input_len = network.add_input("input_len", trt.int32, (1,))

    # === FSQ dequantize ===
    fsq_table = graph_ops.add_constant(
        network, (FSQ_TOTAL_CODES, FSQ_DIM_PER_GROUP),
        _make_fsq_lookup_table(FSQ_LEVELS))
    group_latents = []
    for g in range(NUM_FSQ_GROUPS):
        gs = network.add_slice(
            codec_tokens, start=(g, 0), shape=(1, max_frames), stride=(1, 1))
        gf = network.add_shuffle(gs.get_output(0))
        gf.reshape_dims = (max_frames,)
        group_latents.append(
            network.add_gather(fsq_table, gf.get_output(0), 0).get_output(0))
    cat = network.add_concatenation(group_latents)
    cat.axis = 1
    perm = network.add_shuffle(cat.get_output(0))
    perm.first_transpose = trt.Permutation([1, 0])
    x3 = network.add_shuffle(perm.get_output(0))
    x3.reshape_dims = (1, FSQ_TOTAL_DIM, max_frames)
    x = x3.get_output(0)

    # === Length tracking ===
    cur_len = input_len
    cur_max = max_frames
    mask = _build_length_mask(network, cur_len, cur_max)

    # Zero padded FSQ positions
    x = _apply_mask(network, x, mask)

    # === Pre-conv + mask ===
    pp = "audio_decoder.pre_conv.conv"
    pv = sd[f"{pp}.parametrizations.weight.original1"]
    x = _add_causal_conv1d_wn(
        network, x, sd, pp, pv.shape[0], pv.shape[2], mask)
    cur_ch = pv.shape[0]

    # === 5 upsample stages ===
    # NeMo order: Snake -> ConvTranspose -> ResBlocks (NOT ConvT -> Snake -> RB)
    for si, rate in enumerate(UPSAMPLE_RATES):
        # Scale length for this upsample
        cur_len = _scale_len(network, cur_len, rate)
        cur_max *= rate
        mask = _build_length_mask(network, cur_len, cur_max)

        up = f"audio_decoder.up_sample_conv_layers.{si}.conv"
        uv = sd[f"{up}.parametrizations.weight.original1"]
        ub = sd.get(f"{up}.bias")
        oc = ub.shape[0] if ub is not None else cur_ch // 2
        uk = uv.shape[2]

        # Snake FIRST (on pre-upsample channels)
        sk = (f"audio_decoder.activations.{si}"
              f".activation.snake_act.alpha")
        x = _add_snake(network, x, sd[sk], cur_ch)

        # Then ConvTranspose
        x = _add_causal_conv_t1d_wn(
            network, x, sd, up, oc, uk, rate, groups=oc, mask=mask)
        cur_ch = oc

        # ResBlocks: parallel groups, averaged
        gouts = []
        for j in range(num_res_groups):
            gx = x
            for k, dil in enumerate(RES_DILATIONS):
                r = (f"audio_decoder.res_layers.{si}"
                     f".res_blocks.{j}.res_blocks.{k}")
                res = gx

                gx = _add_snake(
                    network, gx,
                    sd[f"{r}.input_activation.activation.snake_act.alpha"],
                    cur_ch)
                ic = f"{r}.input_conv.conv"
                ick = sd[f"{ic}.parametrizations.weight.original1"].shape[2]
                gx = _add_causal_conv1d_wn(
                    network, gx, sd, ic, cur_ch, ick, mask, dilation=dil)

                gx = _add_snake(
                    network, gx,
                    sd[f"{r}.skip_activation.activation.snake_act.alpha"],
                    cur_ch)
                sc = f"{r}.skip_conv.conv"
                sck = sd[f"{sc}.parametrizations.weight.original1"].shape[2]
                gx = _add_causal_conv1d_wn(
                    network, gx, sd, sc, cur_ch, sck, mask)

                gx = network.add_elementwise(
                    res, gx,
                    trt.ElementWiseOperation.SUM).get_output(0)

            gouts.append(gx)

        x = gouts[0]
        for go in gouts[1:]:
            x = network.add_elementwise(
                x, go, trt.ElementWiseOperation.SUM).get_output(0)
        nc = graph_ops.add_constant(
            network, (1, 1, 1),
            np.array([1.0 / num_res_groups], dtype=np.float32))
        x = network.add_elementwise(
            x, nc, trt.ElementWiseOperation.PROD).get_output(0)

    # === Post ===
    psk = "audio_decoder.post_activation.activation.snake_act.alpha"
    x = _add_snake(network, x, sd[psk], cur_ch)

    pp2 = "audio_decoder.post_conv.conv"
    pv2 = sd[f"{pp2}.parametrizations.weight.original1"]
    x = _add_causal_conv1d_wn(
        network, x, sd, pp2, pv2.shape[0], pv2.shape[2], mask)

    clip = network.add_activation(x, trt.ActivationType.CLIP)
    clip.alpha = -1.0
    clip.beta = 1.0
    x = clip.get_output(0)

    sq = network.add_shuffle(x)
    sq.reshape_dims = (1, -1)
    wf = sq.get_output(0)
    wf.name = "waveform"
    network.mark_output(wf)

    if verbose:
        ns = max_frames * TOTAL_UPSAMPLE
        print(f"[trtmc build] Building NanoCodec decoder engine "
              f"(max_frames={max_frames}, samples={ns}, "
              f"res_groups={num_res_groups}) ...", file=sys.stderr)

    plan = builder.build_serialized_network(network, tc)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed for NanoCodec")
    return bytes(plan)
