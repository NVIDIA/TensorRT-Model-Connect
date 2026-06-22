"""FLUX VAE decoder engine builder.

Builds a TensorRT engine for the FLUX AutoencoderKL decoder using the
TensorRT Python API directly (no ONNX).

Supports both FLUX.1 (AutoencoderKL, 16 latent channels, patch_size=(1,1))
and FLUX.2 (AutoencoderKLFlux2, 32 latent channels, patch_size=(2,2)).

Engine I/O:
    Input:  latents [1, latent_channels, H_lat, W_lat] float32
    Output: image   [1, 3, H_out, W_out] float32 (pixel values in [-1, 1])

    For patch_size=(1,1): H_out = H_lat * 8,  W_out = W_lat * 8
    For patch_size=(2,2): H_out = H_lat * 16, W_out = W_lat * 16

The engine includes the scaling transform: latents / scale_factor + shift_factor.

AutoencoderKL Decoder Architecture (FLUX style):
    post_quant_conv: Conv2d(latent_ch, last_block_ch, 1x1)  [optional]
    mid_block: ResNetBlock2D + SelfAttention2D + ResNetBlock2D
    up_blocks (N blocks, reversed channels):
        Each: (layers_per_block+1) ResNetBlock2D, last N-1 have 2x upsample
    conv_norm_out: GroupNorm + SiLU
    conv_out: Conv2d(first_block_ch, out_ch, 3x3, pad=1)
    unpatchify (if patch_size != (1,1)): pixel-shuffle [B, C*ph*pw, H, W] -> [B, C, H*ph, W*pw]
"""

from __future__ import annotations

import sys
import time

import numpy as np
from tensorrt_model_connect import trt_compat

from ...build_timing import add_trt_compile_timing

trt = trt_compat.get_trt()

# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _load_flux_vae_weights(vae_dir: str) -> dict[str, np.ndarray]:
    """Load AutoencoderKL decoder weights from a diffusers VAE directory.

    Returns a flat dict mapping weight key -> numpy float32 array.
    Only loads decoder + post_quant_conv weights (no encoder).
    """
    from pathlib import Path
    try:
        import ml_dtypes  # noqa: F401 — registers bfloat16 with numpy
    except ImportError:
        pass
    from safetensors import safe_open

    model_path = Path(vae_dir)
    weights: dict[str, np.ndarray] = {}

    # Find safetensors files
    st_files = sorted(model_path.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(
            f"No safetensors files in {vae_dir}")

    readers = []
    for f in st_files:
        readers.append(safe_open(str(f), framework="numpy"))

    def _load(name: str) -> np.ndarray:
        for r in readers:
            if name in r.keys():
                arr = r.get_tensor(name)
                if arr.dtype == np.float16:
                    arr = arr.astype(np.float32)
                elif hasattr(arr.dtype, 'name') and 'bfloat16' in arr.dtype.name:
                    arr = arr.astype(np.float32)
                else:
                    arr = arr.astype(np.float32)
                return arr
        raise KeyError(f"Weight '{name}' not found in safetensors")

    def _maybe(name: str) -> np.ndarray | None:
        for r in readers:
            if name in r.keys():
                return _load(name)
        return None

    # Read VAE config to get architecture params
    import json
    config_path = model_path / "config.json"
    if config_path.exists():
        vae_cfg = json.loads(config_path.read_text())
    else:
        vae_cfg = {}

    block_out_channels = vae_cfg.get("block_out_channels", [128, 256, 512, 512])
    layers_per_block = vae_cfg.get("layers_per_block", 2)
    norm_num_groups = vae_cfg.get("norm_num_groups", 32)

    use_post_quant_conv = vae_cfg.get("use_post_quant_conv", True)

    # Store config for use during build
    weights["_block_out_channels"] = np.array(block_out_channels, dtype=np.int32)
    weights["_layers_per_block"] = np.array([layers_per_block], dtype=np.int32)
    weights["_norm_num_groups"] = np.array([norm_num_groups], dtype=np.int32)
    weights["_use_post_quant_conv"] = np.array(
        [1 if use_post_quant_conv else 0], dtype=np.int32)

    # post_quant_conv (optional — FLUX VAE sets use_post_quant_conv=False)
    if use_post_quant_conv:
        weights["post_quant_conv.weight"] = _load("post_quant_conv.weight")
        weights["post_quant_conv.bias"] = _load("post_quant_conv.bias")

    # decoder.conv_in (maps latent_channels -> last block channels)
    weights["decoder.conv_in.weight"] = _load("decoder.conv_in.weight")
    weights["decoder.conv_in.bias"] = _load("decoder.conv_in.bias")

    # mid_block: 2 resnets + 1 attention
    for i in range(2):
        p = f"decoder.mid_block.resnets.{i}"
        weights[f"{p}.norm1.weight"] = _load(f"{p}.norm1.weight")
        weights[f"{p}.norm1.bias"] = _load(f"{p}.norm1.bias")
        weights[f"{p}.conv1.weight"] = _load(f"{p}.conv1.weight")
        weights[f"{p}.conv1.bias"] = _load(f"{p}.conv1.bias")
        weights[f"{p}.norm2.weight"] = _load(f"{p}.norm2.weight")
        weights[f"{p}.norm2.bias"] = _load(f"{p}.norm2.bias")
        weights[f"{p}.conv2.weight"] = _load(f"{p}.conv2.weight")
        weights[f"{p}.conv2.bias"] = _load(f"{p}.conv2.bias")

    # mid_block attention
    ap = "decoder.mid_block.attentions.0"
    weights[f"{ap}.group_norm.weight"] = _load(f"{ap}.group_norm.weight")
    weights[f"{ap}.group_norm.bias"] = _load(f"{ap}.group_norm.bias")
    for proj in ("to_q", "to_k", "to_v", "to_out.0"):
        weights[f"{ap}.{proj}.weight"] = _load(f"{ap}.{proj}.weight")
        weights[f"{ap}.{proj}.bias"] = _load(f"{ap}.{proj}.bias")

    # up_blocks (4 blocks in reversed channel order)
    num_blocks = len(block_out_channels)

    for block_idx in range(num_blocks):
        num_resnets = layers_per_block + 1

        for res_idx in range(num_resnets):
            p = f"decoder.up_blocks.{block_idx}.resnets.{res_idx}"
            weights[f"{p}.norm1.weight"] = _load(f"{p}.norm1.weight")
            weights[f"{p}.norm1.bias"] = _load(f"{p}.norm1.bias")
            weights[f"{p}.conv1.weight"] = _load(f"{p}.conv1.weight")
            weights[f"{p}.conv1.bias"] = _load(f"{p}.conv1.bias")
            weights[f"{p}.norm2.weight"] = _load(f"{p}.norm2.weight")
            weights[f"{p}.norm2.bias"] = _load(f"{p}.norm2.bias")
            weights[f"{p}.conv2.weight"] = _load(f"{p}.conv2.weight")
            weights[f"{p}.conv2.bias"] = _load(f"{p}.conv2.bias")

            # Conv shortcut if channels change
            sc = _maybe(f"{p}.conv_shortcut.weight")
            if sc is not None:
                weights[f"{p}.conv_shortcut.weight"] = sc
                weights[f"{p}.conv_shortcut.bias"] = _load(f"{p}.conv_shortcut.bias")

        # Upsamplers (all blocks except last have an upsampler)
        up_w = _maybe(f"decoder.up_blocks.{block_idx}.upsamplers.0.conv.weight")
        if up_w is not None:
            weights[f"decoder.up_blocks.{block_idx}.upsamplers.0.conv.weight"] = up_w
            weights[f"decoder.up_blocks.{block_idx}.upsamplers.0.conv.bias"] = _load(
                f"decoder.up_blocks.{block_idx}.upsamplers.0.conv.bias")

    # Final norm + conv
    weights["decoder.conv_norm_out.weight"] = _load("decoder.conv_norm_out.weight")
    weights["decoder.conv_norm_out.bias"] = _load("decoder.conv_norm_out.bias")
    weights["decoder.conv_out.weight"] = _load("decoder.conv_out.weight")
    weights["decoder.conv_out.bias"] = _load("decoder.conv_out.bias")

    return weights


# ---------------------------------------------------------------------------
# 4D GroupNorm (NCHW)
# ---------------------------------------------------------------------------

def _add_group_norm_4d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_channels: int,
    num_groups: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-6,
) -> trt.ITensor:
    """GroupNorm for 4D [B, C, H, W] tensors.

    Reshapes to [B, G, Gs, H, W], normalizes over (Gs, H, W), applies affine.
    """
    b, c, h, w = inp.shape
    group_size = num_channels // num_groups

    # [B, C, H, W] -> [B, G, Gs, H, W]
    reshape_in = network.add_shuffle(inp)
    reshape_in.reshape_dims = (b, num_groups, group_size, h, w)
    x = reshape_in.get_output(0)

    # Reduce over dims 2,3,4 (group_size, H, W)
    reduce_axes = (1 << 2) | (1 << 3) | (1 << 4)
    eps_t = network.add_constant(
        (1, 1, 1, 1, 1),
        trt.Weights(np.array([eps], dtype=np.float32)))

    sq = network.add_elementwise(x, x, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        x, trt.ReduceOperation.AVG, reduce_axes, keep_dims=True)
    mean_sq = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, reduce_axes, keep_dims=True)
    var = network.add_elementwise(
        mean_sq.get_output(0),
        network.add_elementwise(
            mean.get_output(0), mean.get_output(0),
            trt.ElementWiseOperation.PROD).get_output(0),
        trt.ElementWiseOperation.SUB)
    denom = network.add_unary(
        network.add_elementwise(
            var.get_output(0), eps_t.get_output(0),
            trt.ElementWiseOperation.SUM).get_output(0),
        trt.UnaryOperation.SQRT)
    recip = network.add_unary(denom.get_output(0), trt.UnaryOperation.RECIP)
    centered = network.add_elementwise(
        x, mean.get_output(0), trt.ElementWiseOperation.SUB)
    normalized = network.add_elementwise(
        centered.get_output(0), recip.get_output(0),
        trt.ElementWiseOperation.PROD)

    # Reshape back to [B, C, H, W]
    reshape_out = network.add_shuffle(normalized.get_output(0))
    reshape_out.reshape_dims = (b, c, h, w)
    result = reshape_out.get_output(0)

    # Affine: gamma * result + beta, broadcast over spatial dims
    gamma_t = network.add_constant(
        (1, num_channels, 1, 1),
        trt.Weights(np.ascontiguousarray(
            gamma.reshape(1, -1, 1, 1), dtype=np.float32)))
    beta_t = network.add_constant(
        (1, num_channels, 1, 1),
        trt.Weights(np.ascontiguousarray(
            beta.reshape(1, -1, 1, 1), dtype=np.float32)))
    scaled = network.add_elementwise(
        result, gamma_t.get_output(0), trt.ElementWiseOperation.PROD)
    return network.add_elementwise(
        scaled.get_output(0), beta_t.get_output(0),
        trt.ElementWiseOperation.SUM).get_output(0)


# ---------------------------------------------------------------------------
# Primitive ops
# ---------------------------------------------------------------------------

def _add_silu(network: trt.INetworkDefinition, inp: trt.ITensor) -> trt.ITensor:
    """SiLU (Swish): x * sigmoid(x)."""
    sigmoid = network.add_activation(inp, trt.ActivationType.SIGMOID)
    return network.add_elementwise(
        inp, sigmoid.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)


def _add_conv2d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> trt.ITensor:
    """Conv2D layer from weight arrays."""
    conv_w = trt.Weights(np.ascontiguousarray(weight, dtype=np.float32))
    if bias is not None:
        conv_b = trt.Weights(np.ascontiguousarray(bias, dtype=np.float32))
    else:
        conv_b = trt.Weights()

    conv = network.add_convolution_nd(
        inp,
        num_output_maps=out_channels,
        kernel_shape=(kernel_size, kernel_size),
        kernel=conv_w,
        bias=conv_b,
    )
    conv.stride_nd = (stride, stride)
    conv.padding_nd = (padding, padding)
    return conv.get_output(0)


# ---------------------------------------------------------------------------
# Architectural blocks
# ---------------------------------------------------------------------------

def _add_resnet_block_2d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: dict[str, np.ndarray],
    prefix: str,
    in_channels: int,
    out_channels: int,
    num_groups: int = 32,
    eps: float = 1e-6,
) -> trt.ITensor:
    """ResNetBlock2D: norm1->silu->conv1 -> norm2->silu->conv2 + shortcut.

    Weight keys:
        {prefix}.norm1.weight/bias  [in_ch]
        {prefix}.conv1.weight/bias  [out_ch, in_ch, 3, 3]
        {prefix}.norm2.weight/bias  [out_ch]
        {prefix}.conv2.weight/bias  [out_ch, out_ch, 3, 3]
        {prefix}.conv_shortcut.weight/bias  [out_ch, in_ch, 1, 1] (if in!=out)
    """
    # norm1 -> silu -> conv1
    x = _add_group_norm_4d(
        network, inp, in_channels, num_groups,
        weights[f"{prefix}.norm1.weight"],
        weights[f"{prefix}.norm1.bias"], eps)
    x = _add_silu(network, x)
    x = _add_conv2d(
        network, x,
        weights[f"{prefix}.conv1.weight"],
        weights[f"{prefix}.conv1.bias"],
        out_channels, kernel_size=3, padding=1)

    # norm2 -> silu -> conv2
    x = _add_group_norm_4d(
        network, x, out_channels, num_groups,
        weights[f"{prefix}.norm2.weight"],
        weights[f"{prefix}.norm2.bias"], eps)
    x = _add_silu(network, x)
    x = _add_conv2d(
        network, x,
        weights[f"{prefix}.conv2.weight"],
        weights[f"{prefix}.conv2.bias"],
        out_channels, kernel_size=3, padding=1)

    # Shortcut
    if in_channels != out_channels:
        shortcut = _add_conv2d(
            network, inp,
            weights[f"{prefix}.conv_shortcut.weight"],
            weights[f"{prefix}.conv_shortcut.bias"],
            out_channels, kernel_size=1)
    else:
        shortcut = inp

    # Residual
    out = network.add_elementwise(x, shortcut, trt.ElementWiseOperation.SUM)
    return out.get_output(0)


def _add_self_attention_2d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: dict[str, np.ndarray],
    prefix: str,
    channels: int,
    num_groups: int = 32,
    eps: float = 1e-6,
) -> trt.ITensor:
    """VAE mid-block 2D self-attention (single-head).

    Input: [B, C, H, W]
    Weight keys:
        {prefix}.group_norm.weight/bias  [C]
        {prefix}.to_q.weight/bias        [C, C] (Conv2d 1x1 in diffusers)
        {prefix}.to_k.weight/bias        [C, C]
        {prefix}.to_v.weight/bias        [C, C]
        {prefix}.to_out.0.weight/bias    [C, C]
    Output: [B, C, H, W] (with residual)
    """
    b, c, h, w = inp.shape
    hw = h * w
    attn_scale = 1.0 / np.sqrt(max(c, 1))
    identity = inp
    # When ``b`` is a runtime-dynamic ``-1`` we must let TRT infer the merged
    # batch*spatial leading dim via a single ``-1`` placeholder; ``b * hw``
    # would produce ``-hw``, which TRT refuses.
    dynamic_leading = b == -1
    flat_leading = -1 if dynamic_leading else b * hw

    # GroupNorm
    normed = _add_group_norm_4d(
        network, inp, channels, num_groups,
        weights[f"{prefix}.group_norm.weight"],
        weights[f"{prefix}.group_norm.bias"], eps)

    # Reshape [B, C, H, W] -> [B*H*W, C] (2D for matmul compatibility)
    flatten = network.add_shuffle(normed)
    flatten.first_transpose = trt.Permutation([0, 2, 3, 1])  # [B, H, W, C]
    flatten.reshape_dims = (flat_leading, c)

    flat = flatten.get_output(0)  # [B*HW, C]

    # Q, K, V projections: [B*HW, C] @ [C, C] -> [B*HW, C]
    from . import graph_ops

    def _proj(name: str) -> trt.ITensor:
        w = weights[f"{prefix}.{name}.weight"].reshape(c, c)
        bias = weights[f"{prefix}.{name}.bias"]
        # [C, C] in diffusers is [out, in], we need [in, out] for matmul
        out = graph_ops.add_matmul_rhs_constant(
            network, flat, c, c, w.T.copy())
        out = graph_ops.add_bias_sum(network, out, c, bias)
        return out

    q_flat = _proj("to_q")   # [B*HW, C]
    k_flat = _proj("to_k")
    v_flat = _proj("to_v")

    # Reshape to [B, 1, HW, C] for native single-head attention.
    q_r = network.add_shuffle(q_flat)
    q_r.reshape_dims = (b, 1, hw, c)
    k_r = network.add_shuffle(k_flat)
    k_r.reshape_dims = (b, 1, hw, c)
    v_r = network.add_shuffle(v_flat)
    v_r.reshape_dims = (b, 1, hw, c)

    context = graph_ops.add_attention_core(
        network,
        q_r.get_output(0),
        k_r.get_output(0),
        v_r.get_output(0),
        scale=attn_scale)

    # Flatten context to 2D for output projection: [B*HW, C]
    ctx_flat = network.add_shuffle(context)
    ctx_flat.reshape_dims = (flat_leading, c)

    # Output projection: [B*HW, C] @ [C, C] -> [B*HW, C]
    out_w = weights[f"{prefix}.to_out.0.weight"].reshape(c, c)
    out_bias = weights[f"{prefix}.to_out.0.bias"]
    proj_out = graph_ops.add_matmul_rhs_constant(
        network, ctx_flat.get_output(0), c, c, out_w.T.copy())
    proj_out = graph_ops.add_bias_sum(network, proj_out, c, out_bias)

    # Reshape back to [B, C, H, W]
    reshape_out = network.add_shuffle(proj_out)
    reshape_out.reshape_dims = (b, h, w, c)
    reshape_out.second_transpose = trt.Permutation([0, 3, 1, 2])  # [B, C, H, W]

    # Residual
    result = network.add_elementwise(
        reshape_out.get_output(0), identity, trt.ElementWiseOperation.SUM)
    return result.get_output(0)


def _add_upsample_2d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: dict[str, np.ndarray],
    prefix: str,
    out_channels: int,
) -> trt.ITensor:
    """Nearest-neighbor 2x upsample + Conv2d(3x3, pad=1).

    Weight keys:
        {prefix}.conv.weight  [out_ch, in_ch, 3, 3]
        {prefix}.conv.bias    [out_ch]
    """
    # 2x nearest-neighbor upsample
    resize = network.add_resize(inp)
    resize.resize_mode = trt.InterpolationMode.NEAREST
    resize.scales = [1.0, 1.0, 2.0, 2.0]

    # Conv2d 3x3
    return _add_conv2d(
        network, resize.get_output(0),
        weights[f"{prefix}.conv.weight"],
        weights[f"{prefix}.conv.bias"],
        out_channels, kernel_size=3, padding=1)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_flux_vae_decoder_engine(
    vae_dir: str,
    *,
    latent_channels: int = 16,
    h_lat: int = 128,
    w_lat: int = 128,
    scaling_factor: float = 0.3611,
    shift_factor: float = 0.1159,
    patch_size: tuple[int, int] = (1, 1),
    max_batch_size: int = 1,
    opt_batch_size: int | None = None,
    verbose: bool = False,
    build_timing: dict | None = None,
    timing_component: str = "vae_decoder",
) -> bytes:
    """Build FLUX AutoencoderKL decoder TRT engine using TRT Python API.

    Input:  latents [B, latent_channels, h_lat, w_lat] float32
    Output: image   [B, 3, h_lat*8*patch_h, w_lat*8*patch_w] float32

    For patch_size=(1,1) (FLUX.1): standard 8x spatial upsampling.
    For patch_size=(2,2) (FLUX.2): decoder conv_out produces C*ph*pw channels,
    then pixel-shuffle unpatchify yields 3-channel output at 16x resolution.

    The engine applies: x = latents / scaling_factor + shift_factor,
    then runs through the full decoder network.

    When ``max_batch_size > 1`` the leading batch dim of ``latents`` is
    dynamic (``-1``) so the runtime binding is uniform with the other
    diffusion components. Per design Decision E, the VAE *always* caps at
    ``max_batch = opt_batch = 1`` and the pipeline loops one latent at a time
    (peak-memory bound). ``max_batch_size == 1`` (the default) preserves the
    legacy static-shape engine byte-for-byte.
    """
    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be >= 1 (got {max_batch_size})")
    from ...build_timing import timed_weight_loading

    total_t0 = time.monotonic()
    weights_before = _timing_phase(build_timing, "weights_loading_s")
    patch_h, patch_w = patch_size
    print(f"[flux-vae] Loading VAE weights from {vae_dir} ...", file=sys.stderr)
    with timed_weight_loading(build_timing, timing_component):
        weights = _load_flux_vae_weights(vae_dir)

    # Read architecture params from loaded config
    block_out_channels = weights["_block_out_channels"].tolist()
    layers_per_block = int(weights["_layers_per_block"][0])
    num_groups = int(weights["_norm_num_groups"][0])
    use_post_quant_conv = bool(weights["_use_post_quant_conv"][0])
    num_blocks = len(block_out_channels)
    reversed_channels = list(reversed(block_out_channels))
    last_ch = block_out_channels[-1]
    eps = 1e-6

    h_out = h_lat * 8 * patch_h
    w_out = w_lat * 8 * patch_w

    print(f"[flux-vae] Architecture: block_out_channels={block_out_channels}, "
          f"layers_per_block={layers_per_block}, groups={num_groups}, "
          f"patch_size=({patch_h},{patch_w})",
          file=sys.stderr)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)

    # --- Input ---
    # When dynamic batching is enabled the leading dim is ``-1`` even though
    # the profile clamps both kMIN/kOPT/kMAX to 1 (Decision E). This keeps the
    # binding shape uniform across all diffusion components — the runtime
    # treats every dim 0 as dynamic and only the VAE attaches a (1, 1, 1)
    # profile.
    if max_batch_size > 1:
        from ...engine_builder import add_dynamic_batch_profile

        latents = network.add_input(
            "latents", trt.float32, (-1, latent_channels, h_lat, w_lat))
        add_dynamic_batch_profile(
            builder,
            config,
            network,
            input_names=["latents"],
            max_batch=1,  # VAE always caps at 1 — Decision E.
            opt_batch=1,
            static_shape={"latents": (latent_channels, h_lat, w_lat)},
        )
    else:
        latents = network.add_input(
            "latents", trt.float32, (1, latent_channels, h_lat, w_lat))

    # --- Scaling transform: x = latents / scaling_factor + shift_factor ---
    scale_t = network.add_constant(
        (1, 1, 1, 1),
        trt.Weights(np.array([1.0 / scaling_factor], dtype=np.float32)))
    shift_t = network.add_constant(
        (1, 1, 1, 1),
        trt.Weights(np.array([shift_factor], dtype=np.float32)))
    x = network.add_elementwise(
        latents, scale_t.get_output(0),
        trt.ElementWiseOperation.PROD).get_output(0)
    x = network.add_elementwise(
        x, shift_t.get_output(0),
        trt.ElementWiseOperation.SUM).get_output(0)

    # --- post_quant_conv (optional) + conv_in ---
    if use_post_quant_conv:
        pqc_out_ch = weights["post_quant_conv.weight"].shape[0]
        x = _add_conv2d(
            network, x,
            weights["post_quant_conv.weight"],
            weights["post_quant_conv.bias"],
            pqc_out_ch, kernel_size=1)
        print(f"[flux-vae] post_quant_conv: {latent_channels}->{pqc_out_ch}",
              file=sys.stderr)

    # conv_in: Conv2d(latent_channels or last_ch, last_ch, 3x3, pad=1)
    x = _add_conv2d(
        network, x,
        weights["decoder.conv_in.weight"],
        weights["decoder.conv_in.bias"],
        last_ch, kernel_size=3, padding=1)
    print(f"[flux-vae] conv_in -> {last_ch}", file=sys.stderr)

    cur_h, cur_w = h_lat, w_lat

    # --- Mid block: resnet.0 -> attention -> resnet.1 ---
    x = _add_resnet_block_2d(
        network, x, weights, "decoder.mid_block.resnets.0",
        last_ch, last_ch, num_groups, eps)
    x = _add_self_attention_2d(
        network, x, weights, "decoder.mid_block.attentions.0",
        last_ch, num_groups, eps)
    x = _add_resnet_block_2d(
        network, x, weights, "decoder.mid_block.resnets.1",
        last_ch, last_ch, num_groups, eps)
    print(f"[flux-vae] mid_block done: ch={last_ch}, {cur_h}x{cur_w}",
          file=sys.stderr)

    # --- Up blocks ---
    # Standard diffusers decoder ordering:
    # up_blocks[0]: resnets with reversed_channels[0] (512), has upsampler
    # up_blocks[1]: resnets with reversed_channels[1] (512), has upsampler
    # up_blocks[2]: resnets with reversed_channels[2] (256), has upsampler
    # up_blocks[3]: resnets with reversed_channels[3] (128), no upsampler
    prev_ch = last_ch
    for block_idx in range(num_blocks):
        out_ch = reversed_channels[block_idx]
        has_upsample = block_idx < num_blocks - 1
        num_resnets = layers_per_block + 1

        for res_idx in range(num_resnets):
            prefix = f"decoder.up_blocks.{block_idx}.resnets.{res_idx}"
            in_ch = prev_ch if res_idx == 0 else out_ch
            x = _add_resnet_block_2d(
                network, x, weights, prefix,
                in_ch, out_ch, num_groups, eps)
            prev_ch = out_ch

        if has_upsample:
            x = _add_upsample_2d(
                network, x, weights,
                f"decoder.up_blocks.{block_idx}.upsamplers.0",
                out_ch)
            cur_h *= 2
            cur_w *= 2

        print(f"[flux-vae] up_block {block_idx}: ch={out_ch}, "
              f"{cur_h}x{cur_w}, upsample={has_upsample}", file=sys.stderr)

    # --- conv_norm_out -> SiLU -> conv_out ---
    x = _add_group_norm_4d(
        network, x, prev_ch, num_groups,
        weights["decoder.conv_norm_out.weight"],
        weights["decoder.conv_norm_out.bias"], eps)
    x = _add_silu(network, x)

    # conv_out output channels: 3 for patch_size=(1,1), 3*ph*pw for patched VAEs
    conv_out_weight = weights["decoder.conv_out.weight"]
    conv_out_channels = conv_out_weight.shape[0]
    x = _add_conv2d(
        network, x,
        conv_out_weight,
        weights["decoder.conv_out.bias"],
        conv_out_channels, kernel_size=3, padding=1)

    print(f"[flux-vae] conv_out: {prev_ch}->{conv_out_channels}, "
          f"spatial {cur_h}x{cur_w}", file=sys.stderr)

    # --- Unpatchify (pixel shuffle) for patched VAEs ---
    if patch_h > 1 or patch_w > 1:
        # conv_out produces [B, out_ch * ph * pw, H, W]
        # Unpatchify to [B, out_ch, H * ph, W * pw]
        out_ch = conv_out_channels // (patch_h * patch_w)
        # ``-1`` propagates the (possibly dynamic) batch dim through both
        # shuffles — see the comment on ``flat_leading`` in
        # ``_add_self_attention_2d`` for why we cannot substitute the static
        # value ``1`` here.
        leading = -1 if x.shape[0] == -1 else x.shape[0]

        # Reshape: [B, out_ch, ph, pw, H, W]
        reshape1 = network.add_shuffle(x)
        reshape1.reshape_dims = (leading, out_ch, patch_h, patch_w, cur_h, cur_w)

        # Transpose: [B, out_ch, H, ph, W, pw]
        reshape1.second_transpose = trt.Permutation([0, 1, 4, 2, 5, 3])

        # Reshape: [B, out_ch, H * ph, W * pw]
        reshape2 = network.add_shuffle(reshape1.get_output(0))
        reshape2.reshape_dims = (
            leading, out_ch, cur_h * patch_h, cur_w * patch_w)
        x = reshape2.get_output(0)

        print(f"[flux-vae] unpatchify: [{conv_out_channels},{cur_h},{cur_w}] -> "
              f"[{out_ch},{cur_h * patch_h},{cur_w * patch_w}]", file=sys.stderr)

    # --- Mark output ---
    cast_x = network.add_cast(x, trt.float32)
    x_out = cast_x.get_output(0)
    x_out.name = "image"
    network.mark_output(x_out)

    print(f"[flux-vae] Building TRT engine: output [1, 3, {h_out}, {w_out}] ...",
          file=sys.stderr)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for VAE decoder")

    plan_bytes = bytes(plan)
    weights_after = _timing_phase(build_timing, "weights_loading_s")
    compile_elapsed = max(
        0.0, time.monotonic() - total_t0 - max(0.0, weights_after - weights_before))
    add_trt_compile_timing(build_timing, timing_component, compile_elapsed)
    print(f"[flux-vae] Engine built: {len(plan_bytes) / 1e6:.1f} MB", file=sys.stderr)
    return plan_bytes


def _timing_phase(timing: dict | None, key: str) -> float:
    if timing is None:
        return 0.0
    phases = timing.setdefault("phases", {})
    try:
        return float(phases.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
