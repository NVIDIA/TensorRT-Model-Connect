"""Wan 2.2 Causal 3D VAE decoder engine builder.

Sibling builder to ``causal_vae_3d_builder`` targeting Wan 2.2 TI2V-5B's
VAE (``AutoencoderKLWan`` v2 with the Wan 2.2 config). The decoder
structure is the same recipe as Wan 2.1 (CausalConv3D + WanRMS_norm
``.gamma`` 3D-channel-norm + ResBlocks + spatial 2x + temporal 2x via
pixel-shuffle) but with four concrete deltas pulled directly from the
shipped weight names and shapes:

  1. **Upsampler weight prefix is ``.upsampler.`` (singular, no index)**.
     Wan 2.1 uses ``.upsamplers.0.``.
  2. **Channels are driven by ``decoder_base_dim`` (256 for TI2V-5B), not
     by the encoder ``base_dim`` (which is 160)**. Resulting decoder
     channels-list (reversed) is [1024, 1024, 512, 256] for the 4 levels.
  3. **conv_out outputs 12 channels**, not 3.
  4. **Final pixel-shuffle (factor 2 spatial)** un-patchifies 12 channels
     into 3 channels at 2x spatial resolution. Combined with the three
     internal spatial 2x upsamples this yields the Wan 2.2 16x total
     spatial scale (vs 8x in Wan 2.1).

Temporal upsampling pattern is identical to Wan 2.1: levels 0 and 1 in
decoder order have ``time_conv`` (i.e. encoder ``temporal_downsample =
(False, False, True, True)``), giving scale_factor_temporal=4.

is_residual=True in the VAE config refers to the up_block-level skip
contract; the per-resnet structure on disk is the same as Wan 2.1's
(norm1+conv1+norm2+conv2+optional conv_shortcut), so the existing
``add_vae_resblock_3d`` is reused.

Tensor contract (matches the C++ runtime VAE):
  Inputs:
    latent_frame    fp32 (1, z_dim=48, 1, h_lat, w_lat)
    cache_i         fp32 (1, channels_i, t_cache=2, h_i, w_i)
  Outputs:
    video_frame     fp32 (1, 3, T_out, H_out, W_out)
    cache_out_i     fp32 (matching cache_i)
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict


# ---------------------------------------------------------------------------
# Weight loader
# ---------------------------------------------------------------------------


def load_vae_weights_wan22(
    model_dir: str,
    *,
    z_dim: int = 48,
    decoder_base_dim: int = 256,
    dim_mult: tuple[int, ...] = (1, 2, 4, 4),
    num_res_blocks: int = 2,
) -> "WeightDict":
    """Load Wan 2.2 VAE decoder weights from a diffusers vae directory.

    Same shape as ``load_vae_weights`` (Wan 2.1) but uses
    ``.upsampler.`` instead of ``.upsamplers.0.`` and computes channels
    from ``decoder_base_dim``.
    """
    from pathlib import Path
    from ...checkpoint_mapper import (
        WeightDict, _open_safetensors, _load_tensor, _has_tensor,
    )

    model_path = Path(model_dir)
    readers = _open_safetensors(model_path)
    weights = WeightDict()

    def _w(name: str) -> np.ndarray:
        return _load_tensor(readers, name).astype(np.float32)

    def _maybe(name: str) -> np.ndarray | None:
        if _has_tensor(readers, name):
            return _w(name)
        return None

    # post_quant_conv: [z_dim, z_dim, 1, 1, 1]
    weights["post_quant_conv.weight"] = _w("post_quant_conv.weight")
    weights["post_quant_conv.bias"] = _w("post_quant_conv.bias")

    # conv_in: [mid_ch, z_dim, 3, 3, 3]
    weights["decoder.conv_in.weight"] = _w("decoder.conv_in.weight")
    weights["decoder.conv_in.bias"] = _w("decoder.conv_in.bias")

    channels_list = [decoder_base_dim * m for m in dim_mult]
    mid_ch = channels_list[-1]

    # mid_block: 2 resnets + 1 attention (1024-channel for 5B)
    for i in range(2):
        p = f"decoder.mid_block.resnets.{i}"
        weights[f"{p}.norm1.gamma"] = _w(f"{p}.norm1.gamma")
        weights[f"{p}.norm2.gamma"] = _w(f"{p}.norm2.gamma")
        weights[f"{p}.conv1.weight"] = _w(f"{p}.conv1.weight")
        weights[f"{p}.conv1.bias"] = _w(f"{p}.conv1.bias")
        weights[f"{p}.conv2.weight"] = _w(f"{p}.conv2.weight")
        weights[f"{p}.conv2.bias"] = _w(f"{p}.conv2.bias")

    attn_prefix = "decoder.mid_block.attentions.0"
    weights[f"{attn_prefix}.norm.gamma"] = _w(f"{attn_prefix}.norm.gamma")
    weights[f"{attn_prefix}.to_qkv.weight"] = _w(f"{attn_prefix}.to_qkv.weight")
    weights[f"{attn_prefix}.to_qkv.bias"] = _w(f"{attn_prefix}.to_qkv.bias")
    weights[f"{attn_prefix}.proj.weight"] = _w(f"{attn_prefix}.proj.weight")
    weights[f"{attn_prefix}.proj.bias"] = _w(f"{attn_prefix}.proj.bias")

    num_levels = len(dim_mult)
    for level in range(num_levels):
        # num_res_blocks + 1 resnets per up_block
        for blk in range(num_res_blocks + 1):
            p = f"decoder.up_blocks.{level}.resnets.{blk}"
            weights[f"{p}.norm1.gamma"] = _w(f"{p}.norm1.gamma")
            weights[f"{p}.norm2.gamma"] = _w(f"{p}.norm2.gamma")
            weights[f"{p}.conv1.weight"] = _w(f"{p}.conv1.weight")
            weights[f"{p}.conv1.bias"] = _w(f"{p}.conv1.bias")
            weights[f"{p}.conv2.weight"] = _w(f"{p}.conv2.weight")
            weights[f"{p}.conv2.bias"] = _w(f"{p}.conv2.bias")
            sc_w = _maybe(f"{p}.conv_shortcut.weight")
            if sc_w is not None:
                weights[f"{p}.conv_shortcut.weight"] = sc_w
                weights[f"{p}.conv_shortcut.bias"] = _w(f"{p}.conv_shortcut.bias")

        # NOTE: Wan 2.2 uses ".upsampler." (singular), not ".upsamplers.0."
        sp_w = _maybe(f"decoder.up_blocks.{level}.upsampler.resample.1.weight")
        if sp_w is not None:
            weights[f"decoder.up_blocks.{level}.upsampler.resample.1.weight"] = sp_w
            weights[f"decoder.up_blocks.{level}.upsampler.resample.1.bias"] = _w(
                f"decoder.up_blocks.{level}.upsampler.resample.1.bias")
        tc_w = _maybe(f"decoder.up_blocks.{level}.upsampler.time_conv.weight")
        if tc_w is not None:
            weights[f"decoder.up_blocks.{level}.upsampler.time_conv.weight"] = tc_w
            weights[f"decoder.up_blocks.{level}.upsampler.time_conv.bias"] = _w(
                f"decoder.up_blocks.{level}.upsampler.time_conv.bias")

    # Output norm + conv (norm on smallest decoder channel = decoder_base_dim)
    weights["decoder.norm_out.gamma"] = _w("decoder.norm_out.gamma")
    weights["decoder.conv_out.weight"] = _w("decoder.conv_out.weight")
    weights["decoder.conv_out.bias"] = _w("decoder.conv_out.bias")

    return weights


# ---------------------------------------------------------------------------
# Cache count
# ---------------------------------------------------------------------------


def count_vae_caches_wan22(
    dim_mult: tuple[int, ...] = (1, 2, 4, 4),
    num_res_blocks: int = 2,
    temporal_upsample: tuple[bool, ...] = (False, True, True),
) -> int:
    """Same cache-count formula as Wan 2.1.

    Wan 2.2's decoder has the same number of CausalConv3D blocks as 2.1's
    (conv_in + 2 mid resnets + (num_res_blocks+1) resnets per up_block +
    optional time_conv per upsample + conv_out). The encoder/decoder
    asymmetric channel dim doesn't add or remove caches.
    """
    count = 0
    num_levels = len(dim_mult)
    temp_up = list(reversed(temporal_upsample))

    # conv_in: CausalConv3d(kt=3)
    count += 1
    # mid_block: 2 resnets x 2 caches
    count += 4
    for level in range(num_levels):
        count += (num_res_blocks + 1) * 2
        if level < num_levels - 1:
            if level < len(temp_up) and temp_up[level]:
                count += 1
    # conv_out
    count += 1
    return count


# ---------------------------------------------------------------------------
# Pixel un-shuffle (depth-to-space) along spatial dims of a 5D tensor.
# ---------------------------------------------------------------------------


def _spatial_pixel_shuffle_5d(network, x, factor: int, *, c_in: int):
    """Reshape (B, C, T, H, W) -> (B, C // factor^2, T, H*factor, W*factor).

    Implements the depth-to-space operation along the spatial dims only,
    preserving the temporal dim. The output channel count must be
    ``c_in // (factor * factor)``.
    """
    from tensorrt_model_connect import trt_compat
    trt = trt_compat.get_trt()

    if c_in % (factor * factor) != 0:
        raise ValueError(
            f"pixel-shuffle factor={factor} requires C divisible by "
            f"factor^2={factor*factor}; got C={c_in}"
        )
    c_out = c_in // (factor * factor)

    # Step 1: reshape (B, C, T, H, W) -> (B, c_out, factor, factor, T, H, W)
    # Use dynamic-shape API (set_input) so T, H, W flow through symbolically.
    from ... import graph_ops
    shape_t = network.add_shape(x).get_output(0)  # [5] int64: [B, C, T, H, W]
    one_const = graph_ops.add_constant(
        network, (1,), np.array([1], dtype=np.int64), dtype=np.int64)
    co_const = graph_ops.add_constant(
        network, (1,), np.array([c_out], dtype=np.int64), dtype=np.int64)
    f_const = graph_ops.add_constant(
        network, (1,), np.array([factor], dtype=np.int64), dtype=np.int64)
    t_slice = network.add_slice(shape_t, start=(2,), shape=(1,), stride=(1,))
    h_slice = network.add_slice(shape_t, start=(3,), shape=(1,), stride=(1,))
    w_slice = network.add_slice(shape_t, start=(4,), shape=(1,), stride=(1,))
    new_shape_pre = network.add_concatenation([
        one_const, co_const, f_const, f_const,
        t_slice.get_output(0), h_slice.get_output(0), w_slice.get_output(0),
    ])
    new_shape_pre.axis = 0
    shuf1 = network.add_shuffle(x)
    shuf1.set_input(1, new_shape_pre.get_output(0))
    x1 = shuf1.get_output(0)  # (1, c_out, f, f, T, H, W)

    # Step 2: permute to (B, c_out, T, H, f, W, f)
    # Original layout: [B=0, c_out=1, f_h=2, f_w=3, T=4, H=5, W=6]
    # We want:        [B=0, c_out=1, T=4,  H=5,  f_h=2, W=6, f_w=3]
    shuf2 = network.add_shuffle(x1)
    shuf2.first_transpose = (0, 1, 4, 5, 2, 6, 3)
    x2 = shuf2.get_output(0)  # (1, c_out, T, H, f, W, f)

    # Step 3: reshape to (B, c_out, T, H*f, W*f)
    shape2 = network.add_shape(x2).get_output(0)
    # We need [1, c_out, T, H*f, W*f]. T at index 2, H at 3, f at 4, W at 5, f at 6.
    t_s2 = network.add_slice(shape2, start=(2,), shape=(1,), stride=(1,))
    h_s2 = network.add_slice(shape2, start=(3,), shape=(1,), stride=(1,))
    w_s2 = network.add_slice(shape2, start=(5,), shape=(1,), stride=(1,))
    hf = network.add_elementwise(
        h_s2.get_output(0), f_const, trt.ElementWiseOperation.PROD)
    wf = network.add_elementwise(
        w_s2.get_output(0), f_const, trt.ElementWiseOperation.PROD)
    final_shape = network.add_concatenation([
        one_const, co_const,
        t_s2.get_output(0), hf.get_output(0), wf.get_output(0),
    ])
    final_shape.axis = 0
    shuf3 = network.add_shuffle(x2)
    shuf3.set_input(1, final_shape.get_output(0))
    return shuf3.get_output(0)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_wan22_causal_vae_3d_decoder_engine(
    weights: "WeightDict",
    *,
    z_dim: int = 48,
    decoder_base_dim: int = 256,
    dim_mult: tuple[int, ...] = (1, 2, 4, 4),
    num_res_blocks: int = 2,
    temporal_upsample: tuple[bool, ...] = (False, True, True),
    h_lat: int = 24,           # for 384/16 spatial scale
    w_lat: int = 42,           # for 672/16 spatial scale
    out_channels_conv: int = 12,
    final_out_channels: int = 3,
    patch_size: int = 2,
    num_groups: int = 32,
    eps: float = 1e-6,
    verbose: bool = False,
) -> bytes:
    """Build Wan 2.2 causal 3D VAE decoder TRT engine plan.

    Mirrors ``build_causal_vae_3d_engine`` for Wan 2.1, with four deltas:
      - upsampler weight prefix is ``.upsampler.`` (singular)
      - channels driven by ``decoder_base_dim``
      - ``out_channels_conv`` is 12 (not 3); pixel-shuffle factor 2 yields
        ``final_out_channels=3`` at 2x spatial.
      - ``patch_size=2`` configures the final spatial pixel-shuffle.
    """
    from tensorrt_model_connect import trt_compat
    trt = trt_compat.get_trt()
    from ... import graph_ops, graph_blocks

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    num_levels = len(dim_mult)
    channels_list = [decoder_base_dim * m for m in dim_mult]
    dec_channels = list(reversed(channels_list))
    mid_ch = dec_channels[0]
    temp_up = list(reversed(temporal_upsample))

    cur_h, cur_w = h_lat, w_lat
    cur_t = 1

    latent = network.add_input(
        "latent_frame", trt.float32, (1, z_dim, 1, h_lat, w_lat))

    cache_idx = 0
    cache_inputs: dict = {}
    cache_outputs: dict = {}

    def _add_cache_input(channels: int, t_cache: int, h_c: int, w_c: int):
        nonlocal cache_idx
        name = f"cache_{cache_idx}"
        shape = (1, channels, t_cache, h_c, w_c)
        t = network.add_input(name, trt.float32, shape)
        cache_inputs[cache_idx] = t
        cache_idx += 1
        return t

    def _set_cache_output(idx: int, tensor) -> None:
        cache_outputs[idx] = tensor

    # post_quant_conv (1x1x1)
    x = graph_ops.add_conv3d_as_conv2d(
        network, latent,
        weight=weights["post_quant_conv.weight"],
        bias=weights["post_quant_conv.bias"],
        out_channels=z_dim, kernel_size=(1, 1, 1),
    )

    # conv_in (3,3,3) CausalConv3D
    ci_cache = _add_cache_input(z_dim, 2, cur_h, cur_w)
    x, ci_cache_out = graph_ops.add_causal_conv3d(
        network, x, ci_cache,
        weight=weights["decoder.conv_in.weight"],
        bias=weights["decoder.conv_in.bias"],
        out_channels=mid_ch, kernel_size=(3, 3, 3), padding_hw=(1, 1),
    )
    _set_cache_output(cache_idx - 1, ci_cache_out)
    print(f"[wan22-vae-3d] conv_in: [{z_dim}]->[{mid_ch}], "
          f"T={cur_t}, {cur_h}x{cur_w}", file=sys.stderr)

    # mid_block: resnet.0 -> attention -> resnet.1
    for mi in range(2):
        prefix = f"decoder.mid_block.resnets.{mi}"
        c1 = _add_cache_input(mid_ch, 2, cur_h, cur_w)
        c2 = _add_cache_input(mid_ch, 2, cur_h, cur_w)
        x, co1, co2 = graph_blocks.add_vae_resblock_3d(
            network, x, c1, c2,
            weights=weights, prefix=prefix,
            in_channels=mid_ch, out_channels=mid_ch,
            norm_type="l2_channel_norm", num_groups=num_groups, eps=eps)
        _set_cache_output(cache_idx - 2, co1)
        _set_cache_output(cache_idx - 1, co2)

        if mi == 0:
            x = graph_blocks.add_vae_spatial_attention(
                network, x,
                weights=weights,
                prefix="decoder.mid_block.attentions.0",
                channels=mid_ch,
                norm_type="l2_channel_norm", num_groups=num_groups, eps=eps)
    print(f"[wan22-vae-3d] mid_block done, T={cur_t}, {cur_h}x{cur_w}",
          file=sys.stderr)

    # up_blocks
    prev_ch = mid_ch
    for level in range(num_levels):
        out_ch = dec_channels[level]
        has_spatial = level < num_levels - 1
        has_temporal = (level < len(temp_up) and temp_up[level]
                        and level < num_levels - 1)

        for blk in range(num_res_blocks + 1):
            prefix = f"decoder.up_blocks.{level}.resnets.{blk}"
            in_ch = prev_ch if blk == 0 else out_ch
            c1 = _add_cache_input(in_ch, 2, cur_h, cur_w)
            c2 = _add_cache_input(out_ch, 2, cur_h, cur_w)
            x, co1, co2 = graph_blocks.add_vae_resblock_3d(
                network, x, c1, c2,
                weights=weights, prefix=prefix,
                in_channels=in_ch, out_channels=out_ch,
                norm_type="l2_channel_norm", num_groups=num_groups, eps=eps)
            _set_cache_output(cache_idx - 2, co1)
            _set_cache_output(cache_idx - 1, co2)

        prev_ch = out_ch
        print(f"[wan22-vae-3d] up_block {level}: ch={out_ch}, T={cur_t}, "
              f"{cur_h}x{cur_w}", file=sys.stderr)

        # NOTE: Wan 2.2 uses ".upsampler." (singular) — different from Wan 2.1.
        if has_temporal:
            tc_prefix = f"decoder.up_blocks.{level}.upsampler.time_conv"
            tc_w = weights[f"{tc_prefix}.weight"]
            tc_in_ch = tc_w.shape[1]
            tc_out_ch = tc_w.shape[0]
            tc_cache = _add_cache_input(tc_in_ch, 2, cur_h, cur_w)
            x, tc_cache_out = graph_ops.add_causal_conv3d(
                network, x, tc_cache,
                weight=tc_w, bias=weights[f"{tc_prefix}.bias"],
                out_channels=tc_out_ch, kernel_size=(3, 1, 1),
                padding_hw=(0, 0),
            )
            _set_cache_output(cache_idx - 1, tc_cache_out)
            x = graph_ops.add_temporal_pixel_shuffle(network, x, factor=2)
            prev_ch = tc_in_ch
            cur_t *= 2
            print(f"[wan22-vae-3d]   temporal 2x -> {tc_in_ch}ch, T={cur_t}",
                  file=sys.stderr)

        if has_spatial:
            sp_prefix = f"decoder.up_blocks.{level}.upsampler.resample.1"
            sp_w = weights[f"{sp_prefix}.weight"]
            sp_out_ch = sp_w.shape[0]
            x = graph_ops.add_spatial_upsample_with_conv(
                network, x,
                weight=sp_w, bias=weights[f"{sp_prefix}.bias"], scale=2)
            cur_h *= 2
            cur_w *= 2
            prev_ch = sp_out_ch
            print(f"[wan22-vae-3d]   spatial 2x -> {sp_out_ch}ch, "
                  f"{cur_h}x{cur_w}", file=sys.stderr)

    # norm_out + SiLU
    x = graph_ops.add_l2_channel_norm(
        network, x, prev_ch, weights["decoder.norm_out.gamma"], eps)
    x = graph_ops.add_silu(network, x)

    # conv_out: (3,3,3) CausalConv3D, prev_ch -> out_channels_conv (= 12)
    co_cache = _add_cache_input(prev_ch, 2, cur_h, cur_w)
    x, co_cache_out = graph_ops.add_causal_conv3d(
        network, x, co_cache,
        weight=weights["decoder.conv_out.weight"],
        bias=weights["decoder.conv_out.bias"],
        out_channels=out_channels_conv, kernel_size=(3, 3, 3),
        padding_hw=(1, 1),
    )
    _set_cache_output(cache_idx - 1, co_cache_out)

    # Final pixel-shuffle: (1, 12, T, H, W) -> (1, 3, T, H*2, W*2).
    if patch_size > 1:
        if out_channels_conv != final_out_channels * patch_size * patch_size:
            raise ValueError(
                f"out_channels_conv={out_channels_conv} must equal "
                f"final_out_channels={final_out_channels} * patch_size^2="
                f"{patch_size * patch_size}"
            )
        x = _spatial_pixel_shuffle_5d(
            network, x, factor=patch_size, c_in=out_channels_conv)
        cur_h *= patch_size
        cur_w *= patch_size
        print(f"[wan22-vae-3d]   pixel-shuffle x{patch_size} -> "
              f"{final_out_channels}ch, {cur_h}x{cur_w}", file=sys.stderr)

    # Mark output
    cast_x = network.add_cast(x, trt.float32)
    x_out = cast_x.get_output(0)
    x_out.name = "video_frame"
    network.mark_output(x_out)

    for idx in sorted(cache_outputs.keys()):
        t = cache_outputs[idx]
        cast_t = network.add_cast(t, trt.float32)
        t_out = cast_t.get_output(0)
        t_out.name = f"cache_out_{idx}"
        network.mark_output(t_out)

    total_caches = cache_idx
    print(f"[wan22-vae-3d] Building TRT engine: {total_caches} caches, "
          f"output [1, {final_out_channels}, {cur_t}, {cur_h}, {cur_w}]",
          file=sys.stderr)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for Wan 2.2 VAE")
    return bytes(plan)
