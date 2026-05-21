"""AutoencoderKL 2D VAE decoder builder.

Builds a TRT engine for a standard AutoencoderKL VAE decoder (FLUX/Z-Image style)
using the TensorRT Python API directly. No ONNX export.

Engine I/O:
    Input:  latent_input    [1, latent_channels, h_lat, w_lat] float32
    Output: decoder_output  [1, 3, h_out, w_out]               float32

Architecture:
    conv_in(3x3) -> mid_block -> 4 up_blocks -> norm -> SiLU -> conv_out
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from ...build_timing import add_trt_compile_timing


class _TimedWeightReaders(list):
    def __init__(self, readers: list, build_timing: dict | None, component: str):
        super().__init__(readers)
        self.build_timing = build_timing
        self.component = component


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _open_vae_safetensors(model_dir: str) -> list:
    """Open safetensors files from a VAE model directory."""
    from safetensors import safe_open

    try:
        import torch  # noqa: F401
        fw = "torch"
    except ImportError:
        fw = "numpy"

    d = Path(model_dir)

    # diffusers format
    single = d / "diffusion_pytorch_model.safetensors"
    if single.exists():
        return [safe_open(str(single), framework=fw)]

    index = d / "diffusion_pytorch_model.safetensors.index.json"
    if index.exists():
        idx = json.loads(index.read_text())
        shards = sorted(set(idx.get("weight_map", {}).values()))
        return [safe_open(str(d / f), framework=fw) for f in shards]

    # standard HF format
    single2 = d / "model.safetensors"
    if single2.exists():
        return [safe_open(str(single2), framework=fw)]

    raise FileNotFoundError(
        f"No safetensors found in {model_dir}")


def _get_weight(readers: list, name: str) -> np.ndarray:
    """Get a tensor from safetensors readers as float32 numpy."""
    from ...build_timing import timed_weight_loading

    timing = getattr(readers, "build_timing", None)
    component = getattr(readers, "component", "vae_decoder")
    with timed_weight_loading(timing, component):
        for r in readers:
            if name in r.keys():
                t = r.get_tensor(name)
                if hasattr(t, "numpy"):
                    return t.float().numpy()
                return np.array(t, dtype=np.float32)
    raise KeyError(f"Weight not found: {name}")


# ---------------------------------------------------------------------------
# 4D GroupNorm (for [N, C, H, W] tensors)
# ---------------------------------------------------------------------------

def _add_group_norm_4d(
    network,
    inp,
    num_channels: int,
    num_groups: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-6,
):
    """GroupNorm for 4D tensor [N, C, H, W].

    Reshapes to [N, G, Gs, H, W], normalizes over (Gs, H, W), reshapes back,
    then applies affine transform.
    """
    from tensorrt_model_connect import trt_compat
    trt = trt_compat.get_trt()
    from ...graph_ops import add_constant

    n, c, h, w = inp.shape
    group_size = num_channels // num_groups

    # Reshape [N, C, H, W] -> [N, G, Gs, H, W]
    reshape_in = network.add_shuffle(inp)
    reshape_in.reshape_dims = (n, num_groups, group_size, h, w)
    x = reshape_in.get_output(0)

    # Reduce over dims 2,3,4 (group_size, H, W)
    reduce_axes = (1 << 2) | (1 << 3) | (1 << 4)
    eps_t = add_constant(network, (1, 1, 1, 1, 1),
                         np.array([eps], dtype=np.float32))

    sq = network.add_elementwise(x, x, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        x, trt.ReduceOperation.AVG, reduce_axes, keep_dims=True)
    mean_sq = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG,
        reduce_axes, keep_dims=True)
    var = network.add_elementwise(
        mean_sq.get_output(0),
        network.add_elementwise(
            mean.get_output(0), mean.get_output(0),
            trt.ElementWiseOperation.PROD).get_output(0),
        trt.ElementWiseOperation.SUB)
    denom = network.add_unary(
        network.add_elementwise(
            var.get_output(0), eps_t,
            trt.ElementWiseOperation.SUM).get_output(0),
        trt.UnaryOperation.SQRT)
    recip = network.add_unary(
        denom.get_output(0), trt.UnaryOperation.RECIP)
    centered = network.add_elementwise(
        x, mean.get_output(0), trt.ElementWiseOperation.SUB)
    normalized = network.add_elementwise(
        centered.get_output(0), recip.get_output(0),
        trt.ElementWiseOperation.PROD)

    # Reshape back to [N, C, H, W]
    reshape_out = network.add_shuffle(normalized.get_output(0))
    reshape_out.reshape_dims = (n, c, h, w)
    result = reshape_out.get_output(0)

    # Affine: gamma * result + beta
    gamma_t = add_constant(
        network, (1, num_channels, 1, 1),
        gamma.reshape(1, -1, 1, 1).astype(np.float32))
    beta_t = add_constant(
        network, (1, num_channels, 1, 1),
        beta.reshape(1, -1, 1, 1).astype(np.float32))
    scaled = network.add_elementwise(
        result, gamma_t, trt.ElementWiseOperation.PROD)
    return network.add_elementwise(
        scaled.get_output(0), beta_t,
        trt.ElementWiseOperation.SUM).get_output(0)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _add_resnet_block_2d(network, inp, weights, prefix: str,
                         in_ch: int, out_ch: int, h: int, w: int):
    """ResNetBlock2D: norm1 -> SiLU -> conv1 -> norm2 -> SiLU -> conv2 + shortcut."""
    from tensorrt_model_connect import trt_compat
    trt = trt_compat.get_trt()
    from ...graph_ops import add_conv2d, add_silu

    # norm1 -> SiLU -> conv1
    x = _add_group_norm_4d(
        network, inp, in_ch, 32,
        _get_weight(weights, f"{prefix}.norm1.weight"),
        _get_weight(weights, f"{prefix}.norm1.bias"))
    x = add_silu(network, x)
    x = add_conv2d(
        network, x,
        weight=_get_weight(weights, f"{prefix}.conv1.weight"),
        bias=_get_weight(weights, f"{prefix}.conv1.bias"),
        out_channels=out_ch,
        kernel_size=(3, 3),
        padding=(1, 1))

    # norm2 -> SiLU -> conv2
    x = _add_group_norm_4d(
        network, x, out_ch, 32,
        _get_weight(weights, f"{prefix}.norm2.weight"),
        _get_weight(weights, f"{prefix}.norm2.bias"))
    x = add_silu(network, x)
    x = add_conv2d(
        network, x,
        weight=_get_weight(weights, f"{prefix}.conv2.weight"),
        bias=_get_weight(weights, f"{prefix}.conv2.bias"),
        out_channels=out_ch,
        kernel_size=(3, 3),
        padding=(1, 1))

    # Shortcut
    if in_ch != out_ch:
        shortcut = add_conv2d(
            network, inp,
            weight=_get_weight(weights, f"{prefix}.conv_shortcut.weight"),
            bias=_get_weight(weights, f"{prefix}.conv_shortcut.bias"),
            out_channels=out_ch,
            kernel_size=(1, 1))
    else:
        shortcut = inp

    # Residual sum
    return network.add_elementwise(
        shortcut, x, trt.ElementWiseOperation.SUM).get_output(0)


def _linear_weight_to_conv2d(w: np.ndarray) -> np.ndarray:
    """Reshape [out, in] Linear weight to [out, in, 1, 1] Conv2d weight."""
    if w.ndim == 2:
        return w.reshape(w.shape[0], w.shape[1], 1, 1)
    return w  # already 4D


def _add_self_attention_2d(network, inp, weights, prefix: str,
                           ch: int, h: int, w: int):
    """SelfAttention2D (mid block): GroupNorm -> Q/K/V -> softmax(QK^T/sqrt(d)) @ V -> out + residual.

    Uses 1 attention head (head_dim = ch). Q/K/V are Linear layers stored as
    [out, in] in safetensors; we reshape to [out, in, 1, 1] for 1x1 Conv2d.
    """
    from tensorrt_model_connect import trt_compat
    trt = trt_compat.get_trt()
    from ... import graph_ops
    from ...graph_ops import add_conv2d

    residual = inp

    # Group norm
    x = _add_group_norm_4d(
        network, inp, ch, 32,
        _get_weight(weights, f"{prefix}.group_norm.weight"),
        _get_weight(weights, f"{prefix}.group_norm.bias"))

    # Q, K, V via 1x1 conv (Linear weights reshaped to Conv2d format)
    q = add_conv2d(
        network, x,
        weight=_linear_weight_to_conv2d(
            _get_weight(weights, f"{prefix}.to_q.weight")),
        bias=_get_weight(weights, f"{prefix}.to_q.bias"),
        out_channels=ch, kernel_size=(1, 1))
    k = add_conv2d(
        network, x,
        weight=_linear_weight_to_conv2d(
            _get_weight(weights, f"{prefix}.to_k.weight")),
        bias=_get_weight(weights, f"{prefix}.to_k.bias"),
        out_channels=ch, kernel_size=(1, 1))
    v = add_conv2d(
        network, x,
        weight=_linear_weight_to_conv2d(
            _get_weight(weights, f"{prefix}.to_v.weight")),
        bias=_get_weight(weights, f"{prefix}.to_v.bias"),
        out_channels=ch, kernel_size=(1, 1))

    # Reshape [1, ch, H, W] -> [1, ch, H*W] then transpose to [1, H*W, ch]
    seq_len = h * w

    q_r = network.add_shuffle(q)
    q_r.reshape_dims = (1, ch, seq_len)
    q_t = network.add_shuffle(q_r.get_output(0))
    q_t.second_transpose = (0, 2, 1)  # [1, seq, ch]

    k_r = network.add_shuffle(k)
    k_r.reshape_dims = (1, ch, seq_len)
    k_t = network.add_shuffle(k_r.get_output(0))
    k_t.second_transpose = (0, 2, 1)  # [1, seq, ch]

    v_r = network.add_shuffle(v)
    v_r.reshape_dims = (1, ch, seq_len)
    v_t = network.add_shuffle(v_r.get_output(0))
    v_t.second_transpose = (0, 2, 1)  # [1, seq, ch]

    q_4d = network.add_shuffle(q_t.get_output(0))
    q_4d.reshape_dims = (1, 1, seq_len, ch)
    k_4d = network.add_shuffle(k_t.get_output(0))
    k_4d.reshape_dims = (1, 1, seq_len, ch)
    v_4d = network.add_shuffle(v_t.get_output(0))
    v_4d.reshape_dims = (1, 1, seq_len, ch)
    attn_out = graph_ops.add_attention_core(
        network,
        q_4d.get_output(0),
        k_4d.get_output(0),
        v_4d.get_output(0),
        scale=1.0 / np.sqrt(ch))

    # Reshape back: [1, 1, seq, ch] -> [1, seq, ch] -> [1, ch, seq] -> [1, ch, H, W]
    attn_3d = network.add_shuffle(attn_out)
    attn_3d.reshape_dims = (1, seq_len, ch)
    attn_tr = network.add_shuffle(attn_3d.get_output(0))
    attn_tr.second_transpose = (0, 2, 1)  # [1, ch, seq]
    attn_4d = network.add_shuffle(attn_tr.get_output(0))
    attn_4d.reshape_dims = (1, ch, h, w)

    # Output projection (1x1 conv; Linear weights reshaped to Conv2d format)
    out = add_conv2d(
        network, attn_4d.get_output(0),
        weight=_linear_weight_to_conv2d(
            _get_weight(weights, f"{prefix}.to_out.0.weight")),
        bias=_get_weight(weights, f"{prefix}.to_out.0.bias"),
        out_channels=ch, kernel_size=(1, 1))

    # Residual
    return network.add_elementwise(
        residual, out, trt.ElementWiseOperation.SUM).get_output(0)


def _add_upsample_2d(network, inp, weights, prefix: str,
                      ch: int, h: int, w: int):
    """Nearest-neighbor 2x upsample + Conv2d(3x3, pad=1)."""
    from tensorrt_model_connect import trt_compat
    trt = trt_compat.get_trt()
    from ...graph_ops import add_conv2d

    n, c = inp.shape[0], inp.shape[1]
    resize = network.add_resize(inp)
    resize.resize_mode = trt.InterpolationMode.NEAREST
    resize.shape = (n, c, h * 2, w * 2)

    return add_conv2d(
        network, resize.get_output(0),
        weight=_get_weight(weights, f"{prefix}.conv.weight"),
        bias=_get_weight(weights, f"{prefix}.conv.bias"),
        out_channels=ch, kernel_size=(3, 3), padding=(1, 1))


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

# AutoencoderKL decoder default config (matching FLUX/Z-Image VAE)
_BLOCK_OUT_CHANNELS = (128, 256, 512, 512)
_NUM_GROUPS = 32


def build_vae_2d_decoder_engine(
    model_dir: str,
    *,
    latent_channels: int = 16,
    h_lat: int,
    w_lat: int,
    scaling_factor: float = 0.3611,
    shift_factor: float = 0.1159,
    verbose: bool = False,
    build_timing: dict | None = None,
    timing_component: str = "vae_decoder",
) -> bytes:
    """Build a TRT engine for AutoencoderKL decoder using the TensorRT Python API.

    Loads weights from safetensors and constructs the full decoder graph:
    conv_in -> mid_block -> up_blocks -> norm -> SiLU -> conv_out.

    Scaling (latents / scale_factor + shift_factor) is NOT done inside the engine;
    the C++ runtime handles it.

    Input tensor:  "latent_input"    [1, latent_channels, h_lat, w_lat]
    Output tensor: "decoder_output"  [1, 3, h_out, w_out]
    """
    from tensorrt_model_connect import trt_compat
    trt = trt_compat.get_trt()
    from ...graph_ops import add_conv2d, add_silu

    total_t0 = time.monotonic()
    weights_before = _timing_phase(build_timing, "weights_loading_s")
    h_out = h_lat * 8
    w_out = w_lat * 8

    print(f"[vae-2d] Loading VAE weights from {model_dir} ...", file=sys.stderr)
    readers = _TimedWeightReaders(
        _open_vae_safetensors(model_dir), build_timing, timing_component)

    # Reversed block_out_channels for decoder (up path)
    reversed_channels = list(reversed(_BLOCK_OUT_CHANNELS))  # [512, 512, 256, 128]
    ch_last = reversed_channels[0]  # 512

    print(f"[vae-2d] Building TRT graph: latent [{latent_channels},{h_lat},{w_lat}] "
          f"-> image [3,{h_out},{w_out}]", file=sys.stderr)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 << 30)

    # Input
    inp = network.add_input("latent_input", trt.float32,
                            (1, latent_channels, h_lat, w_lat))

    # post_quant_conv: Conv2d(latent_channels, latent_channels, 1x1)
    # Applied before the decoder in AutoencoderKL.decode().
    try:
        pqc_w = _get_weight(readers, "post_quant_conv.weight")
        pqc_b = _get_weight(readers, "post_quant_conv.bias")
        x = add_conv2d(
            network, inp,
            weight=pqc_w, bias=pqc_b,
            out_channels=latent_channels,
            kernel_size=(1, 1))
    except KeyError:
        x = inp  # No post_quant_conv (some VAEs omit it)

    # decoder.conv_in: Conv2d(latent_channels, 512, 3x3, pad=1)
    x = add_conv2d(
        network, x,
        weight=_get_weight(readers, "decoder.conv_in.weight"),
        bias=_get_weight(readers, "decoder.conv_in.bias"),
        out_channels=ch_last,
        kernel_size=(3, 3),
        padding=(1, 1))

    cur_h, cur_w = h_lat, w_lat

    # ---------- Mid block ----------
    # ResNetBlock2D(512, 512)
    x = _add_resnet_block_2d(
        network, x, readers,
        "decoder.mid_block.resnets.0", ch_last, ch_last, cur_h, cur_w)

    # SelfAttention2D(512)
    x = _add_self_attention_2d(
        network, x, readers,
        "decoder.mid_block.attentions.0", ch_last, cur_h, cur_w)

    # ResNetBlock2D(512, 512)
    x = _add_resnet_block_2d(
        network, x, readers,
        "decoder.mid_block.resnets.1", ch_last, ch_last, cur_h, cur_w)

    # ---------- Up blocks ----------
    # 4 up blocks with reversed_channels = [512, 512, 256, 128]
    # Block 0 (512->512): 3 resnets + upsample
    # Block 1 (512->256): first resnet 512->256, next 2 256->256, + upsample
    # Block 2 (256->128): first resnet 256->128, next 2 128->128, + upsample
    # Block 3 (128->128): 3 resnets, no upsample
    prev_ch = ch_last  # starts at 512
    for block_idx in range(4):
        out_ch = reversed_channels[block_idx]

        for resnet_idx in range(3):
            if resnet_idx == 0:
                in_ch = prev_ch
            else:
                in_ch = out_ch

            x = _add_resnet_block_2d(
                network, x, readers,
                f"decoder.up_blocks.{block_idx}.resnets.{resnet_idx}",
                in_ch, out_ch, cur_h, cur_w)

        # Upsample for blocks 0,1,2 (not block 3)
        if block_idx < 3:
            x = _add_upsample_2d(
                network, x, readers,
                f"decoder.up_blocks.{block_idx}.upsamplers.0",
                out_ch, cur_h, cur_w)
            cur_h *= 2
            cur_w *= 2

        prev_ch = out_ch

    # ---------- Final norm + SiLU + conv_out ----------
    final_ch = reversed_channels[-1]  # 128

    x = _add_group_norm_4d(
        network, x, final_ch, _NUM_GROUPS,
        _get_weight(readers, "decoder.conv_norm_out.weight"),
        _get_weight(readers, "decoder.conv_norm_out.bias"))

    x = add_silu(network, x)

    x = add_conv2d(
        network, x,
        weight=_get_weight(readers, "decoder.conv_out.weight"),
        bias=_get_weight(readers, "decoder.conv_out.bias"),
        out_channels=3,
        kernel_size=(3, 3),
        padding=(1, 1))

    # Mark output
    cast_x = network.add_cast(x, trt.float32)
    x_out = cast_x.get_output(0)
    x_out.name = "decoder_output"
    network.mark_output(x_out)

    print("[vae-2d] Building TRT engine ...", file=sys.stderr)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine build failed for VAE decoder")

    plan_bytes = bytes(plan)
    weights_after = _timing_phase(build_timing, "weights_loading_s")
    compile_elapsed = max(
        0.0, time.monotonic() - total_t0 - max(0.0, weights_after - weights_before))
    add_trt_compile_timing(build_timing, timing_component, compile_elapsed)
    print(f"[vae-2d] Engine built: {len(plan_bytes) / 1e6:.1f} MB", file=sys.stderr)
    return plan_bytes


def _timing_phase(timing: dict | None, key: str) -> float:
    if timing is None:
        return 0.0
    phases = timing.setdefault("phases", {})
    try:
        return float(phases.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Placeholder (for testing without real weights)
# ---------------------------------------------------------------------------

def build_vae_2d_placeholder(
    latent_channels: int,
    h_lat: int,
    w_lat: int,
    verbose: bool = False,
) -> bytes:
    """Build a minimal VAE placeholder engine (zeros output, for testing).

    Used when the real VAE can't be built (no weights available).
    """
    from tensorrt_model_connect import trt_compat
    trt = trt_compat.get_trt()

    h_out = h_lat * 8
    w_out = w_lat * 8

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)

    network.add_input("latent_input", trt.float32,
                      (1, latent_channels, h_lat, w_lat))

    zeros = np.zeros((1, 3, h_out, w_out), dtype=np.float32)
    const = network.add_constant((1, 3, h_out, w_out), trt.Weights(zeros))
    out = const.get_output(0)
    cast_out = network.add_cast(out, trt.float32)
    out_final = cast_out.get_output(0)
    out_final.name = "decoder_output"
    network.mark_output(out_final)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("VAE placeholder engine build failed")
    return bytes(plan)
