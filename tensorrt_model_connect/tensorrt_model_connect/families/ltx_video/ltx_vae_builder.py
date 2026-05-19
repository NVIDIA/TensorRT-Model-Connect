"""LTX-Video VAE builder using the raw TensorRT network API.

The encoder/decoder are the ``LTXVideoEncoder3d`` and non-causal
``LTXVideoDecoder3d`` from ``AutoencoderKLLTXVideo``. They are built directly
with TensorRT layers: 3D convolutions, channel RMSNorm/LayerNorm, SiLU,
temporal/spatial pixel shuffle, and patchify/unpatchify reshapes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tensorrt_model_connect import trt_compat
from ...checkpoint_mapper import WeightDict, _has_tensor, _load_tensor, _open_safetensors

if TYPE_CHECKING:
    from collections.abc import Mapping

trt: Any = None
graph_ops: Any = None


def _ensure_trt() -> Any:
    global trt
    if trt is None:
        trt = trt_compat.get_trt()
    return trt


def _ensure_graph_ops() -> Any:
    global graph_ops
    if graph_ops is None:
        from ... import graph_ops as graph_ops_module

        graph_ops = graph_ops_module
    return graph_ops


def _target_np_dtype(precision: str) -> np.dtype:
    return np.float16 if precision == "fp16" else np.float32


def _trt_dtype(precision: str) -> trt.DataType:
    trt_module = _ensure_trt()
    return trt_module.float16 if precision == "fp16" else trt_module.float32


def _cast_back(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    dtype: trt.DataType,
) -> trt.ITensor:
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def load_ltx_vae_weights(
    model_dir: str | Path,
    *,
    precision: str = "fp16",
) -> WeightDict:
    """Load LTX VAE decoder weights from a diffusers VAE directory."""
    readers = _open_safetensors(Path(model_dir))
    dtype = _target_np_dtype(precision)
    weights = WeightDict()

    def f(name: str, *, norm: bool = False) -> np.ndarray:
        return np.ascontiguousarray(
            _load_tensor(readers, name), dtype=np.float32 if norm else dtype
        )

    def maybe(name: str, *, norm: bool = False) -> np.ndarray | None:
        if not _has_tensor(readers, name):
            return None
        return f(name, norm=norm)

    for name in ("latents_mean", "latents_std"):
        value = maybe(name, norm=True)
        if value is not None:
            weights[name] = value

    weights["decoder.conv_in.conv.weight"] = f("decoder.conv_in.conv.weight")
    weights["decoder.conv_in.conv.bias"] = f("decoder.conv_in.conv.bias")

    # Mid block: four ResNet blocks in the default 2B LTX decoder.
    _load_resnet_series(weights, readers, "decoder.mid_block.resnets", dtype)

    # Up blocks: keys are sparse because only channel-changing blocks have
    # conv_in, and block 0 has no upsampler.
    block = 0
    while _has_tensor(readers, f"decoder.up_blocks.{block}.resnets.0.conv1.conv.weight"):
        conv_in = f"decoder.up_blocks.{block}.conv_in"
        if _has_tensor(readers, f"{conv_in}.conv1.conv.weight"):
            _load_resnet_block(weights, readers, conv_in, dtype)

        up = f"decoder.up_blocks.{block}.upsamplers.0.conv.conv"
        if _has_tensor(readers, f"{up}.weight"):
            weights[f"{up}.weight"] = f(f"{up}.weight")
            weights[f"{up}.bias"] = f(f"{up}.bias")

        _load_resnet_series(
            weights, readers, f"decoder.up_blocks.{block}.resnets", dtype
        )
        block += 1

    weights["decoder.conv_out.conv.weight"] = f("decoder.conv_out.conv.weight")
    weights["decoder.conv_out.conv.bias"] = f("decoder.conv_out.conv.bias")
    return weights


def load_ltx_vae_encoder_weights(
    model_dir: str | Path,
    *,
    precision: str = "fp16",
) -> WeightDict:
    """Load LTX VAE encoder weights from a diffusers VAE directory."""
    readers = _open_safetensors(Path(model_dir))
    dtype = _target_np_dtype(precision)
    weights = WeightDict()

    def f(name: str, *, norm: bool = False) -> np.ndarray:
        return np.ascontiguousarray(
            _load_tensor(readers, name), dtype=np.float32 if norm else dtype
        )

    def maybe(name: str, *, norm: bool = False) -> np.ndarray | None:
        if not _has_tensor(readers, name):
            return None
        return f(name, norm=norm)

    for name in ("latents_mean", "latents_std"):
        value = maybe(name, norm=True)
        if value is not None:
            weights[name] = value

    weights["encoder.conv_in.conv.weight"] = f("encoder.conv_in.conv.weight")
    weights["encoder.conv_in.conv.bias"] = f("encoder.conv_in.conv.bias")

    block = 0
    while _has_tensor(readers, f"encoder.down_blocks.{block}.resnets.0.conv1.conv.weight"):
        _load_resnet_series(
            weights, readers, f"encoder.down_blocks.{block}.resnets", dtype
        )
        down = f"encoder.down_blocks.{block}.downsamplers.0.conv"
        if _has_tensor(readers, f"{down}.weight"):
            weights[f"{down}.weight"] = f(f"{down}.weight")
            weights[f"{down}.bias"] = f(f"{down}.bias")

        conv_out = f"encoder.down_blocks.{block}.conv_out"
        if _has_tensor(readers, f"{conv_out}.conv1.conv.weight"):
            _load_resnet_block(weights, readers, conv_out, dtype)
        block += 1

    _load_resnet_series(weights, readers, "encoder.mid_block.resnets", dtype)
    weights["encoder.conv_out.conv.weight"] = f("encoder.conv_out.conv.weight")
    weights["encoder.conv_out.conv.bias"] = f("encoder.conv_out.conv.bias")
    return weights


def build_ltx_vae_decoder_engine(
    weights: "Mapping[str, np.ndarray]",
    *,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    latent_channels: int = 128,
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
    layers_per_block: tuple[int, ...] = (4, 3, 3, 3, 4),
    spatio_temporal_scaling: tuple[bool, ...] = (True, True, True, False),
    patch_size: int = 4,
    patch_size_t: int = 1,
    out_channels: int = 3,
    precision: str = "fp16",
    denormalize_input: bool = False,
    scaling_factor: float = 1.0,
    verbose: bool = False,
) -> bytes:
    """Build the LTX VAE decoder as a TensorRT plan."""
    if precision not in ("fp16", "fp32"):
        raise ValueError("LTX VAE raw builder currently supports fp16 or fp32")

    _ensure_trt()
    _ensure_graph_ops()
    trt_dtype = _trt_dtype(precision)
    dtype = _target_np_dtype(precision)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )

    x = network.add_input(
        "latents",
        trt_dtype,
        (1, latent_channels, latent_frames, latent_height, latent_width),
    )
    if denormalize_input:
        x = _denormalize_ltx_latents(
            network,
            x,
            weights,
            latent_channels,
            scaling_factor=scaling_factor,
        )

    dec_channels = list(reversed(block_out_channels))
    dec_scaling = list(reversed(spatio_temporal_scaling))
    dec_layers = list(reversed(layers_per_block))
    cur_t = latent_frames
    cur_h = latent_height
    cur_w = latent_width

    x = _conv3d_noncausal(
        network,
        x,
        weights["decoder.conv_in.conv.weight"],
        weights["decoder.conv_in.conv.bias"],
        dtype=dtype,
    )

    mid_layers = _count_resnets(weights, "decoder.mid_block.resnets")
    for i in range(mid_layers):
        x = _resnet_block(
            network,
            x,
            weights,
            f"decoder.mid_block.resnets.{i}",
            in_channels=dec_channels[0],
            out_channels=dec_channels[0],
            dtype=dtype,
        )

    prev_ch = dec_channels[0]
    for block_idx, out_ch in enumerate(dec_channels):
        conv_in_prefix = f"decoder.up_blocks.{block_idx}.conv_in"
        if f"{conv_in_prefix}.conv1.conv.weight" in weights:
            x = _resnet_block(
                network,
                x,
                weights,
                conv_in_prefix,
                in_channels=prev_ch,
                out_channels=out_ch,
                dtype=dtype,
            )
            prev_ch = out_ch

        if dec_scaling[block_idx]:
            up_prefix = f"decoder.up_blocks.{block_idx}.upsamplers.0.conv.conv"
            x = _conv3d_noncausal(
                network,
                x,
                weights[f"{up_prefix}.weight"],
                weights[f"{up_prefix}.bias"],
                dtype=dtype,
            )
            x = _pixel_shuffle_3d(
                network,
                x,
                channels=out_ch,
                frames=cur_t,
                height=cur_h,
                width=cur_w,
                stride=(2, 2, 2),
            )
            cur_t = cur_t * 2 - 1
            cur_h *= 2
            cur_w *= 2
            prev_ch = out_ch

        resnet_count = _count_resnets(
            weights, f"decoder.up_blocks.{block_idx}.resnets"
        )
        if block_idx + 1 < len(dec_layers):
            expected = dec_layers[block_idx + 1]
            if resnet_count and resnet_count != expected:
                print(
                    f"[ltx-vae] warning: up block {block_idx} has "
                    f"{resnet_count} resnets, config expected {expected}",
                    file=sys.stderr,
                )
        for res_idx in range(resnet_count):
            x = _resnet_block(
                network,
                x,
                weights,
                f"decoder.up_blocks.{block_idx}.resnets.{res_idx}",
                in_channels=prev_ch,
                out_channels=out_ch,
                dtype=dtype,
            )
            prev_ch = out_ch

    x = _rms_norm_channels(network, x, prev_ch, eps=1e-8)
    x = graph_ops.add_silu(network, x)
    x = _conv3d_noncausal(
        network,
        x,
        weights["decoder.conv_out.conv.weight"],
        weights["decoder.conv_out.conv.bias"],
        dtype=dtype,
    )

    x = _unpatchify(
        network,
        x,
        batch=1,
        out_channels=out_channels,
        frames=cur_t,
        height=cur_h,
        width=cur_w,
        patch_size=patch_size,
        patch_size_t=patch_size_t,
    )
    x = network.add_cast(x, trt.float32).get_output(0)
    x.name = "sample"
    network.mark_output(x)

    print(
        "[ltx-vae] Building TRT engine "
        f"(precision={precision}, latent={latent_frames}x{latent_height}x"
        f"{latent_width}, output={cur_t * patch_size_t}x"
        f"{cur_h * patch_size}x{cur_w * patch_size}) ...",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for LTX VAE decoder")
    return bytes(plan)


def build_ltx_vae_encoder_engine(
    weights: "Mapping[str, np.ndarray]",
    *,
    sample_frames: int,
    sample_height: int,
    sample_width: int,
    in_channels: int = 3,
    latent_channels: int = 128,
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
    layers_per_block: tuple[int, ...] = (4, 3, 3, 3, 4),
    spatio_temporal_scaling: tuple[bool, ...] = (True, True, True, False),
    patch_size: int = 4,
    patch_size_t: int = 1,
    precision: str = "fp16",
    normalize_output: bool = True,
    scaling_factor: float = 1.0,
    verbose: bool = False,
) -> bytes:
    """Build the LTX VAE encoder as a TensorRT plan."""
    if precision not in ("fp16", "fp32"):
        raise ValueError("LTX VAE raw builder currently supports fp16 or fp32")
    if sample_frames % patch_size_t != 0:
        raise ValueError("sample_frames must be divisible by patch_size_t")
    if sample_height % patch_size != 0 or sample_width % patch_size != 0:
        raise ValueError("sample dimensions must be divisible by patch_size")

    _ensure_trt()
    _ensure_graph_ops()
    trt_dtype = _trt_dtype(precision)
    dtype = _target_np_dtype(precision)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )

    x = network.add_input(
        "sample",
        trt_dtype,
        (1, in_channels, sample_frames, sample_height, sample_width),
    )
    cur_t = sample_frames // patch_size_t
    cur_h = sample_height // patch_size
    cur_w = sample_width // patch_size
    x = _patchify(
        network,
        x,
        batch=1,
        in_channels=in_channels,
        frames=cur_t,
        height=cur_h,
        width=cur_w,
        patch_size=patch_size,
        patch_size_t=patch_size_t,
    )
    x = _conv3d_causal(
        network,
        x,
        weights["encoder.conv_in.conv.weight"],
        weights["encoder.conv_in.conv.bias"],
        dtype=dtype,
    )

    prev_ch = block_out_channels[0]
    for block_idx, out_ch in enumerate(block_out_channels):
        if block_idx + 1 < len(block_out_channels):
            out_ch = block_out_channels[block_idx + 1]

        resnet_count = _count_resnets(weights, f"encoder.down_blocks.{block_idx}.resnets")
        if block_idx < len(layers_per_block):
            expected = layers_per_block[block_idx]
            if resnet_count and resnet_count != expected:
                print(
                    f"[ltx-vae] warning: down block {block_idx} has "
                    f"{resnet_count} resnets, config expected {expected}",
                    file=sys.stderr,
                )
        for res_idx in range(resnet_count):
            x = _resnet_block(
                network,
                x,
                weights,
                f"encoder.down_blocks.{block_idx}.resnets.{res_idx}",
                in_channels=prev_ch,
                out_channels=prev_ch,
                dtype=dtype,
                causal=True,
            )

        down_prefix = f"encoder.down_blocks.{block_idx}.downsamplers.0.conv"
        if spatio_temporal_scaling[block_idx] and f"{down_prefix}.weight" in weights:
            x = _conv3d_causal(
                network,
                x,
                weights[f"{down_prefix}.weight"],
                weights[f"{down_prefix}.bias"],
                dtype=dtype,
                stride=(2, 2, 2),
            )
            cur_t = (cur_t - 1) // 2 + 1
            cur_h = (cur_h - 1) // 2 + 1
            cur_w = (cur_w - 1) // 2 + 1

        conv_out_prefix = f"encoder.down_blocks.{block_idx}.conv_out"
        if f"{conv_out_prefix}.conv1.conv.weight" in weights:
            x = _resnet_block(
                network,
                x,
                weights,
                conv_out_prefix,
                in_channels=prev_ch,
                out_channels=out_ch,
                dtype=dtype,
                causal=True,
            )
            prev_ch = out_ch

    mid_layers = _count_resnets(weights, "encoder.mid_block.resnets")
    if mid_layers and len(layers_per_block) > len(block_out_channels):
        expected = layers_per_block[-1]
        if mid_layers != expected:
            print(
                f"[ltx-vae] warning: mid block has {mid_layers} resnets, "
                f"config expected {expected}",
                file=sys.stderr,
            )
    for i in range(mid_layers):
        x = _resnet_block(
            network,
            x,
            weights,
            f"encoder.mid_block.resnets.{i}",
            in_channels=prev_ch,
            out_channels=prev_ch,
            dtype=dtype,
            causal=True,
        )

    x = _rms_norm_channels(network, x, prev_ch, eps=1e-8)
    x = graph_ops.add_silu(network, x)
    x = _conv3d_causal(
        network,
        x,
        weights["encoder.conv_out.conv.weight"],
        weights["encoder.conv_out.conv.bias"],
        dtype=dtype,
    )
    x = network.add_slice(
        x,
        (0, 0, 0, 0, 0),
        (1, latent_channels, cur_t, cur_h, cur_w),
        (1, 1, 1, 1, 1),
    ).get_output(0)
    if normalize_output:
        x = _normalize_ltx_latents(
            network,
            x,
            weights,
            latent_channels,
            scaling_factor=scaling_factor,
        )
    x = network.add_cast(x, trt.float32).get_output(0)
    x.name = "latent"
    network.mark_output(x)

    print(
        "[ltx-vae] Building TRT encoder "
        f"(precision={precision}, input={sample_frames}x{sample_height}x"
        f"{sample_width}, latent={cur_t}x{cur_h}x{cur_w}) ...",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for LTX VAE encoder")
    return bytes(plan)


def _load_resnet_series(
    weights: WeightDict,
    readers: list,
    prefix: str,
    dtype: np.dtype,
) -> None:
    idx = 0
    while _has_tensor(readers, f"{prefix}.{idx}.conv1.conv.weight"):
        _load_resnet_block(weights, readers, f"{prefix}.{idx}", dtype)
        idx += 1


def _load_resnet_block(
    weights: WeightDict,
    readers: list,
    prefix: str,
    dtype: np.dtype,
) -> None:
    def f(name: str, *, norm: bool = False) -> np.ndarray:
        return np.ascontiguousarray(
            _load_tensor(readers, name), dtype=np.float32 if norm else dtype
        )

    for conv in ("conv1", "conv2"):
        weights[f"{prefix}.{conv}.conv.weight"] = f(f"{prefix}.{conv}.conv.weight")
        weights[f"{prefix}.{conv}.conv.bias"] = f(f"{prefix}.{conv}.conv.bias")

    if _has_tensor(readers, f"{prefix}.norm3.weight"):
        weights[f"{prefix}.norm3.weight"] = f(f"{prefix}.norm3.weight", norm=True)
        weights[f"{prefix}.norm3.bias"] = f(f"{prefix}.norm3.bias", norm=True)
    if _has_tensor(readers, f"{prefix}.conv_shortcut.conv.weight"):
        weights[f"{prefix}.conv_shortcut.conv.weight"] = f(
            f"{prefix}.conv_shortcut.conv.weight"
        )
        weights[f"{prefix}.conv_shortcut.conv.bias"] = f(
            f"{prefix}.conv_shortcut.conv.bias"
        )


def _count_resnets(weights: "Mapping[str, np.ndarray]", prefix: str) -> int:
    idx = 0
    while f"{prefix}.{idx}.conv1.conv.weight" in weights:
        idx += 1
    return idx


def _conv3d_noncausal(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    *,
    dtype: np.dtype,
) -> trt.ITensor:
    b, _c, t, _h, _w = inp.shape
    out_channels, _in_channels, kt, kh, kw = weight.shape
    x = inp
    if kt > 1:
        left = network.add_slice(
            inp, (0, 0, 0, 0, 0), (b, inp.shape[1], 1, inp.shape[3], inp.shape[4]), (1, 1, 1, 1, 1)
        ).get_output(0)
        right = network.add_slice(
            inp,
            (0, 0, t - 1, 0, 0),
            (b, inp.shape[1], 1, inp.shape[3], inp.shape[4]),
            (1, 1, 1, 1, 1),
        ).get_output(0)
        concat = network.add_concatenation([left, inp, right])
        concat.axis = 2
        x = concat.get_output(0)

    conv = network.add_convolution_nd(
        x,
        num_output_maps=out_channels,
        kernel_shape=(kt, kh, kw),
        kernel=trt.Weights(np.ascontiguousarray(weight, dtype=dtype)),
        bias=trt.Weights(np.ascontiguousarray(bias, dtype=dtype))
        if bias is not None
        else trt.Weights(),
    )
    conv.stride_nd = (1, 1, 1)
    conv.padding_nd = (0, kh // 2, kw // 2)
    return conv.get_output(0)


def _conv3d_causal(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    *,
    dtype: np.dtype,
    stride: tuple[int, int, int] = (1, 1, 1),
) -> trt.ITensor:
    b, _c, _t, _h, _w = inp.shape
    out_channels, _in_channels, kt, kh, kw = weight.shape
    x = inp
    if kt > 1:
        pad = network.add_slice(
            inp,
            (0, 0, 0, 0, 0),
            (b, inp.shape[1], 1, inp.shape[3], inp.shape[4]),
            (1, 1, 1, 1, 1),
        ).get_output(0)
        pads = [pad for _ in range(kt - 1)]
        concat = network.add_concatenation([*pads, inp])
        concat.axis = 2
        x = concat.get_output(0)

    conv = network.add_convolution_nd(
        x,
        num_output_maps=out_channels,
        kernel_shape=(kt, kh, kw),
        kernel=trt.Weights(np.ascontiguousarray(weight, dtype=dtype)),
        bias=trt.Weights(np.ascontiguousarray(bias, dtype=dtype))
        if bias is not None
        else trt.Weights(),
    )
    conv.stride_nd = stride
    conv.padding_nd = (0, kh // 2, kw // 2)
    return conv.get_output(0)


def _rms_norm_channels(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    channels: int,
    *,
    eps: float,
) -> trt.ITensor:
    out_dtype = inp.dtype
    x = inp if inp.dtype == trt.float32 else network.add_cast(inp, trt.float32).get_output(0)
    sq = network.add_elementwise(x, x, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True
    )
    eps_t = graph_ops.add_constant(
        network, (1, 1, 1, 1, 1), np.array([eps], dtype=np.float32)
    )
    denom = network.add_elementwise(
        mean.get_output(0), eps_t, trt.ElementWiseOperation.SUM
    )
    sqrt = network.add_unary(denom.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt.get_output(0), trt.UnaryOperation.RECIP)
    out = network.add_elementwise(
        x, recip.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)
    if channels <= 0:
        raise ValueError("channels must be positive")
    return _cast_back(network, out, out_dtype)


def _layer_norm_channels(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    channels: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    *,
    eps: float,
) -> trt.ITensor:
    out_dtype = inp.dtype
    x = inp if inp.dtype == trt.float32 else network.add_cast(inp, trt.float32).get_output(0)
    mean = network.add_reduce(x, trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    centered = network.add_elementwise(
        x, mean.get_output(0), trt.ElementWiseOperation.SUB
    ).get_output(0)
    sq = network.add_elementwise(centered, centered, trt.ElementWiseOperation.PROD)
    var = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True
    )
    eps_t = graph_ops.add_constant(
        network, (1, 1, 1, 1, 1), np.array([eps], dtype=np.float32)
    )
    denom = network.add_elementwise(
        var.get_output(0), eps_t, trt.ElementWiseOperation.SUM
    )
    sqrt = network.add_unary(denom.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt.get_output(0), trt.UnaryOperation.RECIP)
    norm = network.add_elementwise(
        centered, recip.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)
    gamma_t = graph_ops.add_constant(
        network,
        (1, channels, 1, 1, 1),
        gamma.reshape(1, channels, 1, 1, 1),
        dtype=np.float32,
    )
    beta_t = graph_ops.add_constant(
        network,
        (1, channels, 1, 1, 1),
        beta.reshape(1, channels, 1, 1, 1),
        dtype=np.float32,
    )
    scaled = network.add_elementwise(norm, gamma_t, trt.ElementWiseOperation.PROD)
    out = network.add_elementwise(
        scaled.get_output(0), beta_t, trt.ElementWiseOperation.SUM
    ).get_output(0)
    return _cast_back(network, out, out_dtype)


def _resnet_block(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: "Mapping[str, np.ndarray]",
    prefix: str,
    *,
    in_channels: int,
    out_channels: int,
    dtype: np.dtype,
    causal: bool = False,
) -> trt.ITensor:
    conv3d = _conv3d_causal if causal else _conv3d_noncausal
    h = _rms_norm_channels(network, inp, in_channels, eps=1e-8)
    h = graph_ops.add_silu(network, h)
    h = conv3d(
        network,
        h,
        weights[f"{prefix}.conv1.conv.weight"],
        weights[f"{prefix}.conv1.conv.bias"],
        dtype=dtype,
    )
    h = _rms_norm_channels(network, h, out_channels, eps=1e-8)
    h = graph_ops.add_silu(network, h)
    h = conv3d(
        network,
        h,
        weights[f"{prefix}.conv2.conv.weight"],
        weights[f"{prefix}.conv2.conv.bias"],
        dtype=dtype,
    )

    shortcut = inp
    if in_channels != out_channels:
        shortcut = _layer_norm_channels(
            network,
            shortcut,
            in_channels,
            weights[f"{prefix}.norm3.weight"],
            weights[f"{prefix}.norm3.bias"],
            eps=1e-6,
        )
        shortcut = conv3d(
            network,
            shortcut,
            weights[f"{prefix}.conv_shortcut.conv.weight"],
            weights[f"{prefix}.conv_shortcut.conv.bias"],
            dtype=dtype,
        )

    return network.add_elementwise(
        h, shortcut, trt.ElementWiseOperation.SUM
    ).get_output(0)


def _patchify(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    batch: int,
    in_channels: int,
    frames: int,
    height: int,
    width: int,
    patch_size: int,
    patch_size_t: int,
) -> trt.ITensor:
    r1 = network.add_shuffle(inp)
    r1.reshape_dims = (
        batch,
        in_channels,
        frames,
        patch_size_t,
        height,
        patch_size,
        width,
        patch_size,
    )
    r2 = network.add_shuffle(r1.get_output(0))
    r2.first_transpose = trt.Permutation([0, 1, 3, 7, 5, 2, 4, 6])
    r2.reshape_dims = (
        batch,
        in_channels * patch_size_t * patch_size * patch_size,
        frames,
        height,
        width,
    )
    return r2.get_output(0)


def _pixel_shuffle_3d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    channels: int,
    frames: int,
    height: int,
    width: int,
    stride: tuple[int, int, int],
) -> trt.ITensor:
    st, sh, sw = stride
    r1 = network.add_shuffle(inp)
    r1.reshape_dims = (1, channels, st, sh, sw, frames, height, width)
    r2 = network.add_shuffle(r1.get_output(0))
    r2.first_transpose = trt.Permutation([0, 1, 5, 2, 6, 3, 7, 4])
    r2.reshape_dims = (1, channels, frames * st, height * sh, width * sw)
    return network.add_slice(
        r2.get_output(0),
        (0, 0, st - 1, 0, 0),
        (1, channels, frames * st - (st - 1), height * sh, width * sw),
        (1, 1, 1, 1, 1),
    ).get_output(0)


def _unpatchify(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    batch: int,
    out_channels: int,
    frames: int,
    height: int,
    width: int,
    patch_size: int,
    patch_size_t: int,
) -> trt.ITensor:
    r1 = network.add_shuffle(inp)
    r1.reshape_dims = (
        batch,
        out_channels,
        patch_size_t,
        patch_size,
        patch_size,
        frames,
        height,
        width,
    )
    r2 = network.add_shuffle(r1.get_output(0))
    r2.first_transpose = trt.Permutation([0, 1, 5, 2, 6, 4, 7, 3])
    r2.reshape_dims = (
        batch,
        out_channels,
        frames * patch_size_t,
        height * patch_size,
        width * patch_size,
    )
    return r2.get_output(0)


def _latents_stats(
    weights: "Mapping[str, np.ndarray]",
    channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean = weights.get("latents_mean")
    std = weights.get("latents_std")
    if mean is None or std is None:
        return np.zeros((channels,), dtype=np.float32), np.ones((channels,), dtype=np.float32)
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(-1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(-1)
    if mean_arr.size < channels or std_arr.size < channels:
        raise ValueError("LTX VAE latent statistics do not cover all latent channels")
    return mean_arr[:channels], std_arr[:channels]


def _normalize_ltx_latents(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: "Mapping[str, np.ndarray]",
    channels: int,
    *,
    scaling_factor: float,
) -> trt.ITensor:
    out_dtype = inp.dtype
    x = inp if inp.dtype == trt.float32 else network.add_cast(inp, trt.float32).get_output(0)
    mean, std = _latents_stats(weights, channels)
    mean_t = graph_ops.add_constant(
        network, (1, channels, 1, 1, 1), mean.reshape(1, channels, 1, 1, 1)
    )
    scale = (float(scaling_factor) / std).reshape(1, channels, 1, 1, 1)
    scale_t = graph_ops.add_constant(network, (1, channels, 1, 1, 1), scale)
    centered = network.add_elementwise(x, mean_t, trt.ElementWiseOperation.SUB)
    out = network.add_elementwise(centered.get_output(0), scale_t, trt.ElementWiseOperation.PROD)
    return _cast_back(network, out.get_output(0), out_dtype)


def _denormalize_ltx_latents(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: "Mapping[str, np.ndarray]",
    channels: int,
    *,
    scaling_factor: float,
) -> trt.ITensor:
    out_dtype = inp.dtype
    x = inp if inp.dtype == trt.float32 else network.add_cast(inp, trt.float32).get_output(0)
    mean, std = _latents_stats(weights, channels)
    scale = (std / float(scaling_factor or 1.0)).reshape(1, channels, 1, 1, 1)
    scale_t = graph_ops.add_constant(network, (1, channels, 1, 1, 1), scale)
    mean_t = graph_ops.add_constant(
        network, (1, channels, 1, 1, 1), mean.reshape(1, channels, 1, 1, 1)
    )
    scaled = network.add_elementwise(x, scale_t, trt.ElementWiseOperation.PROD)
    out = network.add_elementwise(scaled.get_output(0), mean_t, trt.ElementWiseOperation.SUM)
    return _cast_back(network, out.get_output(0), out_dtype)
