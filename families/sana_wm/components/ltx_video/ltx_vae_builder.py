# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
import tensorrt as trt
from ...builder_lifetime import get_process_trt_logger
from .checkpoint_mapper import WeightDict, _has_tensor, _load_tensor, _open_safetensors

if TYPE_CHECKING:
    from collections.abc import Mapping

graph_ops: Any = None
_weight_refs: list[np.ndarray] = []
_torch_conv3d_plugin_index = 0
_vae_rms_silu_plugin_index = 0
_vae_layer_norm_plugin_index = 0


def _ensure_trt() -> Any:
    return trt


def _ensure_graph_ops() -> Any:
    global graph_ops
    if graph_ops is None:
        from . import graph_ops as graph_ops_module

        graph_ops = graph_ops_module
    return graph_ops


def _target_np_dtype(precision: str) -> np.dtype:
    if precision == "bf16":
        try:
            import ml_dtypes
        except ModuleNotFoundError as exc:
            raise RuntimeError("LTX VAE bf16 builds require the ml_dtypes package") from exc
        return ml_dtypes.bfloat16
    return np.float16 if precision == "fp16" else np.float32


def _trt_dtype(precision: str) -> trt.DataType:
    trt_module = _ensure_trt()
    if precision == "bf16":
        return trt_module.bfloat16
    return trt_module.float16 if precision == "fp16" else trt_module.float32


def _is_bfloat16_dtype(dtype: np.dtype | type) -> bool:
    return np.dtype(dtype).name == "bfloat16"


def _trt_weights(values: np.ndarray, dtype: np.dtype | type) -> trt.Weights:
    arr = np.ascontiguousarray(values, dtype=dtype)
    if _is_bfloat16_dtype(arr.dtype):
        trt_module = _ensure_trt()
        _weight_refs.append(arr)
        return trt_module.Weights(trt_module.bfloat16, arr.ctypes.data, arr.size)
    return _ensure_trt().Weights(arr)


def _add_constant_for_trt_dtype(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    dtype: trt.DataType,
) -> trt.ITensor:
    trt_module = _ensure_trt()
    if dtype == trt_module.float32:
        return graph_ops.add_constant(network, shape, values, dtype=np.float32)
    if dtype == trt_module.float16:
        return graph_ops.add_constant(network, shape, values, dtype=np.float16)
    if dtype == trt_module.bfloat16:
        try:
            import ml_dtypes
        except ModuleNotFoundError as exc:
            raise RuntimeError("LTX VAE bf16 constants require the ml_dtypes package") from exc
        arr = np.ascontiguousarray(values, dtype=ml_dtypes.bfloat16)
        _weight_refs.append(arr)
        layer = network.add_constant(
            shape, trt_module.Weights(trt_module.bfloat16, arr.ctypes.data, arr.size)
        )
        return layer.get_output(0)
    raise TypeError(f"unsupported TensorRT dtype for LTX VAE constant: {dtype}")


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

        _load_resnet_series(weights, readers, f"decoder.up_blocks.{block}.resnets", dtype)
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
        _load_resnet_series(weights, readers, f"encoder.down_blocks.{block}.resnets", dtype)
        down = f"encoder.down_blocks.{block}.downsamplers.0.conv"
        down_weight = f"{down}.weight"
        if not _has_tensor(readers, down_weight):
            down_weight = f"{down}.conv.weight"
        if _has_tensor(readers, down_weight):
            down_prefix = down_weight.removesuffix(".weight")
            weights[f"{down}.weight"] = f(down_weight)
            weights[f"{down}.bias"] = f(f"{down_prefix}.bias")

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
    upsample_type: tuple[str, ...] | None = None,
    upsample_factor: tuple[int, ...] | None = None,
    upsample_residual: tuple[bool, ...] | None = None,
    patch_size: int = 4,
    patch_size_t: int = 1,
    out_channels: int = 3,
    precision: str = "fp16",
    denormalize_input: bool = False,
    scaling_factor: float = 1.0,
    spatial_padding_mode: str = "zeros",
    verbose: bool = False,
) -> bytes:
    """Build the LTX VAE decoder as a TensorRT plan."""
    global _torch_conv3d_plugin_index, _vae_rms_silu_plugin_index
    global _vae_layer_norm_plugin_index, _weight_refs
    _weight_refs = []
    _torch_conv3d_plugin_index = 0
    _vae_rms_silu_plugin_index = 0
    _vae_layer_norm_plugin_index = 0
    if precision not in ("bf16", "fp16", "fp32"):
        raise ValueError("LTX VAE raw builder currently supports bf16, fp16, or fp32")
    spatial_padding_mode = spatial_padding_mode.lower()
    if spatial_padding_mode not in ("zeros", "reflect"):
        raise ValueError(
            "LTX VAE raw decoder supports zeros or reflect spatial padding, "
            f"got {spatial_padding_mode!r}"
        )

    _ensure_trt()
    _ensure_graph_ops()
    trt_dtype = _trt_dtype(precision)
    dtype = _target_np_dtype(precision)

    logger = get_process_trt_logger(trt, verbose=verbose)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    debug_outputs = False

    def mark_debug(tensor: trt.ITensor, name: str) -> None:
        if not debug_outputs:
            return
        output = network.add_cast(tensor, trt.float32).get_output(0)
        output.name = name
        network.mark_output(output)

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
    mark_debug(x, "debug_denormalized_input")

    dec_scaling = list(reversed(spatio_temporal_scaling))
    dec_layers = list(reversed(layers_per_block))
    dec_upsample_type = list(reversed(upsample_type)) if upsample_type is not None else None
    dec_upsample_factor = list(reversed(upsample_factor)) if upsample_factor is not None else None
    dec_upsample_residual = (
        list(reversed(upsample_residual)) if upsample_residual is not None else None
    )
    cur_t = latent_frames
    cur_h = latent_height
    cur_w = latent_width

    x = _conv3d_noncausal(
        network,
        x,
        weights["decoder.conv_in.conv.weight"],
        weights["decoder.conv_in.conv.bias"],
        dtype=dtype,
        spatial_padding_mode=spatial_padding_mode,
    )
    mark_debug(x, "debug_conv_in")

    prev_ch = int(weights["decoder.conv_in.conv.weight"].shape[0])
    mid_layers = _count_resnets(weights, "decoder.mid_block.resnets")
    for i in range(mid_layers):
        mid_prefix = f"decoder.mid_block.resnets.{i}"
        res_in, res_out = _resnet_channels(weights, mid_prefix)
        x = _resnet_block(
            network,
            x,
            weights,
            mid_prefix,
            in_channels=res_in,
            out_channels=res_out,
            dtype=dtype,
            spatial_padding_mode=spatial_padding_mode,
        )
        prev_ch = res_out

    block_count = _count_indexed_blocks(weights, "decoder.up_blocks")
    for block_idx in range(block_count):
        conv_in_prefix = f"decoder.up_blocks.{block_idx}.conv_in"
        if f"{conv_in_prefix}.conv1.conv.weight" in weights:
            res_in, res_out = _resnet_channels(weights, conv_in_prefix)
            x = _resnet_block(
                network,
                x,
                weights,
                conv_in_prefix,
                in_channels=res_in,
                out_channels=res_out,
                dtype=dtype,
                spatial_padding_mode=spatial_padding_mode,
            )
            prev_ch = res_out

        up_prefix = f"decoder.up_blocks.{block_idx}.upsamplers.0.conv.conv"
        if f"{up_prefix}.weight" in weights and _sequence_at(dec_scaling, block_idx, True):
            stride = _upsample_stride(_sequence_at(dec_upsample_type, block_idx, "spatiotemporal"))
            up_weight = weights[f"{up_prefix}.weight"]
            upsampled_channels = _pixel_shuffle_output_channels(up_weight, stride)
            residual = x
            x = _conv3d_noncausal(
                network,
                x,
                up_weight,
                weights[f"{up_prefix}.bias"],
                dtype=dtype,
                spatial_padding_mode=spatial_padding_mode,
            )
            x = _pixel_shuffle_3d(
                network,
                x,
                channels=upsampled_channels,
                frames=cur_t,
                height=cur_h,
                width=cur_w,
                stride=stride,
            )
            if _sequence_at(dec_upsample_residual, block_idx, False):
                residual = _ltx2_upsample_residual(
                    network,
                    residual,
                    frames=cur_t,
                    height=cur_h,
                    width=cur_w,
                    stride=stride,
                    upscale_factor=int(_sequence_at(dec_upsample_factor, block_idx, 1)),
                )
                x = network.add_elementwise(x, residual, trt.ElementWiseOperation.SUM).get_output(0)
            cur_t = cur_t * stride[0] - (stride[0] - 1)
            cur_h *= stride[1]
            cur_w *= stride[2]
            prev_ch = upsampled_channels

        resnet_count = _count_resnets(weights, f"decoder.up_blocks.{block_idx}.resnets")
        if block_idx + 1 < len(dec_layers):
            expected = dec_layers[block_idx + 1]
            if resnet_count and resnet_count != expected:
                print(
                    f"[ltx-vae] warning: up block {block_idx} has "
                    f"{resnet_count} resnets, config expected {expected}",
                    file=sys.stderr,
                )
        for res_idx in range(resnet_count):
            res_prefix = f"decoder.up_blocks.{block_idx}.resnets.{res_idx}"
            res_in, res_out = _resnet_channels(weights, res_prefix)
            x = _resnet_block(
                network,
                x,
                weights,
                res_prefix,
                in_channels=res_in,
                out_channels=res_out,
                dtype=dtype,
                spatial_padding_mode=spatial_padding_mode,
            )
            prev_ch = res_out

    if _is_bfloat16_dtype(dtype):
        x = _add_vae_rms_silu(network, x, eps=1e-8)
    else:
        x = _rms_norm_channels(network, x, prev_ch, eps=1e-8)
        x = graph_ops.add_silu(network, x)
    x = _conv3d_noncausal(
        network,
        x,
        weights["decoder.conv_out.conv.weight"],
        weights["decoder.conv_out.conv.bias"],
        dtype=dtype,
        spatial_padding_mode=spatial_padding_mode,
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
    downsample_type: tuple[str, ...] | None = None,
    patch_size: int = 4,
    patch_size_t: int = 1,
    precision: str = "fp16",
    normalize_output: bool = True,
    scaling_factor: float = 1.0,
    spatial_tiling: bool = False,
    tile_sample_min_height: int = 512,
    tile_sample_min_width: int = 512,
    tile_sample_stride_height: int = 448,
    tile_sample_stride_width: int = 448,
    use_torch_conv3d: bool = False,
    verbose: bool = False,
) -> bytes:
    """Build the LTX VAE encoder as a TensorRT plan."""
    global _torch_conv3d_plugin_index, _vae_rms_silu_plugin_index, _weight_refs
    _weight_refs = []
    _torch_conv3d_plugin_index = 0
    _vae_rms_silu_plugin_index = 0
    if precision not in ("bf16", "fp16", "fp32"):
        raise ValueError("LTX VAE raw builder currently supports bf16, fp16, or fp32")
    if sample_frames % patch_size_t != 0:
        raise ValueError("sample_frames must be divisible by patch_size_t")
    if sample_height % patch_size != 0 or sample_width % patch_size != 0:
        raise ValueError("sample dimensions must be divisible by patch_size")

    _ensure_trt()
    _ensure_graph_ops()
    trt_dtype = _trt_dtype(precision)
    dtype = _target_np_dtype(precision)

    logger = get_process_trt_logger(trt, verbose=verbose)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    x = network.add_input(
        "sample",
        trt_dtype,
        (1, in_channels, sample_frames, sample_height, sample_width),
    )

    if spatial_tiling and (
        sample_width > tile_sample_min_width or sample_height > tile_sample_min_height
    ):
        x, cur_t, cur_h, cur_w = _ltx_vae_encoder_tiled(
            network,
            x,
            weights,
            sample_frames=sample_frames,
            sample_height=sample_height,
            sample_width=sample_width,
            in_channels=in_channels,
            latent_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            spatio_temporal_scaling=spatio_temporal_scaling,
            downsample_type=downsample_type,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            dtype=dtype,
            tile_sample_min_height=tile_sample_min_height,
            tile_sample_min_width=tile_sample_min_width,
            tile_sample_stride_height=tile_sample_stride_height,
            tile_sample_stride_width=tile_sample_stride_width,
            use_torch_conv3d=use_torch_conv3d,
            verbose=verbose,
        )
    else:
        x, cur_t, cur_h, cur_w = _ltx_vae_encoder_body(
            network,
            x,
            weights,
            sample_frames=sample_frames,
            sample_height=sample_height,
            sample_width=sample_width,
            in_channels=in_channels,
            latent_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            spatio_temporal_scaling=spatio_temporal_scaling,
            downsample_type=downsample_type,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            dtype=dtype,
            use_torch_conv3d=use_torch_conv3d,
            verbose=verbose,
        )
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
        f"{sample_width}, latent={cur_t}x{cur_h}x{cur_w}"
        f"{', tiled' if spatial_tiling else ''}) ...",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for LTX VAE encoder")
    return bytes(plan)


def _ltx_vae_encoder_body(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    weights: "Mapping[str, np.ndarray]",
    *,
    sample_frames: int,
    sample_height: int,
    sample_width: int,
    in_channels: int,
    latent_channels: int,
    block_out_channels: tuple[int, ...],
    layers_per_block: tuple[int, ...],
    spatio_temporal_scaling: tuple[bool, ...],
    downsample_type: tuple[str, ...] | None,
    patch_size: int,
    patch_size_t: int,
    dtype: np.dtype,
    verbose: bool,
    use_torch_conv3d: bool = False,
) -> tuple[trt.ITensor, int, int, int]:
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
        use_torch_conv3d=use_torch_conv3d,
    )

    block_count = _count_indexed_blocks(weights, "encoder.down_blocks")
    for block_idx in range(block_count):
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
            res_prefix = f"encoder.down_blocks.{block_idx}.resnets.{res_idx}"
            res_in, res_out = _resnet_channels(weights, res_prefix)
            x = _resnet_block(
                network,
                x,
                weights,
                res_prefix,
                in_channels=res_in,
                out_channels=res_out,
                dtype=dtype,
                causal=True,
                use_torch_conv3d=use_torch_conv3d,
            )

        down_prefix = f"encoder.down_blocks.{block_idx}.downsamplers.0.conv"
        if f"{down_prefix}.weight" in weights and _sequence_at(
            spatio_temporal_scaling, block_idx, True
        ):
            kind = _sequence_at(downsample_type, block_idx, "conv")
            if kind == "conv":
                stride = (2, 2, 2)
                x = _conv3d_causal(
                    network,
                    x,
                    weights[f"{down_prefix}.weight"],
                    weights[f"{down_prefix}.bias"],
                    dtype=dtype,
                    stride=stride,
                    use_torch_conv3d=use_torch_conv3d,
                )
            else:
                stride = _downsample_stride(kind)
                x = _ltx2_downsample3d(
                    network,
                    x,
                    weights[f"{down_prefix}.weight"],
                    weights[f"{down_prefix}.bias"],
                    dtype=dtype,
                    stride=stride,
                    use_torch_conv3d=use_torch_conv3d,
                )
            if verbose:
                print(
                    f"[ltx-vae] down block {block_idx} downsample "
                    f"type={kind} stride={stride} shape={tuple(x.shape)}",
                    file=sys.stderr,
                )
            cur_t = (cur_t + stride[0] - 1) // stride[0]
            cur_h //= stride[1]
            cur_w //= stride[2]

        conv_out_prefix = f"encoder.down_blocks.{block_idx}.conv_out"
        if f"{conv_out_prefix}.conv1.conv.weight" in weights:
            res_in, res_out = _resnet_channels(weights, conv_out_prefix)
            x = _resnet_block(
                network,
                x,
                weights,
                conv_out_prefix,
                in_channels=res_in,
                out_channels=res_out,
                dtype=dtype,
                causal=True,
                use_torch_conv3d=use_torch_conv3d,
            )

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
        mid_prefix = f"encoder.mid_block.resnets.{i}"
        res_in, res_out = _resnet_channels(weights, mid_prefix)
        x = _resnet_block(
            network,
            x,
            weights,
            mid_prefix,
            in_channels=res_in,
            out_channels=res_out,
            dtype=dtype,
            causal=True,
            use_torch_conv3d=use_torch_conv3d,
        )

    out_channels = int(weights["encoder.conv_out.conv.weight"].shape[1])
    if use_torch_conv3d:
        x = _add_vae_rms_silu(network, x, eps=1e-8)
    else:
        x = _rms_norm_channels(network, x, out_channels, eps=1e-8)
        x = graph_ops.add_silu(network, x)
    x = _conv3d_causal(
        network,
        x,
        weights["encoder.conv_out.conv.weight"],
        weights["encoder.conv_out.conv.bias"],
        dtype=dtype,
        use_torch_conv3d=use_torch_conv3d,
    )
    x = network.add_slice(
        x,
        (0, 0, 0, 0, 0),
        (1, latent_channels, cur_t, cur_h, cur_w),
        (1, 1, 1, 1, 1),
    ).get_output(0)
    return x, cur_t, cur_h, cur_w


def _weighted_sum(
    network: trt.INetworkDefinition,
    a: trt.ITensor,
    b: trt.ITensor,
    a_weight: float,
    b_weight: float,
) -> trt.ITensor:
    if a_weight == 1.0 and b_weight == 0.0:
        return a
    if a_weight == 0.0 and b_weight == 1.0:
        return b
    a_scale = graph_ops.add_constant(
        network, (1, 1, 1, 1, 1), np.array([a_weight], dtype=np.float32)
    )
    b_scale = graph_ops.add_constant(
        network, (1, 1, 1, 1, 1), np.array([b_weight], dtype=np.float32)
    )
    if a_scale.dtype != a.dtype:
        a_scale = network.add_cast(a_scale, a.dtype).get_output(0)
    if b_scale.dtype != b.dtype:
        b_scale = network.add_cast(b_scale, b.dtype).get_output(0)
    a_scaled = network.add_elementwise(a, a_scale, trt.ElementWiseOperation.PROD).get_output(0)
    b_scaled = network.add_elementwise(b, b_scale, trt.ElementWiseOperation.PROD).get_output(0)
    return network.add_elementwise(a_scaled, b_scaled, trt.ElementWiseOperation.SUM).get_output(0)


def _ltx_vae_blend_v(
    network: trt.INetworkDefinition,
    above: trt.ITensor,
    current: trt.ITensor,
    blend_extent: int,
) -> trt.ITensor:
    out = current
    blend_extent = min(int(above.shape[3]), int(current.shape[3]), blend_extent)
    for y in range(blend_extent):
        width = int(out.shape[4])
        above_slice = network.add_slice(
            above,
            (0, 0, 0, int(above.shape[3]) - blend_extent + y, 0),
            (1, int(above.shape[1]), int(above.shape[2]), 1, width),
            (1, 1, 1, 1, 1),
        ).get_output(0)
        cur_slice = network.add_slice(
            out,
            (0, 0, 0, y, 0),
            (1, int(out.shape[1]), int(out.shape[2]), 1, width),
            (1, 1, 1, 1, 1),
        ).get_output(0)
        alpha = float(y) / float(blend_extent)
        blended = _weighted_sum(network, above_slice, cur_slice, 1.0 - alpha, alpha)
        if y == 0:
            bottom_h = int(out.shape[3]) - 1
            parts = [blended]
            if bottom_h > 0:
                parts.append(
                    network.add_slice(
                        out,
                        (0, 0, 0, 1, 0),
                        (1, int(out.shape[1]), int(out.shape[2]), bottom_h, int(out.shape[4])),
                        (1, 1, 1, 1, 1),
                    ).get_output(0)
                )
            concat = network.add_concatenation(parts)
            concat.axis = 3
            out = concat.get_output(0)
        else:
            top = network.add_slice(
                out,
                (0, 0, 0, 0, 0),
                (1, int(out.shape[1]), int(out.shape[2]), y, int(out.shape[4])),
                (1, 1, 1, 1, 1),
            ).get_output(0)
            bottom_h = int(out.shape[3]) - y - 1
            parts = [top, blended]
            if bottom_h > 0:
                parts.append(
                    network.add_slice(
                        out,
                        (0, 0, 0, y + 1, 0),
                        (1, int(out.shape[1]), int(out.shape[2]), bottom_h, int(out.shape[4])),
                        (1, 1, 1, 1, 1),
                    ).get_output(0)
                )
            concat = network.add_concatenation(parts)
            concat.axis = 3
            out = concat.get_output(0)
    return out


def _ltx_vae_blend_h(
    network: trt.INetworkDefinition,
    left: trt.ITensor,
    current: trt.ITensor,
    blend_extent: int,
) -> trt.ITensor:
    out = current
    blend_extent = min(int(left.shape[4]), int(current.shape[4]), blend_extent)
    for x in range(blend_extent):
        height = int(out.shape[3])
        left_slice = network.add_slice(
            left,
            (0, 0, 0, 0, int(left.shape[4]) - blend_extent + x),
            (1, int(left.shape[1]), int(left.shape[2]), height, 1),
            (1, 1, 1, 1, 1),
        ).get_output(0)
        cur_slice = network.add_slice(
            out,
            (0, 0, 0, 0, x),
            (1, int(out.shape[1]), int(out.shape[2]), height, 1),
            (1, 1, 1, 1, 1),
        ).get_output(0)
        alpha = float(x) / float(blend_extent)
        blended = _weighted_sum(network, left_slice, cur_slice, 1.0 - alpha, alpha)
        if x == 0:
            right_w = int(out.shape[4]) - 1
            parts = [blended]
            if right_w > 0:
                parts.append(
                    network.add_slice(
                        out,
                        (0, 0, 0, 0, 1),
                        (1, int(out.shape[1]), int(out.shape[2]), int(out.shape[3]), right_w),
                        (1, 1, 1, 1, 1),
                    ).get_output(0)
                )
            concat = network.add_concatenation(parts)
            concat.axis = 4
            out = concat.get_output(0)
        else:
            left_part = network.add_slice(
                out,
                (0, 0, 0, 0, 0),
                (1, int(out.shape[1]), int(out.shape[2]), int(out.shape[3]), x),
                (1, 1, 1, 1, 1),
            ).get_output(0)
            right_w = int(out.shape[4]) - x - 1
            parts = [left_part, blended]
            if right_w > 0:
                parts.append(
                    network.add_slice(
                        out,
                        (0, 0, 0, 0, x + 1),
                        (1, int(out.shape[1]), int(out.shape[2]), int(out.shape[3]), right_w),
                        (1, 1, 1, 1, 1),
                    ).get_output(0)
                )
            concat = network.add_concatenation(parts)
            concat.axis = 4
            out = concat.get_output(0)
    return out


def _ltx_vae_encoder_tiled(
    network: trt.INetworkDefinition,
    sample: trt.ITensor,
    weights: "Mapping[str, np.ndarray]",
    *,
    sample_frames: int,
    sample_height: int,
    sample_width: int,
    in_channels: int,
    latent_channels: int,
    block_out_channels: tuple[int, ...],
    layers_per_block: tuple[int, ...],
    spatio_temporal_scaling: tuple[bool, ...],
    downsample_type: tuple[str, ...] | None,
    patch_size: int,
    patch_size_t: int,
    dtype: np.dtype,
    tile_sample_min_height: int,
    tile_sample_min_width: int,
    tile_sample_stride_height: int,
    tile_sample_stride_width: int,
    verbose: bool,
    use_torch_conv3d: bool = False,
) -> tuple[trt.ITensor, int, int, int]:
    spatial_compression = _encoder_spatial_compression(
        weights,
        patch_size=patch_size,
        spatio_temporal_scaling=spatio_temporal_scaling,
        downsample_type=downsample_type,
    )
    latent_height = sample_height // spatial_compression
    latent_width = sample_width // spatial_compression
    tile_latent_min_height = tile_sample_min_height // spatial_compression
    tile_latent_min_width = tile_sample_min_width // spatial_compression
    tile_latent_stride_height = tile_sample_stride_height // spatial_compression
    tile_latent_stride_width = tile_sample_stride_width // spatial_compression
    blend_height = tile_latent_min_height - tile_latent_stride_height
    blend_width = tile_latent_min_width - tile_latent_stride_width

    rows: list[list[trt.ITensor]] = []
    result_rows: list[trt.ITensor] = []
    cur_t = sample_frames
    for i in range(0, sample_height, tile_sample_stride_height):
        row: list[trt.ITensor] = []
        for j in range(0, sample_width, tile_sample_stride_width):
            tile_h = min(tile_sample_min_height, sample_height - i)
            tile_w = min(tile_sample_min_width, sample_width - j)
            tile = network.add_slice(
                sample,
                (0, 0, 0, i, j),
                (1, in_channels, sample_frames, tile_h, tile_w),
                (1, 1, 1, 1, 1),
            ).get_output(0)
            encoded, enc_t, _enc_h, _enc_w = _ltx_vae_encoder_body(
                network,
                tile,
                weights,
                sample_frames=sample_frames,
                sample_height=tile_h,
                sample_width=tile_w,
                in_channels=in_channels,
                latent_channels=latent_channels,
                block_out_channels=block_out_channels,
                layers_per_block=layers_per_block,
                spatio_temporal_scaling=spatio_temporal_scaling,
                downsample_type=downsample_type,
                patch_size=patch_size,
                patch_size_t=patch_size_t,
                dtype=dtype,
                use_torch_conv3d=use_torch_conv3d,
                verbose=verbose,
            )
            cur_t = enc_t
            row.append(encoded)
        rows.append(row)

    for i, row in enumerate(rows):
        result_row: list[trt.ITensor] = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = _ltx_vae_blend_v(network, rows[i - 1][j], tile, blend_height)
            if j > 0:
                tile = _ltx_vae_blend_h(network, row[j - 1], tile, blend_width)
            row[j] = tile
            take_h = min(tile_latent_stride_height, int(tile.shape[3]))
            take_w = min(tile_latent_stride_width, int(tile.shape[4]))
            result_row.append(
                network.add_slice(
                    tile,
                    (0, 0, 0, 0, 0),
                    (1, latent_channels, int(tile.shape[2]), take_h, take_w),
                    (1, 1, 1, 1, 1),
                ).get_output(0)
            )
        row_concat = network.add_concatenation(result_row)
        row_concat.axis = 4
        result_rows.append(row_concat.get_output(0))

    out_concat = network.add_concatenation(result_rows)
    out_concat.axis = 3
    out = network.add_slice(
        out_concat.get_output(0),
        (0, 0, 0, 0, 0),
        (1, latent_channels, cur_t, latent_height, latent_width),
        (1, 1, 1, 1, 1),
    ).get_output(0)
    return out, cur_t, latent_height, latent_width


def _encoder_spatial_compression(
    weights: "Mapping[str, np.ndarray]",
    *,
    patch_size: int,
    spatio_temporal_scaling: tuple[bool, ...],
    downsample_type: tuple[str, ...] | None,
) -> int:
    compression = int(patch_size)
    block_count = _count_indexed_blocks(weights, "encoder.down_blocks")
    for block_idx in range(block_count):
        down_prefix = f"encoder.down_blocks.{block_idx}.downsamplers.0.conv"
        if f"{down_prefix}.weight" not in weights:
            continue
        if not _sequence_at(spatio_temporal_scaling, block_idx, True):
            continue
        kind = _sequence_at(downsample_type, block_idx, "conv")
        stride = (2, 2, 2) if kind == "conv" else _downsample_stride(kind)
        compression *= stride[1]
    return compression


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
        weights[f"{prefix}.conv_shortcut.conv.weight"] = f(f"{prefix}.conv_shortcut.conv.weight")
        weights[f"{prefix}.conv_shortcut.conv.bias"] = f(f"{prefix}.conv_shortcut.conv.bias")


def _count_resnets(weights: "Mapping[str, np.ndarray]", prefix: str) -> int:
    idx = 0
    while f"{prefix}.{idx}.conv1.conv.weight" in weights:
        idx += 1
    return idx


def _count_indexed_blocks(weights: "Mapping[str, np.ndarray]", prefix: str) -> int:
    idx = 0
    while (
        _count_resnets(weights, f"{prefix}.{idx}.resnets")
        or f"{prefix}.{idx}.conv_in.conv1.conv.weight" in weights
        or f"{prefix}.{idx}.conv_out.conv1.conv.weight" in weights
        or f"{prefix}.{idx}.downsamplers.0.conv.weight" in weights
        or f"{prefix}.{idx}.upsamplers.0.conv.conv.weight" in weights
    ):
        idx += 1
    return idx


def _resnet_channels(weights: "Mapping[str, np.ndarray]", prefix: str) -> tuple[int, int]:
    conv1 = weights[f"{prefix}.conv1.conv.weight"]
    return int(conv1.shape[1]), int(conv1.shape[0])


def _sequence_at(seq, idx: int, default):
    if seq is None or idx >= len(seq):
        return default
    return seq[idx]


def _downsample_stride(kind: str) -> tuple[int, int, int]:
    kind = kind.lower()
    if kind == "spatial":
        return (1, 2, 2)
    if kind == "temporal":
        return (2, 1, 1)
    if kind == "spatiotemporal":
        return (2, 2, 2)
    raise ValueError(f"Unsupported LTX2 VAE downsample type: {kind!r}")


def _upsample_stride(kind: str) -> tuple[int, int, int]:
    kind = kind.lower()
    if kind == "spatial":
        return (1, 2, 2)
    if kind == "temporal":
        return (2, 1, 1)
    if kind == "spatiotemporal":
        return (2, 2, 2)
    raise ValueError(f"Unsupported LTX2 VAE upsample type: {kind!r}")


def _pixel_shuffle_output_channels(weight: np.ndarray, stride: tuple[int, int, int]) -> int:
    factor = stride[0] * stride[1] * stride[2]
    if int(weight.shape[0]) % factor != 0:
        raise ValueError(
            f"upsampler output channels {weight.shape[0]} are not divisible by {factor}"
        )
    return int(weight.shape[0]) // factor


def _conv3d_noncausal(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    *,
    dtype: np.dtype,
    spatial_padding_mode: str = "zeros",
) -> trt.ITensor:
    b, _c, t, _h, _w = inp.shape
    out_channels, _in_channels, kt, kh, kw = weight.shape
    x = inp
    if kt > 1:
        left = network.add_slice(
            inp,
            (0, 0, 0, 0, 0),
            (b, inp.shape[1], 1, inp.shape[3], inp.shape[4]),
            (1, 1, 1, 1, 1),
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

    pad_h = kh // 2
    pad_w = kw // 2
    spatial_padding_mode = spatial_padding_mode.lower()
    if spatial_padding_mode == "reflect":
        x = _reflect_spatial_pad_5d(network, x, pad_h, pad_w)
        conv_spatial_padding = (0, 0)
    elif spatial_padding_mode == "zeros":
        conv_spatial_padding = (pad_h, pad_w)
    else:
        raise ValueError(
            "LTX VAE Conv3D supports zeros or reflect spatial padding, "
            f"got {spatial_padding_mode!r}"
        )

    if _is_bfloat16_dtype(dtype):
        return _add_torch_conv3d(
            network,
            x,
            weight,
            bias,
            stride=(1, 1, 1),
            padding=(0, *conv_spatial_padding),
        )

    conv = network.add_convolution_nd(
        x,
        num_output_maps=out_channels,
        kernel_shape=(kt, kh, kw),
        kernel=_trt_weights(weight, dtype),
        bias=_trt_weights(bias, dtype) if bias is not None else trt.Weights(),
    )
    conv.stride_nd = (1, 1, 1)
    conv.padding_nd = (0, *conv_spatial_padding)
    return conv.get_output(0)


def _slice_single_index(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    axis: int,
    index: int,
) -> trt.ITensor:
    start = [0] * len(inp.shape)
    size = [int(dim) for dim in inp.shape]
    stride = [1] * len(inp.shape)
    start[axis] = index
    size[axis] = 1
    return network.add_slice(
        inp,
        tuple(start),
        tuple(size),
        tuple(stride),
    ).get_output(0)


def _reflect_pad_axis(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    axis: int,
    pad: int,
) -> trt.ITensor:
    if pad <= 0:
        return inp
    axis_len = int(inp.shape[axis])
    if pad >= axis_len:
        raise ValueError(
            f"reflect padding {pad} on axis {axis} requires input size > padding, got {axis_len}"
        )
    left = [
        _slice_single_index(network, inp, axis=axis, index=index) for index in range(pad, 0, -1)
    ]
    right = [
        _slice_single_index(network, inp, axis=axis, index=index)
        for index in range(axis_len - 2, axis_len - pad - 2, -1)
    ]
    concat = network.add_concatenation([*left, inp, *right])
    concat.axis = axis
    return concat.get_output(0)


def _reflect_spatial_pad_5d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    pad_h: int,
    pad_w: int,
) -> trt.ITensor:
    x = _reflect_pad_axis(network, inp, axis=3, pad=pad_h)
    return _reflect_pad_axis(network, x, axis=4, pad=pad_w)


def _add_torch_conv3d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    *,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int] = (1, 1, 1),
    groups: int = 1,
) -> trt.ITensor:
    global _torch_conv3d_plugin_index
    trt_module = _ensure_trt()
    from ...stage1_dit_builder import (
        _get_sana_wm_plugin_creator,
    )

    creator = _get_sana_wm_plugin_creator(trt_module, "SanaWmTorchConv3d")
    if creator is None:
        raise RuntimeError("SanaWmTorchConv3d is required for exact BF16 VAE encoder builds")
    if inp.dtype != trt_module.bfloat16:
        raise TypeError("SanaWmTorchConv3d requires a BF16 input tensor")

    out_channels, in_channels_per_group, kernel_t, kernel_h, kernel_w = (
        int(dim) for dim in weight.shape
    )
    in_channels = in_channels_per_group * groups
    input_t, input_h, input_w = (int(inp.shape[i]) for i in (2, 3, 4))

    def output_size(
        input_size: int, kernel_size: int, stride_size: int, pad: int, dilate: int
    ) -> int:
        return (input_size + 2 * pad - dilate * (kernel_size - 1) - 1) // stride_size + 1

    output_t = output_size(input_t, kernel_t, stride[0], padding[0], dilation[0])
    output_h = output_size(input_h, kernel_h, stride[1], padding[1], dilation[1])
    output_w = output_size(input_w, kernel_w, stride[2], padding[2], dilation[2])

    int_values: list[np.ndarray] = []

    def int_field(name: str, value: int) -> trt.PluginField:
        data = np.ascontiguousarray([value], dtype=np.int32)
        int_values.append(data)
        return trt_module.PluginField(name, data, trt_module.PluginFieldType.INT32)

    weight_field = np.ascontiguousarray(weight, dtype=np.float32)
    fields = [
        int_field("out_channels", out_channels),
        int_field("in_channels", in_channels),
        int_field("kernel_t", kernel_t),
        int_field("kernel_h", kernel_h),
        int_field("kernel_w", kernel_w),
        int_field("stride_t", stride[0]),
        int_field("stride_h", stride[1]),
        int_field("stride_w", stride[2]),
        int_field("pad_t", padding[0]),
        int_field("pad_h", padding[1]),
        int_field("pad_w", padding[2]),
        int_field("dilation_t", dilation[0]),
        int_field("dilation_h", dilation[1]),
        int_field("dilation_w", dilation[2]),
        int_field("groups", groups),
        int_field("output_t", output_t),
        int_field("output_h", output_h),
        int_field("output_w", output_w),
        trt_module.PluginField("weight", weight_field, trt_module.PluginFieldType.FLOAT32),
    ]
    if bias is not None:
        bias_field = np.ascontiguousarray(bias, dtype=np.float32)
        fields.append(
            trt_module.PluginField("bias", bias_field, trt_module.PluginFieldType.FLOAT32)
        )
    collection = trt_module.PluginFieldCollection(fields)
    plugin = creator.create_plugin(
        f"sana_wm_vae_torch_conv3d_{_torch_conv3d_plugin_index}", collection
    )
    _torch_conv3d_plugin_index += 1
    if plugin is None:
        raise RuntimeError("failed to create SanaWmTorchConv3d plugin")
    layer = network.add_plugin_v2([inp], plugin)
    if layer is None:
        raise RuntimeError("failed to add SanaWmTorchConv3d layer")
    return layer.get_output(0)


def _add_vae_rms_silu(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    eps: float,
) -> trt.ITensor:
    global _vae_rms_silu_plugin_index
    trt_module = _ensure_trt()
    from ...stage1_dit_builder import (
        _get_sana_wm_plugin_creator,
    )

    creator = _get_sana_wm_plugin_creator(trt_module, "SanaWmVaeRmsSilu")
    if creator is None:
        raise RuntimeError("SanaWmVaeRmsSilu is required for exact BF16 VAE encoder builds")
    eps_field = np.ascontiguousarray([eps], dtype=np.float32)
    plugin = creator.create_plugin(
        f"sana_wm_vae_rms_silu_{_vae_rms_silu_plugin_index}",
        trt_module.PluginFieldCollection(
            [trt_module.PluginField("eps", eps_field, trt_module.PluginFieldType.FLOAT32)]
        ),
    )
    _vae_rms_silu_plugin_index += 1
    if plugin is None:
        raise RuntimeError("failed to create SanaWmVaeRmsSilu plugin")
    layer = network.add_plugin_v2([inp], plugin)
    if layer is None:
        raise RuntimeError("failed to add SanaWmVaeRmsSilu layer")
    return layer.get_output(0)


def _add_vae_denormalize(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: "Mapping[str, np.ndarray]",
    channels: int,
    *,
    scaling_factor: float,
) -> trt.ITensor:
    trt_module = _ensure_trt()
    from ...stage1_dit_builder import (
        _get_sana_wm_plugin_creator,
    )

    creator = _get_sana_wm_plugin_creator(trt_module, "SanaWmVaeDenormalize")
    if creator is None:
        raise RuntimeError("SanaWmVaeDenormalize is required for exact BF16 VAE decoder builds")
    mean, std = _latents_stats(weights, channels)
    channels_field = np.ascontiguousarray([channels], dtype=np.int32)
    scaling_field = np.ascontiguousarray([scaling_factor], dtype=np.float32)
    mean_field = np.ascontiguousarray(mean, dtype=np.float32)
    std_field = np.ascontiguousarray(std, dtype=np.float32)
    fields = [
        trt_module.PluginField("channels", channels_field, trt_module.PluginFieldType.INT32),
        trt_module.PluginField("scaling_factor", scaling_field, trt_module.PluginFieldType.FLOAT32),
        trt_module.PluginField("mean", mean_field, trt_module.PluginFieldType.FLOAT32),
        trt_module.PluginField("std", std_field, trt_module.PluginFieldType.FLOAT32),
    ]
    plugin = creator.create_plugin(
        "sana_wm_vae_denormalize", trt_module.PluginFieldCollection(fields)
    )
    if plugin is None:
        raise RuntimeError("failed to create SanaWmVaeDenormalize plugin")
    layer = network.add_plugin_v2([inp], plugin)
    if layer is None:
        raise RuntimeError("failed to add SanaWmVaeDenormalize layer")
    return layer.get_output(0)


def _add_vae_layer_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    channels: int,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float,
) -> trt.ITensor:
    global _vae_layer_norm_plugin_index
    trt_module = _ensure_trt()
    from ...stage1_dit_builder import (
        _get_sana_wm_plugin_creator,
    )

    creator = _get_sana_wm_plugin_creator(trt_module, "SanaWmVaeLayerNorm")
    if creator is None:
        raise RuntimeError("SanaWmVaeLayerNorm is required for exact BF16 VAE decoder builds")
    channels_field = np.ascontiguousarray([channels], dtype=np.int32)
    eps_field = np.ascontiguousarray([eps], dtype=np.float32)
    weight_field = np.ascontiguousarray(weight, dtype=np.float32)
    bias_field = np.ascontiguousarray(bias, dtype=np.float32)
    fields = [
        trt_module.PluginField("channels", channels_field, trt_module.PluginFieldType.INT32),
        trt_module.PluginField("eps", eps_field, trt_module.PluginFieldType.FLOAT32),
        trt_module.PluginField("weight", weight_field, trt_module.PluginFieldType.FLOAT32),
        trt_module.PluginField("bias", bias_field, trt_module.PluginFieldType.FLOAT32),
    ]
    plugin = creator.create_plugin(
        f"sana_wm_vae_layer_norm_{_vae_layer_norm_plugin_index}",
        trt_module.PluginFieldCollection(fields),
    )
    _vae_layer_norm_plugin_index += 1
    if plugin is None:
        raise RuntimeError("failed to create SanaWmVaeLayerNorm plugin")
    layer = network.add_plugin_v2([inp], plugin)
    if layer is None:
        raise RuntimeError("failed to add SanaWmVaeLayerNorm layer")
    return layer.get_output(0)


def _conv3d_causal(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    *,
    dtype: np.dtype,
    stride: tuple[int, int, int] = (1, 1, 1),
    use_torch_conv3d: bool = False,
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

    if use_torch_conv3d:
        return _add_torch_conv3d(
            network,
            x,
            weight,
            bias,
            stride=stride,
            padding=(0, kh // 2, kw // 2),
        )

    conv = network.add_convolution_nd(
        x,
        num_output_maps=out_channels,
        kernel_shape=(kt, kh, kw),
        kernel=_trt_weights(weight, dtype),
        bias=_trt_weights(bias, dtype) if bias is not None else trt.Weights(),
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
    mean = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    eps_t = graph_ops.add_constant(network, (1, 1, 1, 1, 1), np.array([eps], dtype=np.float32))
    denom = network.add_elementwise(mean.get_output(0), eps_t, trt.ElementWiseOperation.SUM)
    sqrt = network.add_unary(denom.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt.get_output(0), trt.UnaryOperation.RECIP)
    out = network.add_elementwise(x, recip.get_output(0), trt.ElementWiseOperation.PROD).get_output(
        0
    )
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
    var = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    eps_t = graph_ops.add_constant(network, (1, 1, 1, 1, 1), np.array([eps], dtype=np.float32))
    denom = network.add_elementwise(var.get_output(0), eps_t, trt.ElementWiseOperation.SUM)
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
    spatial_padding_mode: str = "zeros",
    use_torch_conv3d: bool = False,
) -> trt.ITensor:
    def conv3d(
        network: trt.INetworkDefinition,
        inp: trt.ITensor,
        weight: np.ndarray,
        bias: np.ndarray | None,
        *,
        dtype: np.dtype,
    ) -> trt.ITensor:
        if causal:
            return _conv3d_causal(
                network,
                inp,
                weight,
                bias,
                dtype=dtype,
                use_torch_conv3d=use_torch_conv3d,
            )
        return _conv3d_noncausal(
            network,
            inp,
            weight,
            bias,
            dtype=dtype,
            spatial_padding_mode=spatial_padding_mode,
        )

    exact_bf16 = _is_bfloat16_dtype(dtype)
    if exact_bf16:
        h = _add_vae_rms_silu(network, inp, eps=1e-8)
    else:
        h = _rms_norm_channels(network, inp, in_channels, eps=1e-8)
        h = graph_ops.add_silu(network, h)
    h = conv3d(
        network,
        h,
        weights[f"{prefix}.conv1.conv.weight"],
        weights[f"{prefix}.conv1.conv.bias"],
        dtype=dtype,
    )
    if exact_bf16:
        h = _add_vae_rms_silu(network, h, eps=1e-8)
    else:
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
        if exact_bf16:
            shortcut = _add_vae_layer_norm(
                network,
                shortcut,
                channels=in_channels,
                weight=weights[f"{prefix}.norm3.weight"],
                bias=weights[f"{prefix}.norm3.bias"],
                eps=1e-6,
            )
        else:
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

    return network.add_elementwise(h, shortcut, trt.ElementWiseOperation.SUM).get_output(0)


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


def _prepend_temporal_stride_frames(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    count: int,
) -> trt.ITensor:
    if count <= 0:
        return inp
    b, c, _t, h, w = inp.shape
    prefix = network.add_slice(
        inp,
        (0, 0, 0, 0, 0),
        (b, c, count, h, w),
        (1, 1, 1, 1, 1),
    ).get_output(0)
    concat = network.add_concatenation([prefix, inp])
    concat.axis = 2
    return concat.get_output(0)


def _pixel_unshuffle_3d(
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
    if frames % st or height % sh or width % sw:
        raise ValueError("LTX2 VAE pixel unshuffle dimensions must be divisible by stride")
    r1 = network.add_shuffle(inp)
    r1.reshape_dims = (
        1,
        channels,
        frames // st,
        st,
        height // sh,
        sh,
        width // sw,
        sw,
    )
    r2 = network.add_shuffle(r1.get_output(0))
    r2.first_transpose = trt.Permutation([0, 1, 3, 5, 7, 2, 4, 6])
    r2.reshape_dims = (
        1,
        channels * st * sh * sw,
        frames // st,
        height // sh,
        width // sw,
    )
    return r2.get_output(0)


def _channel_group_mean(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    out_channels: int,
    group_size: int,
    frames: int,
    height: int,
    width: int,
) -> trt.ITensor:
    if group_size == 1:
        return inp
    r = network.add_shuffle(inp)
    r.reshape_dims = (1, out_channels, group_size, frames, height, width)
    mean = network.add_reduce(r.get_output(0), trt.ReduceOperation.AVG, 1 << 2, keep_dims=False)
    return mean.get_output(0)


def _ltx2_downsample3d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    dtype: np.dtype,
    stride: tuple[int, int, int],
    use_torch_conv3d: bool = False,
) -> trt.ITensor:
    st, sh, sw = stride
    padded = _prepend_temporal_stride_frames(network, inp, st - 1)
    padded_t = int(padded.shape[2])
    padded_h = int(padded.shape[3])
    padded_w = int(padded.shape[4])

    hidden = _conv3d_causal(
        network,
        padded,
        weight,
        bias,
        dtype=dtype,
        use_torch_conv3d=use_torch_conv3d,
    )
    conv_channels = int(weight.shape[0])
    out_channels = conv_channels * st * sh * sw
    hidden = _pixel_unshuffle_3d(
        network,
        hidden,
        channels=conv_channels,
        frames=padded_t,
        height=padded_h,
        width=padded_w,
        stride=stride,
    )

    residual_channels = int(padded.shape[1])
    residual = _pixel_unshuffle_3d(
        network,
        padded,
        channels=residual_channels,
        frames=padded_t,
        height=padded_h,
        width=padded_w,
        stride=stride,
    )
    residual_total_channels = residual_channels * st * sh * sw
    if residual_total_channels % out_channels != 0:
        raise ValueError("LTX2 VAE downsample residual channels do not align")
    residual = _channel_group_mean(
        network,
        residual,
        out_channels=out_channels,
        group_size=residual_total_channels // out_channels,
        frames=padded_t // st,
        height=padded_h // sh,
        width=padded_w // sw,
    )
    return network.add_elementwise(hidden, residual, trt.ElementWiseOperation.SUM).get_output(0)


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


def _repeat_channels(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    repeats: int,
) -> trt.ITensor:
    if repeats <= 1:
        return inp
    concat = network.add_concatenation([inp for _ in range(repeats)])
    concat.axis = 1
    return concat.get_output(0)


def _ltx2_upsample_residual(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    frames: int,
    height: int,
    width: int,
    stride: tuple[int, int, int],
    upscale_factor: int,
) -> trt.ITensor:
    st, sh, sw = stride
    factor = st * sh * sw
    if factor % upscale_factor != 0:
        raise ValueError("LTX2 VAE upsample residual upscale factor is invalid")
    in_channels = int(inp.shape[1])
    if in_channels % factor != 0:
        raise ValueError("LTX2 VAE upsample residual channels do not align")
    residual = _pixel_shuffle_3d(
        network,
        inp,
        channels=in_channels // factor,
        frames=frames,
        height=height,
        width=width,
        stride=stride,
    )
    return _repeat_channels(network, residual, factor // upscale_factor)


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
    x = inp
    mean, std = _latents_stats(weights, channels)
    mean_t = _add_constant_for_trt_dtype(
        network, (1, channels, 1, 1, 1), mean.reshape(1, channels, 1, 1, 1), out_dtype
    )
    scale_t = _add_constant_for_trt_dtype(
        network,
        (1, 1, 1, 1, 1),
        np.array([scaling_factor], dtype=np.float32),
        out_dtype,
    )
    std_t = _add_constant_for_trt_dtype(
        network, (1, channels, 1, 1, 1), std.reshape(1, channels, 1, 1, 1), out_dtype
    )
    centered = network.add_elementwise(x, mean_t, trt.ElementWiseOperation.SUB)
    scaled = network.add_elementwise(centered.get_output(0), scale_t, trt.ElementWiseOperation.PROD)
    out = network.add_elementwise(scaled.get_output(0), std_t, trt.ElementWiseOperation.DIV)
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
    trt_module = _ensure_trt()
    if out_dtype == trt_module.bfloat16:
        return _add_vae_denormalize(
            network,
            inp,
            weights,
            channels,
            scaling_factor=scaling_factor,
        )
    x = inp
    mean, std = _latents_stats(weights, channels)
    scale = (std / float(scaling_factor or 1.0)).reshape(1, channels, 1, 1, 1)
    scale_t = _add_constant_for_trt_dtype(network, (1, channels, 1, 1, 1), scale, out_dtype)
    mean_t = _add_constant_for_trt_dtype(
        network, (1, channels, 1, 1, 1), mean.reshape(1, channels, 1, 1, 1), out_dtype
    )
    scaled = network.add_elementwise(x, scale_t, trt.ElementWiseOperation.PROD)
    out = network.add_elementwise(scaled.get_output(0), mean_t, trt.ElementWiseOperation.SUM)
    return _cast_back(network, out.get_output(0), out_dtype)
