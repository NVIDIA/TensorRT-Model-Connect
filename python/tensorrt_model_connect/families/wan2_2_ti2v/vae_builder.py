# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-profile TensorRT builder for the Wan2.2 TI2V-5B VAE decoder.

The official decoder is an FP32, causal 3D VAE.  Its Python implementation
decodes one latent frame at a time while carrying mutable temporal caches.  A
fixed-shape ONNX export records those cache transitions in the graph, leaving a
single TensorRT runtime contract:

``latents [1, 48, 31, 44, 80] -> video [1, 3, 121, 704, 1280]``.

PyTorch and Diffusers are build-time dependencies only.  The serialized plan
contains the complete decoder and requires neither at runtime.
"""

from __future__ import annotations

import ctypes
import gc
import tempfile
from dataclasses import dataclass
from pathlib import Path

from tensorrt_model_connect import trt_compat

from .checkpoint_mapper import VAE22_CONFIG


@dataclass(frozen=True)
class Wan22VaeDecoderProfile:
    """Static latent geometry for one Wan2.2 VAE decoder plan."""

    latent_frames: int
    latent_height: int
    latent_width: int
    batch_size: int = 1
    latent_channels: int = 48

    def __post_init__(self) -> None:
        for name, value in (
            ("batch_size", self.batch_size),
            ("latent_channels", self.latent_channels),
            ("latent_frames", self.latent_frames),
            ("latent_height", self.latent_height),
            ("latent_width", self.latent_width),
        ):
            if value <= 0:
                raise ValueError(f"Wan2.2 VAE {name} must be positive, got {value}")
        if self.batch_size != 1:
            raise ValueError("Wan2.2 TI2V-5B VAE support requires batch size 1")
        if self.latent_channels != 48:
            raise ValueError("Wan2.2 TI2V-5B VAE support requires 48 latent channels")

    @property
    def input_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.batch_size,
            self.latent_channels,
            self.latent_frames,
            self.latent_height,
            self.latent_width,
        )

    @property
    def output_shape(self) -> tuple[int, int, int, int, int]:
        # The first latent emits one frame.  Each later latent emits four.
        frames = 1 + (self.latent_frames - 1) * 4
        return (
            self.batch_size,
            3,
            frames,
            self.latent_height * 16,
            self.latent_width * 16,
        )


OFFICIAL_VAE_DECODER_PROFILE = Wan22VaeDecoderProfile(
    latent_frames=31,
    latent_height=44,
    latent_width=80,
)

# Keep the official-profile build bounded below Jetson Thor's 128 GB capacity.
# TensorRT 11 requested the configured limit plus about 0.41 GiB when a causal
# convolution was fused with its normalization/padding prelude, at both 64 and
# 72 GiB limits.  The semantic pre-convolution barriers below isolate that
# compiler region instead of chasing the pool limit upward.
OFFICIAL_VAE_WORKSPACE_GIB = 64


@dataclass(frozen=True)
class VaeBarrierInsertion:
    """One parsed-network edge replaced by an FP32 CUDA barrier."""

    target_tensor: str
    barrier_layer: str
    barrier_output: str
    tensor_shape: tuple[int, ...]
    consumer_count: int


@dataclass(frozen=True)
class VaeBarrierSpec:
    """Auditable template for one compiler-fusion boundary."""

    target_template: str
    label_template: str
    reason: str


@dataclass(frozen=True)
class VaeRmsNormReplacement:
    """One exported RMS-normalization subgraph replaced by a CUDA plugin."""

    target_tensor: str
    source_tensor: str
    gamma_tensor: str
    plugin_layer: str
    plugin_output: str
    tensor_shape: tuple[int, ...]
    consumer_count: int


@dataclass(frozen=True)
class VaeConv3dReplacement:
    """One final-up-block Conv3d replaced by the bounded-workspace cuDNN plugin."""

    logical_scope: str
    frame: int
    target_tensor: str
    source_tensor: str
    weight_initializer: str
    bias_initializer: str
    plugin_layer: str
    plugin_output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    consumer_count: int


_VAE_STAGE_BARRIER_MANIFEST = (
    VaeBarrierSpec(
        target_template="/decoder/mid_block/resnets.0_{frame}/Add_output_0",
        label_template="pre_mid_attention_frame_{frame}",
        reason="isolate each unrolled mid-block attention input",
    ),
    VaeBarrierSpec(
        target_template="/decoder/mid_block/attentions.0_{frame}/Add_1_output_0",
        label_template="post_mid_attention_frame_{frame}",
        reason="isolate each unrolled mid-block attention output",
    ),
    VaeBarrierSpec(
        target_template="/decoder/mid_block/resnets.1_{frame}/Add_output_0",
        label_template="post_mid_resnet_1_frame_{frame}",
        reason="isolate the post-attention resnet from the first decoder up block",
    ),
    VaeBarrierSpec(
        target_template="/decoder/up_blocks.0/resnets.0_{frame}/Add_output_0",
        label_template="post_up_block_0_resnet_0_frame_{frame}",
        reason="bound decoder up block 0 resnet 0 as one compiler region",
    ),
    VaeBarrierSpec(
        target_template="/decoder/up_blocks.0/resnets.1_{frame}/Add_output_0",
        label_template="post_up_block_0_resnet_1_frame_{frame}",
        reason="bound decoder up block 0 resnet 1 as one compiler region",
    ),
    VaeBarrierSpec(
        target_template="/decoder/up_blocks.0/resnets.2_{frame}/Add_output_0",
        label_template="post_up_block_0_resnet_2_frame_{frame}",
        reason="separate decoder up block 0 resnets from its upsampler",
    ),
    VaeBarrierSpec(
        target_template="/decoder/up_blocks.0_{frame}/Add_output_0",
        label_template="post_up_block_0_upsampler_frame_{frame}",
        reason="separate decoder up block 0 upsampler from up block 1",
    ),
    VaeBarrierSpec(
        target_template="/decoder/up_blocks.1/resnets.0_{frame}/Add_output_0",
        label_template="post_up_block_1_resnet_0_frame_{frame}",
        reason="bound decoder up block 1 resnet 0 as one compiler region",
    ),
    VaeBarrierSpec(
        target_template="/decoder/up_blocks.1/resnets.1_{frame}/Add_output_0",
        label_template="post_up_block_1_resnet_1_frame_{frame}",
        reason="bound decoder up block 1 resnet 1 as one compiler region",
    ),
    VaeBarrierSpec(
        target_template="/decoder/up_blocks.1/resnets.2_{frame}/Add_output_0",
        label_template="post_up_block_1_resnet_2_frame_{frame}",
        reason="separate decoder up block 1 resnets from its upsampler",
    ),
    VaeBarrierSpec(
        target_template="/decoder/up_blocks.1_{frame}/Add_output_0",
        label_template="post_up_block_1_upsampler_frame_{frame}",
        reason="separate decoder up block 1 upsampler from up block 2",
    ),
    VaeBarrierSpec(
        target_template="/decoder/up_blocks.2/resnets.0_{frame}/Add_output_0",
        label_template="post_up_block_2_resnet_0_frame_{frame}",
        reason="bound decoder up block 2 resnet 0 as one compiler region",
    ),
    VaeBarrierSpec(
        target_template="/decoder/up_blocks.2/resnets.1_{frame}/Add_output_0",
        label_template="post_up_block_2_resnet_1_frame_{frame}",
        reason="bound decoder up block 2 resnet 1 as one compiler region",
    ),
    VaeBarrierSpec(
        target_template="/decoder/up_blocks.2/resnets.2_{frame}/Add_output_0",
        label_template="post_up_block_2_resnet_2_frame_{frame}",
        reason="separate decoder up block 2 resnets from its upsampler",
    ),
    VaeBarrierSpec(
        target_template="/decoder/up_blocks.2_{frame}/Add_output_0",
        label_template="post_up_block_2_upsampler_frame_{frame}",
        reason="separate decoder up block 2 upsampler from up block 3",
    ),
    VaeBarrierSpec(
        target_template="/decoder/up_blocks.3/resnets.0_{frame}/Add_output_0",
        label_template="post_up_block_3_resnet_0_frame_{frame}",
        reason="bound decoder up block 3 resnet 0 as one compiler region",
    ),
    VaeBarrierSpec(
        target_template="/decoder/up_blocks.3/resnets.1_{frame}/Add_output_0",
        label_template="post_up_block_3_resnet_1_frame_{frame}",
        reason="bound decoder up block 3 resnet 1 as one compiler region",
    ),
    VaeBarrierSpec(
        target_template="/decoder/up_blocks.3/resnets.2_{frame}/Add_output_0",
        label_template="post_up_block_3_resnet_2_frame_{frame}",
        reason="separate the final decoder resnet from the output head",
    ),
    VaeBarrierSpec(
        target_template="/decoder/nonlinearity_{frame}/Mul_output_0",
        label_template="post_final_norm_activation_frame_{frame}",
        reason="separate final normalization and activation from output convolution",
    ),
    VaeBarrierSpec(
        target_template="/decoder/conv_out_{frame}/Conv_output_0",
        label_template="post_final_conv_frame_{frame}",
        reason="separate final convolution from frame concatenation and unpatchify",
    ),
)

_VAE_RESNET_SCOPES = (
    ("mid_block/resnets.0", "mid_resnet_0"),
    ("mid_block/resnets.1", "mid_resnet_1"),
    *(
        (f"up_blocks.{block}/resnets.{resnet}", f"up_block_{block}_resnet_{resnet}")
        for block in range(4)
        for resnet in range(3)
    ),
)

# This is intentionally the smallest graph substitution that addresses the
# observed full-resolution build failure.  Do not silently expand it to other
# VAE convolutions: each added operator needs its own source-parity evidence.
_VAE_NATIVE_CONV_SCOPES = tuple(
    (resnet, conv, f"up_block_3_resnet_{resnet}_conv_{conv}")
    for resnet in range(3)
    for conv in (1, 2)
)

_VAE_NATIVE_CONV_PROFILES = {
    (2, 2, 2),
    (
        OFFICIAL_VAE_DECODER_PROFILE.latent_frames,
        OFFICIAL_VAE_DECODER_PROFILE.latent_height,
        OFFICIAL_VAE_DECODER_PROFILE.latent_width,
    ),
}

_VAE_RESNET_INTERNAL_BARRIER_MANIFEST = tuple(
    VaeBarrierSpec(
        target_template=(
            f"/decoder/{scope}/{{prepad_conv{conv}}}"
            if operator is None
            else f"/decoder/{scope}/conv{conv}_{{frame}}/{operator}_output_0"
        ),
        label_template=f"{boundary}_{label}_conv_{conv}_frame_{{frame}}",
        reason=(
            f"isolate {label.replace('_', ' ')} convolution {conv} at its {description} boundary"
        ),
    )
    for scope, label in _VAE_RESNET_SCOPES
    for conv in (1, 2)
    for operator, boundary, description in (
        (None, "pre_pad", "pre-padding"),
        ("Pad", "pre_conv", "post-padding"),
        ("Conv", "post_conv", "post-convolution"),
    )
)

# The stage exits prevent cross-block and cross-frame fusion.  The convolution
# exits bound the compiler region inside each residual block.  Both sides of
# causal padding are explicit boundaries because each fused ForeignNode
# otherwise requests the configured workspace limit plus additional unaccounted
# activation memory.  RMS normalization itself is replaced below: even as an
# isolated ForeignNode its TensorRT compiler tactic requests that same excess.
VAE_BARRIER_MANIFEST = _VAE_STAGE_BARRIER_MANIFEST + _VAE_RESNET_INTERNAL_BARRIER_MANIFEST


_LOADED_VAE_PLUGIN_HANDLES: dict[Path, ctypes.CDLL] = {}


def load_vae_cuda_plugin(*, verbose: bool = False) -> Path:
    """Load the Wan2.2 VAE barrier plugin and return its shared library."""

    from .vae_cuda_plugin_builder import ensure_vae_cuda_plugin

    path = ensure_vae_cuda_plugin(verbose=verbose).resolve()
    if path not in _LOADED_VAE_PLUGIN_HANDLES:
        _LOADED_VAE_PLUGIN_HANDLES[path] = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    return path


def _replace_tensor_consumers_with_barrier(trt, network, target_name: str, label: str):
    tensors, consumers = _index_network_edges(network)
    return _replace_indexed_tensor_consumers_with_barrier(
        trt,
        network,
        target_name,
        label,
        tensors,
        consumers,
    )


def _format_barrier_target(spec: VaeBarrierSpec, frame: int) -> str:
    """Resolve the exporter name for one unrolled decoder frame.

    ONNX keeps the first invocation's module scope unsuffixed and appends
    ``_1`` through ``_N`` to later invocations.  Keeping that rule here makes
    the semantic manifest apply uniformly to every temporal branch.
    """

    if frame < 0:
        raise ValueError(f"Wan2.2 VAE barrier frame must be non-negative, got {frame}")
    template = spec.target_template
    if frame == 0:
        template = template.replace("_{frame}", "")
    return template.format(
        frame=frame,
        # The first causal-convolution invocation pads the activation directly
        # after its nonlinearity.  Later unrolled invocations concatenate the
        # temporal cache first.  These are Pad's FP32 data inputs; the nearby
        # Cast tensors are INT64 padding-shape inputs and must not be treated as
        # compiler boundaries for the activation path.
        prepad_conv1=(
            "nonlinearity/Mul_output_0" if frame == 0 else f"conv1_{frame}/Concat_output_0"
        ),
        prepad_conv2=(
            "nonlinearity_1/Mul_output_0" if frame == 0 else f"conv2_{frame}/Concat_output_0"
        ),
    )


def _index_network_edges(network):
    """Index parsed TensorRT tensors and their original consumers in one pass."""

    tensors = {}
    consumers = {}
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        for output_index in range(layer.num_outputs):
            output = layer.get_output(output_index)
            if output is None:
                continue
            if output.name in tensors:
                raise RuntimeError(f"Duplicate Wan2.2 VAE tensor name {output.name!r}")
            tensors[output.name] = output
        for input_index in range(layer.num_inputs):
            input_tensor = layer.get_input(input_index)
            if input_tensor is not None:
                consumers.setdefault(input_tensor.name, []).append((layer, input_index))
    return tensors, consumers


def _format_rms_norm_names(
    scope: str,
    norm: int,
    frame: int,
) -> tuple[str, str, str]:
    """Return target, source-layer, and gamma names for one unrolled RMS norm."""

    if norm not in (1, 2):
        raise ValueError(f"Wan2.2 VAE resnet norm must be 1 or 2, got {norm}")
    if frame < 0:
        raise ValueError(f"Wan2.2 VAE RMS norm frame must be non-negative, got {frame}")
    suffix = "" if frame == 0 else f"_{frame}"
    norm_scope = f"/decoder/{scope}/norm{norm}{suffix}"
    return (
        f"{norm_scope}/Add_output_0",
        f"{norm_scope}/ReduceL2",
        f"vae.decoder.{scope.replace('/', '.')}.norm{norm}.gamma",
    )


def _format_final_rms_norm_names(frame: int) -> tuple[str, str, str]:
    """Return target, source-layer, and gamma names for decoder norm_out."""

    if frame < 0:
        raise ValueError(f"Wan2.2 VAE RMS norm frame must be non-negative, got {frame}")
    suffix = "" if frame == 0 else f"_{frame}"
    norm_scope = f"/decoder/norm_out{suffix}"
    return (
        f"{norm_scope}/Add_output_0",
        f"{norm_scope}/ReduceL2",
        "vae.decoder.norm_out.gamma",
    )


def _format_up_block3_conv_names(
    resnet: int,
    conv: int,
    frame: int,
) -> tuple[str, str, str, str]:
    """Return parsed layer/output and shared initializer names for one Conv3d."""

    if resnet not in (0, 1, 2):
        raise ValueError(f"Wan2.2 VAE up block 3 resnet must be 0, 1, or 2, got {resnet}")
    if conv not in (1, 2):
        raise ValueError(f"Wan2.2 VAE up block 3 convolution must be 1 or 2, got {conv}")
    if frame < 0:
        raise ValueError(f"Wan2.2 VAE Conv3d frame must be non-negative, got {frame}")
    suffix = "" if frame == 0 else f"_{frame}"
    scope = f"up_blocks.3/resnets.{resnet}/conv{conv}"
    layer = f"/decoder/{scope}{suffix}/Conv"
    initializer_scope = scope.replace("/", ".")
    return (
        layer,
        f"{layer}_output_0",
        f"vae.decoder.{initializer_scope}.weight",
        f"vae.decoder.{initializer_scope}.bias",
    )


def _load_up_block3_conv_initializers(onnx_path: Path):
    """Load only the twelve initializers needed by the six native Conv3d layers."""

    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path), load_external_data=False)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    requested = {
        name
        for resnet, conv, _label in _VAE_NATIVE_CONV_SCOPES
        for name in _format_up_block3_conv_names(resnet, conv, 0)[2:]
    }
    missing = sorted(requested - initializers.keys())
    if missing:
        raise RuntimeError(
            "Wan2.2 VAE ONNX is missing native Conv3d initializers: " + ", ".join(missing)
        )

    arrays = {}
    for name in sorted(requested):
        initializer = initializers[name]
        arrays[name] = np.ascontiguousarray(
            numpy_helper.to_array(initializer, base_dir=str(onnx_path.parent)),
            dtype=np.float32,
        )
    return arrays


def _replace_up_block3_conv3d(
    trt,
    network,
    profile: Wan22VaeDecoderProfile,
    onnx_path: Path,
):
    """Replace exactly six logical final-up-block convolutions across all frames."""

    profile_key = (profile.latent_frames, profile.latent_height, profile.latent_width)
    if profile_key not in _VAE_NATIVE_CONV_PROFILES:
        raise RuntimeError(
            "Wan22VaeConv3d is qualified only for the 2x2x2 validation profile and "
            f"the 31x44x80 official profile, got {profile_key}"
        )

    import numpy as np

    load_vae_cuda_plugin()
    creator = trt.get_plugin_registry().get_creator("Wan22VaeConv3d", "1", "")
    if creator is None:
        raise RuntimeError("Wan22VaeConv3d plugin creator is not registered")

    tensors, consumers = _index_network_edges(network)
    layers = {
        network.get_layer(index).name: network.get_layer(index)
        for index in range(network.num_layers)
    }
    initializer_arrays = _load_up_block3_conv_initializers(onnx_path)
    constant_tensors = {}
    for resnet, conv, label in _VAE_NATIVE_CONV_SCOPES:
        _layer_name, _target_name, weight_name, bias_name = _format_up_block3_conv_names(
            resnet, conv, 0
        )
        input_channels = 512 if (resnet, conv) == (0, 1) else 256
        expected_weight = (256, input_channels, 3, 3, 3)
        expected_bias = (256,)
        weight = initializer_arrays[weight_name]
        bias = initializer_arrays[bias_name]
        if weight.dtype != np.float32 or tuple(weight.shape) != expected_weight:
            raise RuntimeError(
                f"Invalid Wan2.2 VAE Conv3d weight {weight_name!r}: "
                f"{tuple(weight.shape)}/{weight.dtype}, expected {expected_weight}/float32"
            )
        if bias.dtype != np.float32 or tuple(bias.shape) != expected_bias:
            raise RuntimeError(
                f"Invalid Wan2.2 VAE Conv3d bias {bias_name!r}: "
                f"{tuple(bias.shape)}/{bias.dtype}, expected {expected_bias}/float32"
            )
        weight_layer = network.add_constant(expected_weight, weight)
        bias_layer = network.add_constant(expected_bias, bias)
        if weight_layer is None or bias_layer is None:
            raise RuntimeError(f"Could not add shared Wan2.2 VAE Conv3d constants for {label}")
        weight_layer.name = f"wan22_vae_conv3d_{label}_shared_weight"
        bias_layer.name = f"wan22_vae_conv3d_{label}_shared_bias"
        weight_tensor = weight_layer.get_output(0)
        bias_tensor = bias_layer.get_output(0)
        weight_tensor.name = f"{weight_layer.name}_output"
        bias_tensor.name = f"{bias_layer.name}_output"
        constant_tensors[(resnet, conv)] = (weight_tensor, bias_tensor)

    replacements = []
    spatial_input = (profile.latent_height * 8 + 2, profile.latent_width * 8 + 2)
    for frame in range(profile.latent_frames):
        input_depth = 3 if frame == 0 else 6
        for resnet, conv, label in _VAE_NATIVE_CONV_SCOPES:
            layer_name, target_name, weight_name, bias_name = _format_up_block3_conv_names(
                resnet, conv, frame
            )
            original_layer = layers.get(layer_name)
            target = tensors.get(target_name)
            if original_layer is None or target is None or original_layer.num_inputs < 1:
                raise RuntimeError(
                    f"Could not resolve Wan2.2 VAE Conv3d layer/output {layer_name!r}"
                )
            source = original_layer.get_input(0)
            original_consumers = consumers.get(target_name, ())
            input_channels = 512 if (resnet, conv) == (0, 1) else 256
            expected_input = (1, input_channels, input_depth, *spatial_input)
            expected_output = (
                1,
                256,
                input_depth - 2,
                spatial_input[0] - 2,
                spatial_input[1] - 2,
            )
            if (
                source is None
                or source.dtype != trt.float32
                or target.dtype != trt.float32
                or tuple(source.shape) != expected_input
                or tuple(target.shape) != expected_output
                or not original_consumers
            ):
                raise RuntimeError(
                    f"Invalid Wan2.2 VAE Conv3d contract for {target_name!r}: "
                    f"source={None if source is None else tuple(source.shape)}/"
                    f"{None if source is None else source.dtype}, "
                    f"target={tuple(target.shape)}/{target.dtype}, consumers={len(original_consumers)}, "
                    f"expected={expected_input}->{expected_output}"
                )

            field_values = {
                "batch": np.asarray([1], dtype=np.int32),
                "input_channels": np.asarray([input_channels], dtype=np.int32),
                "output_channels": np.asarray([256], dtype=np.int32),
                "input_depth": np.asarray([input_depth], dtype=np.int32),
                "input_height": np.asarray([spatial_input[0]], dtype=np.int32),
                "input_width": np.asarray([spatial_input[1]], dtype=np.int32),
            }
            fields = [
                trt.PluginField(name, value, trt.PluginFieldType.INT32)
                for name, value in field_values.items()
            ]
            plugin_layer_name = f"wan22_vae_conv3d_{label}_frame_{frame}"
            plugin = creator.create_plugin(
                plugin_layer_name,
                trt.PluginFieldCollection(fields),
            )
            if plugin is None:
                raise RuntimeError(f"Could not create {plugin_layer_name}")
            weight_tensor, bias_tensor = constant_tensors[(resnet, conv)]
            native_layer = network.add_plugin_v2(
                [source, weight_tensor, bias_tensor],
                plugin,
            )
            if native_layer is None:
                raise RuntimeError(f"Could not add {plugin_layer_name}")
            native_layer.name = plugin_layer_name
            output = native_layer.get_output(0)
            output.name = f"{plugin_layer_name}_output"
            if tuple(output.shape) != expected_output or output.dtype != trt.float32:
                raise RuntimeError(
                    f"Wan2.2 VAE Conv3d {plugin_layer_name} output is "
                    f"{tuple(output.shape)}/{output.dtype}, expected {expected_output}/{trt.float32}"
                )
            for consumer, input_index in original_consumers:
                consumer.set_input(input_index, output)
            for consumer, input_index in original_consumers:
                if consumer.get_input(input_index).name != output.name:
                    raise RuntimeError(
                        f"TensorRT refused to rewire {target_name!r} through {plugin_layer_name}"
                    )

            replacements.append(
                VaeConv3dReplacement(
                    logical_scope=f"up_blocks.3.resnets.{resnet}.conv{conv}",
                    frame=frame,
                    target_tensor=target_name,
                    source_tensor=source.name,
                    weight_initializer=weight_name,
                    bias_initializer=bias_name,
                    plugin_layer=plugin_layer_name,
                    plugin_output=output.name,
                    input_shape=expected_input,
                    output_shape=expected_output,
                    consumer_count=len(original_consumers),
                )
            )

    expected_count = profile.latent_frames * len(_VAE_NATIVE_CONV_SCOPES)
    if len(replacements) != expected_count:
        raise RuntimeError(
            f"Replaced {len(replacements)} Wan2.2 VAE Conv3d nodes, expected {expected_count}"
        )
    # Keep the twelve NumPy allocations alive through TensorRT's network build.
    return tuple(replacements), tuple(initializer_arrays.values())


def _replace_decoder_rms_norms(
    trt,
    network,
    profile: Wan22VaeDecoderProfile,
) -> tuple[VaeRmsNormReplacement, ...]:
    """Replace source-equivalent per-position RMS norms with zero-workspace CUDA."""

    load_vae_cuda_plugin()
    tensors, consumers = _index_network_edges(network)
    layers = {
        network.get_layer(index).name: network.get_layer(index)
        for index in range(network.num_layers)
    }
    creator = trt.get_plugin_registry().get_creator("Wan22VaeRmsNorm", "1", "")
    if creator is None:
        raise RuntimeError("Wan22VaeRmsNorm plugin creator is not registered")

    replacements = []
    for frame in range(profile.latent_frames):
        frame_norms = [
            (*_format_rms_norm_names(scope, norm, frame), f"{label}_{norm}")
            for scope, label in _VAE_RESNET_SCOPES
            for norm in (1, 2)
        ]
        frame_norms.append((*_format_final_rms_norm_names(frame), "final_output"))
        for target_name, reduce_layer_name, gamma_name, label in frame_norms:
            target = tensors.get(target_name)
            reduce_layer = layers.get(reduce_layer_name)
            gamma_layer = layers.get(gamma_name)
            if target is None:
                raise RuntimeError(f"Could not find Wan2.2 VAE RMS norm target {target_name!r}")
            if reduce_layer is None or reduce_layer.num_inputs < 1:
                raise RuntimeError(
                    f"Could not find Wan2.2 VAE RMS norm source layer {reduce_layer_name!r}"
                )
            source = reduce_layer.get_input(0)
            gamma = (
                gamma_layer.get_output(0)
                if gamma_layer is not None and gamma_layer.num_outputs == 1
                else None
            )
            if source is None or gamma is None:
                raise RuntimeError(
                    f"Could not resolve Wan2.2 VAE RMS norm inputs for {target_name!r}"
                )
            target_shape = tuple(target.shape)
            if (
                target.dtype != trt.float32
                or source.dtype != trt.float32
                or gamma.dtype != trt.float32
                or len(target_shape) != 5
                or tuple(source.shape) != target_shape
                or len(tuple(gamma.shape)) != 4
                or tuple(gamma.shape)[0] != target_shape[1]
            ):
                raise RuntimeError(
                    f"Invalid Wan2.2 VAE RMS norm contract for {target_name!r}: "
                    f"target={target_shape}/{target.dtype}, source={tuple(source.shape)}/{source.dtype}, "
                    f"gamma={tuple(gamma.shape)}/{gamma.dtype}"
                )
            original_consumers = consumers.get(target_name, ())
            if not original_consumers:
                raise RuntimeError(f"Wan2.2 VAE RMS norm target {target_name!r} has no consumers")

            layer_name = f"wan22_vae_rms_norm_{label}_frame_{frame}"
            plugin = creator.create_plugin(layer_name, trt.PluginFieldCollection([]))
            layer = network.add_plugin_v2([source, gamma], plugin)
            if layer is None:
                raise RuntimeError(f"Could not add {layer_name}")
            layer.name = layer_name
            output = layer.get_output(0)
            output.name = f"{layer_name}_output"
            if tuple(output.shape) != target_shape or output.dtype != trt.float32:
                raise RuntimeError(
                    f"Wan2.2 VAE RMS norm {layer_name} output is "
                    f"{tuple(output.shape)}/{output.dtype}, expected {target_shape}/{trt.float32}"
                )
            for consumer, input_index in original_consumers:
                consumer.set_input(input_index, output)

            replacements.append(
                VaeRmsNormReplacement(
                    target_tensor=target_name,
                    source_tensor=source.name,
                    gamma_tensor=gamma.name,
                    plugin_layer=layer_name,
                    plugin_output=output.name,
                    tensor_shape=target_shape,
                    consumer_count=len(original_consumers),
                )
            )
    return tuple(replacements)


def _replace_indexed_tensor_consumers_with_barrier(
    trt,
    network,
    target_name: str,
    label: str,
    tensors,
    consumers_by_tensor,
):
    """Insert one barrier using the immutable pre-insertion network index."""

    target = tensors.get(target_name)
    if target is None:
        raise RuntimeError(f"Could not find Wan2.2 VAE barrier tensor {target_name!r}")
    consumers = consumers_by_tensor.get(target_name, ())
    if target.dtype != trt.float32:
        raise RuntimeError(f"Wan2.2 VAE barrier tensor {target_name!r} is not FP32")
    if not consumers:
        raise RuntimeError(f"Wan2.2 VAE barrier tensor {target_name!r} has no consumers")
    target_shape = tuple(target.shape)
    if len(target_shape) != 5 or any(dimension <= 0 for dimension in target_shape):
        raise RuntimeError(
            f"Wan2.2 VAE barrier tensor {target_name!r} must have a static 5D shape, "
            f"got {target_shape}"
        )

    creator = trt.get_plugin_registry().get_creator("Wan22VaeFp32Barrier", "1", "")
    if creator is None:
        raise RuntimeError("Wan22VaeFp32Barrier plugin creator is not registered")
    layer_name = f"wan22_vae_fp32_barrier_{label}"
    plugin = creator.create_plugin(layer_name, trt.PluginFieldCollection([]))
    if plugin is None:
        raise RuntimeError(f"Could not create {layer_name}")
    barrier = network.add_plugin_v2([target], plugin)
    if barrier is None:
        raise RuntimeError(f"Could not add {layer_name}")
    barrier.name = layer_name
    barrier_output = barrier.get_output(0)
    barrier_output.name = f"{layer_name}_output"
    if tuple(barrier_output.shape) != target_shape:
        raise RuntimeError(
            f"Wan2.2 VAE barrier {layer_name} has shape {tuple(barrier_output.shape)}, "
            f"expected passthrough shape {target_shape}"
        )

    for consumer, input_index in consumers:
        consumer.set_input(input_index, barrier_output)
    for consumer, input_index in consumers:
        if consumer.get_input(input_index).name != barrier_output.name:
            raise RuntimeError(f"TensorRT refused to rewire {target_name!r} through {layer_name}")

    return VaeBarrierInsertion(
        target_tensor=target_name,
        barrier_layer=layer_name,
        barrier_output=barrier_output.name,
        tensor_shape=target_shape,
        consumer_count=len(consumers),
    )


def _insert_decoder_barriers(
    trt,
    network,
    profile: Wan22VaeDecoderProfile,
) -> tuple[VaeBarrierInsertion, ...]:
    """Bound every semantic stage of every unrolled decoder frame.

    TensorRT can otherwise fuse across the cache-carrying temporal branches.
    A last-frame-only manifest merely moves the oversized compiler region to an
    earlier frame, so the same audited boundaries must cover all invocations.
    """

    load_vae_cuda_plugin()
    tensors, consumers = _index_network_edges(network)
    return tuple(
        _replace_indexed_tensor_consumers_with_barrier(
            trt,
            network,
            _format_barrier_target(spec, frame),
            spec.label_template.format(frame=frame),
            tensors,
            consumers,
        )
        for frame in range(profile.latent_frames)
        for spec in VAE_BARRIER_MANIFEST
    )


def _load_converted_vae(checkpoint: Path):
    """Load the native Wan2.2 VAE checkpoint into its canonical module."""

    from diffusers import AutoencoderKLWan

    from .checkpoint_mapper import convert_vae_state_dict, load_native_vae_state_dict

    vae = AutoencoderKLWan(**VAE22_CONFIG).eval().requires_grad_(False)
    vae.load_state_dict(
        convert_vae_state_dict(load_native_vae_state_dict(checkpoint)),
        strict=True,
    )
    return vae


def export_vae_decoder_onnx(
    checkpoint: str | Path,
    output: str | Path,
    *,
    profile: Wan22VaeDecoderProfile = OFFICIAL_VAE_DECODER_PROFILE,
) -> Path:
    """Export a complete fixed-shape Wan2.2 VAE decoder to external-data ONNX."""

    import torch
    from torch.onnx._internal.torchscript_exporter import symbolic_helper

    checkpoint = Path(checkpoint)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    class Decoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, normalized_latents: torch.Tensor) -> torch.Tensor:
            mean = normalized_latents.new_tensor(VAE22_CONFIG["latents_mean"]).view(1, 48, 1, 1, 1)
            std = normalized_latents.new_tensor(VAE22_CONFIG["latents_std"]).view(1, 48, 1, 1, 1)
            latents = normalized_latents * std + mean
            return self.vae.decode(latents, return_dict=False)[0]

    vae = _load_converted_vae(checkpoint)
    wrapper = Decoder(vae).eval()
    sample = torch.zeros(profile.input_shape, dtype=torch.float32)

    with torch.inference_mode():
        # The decoder only uses integer 2x spatial upsampling.  For that case,
        # nearest-exact and ONNX asymmetric-nearest/floor select identical
        # source pixels; the legacy exporter lacks only the operator spelling.
        torch.onnx.register_custom_op_symbolic(
            "aten::_upsample_nearest_exact2d",
            symbolic_helper._interpolate_helper("_upsample_nearest_exact2d", 4, "nearest"),
            20,
        )
        torch.onnx.export(
            wrapper,
            (sample,),
            str(output),
            input_names=["latents"],
            output_names=["video"],
            opset_version=20,
            do_constant_folding=True,
            dynamo=False,
            external_data=True,
        )

    del sample, wrapper, vae
    gc.collect()
    return output


def _configure_builder_config(trt, config, *, workspace_gib: int) -> None:
    if workspace_gib <= 0:
        raise ValueError(f"workspace_gib must be positive, got {workspace_gib}")
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)

    # Preserve TensorRT's default TF32 setting.  A controlled GB300 A/B build
    # showed that clearing TF32 increased every decoded-frame error metric for
    # this graph, so overriding the default would move us farther from the
    # converted Diffusers reference rather than improving source parity.


def _validate_network_contract(trt, network, profile: Wan22VaeDecoderProfile) -> None:
    if network.num_inputs != 1 or network.num_outputs != 1:
        raise RuntimeError(
            "Wan2.2 VAE ONNX must have exactly one input and one output, got "
            f"{network.num_inputs} inputs and {network.num_outputs} outputs"
        )

    input_tensor = network.get_input(0)
    output_tensor = network.get_output(0)
    if input_tensor.name != "latents" or output_tensor.name != "video":
        raise RuntimeError(
            "Wan2.2 VAE ONNX tensor names must be 'latents' and 'video', got "
            f"{input_tensor.name!r} and {output_tensor.name!r}"
        )
    if tuple(input_tensor.shape) != profile.input_shape:
        raise RuntimeError(
            f"Wan2.2 VAE input shape is {tuple(input_tensor.shape)}, expected {profile.input_shape}"
        )
    if tuple(output_tensor.shape) != profile.output_shape:
        raise RuntimeError(
            f"Wan2.2 VAE output shape is {tuple(output_tensor.shape)}, "
            f"expected {profile.output_shape}"
        )
    if input_tensor.dtype != trt.float32 or output_tensor.dtype != trt.float32:
        raise RuntimeError(
            "Wan2.2 VAE TensorRT contract must use FP32 input and output, got "
            f"{input_tensor.dtype} and {output_tensor.dtype}"
        )


def build_onnx_engine(
    onnx_path: str | Path,
    *,
    profile: Wan22VaeDecoderProfile | None = None,
    workspace_gib: int = OFFICIAL_VAE_WORKSPACE_GIB,
    verbose: bool = False,
) -> bytes:
    """Parse an ONNX graph and build a TensorRT plan using default tactics."""

    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(Path(onnx_path).resolve())):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parse failed:\n{errors}")
    if profile is not None:
        _validate_network_contract(trt, network, profile)
        _replace_decoder_rms_norms(trt, network, profile)
        _insert_decoder_barriers(trt, network, profile)
        _conv_replacements, conv_initializer_owners = _replace_up_block3_conv3d(
            trt,
            network,
            profile,
            Path(onnx_path).resolve(),
        )
    else:
        conv_initializer_owners = ()

    config = builder.create_builder_config()
    _configure_builder_config(trt, config, workspace_gib=workspace_gib)
    plan = builder.build_serialized_network(network, config)
    # The IConstantLayer weights are borrowed from these NumPy allocations
    # until build_serialized_network has completed.
    del conv_initializer_owners
    if plan is None:
        raise RuntimeError("TensorRT build returned no serialized Wan2.2 VAE network")
    return bytes(plan)


def build_onnx_engine_file(
    onnx_path: str | Path,
    output: str | Path,
    *,
    profile: Wan22VaeDecoderProfile | None = None,
    workspace_gib: int = OFFICIAL_VAE_WORKSPACE_GIB,
    verbose: bool = False,
) -> Path:
    """Build an ONNX plan and write it to ``output``."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(
        build_onnx_engine(
            onnx_path,
            profile=profile,
            workspace_gib=workspace_gib,
            verbose=verbose,
        )
    )
    return output


def build_vae_decoder_engine(
    checkpoint: str | Path,
    *,
    profile: Wan22VaeDecoderProfile = OFFICIAL_VAE_DECODER_PROFILE,
    workspace_gib: int = OFFICIAL_VAE_WORKSPACE_GIB,
    verbose: bool = False,
) -> bytes:
    """Build the complete Wan2.2 VAE plan without retaining ONNX intermediates."""

    with tempfile.TemporaryDirectory(prefix="trtmc-wan2-2-vae-") as temp_dir:
        onnx_path = Path(temp_dir) / "wan2_2_vae_decoder.onnx"
        export_vae_decoder_onnx(checkpoint, onnx_path, profile=profile)
        return build_onnx_engine(
            onnx_path,
            profile=profile,
            workspace_gib=workspace_gib,
            verbose=verbose,
        )


__all__ = [
    "OFFICIAL_VAE_DECODER_PROFILE",
    "OFFICIAL_VAE_WORKSPACE_GIB",
    "VAE_BARRIER_MANIFEST",
    "VaeBarrierInsertion",
    "VaeBarrierSpec",
    "VaeConv3dReplacement",
    "VaeRmsNormReplacement",
    "Wan22VaeDecoderProfile",
    "build_onnx_engine",
    "build_onnx_engine_file",
    "build_vae_decoder_engine",
    "export_vae_decoder_onnx",
    "load_vae_cuda_plugin",
]
