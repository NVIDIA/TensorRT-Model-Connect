# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT Depth Anything V2 monocular-depth family.

Network construction uses TensorRT's Python Network API directly; there is no
ONNX export or parser. The architecture is exactly HF Transformers'
`DepthAnythingForDepthEstimation`: a DINOv2 ViT-S backbone (absolute position
embeddings, no RoPE, no register tokens - unlike DINOv3), four hidden states
tapped after encoder layers 3/6/9/12 and independently re-normalized by the
backbone's shared final LayerNorm, a DPT-style reassemble/fusion neck, and a
3-conv depth head.

This first cut fixes the input to one square resolution (the checkpoint's own
training resolution, `backbone_config.image_size`) rather than the
aspect-ratio-preserving dynamic shape the reference `DPTImageProcessor` uses -
the same simplification `families/dinov3` makes for its own ViT. Every
intermediate spatial size the neck touches (each reassemble level, each
fusion stage, the head's upsample target) is therefore a static Python int
computed once at build time from that one resolution, not a runtime shape.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import tensorrt as trt

from . import graph_ops
from .checkpoint import WeightDict, encoder_layer_count, load_tensor, open_checkpoint

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


# HF `Dinov2Config` defaults that the small-hf `backbone_config` leaves
# implicit (it sets only hidden_size, image_size, num_attention_heads,
# patch_size, out_features/out_indices, and reshape_hidden_states).
_LAYER_NORM_EPS = 1.0e-6
_HIDDEN_ACT = "gelu"


def _transpose_linear(value: np.ndarray, name: str) -> np.ndarray:
    if value.ndim != 2:
        raise ValueError(f"Expected rank-2 linear weight for {name}, got {value.shape}")
    return np.ascontiguousarray(value.T)


def resolve_config(raw: dict, readers) -> dict:
    backbone = raw.get("backbone_config") or {}
    if str(backbone.get("model_type", "")).lower() != "dinov2":
        raise ValueError(f"Depth Anything V2 requires a DINOv2 backbone, got {backbone.get('model_type')!r}")

    hidden_size = int(backbone.get("hidden_size", 384))
    num_heads = int(backbone.get("num_attention_heads", 6))
    patch_size = int(backbone.get("patch_size", 14))
    image_size = int(backbone.get("image_size", 518))
    if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads != 0:
        raise ValueError("Depth Anything V2 hidden_size must be positive and divisible by num_attention_heads")
    if image_size <= 0 or image_size % patch_size != 0:
        raise ValueError("Depth Anything V2 image_size must be a positive multiple of patch_size")
    if bool(backbone.get("reshape_hidden_states", False)):
        raise NotImplementedError("Depth Anything V2 only supports reshape_hidden_states=false backbones")

    out_indices = backbone.get("out_indices")
    if not isinstance(out_indices, list) or not out_indices:
        raise ValueError("Depth Anything V2 backbone_config.out_indices is required")
    out_layers = tuple(int(index) for index in out_indices)

    neck_hidden_sizes = raw.get("neck_hidden_sizes")
    reassemble_factors = raw.get("reassemble_factors")
    if not isinstance(neck_hidden_sizes, list) or not isinstance(reassemble_factors, list):
        raise ValueError("Depth Anything V2 requires neck_hidden_sizes and reassemble_factors")
    if len(neck_hidden_sizes) != len(out_layers) or len(reassemble_factors) != len(out_layers):
        raise ValueError("Depth Anything V2 neck_hidden_sizes/reassemble_factors must match out_indices")

    num_layers = encoder_layer_count(readers)
    if max(out_layers) > num_layers:
        raise ValueError(f"Depth Anything V2 out_indices {out_layers} exceed the {num_layers}-layer checkpoint")

    depth_estimation_type = str(raw.get("depth_estimation_type", "relative"))
    if depth_estimation_type != "relative":
        raise NotImplementedError("Depth Anything V2 only supports depth_estimation_type=relative")

    return {
        "hidden_size": hidden_size,
        "num_attention_heads": num_heads,
        "head_dim": hidden_size // num_heads,
        "num_hidden_layers": num_layers,
        "patch_size": patch_size,
        "image_size": image_size,
        "layer_norm_eps": _LAYER_NORM_EPS,
        "hidden_act": _HIDDEN_ACT,
        "out_layers": out_layers,
        "neck_hidden_sizes": tuple(int(v) for v in neck_hidden_sizes),
        "reassemble_factors": tuple(float(v) for v in reassemble_factors),
        "reassemble_hidden_size": int(raw.get("reassemble_hidden_size", hidden_size)),
        "fusion_hidden_size": int(raw.get("fusion_hidden_size", 64)),
        "head_hidden_size": int(raw.get("head_hidden_size", 32)),
        "max_depth": float(raw.get("max_depth") or 1.0),
    }


def load_weights(model_dir: str, cfg: dict, precision: str) -> WeightDict:
    readers = open_checkpoint(model_dir)
    dtype = _target_dtype(precision)
    weights = WeightDict()

    def store(logical: str, checkpoint_name: str, *, transpose: bool = False):
        value = load_tensor(readers, checkpoint_name)
        if transpose:
            value = _transpose_linear(value, checkpoint_name)
        weights[logical] = np.ascontiguousarray(value, dtype=dtype)

    store("backbone.patch.weight", "backbone.embeddings.patch_embeddings.projection.weight")
    store("backbone.patch.bias", "backbone.embeddings.patch_embeddings.projection.bias")
    store("backbone.cls_token", "backbone.embeddings.cls_token")
    store("backbone.position_embeddings", "backbone.embeddings.position_embeddings")
    store("backbone.norm.weight", "backbone.layernorm.weight")
    store("backbone.norm.bias", "backbone.layernorm.bias")

    for layer in range(cfg["num_hidden_layers"]):
        source = f"backbone.encoder.layer.{layer}"
        target = f"backbone.layer.{layer}"
        store(f"{target}.norm1.weight", f"{source}.norm1.weight")
        store(f"{target}.norm1.bias", f"{source}.norm1.bias")
        store(f"{target}.norm2.weight", f"{source}.norm2.weight")
        store(f"{target}.norm2.bias", f"{source}.norm2.bias")
        store(f"{target}.layer_scale1", f"{source}.layer_scale1.lambda1")
        store(f"{target}.layer_scale2", f"{source}.layer_scale2.lambda1")
        for name, checkpoint_name in (
            ("q_proj", "query"),
            ("k_proj", "key"),
            ("v_proj", "value"),
        ):
            store(
                f"{target}.attention.{name}.weight",
                f"{source}.attention.attention.{checkpoint_name}.weight",
                transpose=True,
            )
            store(f"{target}.attention.{name}.bias", f"{source}.attention.attention.{checkpoint_name}.bias")
        store(f"{target}.attention.o_proj.weight", f"{source}.attention.output.dense.weight", transpose=True)
        store(f"{target}.attention.o_proj.bias", f"{source}.attention.output.dense.bias")
        store(f"{target}.mlp.fc1.weight", f"{source}.mlp.fc1.weight", transpose=True)
        store(f"{target}.mlp.fc1.bias", f"{source}.mlp.fc1.bias")
        store(f"{target}.mlp.fc2.weight", f"{source}.mlp.fc2.weight", transpose=True)
        store(f"{target}.mlp.fc2.bias", f"{source}.mlp.fc2.bias")

    for index, factor in enumerate(cfg["reassemble_factors"]):
        prefix = f"neck.reassemble_stage.layers.{index}"
        store(f"neck.reassemble.{index}.projection.weight", f"{prefix}.projection.weight")
        store(f"neck.reassemble.{index}.projection.bias", f"{prefix}.projection.bias")
        if factor != 1.0:
            store(f"neck.reassemble.{index}.resize.weight", f"{prefix}.resize.weight")
            store(f"neck.reassemble.{index}.resize.bias", f"{prefix}.resize.bias")

    for index in range(len(cfg["neck_hidden_sizes"])):
        store(f"neck.convs.{index}.weight", f"neck.convs.{index}.weight")

    for index in range(len(cfg["neck_hidden_sizes"])):
        prefix = f"neck.fusion_stage.layers.{index}"
        store(f"neck.fusion.{index}.projection.weight", f"{prefix}.projection.weight")
        store(f"neck.fusion.{index}.projection.bias", f"{prefix}.projection.bias")
        for residual in ("residual_layer1", "residual_layer2"):
            for conv in ("convolution1", "convolution2"):
                store(
                    f"neck.fusion.{index}.{residual}.{conv}.weight",
                    f"{prefix}.{residual}.{conv}.weight",
                )
                store(
                    f"neck.fusion.{index}.{residual}.{conv}.bias",
                    f"{prefix}.{residual}.{conv}.bias",
                )

    for name in ("conv1", "conv2", "conv3"):
        store(f"head.{name}.weight", f"head.{name}.weight")
        store(f"head.{name}.bias", f"head.{name}.bias")

    return weights


def _target_dtype(precision: str) -> np.dtype:
    if precision == "fp16":
        return np.dtype(np.float16)
    if precision == "fp32":
        return np.dtype(np.float32)
    raise ValueError(f"Unsupported Depth Anything V2 precision: {precision}")


def _reassembled_size(grid: int, factor: float) -> int:
    if factor > 1:
        # ConvTranspose2d(kernel=factor, stride=factor, padding=0).
        return (grid - 1) * int(factor) + int(factor)
    if factor == 1:
        return grid
    # Conv2d(kernel=3, stride=int(1/factor), padding=1).
    stride = int(round(1.0 / factor))
    return (grid + 2 * 1 - 3) // stride + 1


def build_engine(cfg: dict, weights: WeightDict, *, precision: str, verbose: bool) -> bytes:
    dtype = _target_dtype(precision)
    trt_dtype = trt.float16 if dtype == np.dtype(np.float16) else trt.float32
    builder, network, builder_config = graph_ops.new_network(verbose)

    image_size = cfg["image_size"]
    patch_size = cfg["patch_size"]
    hidden_size = cfg["hidden_size"]
    num_heads = cfg["num_attention_heads"]
    head_dim = cfg["head_dim"]
    grid = image_size // patch_size
    num_patches = grid * grid
    seq_len = 1 + num_patches
    out_layers = cfg["out_layers"]
    fusion_hidden = cfg["fusion_hidden_size"]

    if verbose:
        print(
            "[trtmc build] depth_anything_v2: "
            f"image={image_size}x{image_size}, patch={patch_size}, grid={grid}x{grid}, "
            f"hidden={hidden_size}, layers={cfg['num_hidden_layers']}, heads={num_heads}, "
            f"taps={out_layers}, precision={precision}",
            file=sys.stderr,
        )

    pixel_values = network.add_input("pixel_values", trt.float32, (1, 3, image_size, image_size))
    pixels = graph_ops.cast(network, pixel_values, trt_dtype)
    patch = graph_ops.conv2d(
        network,
        pixels,
        weights["backbone.patch.weight"],
        weights["backbone.patch.bias"],
        stride=patch_size,
        padding=0,
        dtype=dtype,
    )
    patch_tokens = graph_ops.shuffle(
        network, patch, first_transpose=(0, 2, 3, 1), reshape_dims=(1, num_patches, hidden_size)
    )
    cls = graph_ops.cast(
        network, graph_ops.constant(network, weights["backbone.cls_token"], (1, 1, hidden_size), dtype), trt_dtype
    )
    concat = network.add_concatenation([cls, patch_tokens])
    concat.axis = 1
    hidden = concat.get_output(0)
    position = graph_ops.cast(
        network,
        graph_ops.constant(network, weights["backbone.position_embeddings"], (1, seq_len, hidden_size), dtype),
        trt_dtype,
    )
    hidden = network.add_elementwise(hidden, position, trt.ElementWiseOperation.SUM).get_output(0)

    def to_heads(tensor):
        return graph_ops.shuffle(
            network,
            tensor,
            reshape_dims=(1, seq_len, num_heads, head_dim),
            second_transpose=(0, 2, 1, 3),
        )

    taps: dict[int, object] = {}
    for layer in range(cfg["num_hidden_layers"]):
        prefix = f"backbone.layer.{layer}"

        def normalize(name: str, tensor):
            key = f"{prefix}.{name}"
            return graph_ops.layer_norm(
                network, tensor, hidden_size, weights[f"{key}.weight"], weights[f"{key}.bias"],
                cfg["layer_norm_eps"], dtype,
            )

        residual = hidden
        normalized = normalize("norm1", hidden)
        heads = {}
        for name in ("q_proj", "k_proj", "v_proj"):
            projected = graph_ops.linear_with_bias(
                network, normalized, weights, f"{prefix}.attention.{name}", dtype
            )
            heads[name] = to_heads(projected)
        context = graph_ops.attention(network, heads["q_proj"], heads["k_proj"], heads["v_proj"], head_dim, dtype)
        merged = graph_ops.shuffle(
            network, context, first_transpose=(0, 2, 1, 3), reshape_dims=(1, seq_len, hidden_size)
        )
        attention_out = graph_ops.linear_with_bias(network, merged, weights, f"{prefix}.attention.o_proj", dtype)
        hidden = graph_ops.add_scaled_residual(
            network, residual, attention_out, weights[f"{prefix}.layer_scale1"], dtype
        )

        residual = hidden
        normalized = normalize("norm2", hidden)
        activated = graph_ops.linear_with_bias(network, normalized, weights, f"{prefix}.mlp.fc1", dtype)
        activated = graph_ops.gelu(network, activated, dtype)
        mlp = graph_ops.linear_with_bias(network, activated, weights, f"{prefix}.mlp.fc2", dtype)
        hidden = graph_ops.add_scaled_residual(network, residual, mlp, weights[f"{prefix}.layer_scale2"], dtype)

        layer_number = layer + 1
        if layer_number in out_layers:
            taps[layer_number] = graph_ops.layer_norm(
                network, hidden, hidden_size, weights["backbone.norm.weight"], weights["backbone.norm.bias"],
                cfg["layer_norm_eps"], dtype,
            )

    # --- Neck: reassemble each tap into a spatial feature map. ---
    reassembled = []
    sizes = []
    for index, layer_number in enumerate(out_layers):
        tap = taps[layer_number]
        patches_only = graph_ops.slice_tensor(network, tap, (0, 1, 0), (1, num_patches, hidden_size))
        spatial = graph_ops.shuffle(
            network, patches_only, reshape_dims=(1, grid, grid, hidden_size), second_transpose=(0, 3, 1, 2)
        )
        channels = cfg["neck_hidden_sizes"][index]
        projected = graph_ops.conv2d(
            network, spatial, weights[f"neck.reassemble.{index}.projection.weight"],
            weights[f"neck.reassemble.{index}.projection.bias"], stride=1, padding=0, dtype=dtype,
        )
        factor = cfg["reassemble_factors"][index]
        if factor > 1:
            resized = graph_ops.conv_transpose2d(
                network, projected, weights[f"neck.reassemble.{index}.resize.weight"],
                weights[f"neck.reassemble.{index}.resize.bias"], stride=int(factor), dtype=dtype,
            )
        elif factor == 1:
            resized = projected
        else:
            resized = graph_ops.conv2d(
                network, projected, weights[f"neck.reassemble.{index}.resize.weight"],
                weights[f"neck.reassemble.{index}.resize.bias"],
                stride=int(round(1.0 / factor)), padding=1, dtype=dtype,
            )
        reassembled.append(resized)
        sizes.append(_reassembled_size(grid, factor))
        del channels  # channel count is carried by the tensor itself; kept for readability above.

    features = [
        graph_ops.conv2d(
            network, reassembled[index], weights[f"neck.convs.{index}.weight"], None,
            stride=1, padding=1, dtype=dtype,
        )
        for index in range(len(out_layers))
    ]

    def preact_residual(tensor, prefix: str):
        activated = graph_ops.relu(network, tensor)
        activated = graph_ops.conv2d(
            network, activated, weights[f"{prefix}.convolution1.weight"], weights[f"{prefix}.convolution1.bias"],
            stride=1, padding=1, dtype=dtype,
        )
        activated = graph_ops.relu(network, activated)
        activated = graph_ops.conv2d(
            network, activated, weights[f"{prefix}.convolution2.weight"], weights[f"{prefix}.convolution2.bias"],
            stride=1, padding=1, dtype=dtype,
        )
        return network.add_elementwise(activated, tensor, trt.ElementWiseOperation.SUM).get_output(0)

    reversed_features = list(reversed(features))
    reversed_sizes = list(reversed(sizes))
    fused = None
    fused_size = 0
    for idx, current in enumerate(reversed_features):
        is_last = idx == len(reversed_features) - 1
        if fused is None:
            hidden_state = current
        else:
            if fused_size != reversed_sizes[idx]:
                raise ValueError(
                    "Depth Anything V2 fusion residual shape mismatch: "
                    f"fused={fused_size} current={reversed_sizes[idx]}. This build only "
                    "supports the checkpoint's native resolution, where every fusion "
                    "residual matches its skip connection without interpolation."
                )
            skip = preact_residual(fused, f"neck.fusion.{idx}.residual_layer1")
            hidden_state = network.add_elementwise(current, skip, trt.ElementWiseOperation.SUM).get_output(0)
        hidden_state = preact_residual(hidden_state, f"neck.fusion.{idx}.residual_layer2")

        if is_last:
            target_size = reversed_sizes[idx] * 2
        else:
            target_size = reversed_sizes[idx + 1]
        hidden_state = graph_ops.resize_bilinear(
            network, hidden_state, (1, fusion_hidden, target_size, target_size), align_corners=True
        )
        hidden_state = graph_ops.conv2d(
            network, hidden_state, weights[f"neck.fusion.{idx}.projection.weight"],
            weights[f"neck.fusion.{idx}.projection.bias"], stride=1, padding=0, dtype=dtype,
        )
        fused = hidden_state
        fused_size = target_size

    # --- Head: two convs, an upsample to the input resolution, two more convs. ---
    head = graph_ops.conv2d(
        network, fused, weights["head.conv1.weight"], weights["head.conv1.bias"], stride=1, padding=1, dtype=dtype
    )
    head = graph_ops.resize_bilinear(
        network, head, (1, cfg["fusion_hidden_size"] // 2, image_size, image_size), align_corners=True
    )
    head = graph_ops.conv2d(
        network, head, weights["head.conv2.weight"], weights["head.conv2.bias"], stride=1, padding=1, dtype=dtype
    )
    head = graph_ops.relu(network, head)
    head = graph_ops.conv2d(
        network, head, weights["head.conv3.weight"], weights["head.conv3.bias"], stride=1, padding=0, dtype=dtype
    )
    head = graph_ops.relu(network, head)  # depth_estimation_type == "relative"
    if cfg["max_depth"] != 1.0:
        scale = graph_ops.cast(
            network, graph_ops.constant(network, np.asarray(cfg["max_depth"]), (1, 1, 1, 1), dtype), head.dtype
        )
        head = network.add_elementwise(head, scale, trt.ElementWiseOperation.PROD).get_output(0)

    depth = graph_ops.cast(network, head, trt.float32)
    depth = graph_ops.shuffle(network, depth, reshape_dims=(1, image_size, image_size))
    depth.name = "predicted_depth"
    network.mark_output(depth)

    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT Depth Anything V2 engine build failed")
    return bytes(plan)


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one Depth Anything V2 bundle."""
    if request.task != "monocular_geometry":
        raise ValueError("depth_anything_v2 supports only task=monocular_geometry")
    if request.dynamic_kv_cache:
        raise NotImplementedError("depth_anything_v2 does not support dynamic_kv_cache")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError(
            "depth_anything_v2 builds a fixed square resolution from the checkpoint's own "
            "backbone_config.image_size; --image-height/--image-width are not supported"
        )
    if request.video_num_frames is not None:
        raise NotImplementedError("depth_anything_v2 does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("depth_anything_v2 does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("depth_anything_v2 does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise ValueError("depth_anything_v2 does not support context parallelism")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("depth_anything_v2 does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("depth_anything_v2 does not support mixed-precision layers")
    if request.max_sequence_length not in (None, 1):
        raise NotImplementedError("depth_anything_v2 does not support max_sequence_length")

    precision = str(request.precision).lower()
    model_dir = Path(request.model_dir)
    import json

    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if str(raw.get("model_type", "")).lower() != "depth_anything":
        raise ValueError(f"depth_anything_v2 does not support model_type={raw.get('model_type')!r}")

    readers = open_checkpoint(model_dir)
    cfg = resolve_config(raw, readers)
    weights = load_weights(str(model_dir), cfg, precision)

    writer.set_header(family="depth_anything_v2", task=request.task, backend=request.backend)
    plan = build_engine(cfg, weights, precision=precision, verbose=bool(request.verbose))
    writer.add_bytes("engine.plan", plan)

    preprocessor_path = model_dir / "preprocessor_config.json"
    preprocessor = (
        json.loads(preprocessor_path.read_text(encoding="utf-8")) if preprocessor_path.is_file() else {}
    )
    writer.add_json(
        "runtime.json",
        {
            "input_image_size": cfg["image_size"],
            "image_mean": preprocessor.get("image_mean", [0.485, 0.456, 0.406]),
            "image_std": preprocessor.get("image_std", [0.229, 0.224, 0.225]),
        },
    )
