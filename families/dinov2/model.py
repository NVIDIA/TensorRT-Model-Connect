# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT DINOv2 image-feature family.

This owner implements the Transformers ``Dinov2Model`` and
``Dinov2WithRegistersModel`` encoders.  Network construction uses TensorRT's
Python Network API directly; there is no ONNX export or parser.  The engine
has one fixed input resolution: the checkpoint image processor's center-crop
size.  Learned position embeddings are resized to that grid at build time with
the same bicubic resampling the Transformers model applies at run time.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from safetensors import safe_open

import tensorrt as trt

from . import graph_ops


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


TASK = "image_to_token_and_pooled_features"
_ARCHITECTURES = {
    "dinov2": "Dinov2Model",
    "dinov2_with_registers": "Dinov2WithRegistersModel",
}
_PIL_BICUBIC = 3


def _positive_int(raw: dict, name: str, default: int | None = None) -> int:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"DINOv2 {name} must be a positive integer, got {value!r}")
    return value


def _square(raw: dict, name: str) -> int:
    value = raw.get(name)
    if isinstance(value, (list, tuple)):
        if len(value) != 2 or value[0] != value[1]:
            raise ValueError(f"DINOv2 {name} must be square, got {value!r}")
        value = value[0]
    return _positive_int({name: value}, name)


def resolve_model_config(raw: dict) -> dict:
    """Validate the exact Transformers DINOv2 encoder contract."""
    model_type = str(raw.get("model_type", ""))
    if model_type not in _ARCHITECTURES:
        raise ValueError(f"DINOv2 does not support model_type={model_type!r}")
    architectures = list(raw.get("architectures") or [])
    if architectures != [_ARCHITECTURES[model_type]]:
        raise ValueError(
            f"DINOv2 {model_type} builds only {_ARCHITECTURES[model_type]} encoder checkpoints, "
            f"got architectures={architectures!r}"
        )
    hidden_size = _positive_int(raw, "hidden_size")
    num_heads = _positive_int(raw, "num_attention_heads")
    if hidden_size % num_heads:
        raise ValueError("DINOv2 hidden_size must be divisible by num_attention_heads")
    if raw.get("num_channels", 3) != 3:
        raise ValueError("DINOv2 supports only three-channel RGB input")
    hidden_act = str(raw.get("hidden_act", "gelu"))
    use_swiglu = bool(raw.get("use_swiglu_ffn", False))
    if not use_swiglu and hidden_act != "gelu":
        raise ValueError(f"Unsupported DINOv2 hidden_act: {hidden_act!r}")
    mlp_ratio = float(raw.get("mlp_ratio", 4))
    intermediate = int(hidden_size * mlp_ratio)
    if use_swiglu:
        # Dinov2SwiGLUFFN rounds two thirds of the dense width up to a multiple of eight.
        intermediate = (int(intermediate * 2 / 3) + 7) // 8 * 8
    return {
        "model_type": model_type,
        "pretrained_image_size": _square(raw, "image_size"),
        "patch_size": _square(raw, "patch_size"),
        "hidden_size": hidden_size,
        "num_hidden_layers": _positive_int(raw, "num_hidden_layers"),
        "num_attention_heads": num_heads,
        "head_dim": hidden_size // num_heads,
        "intermediate_size": intermediate,
        "hidden_act": hidden_act,
        "use_swiglu_ffn": use_swiglu,
        "qkv_bias": bool(raw.get("qkv_bias", True)),
        "layer_norm_eps": float(raw.get("layer_norm_eps", 1e-6)),
        "num_register_tokens": (
            _positive_int(raw, "num_register_tokens")
            if model_type == "dinov2_with_registers"
            else 0
        ),
        # Dinov2WithRegistersEmbeddings always resizes with antialiasing;
        # Dinov2Embeddings never does. Neither reads a config switch.
        "antialias": model_type == "dinov2_with_registers",
    }


def resolve_preprocess_config(raw: dict, patch_size: int) -> dict:
    """Validate the checkpoint's BitImageProcessor resize/crop/normalize contract."""
    # The runtime reproduces the Pillow resampler exactly; the torchvision-based
    # BitImageProcessorFast resamples differently and is not claimed.
    processor = str(raw.get("image_processor_type", ""))
    if processor != "BitImageProcessor":
        raise ValueError(f"DINOv2 requires BitImageProcessor preprocessing, got {processor!r}")
    for flag in ("do_resize", "do_center_crop", "do_rescale", "do_normalize"):
        if raw.get(flag) is not True:
            raise ValueError(f"DINOv2 preprocessing requires {flag}=true")
    if raw.get("resample") != _PIL_BICUBIC:
        raise ValueError(
            f"DINOv2 preprocessing supports bicubic resample, got {raw.get('resample')!r}"
        )
    if not math.isclose(float(raw.get("rescale_factor", 0.0)), 1.0 / 255.0, rel_tol=1e-6):
        raise ValueError("DINOv2 preprocessing requires rescale_factor=1/255")
    size = raw.get("size") or {}
    if set(size) != {"shortest_edge"}:
        raise ValueError(f"DINOv2 preprocessing requires size.shortest_edge, got {size!r}")
    crop = raw.get("crop_size") or {}
    if set(crop) != {"height", "width"}:
        raise ValueError(f"DINOv2 preprocessing requires crop_size height/width, got {crop!r}")
    shortest_edge = _positive_int(size, "shortest_edge")
    crop_h = _positive_int(crop, "height")
    crop_w = _positive_int(crop, "width")
    if crop_h % patch_size or crop_w % patch_size:
        raise ValueError("DINOv2 crop size must be divisible by patch_size")
    if crop_h > shortest_edge or crop_w > shortest_edge:
        raise ValueError("DINOv2 crop size must fit inside the resized shortest edge")
    mean = [float(value) for value in raw.get("image_mean", [])]
    std = [float(value) for value in raw.get("image_std", [])]
    if len(mean) != 3 or len(std) != 3 or not all(value > 0.0 for value in std):
        raise ValueError("DINOv2 preprocessing requires three-channel mean and positive std")
    return {
        "input_image_h": crop_h,
        "input_image_w": crop_w,
        "resize_shortest_edge": shortest_edge,
        "image_mean": mean,
        "image_std": std,
    }


def _torch_cubic_weights(t: np.ndarray) -> np.ndarray:
    """Four-tap Keys cubic weights (A=-0.75) used by non-antialiased torch bicubic."""
    a = -0.75

    def near(x):
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0

    def far(x):
        return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a

    return np.stack([far(t + 1.0), near(t), near(1.0 - t), far(2.0 - t)], axis=-1)


def _bicubic_matrix(input_size: int, output_size: int) -> np.ndarray:
    """torch.nn.functional.interpolate(mode="bicubic", align_corners=False) as a matrix."""
    scale = input_size / output_size
    source = scale * (np.arange(output_size, dtype=np.float64) + 0.5) - 0.5
    first = np.floor(source).astype(np.int64)
    weights = _torch_cubic_weights(source - first)
    matrix = np.zeros((output_size, input_size), dtype=np.float64)
    for tap in range(4):
        index = np.clip(first - 1 + tap, 0, input_size - 1)
        np.add.at(matrix, (np.arange(output_size), index), weights[:, tap])
    return matrix


def _antialiased_bicubic_matrix(input_size: int, output_size: int) -> np.ndarray:
    """torch bicubic interpolation with antialias=True (Pillow-style, A=-0.5) as a matrix."""

    def cubic(x):
        a = -0.5
        x = np.abs(x)
        return np.where(
            x < 1.0,
            ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0,
            np.where(x < 2.0, ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a, 0.0),
        )

    scale = input_size / output_size
    support = 2.0 * scale if scale >= 1.0 else 2.0
    inverse = 1.0 / scale if scale >= 1.0 else 1.0
    matrix = np.zeros((output_size, input_size), dtype=np.float64)
    for output_index in range(output_size):
        center = scale * (output_index + 0.5)
        begin = max(int(center - support + 0.5), 0)
        end = min(int(center + support + 0.5), input_size)
        taps = cubic((np.arange(begin, end) - center + 0.5) * inverse)
        total = taps.sum()
        if total != 0.0:
            taps = taps / total
        matrix[output_index, begin:end] = taps
    return matrix


def interpolate_position_embeddings(
    position_embeddings: np.ndarray,
    grid_h: int,
    grid_w: int,
    *,
    antialias: bool,
) -> np.ndarray:
    """Return ``[1, 1 + grid_h * grid_w, hidden]`` for the engine's fixed patch grid."""
    if position_embeddings.ndim != 3 or position_embeddings.shape[0] != 1:
        raise ValueError(
            f"DINOv2 position embeddings must be [1, N, D], got {position_embeddings.shape}"
        )
    num_positions = position_embeddings.shape[1] - 1
    source = math.isqrt(num_positions)
    if source * source != num_positions:
        raise ValueError(f"DINOv2 position grid is not square: {num_positions} patches")
    class_embedding = position_embeddings[:, :1].astype(np.float32)
    patches = position_embeddings[0, 1:].astype(np.float32)
    if (grid_h, grid_w) == (source, source):
        # Transformers returns the learned table unchanged when the grid already matches.
        return np.ascontiguousarray(position_embeddings, dtype=np.float32)
    build = _antialiased_bicubic_matrix if antialias else _bicubic_matrix
    rows = build(source, grid_h)
    columns = build(source, grid_w)
    grid = patches.astype(np.float64).reshape(source, source, -1)
    resized = np.einsum("ys,sth,xt->yxh", rows, grid, columns, optimize=True)
    resized = resized.reshape(1, grid_h * grid_w, -1).astype(np.float32)
    return np.ascontiguousarray(np.concatenate([class_embedding, resized], axis=1))


def _target_dtype(precision: str) -> tuple[np.dtype, trt.DataType]:
    if precision == "fp16":
        return np.dtype(np.float16), trt.float16
    if precision == "fp32":
        return np.dtype(np.float32), trt.float32
    raise ValueError(f"DINOv2 supports fp16 and fp32 precision, got {precision!r}")


class _Checkpoint:
    def __init__(self, model_dir: Path):
        single = model_dir / "model.safetensors"
        if single.is_file():
            files = [single]
            owner = {}
        else:
            index_path = model_dir / "model.safetensors.index.json"
            if not index_path.is_file():
                raise FileNotFoundError(
                    f"DINOv2 requires model.safetensors or model.safetensors.index.json in {model_dir}"
                )
            owner = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
            files = [model_dir / name for name in sorted(set(owner.values()))]
        readers = {path.name: safe_open(str(path), framework="numpy") for path in files}
        self._readers = {
            name: readers[owner.get(name, files[0].name)]
            for reader in readers.values()
            for name in reader.keys()
        }

    def __contains__(self, name: str) -> bool:
        return name in self._readers

    def get(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        reader = self._readers.get(name)
        if reader is None:
            raise KeyError(f"DINOv2 checkpoint tensor not found: {name}")
        value = reader.get_tensor(name)
        if tuple(value.shape) != shape:
            raise ValueError(f"DINOv2 tensor {name} has shape {value.shape}, expected {shape}")
        return np.asarray(value, dtype=np.float32)


def load_weights(model_dir: Path, cfg: dict) -> dict[str, np.ndarray]:
    """Map the Transformers checkpoint to logical float32 weights."""
    checkpoint = _Checkpoint(model_dir)
    hidden = cfg["hidden_size"]
    patch = cfg["patch_size"]
    grid = cfg["pretrained_image_size"] // patch
    weights = {
        "cls_token": checkpoint.get("embeddings.cls_token", (1, 1, hidden)),
        "position_embeddings": checkpoint.get(
            "embeddings.position_embeddings", (1, 1 + grid * grid, hidden)
        ),
        "patch.weight": checkpoint.get(
            "embeddings.patch_embeddings.projection.weight", (hidden, 3, patch, patch)
        ),
        "patch.bias": checkpoint.get("embeddings.patch_embeddings.projection.bias", (hidden,)),
        "norm.weight": checkpoint.get("layernorm.weight", (hidden,)),
        "norm.bias": checkpoint.get("layernorm.bias", (hidden,)),
    }
    if cfg["num_register_tokens"]:
        weights["register_tokens"] = checkpoint.get(
            "embeddings.register_tokens", (1, cfg["num_register_tokens"], hidden)
        )
    elif "embeddings.register_tokens" in checkpoint:
        raise ValueError("DINOv2 checkpoint has register tokens its config does not declare")

    intermediate = cfg["intermediate_size"]
    for layer in range(cfg["num_hidden_layers"]):
        source = f"encoder.layer.{layer}"
        target = f"layer.{layer}"

        def linear(name: str, out_features: int, in_features: int) -> np.ndarray:
            # Store as [in, out] for a right-hand matrix multiply.
            return checkpoint.get(f"{source}.{name}.weight", (out_features, in_features)).T

        def vector(name: str, size: int) -> np.ndarray:
            return checkpoint.get(f"{source}.{name}", (size,))

        for norm in ("norm1", "norm2"):
            weights[f"{target}.{norm}.weight"] = vector(f"{norm}.weight", hidden)
            weights[f"{target}.{norm}.bias"] = vector(f"{norm}.bias", hidden)
        for scale in ("layer_scale1", "layer_scale2"):
            weights[f"{target}.{scale}"] = vector(f"{scale}.lambda1", hidden)

        weights[f"{target}.qkv.weight"] = np.concatenate(
            [
                linear(f"attention.attention.{name}", hidden, hidden)
                for name in ("query", "key", "value")
            ],
            axis=1,
        )
        if cfg["qkv_bias"]:
            weights[f"{target}.qkv.bias"] = np.concatenate(
                [
                    vector(f"attention.attention.{name}.bias", hidden)
                    for name in ("query", "key", "value")
                ]
            )
        weights[f"{target}.attention_output.weight"] = linear(
            "attention.output.dense", hidden, hidden
        )
        weights[f"{target}.attention_output.bias"] = vector("attention.output.dense.bias", hidden)

        if cfg["use_swiglu_ffn"]:
            weights[f"{target}.mlp_in.weight"] = linear("mlp.weights_in", 2 * intermediate, hidden)
            weights[f"{target}.mlp_in.bias"] = vector("mlp.weights_in.bias", 2 * intermediate)
            weights[f"{target}.mlp_out.weight"] = linear("mlp.weights_out", hidden, intermediate)
            weights[f"{target}.mlp_out.bias"] = vector("mlp.weights_out.bias", hidden)
        else:
            weights[f"{target}.mlp_in.weight"] = linear("mlp.fc1", intermediate, hidden)
            weights[f"{target}.mlp_in.bias"] = vector("mlp.fc1.bias", intermediate)
            weights[f"{target}.mlp_out.weight"] = linear("mlp.fc2", hidden, intermediate)
            weights[f"{target}.mlp_out.bias"] = vector("mlp.fc2.bias", hidden)
    return weights


def build_engine(
    cfg: dict,
    weights: dict[str, np.ndarray],
    image_h: int,
    image_w: int,
    *,
    precision: str,
    verbose: bool,
) -> bytes:
    work_dtype, work_trt_dtype = _target_dtype(precision)
    builder, network, builder_config = graph_ops.new_network(verbose)

    patch_size = cfg["patch_size"]
    hidden_size = cfg["hidden_size"]
    num_heads = cfg["num_attention_heads"]
    head_dim = cfg["head_dim"]
    num_registers = cfg["num_register_tokens"]
    grid_h = image_h // patch_size
    grid_w = image_w // patch_size
    num_patches = grid_h * grid_w
    sequence_length = 1 + num_registers + num_patches
    if verbose:
        print(
            f"[trtmc build] {cfg['model_type']}: image={image_h}x{image_w}, patch={patch_size}, "
            f"tokens={sequence_length}, hidden={hidden_size}, "
            f"layers={cfg['num_hidden_layers']}, heads={num_heads}, "
            f"registers={num_registers}, precision={precision}",
            file=sys.stderr,
        )

    pixel_values = network.add_input("pixel_values", trt.float32, (1, 3, image_h, image_w))
    pixels = graph_ops.cast(network, pixel_values, work_trt_dtype)
    patch = network.add_convolution_nd(
        pixels,
        hidden_size,
        (patch_size, patch_size),
        trt.Weights(np.ascontiguousarray(weights["patch.weight"], dtype=work_dtype)),
        trt.Weights(np.ascontiguousarray(weights["patch.bias"], dtype=work_dtype)),
    )
    patch.stride_nd = (patch_size, patch_size)
    patch_tokens = graph_ops.shuffle(
        network,
        patch.get_output(0),
        first_transpose=(0, 2, 3, 1),
        reshape_dims=(1, num_patches, hidden_size),
    )

    # The residual stream, LayerNorm, LayerScale, attention scores and the MLP
    # down-projection stay in float32 at every precision; the remaining matrix
    # multiplies run at `precision`.
    stream = np.dtype(np.float32)

    def stream_constant(name: str, shape: tuple[int, ...]):
        return graph_ops.constant(network, weights[name], shape, stream)

    def to_stream(tensor):
        return graph_ops.cast(network, tensor, trt.float32)

    def to_work(tensor):
        return graph_ops.cast(network, tensor, work_trt_dtype)

    # Transformers adds position embeddings to [CLS, patches] and only then
    # inserts register tokens, which carry no position embedding.
    concat = network.add_concatenation(
        [stream_constant("cls_token", (1, 1, hidden_size)), to_stream(patch_tokens)]
    )
    concat.axis = 1
    positions = stream_constant("position_embeddings", (1, 1 + num_patches, hidden_size))
    embedded = network.add_elementwise(
        concat.get_output(0), positions, trt.ElementWiseOperation.SUM
    ).get_output(0)
    if num_registers:
        cls = graph_ops.slice_tensor(network, embedded, (0, 0, 0), (1, 1, hidden_size))
        patches = graph_ops.slice_tensor(
            network, embedded, (0, 1, 0), (1, num_patches, hidden_size)
        )
        concat = network.add_concatenation(
            [cls, stream_constant("register_tokens", (1, num_registers, hidden_size)), patches]
        )
        concat.axis = 1
        embedded = concat.get_output(0)
    hidden = embedded

    def to_heads(tensor):
        return graph_ops.shuffle(
            network,
            tensor,
            reshape_dims=(1, sequence_length, num_heads, head_dim),
            second_transpose=(0, 2, 1, 3),
        )

    def normalize(name: str, tensor):
        return graph_ops.layer_norm(
            network,
            tensor,
            hidden_size,
            weights[f"{name}.weight"],
            weights[f"{name}.bias"],
            cfg["layer_norm_eps"],
            stream,
        )

    for layer in range(cfg["num_hidden_layers"]):
        prefix = f"layer.{layer}"

        def project(name: str, tensor, dtype=work_dtype):
            return graph_ops.linear_with_bias(network, tensor, weights, f"{prefix}.{name}", dtype)

        def add_residual(residual, tensor, scale: str):
            return graph_ops.add_scaled_residual(
                network, residual, to_stream(tensor), weights[f"{prefix}.{scale}"], stream
            )

        qkv = project("qkv", to_work(normalize(f"{prefix}.norm1", hidden)))
        q, k, v = (
            to_heads(
                graph_ops.slice_tensor(
                    network, qkv, (0, 0, index * hidden_size), (1, sequence_length, hidden_size)
                )
            )
            for index in range(3)
        )
        context = graph_ops.attention(network, q, k, v, head_dim)
        merged = graph_ops.shuffle(
            network,
            context,
            first_transpose=(0, 2, 1, 3),
            reshape_dims=(1, sequence_length, hidden_size),
        )
        hidden = add_residual(hidden, project("attention_output", merged), "layer_scale1")

        expanded = project("mlp_in", to_work(normalize(f"{prefix}.norm2", hidden)))
        if cfg["use_swiglu_ffn"]:
            width = cfg["intermediate_size"]
            gate = graph_ops.slice_tensor(network, expanded, (0, 0, 0), (1, sequence_length, width))
            value = graph_ops.slice_tensor(
                network, expanded, (0, 0, width), (1, sequence_length, width)
            )
            activated = network.add_elementwise(
                graph_ops.silu(network, gate), value, trt.ElementWiseOperation.PROD
            ).get_output(0)
        else:
            activated = graph_ops.activation(network, expanded, cfg["hidden_act"], work_dtype)
        # The MLP down-projection writes the largest-magnitude updates to the
        # stream; rounding them to half precision dominates the fp16 error.
        hidden = add_residual(
            hidden, project("mlp_out", to_stream(activated), stream), "layer_scale2"
        )

    hidden = normalize("norm", hidden)
    last_hidden_state = graph_ops.cast(network, hidden, trt.float32)
    last_hidden_state.name = "last_hidden_state"
    network.mark_output(last_hidden_state)

    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT DINOv2 engine build failed")
    return bytes(plan)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"DINOv2 requires {path.name} in {path.parent}")
    return json.loads(path.read_text(encoding="utf-8"))


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one DINOv2 image-feature bundle."""
    if request.task != TASK:
        raise ValueError(f"dinov2 supports only task={TASK}")
    if request.backend != "trt":
        raise ValueError("dinov2 supports only the trt backend")
    if request.dynamic_kv_cache:
        raise NotImplementedError("dinov2 does not support dynamic_kv_cache")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError(
            "dinov2 builds the checkpoint processor's crop size; image_height/image_width are unsupported"
        )
    if request.video_num_frames is not None:
        raise NotImplementedError("dinov2 does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("dinov2 does not support max_batch_size")
    if request.max_sequence_length not in (None, 1):
        raise NotImplementedError("dinov2 has no sequence-length option")
    if request.context_parallel_size != 1:
        raise ValueError("dinov2 does not support context parallelism")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("dinov2 does not support tensor parallelism")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("dinov2 does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("dinov2 does not support mixed-precision layers")
    precision = str(request.precision).lower()
    _target_dtype(precision)

    model_dir = Path(request.model_dir)
    cfg = resolve_model_config(_read_json(model_dir / "config.json"))
    preprocess = resolve_preprocess_config(
        _read_json(model_dir / "preprocessor_config.json"), cfg["patch_size"]
    )
    image_h = preprocess["input_image_h"]
    image_w = preprocess["input_image_w"]
    weights = load_weights(model_dir, cfg)
    weights["position_embeddings"] = interpolate_position_embeddings(
        weights["position_embeddings"],
        image_h // cfg["patch_size"],
        image_w // cfg["patch_size"],
        antialias=cfg["antialias"],
    )

    writer.set_header(family="dinov2", task=request.task, backend=request.backend)
    writer.add_bytes(
        "engine.plan",
        build_engine(
            cfg,
            weights,
            image_h,
            image_w,
            precision=precision,
            verbose=bool(request.verbose),
        ),
    )
    writer.add_json(
        "runtime.json",
        {
            **preprocess,
            "patch_size": cfg["patch_size"],
            "hidden_size": cfg["hidden_size"],
            "num_register_tokens": cfg["num_register_tokens"],
        },
    )
