# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ViTPose checkpoint this family builds, read and checked in one place.

Only what the graph reproduces is accepted: the plain ViT backbone of `VitPoseBackbone` (one expert, exact GELU,
biased QKV) read at its last layer, the simple decoder head (ReLU, bilinear upsampling, 3 x 3 convolution), float32
weights, and the box-crop image processor. The configuration is checked against the tensor header before any weight
is read, so a file whose tensors disagree with its configuration is rejected rather than built from its weights.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WEIGHTS_FILE = "model.safetensors"
# The processor encodes boxes and warps crops in units of this many pixels (VitPoseImageProcessor.normalize_factor
# and the fixed 200 of its affine_transform and transform_preds).
PIXEL_STD = 200.0
# box_to_center_and_scale pads every box by this factor; post_process_pose_estimation refines with this kernel.
BOX_PADDING = 1.25
DARK_KERNEL = 11
# VitPoseBackboneConfig defaults, used where the checkpoint's backbone_config leaves a value out.
_BACKBONE_DEFAULTS = {
    "image_size": [256, 192],
    "patch_size": [16, 16],
    "num_channels": 3,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "mlp_ratio": 4,
    "num_experts": 1,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-12,
    "qkv_bias": True,
}
_PROCESSOR_TYPES = ("VitPoseImageProcessor", "VitPoseImageProcessorFast")
_PROCESSOR_KEYS = frozenset(
    {
        "do_affine_transform",
        "do_normalize",
        "do_rescale",
        "image_mean",
        "image_processor_type",
        "image_std",
        "normalize_factor",
        "processor_class",
        "rescale_factor",
        "size",
    }
)
# The patch embedding's padding in VitPoseBackbonePatchEmbeddings.
PATCH_PADDING = 2
_INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class VitPoseConfig:
    image_height: int
    image_width: int
    patch_height: int
    patch_width: int
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    layer_norm_eps: float
    scale_factor: int
    labels: tuple[str, ...]
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    rescale_factor: float

    @property
    def grid(self) -> tuple[int, int]:
        return self.image_height // self.patch_height, self.image_width // self.patch_width

    @property
    def heatmap_size(self) -> tuple[int, int]:
        rows, columns = self.grid
        return rows * self.scale_factor, columns * self.scale_factor

    @property
    def num_keypoints(self) -> int:
        return len(self.labels)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"repeats the key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"vitpose checkpoint is missing {path.name}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    except ValueError as error:
        raise ValueError(f"vitpose {path.name} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"vitpose {path.name} must be a JSON object")
    return value


def _unsupported(condition: bool, message: str) -> None:
    if not condition:
        raise NotImplementedError(f"vitpose does not support this checkpoint: {message}")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value, name: str) -> int:
    if not _is_int(value) or value <= 0:
        raise ValueError(f"vitpose {name} must be a positive integer, got {value!r}")
    return value


def _finite_number(value, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"vitpose {name} must be a number, got {value!r}")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"vitpose {name} must be finite, got {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"vitpose {name} must be finite, got {value!r}")
    return number


# Like the runtime, a value narrowed to float32 is compared with FLT_MAX in double: numpy would round a value a little
# above it down to FLT_MAX, which the runtime refuses.
_FLOAT32_MAX = float(np.finfo(np.float32).max)


def _fits_float32(number: float) -> bool:
    return math.isfinite(number) and abs(number) <= _FLOAT32_MAX


def _positive_float32(value, name: str) -> float:
    """A positive finite number that stays positive and finite as the float32 the runtime stores."""

    number = _finite_number(value, name)
    with np.errstate(under="ignore"):
        stored = np.float32(number) if _fits_float32(number) else np.float32(0)
    if not (number > 0 and stored > 0):
        raise ValueError(f"vitpose {name} must be a positive float32, got {value!r}")
    return number


def _bool(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"vitpose {name} must be true or false, got {value!r}")
    return value


def _pair(value, name: str) -> tuple[int, int]:
    """An (height, width) pair, written as one integer or a list of two as VitPoseBackboneConfig accepts them."""

    if _is_int(value):
        value = [value, value]
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"vitpose {name} must be an integer or a list of two, got {value!r}")
    return _positive_int(value[0], name), _positive_int(value[1], name)


def _channels(value, name: str, *, positive: bool) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"vitpose {name} must hold three numbers, got {value!r}")
    if positive:
        return tuple(_positive_float32(item, name) for item in value)
    result = []
    for item in value:
        number = _finite_number(item, name)
        if not _fits_float32(number):
            raise ValueError(f"vitpose {name} must fit float32, got {item!r}")
        result.append(number)
    return tuple(result)


def _get(config: dict, key: str, default):
    """A key's value or its default; an explicit null is not an absent key."""

    if key not in config:
        return default
    if config[key] is None:
        raise ValueError(f"vitpose config value {key} must not be null")
    return config[key]


def _labels(config: dict) -> tuple[str, ...]:
    id2label = config.get("id2label")
    if not isinstance(id2label, dict) or not id2label:
        raise ValueError("vitpose config.json must name every keypoint in id2label")
    labels = []
    for index in range(len(id2label)):
        key = str(index)
        if key not in id2label:
            raise ValueError(
                f"vitpose id2label must be keyed 0..{len(id2label) - 1}; {key} is missing"
            )
        name = id2label[key]
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError(f"vitpose id2label[{key}] must be a non-empty name, got {name!r}")
        labels.append(name)
    if len(set(labels)) != len(labels):
        raise ValueError("vitpose id2label repeats a keypoint name")
    return tuple(labels)


def _selected_stage(backbone: dict, layers: int) -> None:
    """The head reads the last of the backbone's selected stages; only the final layer is built."""

    last_stage = f"stage{layers}"
    indices = backbone.get("out_indices")
    features = backbone.get("out_features")
    if indices is None and features is None:
        return
    if indices is not None:
        if not isinstance(indices, list) or not all(_is_int(item) for item in indices):
            raise ValueError(
                f"vitpose backbone out_indices must be a list of integers, got {indices!r}"
            )
        _unsupported(
            indices == [layers], f"the head must read stage {layers}, not out_indices {indices}"
        )
    if features is not None:
        if not isinstance(features, list) or not all(isinstance(item, str) for item in features):
            raise ValueError(
                f"vitpose backbone out_features must be a list of names, got {features!r}"
            )
        _unsupported(
            features == [last_stage],
            f"the head must read {last_stage}, not out_features {features}",
        )


def _check_layer_count(header: dict, layers: int) -> None:
    """The encoder layers of the header, counted before anything is sized from the configuration's count."""

    prefix = "backbone.encoder.layer."
    indices = set()
    for name in header:
        if name.startswith(prefix):
            index = name[len(prefix) :].split(".", 1)[0]
            if not index.isdecimal() or index != str(int(index)):
                raise ValueError(f"vitpose checkpoint has a malformed encoder layer tensor: {name}")
            indices.add(int(index))
    if indices != set(range(len(indices))):
        raise ValueError("vitpose checkpoint encoder layers are not numbered 0 .. n - 1")
    if len(indices) != layers:
        raise ValueError(
            f"vitpose config.json says {layers} layers but the checkpoint holds {len(indices)}"
        )


def _model_config(config: dict) -> dict:
    _unsupported(config.get("model_type") == "vitpose", f"model_type {config.get('model_type')!r}")
    architectures = _get(config, "architectures", ["VitPoseForPoseEstimation"])
    _unsupported(
        isinstance(architectures, list) and "VitPoseForPoseEstimation" in architectures,
        f"architectures {architectures!r}",
    )
    _unsupported(
        _bool(_get(config, "use_simple_decoder", True), "use_simple_decoder"),
        "only the simple decoder head is built",
    )
    _unsupported(config.get("backbone") is None, "a named backbone replaces the VitPose backbone")
    _unsupported(
        not _bool(_get(config, "use_timm_backbone", False), "use_timm_backbone"),
        "a timm backbone replaces the VitPose backbone",
    )
    backbone = config.get("backbone_config")
    # Without backbone_config the reference reads stage 4 of a default backbone, not its last layer.
    _unsupported(isinstance(backbone, dict), "config.json must state backbone_config")
    _unsupported(
        backbone.get("model_type") == "vitpose_backbone",
        f"backbone model_type {backbone.get('model_type')!r}",
    )
    return backbone


def _expected_tensors(cfg: VitPoseConfig) -> dict[str, tuple[int, ...]]:
    hidden, inner = cfg.hidden_size, cfg.intermediate_size
    rows, columns = cfg.grid
    tensors = {
        "backbone.embeddings.patch_embeddings.projection.weight": (
            hidden,
            3,
            cfg.patch_height,
            cfg.patch_width,
        ),
        "backbone.embeddings.patch_embeddings.projection.bias": (hidden,),
        "backbone.embeddings.position_embeddings": (1, rows * columns + 1, hidden),
        "backbone.layernorm.weight": (hidden,),
        "backbone.layernorm.bias": (hidden,),
        "head.conv.weight": (cfg.num_keypoints, hidden, 3, 3),
        "head.conv.bias": (cfg.num_keypoints,),
    }
    for index in range(cfg.num_layers):
        layer = f"backbone.encoder.layer.{index}"
        for name in ("query", "key", "value"):
            tensors[f"{layer}.attention.attention.{name}.weight"] = (hidden, hidden)
            tensors[f"{layer}.attention.attention.{name}.bias"] = (hidden,)
        tensors[f"{layer}.attention.output.dense.weight"] = (hidden, hidden)
        tensors[f"{layer}.attention.output.dense.bias"] = (hidden,)
        for norm in ("layernorm_before", "layernorm_after"):
            tensors[f"{layer}.{norm}.weight"] = (hidden,)
            tensors[f"{layer}.{norm}.bias"] = (hidden,)
        tensors[f"{layer}.mlp.fc1.weight"] = (inner, hidden)
        tensors[f"{layer}.mlp.fc1.bias"] = (inner,)
        tensors[f"{layer}.mlp.fc2.weight"] = (hidden, inner)
        tensors[f"{layer}.mlp.fc2.bias"] = (hidden,)
    return tensors


def check_tensors(cfg: VitPoseConfig, header: dict[str, tuple[str, tuple[int, ...]]]) -> None:
    """Every tensor the configuration implies, with its shape, as float32, and no other tensor."""

    expected = _expected_tensors(cfg)
    missing = sorted(set(expected) - set(header))
    extra = sorted(set(header) - set(expected))
    if missing:
        raise ValueError(f"vitpose checkpoint lacks {len(missing)} tensors, first {missing[0]}")
    if extra:
        raise NotImplementedError(
            f"vitpose checkpoint has tensors this graph does not use, first {extra[0]}"
        )
    for name, shape in expected.items():
        dtype, stored = header[name]
        if dtype != "F32":
            raise NotImplementedError(f"vitpose builds float32 checkpoints; {name} is {dtype}")
        if tuple(stored) != shape:
            raise ValueError(
                f"vitpose {name} has shape {tuple(stored)}, the configuration implies {shape}"
            )


def _check_normalization(processor: dict) -> None:
    """Every 8-bit value must normalize to a finite float32 the way the runtime computes it.

    The runtime folds the rescale into the statistics in float32, ``mean * (1 / rescale)`` and
    ``std * (1 / rescale)``, then computes ``(value - mean') / std'`` (runtime/image_ops.cpp).
    """

    if not _fits_float32(1.0 / processor["rescale_factor"]):
        raise ValueError("vitpose 1 / rescale_factor must fit float32")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        factor = np.float32(1.0 / processor["rescale_factor"])
        mean = np.asarray(processor["image_mean"], dtype=np.float32) * factor
        std = np.asarray(processor["image_std"], dtype=np.float32) * factor
        values = (np.arange(256, dtype=np.float32)[:, None] - mean[None, :]) / std[None, :]
    if not (
        np.isfinite(factor)
        and np.isfinite(mean).all()
        and np.isfinite(std).all()
        and (std != 0).all()
        and np.isfinite(values).all()
    ):
        raise ValueError(
            "vitpose rescale_factor, image_mean and image_std must normalize 8-bit values to finite float32 values"
        )


def _processor(model_dir: Path, image_size: tuple[int, int]) -> dict:
    processor = _read_json(model_dir / "preprocessor_config.json")
    unknown = sorted(set(processor) - _PROCESSOR_KEYS)
    _unsupported(not unknown, f"preprocessor_config.json sets {unknown}")
    _unsupported(
        processor.get("image_processor_type") in _PROCESSOR_TYPES,
        f"image_processor_type {processor.get('image_processor_type')!r}",
    )
    for key in ("do_affine_transform", "do_rescale", "do_normalize"):
        _unsupported(_bool(_get(processor, key, True), key), f"{key} is off")
    size = processor.get("size")
    if not isinstance(size, dict) or set(size) != {"height", "width"}:
        raise ValueError(f"vitpose preprocessor size must be {{height, width}}, got {size!r}")
    crop = _positive_int(size["height"], "size.height"), _positive_int(size["width"], "size.width")
    _unsupported(
        crop == image_size, f"the processor crops {crop} but the backbone takes {image_size}"
    )
    normalize_factor = _finite_number(
        _get(processor, "normalize_factor", PIXEL_STD), "normalize_factor"
    )
    _unsupported(
        normalize_factor == PIXEL_STD,
        f"normalize_factor {normalize_factor}: the warp and keypoint transforms use {PIXEL_STD}",
    )
    return {
        "rescale_factor": _positive_float32(
            _get(processor, "rescale_factor", 1 / 255), "rescale_factor"
        ),
        "image_mean": _channels(
            _get(processor, "image_mean", [0.485, 0.456, 0.406]), "image_mean", positive=False
        ),
        "image_std": _channels(
            _get(processor, "image_std", [0.229, 0.224, 0.225]), "image_std", positive=True
        ),
    }


def resolve(model_dir: Path, header: dict[str, tuple[str, tuple[int, ...]]]) -> VitPoseConfig:
    """Read config.json and preprocessor_config.json and check them against the tensor header."""

    model_dir = Path(model_dir)
    config = _read_json(model_dir / "config.json")
    backbone = _model_config(config)
    values = {key: _get(backbone, key, default) for key, default in _BACKBONE_DEFAULTS.items()}

    image_size = _pair(values["image_size"], "image_size")
    patch = _pair(values["patch_size"], "patch_size")
    _unsupported(
        _positive_int(values["num_channels"], "num_channels") == 3, "only RGB input is built"
    )
    hidden = _positive_int(values["hidden_size"], "hidden_size")
    layers = _positive_int(values["num_hidden_layers"], "num_hidden_layers")
    heads = _positive_int(values["num_attention_heads"], "num_attention_heads")
    if hidden % heads:
        raise ValueError(f"vitpose hidden_size {hidden} is not divisible by {heads} heads")
    ratio = _finite_number(values["mlp_ratio"], "mlp_ratio")
    intermediate = int(hidden * ratio)
    if intermediate <= 0:
        raise ValueError(f"vitpose mlp_ratio {ratio} leaves no MLP width")
    _unsupported(
        _positive_int(values["num_experts"], "num_experts") == 1,
        "the mixture-of-experts MLP (ViTPose+)",
    )
    _unsupported(
        values["hidden_act"] == "gelu",
        f"hidden_act {values['hidden_act']!r}; the graph is the exact GELU",
    )
    _unsupported(_bool(values["qkv_bias"], "qkv_bias"), "an unbiased QKV projection")
    eps = _positive_float32(values["layer_norm_eps"], "layer_norm_eps")
    _check_layer_count(header, layers)
    _selected_stage(backbone, layers)

    # The patch convolution (padding 2) must produce the grid the head reshapes to, image_size // patch_size.
    for axis, (size, step) in enumerate(zip(image_size, patch)):
        produced = (size + 2 * PATCH_PADDING - step) // step + 1
        _unsupported(
            produced == size // step,
            f"axis {axis}: the patch convolution gives {produced} rows of patches, the head reads {size // step}",
        )
    scale_factor = _positive_int(_get(config, "scale_factor", 4), "scale_factor")
    # The native runtime refuses crops and heatmaps smaller than 2 x 2 (the warp and the keypoint transform divide
    # by size - 1) and sizes beyond int32; a bundle that builds must also load.
    for name, size in (
        ("image_size", image_size),
        (
            "heatmap size",
            tuple(
                step * scale_factor
                for step in (image_size[0] // patch[0], image_size[1] // patch[1])
            ),
        ),
    ):
        _unsupported(
            all(2 <= value <= _INT32_MAX for value in size),
            f"{name} {size} is outside 2 .. {_INT32_MAX}",
        )
    processor = _processor(model_dir, image_size)
    _check_normalization(processor)

    cfg = VitPoseConfig(
        image_height=image_size[0],
        image_width=image_size[1],
        patch_height=patch[0],
        patch_width=patch[1],
        hidden_size=hidden,
        num_layers=layers,
        num_heads=heads,
        intermediate_size=intermediate,
        layer_norm_eps=eps,
        scale_factor=scale_factor,
        labels=_labels(config),
        image_mean=processor["image_mean"],
        image_std=processor["image_std"],
        rescale_factor=processor["rescale_factor"],
    )
    check_tensors(cfg, header)
    return cfg


def runtime_metadata(cfg: VitPoseConfig) -> dict:
    """The bundle's runtime.json: what the native pipeline needs to crop, normalize and decode."""

    heatmap_height, heatmap_width = cfg.heatmap_size
    return {
        "input_height": cfg.image_height,
        "input_width": cfg.image_width,
        "rescale_factor": cfg.rescale_factor,
        "image_mean": list(cfg.image_mean),
        "image_std": list(cfg.image_std),
        "box_padding": BOX_PADDING,
        "heatmap_height": heatmap_height,
        "heatmap_width": heatmap_width,
        "dark_kernel": DARK_KERNEL,
        "labels": list(cfg.labels),
    }
