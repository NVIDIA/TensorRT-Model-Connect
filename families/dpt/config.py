# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and check the DPT-Hybrid fields this family builds against.

Absent fields take the defaults Transformers 5.2.0 fills in (``DPTConfig``,
``BitConfig`` and ``DPTImageProcessor``). A field that is present keeps its
JSON type: ``768.0`` is not an integer, ``"false"`` not a boolean, and an
explicit ``null`` is a value, accepted only where Transformers accepts it.
Every value that would change the graph or the image pipeline outside what this
family implements is refused rather than guessed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

ARCHITECTURE = "DPTForDepthEstimation"

# DPTConfig defaults for the fields this family reads.
_DPT_DEFAULTS = {
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-12,
    "image_size": 384,
    "patch_size": 16,
    "num_channels": 3,
    "is_hybrid": False,
    "qkv_bias": True,
    "backbone_out_indices": [2, 5, 8, 11],
    "readout_type": "project",
    "reassemble_factors": [4, 2, 1, 0.5],
    "neck_hidden_sizes": [96, 192, 384, 768],
    "fusion_hidden_size": 256,
    "head_in_index": -1,
    "use_batch_norm_in_fusion_residual": False,
    "use_bias_in_fusion_residual": None,
    "add_projection": False,
    "backbone_featmap_shape": [1, 1024, 24, 24],
    "neck_ignore_stages": [0, 1],
    "backbone": None,
}

# BitConfig defaults, and the ones DPTConfig substitutes when a hybrid config
# carries no backbone_config at all (or null).
_BIT_DEFAULTS = {
    "num_channels": 3,
    "embedding_size": 64,
    "hidden_sizes": [256, 512, 1024, 2048],
    "depths": [3, 4, 6, 3],
    "layer_type": "preactivation",
    "hidden_act": "relu",
    "global_padding": None,
    "num_groups": 32,
    "embedding_dynamic_padding": False,
    "output_stride": 32,
    "width_factor": 1,
    "out_features": None,
    "out_indices": None,
}
_BIT_HYBRID_DEFAULTS = {
    "global_padding": "same",
    "layer_type": "bottleneck",
    "depths": [3, 4, 9],
    "out_features": ["stage1", "stage2", "stage3"],
    "embedding_dynamic_padding": True,
}

# DPTImageProcessor defaults. resample 2 is PIL bilinear, 3 bicubic.
_PROCESSOR_DEFAULTS = {
    "do_resize": True,
    "size": {"height": 384, "width": 384},
    "resample": 3,
    "keep_aspect_ratio": False,
    "ensure_multiple_of": 1,
    "do_rescale": True,
    "rescale_factor": 1 / 255,
    "do_normalize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "do_pad": False,
    "size_divisor": None,
}
_PIL_BILINEAR = 2

# BitEmbeddings and every BiT bottleneck convolution standardize their weights
# with these epsilons; GroupNorm uses its own.
STEM_WEIGHT_EPS = 1e-8
LAYER_WEIGHT_EPS = 1e-8
GROUP_NORM_EPS = 1e-5

# Engine dimensions are int32.
_INT32_MAX = 2**31 - 1
BIT_PREFIX = "dpt.embeddings.backbone.bit"
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT16_MAX = float(np.finfo(np.float16).max)


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"DPT JSON repeats the key {key!r}")
        result[key] = value
    return result


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"DPT JSON holds the non-finite number {text}")
    return value


def _no_constant(name: str):
    raise ValueError(f"DPT JSON holds the non-finite number {name}")


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing DPT file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys,
                       parse_float=_finite_float, parse_constant=_no_constant)
    if not isinstance(value, dict):
        raise ValueError(f"DPT file must hold a JSON object: {path}")
    return value


def load(model_dir: str | Path) -> dict:
    """Resolve the build configuration of the checkpoint under ``model_dir``."""
    root = Path(model_dir)
    return resolve(_read_json(root / "config.json"),
                   _read_json(root / "preprocessor_config.json"))


def _values(raw: dict, defaults: dict) -> dict:
    """Every field of ``defaults``: the raw value when the key is present, null included."""
    return {key: raw[key] if key in raw else default for key, default in defaults.items()}


def _integer(value, name: str, *, minimum: int | None = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"DPT {name} must be an integer; got {value!r}")
    if (minimum is not None and value < minimum) or abs(value) > _INT32_MAX:
        raise ValueError(f"DPT {name} is out of range: {value}")
    return value


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"DPT {name} must be a finite number; got {value!r}")
    return float(value)


def _flag(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"DPT {name} must be true or false; got {value!r}")
    return value


def _text(value, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"DPT {name} must be a string; got {value!r}")
    return value


def _integers(value, name: str, *, minimum: int | None = 1) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"DPT {name} must be a list; got {value!r}")
    return [_integer(item, name, minimum=minimum) for item in value]


def _numbers(value, name: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"DPT {name} must be a list; got {value!r}")
    return [_number(item, name) for item in value]


def _same_padding(size: int, kernel: int, stride: int) -> tuple[int, int]:
    """Before/after padding of BiT's dynamic TF-style SAME padding."""
    total = max((math.ceil(size / stride) - 1) * stride + kernel - size, 0)
    return total // 2, total - total // 2


def _conv_padding(size: int, kernel: int, stride: int) -> tuple[int, int]:
    """Padding of a BiT convolution: static when stride 1 and odd kernel, else dynamic."""
    if stride == 1 and (kernel - 1) % 2 == 0:
        return (kernel - 1) // 2, (kernel - 1) // 2
    return _same_padding(size, kernel, stride)


def _make_div(value: float, divisor: int = 8) -> int:
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _bit_out_features(values: dict, stage_count: int) -> list[str]:
    """The stages a BiT backbone returns, from out_features and out_indices as Transformers
    5.2.0 aligns them: either one alone decides, both must agree, neither means the last."""
    stage_names = ["stem"] + [f"stage{index}" for index in range(1, stage_count + 1)]
    features, indices = values["out_features"], values["out_indices"]
    if features is not None:
        if not isinstance(features, list) or any(not isinstance(name, str) for name in features) \
                or features != [name for name in stage_names if name in features]:
            raise ValueError(
                f"DPT BiT out_features must name distinct stages of {stage_names} in order; got "
                f"{features!r}")
    if indices is not None:
        indices = _integers(indices, "BiT out_indices", minimum=None)
        if any(not -len(stage_names) <= index < len(stage_names) for index in indices):
            raise ValueError(f"DPT BiT out_indices must index {stage_names}; got {indices}")
        positive = [index % len(stage_names) for index in indices]
        if positive != sorted(set(positive)):
            raise ValueError(f"DPT BiT out_indices must be distinct and in order; got {indices}")
        named = [stage_names[index] for index in positive]
        if features is not None and features != named:
            raise ValueError(
                f"DPT BiT out_features {features} and out_indices {indices} name different stages")
        features = named
    return features if features is not None else [stage_names[-1]]


def _resolve_backbone(raw: dict, image_size: int) -> dict:
    # DPTConfig builds the hybrid BiT defaults for an absent or null backbone_config.
    backbone = raw.get("backbone_config")
    if backbone is None:
        values = dict(_BIT_DEFAULTS, **_BIT_HYBRID_DEFAULTS)
    elif isinstance(backbone, dict):
        model_type = backbone.get("model_type", "bit")
        if model_type != "bit":
            raise NotImplementedError(
                f"dpt builds the BiT backbone of DPT-Hybrid only; got backbone model_type={model_type!r}")
        values = _values(backbone, _BIT_DEFAULTS)
    else:
        raise ValueError("DPT backbone_config must be a JSON object")

    if _text(values["layer_type"], "BiT layer_type") != "bottleneck":
        raise NotImplementedError(
            f"dpt supports the BiT bottleneck layer only; got {values['layer_type']!r}")
    padding = values["global_padding"]
    if not isinstance(padding, str) or padding.upper() != "SAME":
        raise NotImplementedError(f"dpt supports BiT global_padding='SAME' only; got {padding!r}")
    if not _flag(values["embedding_dynamic_padding"], "BiT embedding_dynamic_padding"):
        raise NotImplementedError("dpt supports BiT embedding_dynamic_padding=true only")
    if _text(values["hidden_act"], "BiT hidden_act") != "relu":
        raise NotImplementedError(f"dpt supports BiT hidden_act='relu' only; got {values['hidden_act']!r}")
    if _integer(values["num_channels"], "BiT num_channels") != 3:
        raise ValueError("DPT BiT backbone must take three input channels")

    depths = _integers(values["depths"], "BiT depths")
    hidden_sizes = _integers(values["hidden_sizes"], "BiT hidden_sizes")
    # BitConfig names a stage per depth; the encoder pairs them with hidden_sizes.
    stage_count = len(depths)
    if len(hidden_sizes) < stage_count:
        raise ValueError(f"DPT BiT needs a hidden size per stage; got {hidden_sizes} for {depths}")
    out_features = _bit_out_features(values, stage_count)
    if out_features != [f"stage{index}" for index in range(1, stage_count + 1)] or stage_count != 3:
        raise NotImplementedError(
            "dpt needs a three-stage BiT backbone that returns every stage; got "
            f"depths={depths} out_features={out_features!r}")

    width_factor = _number(values["width_factor"], "BiT width_factor")
    if width_factor <= 0.0:
        raise ValueError(f"DPT BiT width_factor must be positive; got {width_factor}")
    output_stride = _integer(values["output_stride"], "BiT output_stride")
    groups = _integer(values["num_groups"], "BiT num_groups")
    embedding = _integer(values["embedding_size"], "BiT embedding_size")

    # The stem: 7x7 stride-2 convolution, then a 3x3 stride-2 max pool, both with
    # SAME padding computed from the input size.
    side = image_size
    stem_pad = _same_padding(side, 7, 2)
    side = (side + sum(stem_pad) - 7) // 2 + 1
    pool_pad = _same_padding(side, 3, 2)
    side = (side + sum(pool_pad) - 3) // 2 + 1

    stages = []
    channels = embedding
    current_stride = 4
    dilation = 1
    for index in range(stage_count):
        out_channels = _make_div(hidden_sizes[index] * width_factor)
        stride = 1 if index == 0 else 2
        if current_stride >= output_stride:
            dilation *= stride
            stride = 1
        if dilation != 1:
            raise NotImplementedError("dpt does not support dilated BiT stages")
        mid_channels = _make_div(out_channels * 0.25)
        for value in (mid_channels, out_channels):
            if value % groups:
                raise ValueError(f"BiT GroupNorm needs {groups} groups to divide {value} channels")
        stages.append({
            "depth": depths[index],
            "in_channels": channels,
            "out_channels": out_channels,
            "mid_channels": mid_channels,
            "stride": stride,
            "in_side": side,
            "conv2_padding": _conv_padding(side, 3, stride),
            "downsample_padding": _conv_padding(side, 1, stride),
        })
        side = (side + sum(stages[-1]["conv2_padding"]) - 3) // stride + 1
        stages[-1]["out_side"] = side
        channels = out_channels
        current_stride *= stride
    if embedding % groups:
        raise ValueError(f"BiT GroupNorm needs {groups} groups to divide {embedding} channels")

    return {
        "embedding_size": embedding,
        "num_groups": groups,
        "stem_padding": stem_pad,
        "pool_padding": pool_pad,
        "stages": stages,
    }


def _normalized(rescale_factor: float, mean: list[float], std: list[float]) -> np.ndarray:
    """Every 8-bit level of every channel, normalized as the runtime computes it: the
    product in float64 stored as float32, then subtraction and division in float32."""
    mean32 = np.asarray(mean, dtype=np.float32)[:, None]
    std32 = np.asarray(std, dtype=np.float32)[:, None]
    levels = (np.arange(256, dtype=np.float64) * rescale_factor).astype(np.float32)[None, :]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return (levels - mean32) / std32


def _check_normalization(rescale_factor: float, mean: list[float], std: list[float]) -> None:
    """Every 8-bit sample must rescale and normalize to a finite float32."""
    if not 0.0 < rescale_factor <= _FLOAT32_MAX / 255.0:
        raise ValueError(
            f"DPT rescale_factor must be positive and keep 255 within float32; got {rescale_factor}")
    if any(abs(value) > _FLOAT32_MAX for value in (*mean, *std)):
        raise ValueError("DPT image_mean and image_std must fit float32")
    if (np.asarray(std, dtype=np.float32) == 0.0).any():
        raise ValueError(f"DPT image_std must be non-zero in float32; got {std}")
    if not np.isfinite(_normalized(rescale_factor, mean, std)).all():
        raise ValueError("DPT image_mean and image_std overflow float32 for an 8-bit sample")


def _resolve_processor(processor: dict, image_size: int) -> dict:
    values = _values(processor, _PROCESSOR_DEFAULTS)
    # DPTImageProcessor replaces a null size, image_mean or image_std with its default.
    for key in ("size", "image_mean", "image_std"):
        if values[key] is None:
            values[key] = _PROCESSOR_DEFAULTS[key]
    size = values["size"]
    if not _flag(values["do_resize"], "do_resize") or not isinstance(size, dict) \
            or set(size) != {"height", "width"} \
            or _integer(size["height"], "size.height") != image_size \
            or _integer(size["width"], "size.width") != image_size:
        raise NotImplementedError(
            f"dpt needs the image processor to resize to the model's {image_size}x{image_size}")
    if _flag(values["keep_aspect_ratio"], "keep_aspect_ratio") \
            or _integer(values["ensure_multiple_of"], "ensure_multiple_of") != 1:
        raise NotImplementedError(
            "dpt supports keep_aspect_ratio=false and ensure_multiple_of=1 only: the engine "
            "takes one fixed input size")
    if _integer(values["resample"], "resample", minimum=0) != _PIL_BILINEAR:
        raise NotImplementedError(
            f"dpt reproduces PIL bilinear resampling only; got resample={values['resample']}")
    if _flag(values["do_pad"], "do_pad") or values["size_divisor"] is not None:
        raise NotImplementedError("dpt does not support padded preprocessing")
    if not _flag(values["do_rescale"], "do_rescale") \
            or not _flag(values["do_normalize"], "do_normalize"):
        raise NotImplementedError("dpt needs do_rescale=true and do_normalize=true")
    mean = _numbers(values["image_mean"], "image_mean")
    std = _numbers(values["image_std"], "image_std")
    if len(mean) != 3 or len(std) != 3 or any(value == 0.0 for value in std):
        raise ValueError("DPT image_mean and image_std need three values, std non-zero")
    rescale_factor = _number(values["rescale_factor"], "rescale_factor")
    _check_normalization(rescale_factor, mean, std)
    return {
        "rescale_factor": rescale_factor,
        "image_mean": mean,
        "image_std": std,
    }


def resolve(raw: dict, processor: dict) -> dict:
    """The subset of the model and processor configs this family builds against."""
    if raw.get("model_type") != "dpt":
        raise ValueError(f"dpt does not support model_type={raw.get('model_type')!r}")
    # An absent or null list states nothing; the model class is DPTForDepthEstimation.
    architectures = raw.get("architectures")
    if architectures is not None and architectures != [ARCHITECTURE]:
        raise NotImplementedError(
            f"dpt builds {ARCHITECTURE} only; got architectures={architectures!r}")
    values = _values(raw, _DPT_DEFAULTS)
    if _flag(values["is_hybrid"], "is_hybrid") is not True:
        raise NotImplementedError(
            "dpt builds DPT-Hybrid checkpoints (is_hybrid=true, BiT backbone) only")
    if values["backbone"] is not None:
        raise NotImplementedError(
            "dpt builds the BiT backbone described by backbone_config, not a named backbone")
    if _text(values["readout_type"], "readout_type") != "project":
        raise NotImplementedError("DPT-Hybrid requires readout_type='project'")
    if _text(values["hidden_act"], "hidden_act") != "gelu":
        raise NotImplementedError(f"dpt supports hidden_act='gelu' only; got {values['hidden_act']!r}")
    if _flag(values["use_batch_norm_in_fusion_residual"], "use_batch_norm_in_fusion_residual"):
        raise NotImplementedError("dpt does not support batch norm in the fusion residual units")
    if _flag(values["add_projection"], "add_projection"):
        raise NotImplementedError("dpt does not support the depth head's add_projection")
    fusion_bias = values["use_bias_in_fusion_residual"]
    if fusion_bias is not None:
        _flag(fusion_bias, "use_bias_in_fusion_residual")
    if _integer(values["num_channels"], "num_channels") != 3:
        raise ValueError("DPT must take three input channels")

    hidden = _integer(values["hidden_size"], "hidden_size")
    heads = _integer(values["num_attention_heads"], "num_attention_heads")
    if hidden % heads:
        raise ValueError("DPT hidden_size must divide by num_attention_heads")
    image_size = _integer(values["image_size"], "image_size")
    # The runtime refuses an engine input of more than INT32_MAX / 4 pixels.
    if image_size * image_size > _INT32_MAX // 4:
        raise ValueError(f"DPT image_size {image_size} exceeds the runtime's input limit")
    patch = _integer(values["patch_size"], "patch_size")
    if image_size % patch:
        raise ValueError("DPT image_size must be a whole number of patches")
    layers = _integer(values["num_hidden_layers"], "num_hidden_layers")
    layer_norm_eps = _number(values["layer_norm_eps"], "layer_norm_eps")
    # TensorRT stores the epsilon as float32: it must stay positive and finite there.
    if not 0.0 < layer_norm_eps <= _FLOAT32_MAX or float(np.float32(layer_norm_eps)) == 0.0:
        raise ValueError(
            f"DPT layer_norm_eps must be a positive float32 value; got {layer_norm_eps}")

    if _integers(values["neck_ignore_stages"], "neck_ignore_stages", minimum=0) != [0, 1]:
        raise NotImplementedError(
            "DPT-Hybrid feeds BiT stages 1-2 straight into the neck: neck_ignore_stages "
            f"must be [0, 1]; got {values['neck_ignore_stages']!r}")
    out_indices = _integers(values["backbone_out_indices"], "backbone_out_indices", minimum=0)
    neck_sizes = _integers(values["neck_hidden_sizes"], "neck_hidden_sizes")
    factors = _numbers(values["reassemble_factors"], "reassemble_factors")
    if len(out_indices) != 4 or len(neck_sizes) != 4 or len(factors) != 4:
        raise ValueError("DPT needs four backbone_out_indices, neck_hidden_sizes and factors")
    # The reference walks the encoder layers in order and keeps those named here,
    # so the taps reach the neck in ascending layer order whatever the list order.
    taps = sorted(out_indices[2:])
    if len(set(taps)) != len(taps):
        raise ValueError(f"DPT backbone_out_indices {out_indices} name a ViT layer twice")
    if any(index >= layers for index in taps):
        raise ValueError(f"DPT backbone_out_indices {taps} name a layer past the encoder")
    # A factor 1/n is a stride-n convolution over the patch grid: n from 2 to the grid side.
    grid = image_size // patch
    for factor in factors[2:]:
        step = 1.0 / factor if 0.0 < factor < 1.0 else 0.0
        if factor != 1.0 and not (step.is_integer() and 2 <= step <= grid):
            raise NotImplementedError(
                f"dpt supports reassemble factors 1 and 1/n (n from 2 to {grid}) on the ViT "
                f"taps; got {factor}")
    if _integer(values["head_in_index"], "head_in_index", minimum=None) not in (-1, 3):
        raise NotImplementedError("dpt reads the depth head from the last fusion layer only")

    backbone = _resolve_backbone(raw, image_size)
    stage3 = backbone["stages"][-1]
    if stage3["out_side"] != grid:
        raise NotImplementedError(
            f"DPT-Hybrid needs BiT's last stage at the patch grid ({grid}); it is "
            f"{stage3['out_side']}")
    featmap = _integers(values["backbone_featmap_shape"], "backbone_featmap_shape")
    if len(featmap) != 4 or featmap[1] != stage3["out_channels"]:
        raise ValueError(
            f"DPT backbone_featmap_shape {featmap} disagrees with BiT's "
            f"{stage3['out_channels']} output channels")
    if neck_sizes[0] != backbone["stages"][0]["out_channels"] \
            or neck_sizes[1] != backbone["stages"][1]["out_channels"]:
        raise ValueError("DPT neck_hidden_sizes[0:2] must match BiT stages 1 and 2")

    return {
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "head_dim": hidden // heads,
        "num_hidden_layers": layers,
        "intermediate_size": _integer(values["intermediate_size"], "intermediate_size"),
        "layer_norm_eps": layer_norm_eps,
        "qkv_bias": _flag(values["qkv_bias"], "qkv_bias"),
        "image_size": image_size,
        "patch_size": patch,
        "patch_grid": grid,
        "tap_layers": taps,
        "reassemble_factors": factors,
        "neck_hidden_sizes": neck_sizes,
        "fusion_hidden_size": _integer(values["fusion_hidden_size"], "fusion_hidden_size"),
        "fusion_bias": fusion_bias is not False,
        "backbone": backbone,
        "processor": _resolve_processor(processor, image_size),
    }


def bit_layer_shapes(stage: dict, layer_index: int) -> dict[str, tuple[int, int, int, int]]:
    """The convolution weight shapes of one BiT bottleneck layer, by module name."""
    mid, out = stage["mid_channels"], stage["out_channels"]
    in_channels = stage["in_channels"] if layer_index == 0 else out
    shapes = {"downsample.conv": (out, in_channels, 1, 1)} if layer_index == 0 else {}
    shapes.update({"conv1": (mid, in_channels, 1, 1), "conv2": (mid, mid, 3, 3),
                   "conv3": (out, mid, 1, 1)})
    return shapes


def bit_weight_shapes(resolved: dict) -> Iterator[tuple[str, tuple[int, ...]]]:
    """Every BiT convolution weight the graph reads, with the shape the configuration implies.

    A generator in graph order, so that a check against a checkpoint stops at the
    first missing weight however deep the configured stages are.
    """
    backbone = resolved["backbone"]
    yield f"{BIT_PREFIX}.embedder.convolution.weight", (backbone["embedding_size"], 3, 7, 7)
    for stage_index, stage in enumerate(backbone["stages"]):
        for layer_index in range(stage["depth"]):
            prefix = f"{BIT_PREFIX}.encoder.stages.{stage_index}.layers.{layer_index}"
            for module, shape in bit_layer_shapes(stage, layer_index).items():
                yield f"{prefix}.{module}.weight", shape


def check_weight_shapes(tensors: dict, expected: Iterable[tuple[str, tuple[int, ...]]]) -> None:
    """Refuse a checkpoint whose tensors lack the shapes its configuration implies."""
    for name, shape in expected:
        if name not in tensors:
            raise KeyError(f"DPT checkpoint is missing {name}")
        actual = tuple(int(size) for size in tensors[name].shape)
        if actual != shape:
            raise ValueError(
                f"DPT checkpoint does not match its configuration: {name} has shape {actual}, "
                f"the configuration implies {shape}")


def check_requested_image_size(label: str, value, image_size: int) -> None:
    """A requested build image size must be the checkpoint's own, as an integer (384.5 is not 384)."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"dpt {label} must be an integer; got {value!r}")
    if value != image_size:
        raise NotImplementedError(
            f"dpt builds the checkpoint's native {label} ({image_size}) only; images of any size "
            "are resized to it at run time")


def check_engine_input(processor: dict, precision: str) -> None:
    """The fp16 engine casts its float32 input to fp16 first: every normalized sample must fit."""
    if precision != "fp16":
        return
    peak = float(np.abs(_normalized(processor["rescale_factor"], processor["image_mean"],
                                    processor["image_std"])).max())
    if peak > _FLOAT16_MAX:
        raise ValueError(
            f"DPT normalization reaches |{peak:.6g}|, past fp16's {_FLOAT16_MAX:g}: the fp16 "
            "engine's input would overflow; build fp32")
