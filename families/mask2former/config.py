# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Mask2Former checkpoint configuration this family builds, read and checked in one place.

Only the structure the graph reproduces is accepted: a Swin backbone whose stage resolutions are whole multiples of
its window size, the post-norm masked-attention decoder, one extra FPN level and a fixed-size image processor. Any
other value is rejected here instead of being silently built into a different graph.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The reference semantic post-processor resizes the mask logits to this size before it combines them with the class
# probabilities, whatever the preprocessed image size (Mask2FormerImageProcessor.post_process_semantic_segmentation).
SEMANTIC_SCORE_SIZE = 384
# PIL's bilinear filter, the processor's `resample` value.
_PIL_BILINEAR = 2
# Fixed structure of the pixel decoder and the transformer module.
FEATURE_LEVELS = 3
ENCODER_POINTS = 4
GROUP_NORM_GROUPS = 32
NORM_EPS = 1e-5


@dataclass(frozen=True)
class Mask2FormerConfig:
    image_height: int
    image_width: int
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    rescale_factor: float
    patch_size: int
    embed_dim: int
    depths: tuple[int, ...]
    num_heads: tuple[int, ...]
    window_size: int
    mlp_ratio: float
    backbone_eps: float
    feature_size: int
    mask_feature_size: int
    hidden_dim: int
    num_attention_heads: int
    encoder_layers: int
    encoder_feedforward_dim: int
    decoder_layers: int
    dim_feedforward: int
    num_queries: int
    labels: tuple[str, ...]

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    @property
    def stage_channels(self) -> tuple[int, ...]:
        return tuple(self.embed_dim * 2**stage for stage in range(len(self.depths)))

    @property
    def stage_resolutions(self) -> tuple[tuple[int, int], ...]:
        height = self.image_height // self.patch_size
        width = self.image_width // self.patch_size
        return tuple((height >> stage, width >> stage) for stage in range(len(self.depths)))


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"repeats the key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"mask2former checkpoint is missing {path.name}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    except ValueError as error:
        raise ValueError(f"mask2former {path.name} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"mask2former {path.name} must be a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NotImplementedError(f"mask2former does not support this checkpoint: {message}")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value) -> bool:
    """A JSON number (not a boolean) that converts to a finite float."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _int_value(value, name: str) -> int:
    if not _is_int(value):
        raise ValueError(f"mask2former {name} must be an integer, got {value!r}")
    return value


def _positive_int(value, name: str) -> int:
    if not _is_int(value) or value <= 0:
        raise ValueError(f"mask2former {name} must be a positive integer, got {value!r}")
    return value


def _positive_float32(value, name: str) -> float:
    """A positive finite number that stays positive and finite as the float32 the engine stores."""

    if not _is_finite_number(value) or value <= 0:
        raise ValueError(f"mask2former {name} must be a positive number, got {value!r}")
    with np.errstate(over="ignore", under="ignore"):
        stored = np.float32(value)
    if not (np.isfinite(stored) and stored > 0):
        raise ValueError(f"mask2former {name} {value!r} does not fit a positive float32")
    return float(value)


def _int_list(value, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"mask2former {name} must be a non-empty list")
    return tuple(_positive_int(item, name) for item in value)


def _channel_values(value, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"mask2former {name} must hold three numbers")
    if not all(_is_finite_number(item) for item in value):
        raise ValueError(f"mask2former {name} must hold three finite numbers, got {value!r}")
    return tuple(float(item) for item in value)


def _require_finite_normalization(
    rescale_factor: float, mean: tuple[float, ...], std: tuple[float, ...]
) -> None:
    """Normalize every 8-bit value the way the runtime does and require finite float32 results.

    The runtime stores the mean and std as float32 and refuses values beyond its range, rescales in float64, casts
    to float32 and normalizes in float32 (image_ops.cpp); a finite but tiny std, a huge mean or a huge rescale factor
    would otherwise feed infinities to the engine.
    """

    limit = float(np.finfo(np.float32).max)
    for name, values in (("image_mean", mean), ("image_std", std)):
        if any(abs(value) > limit for value in values):
            raise ValueError(f"mask2former {name} {list(values)!r} does not fit float32")
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        rescaled = (np.arange(256, dtype=np.float64) * rescale_factor).astype(np.float32)
        for channel in range(3):
            normalized = (rescaled - np.float32(mean[channel])) / np.float32(std[channel])
            if not np.all(np.isfinite(normalized)):
                raise ValueError(
                    f"mask2former rescale_factor {rescale_factor!r}, image_mean {mean[channel]!r} and image_std "
                    f"{std[channel]!r} do not normalize 8-bit values to finite float32 values"
                )


_CLASS_INDEX = re.compile(r"0|[1-9][0-9]*")


def _labels(config: dict) -> tuple[str, ...]:
    id2label = config.get("id2label")
    if not isinstance(id2label, dict) or not id2label:
        raise ValueError("mask2former config.json must name its classes in id2label")
    # Only canonical decimal keys: int() would also read "00", "+1" or " 1" and merge two entries into one index.
    if not all(_CLASS_INDEX.fullmatch(key) for key in id2label):
        raise ValueError(
            f"mask2former id2label keys must be canonical class indices, got {sorted(id2label)!r}"
        )
    indexed = {int(key): value for key, value in id2label.items()}
    if sorted(indexed) != list(range(len(indexed))):
        raise ValueError("mask2former id2label must cover the class indices 0..N-1")
    labels = tuple(indexed[index] for index in range(len(indexed)))
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("mask2former id2label values must be non-empty strings")
    return labels


def _backbone(config: dict) -> dict:
    _require(not config.get("use_timm_backbone"), "a timm backbone")
    _require(not config.get("backbone"), "a named backbone instead of backbone_config")
    backbone = config.get("backbone_config")
    if not isinstance(backbone, dict):
        raise ValueError("mask2former config.json must carry a backbone_config object")
    _require(backbone.get("model_type") == "swin", f"backbone {backbone.get('model_type')!r}")
    num_channels = _int_value(backbone.get("num_channels", 3), "backbone num_channels")
    _require(num_channels == 3, "a backbone that does not read RGB")
    _require(not backbone.get("use_absolute_embeddings", False), "absolute position embeddings")
    _require(
        backbone.get("hidden_act", "gelu") == "gelu", f"activation {backbone.get('hidden_act')!r}"
    )
    _require(backbone.get("qkv_bias", True) is True, "attention projections without bias")
    return backbone


def load(model_dir: Path) -> Mask2FormerConfig:
    """Read config.json and preprocessor_config.json of one checkpoint directory."""

    model_dir = Path(model_dir)
    config = _read_json(model_dir / "config.json")
    processor = _read_json(model_dir / "preprocessor_config.json")
    _require(config.get("model_type") == "mask2former", f"model_type {config.get('model_type')!r}")
    _require(
        "Mask2FormerForUniversalSegmentation" in (config.get("architectures") or []),
        f"architectures {config.get('architectures')!r}",
    )

    backbone = _backbone(config)
    depths = _int_list(backbone.get("depths"), "backbone depths")
    num_heads = _int_list(backbone.get("num_heads"), "backbone num_heads")
    _require(len(depths) == 4 and len(num_heads) == 4, "a backbone without four stages")
    _require(
        backbone.get("out_features") == ["stage1", "stage2", "stage3", "stage4"],
        f"backbone out_features {backbone.get('out_features')!r}",
    )
    patch_size = _positive_int(backbone.get("patch_size"), "backbone patch_size")
    embed_dim = _positive_int(backbone.get("embed_dim"), "backbone embed_dim")
    window_size = _positive_int(backbone.get("window_size"), "backbone window_size")
    for stage, heads in enumerate(num_heads):
        _require(
            (embed_dim * 2**stage) % heads == 0,
            f"stage {stage + 1} width not divisible by its heads",
        )
    mlp_ratio = backbone.get("mlp_ratio", 4.0)
    if not _is_finite_number(mlp_ratio) or mlp_ratio <= 0:
        raise ValueError(
            f"mask2former backbone mlp_ratio must be a positive number, got {mlp_ratio!r}"
        )
    mlp_width = float(mlp_ratio) * embed_dim
    _require(mlp_width == int(mlp_width), f"mlp_ratio {mlp_ratio!r}")
    backbone_eps = _positive_float32(
        backbone.get("layer_norm_eps", 1e-5), "backbone layer_norm_eps"
    )

    _require(config.get("pre_norm") is False, "the pre-norm decoder")
    _require(
        config.get("activation_function") == "relu",
        f"activation {config.get('activation_function')!r}",
    )
    _require(config.get("enforce_input_projection", False) is False, "decoder input projections")
    feature_size = _positive_int(config.get("feature_size"), "feature_size")
    hidden_dim = _positive_int(config.get("hidden_dim"), "hidden_dim")
    mask_feature_size = _positive_int(config.get("mask_feature_size"), "mask_feature_size")
    # The transformer module projects the pixel-decoder levels only when their width differs from its own.
    _require(feature_size == hidden_dim, "decoder input projections (feature_size != hidden_dim)")
    _require(
        feature_size % GROUP_NORM_GROUPS == 0, "a pixel-decoder width not divisible by 32 groups"
    )
    num_attention_heads = _positive_int(config.get("num_attention_heads"), "num_attention_heads")
    _require(
        feature_size % num_attention_heads == 0 and hidden_dim % num_attention_heads == 0,
        "attention widths not divisible by num_attention_heads",
    )
    strides = _int_list(config.get("feature_strides"), "feature_strides")
    _require(
        strides == tuple(patch_size * 2**stage for stage in range(4)),
        f"feature_strides {list(strides)} for patch size {patch_size}",
    )
    common_stride = _positive_int(config.get("common_stride"), "common_stride")
    # The pixel decoder adds log2(min transformer stride) - log2(common stride) FPN levels; the graph owns one.
    _require(strides[-FEATURE_LEVELS] == 2 * common_stride, f"common_stride {common_stride}")

    _require(processor.get("do_resize") is True, "a processor that does not resize")
    _require(processor.get("do_rescale") is True, "a processor that does not rescale")
    _require(processor.get("do_normalize") is True, "a processor that does not normalize")
    resample = _int_value(processor.get("resample", _PIL_BILINEAR), "processor resample")
    _require(resample == _PIL_BILINEAR, "a resampling filter other than bilinear")
    _require(processor.get("pad_size") is None, "a processor pad size")
    size = processor.get("size")
    _require(
        isinstance(size, dict) and set(size) == {"height", "width"},
        f"processor size {size!r}; only a fixed height and width",
    )
    image_height = _positive_int(size["height"], "processor height")
    image_width = _positive_int(size["width"], "processor width")
    size_divisor = _int_value(processor.get("size_divisor", 32), "processor size_divisor")
    _require(size_divisor >= 0, f"size_divisor {size_divisor!r}")
    if size_divisor:
        _require(
            image_height % size_divisor == 0 and image_width % size_divisor == 0,
            "a processor size the size divisor would change",
        )
    downsampling = patch_size * 2 ** (len(depths) - 1)
    _require(
        image_height % downsampling == 0 and image_width % downsampling == 0,
        f"a {image_height}x{image_width} input that the backbone pads",
    )
    for stage, (height, width) in enumerate(
        (image_height // patch_size >> stage, image_width // patch_size >> stage)
        for stage in range(len(depths))
    ):
        _require(
            height % window_size == 0 and width % window_size == 0,
            f"stage {stage + 1} resolution {height}x{width} is not a multiple of window {window_size}",
        )
    rescale_factor = processor.get("rescale_factor")
    if (
        not isinstance(rescale_factor, float)
        or not math.isfinite(rescale_factor)
        or rescale_factor <= 0.0
    ):
        raise ValueError(
            f"mask2former rescale_factor must be a positive float, got {rescale_factor!r}"
        )
    image_mean = _channel_values(processor.get("image_mean"), "image_mean")
    image_std = _channel_values(processor.get("image_std"), "image_std")
    if any(value == 0.0 for value in image_std):
        raise ValueError("mask2former image_std must not contain zero")
    _require_finite_normalization(rescale_factor, image_mean, image_std)

    labels = _labels(config)
    return Mask2FormerConfig(
        image_height=image_height,
        image_width=image_width,
        image_mean=image_mean,
        image_std=image_std,
        rescale_factor=rescale_factor,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=float(mlp_ratio),
        backbone_eps=backbone_eps,
        feature_size=feature_size,
        mask_feature_size=mask_feature_size,
        hidden_dim=hidden_dim,
        num_attention_heads=num_attention_heads,
        encoder_layers=_positive_int(config.get("encoder_layers"), "encoder_layers"),
        encoder_feedforward_dim=_positive_int(
            config.get("encoder_feedforward_dim"), "encoder_feedforward_dim"
        ),
        decoder_layers=_positive_int(config.get("decoder_layers"), "decoder_layers"),
        dim_feedforward=_positive_int(config.get("dim_feedforward"), "dim_feedforward"),
        num_queries=_positive_int(config.get("num_queries"), "num_queries"),
        labels=labels,
    )
