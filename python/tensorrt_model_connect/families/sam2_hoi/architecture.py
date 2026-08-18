# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed architecture contract for the reviewed SAM2.1 HOI model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Sam2HoiArchitecture:
    variant: str = "sam2.1_hiera_small_hoi_c4"
    image_size: int = 1024
    original_height: int = 1280
    original_width: int = 1088
    fixture_frames: int = 5
    object_batch: int = 2
    hidden_size: int = 256
    memory_channels: int = 64
    num_mask_memory_frames: int = 7
    max_object_pointers: int = 16
    hoi_num_queries: int = 1500
    hoi_num_classes: int = 4
    hoi_num_feature_levels: int = 3
    hoi_encoder_layers: int = 6
    hoi_decoder_layers: int = 6
    tracker_memory_attention_layers: int = 4
    mask_logit_threshold: float = 0.01
    score_threshold: float = 0.35
    class_nms_threshold: float = 0.5
    global_nms_threshold: float = 0.75
    hand_nms_threshold: float = 0.25
    interaction_threshold: float = 0.5

    @property
    def tracker_feature_shapes(self) -> tuple[tuple[int, ...], ...]:
        return (
            (1, 32, 256, 256),
            (1, 64, 128, 128),
            (1, 256, 64, 64),
        )

    @property
    def tracker_position_shape(self) -> tuple[int, ...]:
        return (1, 256, 64, 64)

    @property
    def detector_feature_shapes(self) -> tuple[tuple[int, ...], ...]:
        return (
            (1, 256, 128, 128),
            (1, 256, 64, 64),
            (1, 256, 32, 32),
        )

    def bundle_config(self) -> dict[str, object]:
        result = {f"sam2_hoi_{key}": value for key, value in asdict(self).items()}
        result.update(
            {
                "input_image_h": self.image_size,
                "input_image_w": self.image_size,
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225],
                # PIL.Image.resize() defaults to BICUBIC for RGB images.  The
                # native runtime reproduces Pillow's fixed-point uint8 path.
                "image_interpolation": "bicubic",
                "image_color_order": "rgb",
                "sam2_hoi_tracker_feature_shapes": [
                    list(shape) for shape in self.tracker_feature_shapes
                ],
                "sam2_hoi_detector_feature_shapes": [
                    list(shape) for shape in self.detector_feature_shapes
                ],
                "sam2_hoi_tracker_position_shape": list(self.tracker_position_shape),
            }
        )
        return result


ARCHITECTURE = Sam2HoiArchitecture()


_REQUIRED_RAW_VALUES: tuple[tuple[str, Any], ...] = (
    ("variant", ARCHITECTURE.variant),
    ("hiera_embed_dim", 96),
    ("hiera_stages", [1, 2, 11, 2]),
    ("hiera_global_attention_blocks", [7, 10, 13]),
    ("fpn_hidden_size", ARCHITECTURE.hidden_size),
    ("hoi_num_queries", ARCHITECTURE.hoi_num_queries),
    ("hoi_num_classes", ARCHITECTURE.hoi_num_classes),
    ("hoi_num_feature_levels", ARCHITECTURE.hoi_num_feature_levels),
    ("hoi_encoder_layers", ARCHITECTURE.hoi_encoder_layers),
    ("hoi_decoder_layers", ARCHITECTURE.hoi_decoder_layers),
    ("memory_attention_layers", ARCHITECTURE.tracker_memory_attention_layers),
    ("memory_channels", ARCHITECTURE.memory_channels),
    ("num_mask_memory_frames", ARCHITECTURE.num_mask_memory_frames),
    ("score_threshold", ARCHITECTURE.score_threshold),
    ("class_nms_threshold", ARCHITECTURE.class_nms_threshold),
    ("global_nms_threshold", ARCHITECTURE.global_nms_threshold),
    ("hand_nms_threshold", ARCHITECTURE.hand_nms_threshold),
    ("interaction_threshold", ARCHITECTURE.interaction_threshold),
    ("mask_logit_threshold", ARCHITECTURE.mask_logit_threshold),
)


def validate_architecture(raw_config: dict[str, Any]) -> Sam2HoiArchitecture:
    """Reject variants that do not match the graph rebuilt by this family."""

    raw = raw_config.get("sam2_hoi")
    if not isinstance(raw, dict):
        raise RuntimeError("SAM2 HOI config is missing the 'sam2_hoi' architecture block")
    mismatches: list[str] = []
    for key, expected in _REQUIRED_RAW_VALUES:
        if key not in raw:
            mismatches.append(f"missing {key}")
            continue
        actual = raw[key]
        if isinstance(expected, float):
            try:
                matches = float(actual) == expected
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            mismatches.append(f"{key} expected {expected!r}, got {actual!r}")
    if raw_config.get("image_size") != ARCHITECTURE.image_size:
        mismatches.append(
            f"image_size expected {ARCHITECTURE.image_size}, got {raw_config.get('image_size')!r}"
        )
    if mismatches:
        raise RuntimeError("Unsupported SAM2 HOI architecture: " + "; ".join(mismatches))
    return ARCHITECTURE
