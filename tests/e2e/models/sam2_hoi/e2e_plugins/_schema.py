# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Five-frame structured-output schema for SAM2 HOI tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


FRAME_COUNT = 5
FRAME_FIELDS = (
    "object_ids",
    "binary_masks",
    "det_bboxes",
    "det_labels",
    "det_scores",
    "interaction_pairs",
)
PROJECT_ROOT = Path(__file__).resolve().parents[5]


def frame_key(frame: int, field: str) -> str:
    return f"frame_{frame:06d}_{field}"


def expected_keys() -> set[str]:
    return {frame_key(frame, field) for frame in range(FRAME_COUNT) for field in FRAME_FIELDS}


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = PROJECT_ROOT / path
    return candidate if candidate.exists() else path


def load_npz_arrays(path: str | Path) -> dict[str, np.ndarray]:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"SAM2 HOI output NPZ does not exist: {resolved}")
    with np.load(resolved, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    validate_arrays(arrays)
    return arrays


def validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    keys = set(arrays)
    required = expected_keys()
    if keys != required:
        missing = sorted(required - keys)
        unexpected = sorted(keys - required)
        raise ValueError(
            "SAM2 HOI output must contain exactly the 30 five-frame arrays; "
            f"missing={missing}, unexpected={unexpected}"
        )

    for frame in range(FRAME_COUNT):
        object_ids = np.asarray(arrays[frame_key(frame, "object_ids")])
        masks = np.asarray(arrays[frame_key(frame, "binary_masks")])
        boxes = np.asarray(arrays[frame_key(frame, "det_bboxes")])
        labels = np.asarray(arrays[frame_key(frame, "det_labels")])
        scores = np.asarray(arrays[frame_key(frame, "det_scores")])
        pairs = np.asarray(arrays[frame_key(frame, "interaction_pairs")])
        object_count = object_ids.shape[0] if object_ids.ndim == 1 else -1

        if object_ids.ndim != 1 or not np.issubdtype(object_ids.dtype, np.integer):
            raise ValueError(f"frame {frame}: object_ids must be a rank-1 integer array")
        if len(np.unique(object_ids)) != object_count:
            raise ValueError(f"frame {frame}: object_ids must be unique")
        if masks.ndim != 4 or masks.shape[:2] != (object_count, 1):
            raise ValueError(f"frame {frame}: binary_masks must have shape [objects, 1, H, W]")
        if masks.dtype != np.bool_ and not np.issubdtype(masks.dtype, np.integer):
            raise ValueError(f"frame {frame}: binary_masks must be binary integer or bool")
        if not np.all((masks == 0) | (masks == 1)):
            raise ValueError(f"frame {frame}: binary_masks contains values outside 0 and 1")
        if boxes.shape != (object_count, 4) or not np.issubdtype(boxes.dtype, np.floating):
            raise ValueError(
                f"frame {frame}: det_bboxes must have shape [objects, 4] and float dtype"
            )
        if not np.all(np.isfinite(boxes)) or not np.all(boxes[:, 2:] >= boxes[:, :2]):
            raise ValueError(f"frame {frame}: det_bboxes must be finite xyxy boxes")
        if labels.shape != (object_count,) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"frame {frame}: det_labels must be one integer per object")
        if scores.shape != (object_count,) or not np.issubdtype(scores.dtype, np.floating):
            raise ValueError(f"frame {frame}: det_scores must be one float per object")
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"frame {frame}: det_scores must be finite")
        if pairs.ndim != 2 or pairs.shape[1:] != (2,) or not np.issubdtype(pairs.dtype, np.integer):
            raise ValueError(
                f"frame {frame}: interaction_pairs must have shape [pairs, 2] and integer dtype"
            )
        if pairs.size and not np.all(np.isin(pairs, object_ids)):
            raise ValueError(f"frame {frame}: interaction_pairs must refer to declared object_ids")


def validate_dimensions(
    arrays: Mapping[str, np.ndarray],
    *,
    height: int,
    width: int,
) -> None:
    validate_arrays(arrays)
    for frame in range(FRAME_COUNT):
        shape = arrays[frame_key(frame, "binary_masks")].shape
        if shape[2:] != (height, width):
            raise ValueError(
                f"frame {frame}: binary mask spatial shape must be "
                f"{height}x{width}, got {shape[2]}x{shape[3]}"
            )


def structured_summary(arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    validate_arrays(arrays)
    frames: list[dict[str, Any]] = []
    for frame in range(FRAME_COUNT):
        masks = np.asarray(arrays[frame_key(frame, "binary_masks")])
        frames.append(
            {
                "frame_index": frame,
                "object_ids": arrays[frame_key(frame, "object_ids")].tolist(),
                "det_bboxes": arrays[frame_key(frame, "det_bboxes")].tolist(),
                "det_labels": arrays[frame_key(frame, "det_labels")].tolist(),
                "det_scores": arrays[frame_key(frame, "det_scores")].tolist(),
                "interaction_pairs": arrays[frame_key(frame, "interaction_pairs")].tolist(),
                "binary_masks": {
                    "npz_key": frame_key(frame, "binary_masks"),
                    "shape": list(masks.shape),
                    "dtype": str(masks.dtype),
                },
            }
        )
    return frames


def normalize_runtime_json(json_path: str | Path, output_npz: str | Path) -> Path:
    source = Path(json_path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read SAM2 HOI runtime JSON {source}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("SAM2 HOI runtime JSON must declare schema_version=1")
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != FRAME_COUNT:
        raise ValueError("SAM2 HOI runtime JSON must contain exactly five frames")

    arrays: dict[str, np.ndarray] = {}
    observed_indices: list[int] = []
    for item in frames:
        if not isinstance(item, dict):
            raise ValueError("SAM2 HOI runtime frame entries must be objects")
        frame = item.get("frame_index")
        if not isinstance(frame, int) or isinstance(frame, bool):
            raise ValueError("SAM2 HOI runtime frame_index values must be integers")
        observed_indices.append(frame)
        mask_value = item.get("binary_masks_path")
        if not isinstance(mask_value, str) or not mask_value:
            raise ValueError(f"frame {frame}: binary_masks_path is required")
        mask_path = Path(mask_value)
        if not mask_path.is_absolute():
            mask_path = source.parent / mask_path
        if not mask_path.is_file():
            raise FileNotFoundError(f"frame {frame}: mask array does not exist: {mask_path}")

        arrays[frame_key(frame, "object_ids")] = np.asarray(item.get("object_ids"), dtype=np.int64)
        arrays[frame_key(frame, "binary_masks")] = np.load(mask_path, allow_pickle=False)
        arrays[frame_key(frame, "det_bboxes")] = np.asarray(
            item.get("det_bboxes"), dtype=np.float32
        )
        arrays[frame_key(frame, "det_labels")] = np.asarray(item.get("det_labels"), dtype=np.int64)
        arrays[frame_key(frame, "det_scores")] = np.asarray(
            item.get("det_scores"), dtype=np.float32
        )
        arrays[frame_key(frame, "interaction_pairs")] = np.asarray(
            item.get("interaction_pairs"), dtype=np.int64
        )

    if observed_indices != list(range(FRAME_COUNT)):
        raise ValueError(
            f"SAM2 HOI runtime frames must be ordered exactly 0..4, got {observed_indices}"
        )
    validate_arrays(arrays)
    destination = Path(output_npz)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **arrays)
    return destination
