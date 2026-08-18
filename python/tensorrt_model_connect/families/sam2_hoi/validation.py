# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Final-output accuracy contract for SAM2 HOI five-frame tracking."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ACCURACY_CONTRACT_ID = "sam2-hoi-full-chain-accuracy-v2"
EXACT_DECISION_FIELDS = ("object_ids", "det_labels", "interaction_pairs")


@dataclass(frozen=True)
class AccuracyThresholds:
    """Reviewed gates added because the source package provides no tolerances."""

    detection_score_max_abs: float = 0.01
    detection_box_max_abs_pixels: float = 2.0
    detection_box_min_iou: float = 0.99
    mask_min_iou: float = 0.99
    # Dice is the mathematically equivalent boundary for IoU == 0.99:
    # 2 * IoU / (1 + IoU).
    mask_min_dice: float = 0.9949748743718593
    mask_min_pixel_agreement: float = 0.999


def _frame_indices(keys: set[str]) -> list[int]:
    frames: set[int] = set()
    for key in keys:
        if not key.startswith("frame_"):
            continue
        parts = key.split("_", 2)
        if len(parts) < 3 or not parts[1].isdigit():
            raise ValueError(f"Invalid SAM2 HOI reference key: {key}")
        frames.add(int(parts[1]))
    return sorted(frames)


def _box_iou_aligned(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] != 4:
        raise ValueError(
            "Aligned SAM2 HOI boxes must both have shape [detections, 4], got "
            f"{reference.shape} and {candidate.shape}"
        )
    top_left = np.maximum(reference[:, :2], candidate[:, :2])
    bottom_right = np.minimum(reference[:, 2:], candidate[:, 2:])
    intersection_size = np.maximum(bottom_right - top_left, 0.0)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    reference_size = np.maximum(reference[:, 2:] - reference[:, :2], 0.0)
    candidate_size = np.maximum(candidate[:, 2:] - candidate[:, :2], 0.0)
    reference_area = reference_size[:, 0] * reference_size[:, 1]
    candidate_area = candidate_size[:, 0] * candidate_size[:, 1]
    union = reference_area + candidate_area - intersection
    return np.divide(
        intersection,
        union,
        out=np.ones_like(intersection, dtype=np.float64),
        where=union > 0,
    )


def _mask_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"SAM2 HOI mask shape mismatch: {reference.shape} != {candidate.shape}")
    if reference.ndim != 4:
        raise ValueError(f"SAM2 HOI masks must have shape [objects, 1, H, W]: {reference.shape}")
    reference_bool = reference.astype(bool, copy=False)
    candidate_bool = candidate.astype(bool, copy=False)
    intersection = np.logical_and(reference_bool, candidate_bool).sum(axis=(1, 2, 3))
    union = np.logical_or(reference_bool, candidate_bool).sum(axis=(1, 2, 3))
    reference_sum = reference_bool.sum(axis=(1, 2, 3))
    candidate_sum = candidate_bool.sum(axis=(1, 2, 3))
    iou = np.divide(
        intersection,
        union,
        out=np.ones_like(intersection, dtype=np.float64),
        where=union > 0,
    )
    dice_denominator = reference_sum + candidate_sum
    dice = np.divide(
        2 * intersection,
        dice_denominator,
        out=np.ones_like(intersection, dtype=np.float64),
        where=dice_denominator > 0,
    )
    agreement = np.equal(reference_bool, candidate_bool).mean(axis=(1, 2, 3))
    return {
        "min_iou": float(iou.min(initial=1.0)),
        "min_dice": float(dice.min(initial=1.0)),
        "min_pixel_agreement": float(agreement.min(initial=1.0)),
    }


def compare_outputs(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    thresholds: AccuracyThresholds = AccuracyThresholds(),
) -> dict[str, object]:
    reference_keys = set(reference)
    candidate_keys = set(candidate)
    if reference_keys != candidate_keys:
        missing = sorted(reference_keys - candidate_keys)
        unexpected = sorted(candidate_keys - reference_keys)
        raise ValueError(f"SAM2 HOI output keys differ; missing={missing}, unexpected={unexpected}")

    frames = _frame_indices(reference_keys)
    if frames != list(range(5)):
        raise ValueError(f"SAM2 HOI accuracy contract requires frames 0..4, got {frames}")

    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for frame in frames:
        prefix = f"frame_{frame:06d}"
        exact_fields = EXACT_DECISION_FIELDS
        exact: dict[str, bool] = {}
        for field in exact_fields:
            key = f"{prefix}_{field}"
            matches = bool(np.array_equal(reference[key], candidate[key]))
            exact[field] = matches
            if not matches:
                failures.append(f"{key} differs")

        reference_boxes = np.asarray(reference[f"{prefix}_det_bboxes"], dtype=np.float64)
        candidate_boxes = np.asarray(candidate[f"{prefix}_det_bboxes"], dtype=np.float64)
        if reference_boxes.shape != candidate_boxes.shape:
            raise ValueError(
                f"{prefix}_det_bboxes shape mismatch: "
                f"{reference_boxes.shape} != {candidate_boxes.shape}"
            )
        box_max_abs = float(np.max(np.abs(reference_boxes - candidate_boxes), initial=0.0))
        box_min_iou = float(_box_iou_aligned(reference_boxes, candidate_boxes).min(initial=1.0))

        reference_scores = np.asarray(reference[f"{prefix}_det_scores"], dtype=np.float64)
        candidate_scores = np.asarray(candidate[f"{prefix}_det_scores"], dtype=np.float64)
        if reference_scores.shape != candidate_scores.shape:
            raise ValueError(
                f"{prefix}_det_scores shape mismatch: "
                f"{reference_scores.shape} != {candidate_scores.shape}"
            )
        score_max_abs = float(np.max(np.abs(reference_scores - candidate_scores), initial=0.0))
        masks = _mask_metrics(
            reference[f"{prefix}_binary_masks"], candidate[f"{prefix}_binary_masks"]
        )

        gates = {
            "detection_score": score_max_abs <= thresholds.detection_score_max_abs,
            "detection_box_abs": box_max_abs <= thresholds.detection_box_max_abs_pixels,
            "detection_box_iou": box_min_iou >= thresholds.detection_box_min_iou,
            "mask_iou": masks["min_iou"] >= thresholds.mask_min_iou,
            "mask_dice": masks["min_dice"] >= thresholds.mask_min_dice,
            "mask_pixel_agreement": (
                masks["min_pixel_agreement"] >= thresholds.mask_min_pixel_agreement
            ),
            **exact,
        }
        for gate, passed in gates.items():
            if not passed:
                failures.append(f"{prefix} failed {gate}")
        rows.append(
            {
                "frame": frame,
                "detection_score_max_abs": score_max_abs,
                "detection_box_max_abs_pixels": box_max_abs,
                "detection_box_min_iou": box_min_iou,
                **masks,
                "gates": gates,
            }
        )

    return {
        "schema_version": 2,
        "accuracy_contract": ACCURACY_CONTRACT_ID,
        "decision_contract": {
            "scope": "observable_full_chain_topology",
            "exact_fields": list(EXACT_DECISION_FIELDS),
        },
        "status": "pass" if not failures else "fail",
        "thresholds": asdict(thresholds),
        "frames": rows,
        "failures": failures,
    }


def compare_npz(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    thresholds: AccuracyThresholds = AccuracyThresholds(),
) -> dict[str, object]:
    with np.load(reference_path) as reference_npz, np.load(candidate_path) as candidate_npz:
        reference = {name: reference_npz[name] for name in reference_npz.files}
        candidate = {name: candidate_npz[name] for name in candidate_npz.files}
    return compare_outputs(reference, candidate, thresholds=thresholds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_npz(args.reference, args.candidate)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
