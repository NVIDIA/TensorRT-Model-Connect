"""Sam3 model-owned prompted segmentation contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.e2e_harness.contracts import MetricResult
from tests.e2e_harness.plugins.base import make_error, make_fail, make_pass


def _resolve_mask_list(data):
    masks = data.get("masks") or []
    if masks:
        return masks

    masks_path = data.get("masks_path")
    if not masks_path:
        return []

    path = Path(masks_path)
    if not path.is_file():
        return []

    loaded = np.load(path, allow_pickle=False)
    if loaded.ndim == 2:
        return [loaded]
    return [loaded[i] for i in range(loaded.shape[0])]


def _resolve_score_list(data):
    scores = data.get("mask_scores")
    if scores is None:
        scores = data.get("scores")
    if scores is None:
        scores = data.get("iou_scores")
    if scores is None:
        return []
    return [float(score) for score in scores]


def _resolve_box_list(data):
    boxes = data.get("boxes")
    if boxes is None:
        return []
    resolved = []
    for box in boxes:
        values = [float(value) for value in box]
        if len(values) == 4:
            resolved.append(values)
    return resolved


def _compute_binary_iou(pred, gt):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def _compute_box_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _verify_sam3_instance_metadata(trt_output, ref_output, threshold, metrics):
    trt_boxes = _resolve_box_list(trt_output.data)
    ref_boxes = _resolve_box_list(ref_output.data)
    trt_scores = _resolve_score_list(trt_output.data)
    ref_scores = _resolve_score_list(ref_output.data)

    if not trt_boxes or not ref_boxes:
        return "Missing prompted segmentation boxes"
    if not trt_scores or not ref_scores:
        return "Missing prompted segmentation scores"

    same_box_count = len(trt_boxes) == len(ref_boxes)
    metrics["box_count_consistency"] = MetricResult(
        value=1.0 if same_box_count else 0.0,
        threshold=1.0,
        operator="==",
        passed=same_box_count,
    )
    box_ious = [
        _compute_box_iou(trt_box, ref_box)
        for trt_box, ref_box in zip(trt_boxes, ref_boxes)
    ]
    if box_ious:
        mean_box_iou = float(sum(box_ious) / len(box_ious))
        box_thresh = threshold.metrics.get("box_iou_mean", 0.95)
        metrics["box_iou_mean"] = MetricResult(
            value=mean_box_iou,
            threshold=box_thresh,
            operator=">=",
            passed=mean_box_iou >= box_thresh,
        )

    same_score_count = len(trt_scores) == len(ref_scores)
    metrics["score_count_consistency"] = MetricResult(
        value=1.0 if same_score_count else 0.0,
        threshold=1.0,
        operator="==",
        passed=same_score_count,
    )
    score_errors = [
        abs(float(trt_score) - float(ref_score))
        for trt_score, ref_score in zip(trt_scores, ref_scores)
    ]
    if score_errors:
        mean_score_error = float(sum(score_errors) / len(score_errors))
        score_thresh = threshold.metrics.get("score_abs_error_mean", 0.05)
        metrics["score_abs_error_mean"] = MetricResult(
            value=mean_score_error,
            threshold=score_thresh,
            operator="<=",
            passed=mean_score_error <= score_thresh,
        )

    return None


def _verify_sam3_prompted_masks(trt_output, ref_output, threshold):
    trt_masks = _resolve_mask_list(trt_output.data)
    ref_masks = _resolve_mask_list(ref_output.data)

    if not trt_masks or not ref_masks:
        return make_error("full_inference", "Missing prompted segmentation masks")

    metrics = {
        "trt_num_masks": MetricResult(
            value=float(len(trt_masks)), threshold=None, operator=">=", passed=True),
        "ref_num_masks": MetricResult(
            value=float(len(ref_masks)), threshold=None, operator=">=", passed=True),
    }

    num_masks_threshold = threshold.metrics.get("num_masks_consistency")
    if num_masks_threshold is not None:
        same_count = len(trt_masks) == len(ref_masks)
        metrics["num_masks_consistency"] = MetricResult(
            value=1.0 if same_count else 0.0,
            threshold=1.0,
            operator="==",
            passed=same_count,
        )

    iou_values = []
    for i in range(min(len(trt_masks), len(ref_masks))):
        trt_mask = np.asarray(trt_masks[i], dtype=bool)
        ref_mask = np.asarray(ref_masks[i], dtype=bool)
        if trt_mask.shape != ref_mask.shape:
            try:
                from PIL import Image
                trt_img = Image.fromarray(trt_mask.astype(np.uint8) * 255)
                trt_img = trt_img.resize((ref_mask.shape[1], ref_mask.shape[0]), Image.NEAREST)
                trt_mask = np.asarray(trt_img, dtype=np.uint8).astype(bool)
            except ImportError:
                return make_error(
                    "full_inference",
                    f"Shape mismatch {trt_mask.shape} vs {ref_mask.shape} and PIL unavailable",
                )
        iou = _compute_binary_iou(trt_mask, ref_mask)
        iou_values.append(iou)
        metrics[f"mask_{i}_iou"] = MetricResult(
            value=iou, threshold=None, operator=">=", passed=True,
            note="per-mask informational",
        )

    if not iou_values:
        return make_error("full_inference", "No prompted segmentation masks were comparable")

    mean_iou = sum(iou_values) / len(iou_values)
    iou_threshold = threshold.metrics.get("iou_per_prompt", 0.5)
    metrics["iou_per_prompt"] = MetricResult(
        value=mean_iou,
        threshold=iou_threshold,
        operator=">=",
        passed=mean_iou >= iou_threshold,
    )

    metadata_error = _verify_sam3_instance_metadata(
        trt_output, ref_output, threshold, metrics)
    if metadata_error:
        return make_error("full_inference", metadata_error)

    rule = "mean prompted-mask IoU >= threshold with Sam3 boxes and scores"
    gated = [m for m in metrics.values() if m.threshold is not None]
    passed = all(m.passed for m in gated)
    if passed:
        return make_pass("full_inference", metrics, rule)
    return make_fail(
        "full_inference",
        metrics,
        rule,
        f"Sam3 prompted segmentation quality: mean_iou={mean_iou:.3f}",
    )


class Sam3SegmentationPlugin:
    reference_families = ["prompted_segmentation_sam3"]
    user_contract = "prompted_mask"

    def configure_reference(self, case):
        return {"sam_mode": True}

    def verify(self, trt_output, ref_output, case, threshold):
        return _verify_sam3_prompted_masks(trt_output, ref_output, threshold)


plugin = Sam3SegmentationPlugin()
