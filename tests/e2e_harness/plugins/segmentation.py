"""Contract test plugin for semantic segmentation models."""
from __future__ import annotations

import numpy as np
from ..contracts import MetricResult
from .base import make_pass, make_fail, make_error


def _compute_iou(pred, gt):
    """Compute mean IoU between prediction and ground truth class maps."""
    pred = np.asarray(pred, dtype=np.int32)
    gt = np.asarray(gt, dtype=np.int32)
    classes = np.union1d(np.unique(pred), np.unique(gt))
    if len(classes) == 0:
        return 1.0
    ious = []
    for c in classes:
        p = (pred == c)
        g = (gt == c)
        intersection = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        if union > 0:
            ious.append(float(intersection / union))
    return float(np.mean(ious)) if ious else 0.0


def _pixel_accuracy(pred, gt):
    """Fraction of pixels with matching class."""
    pred = np.asarray(pred, dtype=np.int32).flatten()
    gt = np.asarray(gt, dtype=np.int32).flatten()
    if len(pred) != len(gt):
        return 0.0
    return float((pred == gt).mean())


class SegmentationPlugin:
    reference_families = [
        "semantic_segmentation",
    ]
    user_contract = "segmentation_mask"

    def configure_reference(self, case):
        return {}

    def verify(self, trt_output, ref_output, case, threshold):
        trt_mask = trt_output.data.get("class_map")
        if trt_mask is None:
            trt_mask = trt_output.data.get("mask")
        ref_mask = ref_output.data.get("class_map")
        if ref_mask is None:
            ref_mask = ref_output.data.get("mask")

        if trt_mask is None or ref_mask is None:
            return make_error("full_inference", "Missing mask/class_map in output data")

        trt_arr = np.asarray(trt_mask, dtype=np.int32)
        ref_arr = np.asarray(ref_mask, dtype=np.int32)

        # Resize if shapes differ
        if trt_arr.shape != ref_arr.shape:
            try:
                from PIL import Image
                ref_img = Image.fromarray(ref_arr.astype(np.uint8))
                ref_img = ref_img.resize((trt_arr.shape[1], trt_arr.shape[0]), Image.NEAREST)
                ref_arr = np.array(ref_img, dtype=np.int32)
            except ImportError:
                return make_error("full_inference", f"Shape mismatch {trt_arr.shape} vs {ref_arr.shape} and PIL unavailable")

        miou = _compute_iou(trt_arr, ref_arr)
        pixel_acc = _pixel_accuracy(trt_arr, ref_arr)

        miou_threshold = threshold.metrics.get("contract_miou_threshold", 0.5)
        pixel_threshold = threshold.metrics.get("contract_pixel_accuracy", 0.85)

        metrics = {
            "mIoU": MetricResult(value=miou, threshold=miou_threshold, operator=">=", passed=miou >= miou_threshold),
            "pixel_accuracy": MetricResult(value=pixel_acc, threshold=pixel_threshold, operator=">=", passed=pixel_acc >= pixel_threshold),
        }

        passed = miou >= miou_threshold and pixel_acc >= pixel_threshold
        rule = "mIoU >= threshold AND pixel_accuracy >= threshold"
        if passed:
            return make_pass("full_inference", metrics, rule)
        return make_fail("full_inference", metrics, rule,
                        f"Segmentation quality: mIoU={miou:.3f} pixel_acc={pixel_acc:.3f}")


plugin = SegmentationPlugin()
