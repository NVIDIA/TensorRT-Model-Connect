"""SegFormer semantic segmentation comparator."""

from __future__ import annotations

import logging
from pathlib import Path

from ..contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared numerical helpers
# ---------------------------------------------------------------------------

def _safe_import_numpy():
    import numpy as np
    return np


def _resolve_mask_list(data: dict) -> list:
    masks = data.get("masks", [])
    if masks:
        return masks

    masks_path = data.get("masks_path")
    if not masks_path:
        return []

    path = Path(masks_path)
    if not path.is_file():
        return []

    np = _safe_import_numpy()
    loaded = np.load(path, allow_pickle=False)
    if loaded.ndim == 2:
        return [loaded]
    return [loaded[i] for i in range(loaded.shape[0])]


def _compute_iou(mask_a, mask_b) -> float:
    """Compute IoU between two binary masks (numpy arrays)."""
    np = _safe_import_numpy()
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def _compute_pixel_accuracy(pred, gt) -> float:
    """Pixel-wise accuracy between predicted and ground-truth class maps."""
    np = _safe_import_numpy()
    pred = np.asarray(pred, dtype=np.int32)
    gt = np.asarray(gt, dtype=np.int32)
    if pred.shape != gt.shape:
        # Resize pred to gt shape via nearest-neighbor
        try:
            from PIL import Image
            pred_pil = Image.fromarray(pred.astype(np.uint8))
            pred_resized = pred_pil.resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST)
            pred = np.array(pred_resized).astype(np.int32)
        except ImportError:
            logger.warning("PIL not available for resize; shape mismatch")
            return 0.0
    return float((pred == gt).mean())


def _compute_miou(pred, gt, num_classes: int | None = None) -> float:
    """Mean Intersection-over-Union across all present classes."""
    np = _safe_import_numpy()
    pred = np.asarray(pred, dtype=np.int32)
    gt = np.asarray(gt, dtype=np.int32)

    if pred.shape != gt.shape:
        try:
            from PIL import Image
            pred_pil = Image.fromarray(pred.astype(np.uint8))
            pred_resized = pred_pil.resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST)
            pred = np.array(pred_resized).astype(np.int32)
        except ImportError:
            return 0.0

    if num_classes is None:
        all_ids = np.union1d(np.unique(pred), np.unique(gt))
    else:
        all_ids = np.arange(num_classes)

    iou_sum = 0.0
    count = 0
    for cls in all_ids:
        pred_mask = pred == cls
        gt_mask = gt == cls
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        if union == 0:
            continue
        iou_sum += intersection / union
        count += 1

    return float(iou_sum / count) if count > 0 else 0.0


def _compute_boundary_f_score(pred, gt, tolerance: int = 2) -> float:
    """Boundary F-score: precision/recall of predicted boundary pixels.

    Uses morphological gradient to extract boundaries, then computes
    F1 between predicted and ground-truth boundary pixels within a
    tolerance band.
    """
    np = _safe_import_numpy()
    pred = np.asarray(pred, dtype=np.int32)
    gt = np.asarray(gt, dtype=np.int32)

    if pred.shape != gt.shape:
        try:
            from PIL import Image
            pred_pil = Image.fromarray(pred.astype(np.uint8))
            pred_resized = pred_pil.resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST)
            pred = np.array(pred_resized).astype(np.int32)
        except ImportError:
            return 0.0

    # Extract boundaries via simple gradient (pixel differs from neighbor)
    def _boundary(seg):
        b = np.zeros_like(seg, dtype=bool)
        b[:-1, :] |= seg[:-1, :] != seg[1:, :]
        b[1:, :] |= seg[:-1, :] != seg[1:, :]
        b[:, :-1] |= seg[:, :-1] != seg[:, 1:]
        b[:, 1:] |= seg[:, :-1] != seg[:, 1:]
        return b

    pred_boundary = _boundary(pred)
    gt_boundary = _boundary(gt)

    if not gt_boundary.any() and not pred_boundary.any():
        return 1.0
    if not gt_boundary.any() or not pred_boundary.any():
        return 0.0

    # Dilate ground-truth boundary for tolerance
    from scipy.ndimage import binary_dilation
    struct = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
    gt_dilated = binary_dilation(gt_boundary, structure=struct)
    pred_dilated = binary_dilation(pred_boundary, structure=struct)

    precision = float(pred_boundary[gt_dilated].sum() / pred_boundary.sum())
    recall = float(gt_boundary[pred_dilated].sum() / gt_boundary.sum())

    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


class SegmentationComparator:
    """Compare TRT vs reference semantic segmentation outputs."""

    @property
    def task_strategy(self) -> str:
        return "segmentation"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        np = _safe_import_numpy()

        metrics: dict[str, MetricResult] = {}

        trt_map = trt.data.get("class_map")
        ref_map = ref.data.get("class_map")

        if trt_map is None:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics=metrics,
                message="No TRT segmentation output",
            )
        if ref_map is None:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics=metrics,
                message="No reference segmentation output",
            )

        trt_map = np.asarray(trt_map, dtype=np.int32)
        ref_map = np.asarray(ref_map, dtype=np.int32)

        # Pixel accuracy
        pixel_acc = _compute_pixel_accuracy(trt_map, ref_map)
        pixel_acc_thresh = threshold.metrics.get("pixel_accuracy", 0.85)
        metrics["pixel_accuracy"] = MetricResult(
            value=pixel_acc, threshold=pixel_acc_thresh,
            operator=">=", passed=pixel_acc >= pixel_acc_thresh,
        )

        # mIoU
        miou = _compute_miou(trt_map, ref_map)
        miou_thresh = threshold.metrics.get("mIoU", 0.5)
        metrics["mIoU"] = MetricResult(
            value=miou, threshold=miou_thresh,
            operator=">=", passed=miou >= miou_thresh,
        )

        # Boundary F-score (optional, graceful fallback if scipy unavailable)
        try:
            bf = _compute_boundary_f_score(trt_map, ref_map)
            bf_thresh = threshold.metrics.get("boundary_f_score")
            if bf_thresh is not None:
                metrics["boundary_f_score"] = MetricResult(
                    value=bf, threshold=bf_thresh,
                    operator=">=", passed=bf >= bf_thresh,
                )
            else:
                metrics["boundary_f_score"] = MetricResult(
                    value=bf, threshold=None, operator=">=", passed=True,
                    note="informational (no threshold configured)",
                )
        except ImportError:
            pass  # scipy not available; skip boundary_f_score entirely

        # Class distribution summary (informational)
        trt_classes = int(len(np.unique(trt_map)))
        ref_classes = int(len(np.unique(ref_map)))
        metrics["trt_num_classes"] = MetricResult(
            value=float(trt_classes), threshold=None, operator=">=", passed=True,
        )
        metrics["ref_num_classes"] = MetricResult(
            value=float(ref_classes), threshold=None, operator=">=", passed=True,
        )

        overall = all(m.passed for m in metrics.values() if m.threshold is not None)
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if overall else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all metrics must pass",
            message=f"Segmentation: {'PASS' if overall else 'FAIL'} "
                    f"(pixel_acc={pixel_acc:.4f}, mIoU={miou:.4f})",
        )


# ---------------------------------------------------------------------------
# PromptedSegmentationComparator
# ---------------------------------------------------------------------------


plugin = SegmentationComparator()
