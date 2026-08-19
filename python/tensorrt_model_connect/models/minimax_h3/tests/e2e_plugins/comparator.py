# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed human-visible quality comparator for MiniMax-H3."""

from __future__ import annotations

from pathlib import Path
import re

from .contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)
from tensorrt_model_connect.models.minimax_h3.tests.visual_metrics import (
    compute_decoded_visual_metrics,
    evaluate_visual_quality,
    visual_block_size,
    visual_quality_passed,
)


def _checkpoint_inventory_sha256(receipt: dict) -> str | None:
    digest = receipt.get("checkpoint_inventory_sha256")
    if isinstance(digest, str):
        return digest
    snapshot = receipt.get("checkpoint_snapshot")
    return snapshot.get("inventory_sha256") if isinstance(snapshot, dict) else None


class MiniMaxH3DecodedVideoComparator:
    @property
    def task_strategy(self) -> str:
        return "diffusion_media_generation"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        if int(trt.data.get("returncode", -1)) != 0:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message=f"Native MiniMax-H3 failed (rc={trt.data.get('returncode')})",
            )
        if int(ref.data.get("returncode", -1)) != 0:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message=f"HF MiniMax-H3 reference failed (rc={ref.data.get('returncode')})",
            )
        if trt.data.get("source_revision") != ref.data.get("source_revision"):
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message="TRT and HF receipts do not identify the same source revision",
            )
        trt_receipt = trt.data.get("receipt")
        ref_receipt = ref.data.get("receipt")
        if not isinstance(trt_receipt, dict) or trt_receipt.get("status") != "passed":
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message="Native MiniMax-H3 did not produce a passed receipt",
            )
        if not isinstance(ref_receipt, dict) or ref_receipt.get("status") != "passed":
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message="HF MiniMax-H3 did not produce a passed receipt",
            )
        if (
            trt_receipt.get("backend") != "tensorrt_native_single_device"
            or trt_receipt.get("world_size") != 1
            or trt_receipt.get("collective_transport") != "none"
        ):
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message="Native MiniMax-H3 receipt is not a no-collective single-device run",
            )
        expected_revision = trt.data.get("source_revision")
        if (
            trt_receipt.get("source_revision") != expected_revision
            or ref_receipt.get("source_revision") != expected_revision
        ):
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message="TRT and HF run receipts identify different source revisions",
            )
        trt_inventory = _checkpoint_inventory_sha256(trt_receipt)
        ref_inventory = _checkpoint_inventory_sha256(ref_receipt)
        if (
            not isinstance(trt_inventory, str)
            or re.fullmatch(r"[0-9a-f]{64}", trt_inventory) is None
            or trt_inventory != ref_inventory
        ):
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message="TRT and HF run receipts identify different checkpoints",
            )

        reference_path = Path(str(ref.data.get("frames_path", "")))
        candidate_path = Path(str(trt.data.get("frames_path", "")))
        if not reference_path.is_file() or not candidate_path.is_file():
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message="MiniMax-H3 decoded frame arrays are missing",
            )

        metrics_config = threshold.metrics
        decoded = compute_decoded_visual_metrics(
            reference_path,
            candidate_path,
            block_size=visual_block_size(metrics_config),
        )
        visual_gates = evaluate_visual_quality(decoded, metrics_config)
        metrics = {
            name: MetricResult(
                value=result.value,
                threshold=result.threshold,
                operator=result.operator,
                passed=result.passed,
                note=result.note,
            )
            for name, result in visual_gates.items()
        }
        passed = visual_quality_passed(visual_gates)
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule=(
                "exact finite decoded RGB shape AND low-frequency frame structure AND "
                "brightness profile AND temporal activity/profile AND non-degenerate "
                "frame contrast; PSNR/MAE are diagnostic only"
            ),
            message=(
                f"{'PASS' if passed else 'FAIL'}: low_frequency_correlation="
                f"{decoded.frame_low_frequency_correlation_minimum:.4f}/"
                f"{decoded.frame_low_frequency_correlation_mean:.4f} (min/mean), "
                f"temporal_correlation={decoded.temporal_activity_correlation:.4f}, "
                f"PSNR={decoded.psnr_db:.4f} dB (diagnostic), "
                f"MAE={decoded.mean_absolute_error:.8f} (diagnostic)"
            ),
        )


comparator = MiniMaxH3DecodedVideoComparator()
