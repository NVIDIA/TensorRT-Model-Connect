# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed human-visible quality comparator for MiniMax-H3."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from .contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)
from tests.e2e.models.minimax_h3.visual_metrics import (
    compute_decoded_visual_metrics,
    evaluate_visual_quality,
    visual_block_size,
    visual_quality_passed,
)
from tests.e2e.models.minimax_h3.audio_metrics import (
    audio_quality_passed,
    compute_decoded_audio_metrics,
    evaluate_audio_quality,
    read_float32_wav,
)
from tests.e2e.models.minimax_h3.receipt_contracts import (
    validate_ref2va_receipt_contract,
)
from tensorrt_model_connect.families.minimax_h3.provenance import stable_file_record


def _checkpoint_inventory_sha256(receipt: dict) -> str | None:
    digest = receipt.get("checkpoint_inventory_sha256")
    if isinstance(digest, str):
        return digest
    snapshot = receipt.get("checkpoint_snapshot")
    return snapshot.get("inventory_sha256") if isinstance(snapshot, dict) else None


def _validated_audio_evidence(
    output: StageOutput,
    receipt: dict,
    *,
    label: str,
) -> tuple[Path, int]:
    audio_path = Path(str(output.data.get("audio_path", "")))
    wav_path = Path(str(output.data.get("wav_path", "")))
    if not audio_path.is_file() or not wav_path.is_file():
        raise ValueError(f"MiniMax-H3 {label} stereo audio artifacts are missing")

    audio_record, _ = stable_file_record(audio_path, f"{label} decoded audio")
    wav_record, _ = stable_file_record(wav_path, f"{label} decoded audio WAV")
    if receipt.get("audio") != audio_record or receipt.get("audio_wav") != wav_record:
        raise ValueError(f"MiniMax-H3 {label} audio artifacts do not match their receipt")

    samples = np.load(audio_path, mmap_mode="r", allow_pickle=False)
    wav = read_float32_wav(wav_path)
    if samples.shape != wav.samples.shape or not np.array_equal(samples, wav.samples):
        raise ValueError(f"MiniMax-H3 {label} WAV does not preserve its channel-major audio array")
    sample_rate = receipt.get("audio_sample_rate_hz")
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, int)
        or sample_rate <= 0
        or wav.sample_rate != sample_rate
        or output.data.get("sample_rate") != sample_rate
        or receipt.get("audio_shape") != [int(value) for value in samples.shape]
        or receipt.get("audio_num_samples_per_channel") != samples.shape[1]
        or receipt.get("audio_all_finite") is not True
        or receipt.get("audio_layout") != "channel_major"
        or receipt.get("audio_encoding") != "float32"
        or receipt.get("audio_wav_encoding") != "ieee_float32le"
    ):
        raise ValueError(f"MiniMax-H3 {label} audio metadata is inconsistent")
    return audio_path, sample_rate


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

        try:
            validate_ref2va_receipt_contract(trt_receipt, ref_receipt)
        except (TypeError, ValueError) as error:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message=str(error),
            )

        reference_path = Path(str(ref.data.get("frames_path", "")))
        candidate_path = Path(str(trt.data.get("frames_path", "")))
        if not reference_path.is_file() or not candidate_path.is_file():
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message="MiniMax-H3 decoded frame arrays are missing",
            )

        try:
            reference_audio_path, reference_sample_rate = _validated_audio_evidence(
                ref,
                ref_receipt,
                label="HF reference",
            )
            candidate_audio_path, candidate_sample_rate = _validated_audio_evidence(
                trt,
                trt_receipt,
                label="native candidate",
            )
        except (OSError, TypeError, ValueError) as error:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message=str(error),
            )

        metrics_config = threshold.metrics
        decoded = compute_decoded_visual_metrics(
            reference_path,
            candidate_path,
            block_size=visual_block_size(metrics_config),
        )
        visual_gates = evaluate_visual_quality(decoded, metrics_config)
        decoded_audio = compute_decoded_audio_metrics(
            reference_audio_path,
            candidate_audio_path,
            reference_sample_rate=reference_sample_rate,
            candidate_sample_rate=candidate_sample_rate,
        )
        audio_gates = evaluate_audio_quality(decoded_audio, metrics_config)
        metrics = {
            name: MetricResult(
                value=result.value,
                threshold=result.threshold,
                operator=result.operator,
                passed=result.passed,
                note=result.note,
            )
            for name, result in {**visual_gates, **audio_gates}.items()
        }
        passed = visual_quality_passed(visual_gates) and audio_quality_passed(audio_gates)
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule=(
                "exact finite decoded RGB shape AND low-frequency frame structure AND "
                "brightness profile AND temporal activity/profile AND non-degenerate "
                "frame contrast AND exact finite stereo audio shape/rate/duration AND "
                "direct per-channel HF waveform correlation/NRMSE/SI-SDR; visual "
                "PSNR/MAE and audio maximum absolute error are diagnostic only"
            ),
            message=(
                f"{'PASS' if passed else 'FAIL'}: low_frequency_correlation="
                f"{decoded.frame_low_frequency_correlation_minimum:.4f}/"
                f"{decoded.frame_low_frequency_correlation_mean:.4f} (min/mean), "
                f"temporal_correlation={decoded.temporal_activity_correlation:.4f}, "
                f"PSNR={decoded.psnr_db:.4f} dB (diagnostic), "
                f"MAE={decoded.mean_absolute_error:.8f} (diagnostic), "
                f"audio_correlation={decoded_audio.waveform_correlation_minimum:.6f}, "
                f"audio_NRMSE={decoded_audio.normalized_rmse_maximum:.6f}, "
                f"audio_SI-SDR={decoded_audio.si_sdr_db_minimum:.3f} dB"
            ),
        )


comparator = MiniMaxH3DecodedVideoComparator()
