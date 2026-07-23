# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-profile functional comparator for Wan2.2 TI2V-5B video output."""

from __future__ import annotations

from ..contracts import CompareResult, MetricResult, StageOutput, StageSpec, ThresholdProfile
from .frame_accuracy import compare_png_sequences

_REQUIRED_FRAME_THRESHOLDS = (
    "exact_num_frames",
    "exact_video_width",
    "exact_video_height",
    "min_pixel_mean",
    "max_pixel_mean",
    "min_pixel_std",
)

_REQUIRED_REFERENCE_THRESHOLDS = (
    "min_cosine_uint8",
    "min_frame_cosine_uint8",
    "max_rmse_uint8",
    "min_temporal_motion_ratio",
    "max_temporal_motion_ratio",
    "min_temporal_profile_correlation",
    "min_active_transition_fraction",
)

_DIAGNOSTIC_PIXEL_METRICS = {
    "cosine_uint8",
    "minimum_frame_cosine_uint8",
    "rmse_uint8",
    "maximum_frame_rmse_uint8",
}

_NATIVE_ACCEPTANCE_VALUES = {
    "kind": "native_visual_semantic_acceptance",
    "reference_role": "diagnostic",
    "requires_nightly_vlm": True,
    "vlm_frame_samples": 6,
}

_DIAGNOSTIC_NOTE = "diagnostic under native visual semantic acceptance; threshold unchanged"


def _metric(
    value: float,
    threshold: float,
    operator: str,
    passed: bool,
    *,
    note: str = "",
) -> MetricResult:
    return MetricResult(
        value=value,
        threshold=threshold,
        operator=operator,
        passed=passed,
        note=note,
    )


def _valid_native_acceptance(policy: object) -> bool:
    if not isinstance(policy, dict):
        return False
    if any(policy.get(key) != value for key, value in _NATIVE_ACCEPTANCE_VALUES.items()):
        return False
    return isinstance(policy.get("rationale"), str) and bool(policy["rationale"].strip())


class DiffusionComparator:
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
        reference_data = ref.data or {}
        has_external_reference = not bool(reference_data.get("_invariant_only", False))
        native_acceptance_value = reference_data.get("native_acceptance")
        if native_acceptance_value is not None and not _valid_native_acceptance(
            native_acceptance_value
        ):
            return CompareResult(
                stage_name=stage.name,
                status="failed",
                composite_rule="native visual acceptance policy must be complete and valid",
                message="Wan2.2 TI2V native_acceptance policy is invalid",
            )
        native_acceptance = _valid_native_acceptance(native_acceptance_value)
        if native_acceptance and not has_external_reference:
            return CompareResult(
                stage_name=stage.name,
                status="failed",
                composite_rule="native visual acceptance requires the official reference",
                message=(
                    "Wan2.2 TI2V native_acceptance cannot be used with an invariant-only reference"
                ),
            )
        required_thresholds = set(_REQUIRED_FRAME_THRESHOLDS)
        if has_external_reference:
            required_thresholds.update(_REQUIRED_REFERENCE_THRESHOLDS)
        missing = sorted(required_thresholds - set(threshold.metrics))
        if missing:
            return CompareResult(
                stage_name=stage.name,
                status="failed",
                composite_rule="all model-owned frame thresholds must be loaded",
                message=f"Wan2.2 TI2V threshold sidecar is incomplete: {missing}",
            )

        expected_frames = int(threshold.metrics["exact_num_frames"])
        expected_width = int(threshold.metrics["exact_video_width"])
        expected_height = int(threshold.metrics["exact_video_height"])
        min_mean = float(threshold.metrics["min_pixel_mean"])
        max_mean = float(threshold.metrics["max_pixel_mean"])
        min_std = float(threshold.metrics["min_pixel_std"])

        data = trt.data or {}
        stats = data.get("frame_stats") or {}
        returncode = int(data.get("returncode", -1))
        frame_count = int(data.get("num_frames", 0))
        width = int(stats.get("width", 0))
        height = int(stats.get("height", 0))
        dimensions_consistent = bool(stats.get("dimensions_consistent", False))
        pixel_mean = float(stats.get("mean", 0.0))
        pixel_std = float(stats.get("std", 0.0))

        metrics = {
            "returncode": _metric(float(returncode), 0.0, "==", returncode == 0),
            "exact_num_frames": _metric(
                float(frame_count), float(expected_frames), "==", frame_count == expected_frames
            ),
            "exact_video_width": _metric(
                float(width), float(expected_width), "==", width == expected_width
            ),
            "exact_video_height": _metric(
                float(height), float(expected_height), "==", height == expected_height
            ),
            "frame_dimensions_consistent": _metric(
                float(dimensions_consistent), 1.0, "==", dimensions_consistent
            ),
            "pixel_mean_min": _metric(pixel_mean, min_mean, ">=", pixel_mean >= min_mean),
            "pixel_mean_max": _metric(pixel_mean, max_mean, "<=", pixel_mean <= max_mean),
            "pixel_std_min": _metric(pixel_std, min_std, ">=", pixel_std >= min_std),
        }
        if has_external_reference:
            error = self._add_reference_metrics(
                metrics,
                data,
                reference_data,
                threshold,
                expected_frames,
                native_acceptance=native_acceptance,
            )
            if error:
                return CompareResult(
                    stage_name=stage.name,
                    status="failed",
                    metrics=metrics,
                    composite_rule="all external-reference frames must be present and comparable",
                    message=f"Wan2.2 TI2V all-frame reference comparison failed: {error}",
                )
        passed = all(
            metric.passed
            for name, metric in metrics.items()
            if not (native_acceptance and name in _DIAGNOSTIC_PIXEL_METRICS)
        )
        reference_rule = ""
        if native_acceptance:
            reference_rule = (
                f" AND all {expected_frames} official-Wan/TRT frames are retained "
                "for diagnostic pixel comparison AND temporal motion remains aligned; "
                "raw pixel parity is not claimed; "
                "six-frame Nightly VLM semantic acceptance is required"
            )
        elif has_external_reference:
            reference_rule = f" AND all {expected_frames} official-Wan/TRT frames meet cosine and RMSE thresholds"
        return CompareResult(
            stage_name=stage.name,
            status="passed" if passed else "failed",
            metrics=metrics,
            composite_rule=(
                f"native command succeeds AND output is exactly {expected_frames} "
                f"{expected_width}x{expected_height} frames AND pixels are non-degenerate"
                f"{reference_rule}"
            ),
            message=f"Wan2.2 TI2V fixed-profile qualification: {'PASS' if passed else 'FAIL'}",
        )

    @staticmethod
    def _add_reference_metrics(
        metrics: dict[str, MetricResult],
        trt_data: dict,
        reference_data: dict,
        threshold: ThresholdProfile,
        expected_frames: int,
        *,
        native_acceptance: bool,
    ) -> str:
        reference_returncode = int(reference_data.get("returncode", -1))
        reference_frames = int(reference_data.get("num_frames", 0))
        metrics["reference_returncode"] = _metric(
            float(reference_returncode), 0.0, "==", reference_returncode == 0
        )
        metrics["reference_exact_num_frames"] = _metric(
            float(reference_frames),
            float(expected_frames),
            "==",
            reference_frames == expected_frames,
        )
        try:
            accuracy = compare_png_sequences(
                reference_data.get("frame_paths") or [],
                trt_data.get("frame_paths") or [],
            )
        except (OSError, ValueError) as exc:
            metrics["all_reference_frames_compared"] = _metric(
                0.0, float(expected_frames), "==", False
            )
            return str(exc)

        min_cosine = float(threshold.metrics["min_cosine_uint8"])
        min_frame_cosine = float(threshold.metrics["min_frame_cosine_uint8"])
        max_rmse = float(threshold.metrics["max_rmse_uint8"])
        min_temporal_motion_ratio = float(
            threshold.metrics["min_temporal_motion_ratio"])
        max_temporal_motion_ratio = float(
            threshold.metrics["max_temporal_motion_ratio"])
        min_temporal_profile_correlation = float(
            threshold.metrics["min_temporal_profile_correlation"])
        min_active_transition_fraction = float(
            threshold.metrics["min_active_transition_fraction"])
        compared_frames = float(accuracy["frame_count"])
        cosine = float(accuracy["cosine_uint8"])
        frame_cosine = float(accuracy["minimum_frame_cosine_uint8"])
        rmse = float(accuracy["rmse_uint8"])
        frame_rmse = float(accuracy["maximum_frame_rmse_uint8"])
        reference_temporal_mae = float(
            accuracy["reference_temporal_mae_uint8"])
        trt_temporal_mae = float(accuracy["trt_temporal_mae_uint8"])
        temporal_motion_ratio = float(accuracy["temporal_motion_ratio"])
        temporal_profile_correlation = float(
            accuracy["temporal_profile_correlation"])
        reference_active_transition_fraction = float(
            accuracy["reference_active_transition_fraction"])
        trt_active_transition_fraction = float(
            accuracy["trt_active_transition_fraction"])
        diagnostic_note = _DIAGNOSTIC_NOTE if native_acceptance else ""
        metrics.update(
            {
                "all_reference_frames_compared": _metric(
                    compared_frames,
                    float(expected_frames),
                    "==",
                    compared_frames == expected_frames,
                ),
                "cosine_uint8": _metric(
                    cosine,
                    min_cosine,
                    ">=",
                    cosine >= min_cosine,
                    note=diagnostic_note,
                ),
                "minimum_frame_cosine_uint8": _metric(
                    frame_cosine,
                    min_frame_cosine,
                    ">=",
                    frame_cosine >= min_frame_cosine,
                    note=diagnostic_note,
                ),
                "rmse_uint8": _metric(
                    rmse,
                    max_rmse,
                    "<=",
                    rmse <= max_rmse,
                    note=diagnostic_note,
                ),
                "maximum_frame_rmse_uint8": _metric(
                    frame_rmse,
                    max_rmse,
                    "<=",
                    frame_rmse <= max_rmse,
                    note=diagnostic_note,
                ),
                "reference_temporal_mae_uint8": _metric(
                    reference_temporal_mae,
                    0.0,
                    ">",
                    reference_temporal_mae > 0.0,
                ),
                "trt_temporal_mae_uint8": _metric(
                    trt_temporal_mae,
                    0.0,
                    ">",
                    trt_temporal_mae > 0.0,
                ),
                "reference_active_transition_fraction": _metric(
                    reference_active_transition_fraction,
                    min_active_transition_fraction,
                    ">=",
                    reference_active_transition_fraction
                    >= min_active_transition_fraction,
                ),
                "trt_active_transition_fraction": _metric(
                    trt_active_transition_fraction,
                    min_active_transition_fraction,
                    ">=",
                    trt_active_transition_fraction
                    >= min_active_transition_fraction,
                ),
                "temporal_motion_ratio_min": _metric(
                    temporal_motion_ratio,
                    min_temporal_motion_ratio,
                    ">=",
                    temporal_motion_ratio >= min_temporal_motion_ratio,
                ),
                "temporal_motion_ratio_max": _metric(
                    temporal_motion_ratio,
                    max_temporal_motion_ratio,
                    "<=",
                    temporal_motion_ratio <= max_temporal_motion_ratio,
                ),
                "temporal_profile_correlation": _metric(
                    temporal_profile_correlation,
                    min_temporal_profile_correlation,
                    ">=",
                    temporal_profile_correlation
                    >= min_temporal_profile_correlation,
                ),
            }
        )
        return ""


plugin = DiffusionComparator()
