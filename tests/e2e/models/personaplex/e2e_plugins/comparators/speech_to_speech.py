# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PersonaPlex speech-to-speech accuracy comparator.

The official and TRTMC free-running decoders are autoregressive. A one-logit
numeric difference can therefore change all later tokens even when the model's
conditional predictions remain equivalent. Free-generation agreement is kept
for diagnosis; teacher-forced top-1 agreement is the accuracy gate.
"""

from __future__ import annotations

import logging

from ..contracts import CompareResult, MetricResult, StageOutput, StageSpec, StageStatus, ThresholdProfile

logger = logging.getLogger(__name__)


def _first_mismatch_frame(trt_arr, ref_arr) -> int | None:
    """Return the first differing frame, including a trailing length mismatch."""
    import numpy as np

    if trt_arr.ndim == 0 or ref_arr.ndim == 0:
        return None if np.array_equal(trt_arr, ref_arr) else 0
    if trt_arr.shape[1:] != ref_arr.shape[1:]:
        return 0
    common_frames = min(trt_arr.shape[0], ref_arr.shape[0])
    for frame in range(common_frames):
        if not np.array_equal(trt_arr[frame], ref_arr[frame]):
            return frame
    if trt_arr.shape[0] != ref_arr.shape[0]:
        return common_frames
    return None


def _compare_exact_stream(
    trt_tokens,
    ref_tokens,
    *,
    availability_metric: str,
    frame_metric: str,
    token_metric: str,
    token_threshold: float,
    token_gates: bool = True,
    required: bool = False,
) -> tuple[dict[str, MetricResult], bool, int | None]:
    """Compare an optional frame-major token stream exactly."""
    import numpy as np

    if trt_tokens is None and ref_tokens is None:
        if required:
            return {
                availability_metric: MetricResult(
                    value=0.0,
                    threshold=1.0,
                    operator="==",
                    passed=False,
                    note="TRT available=False, reference available=False",
                )
            }, False, 0
        return {}, True, None
    if trt_tokens is None or ref_tokens is None:
        return {
            availability_metric: MetricResult(
                value=0.0,
                threshold=1.0,
                operator="==",
                passed=False,
                note=(
                    f"TRT available={trt_tokens is not None}, "
                    f"reference available={ref_tokens is not None}"
                ),
            )
        }, False, 0

    trt_arr = np.asarray(trt_tokens)
    ref_arr = np.asarray(ref_tokens)
    mismatch = _first_mismatch_frame(trt_arr, ref_arr)
    frame_count_ok = (
        trt_arr.ndim >= 1
        and ref_arr.ndim >= 1
        and trt_arr.shape[0] == ref_arr.shape[0]
    )
    common_frames = (
        min(trt_arr.shape[0], ref_arr.shape[0])
        if trt_arr.ndim >= 1 and ref_arr.ndim >= 1
        else 0
    )
    layout_ok = trt_arr.shape[1:] == ref_arr.shape[1:]
    match_rate = (
        float(np.mean(trt_arr[:common_frames] == ref_arr[:common_frames]))
        if common_frames > 0 and layout_ok
        else 0.0
    )
    tokens_ok = match_rate >= token_threshold
    metrics = {
        frame_metric: MetricResult(
            value=1.0 if frame_count_ok else 0.0,
            threshold=1.0,
            operator="==",
            passed=frame_count_ok,
            note=(
                f"TRT frames={trt_arr.shape[0] if trt_arr.ndim else 0}, "
                f"reference frames={ref_arr.shape[0] if ref_arr.ndim else 0}"
            ),
        ),
        token_metric: MetricResult(
            value=match_rate,
            threshold=token_threshold if token_gates else None,
            operator=">=" if token_gates else "info",
            passed=tokens_ok if token_gates else True,
            note=(
                "exact match"
                if mismatch is None
                else (
                    f"first mismatch frame={mismatch}"
                    + ("" if token_gates else "; free-running diagnostic")
                )
            ),
        ),
    }
    return metrics, frame_count_ok and (tokens_ok or not token_gates), mismatch


class SpeechToSpeechComparator:
    """Compares TRT speech-to-speech output against reference tokens/audio."""

    @property
    def task_strategy(self) -> str:
        return "speech_to_speech"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        import numpy as np

        metrics: dict[str, MetricResult] = {}
        thresholds = threshold.metrics
        all_pass = True
        first_divergence: tuple[str, int] | None = None

        # Check TRT returncode
        if trt.data.get("returncode", -1) != 0:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics={},
                message=f"TRT speech-to-speech failed (rc={trt.data.get('returncode')})",
            )

        # The codec is FP32 on both sides, so input Mimi tokens must be exact.
        input_metrics, input_ok, input_mismatch = _compare_exact_stream(
            trt.data.get("input_codec_tokens"),
            ref.data.get("reference_input_codec_tokens"),
            availability_metric="input_mimi_tokens_available",
            frame_metric="input_mimi_frame_count_match",
            token_metric="input_mimi_token_match_rate",
            token_threshold=thresholds.get("input_mimi_token_match_rate", 1.0),
            required="input_mimi_token_match_rate" in thresholds,
        )
        metrics.update(input_metrics)
        all_pass = all_pass and input_ok
        if input_mismatch is not None:
            first_divergence = ("input_mimi", input_mismatch)

        # The free-running temporal stream is useful for locating the first
        # divergence, but is not an accuracy gate: autoregressive divergence
        # amplifies a single numeric difference into a different token trace.
        trt_text_tokens = trt.data.get("output_text_tokens")
        ref_text_tokens = ref.data.get("reference_text_tokens")
        if trt_text_tokens is not None:
            trt_text_tokens = np.asarray(trt_text_tokens).reshape(-1)
        if ref_text_tokens is not None:
            ref_text_tokens = np.asarray(ref_text_tokens).reshape(-1)
        text_metrics, text_ok, text_mismatch = _compare_exact_stream(
            trt_text_tokens,
            ref_text_tokens,
            availability_metric="output_text_tokens_available",
            frame_metric="output_text_frame_count_match",
            token_metric="free_generation_text_token_match_rate",
            token_threshold=1.0,
            token_gates=False,
        )
        metrics.update(text_metrics)
        all_pass = all_pass and text_ok
        if first_divergence is None and text_mismatch is not None:
            first_divergence = ("temporal_text", text_mismatch)

        # Get final audio token arrays.
        trt_tokens = trt.data.get("output_tokens")
        ref_tokens = ref.data.get("reference_tokens")
        audio_tokens_available = trt_tokens is not None and ref_tokens is not None
        metrics["free_generation_audio_tokens_available"] = MetricResult(
            value=1.0 if audio_tokens_available else 0.0,
            threshold=1.0,
            operator="==",
            passed=audio_tokens_available,
            note=(
                f"TRT available={trt_tokens is not None}, "
                f"reference available={ref_tokens is not None}"
            ),
        )
        all_pass = all_pass and audio_tokens_available

        if audio_tokens_available:
            trt_arr = np.asarray(trt_tokens)
            ref_arr = np.asarray(ref_tokens)
            output_mismatch = _first_mismatch_frame(trt_arr, ref_arr)
            if first_divergence is None and output_mismatch is not None:
                first_divergence = ("depth_audio", output_mismatch)

            frame_count_ok = trt_arr.shape[0] == ref_arr.shape[0]
            metrics["free_generation_frame_count_match"] = MetricResult(
                value=1.0 if frame_count_ok else 0.0,
                threshold=1.0,
                operator="==",
                passed=frame_count_ok,
                note=(
                    f"TRT frames={trt_arr.shape[0]}, "
                    f"reference frames={ref_arr.shape[0]}"
                ),
            )
            if not frame_count_ok:
                all_pass = False

            # Align frame count
            n_frames = min(trt_arr.shape[0], ref_arr.shape[0])
            if n_frames == 0:
                return CompareResult(
                    stage_name=stage.name,
                    status=StageStatus.ERROR.value,
                    metrics={},
                    message="No frames to compare (empty token arrays)",
                )

            trt_aligned = trt_arr[:n_frames]
            ref_aligned = ref_arr[:n_frames]

            # Depth token match (first column if multi-column)
            if trt_aligned.ndim >= 2 and trt_aligned.shape[1] >= 1:
                depth_matches = np.sum(trt_aligned[:, 0] == ref_aligned[:, 0])
                depth_rate = float(depth_matches) / n_frames
                metrics["free_generation_depth_token_match_rate"] = MetricResult(
                    value=depth_rate,
                    threshold=None,
                    operator="info",
                    passed=True,
                    note="free-running diagnostic",
                )

                # Audio token match (remaining columns)
                if trt_aligned.shape[1] > 1:
                    audio_cols = trt_aligned[:, 1:]
                    ref_audio_cols = ref_aligned[:, 1:]
                    audio_matches = np.sum(audio_cols == ref_audio_cols)
                    total_audio = audio_cols.size
                    audio_rate = float(audio_matches) / total_audio if total_audio > 0 else 0.0
                    metrics["free_generation_audio_token_match_rate"] = MetricResult(
                        value=audio_rate,
                        threshold=None,
                        operator="info",
                        passed=True,
                        note="free-running diagnostic",
                    )
            else:
                # 1D token comparison (flat match)
                flat_trt = trt_aligned.flatten()
                flat_ref = ref_aligned.flatten()
                n_compare = min(len(flat_trt), len(flat_ref))
                if n_compare > 0:
                    match_rate = float(np.sum(flat_trt[:n_compare] == flat_ref[:n_compare])) / n_compare
                    metrics["free_generation_token_match_rate"] = MetricResult(
                        value=match_rate,
                        threshold=None,
                        operator="info",
                        passed=True,
                        note="free-running diagnostic",
                    )

            # Frame exact match rate
            frame_exact = 0
            for i in range(n_frames):
                if np.array_equal(trt_aligned[i], ref_aligned[i]):
                    frame_exact += 1
            frame_rate = float(frame_exact) / n_frames
            metrics["free_generation_frame_exact_match_rate"] = MetricResult(
                value=frame_rate,
                threshold=None,
                operator="info",
                passed=True,
                note=(
                    "exact match; free-running diagnostic"
                    if output_mismatch is None
                    else (
                        f"first mismatch frame={output_mismatch}; "
                        "free-running diagnostic"
                    )
                ),
            )

        elif trt_tokens is None:
            logger.warning("No TRT output tokens available")
        elif ref_tokens is None:
            logger.warning("No reference tokens available for comparison")

        # Teacher-forced replay preserves the official history and measures
        # the TRT model's conditional top-1 choice before forcing the target.
        # Trace/reference identity protects the validity of the replay. The
        # conditional top-1 rates then gate the model decision accuracy without
        # allowing free-running cascade amplification to dominate the result.
        teacher_fields = (
            "teacher_text_target_tokens",
            "teacher_text_predicted_tokens",
            "teacher_audio_target_tokens",
            "teacher_audio_predicted_tokens",
        )
        teacher_values = [trt.data.get(name) for name in teacher_fields]
        teacher_required = any(
            name in thresholds
            for name in (
                "teacher_text_top1_match_rate",
                "teacher_depth_top1_match_rate",
                "teacher_audio_top1_match_rate",
            )
        )
        if teacher_required or any(value is not None for value in teacher_values):
            teacher_available = all(value is not None for value in teacher_values)
            metrics["teacher_replay_available"] = MetricResult(
                value=1.0 if teacher_available else 0.0,
                threshold=1.0,
                operator="==",
                passed=teacher_available,
            )
            all_pass = all_pass and teacher_available
            if teacher_available:
                text_target = np.asarray(teacher_values[0]).reshape(-1)
                text_predicted = np.asarray(teacher_values[1]).reshape(-1)
                audio_target = np.asarray(teacher_values[2])
                audio_predicted = np.asarray(teacher_values[3])
                replay_shapes_ok = (
                    text_target.shape == text_predicted.shape
                    and audio_target.shape == audio_predicted.shape
                    and audio_target.ndim == 2
                    and audio_target.shape[0] == text_target.shape[0]
                    and audio_target.shape[1] >= 1
                    and text_target.size > 0
                )
                metrics["teacher_replay_shape_valid"] = MetricResult(
                    value=1.0 if replay_shapes_ok else 0.0,
                    threshold=1.0,
                    operator="==",
                    passed=replay_shapes_ok,
                )
                all_pass = all_pass and replay_shapes_ok
                if replay_shapes_ok:
                    ref_text_array = (
                        np.asarray(ref_text_tokens).reshape(-1)
                        if ref_text_tokens is not None
                        else np.empty(0, dtype=np.int32)
                    )
                    ref_audio_array = (
                        np.asarray(ref_tokens)
                        if ref_tokens is not None
                        else np.empty((0, audio_target.shape[1]), dtype=np.int32)
                    )
                    trace_matches_reference = (
                        np.array_equal(text_target, ref_text_array)
                        and np.array_equal(audio_target, ref_audio_array)
                    )
                    metrics["teacher_trace_reference_exact"] = MetricResult(
                        value=1.0 if trace_matches_reference else 0.0,
                        threshold=1.0,
                        operator="==",
                        passed=trace_matches_reference,
                    )
                    all_pass = all_pass and trace_matches_reference

                    text_rate = float(np.mean(text_predicted == text_target))
                    depth_rate = float(
                        np.mean(audio_predicted[:, 0] == audio_target[:, 0])
                    )
                    if audio_target.shape[1] > 1:
                        audio_rate = float(
                            np.mean(audio_predicted[:, 1:] == audio_target[:, 1:])
                        )
                    else:
                        audio_rate = depth_rate
                    frame_rate = float(
                        np.mean(np.all(audio_predicted == audio_target, axis=1))
                    )
                    for name, value, default_threshold, note in (
                        (
                            "teacher_text_top1_match_rate",
                            text_rate,
                            0.99,
                            "conditional temporal top-1",
                        ),
                        (
                            "teacher_depth_top1_match_rate",
                            depth_rate,
                            0.99,
                            "conditional first audio codebook top-1",
                        ),
                        (
                            "teacher_audio_top1_match_rate",
                            audio_rate,
                            0.98,
                            "conditional remaining audio codebooks top-1",
                        ),
                    ):
                        metric_threshold = thresholds.get(name, default_threshold)
                        metrics[name] = MetricResult(
                            value=value,
                            threshold=metric_threshold,
                            operator=">=",
                            passed=value >= metric_threshold,
                            note=note,
                        )
                        all_pass = all_pass and value >= metric_threshold
                    metrics["teacher_frame_top1_exact_rate"] = MetricResult(
                        value=frame_rate,
                        threshold=None,
                        operator="info",
                        passed=True,
                        note="all audio codebooks conditionally exact; diagnostic",
                    )

        # RMS floor check
        rms = trt.data.get("rms", 0.0)
        if rms > 0 or trt.data.get("wav_exists", False):
            rms_thresh = thresholds.get("speech_min_rms", 0.001)
            ref_rms = ref.data.get("rms")
            reference_is_below_floor = (
                ref_rms is not None and float(ref_rms) < rms_thresh
            )
            rms_ok = rms >= rms_thresh or reference_is_below_floor
            note = ""
            if reference_is_below_floor:
                note = (
                    f"reference is also below the floor "
                    f"(reference_rms={float(ref_rms):.8g})"
                )
            metrics["rms_floor"] = MetricResult(
                value=rms,
                threshold=rms_thresh,
                operator=">=",
                passed=rms_ok,
                note=note,
            )
            if not rms_ok:
                all_pass = False

        n_passed = sum(1 for m in metrics.values() if m.passed)
        divergence_message = (
            "first divergence=none"
            if first_divergence is None
            else (
                f"first divergence={first_divergence[0]} "
                f"frame={first_divergence[1]}"
            )
        )
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if all_pass else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all metrics must pass",
            message=f"{'PASS' if all_pass else 'FAIL'}: "
                    f"{n_passed}/{len(metrics)} metrics passed; {divergence_message}",
        )


plugin = SpeechToSpeechComparator()
