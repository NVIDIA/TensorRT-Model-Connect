# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speech-to-speech comparator.

Compares TRT PersonaPlex-style speech output against reference with metrics:
- Frame count match
- Depth token match rate
- Audio token match rate
- Frame exact match rate
- RMS floor check
- Optional ASR consistency (transcript similarity)
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
) -> tuple[dict[str, MetricResult], bool, int | None]:
    """Compare an optional frame-major token stream exactly."""
    import numpy as np

    if trt_tokens is None and ref_tokens is None:
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
            threshold=token_threshold,
            operator=">=",
            passed=tokens_ok,
            note=(
                "exact match"
                if mismatch is None
                else f"first mismatch frame={mismatch}"
            ),
        ),
    }
    return metrics, frame_count_ok and tokens_ok, mismatch


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
        )
        metrics.update(input_metrics)
        all_pass = all_pass and input_ok
        if input_mismatch is not None:
            first_divergence = ("input_mimi", input_mismatch)

        # The temporal text stream precedes depth/audio generation.
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
            token_metric="output_text_token_match_rate",
            token_threshold=thresholds.get("output_text_token_match_rate", 1.0),
        )
        metrics.update(text_metrics)
        all_pass = all_pass and text_ok
        if first_divergence is None and text_mismatch is not None:
            first_divergence = ("temporal_text", text_mismatch)

        # Get final audio token arrays.
        trt_tokens = trt.data.get("output_tokens")
        ref_tokens = ref.data.get("reference_tokens")

        if trt_tokens is not None and ref_tokens is not None:
            trt_arr = np.asarray(trt_tokens)
            ref_arr = np.asarray(ref_tokens)
            output_mismatch = _first_mismatch_frame(trt_arr, ref_arr)
            if first_divergence is None and output_mismatch is not None:
                first_divergence = ("depth_audio", output_mismatch)

            frame_count_ok = trt_arr.shape[0] == ref_arr.shape[0]
            metrics["frame_count_match"] = MetricResult(
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
                depth_thresh = thresholds.get("depth_token_match_rate", 0.7)
                depth_ok = depth_rate >= depth_thresh
                metrics["depth_token_match_rate"] = MetricResult(
                    value=depth_rate, threshold=depth_thresh, operator=">=", passed=depth_ok)
                if not depth_ok:
                    all_pass = False

                # Audio token match (remaining columns)
                if trt_aligned.shape[1] > 1:
                    audio_cols = trt_aligned[:, 1:]
                    ref_audio_cols = ref_aligned[:, 1:]
                    audio_matches = np.sum(audio_cols == ref_audio_cols)
                    total_audio = audio_cols.size
                    audio_rate = float(audio_matches) / total_audio if total_audio > 0 else 0.0
                    audio_thresh = thresholds.get("audio_token_match_rate", 0.7)
                    audio_ok = audio_rate >= audio_thresh
                    metrics["audio_token_match_rate"] = MetricResult(
                        value=audio_rate, threshold=audio_thresh, operator=">=", passed=audio_ok)
                    if not audio_ok:
                        all_pass = False
            else:
                # 1D token comparison (flat match)
                flat_trt = trt_aligned.flatten()
                flat_ref = ref_aligned.flatten()
                n_compare = min(len(flat_trt), len(flat_ref))
                if n_compare > 0:
                    match_rate = float(np.sum(flat_trt[:n_compare] == flat_ref[:n_compare])) / n_compare
                    token_thresh = thresholds.get("speech_min_token_match", 0.8)
                    token_ok = match_rate >= token_thresh
                    metrics["speech_min_token_match"] = MetricResult(
                        value=match_rate, threshold=token_thresh, operator=">=", passed=token_ok)
                    if not token_ok:
                        all_pass = False

            # Frame exact match rate
            frame_exact = 0
            for i in range(n_frames):
                if np.array_equal(trt_aligned[i], ref_aligned[i]):
                    frame_exact += 1
            frame_rate = float(frame_exact) / n_frames
            frame_thresh = thresholds.get(
                "frame_exact_match_rate",
                thresholds.get("speech_min_frame_exact", 0.7),
            )
            frame_ok = frame_rate >= frame_thresh
            metrics["frame_exact_match_rate"] = MetricResult(
                value=frame_rate,
                threshold=frame_thresh,
                operator=">=",
                passed=frame_ok,
                note=(
                    "exact match"
                    if output_mismatch is None
                    else f"first mismatch frame={output_mismatch}"
                ),
            )
            if not frame_ok:
                all_pass = False

        elif trt_tokens is None:
            logger.warning("No TRT output tokens available")
        elif ref_tokens is None:
            logger.warning("No reference tokens available for comparison")

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
