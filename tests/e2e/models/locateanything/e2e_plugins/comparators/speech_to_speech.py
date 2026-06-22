"""Speech-to-speech comparator.

Compares TRT PersonaPlex-style speech output against reference with metrics:
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

        # Check TRT returncode
        if trt.data.get("returncode", -1) != 0:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics={},
                message=f"TRT speech-to-speech failed (rc={trt.data.get('returncode')})",
            )

        # Get token arrays
        trt_tokens = trt.data.get("output_tokens")
        ref_tokens = ref.data.get("reference_tokens")

        if trt_tokens is not None and ref_tokens is not None:
            trt_arr = np.asarray(trt_tokens)
            ref_arr = np.asarray(ref_tokens)

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
                value=frame_rate, threshold=frame_thresh, operator=">=", passed=frame_ok)
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
            rms_ok = rms >= rms_thresh
            metrics["rms_floor"] = MetricResult(
                value=rms, threshold=rms_thresh, operator=">=", passed=rms_ok)
            if not rms_ok:
                all_pass = False

        n_passed = sum(1 for m in metrics.values() if m.passed)
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if all_pass else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all metrics must pass",
            message=f"{'PASS' if all_pass else 'FAIL'}: "
                    f"{n_passed}/{len(metrics)} metrics passed",
        )


plugin = SpeechToSpeechComparator()
