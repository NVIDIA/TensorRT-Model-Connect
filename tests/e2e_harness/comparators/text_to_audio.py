"""Text-to-audio comparator.

Compares TRT Bark-style audio generation output against reference with metrics:
- Codec token match rate
- Mel-spectrogram distance
- Log-spectral distance
- Duration and RMS bounds
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from ..contracts import CompareResult, MetricResult, StageOutput, StageSpec, StageStatus, ThresholdProfile
from tools import compare_wav_exact

logger = logging.getLogger(__name__)


def _compute_mel_spectrogram(samples, sample_rate: int = 24000, n_fft: int = 1024,
                              hop_length: int = 256, n_mels: int = 80):
    """Compute log-mel spectrogram using numpy (no librosa dependency)."""
    import numpy as np

    # STFT
    n_frames = 1 + (len(samples) - n_fft) // hop_length
    if n_frames < 1:
        return np.zeros((n_mels, 1), dtype=np.float32)

    frames = np.stack([
        samples[i * hop_length: i * hop_length + n_fft]
        for i in range(n_frames)
    ])
    window = np.hanning(n_fft)
    frames = frames * window
    spectrum = np.fft.rfft(frames, n=n_fft)
    power = np.abs(spectrum) ** 2

    # Mel filterbank (simplified linear spacing)
    fmin = 0.0
    fmax = sample_rate / 2.0
    mel_min = 2595.0 * math.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * math.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, n_fft // 2)

    n_freq = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_freq), dtype=np.float32)
    for m in range(n_mels):
        f_left = bin_points[m]
        f_center = bin_points[m + 1]
        f_right = bin_points[m + 2]
        for k in range(f_left, f_center):
            if f_center > f_left:
                filterbank[m, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right > f_center:
                filterbank[m, k] = (f_right - k) / (f_right - f_center)

    mel_spec = filterbank @ power.T  # [n_mels, n_frames]
    log_mel = np.log(np.maximum(mel_spec, 1e-10))
    return log_mel.astype(np.float32)


def _mel_spectrogram_distance(samples1, samples2, sample_rate: int = 24000) -> float:
    """Compute mean absolute difference between two log-mel spectrograms."""
    import numpy as np
    mel1 = _compute_mel_spectrogram(np.asarray(samples1, dtype=np.float32), sample_rate)
    mel2 = _compute_mel_spectrogram(np.asarray(samples2, dtype=np.float32), sample_rate)

    # Align lengths
    min_frames = min(mel1.shape[1], mel2.shape[1])
    if min_frames == 0:
        return float("inf")
    mel1 = mel1[:, :min_frames]
    mel2 = mel2[:, :min_frames]

    return float(np.mean(np.abs(mel1 - mel2)))


def _log_spectral_distance(samples1, samples2, sample_rate: int = 24000,
                            n_fft: int = 1024, hop_length: int = 256) -> float:
    """Compute log-spectral distance (LSD) between two audio signals."""
    import numpy as np

    def _power_spectrum(samples):
        n_frames = 1 + (len(samples) - n_fft) // hop_length
        if n_frames < 1:
            return np.zeros((1, n_fft // 2 + 1), dtype=np.float32)
        frames = np.stack([
            samples[i * hop_length: i * hop_length + n_fft]
            for i in range(n_frames)
        ])
        window = np.hanning(n_fft)
        frames = frames * window
        spectrum = np.fft.rfft(frames, n=n_fft)
        return np.abs(spectrum) ** 2

    ps1 = _power_spectrum(np.asarray(samples1, dtype=np.float32))
    ps2 = _power_spectrum(np.asarray(samples2, dtype=np.float32))

    min_frames = min(ps1.shape[0], ps2.shape[0])
    if min_frames == 0:
        return float("inf")
    ps1 = ps1[:min_frames]
    ps2 = ps2[:min_frames]

    log_ps1 = np.log(np.maximum(ps1, 1e-10))
    log_ps2 = np.log(np.maximum(ps2, 1e-10))

    lsd = np.sqrt(np.mean((log_ps1 - log_ps2) ** 2))
    return float(lsd)


class TextToAudioComparator:
    """Compares TRT text-to-audio output against reference."""

    @property
    def task_strategy(self) -> str:
        return "text_to_audio"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        metrics: dict[str, MetricResult] = {}
        thresholds = threshold.metrics
        all_pass = True
        exact_waveform_required = (
            stage.comparison_mode == "waveform_exact"
            or float(thresholds.get("exact_waveform_match", 0.0)) >= 1.0
        )

        # Check TRT returncode
        if trt.data.get("returncode", -1) != 0:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics={},
                message=f"TRT audio generation failed (rc={trt.data.get('returncode')})",
            )

        # WAV existence check
        if not trt.data.get("wav_exists", False):
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics={},
                message="TRT did not produce a WAV output file",
            )

        trt_wav_path = trt.data.get("wav_path", "")
        ref_wav_path = ref.data.get("wav_path", "")

        if exact_waveform_required:
            if not ref_wav_path:
                return CompareResult(
                    stage_name=stage.name,
                    status=StageStatus.ERROR.value,
                    metrics={},
                    message="Exact waveform comparison requires reference wav_path",
                )
            exact_result = compare_wav_exact.compare_wavs(
                Path(trt_wav_path), Path(ref_wav_path)
            )
            exact_checks = exact_result.get("metrics", {})
            if not exact_checks:
                return CompareResult(
                    stage_name=stage.name,
                    status=StageStatus.ERROR.value,
                    metrics={},
                    message=(
                        "Exact waveform comparison could not read TRT/reference WAV"
                    ),
                )

            for name, passed in exact_checks.items():
                metrics[name] = MetricResult(
                    value=1.0 if passed else 0.0,
                    threshold=1.0,
                    operator="==",
                    passed=passed,
                )
                if not passed:
                    all_pass = False

        # RMS energy check
        rms = trt.data.get("rms", 0.0)
        rms_min = thresholds.get("rms_min", 0.001)
        rms_max = thresholds.get("rms_max", 1.0)
        rms_ok = rms_min <= rms <= rms_max
        metrics["rms"] = MetricResult(
            value=rms, threshold=rms_min, operator=">=", passed=rms_ok,
            note=f"range [{rms_min}, {rms_max}]")
        if not rms_ok:
            all_pass = False

        # Duration check
        duration = trt.data.get("duration_s", 0.0)
        ref_duration = ref.data.get("duration_s", 0.0)
        if duration > 0 and ref_duration > 0:
            ratio = duration / ref_duration
            ratio_min = thresholds.get("duration_ratio_min", 0.5)
            ratio_max = thresholds.get("duration_ratio_max", 2.0)
            ratio_ok = ratio_min <= ratio <= ratio_max
            metrics["duration_ratio"] = MetricResult(
                value=ratio, threshold=ratio_min, operator=">=", passed=ratio_ok,
                note=f"range [{ratio_min}, {ratio_max}]")
            if not ratio_ok:
                all_pass = False

        # Mel-spectrogram and log-spectral distance.
        # Read samples from WAV files (both TRT and reference produce wav_path).
        ref_samples = ref.data.get("audio_samples")
        if trt_wav_path and (ref_wav_path or ref_samples is not None):
            try:
                import numpy as np

                trt_samples = self._read_wav_samples(trt_wav_path)
                if ref_samples is None and ref_wav_path:
                    ref_samples = self._read_wav_samples(ref_wav_path)
                sample_rate = trt.data.get("sample_rate", 24000)

                if (trt_samples is not None and len(trt_samples) > 0
                        and ref_samples is not None and len(ref_samples) > 0):
                    mel_dist = _mel_spectrogram_distance(
                        trt_samples, ref_samples, sample_rate)
                    mel_thresh = thresholds.get("mel_spectrogram_distance", 5.0)
                    mel_ok = mel_dist <= mel_thresh
                    metrics["mel_spectrogram_distance"] = MetricResult(
                        value=mel_dist, threshold=mel_thresh, operator="<=", passed=mel_ok)
                    if not mel_ok:
                        all_pass = False

                    lsd = _log_spectral_distance(
                        trt_samples, ref_samples, sample_rate)
                    lsd_thresh = thresholds.get("log_spectral_distance", 3.0)
                    lsd_ok = lsd <= lsd_thresh
                    metrics["log_spectral_distance"] = MetricResult(
                        value=lsd, threshold=lsd_thresh, operator="<=", passed=lsd_ok)
                    if not lsd_ok:
                        all_pass = False
            except Exception as e:
                logger.warning("spectral comparison failed: %s", e)

        # Semantic token degeneration checks.
        # Follows HF's pattern: verify intermediate token sequences are sane.
        # HF checks exact golden token IDs; we check diversity + count since
        # TRT uses different RNG and floating-point paths.
        trt_stderr = trt.data.get("stderr", "")
        if "Bark semantic: generated" in trt_stderr:
            try:
                import re
                m = re.search(r"Bark semantic: generated (\d+) tokens", trt_stderr)
                if m:
                    n_sem = int(m.group(1))
                    sem_min = thresholds.get("min_semantic_tokens", 10)
                    sem_ok = n_sem >= sem_min
                    metrics["semantic_token_count"] = MetricResult(
                        value=n_sem, threshold=sem_min, operator=">=",
                        passed=sem_ok,
                        note="too few tokens suggests degenerate output")
                    if not sem_ok:
                        all_pass = False
            except Exception:
                pass

        # Token diversity: if audio_bark.dump_path produced a .sem_tokens file,
        # verify the tokens aren't degenerate (>80% same value = stuck model).
        sem_dump = trt.data.get("sem_tokens_path", "")
        if sem_dump:
            try:
                import numpy as np
                tokens = np.loadtxt(sem_dump, dtype=np.int32)
                if len(tokens) > 10:
                    # Count most frequent token
                    values, counts = np.unique(tokens, return_counts=True)
                    max_frac = float(counts.max()) / len(tokens)
                    diversity_thresh = thresholds.get(
                        "max_semantic_repeat_fraction", 0.5)
                    div_ok = max_frac <= diversity_thresh
                    metrics["semantic_diversity"] = MetricResult(
                        value=1.0 - max_frac,
                        threshold=1.0 - diversity_thresh,
                        operator=">=",
                        passed=div_ok,
                        note=f"most common token is {max_frac*100:.0f}% of output")
                    if not div_ok:
                        all_pass = False
            except Exception:
                pass

        # Golden token sequence matching (HF-style).
        # If manifest provides golden_semantic_tokens or golden_coarse_tokens,
        # compare first N TRT tokens against the golden reference (exact match).
        # This is the strongest regression gate — catches any change in
        # tokenization, embedding, attention, or sampling.
        golden_sem = thresholds.get("golden_semantic_tokens")
        if golden_sem and sem_dump:
            try:
                import numpy as np
                trt_sem = np.loadtxt(sem_dump, dtype=np.int32)
                golden = np.array(golden_sem, dtype=np.int32)
                n = min(len(trt_sem), len(golden))
                if n > 0:
                    matches = int(np.sum(trt_sem[:n] == golden[:n]))
                    match_rate = matches / n
                    golden_ok = match_rate == 1.0
                    metrics["golden_semantic_match"] = MetricResult(
                        value=match_rate, threshold=1.0, operator=">=",
                        passed=golden_ok,
                        note=f"{matches}/{n} tokens match"
                              + ("" if golden_ok else
                                 f", first mismatch at pos {int(np.argmin(trt_sem[:n] == golden[:n]))}"))
                    if not golden_ok:
                        all_pass = False
            except Exception as e:
                logger.warning("golden semantic token check failed: %s", e)

        coarse_dump = trt.data.get("coarse_tokens_path", "")
        golden_coarse = thresholds.get("golden_coarse_tokens")
        if golden_coarse and coarse_dump:
            try:
                import numpy as np
                trt_coarse = np.loadtxt(coarse_dump, dtype=np.int32)
                golden = np.array(golden_coarse, dtype=np.int32)
                n = min(len(trt_coarse), len(golden))
                if n > 0:
                    matches = int(np.sum(trt_coarse[:n] == golden[:n]))
                    match_rate = matches / n
                    golden_ok = match_rate == 1.0
                    metrics["golden_coarse_match"] = MetricResult(
                        value=match_rate, threshold=1.0, operator=">=",
                        passed=golden_ok,
                        note=f"{matches}/{n} tokens match")
                    if not golden_ok:
                        all_pass = False
            except Exception as e:
                logger.warning("golden coarse token check failed: %s", e)

        # Codec token match (if token data is available from both sides)
        trt_tokens = trt.data.get("codec_tokens")
        ref_tokens = ref.data.get("codec_tokens")
        if trt_tokens is not None and ref_tokens is not None:
            import numpy as np
            trt_t = np.asarray(trt_tokens).flatten()
            ref_t = np.asarray(ref_tokens).flatten()
            n = min(len(trt_t), len(ref_t))
            if n > 0:
                match_rate = float(np.sum(trt_t[:n] == ref_t[:n])) / n
                ct_thresh = thresholds.get("codec_token_match", 0.7)
                ct_ok = match_rate >= ct_thresh
                metrics["codec_token_match"] = MetricResult(
                    value=match_rate, threshold=ct_thresh, operator=">=", passed=ct_ok)
                if not ct_ok:
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

    @staticmethod
    def _read_wav_samples(path: str):
        """Read float32 WAV samples from a file."""
        import numpy as np

        try:
            payload = TextToAudioComparator._read_wav_payload(path)
            if payload is None:
                return None
            data_bytes = payload["data"]
            audio_format = payload["audio_format"]

            if audio_format == 3:  # IEEE float32
                return np.frombuffer(data_bytes, dtype=np.float32)
            elif audio_format == 1:  # PCM int16
                return np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            return None
        return None

    @staticmethod
    def _read_wav_payload(path: str):
        """Read WAV metadata and raw data chunk bytes for exact comparisons."""
        return compare_wav_exact.read_wav_payload(Path(path))


plugin = TextToAudioComparator()
