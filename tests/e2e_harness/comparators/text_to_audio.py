"""Text-to-audio comparator.

Compares TRT audio generation output against reference with generic metrics:
- Codec token match rate
- Mel-spectrogram distance
- Log-spectral distance
- Duration and RMS bounds
"""

from __future__ import annotations

import logging
import math

from ..contracts import CompareResult, MetricResult, StageOutput, StageSpec, StageStatus, ThresholdProfile

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
        trt_wav_path = trt.data.get("wav_path", "")
        ref_wav_path = ref.data.get("wav_path", "")
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
        import struct
        import numpy as np

        try:
            with open(path, "rb") as f:
                riff = f.read(4)
                if riff != b"RIFF":
                    return None
                f.read(4)  # chunk size
                f.read(4)  # WAVE

                data_bytes = b""
                audio_format = 1
                while True:
                    chunk_id = f.read(4)
                    if len(chunk_id) < 4:
                        break
                    chunk_size = struct.unpack("<I", f.read(4))[0]
                    if chunk_id == b"fmt ":
                        fmt_data = f.read(chunk_size)
                        audio_format = struct.unpack("<H", fmt_data[0:2])[0]
                    elif chunk_id == b"data":
                        data_bytes = f.read(chunk_size)
                    else:
                        f.read(chunk_size)

            if audio_format == 3:  # IEEE float32
                return np.frombuffer(data_bytes, dtype=np.float32)
            elif audio_format == 1:  # PCM int16
                return np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            return None
        return None


plugin = TextToAudioComparator()
