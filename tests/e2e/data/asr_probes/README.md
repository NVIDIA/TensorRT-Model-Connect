# ASR Probe Inputs

These WAV files are generated from the existing ASR fixture at
`tests/e2e/data/Recording.wav`. They exercise TensorRT Model Connect audio
input handling paths and TRT-vs-reference transcript parity. They are not
intended to benchmark ASR model quality.

| ID | File | SR | Ch | Duration | Readiness | Model Connect path |
|---|---|---:|---:|---:|---|---|
| probe_01_clean_48k_stereo_baseline | `probe_01_clean_48k_stereo_baseline.wav` | 48000 | 2 | 4.16s | ready_now | WAV decode + stereo input + model-specific resample. |
| probe_02_clean_16k_mono_no_resample | `probe_02_clean_16k_mono_no_resample.wav` | 16000 | 1 | 4.16s | ready_now | WAV decode without model-side downmix; should avoid 48k->16k resample in ASR preprocessing. |
| probe_03_clean_48k_mono_resample | `probe_03_clean_48k_mono_resample.wav` | 48000 | 1 | 4.16s | ready_now | WAV decode + 48k->16k resample, without stereo downmix. |
| probe_04_clean_48k_stereo_gain_skew | `probe_04_clean_48k_stereo_gain_skew.wav` | 48000 | 2 | 4.16s | ready_now | Stereo downmix should be robust when L/R channels are not identical. |
| probe_05_low_volume_48k_stereo | `probe_05_low_volume_48k_stereo.wav` | 48000 | 2 | 4.16s | ready_now | PCM int16 decode + float normalization + ASR preprocessing. |
| probe_06_leading_trailing_silence_48k_stereo | `probe_06_leading_trailing_silence_48k_stereo.wav` | 48000 | 2 | 6.16s | ready_now | Longer waveform decode + silence handling + transcript stability. |
| probe_08_noisy_48k_stereo_snr20 | `probe_08_noisy_48k_stereo_snr20.wav` | 48000 | 2 | 4.16s | nightly_optional | PCM decode + preprocessing under non-clean waveform. |

Use `manifest.json` for detailed purpose and notes. Re-run
`generate_asr_probe_inputs.py` from this directory to regenerate the WAVs.
