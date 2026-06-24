#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

struct MelResult {
    std::vector<float> data; // [n_mels, n_frames] row-major
    int32_t n_mels{0};
    int32_t n_frames{0};
};

// Extract mel spectrogram from audio samples, matching HuggingFace
// WhisperFeatureExtractor._np_extract_fbank_features exactly.
//
// Parameters:
//   samples:        mono float32 audio samples (16kHz expected)
//   n_samples:      number of samples
//   mel_filters:    mel filterbank matrix [n_freq_bins * n_mel_bins] row-major
//                   where rows = frequency bins, cols = mel bins
//   n_freq_bins:    number of frequency bins (n_fft/2 + 1, typically 201)
//   n_mel_bins:     number of mel filter outputs (80 or 128)
//   n_fft:          FFT window size (400 for Whisper)
//   hop_length:     hop between frames (160 for Whisper)
//   chunk_length_s: audio chunk length in seconds (30 for Whisper)
//   sample_rate:    sample rate (16000 for Whisper)
//
// Returns mel spectrogram [n_mel_bins, n_frames].
MelResult extract_mel_spectrogram(const float* samples, int32_t n_samples, const float* mel_filters,
                                  int32_t n_freq_bins, int32_t n_mel_bins, int32_t n_fft,
                                  int32_t hop_length, int32_t chunk_length_s, int32_t sample_rate);

// Extract NeMo ASR-style log-mel features.
//
// Matches AudioToMelSpectrogramPreprocessor for NeMo ASR configs:
// center=True STFT, win_length < n_fft, preemphasis, power mel projection,
// natural log with additive guard, and optional per-feature normalization over
// the valid audio frames.
MelResult extract_nemo_mel_spectrogram(const float* samples, int32_t n_samples,
                                       const float* mel_filters, int32_t n_freq_bins,
                                       int32_t n_mel_bins, int32_t n_fft, int32_t win_length,
                                       int32_t hop_length, int32_t chunk_length_s,
                                       int32_t sample_rate, float preemph,
                                       bool normalize_per_feature = false);

} // namespace trtmc
