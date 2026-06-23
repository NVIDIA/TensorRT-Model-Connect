#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

struct MelResult {
    std::vector<float> data; // [n_mels, n_frames] row-major
    int32_t n_mels{0};
    int32_t n_frames{0};
};

enum class MelLogScale {
    kLog10Normalized,
    kNaturalLog,
};

struct MelSpectrogramOptions {
    int32_t n_fft{400};
    int32_t win_length{0};
    int32_t hop_length{160};
    int32_t chunk_length_s{30};
    int32_t sample_rate{16000};
    bool symmetric_window{false};
    bool center_window_in_fft{false};
    float preemphasis{0.0F};
    MelLogScale log_scale{MelLogScale::kLog10Normalized};
    bool normalize_per_feature{false};
};

// Extract log-mel spectrogram features from audio samples, matching the
// HuggingFace feature-extractor contract used by compatible ASR models.
//
// Parameters:
//   samples:        mono float32 audio samples
//   n_samples:      number of samples
//   mel_filters:    mel filterbank matrix [n_freq_bins * n_mel_bins] row-major
//                   where rows = frequency bins, cols = mel bins
//   n_freq_bins:    number of frequency bins (n_fft/2 + 1, typically 201)
//   n_mel_bins:     number of mel filter outputs (80 or 128)
//   n_fft:          FFT window size
//   hop_length:     hop between frames
//   chunk_length_s: audio chunk length in seconds
//   sample_rate:    sample rate
//
// Returns mel spectrogram [n_mel_bins, n_frames].
MelResult extract_mel_spectrogram(const float* samples, int32_t n_samples, const float* mel_filters,
                                  int32_t n_freq_bins, int32_t n_mel_bins, int32_t n_fft,
                                  int32_t hop_length, int32_t chunk_length_s, int32_t sample_rate);

// Extract log-mel spectrogram features with explicit low-level signal options.
// Model-owned code selects the concrete option set required by that model.
MelResult extract_configured_mel_spectrogram(const float* samples, int32_t n_samples,
                                             const float* mel_filters, int32_t n_freq_bins,
                                             int32_t n_mel_bins,
                                             const MelSpectrogramOptions& options);

} // namespace trtmc
